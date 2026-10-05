//! Where a GPU nice-only field runs, decided once per field for every
//! backend.
//!
//! A backend offers its pipelines through [`NiceonlyGpu`]: the MSD/stride
//! pipeline, the overlap join if it has one, and finishing a field from
//! either. [`begin_niceonly`] decides a field's route (answered on the spot,
//! the join, or the stride pipeline) and hands it to that pipeline, which
//! returns a [`FieldTicket`]. The caller gives the ticket back to
//! [`NiceonlyGpu::finish`] for the field's results. Each pipeline is first
//! in, first out and checks that its tickets come back in order, so a caller
//! can keep several fields in flight with no routing bookkeeping of its own.
#![cfg(any(feature = "cuda", feature = "vulkan", feature = "cubecl"))]

use crate::client_process::process_range_niceonly;
use crate::gpu_config::gpu_supports_base;
use crate::gpu_niceonly::{GPU_LSD_K, NiceonlyStats, residue_empty_result};
use crate::join_plan::{JoinField, JoinLimits, plan_join};
use crate::overlap_join::{StrideReason, join_verdict};
use crate::stride_filter::StrideTable;
use crate::{FieldResults, FieldSize};
use anyhow::{Result, anyhow, ensure};
use log::{debug, warn};

/// Which of a GPU's pipelines a field went to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Route {
    /// The MSD/stride pipeline (`gpu_niceonly::NiceonlyPipeline`).
    Stride,
    /// The overlap join (see `crate::join_plan`).
    Join,
}

/// A field queued in one of a GPU's pipelines: give it back to
/// [`NiceonlyGpu::finish`] for the field's results. Tickets of one route
/// must come back in the order they were issued; the pipeline checks.
#[derive(Debug)]
#[must_use = "a queued field must be finished with its ticket"]
pub struct FieldTicket {
    route: Route,
    /// The field's place in its pipeline.
    seq: u64,
    /// Why a stride field did not take the join.
    stride_reason: Option<StrideReason>,
}

impl FieldTicket {
    pub(crate) fn new(route: Route, seq: u64) -> Self {
        Self {
            route,
            seq,
            stride_reason: None,
        }
    }

    /// The pipeline holding the field.
    #[must_use]
    pub fn route(&self) -> Route {
        self.route
    }

    /// Why the field took the stride pipeline, if it did.
    #[must_use]
    pub fn stride_reason(&self) -> Option<StrideReason> {
        self.stride_reason
    }

    fn because(mut self, reason: StrideReason) -> Self {
        self.stride_reason = Some(reason);
        self
    }

    /// Hand the ticket back to the `route` pipeline, whose oldest field is
    /// number `oldest`.
    ///
    /// # Errors
    /// The ticket is another pipeline's, or not for the oldest field.
    pub(crate) fn redeem(self, route: Route, oldest: u64) -> Result<()> {
        ensure!(
            self.route == route && self.seq == oldest,
            "{:?} field #{} finished out of order: the {route:?} pipeline's oldest is #{oldest}",
            self.route,
            self.seq
        );
        Ok(())
    }

    #[cfg(test)]
    pub(crate) fn seq(&self) -> u64 {
        self.seq
    }
}

/// What [`begin_niceonly`] did with a field: answered it on the spot, or
/// queued it in one of the GPU's pipelines.
pub enum NiceonlyStarted {
    Immediate(FieldResults),
    Queued(FieldTicket),
}

/// A GPU backend's nice-only pipelines. Fields go in through
/// [`begin_niceonly`], which picks the pipeline.
pub trait NiceonlyGpu {
    /// The device's limits for the overlap join, or `None` if this backend
    /// has no join; then every field takes the stride pipeline.
    fn join_limits(&self) -> Option<JoinLimits> {
        None
    }

    /// Queue a field in the MSD/stride pipeline.
    ///
    /// # Errors
    /// The pipeline's threads have exited.
    fn begin_stride(&self, range: &FieldSize, base: u32) -> Result<FieldTicket>;

    /// Queue a field that [`plan_join`] prepared for this device.
    ///
    /// # Errors
    /// The join's worker has exited, or this backend has no join.
    fn begin_join(&self, field: JoinField) -> Result<FieldTicket> {
        let _ = field;
        Err(anyhow!("this backend has no overlap join"))
    }

    /// Wait for a queued field and return its results.
    ///
    /// # Errors
    /// The field's own error (device failure, output overflow), its
    /// pipeline having stopped, or a ticket out of order.
    fn finish(&self, ticket: FieldTicket) -> Result<(FieldResults, NiceonlyStats)>;
}

/// Start a nice-only field on `gpu`. This is the one place a field's route
/// is decided:
/// - a residue-empty base has no candidates: answered on the spot;
/// - a base the GPU kernels cannot take runs on the CPU, on the spot;
/// - a field the overlap join takes on this device ([`plan_join`]) goes to
///   the join;
/// - every other field goes to the stride pipeline.
///
/// # Errors
/// The chosen pipeline's error (see [`NiceonlyGpu`]).
pub fn begin_niceonly(
    gpu: &dyn NiceonlyGpu,
    range: &FieldSize,
    base: u32,
) -> Result<NiceonlyStarted> {
    if let Some(empty) = residue_empty_result(base) {
        return Ok(NiceonlyStarted::Immediate(empty));
    }
    if !gpu_supports_base(base) {
        warn!("base {base} not supported on GPU, falling back to CPU for this field");
        let table = StrideTable::new(base, GPU_LSD_K);
        return Ok(NiceonlyStarted::Immediate(process_range_niceonly(
            range, base, &table,
        )));
    }
    let verdict = match gpu.join_limits() {
        Some(lim) => plan_join(base, range, lim),
        // The field's own reason if it has one, else the backend's.
        None => Err(join_verdict(base, range)
            .err()
            .unwrap_or(StrideReason::NoJoin)),
    };
    let ticket = match verdict {
        Ok(field) => {
            debug!(
                "b{base} [{}, {}): overlap join, {} slice(s) of {} partitions, {} per launch",
                range.start(),
                range.end(),
                field.slice_count(),
                field.partitions(),
                field.plan.slots
            );
            gpu.begin_join(field)?
        }
        Err(reason) => {
            debug!(
                "b{base} [{}, {}): stride pipeline, {}",
                range.start(),
                range.end(),
                reason.label()
            );
            gpu.begin_stride(range, base)?.because(reason)
        }
    };
    Ok(NiceonlyStarted::Queued(ticket))
}

/// [`NiceonlyGpu::finish`], with the route's reason from the ticket put
/// into the field's stats (for telemetry).
///
/// # Errors
/// See [`NiceonlyGpu::finish`].
pub fn finish_niceonly(
    gpu: &dyn NiceonlyGpu,
    ticket: FieldTicket,
) -> Result<(FieldResults, NiceonlyStats)> {
    let reason = ticket.stride_reason();
    let (results, mut stats) = gpu.finish(ticket)?;
    stats.route_reason = reason;
    Ok((results, stats))
}

/// [`begin_niceonly`] and [`NiceonlyGpu::finish`] in one, for callers that
/// process one field at a time (the benchmark, tests).
///
/// # Errors
/// See [`begin_niceonly`] and [`NiceonlyGpu::finish`].
pub fn process_niceonly(
    gpu: &dyn NiceonlyGpu,
    range: &FieldSize,
    base: u32,
) -> Result<FieldResults> {
    match begin_niceonly(gpu, range, base)? {
        NiceonlyStarted::Immediate(results) => Ok(results),
        NiceonlyStarted::Queued(ticket) => finish_niceonly(gpu, ticket).map(|(results, _)| results),
    }
}
