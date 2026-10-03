//! A field fitted to a device for the overlap join's GPU stage.
//!
//! - [`JoinPlan`]: how a field's buffers fit one device. Partitions per
//!   launch and the survivor list's length are chosen so that each buffer
//!   fits the device's largest binding and all of them a budget
//!   ([`JoinLimits`]).
//! - [`plan_join`]: the join's verdict on a field for a device, which
//!   decides the field's route ([`crate::gpu_route::begin_niceonly`]).
//!
//! The field's host side, which every partition value starts from, is
//! [`FieldSetup`]. Nothing here depends on a GPU runtime: the `CubeCL` stage
//! (`crate::cubecl_join`) allocates exactly the buffers [`Footprint`]
//! counts.
#![cfg(any(feature = "cuda", feature = "vulkan", feature = "cubecl"))]
// Only `CubeCL` runs the join so far: without it a field is still planned
// (the route is decided the same way everywhere) but nothing reads the plan.
#![cfg_attr(not(feature = "cubecl"), allow(dead_code))]

use crate::FieldSize;
use crate::gpu_config::n_limbs;
use crate::overlap_join::{FieldSetup, JoinParams, join_params_for};
use anyhow::Result;
use log::warn;

/// Partitions per launch (device slots), bounded below by memory. A field
/// is b^p partitions (3,249 at base 57), so this sets the launch count: 16
/// measured 1.11-1.37x faster per field than 4 on an RTX 3080 and an RTX
/// 4090, where launch overhead was most of the join's fixed per-field cost.
const SLOTS: usize = 16;
/// Launched batches kept in flight before the driver waits for the oldest
/// (each holds a little scratch of its own, counted in [`Footprint`]).
pub(crate) const BATCHES_IN_FLIGHT: usize = 4;
/// Prefilter survivors one batch may produce (each 8 bytes). A base-57
/// partition yields 1.5-2.7e6 join survivors, of which ~6% pass the
/// prefilter, so 16 partitions make ~2.6e6 at most.
const SURV_CAP: u32 = 1 << 24;
/// The same for the overflow re-run (one partition per launch).
const SURV_CAP_RETRY: u32 = 1 << 26;
/// The smallest survivor list the overflow re-run may have; a device that
/// cannot hold one partition with a list this long does not get the join.
/// A partition that overflows the re-run's list is re-run on halves of its
/// top layer, down to single top prefixes if need be, and one top of one
/// partition is one block of `b^f0` numbers: at most 2^18 at base 64 with
/// the production `f0 = 3`, so this always fits it.
pub(crate) const MIN_RETRY_CAP: u32 = 1 << 18;
/// Nice numbers one field may report.
const NICE_CAP: u32 = 1 << 10;
/// Bytes per hit in the output list: `n` as four `u32` limbs, as the check
/// kernels write it.
pub(crate) const NICE_RECORD_BYTES: usize = 16;
/// The join's whole device working set may take at most this much; a field
/// at bases 40-64 needs 0.3-0.9 GiB at 16 partitions per launch. See
/// [`JoinLimits`].
const JOIN_MEMORY: usize = 1 << 30;

/// What a device allows the join: its largest buffer, and a budget for all
/// of one field's buffers together.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct JoinLimits {
    pub(crate) max_binding: usize,
    pub(crate) budget: usize,
}

impl JoinLimits {
    /// The limits of a device whose largest buffer is `max_buffer` bytes:
    /// on wgpu the adapter's `max_storage_buffer_binding_size` (128 MiB on
    /// lavapipe, 1-4 GiB on desktop drivers); on CUDA, and on Vulkan with
    /// 64-bit indexing, `CubeCL` reports a quarter of the device's memory.
    /// The budget is [`JOIN_MEMORY`] but at most twice the largest buffer:
    /// half the device where the runtime knows its size, and a small working
    /// set where buffers are small (software rasterizers, embedded GPUs).
    #[must_use]
    pub fn for_buffer(max_buffer: u64) -> Self {
        let max_binding = usize::try_from(max_buffer).unwrap_or(usize::MAX).max(1);
        Self {
            max_binding,
            budget: JOIN_MEMORY.min(max_binding.saturating_mul(2)),
        }
    }
}

/// Bytes in each of one field's device buffers, as the GPU stage allocates
/// them (`cubecl_join`'s `JoinDevice::new`): per slot for those that grow
/// with the partitions per launch, then the field's own tables.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Footprint {
    pub(crate) ext_m: usize,
    pub(crate) ext_pk: usize,
    pub(crate) lists: usize,
    pub(crate) counts: usize,
    /// Each of `top_p`, `top_m`, `top_r` and `top_x`.
    pub(crate) top: usize,
    pub(crate) top_b: usize,
    pub(crate) tl: usize,
    pub(crate) work: usize,
    pub(crate) cursor: usize,
    /// What a launched batch allocates for itself (`bucket_cnt`), for every
    /// batch in flight.
    pub(crate) scratch: usize,
    /// The field's tables and the output list.
    pub(crate) fixed: usize,
    pub(crate) largest_fixed: usize,
}

impl Footprint {
    pub(crate) fn of(fs: &FieldSetup, nice_cap: u32) -> Self {
        let b = fs.b as usize;
        let nbp = fs.bp_r.len();
        let ntl = fs.tlay.len().max(1);
        let nroots = fs.base.roots.len();
        let nkeys = if fs.key_level { b } else { 1 };
        let nl3 = 3 * n_limbs(fs.b).unwrap_or(3) as usize;
        let tables = [
            nbp * 4,                               // bp_r
            nbp * 8,                               // bp_m
            fs.seg.len() * 4,                      // seg
            fs.tlay.len() * 5 * 4,                 // tlay
            10 * 4,                                // fb
            (fs.base.s3 as usize + 1) * nl3 * 4,   // pw
            nroots * 4,                            // roots
            nice_cap as usize * NICE_RECORD_BYTES, // nice_out
            4,                                     // nice_count
        ];
        Self {
            ext_m: nbp * 8,
            ext_pk: nbp.max(1) * 4,
            lists: nbp * b * 4,
            counts: nkeys * (b - 1) * 4,
            top: ntl * 8,
            top_b: ntl * 4,
            tl: ntl * nroots * 4,
            work: fs.nb * 16,
            cursor: fs.nb * 4,
            scratch: fs.nb * 4 * BATCHES_IN_FLIGHT,
            fixed: tables.iter().sum(),
            largest_fixed: tables.into_iter().max().unwrap_or(0),
        }
    }

    pub(crate) fn per_slot(&self) -> usize {
        self.ext_m
            + self.ext_pk
            + self.lists
            + self.counts
            + 4 * self.top
            + self.top_b
            + self.tl
            + self.work
            + self.cursor
            + self.scratch
    }

    /// The largest buffer one slot adds to (all of them grow linearly).
    pub(crate) fn largest_per_slot(&self) -> usize {
        [
            self.ext_m,
            self.ext_pk,
            self.lists,
            self.counts,
            self.top,
            self.top_b,
            self.tl,
            self.work,
            self.cursor,
        ]
        .into_iter()
        .max()
        .unwrap_or(0)
    }
}

/// One field's layout on a device: partitions per launch and the lengths
/// of the survivor lists, chosen so that every buffer fits the device's
/// binding limit and all of them together fit the budget ([`JoinLimits`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct JoinPlan {
    /// Partitions per launch.
    pub slots: usize,
    /// Entries in the survivor list: the prefilter's survivors, which the
    /// full check reads.
    pub list_cap: u32,
    pub nice_cap: u32,
    /// Every buffer of the field together, the batches in flight included.
    pub bytes: usize,
}

impl JoinPlan {
    /// The layout for `fs` with at most `slots` partitions per launch and a
    /// survivor list of at most `cap` entries (8 bytes each), fitted to
    /// `lim`: the list gets at most `list_budget` bytes, then as many slots
    /// as fit the rest of the budget. `None` if one slot with a list of
    /// `min_cap` does not.
    pub(crate) fn new(
        fs: &FieldSetup,
        lim: JoinLimits,
        slots: usize,
        cap: u32,
        min_cap: u32,
        list_budget: usize,
    ) -> Option<Self> {
        let fp = Footprint::of(fs, NICE_CAP);
        let list_cap = u32::try_from((cap as usize).min(lim.max_binding / 8).min(list_budget / 8))
            .unwrap_or(u32::MAX);
        if list_cap < min_cap.max(1) || fp.largest_fixed > lim.max_binding {
            return None;
        }
        let list = list_cap as usize * 8;
        let room = lim.budget.checked_sub(fp.fixed + list)?;
        let slots = slots
            .min(lim.max_binding / fp.largest_per_slot().max(1))
            .min(room / fp.per_slot().max(1));
        (slots > 0).then(|| JoinPlan {
            slots,
            list_cap,
            nice_cap: NICE_CAP,
            bytes: fp.fixed + list + slots * fp.per_slot(),
        })
    }

    /// The production layout of `fs` on a device with `lim` (the lists get
    /// a quarter of the budget), and the one for re-running a partition that
    /// overflowed it: one slot, and the longest list that fits beside it.
    /// `None` if the device cannot hold one partition with a re-run list of
    /// [`MIN_RETRY_CAP`].
    pub(crate) fn for_field(fs: &FieldSetup, lim: JoinLimits) -> Option<(Self, Self)> {
        let main = Self::new(fs, lim, SLOTS, SURV_CAP, 1, lim.budget / 4)?;
        let fp = Footprint::of(fs, NICE_CAP);
        let spare = lim.budget.saturating_sub(fp.fixed + fp.per_slot());
        let retry = Self::new(fs, lim, 1, SURV_CAP_RETRY, MIN_RETRY_CAP, spare)?;
        Some((main, retry))
    }
}

/// A nice-only field ready for the join on one device: its host side and its
/// two device layouts, the batches' and the overflow re-run's. Preparing it
/// decides the field's route ([`plan_join`]), and the device stage then has
/// no host work left before its first launch.
pub struct JoinField {
    pub(crate) fs: FieldSetup,
    pub(crate) plan: JoinPlan,
    pub(crate) retry: JoinPlan,
}

impl JoinField {
    /// Prepare `[range)` at `base` with parameters `jp` for a device with
    /// `lim`; `Ok(None)` if the device cannot hold one partition of it.
    ///
    /// # Errors
    /// A field or parameters the join cannot take.
    pub(crate) fn prepare(
        base: u32,
        range: &FieldSize,
        jp: JoinParams,
        lim: JoinLimits,
    ) -> Result<Option<Self>> {
        let fs = FieldSetup::new(base, range.start(), range.end(), jp)?;
        Ok(JoinPlan::for_field(&fs, lim).map(|(plan, retry)| Self { fs, plan, retry }))
    }

    /// A field with layouts of the test's choosing.
    #[cfg(test)]
    pub(crate) fn with_plans(fs: FieldSetup, plan: JoinPlan, retry: JoinPlan) -> Self {
        Self { fs, plan, retry }
    }
}

/// The overlap join's verdict on a nice-only field for a device with `lim`:
/// the field prepared for the join, or `None` for the stride pipeline. The
/// join takes production-size fields at bases 40-64 ([`join_params_for`])
/// of which the device can hold at least one partition
/// ([`JoinPlan::for_field`]); a device that cannot is said so once in the
/// log. Preparing the field (20-110 ms) happens on the caller's thread, so
/// in the client it overlaps the device's work on the previous field.
#[must_use]
pub fn plan_join(base: u32, range: &FieldSize, lim: JoinLimits) -> Option<JoinField> {
    static TOO_SMALL: std::sync::Once = std::sync::Once::new();
    let jp = join_params_for(base, range)?;
    match JoinField::prepare(base, range, jp, lim) {
        Ok(Some(field)) => Some(field),
        Ok(None) => {
            TOO_SMALL.call_once(|| {
                warn!(
                    "overlap join: this device ({} MiB buffer limit, {} MiB budget) cannot hold \
                     one partition of a base-{base} field; those fields use the stride pipeline",
                    lim.max_binding >> 20,
                    lim.budget >> 20
                );
            });
            None
        }
        Err(e) => {
            warn!(
                "overlap join cannot take base {base} {range:?} ({e:#}); using the stride pipeline"
            );
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::overlap_join::test_fields::{FRONTIER_57, PLAN_FIELDS};

    fn plan_field(base: u32, start: u128, size: u128) -> FieldSetup {
        let range = FieldSize::new(start, start + size);
        let jp = crate::overlap_join::join_params_for(base, &range).expect("a join field");
        FieldSetup::new(base, range.start(), range.end(), jp).expect("field setup")
    }

    /// On a desktop GPU (CUDA reports a quarter of an 8 GB card as its
    /// largest buffer) every production field gets the measured layout: 16
    /// partitions per launch and full lists, within the budget.
    #[test]
    fn production_fields_get_the_full_layout() {
        let lim = JoinLimits {
            max_binding: 2 << 30,
            budget: JOIN_MEMORY,
        };
        for &(base, start, size) in PLAN_FIELDS {
            let fs = plan_field(base, start, size);
            let (plan, retry) = JoinPlan::for_field(&fs, lim).expect("fits a desktop GPU");
            assert_eq!((plan.slots, plan.list_cap), (SLOTS, SURV_CAP), "b{base}");
            assert_eq!(
                (retry.slots, retry.list_cap),
                (1, SURV_CAP_RETRY),
                "b{base}"
            );
            assert!(
                plan.bytes <= lim.budget && retry.bytes <= lim.budget,
                "b{base}"
            );
        }
    }

    /// A device with small buffers gets a smaller layout that respects them,
    /// and one that cannot hold a single partition gets none (its fields
    /// stay on the stride pipeline).
    #[test]
    fn small_devices_get_smaller_layouts_or_none() {
        let fs = plan_field(57, FRONTIER_57, 100_000_000_000_000);
        let fp = Footprint::of(&fs, NICE_CAP);
        // lavapipe and other software rasterizers: 128 MiB buffers.
        let lim = JoinLimits {
            max_binding: 128 << 20,
            budget: 256 << 20,
        };
        let (plan, retry) = JoinPlan::for_field(&fs, lim).expect("lavapipe holds base 57");
        assert!((1..SLOTS).contains(&plan.slots), "{plan:?}");
        assert!(plan.slots * fp.largest_per_slot() <= lim.max_binding);
        assert!(plan.list_cap as usize * 8 <= lim.max_binding);
        assert!(retry.list_cap >= MIN_RETRY_CAP);
        assert!(plan.bytes <= lim.budget && retry.bytes <= lim.budget);
        // Buffers smaller than one partition's bucket lists (29 MiB here).
        let lim = JoinLimits {
            max_binding: 16 << 20,
            budget: 1 << 30,
        };
        assert_eq!(JoinPlan::for_field(&fs, lim), None);
        // A budget smaller than one partition and the re-run's list.
        let lim = JoinLimits {
            max_binding: 1 << 30,
            budget: 32 << 20,
        };
        assert_eq!(JoinPlan::for_field(&fs, lim), None);
    }
}
