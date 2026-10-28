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
#![cfg(any(feature = "cuda", feature = "cubecl"))]
// Only `CubeCL` runs the join so far: without it a field is still planned
// (the route is decided the same way everywhere) but nothing reads the plan.
#![cfg_attr(not(feature = "cubecl"), allow(dead_code))]

use crate::FieldSize;
use crate::gpu_config::n_limbs;
use crate::overlap_join::{
    FieldSetup, StrideReason, join_slices, join_verdict, ndigits, prefix_block,
};
use anyhow::Result;
use log::warn;
use std::sync::atomic::{AtomicUsize, Ordering};

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
/// An upper bound on the prefilter survivors one partition yields per
/// top-layer prefix of the field: measured 0.8-1.6 at bases 57-64 on fields
/// of 1e14-1e15 (b57 frontier 1.08, b58 1.09-1.11, b60 0.79, Dan's dense
/// b57 fields up to about 1.6). A batch is planned to fill at most half its
/// survivor list on this bound; one that still overflows is re-run in
/// smaller batches (`cubecl_join::run_field`).
const SURV_PER_PREFIX: usize = 2;
/// Partitions per launch a slice of a large field is cut for
/// ([`max_slice_prefixes`]). A slice of that size usually gets two or three
/// times as many, since only a third to a half of its top-layer prefixes
/// are certified.
const SLICE_SLOTS: usize = 2;

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

/// The largest layout one device may get, learned from the device: one
/// that does not run a layout its limits allow gets at most half of that
/// layout from then on. The macOS runner's paravirtual Metal device reports
/// a 3.5 GiB buffer limit and runs a 626 MiB layout, but drops every
/// dispatch of a 697 MiB one without an error. A device's planning (through
/// its limits) and its join's worker share one ceiling, so every later
/// field is planned within it.
#[derive(Debug)]
pub struct JoinCeiling(AtomicUsize);

impl Default for JoinCeiling {
    fn default() -> Self {
        Self(AtomicUsize::new(usize::MAX))
    }
}

impl JoinCeiling {
    /// `lim` with neither its budget nor its largest buffer above the
    /// ceiling.
    pub(crate) fn cap(&self, lim: JoinLimits) -> JoinLimits {
        let c = self.get();
        JoinLimits {
            max_binding: lim.max_binding.min(c),
            budget: lim.budget.min(c),
        }
    }

    /// The device did not run a layout of `bytes`: at most half of it from
    /// now on. Returns the ceiling.
    pub(crate) fn lower(&self, bytes: usize) -> usize {
        let half = bytes / 2;
        self.0.fetch_min(half, Ordering::Relaxed).min(half)
    }

    pub(crate) fn get(&self) -> usize {
        self.0.load(Ordering::Relaxed)
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
        Self::with_prefixes(fs, fs.tlay.len(), nice_cap)
    }

    /// The footprint `fs`'s field would have with `ntl` top-layer prefixes:
    /// every buffer is affine in it.
    pub(crate) fn with_prefixes(fs: &FieldSetup, ntl: usize, nice_cap: u32) -> Self {
        let b = fs.b as usize;
        let nbp = fs.bp_r.len();
        let ntl = ntl.max(1);
        let nroots = fs.base.roots.len();
        let nkeys = if fs.key_level { b } else { 1 };
        let nl3 = 3 * n_limbs(fs.b).unwrap_or(3) as usize;
        let tables = [
            nbp * 4,                               // bp_r
            nbp * 8,                               // bp_m
            fs.seg.len() * 4,                      // seg
            ntl * 5 * 4,                           // tlay
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
        // The batch's expected prefilter survivors fill at most half its list.
        let by_survivors =
            (list_cap as usize / 2 / (SURV_PER_PREFIX * fs.tlay.len().max(1))).max(1);
        let slots = slots
            .min(lim.max_binding / fp.largest_per_slot().max(1))
            .min(room / fp.per_slot().max(1))
            .min(by_survivors);
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

/// The most top-layer prefixes one slice of `fs`'s field may have on a
/// device with `lim`: [`SLICE_SLOTS`] partitions per launch must fit the
/// budget and the largest binding, with their expected survivors in half the
/// list. 0 if not even one prefix fits.
///
/// Every buffer that grows with the field grows with its top layer, so this
/// bounds them all, whatever the field's size ([`plan_join`]).
pub(crate) fn max_slice_prefixes(fs: &FieldSetup, lim: JoinLimits) -> usize {
    let (one, two) = (
        Footprint::with_prefixes(fs, 1, NICE_CAP),
        Footprint::with_prefixes(fs, 2, NICE_CAP),
    );
    // Bytes per prefix: in the field's tables, and per slot.
    let per_fixed = two.fixed - one.fixed;
    let per_slot = two.per_slot() - one.per_slot();
    let per_largest = two.top.max(two.tl) - one.top.max(one.tl);
    let list_cap = (SURV_CAP as usize)
        .min(lim.max_binding / 8)
        .min(lim.budget / 4 / 8);
    let s = SLICE_SLOTS;
    let base_bytes = one.fixed - per_fixed + list_cap * 8 + s * (one.per_slot() - per_slot);
    let by_budget = lim.budget.saturating_sub(base_bytes) / (per_fixed + s * per_slot).max(1);
    let by_binding =
        (lim.max_binding / s / per_largest.max(1)).min(lim.max_binding / per_fixed.max(1));
    let by_survivors = list_cap / 2 / (s * SURV_PER_PREFIX);
    by_budget.min(by_binding).min(by_survivors)
}

/// A nice-only field ready for the join on one device: its slices (one, for
/// any field below about 1e15), the first slice's host side and its two
/// device layouts, the batches' and the overflow re-run's. Preparing it
/// decides the field's route ([`plan_join`]), and the device stage then has
/// no host work left before its first launch; it prepares later slices
/// itself, each while the previous one runs.
pub struct JoinField {
    /// The field's slices in order, and the device they were cut for.
    pub(crate) slices: Vec<FieldSize>,
    pub(crate) lim: JoinLimits,
    /// The first slice's setup and layouts.
    pub(crate) fs: FieldSetup,
    pub(crate) plan: JoinPlan,
    pub(crate) retry: JoinPlan,
}

impl JoinField {
    /// Prepare `[range)` at `base` with parameters `jp` for a device with
    /// `lim`, as one slice; `Ok(None)` if the device cannot hold one
    /// partition of it.
    ///
    /// # Errors
    /// A field or parameters the join cannot take.
    #[cfg(test)]
    pub(crate) fn prepare(
        base: u32,
        range: &FieldSize,
        jp: crate::overlap_join::JoinParams,
        lim: JoinLimits,
    ) -> Result<Option<Self>> {
        let fs = FieldSetup::new(base, range.start(), range.end(), jp)?;
        Ok(JoinPlan::for_field(&fs, lim).map(|(plan, retry)| Self {
            slices: vec![*range],
            lim,
            fs,
            plan,
            retry,
        }))
    }

    /// A field with layouts of the test's choosing, as one slice.
    #[cfg(test)]
    pub(crate) fn with_plans(fs: FieldSetup, plan: JoinPlan, retry: JoinPlan) -> Self {
        Self {
            slices: vec![FieldSize::new(fs.s, fs.e)],
            lim: JoinLimits {
                max_binding: usize::MAX,
                budget: usize::MAX,
            },
            fs,
            plan,
            retry,
        }
    }

    /// The field cut into slices of at most `max_prefixes` top-layer
    /// prefixes (tests force small ones).
    #[cfg(test)]
    pub(crate) fn resliced(mut self, max_prefixes: u128) -> Result<Self> {
        let (s, e) = (
            self.slices[0].start(),
            self.slices[self.slices.len() - 1].end(),
        );
        let block = prefix_block(self.fs.b, self.fs.base.l, self.fs.jp);
        self.slices = join_slices(&FieldSize::new(s, e), block, max_prefixes);
        self.fs = self
            .fs
            .sub_range(self.slices[0].start(), self.slices[0].end())?;
        Ok(self)
    }

    /// The number of slices.
    #[must_use]
    pub fn slice_count(&self) -> usize {
        self.slices.len()
    }

    /// Partitions in each slice (`b^p`).
    #[must_use]
    pub fn partitions(&self) -> u128 {
        self.fs.nparts
    }
}

/// The overlap join's verdict on a nice-only field for a device with `lim`:
/// the field prepared for the join, or why it takes the stride pipeline.
///
/// - **The field:** production-size fields at bases 40-64 ([`join_verdict`]).
/// - **The device:** the field is cut into slices of at most
///   [`max_slice_prefixes`] top-layer prefixes, so that whatever its size each
///   slice fits the device with a few partitions per launch. A device that
///   cannot hold one partition of a slice gets none, and says so once in the
///   log.
///
/// Preparing the first slice (20-300 ms) happens on the caller's thread, so
/// in the client it overlaps the device's work on the previous field.
///
/// # Errors
/// The reason the field takes the stride pipeline.
pub fn plan_join(base: u32, range: &FieldSize, lim: JoinLimits) -> Result<JoinField, StrideReason> {
    static TOO_SMALL: std::sync::Once = std::sync::Once::new();
    let too_small = || {
        TOO_SMALL.call_once(|| {
            warn!(
                "overlap join: this device ({} MiB buffer limit, {} MiB budget) cannot hold \
                 one partition of a base-{base} field; those fields use the stride pipeline",
                lim.max_binding >> 20,
                lim.budget >> 20
            );
        });
        StrideReason::DeviceTooSmall
    };
    let cannot = |e: anyhow::Error| {
        warn!("overlap join cannot take base {base} {range:?} ({e:#}); using the stride pipeline");
        StrideReason::Setup
    };
    let jp = join_verdict(base, range)?;
    let block = prefix_block(base, ndigits(range.last(), base), jp);
    // The bottom side, and from it the slice size, from the first block.
    let first_end = (range.start() / block * block + block).min(range.end());
    let probe = FieldSetup::new(base, range.start(), first_end, jp).map_err(cannot)?;
    let cap = max_slice_prefixes(&probe, lim);
    if cap == 0 {
        return Err(too_small());
    }
    let slices = join_slices(range, block, cap as u128);
    let mut fs = probe
        .sub_range(slices[0].start(), slices[0].end())
        .map_err(cannot)?;
    // The first slice's setup includes the bottom list, built for the probe.
    fs.secs += probe.secs;
    let (plan, retry) = JoinPlan::for_field(&fs, lim).ok_or_else(too_small)?;
    Ok(JoinField {
        slices,
        lim,
        fs,
        plan,
        retry,
    })
}

#[cfg(test)]
pub(crate) mod tests {
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

    /// The densest base-58 region the probes found (2026-10-05), where a 1e15
    /// field overran the join kernel's survivor stage before it flushed per
    /// chunk.
    pub(crate) const DENSE_58: u128 = 102_121_216_587_923_470_200;

    /// A field of 1e14 at base `base`, from the middle of its band.
    fn mid_band(base: u32, size: u128) -> FieldSize {
        let r = crate::base_range::get_base_range_u128(base)
            .unwrap()
            .unwrap();
        let s = r.range_start + (r.range_end - r.range_start - size) / 2;
        FieldSize::new(s, s + size)
    }

    /// Fields of the sizes production uses and will use (1e14 at base 57,
    /// 1e15 at 58, 1e16 at 60, about 1e17 at 62) get the join on an 8-10 GB
    /// card, cut into as few slices as fit: one up to 1e15. Every slice's
    /// buffers fit the budget, and its expected survivors half its list.
    #[test]
    #[allow(clippy::cast_precision_loss)] // the field size, printed
    fn fields_of_every_production_size_are_cut_into_slices_that_fit() {
        let lim = JoinLimits {
            max_binding: 10 << 28,
            budget: JOIN_MEMORY,
        };
        for (base, range, most) in [
            (
                57,
                FieldSize::new(FRONTIER_57, FRONTIER_57 + 100_000_000_000_000),
                1,
            ),
            (
                58,
                FieldSize::new(DENSE_58, DENSE_58 + 1_000_000_000_000_000),
                1,
            ),
            (60, mid_band(60, 10_000_000_000_000_000), 8),
            (62, mid_band(62, 100_000_000_000_000_000), 64),
        ] {
            let field = plan_join(base, &range, lim).expect("the join takes it");
            let n = field.slices.len();
            assert!((1..=most).contains(&n), "b{base}: {n} slices");
            assert_eq!(field.slices[0].start(), range.start());
            assert_eq!(field.slices[n - 1].end(), range.end());
            for w in field.slices.windows(2) {
                assert_eq!(w[0].end(), w[1].start());
            }
            let cap = max_slice_prefixes(&field.fs, lim);
            let block = prefix_block(base, field.fs.base.l, field.fs.jp);
            for sl in &field.slices {
                assert!((sl.last() / block - sl.start() / block) < cap as u128);
            }
            let (plan, ntl) = (field.plan, field.fs.tlay.len());
            assert!(plan.bytes <= lim.budget, "b{base}: {plan:?}");
            assert!(plan.slots >= SLICE_SLOTS, "b{base}: {plan:?}");
            assert!(
                plan.slots * ntl * SURV_PER_PREFIX
                    <= plan.list_cap as usize / 2 + ntl * SURV_PER_PREFIX
            );
            println!(
                "b{base} {:.0e}: {n} slice(s), first {ntl} prefixes, {} per launch, {} MiB",
                range.size() as f64,
                plan.slots,
                plan.bytes >> 20
            );
        }
    }

    /// A small device cuts the same field finer, and still holds each slice.
    #[test]
    fn small_devices_cut_large_fields_finer() {
        let range = FieldSize::new(DENSE_58, DENSE_58 + 1_000_000_000_000_000);
        let desktop = plan_join(58, &range, JoinLimits::for_buffer(2 << 30)).expect("fits");
        let soft = JoinLimits {
            max_binding: 128 << 20,
            budget: 256 << 20,
        };
        let field = plan_join(58, &range, soft).expect("lavapipe holds a slice");
        assert!(field.slices.len() > desktop.slices.len());
        assert!(field.plan.bytes <= soft.budget && field.retry.bytes <= soft.budget);
        let tiny = JoinLimits {
            max_binding: 16 << 20,
            budget: 32 << 20,
        };
        assert_eq!(
            plan_join(58, &range, tiny).err(),
            Some(StrideReason::DeviceTooSmall)
        );
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
