//! A field prepared for the overlap join's GPU stage, and fitted to a device.
//!
//! - [`FieldSetup`]: the host side of a field, which every partition value
//!   starts from on the device: the top layer at depth `t − p` and the bottom
//!   prefixes up to depth `f0`, sorted by digit-sum class.
//! - [`JoinPlan`]: how the field's buffers fit one device. Partitions per
//!   launch and survivor-list lengths are chosen so that each buffer fits the
//!   device's largest binding and all of them a budget ([`JoinLimits`]).
//! - [`plan_join`]: the join's verdict on a field for a device, which
//!   decides the field's route ([`crate::gpu_route::begin_niceonly`]).
//!
//! Nothing here depends on a GPU runtime: the `CubeCL` stage
//! (`crate::cubecl_join`) allocates exactly the buffers [`Footprint`]
//! counts.
#![cfg(any(feature = "cuda", feature = "vulkan", feature = "cubecl"))]
// Only `CubeCL` runs the join so far: without it a field is still planned
// (the route is decided the same way everywhere) but nothing reads the plan.
#![cfg_attr(not(feature = "cubecl"), allow(dead_code))]

use crate::FieldSize;
use crate::gpu_config::n_limbs;
use crate::overlap_join::{Base, JoinParams, join_params_for};
use anyhow::{Result, anyhow, ensure};
use log::warn;
use std::time::Instant;

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

/// The host side of a field: everything that does not depend on the
/// partition value.
#[derive(Clone)]
pub(crate) struct FieldSetup {
    pub base: Base,
    pub b: u32,
    pub s: u128,
    pub e: u128,
    pub jp: JoinParams,
    pub f0: u32,
    pub key_level: bool,
    /// `b^f0`: the width of one top prefix block.
    pub w: u128,
    /// `b^p`: the number of partition values.
    pub nparts: u128,
    pub m1: u32,
    /// Buckets per partition: key values × digit-sum classes.
    pub nb: usize,
    pub plo: u128,
    pub phi: u128,
    /// Top layer at depth `t − p`.
    pub tlay: Vec<(u128, u64)>,
    /// `bpre`: residues mod `b^f0` whose `2·f0` low output digits are
    /// distinct, sorted by digit-sum class mod `b − 1`; `seg[c]..seg[c + 1]`
    /// is class `c`.
    pub bp_r: Vec<u32>,
    pub bp_m: Vec<u64>,
    pub seg: Vec<u32>,
    /// Middle-digit prefilter depth above `k` (0 = off) and `k2 = k + mid`.
    pub mid: u32,
    pub k2: u32,
    /// Certificate floor of every full-width block in the field (monotone in
    /// `P`, so the first full block bounds them all).
    pub full_floor: u32,
    pub secs: f64,
}

impl FieldSetup {
    /// The same field with only the top-layer prefixes `tlay` (a subset of
    /// its own). The tops keep their intervals, so a partition's survivors
    /// split exactly between complementary subsets.
    pub(crate) fn with_tops(&self, tlay: Vec<(u128, u64)>) -> Self {
        let mut sub = self.clone();
        sub.tlay = tlay;
        sub
    }

    /// # Errors
    /// A field or parameters the device stage cannot take.
    #[allow(clippy::many_single_char_names)] // b, s, e, t, k, o, w as in `overlap_join`
    pub fn new(b: u32, s: u128, e: u128, jp: JoinParams) -> Result<Self> {
        let t0 = Instant::now();
        ensure!(s < e, "empty field");
        let base = Base::try_new(b, s, e - 1)
            .ok_or_else(|| anyhow!("[{s}, {e}) crosses a digit-length boundary in base {b}"))?;
        let l = base.l;
        ensure!(
            jp.supported(b, l),
            "{jp:?} not supported at base {b}, L = {l}"
        );
        ensure!(e - 1 < 1 << 96, "device tops need n < 2^96");
        let (t, k, pp) = (jp.t, jp.k, jp.p);
        let f0 = l - t;
        let o = t + k - l;
        let w = base.powu(f0);
        let tlay = base.top_layer(s, e - 1, t - pp, k);
        let mut bpre: Vec<(u64, u64)> = Vec::new();
        base.bot_dfs(0, 0, 0, f0, f0, 0, 0, &mut bpre);
        ensure!(!bpre.is_empty(), "empty bottom list");
        let m1 = b - 1;
        bpre.sort_unstable_by_key(|&(r, _)| (r % u64::from(m1), r));
        let mut seg = vec![0u32; m1 as usize + 1];
        for &(r, _) in &bpre {
            seg[usize::try_from(r % u64::from(m1))? + 1] += 1;
        }
        for c in 0..m1 as usize {
            seg[c + 1] += seg[c];
        }
        // Residues below b^f0 < 2^32 (`JoinParams::supported`).
        let bp_r = bpre
            .iter()
            .map(|&(r, _)| u32::try_from(r))
            .collect::<Result<_, _>>()?;
        let bp_m = bpre.iter().map(|&(_, m)| m).collect();
        let keyspace = base.powu(o - pp);
        // The prefilter tests digits k..k2 too; it needs both powers to have
        // at least k2 digits and P mod b^(k2 - f0) to fit u32.
        let mut mid = 2;
        while mid > 0
            && (k + mid > base.s2 || u64::from(b).pow(k + mid - f0) >= 1 << 32 || k + mid > 12)
        {
            mid -= 1;
        }
        let first_full = s.div_ceil(w);
        let full_floor = if first_full * w + w - 1 < e {
            base.cert_floor(first_full * w, first_full * w + w - 1, k)
        } else {
            0
        };
        Ok(Self {
            mid,
            k2: k + mid,
            full_floor,
            b,
            s,
            e,
            jp,
            f0,
            key_level: o - pp == 1,
            w,
            nparts: base.powu(pp),
            m1,
            nb: keyspace as usize * m1 as usize,
            plo: s / w,
            phi: (e - 1) / w,
            tlay,
            bp_r,
            bp_m,
            seg,
            secs: t0.elapsed().as_secs_f64(),
            base,
        })
    }
}

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
    /// The prefilter runs in the join kernel's flush (production) instead
    /// of as a separate pass (the tests' cross-check).
    pub fused: bool,
    /// The join's own survivor list; fused, none is written (0).
    pub surv_cap: u32,
    /// The list the full check reads: the prefilter's survivors, or the
    /// join's own when there is no prefilter.
    pub list2_cap: u32,
    pub nice_cap: u32,
    /// Every buffer of the field together, the batches in flight included.
    pub bytes: usize,
}

impl JoinPlan {
    /// The layout for `fs` with at most `slots` partitions per launch and
    /// survivor lists of at most `cap` entries, fitted to `lim`: the lists
    /// get at most `list_budget` bytes, then as many slots as fit the rest
    /// of the budget. `None` if one slot with lists of `min_cap` does not.
    pub(crate) fn new(
        fs: &FieldSetup,
        lim: JoinLimits,
        slots: usize,
        fused: bool,
        cap: u32,
        min_cap: u32,
        list_budget: usize,
    ) -> Option<Self> {
        let fused = fused && fs.mid > 0;
        let fp = Footprint::of(fs, NICE_CAP);
        // Fused: the prefilter's list only. Unfused: the join's own list and,
        // with a prefilter, its output, a quarter as long for long lists (the
        // prefilter keeps ~6% at bases 57-64; short lists keep their full
        // length for the weak seeds of small bases).
        let per_entry = if fused || fs.mid == 0 { 8 } else { 16 };
        let cap = u32::try_from(
            (cap as usize)
                .min(lim.max_binding / 8)
                .min(list_budget.saturating_sub(16) / per_entry),
        )
        .unwrap_or(u32::MAX);
        if cap < min_cap.max(1) || fp.largest_fixed > lim.max_binding {
            return None;
        }
        let (surv_cap, list2_cap) = if fused {
            (0, cap)
        } else if fs.mid == 0 {
            (cap, 0)
        } else if cap <= 1 << 22 {
            (cap, cap)
        } else {
            (cap, cap / 4)
        };
        // An unused list is still an 8-byte buffer.
        let lists = (surv_cap as usize * 8).max(8) + (list2_cap as usize * 8).max(8);
        let room = lim.budget.checked_sub(fp.fixed + lists)?;
        let slots = slots
            .min(lim.max_binding / fp.largest_per_slot().max(1))
            .min(room / fp.per_slot().max(1));
        (slots > 0).then(|| JoinPlan {
            slots,
            fused,
            surv_cap,
            list2_cap,
            nice_cap: NICE_CAP,
            bytes: fp.fixed + lists + slots * fp.per_slot(),
        })
    }

    /// The production layout of `fs` on a device with `lim` (the lists get
    /// a quarter of the budget), and the one for re-running a partition that
    /// overflowed it: one slot, and the longest list that fits beside it.
    /// `None` if the device cannot hold one partition with a re-run list of
    /// [`MIN_RETRY_CAP`].
    pub(crate) fn for_field(fs: &FieldSetup, lim: JoinLimits) -> Option<(Self, Self)> {
        let main = Self::new(fs, lim, SLOTS, true, SURV_CAP, 1, lim.budget / 4)?;
        let fp = Footprint::of(fs, NICE_CAP);
        let spare = lim.budget.saturating_sub(fp.fixed + fp.per_slot());
        let retry = Self::new(fs, lim, 1, true, SURV_CAP_RETRY, MIN_RETRY_CAP, spare)?;
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

/// Fields the tests share: production-size fields that the join takes.
#[cfg(test)]
pub(crate) mod test_fields {
    /// A frontier field of base 57.
    pub(crate) const FRONTIER_57: u128 = 28_151_599_893_042_801_193;

    /// Production-size fields at bases 42-64 (gate-sized at 42, 1e14
    /// elsewhere), from the middle of their bands and the frontier.
    pub(crate) const PLAN_FIELDS: &[(u32, u128, u128)] = &[
        (42, 9_682_651_996_416, 10_000_000_000_000),
        (50, 62_082_117_268_529_817, 100_000_000_000_000),
        (57, FRONTIER_57, 100_000_000_000_000),
        (60, 1_366_405_974_057_412_100_454, 100_000_000_000_000),
        (62, 7_997_740_455_941_656_911_841, 100_000_000_000_000),
        (64, 41_242_006_262_957_161_709_568, 100_000_000_000_000),
    ];
}

#[cfg(test)]
mod tests {
    use super::test_fields::{FRONTIER_57, PLAN_FIELDS};
    use super::*;

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
            assert_eq!(
                (plan.slots, plan.fused, plan.list2_cap),
                (SLOTS, true, SURV_CAP),
                "b{base}"
            );
            assert_eq!(
                (retry.slots, retry.list2_cap),
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
        assert!(plan.list2_cap as usize * 8 <= lim.max_binding);
        assert!(retry.list2_cap >= MIN_RETRY_CAP);
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
