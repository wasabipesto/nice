//! GPU stage of the overlap join (see [`crate::overlap_join`] for the idea and
//! the soundness argument), as `CubeCL` kernels: one source for the wgpu
//! (Vulkan, Metal, DX12), CUDA and HIP runtimes.
//!
//! The kernels and the host driver are Dan Stoyell's (wasabipesto/nice#177;
//! `overlap-join/gpu` in `danstoyell/nice_numbers_research` at `ba53f5c`),
//! ported with the device-side top certification as the only path. Per field
//! the host builds the top layer at depth `t − p` and the class-sorted `bpre`
//! list (the bottom list up to depth `f0`), 20-110 ms on the thread that
//! begins the field, and then every partition value goes through the device
//! in batches:
//!
//! 1. [`top_kernel`]: certify each top `P = P0·b^p + v` of the slot's
//!    partition `v` (96-bit endpoints, their powers in 16-bit limbs; the low
//!    endpoint's digits and a carry test, as `Base::cert`), and append it to
//!    its prefix's group: the host sorts the top layer by (key digit, P0 mod
//!    b - 1), and a bucket (key digit, digit-sum class) of any slot is the
//!    union of one group per root, so the tops need no bucketing pass. Where
//!    it pays, [`top_fd_kernel`] does the certifying:
//!    consecutive partitions shift a prefix's interval by a power of the
//!    base, so its square's and cube's digits step from slot to slot by
//!    additions, and only a batch's first slot divides them out.
//! 2. [`bucket_kernel`]: extend every `bpre` entry by `v`'s digits with exact
//!    digit arithmetic mod `b^k`, and list it under every key digit whose two
//!    output digits are distinct and new.
//! 3. [`join_kernel`]: one cube per bucket; the bucket's tops in shared
//!    memory, `ept` entries per thread in registers, a branch-free AND over
//!    32 tops at a time, formed from a digit-major copy of the tops' masks
//!    (one OR per digit of the entry) for large batches of tops. Each pair
//!    that passes it goes through the
//!    middle-digit prefilter as the walk over the AND's bits reaches it:
//!    digits `k..k2` of `n²` and `n³` must be distinct, new, and disjoint
//!    from the top certificate when it provably sits above them. It is
//!    affine in the top's digits there (as on the CPU, `cpu_join`), so the
//!    entry's part is formed once per wave and the top's once per chunk. It
//!    keeps about 6% at bases 57-64, and only those are staged in shared
//!    memory and written out. (Dan's design ran the prefilter as a separate
//!    pass over a full survivor list, which was never small.)
//! 4. [`check_kernel`]: the client's own `candidate_check`.
//!
//! Buffers are sized per field to the device (`join_plan::JoinPlan`): each
//! within the device's largest binding and all of them within a budget. A
//! field too large for that is cut into slices of a bounded top layer and
//! run slice by slice, so any field size fits (`join_plan::plan_join`). A
//! device that cannot hold one partition does not get the join, and its
//! fields stay on the stride pipeline. Every list is bounded and checked:
//! the batches whose survivors overflow are re-run in batches half the
//! size, down to one partition with a longer list, and a partition that
//! overflows that is re-run on halves of its top layer.
#![cfg(feature = "cubecl")]

use crate::cubecl_backend::{LaunchFence, NICEONLY_STRIDE, launch_fence, wide_chunk_for};
use crate::gpu_config::{chunk_constants, chunk_constants_u16, n_limbs};
use crate::gpu_niceonly::{FIELDS_IN_FLIGHT, NiceonlyStats};
use crate::gpu_route::{FieldTicket, Route};
use crate::join_plan::{
    BATCHES_IN_FLIGHT, Footprint, JoinCeiling, JoinField, JoinLimits, JoinPlan, NICE_RECORD_BYTES,
    SortedLayer, TOP_RECORD_BYTES, sorted_layer,
};
use crate::overlap_join::{FieldSetup, JoinTelemetry};
use crate::progress::FieldProgress;
use crate::{FieldSize, NiceNumberSimple};
use anyhow::{Result, anyhow, ensure};
use cubecl::prelude::*;
use cubecl::server::Handle;
use log::{debug, warn};
use std::collections::VecDeque;
use std::sync::Arc;
use std::sync::mpsc::{Receiver, SyncSender, sync_channel};
use web_time::Instant;

// The output list's layout is shared with the stride kernels' check.
const _: () = assert!(NICEONLY_STRIDE as usize * 4 == NICE_RECORD_BYTES);
const _: () = assert!(TOP_REC_WORDS as usize * 4 == TOP_RECORD_BYTES);

/// Threads per cube for every kernel here (the client's `WORKGROUP_SIZE`).
pub const JOIN_WG: u32 = 256;
/// Tops held in shared memory per pass over a bucket's entries.
pub const TOP_CAP: u32 = 256;
/// Words per top in `top_rec`, one 32-byte record per top: P (lo, hi), the
/// certificate mask (lo, hi), the bottom residues' range (lo, hi), then
/// P mod b^(k2 - f0) and the seed flag. The join reads a bucket's tops
/// record by record, and the check reads one by its index, so a top's words
/// are one sector of one array rather than one from each of three.
pub const TOP_REC_WORDS: u32 = 8;
/// Top batches of at least this many tops get [`join_kernel`]'s transposed
/// AND; smaller ones test each entry against each top, which is cheaper than
/// the digit-major copy on a few tops (the fastest of 128, 192 and 256 on
/// fields of 1e14-1e17 at bases 58-64).
pub const TRANSPOSE_MIN: u32 = 192;
/// Chunks of 32 tops whose pass masks the transposed AND forms per pass over
/// an entry's digits: 4 accumulators per entry slot (8 loaded more words
/// than a partial batch needs and more registers; 1 re-walks the digits per
/// chunk).
const AND_GROUP: u32 = 4;
// The transposed copy maps thread u to chunk u / 32 of the batch, and the
// groups tile the batch's chunks.
const _: () = assert!(
    JOIN_WG == TOP_CAP && TOP_CAP.is_multiple_of(32 * AND_GROUP) && TRANSPOSE_MIN <= TOP_CAP
);
/// Prefilter survivors staged in shared memory before a flush to the global
/// list.
pub const SURV_SH_CAP: u32 = 1024;
/// Staged-survivor level that triggers a flush at the next barrier.
pub const SURV_FLUSH: u32 = 512;
/// Sentinel mask of an extended entry that failed its fixed positions.
pub const DEAD_MASK: u64 = 0xFFFF_FFFF_FFFF_FFFF;
/// Bottom entries each thread holds in registers in [`join_kernel`]: a wave
/// covers `ENTRIES_PER_THREAD · JOIN_WG` = 512 entries of a bucket. Since
/// the field prune (`FieldSetup::fc`) a bucket holds 3-4x fewer entries,
/// about 280-420 on fields of 1e16 at bases 58-64 and fewer at 1e14, so
/// waves of 1,024 or 1,536 entries (4 below base 62 and 6 from it before)
/// ran mostly empty lanes. Two are the fastest at every base measured.
const ENTRIES_PER_THREAD: u32 = 2;
/// Cubes for the prefilter and check kernels (grid-stride loops).
const CHECK_CUBES: u32 = 1024;
/// How deep the halving of an overflowing partition's top layer may go: a
/// layer of up to 2^32 prefixes reaches single prefixes well within it.
const MAX_SPLITS: u32 = 32;
/// Kernels a batch launches (top, bucket, join, check), each counting its
/// finished cubes.
const STAGES: usize = 4;
/// The most digit-sum roots of any base the join takes (8, at bases 46, 50
/// and 64): [`join_kernel`] holds a bucket's group segments in shared memory.
const MAX_ROOTS: u32 = 8;
/// Units [`top_fd_kernel`] should have: a smaller top layer splits a batch's slots into groups.
const TOP_FD_MIN_UNITS: u32 = 1 << 17;
/// FD at power-of-two bases too: no, `top_kernel`'s division is shifts there and FD measured slower.
const TOP_FD_AT_POW2_BASES: bool = false;

/// Whether [`top_fd_kernel`]'s append ranks a plane's kept tops with one
/// ballot (`top_slot`): only on devices whose planes have at most 32 lanes.
/// `CubeCL`'s C++ backends (CUDA, HIP, Metal) return a ballot's first 32
/// bits, and HIP's wave64 devices (CDNA) and Metal on AMD GPUs have 64-lane
/// planes, where a ballot would miss the upper half; those keep the sum and
/// exclusive scan.
pub(crate) const fn ballot_append(plane_size_max: u32) -> bool {
    plane_size_max <= 32
}

// ---------------------------------------------------------------------------
// Device kernels
// ---------------------------------------------------------------------------

pub use kernels::{bucket_kernel, check_kernel, join_kernel, top_fd_kernel, top_kernel};

/// The kernels. The cube macro evaluates `comptime!` expressions host-side,
/// where fn-level allows do not reach, and the bodies are ported integer
/// arithmetic with deliberate casts, so the lints below are allowed for this
/// module (as for `cubecl_backend`'s kernels). The host side has none.
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_lossless,
    clippy::used_underscore_binding,
    clippy::too_many_arguments,
    clippy::too_many_lines,
    clippy::similar_names,
    clippy::many_single_char_names,
    clippy::collapsible_if,
    clippy::fn_params_excessive_bools,
    clippy::needless_range_loop,
    clippy::unreadable_literal,
    // Shared memory is passed by reference, the comptime `if` must stay
    // separate from the runtime one, and the integer helpers clippy suggests
    // (midpoint, is_power_of_two, is_multiple_of) are not cube intrinsics.
    clippy::trivially_copy_pass_by_ref,
    clippy::collapsible_else_if,
    clippy::manual_midpoint,
    clippy::manual_is_power_of_two,
    clippy::manual_is_multiple_of
)]
mod kernels {
    use super::{
        AND_GROUP, DEAD_MASK, JOIN_WG, MAX_ROOTS, SURV_FLUSH, SURV_SH_CAP, TOP_CAP, TOP_REC_WORDS,
        TRANSPOSE_MIN,
    };
    use crate::cubecl_backend::candidate_check;
    use crate::cubecl_detailed_fd::fd_add_top;
    use cubecl::prelude::*;

    /// Count the cube in `ran[stage]` once every unit of it is done (see
    /// `JoinDevice::ensure_ran`). Every unit must reach it.
    #[cube]
    fn cube_done(ran: &mut Array<Atomic<u32>>, #[comptime] stage: u32) {
        sync_cube();
        if UNIT_POS_X == 0u32 {
            ran[stage as usize].fetch_add(1u32);
        }
    }

    /// Step 1+ (v3): extension fused with bucketization. One cube per
    /// (class c, slot): its bpre segment is extended by the slot's partition
    /// digits (the extended entries are written to `ext_m`/`ext_q`, where the
    /// join kernel reads them), and each surviving entry's two
    /// key-position digits are stepped through every key value d incrementally
    /// (+s2, +s3 mod b); the entry's index goes onto the (slot, d, c) list for
    /// every d where they are distinct and new. Every digit is also tested
    /// against the field's certificate `fc_lo | fc_hi << 32`, which every
    /// top's mask contains: an entry that meets it would fail every AND.
    /// List layout per slot:
    /// `slot*b*nbp + b*seg[c] + d*len(c) + i`; counts at `(slot*b + d)*m1 + c`.
    ///
    /// The digit columns run to `k2`, past the entry's `k` digits, for the
    /// prefilter's terms in `ext_q` (two words per entry; see
    /// [`join_kernel`]). With `mid = k2 - k`, `M = b^(mid + 1)` (at most
    /// `64^3 = 2^18`) and `R'` the entry with key digit 0:
    /// - key level: `⌊R'²/b^(k-1)⌋ mod M | c3 << 18` and
    ///   `⌊R'³/b^(k-1)⌋ mod M | (c3 >> 14) << 18`, `c3 = 3R'² mod M`: the
    ///   key digit's output digits and the prefilter's carries for every key
    ///   digit (`cpu_join`'s `Ext`);
    /// - otherwise (`R'` is the whole entry R): `⌊R²/b^k⌋ | ⌊R³/b^k⌋ << 16`
    ///   and `2R | 3R² << 16`, each mod `b^mid` (with `R mod b^mid` for `2R`
    ///   when `k < mid`).
    #[cube(launch_unchecked)]
    pub fn bucket_kernel(
        bp_r: &Array<u32>,
        bp_m: &Array<u32>,
        seg: &Array<u32>,
        vs: &Array<u32>,
        ext_m: &mut Array<u32>,
        ext_q: &mut Array<u32>,
        lists: &mut Array<u32>,
        counts: &mut Array<u32>,
        ran: &mut Array<Atomic<u32>>,
        nbp: u32,
        fc_lo: u32,
        fc_hi: u32,
        #[comptime] base: u32,
        #[comptime] f0: u32,
        #[comptime] pp: u32,
        #[comptime] k: u32,
        #[comptime] k2: u32,
        #[comptime] key_level: bool,
        #[comptime] key_at_zero: bool,
    ) {
        let m1 = comptime!(base - 1);
        let mid = comptime!(k2 - k);
        let bm = comptime!(base.pow(k2 - k));
        let big_m = comptime!(base.pow(k2 - k + 1));
        let fc = (u64::cast_from(fc_hi) << 32u64) | u64::cast_from(fc_lo);
        let nkeys = comptime!(if key_level { base } else { 1 });
        let s_cnt = SharedMemory::<Atomic<u32>>::new(64usize);
        let c = CUBE_POS_X;
        let slot = CUBE_POS_Y;
        if UNIT_POS_X < 64u32 {
            s_cnt[UNIT_POS_X as usize].store(0u32);
        }
        sync_cube();
        let cs = seg[c as usize];
        let ce = seg[(c + 1u32) as usize];
        let clen = ce - cs;
        let lbase = slot * base * nbp + base * cs;
        let v = vs[slot as usize];
        let mut i = cs + UNIT_POS_X;
        while i < ce {
            let r0 = bp_r[i as usize];
            let mut m = (u64::cast_from(bp_m[(2u32 * i + 1u32) as usize]) << 32u64)
                | u64::cast_from(bp_m[(2u32 * i) as usize]);
            let mut d = Array::<u32>::new(k2 as usize);
            let mut x = r0;
            #[unroll]
            for j in 0..f0 {
                d[j as usize] = x % base;
                x /= base;
            }
            let mut y = v;
            #[unroll]
            for j in f0..comptime!(f0 + pp) {
                d[j as usize] = y % base;
                y /= base;
            }
            #[unroll]
            for j in comptime!(f0 + pp)..k2 {
                d[j as usize] = 0u32;
            }
            let mut sq = Array::<u32>::new(k2 as usize);
            let mut carry = 0u32;
            #[unroll]
            for pos in 0..k2 {
                let mut col = carry;
                #[unroll]
                for i2 in 0..comptime!(pos + 1) {
                    col += d[i2 as usize] * d[comptime!(pos - i2) as usize];
                }
                sq[pos as usize] = col % base;
                carry = col / base;
            }
            let mut cu = Array::<u32>::new(k2 as usize);
            carry = 0u32;
            #[unroll]
            for pos in 0..k2 {
                let mut col = carry;
                #[unroll]
                for i2 in 0..comptime!(pos + 1) {
                    col += sq[i2 as usize] * d[comptime!(pos - i2) as usize];
                }
                cu[pos as usize] = col % base;
                carry = col / base;
            }
            let mut ok = true;
            #[unroll]
            for j in f0..comptime!(f0 + pp) {
                let g2 = sq[j as usize];
                let g3 = cu[j as usize];
                let bb = (1u64 << u64::cast_from(g2)) | (1u64 << u64::cast_from(g3));
                if g2 == g3 || ((m | fc) & bb) != 0u64 {
                    ok = false;
                }
                m |= bb;
            }
            if !ok {
                m = DEAD_MASK;
            }
            // What the key digit must miss: the entry's digits and the
            // field's.
            let mf = m | fc;
            let idx = slot * nbp + i;
            ext_m[(2u32 * idx) as usize] = u32::cast_from(m);
            ext_m[(2u32 * idx + 1u32) as usize] = u32::cast_from(m >> 32u64);
            if key_level {
                // Digits k-1..k2 of R'^2 and R'^3 as numbers, and R'^2 mod M
                // from its low digits (k2 >= mid + 1 of them).
                let mut q2 = 0u32;
                let mut q3 = 0u32;
                #[unroll]
                for jj in 0..comptime!(mid + 1) {
                    let j = comptime!(k2 - 1 - jj);
                    q2 = q2 * base + sq[j as usize];
                    q3 = q3 * base + cu[j as usize];
                }
                let mut lo2 = 0u32;
                #[unroll]
                for jj in 0..comptime!(mid + 1) {
                    lo2 = lo2 * base + sq[comptime!(mid - jj) as usize];
                }
                let c3 = (3u32 * lo2) % big_m;
                ext_q[(2u32 * idx) as usize] = q2 | ((c3 & 0x3FFFu32) << 18u32);
                ext_q[(2u32 * idx + 1u32) as usize] = q3 | ((c3 >> 14u32) << 18u32);
            } else {
                // Digits k..k2 of R^2 and R^3 as numbers, and R mod b^mid.
                let mut a2 = 0u32;
                let mut a3 = 0u32;
                #[unroll]
                for jj in 0..mid {
                    let j = comptime!(k2 - 1 - jj);
                    a2 = a2 * base + sq[j as usize];
                    a3 = a3 * base + cu[j as usize];
                }
                let mut rho = 0u32;
                #[unroll]
                for jj in 0..mid {
                    rho = rho * base + d[comptime!(mid - 1 - jj) as usize];
                }
                let p3 = (3u32 * ((rho * rho) % bm)) % bm;
                let mut p2 = (2u32 * rho) % bm;
                if comptime!(k < mid) {
                    p2 = rho;
                }
                ext_q[(2u32 * idx) as usize] = a2 | (a3 << 16u32);
                ext_q[(2u32 * idx + 1u32) as usize] = p2 | (p3 << 16u32);
            }
            if key_level {
                let u2 = sq[comptime!(k - 1) as usize];
                let u3 = cu[comptime!(k - 1) as usize];
                let d0 = d[0];
                let s2 = (2u32 * d0) % base;
                let s3 = (3u32 * ((d0 * d0) % base)) % base;
                if ok {
                    // Key digit dk: (u2 + dk s2, u3 + dk s3) mod b, stepped.
                    let mut g2 = u2;
                    let mut g3 = u3;
                    let mut dk = 0u32;
                    while dk < nkeys {
                        let mut a2 = g2;
                        let mut a3 = g3;
                        if key_at_zero {
                            a2 = (dk * dk) % base;
                            a3 = (a2 * dk) % base;
                        }
                        let bb = (1u64 << u64::cast_from(a2)) | (1u64 << u64::cast_from(a3));
                        if a2 != a3 && (mf & bb) == 0u64 {
                            let at = s_cnt[dk as usize].fetch_add(1u32);
                            lists[(lbase + dk * clen + at) as usize] = i;
                        }
                        g2 += s2;
                        if g2 >= base {
                            g2 -= base;
                        }
                        g3 += s3;
                        if g3 >= base {
                            g3 -= base;
                        }
                        dk += 1u32;
                    }
                }
            } else {
                if ok {
                    let at = s_cnt[0].fetch_add(1u32);
                    lists[(lbase + at) as usize] = i;
                }
            }
            i += CUBE_DIM_X;
        }
        sync_cube();
        if UNIT_POS_X < nkeys {
            counts[((slot * nkeys + UNIT_POS_X) * m1 + c) as usize] =
                s_cnt[UNIT_POS_X as usize].load();
        }
        cube_done(ran, 1u32);
    }

    /// Step 2 (v3): one cube per work item, a (slot, d, c) bucket: its entries
    /// from the list built by [`bucket_kernel`], its tops from `nroots`
    /// group segments of `top_rec` (read in order, see the work item); `ept`
    /// entries per thread per wave, every thread runs every wave (uniform).
    /// A branch-free AND over 32 tops at a time gives each entry a bitmask of
    /// the tops it passes, and each lane walks its set bits.
    ///
    /// A pair passes the AND iff the entry's and the top's digit masks are
    /// disjoint, i.e. iff no digit of the entry is a digit of the top. So for
    /// a batch of at least [`TRANSPOSE_MIN`] tops the kernel first transposes
    /// the tops' masks into `t_tr`, one word per (chunk of 32 tops, digit)
    /// with bit i set iff the chunk's top i has the digit, and an entry's
    /// pass mask for a chunk is the complement of the OR of the words of its
    /// digits: about 12 OR-loads per entry per chunk (the entry's 2k digits)
    /// instead of 32 two-word tests, done for `AND_GROUP` chunks per pass
    /// over the digits. Smaller batches test each top.
    ///
    /// A pair inside the top's range is a join survivor
    /// (counted in `and_count`) and goes through the middle-digit prefilter
    /// there and then ([`walk_pf_lean`], or [`walk_pf`] for small test
    /// shapes); only the pairs that pass it, about 6% at bases 57-64, are
    /// staged in shared memory, and the stage goes to the list once it is
    /// half full, checked after every batch of tops. The range test is done per
    /// chunk, not per pair: only a clipped top (a block at the field's ends,
    /// whose range is not the whole `[0, w)`) can reject a residue, so the
    /// batch load marks the clipped tops in a word per chunk (`s_cw`), and a
    /// chunk with one clears their out-of-range pass bits before the walk.
    /// Every bit walked is then a join survivor, counted by popcount.
    ///
    /// The prefilter tests n's output digits at positions `k..k2` (`mid =
    /// k2 - k` of them). With `n = R + D·b^k` (`R` the entry's `k` digits,
    /// `D` the top's next `mid`), they are the digits of
    /// `(⌊R²/b^k⌋ + 2R·D) mod b^mid` and `(⌊R³/b^k⌋ + 3R²·D) mod b^mid`, as
    /// in `cpu_join`'s prefilter (plus terms in `b^k` that vanish unless
    /// `k < mid`, which only small test shapes have). So the work splits by
    /// what it depends on: the wave load forms each entry's part once per
    /// 256 tops, the batch load the top's, and a pair costs two
    /// multiply-adds and the digit tests. At `mid = 2` (every production
    /// shape) the reduction mod `b²` and the two digits come from
    /// fixed-point arithmetic ([`walk_pf_lean`]): the entry holds `A·C` and
    /// `P` for each power (`A` the carry, `P` = `2R` or `3R²`, `C =
    /// ceil(2^32/b²)`), the top `ceil(D·2^32/b²)` (`t_dm`) and its
    /// certificate gated by its seed flag (`t_glo`, `t_ghi`), and a pair
    /// costs about 25 instructions. Elsewhere `ac`/`pc` pack the carries and
    /// `2R | 3R² << 16`, `t_dm` is `D | seed << 16`, and [`walk_pf`] reduces
    /// by division. Rebuilding all `k2` digit columns of n² and n³ for every
    /// pair instead, as this kernel used to when it flushed the stage, was
    /// about 290.
    ///
    /// The digit masks are held bit-reversed in this kernel when the lean
    /// prefilter runs (`lean`: digit d at bit 63 - d), entries' and tops'
    /// alike, and so `t_tr`: the AND is the same disjointness test under any
    /// labelling of the bits, and the prefilter reads digit g as the sign bit
    /// of `hi32(seen << g)`. The global buffers (`ext_m`, `top_rec`) and every
    /// other kernel keep digit d at bit d.
    #[cube(launch_unchecked)]
    pub fn join_kernel(
        ext_m: &Array<u32>,
        ext_q: &Array<u32>,
        bp_r: &Array<u32>,
        seg: &Array<u32>,
        lists: &Array<u32>,
        counts: &Array<u32>,
        vs: &Array<u32>,
        roots: &Array<u32>,
        goff: &Array<u32>,
        grp_cnt: &Array<u32>,
        top_rec: &Array<u32>,
        surv: &mut Array<u32>,
        surv_count: &mut Array<Atomic<u32>>,
        and_count: &mut Array<Atomic<u32>>,
        ran: &mut Array<Atomic<u32>>,
        nwork: u32,
        nbp: u32,
        surv_cap: u32,
        w_f0: u32,
        ntlay: u32,
        #[comptime] base: u32,
        #[comptime] key_level: bool,
        #[comptime] nroots: u32,
        #[comptime] ept: u32,
        #[comptime] f0: u32,
        #[comptime] k: u32,
        #[comptime] k2: u32,
        #[comptime] walk_sync: bool,
    ) {
        let m1 = comptime!(base - 1);
        let nkeys = comptime!(if key_level { base } else { 1 });
        let wave = comptime!(ept * JOIN_WG);
        let mid = comptime!(k2 - k);
        let bm = comptime!(base.pow(k2 - k));
        // M = b^(mid + 1): the key digit's output digit and the carries above.
        let big_m = comptime!(base.pow(k2 - k + 1));
        // n's digits k..k2 from the top's P mod b^(k2 - f0).
        let dm_div = comptime!(base.pow(k - f0));
        // b^(k-1) and b^(2(k-1)) mod M, and b^(k-1) mod b^mid: the key digit's
        // terms beyond the affine ones (`cpu_join`'s e1, e2 and kb). All zero
        // at k >= mid + 2, the production shape's k = 6 among them.
        let e1 = comptime!(if k < mid + 2 { base.pow(k - 1) } else { 0 });
        let e2 = comptime!(if 2 * k < mid + 3 {
            base.pow(2 * k - 2)
        } else {
            0
        });
        let kb = comptime!(if k <= mid { base.pow(k - 1) } else { 0 });
        // R' mod M (R' the entry with key digit 0, so R = R' + d·b^(k-1)) is
        // r0 mod M when the bottom list's f0 digits reach M (the production
        // shape: f0 = 3 = mid + 1); otherwise it takes the slot's partition
        // digits below M too, which every top's P mod b^(k2 - f0) begins with.
        let v_digits = comptime!(if key_level && f0 <= mid {
            (k - 1 - f0).min(mid + 1 - f0)
        } else {
            0
        });
        // The fixed-point prefilter ([`walk_pf_lean`]) at mid = 2, k >= 2:
        // C = ceil(2^32/b²), and 2^32 = Q·b² + R.
        let lean = comptime!(mid == 2 && k >= 2);
        let fx_c = comptime!(fx_consts(base).0);
        let fx_q = comptime!(fx_consts(base).1);
        let fx_r = comptime!(fx_consts(base).2);
        let bsq = comptime!(base * base);
        let mut t_lo = SharedMemory::<u32>::new(comptime!(TOP_CAP as usize));
        let mut t_hi = SharedMemory::<u32>::new(comptime!(TOP_CAP as usize));
        let mut t_rlo = SharedMemory::<u32>::new(comptime!(TOP_CAP as usize));
        let mut t_rhi = SharedMemory::<u32>::new(comptime!(TOP_CAP as usize));
        let mut t_id = SharedMemory::<u32>::new(comptime!(TOP_CAP as usize));
        let mut t_dm = SharedMemory::<u32>::new(comptime!(TOP_CAP as usize));
        // The tops' certificates where their seed flag is set, else 0.
        let mut t_glo =
            SharedMemory::<u32>::new(comptime!(if lean { TOP_CAP as usize } else { 1 }));
        let mut t_ghi =
            SharedMemory::<u32>::new(comptime!(if lean { TOP_CAP as usize } else { 1 }));
        // Per chunk of 32 tops: bit i set iff top i is clipped.
        let s_cw = SharedMemory::<Atomic<u32>>::new(comptime!((TOP_CAP / 32) as usize));
        // The batch's masks digit-major: bit i of `t_tr[c·64 + d]` is set iff
        // top 32c + i has the digit at mask bit d (digit d, or digit 63 - d
        // where `lean` holds the masks bit-reversed, see the doc above).
        let mut t_tr = SharedMemory::<u32>::new(comptime!((TOP_CAP / 32 * 64) as usize));
        let mut s_t = SharedMemory::<u32>::new(comptime!(SURV_SH_CAP as usize));
        let mut s_r = SharedMemory::<u32>::new(comptime!(SURV_SH_CAP as usize));
        let mut s_cnt = SharedMemory::<Atomic<u32>>::new(1usize);
        let mut s_base = SharedMemory::<u32>::new(1usize);
        // The work item's tops (see the work item below): segment r holds
        // its tops s_cum[r]..s_cum[r + 1], top q of it at record s_db[r] + q.
        let mut s_cum = SharedMemory::<u32>::new(comptime!(MAX_ROOTS as usize + 1));
        let mut s_db = SharedMemory::<u32>::new(comptime!(MAX_ROOTS as usize));
        // The work item's list offset, entry base, entries and key digit.
        let mut s_hd = SharedMemory::<u32>::new(4usize);
        // This thread's join survivors (pairs that pass the AND in range).
        let mut nand = 0u32;

        if UNIT_POS_X == 0u32 {
            s_cnt[0].store(0u32);
        }
        if UNIT_POS_X < comptime!(TOP_CAP / 32) {
            s_cw[UNIT_POS_X as usize].store(0u32);
        }
        sync_cube();

        // Work item w is bucket (key dkey, class cls) of slot w / nb. Its tops
        // are the kept tops of `nroots` groups of the top layer, one per
        // digit-sum root r: group (dkey, (root_r - cls - v) mod (b-1)), whose
        // slot-v tops [`top_kernel`] appended into
        // `top_rec[slot·ntlay + goff[g]..][..grp_cnt[slot·nb + g]]`. The
        // batches take them in that order, across the groups' boundaries.
        //
        // The item's entry list (its offset, the slot's extended entries, its
        // length) and key digit go to shared memory with the segments, and
        // each wave reads them there: held in registers they would be live
        // across the whole batch loop, and the kernel sits at the 48-register
        // line.
        let nbk = comptime!(nkeys * m1);
        let mut w = CUBE_POS_X;
        while w < nwork {
            let slot = w / nbk;
            let bx = w - slot * nbk;
            let dkey = bx / m1;
            let cls = bx - dkey * m1;
            let v = vs[slot as usize];
            // Thread r < nroots takes segment r, thread 0 places them. Every
            // thread is past the previous item's reads of the table (the
            // first barrier).
            sync_cube();
            if UNIT_POS_X < nroots {
                let c0 = (roots[UNIT_POS_X as usize] + comptime!(2 * m1) - cls - v % m1) % m1;
                let g = dkey * m1 + c0;
                s_db[UNIT_POS_X as usize] = slot * ntlay + goff[g as usize];
                s_cum[(UNIT_POS_X + 1u32) as usize] = grp_cnt[(slot * nbk + g) as usize];
            }
            if UNIT_POS_X == 0u32 {
                let cs = seg[cls as usize];
                let clen = seg[(cls + 1u32) as usize] - cs;
                s_hd[0] = slot * base * nbp + base * cs + dkey * clen;
                s_hd[1] = slot * nbp;
                s_hd[2] = counts[((slot * nkeys + dkey) * m1 + cls) as usize];
                s_hd[3] = dkey;
            }
            sync_cube();
            if UNIT_POS_X == 0u32 {
                s_cum[0] = 0u32;
                #[unroll]
                for r in 0..nroots {
                    let c = s_cum[r as usize];
                    s_cum[comptime!(r + 1) as usize] += c;
                    s_db[r as usize] -= c;
                }
            }
            sync_cube();
            let mut v_part = 0u32;
            if comptime!(v_digits > 0) {
                // The slot's partition digits below M (`v_digits <= pp`).
                v_part = (v % comptime!(base.pow(v_digits))) * comptime!(base.pow(f0));
            }

            let mut tc = 0u32;
            while tc < s_cum[nroots as usize] {
                let mut nt = s_cum[nroots as usize] - tc;
                if nt > TOP_CAP {
                    nt = TOP_CAP;
                }
                if UNIT_POS_X < nt {
                    // Top q of the bucket: in the last segment that starts
                    // at or before it (the ones between are empty).
                    let q = tc + UNIT_POS_X;
                    let mut sr = 0u32;
                    #[unroll]
                    for j in 1..nroots {
                        sr = select(s_cum[j as usize] <= q, j, sr);
                    }
                    let t = s_db[sr as usize] + q;
                    // Words 2-7 of the top's record ([`TOP_REC_WORDS`]).
                    let r8 = TOP_REC_WORDS * t;
                    let mut lo = top_rec[(r8 + 2u32) as usize];
                    let mut hi = top_rec[(r8 + 3u32) as usize];
                    if comptime!(lean) {
                        let l0 = lo;
                        lo = hi.reverse_bits();
                        hi = l0.reverse_bits();
                    }
                    t_lo[UNIT_POS_X as usize] = lo;
                    t_hi[UNIT_POS_X as usize] = hi;
                    let rlo = top_rec[(r8 + 4u32) as usize];
                    let rhi = top_rec[(r8 + 5u32) as usize];
                    t_rlo[UNIT_POS_X as usize] = rlo;
                    t_rhi[UNIT_POS_X as usize] = rhi;
                    if rlo != 0u32 || rhi != w_f0 {
                        s_cw[(UNIT_POS_X / 32u32) as usize].fetch_or(1u32 << (UNIT_POS_X % 32u32));
                    }
                    t_id[UNIT_POS_X as usize] = t;
                    // D (below b^mid <= 2^12) and the seed flag.
                    let dm = top_rec[(r8 + 6u32) as usize] / dm_div;
                    let seed = top_rec[(r8 + 7u32) as usize];
                    if comptime!(lean) {
                        // ceil(D·2^32/b²) = D·Q + ceil(D·R/b²).
                        t_dm[UNIT_POS_X as usize] =
                            dm * fx_q + (dm * fx_r + comptime!(bsq - 1)) / bsq;
                        t_glo[UNIT_POS_X as usize] = select(seed != 0u32, lo, 0u32);
                        t_ghi[UNIT_POS_X as usize] = select(seed != 0u32, hi, 0u32);
                    } else {
                        t_dm[UNIT_POS_X as usize] = dm | (seed << 16u32);
                    }
                }
                sync_cube();
                if nt >= TRANSPOSE_MIN {
                    // Thread u writes chunk u / 32's words for digits u % 32
                    // and u % 32 + 32, over the chunk's 32 tops unrolled (the
                    // loads pipeline); bits past the batch's last top are
                    // clear, and chunks past it are zero.
                    let cc = UNIT_POS_X / 32u32;
                    let dl = UNIT_POS_X % 32u32;
                    let t0 = cc * 32u32;
                    let mut w0 = 0u32;
                    let mut w1 = 0u32;
                    if t0 < nt {
                        #[unroll]
                        for i in 0..32u32 {
                            let lo = t_lo[(t0 + i) as usize];
                            let hi = t_hi[(t0 + i) as usize];
                            w0 |= ((lo >> dl) & 1u32) << i;
                            w1 |= ((hi >> dl) & 1u32) << i;
                        }
                        let n = nt - t0;
                        if n < 32u32 {
                            let vm = (1u32 << n) - 1u32;
                            w0 &= vm;
                            w1 &= vm;
                        }
                    }
                    t_tr[(cc * 64u32 + dl) as usize] = w0;
                    t_tr[(cc * 64u32 + 32u32 + dl) as usize] = w1;
                }
                sync_cube();

                let mut ws = 0u32;
                while ws < s_hd[2] {
                    let lbase = s_hd[0];
                    let ebase = s_hd[1];
                    let ne = s_hd[2];
                    let dkey = s_hd[3];
                    let mut mlo = Array::<u32>::new(ept as usize);
                    let mut mhi = Array::<u32>::new(ept as usize);
                    let mut rr = Array::<u32>::new(ept as usize);
                    // Per-slot validity: an empty slot must never pass, whatever
                    // the top's mask (a top may certify nothing at small t).
                    let mut vv = Array::<u32>::new(ept as usize);
                    // The entry's part of the prefilter (see above).
                    let mut ac = Array::<u32>::new(ept as usize);
                    let mut pc = Array::<u32>::new(ept as usize);
                    // Lean: `ac`/`pc` the n² terms (A·C, P), `ac3`/`pc3` n³'s.
                    let mut ac3 = Array::<u32>::new(ept as usize);
                    let mut pc3 = Array::<u32>::new(ept as usize);
                    #[unroll]
                    for j in 0..ept {
                        let sl = ws + UNIT_POS_X + comptime!(j * JOIN_WG);
                        mlo[j as usize] = 0xFFFF_FFFFu32;
                        mhi[j as usize] = 0xFFFF_FFFFu32;
                        rr[j as usize] = 0u32;
                        vv[j as usize] = 0u32;
                        ac[j as usize] = 0u32;
                        pc[j as usize] = 0u32;
                        ac3[j as usize] = 0u32;
                        pc3[j as usize] = 0u32;
                        if sl < ne {
                            vv[j as usize] = 0xFFFF_FFFFu32;
                            let e = lists[(lbase + sl) as usize];
                            let gi = ebase + e;
                            let r0 = bp_r[e as usize];
                            let mut m = (u64::cast_from(ext_m[(2u32 * gi + 1u32) as usize])
                                << 32u64)
                                | u64::cast_from(ext_m[(2u32 * gi) as usize]);
                            let q0 = ext_q[(2u32 * gi) as usize];
                            let q1 = ext_q[(2u32 * gi + 1u32) as usize];
                            if key_level {
                                // t2, t3 = ⌊R²/b^(k-1)⌋, ⌊R³/b^(k-1)⌋ mod M for
                                // this key digit: q + d·c (+ d²·e1, ...) from
                                // the bucket kernel's terms at d = 0 (`cpu_join`'s
                                // `advance`). The low digit is the key digit's
                                // output digit, the two above the carries.
                                let mut sigma = r0;
                                if comptime!(v_digits > 0) {
                                    sigma += v_part;
                                }
                                if comptime!(f0 > mid + 1) {
                                    sigma = r0 % big_m;
                                }
                                let c3 = (q0 >> 18u32) | ((q1 >> 18u32) << 14u32);
                                let mut t2 = (q0 & 0x3FFFFu32) + dkey * (2u32 * sigma);
                                let mut t3 = (q1 & 0x3FFFFu32) + dkey * c3;
                                if comptime!(e1 != 0) {
                                    let dd = dkey * dkey;
                                    t2 += (dd % comptime!(big_m / e1.max(1))) * e1;
                                    let c3b = ((3u32 * sigma) % comptime!(big_m / e1.max(1))) * e1;
                                    t3 += (dd * c3b) % big_m;
                                    if comptime!(e2 != 0) {
                                        t3 += ((dd * dkey) % comptime!(big_m / e2.max(1))) * e2;
                                    }
                                }
                                t2 %= big_m;
                                t3 %= big_m;
                                let g2 = t2 % base;
                                let g3 = t3 % base;
                                m |= (1u64 << u64::cast_from(g2)) | (1u64 << u64::cast_from(g3));
                                if comptime!(lean) {
                                    ac[j as usize] = (t2 / base) * fx_c;
                                    ac3[j as usize] = (t3 / base) * fx_c;
                                } else {
                                    ac[j as usize] = (t2 / base) | ((t3 / base) << 16u32);
                                }
                                // 2R and 3R² mod b^mid (R mod b^mid itself
                                // when k < mid).
                                let mut rho = sigma % bm;
                                if comptime!(kb != 0) {
                                    rho = (rho + dkey * kb) % bm;
                                }
                                let p3 = (3u32 * ((rho * rho) % bm)) % bm;
                                let mut p2 = (2u32 * rho) % bm;
                                if comptime!(k < mid) {
                                    p2 = rho;
                                }
                                if comptime!(lean) {
                                    pc[j as usize] = p2;
                                    pc3[j as usize] = p3;
                                } else {
                                    pc[j as usize] = p2 | (p3 << 16u32);
                                }
                            } else if comptime!(lean) {
                                ac[j as usize] = (q0 & 0xFFFFu32) * fx_c;
                                ac3[j as usize] = (q0 >> 16u32) * fx_c;
                                pc[j as usize] = q1 & 0xFFFFu32;
                                pc3[j as usize] = q1 >> 16u32;
                            } else {
                                ac[j as usize] = q0;
                                pc[j as usize] = q1;
                            }
                            if comptime!(lean) {
                                mlo[j as usize] = u32::cast_from(m >> 32u64).reverse_bits();
                                mhi[j as usize] = u32::cast_from(m).reverse_bits();
                            } else {
                                mlo[j as usize] = u32::cast_from(m);
                                mhi[j as usize] = u32::cast_from(m >> 32u64);
                            }
                            rr[j as usize] = r0;
                        }
                    }
                    // Branch-free AND over a chunk of up to 32 tops into a pass
                    // bitmask per entry, then only the (rare) set bits are
                    // walked: the survivor path diverges once per chunk, not
                    // once per top.
                    // The transposed AND's ORs of the next AND_GROUP chunks,
                    // per entry slot (chunk-major).
                    let ag = comptime!(AND_GROUP);
                    let mut acc = Array::<u32>::new(comptime!((ag * ept) as usize));
                    let mut jc = 0u32;
                    while jc < nt {
                        let mut cend = jc + 32u32;
                        if cend > nt {
                            cend = nt;
                        }
                        let mut pm = Array::<u32>::new(ept as usize);
                        if nt >= TRANSPOSE_MIN {
                            let cb = jc / 32u32;
                            if cb % ag == 0u32 {
                                #[unroll]
                                for i in 0..comptime!(ag * ept) {
                                    acc[i as usize] = 0u32;
                                }
                                let ab = cb * 64u32;
                                #[unroll]
                                for j in 0..ept {
                                    // The entry's digits, low word first; an
                                    // empty slot (vv = 0) has none.
                                    let mut x = mlo[j as usize] & vv[j as usize];
                                    let mut y = mhi[j as usize] & vv[j as usize];
                                    while (x | y) != 0u32 {
                                        let lo = x != 0u32;
                                        let v = select(lo, x, y);
                                        // v != 0: its lowest set bit.
                                        let d =
                                            (v.find_first_set() - 1u32) + select(lo, 0u32, 32u32);
                                        let v1 = v & (v - 1u32);
                                        x = select(lo, v1, x);
                                        y = select(lo, y, v1);
                                        let a = ab + d;
                                        #[unroll]
                                        for c in 0..ag {
                                            acc[comptime!(c * ept + j) as usize] |=
                                                t_tr[(a + comptime!(c * 64)) as usize];
                                        }
                                    }
                                }
                            }
                            // Pass: no digit of the entry in the top, and a
                            // top of the batch.
                            let nb = cend - jc;
                            let cm =
                                select(nb >= 32u32, 0xFFFF_FFFFu32, (1u32 << (nb & 31u32)) - 1u32);
                            #[unroll]
                            for j in 0..ept {
                                pm[j as usize] = (acc[j as usize] ^ cm) & cm;
                            }
                            #[unroll]
                            for i in 0..comptime!((ag - 1) * ept) {
                                acc[i as usize] = acc[comptime!(i + ept) as usize];
                            }
                        } else {
                            #[unroll]
                            for j in 0..ept {
                                pm[j as usize] = 0u32;
                            }
                            let mut jt = jc;
                            let mut bit = 1u32;
                            while jt < cend {
                                let tlo = t_lo[jt as usize];
                                let thi = t_hi[jt as usize];
                                #[unroll]
                                for j in 0..ept {
                                    let z = (mlo[j as usize] & tlo) | (mhi[j as usize] & thi);
                                    pm[j as usize] |= select(z == 0u32, bit, 0u32);
                                }
                                bit <<= 1u32;
                                jt += 1u32;
                            }
                        }
                        // The range test, on the chunk's clipped tops only
                        // (none but in a field's edge blocks): their pass
                        // bits for residues outside the range are cleared,
                        // so every bit walked is a join survivor.
                        let cw = s_cw[(jc / 32u32) as usize].load();
                        if cw != 0u32 {
                            #[unroll]
                            for j in 0..ept {
                                let r0 = rr[j as usize];
                                let mut y = pm[j as usize] & vv[j as usize] & cw;
                                while y != 0u32 {
                                    let i = y.trailing_zeros();
                                    y &= y - 1u32;
                                    if r0 < t_rlo[(jc + i) as usize]
                                        || r0 >= t_rhi[(jc + i) as usize]
                                    {
                                        pm[j as usize] ^= 1u32 << i;
                                    }
                                }
                            }
                        }
                        #[unroll]
                        for j in 0..ept {
                            nand += (pm[j as usize] & vv[j as usize]).count_ones();
                        }
                        // Each lane walks its set bits, one per round while any
                        // lane of the plane has one left. With an exit that
                        // diverged per lane, CUDA's independent thread
                        // scheduling (sm_70+) let the lanes drift apart, and
                        // the AND loop after the walk ran with the plane partly
                        // idle: 1.6-1.7x on this kernel at base 57. On CUDA
                        // (`walk_sync`) each round also ends with a plane
                        // barrier, which drops the YIELD ptxas otherwise puts in
                        // the loop and lets it keep fewer registers: another
                        // 2-11%.
                        #[unroll]
                        for j in 0..ept {
                            let mut x = pm[j as usize] & vv[j as usize];
                            let r0 = rr[j as usize];
                            while plane_any(x != 0u32) {
                                if x != 0u32 {
                                    let jt2 = jc + x.trailing_zeros();
                                    x &= x - 1u32;
                                    let pass = if comptime!(lean) {
                                        walk_pf_lean(
                                            ac[j as usize],
                                            ac3[j as usize],
                                            pc[j as usize],
                                            pc3[j as usize],
                                            t_dm[jt2 as usize],
                                            mlo[j as usize],
                                            mhi[j as usize],
                                            t_glo[jt2 as usize],
                                            t_ghi[jt2 as usize],
                                            base,
                                        )
                                    } else {
                                        walk_pf(
                                            ac[j as usize],
                                            pc[j as usize],
                                            t_dm[jt2 as usize],
                                            mlo[j as usize],
                                            mhi[j as usize],
                                            t_lo[jt2 as usize],
                                            t_hi[jt2 as usize],
                                            base,
                                            k,
                                            k2,
                                        )
                                    };
                                    if pass {
                                        let at = s_cnt[0].fetch_add(1u32);
                                        if at < SURV_SH_CAP {
                                            s_t[at as usize] = t_id[jt2 as usize];
                                            s_r[at as usize] = r0;
                                        } else {
                                            let g = surv_count[0].fetch_add(1u32);
                                            if g < surv_cap {
                                                surv[(2u32 * g) as usize] = t_id[jt2 as usize];
                                                surv[(2u32 * g + 1u32) as usize] = r0;
                                            }
                                        }
                                    }
                                }
                                if walk_sync {
                                    sync_plane();
                                }
                            }
                        }
                        jc += 32u32;
                    }
                    ws += wave;
                }
                // Write the stage out once it is half full, checked once per
                // batch of tops (per chunk, its two barriers cost 2 % of a
                // field). A batch can add more than the half left (up to 256
                // tops times the bucket's entries): the pairs past a full
                // stage go to the list directly, above. They passed the
                // prefilter too, so the list holds the same entries either
                // way.
                //
                // Every thread reads the count before any appends the next
                // batch's survivors (the second barrier): a plane that read
                // it after a faster plane's appends could decide alone to
                // flush, and from then on the cube's barriers would pair up
                // at different call sites. The lone plane's flush writes at a
                // list offset no one reserved for it, which costs full checks
                // of stale list entries and can overwrite ones another flush
                // reserved. The first barrier also puts every thread past its
                // last read of the batch's clip words, cleared below.
                sync_cube();
                let sc = s_cnt[0].load();
                sync_cube();
                if sc >= SURV_FLUSH {
                    flush_survivors(
                        &s_t,
                        &s_r,
                        &mut s_cnt,
                        &mut s_base,
                        surv,
                        surv_count,
                        surv_cap,
                        sc,
                    );
                    sync_cube();
                }
                if UNIT_POS_X < comptime!(TOP_CAP / 32) {
                    s_cw[UNIT_POS_X as usize].store(0u32);
                }
                tc += TOP_CAP;
                sync_cube();
            }
            w += CUBE_COUNT_X;
        }
        sync_cube();
        let sc = s_cnt[0].load();
        if sc > 0u32 {
            flush_survivors(
                &s_t,
                &s_r,
                &mut s_cnt,
                &mut s_base,
                surv,
                surv_count,
                surv_cap,
                sc,
            );
        }
        // One atomic per plane for the join survivors.
        let planes_and = plane_sum(nand);
        if UNIT_POS_PLANE == 0u32 {
            and_count[0].fetch_add(planes_and);
        }
        cube_done(ran, 2u32);
    }

    /// 16-bit limbs of `k^e`, least significant first: the host-side
    /// constants `cert_power` multiplies by.
    // Every limb is masked to 16 bits.
    #[allow(clippy::cast_possible_truncation)]
    fn pow_limbs16(k: u32, e: u32) -> Vec<u32> {
        let mut v = vec![1u32];
        for _ in 0..e {
            let mut c = 0u64;
            for l in &mut v {
                let t = u64::from(*l) * u64::from(k) + c;
                *l = (t & 0xFFFF) as u32;
                c = t >> 16;
            }
            while c != 0 {
                v.push((c & 0xFFFF) as u32);
                c >>= 16;
            }
        }
        v
    }

    /// u32 words of `k^e`, least significant first.
    fn pow_words32(k: u32, e: u32) -> Vec<u32> {
        pow_limbs16(k, e)
            .chunks(2)
            .map(|c| c[0] | c.get(1).map_or(0, |&h| h << 16))
            .collect()
    }

    /// (m, s, a) with ((L·m + a) >> s) = `ceil(L·log_b(2))`, the smallest i
    /// with b^i >= 2^L, for every bit length L in 0..=`lmax`, and `lmax·m +
    /// a` below 2^32: the host-side constants of [`ndig_fast`]. Searched
    /// near m = `2^s·log_b(2)` and checked against the powers' bit lengths;
    /// every base 10-64 has one for every width the kernels use (a test
    /// checks it), so the panic is unreachable there.
    #[allow(clippy::cast_precision_loss, clippy::cast_sign_loss)]
    pub(super) fn ndig_magic(base: u32, lmax: u32) -> (u32, u32, u32) {
        // want[L] = the smallest i with b^i >= 2^L, i.e. bits(b^i) > L.
        let mut want = Vec::with_capacity(lmax as usize + 1);
        let mut i = 0u32;
        let mut bits = 1u32;
        for l in 0..=lmax {
            while bits <= l {
                i += 1;
                let w = pow_limbs16(base, i);
                bits = 16 * (w.len() as u32 - 1) + (32 - w[w.len() - 1].leading_zeros());
            }
            want.push(i);
        }
        let lg = 1.0 / f64::from(base).log2();
        for sh in 12..=24u32 {
            let m0 = ((1u64 << sh) as f64 * lg) as u64;
            for m in m0.saturating_sub(3).max(1)..=m0 + 3 {
                for t in 0..64u64 {
                    let a = (1u64 << sh) - 1 - t;
                    if u64::from(lmax) * m + a >= 1 << 32 {
                        continue;
                    }
                    if (0..=lmax).all(|l| ((u64::from(l) * m + a) >> sh) as u32 == want[l as usize])
                    {
                        return (m as u32, sh, a as u32);
                    }
                }
            }
        }
        panic!("b{base}: no digit-count constants up to {lmax} bits");
    }

    /// min(ndig(d), sp) for d of `nl` u32 words, where ndig(d) is the
    /// smallest i with b^i > d, from d's bit length L and one compare with
    /// the power table `pw` (`nl3` words a power): 2^(L-1) <= d < 2^L, and
    /// at most one power of b lies in (2^(L-1), 2^L], so ndig(d) is c or
    /// c + 1 for c = `ceil(L·log_b(2)) - 1`, and it is c iff d < b^c. c is
    /// clamped to sp before the compare. Its constants are [`ndig_magic`]'s.
    #[cube]
    pub(super) fn ndig_fast(
        d: &Array<u32>,
        pw: &Array<u32>,
        #[comptime] nl: u32,
        #[comptime] nl3: u32,
        #[comptime] sp: u32,
        #[comptime] base: u32,
    ) -> u32 {
        let mut bl = 0u32;
        #[unroll]
        for j in 0..nl {
            let w = d[j as usize];
            bl = select(w != 0u32, comptime!(32 * (j + 1)) - w.leading_zeros(), bl);
        }
        let m = comptime!(ndig_magic(base, 32 * nl).0);
        let sh = comptime!(ndig_magic(base, 32 * nl).1);
        let a = comptime!(ndig_magic(base, 32 * nl).2);
        let ce = (bl * m + a) >> sh;
        let mut c = select(ce > 0u32, ce - 1u32, 0u32);
        c = select(c > sp, sp.runtime(), c);
        // d < b^c ?
        let mut lt = false;
        let mut eq = true;
        #[unroll]
        for jj in 0..nl3 {
            let j = comptime!(nl3 - 1 - jj);
            let pj = pw[(c * nl3 + j) as usize];
            let mut dj = 0u32;
            if comptime!(j < nl) {
                dj = d[j as usize];
            }
            lt = lt || (eq && dj < pj);
            eq = eq && dj == pj;
        }
        let nd = select(lt, c, c + 1u32);
        select(nd > sp, sp.runtime(), nd)
    }

    /// One power's share of the top certificate, the device form of
    /// `Base::cert`'s inner loop: x is the power of the interval's low
    /// endpoint (`2·nl` 16-bit limbs, one to a u32; destroyed; `nl` is the
    /// width in u32 words) and d = y - x (`nl` u32 words, from
    /// [`powers_diff`]), where y is the high endpoint's; `sp` is the power's
    /// digit count. Returns the mask of the digits `cert` certifies, the ones
    /// common to x and y at positions >= c0 = max(cap, ndig(d)) from the top
    /// down to their first disagreement; `c0`, and `dup` set if two of those
    /// digits repeat, via the out-arrays; and in `out_dom` the digits of x
    /// and y at that topmost disagreement, which bound every n's digit there
    /// (`Base::cert`), or (1, 0) if there is none. ndig(d) comes from d's
    /// bit length ([`ndig_fast`]).
    ///
    /// Only x's digits are divided out, as in `Base::cert`: y - x < b^c0,
    /// so floor(y / b^c0) is floor(x / b^c0) or one more, and it is one more
    /// exactly when (x mod b^c0) + (y - x) >= b^c0. If not, every digit from
    /// c0 up is common. If so, the two differ on x's run of (b - 1)s from c0
    /// up and on the digit d above it, which is d + 1 in y, and agree above
    /// that. The scan runs least significant first, so it keeps both
    /// answers (every digit from c0 up; the digits above the first that is
    /// not b - 1) and the remainder test picks one at the end. x mod b^c0
    /// is put together in 16-bit limbs from the digits below c0, which the
    /// scan divides out anyway: each pass adds its chunk's digits below c0,
    /// times B^pass (B = `chunk_div`, whose powers' limbs are constants).
    ///
    /// The limbs are 16-bit so that no constant division here divides a
    /// bare half-width right shift. Each pass's long division by `chunk_div`
    /// starts with the remainder 0, and the loops are unrolled, so with u32
    /// limbs split in 16-bit halves the first dividend of every pass was
    /// exactly `x[top] >> 16`, and in pass 0 `x[top]` was the high half of a
    /// u64 product sum. Apple's GPU compiler (M4, macOS 26.3, through naga's
    /// MSL) miscompiles `naga_div(x >> 16u, K)`, `(x >> 16u) % K` and
    /// `uint(ulong p >> 32) / K` for a constant K that is not a power of
    /// two. The detailed FD seed of #225 came out wrong at b49, b50 and
    /// b52-55 from the same division, and so would the tops' certificates
    /// wherever the top word of n² or n³ is not zero (b45-55 of the join's
    /// bases; it is zero at b40-44 and b57-64). Here the first step of each
    /// pass divides the top two limbs that can be nonzero as one u32, both
    /// stored 16-bit limbs (the schoolbook's in `powers_diff` in pass 0, the
    /// previous quotient's later), never a half-width shift of a sum.
    ///
    /// x < b^sp (true whenever the low endpoint has the field's digit
    /// counts), so before pass p, x < b^(sp - p·cd) and only that power's
    /// limb count of limbs can be nonzero; each pass divides only those (a
    /// comptime bound, so no branch). At b62 that is 185 division steps a
    /// certificate instead of 466.
    #[cube]
    pub(super) fn cert_power(
        x: &mut Array<u32>,
        d: &Array<u32>,
        pw: &Array<u32>,
        out_c0: &mut Array<u32>,
        out_dup: &mut Array<u32>,
        out_dom: &mut Array<u32>,
        #[comptime] nl: u32,
        #[comptime] nl3: u32,
        #[comptime] sp: u32,
        #[comptime] cap: u32,
        #[comptime] base: u32,
        #[comptime] chunk_digits: u32,
        #[comptime] chunk_div: u32,
    ) -> u64 {
        let nh = comptime!(2 * nl);
        let mut c0 = ndig_fast(d, pw, nl, nl3, sp, base);
        if c0 < cap {
            c0 = cap.runtime();
        }
        // x mod b^c0, in 16-bit limbs.
        let mut rem = Array::<u32>::new(nh as usize);
        #[unroll]
        for j in 0..nh {
            rem[j as usize] = 0u32;
        }
        // Every digit from c0 up (`run`, `dup`); and the digits above x's
        // run of (b - 1)s from c0 up (`runc`, `dupc`), with the digit `dlo`
        // that ends that run.
        let mut run = 0u64;
        let mut dup = 0u32;
        let mut runc = 0u64;
        let mut dupc = 0u32;
        let mut in_run = true;
        let mut dlo = 0u32;
        let passes = comptime!(sp.div_ceil(chunk_digits));
        #[unroll]
        for pass in 0..passes {
            // Only the limbs that can be nonzero: x < b^(sp - pass·cd) here,
            // so its limbs from that power's limb count up are zero. The top
            // two of them as one u32 (see above), then one limb a step.
            let lp =
                comptime!((pow_limbs16(base, sp - pass * chunk_digits).len() as u32).clamp(2, nh));
            let top = comptime!(lp - 1);
            let next = comptime!(lp - 2);
            let cx = (x[top as usize] << 16u32) | x[next as usize];
            let qx = cx / chunk_div;
            let mut rx = cx - qx * chunk_div;
            x[top as usize] = qx >> 16u32;
            x[next as usize] = qx & 0xFFFFu32;
            #[unroll]
            for jj in 2..lp {
                let j = comptime!(lp - 1 - jj);
                let c1 = (rx << 16u32) | x[j as usize];
                let q1 = c1 / chunk_div;
                rx = c1 - q1 * chunk_div;
                x[j as usize] = q1;
            }
            // The chunk's digits below c0, as a value below B: the chunk
            // itself if it lies below c0 (and below sp; the last chunk can
            // reach past sp, so it always takes the digit path). Otherwise
            // its digits one at a time, with selects rather than branches:
            // CubeCL's SPIR-V compiler recurses once per basic block, on a
            // thread with a 2 MiB stack.
            let start = comptime!(pass * chunk_digits);
            let digit_path = comptime!(if start + chunk_digits > sp {
                u32::MAX
            } else {
                start + chunk_digits
            });
            let mut low = rx;
            if c0 < digit_path {
                low = 0u32;
                #[unroll]
                for q in 0..chunk_digits {
                    let pos = comptime!(start + q);
                    let dx = rx % base;
                    rx /= base;
                    if comptime!(pos < sp) {
                        let above = c0 <= pos;
                        let bit = select(above, 1u64 << u64::cast_from(dx), 0u64);
                        dup |= select((run & bit) != 0u64, 1u32, 0u32);
                        run |= bit;
                        // Above the end of x's run of (b - 1)s.
                        let cbit = select(in_run, 0u64, bit);
                        dupc |= select((runc & cbit) != 0u64, 1u32, 0u32);
                        runc |= cbit;
                        let ends = above && in_run && dx != comptime!(base - 1);
                        dlo = select(ends, dx, dlo);
                        in_run = in_run && !ends;
                        low += select(above, 0u32, dx * comptime!(base.pow(q)));
                    }
                }
            }
            // rem += low · B^pass. The sum is x mod b^min(c0, start + cd),
            // below B^(pass + 1), and B^pass fits x's limbs (start < sp).
            if start < c0 {
                let len = comptime!((pow_limbs16(chunk_div, pass).len() as u32).min(nh));
                let topk = comptime!((pow_limbs16(chunk_div, pass + 1).len() as u32).min(nh));
                let mut c = 0u32;
                #[unroll]
                for i in 0..len {
                    let t = low * comptime!(pow_limbs16(chunk_div, pass)[i as usize])
                        + rem[i as usize]
                        + c;
                    rem[i as usize] = t & 0xFFFFu32;
                    c = t >> 16u32;
                }
                #[unroll]
                for i in len..topk {
                    let t = rem[i as usize] + c;
                    rem[i as usize] = t & 0xFFFFu32;
                    c = t >> 16u32;
                }
            }
        }
        // floor(y / b^c0) is one more than floor(x / b^c0) iff
        // (x mod b^c0) + (y - x) >= b^c0. The sum is below 2·b^c0 <= 2x, so
        // it can carry out of nl words, and is then above b^c0.
        let mut one_more = false;
        if c0 < sp {
            let mut s = Array::<u32>::new(nl as usize);
            let mut carry = 0u32;
            #[unroll]
            for j in 0..nl {
                let rj =
                    rem[comptime!(2 * j) as usize] | (rem[comptime!(2 * j + 1) as usize] << 16u32);
                let t = rj + d[j as usize];
                let t2 = t + carry;
                let mut nc = 0u32;
                if t < rj || t2 < t {
                    nc = 1u32;
                }
                s[j as usize] = t2;
                carry = nc;
            }
            let mut ge = true;
            let mut eq = true;
            #[unroll]
            for jj in 0..nl3 {
                let j = comptime!(nl3 - 1 - jj);
                let pj = pw[(c0 * nl3 + j) as usize];
                let mut sj = 0u32;
                if comptime!(j < nl) {
                    sj = s[j as usize];
                }
                if eq {
                    if sj < pj {
                        ge = false;
                        eq = false;
                    } else if sj > pj {
                        eq = false;
                    }
                }
            }
            one_more = carry != 0u32 || ge;
        }
        let mut dom_lo = 1u32;
        let mut dom_hi = 0u32;
        if one_more {
            run = runc;
            dup = dupc;
            dom_lo = dlo;
            dom_hi = dlo + 1u32;
        }
        out_c0[0] = c0;
        out_dup[0] = dup;
        out_dom[0] = dom_lo;
        out_dom[1] = dom_hi;
        run
    }

    /// Digits `lo..=hi` as a mask, 0 if `hi < lo` (`digit_range_mask`).
    #[cube]
    fn dom_mask(lo: u32, hi: u32) -> u64 {
        let mut m = 0u64;
        if hi >= lo {
            let w = hi - lo + 1u32;
            if w >= 64u32 {
                m = 0xFFFF_FFFF_FFFF_FFFFu64 << u64::cast_from(lo);
            } else {
                m = ((1u64 << u64::cast_from(w)) - 1u64) << u64::cast_from(lo);
            }
        }
        m
    }

    /// 16-bit limbs (one to a u32) packed into `nl` u32 words.
    #[cube]
    pub(super) fn pack16(x: &Array<u32>, w: &mut Array<u32>, #[comptime] nl: u32) {
        #[unroll]
        for j in 0..nl {
            w[j as usize] =
                x[comptime!(2 * j) as usize] | (x[comptime!(2 * j + 1) as usize] << 16u32);
        }
    }

    /// The powers a top certificate needs, for an interval [a, e] (`limbs`
    /// u32 words each) whose width δ = e - a fits `dw` words: a² (4L limbs)
    /// and a³ (6L limbs) in 16-bit limbs one to a u32, and the differences
    /// e² - a² = δ(2a + δ) (2L u32 words) and e³ - a³ = δ(3a² + 3aδ + δ²)
    /// (3L words), without forming e² or e³. All by schoolbook in u32 on
    /// 16-bit limbs (`cert_power` says why): a limb product plus two values
    /// below 2^16 is at most 2^32 - 1. In the top kernel δ < w < 2^32, so
    /// `dw` is 1 and a difference costs about a quarter of the high
    /// endpoint's power.
    #[cube]
    pub(super) fn powers_diff(
        a: &Array<u32>,
        e: &Array<u32>,
        a2: &mut Array<u32>,
        a3: &mut Array<u32>,
        d2: &mut Array<u32>,
        d3: &mut Array<u32>,
        #[comptime] limbs: u32,
        #[comptime] dw: u32,
        #[comptime] cubes: bool,
    ) {
        let nh = comptime!(2 * limbs);
        let nh2 = comptime!(4 * limbs);
        let nh3 = comptime!(6 * limbs);
        let dh = comptime!(2 * dw);
        let mut ah = Array::<u32>::new(nh as usize);
        #[unroll]
        for j in 0..limbs {
            ah[comptime!(2 * j) as usize] = a[j as usize] & 0xFFFFu32;
            ah[comptime!(2 * j + 1) as usize] = a[j as usize] >> 16u32;
        }
        // δ = e - a over its dw low words, in 16-bit limbs.
        let mut dl = Array::<u32>::new(dh as usize);
        let mut borrow = 0u32;
        #[unroll]
        for j in 0..dw {
            let ej = e[j as usize];
            let aj = a[j as usize];
            let t = ej - aj - borrow;
            borrow = select(ej < aj || (ej == aj && borrow != 0u32), 1u32, 0u32);
            dl[comptime!(2 * j) as usize] = t & 0xFFFFu32;
            dl[comptime!(2 * j + 1) as usize] = t >> 16u32;
        }
        // a² and a³.
        #[unroll]
        for j in 0..nh2 {
            a2[j as usize] = 0u32;
        }
        #[unroll]
        for i1 in 0..nh {
            let mut ca = 0u32;
            #[unroll]
            for j in 0..nh {
                let kk = comptime!(i1 + j);
                let ta = ah[i1 as usize] * ah[j as usize] + a2[kk as usize] + ca;
                a2[kk as usize] = ta & 0xFFFFu32;
                ca = ta >> 16u32;
            }
            a2[comptime!(i1 + nh) as usize] = ca;
        }
        // a³ only if the caller divides it (`cubes`; the FD top kernel
        // steps its digits instead).
        if comptime!(cubes) {
            #[unroll]
            for j in 0..nh3 {
                a3[j as usize] = 0u32;
            }
            #[unroll]
            for i1 in 0..nh2 {
                let mut ca = 0u32;
                #[unroll]
                for j in 0..nh {
                    let kk = comptime!(i1 + j);
                    let ta = a2[i1 as usize] * ah[j as usize] + a3[kk as usize] + ca;
                    a3[kk as usize] = ta & 0xFFFFu32;
                    ca = ta >> 16u32;
                }
                a3[comptime!(i1 + nh) as usize] = ca;
            }
        }
        // u = 2a + δ (nh + 1 limbs); d2 = δ·u.
        let nu = comptime!(nh + 1);
        let mut u = Array::<u32>::new(nu as usize);
        let mut c = 0u32;
        #[unroll]
        for j in 0..nh {
            let mut t = (ah[j as usize] << 1u32) + c;
            if comptime!(j < dh) {
                t += dl[j as usize];
            }
            u[j as usize] = t & 0xFFFFu32;
            c = t >> 16u32;
        }
        u[nh as usize] = c;
        let np2 = comptime!(if dh + nu > nh2 { dh + nu } else { nh2 });
        let mut t2 = Array::<u32>::new(np2 as usize);
        #[unroll]
        for j in 0..np2 {
            t2[j as usize] = 0u32;
        }
        #[unroll]
        for i1 in 0..dh {
            let mut cc = 0u32;
            #[unroll]
            for j in 0..nu {
                let kk = comptime!(i1 + j);
                let t = dl[i1 as usize] * u[j as usize] + t2[kk as usize] + cc;
                t2[kk as usize] = t & 0xFFFFu32;
                cc = t >> 16u32;
            }
            t2[comptime!(i1 + nu) as usize] = cc;
        }
        pack16(&t2, d2, comptime!(2 * limbs));
        // aδ (nh + dh <= nh2 limbs) and δ² (2·dh limbs).
        let mut ad = Array::<u32>::new(nh2 as usize);
        #[unroll]
        for j in 0..nh2 {
            ad[j as usize] = 0u32;
        }
        #[unroll]
        for i1 in 0..dh {
            let mut cc = 0u32;
            #[unroll]
            for j in 0..nh {
                let kk = comptime!(i1 + j);
                let t = dl[i1 as usize] * ah[j as usize] + ad[kk as usize] + cc;
                ad[kk as usize] = t & 0xFFFFu32;
                cc = t >> 16u32;
            }
            ad[comptime!(i1 + nh) as usize] = cc;
        }
        let ndd = comptime!(2 * dh);
        let mut dd = Array::<u32>::new(ndd as usize);
        #[unroll]
        for j in 0..ndd {
            dd[j as usize] = 0u32;
        }
        #[unroll]
        for i1 in 0..dh {
            let mut cc = 0u32;
            #[unroll]
            for j in 0..dh {
                let kk = comptime!(i1 + j);
                let t = dl[i1 as usize] * dl[j as usize] + dd[kk as usize] + cc;
                dd[kk as usize] = t & 0xFFFFu32;
                cc = t >> 16u32;
            }
            dd[comptime!(i1 + dh) as usize] = cc;
        }
        // v = 3(a² + aδ) + δ² (below 3e², nh2 + 1 limbs); a² + aδ = ae
        // fits nh2 limbs.
        let nv = comptime!(nh2 + 1);
        let mut v = Array::<u32>::new(nv as usize);
        let mut cs = 0u32;
        let mut cv = 0u32;
        #[unroll]
        for j in 0..nh2 {
            let sj = a2[j as usize] + ad[j as usize] + cs;
            cs = sj >> 16u32;
            let mut t = (sj & 0xFFFFu32) * 3u32 + cv;
            if comptime!(j < ndd) {
                t += dd[j as usize];
            }
            v[j as usize] = t & 0xFFFFu32;
            cv = t >> 16u32;
        }
        v[nh2 as usize] = cv + cs * 3u32;
        // d3 = δ·v.
        let np3 = comptime!(if dh + nv > nh3 { dh + nv } else { nh3 });
        let mut t3 = Array::<u32>::new(np3 as usize);
        #[unroll]
        for j in 0..np3 {
            t3[j as usize] = 0u32;
        }
        #[unroll]
        for i1 in 0..dh {
            let mut cc = 0u32;
            #[unroll]
            for j in 0..nv {
                let kk = comptime!(i1 + j);
                let t = dl[i1 as usize] * v[j as usize] + t3[kk as usize] + cc;
                t3[kk as usize] = t & 0xFFFFu32;
                cc = t >> 16u32;
            }
            t3[comptime!(i1 + nv) as usize] = cc;
        }
        pack16(&t3, d3, comptime!(3 * limbs));
    }

    /// The top certificate of [a, e] (`limbs` u32 words each; e - a fits
    /// `dw` words), the device form of `Base::cert`: returns its mask, and
    /// in `res` whether the interval is kept (1) or not (0) and the square's
    /// and the cube's c0 (the terms of `Base::cert_floor`).
    #[cube]
    pub(super) fn top_cert(
        a: &Array<u32>,
        e: &Array<u32>,
        pw: &Array<u32>,
        res: &mut Array<u32>,
        #[comptime] base: u32,
        #[comptime] limbs: u32,
        #[comptime] cap: u32,
        #[comptime] s2: u32,
        #[comptime] s3: u32,
        #[comptime] chunk_digits: u32,
        #[comptime] chunk_div: u32,
        #[comptime] dw: u32,
    ) -> u64 {
        let nl2 = comptime!(2 * limbs);
        let nl3 = comptime!(3 * limbs);
        let nh2 = comptime!(4 * limbs);
        let nh3 = comptime!(6 * limbs);
        let mut a2 = Array::<u32>::new(nh2 as usize);
        let mut a3 = Array::<u32>::new(nh3 as usize);
        let mut d2 = Array::<u32>::new(nl2 as usize);
        let mut d3 = Array::<u32>::new(nl3 as usize);
        powers_diff(a, e, &mut a2, &mut a3, &mut d2, &mut d3, limbs, dw, true);
        let mut c0a = Array::<u32>::new(1usize);
        let mut dupa = Array::<u32>::new(1usize);
        let mut c0b = Array::<u32>::new(1usize);
        let mut dupb = Array::<u32>::new(1usize);
        let mut doma = Array::<u32>::new(2usize);
        let mut domb = Array::<u32>::new(2usize);
        let msq = cert_power(
            &mut a2,
            &d2,
            pw,
            &mut c0a,
            &mut dupa,
            &mut doma,
            nl2,
            nl3,
            s2,
            cap,
            base,
            chunk_digits,
            chunk_div,
        );
        let mcu = cert_power(
            &mut a3,
            &d3,
            pw,
            &mut c0b,
            &mut dupb,
            &mut domb,
            nl3,
            nl3,
            s3,
            cap,
            base,
            chunk_digits,
            chunk_div,
        );
        close_cert(
            msq, mcu, dupa[0], dupb[0], &doma, &domb, c0a[0], c0b[0], res,
        )
    }

    /// The end of a top certificate, shared by [`top_cert`] and
    /// [`top_cert_fd`]: both powers' masks (`msq`, `mcu`), repeat flags and
    /// first-disagreement digits closed into the certificate's mask; `res`
    /// gets kept (1) or not (0) and the two c0s.
    #[cube]
    fn close_cert(
        msq: u64,
        mcu: u64,
        dupa: u32,
        dupb: u32,
        doma: &Array<u32>,
        domb: &Array<u32>,
        c0a: u32,
        c0b: u32,
        res: &mut Array<u32>,
    ) -> u64 {
        let mut mask = msq | mcu;
        let mut keep = dupa == 0u32 && dupb == 0u32 && (msq & mcu) == 0u64;
        // `close_certificate` over the two powers' first-disagreement
        // intervals: an interval with no free digit rejects, and one with a
        // single free digit forces it. Forcing the cube's digit can only
        // shrink the square's free set, so one recheck of the square reaches
        // the same fixed point.
        let d0 = dom_mask(doma[0], doma[1]);
        let d1 = dom_mask(domb[0], domb[1]);
        let mut f0 = false;
        let mut f1 = false;
        if keep && d0 != 0u64 {
            let fr = d0 & !mask;
            if fr == 0u64 {
                keep = false;
            } else if (fr & (fr - 1u64)) == 0u64 {
                mask |= fr;
                f0 = true;
            }
        }
        if keep && d1 != 0u64 {
            let fr = d1 & !mask;
            if fr == 0u64 {
                keep = false;
            } else if (fr & (fr - 1u64)) == 0u64 {
                mask |= fr;
                f1 = true;
            }
        }
        if keep && d0 != 0u64 && !f0 && f1 {
            let fr = d0 & !mask;
            if fr == 0u64 {
                keep = false;
            } else if (fr & (fr - 1u64)) == 0u64 {
                mask |= fr;
            }
        }
        let mut kept = 0u32;
        if keep {
            kept = 1u32;
        }
        res[0] = kept;
        res[1] = c0a;
        res[2] = c0b;
        mask
    }

    /// Append a plane's kept tops of one slot into their groups' segments of
    /// `top_rec` (see [`top_kernel`], which takes one atomic per top) and
    /// return a kept lane's record index:
    /// `slot·ntlay + goff + at`, `at` its place among the group's kept tops
    /// (`grp_cnt[slot·nb + g]`). The host sorts the top layer by group, so a
    /// plane's kept tops are almost always of one group, and take one atomic;
    /// a plane that straddles groups takes one per kept lane. Every lane of
    /// the plane must call it.
    ///
    /// The group tested is lane 0's, kept or not: lane 0 holds the plane's
    /// lowest prefix index, so a dead lane 0 means a dead plane (nothing
    /// kept, no atomic), and lane 0's group differs from its kept lanes' only
    /// where the plane straddles a group boundary. Only such a plane can take
    /// per-lane atomics where one would have done.
    ///
    /// With `ballot` (`ballot_append`: planes of at most 32 lanes) the
    /// plane's count and each lane's rank are popcounts of the ballot's first
    /// word, instead of a sum and an exclusive scan over the plane. `CubeCL`'s
    /// C++ backends return only that word (HIP truncates a 64-lane ballot to
    /// it), so wider planes keep the scans.
    #[cube]
    fn top_slot(
        kept: bool,
        g: u32,
        goff: u32,
        slot: u32,
        ntlay: u32,
        grp_cnt: &mut Array<Atomic<u32>>,
        #[comptime] nb: u32,
        #[comptime] ballot: bool,
    ) -> u32 {
        let g0 = plane_broadcast(g, 0u32);
        let one = plane_all(!kept || g == g0);
        let kc = select(kept && one, 1u32, 0u32);
        let b0 = if comptime!(ballot) {
            let bal = plane_ballot(kept && one);
            bal[0usize]
        } else {
            kc
        };
        let n = if comptime!(ballot) {
            b0.count_ones()
        } else {
            plane_sum(kc)
        };
        let pre = if comptime!(ballot) {
            (b0 & ((1u32 << UNIT_POS_PLANE) - 1u32)).count_ones()
        } else {
            plane_exclusive_sum(kc)
        };
        let mut a0 = 0u32;
        if UNIT_POS_PLANE == 0u32 && n > 0u32 {
            a0 = grp_cnt[(slot * nb + g0) as usize].fetch_add(n);
        }
        let mut at = plane_broadcast(a0, 0u32) + pre;
        if kept && !one {
            at = grp_cnt[(slot * nb + g) as usize].fetch_add(1u32);
        }
        slot * ntlay + goff + at
    }

    /// Stage 0a: certify every top prefix of every slot's partition. Grid: x
    /// over the field's top layer (depth t - pp, host-built once per slice
    /// and sorted by group), y over slots. With `only_clipped`, only the
    /// blocks clipped to the slice's ends (beside [`top_fd_kernel`], which
    /// leaves them out). Per tlay entry (5 words): P0 lo, hi, its group `g =
    /// key·(b-1) + P0·b^pp mod (b-1)` (key = P0 mod b at the key level, else
    /// 0), the group's first index `goff[g]` in the sorted layer, and P0 mod
    /// b^(k2-f0-pp). `fb` = s (3 limbs), e-1 (3 limbs), the P range lo (2
    /// words) and hi (2 words). Passing tops are appended into their group's
    /// segment of the slot's records, one record each in `top_rec`
    /// ([`TOP_REC_WORDS`]), counted in `grp_cnt`: a top of group `g` at slot
    /// `v` belongs to bucket `(key, (root - (g mod (b-1)) - v) mod (b-1))`
    /// for each digit-sum root, so [`join_kernel`] reads a bucket's tops as
    /// the segments of `nroots` groups.
    #[cube(launch_unchecked)]
    pub fn top_kernel(
        tlay: &Array<u32>,
        vs: &Array<u32>,
        fb: &Array<u32>,
        pw: &Array<u32>,
        top_rec: &mut Array<u32>,
        grp_cnt: &mut Array<Atomic<u32>>,
        ran: &mut Array<Atomic<u32>>,
        ntlay: u32,
        w: u32,
        pdiv: u32,
        full_floor: u32, // certificate floor bound of every unclipped block (host rule)
        #[comptime] base: u32,
        #[comptime] limbs: u32,
        #[comptime] cap: u32,
        #[comptime] k2: u32,
        #[comptime] s2: u32,
        #[comptime] s3: u32,
        #[comptime] chunk_digits: u32,
        #[comptime] chunk_div: u32,
        #[comptime] nb: u32,
        #[comptime] only_clipped: bool,
    ) {
        let i = ABSOLUTE_POS_X;
        let slot = CUBE_POS_Y;
        if i < ntlay {
            let p0 = (u64::cast_from(tlay[(5u32 * i + 1u32) as usize]) << 32u64)
                | u64::cast_from(tlay[(5u32 * i) as usize]);
            let v = vs[slot as usize];
            let p = p0 * u64::cast_from(pdiv) + u64::cast_from(v);
            let plo = (u64::cast_from(fb[7]) << 32u64) | u64::cast_from(fb[6]);
            let phi = (u64::cast_from(fb[9]) << 32u64) | u64::cast_from(fb[8]);
            if p >= plo && p <= phi {
                // a = p*w (< 2^96, 3 limbs), e = a + w - 1, clipped to [s, e-1].
                let t0 = (p & 0xFFFF_FFFFu64) * u64::cast_from(w);
                let t1 = (p >> 32u64) * u64::cast_from(w) + (t0 >> 32u64);
                let pw0 = u32::cast_from(t0);
                let mut a = Array::<u32>::new(limbs as usize);
                let mut e = Array::<u32>::new(limbs as usize);
                #[unroll]
                for j in 0..limbs {
                    if comptime!(j == 0) {
                        a[j as usize] = pw0;
                    } else if comptime!(j == 1) {
                        a[j as usize] = u32::cast_from(t1);
                    } else if comptime!(j == 2) {
                        a[j as usize] = u32::cast_from(t1 >> 32u64);
                    } else {
                        a[j as usize] = 0u32;
                    }
                }
                let mut carry = w - 1u32;
                #[unroll]
                for j in 0..limbs {
                    let t = a[j as usize] + carry;
                    let mut c = 0u32;
                    if t < carry {
                        c = 1u32;
                    }
                    e[j as usize] = t;
                    carry = c;
                }
                // a = max(a, s); e = min(e, e_incl) (limbs <= 3: fb words 0..2, 3..5)
                let mut a_lt_s = false;
                let mut e_gt_e = false;
                let mut eq_a = true;
                let mut eq_e = true;
                #[unroll]
                for jj in 0..limbs {
                    let j = comptime!(limbs - 1 - jj);
                    let sj = fb[j as usize];
                    let ej = fb[comptime!(3 + j) as usize];
                    if eq_a {
                        if a[j as usize] < sj {
                            a_lt_s = true;
                            eq_a = false;
                        } else if a[j as usize] > sj {
                            eq_a = false;
                        }
                    }
                    if eq_e {
                        if e[j as usize] > ej {
                            e_gt_e = true;
                            eq_e = false;
                        } else if e[j as usize] < ej {
                            eq_e = false;
                        }
                    }
                }
                // `only_clipped`: the blocks [`top_fd_kernel`] leaves out.
                let mut go = true;
                if comptime!(only_clipped) {
                    go = a_lt_s || e_gt_e;
                }
                if go {
                    if a_lt_s {
                        #[unroll]
                        for j in 0..limbs {
                            a[j as usize] = fb[j as usize];
                        }
                    }
                    if e_gt_e {
                        #[unroll]
                        for j in 0..limbs {
                            e[j as usize] = fb[comptime!(3 + j) as usize];
                        }
                    }
                    // e - a < w: one word.
                    let mut res = Array::<u32>::new(3usize);
                    let mask = top_cert(
                        &a,
                        &e,
                        pw,
                        &mut res,
                        base,
                        limbs,
                        cap,
                        s2,
                        s3,
                        chunk_digits,
                        chunk_div,
                        1u32,
                    );
                    if res[0] != 0u32 {
                        let mut floor = res[1];
                        if res[2] < floor {
                            floor = res[2];
                        }
                        // Same seed rule as the host: an unclipped block uses the
                        // field bound (a lower bound of its exact floor).
                        if !a_lt_s && !e_gt_e {
                            floor = full_floor;
                        }
                        // Into the group's segment, one atomic per top.
                        let g = tlay[(5u32 * i + 2u32) as usize];
                        let at = grp_cnt[(slot * nb + g) as usize].fetch_add(1u32);
                        let r8 =
                            TOP_REC_WORDS * (slot * ntlay + tlay[(5u32 * i + 3u32) as usize] + at);
                        top_rec[r8 as usize] = u32::cast_from(p);
                        top_rec[(r8 + 1u32) as usize] = u32::cast_from(p >> 32u64);
                        top_rec[(r8 + 2u32) as usize] = u32::cast_from(mask);
                        top_rec[(r8 + 3u32) as usize] = u32::cast_from(mask >> 32u64);
                        top_rec[(r8 + 4u32) as usize] = a[0] - pw0;
                        top_rec[(r8 + 5u32) as usize] = e[0] - pw0 + 1u32;
                        let pmod = tlay[(5u32 * i + 4u32) as usize] * pdiv + v;
                        top_rec[(r8 + 6u32) as usize] = pmod;
                        let mut seed = 0u32;
                        if floor >= k2 {
                            seed = 1u32;
                        }
                        top_rec[(r8 + 7u32) as usize] = seed;
                    }
                }
            }
        }
        cube_done(ran, 0u32);
    }

    // -----------------------------------------------------------------
    // The top kernel by finite differences over a batch's partitions.
    // -----------------------------------------------------------------

    /// `x += c` for a digit constant `c < b`, the packed digits of `x` over
    /// `kd` words (four to a u32, as `cubecl_detailed_fd`'s): `fd_add_top`
    /// with a one-digit addend, the carry out of word 0 rippled through the
    /// others only when there is one.
    #[cube]
    fn fd_add_small(
        x: &mut Array<u32>,
        #[comptime] c: u32,
        #[comptime] kd: u32,
        #[comptime] base: u32,
    ) {
        let bias = comptime!((256 - base) * 0x0101_0101u32);
        let fix = comptime!(256 - base);
        let t0 = x[0] + comptime!(bias + c);
        let mut cin = (t0 >> 31u32) ^ 1u32;
        x[0] = t0 - ((t0 >> 7u32) & 0x0101_0101u32) * fix;
        // A word plus the bias and no carry comes back unchanged.
        if comptime!(kd > 1) {
            if cin != 0u32 {
                #[unroll]
                for w in 1..kd {
                    let t = x[w as usize] + bias + cin;
                    cin = (t >> 31u32) ^ 1u32;
                    x[w as usize] = t - ((t >> 7u32) & 0x0101_0101u32) * fix;
                }
            }
        }
    }

    /// `fd_add_top`'s biased add of `y`'s `wy` packed-digit words into `x`
    /// (`kd` words), with the carry past `y`'s words propagated only when
    /// there is one.
    #[cube]
    fn fd_add_carry(
        x: &mut Array<u32>,
        y: &Array<u32>,
        #[comptime] wy: u32,
        #[comptime] kd: u32,
        #[comptime] base: u32,
    ) {
        let bias = comptime!((256 - base) * 0x0101_0101u32);
        let fix = comptime!(256 - base);
        let mut cin = 0u32;
        #[unroll]
        for w in 0..wy {
            let t = x[w as usize] + y[w as usize] + bias + cin;
            cin = (t >> 31u32) ^ 1u32;
            x[w as usize] = t - ((t >> 7u32) & 0x0101_0101u32) * fix;
        }
        if comptime!(wy < kd) {
            if cin != 0u32 {
                #[unroll]
                for w in wy..kd {
                    let t = x[w as usize] + bias + cin;
                    cin = (t >> 31u32) ^ 1u32;
                    x[w as usize] = t - ((t >> 7u32) & 0x0101_0101u32) * fix;
                }
            }
        }
    }

    /// The `q` packed base-b digits (four to a u32) of `v`, a value below
    /// b^`q` in 16-bit limbs one to a u32 (`nl` of them, destroyed), into
    /// `out` (pre-zeroed): split16 long division by `chunk_div` with
    /// `cert_power`'s limb trim (before pass i the value is below
    /// b^(q - i·cd), so only that power's limb count can be nonzero) and its
    /// Apple-safe first step (the top two stored limbs as one u32).
    #[cube]
    fn fd_digits_trim(
        v: &mut Array<u32>,
        out: &mut Array<u32>,
        #[comptime] nl: u32,
        #[comptime] q: u32,
        #[comptime] base: u32,
        #[comptime] chunk_digits: u32,
        #[comptime] chunk_div: u32,
    ) {
        let passes = comptime!(q.div_ceil(chunk_digits));
        #[unroll]
        for pass in 0..passes {
            let lp =
                comptime!((pow_limbs16(base, q - pass * chunk_digits).len() as u32).clamp(2, nl));
            let top = comptime!(lp - 1);
            let next = comptime!(lp - 2);
            let cx = (v[top as usize] << 16u32) | v[next as usize];
            let qx = cx / chunk_div;
            let mut rx = cx - qx * chunk_div;
            v[top as usize] = qx >> 16u32;
            v[next as usize] = qx & 0xFFFFu32;
            #[unroll]
            for jj in 2..lp {
                let j = comptime!(lp - 1 - jj);
                let c1 = (rx << 16u32) | v[j as usize];
                let q1 = c1 / chunk_div;
                rx = c1 - q1 * chunk_div;
                v[j as usize] = q1;
            }
            #[unroll]
            for i in 0..chunk_digits {
                let pos = comptime!(pass * chunk_digits + i);
                if comptime!(pos < q) {
                    let dx = rx % base;
                    rx /= base;
                    out[comptime!(pos / 4) as usize] |= dx << comptime!(8 * (pos % 4));
                }
            }
        }
    }

    /// Seed the finite-difference state at P (< 2^64; the block's P^k below
    /// `b^(q_k)`, P below b^`tq`): packed digits of P² (`sq`), P³ (`cu`),
    /// 2P + 1 (`ds`), 3P² + 3P + 1 (`dc`) and 6P + 6 (`ddc`). P, P², P³ by
    /// schoolbook in 16-bit limbs and [`fd_digits_trim`]; the three
    /// differences from those digits by biased adds.
    #[cube]
    fn top_fd_seed(
        p: u64,
        sq: &mut Array<u32>,
        cu: &mut Array<u32>,
        ds: &mut Array<u32>,
        dc: &mut Array<u32>,
        ddc: &mut Array<u32>,
        #[comptime] base: u32,
        #[comptime] tq: u32,
        #[comptime] q2: u32,
        #[comptime] q3: u32,
        #[comptime] chunk_digits: u32,
        #[comptime] chunk_div: u32,
    ) {
        let wq1 = comptime!((tq + 1).div_ceil(4));
        let wq2 = comptime!(q2.div_ceil(4));
        let wdc = comptime!((q2 + 1).div_ceil(4));
        let wq3 = comptime!(q3.div_ceil(4));
        let lo = u32::cast_from(p);
        let hi = u32::cast_from(p >> 32u64);
        let mut ph = Array::<u32>::new(4usize);
        ph[0] = lo & 0xFFFFu32;
        ph[1] = lo >> 16u32;
        ph[2] = hi & 0xFFFFu32;
        ph[3] = hi >> 16u32;
        let mut p2 = Array::<u32>::new(8usize);
        #[unroll]
        for j in 0..8u32 {
            p2[j as usize] = 0u32;
        }
        #[unroll]
        for i1 in 0..4u32 {
            let mut ca = 0u32;
            #[unroll]
            for j in 0..4u32 {
                let kk = comptime!(i1 + j);
                let t = ph[i1 as usize] * ph[j as usize] + p2[kk as usize] + ca;
                p2[kk as usize] = t & 0xFFFFu32;
                ca = t >> 16u32;
            }
            p2[comptime!(i1 + 4) as usize] = ca;
        }
        let mut p3 = Array::<u32>::new(12usize);
        #[unroll]
        for j in 0..12u32 {
            p3[j as usize] = 0u32;
        }
        #[unroll]
        for i1 in 0..8u32 {
            let mut ca = 0u32;
            #[unroll]
            for j in 0..4u32 {
                let kk = comptime!(i1 + j);
                let t = p2[i1 as usize] * ph[j as usize] + p3[kk as usize] + ca;
                p3[kk as usize] = t & 0xFFFFu32;
                ca = t >> 16u32;
            }
            p3[comptime!(i1 + 4) as usize] = ca;
        }
        let mut pd = Array::<u32>::new(wq1 as usize);
        #[unroll]
        for w in 0..wq1 {
            pd[w as usize] = 0u32;
        }
        #[unroll]
        for w in 0..wq2 {
            sq[w as usize] = 0u32;
        }
        #[unroll]
        for w in 0..wq3 {
            cu[w as usize] = 0u32;
        }
        fd_digits_trim(&mut ph, &mut pd, 4u32, tq, base, chunk_digits, chunk_div);
        fd_digits_trim(&mut p2, sq, 8u32, q2, base, chunk_digits, chunk_div);
        fd_digits_trim(&mut p3, cu, 12u32, q3, base, chunk_digits, chunk_div);
        // 2P + 1.
        #[unroll]
        for w in 0..wq1 {
            ds[w as usize] = pd[w as usize];
        }
        fd_add_top(ds, &pd, wq1, wq1, base);
        fd_add_small(ds, 1u32, wq1, base);
        // 3(P² + P) + 1.
        let mut u = Array::<u32>::new(wdc as usize);
        #[unroll]
        for w in 0..wdc {
            if comptime!(w < wq2) {
                u[w as usize] = sq[w as usize];
            } else {
                u[w as usize] = 0u32;
            }
        }
        fd_add_top(&mut u, &pd, wq1, wdc, base);
        #[unroll]
        for w in 0..wdc {
            dc[w as usize] = u[w as usize];
        }
        fd_add_top(dc, &u, wdc, wdc, base);
        fd_add_top(dc, &u, wdc, wdc, base);
        fd_add_small(dc, 1u32, wdc, base);
        // 6(P + 1).
        fd_add_small(&mut pd, 1u32, wq1, base);
        #[unroll]
        for w in 0..wq1 {
            ddc[w as usize] = pd[w as usize];
        }
        fd_add_top(ddc, &pd, wq1, wq1, base);
        fd_add_top(ddc, &pd, wq1, wq1, base);
        let mut u3 = Array::<u32>::new(wq1 as usize);
        #[unroll]
        for w in 0..wq1 {
            u3[w as usize] = ddc[w as usize];
        }
        fd_add_top(ddc, &u3, wq1, wq1, base);
    }

    /// One step of the state from P - 1 to P: S += dS, dS += 2, C += dC,
    /// dC += ddC, ddC += 6 (`cubecl_detailed_fd`'s recurrence on P).
    #[cube]
    fn top_fd_step(
        sq: &mut Array<u32>,
        cu: &mut Array<u32>,
        ds: &mut Array<u32>,
        dc: &mut Array<u32>,
        ddc: &mut Array<u32>,
        #[comptime] base: u32,
        #[comptime] tq: u32,
        #[comptime] q2: u32,
        #[comptime] q3: u32,
    ) {
        let wq1 = comptime!((tq + 1).div_ceil(4));
        let wq2 = comptime!(q2.div_ceil(4));
        let wdc = comptime!((q2 + 1).div_ceil(4));
        let wq3 = comptime!(q3.div_ceil(4));
        fd_add_carry(sq, ds, wq1, wq2, base);
        fd_add_small(ds, 2u32, wq1, base);
        fd_add_carry(cu, dc, wdc, wq3, base);
        fd_add_carry(dc, ddc, wq1, wdc, base);
        fd_add_small(ddc, 6u32, wq1, base);
    }

    /// `x += y` over `nx` u32 words (`y` has `ny` <= `nx`), the carry out of
    /// the top word dropped (the caller's values fit); a carry past `y`'s
    /// words is propagated only when there is one.
    #[cube]
    fn add_words(x: &mut Array<u32>, y: &Array<u32>, #[comptime] ny: u32, #[comptime] nx: u32) {
        let mut c = 0u64;
        #[unroll]
        for j in 0..ny {
            c += u64::cast_from(x[j as usize]) + u64::cast_from(y[j as usize]);
            x[j as usize] = u32::cast_from(c);
            c >>= 32u64;
        }
        if comptime!(ny < nx) {
            if c != 0u64 {
                #[unroll]
                for j in ny..nx {
                    c += u64::cast_from(x[j as usize]);
                    x[j as usize] = u32::cast_from(c);
                    c >>= 32u64;
                }
            }
        }
    }

    /// `x -= y` over `nx` u32 words (`y` has `ny` <= `nx`); returns the borrow
    /// out of the top word. A borrow past `y`'s words is propagated only when
    /// there is one.
    #[cube]
    fn sub_words(
        x: &mut Array<u32>,
        y: &Array<u32>,
        #[comptime] ny: u32,
        #[comptime] nx: u32,
    ) -> u32 {
        // x + !y + 1, carried as `add_words` carries.
        let mut c = 1u64;
        #[unroll]
        for j in 0..ny {
            c += u64::cast_from(x[j as usize]) + u64::cast_from(y[j as usize] ^ 0xFFFF_FFFFu32);
            x[j as usize] = u32::cast_from(c);
            c >>= 32u64;
        }
        if comptime!(ny < nx) {
            if c == 0u64 {
                #[unroll]
                for j in ny..nx {
                    c += u64::cast_from(x[j as usize]) + 0xFFFF_FFFFu64;
                    x[j as usize] = u32::cast_from(c);
                    c >>= 32u64;
                }
            }
        }
        1u32 - u32::cast_from(c)
    }

    /// Words that bound the binary steps for blocks of width w = b^f0 (δ =
    /// w - 1), each at most its array's ([`top_fd_consts`]: `c2` 3 words,
    /// `k3` 4) and the difference's (`nl2`, `nl3` = 2·`limbs`, 3·`limbs`):
    /// c2 = 2wδ < 2w², k3 = 6w²δ < 6w³, and Δ3 = 3wδ(2a + 2w - 1) <
    /// 3w²·2^(32·limbs + 2) for a block start a < 2^(32·limbs).
    fn fd_step_words(base: u32, f0: u32, limbs: u32) -> (u32, u32, u32) {
        let w = u128::from(base).pow(f0);
        let bits = |x: u128| 128 - x.leading_zeros();
        (
            bits(2 * w * w).div_ceil(32).min(3).min(2 * limbs),
            bits(6 * w * w * w).div_ceil(32).min(4).min(3 * limbs),
            (bits(3 * w * w) + 32 * limbs + 2)
                .div_ceil(32)
                .min(3 * limbs),
        )
    }

    /// `out = b^c - 1 - x` over `nl` words (mod 2^(32·nl)), b^c from the
    /// power table `pw` (`nl3` words a power); it is its own inverse.
    #[cube]
    fn pw_minus1(
        pw: &Array<u32>,
        c: u32,
        x: &Array<u32>,
        out: &mut Array<u32>,
        #[comptime] nl: u32,
        #[comptime] nl3: u32,
    ) {
        let mut cy = 0u64;
        #[unroll]
        for j in 0..nl {
            cy += u64::cast_from(pw[(c * nl3 + j) as usize])
                + u64::cast_from(x[j as usize] ^ 0xFFFF_FFFFu32);
            out[j as usize] = u32::cast_from(cy);
            cy >>= 32u64;
        }
    }

    /// The base-b value of a word's four packed digits.
    #[cube]
    fn word_value(word: u32, #[comptime] base: u32) -> u32 {
        (word & 0xFFu32)
            + comptime!(base)
                * (((word >> 8u32) & 0xFFu32)
                    + comptime!(base)
                        * (((word >> 16u32) & 0xFFu32) + comptime!(base) * (word >> 24u32)))
    }

    /// `acc += v · b^p0` over `nl` words, for a sum below b^(p0 + 4).
    #[cube]
    fn add_scaled_pow(
        acc: &mut Array<u32>,
        v: u32,
        #[comptime] nl: u32,
        #[comptime] base: u32,
        #[comptime] p0: u32,
    ) {
        let len = comptime!((pow_words32(base, p0).len() as u32).min(nl));
        let topk = comptime!((pow_words32(base, p0 + 4).len() as u32).min(nl));
        let mut c = 0u64;
        #[unroll]
        for k in 0..len {
            let t = u64::cast_from(v) * comptime!(u64::from(pow_words32(base, p0)[k as usize]))
                + u64::cast_from(acc[k as usize])
                + c;
            acc[k as usize] = u32::cast_from(t);
            c = t >> 32u64;
        }
        #[unroll]
        for k in len..topk {
            let t = u64::cast_from(acc[k as usize]) + c;
            acc[k as usize] = u32::cast_from(t);
            c = t >> 32u64;
        }
    }

    /// The binary differences' constants for blocks of width w (δ = w - 1):
    /// `c2` = 2wδ (3 words), the step of d2 = e² - a², and `k3` = 6w²δ (4
    /// words), the second difference of d3 = e³ - a³.
    #[cube]
    fn top_fd_consts(w: u32, c2: &mut Array<u32>, k3: &mut Array<u32>) {
        let wd = u64::cast_from(w - 1u32) * u64::cast_from(w);
        let lo = (wd & 0xFFFF_FFFFu64) * 2u64;
        let hi = (wd >> 32u64) * 2u64 + (lo >> 32u64);
        c2[0] = u32::cast_from(lo);
        c2[1] = u32::cast_from(hi);
        c2[2] = u32::cast_from(hi >> 32u64);
        // w·wd in 3 words, then ·6.
        let m0 = (wd & 0xFFFF_FFFFu64) * u64::cast_from(w);
        let m1 = (wd >> 32u64) * u64::cast_from(w) + (m0 >> 32u64);
        let x0 = u64::cast_from(u32::cast_from(m0)) * 6u64;
        let x1 = u64::cast_from(u32::cast_from(m1)) * 6u64 + (x0 >> 32u64);
        let x2 = (m1 >> 32u64) * 6u64 + (x1 >> 32u64);
        k3[0] = u32::cast_from(x0);
        k3[1] = u32::cast_from(x1);
        k3[2] = u32::cast_from(x2);
        k3[3] = u32::cast_from(x2 >> 32u64);
    }

    /// Δ3 = d3(P + 1) - d3(P) = 3wδ·(2a + 2w - 1) for the unclipped block
    /// a = P·w (`limbs` <= 3 words), into `del` (`nl3` words, zeroed here).
    #[cube]
    fn top_fd_del3(
        a: &Array<u32>,
        w: u32,
        del: &mut Array<u32>,
        #[comptime] limbs: u32,
        #[comptime] nl3: u32,
    ) {
        // X = 2a + 2w - 1 (4 words).
        let mut x = Array::<u32>::new(4usize);
        let mut c = u64::cast_from(w) * 2u64 - 1u64;
        #[unroll]
        for j in 0..4u32 {
            if comptime!(j < limbs) {
                c += u64::cast_from(a[j as usize]) * 2u64;
            }
            x[j as usize] = u32::cast_from(c);
            c >>= 32u64;
        }
        // M = 3wδ (3 words).
        let wd = u64::cast_from(w - 1u32) * u64::cast_from(w);
        let lo = (wd & 0xFFFF_FFFFu64) * 3u64;
        let hi = (wd >> 32u64) * 3u64 + (lo >> 32u64);
        let mut m = Array::<u32>::new(3usize);
        m[0] = u32::cast_from(lo);
        m[1] = u32::cast_from(hi);
        m[2] = u32::cast_from(hi >> 32u64);
        #[unroll]
        for j in 0..nl3 {
            del[j as usize] = 0u32;
        }
        #[unroll]
        for i1 in 0..3u32 {
            let mut cc = 0u64;
            #[unroll]
            for j in 0..4u32 {
                let kk = comptime!(i1 + j);
                if comptime!(kk < nl3) {
                    let t = u64::cast_from(m[i1 as usize]) * u64::cast_from(x[j as usize])
                        + u64::cast_from(del[kk as usize])
                        + cc;
                    del[kk as usize] = u32::cast_from(t);
                    cc = t >> 32u64;
                }
            }
            if comptime!(i1 + 4 < nl3) {
                del[comptime!(i1 + 4) as usize] = u32::cast_from(cc);
            }
        }
    }

    /// [`cert_power`] with the power's digits read from the finite-difference
    /// state instead of divided out: x = Q·b^z (Q's packed digits in `q`,
    /// four to a word; the `z` digits below are zero), `c0` = max(cap,
    /// ndig(y - x) clamped to sp) and `dm` = b^c0 - 1 - (y - x) (`nl` u32
    /// words), both kept by the caller. The same results from the same digits
    /// as `cert_power`, by a scan over Q's words from the bottom: a word
    /// wholly below c0 only adds its value times b^pos to x mod b^c0 (u32
    /// words here; nothing divides it), and the digits of the others go
    /// through `cert_power`'s scan step in order, those of a word above the
    /// end of x's run of (b - 1)s from c0 up without the run's bookkeeping.
    /// The carry is x mod b^c0 > dm. The repeat flags come from
    /// the masks' popcounts at the end (a digit repeats among the n positions
    /// a mask covers iff it has fewer than n bits), and the end of x's run of
    /// (b - 1)s from c0 up is kept as a position. The zero digits at c0..z, if
    /// c0 < z (never for an unclipped block with P >= 1, where d2 >= w² and d3
    /// >= w³), are put in before the scan: the first ends the run, the rest
    /// lie above it.
    #[cube]
    pub(super) fn cert_power_fd(
        q: &Array<u32>,
        dm: &Array<u32>,
        c0: u32,
        out_dup: &mut Array<u32>,
        out_dom: &mut Array<u32>,
        #[comptime] nl: u32,
        #[comptime] sp: u32,
        #[comptime] base: u32,
        #[comptime] z: u32,
    ) -> u64 {
        let mut rem = Array::<u32>::new(nl as usize);
        #[unroll]
        for j in 0..nl {
            rem[j as usize] = 0u32;
        }
        let mut run = 0u64;
        let mut runc = 0u64;
        let mut in_run = true;
        let mut dlo = 0u32;
        let mut qend = 0u32;
        if c0 < z {
            let m = z.runtime() - c0;
            run = 1u64;
            runc = select(m >= 2u32, 1u64, 0u64);
            in_run = false;
            qend = c0;
        }
        let nw = comptime!((sp - z).div_ceil(4));
        #[unroll]
        for j in 0..nw {
            let p0 = comptime!(z + 4 * j);
            let wend = comptime!(if p0 + 4 < sp { p0 + 4 } else { sp });
            let word = q[j as usize];
            let mut low = 0u32;
            if c0 >= wend {
                // Every digit of the word below c0 (those at sp and up are 0).
                low = word_value(word, base);
            } else if in_run {
                #[unroll]
                for i in 0..4u32 {
                    let pos = comptime!(p0 + i);
                    if comptime!(pos < sp) {
                        let dx = (word >> comptime!(8 * i)) & 0xFFu32;
                        let above = c0 <= pos;
                        let bit = select(above, 1u64 << u64::cast_from(dx), 0u64);
                        run |= bit;
                        runc |= select(in_run, 0u64, bit);
                        let ends = above && in_run && dx != comptime!(base - 1);
                        dlo = select(ends, dx, dlo);
                        qend = select(ends, pos.runtime(), qend);
                        in_run = in_run && !ends;
                        low += select(above, 0u32, dx * comptime!(base.pow(i)));
                    }
                }
            } else {
                // x's run has ended (in a lower word, or before the scan when
                // c0 < z): every digit here lies above c0.
                let mut wb = 0u64;
                #[unroll]
                for i in 0..4u32 {
                    let pos = comptime!(p0 + i);
                    if comptime!(pos < sp) {
                        let dx = (word >> comptime!(8 * i)) & 0xFFu32;
                        wb |= 1u64 << u64::cast_from(dx);
                    }
                }
                run |= wb;
                runc |= wb;
            }
            // rem += low · b^p0 (the sum is below b^(p0 + 4)).
            if p0 < c0 {
                add_scaled_pow(&mut rem, low, nl, base, p0);
            }
        }
        let mut nabove = 0u32;
        if c0 < sp {
            nabove = sp.runtime() - c0;
        }
        let prun = u32::cast_from(run).count_ones() + u32::cast_from(run >> 32u64).count_ones();
        let dup = select(prun != nabove, 1u32, 0u32);
        let mut nc = 0u32;
        if !in_run {
            nc = comptime!(sp - 1).runtime() - qend;
        }
        let pruc = u32::cast_from(runc).count_ones() + u32::cast_from(runc >> 32u64).count_ones();
        let dupc = select(pruc != nc, 1u32, 0u32);
        // floor(y / b^c0) is one more than floor(x / b^c0) iff (x mod b^c0) +
        // (y - x) >= b^c0, i.e. x mod b^c0 > dm: dm - rem borrows (only the
        // borrow is kept).
        let mut one_more = false;
        if c0 < sp {
            let mut c = 1u64;
            #[unroll]
            for j in 0..nl {
                c += u64::cast_from(dm[j as usize])
                    + u64::cast_from(rem[j as usize] ^ 0xFFFF_FFFFu32);
                c >>= 32u64;
            }
            one_more = c == 0u64;
        }
        let mut dom_lo = 1u32;
        let mut dom_hi = 0u32;
        let mut dupo = dup;
        if one_more {
            run = runc;
            dupo = dupc;
            dom_lo = dlo;
            dom_hi = dlo + 1u32;
        }
        out_dup[0] = dupo;
        out_dom[0] = dom_lo;
        out_dom[1] = dom_hi;
        run
    }

    /// [`top_cert`] for an unclipped block a = P·b^f0, e = a + b^f0 - 1 from
    /// the state's P² and P³ digits, the c0s (both at least their power's
    /// zero digits) and b^c0 - 1 - (e^k - a^k) of each power (`dm2`, `dm3`).
    #[cube]
    pub(super) fn top_cert_fd(
        sq: &Array<u32>,
        cu: &Array<u32>,
        dm2: &Array<u32>,
        dm3: &Array<u32>,
        c0a: u32,
        c0b: u32,
        res: &mut Array<u32>,
        #[comptime] base: u32,
        #[comptime] limbs: u32,
        #[comptime] s2: u32,
        #[comptime] s3: u32,
        #[comptime] f0: u32,
    ) -> u64 {
        let nl2 = comptime!(2 * limbs);
        let nl3 = comptime!(3 * limbs);
        let mut dupa = Array::<u32>::new(1usize);
        let mut dupb = Array::<u32>::new(1usize);
        let mut doma = Array::<u32>::new(2usize);
        let mut domb = Array::<u32>::new(2usize);
        let msq = cert_power_fd(
            sq,
            dm2,
            c0a,
            &mut dupa,
            &mut doma,
            nl2,
            s2,
            base,
            comptime!(2 * f0),
        );
        let mcu = cert_power_fd(
            cu,
            dm3,
            c0b,
            &mut dupb,
            &mut domb,
            nl3,
            s3,
            base,
            comptime!(3 * f0),
        );
        close_cert(msq, mcu, dupa[0], dupb[0], &doma, &domb, c0a, c0b, res)
    }

    /// [`top_kernel`] by finite differences over the batch's partitions:
    /// grid x over the top layer, y over groups of `slot_k` slots (one group
    /// unless the top layer is small); each thread takes its prefix through
    /// its group's slots in order. An unclipped block a = P·w (w = b^f0) has
    /// a^k = P^k·b^(k·f0), and consecutive partition values are consecutive
    /// P, so the digits of P² and P³ step from slot to slot by additions
    /// ([`top_fd_step`]), as do each power's c0 and b^c0 - 1 - (e^k - a^k) in
    /// binary; they are divided out ([`top_fd_seed`], [`powers_diff`]) only
    /// for a group's first unclipped slot, after a gap in `vs` and after a
    /// clipped or out-of-range slot. Clipped blocks (P = plo, phi: the
    /// slice's ends) are left to [`top_kernel`] with `only_clipped`, launched
    /// beside this one.
    /// The certificate is [`cert_power_fd`]'s, the same function of the same
    /// digits as [`cert_power`]'s; the outputs are [`top_kernel`]'s.
    #[cube(launch_unchecked)]
    pub fn top_fd_kernel(
        tlay: &Array<u32>,
        vs: &Array<u32>,
        fb: &Array<u32>,
        pw: &Array<u32>,
        top_rec: &mut Array<u32>,
        grp_cnt: &mut Array<Atomic<u32>>,
        ran: &mut Array<Atomic<u32>>,
        ntlay: u32,
        w: u32,
        pdiv: u32,
        full_floor: u32,
        nslots: u32,
        slot_k: u32,
        #[comptime] base: u32,
        #[comptime] limbs: u32,
        #[comptime] cap: u32,
        #[comptime] k2: u32,
        #[comptime] s2: u32,
        #[comptime] s3: u32,
        #[comptime] chunk_digits: u32,
        #[comptime] chunk_div: u32,
        #[comptime] nb: u32,
        #[comptime] f0: u32,
        #[comptime] tq: u32,
        #[comptime] ballot: bool,
    ) {
        let nl2 = comptime!(2 * limbs);
        let nl3 = comptime!(3 * limbs);
        let nh2 = comptime!(4 * limbs);
        let nh3 = comptime!(6 * limbs);
        let q2 = comptime!(s2 - 2 * f0);
        let q3 = comptime!(s3 - 3 * f0);
        let wq1 = comptime!((tq + 1).div_ceil(4));
        let wq2 = comptime!(q2.div_ceil(4));
        let wdc = comptime!((q2 + 1).div_ceil(4));
        let wq3 = comptime!(q3.div_ceil(4));
        // Every unit runs the slot loop (the tops are appended per plane at
        // the end of each slot, so every lane of a plane must get there).
        let i = ABSOLUTE_POS_X;
        let live = i < ntlay;
        let mut p0 = 0u64;
        let mut tl_g = 0u32;
        let mut tl_off = 0u32;
        let mut tl_mod = 0u32;
        if live {
            p0 = (u64::cast_from(tlay[(5u32 * i + 1u32) as usize]) << 32u64)
                | u64::cast_from(tlay[(5u32 * i) as usize]);
            tl_g = tlay[(5u32 * i + 2u32) as usize];
            tl_off = tlay[(5u32 * i + 3u32) as usize];
            tl_mod = tlay[(5u32 * i + 4u32) as usize];
        }
        // Unclipped blocks only: r is (0, w) and the seed flag is the
        // field floor's.
        let mut seedf = 0u32;
        if full_floor >= comptime!(k2) {
            seedf = 1u32;
        }
        {
            let plo = (u64::cast_from(fb[7]) << 32u64) | u64::cast_from(fb[6]);
            let phi = (u64::cast_from(fb[9]) << 32u64) | u64::cast_from(fb[8]);
            let mut sq = Array::<u32>::new(wq2 as usize);
            let mut cu = Array::<u32>::new(wq3 as usize);
            let mut ds = Array::<u32>::new(wq1 as usize);
            let mut dc = Array::<u32>::new(wdc as usize);
            let mut ddc = Array::<u32>::new(wq1 as usize);
            #[unroll]
            for j in 0..wq2 {
                sq[j as usize] = 0u32;
            }
            #[unroll]
            for j in 0..wq3 {
                cu[j as usize] = 0u32;
            }
            #[unroll]
            for j in 0..wq1 {
                ds[j as usize] = 0u32;
                ddc[j as usize] = 0u32;
            }
            #[unroll]
            for j in 0..wdc {
                dc[j as usize] = 0u32;
            }
            // d2 = e² - a², d3 = e³ - a³ and Δ3 = d3(P + 1) - d3(P): d2 grows
            // by 2wδ a step, d3 by Δ3, and Δ3 by 6w²δ. What is stepped is
            // each power's c0 = max(cap, ndig(d)) (it rarely moves) and dm
            // = b^c0 - 1 - d, the bound of the carry test.
            let mut d2 = Array::<u32>::new(nl2 as usize);
            let mut d3 = Array::<u32>::new(nl3 as usize);
            let mut del3 = Array::<u32>::new(nl3 as usize);
            let mut c2 = Array::<u32>::new(3usize);
            let mut k3 = Array::<u32>::new(4usize);
            top_fd_consts(w, &mut c2, &mut k3);
            #[unroll]
            for j in 0..nl3 {
                d3[j as usize] = 0u32;
                del3[j as usize] = 0u32;
            }
            #[unroll]
            for j in 0..nl2 {
                d2[j as usize] = 0u32;
            }
            let mut c0a = 0u32;
            let mut c0b = 0u32;
            let mut dm2 = Array::<u32>::new(nl2 as usize);
            let mut dm3 = Array::<u32>::new(nl3 as usize);
            #[unroll]
            for j in 0..nl2 {
                dm2[j as usize] = 0u32;
            }
            #[unroll]
            for j in 0..nl3 {
                dm3[j as usize] = 0u32;
            }
            let mut valid = false;
            let mut prev_v = 0u32;
            // This cube's group of slots (`slot_k` of them; one group, all
            // the slots, when the top layer alone fills the device).
            let mut slot = CUBE_POS_Y * slot_k;
            let mut slot_end = slot + slot_k;
            if slot_end > nslots {
                slot_end = nslots;
            }
            while slot < slot_end {
                let v = vs[slot as usize];
                let p = p0 * u64::cast_from(pdiv) + u64::cast_from(v);
                let mut kept = false;
                let mut mask = 0u64;
                if live && p >= plo && p <= phi {
                    let mut a = Array::<u32>::new(limbs as usize);
                    let mut e = Array::<u32>::new(limbs as usize);
                    let mut a_lt_s = false;
                    let mut e_gt_e = false;
                    // Only P = plo and P = phi can be clipped, and a slot that
                    // steps needs neither a nor e (they feed a seed).
                    let interior_step = valid && v == prev_v + 1u32 && p != plo && p != phi;
                    if !interior_step {
                        // a = p*w (< 2^96, 3 limbs), e = a + w - 1, clipped to [s, e-1].
                        let t0 = (p & 0xFFFF_FFFFu64) * u64::cast_from(w);
                        let t1 = (p >> 32u64) * u64::cast_from(w) + (t0 >> 32u64);
                        let pw0 = u32::cast_from(t0);
                        #[unroll]
                        for j in 0..limbs {
                            if comptime!(j == 0) {
                                a[j as usize] = pw0;
                            } else if comptime!(j == 1) {
                                a[j as usize] = u32::cast_from(t1);
                            } else if comptime!(j == 2) {
                                a[j as usize] = u32::cast_from(t1 >> 32u64);
                            } else {
                                a[j as usize] = 0u32;
                            }
                        }
                        let mut carry = w - 1u32;
                        #[unroll]
                        for j in 0..limbs {
                            let t = a[j as usize] + carry;
                            let mut c = 0u32;
                            if t < carry {
                                c = 1u32;
                            }
                            e[j as usize] = t;
                            carry = c;
                        }
                        let mut eq_a = true;
                        let mut eq_e = true;
                        #[unroll]
                        for jj in 0..limbs {
                            let j = comptime!(limbs - 1 - jj);
                            let sj = fb[j as usize];
                            let ej = fb[comptime!(3 + j) as usize];
                            if eq_a {
                                if a[j as usize] < sj {
                                    a_lt_s = true;
                                    eq_a = false;
                                } else if a[j as usize] > sj {
                                    eq_a = false;
                                }
                            }
                            if eq_e {
                                if e[j as usize] > ej {
                                    e_gt_e = true;
                                    eq_e = false;
                                } else if e[j as usize] < ej {
                                    eq_e = false;
                                }
                            }
                        }
                        if a_lt_s {
                            #[unroll]
                            for j in 0..limbs {
                                a[j as usize] = fb[j as usize];
                            }
                        }
                        if e_gt_e {
                            #[unroll]
                            for j in 0..limbs {
                                e[j as usize] = fb[comptime!(3 + j) as usize];
                            }
                        }
                    }
                    // Clipped blocks (P = plo, P = phi) are `top_kernel`'s
                    // (`only_clipped`), launched beside this one.
                    let clipped = a_lt_s || e_gt_e;
                    let mut res = Array::<u32>::new(3usize);
                    res[0] = 0u32;
                    if !clipped {
                        // The previous slot was P - 1, unclipped: step. Else
                        // (the batch's first slot, a gap, or after a clipped
                        // or out-of-range slot) seed from P by division.
                        let step = valid && v == prev_v + 1u32;
                        if step {
                            top_fd_step(
                                &mut sq, &mut cu, &mut ds, &mut dc, &mut ddc, base, tq, q2, q3,
                            );
                            // dm -= the step of d. A borrow means d reached
                            // b^c0 (rare): d = b^c0 - 1 - dm, c0 and dm anew.
                            let nc2 = comptime!(fd_step_words(base, f0, limbs).0);
                            let nk3 = comptime!(fd_step_words(base, f0, limbs).1);
                            let nd3 = comptime!(fd_step_words(base, f0, limbs).2);
                            let br2 = sub_words(&mut dm2, &c2, nc2, nl2);
                            let br3 = sub_words(&mut dm3, &del3, nd3, nl3);
                            add_words(&mut del3, &k3, nk3, nl3);
                            if c0a < s2 && br2 != 0u32 {
                                pw_minus1(pw, c0a, &dm2, &mut d2, nl2, nl3);
                                c0a = ndig_fast(&d2, pw, nl2, nl3, s2, base);
                                if c0a < cap {
                                    c0a = cap.runtime();
                                }
                                pw_minus1(pw, c0a, &d2, &mut dm2, nl2, nl3);
                            }
                            if c0b < s3 && br3 != 0u32 {
                                pw_minus1(pw, c0b, &dm3, &mut d3, nl3, nl3);
                                c0b = ndig_fast(&d3, pw, nl3, nl3, s3, base);
                                if c0b < cap {
                                    c0b = cap.runtime();
                                }
                                pw_minus1(pw, c0b, &d3, &mut dm3, nl3, nl3);
                            }
                        } else {
                            let mut a2 = Array::<u32>::new(nh2 as usize);
                            let mut a3 = Array::<u32>::new(nh3 as usize);
                            powers_diff(
                                &a, &e, &mut a2, &mut a3, &mut d2, &mut d3, limbs, 1u32, false,
                            );
                            top_fd_del3(&a, w, &mut del3, limbs, nl3);
                            top_fd_seed(
                                p,
                                &mut sq,
                                &mut cu,
                                &mut ds,
                                &mut dc,
                                &mut ddc,
                                base,
                                tq,
                                q2,
                                q3,
                                chunk_digits,
                                chunk_div,
                            );
                            c0a = ndig_fast(&d2, pw, nl2, nl3, s2, base);
                            if c0a < cap {
                                c0a = cap.runtime();
                            }
                            c0b = ndig_fast(&d3, pw, nl3, nl3, s3, base);
                            if c0b < cap {
                                c0b = cap.runtime();
                            }
                            pw_minus1(pw, c0a, &d2, &mut dm2, nl2, nl3);
                            pw_minus1(pw, c0b, &d3, &mut dm3, nl3, nl3);
                        }
                        mask = top_cert_fd(
                            &sq, &cu, &dm2, &dm3, c0a, c0b, &mut res, base, limbs, s2, s3, f0,
                        );
                    }
                    valid = !clipped;
                    prev_v = v;
                    kept = res[0] != 0u32;
                } else {
                    valid = false;
                }
                // Append the plane's kept tops into their groups' segments.
                let rec = top_slot(kept, tl_g, tl_off, slot, ntlay, grp_cnt, nb, ballot);
                if kept {
                    let r8 = TOP_REC_WORDS * rec;
                    top_rec[r8 as usize] = u32::cast_from(p);
                    top_rec[(r8 + 1u32) as usize] = u32::cast_from(p >> 32u64);
                    top_rec[(r8 + 2u32) as usize] = u32::cast_from(mask);
                    top_rec[(r8 + 3u32) as usize] = u32::cast_from(mask >> 32u64);
                    top_rec[(r8 + 4u32) as usize] = 0u32;
                    top_rec[(r8 + 5u32) as usize] = w;
                    top_rec[(r8 + 6u32) as usize] = tl_mod * pdiv + v;
                    top_rec[(r8 + 7u32) as usize] = seedf;
                }
                slot += 1u32;
            }
        }
        cube_done(ran, 0u32);
    }

    /// The middle-digit prefilter of one join survivor, in [`join_kernel`]'s
    /// walk: n's output digits at positions `k..k2` of n² and n³ must be
    /// distinct, and miss the bottom's (`mlo | mhi << 32`, positions `0..k`,
    /// distinct by construction) and the top's certificate (`tlo | thi <<
    /// 32`) when its seed flag says every certified position is `>= k2`.
    /// `ac` = `⌊R²/b^k⌋ | ⌊R³/b^k⌋ << 16` and `pc` = `2R | 3R² << 16` (each
    /// mod `b^mid`; `R mod b^mid` for `2R` when `k < mid`) are the entry's,
    /// `dmv` = `D | seed << 16` the top's. The same test as
    /// `cpu_join::CpuJoin::prefilter`, which has the derivation.
    ///
    /// Every division here divides a sum or a remainder, never a bare
    /// half-width shift (see [`cert_power`] for why that matters).
    #[cube]
    fn walk_pf(
        ac: u32,
        pc: u32,
        dmv: u32,
        mlo: u32,
        mhi: u32,
        tlo: u32,
        thi: u32,
        #[comptime] base: u32,
        #[comptime] k: u32,
        #[comptime] k2: u32,
    ) -> bool {
        let bm = comptime!(base.pow(k2 - k));
        // b^k and b^2k mod b^mid: zero unless k < mid.
        let bk = comptime!(if k < k2 - k { base.pow(k) } else { 0 });
        let b2k = comptime!(if 2 * k < k2 - k { base.pow(2 * k) } else { 0 });
        let dm = dmv & 0xFFFFu32;
        let mut seen = (u64::cast_from(mhi) << 32u64) | u64::cast_from(mlo);
        if (dmv >> 16u32) != 0u32 {
            seen |= (u64::cast_from(thi) << 32u64) | u64::cast_from(tlo);
        }
        let mut x2 = (ac & 0xFFFFu32) + (pc & 0xFFFFu32) * dm;
        let mut x3 = (ac >> 16u32) + (pc >> 16u32) * dm;
        if comptime!(bk != 0) {
            // pc's low half is R mod b^mid here, so x2 holds R·D of 2R·D.
            // The rest: R·D, D²·b^k; 3R·D²·b^k and D³·b^2k (`prefilter_any`).
            let rho = pc & 0xFFFFu32;
            let d2 = (dm * dm) % bm;
            x2 += rho * dm + d2 * bk;
            x3 += ((3u32 * rho * d2) % bm) * bk + ((d2 * dm) % bm) * b2k;
        }
        x2 %= bm;
        x3 %= bm;
        let mut ok = true;
        if comptime!(k2 - k == 2) {
            let g2 = x2 % base;
            let g3 = x3 % base;
            let h2 = x2 / base;
            let h3 = x3 / base;
            let at_k = (1u64 << u64::cast_from(g2)) | (1u64 << u64::cast_from(g3));
            let at_k1 = (1u64 << u64::cast_from(h2)) | (1u64 << u64::cast_from(h3));
            if g2 == g3 || h2 == h3 || (seen & at_k) != 0u64 || ((seen | at_k) & at_k1) != 0u64 {
                ok = false;
            }
        } else {
            let at_k = (1u64 << u64::cast_from(x2)) | (1u64 << u64::cast_from(x3));
            if x2 == x3 || (seen & at_k) != 0u64 {
                ok = false;
            }
        }
        ok
    }

    /// The constants of [`walk_pf_lean`] at base b: `C = ceil(2^32/b²)`, and
    /// `Q`, `R` with `2^32 = Q·b² + R`.
    #[allow(clippy::cast_possible_truncation)]
    pub(super) fn fx_consts(base: u32) -> (u32, u32, u32) {
        let b2 = u64::from(base * base);
        (
            (1u64 << 32).div_ceil(b2) as u32,
            ((1u64 << 32) / b2) as u32,
            ((1u64 << 32) % b2) as u32,
        )
    }

    /// [`walk_pf`] at mid = 2 (k >= 2) without a division: `x mod b²` as a
    /// 32-bit fraction of b², `F = (ah + p·dh) mod 2^32` with `ah = A·C`,
    /// `C = ceil(2^32/b²)` (the entry's) and `dh = ceil(D·2^32/b²)` (the
    /// top's); then `F·b² = (x mod b²)·2^32 + Δ` with `Δ = A·(C·b² − 2^32) +
    /// P·(dh·b² − D·2^32) < 2b⁴ <= 2^25`, so the high words of `F·b` and of
    /// `(F·b mod 2^32)·b` are the two digits of `x mod b²` exactly. The masks
    /// are bit-reversed (digit d at bit 63 - d), so digit g is in `seen` iff
    /// bit 63 of `seen << g` is set: four shifts and a sign test, and six
    /// compares for the digits' distinctness. `tlo`/`thi` are the top's
    /// certificate if its seed flag is set, else 0.
    #[cube]
    fn walk_pf_lean(
        ah2: u32,
        ah3: u32,
        p2: u32,
        p3: u32,
        dh: u32,
        mlo: u32,
        mhi: u32,
        tlo: u32,
        thi: u32,
        #[comptime] base: u32,
    ) -> bool {
        let f2 = ah2 + p2 * dh;
        let f3 = ah3 + p3 * dh;
        let h2 = u32::mul_hi(f2, base);
        let g2 = u32::mul_hi(f2 * base, base);
        let h3 = u32::mul_hi(f3, base);
        let g3 = u32::mul_hi(f3 * base, base);
        let seen = (u64::cast_from(mhi | thi) << 32u64) | u64::cast_from(mlo | tlo);
        let hit = u32::cast_from((seen << u64::cast_from(g2)) >> 32u64)
            | u32::cast_from((seen << u64::cast_from(g3)) >> 32u64)
            | u32::cast_from((seen << u64::cast_from(h2)) >> 32u64)
            | u32::cast_from((seen << u64::cast_from(h3)) >> 32u64);
        let mut ok = true;
        // Keep the sign test last in the chain: placed first, NVVM compiled it
        // to a select and a byte extension in front of the compares.
        if g2 == g3
            || h2 == h3
            || g2 == h2
            || g2 == h3
            || g3 == h2
            || g3 == h3
            || i32::cast_from(hit) < 0i32
        {
            ok = false;
        }
        ok
    }

    /// Write the shared survivor stage to the list. Every staged pair passed
    /// the prefilter in the walk, so one atomic reserves the stage's place
    /// in `out` and the cube copies it there. Must be called by the whole
    /// cube at a point where no thread is still appending (after a barrier);
    /// `sc` is the stage count read after that barrier.
    #[cube]
    fn flush_survivors(
        s_t: &SharedMemory<u32>,
        s_r: &SharedMemory<u32>,
        s_cnt: &mut SharedMemory<Atomic<u32>>,
        s_base: &mut SharedMemory<u32>,
        out: &mut Array<u32>,
        out_count: &mut Array<Atomic<u32>>,
        out_cap: u32,
        sc: u32,
    ) {
        let mut n = sc;
        if n > SURV_SH_CAP {
            n = SURV_SH_CAP; // the rest went to `out` from the walk
        }
        if UNIT_POS_X == 0u32 {
            s_base[0] = out_count[0].fetch_add(n);
        }
        sync_cube();
        let at = s_base[0];
        let mut i = UNIT_POS_X;
        while i < n {
            let g = at + i;
            if g < out_cap {
                out[(2u32 * g) as usize] = s_t[i as usize];
                out[(2u32 * g + 1u32) as usize] = s_r[i as usize];
            }
            i += CUBE_DIM_X;
        }
        sync_cube();
        if UNIT_POS_X == 0u32 {
            s_cnt[0].store(0u32);
        }
    }

    /// Step 4: grid-stride over the survivor list; rebuild n = P*b^f0 + r0 and
    /// run the client's full check (probe: report every n instead).
    #[cube(launch_unchecked)]
    pub fn check_kernel(
        surv: &Array<u32>,
        surv_count: &mut Array<Atomic<u32>>,
        top_rec: &Array<u32>, // per top, P's lo, hi words first (`TOP_REC_WORDS`)
        nice_out: &mut Array<u32>,
        nice_count: &mut Array<Atomic<u32>>,
        ran: &mut Array<Atomic<u32>>,
        surv_cap: u32,
        nice_cap: u32,
        w_f0: u32, // b^f0
        #[comptime] base: u32,
        #[comptime] limbs: u32,
        #[comptime] chunk_digits: u32,
        #[comptime] chunk_div: u32,
        #[comptime] wide_chunk: bool,
        #[comptime] pre_limbs: u32,
        #[comptime] pre_chunk_digits: u32,
        #[comptime] pre_chunk_div: u32,
        #[comptime] probe: bool,
    ) {
        let cu_limbs = comptime!(3 * limbs);
        let sv_pad = comptime!(cu_limbs | 1);
        let mut sv_s = SharedMemory::<u32>::new(comptime!((JOIN_WG * (cu_limbs | 1)) as usize));
        let svb = UNIT_POS_X * sv_pad;
        let mut count = surv_count[0].load();
        if count > surv_cap {
            count = surv_cap;
        }
        let stride = CUBE_COUNT_X * CUBE_DIM_X;
        let mut i = ABSOLUTE_POS_X;
        while i < count {
            let t = surv[(2u32 * i) as usize];
            let r0 = surv[(2u32 * i + 1u32) as usize];
            let p_lo = u64::cast_from(top_rec[(TOP_REC_WORDS * t) as usize]);
            let p_hi = u64::cast_from(top_rec[(TOP_REC_WORDS * t + 1u32) as usize]);
            let wf = u64::cast_from(w_f0);
            // n = (p_hi 2^32 + p_lo) * w + r0 over (lo, hi) u64 halves.
            let a = p_lo * wf;
            let bq = p_hi * wf;
            let lo1 = a + (bq << 32u64);
            let mut hi = bq >> 32u64;
            if lo1 < a {
                hi += 1u64;
            }
            let lo = lo1 + u64::cast_from(r0);
            if lo < lo1 {
                hi += 1u64;
            }
            candidate_check(
                lo,
                hi,
                &mut sv_s,
                svb,
                nice_out,
                nice_count,
                nice_cap,
                base,
                limbs,
                chunk_digits,
                chunk_div,
                wide_chunk,
                pre_limbs,
                pre_chunk_digits,
                pre_chunk_div,
                probe,
            );
            i += stride;
        }
        cube_done(ran, 3u32);
    }
}

// ---------------------------------------------------------------------------
// Host side
// ---------------------------------------------------------------------------

/// `x` as little-endian `u32` words, the device's limb layout.
fn u32_words(x: u128) -> [u32; 4] {
    let b = x.to_le_bytes();
    std::array::from_fn(|i| {
        u32::from_le_bytes([b[4 * i], b[4 * i + 1], b[4 * i + 2], b[4 * i + 3]])
    })
}

/// What `client`'s device allows the join: `CubeCL` reports its largest
/// buffer as `max_page_size`, on CUDA a quarter of the device's memory
/// ([`JoinLimits::vram_budget`]).
pub(crate) fn limits_of<R: Runtime>(client: &ComputeClient<R>) -> JoinLimits {
    let lim = JoinLimits::for_buffer(client.properties().memory.max_page_size);
    if R::name(client).contains("cuda") {
        lim.with_vram_budget()
    } else {
        lim
    }
}

/// Device buffers for one field: the field's tables, the per-slot scratch
/// the kernels fill, and the survivor and output lists.
// The bools are independent per-field choices (key level, walk barrier,
// wide chunks, FD tops), not states of one machine.
#[allow(clippy::struct_excessive_bools)]
pub(crate) struct JoinDevice<R: Runtime> {
    client: ComputeClient<R>,
    b: u32,
    f0: u32,
    pp: u32,
    k: u32,
    k2: u32,
    key_level: bool,
    /// End each round of the join kernel's set-bit walk with a plane
    /// barrier (CUDA only, see the walk).
    walk_sync: bool,
    limbs: u32,
    chunk_digits: u32,
    chunk_div: u32,
    wide: bool,
    /// Certify tops with [`top_fd_kernel`] (where [`top_fd_fits`] the field
    /// and the base is not excluded, see [`TOP_FD_AT_POW2_BASES`]), the top
    /// layer's digit count t, and the partitions holding a clipped block
    /// (P = plo or phi), which `top_kernel` certifies beside it.
    top_fd: bool,
    tq: u32,
    clip_vs: Vec<u32>,
    /// [`TOP_FD_MIN_UNITS`] (the tests lower it to force slot groups).
    top_fd_min: u32,
    /// [`ballot_append`] for the device's largest plane.
    ballot: bool,
    w_f0: u32,
    nbp: u32,
    /// The field's certificate (`FieldSetup::fc`), low and high words.
    fc: [u32; 2],
    max_slots: usize,
    /// The layout's bytes, and its largest buffer (for errors).
    bytes: usize,
    largest: usize,
    bp_r: Handle,
    bp_m: Handle,
    seg: Handle,
    ext_m: Handle,
    /// The prefilter's terms per extended entry (see [`bucket_kernel`]).
    ext_q: Handle,
    lists: Handle,
    counts: Handle,
    nkeys: u32,
    /// The prefilter's survivors, which the full check reads.
    list: Handle,
    list_cap: u32,
    nice_out: Handle,
    nice_count: Handle,
    nice_cap: u32,
    // Device-side tops.
    ntlay: u32,
    nroots: u32,
    nb: u32,
    s2: u32,
    s3: u32,
    pdiv: u32,
    full_floor: u32,
    /// The top layer sorted by group (see [`top_kernel`]), and each group's
    /// first index in it (`nb + 1` words).
    tlay: Handle,
    goff: Handle,
    fb: Handle,
    pw: Handle,
    pw_len: usize,
    roots: Handle,
    top_rec: Handle,
}

/// What a launched batch leaves behind to be read later: its two counts,
/// the pairs that passed the join's AND (in range) and the prefilter's
/// survivors.
pub(crate) struct BatchRec {
    pub survivors: Handle,
    pub checked: Handle,
    /// Cubes of each kernel that finished, and that were launched (see
    /// [`JoinDevice::ensure_ran`]).
    pub ran: Handle,
    pub expect: [u32; STAGES],
}

/// Whether [`top_fd_kernel`]'s state fits `fs`: the biased add needs b <= 128
/// (the mask b <= 64), P², P³ need digits above their zero digits, and the
/// differences' words must fit the arrays they are added into (2P + 1 into
/// P², 3P² + 3P + 1 into P³). P < 2^64 is checked for every device top.
pub(crate) fn top_fd_fits(fs: &FieldSetup) -> bool {
    let (t, f0) = (fs.base.l - fs.f0, fs.f0);
    let (s2, s3) = (fs.base.s2, fs.base.s3);
    if fs.b > 64 || s2 <= 2 * f0 || s3 <= 3 * f0 {
        return false;
    }
    let (q2, q3) = (s2 - 2 * f0, s3 - 3 * f0);
    (t + 1).div_ceil(4) <= q2.div_ceil(4) && (q2 + 1).div_ceil(4) <= q3.div_ceil(4)
}

/// Whether `fs`'s tops are certified by [`top_fd_kernel`]: where it fits,
/// except at power-of-two bases, where the top kernel's division is shifts
/// and masks and the FD kernel measured slower (b64).
pub(crate) fn top_fd_runs(fs: &FieldSetup) -> bool {
    top_fd_fits(fs) && (TOP_FD_AT_POW2_BASES || !fs.b.is_power_of_two())
}

impl<R: Runtime> JoinDevice<R> {
    /// The device side of `fs` with `layer`, its sorted top layer, in
    /// `plan`'s layout.
    ///
    /// # Errors
    /// A base without a `u128` range, `n` too wide for device tops, or a
    /// layer of another length than `fs`'s.
    // One buffer per line, as the plan sized them.
    #[allow(clippy::too_many_lines)]
    pub fn new(
        client: &ComputeClient<R>,
        fs: &FieldSetup,
        layer: SortedLayer,
        plan: &JoinPlan,
    ) -> Result<Self> {
        let limbs = n_limbs(fs.b).ok_or_else(|| anyhow!("base {} has no u128 range", fs.b))?;
        ensure!(limbs <= 3, "device tops: n must fit 96 bits");
        let wide = wide_chunk_for(client);
        let (chunk_digits, chunk_div) = if wide {
            chunk_constants(fs.b)
        } else {
            chunk_constants_u16(fs.b)
        };
        let nbp = fs.bp_r.len();
        // The prefilter's terms pack values below M = b^(k2 - k + 1) into
        // 18 bits (`bucket_kernel`).
        ensure!(
            u64::from(fs.b).pow(fs.k2 - fs.jp.k + 1) <= 1 << 18,
            "base {} with a prefilter depth of {}: its terms exceed 18 bits",
            fs.b,
            fs.k2 - fs.jp.k
        );
        let bp_m_words: Vec<u32> = fs
            .bp_m
            .iter()
            .flat_map(|&m| {
                let w = u32_words(u128::from(m));
                [w[0], w[1]]
            })
            .collect();
        // Sizes and capacities come from the plan, which fitted them to the
        // device (`JoinPlan::new`); `fp` is the same table it used.
        let fp = Footprint::of(fs, plan.nice_cap);
        let max_slots = plan.slots;
        ensure!(max_slots > 0, "a join plan with no slots");
        let nkeys = if fs.key_level { fs.b } else { 1 };
        let (list_cap, nice_cap) = (plan.list_cap, plan.nice_cap);

        // Device-side tops: the top layer with its residues, sorted by
        // group, the field bounds, the power table and the per-slot records.
        let pp = fs.jp.p;
        let SortedLayer {
            words: tl, goff, ..
        } = layer;
        ensure!(
            tl.len() == 5 * fs.tlay.len() && goff.len() == fs.nb + 1,
            "a sorted layer of another slice"
        );
        // n < 2^96 (`FieldSetup::new`) and the prefix blocks below 2^64.
        let (s_w, e_w) = (u32_words(fs.s), u32_words(fs.e - 1));
        let (plo, phi) = (u32_words(fs.plo), u32_words(fs.phi));
        ensure!(
            plo[2..] == [0, 0] && phi[2..] == [0, 0],
            "prefix blocks exceed u64"
        );
        let fb = vec![
            s_w[0], s_w[1], s_w[2], e_w[0], e_w[1], e_w[2], plo[0], plo[1], phi[0], phi[1],
        ];
        let nl3 = 3 * limbs as usize;
        let s3 = fs.base.s3;
        let mut pw = Vec::with_capacity((s3 as usize + 1) * nl3);
        for i in 0..=s3 {
            let words = fs.base.poww_words(i);
            let mut l32: Vec<u32> = words
                .iter()
                .flat_map(|&w| {
                    let w = u32_words(u128::from(w));
                    [w[0], w[1]]
                })
                .collect();
            l32.resize(nl3.max(8), 0);
            ensure!(
                l32[nl3..].iter().all(|&x| x == 0),
                "b^{i} does not fit {nl3} limbs"
            );
            pw.extend_from_slice(&l32[..nl3]);
        }
        let roots: Vec<u32> = fs.base.roots.clone();
        let nroots = roots.len();
        ensure!(
            nroots <= MAX_ROOTS as usize,
            "base {} has {nroots} digit-sum roots",
            fs.b
        );
        let ntlay = fs.tlay.len();
        let nb = fs.nb;
        let pdiv = fs.nparts;
        let dev = Self {
            client: client.clone(),
            b: fs.b,
            f0: fs.f0,
            pp,
            k: fs.jp.k,
            k2: fs.k2,
            key_level: fs.key_level,
            // Only CUDA has independent thread scheduling to correct for, and
            // WGSL has no plane barrier.
            walk_sync: R::name(client).contains("cuda")
                && client
                    .properties()
                    .features
                    .plane
                    .contains(cubecl::ir::features::Plane::Sync),
            limbs,
            chunk_digits,
            chunk_div,
            wide,
            top_fd: top_fd_runs(fs),
            tq: fs.base.l - fs.f0,
            top_fd_min: TOP_FD_MIN_UNITS,
            ballot: ballot_append(client.properties().hardware.plane_size_max),
            clip_vs: {
                let mut c = Vec::new();
                if fs.plo * fs.w < fs.s {
                    c.push(u32::try_from(fs.plo % fs.nparts)?);
                }
                if fs.phi * fs.w + fs.w - 1 > fs.e - 1 {
                    c.push(u32::try_from(fs.phi % fs.nparts)?);
                }
                c
            },
            w_f0: u32::try_from(fs.w)?,
            nbp: u32::try_from(nbp)?,
            fc: {
                let w = u32_words(u128::from(fs.fc));
                [w[0], w[1]]
            },
            max_slots,
            bytes: plan.bytes,
            largest: (max_slots * fp.largest_per_slot())
                .max(fp.largest_fixed)
                .max(list_cap as usize * 8),
            bp_r: client.create(cubecl::bytes::Bytes::from_elems(fs.bp_r.clone())),
            bp_m: client.create(cubecl::bytes::Bytes::from_elems(bp_m_words)),
            seg: client.create(cubecl::bytes::Bytes::from_elems(fs.seg.clone())),
            ext_m: client.empty(max_slots * fp.ext_m),
            ext_q: client.empty(max_slots * fp.ext_q),
            lists: client.empty(max_slots * fp.lists),
            counts: client.empty(max_slots * fp.counts),
            nkeys,
            list: client.empty(list_cap as usize * 8),
            list_cap,
            nice_out: client.create(cubecl::bytes::Bytes::from_elems(vec![
                0u32;
                nice_cap as usize
                    * NICEONLY_STRIDE
                        as usize
            ])),
            nice_count: client.create(cubecl::bytes::Bytes::from_elems(vec![0u32; 1])),
            nice_cap,
            ntlay: u32::try_from(ntlay)?,
            nroots: u32::try_from(nroots)?,
            nb: u32::try_from(nb)?,
            s2: fs.base.s2,
            s3,
            pdiv: u32::try_from(pdiv)?,
            full_floor: fs.full_floor,
            tlay: client.create(cubecl::bytes::Bytes::from_elems(tl)),
            goff: client.create(cubecl::bytes::Bytes::from_elems(goff)),
            fb: client.create(cubecl::bytes::Bytes::from_elems(fb)),
            pw_len: pw.len(),
            pw: client.create(cubecl::bytes::Bytes::from_elems(pw)),
            roots: client.create(cubecl::bytes::Bytes::from_elems(roots)),
            top_rec: client.empty(max_slots * fp.top_rec),
        };
        // The FD top stage takes `top_kernel` (only_clipped) beside it on a
        // batch with a clipped block. Compile that now, on this slice's first
        // clipped partition, rather than inside whichever batch first holds
        // one (NiceMark's warm-up runs partition 0 only).
        if let Some(&v) = dev.clip_vs.first().filter(|_| dev.top_fd) {
            dev.warm_top(v);
        }
        Ok(dev)
    }

    /// Run the top stage once on partition `v` into scratch counters, so
    /// its kernels are compiled (see [`Self::new`]).
    fn warm_top(&self, v: u32) {
        let c = &self.client;
        let vs_h = c.create(cubecl::bytes::Bytes::from_elems(vec![v]));
        let gc = c.create(cubecl::bytes::Bytes::from_elems(vec![
            0u32;
            self.nb as usize
        ]));
        let ran = c.create(cubecl::bytes::Bytes::from_elems(vec![0u32; STAGES]));
        self.launch_top(&[v], &vs_h, &gc, &ran, self.top_fd);
    }

    /// Whether batch `vs` holds a clipped block (see `clip_vs`).
    fn top_fix(&self, vs: &[u32]) -> bool {
        vs.iter().any(|v| self.clip_vs.contains(v))
    }

    /// [`top_fd_kernel`]'s slots per group and groups for a batch of
    /// `nslots`: as few groups as give at least `top_fd_min` units.
    fn top_fd_groups(&self, nslots: u32) -> (u32, u32) {
        let want = self
            .top_fd_min
            .div_ceil(self.ntlay.max(1))
            .clamp(1, nslots.max(1));
        let k = nslots.div_ceil(want).max(1);
        (k, nslots.div_ceil(k).max(1))
    }

    /// The top stage's cubes for batch `vs` (see [`Self::launch_top`]).
    fn top_expect(&self, vs: &[u32], fd: bool) -> u32 {
        let top_cubes = self.ntlay.div_ceil(JOIN_WG).max(1);
        let nslots = u32::try_from(vs.len()).expect("a batch fits u32");
        let per_slot = top_cubes * nslots;
        let fd_cubes = top_cubes * self.top_fd_groups(nslots).1;
        match (fd, self.top_fix(vs)) {
            (false, _) => per_slot,
            (true, true) => fd_cubes + per_slot,
            (true, false) => fd_cubes,
        }
    }

    /// Launch the top stage of batch `vs` (on the device as `vs_h`):
    /// [`top_fd_kernel`] if `fd`, with `top_kernel` on the clipped blocks
    /// alone if the batch has one; else `top_kernel`. Both append into the
    /// slots' group segments of `top_rec`, counted in `grp_cnt` (zeroed,
    /// `vs.len() · nb` words).
    fn launch_top(&self, vs: &[u32], vs_h: &Handle, grp_cnt: &Handle, ran: &Handle, fd: bool) {
        let nslots = vs.len();
        let fix = fd && self.top_fix(vs);
        let c = &self.client;
        let ntl = self.ntlay as usize;
        let nb = self.nb as usize;
        let max_slots = self.max_slots;
        let nran = STAGES;
        let nslots_u32 = u32::try_from(nslots).expect("a batch fits u32");
        let top_cubes = self.ntlay.div_ceil(JOIN_WG).max(1);
        let rec_len = TOP_REC_WORDS as usize * max_slots * ntl;
        let (cd16, cdiv16) = chunk_constants_u16(self.b);
        let vs_h = vs_h.clone();
        let grp_cnt = grp_cnt.clone();
        let ran = ran.clone();
        unsafe {
            if fd {
                let (slot_k, groups) = self.top_fd_groups(nslots_u32);
                top_fd_kernel::launch_unchecked::<R>(
                    c,
                    CubeCount::Static(top_cubes, groups, 1),
                    CubeDim::new_1d(JOIN_WG),
                    ArrayArg::from_raw_parts(self.tlay.clone(), 5 * ntl),
                    ArrayArg::from_raw_parts(vs_h.clone(), nslots),
                    ArrayArg::from_raw_parts(self.fb.clone(), 10),
                    ArrayArg::from_raw_parts(self.pw.clone(), self.pw_len),
                    ArrayArg::from_raw_parts(self.top_rec.clone(), rec_len),
                    ArrayArg::from_raw_parts(grp_cnt.clone(), nslots * nb),
                    ArrayArg::from_raw_parts(ran.clone(), nran),
                    self.ntlay,
                    self.w_f0,
                    self.pdiv,
                    self.full_floor,
                    nslots_u32,
                    slot_k,
                    self.b,
                    self.limbs,
                    self.k,
                    self.k2,
                    self.s2,
                    self.s3,
                    cd16,
                    cdiv16,
                    self.nb,
                    self.f0,
                    self.tq,
                    self.ballot,
                );
            }
            if !fd || fix {
                top_kernel::launch_unchecked::<R>(
                    c,
                    CubeCount::Static(top_cubes, nslots_u32, 1),
                    CubeDim::new_1d(JOIN_WG),
                    ArrayArg::from_raw_parts(self.tlay.clone(), 5 * ntl),
                    ArrayArg::from_raw_parts(vs_h.clone(), nslots),
                    ArrayArg::from_raw_parts(self.fb.clone(), 10),
                    ArrayArg::from_raw_parts(self.pw.clone(), self.pw_len),
                    ArrayArg::from_raw_parts(self.top_rec.clone(), rec_len),
                    ArrayArg::from_raw_parts(grp_cnt.clone(), nslots * nb),
                    ArrayArg::from_raw_parts(ran.clone(), nran),
                    self.ntlay,
                    self.w_f0,
                    self.pdiv,
                    self.full_floor,
                    self.b,
                    self.limbs,
                    self.k,
                    self.k2,
                    self.s2,
                    self.s3,
                    cd16,
                    cdiv16,
                    self.nb,
                    fix,
                );
            }
        }
    }

    /// Launch one batch of partition values (one slot each): certify,
    /// bucket, extend, join, prefilter, check. Returns immediately.
    /// `probe`: the check kernel reports every n it rebuilds instead of
    /// checking it (tests).
    ///
    /// # Errors
    /// An empty or oversized batch, or a failed flush.
    // The launch sequence of one batch, kernel after kernel with its bindings.
    #[allow(clippy::too_many_lines)]
    pub fn launch_batch(&mut self, vs: &[u32], probe: bool) -> Result<BatchRec> {
        let nslots = vs.len();
        ensure!(
            nslots > 0 && nslots <= self.max_slots,
            "batch of {nslots} slots (max {})",
            self.max_slots
        );
        let nslots_u32 = u32::try_from(nslots)?;
        let c = &self.client;
        let nbp = self.nbp as usize;
        let ntl = self.ntlay as usize;
        let nb = self.nb as usize;
        let nwork = nslots * nb;
        let nwork_u32 = u32::try_from(nwork)?;
        let max_slots = self.max_slots;
        let rec_len = TOP_REC_WORDS as usize * max_slots * ntl;
        let vs_h = c.create(cubecl::bytes::Bytes::from_elems(vs.to_vec()));
        // Kept tops per (slot, group), which the top stage appends.
        let grp_cnt = c.create(cubecl::bytes::Bytes::from_elems(vec![0u32; nslots * nb]));
        // The join kernel counts its survivors and writes only those that
        // pass the prefilter to the list, which it counts too. Both counts
        // are exact: every survivor is counted and prefiltered in the walk.
        let survivors = c.create(cubecl::bytes::Bytes::from_elems(vec![0u32; 1]));
        let checked = c.create(cubecl::bytes::Bytes::from_elems(vec![0u32; 1]));
        // Every kernel counts its cubes as they finish, against the cubes
        // it was launched with. A device that drops a dispatch without an
        // error (as the macOS runner's paravirtual GPU does with layouts
        // above about 0.65 GiB) leaves its count short, where the batch
        // would otherwise read as having no survivors.
        let join_cubes = nwork_u32.min(65_535);
        let expect: [u32; STAGES] = [
            self.top_expect(vs, self.top_fd), // top
            (self.b - 1) * nslots_u32,        // bucket
            join_cubes,                       // join
            CHECK_CUBES,                      // check
        ];
        let ran = c.create(cubecl::bytes::Bytes::from_elems(vec![0u32; expect.len()]));
        let nran = expect.len();
        #[cfg(test)]
        if self.bytes > tests::REFUSE_ABOVE.with(std::cell::Cell::get) {
            // A device that runs nothing of a layout this large.
            return Ok(BatchRec {
                survivors,
                checked,
                ran,
                expect,
            });
        }
        self.launch_top(vs, &vs_h, &grp_cnt, &ran, self.top_fd);
        unsafe {
            bucket_kernel::launch_unchecked::<R>(
                c,
                CubeCount::Static(self.b - 1, nslots_u32, 1),
                CubeDim::new_1d(JOIN_WG),
                ArrayArg::from_raw_parts(self.bp_r.clone(), nbp),
                ArrayArg::from_raw_parts(self.bp_m.clone(), 2 * nbp),
                ArrayArg::from_raw_parts(self.seg.clone(), self.b as usize),
                ArrayArg::from_raw_parts(vs_h.clone(), nslots),
                ArrayArg::from_raw_parts(self.ext_m.clone(), 2 * nbp * max_slots),
                ArrayArg::from_raw_parts(self.ext_q.clone(), 2 * nbp.max(1) * max_slots),
                ArrayArg::from_raw_parts(self.lists.clone(), nbp * max_slots * self.b as usize),
                ArrayArg::from_raw_parts(
                    self.counts.clone(),
                    max_slots * self.nkeys as usize * (self.b as usize - 1),
                ),
                ArrayArg::from_raw_parts(ran.clone(), nran),
                self.nbp,
                self.fc[0],
                self.fc[1],
                self.b,
                self.f0,
                self.pp,
                self.k,
                self.k2,
                self.key_level,
                self.key_level && self.k == 1,
            );
            join_kernel::launch_unchecked::<R>(
                c,
                CubeCount::Static(join_cubes, 1, 1),
                CubeDim::new_1d(JOIN_WG),
                ArrayArg::from_raw_parts(self.ext_m.clone(), 2 * nbp * max_slots),
                ArrayArg::from_raw_parts(self.ext_q.clone(), 2 * nbp.max(1) * max_slots),
                ArrayArg::from_raw_parts(self.bp_r.clone(), nbp),
                ArrayArg::from_raw_parts(self.seg.clone(), self.b as usize),
                ArrayArg::from_raw_parts(self.lists.clone(), nbp * max_slots * self.b as usize),
                ArrayArg::from_raw_parts(
                    self.counts.clone(),
                    max_slots * self.nkeys as usize * (self.b as usize - 1),
                ),
                ArrayArg::from_raw_parts(vs_h, nslots),
                ArrayArg::from_raw_parts(self.roots.clone(), self.nroots as usize),
                ArrayArg::from_raw_parts(self.goff.clone(), nb + 1),
                ArrayArg::from_raw_parts(grp_cnt, nslots * nb),
                ArrayArg::from_raw_parts(self.top_rec.clone(), rec_len),
                ArrayArg::from_raw_parts(self.list.clone(), 2 * self.list_cap as usize),
                ArrayArg::from_raw_parts(checked.clone(), 1),
                ArrayArg::from_raw_parts(survivors.clone(), 1),
                ArrayArg::from_raw_parts(ran.clone(), nran),
                nwork_u32,
                self.nbp,
                self.list_cap,
                self.w_f0,
                self.ntlay,
                self.b,
                self.key_level,
                self.nroots,
                ENTRIES_PER_THREAD,
                self.f0,
                self.k,
                self.k2,
                self.walk_sync,
            );
        }
        unsafe {
            check_kernel::launch_unchecked::<R>(
                c,
                CubeCount::Static(CHECK_CUBES, 1, 1),
                CubeDim::new_1d(JOIN_WG),
                ArrayArg::from_raw_parts(self.list.clone(), 2 * self.list_cap as usize),
                ArrayArg::from_raw_parts(checked.clone(), 1),
                ArrayArg::from_raw_parts(self.top_rec.clone(), rec_len),
                ArrayArg::from_raw_parts(
                    self.nice_out.clone(),
                    self.nice_cap as usize * NICEONLY_STRIDE as usize,
                ),
                ArrayArg::from_raw_parts(self.nice_count.clone(), 1),
                ArrayArg::from_raw_parts(ran.clone(), nran),
                self.list_cap,
                self.nice_cap,
                self.w_f0,
                self.b,
                self.limbs,
                self.chunk_digits,
                self.chunk_div,
                self.wide,
                0u32,
                0u32,
                0u32,
                probe,
            );
        }
        c.flush().map_err(|e| anyhow!("flush failed: {e:?}"))?;
        Ok(BatchRec {
            survivors,
            checked,
            ran,
            expect,
        })
    }

    /// Whether every kernel of a batch finished every cube it was launched
    /// with (`ran` against `expect`, see [`Self::launch_batch`]). If not,
    /// the batch's counts and lists are not its own: a device that drops a
    /// dispatch without an error must not have the batch read as one with
    /// no survivors.
    fn ensure_ran(&self, ran: &[u32], expect: &[u32; STAGES]) -> Result<(), NotRun> {
        if ran == expect.as_slice() {
            return Ok(());
        }
        Err(NotRun {
            bytes: self.bytes,
            largest: self.largest,
            ran: ran.to_vec(),
            expect: *expect,
        })
    }

    /// Whether a batch dropped prefilter survivors past the end of the list
    /// (unchecked), from its `checked` count.
    fn overflowed(&self, checked: u32) -> bool {
        checked > self.list_cap
    }

    fn read_u32(&self, h: &Handle) -> Result<Vec<u32>> {
        let bytes = self
            .client
            .read_one(h.clone())
            .map_err(|e| anyhow!("read failed: {e:?}"))?;
        Ok(u32::from_bytes(&bytes).to_vec())
    }

    /// Hits (or, after a probe launch, every rebuilt n) so far, sorted.
    ///
    /// # Errors
    /// Device read failure or output overflow.
    pub fn read_hits(&self) -> Result<Vec<u128>> {
        let written = self.read_u32(&self.nice_count)?[0] as usize;
        ensure!(
            written <= self.nice_cap as usize,
            "overlap join output overflow: {written} > {} (this strongly suggests a kernel bug)",
            self.nice_cap
        );
        let words = self.read_u32(&self.nice_out)?;
        let mut v: Vec<u128> = (0..written)
            .map(|i| {
                let o = i * NICEONLY_STRIDE as usize;
                let lo = u128::from(words[o]) | (u128::from(words[o + 1]) << 32);
                let hi = u128::from(words[o + 2]) | (u128::from(words[o + 3]) << 32);
                (hi << 64) | lo
            })
            .collect();
        v.sort_unstable();
        Ok(v)
    }

    /// A batch's two counts: the join's survivors and the prefilter's.
    #[cfg(test)]
    fn read_counts(&self, rec: &BatchRec) -> Result<(u32, u32)> {
        Ok((
            self.read_u32(&rec.survivors)?[0],
            self.read_u32(&rec.checked)?[0],
        ))
    }

    /// The prefilter's survivors of the most recent batch as n, sorted
    /// (tests).
    #[cfg(test)]
    fn read_checked(&self, rec: &BatchRec) -> Result<Vec<u128>> {
        let cnt = self.read_u32(&rec.checked)?[0];
        ensure!(
            cnt <= self.list_cap,
            "survivor list overflow: {cnt} > {}",
            self.list_cap
        );
        let words = self.read_u32(&self.list)?;
        let tops = self.read_u32(&self.top_rec)?;
        let mut v: Vec<u128> = (0..cnt as usize)
            .map(|i| {
                let r8 = TOP_REC_WORDS as usize * words[2 * i] as usize;
                let p = u128::from(tops[r8]) | (u128::from(tops[r8 + 1]) << 32);
                p * u128::from(self.w_f0) + u128::from(words[2 * i + 1])
            })
            .collect();
        v.sort_unstable();
        Ok(v)
    }
}

/// A batch the device did not run in full: a kernel finished fewer cubes
/// than it was launched with, in a layout the device's limits allowed (see
/// [`JoinCeiling`]).
#[derive(Debug)]
pub(crate) struct NotRun {
    /// The layout's bytes, and its largest buffer.
    bytes: usize,
    largest: usize,
    /// Finished and launched cubes, kernel by kernel.
    ran: Vec<u32>,
    expect: [u32; STAGES],
}

impl std::fmt::Display for NotRun {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "overlap join: the device did not run a batch in a layout of {} MiB (buffers up \
             to {} MiB): its kernels finished {:?} of {:?} cubes",
            self.bytes >> 20,
            self.largest >> 20,
            self.ran,
            self.expect
        )
    }
}

impl std::error::Error for NotRun {}

/// What one field through the join did, for telemetry.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct JoinFieldStats {
    /// Host setup of every slice, the first on the thread that began the
    /// field and the rest beside the device's work on the slice before.
    pub setup_secs: f64,
    /// Wall time of the device work, every slice from first launch to last
    /// read.
    pub device_secs: f64,
    pub device_wait_secs: f64,
    /// Slices, and partitions per slice.
    pub slices: usize,
    pub partitions: usize,
    pub batches: u32,
    /// Partitions per launch planned for the first slice, the fewest any
    /// launch used, and the bytes of the first slice's buffers.
    pub slots: usize,
    pub min_slots: usize,
    pub bytes: usize,
    /// Pairs that passed the join's AND inside their top's range, and of
    /// those the prefilter's survivors (what the full check read). Both
    /// depend only on the field and the parameters, not on the device or the
    /// layout: the join kernel counts and prefilters every pair in its walk,
    /// before the shared stage, so a chunk that overruns the stage sends the
    /// list only prefiltered pairs too.
    pub survivors: u64,
    pub checked: u64,
    /// Partitions of batches that overflowed their list and ran again, in a
    /// batch half the size (each time they did).
    pub retried_partitions: usize,
    /// Halvings of a re-run partition's top layer (normally none).
    pub splits: u32,
    /// Layouts the device did not run, each followed by one of at most
    /// half the size (normally none; see [`JoinCeiling`]).
    pub refused: u32,
}

/// Run one prepared field through the join on `client`: slice by slice
/// (each later slice's setup beside the device's work on the one before),
/// every partition of each in batches ([`run_slice`]). A field whose top
/// layer is empty has no candidates and needs no device work at all; about
/// a third of random 1e14 fields at bases 57-64 are such.
///
/// A device that does not run a layout gets the work again in one of at
/// most half the size, and keeps that `ceiling` for every later layout.
///
/// The field reports its progress ([`crate::progress`]) in partitions run,
/// every slice's, from when the device starts on it.
///
/// # Errors
/// Device failures, a device that runs no layout of the field, or a single
/// top prefix of one partition that overflows the re-run's list (which
/// `join_plan::MIN_RETRY_CAP` rules out).
pub(crate) fn run_field<R: Runtime>(
    client: &ComputeClient<R>,
    field: &JoinField,
    ceiling: &JoinCeiling,
) -> Result<(Vec<NiceNumberSimple>, JoinFieldStats)> {
    let parts: Vec<u32> = (0..u32::try_from(field.fs.nparts)?).collect();
    let range = match (field.slices.first(), field.slices.last()) {
        (Some(first), Some(last)) => FieldSize::new(first.start(), last.end()),
        _ => return Ok((Vec::new(), JoinFieldStats::default())),
    };
    let units = (parts.len() * field.slices.len()) as u64;
    let mut progress =
        FieldProgress::with_units(&range, units, range.size() / u128::from(units.max(1)));
    run_slices_with(
        client,
        field,
        field.slices.len(),
        &parts,
        ceiling,
        Some(&mut progress),
    )
}

/// The first `count` slices of `field`, each on the partitions `parts`:
/// [`run_field`], or a sample of it (`count = 1` and a few partitions) for
/// timing the join on a production-size field without running all of it.
///
/// # Errors
/// As [`run_field`].
pub(crate) fn run_slices<R: Runtime>(
    client: &ComputeClient<R>,
    field: &JoinField,
    count: usize,
    parts: &[u32],
    ceiling: &JoinCeiling,
) -> Result<(Vec<NiceNumberSimple>, JoinFieldStats)> {
    run_slices_with(client, field, count, parts, ceiling, None)
}

/// [`run_slices`], counting the partitions run on `progress`.
fn run_slices_with<R: Runtime>(
    client: &ComputeClient<R>,
    field: &JoinField,
    count: usize,
    parts: &[u32],
    ceiling: &JoinCeiling,
    mut progress: Option<&mut FieldProgress>,
) -> Result<(Vec<NiceNumberSimple>, JoinFieldStats)> {
    let mut st = JoinFieldStats {
        partitions: usize::try_from(field.fs.nparts)?,
        slots: field.plan.slots,
        min_slots: field.plan.slots,
        bytes: field.plan.bytes,
        ..JoinFieldStats::default()
    };
    let mut hits = Vec::new();
    let mut slots_hint = field.plan.slots;
    let mut next: Option<Result<(FieldSetup, SortedLayer)>> = None;
    for i in 0..count.min(field.slices.len()) {
        let owned;
        let (fs, sorted) = if i == 0 {
            (&field.fs, field.layer.clone())
        } else {
            owned = next.take().expect("prepared beside the previous slice")?;
            (&owned.0, owned.1)
        };
        st.slices += 1;
        st.setup_secs += fs.secs + sorted.secs;
        // The field's own layouts, unless the ceiling came down since it was
        // planned.
        let lim = ceiling.cap(field.lim);
        let layouts = if i == 0 && lim == field.lim {
            (field.plan, field.retry)
        } else {
            JoinPlan::for_field(fs, lim)
                .ok_or_else(|| anyhow!("slice {i} of the field does not fit the device"))?
        };
        let later = field.slices.get(i + 1).filter(|_| i + 1 < count);
        let start = slots_hint.clamp(1, layouts.0.slots);
        std::thread::scope(|scope| -> Result<()> {
            // The next slice's setup and sorted top layer, beside this one's
            // device work.
            let prep = later.map(|sl| {
                scope.spawn(|| -> Result<(FieldSetup, SortedLayer)> {
                    let fs = field.fs.sub_range(sl.start(), sl.end())?;
                    let sorted = sorted_layer(&fs)?;
                    Ok((fs, sorted))
                })
            });
            let used = run_slice(
                client,
                fs,
                sorted,
                layouts,
                field.lim,
                ceiling,
                slots_hint,
                parts,
                &mut st,
                &mut hits,
                progress.as_deref_mut(),
            )?;
            // A slice that kept its batch size lets the next one try twice
            // that (up to its own plan's), so one dense slice does not hold
            // the rest of the field at a smaller size.
            slots_hint = if used < start {
                used
            } else {
                used.saturating_mul(2)
            };
            next = prep.map(|h| h.join().expect("slice setup panicked"));
            Ok(())
        })?;
    }
    hits.sort_unstable();
    hits.dedup();
    let nice = hits
        .into_iter()
        .map(|number| NiceNumberSimple {
            number,
            num_uniques: field.fs.b,
        })
        .collect();
    Ok((nice, st))
}

/// One slice, on the partitions `parts`, in `layouts` (the batches' and the
/// re-run's): batches of at most `slots_hint` partitions (and of the
/// plan's), all launched before any is read. The batches whose survivors
/// overflowed their list go again in batches half the size, until one
/// partition is left; one partition that still overflows takes the re-run
/// layout ([`run_partition`]). Batches the device did not run go again in
/// layouts within `lim` and a lowered `ceiling` ([`refit`]). Returns the
/// batch size the last pass used, for the next slice to start from.
#[allow(clippy::too_many_arguments)]
fn run_slice<R: Runtime>(
    client: &ComputeClient<R>,
    fs: &FieldSetup,
    layer: SortedLayer,
    layouts: (JoinPlan, JoinPlan),
    lim: JoinLimits,
    ceiling: &JoinCeiling,
    slots_hint: usize,
    parts: &[u32],
    st: &mut JoinFieldStats,
    hits: &mut Vec<u128>,
    mut progress: Option<&mut FieldProgress>,
) -> Result<usize> {
    if fs.tlay.is_empty() || parts.is_empty() {
        // Nothing for the device: the slice's partitions are done.
        if let Some(p) = progress {
            p.add(parts.len() as u64);
        }
        return Ok(slots_hint);
    }
    let t0 = Instant::now();
    let (plan, mut retry) = layouts;
    let mut dev = JoinDevice::new(client, fs, layer, &plan)?;
    let mut slots = slots_hint.clamp(1, dev.max_slots);
    let mut todo = parts.to_vec();
    loop {
        st.min_slots = st.min_slots.min(slots);
        let left = run_pass(client, &mut dev, &todo, slots, st, progress.as_deref_mut())?;
        if let Some(why) = left.refused {
            // What the batches that ran found stays found; the rest goes
            // again, with what overflowed, in a smaller layout.
            hits.extend(dev.read_hits()?);
            drop(dev);
            let (plan, smaller) = refit(client, fs, lim, ceiling, &why, st)?;
            retry = smaller;
            dev = JoinDevice::new(client, fs, sorted_layer(fs)?, &plan)?;
            slots = slots.min(dev.max_slots);
            todo = [left.not_run, left.overflowed].concat();
            continue;
        }
        if left.overflowed.is_empty() {
            break;
        }
        debug!(
            "overlap join b{}: {} partitions overflowed a survivor list in batches of {slots}",
            fs.b,
            left.overflowed.len()
        );
        st.retried_partitions += left.overflowed.len();
        if slots == 1 {
            // Survivors past a list's end were dropped unchecked, so these
            // run again; hits they already found come back again and are
            // deduplicated by the caller.
            hits.extend(dev.read_hits()?);
            drop(dev);
            for &v in &left.overflowed {
                rerun_partition(client, fs, &mut retry, lim, ceiling, v, st, hits)?;
            }
            st.device_secs += t0.elapsed().as_secs_f64();
            return Ok(1);
        }
        slots = (slots / 2).max(1);
        todo = left.overflowed;
    }
    hits.extend(dev.read_hits()?);
    st.device_secs += t0.elapsed().as_secs_f64();
    Ok(slots)
}

/// After the device did not run a layout (`why`): lower its ceiling to half
/// that layout, release what the layout held, and fit `fs` within `lim`
/// and the ceiling. Every refit at least halves the layout, so a device
/// that runs nothing ends here with an error, once one partition no longer
/// fits.
fn refit<R: Runtime>(
    client: &ComputeClient<R>,
    fs: &FieldSetup,
    lim: JoinLimits,
    ceiling: &JoinCeiling,
    why: &NotRun,
    st: &mut JoinFieldStats,
) -> Result<(JoinPlan, JoinPlan)> {
    let cap = ceiling.lower(why.bytes);
    st.refused += 1;
    client.memory_cleanup();
    warn!(
        "{why}; the join takes layouts of at most {} MiB on this device from now on",
        cap >> 20
    );
    JoinPlan::for_field(fs, ceiling.cap(lim))
        .ok_or_else(|| anyhow!("{why}, and one partition of this field needs more than half"))
}

/// Partition `v` alone in the re-run layout ([`run_partition`]), which
/// shrinks for as long as the device does not run it ([`refit`]).
#[allow(clippy::too_many_arguments)]
fn rerun_partition<R: Runtime>(
    client: &ComputeClient<R>,
    fs: &FieldSetup,
    retry: &mut JoinPlan,
    lim: JoinLimits,
    ceiling: &JoinCeiling,
    v: u32,
    st: &mut JoinFieldStats,
    hits: &mut Vec<u128>,
) -> Result<()> {
    loop {
        let (survivors, checked, splits) = (st.survivors, st.checked, st.splits);
        let Err(e) = run_partition(client, fs, retry, v, 0, st, hits) else {
            return Ok(());
        };
        let why = e.downcast::<NotRun>()?;
        // Halves of its top layer that ran are counted again when it does.
        (st.survivors, st.checked, st.splits) = (survivors, checked, splits);
        *retry = refit(client, fs, lim, ceiling, &why, st)?.1;
    }
}

/// What a pass over a slice's partitions left to do.
#[derive(Default)]
struct Leftover {
    /// Partitions of the batches that overflowed their survivor list.
    overflowed: Vec<u32>,
    /// Partitions of the batches the device did not run, and how the first
    /// of those fell short.
    not_run: Vec<u32>,
    refused: Option<NotRun>,
}

/// One pass over `parts` in batches of `slots`, pipelined: every batch is
/// launched before any count is read. Adds the batches that ran and fit
/// their list to `st`, and returns the partitions of the rest.
fn run_pass<R: Runtime>(
    client: &ComputeClient<R>,
    dev: &mut JoinDevice<R>,
    parts: &[u32],
    slots: usize,
    st: &mut JoinFieldStats,
    mut progress: Option<&mut FieldProgress>,
) -> Result<Leftover> {
    // Each fence stands for the partitions launched before it: once it
    // resolves, the device has run them.
    let mut inflight: VecDeque<(LaunchFence, usize)> = VecDeque::new();
    let (mut launched, mut counted) = (0, 0);
    let mut count_to = |upto: usize| {
        if let Some(p) = progress.as_deref_mut() {
            p.add((upto - counted) as u64);
        }
        counted = upto;
    };
    let mut recs = Vec::new();
    for batch in parts.chunks(slots) {
        while inflight.len() >= BATCHES_IN_FLIGHT {
            let tw = Instant::now();
            if let Some((f, upto)) = inflight.pop_front() {
                cubecl::future::block_on(f).map_err(|e| anyhow!("launch fence failed: {e:?}"))?;
                count_to(upto);
            }
            st.device_wait_secs += tw.elapsed().as_secs_f64();
        }
        let rec = dev.launch_batch(batch, false)?;
        launched += batch.len();
        match launch_fence(client)? {
            Some(f) => inflight.push_back((f, launched)),
            // Nothing was pending: the batch has already run.
            None => count_to(launched),
        }
        st.batches += 1;
        recs.push((batch, rec));
    }
    let handles: Vec<Handle> = recs
        .iter()
        .flat_map(|(_, r)| [r.survivors.clone(), r.checked.clone(), r.ran.clone()])
        .collect();
    let counts = cubecl::future::block_on(client.read_async(handles))
        .map_err(|e| anyhow!("read failed: {e:?}"))?;
    count_to(launched);
    let mut left = Leftover::default();
    for (i, (batch, rec)) in recs.iter().enumerate() {
        if let Err(why) = dev.ensure_ran(u32::from_bytes(&counts[3 * i + 2]), &rec.expect) {
            left.not_run.extend_from_slice(batch);
            left.refused.get_or_insert(why);
            continue;
        }
        let surv = u32::from_bytes(&counts[3 * i])[0];
        let checked = u32::from_bytes(&counts[3 * i + 1])[0];
        if dev.overflowed(checked) {
            left.overflowed.extend_from_slice(batch);
        } else {
            st.survivors += u64::from(surv);
            st.checked += u64::from(checked);
        }
    }
    Ok(left)
}

impl JoinFieldStats {
    /// The field's account as telemetry.
    pub(crate) fn telemetry(&self) -> JoinTelemetry {
        JoinTelemetry {
            setup_secs: self.setup_secs,
            run_secs: self.device_secs,
            slices: self.slices,
            partitions: self.partitions,
            slots: self.slots,
            min_slots: self.min_slots,
            retried_partitions: self.retried_partitions,
            splits: self.splits,
            refused: self.refused,
            survivors: self.survivors,
            checked: self.checked,
        }
    }
}

/// The same field with only the top-layer prefixes `tlay` (a subset of its
/// own). The tops keep their intervals, so a partition's survivors split
/// exactly between complementary subsets.
fn with_tops(fs: &FieldSetup, tlay: Vec<(u128, u64)>) -> FieldSetup {
    let mut sub = fs.clone();
    sub.tlay = tlay;
    sub
}

/// Partition `v` of `fs`'s field alone, with `plan` (one slot). If its
/// survivors overflow the list, its top layer is halved and each half runs
/// the partition again: the halves' survivors are exactly the whole's, split,
/// and a single top prefix of one partition always fits the re-run's list
/// (see `join_plan::MIN_RETRY_CAP`).
fn run_partition<R: Runtime>(
    client: &ComputeClient<R>,
    fs: &FieldSetup,
    plan: &JoinPlan,
    v: u32,
    depth: u32,
    st: &mut JoinFieldStats,
    hits: &mut Vec<u128>,
) -> Result<()> {
    let mut dev = JoinDevice::new(client, fs, sorted_layer(fs)?, plan)?;
    let rec = dev.launch_batch(&[v], false)?;
    st.batches += 1;
    dev.ensure_ran(&dev.read_u32(&rec.ran)?, &rec.expect)?;
    let surv = dev.read_u32(&rec.survivors)?[0];
    let checked = dev.read_u32(&rec.checked)?[0];
    if !dev.overflowed(checked) {
        st.survivors += u64::from(surv);
        st.checked += u64::from(checked);
        hits.extend(dev.read_hits()?);
        return Ok(());
    }
    drop(dev);
    ensure!(
        depth < MAX_SPLITS && fs.tlay.len() >= 2,
        "overlap join b{} [{}, {}): partition {v} overflows its survivor list on {} top \
         prefixes ({surv} join survivors, {checked} after the prefilter, room for {})",
        fs.b,
        fs.s,
        fs.e,
        fs.tlay.len(),
        plan.list_cap
    );
    st.splits += 1;
    let (lo, hi) = fs.tlay.split_at(fs.tlay.len() / 2);
    for half in [lo, hi] {
        run_partition(
            client,
            &with_tops(fs, half.to_vec()),
            plan,
            v,
            depth + 1,
            st,
            hits,
        )?;
    }
    Ok(())
}

struct JoinJob {
    field: JoinField,
    pushed_at: Instant,
}

type JoinDone = Result<(NiceonlyStats, Vec<NiceNumberSimple>, JoinFieldStats)>;

/// The join's side of the nice-only pipeline: a thread that takes fields in
/// order, runs each through [`run_field`] and hands the results back in the
/// same order. Same contract as `NiceonlyPipeline`: `push` returns a ticket,
/// and `next_result` takes the tickets back in order.
// Public because it names a field type of the public context enum; its own
// fields stay private, as `NiceonlyPlan`'s do.
pub struct JoinPipeline {
    /// `Option` only so `Drop` can release it before joining the thread.
    tx: Option<SyncSender<JoinJob>>,
    results: Option<Receiver<JoinDone>>,
    /// Fields pushed, and fields returned: the next ticket to issue and the
    /// next one to take back.
    pushed: u64,
    returned: u64,
    thread: Option<std::thread::JoinHandle<()>>,
    /// The join's own account of the field `next_result` last returned,
    /// for the throughput harness.
    #[cfg(test)]
    last: Option<JoinFieldStats>,
}

impl JoinPipeline {
    /// The pipeline for the device `client`, whose layouts stay within
    /// `ceiling` (shared with the device's planning).
    pub(crate) fn start<R: Runtime>(client: ComputeClient<R>, ceiling: Arc<JoinCeiling>) -> Self {
        let depth = FIELDS_IN_FLIGHT + 1;
        let (tx, jobs) = sync_channel::<JoinJob>(depth);
        let (results_tx, results) = sync_channel::<JoinDone>(depth);
        let thread = std::thread::spawn(move || {
            while let Ok(job) = jobs.recv() {
                let (base, jp) = (job.field.fs.b, job.field.fs.jp);
                let out = run_field(&client, &job.field, &ceiling).map(|(hits, js)| {
                    let stats = NiceonlyStats {
                        msd_secs: js.setup_secs,
                        device_secs: js.device_secs,
                        total_secs: job.pushed_at.elapsed().as_secs_f64(),
                        floor: 0,
                        num_ranges: js.partitions * js.slices.max(1),
                        valid_numbers: js.checked,
                        launches: js.batches,
                        cpu_wait_secs: 0.0,
                        device_wait_secs: js.device_wait_secs,
                        device_busy_secs: None,
                        overlap_join: true,
                        join: Some(js.telemetry()),
                        route_reason: None,
                    };
                    debug!(
                        "overlap join b{base} {jp:?}: setup {:.3}s, device {:.3}s ({} slices of \
                         {} partitions, {} batches of {} (at least {}), {} MiB), {} survivors, \
                         {} checked, {} re-run, {} splits, total {:.3}s",
                        js.setup_secs,
                        js.device_secs,
                        js.slices,
                        js.partitions,
                        js.batches,
                        js.slots,
                        js.min_slots,
                        js.bytes >> 20,
                        js.survivors,
                        js.checked,
                        js.retried_partitions,
                        js.splits,
                        stats.total_secs
                    );
                    (stats, hits, js)
                });
                if results_tx.send(out).is_err() {
                    return;
                }
            }
        });
        Self {
            tx: Some(tx),
            results: Some(results),
            pushed: 0,
            returned: 0,
            thread: Some(thread),
            #[cfg(test)]
            last: None,
        }
    }

    /// Queue a prepared field and return its ticket. Returns at once unless
    /// the queue is full.
    ///
    /// # Errors
    /// The worker thread has exited.
    pub(crate) fn push(&mut self, field: JoinField) -> Result<FieldTicket> {
        let tx = self
            .tx
            .as_ref()
            .ok_or_else(|| anyhow!("overlap join pipeline is shut down"))?;
        tx.send(JoinJob {
            field,
            pushed_at: Instant::now(),
        })
        .map_err(|_| anyhow!("overlap join worker is gone"))?;
        self.pushed += 1;
        Ok(FieldTicket::new(Route::Join, self.pushed - 1))
    }

    /// Wait for the field `ticket` stands for, which must be the oldest
    /// queued one.
    ///
    /// # Errors
    /// The field's own error, the worker having exited, or a ticket that is
    /// not the oldest.
    pub(crate) fn next_result(
        &mut self,
        ticket: FieldTicket,
    ) -> Result<(NiceonlyStats, Vec<NiceNumberSimple>)> {
        ensure!(
            self.returned < self.pushed,
            "no field outstanding in the overlap join"
        );
        ticket.redeem(Route::Join, self.returned)?;
        self.returned += 1;
        let (stats, hits, js) = self
            .results
            .as_ref()
            .ok_or_else(|| anyhow!("overlap join pipeline is shut down"))?
            .recv()
            .map_err(|_| anyhow!("overlap join worker is gone"))??;
        #[cfg(test)]
        {
            self.last = Some(js);
        }
        #[cfg(not(test))]
        let _ = js;
        Ok((stats, hits))
    }

    /// The join's own account (re-runs, layout) of the field
    /// [`Self::next_result`] last returned.
    #[cfg(test)]
    pub(crate) fn last_field_stats(&self) -> Option<JoinFieldStats> {
        self.last
    }
}

impl Drop for JoinPipeline {
    fn drop(&mut self) {
        drop(self.tx.take());
        drop(self.results.take());
        if let Some(t) = self.thread.take() {
            let _ = t.join();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::FieldSize;
    use crate::cubecl_backend::{CubeclContext, last_join_stats};
    use crate::gpu_route::{NiceonlyGpu, NiceonlyStarted, begin_niceonly, process_niceonly};
    use crate::overlap_join::test_fields::{
        FRONTIER_57, PLAN_FIELDS, WINDOWS, live_partitions, mid_mirror, reference_field,
    };
    use crate::overlap_join::{JoinParams, join_range};

    thread_local! {
        /// Layouts above this many bytes run nothing (`launch_batch`): the
        /// macOS runner's paravirtual Metal device, on this thread.
        pub(super) static REFUSE_ABOVE: std::cell::Cell<usize> =
            const { std::cell::Cell::new(usize::MAX) };
    }

    fn client() -> (ComputeClient<cubecl::wgpu::WgpuRuntime>, String) {
        let ctx = CubeclContext::new_default().expect("CubeCL init");
        let name = ctx.device_name();
        #[allow(irrefutable_let_patterns)]
        let CubeclContext::Wgpu { client, .. } = ctx else {
            unreachable!("new_default is wgpu")
        };
        (client, name)
    }

    /// The device's limits within the ceiling the client starts it at
    /// ([`JoinCeiling::for_device`]): the limits every client field is
    /// planned within, so the tests' production-size layouts are the ones a
    /// client would run.
    fn device_limits<R: Runtime>(client: &ComputeClient<R>, name: &str) -> JoinLimits {
        JoinCeiling::for_device(name).cap(limits_of(client))
    }

    /// A layout for `slots` partitions per launch whose lists hold every
    /// survivor of a production batch (a base-57 partition has 1.5-2.7e6
    /// join survivors), since the tests compare whole lists: the device's
    /// binding limit, but a larger budget than production's.
    fn test_plan(fs: &FieldSetup, lim: JoinLimits, slots: usize) -> JoinPlan {
        let lim = JoinLimits {
            max_binding: lim.max_binding,
            budget: 4 << 30,
            vram_budget: lim.vram_budget,
        };
        JoinPlan::new(fs, lim, slots, 1 << 24, 1, 2 << 30).expect("the test layout fits")
    }

    /// The join's survivor count, the prefilter's survivors and the hits
    /// must equal the CPU reference, batch by batch. `parts` defaults to the
    /// partitions that hold a top of the window, thinned evenly to at most
    /// 96 so a software rasterizer gets through it.
    fn check_window<R: Runtime>(
        client: &ComputeClient<R>,
        b: u32,
        s: u128,
        e: u128,
        jp: JoinParams,
        parts: Option<Vec<u32>>,
    ) {
        let fs = FieldSetup::new(b, s, e, jp).expect("field setup");
        let parts = parts.unwrap_or_else(|| {
            let live = live_partitions(&fs);
            let step = live.len().div_ceil(96).max(1);
            live.into_iter().step_by(step).collect()
        });
        let plan = test_plan(&fs, limits_of(client), 4);
        let layer = sorted_layer(&fs).expect("sorted layer");
        let mut dev = JoinDevice::new(client, &fs, layer, &plan).expect("device");
        let mut cpu_hits = Vec::new();
        for batch in parts.chunks(dev.max_slots) {
            let vs: Vec<u128> = batch.iter().map(|&v| u128::from(v)).collect();
            let mut cpu = Vec::new();
            let st = join_range(&fs.base, s, e, jp, Some(&vs), Some(&mut cpu));
            cpu.retain(|&n| mid_mirror(&fs, n));
            cpu.sort_unstable();
            cpu_hits.extend(st.hits);
            let rec = dev.launch_batch(batch, false).expect("launch");
            let (survivors, _) = dev.read_counts(&rec).expect("counts");
            assert_eq!(
                u64::from(survivors),
                st.survivors,
                "b{b} [{s}, {e}) {jp:?} partitions {batch:?}: join survivors differ"
            );
            let got = dev.read_checked(&rec).expect("prefilter survivors");
            assert_eq!(
                got, cpu,
                "b{b} [{s}, {e}) {jp:?} partitions {batch:?}: prefilter survivors differ"
            );
        }
        cpu_hits.sort_unstable();
        assert_eq!(
            dev.read_hits().unwrap(),
            cpu_hits,
            "b{b} [{s}, {e}) {jp:?}: hits differ"
        );
    }

    /// Layouts far tighter than production's, against the CPU reference over
    /// every partition of a window: the batches' list holds half the densest
    /// partition, so batches re-run their partitions one at a time, and the
    /// re-run's list holds only the densest single top prefix, so the
    /// densest partition's top layer is halved down to prefixes that fit.
    /// Hits, join survivors and prefilter survivors must come out exactly as
    /// the reference has them.
    fn check_tight_layouts<R: Runtime>(client: &ComputeClient<R>, name: &str) {
        let lim = device_limits(client, name);
        for &(b, s, e, t, k, p) in WINDOWS {
            let jp = JoinParams { t, k, p };
            let fs = FieldSetup::new(b, s, e, jp).expect("field setup");
            if fs.nparts > 64 {
                continue; // keeps a software rasterizer's run short
            }
            // Width of a top-layer prefix, to count the survivors per prefix.
            let pw = fs.base.powu(fs.base.l - (t - p));
            let (mut survivors, mut checked, mut hits) = (0u64, 0u64, Vec::new());
            let (mut densest, mut single) = (0usize, 0usize);
            for v in 0..fs.nparts {
                let mut rec = Vec::new();
                let st = join_range(&fs.base, s, e, jp, Some(&[v]), Some(&mut rec));
                rec.retain(|&n| mid_mirror(&fs, n));
                survivors += st.survivors;
                checked += rec.len() as u64;
                hits.extend(st.hits);
                densest = densest.max(rec.len());
                let mut per_top: std::collections::HashMap<u128, usize> =
                    std::collections::HashMap::new();
                for n in &rec {
                    *per_top.entry(n / pw).or_default() += 1;
                }
                single = single.max(per_top.into_values().max().unwrap_or(0));
            }
            if densest <= single.max(1) {
                continue; // nothing to split
            }
            let cap = |c: usize| u32::try_from(c.max(1)).expect("a small window");
            let plan = JoinPlan::new(&fs, lim, 2, cap(densest / 2), 1, lim.budget)
                .expect("a two-slot layout fits");
            let retry = JoinPlan::new(&fs, lim, 1, cap(single), 1, lim.budget).expect("fits");
            let field = JoinField::with_plans(fs, plan, retry);
            let (nice, st) = run_field(client, &field, &JoinCeiling::default()).expect("tight run");
            hits.sort_unstable();
            hits.dedup();
            let got: Vec<u128> = nice.iter().map(|n| n.number).collect();
            assert_eq!(got, hits, "b{b} [{s}, {e}) {jp:?}: hits differ");
            assert_eq!(
                (st.survivors, st.checked),
                (survivors, checked),
                "b{b} [{s}, {e}) {jp:?}: survivor counts differ"
            );
            assert!(
                st.retried_partitions > 0 && st.splits > 0,
                "b{b} [{s}, {e}) {jp:?}: the tight layouts did not bite ({st:?})"
            );
            println!(
                "{name}: b{b} [{s}, {e}) {jp:?} tight layouts agree ({} partitions re-run, {} splits)",
                st.retried_partitions, st.splits
            );
        }
    }

    fn check_all<R: Runtime>(client: &ComputeClient<R>, name: &str) {
        let jp = JoinParams { t: 2, k: 1, p: 0 };
        let band = FieldSize::new(47, 100);
        let field = JoinField::prepare(10, &band, jp, device_limits(client, name))
            .expect("field setup")
            .expect("base 10 fits any device");
        let (hits, _) = run_field(client, &field, &JoinCeiling::default()).expect("join run");
        assert_eq!(hits.iter().map(|n| n.number).collect::<Vec<_>>(), vec![69]);
        check_window(client, 10, 47, 100, jp, None);
        for &(b, s, e, t, k, p) in WINDOWS {
            check_window(client, b, s, e, JoinParams { t, k, p }, None);
            println!("{name}: b{b} [{s}, {e}) ({t},{k},{p}) agrees");
        }
        let field = FieldSize::new(FRONTIER_57, FRONTIER_57 + 100_000_000_000_000);
        let jp = crate::overlap_join::join_params_for(57, &field).expect("a join field");
        check_window(
            client,
            57,
            field.start(),
            field.end(),
            jp,
            Some(vec![0, 1_000, 3_248]),
        );
        println!("{name}: b57 frontier field {jp:?}, partitions 0, 1000, 3248 agree");
        check_tight_layouts(client, name);
        check_slices(client, name);
        check_refused_layouts(client, name);
        check_production_shape(client, name);
    }

    /// The re-run and refusal paths at the production shape (the b57
    /// frontier field: t = L-3, k = 6, p = 2, key-level buckets), which the
    /// windows above are too small for, on three of its partitions: a
    /// survivor list half the densest partition's, so partitions overflow,
    /// run again alone and split their top layer; then the production
    /// layout refused. Hits and both survivor counts must equal the CPU
    /// reference's.
    fn check_production_shape<R: Runtime>(client: &ComputeClient<R>, name: &str) {
        let range = FieldSize::new(FRONTIER_57, FRONTIER_57 + 100_000_000_000_000);
        let lim = device_limits(client, name);
        let field = crate::join_plan::plan_join(57, &range, lim).expect("a join field");
        let fs = &field.fs;
        let parts = [0u32, 1_000, 3_248];
        let (mut survivors, mut checked, mut hits, mut densest) = (0u64, 0u64, Vec::new(), 0);
        for &v in &parts {
            let mut rec = Vec::new();
            let st = join_range(
                &fs.base,
                range.start(),
                range.end(),
                fs.jp,
                Some(&[u128::from(v)]),
                Some(&mut rec),
            );
            rec.retain(|&n| mid_mirror(fs, n));
            survivors += st.survivors;
            checked += rec.len() as u64;
            densest = densest.max(rec.len());
            hits.extend(st.hits);
        }
        hits.sort_unstable();
        hits.dedup();
        let cap = u32::try_from(densest / 2).expect("a partition's survivors fit u32");
        let plan = JoinPlan::new(fs, lim, 4, cap, 1, lim.budget).expect("a tight layout fits");
        let retry = JoinPlan::new(fs, lim, 1, cap, 1, lim.budget).expect("a tight re-run fits");
        let tight = JoinField::with_plans(fs.clone(), plan, retry);
        let (got, st) =
            run_slices(client, &tight, 1, &parts, &JoinCeiling::default()).expect("tight run");
        let got: Vec<u128> = got.iter().map(|n| n.number).collect();
        assert_eq!(got, hits, "tight layout: hits differ");
        assert_eq!(
            (st.survivors, st.checked),
            (survivors, checked),
            "tight layout: {st:?}"
        );
        assert!(
            st.retried_partitions > 0 && st.splits > 0,
            "tight layout: {st:?}"
        );
        REFUSE_ABOVE.with(|r| r.set(field.plan.bytes - 1));
        let refused = run_slices(client, &field, 1, &parts, &JoinCeiling::default());
        REFUSE_ABOVE.with(|r| r.set(usize::MAX));
        let (got, st_refused) = refused.expect("a smaller layout runs");
        let got: Vec<u128> = got.iter().map(|n| n.number).collect();
        assert_eq!(got, hits, "refused layout: hits differ");
        assert_eq!(
            (st_refused.survivors, st_refused.checked),
            (survivors, checked),
            "refused layout: {st_refused:?}"
        );
        assert!(st_refused.refused >= 1, "refused layout: {st_refused:?}");
        println!(
            "{name}: b57 frontier partitions {parts:?} agree in a tight layout ({} re-run, {} \
             splits) and after a refused one",
            st.retried_partitions, st.splits
        );
    }

    /// A device that does not run its planned layout (it runs nothing above
    /// a size, as the macOS runner's paravirtual Metal device does above
    /// about 0.65 GiB): the field comes out as it does where every layout
    /// runs, in layouts of at most half the refused one, and the ceiling
    /// stays down for the next field. A device that runs no layout at all
    /// fails the field instead of reporting it empty.
    fn check_refused_layouts<R: Runtime>(client: &ComputeClient<R>, name: &str) {
        let refuse_above = |bytes: usize| REFUSE_ABOVE.with(|r| r.set(bytes));
        let lim = device_limits(client, name);
        let mut tried = 0;
        for &(b, s, e, t, k, p) in WINDOWS {
            let (jp, range) = (JoinParams { t, k, p }, FieldSize::new(s, e));
            let Some(field) = JoinField::prepare(b, &range, jp, lim).expect("setup") else {
                continue;
            };
            if field.fs.nparts > 64 || field.fs.nparts < 8 {
                continue; // long on a software rasterizer, or one batch
            }
            let (want, st_want) =
                run_field(client, &field, &JoinCeiling::default()).expect("every layout runs");
            if st_want.batches == 0 {
                continue; // an empty top layer: no device work to refuse
            }
            let ceiling = JoinCeiling::default();
            refuse_above(field.plan.bytes - 1);
            let refused = run_field(client, &field, &ceiling);
            refuse_above(usize::MAX);
            let (got, st) = refused.expect("a smaller layout runs");
            assert_eq!(got, want, "b{b} [{s}, {e}) {jp:?}: hits differ");
            assert_eq!(
                (st.survivors, st.checked),
                (st_want.survivors, st_want.checked),
                "b{b} [{s}, {e}) {jp:?}: survivor counts differ"
            );
            assert!(st.refused >= 1, "b{b} [{s}, {e}) {jp:?}: {st:?}");
            assert!(ceiling.get() <= field.plan.bytes / 2);
            // The next field starts within the ceiling: nothing refused.
            refuse_above(ceiling.get());
            let again = run_field(client, &field, &ceiling);
            refuse_above(0);
            let none = run_field(client, &field, &JoinCeiling::default());
            refuse_above(usize::MAX);
            assert_eq!(again.expect("within the ceiling").1.refused, 0);
            let err = none.expect_err("a device that runs nothing").to_string();
            assert!(err.contains("did not run"), "{err}");
            tried += 1;
            println!(
                "{name}: b{b} [{s}, {e}) {jp:?} agrees after the device refused a {} MiB \
                 layout ({} refusals, ceiling {} MiB)",
                field.plan.bytes >> 20,
                st.refused,
                ceiling.get() >> 20
            );
        }
        assert!(tried > 0, "{name}: no window to refuse a layout of");
    }

    /// A field run slice by slice (here cut into about three) finds what it
    /// finds whole: the same hits, join survivors and prefilter survivors.
    fn check_slices<R: Runtime>(client: &ComputeClient<R>, name: &str) {
        let lim = device_limits(client, name);
        let mut sliced_any = 0;
        for &(b, s, e, t, k, p) in WINDOWS {
            let (jp, range) = (JoinParams { t, k, p }, FieldSize::new(s, e));
            let Some(whole) = JoinField::prepare(b, &range, jp, lim).expect("setup") else {
                continue;
            };
            let block = crate::overlap_join::prefix_block(b, whole.fs.base.l, jp);
            let prefixes = range.last() / block - range.start() / block + 1;
            if prefixes < 3 || whole.fs.nparts > 64 {
                continue; // nothing to cut, or long on a software rasterizer
            }
            let ceiling = JoinCeiling::default();
            let (all, st_all) = run_field(client, &whole, &ceiling).expect("whole run");
            let field = whole.resliced(prefixes.div_ceil(3)).expect("slices");
            let (got, st) = run_field(client, &field, &ceiling).expect("sliced run");
            assert!(st.slices >= 2, "b{b} [{s}, {e}) {jp:?}: {st:?}");
            assert_eq!(got, all, "b{b} [{s}, {e}) {jp:?}: hits differ");
            assert_eq!(
                (st.survivors, st.checked),
                (st_all.survivors, st_all.checked),
                "b{b} [{s}, {e}) {jp:?}: survivor counts differ in {} slices",
                st.slices
            );
            sliced_any += 1;
        }
        assert!(sliced_any > 0, "{name}: no window was cut into slices");
        println!("{name}: {sliced_any} windows agree whole and in slices");
    }

    /// The benchmark's sample of a production field: partitions of the b57
    /// frontier field's first (only) slice, timed through the production
    /// plan, with the CPU join's counts; and a field below the size gate
    /// gets the reason instead. It prints the device's limits, the plan and
    /// any layouts the device refused, and runs the same partition with a
    /// 4-slot test layout too, so a device that mishandles the production
    /// layout shows which part.
    #[test_log::test]
    #[ignore = "requires a wgpu device"]
    fn join_sample_runs_partitions_of_a_production_field() {
        let ctx = CubeclContext::new_default().expect("CubeCL init");
        let range = FieldSize::new(FRONTIER_57, FRONTIER_57 + 100_000_000_000_000);
        let cpu = crate::cpu_join::CpuJoin::for_field(57, &range).expect("a join field");
        let want = cpu.run_partition(1_000, &mut crate::cpu_join::Scratch::default());
        let (client, name) = client();
        // Planned and run within the ceiling the client starts this device
        // at, as `join_sample` below is.
        let ceiling = JoinCeiling::for_device(&name);
        let lim = device_limits(&client, &name);
        let field = crate::join_plan::plan_join(57, &range, lim).expect("a join field");
        println!(
            "{name}: limits {lim:?}; {} slice(s); plan {:?}; re-run {:?}",
            field.slices.len(),
            field.plan,
            field.retry
        );
        let (_, prod) =
            run_slices(&client, &field, 1, &[1_000], &ceiling).expect("production layout");
        let four = JoinField::with_plans(
            field.fs.clone(),
            test_plan(&field.fs, limits_of(&client), 4),
            field.retry,
        );
        let (_, test) =
            run_slices(&client, &four, 1, &[1_000], &JoinCeiling::default()).expect("test layout");
        println!(
            "partition 1000: CPU ({}, {}); production layout ({}, {}, {} re-run, {} refused, \
             ceiling {}); 4-slot layout ({}, {})",
            want.survivors,
            want.checked,
            prod.survivors,
            prod.checked,
            prod.retried_partitions,
            prod.refused,
            match ceiling.get() {
                usize::MAX => "none".to_string(),
                c => format!("{} MiB", c >> 20),
            },
            test.survivors,
            test.checked,
        );
        if let Ok(m) = client.memory_usage() {
            println!("{m}");
        }
        assert_eq!(
            (prod.survivors, prod.checked),
            (want.survivors, want.checked)
        );
        let sample = ctx
            .join_sample(57, &range, &[1_000])
            .expect("device run")
            .expect("a join field");
        assert_eq!(
            (sample.slices, sample.partitions, sample.sampled),
            (1, 3_249, 1)
        );
        assert_eq!(
            (sample.survivors, sample.checked),
            (want.survivors, want.checked)
        );
        assert_eq!(sample.hits, want.hits);
        assert!(sample.setup_secs > 0.0 && sample.run_secs > 0.0);
        let small = FieldSize::new(FRONTIER_57, FRONTIER_57 + 4_000_000_000);
        assert_eq!(
            ctx.join_sample(57, &small, &[1_000])
                .expect("no device work")
                .err(),
            Some(crate::overlap_join::StrideReason::BelowMinSize)
        );
    }

    /// A field with dense buckets (base 58, 1e15), on `NICE_TEST_JOIN_DENSE`
    /// partitions spread over it: the hits and both survivor counts must
    /// equal the CPU join's. This is the field where the join kernel's
    /// shared stage overran most (when it held unprefiltered survivors,
    /// about 2e-5 of them here); a chunk that overruns it now writes only
    /// prefiltered survivors to the list, so the counts stay exact.
    /// `NICE_TEST_JOIN_DENSE_FIELD=base:start:size` runs another field (of
    /// its first slice) instead.
    #[test]
    #[cfg(feature = "cubecl-cuda")]
    #[ignore = "requires an NVIDIA device; opt-in, NICE_TEST_JOIN_DENSE=partitions"]
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    fn cubecl_cuda_join_matches_the_cpu_join_on_a_dense_field() {
        let Ok(n) = std::env::var("NICE_TEST_JOIN_DENSE") else {
            eprintln!("skipping: set NICE_TEST_JOIN_DENSE to a partition count");
            return;
        };
        let n: u32 = n.parse().expect("NICE_TEST_JOIN_DENSE: a count");
        let ctx = CubeclContext::new_cuda(0).expect("CubeCL CUDA init");
        let CubeclContext::Cuda { client, .. } = ctx else {
            unreachable!("new_cuda is CUDA")
        };
        let (b, dense, size) = std::env::var("NICE_TEST_JOIN_DENSE_FIELD").map_or(
            (58, crate::join_plan::tests::DENSE_58, 1_000_000_000_000_000),
            |v| {
                let x: Vec<&str> = v.split(':').map(str::trim).collect();
                (
                    x[0].parse().expect("NICE_TEST_JOIN_DENSE_FIELD: a base"),
                    x[1].parse().expect("NICE_TEST_JOIN_DENSE_FIELD: a start"),
                    x[2].parse::<f64>()
                        .expect("NICE_TEST_JOIN_DENSE_FIELD: a size") as u128,
                )
            },
        );
        let range = FieldSize::new(dense, dense + size);
        let field =
            crate::join_plan::plan_join(b, &range, limits_of(&client)).expect("a join field");
        let total = u32::try_from(field.fs.nparts).unwrap();
        let parts: Vec<u32> = (0..n).map(|i| i * total / n.max(1)).collect();
        let t0 = Instant::now();
        let (hits, st) =
            run_slices(&client, &field, 1, &parts, &JoinCeiling::default()).expect("device run");
        let gpu_secs = t0.elapsed().as_secs_f64();
        let cpu =
            crate::cpu_join::CpuJoin::new(b, &field.slices[0], field.fs.jp).expect("cpu setup");
        let mut sc = crate::cpu_join::Scratch::default();
        let mut want = crate::cpu_join::PartitionResult::default();
        for &v in &parts {
            want.add(cpu.run_partition(v, &mut sc));
        }
        want.hits.sort_unstable();
        let got: Vec<u128> = hits.iter().map(|h| h.number).collect();
        println!(
            "b{b} [{dense}, +{size}), {n} partitions: {} slice(s), {} per launch (fewest {}), \
             {} re-run; survivors {} checked {} on the device in {gpu_secs:.2}s",
            field.slices.len(),
            st.slots,
            st.min_slots,
            st.retried_partitions,
            st.survivors,
            st.checked
        );
        assert_eq!(got, want.hits, "hits differ");
        assert_eq!(
            (st.survivors, st.checked),
            (want.survivors, want.checked),
            "join or prefilter survivors differ from the CPU join's"
        );
    }

    #[test]
    #[ignore = "requires a wgpu device"]
    fn cubecl_join_matches_the_cpu_reference() {
        let (client, name) = client();
        check_all(&client, &name);
    }

    /// The same on `CubeCL`'s CUDA runtime, which is what NVIDIA hosts run.
    #[test]
    #[cfg(feature = "cubecl-cuda")]
    #[ignore = "requires an NVIDIA device"]
    fn cubecl_cuda_join_matches_the_cpu_reference() {
        let ctx = CubeclContext::new_cuda(0).expect("CubeCL CUDA init");
        let name = ctx.device_name();
        let CubeclContext::Cuda { client, .. } = ctx else {
            unreachable!("new_cuda is CUDA")
        };
        check_all(&client, &name);
    }

    fn numbers(results: &crate::FieldResults) -> Vec<u128> {
        let mut v: Vec<u128> = results.nice_numbers.iter().map(|n| n.number).collect();
        v.sort_unstable();
        v
    }

    /// The backend keeps its two pipelines apart by ticket: base 10's band
    /// through the join (forced parameters), the stride pipeline, then the
    /// join again, each finished with its own ticket. A join ticket handed
    /// back ahead of an older join field is refused, and both still finish.
    #[test]
    #[ignore = "requires a wgpu device"]
    fn cubecl_niceonly_keeps_join_and_stride_apart_by_ticket() {
        let ctx = CubeclContext::new_default().expect("CubeCL init");
        let band = FieldSize::new(47, 100);
        let lim = ctx.join_limits().expect("CubeCL has the join");
        let join_field = || {
            JoinField::prepare(10, &band, JoinParams { t: 2, k: 1, p: 0 }, lim)
                .unwrap()
                .expect("base 10 fits any device")
        };
        let tickets = [
            ctx.begin_join(join_field()).unwrap(),
            ctx.begin_stride(&band, 10).unwrap(),
            ctx.begin_join(join_field()).unwrap(),
        ];
        for ticket in tickets {
            let route = ticket.route();
            let (res, stats) = ctx.finish(ticket).unwrap();
            assert_eq!(
                stats.overlap_join,
                route == Route::Join,
                "a ticket brought back the other pipeline's field"
            );
            assert_eq!(numbers(&res), vec![69]);
        }
        let first = ctx.begin_join(join_field()).unwrap();
        let second = ctx.begin_join(join_field()).unwrap();
        let second_again = FieldTicket::new(second.route(), second.seq());
        assert!(
            ctx.finish(second).is_err(),
            "a ticket out of order was accepted"
        );
        ctx.finish(first).unwrap();
        ctx.finish(second_again).unwrap();
        // And the route leaves a band this small to the stride pipeline.
        assert_eq!(
            numbers(&process_niceonly(&ctx, &band, 10).unwrap()),
            vec![69]
        );
    }

    /// The client's route on one field: `begin_niceonly` must send it to the
    /// join. Returns the join's results and its own account of the field.
    fn run_routed(
        ctx: &CubeclContext,
        base: u32,
        range: &FieldSize,
    ) -> (Vec<u128>, NiceonlyStats, JoinFieldStats) {
        let NiceonlyStarted::Queued(ticket) = begin_niceonly(ctx, range, base).expect("begin")
        else {
            panic!("a production field is queued");
        };
        assert_eq!(ticket.route(), Route::Join, "the join takes the field");
        let (res, stats) = ctx.finish(ticket).expect("finish");
        let js = last_join_stats(ctx).expect("join stats");
        (numbers(&res), stats, js)
    }

    /// The stride pipeline on the same field (the client's path before the
    /// join).
    fn run_stride(ctx: &CubeclContext, base: u32, range: &FieldSize) -> Vec<u128> {
        let ticket = ctx.begin_stride(range, base).expect("begin");
        numbers(&ctx.finish(ticket).expect("finish").0)
    }

    /// Opt-in (`NICE_TEST_JOIN_FULL_FIELD=1`; minutes of CPU): one whole
    /// production-size field (base 42, 1e13, the gate's smallest) through
    /// the client's route, against the CPU reference over every partition
    /// (threaded) — the same hits, join survivors and prefilter survivors —
    /// and against the stride pipeline, which must find the same nice
    /// numbers.
    fn full_field_check(ctx: &CubeclContext) {
        if std::env::var("NICE_TEST_JOIN_FULL_FIELD").is_err() {
            eprintln!("skipping: set NICE_TEST_JOIN_FULL_FIELD to run the whole-field comparison");
            return;
        }
        let (base, start, size) = PLAN_FIELDS[0];
        let range = FieldSize::new(start, start + size);
        let jp = crate::overlap_join::join_params_for(base, &range).expect("a join field");
        let fs = FieldSetup::new(base, range.start(), range.end(), jp).expect("field setup");
        let t = std::time::Instant::now();
        let (survivors, checked, hits) = reference_field(&fs);
        let cpu_secs = t.elapsed().as_secs_f64();
        let (got, _, js) = run_routed(ctx, base, &range);
        assert_eq!(got, hits, "b{base} {range:?}: hits differ");
        assert_eq!(
            (js.survivors, js.checked),
            (survivors, checked),
            "b{base} {range:?}: survivor counts differ"
        );
        let t = std::time::Instant::now();
        let stride = run_stride(ctx, base, &range);
        assert_eq!(
            stride, hits,
            "b{base} {range:?}: the stride pipeline differs"
        );
        eprintln!(
            "FULL FIELD device={} b{base} {range:?} {jp:?}: {} partitions, {survivors} join \
             survivors, {checked} checked, hits {hits:?}; reference {cpu_secs:.0}s, join \
             {:.2}s, stride {:.2}s",
            ctx.device_name(),
            fs.nparts,
            js.device_secs,
            t.elapsed().as_secs_f64()
        );
    }

    #[test]
    #[ignore = "requires a wgpu device; opt-in, minutes of CPU"]
    fn join_matches_the_reference_on_a_whole_field() {
        full_field_check(&CubeclContext::new_default().expect("CubeCL init"));
    }

    #[test]
    #[cfg(feature = "cubecl-cuda")]
    #[ignore = "requires an NVIDIA device; opt-in, minutes of CPU"]
    fn cubecl_cuda_join_matches_the_reference_on_a_whole_field() {
        full_field_check(&CubeclContext::new_cuda(0).expect("CubeCL CUDA init"));
    }

    /// Fields of 1e14 per base for the throughput harness: consecutive from
    /// the base-57 frontier, and from one of Dan Stoyell's benchmark fields
    /// at bases 60 and 64.
    const THROUGHPUT_STARTS: &[(u32, u128)] = &[
        (57, FRONTIER_57),
        // EXPERIMENTS item 18's b58 GPU field.
        (58, 130_525_216_587_923_470_200),
        (60, 1_573_714_731_429_953_349_518),
        // `PLAN_FIELDS`' b62 start.
        (62, 7_997_740_455_941_656_911_841),
        (64, 41_242_006_262_957_161_709_568),
    ];

    /// Throughput of production fields through the client's route
    /// (`plan_join`, then the join's pipeline with `FIELDS_IN_FLIGHT`
    /// fields queued, as the client runs them): `NICE_TEST_JOIN_FIELDS`
    /// fields per base (also the opt-in, since the parity workflow runs this
    /// module's ignored tests on lavapipe) after one untimed warm-up field,
    /// for the bases in `NICE_TEST_JOIN_BASES` (default 57,60,64). Prints per
    /// base the rate, per-field wall, setup and device seconds, the layout
    /// (partitions per launch, MiB), re-runs, survivor counts and the
    /// device's memory. `NICE_TEST_JOIN_COMPARE=1` also runs every field
    /// through the stride pipeline and asserts the same nice numbers.
    /// `NICE_TEST_JOIN_SIZE` sets the field size (default 1e14; the warm-up
    /// field stays at most 1e14), and `NICE_TEST_JOIN_STARTS=b:start,...`
    /// overrides the table's starts.
    #[allow(
        clippy::cast_precision_loss,
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        clippy::too_many_lines
    )]
    fn join_throughput(ctx: &CubeclContext) {
        use crate::cubecl_backend::memory_usage;
        use crate::gpu_niceonly::FIELDS_IN_FLIGHT;
        let Ok(n) = std::env::var("NICE_TEST_JOIN_FIELDS") else {
            eprintln!("skipping: set NICE_TEST_JOIN_FIELDS to run the throughput harness");
            return;
        };
        let n: u128 = n.parse().expect("NICE_TEST_JOIN_FIELDS must be a count");
        let bases: Vec<u32> = std::env::var("NICE_TEST_JOIN_BASES")
            .unwrap_or_else(|_| "57,60,64".into())
            .split(',')
            .map(|b| b.trim().parse().expect("NICE_TEST_JOIN_BASES: bases"))
            .collect();
        let compare = std::env::var("NICE_TEST_JOIN_COMPARE").is_ok();
        let size: u128 = std::env::var("NICE_TEST_JOIN_SIZE").map_or(100_000_000_000_000, |v| {
            v.parse::<f64>().expect("NICE_TEST_JOIN_SIZE: a field size") as u128
        });
        let overrides: Vec<(u32, u128)> = std::env::var("NICE_TEST_JOIN_STARTS")
            .unwrap_or_default()
            .split(',')
            .filter(|x| !x.trim().is_empty())
            .map(|x| {
                let (b, s) = x.split_once(':').expect("NICE_TEST_JOIN_STARTS: b:start");
                (
                    b.trim().parse().expect("a base"),
                    s.trim().replace('_', "").parse().expect("a start"),
                )
            })
            .collect();
        let warm_size = size.min(100_000_000_000_000);
        let mib = |b: u64| b >> 20;
        for base in bases {
            let &(_, start) = overrides
                .iter()
                .chain(THROUGHPUT_STARTS)
                .find(|&&(b, _)| b == base)
                .expect("a base with throughput fields (57, 58, 60, 62, 64)");
            let warm = FieldSize::new(start - warm_size, start);
            run_routed(ctx, base, &warm);
            let fields: Vec<FieldSize> = (0..n)
                .map(|i| FieldSize::new(start + i * size, start + (i + 1) * size))
                .collect();
            let lookahead = FIELDS_IN_FLIGHT.saturating_sub(1);
            let t = std::time::Instant::now();
            let (mut queued, mut found) = (std::collections::VecDeque::new(), Vec::new());
            let mut per_field: Vec<(NiceonlyStats, JoinFieldStats)> = Vec::new();
            // Device memory, sampled while the next field is on the device.
            let mut peak = (0u64, 0u64);
            let mut finish = |ticket: FieldTicket, found: &mut Vec<u128>| {
                if let Some((in_use, reserved)) = memory_usage(ctx) {
                    peak = (peak.0.max(in_use), peak.1.max(reserved));
                }
                let (res, stats) = ctx.finish(ticket).expect("finish");
                found.extend(res.nice_numbers.iter().map(|x| x.number));
                per_field.push((stats, last_join_stats(ctx).expect("join stats")));
            };
            for f in &fields {
                let NiceonlyStarted::Queued(ticket) = begin_niceonly(ctx, f, base).expect("begin")
                else {
                    panic!("a production field is queued");
                };
                assert_eq!(ticket.route(), Route::Join, "the join takes the field");
                queued.push_back(ticket);
                while queued.len() > lookahead {
                    finish(queued.pop_front().expect("queued"), &mut found);
                }
            }
            for ticket in queued {
                finish(ticket, &mut found);
            }
            let secs = t.elapsed().as_secs_f64();
            let nf = per_field.len() as f64;
            let mean = |f: &dyn Fn(&(NiceonlyStats, JoinFieldStats)) -> f64| {
                per_field.iter().map(f).sum::<f64>() / nf
            };
            let sum =
                |f: &dyn Fn(&JoinFieldStats) -> u64| per_field.iter().map(|x| f(&x.1)).sum::<u64>();
            let (in_use, reserved) = peak;
            eprintln!(
                "JOIN THROUGHPUT device={} base={base} fields={n} in_flight={} secs={secs:.2} \
                 per_field={:.3} rate={:.3e} n/s | setup={:.3}s device={:.3}s wall={:.3}s \
                 batches={:.0} slots={} plan_mib={} | empty_top={} retried={} splits={} \
                 survivors={} checked={} found={} | peak_in_use_mib={} peak_reserved_mib={}",
                ctx.device_name(),
                FIELDS_IN_FLIGHT,
                secs / nf,
                (n * size) as f64 / secs,
                mean(&|x| x.0.msd_secs),
                mean(&|x| x.0.device_secs),
                mean(&|x| x.0.total_secs),
                mean(&|x| f64::from(x.0.launches)),
                per_field.iter().map(|x| x.1.slots).max().unwrap_or(0),
                per_field.iter().map(|x| x.1.bytes >> 20).max().unwrap_or(0),
                per_field.iter().filter(|x| x.1.batches == 0).count(),
                sum(&|x| x.retried_partitions as u64),
                sum(&|x| u64::from(x.splits)),
                sum(&|x| x.survivors),
                sum(&|x| x.checked),
                found.len(),
                mib(in_use),
                mib(reserved),
            );
            if compare {
                let t = std::time::Instant::now();
                let mut stride = Vec::new();
                for f in &fields {
                    stride.extend(run_stride(ctx, base, f));
                }
                found.sort_unstable();
                stride.sort_unstable();
                assert_eq!(stride, found, "b{base}: the stride pipeline differs");
                eprintln!(
                    "STRIDE base={base} fields={n} secs={:.2} per_field={:.3} (one at a time) \
                     found={}",
                    t.elapsed().as_secs_f64(),
                    t.elapsed().as_secs_f64() / nf,
                    stride.len()
                );
            }
        }
    }

    #[test]
    #[ignore = "requires a wgpu device; prints throughput"]
    fn join_throughput_fixed_fields() {
        join_throughput(&CubeclContext::new_default().expect("CubeCL init"));
    }

    #[test]
    #[cfg(feature = "cubecl-cuda")]
    #[ignore = "requires an NVIDIA device; prints throughput"]
    fn cubecl_cuda_join_throughput_fixed_fields() {
        join_throughput(&CubeclContext::new_cuda(0).expect("CubeCL CUDA init"));
    }

    // ---------------------------------------------------------------------
    // The top certificate: the remainder form against the lockstep scan it
    // replaced and against the CPU's `Base::cert`.
    // ---------------------------------------------------------------------

    /// The device side of the certificate test, under `kernels`' lint
    /// allowances (ported integer arithmetic, comptime-unrolled loops).
    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_lossless,
        clippy::too_many_arguments,
        clippy::too_many_lines,
        clippy::similar_names,
        clippy::many_single_char_names,
        clippy::collapsible_if,
        clippy::needless_range_loop,
        clippy::trivially_copy_pass_by_ref,
        clippy::collapsible_else_if,
        clippy::manual_midpoint
    )]
    mod cert_kernels {
        use super::super::kernels::{cert_power, pack16, powers_diff, top_cert};
        use cubecl::prelude::*;

        /// a² and e² (4L limbs) and a³ and e³ (6L limbs) of a and e (`limbs` =
        /// L u32 words each), in 16-bit limbs one to a u32, by schoolbook in
        /// u32, as the top certificate formed them before `powers_diff`: the
        /// lockstep scan's input and the reference `powers_diff` must match.
        #[cube]
        fn powers16(
            a: &Array<u32>,
            e: &Array<u32>,
            a2: &mut Array<u32>,
            e2: &mut Array<u32>,
            a3: &mut Array<u32>,
            e3: &mut Array<u32>,
            #[comptime] limbs: u32,
        ) {
            let nh = comptime!(2 * limbs);
            let nh2 = comptime!(4 * limbs);
            let nh3 = comptime!(6 * limbs);
            let mut ah = Array::<u32>::new(nh as usize);
            let mut eh = Array::<u32>::new(nh as usize);
            #[unroll]
            for j in 0..limbs {
                ah[comptime!(2 * j) as usize] = a[j as usize] & 0xFFFFu32;
                ah[comptime!(2 * j + 1) as usize] = a[j as usize] >> 16u32;
                eh[comptime!(2 * j) as usize] = e[j as usize] & 0xFFFFu32;
                eh[comptime!(2 * j + 1) as usize] = e[j as usize] >> 16u32;
            }
            #[unroll]
            for j in 0..nh2 {
                a2[j as usize] = 0u32;
                e2[j as usize] = 0u32;
            }
            #[unroll]
            for i1 in 0..nh {
                let mut ca = 0u32;
                let mut ce = 0u32;
                #[unroll]
                for j in 0..nh {
                    let kk = comptime!(i1 + j);
                    let ta = ah[i1 as usize] * ah[j as usize] + a2[kk as usize] + ca;
                    a2[kk as usize] = ta & 0xFFFFu32;
                    ca = ta >> 16u32;
                    let te = eh[i1 as usize] * eh[j as usize] + e2[kk as usize] + ce;
                    e2[kk as usize] = te & 0xFFFFu32;
                    ce = te >> 16u32;
                }
                a2[comptime!(i1 + nh) as usize] = ca;
                e2[comptime!(i1 + nh) as usize] = ce;
            }
            #[unroll]
            for j in 0..nh3 {
                a3[j as usize] = 0u32;
                e3[j as usize] = 0u32;
            }
            #[unroll]
            for i1 in 0..nh2 {
                let mut ca = 0u32;
                let mut ce = 0u32;
                #[unroll]
                for j in 0..nh {
                    let kk = comptime!(i1 + j);
                    let ta = a2[i1 as usize] * ah[j as usize] + a3[kk as usize] + ca;
                    a3[kk as usize] = ta & 0xFFFFu32;
                    ca = ta >> 16u32;
                    let te = e2[i1 as usize] * eh[j as usize] + e3[kk as usize] + ce;
                    e3[kk as usize] = te & 0xFFFFu32;
                    ce = te >> 16u32;
                }
                a3[comptime!(i1 + nh) as usize] = ca;
                e3[comptime!(i1 + nh) as usize] = ce;
            }
        }

        /// y - x (y >= x) in `nl` u32 words, from 16-bit limbs.
        #[cube]
        fn sub_words16(x: &Array<u32>, y: &Array<u32>, d: &mut Array<u32>, #[comptime] nl: u32) {
            let mut xw = Array::<u32>::new(nl as usize);
            let mut yw = Array::<u32>::new(nl as usize);
            pack16(x, &mut xw, nl);
            pack16(y, &mut yw, nl);
            let mut borrow = 0u32;
            #[unroll]
            for j in 0..nl {
                let yj = yw[j as usize];
                let xj = xw[j as usize];
                d[j as usize] = yj - xj - borrow;
                borrow = select(yj < xj || (yj == xj && borrow != 0u32), 1u32, 0u32);
            }
        }

        /// `kernels::cert_power` as it was before it divided out only x's
        /// digits: both endpoints' powers turned into digits in lockstep, least
        /// significant first, where a position whose digits differ resets the
        /// run. The reference the remainder form must reproduce exactly.
        #[cube]
        fn cert_power_lockstep(
            x: &mut Array<u32>,
            y: &mut Array<u32>,
            pw: &Array<u32>,
            out_c0: &mut Array<u32>,
            out_dup: &mut Array<u32>,
            out_dom: &mut Array<u32>,
            #[comptime] nl: u32,
            #[comptime] nl3: u32,
            #[comptime] sp: u32,
            #[comptime] cap: u32,
            #[comptime] base: u32,
            #[comptime] chunk_digits: u32,
            #[comptime] chunk_div: u32,
        ) -> u64 {
            let nh = comptime!(2 * nl);
            // d = y - x (y >= x), over u32 words packed from the limb pairs.
            let mut d = Array::<u32>::new(nl as usize);
            let mut borrow = 0u32;
            #[unroll]
            for j in 0..nl {
                let yj = y[comptime!(2 * j) as usize] | (y[comptime!(2 * j + 1) as usize] << 16u32);
                let xj = x[comptime!(2 * j) as usize] | (x[comptime!(2 * j + 1) as usize] << 16u32);
                let t = yj - xj - borrow;
                let mut nb = 0u32;
                if yj < xj || (yj == xj && borrow != 0u32) {
                    nb = 1u32;
                }
                d[j as usize] = t;
                borrow = nb;
            }
            // ndig(d): smallest i with b^i > d, binary search over the power table.
            let mut lo = 0u32;
            let mut hi = sp.runtime();
            while lo < hi {
                let mid = (lo + hi) >> 1u32;
                // d < b^mid ?
                let mut lt = false;
                let mut eq = true;
                #[unroll]
                for jj in 0..nl3 {
                    let j = comptime!(nl3 - 1 - jj);
                    let pj = pw[(mid * nl3 + j) as usize];
                    let mut dj = 0u32;
                    if comptime!(j < nl) {
                        dj = d[j as usize];
                    }
                    if eq {
                        if dj < pj {
                            lt = true;
                            eq = false;
                        } else if dj > pj {
                            eq = false;
                        }
                    }
                }
                if lt {
                    hi = mid;
                } else {
                    lo = mid + 1u32;
                }
            }
            let mut c0 = lo;
            if c0 < cap {
                c0 = cap.runtime();
            }
            let mut run = 0u64;
            let mut dup = 0u32;
            let mut dlo = 1u32;
            let mut dhi = 0u32;
            let passes = comptime!(sp.div_ceil(chunk_digits));
            #[unroll]
            for pass in 0..passes {
                // The top two limbs as one u32 (see above), then one limb a step.
                let top = comptime!(nh - 1);
                let next = comptime!(nh - 2);
                let cx = (x[top as usize] << 16u32) | x[next as usize];
                let qx = cx / chunk_div;
                let mut rx = cx - qx * chunk_div;
                x[top as usize] = qx >> 16u32;
                x[next as usize] = qx & 0xFFFFu32;
                let cy = (y[top as usize] << 16u32) | y[next as usize];
                let qy = cy / chunk_div;
                let mut ry = cy - qy * chunk_div;
                y[top as usize] = qy >> 16u32;
                y[next as usize] = qy & 0xFFFFu32;
                #[unroll]
                for jj in 2..nh {
                    let j = comptime!(nh - 1 - jj);
                    let c1 = (rx << 16u32) | x[j as usize];
                    let q1 = c1 / chunk_div;
                    rx = c1 - q1 * chunk_div;
                    x[j as usize] = q1;
                    let e1 = (ry << 16u32) | y[j as usize];
                    let s1 = e1 / chunk_div;
                    ry = e1 - s1 * chunk_div;
                    y[j as usize] = s1;
                }
                #[unroll]
                for q in 0..chunk_digits {
                    let pos = comptime!(pass * chunk_digits + q);
                    let dx = rx % base;
                    rx /= base;
                    let dy = ry % base;
                    ry /= base;
                    if comptime!(pos < sp) {
                        if pos >= c0 {
                            if dx == dy {
                                let bit = 1u64 << u64::cast_from(dx);
                                if (run & bit) != 0u64 {
                                    dup = 1u32;
                                }
                                run |= bit;
                            } else {
                                run = 0u64;
                                dup = 0u32;
                                dlo = dx;
                                dhi = dy;
                            }
                        }
                    }
                }
            }
            out_c0[0] = c0;
            out_dup[0] = dup;
            out_dom[0] = dlo;
            out_dom[1] = dhi;
            run
        }

        /// Words per interval in [`cert_test_kernel`]'s output: for the square
        /// and then the cube, `cert_power`'s (run lo, run hi, dup, dom lo, dom
        /// hi, c0) at 0 and 6 and `cert_power_lockstep`'s at 12 and 18; then
        /// `top_cert`'s (kept, mask lo, mask hi, c0 of the square, of the cube)
        /// at 24; at 29, which of `powers_diff`'s d², d³, a² and a³ (bits 0-3)
        /// differ from the full powers' in any word (0).
        pub(super) const CERT_OUT: u32 = 32;

        /// Writes one power's certificate state at `o` (see [`CERT_OUT`]).
        #[cube]
        fn put_power(
            out: &mut Array<u32>,
            o: u32,
            run: u64,
            c0: &Array<u32>,
            dup: &Array<u32>,
            dom: &Array<u32>,
        ) {
            out[o as usize] = u32::cast_from(run);
            out[(o + 1u32) as usize] = u32::cast_from(run >> 32u64);
            out[(o + 2u32) as usize] = dup[0];
            out[(o + 3u32) as usize] = dom[0];
            out[(o + 4u32) as usize] = dom[1];
            out[(o + 5u32) as usize] = c0[0];
        }

        /// Per interval `[a, e]` (`limbs` u32 words each, a then e, in `iv`;
        /// e - a fits `dw` words), one `part` of the output [`CERT_OUT`]
        /// describes: 0, both powers' certificates by `cert_power` from
        /// `powers_diff`, and `powers_diff` against `powers16`; 1, by
        /// `cert_power_lockstep` from `powers16`; 2, `top_cert`'s. One part per kernel: `CubeCL`'s SPIR-V compiler
        /// recurses once per basic block, and one kernel holding all three
        /// (six power scans) overflowed the 2 MiB stack of the thread that
        /// compiles it at the 3-word shapes of b47 and b50. A part is about
        /// the size of a top kernel.
        #[cube(launch_unchecked)]
        pub(super) fn cert_test_kernel(
            iv: &Array<u32>,
            pw: &Array<u32>,
            out: &mut Array<u32>,
            n: u32,
            #[comptime] base: u32,
            #[comptime] limbs: u32,
            #[comptime] cap: u32,
            #[comptime] s2: u32,
            #[comptime] s3: u32,
            #[comptime] chunk_digits: u32,
            #[comptime] chunk_div: u32,
            #[comptime] part: u32,
            #[comptime] dw: u32,
        ) {
            let i = ABSOLUTE_POS_X;
            if i < n {
                let nl2 = comptime!(2 * limbs);
                let nl3 = comptime!(3 * limbs);
                let nh2 = comptime!(4 * limbs);
                let nh3 = comptime!(6 * limbs);
                let mut a = Array::<u32>::new(limbs as usize);
                let mut e = Array::<u32>::new(limbs as usize);
                #[unroll]
                for j in 0..limbs {
                    a[j as usize] = iv[(2u32 * limbs * i + j) as usize];
                    e[j as usize] = iv[(2u32 * limbs * i + limbs + j) as usize];
                }
                let o = i * CERT_OUT;
                if comptime!(part == 2) {
                    let mut res = Array::<u32>::new(3usize);
                    let mask = top_cert(
                        &a,
                        &e,
                        pw,
                        &mut res,
                        base,
                        limbs,
                        cap,
                        s2,
                        s3,
                        chunk_digits,
                        chunk_div,
                        dw,
                    );
                    out[(o + 24u32) as usize] = res[0];
                    out[(o + 25u32) as usize] = u32::cast_from(mask);
                    out[(o + 26u32) as usize] = u32::cast_from(mask >> 32u64);
                    out[(o + 27u32) as usize] = res[1];
                    out[(o + 28u32) as usize] = res[2];
                } else {
                    let mut a2 = Array::<u32>::new(nh2 as usize);
                    let mut e2 = Array::<u32>::new(nh2 as usize);
                    let mut a3 = Array::<u32>::new(nh3 as usize);
                    let mut e3 = Array::<u32>::new(nh3 as usize);
                    powers16(&a, &e, &mut a2, &mut e2, &mut a3, &mut e3, limbs);
                    let mut c0 = Array::<u32>::new(1usize);
                    let mut dup = Array::<u32>::new(1usize);
                    let mut dom = Array::<u32>::new(2usize);
                    if comptime!(part == 0) {
                        // `powers_diff` against the full powers, word for word.
                        let mut r2 = Array::<u32>::new(nl2 as usize);
                        let mut r3 = Array::<u32>::new(nl3 as usize);
                        sub_words16(&a2, &e2, &mut r2, nl2);
                        sub_words16(&a3, &e3, &mut r3, nl3);
                        let mut x2 = Array::<u32>::new(nh2 as usize);
                        let mut x3 = Array::<u32>::new(nh3 as usize);
                        let mut d2 = Array::<u32>::new(nl2 as usize);
                        let mut d3 = Array::<u32>::new(nl3 as usize);
                        powers_diff(&a, &e, &mut x2, &mut x3, &mut d2, &mut d3, limbs, dw, true);
                        let mut bad = 0u32;
                        #[unroll]
                        for j in 0..nl2 {
                            bad |= select(d2[j as usize] != r2[j as usize], 1u32, 0u32);
                        }
                        #[unroll]
                        for j in 0..nl3 {
                            bad |= select(d3[j as usize] != r3[j as usize], 2u32, 0u32);
                        }
                        #[unroll]
                        for j in 0..nh2 {
                            bad |= select(x2[j as usize] != a2[j as usize], 4u32, 0u32);
                        }
                        #[unroll]
                        for j in 0..nh3 {
                            bad |= select(x3[j as usize] != a3[j as usize], 8u32, 0u32);
                        }
                        out[(o + 29u32) as usize] = bad;
                        let m = cert_power(
                            &mut x2,
                            &d2,
                            pw,
                            &mut c0,
                            &mut dup,
                            &mut dom,
                            nl2,
                            nl3,
                            s2,
                            cap,
                            base,
                            chunk_digits,
                            chunk_div,
                        );
                        put_power(out, o, m, &c0, &dup, &dom);
                        let m = cert_power(
                            &mut x3,
                            &d3,
                            pw,
                            &mut c0,
                            &mut dup,
                            &mut dom,
                            nl3,
                            nl3,
                            s3,
                            cap,
                            base,
                            chunk_digits,
                            chunk_div,
                        );
                        put_power(out, o + 6u32, m, &c0, &dup, &dom);
                    } else {
                        let m = cert_power_lockstep(
                            &mut a2,
                            &mut e2,
                            pw,
                            &mut c0,
                            &mut dup,
                            &mut dom,
                            nl2,
                            nl3,
                            s2,
                            cap,
                            base,
                            chunk_digits,
                            chunk_div,
                        );
                        put_power(out, o + 12u32, m, &c0, &dup, &dom);
                        let m = cert_power_lockstep(
                            &mut a3,
                            &mut e3,
                            pw,
                            &mut c0,
                            &mut dup,
                            &mut dom,
                            nl3,
                            nl3,
                            s3,
                            cap,
                            base,
                            chunk_digits,
                            chunk_div,
                        );
                        put_power(out, o + 18u32, m, &c0, &dup, &dom);
                    }
                }
            }
        }
    }

    use cert_kernels::{CERT_OUT, cert_test_kernel};

    /// A splitmix64 stream for the interval generators.
    struct SplitMix(u64);

    impl SplitMix {
        fn next(&mut self) -> u64 {
            self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        }

        /// Below `n` (> 0), near enough uniform for a test.
        fn below(&mut self, n: u128) -> u128 {
            ((u128::from(self.next()) << 64) | u128::from(self.next())) % n
        }
    }

    /// What one group of intervals exercised.
    #[derive(Default)]
    struct CertTally {
        intervals: u64,
        kept: u64,
        /// Powers whose shifted endpoints differ by one (the carry case).
        carried: u64,
        /// Powers with a digit at or above c0 (c0 below the digit count).
        scanned: u64,
    }

    /// Runs `iv` (one comptime shape: base, `limbs`, `cap` and the digit
    /// counts of `base`; every e - a fits `dw` words) through
    /// [`cert_test_kernel`] with the production chunk constants, and asserts
    /// per interval that `powers_diff` equals the full powers and their
    /// difference word for word, that `cert_power` equals
    /// `cert_power_lockstep` on both powers, word for word, and that
    /// `top_cert` equals `Base::cert` and its c0 terms `Base::cert_floor`.
    #[allow(clippy::cast_possible_truncation)]
    fn check_cert_group<R: Runtime>(
        client: &ComputeClient<R>,
        base: &crate::overlap_join::Base,
        limbs: u32,
        cap: u32,
        iv: &[(u128, u128)],
        tally: &mut CertTally,
        dw: u32,
    ) {
        if iv.is_empty() {
            return;
        }
        assert!(
            iv.iter().all(|&(a, e)| (e - a) >> (32 * dw) == 0),
            "e - a must fit {dw} words"
        );
        let b = base.b;
        let nl3 = 3 * limbs as usize;
        let mut pw = Vec::with_capacity((base.s3 as usize + 1) * nl3);
        for i in 0..=base.s3 {
            let mut words: Vec<u32> = base
                .poww_words(i)
                .iter()
                .flat_map(|&w| [w as u32, (w >> 32) as u32])
                .collect();
            words.resize(nl3.max(8), 0);
            assert!(
                words[nl3..].iter().all(|&x| x == 0),
                "b{b}: b^{i} does not fit {nl3} words"
            );
            pw.extend_from_slice(&words[..nl3]);
        }
        let mut words = Vec::with_capacity(iv.len() * 2 * limbs as usize);
        for &(a, e) in iv {
            assert!(
                e >> (32 * limbs) == 0,
                "b{b}: [{a}, {e}] exceeds {limbs} words"
            );
            for x in [a, e] {
                words.extend((0..limbs).map(|j| (x >> (32 * j)) as u32));
            }
        }
        let n = u32::try_from(iv.len()).expect("a test-sized group");
        let (cd, cdiv) = chunk_constants_u16(b);
        let out = client.empty(iv.len() * CERT_OUT as usize * 4);
        let iv_h = client.create(cubecl::bytes::Bytes::from_elems(words.clone()));
        let pw_h = client.create(cubecl::bytes::Bytes::from_elems(pw.clone()));
        for part in 0..3 {
            unsafe {
                cert_test_kernel::launch_unchecked::<R>(
                    client,
                    CubeCount::Static(n.div_ceil(JOIN_WG), 1, 1),
                    CubeDim::new_1d(JOIN_WG),
                    ArrayArg::from_raw_parts(iv_h.clone(), words.len()),
                    ArrayArg::from_raw_parts(pw_h.clone(), pw.len()),
                    ArrayArg::from_raw_parts(out.clone(), iv.len() * CERT_OUT as usize),
                    n,
                    b,
                    limbs,
                    cap,
                    base.s2,
                    base.s3,
                    cd,
                    cdiv,
                    part,
                    dw,
                );
            }
        }
        let bytes = client.read_one(out).expect("read the certificates");
        let got = u32::from_bytes(&bytes);
        for (i, &(a, e)) in iv.iter().enumerate() {
            let o = &got[i * CERT_OUT as usize..(i + 1) * CERT_OUT as usize];
            let at = format!("b{b} [{a}, {e}] cap {cap}");
            assert_eq!(o[29], 0, "{at}: powers_diff differs from the full powers");
            assert_eq!(o[0..6], o[12..18], "{at}: the square's certificate differs");
            assert_eq!(o[6..12], o[18..24], "{at}: the cube's certificate differs");
            let mask = u64::from(o[25]) | (u64::from(o[26]) << 32);
            let want = base.cert(a, e, cap);
            assert_eq!(
                (o[24] == 1).then_some(mask),
                want,
                "{at}: against Base::cert"
            );
            assert_eq!(
                o[27].min(o[28]),
                base.cert_floor(a, e, cap),
                "{at}: against Base::cert_floor"
            );
            tally.intervals += 1;
            tally.kept += u64::from(want.is_some());
            for (p, sp) in [(&o[0..6], base.s2), (&o[6..12], base.s3)] {
                tally.carried += u64::from(p[3] < p[4]);
                tally.scanned += u64::from(p[5] < sp);
            }
        }
    }

    /// The top certificate from the low endpoint's digits and a remainder
    /// (`kernels::cert_power`) equals the lockstep scan of both endpoints'
    /// digits it replaced, power by power and word for word (the mask, the
    /// repeat flag, the digit interval at the first disagreement, c0, which
    /// the lockstep scan finds by a binary search over the power table), the
    /// powers and differences `powers_diff` forms equal the full powers' and
    /// their difference word for word, and `top_cert` equals the CPU's
    /// `Base::cert` and `cert_floor`: on
    /// partition-extended tops of production-shaped fields at bases 58-64,
    /// and on arbitrary intervals and caps at smaller bases, in groups that
    /// share a kernel shape (where c0 is often the cap). More with
    /// `NICE_TEST_CERT_SAMPLES` per field and base (default 20,000).
    #[allow(clippy::cast_possible_truncation)]
    fn check_certificates<R: Runtime>(client: &ComputeClient<R>, name: &str) {
        use crate::overlap_join::{Base, join_params_for};
        let per: u64 = std::env::var("NICE_TEST_CERT_SAMPLES")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(20_000);
        let mut tally = CertTally::default();
        for (b, s0, size) in [
            (58, 183_859_216_587_923_470_200, 1_000_000_000_000_000),
            (60, 795_599_612_114_824_200_908, 10_000_000_000_000_000),
            (62, 12_255_736_762_397_899_821_056, 10_000_000_000_000_000),
            (64, 72_806_461_862_957_161_709_568, 10_000_000_000_000_000),
        ] {
            let range = FieldSize::new(s0, s0 + size);
            let jp = join_params_for(b, &range).expect("a join field");
            let fs = FieldSetup::new(b, s0, s0 + size, jp).expect("field setup");
            let limbs = crate::gpu_config::n_limbs(b).expect("a base range");
            let mut rng = SplitMix(u64::from(b) * 0x51ED_2701);
            let mut iv = Vec::new();
            for _ in 0..per {
                let (p0, _) = fs.tlay[rng.below(fs.tlay.len() as u128) as usize];
                let p = p0 * fs.nparts + rng.below(fs.nparts);
                if p < fs.plo || p > fs.phi {
                    continue;
                }
                let lo = p * fs.w;
                iv.push((lo.max(fs.s), (lo + fs.w - 1).min(fs.e - 1)));
            }
            check_cert_group(client, &fs.base, limbs, jp.k, &iv, &mut tally, 1);
        }
        for b in [10u32, 12, 20, 30, 40, 47, 50, 57] {
            let mut rng = SplitMix(u64::from(b) * 0x2545_F491);
            for _ in 0..4 {
                // n of about `half` + 1 digits, any cap below 6, widths up
                // to 2^40 or n's own: the intervals whose powers keep their
                // digit counts and share the first one's.
                let mut half = 2 + u32::try_from(rng.below(14)).unwrap();
                let cap = u32::try_from(rng.below(6)).unwrap();
                // `Base` takes n below 2^85.
                while 4 * u128::from(b).pow(half) >= 1 << 85 {
                    half -= 1;
                }
                let start = u128::from(b).pow(half);
                let bits = 128 - start.leading_zeros();
                let mut iv = Vec::new();
                let mut shape: Option<(Base, u32)> = None;
                for _ in 0..per / 16 {
                    let a = start + rng.below(start * 3);
                    let shift = rng.below(u128::from(bits.min(40)));
                    let e = a + rng.below(1 << shift);
                    let Some(base) = Base::try_new(b, a, e) else {
                        continue;
                    };
                    // The fewest words that hold e and the power table.
                    let words = (128 - e.leading_zeros()).div_ceil(32).max(1);
                    let top: Vec<u32> = base
                        .poww_words(base.s3)
                        .iter()
                        .flat_map(|&w| [w as u32, (w >> 32) as u32])
                        .collect();
                    let limbs =
                        (words..=3).find(|&l| top.iter().skip(3 * l as usize).all(|&x| x == 0));
                    let Some(limbs) = limbs else { continue };
                    match &shape {
                        None => shape = Some((base, limbs)),
                        Some((s, l)) if (s.s2, s.s3, *l) != (base.s2, base.s3, limbs) => continue,
                        Some(_) => {}
                    }
                    iv.push((a, e));
                }
                if let Some((base, limbs)) = shape {
                    check_cert_group(client, &base, limbs, cap, &iv, &mut tally, limbs);
                }
            }
        }
        println!(
            "{name}: {} intervals, {} kept; {} of the powers scanned above c0, {} of them \
             with the carry",
            tally.intervals, tally.kept, tally.scanned, tally.carried
        );
        assert!(
            tally.kept > per && tally.carried > per,
            "the kept case or the carry is barely tested"
        );
    }

    #[test]
    #[ignore = "requires a wgpu device"]
    fn top_certificates_match_the_lockstep_scan_and_the_cpu() {
        let (client, name) = client();
        check_certificates(&client, &name);
    }

    /// The same on `CubeCL`'s CUDA runtime.
    #[test]
    #[cfg(feature = "cubecl-cuda")]
    #[ignore = "requires an NVIDIA device"]
    fn cubecl_cuda_top_certificates_match_the_lockstep_scan_and_the_cpu() {
        let ctx = CubeclContext::new_cuda(0).expect("CubeCL CUDA init");
        let name = ctx.device_name();
        let CubeclContext::Cuda { client, .. } = ctx else {
            unreachable!("new_cuda is CUDA")
        };
        check_certificates(&client, &name);
    }

    /// `kernels::ndig_fast`'s constants exist for every base 10-64 at every
    /// width the kernels use (1-9 words), and its rule (c from the bit length
    /// L, `ceil(L·log_b(2)) - 1`, plus one if d >= b^c) gives ndig(d) for d
    /// at both ends of every bit length and next to every power of the base.
    #[test]
    #[allow(clippy::cast_possible_truncation)]
    fn ndig_fast_constants_give_the_digit_count() {
        use malachite::base::num::arithmetic::traits::PowerOf2;
        use malachite::base::num::logic::traits::SignificantBits;
        use malachite::natural::Natural;
        for base in 10u32..=64 {
            for nl in 1u32..=9 {
                let lmax = 32 * nl;
                let (m, sh, a) = super::kernels::ndig_magic(base, lmax);
                let b = Natural::from(base);
                let mut pw = vec![Natural::from(1u32)];
                while pw[pw.len() - 1].significant_bits() <= u64::from(lmax) {
                    let next = &pw[pw.len() - 1] * &b;
                    pw.push(next);
                }
                let mut ds = vec![Natural::from(0u32)];
                for l in 1..=u64::from(lmax) {
                    ds.push(Natural::power_of_2(l - 1));
                    ds.push(Natural::power_of_2(l) - Natural::from(1u32));
                }
                for p in &pw {
                    for d in [p - Natural::from(1u32), p.clone(), p + Natural::from(1u32)] {
                        if d.significant_bits() <= u64::from(lmax) {
                            ds.push(d);
                        }
                    }
                }
                for d in &ds {
                    let want = pw.iter().position(|p| p > d).expect("a power above d");
                    let bl = d.significant_bits() as u32;
                    let ce = bl
                        .checked_mul(m)
                        .and_then(|x| x.checked_add(a))
                        .expect("fits u32")
                        >> sh;
                    let c = ce.saturating_sub(1) as usize;
                    let got = if *d < pw[c] { c } else { c + 1 };
                    assert_eq!(got, want, "b{base}, {nl} words, d = {d}");
                }
            }
        }
    }

    /// The batch load's `ceil(D·2^32/b²)`, as [`join_kernel`] forms it:
    /// `D·Q + ceil(D·R/b²)`.
    fn fx_top(d: u32, base: u32) -> u32 {
        let (_, q, r) = super::kernels::fx_consts(base);
        d * q + (d * r).div_ceil(base * base)
    }

    /// [`walk_pf_lean`]'s two digits of `F = ah + p·dh mod 2^32`: the high
    /// words of `F·b` and of `(F·b mod 2^32)·b`.
    #[allow(clippy::cast_possible_truncation)]
    fn fx_digits(ah: u32, p: u32, dh: u32, base: u32) -> (u32, u32) {
        let f = ah.wrapping_add(p.wrapping_mul(dh));
        let fb = u64::from(f) * u64::from(base);
        (
            (fb >> 32) as u32,
            ((u64::from(fb as u32) * u64::from(base)) >> 32) as u32,
        )
    }

    /// [`walk_pf`]'s digit test at mid = 2: four one-hot masks.
    fn pf_masks(seen: u64, g2: u32, g3: u32, h2: u32, h3: u32) -> bool {
        let at_k = (1u64 << g2) | (1u64 << g3);
        let at_k1 = (1u64 << h2) | (1u64 << h3);
        !(g2 == g3 || h2 == h3 || seen & at_k != 0 || (seen | at_k) & at_k1 != 0)
    }

    /// [`walk_pf_lean`]'s digit test on the bit-reversed `seen`: digit g is
    /// in it iff the sign bit of `hi32(seen << g)` is set; six compares.
    #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
    fn pf_lean(rseen: u64, g2: u32, g3: u32, h2: u32, h3: u32) -> bool {
        let hi = |g: u32| ((rseen << g) >> 32) as u32;
        let hit = hi(g2) | hi(g3) | hi(h2) | hi(h3);
        !(g2 == g3 || h2 == h3 || g2 == h2 || g2 == h3 || g3 == h2 || g3 == h3 || (hit as i32) < 0)
    }

    fn xorshift(x: &mut u64) -> u64 {
        *x ^= *x << 13;
        *x ^= *x >> 7;
        *x ^= *x << 17;
        *x
    }

    /// The lean prefilter's fixed-point digits at every base 10-64: `C =
    /// ceil(2^32/b²)` and the batch load's `dh` are the true ceilings (every
    /// D < b²), `A·C` fits a word, and the error `Δ = A·(C·b² − 2^32) +
    /// P·(dh·b² − D·2^32)` stays below 2^32 for every A, P, D < b². Then the
    /// digits themselves: every (P, D) at the largest A, and every (A, D) at
    /// the largest P, against `(A + P·D) mod b²` (the error grows with A and
    /// P; `fx_digits_are_exact_everywhere` runs every triple).
    #[test]
    #[allow(clippy::many_single_char_names)]
    fn fx_digits_match_the_division() {
        for base in 10u32..=64 {
            let b2 = base * base;
            let (c, q, r) = super::kernels::fx_consts(base);
            assert_eq!(
                u128::from(c),
                (1u128 << 32).div_ceil(u128::from(b2)),
                "b{base}"
            );
            assert_eq!(
                u64::from(q) * u64::from(b2) + u64::from(r),
                1u64 << 32,
                "b{base}"
            );
            assert!((b2 - 1).checked_mul(c).is_some(), "b{base}: A·C overflows");
            let e1 = u64::from(c) * u64::from(b2) - (1u64 << 32);
            let mut e2max = 0u64;
            let dh: Vec<u32> = (0..b2).map(|d| fx_top(d, base)).collect();
            for d in 0..b2 {
                let want = (u128::from(d) << 32).div_ceil(u128::from(b2));
                assert_eq!(u128::from(dh[d as usize]), want, "b{base} D={d}");
                e2max = e2max.max(u64::from(dh[d as usize]) * u64::from(b2) - (u64::from(d) << 32));
            }
            let m = u64::from(b2 - 1);
            assert!(m * e1 + m * e2max < 1u64 << 32, "b{base}: Δ bound");
            let amax = b2 - 1;
            for p in 0..b2 {
                for d in 0..b2 {
                    let x = (amax + p * d) % b2;
                    assert_eq!(
                        fx_digits(amax * c, p, dh[d as usize], base),
                        (x / base, x % base),
                        "b{base} A={amax} P={p} D={d}"
                    );
                }
            }
            for a in 0..b2 {
                for d in 0..b2 {
                    let x = (a + amax * d) % b2;
                    assert_eq!(
                        fx_digits(a * c, amax, dh[d as usize], base),
                        (x / base, x % base),
                        "b{base} A={a} P={amax} D={d}"
                    );
                }
            }
        }
    }

    /// The fixed-point digits over every operand the join can give them: all
    /// (A, P, D) below b² at every base 10-64 (6.6e11 triples, minutes on all
    /// cores).
    #[test]
    #[ignore = "exhaustive, minutes on all cores; opt-in, NICE_TEST_FX_EXHAUSTIVE=1"]
    #[allow(clippy::cast_possible_truncation, clippy::many_single_char_names)]
    fn fx_digits_are_exact_everywhere() {
        if std::env::var("NICE_TEST_FX_EXHAUSTIVE").is_err() {
            eprintln!("skipping: set NICE_TEST_FX_EXHAUSTIVE=1 to run every operand triple");
            return;
        }
        let threads = std::thread::available_parallelism().map_or(8, std::num::NonZero::get);
        let t0 = Instant::now();
        let mut total = 0u64;
        for base in 10u32..=64 {
            let b2 = base * base;
            let c = super::kernels::fx_consts(base).0;
            let next = std::sync::atomic::AtomicU32::new(0);
            std::thread::scope(|sc| {
                for _ in 0..threads {
                    sc.spawn(|| {
                        loop {
                            let d = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                            if d >= b2 {
                                break;
                            }
                            let dh = fx_top(d, base);
                            for p in 0..b2 {
                                let u = p.wrapping_mul(dh);
                                let y = p * d % b2;
                                // (h, g): the digits of (A + P·D) mod b², stepped with A.
                                let (mut h, mut g) = (y / base, y % base);
                                let mut bad = 0u32;
                                for a in 0..b2 {
                                    let f = (a * c).wrapping_add(u);
                                    let fb = u64::from(f) * u64::from(base);
                                    let fh = (fb >> 32) as u32;
                                    let fg =
                                        ((u64::from(fb as u32) * u64::from(base)) >> 32) as u32;
                                    bad |= (fh ^ h) | (fg ^ g);
                                    g += 1;
                                    if g == base {
                                        g = 0;
                                        h += 1;
                                        if h == base {
                                            h = 0;
                                        }
                                    }
                                }
                                assert_eq!(bad, 0, "b{base} P={p} D={d}");
                            }
                        }
                    });
                }
            });
            total += u64::from(b2).pow(3);
        }
        println!(
            "fixed-point digits: {total} triples (A, P, D < b², b = 10-64) exact in {:.1}s on {threads} threads",
            t0.elapsed().as_secs_f64()
        );
    }

    /// The lean membership test (bit-reversed masks, four shifts and a sign
    /// test, six compares) is the four-mask test: every digit 4-tuple at
    /// every base 10-64 against a random set of seen digits (about a quarter
    /// of the bits, as the entry's 12 and a top's certificate give), and 10^7
    /// random tuples against sets of every density.
    #[test]
    #[allow(clippy::cast_possible_truncation)]
    fn lean_membership_matches_the_masks() {
        let mut x = 0x9E37_79B9_7F4A_7C15u64;
        let (mut fails, mut pass, mut n) = (0u64, 0u64, 0u64);
        for base in 10u32..=64 {
            for g2 in 0..base {
                for g3 in 0..base {
                    for h2 in 0..base {
                        for h3 in 0..base {
                            let seen = xorshift(&mut x) & xorshift(&mut x);
                            let want = pf_masks(seen, g2, g3, h2, h3);
                            fails +=
                                u64::from(want != pf_lean(seen.reverse_bits(), g2, g3, h2, h3));
                            pass += u64::from(want);
                            n += 1;
                        }
                    }
                }
            }
        }
        for _ in 0..10_000_000 {
            let r = xorshift(&mut x);
            let base = 10 + (r % 55) as u32;
            let dig = |s: u64| (s % u64::from(base)) as u32;
            let (g2, g3, h2, h3) = (dig(r >> 8), dig(r >> 20), dig(r >> 32), dig(r >> 44));
            let mut seen = xorshift(&mut x);
            for _ in 0..(r >> 60) % 4 {
                seen &= xorshift(&mut x);
            }
            seen &= u64::MAX >> (64 - base);
            let want = pf_masks(seen, g2, g3, h2, h3);
            fails += u64::from(want != pf_lean(seen.reverse_bits(), g2, g3, h2, h3));
            pass += u64::from(want);
            n += 1;
        }
        assert_eq!(fails, 0, "{fails} of {n} differ");
        assert!(pass > n / 50 && pass < n / 2, "{pass} of {n} pass");
    }

    /// The lean prefilter end to end against a host model of the division
    /// form ([`walk_pf`] at mid = 2: the digits of `(A + P·D) mod b²` and the
    /// four one-hot masks): random entries (A2, A3, P2, P3 < b², masks) and
    /// tops (D < b², certificate, seed flag), 10^6 per base 10-64. The device
    /// tests compare the kernel itself with the CPU join.
    #[test]
    #[allow(clippy::cast_possible_truncation)]
    fn lean_prefilter_matches_the_division_form() {
        let mut x = 0x2545_F491_4F6C_DD1Du64;
        let mut pass = 0u64;
        for base in 10u32..=64 {
            let b2 = base * base;
            let c = super::kernels::fx_consts(base).0;
            let full = u64::MAX >> (64 - base);
            for _ in 0..1_000_000 {
                let r = xorshift(&mut x);
                let (a2, a3) = (
                    (r % u64::from(b2)) as u32,
                    ((r >> 16) % u64::from(b2)) as u32,
                );
                let (p2, p3) = (
                    ((r >> 32) % u64::from(b2)) as u32,
                    ((r >> 48) % u64::from(b2)) as u32,
                );
                let r2 = xorshift(&mut x);
                let d = (r2 % u64::from(b2)) as u32;
                let flag = (r2 >> 20) & 1 == 1;
                let em = xorshift(&mut x) & xorshift(&mut x) & xorshift(&mut x) & full;
                let tm = xorshift(&mut x) & xorshift(&mut x) & full;
                let seen = em | if flag { tm } else { 0 };
                let (x2, x3) = ((a2 + p2 * d) % b2, (a3 + p3 * d) % b2);
                let want = pf_masks(seen, x2 % base, x3 % base, x2 / base, x3 / base);
                let dh = fx_top(d, base);
                let (h2, g2) = fx_digits(a2 * c, p2, dh, base);
                let (h3, g3) = fx_digits(a3 * c, p3, dh, base);
                let gated = if flag { tm.reverse_bits() } else { 0 };
                assert_eq!(
                    pf_lean(em.reverse_bits() | gated, g2, g3, h2, h3),
                    want,
                    "b{base}"
                );
                pass += u64::from(want);
            }
        }
        assert!(pass > 0);
    }

    // ---------------------------------------------------------------------
    // The top kernel by finite differences against the top kernel.
    // ---------------------------------------------------------------------

    /// One batch's tops from one top kernel alone: per (slot, group) its
    /// count, and per slot its (group, p, mask, r, x, seed) entries sorted
    /// (the kernels append in atomic order within a group). Every counted
    /// record must have been written (none is left at the sentinel), lie in
    /// its group's segment, within the group's size, and its prefix `P / b^p`
    /// must be of that group by the host's rule ([`sorted_layer`]).
    type Tops = (Vec<u32>, Vec<Vec<[u32; 9]>>);

    /// `u32::MAX` over `a[0..n]`, grid-stride.
    #[cube(launch_unchecked)]
    fn fill_sentinel(a: &mut Array<u32>, n: u32) {
        let mut i = ABSOLUTE_POS_X;
        while i < n {
            a[i as usize] = 0xFFFF_FFFFu32;
            i += CUBE_COUNT_X * CUBE_DIM_X;
        }
    }

    impl<R: Runtime> JoinDevice<R> {
        fn run_tops(&self, vs: &[u32], fd: bool) -> Tops {
            let c = &self.client;
            let nslots = vs.len();
            let nb = self.nb as usize;
            // Every record starts as the sentinel, so a top a kernel counts
            // but does not write cannot read back as the other kernel's
            // record at the same index (both append a group's tops in about
            // the same order).
            let rec_len = TOP_REC_WORDS as usize * self.max_slots * self.ntlay as usize;
            unsafe {
                fill_sentinel::launch_unchecked::<R>(
                    c,
                    CubeCount::Static(CHECK_CUBES, 1, 1),
                    CubeDim::new_1d(JOIN_WG),
                    ArrayArg::from_raw_parts(self.top_rec.clone(), rec_len),
                    u32::try_from(rec_len).expect("records fit u32"),
                );
            }
            let vs_h = c.create(cubecl::bytes::Bytes::from_elems(vs.to_vec()));
            let grp_cnt = c.create(cubecl::bytes::Bytes::from_elems(vec![0u32; nslots * nb]));
            let ran = c.create(cubecl::bytes::Bytes::from_elems(vec![0u32; STAGES]));
            self.launch_top(vs, &vs_h, &grp_cnt, &ran, fd);
            let counts = self.read_u32(&grp_cnt).expect("group counts");
            let want = self.top_expect(vs, fd);
            assert_eq!(self.read_u32(&ran).expect("ran")[0], want, "top cubes run");
            let rec = self.read_u32(&self.top_rec).unwrap();
            let goff = self.read_u32(&self.goff).unwrap();
            let ntl = self.ntlay as usize;
            let (base, m1, pdiv) = (
                u64::from(self.b),
                u64::from(self.b - 1),
                u64::from(self.pdiv),
            );
            let group = |p0: u64| {
                let key = if self.key_level { p0 % base } else { 0 };
                key * m1 + (p0 % m1) * (pdiv % m1) % m1
            };
            let entries = (0..nslots)
                .map(|s| {
                    let mut ents: Vec<[u32; 9]> = Vec::new();
                    for (grp, w) in goff.windows(2).enumerate() {
                        let n = counts[s * nb + grp] as usize;
                        assert!(
                            n <= (w[1] - w[0]) as usize,
                            "slot {s} group {grp}: {n} tops overrun the segment"
                        );
                        let gw = u32::try_from(grp).unwrap();
                        for i in 0..n {
                            let at = s * ntl + w[0] as usize + i;
                            let rw = &rec[TOP_REC_WORDS as usize * at..][..8];
                            assert!(rw[7] <= 1, "slot {s} group {grp}: record {i} not written");
                            let pv = u64::from(rw[0]) | (u64::from(rw[1]) << 32);
                            assert_eq!(
                                group(pv / pdiv),
                                u64::from(gw),
                                "slot {s}: P {pv} not in its group"
                            );
                            ents.push([gw, rw[0], rw[1], rw[2], rw[3], rw[4], rw[5], rw[6], rw[7]]);
                        }
                    }
                    ents.sort_unstable();
                    ents
                })
                .collect();
            (counts, entries)
        }
    }

    /// `top_fd_kernel` must give exactly `top_kernel`'s tops: the same count
    /// per slot and group, the same entries in the same segments. On the test
    /// windows (small bases, clipped at both ends, every partition) and on
    /// production plans' first and last slices (`heavy`: also 1e16 fields),
    /// in consecutive batches (the first ones, the middle, the last), in the
    /// windows' thinned live partitions (gaps), and in a batch that steps
    /// back and skips. Of all of these, only the b20 windows' batches step
    /// across a change of a power's c0 (P = 201 for the square, P = 231 for
    /// the cube), the only place the FD kernel re-forms its carried c0 and
    /// dm: keep them (`top_fd_test_windows_step_across_a_c0_change`).
    fn check_top_fd<R: Runtime>(client: &ComputeClient<R>, name: &str, heavy: bool) {
        let lim = device_limits(client, name);
        let mut fields: Vec<(String, FieldSetup, JoinPlan)> = Vec::new();
        for &(b, s, e, t, k, p) in WINDOWS {
            let jp = JoinParams { t, k, p };
            let fs = FieldSetup::new(b, s, e, jp).expect("field setup");
            let plan = test_plan(&fs, limits_of(client), 4);
            fields.push((format!("b{b} [{s}, {e}) ({t},{k},{p})"), fs, plan));
        }
        let mut prod: Vec<(u32, u128, u128)> = vec![
            (57, FRONTIER_57, 100_000_000_000_000),
            (60, 1_366_406_074_057_412_100_454, 100_000_000_000_000),
            (62, 7_997_740_455_941_656_911_841, 100_000_000_000_000),
            (64, 41_242_006_262_957_161_709_568, 100_000_000_000_000),
        ];
        if heavy {
            prod.push((60, 1_366_406_074_057_412_100_454, 10_000_000_000_000_000));
            prod.push((62, 7_997_740_455_941_656_911_841, 10_000_000_000_000_000));
            prod.push((64, 41_242_006_262_957_161_709_568, 10_000_000_000_000_000));
        }
        for (b, start, size) in prod {
            let range = FieldSize::new(start, start + size);
            let field = crate::join_plan::plan_join(b, &range, lim).expect("a join field");
            let label = format!("b{b} {size:e} at {start}");
            fields.push((
                format!("{label}, first slice"),
                field.fs.clone(),
                field.plan,
            ));
            if let Some(last) = field.slices.last().filter(|_| field.slices.len() > 1) {
                let fs = field.fs.sub_range(last.start(), last.end()).expect("slice");
                let plan = JoinPlan::for_field(&fs, lim).expect("the slice fits").0;
                fields.push((format!("{label}, last slice"), fs, plan));
            }
        }
        let (mut nbatches, mut ntops, mut nfields, mut nfix) = (0usize, 0u64, 0usize, 0usize);
        for (label, fs, plan) in fields {
            if !top_fd_fits(&fs) {
                println!("{name}: {label}: the FD state does not fit, skipped");
                continue;
            }
            let layer = sorted_layer(&fs).expect("sorted layer");
            let mut dev = JoinDevice::new(client, &fs, layer, &plan).expect("device");
            let n = u32::try_from(fs.nparts).expect("b^p < 2^32");
            let ms = dev.max_slots;
            let all: Vec<u32> = (0..n).collect();
            let chunks: Vec<&[u32]> = all.chunks(ms).collect();
            let mut batches: Vec<Vec<u32>> = Vec::new();
            if chunks.len() <= 64 {
                batches.extend(chunks.iter().map(|c| c.to_vec()));
            } else if heavy {
                for i in [0, 1, 2, chunks.len() / 2, chunks.len() - 1] {
                    batches.push(chunks[i].to_vec());
                }
            } else {
                // A software rasterizer: the first and last batches only.
                batches.push(chunks[0].to_vec());
                batches.push(chunks[chunks.len() - 1].to_vec());
            }
            let live = live_partitions(&fs);
            let step = live.len().div_ceil(96).max(1);
            let thinned: Vec<u32> = live.into_iter().step_by(step).collect();
            batches.extend(
                thinned
                    .chunks(ms)
                    .take(if heavy { 4 } else { 1 })
                    .map(<[u32]>::to_vec),
            );
            if n >= 12 {
                let odd: Vec<u32> = [3, 2, 4, 5, 7, 8, 9, 11].into_iter().take(ms).collect();
                batches.push(odd);
            }
            for vs in &batches {
                let a = dev.run_tops(vs, false);
                // One group of slots, and groups of about a third (and of
                // one slot each where the batch is short).
                for min in [0, dev.ntlay.max(1) * 3] {
                    dev.top_fd_min = min;
                    let f = dev.run_tops(vs, true);
                    let groups = dev.top_fd_groups(u32::try_from(vs.len()).unwrap());
                    let how = format!("slot groups {groups:?}");
                    assert_eq!(a.0, f.0, "{label} {vs:?} {how}: group counts differ");
                    assert!(a.1 == f.1, "{label} {vs:?} {how}: top entries differ");
                }
                ntops += a.0.iter().map(|&x| u64::from(x)).sum::<u64>();
                nfix += usize::from(dev.top_fix(vs));
            }
            nbatches += batches.len();
            nfields += 1;
            println!(
                "{name}: {label}: {} batches (slots {ms}, {} prefixes) agree",
                batches.len(),
                dev.ntlay
            );
        }
        println!(
            "{name}: top_fd (one slot group and several) = top_kernel on {nfields} \
             fields, {nbatches} batches ({nfix} with a clipped block), {ntops} tops"
        );
        assert!(
            nfields > 10 && ntops > 0 && nfix > 0,
            "the FD comparison ran"
        );
    }

    /// The FD kernel keeps each power's c0 = max(k, ndig(e^k - a^k)) from
    /// slot to slot and re-forms it (with dm) only when e^k - a^k reaches
    /// b^c0. Only small windows step across such a change, and within
    /// `check_top_fd`'s batches only the b20 ones do (P = 201 for the square,
    /// P = 231 for the cube); no production-size field changes c0 within a
    /// slice. This pins that the b20 windows still cross one for each power.
    #[test]
    fn top_fd_test_windows_step_across_a_c0_change() {
        fn ndig(mut x: u128, b: u128) -> u32 {
            let mut n = 0;
            while x > 0 {
                x /= b;
                n += 1;
            }
            n
        }
        let mut crossed = [false; 2];
        for &(b, s, e, t, k, p) in WINDOWS {
            if e >= 1 << 40 {
                continue; // cubes in u128
            }
            let fs = FieldSetup::new(b, s, e, JoinParams { t, k, p }).expect("field setup");
            let c0 = |p: u128, pow: u32| {
                let a = p * fs.w;
                let ee = a + fs.w - 1;
                let sp = if pow == 2 { fs.base.s2 } else { fs.base.s3 };
                let d = ee.pow(pow) - a.pow(pow);
                (fs.jp.k.max(ndig(d, u128::from(fs.b)).min(sp)), sp)
            };
            let unclipped = |p: u128| p * fs.w >= fs.s && p * fs.w + fs.w - 1 < fs.e;
            for q in fs.plo + 1..=fs.phi {
                if !(unclipped(q - 1) && unclipped(q)) {
                    continue;
                }
                for (i, pow) in [2u32, 3].into_iter().enumerate() {
                    let ((c_prev, sp), (c_now, _)) = (c0(q - 1, pow), c0(q, pow));
                    if c_prev < sp && c_now != c_prev && b == 20 {
                        crossed[i] = true;
                    }
                }
            }
        }
        assert_eq!(
            crossed,
            [true, true],
            "the windows no longer cross a c0 change"
        );
    }

    /// The FD kernel's append ranks kept tops with a ballot only on planes
    /// of at most 32 lanes: `CubeCL`'s C++ backends keep a ballot's first
    /// word, and HIP on CDNA (and Metal on AMD GPUs) has 64-lane planes.
    #[test]
    fn the_ballot_append_is_for_planes_of_at_most_32_lanes() {
        for (lanes, ballot) in [(8, true), (16, true), (32, true), (64, false), (128, false)] {
            assert_eq!(ballot_append(lanes), ballot, "{lanes} lanes");
        }
    }

    #[test]
    #[ignore = "requires a wgpu device"]
    fn top_fd_matches_the_top_kernel() {
        let (client, name) = client();
        check_top_fd(&client, &name, false);
    }

    #[test]
    #[cfg(feature = "cubecl-cuda")]
    #[ignore = "requires an NVIDIA device"]
    fn cubecl_cuda_top_fd_matches_the_top_kernel() {
        let ctx = CubeclContext::new_cuda(0).expect("CubeCL CUDA init");
        let name = ctx.device_name();
        let CubeclContext::Cuda { client, .. } = ctx else {
            unreachable!("new_cuda is CUDA")
        };
        check_top_fd(&client, &name, true);
    }

    /// The host side of the static bucket order, without a device: the
    /// sorted layer ([`sorted_layer`]) is a stable permutation of the top
    /// layer by group with the right words and offsets, and for any kept
    /// subset of a partition's tops, appended into their groups' segments,
    /// [`join_kernel`]'s segment walk (simulated in u32, at two slots) reads
    /// each bucket exactly the tops `cpu_join` lists there: key `P0 mod b` at
    /// the key level, class `(root - P mod (b-1)) mod (b-1)` per root. On b10,
    /// the test windows and two 1e14 fields (2 and 8 roots), a few partitions
    /// each, and on the b62 field's layer reversed (every prefix's residues by
    /// division).
    #[test]
    #[allow(
        clippy::too_many_lines,
        clippy::cast_possible_truncation,
        clippy::needless_range_loop
    )]
    fn sorted_layer_and_segments_match_the_cpu_buckets() {
        let mut fields = vec![(
            "b10 [47, 100)".to_string(),
            FieldSetup::new(10, 47, 100, JoinParams { t: 2, k: 1, p: 0 }).expect("field setup"),
        )];
        for &(b, s, e, t, k, p) in WINDOWS {
            let fs = FieldSetup::new(b, s, e, JoinParams { t, k, p }).expect("field setup");
            fields.push((format!("b{b} [{s}, {e}) ({t},{k},{p})"), fs));
        }
        for &(b, s, size) in PLAN_FIELDS.iter().filter(|f| f.0 == 62 || f.0 == 64) {
            let range = FieldSize::new(s, s + size);
            let jp = crate::overlap_join::join_params_for(b, &range).expect("a join field");
            let fs = FieldSetup::new(b, range.start(), range.end(), jp).expect("field setup");
            if b == 62 {
                let mut rev = fs.clone();
                rev.tlay.reverse();
                fields.push((format!("b{b} {size:e}, layer reversed"), rev));
            }
            fields.push((format!("b{b} {size:e}"), fs));
        }
        let mut nbuckets = 0u64;
        for (label, fs) in &fields {
            let (b, m1, m1u) = (u64::from(fs.b), u64::from(fs.m1), fs.m1);
            let SortedLayer {
                words: tl, goff, ..
            } = sorted_layer(fs).expect("sorted layer");
            let (n, nb) = (fs.tlay.len(), fs.nb);
            assert_eq!(
                (goff.len(), goff[0], goff[nb] as usize),
                (nb + 1, 0, n),
                "{label}"
            );
            assert!(goff.windows(2).all(|w| w[0] <= w[1]), "{label}");
            let p0mod = u64::try_from(fs.base.powu(fs.k2 - fs.f0 - fs.jp.p)).unwrap();
            let nparts = u64::try_from(fs.nparts).unwrap();
            let group = |p0: u64| {
                let key = if fs.key_level { p0 % b } else { 0 };
                (key * m1 + p0 % m1 * (nparts % m1) % m1) as usize
            };
            // The layer by group, in its original order within each.
            let mut want: Vec<Vec<u64>> = vec![Vec::new(); nb];
            for &(p0, _) in &fs.tlay {
                let p0 = u64::try_from(p0).unwrap();
                want[group(p0)].push(p0);
            }
            let mut sp0 = Vec::with_capacity(n);
            for w in tl.as_chunks::<5>().0 {
                let p0 = u64::from(w[0]) | (u64::from(w[1]) << 32);
                let g = group(p0);
                assert_eq!((w[2] as usize, w[3]), (g, goff[g]), "{label}: P0 {p0}");
                assert_eq!(u64::from(w[4]), p0 % p0mod, "{label}: P0 {p0}");
                sp0.push(p0);
            }
            for (g, w) in goff.windows(2).enumerate() {
                assert_eq!(&sp0[w[0] as usize..w[1] as usize], &want[g][..], "{label}");
            }
            // A few partitions: the edges and some at random.
            let mut vsel: Vec<u64> = (0..nparts.min(4)).collect();
            vsel.extend(nparts.saturating_sub(4).max(4)..nparts);
            let mut x = 0x9E37_79B9_7F4A_7C15u64 ^ nparts;
            for _ in 0..4 {
                vsel.push(xorshift(&mut x) % nparts);
            }
            let roots = &fs.base.roots;
            let nroots = roots.len();
            for (iv, &v) in vsel.iter().enumerate() {
                // Kept: every in-range top, or about 60 % of them.
                let mut seg: Vec<Vec<u128>> = vec![Vec::new(); nb];
                let mut cpu: Vec<Vec<u128>> = vec![Vec::new(); nb];
                for (i, &p0) in sp0.iter().enumerate() {
                    let p = u128::from(p0) * u128::from(nparts) + u128::from(v);
                    if p < fs.plo || p > fs.phi || (iv % 2 == 1 && xorshift(&mut x) % 10 >= 6) {
                        continue;
                    }
                    seg[tl[5 * i + 2] as usize].push(p);
                    let key = if fs.key_level { p0 % b } else { 0 };
                    let pc = (p % u128::from(m1)) as u32;
                    for &root in roots {
                        cpu[(key * m1) as usize + ((root + m1u - pc) % m1u) as usize].push(p);
                    }
                }
                for c in &mut cpu {
                    c.sort_unstable();
                }
                for slot in [0u32, 3] {
                    let ntl = n as u32;
                    for bx in 0..nb as u32 {
                        let (dkey, cls) = (bx / m1u, bx % m1u);
                        let mut s_db = [0u32; MAX_ROOTS as usize];
                        let mut s_cum = [0u32; MAX_ROOTS as usize + 1];
                        let mut gs = [0usize; MAX_ROOTS as usize];
                        for r in 0..nroots {
                            let c0 = (roots[r] + 2 * m1u - cls - v as u32 % m1u) % m1u;
                            gs[r] = (dkey * m1u + c0) as usize;
                            s_db[r] = slot * ntl + goff[gs[r]];
                            s_cum[r + 1] = seg[gs[r]].len() as u32;
                        }
                        let mut distinct = gs[..nroots].to_vec();
                        distinct.sort_unstable();
                        distinct.dedup();
                        assert_eq!(distinct.len(), nroots, "{label}: a bucket's groups repeat");
                        for r in 0..nroots {
                            let c = s_cum[r];
                            s_cum[r + 1] += c;
                            s_db[r] = s_db[r].wrapping_sub(c);
                        }
                        let mut got = Vec::new();
                        for q in 0..s_cum[nroots] {
                            let sr = (1..nroots).rev().find(|&j| s_cum[j] <= q).unwrap_or(0);
                            let t = s_db[sr].wrapping_add(q);
                            let off = (t - (slot * ntl + goff[gs[sr]])) as usize;
                            got.push(seg[gs[sr]][off]);
                        }
                        got.sort_unstable();
                        assert_eq!(
                            got, cpu[bx as usize],
                            "{label}: v {v} slot {slot} bucket {bx}"
                        );
                        nbuckets += 1;
                    }
                }
            }
        }
        println!("{} fields, {nbuckets} buckets agree", fields.len());
    }

    /// Device seconds of the two top kernels alone, on the first slice of a
    /// production plan (`NICE_TEST_TOPFD_TIMING` = "b:start:size"): its
    /// first 40 batches of consecutive partitions, and 40 of every other
    /// partition (every FD slot seeds), each kernel in turn, three rounds.
    #[test]
    #[cfg(feature = "cubecl-cuda")]
    #[ignore = "requires an NVIDIA device; prints timings"]
    #[allow(
        clippy::cast_precision_loss,
        clippy::cast_sign_loss,
        clippy::cast_possible_truncation,
        clippy::many_single_char_names
    )]
    fn cubecl_cuda_top_fd_timing() {
        let Ok(spec) = std::env::var("NICE_TEST_TOPFD_TIMING") else {
            eprintln!("skipping: set NICE_TEST_TOPFD_TIMING=b:start:size");
            return;
        };
        let f: Vec<&str> = spec.split(':').collect();
        let b: u32 = f[0].parse().expect("base");
        let start: u128 = f[1].parse().expect("start");
        let size = f[2].parse::<f64>().expect("size") as u128;
        let nbat = 40;
        let ctx = CubeclContext::new_cuda(0).expect("CubeCL CUDA init");
        let name = ctx.device_name();
        let CubeclContext::Cuda { client, .. } = ctx else {
            unreachable!("new_cuda is CUDA")
        };
        let lim = device_limits(&client, &name);
        let range = FieldSize::new(start, start + size);
        let field = crate::join_plan::plan_join(b, &range, lim).expect("a join field");
        let fs = &field.fs;
        let dev = JoinDevice::new(&client, fs, field.layer.clone(), &field.plan).expect("device");
        let n = u32::try_from(fs.nparts).expect("small");
        let ms = dev.max_slots;
        let all: Vec<u32> = (0..n).collect();
        let odd: Vec<u32> = (0..n).step_by(2).collect();
        let cons: Vec<Vec<u32>> = all.chunks(ms).take(nbat).map(<[u32]>::to_vec).collect();
        let gap: Vec<Vec<u32>> = odd.chunks(ms).take(nbat).map(<[u32]>::to_vec).collect();
        let nb = dev.nb as usize;
        let run = |sets: &[Vec<u32>], fd: bool| -> f64 {
            let t = Instant::now();
            let mut last = None;
            for vs in sets {
                let c = &dev.client;
                let vs_h = c.create(cubecl::bytes::Bytes::from_elems(vs.clone()));
                let gc = c.create(cubecl::bytes::Bytes::from_elems(vec![0u32; vs.len() * nb]));
                let ran = c.create(cubecl::bytes::Bytes::from_elems(vec![0u32; STAGES]));
                dev.launch_top(vs, &vs_h, &gc, &ran, fd);
                last = Some(ran);
            }
            let _ = dev.read_u32(&last.expect("a batch")).expect("sync");
            t.elapsed().as_secs_f64()
        };
        let slots: usize = cons.iter().map(Vec::len).sum();
        let gslots: usize = gap.iter().map(Vec::len).sum();
        run(&cons[..1], false);
        run(&cons[..1], true);
        let per = |s: f64, k: usize| s / (k as f64 * f64::from(dev.ntlay)) * 1e9;
        for r in 0..3 {
            let (t0, t1) = (run(&cons, false), run(&cons, true));
            let (g0, g1) = (run(&gap, false), run(&gap, true));
            eprintln!(
                "TOPFD TIMING b{b} {size:e} round={r} prefixes={} slots={ms} batches={} \
                 | consecutive ns/slot: top={:.4} fd={:.4} ratio={:.3} \
                 | every other partition ns/slot: top={:.4} fd={:.4} ratio={:.3}",
                dev.ntlay,
                cons.len(),
                per(t0, slots),
                per(t1, slots),
                t0 / t1,
                per(g0, gslots),
                per(g1, gslots),
                g0 / g1,
            );
        }
    }
}
