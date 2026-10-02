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
//! 1. [`top_kernel`], [`top_scan_kernel`], [`top_fill_kernel`]: certify each
//!    top `P = P0·b^p + v` of the slot's partition `v` (96-bit endpoint
//!    powers in `u32` limbs, both endpoints' digits scanned in lockstep), and
//!    bucket it by (key digit, digit-sum class) once per root.
//! 2. [`bucket_kernel`]: extend every `bpre` entry by `v`'s digits with exact
//!    digit arithmetic mod `b^k`, and list it under every key digit whose two
//!    output digits are distinct and new.
//! 3. [`join_kernel`]: one cube per bucket; the bucket's tops in shared
//!    memory, `ept` entries per thread in registers, a branch-free AND over
//!    32 tops at a time, survivors staged in shared memory. As a stage is
//!    flushed the whole cube applies the middle-digit prefilter
//!    (`mid_pass`): digits `0..k+2` of `n²` and `n³` from `n`'s low `k + 2`
//!    digits must be distinct, and disjoint from the top certificate when it
//!    provably sits above them. It keeps about 6% at bases 57-64, and only
//!    those are written out. (Dan's design ran the prefilter as a separate
//!    pass, [`mid_kernel`], over a full survivor list; the tests still run
//!    that placement to compare the join's own survivor set.)
//! 4. [`check_kernel`]: the client's own `candidate_check`.
//!
//! Buffers are sized per field to the device ([`JoinPlan`]): each within the
//! device's largest binding and all of them within a budget. A device that
//! cannot hold one partition does not get the join, and its fields stay on
//! the stride pipeline (`cubecl_backend::plan_join`). Every list is bounded
//! and checked: a batch whose survivors overflow is re-run one partition at
//! a time with a longer list, and a partition that overflows that is re-run
//! on halves of its top layer.
#![cfg(feature = "cubecl")]

use crate::cubecl_backend::{LaunchFence, NICEONLY_STRIDE, launch_fence, wide_chunk_for};
use crate::gpu_config::{chunk_constants, chunk_constants_u16, n_limbs};
use crate::gpu_niceonly::{NiceonlyStats, fields_in_flight};
use crate::overlap_join::{Base, JoinParams};
use crate::{FieldSize, NiceNumberSimple};
use anyhow::{Result, anyhow, ensure};
use cubecl::prelude::*;
use cubecl::server::Handle;
use log::debug;
use std::collections::VecDeque;
use std::sync::mpsc::{Receiver, SyncSender, sync_channel};
use web_time::Instant;

/// Threads per cube for every kernel here (the client's `WORKGROUP_SIZE`).
pub const JOIN_WG: u32 = 256;
/// Tops held in shared memory per pass over a bucket's entries.
pub const TOP_CAP: u32 = 256;
/// Survivors staged in shared memory before a flush to the global list.
pub const SURV_SH_CAP: u32 = 1024;
/// Staged-survivor level that triggers a flush at the next barrier.
pub const SURV_FLUSH: u32 = 512;
/// Sentinel mask of an extended entry that failed its fixed positions.
pub const DEAD_MASK: u64 = 0xFFFF_FFFF_FFFF_FFFF;
/// Bottom entries each thread holds in registers in [`join_kernel`].
const ENTRIES_PER_THREAD: u32 = 4;
/// Partitions per launch (device slots), bounded below by memory. A field
/// is b^p partitions (3,249 at base 57), so this sets the launch count: 16
/// measured 1.11-1.37x faster per field than 4 on an RTX 3080 and an RTX
/// 4090, where launch overhead was most of the join's fixed per-field cost.
const SLOTS: usize = 16;

/// Launched batches kept in flight before the driver waits for the oldest.
const BATCHES_IN_FLIGHT: usize = 4;
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
const MIN_RETRY_CAP: u32 = 1 << 18;
/// Nice numbers one field may report.
const NICE_CAP: u32 = 1 << 10;
/// Cubes for the prefilter and check kernels (grid-stride loops).
const CHECK_CUBES: u32 = 1024;
/// The join's whole device working set (every buffer of [`JoinDevice`]) may
/// take at most this much; a field at bases 40-64 needs 0.3-0.9 GiB at 16
/// partitions per launch. See [`JoinLimits`].
const JOIN_MEMORY: usize = 1 << 30;
/// How deep the halving of an overflowing partition's top layer may go: a
/// layer of up to 2^32 prefixes reaches single prefixes well within it.
const MAX_SPLITS: u32 = 32;

// ---------------------------------------------------------------------------
// Device kernels
// ---------------------------------------------------------------------------

pub use kernels::{
    bucket_kernel, check_kernel, join_kernel, mid_kernel, top_fill_kernel, top_kernel,
    top_scan_kernel,
};

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
    // (midpoint) are not cube intrinsics.
    clippy::trivially_copy_pass_by_ref,
    clippy::collapsible_else_if,
    clippy::manual_midpoint
)]
mod kernels {
    use super::{DEAD_MASK, JOIN_WG, SURV_FLUSH, SURV_SH_CAP, TOP_CAP};
    use crate::cubecl_backend::candidate_check;
    use cubecl::prelude::*;

    /// Flush the shared survivor stage to the global list. Must be called by the
    /// whole cube at a point where no thread is still appending (after a
    /// barrier); `sc` is the stage count read after that barrier.
    #[cube]
    fn flush_survivors(
        s_t: &SharedMemory<u32>,
        s_r: &SharedMemory<u32>,
        s_cnt: &mut SharedMemory<Atomic<u32>>,
        s_base: &mut SharedMemory<u32>,
        surv: &mut Array<u32>,
        surv_count: &mut Array<Atomic<u32>>,
        surv_cap: u32,
        sc: u32,
    ) {
        let mut n = sc;
        if n > SURV_SH_CAP {
            n = SURV_SH_CAP; // the rest went straight to the global list
        }
        if UNIT_POS_X == 0u32 {
            s_base[0] = surv_count[0].fetch_add(n);
        }
        sync_cube();
        let gb = s_base[0];
        let mut i = UNIT_POS_X;
        while i < n {
            let g = gb + i;
            if g < surv_cap {
                surv[(2u32 * g) as usize] = s_t[i as usize];
                surv[(2u32 * g + 1u32) as usize] = s_r[i as usize];
            }
            i += CUBE_DIM_X;
        }
        sync_cube();
        if UNIT_POS_X == 0u32 {
            s_cnt[0].store(0u32);
        }
    }

    /// Step 1+ (v3): extension fused with bucketization. One cube per
    /// (class c, slot): its bpre segment is extended by the slot's partition
    /// digits exactly as in [`bottom_extend_kernel`] (the extended entries are
    /// written to `ext_m`/`ext_pk` as there), and each surviving entry's two
    /// key-position digits are stepped through every key value d incrementally
    /// (+s2, +s3 mod b); the entry's index goes onto the (slot, d, c) list for
    /// every d where they are distinct and new. List layout per slot:
    /// `slot*b*nbp + b*seg[c] + d*len(c) + i`; counts at `(slot*b + d)*m1 + c`.
    #[cube(launch_unchecked)]
    pub fn bucket_kernel(
        bp_r: &Array<u32>,
        bp_m: &Array<u32>,
        seg: &Array<u32>,
        vs: &Array<u32>,
        ext_m: &mut Array<u32>,
        ext_pk: &mut Array<u32>,
        lists: &mut Array<u32>,
        counts: &mut Array<u32>,
        nbp: u32,
        #[comptime] base: u32,
        #[comptime] f0: u32,
        #[comptime] pp: u32,
        #[comptime] k: u32,
        #[comptime] key_level: bool,
        #[comptime] key_at_zero: bool,
    ) {
        let m1 = comptime!(base - 1);
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
            let mut d = Array::<u32>::new(k as usize);
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
            for j in comptime!(f0 + pp)..k {
                d[j as usize] = 0u32;
            }
            let mut sq = Array::<u32>::new(k as usize);
            let mut carry = 0u32;
            #[unroll]
            for pos in 0..k {
                let mut col = carry;
                #[unroll]
                for i2 in 0..comptime!(pos + 1) {
                    col += d[i2 as usize] * d[comptime!(pos - i2) as usize];
                }
                sq[pos as usize] = col % base;
                carry = col / base;
            }
            let mut cu = Array::<u32>::new(k as usize);
            carry = 0u32;
            #[unroll]
            for pos in 0..k {
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
                if g2 == g3 || (m & bb) != 0u64 {
                    ok = false;
                }
                m |= bb;
            }
            if !ok {
                m = DEAD_MASK;
            }
            let idx = slot * nbp + i;
            ext_m[(2u32 * idx) as usize] = u32::cast_from(m);
            ext_m[(2u32 * idx + 1u32) as usize] = u32::cast_from(m >> 32u64);
            if key_level {
                let u2 = sq[comptime!(k - 1) as usize];
                let u3 = cu[comptime!(k - 1) as usize];
                let d0 = d[0];
                let s2 = (2u32 * d0) % base;
                let s3 = (3u32 * ((d0 * d0) % base)) % base;
                ext_pk[idx as usize] = u2 | (u3 << 8u32) | (s2 << 16u32) | (s3 << 24u32);
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
                        if a2 != a3 && (m & bb) == 0u64 {
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
    }

    /// Step 2 (v3): one cube per work item over the dense (slot, d, c) list
    /// built by [`bucket_kernel`]; `ept` entries per thread per wave, every
    /// thread runs every wave (uniform), survivors staged and flushed per wave.
    #[cube(launch_unchecked)]
    pub fn join_kernel(
        ext_m: &Array<u32>,
        ext_pk: &Array<u32>,
        bp_r: &Array<u32>,
        seg: &Array<u32>,
        lists: &Array<u32>,
        counts: &Array<u32>,
        work: &Array<u32>,
        tl: &Array<u32>,
        top_m: &Array<u32>,
        top_r: &Array<u32>,
        surv: &mut Array<u32>,
        surv_count: &mut Array<Atomic<u32>>,
        top_x: &Array<u32>,
        staged: &mut Array<Atomic<u32>>,
        nwork: u32,
        nbp: u32,
        surv_cap: u32,
        #[comptime] base: u32,
        #[comptime] key_level: bool,
        #[comptime] key_at_zero: bool,
        #[comptime] ept: u32,
        #[comptime] fused: bool,
        #[comptime] f0: u32,
        #[comptime] k2: u32,
    ) {
        let m1 = comptime!(base - 1);
        let nkeys = comptime!(if key_level { base } else { 1 });
        let wave = comptime!(ept * JOIN_WG);
        let mut plane_tot = SharedMemory::<u32>::new(comptime!(JOIN_WG as usize));
        let mut t_lo = SharedMemory::<u32>::new(comptime!(TOP_CAP as usize));
        let mut t_hi = SharedMemory::<u32>::new(comptime!(TOP_CAP as usize));
        let mut t_rlo = SharedMemory::<u32>::new(comptime!(TOP_CAP as usize));
        let mut t_rhi = SharedMemory::<u32>::new(comptime!(TOP_CAP as usize));
        let mut t_id = SharedMemory::<u32>::new(comptime!(TOP_CAP as usize));
        let mut s_t = SharedMemory::<u32>::new(comptime!(SURV_SH_CAP as usize));
        let mut s_r = SharedMemory::<u32>::new(comptime!(SURV_SH_CAP as usize));
        let mut s_cnt = SharedMemory::<Atomic<u32>>::new(1usize);
        let mut s_base = SharedMemory::<u32>::new(1usize);

        if UNIT_POS_X == 0u32 {
            s_cnt[0].store(0u32);
        }
        sync_cube();

        let mut w = CUBE_POS_X;
        while w < nwork {
            let slot = work[(4u32 * w) as usize];
            let bx = work[(4u32 * w + 1u32) as usize];
            let ts = work[(4u32 * w + 2u32) as usize];
            let te = work[(4u32 * w + 3u32) as usize];
            let dkey = bx / m1;
            let cls = bx - dkey * m1;
            let cs = seg[cls as usize];
            let clen = seg[(cls + 1u32) as usize] - cs;
            let ne = counts[((slot * nkeys + dkey) * m1 + cls) as usize];
            let lbase = slot * base * nbp + base * cs + dkey * clen;
            let ebase = slot * nbp;

            let mut tc = ts;
            while tc < te {
                let mut nt = te - tc;
                if nt > TOP_CAP {
                    nt = TOP_CAP;
                }
                if UNIT_POS_X < nt {
                    let t = tl[(tc + UNIT_POS_X) as usize];
                    t_lo[UNIT_POS_X as usize] = top_m[(2u32 * t) as usize];
                    t_hi[UNIT_POS_X as usize] = top_m[(2u32 * t + 1u32) as usize];
                    t_rlo[UNIT_POS_X as usize] = top_r[(2u32 * t) as usize];
                    t_rhi[UNIT_POS_X as usize] = top_r[(2u32 * t + 1u32) as usize];
                    t_id[UNIT_POS_X as usize] = t;
                }
                sync_cube();

                let mut ws = 0u32;
                while ws < ne {
                    let mut mlo = Array::<u32>::new(ept as usize);
                    let mut mhi = Array::<u32>::new(ept as usize);
                    let mut rr = Array::<u32>::new(ept as usize);
                    // Per-slot validity: an empty slot must never pass, whatever
                    // the top's mask (a top may certify nothing at small t).
                    let mut vv = Array::<u32>::new(ept as usize);
                    #[unroll]
                    for j in 0..ept {
                        let sl = ws + UNIT_POS_X + comptime!(j * JOIN_WG);
                        mlo[j as usize] = 0xFFFF_FFFFu32;
                        mhi[j as usize] = 0xFFFF_FFFFu32;
                        rr[j as usize] = 0u32;
                        vv[j as usize] = 0u32;
                        if sl < ne {
                            vv[j as usize] = 0xFFFF_FFFFu32;
                            let e = lists[(lbase + sl) as usize];
                            let gi = ebase + e;
                            let mut m = (u64::cast_from(ext_m[(2u32 * gi + 1u32) as usize])
                                << 32u64)
                                | u64::cast_from(ext_m[(2u32 * gi) as usize]);
                            if key_level {
                                let pk = ext_pk[gi as usize];
                                let mut g2 =
                                    ((pk & 255u32) + dkey * ((pk >> 16u32) & 255u32)) % base;
                                let mut g3 =
                                    (((pk >> 8u32) & 255u32) + dkey * (pk >> 24u32)) % base;
                                if key_at_zero {
                                    g2 = (dkey * dkey) % base;
                                    g3 = (g2 * dkey) % base;
                                }
                                m |= (1u64 << u64::cast_from(g2)) | (1u64 << u64::cast_from(g3));
                            }
                            mlo[j as usize] = u32::cast_from(m);
                            mhi[j as usize] = u32::cast_from(m >> 32u64);
                            rr[j as usize] = bp_r[e as usize];
                        }
                    }
                    // Branch-free AND over a chunk of up to 32 tops into a pass
                    // bitmask per entry, then only the (rare) set bits are
                    // walked: the survivor path diverges once per chunk, not
                    // once per top.
                    let mut jc = 0u32;
                    while jc < nt {
                        let mut cend = jc + 32u32;
                        if cend > nt {
                            cend = nt;
                        }
                        let mut pm = Array::<u32>::new(ept as usize);
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
                        #[unroll]
                        for j in 0..ept {
                            let mut x = pm[j as usize] & vv[j as usize];
                            let r0 = rr[j as usize];
                            while x != 0u32 {
                                let jt2 = jc + x.trailing_zeros();
                                x &= x - 1u32;
                                if r0 >= t_rlo[jt2 as usize] && r0 < t_rhi[jt2 as usize] {
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
                        }
                        jc += 32u32;
                    }
                    sync_cube();
                    let sc = s_cnt[0].load();
                    if sc >= SURV_FLUSH {
                        if fused {
                            flush_survivors_mid(
                                &s_t,
                                &s_r,
                                &mut s_cnt,
                                &mut s_base,
                                &mut plane_tot,
                                top_m,
                                top_x,
                                surv,
                                surv_count,
                                staged,
                                surv_cap,
                                sc,
                                base,
                                f0,
                                k2,
                            );
                        } else {
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
                        sync_cube();
                    }
                    ws += wave;
                }
                tc += TOP_CAP;
                sync_cube();
            }
            w += CUBE_COUNT_X;
        }
        sync_cube();
        let sc = s_cnt[0].load();
        if sc > 0u32 {
            if fused {
                flush_survivors_mid(
                    &s_t,
                    &s_r,
                    &mut s_cnt,
                    &mut s_base,
                    &mut plane_tot,
                    top_m,
                    top_x,
                    surv,
                    surv_count,
                    staged,
                    surv_cap,
                    sc,
                    base,
                    f0,
                    k2,
                );
            } else {
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
        }
    }

    /// One power's share of the top certificate, the device form of
    /// `Base::cert`'s inner loop: x and y are the power of the interval's two
    /// endpoints (`nl` u32 limbs, destroyed), `sp` its digit count. Positions
    /// >= c0 = max(cap, ndig(y - x)) are scanned least significant first in
    /// lockstep; a position where the digits differ resets the run, so what
    /// is left at the top is exactly the digits `cert` certifies (from the top
    /// down to the first disagreement), with `dup` set if two of them repeat.
    /// Returns the run mask; `c0` and `dup` via the out-arrays.
    #[cube]
    fn cert_power(
        x: &mut Array<u32>,
        y: &mut Array<u32>,
        pw: &Array<u32>,
        out_c0: &mut Array<u32>,
        out_dup: &mut Array<u32>,
        #[comptime] nl: u32,
        #[comptime] nl3: u32,
        #[comptime] sp: u32,
        #[comptime] cap: u32,
        #[comptime] base: u32,
        #[comptime] chunk_digits: u32,
        #[comptime] chunk_div: u32,
    ) -> u64 {
        // d = y - x (y >= x)
        let mut d = Array::<u32>::new(nl as usize);
        let mut borrow = 0u32;
        #[unroll]
        for j in 0..nl {
            let yj = y[j as usize];
            let xj = x[j as usize];
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
        let passes = comptime!(sp.div_ceil(chunk_digits));
        #[unroll]
        for pass in 0..passes {
            let mut rx = 0u32;
            let mut ry = 0u32;
            #[unroll]
            for jj in 0..nl {
                let j = comptime!(nl - 1 - jj);
                let vx = x[j as usize];
                let c1 = (rx << 16u32) | (vx >> 16u32);
                let q1 = c1 / chunk_div;
                let r1 = c1 - q1 * chunk_div;
                let c2 = (r1 << 16u32) | (vx & 0xFFFFu32);
                let q2 = c2 / chunk_div;
                rx = c2 - q2 * chunk_div;
                x[j as usize] = (q1 << 16u32) | q2;
                let vy = y[j as usize];
                let e1 = (ry << 16u32) | (vy >> 16u32);
                let s1 = e1 / chunk_div;
                let t1 = e1 - s1 * chunk_div;
                let e2 = (t1 << 16u32) | (vy & 0xFFFFu32);
                let s2 = e2 / chunk_div;
                ry = e2 - s2 * chunk_div;
                y[j as usize] = (s1 << 16u32) | s2;
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
                        }
                    }
                }
            }
        }
        out_c0[0] = c0;
        out_dup[0] = dup;
        run
    }

    /// Stage 0a: certify every top prefix of every slot's partition. Grid: x
    /// over the field's top layer (depth t - pp, host-built once per field),
    /// y over slots. Per tlay entry (5 words): P0 lo, hi, P0 mod (b-1),
    /// P0 mod b, P0 mod b^(k2-f0-pp). `fb` = s (3 limbs), e-1 (3 limbs), the P
    /// range lo (2 words) and hi (2 words). Passing tops are appended per slot
    /// (stride `ntlay`) and counted into their buckets, one per digit-sum root.
    #[cube(launch_unchecked)]
    pub fn top_kernel(
        tlay: &Array<u32>,
        vs: &Array<u32>,
        fb: &Array<u32>,
        pw: &Array<u32>,
        roots: &Array<u32>,
        top_p: &mut Array<u32>,
        top_m: &mut Array<u32>,
        top_r: &mut Array<u32>,
        top_x: &mut Array<u32>,
        top_b: &mut Array<u32>,
        top_count: &mut Array<Atomic<u32>>,
        bucket_cnt: &mut Array<Atomic<u32>>,
        ntlay: u32,
        nroots: u32,
        w: u32,
        pdiv: u32,
        pdiv_m1: u32,
        full_floor: u32, // certificate floor bound of every unclipped block (host rule)
        #[comptime] base: u32,
        #[comptime] limbs: u32,
        #[comptime] cap: u32,
        #[comptime] k2: u32,
        #[comptime] s2: u32,
        #[comptime] s3: u32,
        #[comptime] chunk_digits: u32,
        #[comptime] chunk_div: u32,
        #[comptime] key_level: bool,
        #[comptime] nb: u32,
    ) {
        let m1 = comptime!(base - 1);
        let nl2 = comptime!(2 * limbs);
        let nl3 = comptime!(3 * limbs);
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
                // Powers: a2 = a^2, e2 = e^2 (2L limbs), a3, e3 (3L limbs).
                let mut a2 = Array::<u32>::new(nl2 as usize);
                let mut e2 = Array::<u32>::new(nl2 as usize);
                #[unroll]
                for j in 0..nl2 {
                    a2[j as usize] = 0u32;
                    e2[j as usize] = 0u32;
                }
                #[unroll]
                for i1 in 0..limbs {
                    let mut ca = 0u64;
                    let mut ce = 0u64;
                    #[unroll]
                    for j in 0..limbs {
                        let kk = comptime!(i1 + j);
                        let ta = u64::cast_from(a[i1 as usize]) * u64::cast_from(a[j as usize])
                            + u64::cast_from(a2[kk as usize])
                            + ca;
                        a2[kk as usize] = u32::cast_from(ta);
                        ca = ta >> 32u64;
                        let te = u64::cast_from(e[i1 as usize]) * u64::cast_from(e[j as usize])
                            + u64::cast_from(e2[kk as usize])
                            + ce;
                        e2[kk as usize] = u32::cast_from(te);
                        ce = te >> 32u64;
                    }
                    a2[comptime!(i1 + limbs) as usize] = u32::cast_from(ca);
                    e2[comptime!(i1 + limbs) as usize] = u32::cast_from(ce);
                }
                let mut a3 = Array::<u32>::new(nl3 as usize);
                let mut e3 = Array::<u32>::new(nl3 as usize);
                #[unroll]
                for j in 0..nl3 {
                    a3[j as usize] = 0u32;
                    e3[j as usize] = 0u32;
                }
                #[unroll]
                for i1 in 0..nl2 {
                    let mut ca = 0u64;
                    let mut ce = 0u64;
                    #[unroll]
                    for j in 0..limbs {
                        let kk = comptime!(i1 + j);
                        let ta = u64::cast_from(a2[i1 as usize]) * u64::cast_from(a[j as usize])
                            + u64::cast_from(a3[kk as usize])
                            + ca;
                        a3[kk as usize] = u32::cast_from(ta);
                        ca = ta >> 32u64;
                        let te = u64::cast_from(e2[i1 as usize]) * u64::cast_from(e[j as usize])
                            + u64::cast_from(e3[kk as usize])
                            + ce;
                        e3[kk as usize] = u32::cast_from(te);
                        ce = te >> 32u64;
                    }
                    a3[comptime!(i1 + limbs) as usize] = u32::cast_from(ca);
                    e3[comptime!(i1 + limbs) as usize] = u32::cast_from(ce);
                }
                let mut c0a = Array::<u32>::new(1usize);
                let mut dupa = Array::<u32>::new(1usize);
                let mut c0b = Array::<u32>::new(1usize);
                let mut dupb = Array::<u32>::new(1usize);
                let msq = cert_power(
                    &mut a2,
                    &mut e2,
                    pw,
                    &mut c0a,
                    &mut dupa,
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
                    &mut e3,
                    pw,
                    &mut c0b,
                    &mut dupb,
                    nl3,
                    nl3,
                    s3,
                    cap,
                    base,
                    chunk_digits,
                    chunk_div,
                );
                if dupa[0] == 0u32 && dupb[0] == 0u32 && (msq & mcu) == 0u64 {
                    let mask = msq | mcu;
                    let mut floor = c0a[0];
                    if c0b[0] < floor {
                        floor = c0b[0];
                    }
                    // Same seed rule as the host: an unclipped block uses the
                    // field bound (a lower bound of its exact floor).
                    if !a_lt_s && !e_gt_e {
                        floor = full_floor;
                    }
                    let idx = top_count[slot as usize].fetch_add(1u32);
                    let g = slot * ntlay + idx;
                    top_p[(2u32 * g) as usize] = u32::cast_from(p);
                    top_p[(2u32 * g + 1u32) as usize] = u32::cast_from(p >> 32u64);
                    top_m[(2u32 * g) as usize] = u32::cast_from(mask);
                    top_m[(2u32 * g + 1u32) as usize] = u32::cast_from(mask >> 32u64);
                    top_r[(2u32 * g) as usize] = a[0] - pw0;
                    top_r[(2u32 * g + 1u32) as usize] = e[0] - pw0 + 1u32;
                    let pmod = tlay[(5u32 * i + 4u32) as usize] * pdiv + v;
                    top_x[(2u32 * g) as usize] = pmod;
                    let mut seed = 0u32;
                    if floor >= k2 {
                        seed = 1u32;
                    }
                    top_x[(2u32 * g + 1u32) as usize] = seed;
                    let pc = (tlay[(5u32 * i + 2u32) as usize] * pdiv_m1 + v) % m1;
                    let mut key = 0u32;
                    if key_level {
                        key = tlay[(5u32 * i + 3u32) as usize];
                    }
                    top_b[g as usize] = pc | (key << 8u32);
                    let mut r = 0u32;
                    while r < nroots {
                        let cls = (roots[r as usize] + m1 - pc) % m1;
                        bucket_cnt[(slot * nb + key * m1 + cls) as usize].fetch_add(1u32);
                        r += 1u32;
                    }
                }
            }
        }
    }

    /// Stage 0b: per slot (one cube), exclusive scan of the bucket counts into
    /// offsets and fill cursors; writes every (slot, bucket) work item.
    #[cube(launch_unchecked)]
    pub fn top_scan_kernel(
        bucket_cnt: &Array<Atomic<u32>>,
        cursor: &mut Array<Atomic<u32>>,
        work: &mut Array<u32>,
        tl_stride: u32,
        #[comptime] nb: u32,
    ) {
        let per = comptime!(nb.div_ceil(JOIN_WG));
        let mut tot = SharedMemory::<u32>::new(comptime!(JOIN_WG as usize));
        let slot = CUBE_POS_X;
        let first = UNIT_POS_X * per;
        let mut acc = 0u32;
        #[unroll]
        for q in 0..per {
            let bx = first + q;
            if bx < nb {
                acc += bucket_cnt[(slot * nb + bx) as usize].load();
            }
        }
        tot[UNIT_POS_X as usize] = acc;
        sync_cube();
        if UNIT_POS_X == 0u32 {
            let mut run = 0u32;
            let mut t = 0u32;
            while t < CUBE_DIM_X {
                let c = tot[t as usize];
                tot[t as usize] = run;
                run += c;
                t += 1u32;
            }
        }
        sync_cube();
        let mut off = tot[UNIT_POS_X as usize] + slot * tl_stride;
        #[unroll]
        for q in 0..per {
            let bx = first + q;
            if bx < nb {
                let c = bucket_cnt[(slot * nb + bx) as usize].load();
                cursor[(slot * nb + bx) as usize].store(off);
                let wi = slot * nb + bx;
                work[(4u32 * wi) as usize] = slot;
                work[(4u32 * wi + 1u32) as usize] = bx;
                work[(4u32 * wi + 2u32) as usize] = off;
                work[(4u32 * wi + 3u32) as usize] = off + c;
                off += c;
            }
        }
    }

    /// Stage 0c: every certified top onto the list of each bucket it probes.
    #[cube(launch_unchecked)]
    pub fn top_fill_kernel(
        top_b: &Array<u32>,
        top_count: &Array<Atomic<u32>>,
        roots: &Array<u32>,
        cursor: &mut Array<Atomic<u32>>,
        tl: &mut Array<u32>,
        ntlay: u32,
        nroots: u32,
        #[comptime] base: u32,
        #[comptime] nb: u32,
    ) {
        let m1 = comptime!(base - 1);
        let i = ABSOLUTE_POS_X;
        let slot = CUBE_POS_Y;
        if i < top_count[slot as usize].load() {
            let g = slot * ntlay + i;
            let b = top_b[g as usize];
            let pc = b & 255u32;
            let key = b >> 8u32;
            let mut r = 0u32;
            while r < nroots {
                let cls = (roots[r as usize] + m1 - pc) % m1;
                let at = cursor[(slot * nb + key * m1 + cls) as usize].fetch_add(1u32);
                tl[at as usize] = g;
                r += 1u32;
            }
        }
    }

    /// The middle-digit test for one survivor `(top t, r0)`: digits `0..k2` of
    /// n² and n³ from n's low `k2` digits (`r0` and the top's `P mod b^(k2-f0)`)
    /// must be distinct, and disjoint from the top's certificate when its seed
    /// flag says every certified position is `>= k2`.
    #[cube]
    fn mid_pass(
        t: u32,
        r0: u32,
        top_m: &Array<u32>,
        top_x: &Array<u32>,
        #[comptime] base: u32,
        #[comptime] f0: u32,
        #[comptime] k2: u32,
    ) -> bool {
        let pm = top_x[(2u32 * t) as usize];
        let mut m = 0u64;
        if top_x[(2u32 * t + 1u32) as usize] != 0u32 {
            m = (u64::cast_from(top_m[(2u32 * t + 1u32) as usize]) << 32u64)
                | u64::cast_from(top_m[(2u32 * t) as usize]);
        }
        let mut d = Array::<u32>::new(k2 as usize);
        let mut x = r0;
        #[unroll]
        for j in 0..f0 {
            d[j as usize] = x % base;
            x /= base;
        }
        let mut y = pm;
        #[unroll]
        for j in f0..k2 {
            d[j as usize] = y % base;
            y /= base;
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
        let mut ok = true;
        carry = 0u32;
        #[unroll]
        for pos in 0..k2 {
            let mut col = carry;
            #[unroll]
            for i2 in 0..comptime!(pos + 1) {
                col += sq[i2 as usize] * d[comptime!(pos - i2) as usize];
            }
            let g3 = col % base;
            carry = col / base;
            let g2 = sq[pos as usize];
            let bb = (1u64 << u64::cast_from(g2)) | (1u64 << u64::cast_from(g3));
            if g2 == g3 || (m & bb) != 0u64 {
                ok = false;
            }
            m |= bb;
        }
        ok
    }

    /// [`flush_survivors`] with the middle-digit test applied on the way out
    /// (the fused prefilter): the staged survivors are tested by the whole cube
    /// and only those that pass go to `out`, compacted per plane with one atomic
    /// per round. `staged` counts every survivor staged (statistics).
    #[cube]
    fn flush_survivors_mid(
        s_t: &SharedMemory<u32>,
        s_r: &SharedMemory<u32>,
        s_cnt: &mut SharedMemory<Atomic<u32>>,
        s_base: &mut SharedMemory<u32>,
        plane_tot: &mut SharedMemory<u32>,
        top_m: &Array<u32>,
        top_x: &Array<u32>,
        out: &mut Array<u32>,
        out_count: &mut Array<Atomic<u32>>,
        staged: &mut Array<Atomic<u32>>,
        out_cap: u32,
        sc: u32,
        #[comptime] base: u32,
        #[comptime] f0: u32,
        #[comptime] k2: u32,
    ) {
        let mut n = sc;
        if n > SURV_SH_CAP {
            n = SURV_SH_CAP; // the rest went straight to `out`, untested
        }
        if UNIT_POS_X == 0u32 {
            staged[0].fetch_add(n);
        }
        let my_plane = UNIT_POS_X / PLANE_DIM;
        let num_planes = CUBE_DIM_X / PLANE_DIM;
        let mut rb = 0u32;
        while rb < n {
            let i = rb + UNIT_POS_X;
            let mut pass = 0u32;
            let mut t = 0u32;
            let mut r0 = 0u32;
            if i < n {
                t = s_t[i as usize];
                r0 = s_r[i as usize];
                if mid_pass(t, r0, top_m, top_x, base, f0, k2) {
                    pass = 1u32;
                }
            }
            let idx = plane_exclusive_sum(pass);
            let tot = plane_sum(pass);
            if UNIT_POS_PLANE == 0u32 {
                plane_tot[my_plane as usize] = tot;
            }
            sync_cube();
            let mut off = 0u32;
            let mut all = 0u32;
            let mut p = 0u32;
            while p < num_planes {
                let tp = plane_tot[p as usize];
                if p < my_plane {
                    off += tp;
                }
                all += tp;
                p += 1u32;
            }
            if UNIT_POS_X == 0u32 && all > 0u32 {
                s_base[0] = out_count[0].fetch_add(all);
            }
            sync_cube();
            if pass != 0u32 {
                let g = s_base[0] + off + idx;
                if g < out_cap {
                    out[(2u32 * g) as usize] = t;
                    out[(2u32 * g + 1u32) as usize] = r0;
                }
            }
            rb += CUBE_DIM_X;
        }
        sync_cube();
        if UNIT_POS_X == 0u32 {
            s_cnt[0].store(0u32);
        }
    }

    /// Step 3 (optional): middle-digit prefilter with compaction. See the module
    /// docs. Soundness: every digit tested is a real digit of n^2 or n^3 at a
    /// distinct position (the host guarantees both powers have >= k2 digits), and
    /// seed digits sit at positions >= k2, so any repeat proves n is not nice.
    #[cube(launch_unchecked)]
    pub fn mid_kernel(
        surv: &Array<u32>,
        surv_count: &mut Array<Atomic<u32>>,
        top_m: &Array<u32>, // lo, hi certificate words per top
        top_x: &Array<u32>, // P mod b^(k2-f0), seed flag per top
        out: &mut Array<u32>,
        out_count: &mut Array<Atomic<u32>>,
        surv_cap: u32,
        out_cap: u32,
        #[comptime] base: u32,
        #[comptime] f0: u32,
        #[comptime] k2: u32,
    ) {
        let mut plane_tot = SharedMemory::<u32>::new(comptime!(JOIN_WG as usize));
        let mut s_base = SharedMemory::<u32>::new(1usize);
        let my_plane = UNIT_POS_X / PLANE_DIM;
        let num_planes = CUBE_DIM_X / PLANE_DIM;
        let mut count = surv_count[0].load();
        if count > surv_cap {
            count = surv_cap;
        }
        let mut rb = CUBE_POS_X * CUBE_DIM_X;
        let rstride = CUBE_COUNT_X * CUBE_DIM_X;
        while rb < count {
            let i = rb + UNIT_POS_X;
            let mut pass = 0u32;
            let mut t = 0u32;
            let mut r0 = 0u32;
            if i < count {
                t = surv[(2u32 * i) as usize];
                r0 = surv[(2u32 * i + 1u32) as usize];
                if mid_pass(t, r0, top_m, top_x, base, f0, k2) {
                    pass = 1u32;
                }
            }
            let idx = plane_exclusive_sum(pass);
            let tot = plane_sum(pass);
            if UNIT_POS_PLANE == 0u32 {
                plane_tot[my_plane as usize] = tot;
            }
            sync_cube();
            let mut off = 0u32;
            let mut all = 0u32;
            let mut p = 0u32;
            while p < num_planes {
                let tp = plane_tot[p as usize];
                if p < my_plane {
                    off += tp;
                }
                all += tp;
                p += 1u32;
            }
            if UNIT_POS_X == 0u32 && all > 0u32 {
                s_base[0] = out_count[0].fetch_add(all);
            }
            sync_cube();
            if pass != 0u32 {
                let g = s_base[0] + off + idx;
                if g < out_cap {
                    out[(2u32 * g) as usize] = t;
                    out[(2u32 * g + 1u32) as usize] = r0;
                }
            }
            rb += rstride;
        }
    }

    /// Step 4: grid-stride over the survivor list; rebuild n = P*b^f0 + r0 and
    /// run the client's full check (probe: report every n instead).
    #[cube(launch_unchecked)]
    pub fn check_kernel(
        surv: &Array<u32>,
        surv_count: &mut Array<Atomic<u32>>,
        top_p: &Array<u32>, // lo, hi words of P per top
        nice_out: &mut Array<u32>,
        nice_count: &mut Array<Atomic<u32>>,
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
            let p_lo = u64::cast_from(top_p[(2u32 * t) as usize]);
            let p_hi = u64::cast_from(top_p[(2u32 * t + 1u32) as usize]);
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

/// Everything about a field that does not depend on the partition value.
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
    fn with_tops(&self, tlay: Vec<(u128, u64)>) -> Self {
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
pub(crate) struct JoinLimits {
    pub max_binding: usize,
    pub budget: usize,
}

impl JoinLimits {
    /// The largest buffer is the runtime's `max_page_size`: on wgpu the
    /// adapter's `max_storage_buffer_binding_size` (128 MiB on lavapipe, 1-4
    /// GiB on desktop drivers); on CUDA, and on Vulkan with 64-bit indexing,
    /// a quarter of the device's memory. The budget is [`JOIN_MEMORY`] but at
    /// most twice that: half the device where the runtime knows its size, and
    /// a small working set where buffers are small (software rasterizers,
    /// embedded GPUs).
    pub(crate) fn of<R: Runtime>(client: &ComputeClient<R>) -> Self {
        let page = usize::try_from(client.properties().memory.max_page_size)
            .unwrap_or(usize::MAX)
            .max(1);
        Self {
            max_binding: page,
            budget: JOIN_MEMORY.min(page.saturating_mul(2)),
        }
    }
}

/// Bytes in each of one field's device buffers, as [`JoinDevice::new`]
/// allocates them: per slot for those that grow with the partitions per
/// launch, then the field's own tables.
#[derive(Clone, Copy, Debug)]
struct Footprint {
    ext_m: usize,
    ext_pk: usize,
    lists: usize,
    counts: usize,
    /// Each of `top_p`, `top_m`, `top_r` and `top_x`.
    top: usize,
    top_b: usize,
    tl: usize,
    work: usize,
    cursor: usize,
    /// What a launched batch allocates for itself (`bucket_cnt`), for every
    /// batch in flight.
    scratch: usize,
    /// The field's tables and the output list.
    fixed: usize,
    largest_fixed: usize,
}

impl Footprint {
    fn of(fs: &FieldSetup, nice_cap: u32) -> Self {
        let b = fs.b as usize;
        let nbp = fs.bp_r.len();
        let ntl = fs.tlay.len().max(1);
        let nroots = fs.base.roots.len();
        let nkeys = if fs.key_level { b } else { 1 };
        let nl3 = 3 * n_limbs(fs.b).unwrap_or(3) as usize;
        let tables = [
            nbp * 4,                                          // bp_r
            nbp * 8,                                          // bp_m
            fs.seg.len() * 4,                                 // seg
            fs.tlay.len() * 5 * 4,                            // tlay
            10 * 4,                                           // fb
            (fs.base.s3 as usize + 1) * nl3 * 4,              // pw
            nroots * 4,                                       // roots
            nice_cap as usize * NICEONLY_STRIDE as usize * 4, // nice_out
            4,                                                // nice_count
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

    fn per_slot(&self) -> usize {
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
    fn largest_per_slot(&self) -> usize {
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

/// A niceonly field ready for the join on one device: its host-side setup
/// (the top layer and the bottom prefixes) and its two device layouts.
/// Preparing it decides the field's route (see
/// `cubecl_backend::plan_join`), and the join's worker thread then has no
/// host work left before its first launch.
pub struct JoinField {
    fs: FieldSetup,
    plan: JoinPlan,
    retry: JoinPlan,
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

    pub(crate) fn base(&self) -> u32 {
        self.fs.b
    }

    pub(crate) fn range(&self) -> FieldSize {
        FieldSize::new(self.fs.s, self.fs.e)
    }

    /// A field with layouts of the test's choosing.
    #[cfg(test)]
    fn with_plans(fs: FieldSetup, plan: JoinPlan, retry: JoinPlan) -> Self {
        Self { fs, plan, retry }
    }
}

/// Device buffers for one field: the field's tables, the per-slot scratch
/// the kernels fill, and the survivor and output lists.
pub(crate) struct JoinDevice<R: Runtime> {
    client: ComputeClient<R>,
    b: u32,
    f0: u32,
    pp: u32,
    k: u32,
    k2: u32,
    mid: u32,
    /// Prefilter in the join kernel's flush instead of a separate pass.
    fused: bool,
    key_level: bool,
    limbs: u32,
    chunk_digits: u32,
    chunk_div: u32,
    wide: bool,
    w_f0: u32,
    nbp: u32,
    max_slots: usize,
    bp_r: Handle,
    bp_m: Handle,
    seg: Handle,
    ext_m: Handle,
    ext_pk: Handle,
    lists: Handle,
    counts: Handle,
    nkeys: u32,
    surv: Handle,
    surv_cap: u32,
    list2: Handle,
    list2_cap: u32,
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
    pdiv_m1: u32,
    full_floor: u32,
    tlay: Handle,
    fb: Handle,
    pw: Handle,
    pw_len: usize,
    roots: Handle,
    top_p: Handle,
    top_m: Handle,
    top_r: Handle,
    top_x: Handle,
    top_b: Handle,
    tl: Handle,
    work: Handle,
    cursor: Handle,
}

/// What a launched batch leaves behind to be read later.
pub(crate) struct BatchRec {
    pub surv_count: Handle,
    pub list2_count: Handle,
    /// Survivors were prefiltered in the join kernel: `surv_count` counts
    /// what was staged, and no unfiltered list exists to overflow.
    pub fused: bool,
}

impl<R: Runtime> JoinDevice<R> {
    /// # Errors
    /// A base without a `u128` range, or `n` too wide for device tops.
    // One buffer per line, as the plan sized them.
    #[allow(clippy::too_many_lines)]
    pub fn new(client: &ComputeClient<R>, fs: &FieldSetup, plan: &JoinPlan) -> Result<Self> {
        let limbs = n_limbs(fs.b).ok_or_else(|| anyhow!("base {} has no u128 range", fs.b))?;
        ensure!(limbs <= 3, "device tops: n must fit 96 bits");
        let wide = wide_chunk_for(client);
        let (chunk_digits, chunk_div) = if wide {
            chunk_constants(fs.b)
        } else {
            chunk_constants_u16(fs.b)
        };
        let nbp = fs.bp_r.len();
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
        let (fused, surv_cap, list2_cap, nice_cap) =
            (plan.fused, plan.surv_cap, plan.list2_cap, plan.nice_cap);

        // Device-side tops: the top layer with its residues, the field
        // bounds, the power table and the per-slot top and bucket lists.
        let (b128, pp) = (u128::from(fs.b), fs.jp.p);
        let m1 = u128::from(fs.m1);
        let p0mod = fs.base.powu(fs.k2 - fs.f0 - pp);
        let mut tl = Vec::with_capacity(5 * fs.tlay.len());
        for &(p0, _) in &fs.tlay {
            let w = u32_words(p0);
            ensure!(w[2] == 0 && w[3] == 0, "top-layer prefix exceeds u64");
            // p0mod = b^(k2 - f0 - p) < 2^32, as `FieldSetup::new` chose k2.
            tl.extend_from_slice(&[
                w[0],
                w[1],
                u32::try_from(p0 % m1)?,
                u32::try_from(p0 % b128)?,
                u32::try_from(p0 % p0mod)?,
            ]);
        }
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
        let ntlay = fs.tlay.len();
        let nb = fs.nb;
        let pdiv = fs.nparts;
        Ok(Self {
            client: client.clone(),
            b: fs.b,
            f0: fs.f0,
            pp,
            k: fs.jp.k,
            k2: fs.k2,
            mid: fs.mid,
            fused,
            key_level: fs.key_level,
            limbs,
            chunk_digits,
            chunk_div,
            wide,
            w_f0: u32::try_from(fs.w)?,
            nbp: u32::try_from(nbp)?,
            max_slots,
            bp_r: client.create(cubecl::bytes::Bytes::from_elems(fs.bp_r.clone())),
            bp_m: client.create(cubecl::bytes::Bytes::from_elems(bp_m_words)),
            seg: client.create(cubecl::bytes::Bytes::from_elems(fs.seg.clone())),
            ext_m: client.empty(max_slots * fp.ext_m),
            ext_pk: client.empty(max_slots * fp.ext_pk),
            lists: client.empty(max_slots * fp.lists),
            counts: client.empty(max_slots * fp.counts),
            nkeys,
            // An unused list (fused: the join's own; no prefilter: its
            // output) is a placeholder.
            surv: client.empty((surv_cap as usize * 8).max(8)),
            surv_cap,
            list2: client.empty((list2_cap as usize * 8).max(8)),
            list2_cap,
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
            pdiv_m1: u32::try_from(pdiv % m1)?,
            full_floor: fs.full_floor,
            tlay: client.create(cubecl::bytes::Bytes::from_elems(tl)),
            fb: client.create(cubecl::bytes::Bytes::from_elems(fb)),
            pw_len: pw.len(),
            pw: client.create(cubecl::bytes::Bytes::from_elems(pw)),
            roots: client.create(cubecl::bytes::Bytes::from_elems(roots)),
            top_p: client.empty(max_slots * fp.top),
            top_m: client.empty(max_slots * fp.top),
            top_r: client.empty(max_slots * fp.top),
            top_x: client.empty(max_slots * fp.top),
            top_b: client.empty(max_slots * fp.top_b),
            tl: client.empty(max_slots * fp.tl),
            work: client.empty(max_slots * fp.work),
            cursor: client.empty(max_slots * fp.cursor),
        })
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
        let tl_len = max_slots * ntl * self.nroots as usize;
        let vs_h = c.create(cubecl::bytes::Bytes::from_elems(vs.to_vec()));
        let top_count = c.create(cubecl::bytes::Bytes::from_elems(vec![0u32; nslots]));
        let bucket_cnt = c.create(cubecl::bytes::Bytes::from_elems(vec![0u32; nslots * nb]));
        let surv_count = c.create(cubecl::bytes::Bytes::from_elems(vec![0u32; 1]));
        let list2_count = c.create(cubecl::bytes::Bytes::from_elems(vec![0u32; 1]));
        // Fused prefilter: the join kernel tests survivors as it flushes them
        // and writes only the passing ones, straight to list2.
        let fused = self.fused;
        let staged = c.create(cubecl::bytes::Bytes::from_elems(vec![0u32; 1]));
        let (cd16, cdiv16) = chunk_constants_u16(self.b);
        unsafe {
            top_kernel::launch_unchecked::<R>(
                c,
                CubeCount::Static(self.ntlay.div_ceil(JOIN_WG).max(1), nslots_u32, 1),
                CubeDim::new_1d(JOIN_WG),
                ArrayArg::from_raw_parts(self.tlay.clone(), 5 * ntl),
                ArrayArg::from_raw_parts(vs_h.clone(), nslots),
                ArrayArg::from_raw_parts(self.fb.clone(), 10),
                ArrayArg::from_raw_parts(self.pw.clone(), self.pw_len),
                ArrayArg::from_raw_parts(self.roots.clone(), self.nroots as usize),
                ArrayArg::from_raw_parts(self.top_p.clone(), 2 * max_slots * ntl),
                ArrayArg::from_raw_parts(self.top_m.clone(), 2 * max_slots * ntl),
                ArrayArg::from_raw_parts(self.top_r.clone(), 2 * max_slots * ntl),
                ArrayArg::from_raw_parts(self.top_x.clone(), 2 * max_slots * ntl),
                ArrayArg::from_raw_parts(self.top_b.clone(), max_slots * ntl),
                ArrayArg::from_raw_parts(top_count.clone(), nslots),
                ArrayArg::from_raw_parts(bucket_cnt.clone(), nslots * nb),
                self.ntlay,
                self.nroots,
                self.w_f0,
                self.pdiv,
                self.pdiv_m1,
                self.full_floor,
                self.b,
                self.limbs,
                self.k,
                self.k2,
                self.s2,
                self.s3,
                cd16,
                cdiv16,
                self.key_level,
                self.nb,
            );
            top_scan_kernel::launch_unchecked::<R>(
                c,
                CubeCount::Static(nslots_u32, 1, 1),
                CubeDim::new_1d(JOIN_WG),
                ArrayArg::from_raw_parts(bucket_cnt.clone(), nslots * nb),
                ArrayArg::from_raw_parts(self.cursor.clone(), max_slots * nb),
                ArrayArg::from_raw_parts(self.work.clone(), 4 * max_slots * nb),
                self.ntlay * self.nroots,
                self.nb,
            );
            top_fill_kernel::launch_unchecked::<R>(
                c,
                CubeCount::Static(self.ntlay.div_ceil(JOIN_WG).max(1), nslots_u32, 1),
                CubeDim::new_1d(JOIN_WG),
                ArrayArg::from_raw_parts(self.top_b.clone(), max_slots * ntl),
                ArrayArg::from_raw_parts(top_count.clone(), nslots),
                ArrayArg::from_raw_parts(self.roots.clone(), self.nroots as usize),
                ArrayArg::from_raw_parts(self.cursor.clone(), max_slots * nb),
                ArrayArg::from_raw_parts(self.tl.clone(), tl_len),
                self.ntlay,
                self.nroots,
                self.b,
                self.nb,
            );
            bucket_kernel::launch_unchecked::<R>(
                c,
                CubeCount::Static(self.b - 1, nslots_u32, 1),
                CubeDim::new_1d(JOIN_WG),
                ArrayArg::from_raw_parts(self.bp_r.clone(), nbp),
                ArrayArg::from_raw_parts(self.bp_m.clone(), 2 * nbp),
                ArrayArg::from_raw_parts(self.seg.clone(), self.b as usize),
                ArrayArg::from_raw_parts(vs_h, nslots),
                ArrayArg::from_raw_parts(self.ext_m.clone(), 2 * nbp * max_slots),
                ArrayArg::from_raw_parts(self.ext_pk.clone(), (nbp * max_slots).max(1)),
                ArrayArg::from_raw_parts(self.lists.clone(), nbp * max_slots * self.b as usize),
                ArrayArg::from_raw_parts(
                    self.counts.clone(),
                    max_slots * self.nkeys as usize * (self.b as usize - 1),
                ),
                self.nbp,
                self.b,
                self.f0,
                self.pp,
                self.k,
                self.key_level,
                self.key_level && self.k == 1,
            );
            join_kernel::launch_unchecked::<R>(
                c,
                CubeCount::Static(nwork_u32.min(65_535), 1, 1),
                CubeDim::new_1d(JOIN_WG),
                ArrayArg::from_raw_parts(self.ext_m.clone(), 2 * nbp * max_slots),
                ArrayArg::from_raw_parts(self.ext_pk.clone(), (nbp * max_slots).max(1)),
                ArrayArg::from_raw_parts(self.bp_r.clone(), nbp),
                ArrayArg::from_raw_parts(self.seg.clone(), self.b as usize),
                ArrayArg::from_raw_parts(self.lists.clone(), nbp * max_slots * self.b as usize),
                ArrayArg::from_raw_parts(
                    self.counts.clone(),
                    max_slots * self.nkeys as usize * (self.b as usize - 1),
                ),
                ArrayArg::from_raw_parts(self.work.clone(), 4 * max_slots * nb),
                ArrayArg::from_raw_parts(self.tl.clone(), tl_len),
                ArrayArg::from_raw_parts(self.top_m.clone(), 2 * max_slots * ntl),
                ArrayArg::from_raw_parts(self.top_r.clone(), 2 * max_slots * ntl),
                ArrayArg::from_raw_parts(
                    if fused {
                        self.list2.clone()
                    } else {
                        self.surv.clone()
                    },
                    2 * (if fused { self.list2_cap } else { self.surv_cap }) as usize,
                ),
                ArrayArg::from_raw_parts(
                    if fused {
                        list2_count.clone()
                    } else {
                        surv_count.clone()
                    },
                    1,
                ),
                ArrayArg::from_raw_parts(self.top_x.clone(), 2 * max_slots * ntl),
                ArrayArg::from_raw_parts(staged.clone(), 1),
                nwork_u32,
                self.nbp,
                if fused { self.list2_cap } else { self.surv_cap },
                self.b,
                self.key_level,
                self.key_level && self.k == 1,
                ENTRIES_PER_THREAD,
                fused,
                self.f0,
                self.k2,
            );
        }
        let (chk_list, chk_count, chk_cap) = if fused {
            (self.list2.clone(), list2_count.clone(), self.list2_cap)
        } else if self.mid > 0 {
            unsafe {
                mid_kernel::launch_unchecked::<R>(
                    c,
                    CubeCount::Static(CHECK_CUBES, 1, 1),
                    CubeDim::new_1d(JOIN_WG),
                    ArrayArg::from_raw_parts(self.surv.clone(), 2 * self.surv_cap as usize),
                    ArrayArg::from_raw_parts(surv_count.clone(), 1),
                    ArrayArg::from_raw_parts(self.top_m.clone(), 2 * max_slots * ntl),
                    ArrayArg::from_raw_parts(self.top_x.clone(), 2 * max_slots * ntl),
                    ArrayArg::from_raw_parts(self.list2.clone(), 2 * self.list2_cap as usize),
                    ArrayArg::from_raw_parts(list2_count.clone(), 1),
                    self.surv_cap,
                    self.list2_cap,
                    self.b,
                    self.f0,
                    self.k2,
                );
            }
            (self.list2.clone(), list2_count.clone(), self.list2_cap)
        } else {
            (self.surv.clone(), surv_count.clone(), self.surv_cap)
        };
        unsafe {
            check_kernel::launch_unchecked::<R>(
                c,
                CubeCount::Static(CHECK_CUBES, 1, 1),
                CubeDim::new_1d(JOIN_WG),
                ArrayArg::from_raw_parts(chk_list, 2 * chk_cap as usize),
                ArrayArg::from_raw_parts(chk_count, 1),
                ArrayArg::from_raw_parts(self.top_p.clone(), 2 * max_slots * ntl),
                ArrayArg::from_raw_parts(
                    self.nice_out.clone(),
                    self.nice_cap as usize * NICEONLY_STRIDE as usize,
                ),
                ArrayArg::from_raw_parts(self.nice_count.clone(), 1),
                chk_cap,
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
            surv_count: if fused { staged } else { surv_count },
            list2_count,
            fused,
        })
    }

    /// Whether a batch dropped survivors past the end of a list, from its
    /// two counts (`BatchRec::surv_count`, `BatchRec::list2_count`).
    fn overflowed(&self, rec: &BatchRec, surv: u32, checked: u32) -> bool {
        (!rec.fused && surv > self.surv_cap) || checked > self.list2_cap
    }

    /// How many survivors the full check read, from a batch's two counts.
    fn checked_of(&self, surv: u32, checked: u32) -> u32 {
        if self.mid > 0 { checked } else { surv }
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

    /// The (prefilter, with `second`) survivors of the most recent batch as
    /// n, sorted (tests).
    #[cfg(test)]
    fn read_survivors(&self, rec: &BatchRec, second: bool) -> Result<Vec<u128>> {
        let (cnt_h, list, cap) = if second {
            (&rec.list2_count, &self.list2, self.list2_cap)
        } else {
            (&rec.surv_count, &self.surv, self.surv_cap)
        };
        let cnt = self.read_u32(cnt_h)?[0];
        ensure!(cnt <= cap, "survivor list overflow: {cnt} > {cap}");
        let words = self.read_u32(list)?;
        let tp = self.read_u32(&self.top_p)?;
        let mut v: Vec<u128> = (0..cnt as usize)
            .map(|i| {
                let t = words[2 * i] as usize;
                let p = u128::from(tp[2 * t]) | (u128::from(tp[2 * t + 1]) << 32);
                p * u128::from(self.w_f0) + u128::from(words[2 * i + 1])
            })
            .collect();
        v.sort_unstable();
        Ok(v)
    }
}

/// What one field through the join did, for telemetry.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct JoinFieldStats {
    pub setup_secs: f64,
    pub device_secs: f64,
    pub device_wait_secs: f64,
    pub partitions: usize,
    pub batches: u32,
    /// Partitions per launch, and the bytes of the field's buffers.
    pub slots: usize,
    pub bytes: usize,
    /// Pairs that passed the join's AND, and the prefilter's survivors
    /// (what the full check read). Both depend only on the field and the
    /// parameters, not on the device or the layout, unless a wave overflows
    /// the join kernel's shared stage: then the overflow goes to the list
    /// unprefiltered and uncounted. No production field has done so (the
    /// counts agree across layouts and prefilter placements), and the full
    /// check is exact either way.
    pub survivors: u64,
    pub checked: u64,
    pub retried_partitions: usize,
    /// Halvings of a re-run partition's top layer (normally none).
    pub splits: u32,
}

/// Run one prepared field through the join on `client`: every partition
/// value in batches, then every batch's survivor counts are checked and any
/// batch that overflowed its list is re-run a partition at a time, on
/// halves of its top layer if one partition alone overflows (see
/// [`run_partition`]). A field whose top layer is empty has no candidates
/// and needs no device work at all; about a third of random 1e14 fields at
/// bases 57-64 are such.
///
/// # Errors
/// Device failures, or a single top prefix of one partition that overflows
/// the re-run's list (which [`MIN_RETRY_CAP`] rules out).
pub(crate) fn run_field<R: Runtime>(
    client: &ComputeClient<R>,
    field: &JoinField,
) -> Result<(Vec<NiceNumberSimple>, JoinFieldStats)> {
    let fs = &field.fs;
    let mut st = JoinFieldStats {
        setup_secs: fs.secs,
        ..JoinFieldStats::default()
    };
    if fs.tlay.is_empty() {
        return Ok((Vec::new(), st));
    }
    let t0 = Instant::now();
    let parts: Vec<u32> = (0..u32::try_from(fs.nparts)?).collect();
    st.partitions = parts.len();
    st.slots = field.plan.slots;
    st.bytes = field.plan.bytes;
    let mut dev = JoinDevice::new(client, fs, &field.plan)?;
    let mut inflight: VecDeque<LaunchFence> = VecDeque::new();
    let mut recs = Vec::new();
    for batch in parts.chunks(dev.max_slots) {
        while inflight.len() >= BATCHES_IN_FLIGHT {
            let tw = Instant::now();
            if let Some(f) = inflight.pop_front() {
                cubecl::future::block_on(f).map_err(|e| anyhow!("launch fence failed: {e:?}"))?;
            }
            st.device_wait_secs += tw.elapsed().as_secs_f64();
        }
        let rec = dev.launch_batch(batch, false)?;
        if let Some(f) = launch_fence(client)? {
            inflight.push_back(f);
        }
        st.batches += 1;
        recs.push((batch, rec));
    }
    let handles: Vec<Handle> = recs
        .iter()
        .flat_map(|(_, r)| [r.surv_count.clone(), r.list2_count.clone()])
        .collect();
    let counts = cubecl::future::block_on(client.read_async(handles))
        .map_err(|e| anyhow!("read failed: {e:?}"))?;
    let mut retry: Vec<u32> = Vec::new();
    for (i, (batch, rec)) in recs.iter().enumerate() {
        let surv = u32::from_bytes(&counts[2 * i])[0];
        let checked = u32::from_bytes(&counts[2 * i + 1])[0];
        if dev.overflowed(rec, surv, checked) {
            retry.extend_from_slice(batch);
        } else {
            st.survivors += u64::from(surv);
            st.checked += u64::from(dev.checked_of(surv, checked));
        }
    }
    let mut hits = dev.read_hits()?;
    drop(dev);
    if !retry.is_empty() {
        // Survivors past a list's end were dropped unchecked, so the whole
        // batch runs again; hits it already found come back again and are
        // deduplicated below.
        debug!(
            "overlap join b{}: {} partitions overflowed a survivor list, re-running one at a time",
            fs.b,
            retry.len()
        );
        st.retried_partitions = retry.len();
        for &v in &retry {
            run_partition(client, fs, &field.retry, v, 0, &mut st, &mut hits)?;
        }
        hits.sort_unstable();
        hits.dedup();
    }
    st.device_secs = t0.elapsed().as_secs_f64();
    let nice = hits
        .into_iter()
        .map(|number| NiceNumberSimple {
            number,
            num_uniques: fs.b,
        })
        .collect();
    Ok((nice, st))
}

/// Partition `v` of `fs`'s field alone, with `plan` (one slot). If its
/// survivors overflow the list, its top layer is halved and each half runs
/// the partition again: the halves' survivors are exactly the whole's, split,
/// and a single top prefix of one partition always fits the re-run's list
/// (see [`MIN_RETRY_CAP`]).
fn run_partition<R: Runtime>(
    client: &ComputeClient<R>,
    fs: &FieldSetup,
    plan: &JoinPlan,
    v: u32,
    depth: u32,
    st: &mut JoinFieldStats,
    hits: &mut Vec<u128>,
) -> Result<()> {
    let mut dev = JoinDevice::new(client, fs, plan)?;
    let rec = dev.launch_batch(&[v], false)?;
    st.batches += 1;
    let surv = dev.read_u32(&rec.surv_count)?[0];
    let checked = dev.read_u32(&rec.list2_count)?[0];
    if !dev.overflowed(&rec, surv, checked) {
        st.survivors += u64::from(surv);
        st.checked += u64::from(dev.checked_of(surv, checked));
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
        plan.list2_cap.max(plan.surv_cap)
    );
    st.splits += 1;
    let (lo, hi) = fs.tlay.split_at(fs.tlay.len() / 2);
    for half in [lo, hi] {
        run_partition(
            client,
            &fs.with_tops(half.to_vec()),
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

/// The join's side of the niceonly pipeline: a thread that takes fields in
/// order, runs each through [`run_field`] and hands the results back in the
/// same order. Same contract as `NiceonlyPipeline` (push, then
/// `next_result` per field), so the backend can interleave the two.
// Public because it names a field type of the public context enum; its own
// fields stay private, as `NiceonlyPlan`'s do.
pub struct JoinPipeline {
    /// `Option` only so `Drop` can release it before joining the thread.
    tx: Option<SyncSender<JoinJob>>,
    results: Option<Receiver<JoinDone>>,
    outstanding: usize,
    thread: Option<std::thread::JoinHandle<()>>,
    /// The join's own account of the field `next_result` last returned,
    /// for the throughput harness.
    #[cfg(test)]
    last: Option<JoinFieldStats>,
}

impl JoinPipeline {
    pub(crate) fn start<R: Runtime>(client: ComputeClient<R>) -> Self {
        let depth = fields_in_flight() + 1;
        let (tx, jobs) = sync_channel::<JoinJob>(depth);
        let (results_tx, results) = sync_channel::<JoinDone>(depth);
        let thread = std::thread::spawn(move || {
            while let Ok(job) = jobs.recv() {
                let (base, jp) = (job.field.fs.b, job.field.fs.jp);
                let out = run_field(&client, &job.field).map(|(hits, js)| {
                    let stats = NiceonlyStats {
                        msd_secs: js.setup_secs,
                        device_secs: js.device_secs,
                        total_secs: job.pushed_at.elapsed().as_secs_f64(),
                        floor: 0,
                        num_ranges: js.partitions,
                        valid_numbers: js.checked,
                        launches: js.batches,
                        cpu_wait_secs: 0.0,
                        device_wait_secs: js.device_wait_secs,
                        device_busy_secs: None,
                        overlap_join: true,
                    };
                    debug!(
                        "overlap join b{base} {jp:?}: setup {:.3}s, device {:.3}s ({} partitions, \
                         {} batches of {}, {} MiB), {} survivors, {} checked, {} re-run, \
                         {} splits, total {:.3}s",
                        js.setup_secs,
                        js.device_secs,
                        js.partitions,
                        js.batches,
                        js.slots,
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
            outstanding: 0,
            thread: Some(thread),
            #[cfg(test)]
            last: None,
        }
    }

    /// Queue a prepared field. Returns at once unless the queue is full.
    ///
    /// # Errors
    /// The worker thread has exited.
    pub(crate) fn push(&mut self, field: JoinField) -> Result<()> {
        let tx = self
            .tx
            .as_ref()
            .ok_or_else(|| anyhow!("overlap join pipeline is shut down"))?;
        tx.send(JoinJob {
            field,
            pushed_at: Instant::now(),
        })
        .map_err(|_| anyhow!("overlap join worker is gone"))?;
        self.outstanding += 1;
        Ok(())
    }

    /// Wait for the oldest queued field.
    ///
    /// # Errors
    /// The field's own error, or the worker having exited.
    pub(crate) fn next_result(&mut self) -> Result<(NiceonlyStats, Vec<NiceNumberSimple>)> {
        ensure!(
            self.outstanding > 0,
            "no field outstanding in the overlap join"
        );
        self.outstanding -= 1;
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
    use crate::cubecl_backend::{
        CubeclContext, finish_niceonly_cubecl, process_range_niceonly_cubecl,
    };
    use crate::overlap_join::join_range;

    fn client() -> (ComputeClient<cubecl::wgpu::WgpuRuntime>, String) {
        let ctx = CubeclContext::new_default().expect("CubeCL init");
        let name = ctx.device_name();
        #[allow(irrefutable_let_patterns)]
        let CubeclContext::Wgpu { client, .. } = ctx else {
            unreachable!("new_default is wgpu")
        };
        (client, name)
    }

    /// The prefilter by its definition: the digits at positions `0..k2` of
    /// n² and n³ (mod `b^k2` in u128) are distinct, and disjoint from the
    /// top certificate when every certified position is `>= k2`.
    #[allow(clippy::many_single_char_names)] // b, n, r, p, a as in the docs
    fn mid_mirror(fs: &FieldSetup, n: u128) -> bool {
        let b = u128::from(fs.b);
        let bk = b.pow(fs.k2);
        let r = n % bk;
        let sq = r * r % bk;
        let cu = sq * r % bk;
        let p = n / fs.w;
        let a = (p * fs.w).max(fs.s);
        let ee = (p * fs.w + fs.w - 1).min(fs.e - 1);
        let mut seen = 0u64;
        let floor = if a == p * fs.w && ee == p * fs.w + fs.w - 1 {
            fs.full_floor
        } else {
            fs.base.cert_floor(a, ee, fs.jp.k)
        };
        if floor >= fs.k2 {
            seen = fs
                .base
                .cert(a, ee, fs.jp.k)
                .expect("a survivor's top is certified");
        }
        let (mut x2, mut x3) = (sq, cu);
        for _ in 0..fs.k2 {
            for d in [x2 % b, x3 % b] {
                let bit = 1u64 << d;
                if seen & bit != 0 {
                    return false;
                }
                seen |= bit;
            }
            x2 /= b;
            x3 /= b;
        }
        true
    }

    /// Partition values that hold at least one top prefix of the field.
    fn live_partitions(fs: &FieldSetup) -> Vec<u32> {
        let mut v: Vec<u32> = (fs.plo..=fs.phi)
            .map(|p| u32::try_from(p % fs.nparts).expect("b^p < 2^32"))
            .take(usize::try_from(fs.nparts).expect("b^p < 2^32") + 1)
            .collect();
        v.sort_unstable();
        v.dedup();
        v
    }

    /// A layout for `slots` partitions per launch whose lists hold every
    /// survivor of a production batch (a base-57 partition has 1.5-2.7e6
    /// join survivors), since the tests compare whole lists: the device's
    /// binding limit, but a larger budget than production's.
    fn test_plan(fs: &FieldSetup, lim: JoinLimits, slots: usize, fused: bool) -> JoinPlan {
        let lim = JoinLimits {
            max_binding: lim.max_binding,
            budget: 4 << 30,
        };
        JoinPlan::new(fs, lim, slots, fused, 1 << 24, 1, 2 << 30).expect("the test layout fits")
    }

    /// Production-size fields at bases 42-64 (gate-sized at 42, 1e14
    /// elsewhere), from the middle of their bands and the frontier.
    const PLAN_FIELDS: &[(u32, u128, u128)] = &[
        (42, 9_682_651_996_416, 10_000_000_000_000),
        (50, 62_082_117_268_529_817, 100_000_000_000_000),
        (57, FRONTIER_57, 100_000_000_000_000),
        (60, 1_366_405_974_057_412_100_454, 100_000_000_000_000),
        (62, 7_997_740_455_941_656_911_841, 100_000_000_000_000),
        (64, 41_242_006_262_957_161_709_568, 100_000_000_000_000),
    ];

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

    /// GPU survivors (after the AND, and after the prefilter) and hits must
    /// equal the CPU reference, batch by batch. `parts` defaults to the
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
        // Both prefilter placements: unfused, the join's own survivor set is
        // compared too; fused (production), what reaches the full check.
        for fused in [false, true] {
            let plan = test_plan(&fs, JoinLimits::of(client), 4, fused);
            let mut dev = JoinDevice::new(client, &fs, &plan).expect("device");
            let mut cpu_hits = Vec::new();
            for batch in parts.chunks(dev.max_slots) {
                let vs: Vec<u128> = batch.iter().map(|&v| u128::from(v)).collect();
                let mut cpu = Vec::new();
                let st = join_range(&fs.base, s, e, jp, Some(&vs), Some(&mut cpu));
                cpu.sort_unstable();
                cpu_hits.extend(st.hits);
                let rec = dev.launch_batch(batch, false).expect("launch");
                if !rec.fused {
                    let gpu = dev.read_survivors(&rec, false).expect("survivors");
                    assert_eq!(
                        gpu, cpu,
                        "b{b} [{s}, {e}) {jp:?} partitions {batch:?}: survivors differ"
                    );
                }
                if fs.mid > 0 {
                    let want: Vec<u128> = cpu
                        .iter()
                        .copied()
                        .filter(|&n| mid_mirror(&fs, n))
                        .collect();
                    let got = dev.read_survivors(&rec, true).expect("prefilter survivors");
                    assert_eq!(
                        got, want,
                        "b{b} [{s}, {e}) {jp:?} fused={fused} partitions {batch:?}: prefilter differs"
                    );
                }
            }
            cpu_hits.sort_unstable();
            assert_eq!(
                dev.read_hits().unwrap(),
                cpu_hits,
                "b{b} [{s}, {e}) {jp:?} fused={fused}: hits differ"
            );
        }
    }

    /// Layouts far tighter than production's, against the CPU reference over
    /// every partition of a window: the batches' list holds half the densest
    /// partition, so batches re-run their partitions one at a time, and the
    /// re-run's list holds only the densest single top prefix, so the
    /// densest partition's top layer is halved down to prefixes that fit.
    /// Hits, join survivors and prefilter survivors must come out exactly as
    /// the reference has them.
    fn check_tight_layouts<R: Runtime>(client: &ComputeClient<R>, name: &str) {
        let lim = JoinLimits::of(client);
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
                if fs.mid > 0 {
                    rec.retain(|&n| mid_mirror(&fs, n));
                }
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
            let plan = JoinPlan::new(&fs, lim, 2, true, cap(densest / 2), 1, lim.budget)
                .expect("a two-slot layout fits");
            let retry = JoinPlan::new(&fs, lim, 1, true, cap(single), 1, lim.budget).expect("fits");
            let field = JoinField::with_plans(fs, plan, retry);
            let (nice, st) = run_field(client, &field).expect("tight run");
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

    /// Dan Stoyell's exactness windows: several shapes, aligned and
    /// unaligned, some where the client's MSD filter lets candidates through.
    const WINDOWS: &[(u32, u128, u128, u32, u32, u32)] = &[
        (20, 58_945, 160_000, 3, 2, 0),
        (20, 58_945, 160_000, 2, 3, 1),
        (20, 60_001, 150_003, 2, 3, 1),
        (25, 3_339_797, 4_339_797, 4, 3, 1),
        (25, 5_000_123, 5_700_456, 3, 4, 2),
        (30, 300_000_000, 301_000_000, 5, 4, 2),
        (34, 12_000_000_017, 12_001_000_017, 5, 5, 2),
        (40, 3_000_000_000_000, 3_000_002_000_000, 6, 5, 2),
        (40, 3_000_000_000_000, 3_000_002_000_000, 7, 4, 2),
        (
            57,
            30_000_000_000_000_000_000,
            30_000_000_000_000_300_000,
            10,
            4,
            1,
        ),
        (
            57,
            20_635_899_893_042_801_193,
            20_635_899_893_043_101_193,
            9,
            5,
            1,
        ),
        (
            57,
            78_920_310_198_429_586_458,
            78_920_310_198_429_886_458,
            9,
            5,
            2,
        ),
        (
            58,
            114_041_927_169_846_138_720,
            114_041_927_169_846_438_720,
            10,
            4,
            2,
        ),
        (
            60,
            1_573_714_731_429_953_349_518,
            1_573_714_731_429_953_649_518,
            10,
            4,
            2,
        ),
        (
            64,
            52_125_117_810_081_128_433_988,
            52_125_117_810_081_128_733_988,
            11,
            4,
            2,
        ),
    ];

    /// A frontier field of base 57, with the parameters the gate picks.
    const FRONTIER_57: u128 = 28_151_599_893_042_801_193;

    fn check_all<R: Runtime>(client: &ComputeClient<R>, name: &str) {
        let jp = JoinParams { t: 2, k: 1, p: 0 };
        let band = FieldSize::new(47, 100);
        let field = JoinField::prepare(10, &band, jp, JoinLimits::of(client))
            .expect("field setup")
            .expect("base 10 fits any device");
        let (hits, _) = run_field(client, &field).expect("join run");
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

    /// The backend interleaves the two pipelines and still returns fields in
    /// the order they were begun: base 10's band through the join (forced
    /// parameters), then through the stride pipeline, then the join again.
    #[test]
    #[ignore = "requires a wgpu device"]
    fn cubecl_niceonly_interleaves_join_and_stride_in_order() {
        use crate::cubecl_backend::begin_routed;
        use crate::gpu_niceonly::NiceonlyStarted;
        let ctx = CubeclContext::new_default().expect("CubeCL init");
        let band = FieldSize::new(47, 100);
        let jp = JoinParams { t: 2, k: 1, p: 0 };
        let routes = [Some(jp), None, Some(jp)];
        for &r in &routes {
            let field = r.map(|jp| {
                JoinField::prepare(10, &band, jp, ctx.join_limits())
                    .unwrap()
                    .expect("base 10 fits any device")
            });
            assert!(matches!(
                begin_routed(&ctx, &band, 10, field).unwrap(),
                NiceonlyStarted::Queued
            ));
        }
        for &r in &routes {
            let (res, stats) = finish_niceonly_cubecl(&ctx).unwrap();
            assert_eq!(
                stats.overlap_join,
                r.is_some(),
                "fields came back out of order"
            );
            assert_eq!(
                res.nice_numbers
                    .iter()
                    .map(|n| n.number)
                    .collect::<Vec<_>>(),
                vec![69]
            );
        }
        // And the public entry point leaves a band this small to the stride
        // pipeline.
        let r = process_range_niceonly_cubecl(&ctx, &band, 10).unwrap();
        assert_eq!(
            r.nice_numbers.iter().map(|n| n.number).collect::<Vec<_>>(),
            vec![69]
        );
    }

    /// The client's route on one field: `plan_join` must take it, and the
    /// join's results and own account come back.
    fn run_routed(
        ctx: &CubeclContext,
        base: u32,
        range: &FieldSize,
    ) -> (Vec<u128>, NiceonlyStats, JoinFieldStats) {
        use crate::cubecl_backend::{begin_join, last_join_stats, plan_join};
        let field = plan_join(ctx, base, range).expect("the join takes the field");
        begin_join(ctx, field).expect("begin");
        let (res, stats) = finish_niceonly_cubecl(ctx).expect("finish");
        let js = last_join_stats(ctx).expect("join stats");
        (
            res.nice_numbers.iter().map(|n| n.number).collect(),
            stats,
            js,
        )
    }

    /// The stride pipeline on the same field (the client's path before the
    /// join).
    fn run_stride(ctx: &CubeclContext, base: u32, range: &FieldSize) -> Vec<u128> {
        use crate::cubecl_backend::begin_routed;
        begin_routed(ctx, range, base, None).expect("begin");
        let (res, _) = finish_niceonly_cubecl(ctx).expect("finish");
        let mut v: Vec<u128> = res.nice_numbers.iter().map(|n| n.number).collect();
        v.sort_unstable();
        v
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
        let threads = std::thread::available_parallelism().map_or(4, usize::from);
        let parts: Vec<u128> = (0..fs.nparts).collect();
        let (survivors, checked, mut hits) = std::thread::scope(|sc| {
            let fs = &fs;
            let workers: Vec<_> = parts
                .chunks(parts.len().div_ceil(threads))
                .map(|chunk| {
                    sc.spawn(move || {
                        // One partition at a time: a whole field's survivors
                        // (4.5e8 at base 42) would not fit in memory.
                        let (mut survivors, mut kept, mut hits) = (0u64, 0u64, Vec::new());
                        let mut rec = Vec::new();
                        for &v in chunk {
                            rec.clear();
                            let st =
                                join_range(&fs.base, fs.s, fs.e, fs.jp, Some(&[v]), Some(&mut rec));
                            survivors += st.survivors;
                            hits.extend(st.hits);
                            kept += if fs.mid > 0 {
                                rec.iter().filter(|&&n| mid_mirror(fs, n)).count() as u64
                            } else {
                                rec.len() as u64
                            };
                        }
                        (survivors, kept, hits)
                    })
                })
                .collect();
            workers
                .into_iter()
                .map(|w| w.join().expect("reference worker"))
                .fold((0u64, 0u64, Vec::new()), |(a, b, mut h), (x, y, z)| {
                    h.extend(z);
                    (a + x, b + y, h)
                })
        });
        hits.sort_unstable();
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
             survivors, {checked} checked, hits {hits:?}; reference {cpu_secs:.0}s on {threads} \
             threads, join {:.2}s, stride {:.2}s",
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
        (60, 1_573_714_731_429_953_349_518),
        (64, 41_242_006_262_957_161_709_568),
    ];

    /// Throughput of production fields through the client's route
    /// (`plan_join`, then the join's pipeline with `fields_in_flight()`
    /// fields queued, as the client runs them): `NICE_TEST_JOIN_FIELDS`
    /// fields per base (also the opt-in, since the parity workflow runs this
    /// module's ignored tests on lavapipe) after one untimed warm-up field,
    /// for the bases in `NICE_TEST_JOIN_BASES` (default 57,60,64). Prints per
    /// base the rate, per-field wall, setup and device seconds, the layout
    /// (partitions per launch, MiB), re-runs, survivor counts and the
    /// device's memory. `NICE_TEST_JOIN_COMPARE=1` also runs every field
    /// through the stride pipeline and asserts the same nice numbers.
    #[allow(clippy::cast_precision_loss, clippy::too_many_lines)]
    fn join_throughput(ctx: &CubeclContext) {
        use crate::cubecl_backend::{begin_join, last_join_stats, memory_usage, plan_join};
        use crate::gpu_niceonly::fields_in_flight;
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
        let size = 100_000_000_000_000u128;
        let mib = |b: u64| b >> 20;
        for base in bases {
            let &(_, start) = THROUGHPUT_STARTS
                .iter()
                .find(|&&(b, _)| b == base)
                .expect("a base with throughput fields (57, 60, 64)");
            let warm = FieldSize::new(start - size, start);
            run_routed(ctx, base, &warm);
            let fields: Vec<FieldSize> = (0..n)
                .map(|i| FieldSize::new(start + i * size, start + (i + 1) * size))
                .collect();
            let lookahead = fields_in_flight().saturating_sub(1);
            let t = std::time::Instant::now();
            let (mut queued, mut found) = (0usize, Vec::new());
            let mut per_field: Vec<(NiceonlyStats, JoinFieldStats)> = Vec::new();
            // Device memory, sampled while the next field is on the device.
            let mut peak = (0u64, 0u64);
            let mut finish = |found: &mut Vec<u128>| {
                if let Some((in_use, reserved)) = memory_usage(ctx) {
                    peak = (peak.0.max(in_use), peak.1.max(reserved));
                }
                let (res, stats) = finish_niceonly_cubecl(ctx).expect("finish");
                found.extend(res.nice_numbers.iter().map(|x| x.number));
                per_field.push((stats, last_join_stats(ctx).expect("join stats")));
            };
            for f in &fields {
                let field = plan_join(ctx, base, f).expect("the join takes the field");
                begin_join(ctx, field).expect("begin");
                queued += 1;
                while queued > lookahead {
                    finish(&mut found);
                    queued -= 1;
                }
            }
            while queued > 0 {
                finish(&mut found);
                queued -= 1;
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
                fields_in_flight(),
                secs / nf,
                (n * size) as f64 / secs,
                mean(&|x| x.0.msd_secs),
                mean(&|x| x.0.device_secs),
                mean(&|x| x.0.total_secs),
                mean(&|x| f64::from(x.0.launches)),
                per_field.iter().map(|x| x.1.slots).max().unwrap_or(0),
                per_field.iter().map(|x| x.1.bytes >> 20).max().unwrap_or(0),
                per_field.iter().filter(|x| x.1.partitions == 0).count(),
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
}
