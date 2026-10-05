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

use crate::NiceNumberSimple;
use crate::cubecl_backend::{LaunchFence, NICEONLY_STRIDE, launch_fence, wide_chunk_for};
use crate::gpu_config::{chunk_constants, chunk_constants_u16, n_limbs};
use crate::gpu_niceonly::{NiceonlyStats, fields_in_flight};
use crate::gpu_route::{FieldTicket, Route};
use crate::join_plan::{
    BATCHES_IN_FLIGHT, Footprint, JoinField, JoinLimits, JoinPlan, NICE_RECORD_BYTES,
};
use crate::overlap_join::{FieldSetup, JoinTelemetry};
use anyhow::{Result, anyhow, ensure};
use cubecl::prelude::*;
use cubecl::server::Handle;
use log::debug;
use std::collections::VecDeque;
use std::sync::mpsc::{Receiver, SyncSender, sync_channel};
use web_time::Instant;

// The output list's layout is shared with the stride kernels' check.
const _: () = assert!(NICEONLY_STRIDE as usize * 4 == NICE_RECORD_BYTES);

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
/// Cubes for the prefilter and check kernels (grid-stride loops).
const CHECK_CUBES: u32 = 1024;
/// How deep the halving of an overflowing partition's top layer may go: a
/// layer of up to 2^32 prefixes reaches single prefixes well within it.
const MAX_SPLITS: u32 = 32;

// ---------------------------------------------------------------------------
// Device kernels
// ---------------------------------------------------------------------------

pub use kernels::{
    bucket_kernel, check_kernel, join_kernel, top_fill_kernel, top_kernel, top_scan_kernel,
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
    /// thread runs every wave (uniform), survivors staged and flushed through
    /// the prefilter whenever the stage is half full, checked after every 32
    /// tops.
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
        #[comptime] f0: u32,
        #[comptime] k2: u32,
        #[comptime] walk_sync: bool,
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
                                if walk_sync {
                                    sync_plane();
                                }
                            }
                        }
                        // Flush the stage once it is half full, after every
                        // chunk of 32 tops. Once a wave pairs its 1,024
                        // entries with a full 256 tops (buckets that dense
                        // come with fields of about 1e15 and up), checking
                        // only at the wave's end let the stage overrun, and
                        // its overflow went to the list unfiltered: 11x the
                        // full checks on a base-58 field of 1e15, and lists
                        // that overflowed. A chunk can still overrun the
                        // half left (about 2e-5 of the survivors there, and
                        // no more on larger fields, since a chunk is the same
                        // size); that overflow takes the same unfiltered
                        // path, which costs only its full checks.
                        sync_cube();
                        let sc = s_cnt[0].load();
                        if sc >= SURV_FLUSH {
                            flush_survivors(
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
                            sync_cube();
                        }
                        jc += 32u32;
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
            flush_survivors(
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

    /// Flush the shared survivor stage through the middle-digit prefilter:
    /// the whole cube tests the staged survivors and only those that pass go
    /// to `out`, compacted per plane with one atomic per round. `staged`
    /// counts every survivor staged (statistics). Must be called by the whole
    /// cube at a point where no thread is still appending (after a barrier);
    /// `sc` is the stage count read after that barrier.
    #[cube]
    fn flush_survivors(
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

/// What `client`'s device allows the join: `CubeCL` reports its largest
/// buffer as `max_page_size`.
pub(crate) fn limits_of<R: Runtime>(client: &ComputeClient<R>) -> JoinLimits {
    JoinLimits::for_buffer(client.properties().memory.max_page_size)
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
    key_level: bool,
    /// End each round of the join kernel's set-bit walk with a plane
    /// barrier (CUDA only, see the walk).
    walk_sync: bool,
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

/// What a launched batch leaves behind to be read later: its two counts,
/// the pairs that passed the join's AND and the prefilter's survivors.
pub(crate) struct BatchRec {
    pub survivors: Handle,
    pub checked: Handle,
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
        let (list_cap, nice_cap) = (plan.list_cap, plan.nice_cap);

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
        // The join kernel counts its survivors and writes only those that
        // pass the prefilter to the list, which it counts too.
        let survivors = c.create(cubecl::bytes::Bytes::from_elems(vec![0u32; 1]));
        let checked = c.create(cubecl::bytes::Bytes::from_elems(vec![0u32; 1]));
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
                ArrayArg::from_raw_parts(self.list.clone(), 2 * self.list_cap as usize),
                ArrayArg::from_raw_parts(checked.clone(), 1),
                ArrayArg::from_raw_parts(self.top_x.clone(), 2 * max_slots * ntl),
                ArrayArg::from_raw_parts(survivors.clone(), 1),
                nwork_u32,
                self.nbp,
                self.list_cap,
                self.b,
                self.key_level,
                self.key_level && self.k == 1,
                ENTRIES_PER_THREAD,
                self.f0,
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
                ArrayArg::from_raw_parts(self.top_p.clone(), 2 * max_slots * ntl),
                ArrayArg::from_raw_parts(
                    self.nice_out.clone(),
                    self.nice_cap as usize * NICEONLY_STRIDE as usize,
                ),
                ArrayArg::from_raw_parts(self.nice_count.clone(), 1),
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
        Ok(BatchRec { survivors, checked })
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
    /// Pairs that passed the join's AND, and the prefilter's survivors
    /// (what the full check read). Both depend only on the field and the
    /// parameters, not on the device or the layout, unless a wave overflows
    /// the join kernel's shared stage: then the overflow goes to the list
    /// unprefiltered and uncounted. No production field has done so (the
    /// counts matched a separate prefilter pass on ten production fields),
    /// and the full check is exact either way.
    pub survivors: u64,
    pub checked: u64,
    /// Partitions of batches that overflowed their list and ran again, in a
    /// batch half the size (each time they did).
    pub retried_partitions: usize,
    /// Halvings of a re-run partition's top layer (normally none).
    pub splits: u32,
}

/// Run one prepared field through the join on `client`: slice by slice
/// (each later slice's setup beside the device's work on the one before),
/// every partition of each in batches ([`run_slice`]). A field whose top
/// layer is empty has no candidates and needs no device work at all; about
/// a third of random 1e14 fields at bases 57-64 are such.
///
/// # Errors
/// Device failures, or a single top prefix of one partition that overflows
/// the re-run's list (which `join_plan::MIN_RETRY_CAP` rules out).
pub(crate) fn run_field<R: Runtime>(
    client: &ComputeClient<R>,
    field: &JoinField,
) -> Result<(Vec<NiceNumberSimple>, JoinFieldStats)> {
    let parts: Vec<u32> = (0..u32::try_from(field.fs.nparts)?).collect();
    run_slices(client, field, field.slices.len(), &parts)
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
    let mut next: Option<Result<FieldSetup>> = None;
    for i in 0..count.min(field.slices.len()) {
        let owned;
        let fs = if i == 0 {
            &field.fs
        } else {
            owned = next.take().expect("prepared beside the previous slice")?;
            &owned
        };
        st.slices += 1;
        st.setup_secs += fs.secs;
        let (plan, retry) = if i == 0 {
            (field.plan, field.retry)
        } else {
            JoinPlan::for_field(fs, field.lim)
                .ok_or_else(|| anyhow!("slice {i} of the field does not fit the device"))?
        };
        let later = field.slices.get(i + 1).filter(|_| i + 1 < count);
        std::thread::scope(|scope| -> Result<()> {
            let prep = later.map(|sl| scope.spawn(|| field.fs.sub_range(sl.start(), sl.end())));
            let used = run_slice(
                client, fs, &plan, &retry, slots_hint, parts, &mut st, &mut hits,
            )?;
            slots_hint = used;
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

/// One slice, on the partitions `parts`: batches of at most `slots_hint`
/// partitions (and of the plan's), all launched before any is read. The
/// batches whose survivors overflowed their list go again in batches half
/// the size, until one partition is left; one partition that still
/// overflows takes the re-run layout ([`run_partition`]). Returns the batch
/// size the last pass used, for the next slice to start from.
#[allow(clippy::too_many_arguments)]
fn run_slice<R: Runtime>(
    client: &ComputeClient<R>,
    fs: &FieldSetup,
    plan: &JoinPlan,
    retry: &JoinPlan,
    slots_hint: usize,
    parts: &[u32],
    st: &mut JoinFieldStats,
    hits: &mut Vec<u128>,
) -> Result<usize> {
    if fs.tlay.is_empty() || parts.is_empty() {
        return Ok(slots_hint);
    }
    let t0 = Instant::now();
    let mut dev = JoinDevice::new(client, fs, plan)?;
    let mut slots = slots_hint.clamp(1, dev.max_slots);
    let mut todo = parts.to_vec();
    loop {
        st.min_slots = st.min_slots.min(slots);
        let overflowed = run_pass(client, &mut dev, &todo, slots, st)?;
        if overflowed.is_empty() {
            break;
        }
        debug!(
            "overlap join b{}: {} partitions overflowed a survivor list in batches of {slots}",
            fs.b,
            overflowed.len()
        );
        st.retried_partitions += overflowed.len();
        if slots == 1 {
            // Survivors past a list's end were dropped unchecked, so these
            // run again; hits they already found come back again and are
            // deduplicated by the caller.
            hits.extend(dev.read_hits()?);
            drop(dev);
            for &v in &overflowed {
                run_partition(client, fs, retry, v, 0, st, hits)?;
            }
            st.device_secs += t0.elapsed().as_secs_f64();
            return Ok(1);
        }
        slots = (slots / 2).max(1);
        todo = overflowed;
    }
    hits.extend(dev.read_hits()?);
    st.device_secs += t0.elapsed().as_secs_f64();
    Ok(slots)
}

/// One pass over `parts` in batches of `slots`, pipelined: every batch is
/// launched before any count is read. Adds the batches that fit their list
/// to `st` and returns the partitions of those that did not.
fn run_pass<R: Runtime>(
    client: &ComputeClient<R>,
    dev: &mut JoinDevice<R>,
    parts: &[u32],
    slots: usize,
    st: &mut JoinFieldStats,
) -> Result<Vec<u32>> {
    let mut inflight: VecDeque<LaunchFence> = VecDeque::new();
    let mut recs = Vec::new();
    for batch in parts.chunks(slots) {
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
        .flat_map(|(_, r)| [r.survivors.clone(), r.checked.clone()])
        .collect();
    let counts = cubecl::future::block_on(client.read_async(handles))
        .map_err(|e| anyhow!("read failed: {e:?}"))?;
    let mut overflowed = Vec::new();
    for (i, (batch, _)) in recs.iter().enumerate() {
        let surv = u32::from_bytes(&counts[2 * i])[0];
        let checked = u32::from_bytes(&counts[2 * i + 1])[0];
        if dev.overflowed(checked) {
            overflowed.extend_from_slice(batch);
        } else {
            st.survivors += u64::from(surv);
            st.checked += u64::from(checked);
        }
    }
    Ok(overflowed)
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
    let mut dev = JoinDevice::new(client, fs, plan)?;
    let rec = dev.launch_batch(&[v], false)?;
    st.batches += 1;
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

    fn client() -> (ComputeClient<cubecl::wgpu::WgpuRuntime>, String) {
        let ctx = CubeclContext::new_default().expect("CubeCL init");
        let name = ctx.device_name();
        #[allow(irrefutable_let_patterns)]
        let CubeclContext::Wgpu { client, .. } = ctx else {
            unreachable!("new_default is wgpu")
        };
        (client, name)
    }

    /// A layout for `slots` partitions per launch whose lists hold every
    /// survivor of a production batch (a base-57 partition has 1.5-2.7e6
    /// join survivors), since the tests compare whole lists: the device's
    /// binding limit, but a larger budget than production's.
    fn test_plan(fs: &FieldSetup, lim: JoinLimits, slots: usize) -> JoinPlan {
        let lim = JoinLimits {
            max_binding: lim.max_binding,
            budget: 4 << 30,
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
        let mut dev = JoinDevice::new(client, &fs, &plan).expect("device");
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
        let lim = limits_of(client);
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

    fn check_all<R: Runtime>(client: &ComputeClient<R>, name: &str) {
        let jp = JoinParams { t: 2, k: 1, p: 0 };
        let band = FieldSize::new(47, 100);
        let field = JoinField::prepare(10, &band, jp, limits_of(client))
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
        check_slices(client, name);
    }

    /// A field run slice by slice (here cut into about three) finds what it
    /// finds whole: the same hits, join survivors and prefilter survivors.
    fn check_slices<R: Runtime>(client: &ComputeClient<R>, name: &str) {
        let lim = limits_of(client);
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
            let (all, st_all) = run_field(client, &whole).expect("whole run");
            let field = whole.resliced(prefixes.div_ceil(3)).expect("slices");
            let (got, st) = run_field(client, &field).expect("sliced run");
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
    /// gets the reason instead.
    #[test]
    #[ignore = "requires a wgpu device"]
    fn join_sample_runs_partitions_of_a_production_field() {
        let ctx = CubeclContext::new_default().expect("CubeCL init");
        let range = FieldSize::new(FRONTIER_57, FRONTIER_57 + 100_000_000_000_000);
        let parts = [1_000u32];
        let sample = ctx
            .join_sample(57, &range, &parts)
            .expect("device run")
            .expect("a join field");
        assert_eq!(
            (sample.slices, sample.partitions, sample.sampled),
            (1, 3_249, 1)
        );
        let cpu = crate::cpu_join::CpuJoin::for_field(57, &range).expect("a join field");
        let want = cpu.run_partition(1_000, &mut crate::cpu_join::Scratch::default());
        assert_eq!(
            (sample.survivors, sample.checked),
            (want.survivors, want.checked)
        );
        assert_eq!(sample.hits, want.hits);
        assert!(sample.setup_secs > 0.0 && sample.run_secs > 0.0);
        let small = FieldSize::new(FRONTIER_57, FRONTIER_57 + 4_000_000_000);
        assert_eq!(
            ctx.join_sample(57, &small, &parts)
                .expect("no device work")
                .err(),
            Some(crate::overlap_join::StrideReason::BelowMinSize)
        );
    }

    /// A field with dense buckets (base 58, 1e15), on `NICE_TEST_JOIN_DENSE`
    /// partitions spread over it: the hits must equal the CPU join's, and
    /// the survivor counts nearly so. A chunk that overruns the stage's
    /// headroom still sends its overflow to the list unfiltered (and
    /// uncounted as a join survivor); with the stage flushed per chunk that
    /// is about 2e-5 of the survivors here (7.6k of 323M on 32 partitions),
    /// where flushing per wave sent two thirds (11x the full checks).
    #[test]
    #[cfg(feature = "cubecl-cuda")]
    #[ignore = "requires an NVIDIA device; opt-in, NICE_TEST_JOIN_DENSE=partitions"]
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
        let dense = crate::join_plan::tests::DENSE_58;
        let range = FieldSize::new(dense, dense + 1_000_000_000_000_000);
        let field =
            crate::join_plan::plan_join(58, &range, limits_of(&client)).expect("a join field");
        let total = u32::try_from(field.fs.nparts).unwrap();
        let parts: Vec<u32> = (0..n).map(|i| i * total / n.max(1)).collect();
        let t0 = Instant::now();
        let (hits, st) = run_slices(&client, &field, 1, &parts).expect("device run");
        let gpu_secs = t0.elapsed().as_secs_f64();
        let cpu =
            crate::cpu_join::CpuJoin::new(58, &field.slices[0], field.fs.jp).expect("cpu setup");
        let mut sc = crate::cpu_join::Scratch::default();
        let mut want = crate::cpu_join::PartitionResult::default();
        for &v in &parts {
            want.add(cpu.run_partition(v, &mut sc));
        }
        want.hits.sort_unstable();
        let got: Vec<u128> = hits.iter().map(|h| h.number).collect();
        println!(
            "b58 dense 1e15, {n} partitions: {} slice(s), {} per launch (fewest {}), {} re-run; \
             survivors {} checked {} on the device in {gpu_secs:.2}s",
            field.slices.len(),
            st.slots,
            st.min_slots,
            st.retried_partitions,
            st.survivors,
            st.checked
        );
        assert_eq!(got, want.hits, "hits differ");
        // Each survivor that overran the stage is missing from the join's
        // count and went to the list unfiltered: at most one more checked.
        assert!(
            st.survivors <= want.survivors,
            "more join survivors than the CPU join"
        );
        let spilled = want.survivors - st.survivors;
        #[allow(clippy::cast_precision_loss)]
        let share = spilled as f64 / want.survivors.max(1) as f64;
        println!("  {spilled} survivors overran the stage ({share:.1e} of them)");
        assert!(
            share <= 1e-4,
            "{spilled} of {} survivors overran the stage",
            want.survivors
        );
        assert!(
            st.checked >= want.checked && st.checked - want.checked <= spilled,
            "prefilter survivors differ beyond the overrun"
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
        use crate::cubecl_backend::memory_usage;
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
