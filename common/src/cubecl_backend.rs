//! `CubeCL` detailed-mode backend: the `vulkan/codegen.rs` detailed kernel
//! expressed as a `#[cube]` Rust function instead of generated WGSL.
//!
//! This is a benchmark-grade port for evaluating `CubeCL` as a replacement for
//! the string-codegen approach (see PR #96 discussion). It mirrors the Vulkan
//! backend's structure exactly: same per-base constants from
//! [`crate::gpu_config`], same split16 chunk scan (all divisions 32-bit with
//! comptime-constant divisors), same 4-copy workgroup histogram, same
//! `MISS_STRIDE` near-miss records, same `FieldResults` out the other end.
//! What used to be per-base source generation is `#[comptime]` parameters:
//! `CubeCL` JIT-specializes one Rust function per base at first launch.
//!
//! Niceonly drives the shared [`crate::gpu_niceonly`] pipeline like the other
//! backends: the CPU's MSD prefix filter streams range descriptors in, and
//! [`niceonly_kernel`] reconstructs the stride filter's candidates on-device
//! from the residue table — the same reconstruction, offset reduction, lane
//! tiling, and low-digit prefilter as the Vulkan shader, as Rust.

#![cfg(feature = "cubecl")]
// The cube macro evaluates comptime! expressions host-side, where fn-level
// allows do not reach; these lints are all macro-expansion artifacts.
#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_lossless,
    clippy::used_underscore_binding
)]

use crate::client_process::{process_range_detailed, process_range_niceonly};
use crate::gpu_config::{
    VulkanPrefilterParams, chunk_constants_u16, gpu_supports_base, n_limbs, vulkan_prefilter_params,
};
use crate::gpu_niceonly::{
    DeviceResult, GPU_LSD_K, MAX_STRIDE_MODULUS, NiceonlyPipeline, NiceonlyStarted, NiceonlyStats,
    PendingField, RangeSink, batches_in_flight, lane_shift_for, residue_empty_result,
    stride_chunk_bits,
};
use crate::number_stats::get_near_miss_cutoff;
use crate::stride_filter::StrideTable;
use crate::{FieldResults, FieldSize, NiceNumberSimple, UniquesDistributionSimple};
use anyhow::{Context as _, Result, ensure};
use cubecl::prelude::*;
use log::{debug, warn};
use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};
use web_time::Instant;

/// Threads per cube (workgroup); matches the Vulkan backend.
pub const WORKGROUP_SIZE: u32 = 256;

/// Upper bound on cubes per dispatch; threads grid-stride past it.
/// Matches the Vulkan backend's `MAX_WORKGROUPS`.
pub const MAX_CUBES: u32 = 4096;

/// Workgroup histogram copies, spreading atomic contention like the CUDA
/// kernel's per-warp histograms and the Vulkan backend's `HIST_COPIES`.
pub const HIST_COPIES: u32 = 4;

/// u32 slots per near-miss record: n as four u32 limbs plus its unique count.
/// Must match the Vulkan backend's `MISS_STRIDE` decode.
pub const MISS_STRIDE: u32 = 5;

/// Candidates per dispatch; same bound and rationale as `VULKAN_BATCH_SIZE`
/// (a u32 histogram bin must not overflow before the host drains it).
pub const CUBECL_BATCH_SIZE: u128 = 50_000_000;

/// Near-miss records held on the device per field.
const NEAR_MISS_CAPACITY: usize = 1 << 20;

/// The detailed kernel: for each candidate `n` in `[start, start+count)`,
/// count the unique base-b digits of n² and n³ together, histogram the
/// count, and record near-misses above the cutoff.
///
/// Comptime parameters make `CubeCL` specialize one kernel per base, exactly
/// as the WGSL generator bakes literals: every division below is by a
/// comptime constant, and every limb loop unrolls to straight-line code.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
fn detailed_kernel(
    hist: &mut Array<Atomic<u32>>,
    miss_count: &mut Array<Atomic<u32>>,
    miss_data: &mut Array<u32>,
    s0: u32,
    s1: u32,
    s2: u32,
    s3: u32,
    cnt_lo: u32,
    cnt_hi: u32,
    miss_cap: u32,
    #[comptime] base: u32,
    #[comptime] limbs: u32,
    #[comptime] chunk_digits: u32,
    #[comptime] chunk_div: u32,
    #[comptime] cutoff: u32,
    #[comptime] wide_chunk: bool,
) {
    let hist_bins = comptime!(base + 1);
    let sq_limbs = comptime!(2 * limbs);
    let cu_limbs = comptime!(3 * limbs);
    let two_masks = comptime!(base > 64);

    let hist_s = SharedMemory::<Atomic<u32>>::new(comptime!((HIST_COPIES * (base + 1)) as usize));

    // Zero the workgroup histogram.
    let mut zi = UNIT_POS_X;
    while zi < HIST_COPIES * hist_bins {
        hist_s[zi as usize].store(0u32);
        zi += CUBE_DIM_X;
    }
    sync_cube();

    let start_lo = (u64::cast_from(s1) << 32u64) | u64::cast_from(s0);
    let start_hi = (u64::cast_from(s3) << 32u64) | u64::cast_from(s2);
    let count = (u64::cast_from(cnt_hi) << 32u64) | u64::cast_from(cnt_lo);
    let copy = (UNIT_POS_X >> 5u32) % HIST_COPIES;
    let stride = u64::cast_from(CUBE_COUNT_X) * u64::cast_from(CUBE_DIM_X);

    // Scratch for the digit scan, in shared memory: the split loop indexes
    // it at runtime, and a register-promoted array makes every such access a
    // compare/select chain over all words (~100 extra SASS compares at b50),
    // while local memory costs a round trip. One shared word per thread per
    // limb, padded to an odd stride so a warp's accesses spread across banks.
    let sv_pad = comptime!(cu_limbs | 1);
    let mut sv_s = SharedMemory::<u32>::new(comptime!((WORKGROUP_SIZE * (cu_limbs | 1)) as usize));
    let svb = UNIT_POS_X * sv_pad;

    let mut idx = u64::cast_from(ABSOLUTE_POS_X);
    while idx < count {
        // n = start + idx over u64 halves (u128 add with carry).
        let n_lo = start_lo + idx;
        let mut n_hi = start_hi;
        if n_lo < start_lo {
            n_hi += 1u64;
        }

        // --- num_unique(n_lo, n_hi), inlined ---------------------------------
        // Unpack n into u32 limbs held in a local array; limb loops below
        // unroll over comptime bounds, so indices are constants.
        let mut nl = Array::<u32>::new(limbs as usize);
        #[unroll]
        for i in 0..limbs {
            let shift = comptime!(((i & 1) * 32) as u64);
            if comptime!(i < 2) {
                nl[i as usize] = u32::cast_from(n_lo >> shift);
            } else {
                nl[i as usize] = u32::cast_from(n_hi >> shift);
            }
        }

        // sq = n * n (schoolbook, u64 accumulation).
        let mut sq = Array::<u32>::new(sq_limbs as usize);
        #[unroll]
        for i in 0..sq_limbs {
            sq[i as usize] = 0u32;
        }
        #[unroll]
        for i in 0..limbs {
            let mut carry = 0u64;
            #[unroll]
            for j in 0..limbs {
                let k = comptime!(i + j);
                let t = u64::cast_from(nl[i as usize]) * u64::cast_from(nl[j as usize])
                    + u64::cast_from(sq[k as usize])
                    + carry;
                sq[k as usize] = u32::cast_from(t);
                carry = t >> 32u64;
            }
            sq[comptime!(i + limbs) as usize] = u32::cast_from(carry);
        }

        // cu = sq * n.
        let mut cu = Array::<u32>::new(cu_limbs as usize);
        #[unroll]
        for i in 0..cu_limbs {
            cu[i as usize] = 0u32;
        }
        #[unroll]
        for i in 0..sq_limbs {
            let mut carry = 0u64;
            #[unroll]
            for j in 0..limbs {
                let k = comptime!(i + j);
                let t = u64::cast_from(sq[i as usize]) * u64::cast_from(nl[j as usize])
                    + u64::cast_from(cu[k as usize])
                    + carry;
                cu[k as usize] = u32::cast_from(t);
                carry = t >> 32u64;
            }
            cu[comptime!(i + limbs) as usize] = u32::cast_from(carry);
        }

        // Digit masks.
        let mut m0 = 0u64;
        let mut m1 = 0u64;

        // Two scans: sq then cu. `pass` selects the source.
        let mut pass: u32 = 0u32;
        while pass < 2u32 {
            if pass == 0u32 {
                #[unroll]
                for i in 0..sq_limbs {
                    sv_s[(svb + u32::cast_from(i)) as usize] = sq[i as usize];
                }
                #[unroll]
                for i in sq_limbs..cu_limbs {
                    sv_s[(svb + u32::cast_from(i)) as usize] = 0u32;
                }
            } else {
                #[unroll]
                for i in 0..cu_limbs {
                    sv_s[(svb + u32::cast_from(i)) as usize] = cu[i as usize];
                }
            }

            // top_limb
            let mut top: i32 = comptime!(cu_limbs as i32 - 1).runtime();
            while top >= 0 {
                if sv_s[(svb + u32::cast_from(top)) as usize] != 0u32 {
                    break;
                }
                top -= 1;
            }

            // Chunked radix scan, destroying sv. Two flavors, chosen at
            // comptime per runtime: `wide_chunk` divides limb-pairs by a
            // sub-2^31 constant with one u64 division per limb (nvcc
            // strength-reduces it; 5 digits/chunk at base 40), while the
            // split16 flavor uses two u32 divisions over 16-bit halves
            // (for drivers that cannot strength-reduce u64, 3 digits/chunk).
            while top >= 0 {
                let mut rem = 0u32;
                if wide_chunk {
                    let mut rem64 = 0u64;
                    let mut i: i32 = top;
                    while i >= 0 {
                        let cur = (rem64 << 32u64)
                            | u64::cast_from(sv_s[(svb + u32::cast_from(i)) as usize]);
                        // Quotient once, remainder by mul-sub — see the
                        // niceonly scan; `%` would cost a second multiply-high
                        // sequence per word.
                        let q = cur / u64::cast_from(chunk_div);
                        sv_s[(svb + u32::cast_from(i)) as usize] = u32::cast_from(q);
                        rem64 = cur - q * u64::cast_from(chunk_div);
                        i -= 1;
                    }
                    rem = u32::cast_from(rem64); // rem < chunk_div < 2^31
                } else {
                    let mut i: i32 = top;
                    while i >= 0 {
                        let vi = sv_s[(svb + u32::cast_from(i)) as usize];
                        let c1 = (rem << 16u32) | (vi >> 16u32);
                        let q1 = c1 / chunk_div;
                        let r1 = c1 - q1 * chunk_div;
                        let c2 = (r1 << 16u32) | (vi & 0xFFFFu32);
                        let q2 = c2 / chunk_div;
                        rem = c2 - q2 * chunk_div;
                        sv_s[(svb + u32::cast_from(i)) as usize] = (q1 << 16u32) | q2;
                        i -= 1;
                    }
                }
                while top >= 0 {
                    if sv_s[(svb + u32::cast_from(top)) as usize] != 0u32 {
                        break;
                    }
                    top -= 1;
                }

                let mut chunk = rem;
                if top >= 0 {
                    // Interior chunk: all chunk_digits digits, zeros included.
                    #[unroll]
                    for _k in 0..chunk_digits {
                        let cq = chunk / base;
                        let d = chunk - cq * base;
                        chunk = cq;
                        if two_masks {
                            if d < 64u32 {
                                m0 |= 1u64 << u64::cast_from(d);
                            } else {
                                m1 |= 1u64 << u64::cast_from(d - 64u32);
                            }
                        } else {
                            m0 |= 1u64 << u64::cast_from(d);
                        }
                    }
                } else {
                    // Most significant chunk: digits until zero.
                    while chunk != 0u32 {
                        let cq = chunk / base;
                        let d = chunk - cq * base;
                        chunk = cq;
                        if two_masks {
                            if d < 64u32 {
                                m0 |= 1u64 << u64::cast_from(d);
                            } else {
                                m1 |= 1u64 << u64::cast_from(d - 64u32);
                            }
                        } else {
                            m0 |= 1u64 << u64::cast_from(d);
                        }
                    }
                }
            }
            pass += 1u32;
        }

        let mut u = u32::cast_from(m0).count_ones() + u32::cast_from(m0 >> 32u64).count_ones();
        if two_masks {
            u += u32::cast_from(m1).count_ones() + u32::cast_from(m1 >> 32u64).count_ones();
        }
        // --- end num_unique --------------------------------------------------

        hist_s[(copy * hist_bins + u) as usize].fetch_add(1u32);
        if u > cutoff {
            let pos = miss_count[0].fetch_add(1u32);
            if pos < miss_cap {
                let o = MISS_STRIDE * pos;
                miss_data[o as usize] = u32::cast_from(n_lo);
                miss_data[(o + 1u32) as usize] = u32::cast_from(n_lo >> 32u64);
                miss_data[(o + 2u32) as usize] = u32::cast_from(n_hi);
                miss_data[(o + 3u32) as usize] = u32::cast_from(n_hi >> 32u64);
                miss_data[(o + 4u32) as usize] = u;
            }
        }

        idx += stride;
    }

    sync_cube();
    // Reduce the workgroup histogram into the global one.
    let mut b = UNIT_POS_X;
    while b < hist_bins {
        let mut acc = 0u32;
        #[unroll]
        for c in 0..HIST_COPIES {
            acc += hist_s[(c * hist_bins + b) as usize].load();
        }
        if acc != 0u32 {
            hist[b as usize].fetch_add(acc);
        }
        b += CUBE_DIM_X;
    }
}

/// u32 slots per niceonly hit: n as four u32 limbs. Must match the Vulkan
/// backend's `NICE_STRIDE` decode.
pub const NICEONLY_STRIDE: u32 = 4;

/// Capacity of the niceonly output buffer (in nice numbers) per field.
/// Genuinely nice numbers are astronomically rare; this is pure headroom.
/// Matches the Vulkan backend's `NICE_OUT_CAPACITY`.
const NICEONLY_OUT_CAPACITY: usize = 1 << 16;

/// The niceonly kernel: the `#[cube]` port of the Vulkan backend's
/// `niceonly_wgsl`, checking the stride-valid candidates of MSD-surviving
/// ranges reconstructed on-device from the residue table.
///
/// Shared per-candidate check: the low-digit modular prefilter, then the
/// full square/cube distinct-digit scan, appending hits (or, in probe
/// builds, prefilter survivors) to the output buffer. Factored out of
/// `niceonly_kernel` so the plain and plane-compacted enumeration paths run
/// the identical body.
// `collapsible_if`: `has_prefilter` is comptime and `ok` is runtime; the
// cube macro wants them in separate `if`s so the outer one folds away.
#[cube]
#[allow(
    clippy::too_many_arguments,
    clippy::too_many_lines,
    clippy::similar_names,
    clippy::many_single_char_names,
    clippy::collapsible_if,
    clippy::fn_params_excessive_bools
)]
fn candidate_check(
    n_lo: u64,
    n_hi: u64,
    sv_s: &mut SharedMemory<u32>,
    svb: u32,
    nice_out: &mut Array<u32>,
    nice_count: &mut Array<Atomic<u32>>,
    nice_cap: u32,
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
    let sq_limbs = comptime!(2 * limbs);
    let cu_limbs = comptime!(3 * limbs);
    let two_masks = comptime!(base > 64);
    let has_prefilter = comptime!(pre_limbs > 0);

    let mut ok = true;
    // Low-digit modular prefilter: are the lowest
    // `pre_limbs * pre_chunk_digits` digits of n² and n³ all distinct?
    // Fixed-length and branch-free — lanes only save work when their
    // whole group is killed, and this kills ~98% of candidates. Held
    // as digit-chunks of base `pre_chunk_div < 2^16` so the truncated
    // multiplies stay in u32 (`x^k mod b^p == (x mod b^p)^k mod b^p`).
    if has_prefilter {
        if ok {
            // v = n's u32 limbs; a[r] = successive `mod pre_chunk_div`
            // chunks peeled off by the same split16 step the digit scan
            // uses.
            let mut v = Array::<u32>::new(limbs as usize);
            #[unroll]
            for i in 0..limbs {
                let shift = comptime!(((i & 1) * 32) as u64);
                if comptime!(i < 2) {
                    v[i as usize] = u32::cast_from(n_lo >> shift);
                } else {
                    v[i as usize] = u32::cast_from(n_hi >> shift);
                }
            }
            let mut a = Array::<u32>::new(pre_limbs as usize);
            #[unroll]
            for rr in 0..pre_limbs {
                let mut rem = 0u32;
                #[unroll]
                for i in 0..limbs {
                    let idx = comptime!(limbs - 1 - i);
                    let vi = v[idx as usize];
                    // Plain paired `%` here, unlike the digit scan: the
                    // mul-sub form of these four prefilter sites
                    // miscompiles under naga's MSL backend (survivors
                    // diverge from the host mirror on Apple GPUs, caught
                    // by the probe test), and the prefilter runs once per
                    // candidate so the pairing costs nothing measurable.
                    let c1 = (rem << 16u32) | (vi >> 16u32);
                    let q1 = c1 / pre_chunk_div;
                    let c2 = ((c1 % pre_chunk_div) << 16u32) | (vi & 0xFFFFu32);
                    rem = c2 % pre_chunk_div;
                    v[idx as usize] = (q1 << 16u32) | (c2 / pre_chunk_div);
                }
                a[rr as usize] = rem;
            }

            // sq = a², cu = sq·a, truncated schoolbook over digit-chunks —
            // dropping products at or above chunk `pre_limbs` is exactly
            // reduction mod pre_chunk_div^pre_limbs.
            let mut psq = Array::<u32>::new(pre_limbs as usize);
            #[unroll]
            for i in 0..pre_limbs {
                psq[i as usize] = 0u32;
            }
            #[unroll]
            for i in 0..pre_limbs {
                let mut carry = 0u32;
                #[unroll]
                for j in 0..comptime!(pre_limbs - i) {
                    let k = comptime!(i + j);
                    let t = a[i as usize] * a[j as usize] + psq[k as usize] + carry;
                    psq[k as usize] = t % pre_chunk_div;
                    carry = t / pre_chunk_div;
                }
            }
            let mut pcu = Array::<u32>::new(pre_limbs as usize);
            #[unroll]
            for i in 0..pre_limbs {
                pcu[i as usize] = 0u32;
            }
            #[unroll]
            for i in 0..pre_limbs {
                let mut carry = 0u32;
                #[unroll]
                for j in 0..comptime!(pre_limbs - i) {
                    let k = comptime!(i + j);
                    let t = psq[i as usize] * a[j as usize] + pcu[k as usize] + carry;
                    pcu[k as usize] = t % pre_chunk_div;
                    carry = t / pre_chunk_div;
                }
            }

            // Duplicate scan over both values' chunks; every chunk holds
            // exactly pre_chunk_digits real digits, leading zeros included
            // (that is what the host's digit-count guarantee buys).
            let mut p0 = 0u64;
            let mut p1 = 0u64;
            let mut dup = 0u64;
            let mut src: u32 = 0u32;
            while src < 2u32 {
                #[unroll]
                for i in 0..pre_limbs {
                    let mut c = if src == 0u32 {
                        psq[i as usize]
                    } else {
                        pcu[i as usize]
                    };
                    #[unroll]
                    for _k in 0..pre_chunk_digits {
                        let d = c % base;
                        c /= base;
                        if two_masks {
                            if d < 64u32 {
                                let bit = 1u64 << u64::cast_from(d);
                                dup |= p0 & bit;
                                p0 |= bit;
                            } else {
                                let bit = 1u64 << u64::cast_from(d - 64u32);
                                dup |= p1 & bit;
                                p1 |= bit;
                            }
                        } else {
                            let bit = 1u64 << u64::cast_from(d);
                            dup |= p0 & bit;
                            p0 |= bit;
                        }
                    }
                }
                src += 1u32;
            }
            if dup != 0u64 {
                ok = false;
            }
        }
    }

    // Full check: unique digits of n² and n³ together must number
    // exactly `base`. Same limb multiply and chunked scan as the
    // detailed kernel, but the scan bails at the first duplicate —
    // almost no candidate survives its n² scan. A probe build skips
    // it and reports the prefilter's survivors instead, which is the
    // only way a device test can see the filter itself (an
    // over-rejecting prefilter agrees with the CPU on any range
    // without a nice number in it — the v3.2.14 bug class).
    let mut m0 = 0u64;
    let mut m1 = 0u64;
    if ok && !probe {
        let mut nl = Array::<u32>::new(limbs as usize);
        #[unroll]
        for i in 0..limbs {
            let shift = comptime!(((i & 1) * 32) as u64);
            if comptime!(i < 2) {
                nl[i as usize] = u32::cast_from(n_lo >> shift);
            } else {
                nl[i as usize] = u32::cast_from(n_hi >> shift);
            }
        }
        let mut sq = Array::<u32>::new(sq_limbs as usize);
        #[unroll]
        for i in 0..sq_limbs {
            sq[i as usize] = 0u32;
        }
        #[unroll]
        for i in 0..limbs {
            let mut carry = 0u64;
            #[unroll]
            for jj in 0..limbs {
                let k = comptime!(i + jj);
                let t = u64::cast_from(nl[i as usize]) * u64::cast_from(nl[jj as usize])
                    + u64::cast_from(sq[k as usize])
                    + carry;
                sq[k as usize] = u32::cast_from(t);
                carry = t >> 32u64;
            }
            sq[comptime!(i + limbs) as usize] = u32::cast_from(carry);
        }

        // Scan n² first; n³ is only multiplied out if n² had no
        // duplicate (the same ordering worth 20-27% in the CUDA
        // kernel — the cube multiply is usually dead work). Straight-line
        // like the hand kernels: the sq scan reads a sq_limbs window, and
        // the cube is built directly in the scratch and destroyed by its
        // scan — no cu array, no extra copies, no runtime pass branch.
        // (The pass-loop version cost ~30% whole-kernel at bases without a
        // prefilter, where every candidate takes this path.)
        #[unroll]
        for i in 0..sq_limbs {
            sv_s[(svb + u32::cast_from(i)) as usize] = sq[i as usize];
        }
        let mut top: i32 = comptime!(sq_limbs as i32 - 1).runtime();
        while top >= 0 {
            if sv_s[(svb + u32::cast_from(top)) as usize] != 0u32 {
                break;
            }
            top -= 1;
        }
        while top >= 0 && ok {
            let mut rem = 0u32;
            if wide_chunk {
                let mut rem64 = 0u64;
                let mut i: i32 = top;
                while i >= 0 {
                    let cur =
                        (rem64 << 32u64) | u64::cast_from(sv_s[(svb + u32::cast_from(i)) as usize]);
                    // Quotient once, remainder by mul-sub: `%` would
                    // lower as a second independent multiply-high
                    // correction sequence (the hand kernels avoid it
                    // the same way).
                    let q = cur / u64::cast_from(chunk_div);
                    sv_s[(svb + u32::cast_from(i)) as usize] = u32::cast_from(q);
                    rem64 = cur - q * u64::cast_from(chunk_div);
                    i -= 1;
                }
                rem = u32::cast_from(rem64); // rem < chunk_div < 2^31
            } else {
                let mut i: i32 = top;
                while i >= 0 {
                    let vi = sv_s[(svb + u32::cast_from(i)) as usize];
                    let c1 = (rem << 16u32) | (vi >> 16u32);
                    let q1 = c1 / chunk_div;
                    let r1 = c1 - q1 * chunk_div;
                    let c2 = (r1 << 16u32) | (vi & 0xFFFFu32);
                    let q2 = c2 / chunk_div;
                    rem = c2 - q2 * chunk_div;
                    sv_s[(svb + u32::cast_from(i)) as usize] = (q1 << 16u32) | q2;
                    i -= 1;
                }
            }
            while top >= 0 {
                if sv_s[(svb + u32::cast_from(top)) as usize] != 0u32 {
                    break;
                }
                top -= 1;
            }

            let mut chunk = rem;
            let mut dup = 0u64;
            if top >= 0 {
                // Interior chunk: all chunk_digits digits, zeros included.
                #[unroll]
                for _k in 0..chunk_digits {
                    let cq = chunk / base;
                    let d = chunk - cq * base;
                    chunk = cq;
                    if two_masks {
                        if d < 64u32 {
                            let bit = 1u64 << u64::cast_from(d);
                            dup |= m0 & bit;
                            m0 |= bit;
                        } else {
                            let bit = 1u64 << u64::cast_from(d - 64u32);
                            dup |= m1 & bit;
                            m1 |= bit;
                        }
                    } else {
                        let bit = 1u64 << u64::cast_from(d);
                        dup |= m0 & bit;
                        m0 |= bit;
                    }
                }
            } else {
                // Most significant chunk: digits until zero.
                while chunk != 0u32 {
                    let cq = chunk / base;
                    let d = chunk - cq * base;
                    chunk = cq;
                    if two_masks {
                        if d < 64u32 {
                            let bit = 1u64 << u64::cast_from(d);
                            dup |= m0 & bit;
                            m0 |= bit;
                        } else {
                            let bit = 1u64 << u64::cast_from(d - 64u32);
                            dup |= m1 & bit;
                            m1 |= bit;
                        }
                    } else {
                        let bit = 1u64 << u64::cast_from(d);
                        dup |= m0 & bit;
                        m0 |= bit;
                    }
                }
            }
            if dup != 0u64 {
                ok = false;
            }
        }

        if ok {
            // cu = sq * n, built directly in the scratch and consumed there.
            #[unroll]
            for i in 0..cu_limbs {
                sv_s[(svb + u32::cast_from(i)) as usize] = 0u32;
            }
            #[unroll]
            for i in 0..sq_limbs {
                let mut carry = 0u64;
                #[unroll]
                for jj in 0..limbs {
                    let k = comptime!(i + jj);
                    let t = u64::cast_from(sq[i as usize]) * u64::cast_from(nl[jj as usize])
                        + u64::cast_from(sv_s[(svb + k) as usize])
                        + carry;
                    sv_s[(svb + k) as usize] = u32::cast_from(t);
                    carry = t >> 32u64;
                }
                sv_s[(svb + comptime!(i + limbs)) as usize] = u32::cast_from(carry);
            }
            let mut topc: i32 = comptime!(cu_limbs as i32 - 1).runtime();
            while topc >= 0 {
                if sv_s[(svb + u32::cast_from(topc)) as usize] != 0u32 {
                    break;
                }
                topc -= 1;
            }
            while topc >= 0 && ok {
                let mut rem = 0u32;
                if wide_chunk {
                    let mut rem64 = 0u64;
                    let mut i: i32 = topc;
                    while i >= 0 {
                        let cur = (rem64 << 32u64)
                            | u64::cast_from(sv_s[(svb + u32::cast_from(i)) as usize]);
                        // Quotient once, remainder by mul-sub: `%` would
                        // lower as a second independent multiply-high
                        // correction sequence (the hand kernels avoid it
                        // the same way).
                        let q = cur / u64::cast_from(chunk_div);
                        sv_s[(svb + u32::cast_from(i)) as usize] = u32::cast_from(q);
                        rem64 = cur - q * u64::cast_from(chunk_div);
                        i -= 1;
                    }
                    rem = u32::cast_from(rem64); // rem < chunk_div < 2^31
                } else {
                    let mut i: i32 = topc;
                    while i >= 0 {
                        let vi = sv_s[(svb + u32::cast_from(i)) as usize];
                        let c1 = (rem << 16u32) | (vi >> 16u32);
                        let q1 = c1 / chunk_div;
                        let r1 = c1 - q1 * chunk_div;
                        let c2 = (r1 << 16u32) | (vi & 0xFFFFu32);
                        let q2 = c2 / chunk_div;
                        rem = c2 - q2 * chunk_div;
                        sv_s[(svb + u32::cast_from(i)) as usize] = (q1 << 16u32) | q2;
                        i -= 1;
                    }
                }
                while topc >= 0 {
                    if sv_s[(svb + u32::cast_from(topc)) as usize] != 0u32 {
                        break;
                    }
                    topc -= 1;
                }

                let mut chunk = rem;
                let mut dup = 0u64;
                if topc >= 0 {
                    // Interior chunk: all chunk_digits digits, zeros included.
                    #[unroll]
                    for _k in 0..chunk_digits {
                        let cq = chunk / base;
                        let d = chunk - cq * base;
                        chunk = cq;
                        if two_masks {
                            if d < 64u32 {
                                let bit = 1u64 << u64::cast_from(d);
                                dup |= m0 & bit;
                                m0 |= bit;
                            } else {
                                let bit = 1u64 << u64::cast_from(d - 64u32);
                                dup |= m1 & bit;
                                m1 |= bit;
                            }
                        } else {
                            let bit = 1u64 << u64::cast_from(d);
                            dup |= m0 & bit;
                            m0 |= bit;
                        }
                    }
                } else {
                    // Most significant chunk: digits until zero.
                    while chunk != 0u32 {
                        let cq = chunk / base;
                        let d = chunk - cq * base;
                        chunk = cq;
                        if two_masks {
                            if d < 64u32 {
                                let bit = 1u64 << u64::cast_from(d);
                                dup |= m0 & bit;
                                m0 |= bit;
                            } else {
                                let bit = 1u64 << u64::cast_from(d - 64u32);
                                dup |= m1 & bit;
                                m1 |= bit;
                            }
                        } else {
                            let bit = 1u64 << u64::cast_from(d);
                            dup |= m0 & bit;
                            m0 |= bit;
                        }
                    }
                }
                if dup != 0u64 {
                    ok = false;
                }
            }
        }
    }

    if ok {
        let mut u = u32::cast_from(m0).count_ones() + u32::cast_from(m0 >> 32u64).count_ones();
        if two_masks {
            u += u32::cast_from(m1).count_ones() + u32::cast_from(m1 >> 32u64).count_ones();
        }
        if probe || u == base {
            let pos = nice_count[0].fetch_add(1u32);
            if pos < nice_cap {
                let o = NICEONLY_STRIDE * pos;
                nice_out[o as usize] = u32::cast_from(n_lo);
                nice_out[(o + 1u32) as usize] = u32::cast_from(n_lo >> 32u64);
                nice_out[(o + 2u32) as usize] = u32::cast_from(n_hi);
                nice_out[(o + 3u32) as usize] = u32::cast_from(n_hi >> 32u64);
            }
        }
    }
}

/// `lanes = 1 << lane_shift` threads cooperate on each range, striding
/// through its candidates by index — pure index arithmetic, no subgroup ops.
/// The g-th valid candidate at or after a range start is
/// `B0 + (g / R) * M + residues[g % R]`.
///
/// The stride modulus `M` and residue count `R` depend only on the base, so
/// they ride along as comptime parameters: every division below is by a
/// comptime constant, including the range offset's chunked Horner reduction
/// (`offset_chunk_bits` wide, u32 throughout — the construct RADV cannot
/// strength-reduce at 64 bits, avoided the same way the WGSL avoids it).
#[cube(launch_unchecked)]
// The single-character and lookalike names (m, g, j, rs/re, ...) deliberately
// match the generated WGSL, so the two kernels review side by side.
// `manual_midpoint`: `u32::midpoint` has no cube-dialect translation.
// manual_midpoint: `u32::midpoint` has no cube translation, so the kernel
// keeps the shift form.
// `collapsible_if`: `has_prefilter` is comptime and `ok` is runtime; the
// cube macro wants them in separate `if`s so the outer one folds away.
#[allow(
    clippy::too_many_arguments,
    clippy::too_many_lines,
    clippy::similar_names,
    clippy::many_single_char_names,
    clippy::manual_midpoint,
    clippy::collapsible_if,
    clippy::fn_params_excessive_bools
)]
fn niceonly_kernel(
    residues: &Array<u32>,
    range_offsets: &Array<u32>, // lo, hi pairs
    range_lens: &Array<u32>,
    nice_out: &mut Array<u32>,
    nice_count: &mut Array<Atomic<u32>>,
    low_masks: &Array<u32>, // per residue: lo,hi words of its exact low-digit mask
    range_masks: &Array<u32>, // per range: lo,hi words of its certificate
    fs0: u32,
    fs1: u32,
    fs2: u32,
    fs3: u32,
    fs_mod_m: u32,
    num_ranges: u32,
    nice_cap: u32,
    lane_shift: u32,
    #[comptime] base: u32,
    #[comptime] limbs: u32,
    #[comptime] chunk_digits: u32,
    #[comptime] chunk_div: u32,
    #[comptime] wide_chunk: bool,
    #[comptime] stride_m: u32,
    #[comptime] stride_r: u32,
    #[comptime] offset_chunk_bits: u32,
    #[comptime] pre_limbs: u32, // 0 disables the low-digit prefilter
    #[comptime] pre_chunk_digits: u32,
    #[comptime] pre_chunk_div: u32,
    #[comptime] probe: bool, // report prefilter survivors instead of nice numbers
    #[comptime] cross: bool, // cross-end residue filter (certificate x low mask)
    #[comptime] compact: bool, // plane-compact mask survivors before checking
    #[comptime] plane_scoped: bool, // compaction queue per plane (needs Plane::Sync)
) {
    let cu_limbs = comptime!(3 * limbs);
    let offset_chunks = comptime!(64 / offset_chunk_bits);
    let per_word = comptime!(32 / offset_chunk_bits);
    let offset_chunk_mask = comptime!((1u32 << offset_chunk_bits) - 1);

    let lanes = 1u32 << lane_shift;
    let lane = ABSOLUTE_POS_X & (lanes - 1u32);
    let nwarps = (CUBE_COUNT_X * CUBE_DIM_X) >> lane_shift;
    let fs_lo = (u64::cast_from(fs1) << 32u64) | u64::cast_from(fs0);
    let fs_hi = (u64::cast_from(fs3) << 32u64) | u64::cast_from(fs2);

    // Scratch for the digit scan, in shared memory: the split loop indexes
    // it at runtime, and a register-promoted array makes every such access a
    // compare/select chain over all words (~100 extra SASS compares at b50),
    // while local memory costs a round trip. One shared word per thread per
    // limb, padded to an odd stride so a warp's accesses spread across banks.
    let sv_pad = comptime!(cu_limbs | 1);
    let mut sv_s = SharedMemory::<u32>::new(comptime!((WORKGROUP_SIZE * (cu_limbs | 1)) as usize));
    let svb = UNIT_POS_X * sv_pad;

    if comptime!(compact && plane_scoped) {
        // Plane-scoped compaction: the same cursors and survivor production
        // as the cube-scoped path below, but each plane owns a private
        // queue (2*PLANE_DIM slots) in shared memory, takes positions
        // straight from plane_exclusive_sum, fences the write->drain handoff
        // with sync_plane, and drains in PLANE_DIM-wide waves. No cube
        // barrier, no serial walk over plane totals, and a plane whose
        // ranges are exhausted leaves as soon as its queue is dry instead
        // of idling through the other planes' iterations. Control flow is
        // plane-uniform: `qn` comes from plane_sum and `done` from
        // plane_all. Requires Plane::Sync (spirv/msl/cuda; not wgsl).
        let mut q_lo = SharedMemory::<u64>::new(comptime!((2 * WORKGROUP_SIZE) as usize));
        let mut q_hi = SharedMemory::<u64>::new(comptime!((2 * WORKGROUP_SIZE) as usize));
        let qbase = PLANE_POS * (2u32 * PLANE_DIM);
        let lane_in_plane = UNIT_POS_X % PLANE_DIM;

        let mut r = ABSOLUTE_POS_X >> lane_shift;
        let mut have_range = r < num_ranges;
        let mut need_setup = have_range;
        let mut rmask = 0u64;
        let mut re_lo = 0u64;
        let mut re_hi = 0u64;
        let mut b0_lo = 0u64;
        let mut b0_hi = 0u64;
        let mut g = 0u32;
        let mut qn = 0u32;
        loop {
            if need_setup {
                need_setup = false;
                let off_lo = range_offsets[(2u32 * r) as usize];
                let off_hi = range_offsets[(2u32 * r + 1u32) as usize];
                let offset = (u64::cast_from(off_hi) << 32u64) | u64::cast_from(off_lo);
                rmask = 0u64;
                if cross {
                    rmask = (u64::cast_from(range_masks[(2u32 * r + 1u32) as usize]) << 32u64)
                        | u64::cast_from(range_masks[(2u32 * r) as usize]);
                }
                let rs_lo = fs_lo + offset;
                let mut rs_hi = fs_hi;
                if rs_lo < fs_lo {
                    rs_hi += 1u64;
                }
                re_lo = rs_lo + u64::cast_from(range_lens[r as usize]);
                re_hi = rs_hi;
                if re_lo < rs_lo {
                    re_hi += 1u64;
                }
                let mut acc = 0u32;
                #[unroll]
                for k in 0..offset_chunks {
                    let word = if comptime!(k < per_word) {
                        off_hi
                    } else {
                        off_lo
                    };
                    let shift =
                        comptime!(32 - offset_chunk_bits - (k % per_word) * offset_chunk_bits);
                    acc = ((acc << offset_chunk_bits) | ((word >> shift) & offset_chunk_mask))
                        % stride_m;
                }
                let m = (fs_mod_m + acc) % stride_m;
                b0_lo = rs_lo - u64::cast_from(m);
                b0_hi = rs_hi;
                if rs_lo < u64::cast_from(m) {
                    b0_hi -= 1u64;
                }
                let mut lb_lo = 0u32;
                let mut lb_hi = stride_r.runtime();
                while lb_lo < lb_hi {
                    let mid = (lb_lo + lb_hi) >> 1u32;
                    if residues[mid as usize] < m {
                        lb_lo = mid + 1u32;
                    } else {
                        lb_hi = mid;
                    }
                }
                g = lb_lo + lane;
            }

            // One candidate ordinal per lane per iteration. A lane whose
            // range just ended spends the iteration advancing its cursor
            // and produces nothing.
            let mut pf = 0u32;
            let mut cand_lo = 0u64;
            let mut cand_hi = 0u64;
            if have_range {
                let cycle = g / stride_r;
                let j = g - cycle * stride_r;
                let add = u64::cast_from(cycle) * u64::cast_from(stride_m)
                    + u64::cast_from(residues[j as usize]);
                let n_lo = b0_lo + add;
                let mut n_hi = b0_hi;
                if n_lo < b0_lo {
                    n_hi += 1u64;
                }
                if n_hi > re_hi || (n_hi == re_hi && n_lo >= re_lo) {
                    r += nwarps;
                    if r < num_ranges {
                        need_setup = true;
                    } else {
                        have_range = false;
                    }
                } else {
                    let mut masked = false;
                    if cross {
                        let lm = (u64::cast_from(low_masks[(2u32 * j + 1u32) as usize]) << 32u64)
                            | u64::cast_from(low_masks[(2u32 * j) as usize]);
                        if (lm & rmask) != 0u64 {
                            masked = true;
                        }
                    }
                    if !masked {
                        pf = 1u32;
                        cand_lo = n_lo;
                        cand_hi = n_hi;
                    }
                    g += lanes;
                }
            }

            let idx_in_plane = plane_exclusive_sum(pf);
            let tot = plane_sum(pf);
            let done = plane_all(!have_range);
            if pf != 0u32 {
                let dst = (qbase + qn + idx_in_plane) as usize;
                q_lo[dst] = cand_lo;
                q_hi[dst] = cand_hi;
            }
            qn += tot;
            // Queue writes become visible to the drain below.
            sync_plane();

            let mut take = 0u32;
            if qn >= PLANE_DIM {
                take = PLANE_DIM;
            } else if done {
                take = qn;
            }
            if lane_in_plane < take {
                let s = (qbase + qn - take + lane_in_plane) as usize;
                candidate_check(
                    q_lo[s],
                    q_hi[s],
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
            }
            qn -= take;
            // Drain reads are ordered before the next iteration reuses slots.
            sync_plane();
            if done && qn == 0u32 {
                break;
            }
        }
    } else if compact {
        // Cube-compacted enumeration (cross filter survivors only).
        //
        // Each lane keeps its own (range, ordinal) cursor - identical
        // candidate coverage to the plain path - but instead of checking a
        // survivor in place (where SIMT would idle every filtered lane
        // through its neighbors' checks), survivors queue in cube-scoped
        // shared memory and are checked in dense CUBE_DIM_X-wide waves.
        //
        // The queue is cube-scoped rather than plane-scoped because WGSL
        // has no plane barrier (cubecl's sync_plane panics in the wgsl
        // compiler), so the write->read handoff is fenced with sync_cube.
        // Queue positions come from a two-level scan: plane_exclusive_sum
        // within a plane, plane totals through shared memory. Control flow
        // is cube-uniform: every thread runs every iteration with exactly
        // two barriers, planes whose ranges are exhausted just stop
        // producing, and everyone leaves together once the queue is dry.
        // Wave-size agnostic: nothing assumes a plane width, only that it
        // divides CUBE_DIM_X.
        let mut q_lo = SharedMemory::<u64>::new(comptime!((2 * WORKGROUP_SIZE) as usize));
        let mut q_hi = SharedMemory::<u64>::new(comptime!((2 * WORKGROUP_SIZE) as usize));
        // Indexed by plane id; sized for the narrowest plane wgpu allows.
        let mut plane_tot = SharedMemory::<u32>::new(comptime!(WORKGROUP_SIZE as usize));
        let mut plane_done = SharedMemory::<u32>::new(comptime!(WORKGROUP_SIZE as usize));
        let my_plane = UNIT_POS_X / PLANE_DIM;
        let num_planes = CUBE_DIM_X / PLANE_DIM;

        let mut r = ABSOLUTE_POS_X >> lane_shift;
        let mut have_range = r < num_ranges;
        let mut need_setup = have_range;
        let mut rmask = 0u64;
        let mut re_lo = 0u64;
        let mut re_hi = 0u64;
        let mut b0_lo = 0u64;
        let mut b0_hi = 0u64;
        let mut g = 0u32;
        let mut qn = 0u32;
        loop {
            if need_setup {
                need_setup = false;
                let off_lo = range_offsets[(2u32 * r) as usize];
                let off_hi = range_offsets[(2u32 * r + 1u32) as usize];
                let offset = (u64::cast_from(off_hi) << 32u64) | u64::cast_from(off_lo);
                rmask = 0u64;
                if cross {
                    rmask = (u64::cast_from(range_masks[(2u32 * r + 1u32) as usize]) << 32u64)
                        | u64::cast_from(range_masks[(2u32 * r) as usize]);
                }
                let rs_lo = fs_lo + offset;
                let mut rs_hi = fs_hi;
                if rs_lo < fs_lo {
                    rs_hi += 1u64;
                }
                re_lo = rs_lo + u64::cast_from(range_lens[r as usize]);
                re_hi = rs_hi;
                if re_lo < rs_lo {
                    re_hi += 1u64;
                }
                let mut acc = 0u32;
                #[unroll]
                for k in 0..offset_chunks {
                    let word = if comptime!(k < per_word) {
                        off_hi
                    } else {
                        off_lo
                    };
                    let shift =
                        comptime!(32 - offset_chunk_bits - (k % per_word) * offset_chunk_bits);
                    acc = ((acc << offset_chunk_bits) | ((word >> shift) & offset_chunk_mask))
                        % stride_m;
                }
                let m = (fs_mod_m + acc) % stride_m;
                b0_lo = rs_lo - u64::cast_from(m);
                b0_hi = rs_hi;
                if rs_lo < u64::cast_from(m) {
                    b0_hi -= 1u64;
                }
                let mut lb_lo = 0u32;
                let mut lb_hi = stride_r.runtime();
                while lb_lo < lb_hi {
                    let mid = (lb_lo + lb_hi) >> 1u32;
                    if residues[mid as usize] < m {
                        lb_lo = mid + 1u32;
                    } else {
                        lb_hi = mid;
                    }
                }
                g = lb_lo + lane;
            }

            // One candidate ordinal per lane per iteration. A lane whose
            // range just ended spends the iteration advancing its cursor
            // and produces nothing.
            let mut pf = 0u32;
            let mut cand_lo = 0u64;
            let mut cand_hi = 0u64;
            if have_range {
                let cycle = g / stride_r;
                let j = g - cycle * stride_r;
                let add = u64::cast_from(cycle) * u64::cast_from(stride_m)
                    + u64::cast_from(residues[j as usize]);
                let n_lo = b0_lo + add;
                let mut n_hi = b0_hi;
                if n_lo < b0_lo {
                    n_hi += 1u64;
                }
                if n_hi > re_hi || (n_hi == re_hi && n_lo >= re_lo) {
                    r += nwarps;
                    if r < num_ranges {
                        need_setup = true;
                    } else {
                        have_range = false;
                    }
                } else {
                    let mut masked = false;
                    if cross {
                        let lm = (u64::cast_from(low_masks[(2u32 * j + 1u32) as usize]) << 32u64)
                            | u64::cast_from(low_masks[(2u32 * j) as usize]);
                        if (lm & rmask) != 0u64 {
                            masked = true;
                        }
                    }
                    if !masked {
                        pf = 1u32;
                        cand_lo = n_lo;
                        cand_hi = n_hi;
                    }
                    g += lanes;
                }
            }

            let idx_in_plane = plane_exclusive_sum(pf);
            let tot = plane_sum(pf);
            let pd = plane_all(!have_range);
            if UNIT_POS_PLANE == 0u32 {
                plane_tot[my_plane as usize] = tot;
                let mut d = 0u32;
                if pd {
                    d = 1u32;
                }
                plane_done[my_plane as usize] = d;
            }
            // Barrier 1: plane totals/done flags become visible, and the
            // previous iteration's drain reads are ordered before this
            // iteration's queue writes reuse those slots.
            sync_cube();

            let mut base_off = qn;
            let mut incoming = 0u32;
            let mut all_done = true;
            let mut p = 0u32;
            while p < num_planes {
                let t = plane_tot[p as usize];
                if p < my_plane {
                    base_off += t;
                }
                incoming += t;
                if plane_done[p as usize] == 0u32 {
                    all_done = false;
                }
                p += 1u32;
            }
            if pf != 0u32 {
                let dst = (base_off + idx_in_plane) as usize;
                q_lo[dst] = cand_lo;
                q_hi[dst] = cand_hi;
            }
            qn += incoming;
            // Barrier 2: queue writes become visible to the drain below.
            sync_cube();

            // qn and all_done are cube-uniform (derived from the shared
            // totals alone), so drain sizing and loop exit stay uniform.
            let mut take = 0u32;
            if qn >= CUBE_DIM_X {
                take = CUBE_DIM_X;
            } else if all_done {
                take = qn;
            }
            if UNIT_POS_X < take {
                let s = (qn - take + UNIT_POS_X) as usize;
                candidate_check(
                    q_lo[s],
                    q_hi[s],
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
            }
            qn -= take;
            if all_done && qn == 0u32 {
                break;
            }
        }
    } else {
        let mut r = ABSOLUTE_POS_X >> lane_shift;
        while r < num_ranges {
            let off_lo = range_offsets[(2u32 * r) as usize];
            let off_hi = range_offsets[(2u32 * r + 1u32) as usize];
            let offset = (u64::cast_from(off_hi) << 32u64) | u64::cast_from(off_lo);

            // Cross-end certificate for this range (0 = no filtering; the
            // no-MSD bypass and mask-less analyses ship exactly that).
            let mut rmask = 0u64;
            if cross {
                rmask = (u64::cast_from(range_masks[(2u32 * r + 1u32) as usize]) << 32u64)
                    | u64::cast_from(range_masks[(2u32 * r) as usize]);
            }

            // range_start = field_start + offset, range_end = + len.
            let rs_lo = fs_lo + offset;
            let mut rs_hi = fs_hi;
            if rs_lo < fs_lo {
                rs_hi += 1u64;
            }
            let re_lo = rs_lo + u64::cast_from(range_lens[r as usize]);
            let mut re_hi = rs_hi;
            if re_lo < rs_lo {
                re_hi += 1u64;
            }

            // m = range_start mod M. The host pushes `field_start mod M`; only the
            // 64-bit offset is reduced here, MSB-first Horner over u32 chunks with
            // every divisor a comptime constant. Exact because acc < M <= 2^(32-c).
            let mut acc = 0u32;
            #[unroll]
            for k in 0..offset_chunks {
                let word = if comptime!(k < per_word) {
                    off_hi
                } else {
                    off_lo
                };
                let shift = comptime!(32 - offset_chunk_bits - (k % per_word) * offset_chunk_bits);
                acc =
                    ((acc << offset_chunk_bits) | ((word >> shift) & offset_chunk_mask)) % stride_m;
            }
            let m = (fs_mod_m + acc) % stride_m;

            // B0 = range_start - m.
            let b0_lo = rs_lo - u64::cast_from(m);
            let mut b0_hi = rs_hi;
            if rs_lo < u64::cast_from(m) {
                b0_hi -= 1u64;
            }

            // First residue index at or after m: lower_bound over the sorted table.
            let mut lb_lo = 0u32;
            let mut lb_hi = stride_r.runtime();
            while lb_lo < lb_hi {
                let mid = (lb_lo + lb_hi) >> 1u32;
                if residues[mid as usize] < m {
                    lb_lo = mid + 1u32;
                } else {
                    lb_hi = mid;
                }
            }

            let mut g = lb_lo + lane;
            loop {
                let cycle = g / stride_r;
                let j = g - cycle * stride_r;
                let add = u64::cast_from(cycle) * u64::cast_from(stride_m)
                    + u64::cast_from(residues[j as usize]);
                let n_lo = b0_lo + add;
                let mut n_hi = b0_hi;
                if n_lo < b0_lo {
                    n_hi += 1u64;
                }
                if n_hi > re_hi || (n_hi == re_hi && n_lo >= re_lo) {
                    break;
                }

                // Cross-end residue filter: residue j's exact low output digits
                // against the range's certain high digits. An intersection is a
                // duplicated digit across two distinct positions - skip the
                // candidate without checking anything.
                let mut masked = false;
                if cross {
                    let lm = (u64::cast_from(low_masks[(2u32 * j + 1u32) as usize]) << 32u64)
                        | u64::cast_from(low_masks[(2u32 * j) as usize]);
                    if (lm & rmask) != 0u64 {
                        masked = true;
                    }
                }
                if !masked {
                    candidate_check(
                        n_lo,
                        n_hi,
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
                }

                g += lanes;
            }
            r += nwarps;
        }
    }
}

/// One-thread smoke kernel for [`CubeclContext::new_cuda`]: proves the
/// runtime can actually compile and run something before init reports
/// success.
#[cube(launch_unchecked)]
fn smoke_kernel(out: &mut Array<u32>) {
    if ABSOLUTE_POS_X == 0 {
        out[0] = 42u32;
    }
}

// ============================================================================
// Host side
// ============================================================================

/// The process-wide wgpu client: `init_setup` registers the device server
/// and is not idempotent, so setup runs exactly once per process and later
/// contexts clone the client — shared by the sync and async constructors.
static WGPU_DEFAULT: OnceLock<(
    cubecl::prelude::ComputeClient<cubecl::wgpu::WgpuRuntime>,
    String,
)> = OnceLock::new();

/// One initialized `CubeCL` device: wgpu everywhere, or the native CUDA
/// runtime when built with `cubecl-cuda` (the meaningful NVIDIA comparison,
/// since it exercises `CubeCL`'s CUDA codegen against the hand kernels).
pub enum CubeclContext {
    Wgpu {
        client: cubecl::prelude::ComputeClient<cubecl::wgpu::WgpuRuntime>,
        device_name: String,
        /// Per-base niceonly plans, cached across fields — see [`NiceonlyPlan`].
        /// Shared with the niceonly pipeline's dispatch thread.
        niceonly_plans: Arc<Mutex<HashMap<u32, Arc<NiceonlyPlan>>>>,
        /// The continuous niceonly pipeline, started on the first niceonly field.
        niceonly_pipeline:
            Mutex<Option<NiceonlyPipeline<CubeclPendingField<cubecl::wgpu::WgpuRuntime>>>>,
    },
    #[cfg(feature = "cubecl-cuda")]
    Cuda {
        client: cubecl::prelude::ComputeClient<cubecl::cuda::CudaRuntime>,
        device_name: String,
        /// Per-base niceonly plans, cached across fields — see [`NiceonlyPlan`].
        /// Shared with the niceonly pipeline's dispatch thread.
        niceonly_plans: Arc<Mutex<HashMap<u32, Arc<NiceonlyPlan>>>>,
        /// The continuous niceonly pipeline, started on the first niceonly field.
        niceonly_pipeline:
            Mutex<Option<NiceonlyPipeline<CubeclPendingField<cubecl::cuda::CudaRuntime>>>>,
    },
    #[cfg(feature = "cubecl-hip")]
    Hip {
        client: cubecl::prelude::ComputeClient<cubecl::hip::HipRuntime>,
        device_name: String,
        /// Per-base niceonly plans, cached across fields — see [`NiceonlyPlan`].
        niceonly_plans: Mutex<HashMap<u32, Arc<NiceonlyPlan>>>,
    },
}

/// The device's real name from the CUDA driver, e.g. `NVIDIA A100-SXM4-40GB`.
///
/// `CubeCL`'s CUDA runtime does not expose one — its `Runtime::name` is the
/// static string `"cuda"` — so this asks the driver directly. That matters
/// well beyond cosmetics: this string is `hardware.gpu_model` in every
/// benchmark report and every `--telemetry` submission, and both the API's
/// `/estimate` and the fleet controller match offers against it. A
/// `gpu_models_match` against a placeholder fails for every real GPU, which
/// drops the estimate to the `floor` stage and makes the sample unusable. And
/// since #133 fixed the runtime headers, `cubecl-cuda` is the backend
/// `--gpu-backend auto` picks for detailed mode on NVIDIA, so a placeholder
/// here would silently blank the GPU identity of the entire detailed corpus.
///
/// Asked at *device* level (`cuDeviceGet` + `cuDeviceGetName`), which touches
/// no context and so cannot disturb the one `CubeCL` has just built. `cuInit`
/// is idempotent and has already run inside `CudaRuntime::client`; it is
/// repeated here only so the helper does not depend on call order.
///
/// The backend itself is *not* folded into this string — the hand-CUDA arm
/// reports the bare driver name too, and `hardware.gpu_backend` already
/// distinguishes the two. Keeping them identical is what lets a `cuda` and a
/// `cubecl-cuda` report of the same card land in the same estimator bucket.
///
/// Falls back to the old placeholder if the driver refuses, which is the only
/// case a caller can still see. cudarc *panics* rather than erroring when
/// `libcuda` cannot be loaded (the same trap the NVRTC probe in
/// `client_process_cuda` documents), so the driver calls run under
/// `catch_unwind` — a name for a report is never worth taking the process down.
#[cfg(feature = "cubecl-cuda")]
fn cuda_device_name(device_index: usize) -> String {
    let name = std::panic::catch_unwind(|| {
        use cudarc::driver::result::{device, init};
        // Idempotent, and `CudaRuntime::client` has already run it; repeated
        // here only so this helper does not depend on call order.
        init().ok()?;
        let ordinal = i32::try_from(device_index).ok()?;
        let dev = device::get(ordinal).ok()?;
        device::get_name(dev).ok()
    });
    match name {
        Ok(Some(name)) if !name.trim().is_empty() => name,
        _ => {
            warn!(
                "CUDA driver did not name device {device_index}; \
                 benchmark and telemetry reports will carry a placeholder"
            );
            format!("cubecl-cuda device {device_index}")
        }
    }
}

impl CubeclContext {
    /// Runtime options for the wgpu backend.
    ///
    /// In a browser this pins `ExclusivePages`, so every buffer is allocated
    /// at exactly its own size. The default `SubSlices` pool instead builds a
    /// ladder of page sizes down from `max_storage_buffer_binding_size` — the
    /// *adapter's* maximum, since cubecl requests adapter limits — so on a
    /// discrete card it will try to allocate pages of hundreds of megabytes
    /// to satisfy a 21 MB request. A browser gives a tab far less GPU memory
    /// than the card has, and refuses: observed on an RX 9070 XT under
    /// `LibreWolf` 149 as "Not enough memory left", then an invalid buffer,
    /// then a panic in `CubeCL`'s buffer mapping. Native builds keep the
    /// default pooling, which is measurably fine on the same card.
    ///
    /// This backend allocates a handful of long-lived buffers per field, so
    /// exclusive pages cost nothing here.
    fn runtime_options() -> cubecl::wgpu::RuntimeOptions {
        #[allow(unused_mut)]
        let mut options = cubecl::wgpu::RuntimeOptions::default();
        #[cfg(target_family = "wasm")]
        {
            options.memory_config = cubecl::wgpu::MemoryConfiguration::ExclusivePages;
        }
        options
    }

    /// Initialize on the runtime's default device (discrete first, then
    /// integrated, then software rasterizers), recording the adapter's
    /// marketing name for benchmark reports and the estimator.
    ///
    /// `init_setup` registers the device server and is not idempotent, so the
    /// setup runs exactly once per process; later contexts clone the client.
    ///
    /// # Errors
    /// Returns an error if no wgpu device is available.
    pub fn new_default() -> Result<Self> {
        let (client, device_name) = WGPU_DEFAULT.get_or_init(|| {
            let device = cubecl::wgpu::WgpuDevice::default();
            let setup = cubecl::wgpu::init_setup::<cubecl::wgpu::AutoGraphicsApi>(
                &device,
                Self::runtime_options(),
            );
            // Include the resolved graphics API: AutoGraphicsApi may pick
            // DX12 on Windows, and reports comparing this backend against
            // the hand-Vulkan one need to know which API actually ran.
            let info = setup.adapter.get_info();
            let client = cubecl::wgpu::WgpuRuntime::client(&device);
            let device_name = format!(
                "{} ({:?}, {})",
                info.name,
                info.backend,
                <cubecl::wgpu::WgpuRuntime as cubecl::prelude::Runtime>::name(&client)
            );
            (client, device_name)
        });
        Ok(Self::Wgpu {
            client: client.clone(),
            device_name: device_name.clone(),
            niceonly_plans: Arc::new(Mutex::new(HashMap::new())),
            niceonly_pipeline: Mutex::new(None),
        })
    }

    /// As [`Self::new_default`], but awaiting adapter acquisition — the only
    /// form a browser permits (`init_setup` panics on wasm by design). Native
    /// callers can use either; both share one process-wide client.
    ///
    /// # Errors
    /// Returns an error if no wgpu device is available.
    ///
    /// # Panics
    /// Cannot in practice: the read of the just-initialized process-wide
    /// client is infallible once the set above has happened.
    pub async fn new_default_async() -> Result<Self> {
        if WGPU_DEFAULT.get().is_none() {
            let device = cubecl::wgpu::WgpuDevice::default();
            let setup = cubecl::wgpu::init_setup_async::<cubecl::wgpu::AutoGraphicsApi>(
                &device,
                Self::runtime_options(),
            )
            .await;
            let info = setup.adapter.get_info();
            let client = cubecl::wgpu::WgpuRuntime::client(&device);
            let device_name = format!(
                "{} ({:?}, {})",
                info.name,
                info.backend,
                <cubecl::wgpu::WgpuRuntime as cubecl::prelude::Runtime>::name(&client)
            );
            // A racing initializer winning this set is fine: both describe
            // the same process-wide runtime registration.
            let _ = WGPU_DEFAULT.set((client, device_name));
        }
        let (client, device_name) = WGPU_DEFAULT
            .get()
            .expect("just initialized the wgpu default client");
        Ok(Self::Wgpu {
            client: client.clone(),
            device_name: device_name.clone(),
            niceonly_plans: Arc::new(Mutex::new(HashMap::new())),
            niceonly_pipeline: Mutex::new(None),
        })
    }

    /// Initialize `CubeCL`'s native CUDA runtime on `device_index`.
    ///
    /// # Errors
    /// Returns an error if CUDA is unavailable.
    /// A working NVIDIA *driver* is not enough: kernel compilation needs
    /// NVRTC from the CUDA toolkit, `CubeCL` compiles lazily at first
    /// launch, and a missing NVRTC panics `CubeCL`'s server thread while
    /// launches keep reporting success — a completed run whose results are
    /// silently zero. So init proves the runtime end-to-end with a smoke
    /// kernel before reporting success, the same contract as the hand
    /// backends' init-time compile checks.
    #[cfg(feature = "cubecl-cuda")]
    pub fn new_cuda(device_index: usize) -> Result<Self> {
        let device = cubecl::cuda::CudaDevice::new(device_index);
        let client = cubecl::cuda::CudaRuntime::client(&device);

        let out = client.create(cubecl::bytes::Bytes::from_elems(vec![0u32; 1]));
        unsafe {
            smoke_kernel::launch_unchecked::<cubecl::cuda::CudaRuntime>(
                &client,
                CubeCount::Static(1, 1, 1),
                CubeDim::new_1d(1),
                ArrayArg::from_raw_parts(out.clone(), 1),
            );
        }
        let bytes = client.read_one(out).map_err(|e| {
            anyhow::anyhow!(
                "CubeCL CUDA smoke test failed to read back: {e:?} \
                 (is the CUDA toolkit, including NVRTC, installed?)"
            )
        })?;
        ensure!(
            u32::from_bytes(&bytes)[0] == 42,
            "CubeCL CUDA smoke kernel did not run \
             (is the CUDA toolkit, including NVRTC, installed?)"
        );

        Ok(Self::Cuda {
            client,
            device_name: cuda_device_name(device_index),
            niceonly_plans: Arc::new(Mutex::new(HashMap::new())),
            niceonly_pipeline: Mutex::new(None),
        })
    }

    /// Initialize on the native HIP (`ROCm`) runtime, device `device_index`.
    ///
    /// Same smoke-kernel contract as [`Self::new_cuda`]: hiprtc must compile
    /// and run something before init reports success.
    ///
    /// # Errors
    /// Returns an error if the HIP runtime cannot compile or run a kernel.
    #[cfg(feature = "cubecl-hip")]
    pub fn new_hip(device_index: usize) -> Result<Self> {
        let device = cubecl::hip::AmdDevice::new(device_index);
        let client = cubecl::hip::HipRuntime::client(&device);

        let out = client.create(cubecl::bytes::Bytes::from_elems(vec![0u32; 1]));
        unsafe {
            smoke_kernel::launch_unchecked::<cubecl::hip::HipRuntime>(
                &client,
                CubeCount::Static(1, 1, 1),
                CubeDim::new_1d(1),
                ArrayArg::from_raw_parts(out.clone(), 1),
            );
        }
        let bytes = client.read_one(out).map_err(|e| {
            anyhow::anyhow!("CubeCL HIP smoke test failed to read back: {e:?} (is ROCm installed?)")
        })?;
        ensure!(
            u32::from_bytes(&bytes)[0] == 42,
            "CubeCL HIP smoke kernel did not run (is ROCm installed?)"
        );

        Ok(Self::Hip {
            client,
            device_name: format!("cubecl-hip device {device_index}"),
            niceonly_plans: Mutex::new(HashMap::new()),
        })
    }

    /// The adapter/device name, for reports.
    #[must_use]
    pub fn device_name(&self) -> String {
        match self {
            Self::Wgpu { device_name, .. } => device_name.clone(),
            #[cfg(feature = "cubecl-cuda")]
            Self::Cuda { device_name, .. } => device_name.clone(),
            #[cfg(feature = "cubecl-hip")]
            Self::Hip { device_name, .. } => device_name.clone(),
        }
    }

    /// Which runtime this context drives, matching the `--gpu-backend` value
    /// that selects it — for benchmark reports and submission telemetry.
    #[must_use]
    pub fn backend_name(&self) -> &'static str {
        match self {
            Self::Wgpu { .. } => "cubecl",
            #[cfg(feature = "cubecl-cuda")]
            Self::Cuda { .. } => "cubecl-cuda",
            #[cfg(feature = "cubecl-hip")]
            Self::Hip { .. } => "cubecl-hip",
        }
    }
}

/// `CubeCL` implementation of `process_range_detailed`.
///
/// **Range semantics**: half-open [`range_start`, `range_end`).
///
/// # Errors
/// Returns an error on any device failure or if the near-miss buffer
/// overflows.
pub fn process_range_detailed_cubecl(
    ctx: &CubeclContext,
    range: &FieldSize,
    base: u32,
) -> Result<FieldResults> {
    cubecl::future::block_on(process_range_detailed_cubecl_async(ctx, range, base))
}

/// As [`process_range_detailed_cubecl`], awaiting device reads instead of
/// blocking on them — the only form a browser permits. The sync wrapper is
/// this future under `block_on`, so both paths run the same body (and the
/// native device tests cover it).
///
/// # Errors
/// Returns an error on any device failure or if the near-miss buffer
/// overflows.
pub async fn process_range_detailed_cubecl_async(
    ctx: &CubeclContext,
    range: &FieldSize,
    base: u32,
) -> Result<FieldResults> {
    if !gpu_supports_base(base) {
        warn!("base {base} not supported on GPU, falling back to CPU for this field");
        return Ok(process_range_detailed(range, base));
    }

    match ctx {
        CubeclContext::Wgpu { client, .. } => {
            detailed_impl(client, range, base, wide_chunk_for(client)).await
        }
        #[cfg(feature = "cubecl-cuda")]
        CubeclContext::Cuda { client, .. } => {
            detailed_impl(client, range, base, wide_chunk_for(client)).await
        }
        #[cfg(feature = "cubecl-hip")]
        CubeclContext::Hip { client, .. } => {
            detailed_impl(client, range, base, wide_chunk_for(client)).await
        }
    }
}

/// Which chunk-scan flavor a client should run.
///
/// The wide flavor divides limb pairs with 64-bit arithmetic and is the
/// CUDA-native form. On wgpu it is *legal* wherever the device exposes
/// `u64` and the shader goes through one of `CubeCL`'s direct compilers
/// (`wgpu<spirv>` / `wgpu<msl>`; naga's MSL backend miscompiles its checked
/// u64 division), but legal is not fast: on an Apple M4 under `wgpu<msl>`
/// the wide flavor measured 4.6x *slower* than split16 (64-bit integer
/// division is emulated on Apple GPUs). So wgpu defaults to split16 and
/// `NICE_CUBECL_WIDE=1` opts in for A/B runs on devices where it is legal;
/// `NICE_CUBECL_WIDE=0` forces split16 anywhere. A forced wide flavor on a
/// device without u64 fails at shader compile time, loudly.
fn wide_chunk_for<R: cubecl::prelude::Runtime>(client: &cubecl::prelude::ComputeClient<R>) -> bool {
    let name = R::name(client);
    let cuda = name.contains("cuda");
    let direct = name.contains("spirv") || name.contains("msl") || name == "hip";
    let u64_ok = client
        .properties()
        .features
        .supports_type(cubecl::ir::Type::scalar(cubecl::ir::ElemType::UInt(
            cubecl::ir::UIntKind::U64,
        )));
    let wide = match std::env::var("NICE_CUBECL_WIDE").ok().as_deref() {
        Some("0") => false,
        Some(_) => true,
        None => cuda,
    };
    if wide && !cuda && !(direct && u64_ok) {
        warn!(
            "NICE_CUBECL_WIDE=1 on {name} (direct compiler {direct}, u64 {u64_ok}): \
             the wide chunk scan is not expected to compile here"
        );
    }
    debug!("CubeCL chunk flavor on {name}: wide {wide} (direct compiler {direct}, u64 {u64_ok})");
    wide
}

/// Read one histogram buffer and fold its bins into the accumulator.
///
/// An async fn rather than a closure (async closures aren't stable), and
/// `read_async` rather than `read_one` so [`detailed_impl`] stays runnable
/// on wasm, where a future cannot block.
pub(crate) async fn drain<R: cubecl::prelude::Runtime>(
    client: &cubecl::prelude::ComputeClient<R>,
    handle: cubecl::server::Handle,
    histogram: &mut [u128],
) -> Result<()> {
    let bytes = client
        .read_async(vec![handle])
        .await
        .map_err(|e| anyhow::anyhow!("histogram read failed: {e:?}"))?
        .remove(0);
    let bins = u32::from_bytes(&bytes);
    for (acc, &bin) in histogram.iter_mut().zip(bins.iter()) {
        *acc += u128::from(bin);
    }
    Ok(())
}

/// Runtime-generic body of [`process_range_detailed_cubecl`].
#[allow(clippy::too_many_lines)]
async fn detailed_impl<R: cubecl::prelude::Runtime>(
    client: &cubecl::prelude::ComputeClient<R>,
    range: &FieldSize,
    base: u32,
    wide_chunk: bool,
) -> Result<FieldResults> {
    /// Batches launched between blocking histogram drains; the overflow
    /// bound is checked below.
    const DRAIN_INTERVAL: usize = 64;
    const _: () = assert!((DRAIN_INTERVAL as u128) * CUBECL_BATCH_SIZE < u32::MAX as u128);

    let start_time = Instant::now();
    let limbs = n_limbs(base).with_context(|| format!("base {base} has no u128 range"))?;
    let (chunk_digits, chunk_div) = if wide_chunk {
        crate::gpu_config::chunk_constants(base)
    } else {
        chunk_constants_u16(base)
    };
    let cutoff = get_near_miss_cutoff(base);
    let hist_bins = (base + 1) as usize;

    let miss_count_handle = client.create(cubecl::bytes::Bytes::from_elems(vec![0u32; 1]));
    let miss_data_handle =
        client.empty(NEAR_MISS_CAPACITY * MISS_STRIDE as usize * core::mem::size_of::<u32>());

    let mut histogram = vec![0u128; hist_bins];

    // Batches launch back-to-back asynchronously — the blocking histogram
    // read is the only stream sync, so doing it per batch put a GPU-idle
    // bubble after every ~12ms of work (~25% whole-field on an RTX 4060,
    // the hand-CUDA backend's entire lead). Each candidate increments
    // exactly one u32 bin, so a bin grows by at most CUBECL_BATCH_SIZE per
    // batch and one drain per DRAIN_INTERVAL batches stays overflow-safe:
    // 64 * 50e6 < u32::MAX. (The hand-CUDA kernel sidesteps this with u64
    // bins; WGSL has no u64 atomics, and this host loop serves both.)
    let mut hist_handle = client.create(cubecl::bytes::Bytes::from_elems(vec![0u32; hist_bins]));
    let mut undrained = 0usize;

    for batch in range.chunks(CUBECL_BATCH_SIZE) {
        let start = batch.start();
        let count = batch.size() as u64;
        let cubes = u32::try_from(count.div_ceil(u64::from(WORKGROUP_SIZE)))
            .unwrap_or(MAX_CUBES)
            .clamp(1, MAX_CUBES);

        unsafe {
            detailed_kernel::launch_unchecked::<R>(
                client,
                CubeCount::Static(cubes, 1, 1),
                CubeDim::new_1d(WORKGROUP_SIZE),
                ArrayArg::from_raw_parts(hist_handle.clone(), hist_bins),
                ArrayArg::from_raw_parts(miss_count_handle.clone(), 1),
                ArrayArg::from_raw_parts(
                    miss_data_handle.clone(),
                    NEAR_MISS_CAPACITY * MISS_STRIDE as usize,
                ),
                start as u32,
                (start >> 32) as u32,
                (start >> 64) as u32,
                (start >> 96) as u32,
                count as u32,
                (count >> 32) as u32,
                NEAR_MISS_CAPACITY as u32,
                base,
                limbs,
                chunk_digits,
                chunk_div,
                cutoff,
                wide_chunk,
            );
        }

        // Submit this batch on its own, because a GPU driver kills a
        // submission that runs too long. CubeCL aggregates dispatches into one
        // command buffer and submits only at `CUBECL_WGPU_MAX_TASKS` tasks (32
        // by default) or at the next read — a bound tuned for ML kernels that
        // run in microseconds. This one is a persistent grid-stride kernel over
        // a whole `CUBECL_BATCH_SIZE`, ~0.12 s per dispatch at base 50 on an
        // AMD 860M, so without this flush a single submission holds the *whole
        // field*: `DETAILED_SEARCH_MAX_FIELD_SIZE` is 20 batches, which is
        // 2.4 s at base 50 and 6.3 s at base 80. Measured on Linux/RADV,
        // amdgpu resets a gfx job at ~2 s ("ring gfx_0.0.0 timeout ... device
        // wedged") and the device is lost for the rest of the process; Windows
        // has the same 2 s TDR, and only survives because the GPUs it was
        // tested on run a field inside it. So the task cap is not the bound
        // that matters — the field is — and raising or lowering it does not
        // help.
        //
        // Flushing per batch is the granularity the hand Vulkan backend has
        // always used, and it is free: `flush` submits without waiting, and one
        // batch already saturates the device (measured interleaved, 80 batches
        // at base 50: 9.76 s median against 10.00 s for the aggregated form).
        client
            .flush()
            .map_err(|e| anyhow::anyhow!("stream flush failed: {e:?}"))?;

        undrained += 1;
        if undrained == DRAIN_INTERVAL {
            drain(client, hist_handle, &mut histogram).await?;
            hist_handle = client.create(cubecl::bytes::Bytes::from_elems(vec![0u32; hist_bins]));
            undrained = 0;
        }
    }
    drain(client, hist_handle, &mut histogram).await?;

    // Every candidate lands in exactly one bin, so a mismatch means the
    // device silently did nothing for some batch — refuse to report
    // fabricated zeros. Observed failure modes differ per runtime, so the
    // hint does too: a CUDA runtime without NVRTC, or a wgpu device the
    // driver's watchdog reset.
    let counted: u128 = histogram.iter().sum();
    let hint = if R::name(client).contains("cuda") {
        "is the CUDA toolkit, including NVRTC, installed?"
    } else {
        "was the device reset by the driver's watchdog? check the kernel log"
    };
    ensure!(
        counted == range.size(),
        "GPU histogram counted {counted} of {} candidates; \
         the device dropped work ({hint})",
        range.size()
    );

    // Near misses.
    let bytes = client
        .read_async(vec![miss_count_handle.clone()])
        .await
        .map_err(|e| anyhow::anyhow!("miss count read failed: {e:?}"))?
        .remove(0);
    let miss_total = u32::from_bytes(&bytes)[0] as usize;
    ensure!(
        miss_total <= NEAR_MISS_CAPACITY,
        "near-miss buffer overflow: {miss_total} > {NEAR_MISS_CAPACITY}"
    );
    let bytes = client
        .read_async(vec![miss_data_handle])
        .await
        .map_err(|e| anyhow::anyhow!("miss data read failed: {e:?}"))?
        .remove(0);
    let words = u32::from_bytes(&bytes);
    let mut nice_numbers: Vec<NiceNumberSimple> = (0..miss_total)
        .map(|i| {
            let o = i * MISS_STRIDE as usize;
            let number = u128::from(words[o])
                | (u128::from(words[o + 1]) << 32)
                | (u128::from(words[o + 2]) << 64)
                | (u128::from(words[o + 3]) << 96);
            NiceNumberSimple {
                number,
                num_uniques: words[o + 4],
            }
        })
        .collect();
    nice_numbers.sort_by_key(|n| n.number);

    let distribution: Vec<UniquesDistributionSimple> = (1..=base)
        .map(|i| UniquesDistributionSimple {
            num_uniques: i,
            count: histogram[i as usize],
        })
        .collect();

    #[allow(clippy::cast_precision_loss)]
    {
        let secs = start_time.elapsed().as_secs_f64();
        debug!(
            "CubeCL detailed b{base}: {:.2e} numbers in {secs:.2}s ({:.2e} n/s), {} near-misses",
            range.size() as f64,
            range.size() as f64 / secs,
            nice_numbers.len(),
        );
    }

    Ok(FieldResults {
        distribution,
        nice_numbers,
    })
}

/// `CubeCL` implementation of `process_range_niceonly`.
///
/// Runs the MSD prefix filter on the CPU (all cores) and checks the surviving
/// ranges' stride-valid candidates on the GPU, which reconstructs them from
/// the residue table on-device. Produces the exact same nice-number set as the
/// CPU path: the coarser MSD floor makes the GPU's candidate set a *superset*,
/// and the per-candidate check is identical.
///
/// **Range semantics**: half-open [`range_start`, `range_end`).
///
/// # Errors
/// Returns an error on any device failure or if the output buffer overflows.
/// Start one niceonly field: hand it to the continuous pipeline, or answer
/// it on the spot for a base the device cannot take. Pair with
/// [`finish_niceonly_cubecl`], which returns fields in the order they were
/// begun.
///
/// # Errors
/// Returns an error if the pipeline's dispatch thread has died.
///
/// # Panics
/// Panics if the pipeline mutex was poisoned by an earlier panic.
pub fn begin_niceonly_cubecl(
    ctx: &CubeclContext,
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
    let software = is_software_rasterizer(&ctx.device_name());
    match ctx {
        CubeclContext::Wgpu {
            client,
            niceonly_plans,
            niceonly_pipeline,
            ..
        } => begin_impl(
            client,
            niceonly_plans,
            niceonly_pipeline,
            software,
            range,
            base,
        ),
        #[cfg(feature = "cubecl-cuda")]
        CubeclContext::Cuda {
            client,
            niceonly_plans,
            niceonly_pipeline,
            ..
        } => begin_impl(
            client,
            niceonly_plans,
            niceonly_pipeline,
            software,
            range,
            base,
        ),
        #[cfg(feature = "cubecl-hip")]
        CubeclContext::Hip {
            client,
            niceonly_plans,
            ..
        } => niceonly_impl(
            client,
            cached_plan(niceonly_plans, client, base, software)?,
            range,
            base,
            wide_chunk_for(client),
        ),
    }
}

/// Wait for the oldest field begun with [`begin_niceonly_cubecl`] that went
/// into the pipeline, and return its results.
///
/// # Errors
/// The field's device error, or an output buffer overflow.
///
/// # Panics
/// Panics if the pipeline mutex was poisoned by an earlier panic.
pub fn finish_niceonly_cubecl(ctx: &CubeclContext) -> Result<(FieldResults, NiceonlyStats)> {
    match ctx {
        CubeclContext::Wgpu {
            niceonly_pipeline, ..
        } => finish_impl(niceonly_pipeline),
        #[cfg(feature = "cubecl-cuda")]
        CubeclContext::Cuda {
            niceonly_pipeline, ..
        } => finish_impl(niceonly_pipeline),
    }
}

/// One field, begun and finished: the synchronous form, for callers that
/// process a single field at a time (the benchmark sweep, tests).
///
/// # Errors
/// See [`begin_niceonly_cubecl`] and [`finish_niceonly_cubecl`].
///
/// # Panics
/// Panics if the pipeline mutex was poisoned by an earlier panic.
pub fn process_range_niceonly_cubecl(
    ctx: &CubeclContext,
    range: &FieldSize,
    base: u32,
) -> Result<FieldResults> {
    match begin_niceonly_cubecl(ctx, range, base)? {
        NiceonlyStarted::Immediate(results) => Ok(results),
        NiceonlyStarted::Queued => finish_niceonly_cubecl(ctx).map(|(results, _)| results),
    }
}

/// Runtime-generic body of [`begin_niceonly_cubecl`].
fn begin_impl<R: cubecl::prelude::Runtime>(
    client: &cubecl::prelude::ComputeClient<R>,
    plans: &Arc<Mutex<HashMap<u32, Arc<NiceonlyPlan>>>>,
    pipeline: &Mutex<Option<NiceonlyPipeline<CubeclPendingField<R>>>>,
    software: bool,
    range: &FieldSize,
    base: u32,
) -> Result<NiceonlyStarted> {
    let mut guard = pipeline.lock().unwrap();
    let pipeline = guard.get_or_insert_with(|| {
        NiceonlyPipeline::start(
            "CubeCL",
            CubeclNiceonlySink {
                client: client.clone(),
                plans: plans.clone(),
                wide_chunk: wide_chunk_for(client),
                software,
                open: HashMap::new(),
                inflight: std::collections::VecDeque::new(),
            },
        )
    });
    pipeline.push(base, range)?;
    Ok(NiceonlyStarted::Queued)
}

/// Runtime-generic body of [`finish_niceonly_cubecl`].
fn finish_impl<R: cubecl::prelude::Runtime>(
    pipeline: &Mutex<Option<NiceonlyPipeline<CubeclPendingField<R>>>>,
) -> Result<(FieldResults, NiceonlyStats)> {
    let mut guard = pipeline.lock().unwrap();
    let pipeline = guard
        .as_mut()
        .ok_or_else(|| anyhow::anyhow!("no niceonly field has been begun"))?;
    let (stats, nice_numbers) = pipeline.next_result()?;
    debug!(
        "CubeCL niceonly pipeline: {} ranges in {} dispatches, found {}",
        stats.num_ranges,
        stats.launches,
        nice_numbers.len()
    );
    Ok((
        FieldResults {
            distribution: Vec::new(),
            nice_numbers,
        },
        stats,
    ))
}

/// A fence on one launched batch: the client's `sync` future taken right
/// after the launch and polled once on the spot. The flush is eager, but on
/// wgpu the submitted-work-done callback is registered inside the future's
/// first poll, and such a callback covers everything submitted *at
/// registration*; polling immediately pins it to the work queued up to this
/// launch, so awaiting it later does not also wait for whatever was launched
/// since.
type LaunchFence = cubecl::future::DynFut<Result<(), cubecl::server::ServerError>>;

/// Take a fence on everything the client has queued so far. `None` if it
/// resolved on the spot (nothing pending).
fn launch_fence<R: cubecl::prelude::Runtime>(
    client: &cubecl::prelude::ComputeClient<R>,
) -> Result<Option<LaunchFence>> {
    use std::task::{Context, Poll, Waker};
    let mut fence = client.sync();
    let mut cx = Context::from_waker(Waker::noop());
    match fence.as_mut().poll(&mut cx) {
        Poll::Ready(r) => {
            r.map_err(|e| anyhow::anyhow!("launch fence failed: {e:?}"))?;
            Ok(None)
        }
        Poll::Pending => Ok(Some(fence)),
    }
}

/// The device end of the niceonly pipeline for a `CubeCL` runtime: one
/// [`CubeclNiceonlyRun`] per open field, plus the launch count that bounds
/// the work in flight.
struct CubeclNiceonlySink<R: cubecl::prelude::Runtime> {
    client: cubecl::prelude::ComputeClient<R>,
    plans: Arc<Mutex<HashMap<u32, Arc<NiceonlyPlan>>>>,
    wide_chunk: bool,
    /// The adapter is a software rasterizer (plans then keep the plane-scoped
    /// queue off — see [`is_software_rasterizer`]).
    software: bool,
    open: HashMap<u64, CubeclNiceonlyRun<R>>,
    /// One fence per launched batch, oldest first: the `CubeCL` analogue of
    /// the CUDA sink's event ring. Before the ring exceeds
    /// [`batches_in_flight`] the oldest fence is awaited, so the dispatch
    /// thread blocks for exactly one batch at a time and the descriptor
    /// channel keeps draining. A full `client.sync()` every N launches was
    /// tried first and is the wrong shape: it parks the dispatch thread for N
    /// batches while the channel fills and the MSD workers park unseen, so
    /// "device behind" reads as saturated and the floor controller drove
    /// the floor to its minimum on a 9070 XT and an M4 (a third of the
    /// pinned throughput).
    inflight: std::collections::VecDeque<LaunchFence>,
}

impl<R: cubecl::prelude::Runtime> RangeSink for CubeclNiceonlySink<R> {
    type Pending = CubeclPendingField<R>;

    fn begin_field(&mut self, seq: u64, base: u32, range: &FieldSize) -> Result<()> {
        let plan = cached_plan(&self.plans, &self.client, base, self.software)?;
        let run =
            CubeclNiceonlyRun::from_plan(&self.client, plan, base, range.start(), self.wide_chunk)?;
        self.open.insert(seq, run);
        Ok(())
    }

    fn launch(&mut self, field: u64, offsets: &[u64], lens: &[u32], masks: &[u64]) -> Result<()> {
        let run = self
            .open
            .get_mut(&field)
            .ok_or_else(|| anyhow::anyhow!("launch for a field that is not open ({field})"))?;
        // Backpressure: wait for the oldest batch before adding another,
        // once `batches_in_flight` are outstanding.
        while self.inflight.len() >= batches_in_flight() {
            if let Some(oldest) = self.inflight.pop_front() {
                cubecl::future::block_on(oldest)
                    .map_err(|e| anyhow::anyhow!("launch fence failed: {e:?}"))?;
            }
        }
        run.launch(field, offsets, lens, masks)?;
        if let Some(fence) = launch_fence(&self.client)? {
            self.inflight.push_back(fence);
        }
        Ok(())
    }

    fn end_field(&mut self, seq: u64) -> Result<Self::Pending> {
        let mut run = self
            .open
            .remove(&seq)
            .ok_or_else(|| anyhow::anyhow!("end of a field that is not open ({seq})"))?;
        run.end_field(seq)
    }
}

/// Whether a wgpu adapter name denotes a software rasterizer. CI runs the
/// device tests on lavapipe, and llvmpipe mishandles a subgroup-scope
/// control barrier inside plane-divergent control flow (the plane-scoped
/// compaction queue returned 2 of 283 survivors there, and was exact with
/// the barriers removed), so that path stays off on such adapters.
pub(crate) fn is_software_rasterizer(device: &str) -> bool {
    let squashed = device.to_lowercase().replace(' ', "");
    ["llvmpipe", "lavapipe", "swiftshader", "softwarerasterizer"]
        .iter()
        .any(|s| squashed.contains(s))
}

/// Fetch the base's plan from the context cache, building it on first use.
///
/// # Panics
/// Panics if the cache mutex was poisoned by an earlier panic.
fn cached_plan<R: cubecl::prelude::Runtime>(
    plans: &Mutex<HashMap<u32, Arc<NiceonlyPlan>>>,
    client: &cubecl::prelude::ComputeClient<R>,
    base: u32,
    software: bool,
) -> Result<Arc<NiceonlyPlan>> {
    if let Some(plan) = plans.lock().unwrap().get(&base) {
        return Ok(plan.clone());
    }
    // Built outside the lock: seconds of CPU work for a large modulus, and
    // another thread asking for a *different* base should not wait on it.
    // A racing build of the same base wastes one table walk, harmlessly.
    let plan = Arc::new(NiceonlyPlan::build(client, base, software)?);
    Ok(plans.lock().unwrap().entry(base).or_insert(plan).clone())
}

/// Per-base niceonly state that survives across fields: the stride-table
/// constants and the residue table already on the device. The analog of the
/// CUDA backend's cached `NiceonlyPlan` and the Vulkan pipeline cache.
///
/// Building this is *expensive*: `StrideTable::new` walks the whole modulus
/// on one CPU thread (8.6e6 steps at base 52). Rebuilt per field, that walk
/// dominated the pipeline at benchmark-window field sizes — measured on an
/// RTX 4060, half of every b50 scenario's wall time was table rebuilds the
/// hand backends never do.
// Public because it names a field type of the public context enum; its own
// fields stay private, so nothing outside this module can touch it.
pub struct NiceonlyPlan {
    stride_m: u32,
    stride_r: u32,
    residues: cubecl::server::Handle,
    prefilter: Option<VulkanPrefilterParams>,
    /// Per-residue exact low-digit masks on device as (lo, hi) u32 words,
    /// for the cross-end residue filter; a 2-word dummy when `cross` is off.
    low_masks: cubecl::server::Handle,
    cross: bool,
    /// Plane-compact the filter's survivors before checking them
    /// (`NICE_CUBECL_COMPACT=0` opts out; requires `cross`).
    compact: bool,
    /// Use the plane-scoped compaction queue instead of the cube-scoped one
    /// (default on under `wgpu<spirv>` and `cuda`; `NICE_CUBECL_PLANE_COMPACT=0|1`
    /// overrides; requires `compact` and a device with plane barriers).
    plane_compact: bool,
}

impl NiceonlyPlan {
    /// # Errors
    /// Returns an error for an unconfigurable base. Residue-empty bases must
    /// be short-circuited by the caller before any stride table is built.
    fn build<R: cubecl::prelude::Runtime>(
        client: &cubecl::prelude::ComputeClient<R>,
        base: u32,
        software: bool,
    ) -> Result<Self> {
        let table = StrideTable::new(base, GPU_LSD_K);
        ensure!(
            !table.valid_residues.is_empty(),
            "no valid stride residues for base {base} (residue-empty base?)"
        );
        ensure!(
            table.modulus <= MAX_STRIDE_MODULUS,
            "stride modulus {} exceeds the kernel's {MAX_STRIDE_MODULUS} bound for base {base}",
            table.modulus
        );
        #[allow(clippy::cast_possible_truncation)]
        let stride_m = table.modulus as u32;
        let stride_r = u32::try_from(table.valid_residues.len())
            .with_context(|| format!("residue count overflows u32 for base {base}"))?;
        // Cross-end residue filter: on by default wherever the low-mask
        // table exists (base <= 64); NICE_CUBECL_CROSS=0 opts out for A/B.
        let cross = !table.low_digit_masks.is_empty()
            && std::env::var("NICE_CUBECL_CROSS").map_or(true, |v| v != "0");
        // Compaction needs plane scan/reduce ops; adapters without them
        // (some older wgpu targets) fall back to the naive skip.
        let plane_ok = client
            .properties()
            .features
            .plane
            .contains(cubecl::ir::features::Plane::Ops);
        let compact =
            cross && plane_ok && std::env::var("NICE_CUBECL_COMPACT").map_or(true, |v| v != "0");
        let plane_sync = client
            .properties()
            .features
            .plane
            .contains(cubecl::ir::features::Plane::Sync);
        // Plane-scoped queue: on by default where it is measured to win —
        // CubeCL's direct SPIR-V compiler on a real Vulkan device (+18-22%
        // on every niceonly scenario on an RX 9070 XT) and the native CUDA
        // runtime (+9-18% on an RTX 3090 once the device is the bottleneck,
        // neutral while the CPU MSD filter is). Off on Metal (b50+ loses),
        // off on software rasterizers (see `is_software_rasterizer`), and
        // opt-in on anything else. `NICE_CUBECL_PLANE_COMPACT=0|1`
        // overrides, except on software rasterizers where the variant
        // cannot run correctly.
        let name = R::name(client);
        let plane_default = name.contains("spirv") || name.contains("cuda");
        let plane_compact = compact
            && plane_sync
            && !software
            && match std::env::var("NICE_CUBECL_PLANE_COMPACT").ok().as_deref() {
                Some("0") => false,
                Some(_) => true,
                None => plane_default,
            };
        debug!(
            "CubeCL niceonly plan base {base}: cross {cross}, compact {compact} \
             (plane ops {plane_ok}), plane-scoped {plane_compact} (plane sync {plane_sync})"
        );
        #[allow(clippy::cast_possible_truncation)]
        let mask_words: Vec<u32> = if cross {
            table
                .low_digit_masks
                .iter()
                .flat_map(|&m| [m as u32, (m >> 32) as u32])
                .collect()
        } else {
            vec![0u32; 2]
        };
        let low_masks = client.create(cubecl::bytes::Bytes::from_elems(mask_words));
        let residues = client.create(cubecl::bytes::Bytes::from_elems(table.valid_residues));
        Ok(Self {
            stride_m,
            stride_r,
            residues,
            prefilter: vulkan_prefilter_params(base),
            low_masks,
            cross,
            compact,
            plane_compact,
        })
    }
}

/// The `CubeCL` end of the niceonly range pipeline: the base's cached plan
/// plus this field's output buffers, kept across every dispatch of a field.
struct CubeclNiceonlyRun<R: cubecl::prelude::Runtime> {
    client: cubecl::prelude::ComputeClient<R>,
    plan: Arc<NiceonlyPlan>,
    nice_out: cubecl::server::Handle,
    nice_count: cubecl::server::Handle,
    field_start: u128,
    /// `field_start mod M`, so the kernel only reduces the 64-bit offset.
    fs_mod_m: u32,
    base: u32,
    limbs: u32,
    chunk_digits: u32,
    chunk_div: u32,
    wide_chunk: bool,
    /// Report prefilter survivors instead of nice numbers (device tests only).
    probe: bool,
    /// Pin the lane tiling instead of sizing it per dispatch (device tests).
    lane_shift_override: Option<u32>,
    /// Force the compact specialization on or off (device tests).
    #[cfg(test)]
    compact_override: Option<bool>,
    /// Force the plane-scoped queue on or off (device tests).
    #[cfg(test)]
    plane_override: Option<bool>,
}

impl<R: cubecl::prelude::Runtime> CubeclNiceonlyRun<R> {
    /// Build a fresh, uncached plan and wrap a run around it (device tests).
    #[cfg(test)]
    fn new(
        client: &cubecl::prelude::ComputeClient<R>,
        base: u32,
        field_start: u128,
        wide_chunk: bool,
    ) -> Result<Self> {
        let plan = Arc::new(NiceonlyPlan::build(client, base, false)?);
        Self::from_plan(client, plan, base, field_start, wide_chunk)
    }

    /// Per-field setup over a cached plan: cheap — two buffer creations.
    ///
    /// # Errors
    /// Returns an error for a base with no u128 range.
    fn from_plan(
        client: &cubecl::prelude::ComputeClient<R>,
        plan: Arc<NiceonlyPlan>,
        base: u32,
        field_start: u128,
        wide_chunk: bool,
    ) -> Result<Self> {
        let limbs = n_limbs(base).with_context(|| format!("base {base} has no u128 range"))?;
        let (chunk_digits, chunk_div) = if wide_chunk {
            crate::gpu_config::chunk_constants(base)
        } else {
            chunk_constants_u16(base)
        };
        let nice_out = client.create(cubecl::bytes::Bytes::from_elems(vec![
            0u32;
            NICEONLY_OUT_CAPACITY
                * NICEONLY_STRIDE
                    as usize
        ]));
        let nice_count = client.create(cubecl::bytes::Bytes::from_elems(vec![0u32; 1]));

        #[allow(clippy::cast_possible_truncation)]
        let fs_mod_m = (field_start % u128::from(plan.stride_m)) as u32;
        Ok(Self {
            client: client.clone(),
            plan,
            nice_out,
            nice_count,
            field_start,
            fs_mod_m,
            base,
            limbs,
            chunk_digits,
            chunk_div,
            wide_chunk,
            probe: false,
            lane_shift_override: None,
            #[cfg(test)]
            compact_override: None,
            #[cfg(test)]
            plane_override: None,
        })
    }

    /// Collect the nice numbers found across the whole field (device tests).
    #[cfg(test)]
    fn finish(&self) -> Result<Vec<NiceNumberSimple>> {
        read_niceonly_hits(&self.client, &self.nice_count, &self.nice_out, self.base)
    }
}

/// Read a field's hits back. The blocking reads double as the device wait: a
/// dispatch still in flight has simply not written its hits yet, and the
/// failure mode is *silently missing solutions* — so the read path orders
/// after every launch on the client's queue.
///
/// # Errors
/// Returns an error if the kernel tried to write more hits than the buffer
/// holds — which, given how rare nice numbers are, means a kernel bug rather
/// than a genuine flood.
fn read_niceonly_hits<R: cubecl::prelude::Runtime>(
    client: &cubecl::prelude::ComputeClient<R>,
    nice_count: &cubecl::server::Handle,
    nice_out: &cubecl::server::Handle,
    base: u32,
) -> Result<Vec<NiceNumberSimple>> {
    let bytes = client
        .read_one(nice_count.clone())
        .map_err(|e| anyhow::anyhow!("nice count read failed: {e:?}"))?;
    let written = u32::from_bytes(&bytes)[0] as usize;
    ensure!(
        written <= NICEONLY_OUT_CAPACITY,
        "niceonly output buffer overflow: {written} > {NICEONLY_OUT_CAPACITY} \
         (this strongly suggests a kernel bug)"
    );
    let bytes = client
        .read_one(nice_out.clone())
        .map_err(|e| anyhow::anyhow!("nice out read failed: {e:?}"))?;
    let words = u32::from_bytes(&bytes);
    let mut hits: Vec<NiceNumberSimple> = (0..written)
        .map(|i| {
            let o = i * NICEONLY_STRIDE as usize;
            let lo = u128::from(words[o]) | (u128::from(words[o + 1]) << 32);
            let hi = u128::from(words[o + 2]) | (u128::from(words[o + 3]) << 32);
            NiceNumberSimple {
                number: (hi << 64) | lo,
                num_uniques: base,
            }
        })
        .collect();
    hits.sort_by_key(|n| n.number);
    Ok(hits)
}

/// Certificates as (lo, hi) u32 words, matching the offset encoding; a
/// 2-word dummy when the cross filter is off.
#[allow(clippy::cast_possible_truncation)]
fn certificate_words(masks: &[u64], cross: bool) -> Vec<u32> {
    if cross {
        masks
            .iter()
            .flat_map(|&m| [m as u32, (m >> 32) as u32])
            .collect()
    } else {
        vec![0u32; 2]
    }
}

impl<R: cubecl::prelude::Runtime> CubeclNiceonlyRun<R> {
    /// Whether this dispatch should use the compacted specialization. The
    /// no-MSD bypass ships all-zero certificates; the compacted kernel can
    /// only cost there (queue traffic and two barriers per iteration for a
    /// filter that never fires), so such dispatches take the plain path.
    /// Mixed batches keep compaction.
    fn dispatch_compact(&self, masks: &[u64]) -> bool {
        #[cfg(test)]
        if let Some(forced) = self.compact_override {
            return forced;
        }
        self.plan.compact && masks.iter().any(|&m| m != 0)
    }

    /// Whether a compacted dispatch uses the plane-scoped queue.
    fn dispatch_plane_scoped(&self, compact: bool) -> bool {
        #[cfg(test)]
        if let Some(forced) = self.plane_override {
            return compact && forced;
        }
        compact && self.plan.plane_compact
    }
}

/// A closed field's device work on a `CubeCL` runtime: the handles its hits
/// land in. Waited for on the client's thread; the blocking reads order after
/// every launch on the client's queue.
pub struct CubeclPendingField<R: cubecl::prelude::Runtime> {
    client: cubecl::prelude::ComputeClient<R>,
    nice_out: cubecl::server::Handle,
    nice_count: cubecl::server::Handle,
    base: u32,
}

impl<R: cubecl::prelude::Runtime> PendingField for CubeclPendingField<R> {
    fn wait(self: Box<Self>) -> Result<DeviceResult> {
        Ok(DeviceResult {
            nice_numbers: read_niceonly_hits(
                &self.client,
                &self.nice_count,
                &self.nice_out,
                self.base,
            )?,
            // No per-submission timing on this runtime without device
            // timestamp queries.
            device_busy_secs: None,
        })
    }
}

impl<R: cubecl::prelude::Runtime> RangeSink for CubeclNiceonlyRun<R> {
    type Pending = CubeclPendingField<R>;

    /// A run is one field; opening is what constructing it did.
    fn begin_field(&mut self, _seq: u64, _base: u32, _range: &FieldSize) -> Result<()> {
        Ok(())
    }

    fn end_field(&mut self, _seq: u64) -> Result<Self::Pending> {
        Ok(CubeclPendingField {
            client: self.client.clone(),
            nice_out: self.nice_out.clone(),
            nice_count: self.nice_count.clone(),
            base: self.base,
        })
    }

    fn launch(&mut self, _field: u64, offsets: &[u64], lens: &[u32], masks: &[u64]) -> Result<()> {
        ensure!(
            offsets.len() == lens.len() && offsets.len() == masks.len(),
            "range descriptor slices have mismatched lengths ({}/{}/{})",
            offsets.len(),
            lens.len(),
            masks.len()
        );
        // Pack offsets as lo/hi u32 pairs; buffers are created per dispatch
        // and sized to the batch (CubeCL pools the allocations). These
        // per-dispatch writes also flush the stream, so each dispatch is
        // submitted on its own — the same watchdog protection detailed mode
        // gets from its explicit `flush`. If these buffers are ever hoisted
        // or pooled to save the allocation, add that flush here.
        #[allow(clippy::cast_possible_truncation)]
        let pairs: Vec<u32> = offsets
            .iter()
            .flat_map(|&o| [o as u32, (o >> 32) as u32])
            .collect();
        let offsets_handle = self.client.create(cubecl::bytes::Bytes::from_elems(pairs));
        let lens_handle = self
            .client
            .create(cubecl::bytes::Bytes::from_elems(lens.to_vec()));
        let mask_words = certificate_words(masks, self.plan.cross);
        let masks_len = mask_words.len();
        let compact = self.dispatch_compact(masks);
        let masks_handle = self
            .client
            .create(cubecl::bytes::Bytes::from_elems(mask_words));

        // Tile the dispatch to this batch's ranges; batches are homogeneous
        // enough for the mean to be a good summary, because the MSD recursion
        // bounds range length by the floor.
        let mean_len = lens.iter().map(|&l| u64::from(l)).sum::<u64>() / lens.len().max(1) as u64;
        // NICE_CUBECL_LANES pins the tiling for A/B measurement, the same
        // shape of knob as NICE_VULKAN_LANES on the hand backend.
        let lane_shift = self
            .lane_shift_override
            .or_else(|| {
                std::env::var("NICE_CUBECL_LANES")
                    .ok()?
                    .parse::<u32>()
                    .ok()
                    .filter(|n| n.is_power_of_two() && *n <= 32)
                    .map(u32::trailing_zeros)
            })
            .unwrap_or_else(|| {
                lane_shift_for(
                    offsets.len() as u64,
                    mean_len,
                    self.plan.stride_m,
                    self.plan.stride_r,
                )
            });
        let num_ranges = u32::try_from(offsets.len()).unwrap_or(u32::MAX);
        let threads = u64::from(num_ranges) << lane_shift;
        #[allow(clippy::cast_possible_truncation)]
        let cubes = threads
            .div_ceil(u64::from(WORKGROUP_SIZE))
            .clamp(1, u64::from(MAX_CUBES)) as u32;

        let (pre_limbs, pre_chunk_digits, pre_chunk_div) = self
            .plan
            .prefilter
            .map_or((0, 0, 0), |p| (p.limbs, p.chunk_digits, p.chunk_div));

        #[allow(clippy::cast_possible_truncation)]
        unsafe {
            niceonly_kernel::launch_unchecked::<R>(
                &self.client,
                CubeCount::Static(cubes, 1, 1),
                CubeDim::new_1d(WORKGROUP_SIZE),
                ArrayArg::from_raw_parts(self.plan.residues.clone(), self.plan.stride_r as usize),
                ArrayArg::from_raw_parts(offsets_handle, offsets.len() * 2),
                ArrayArg::from_raw_parts(lens_handle, lens.len()),
                ArrayArg::from_raw_parts(
                    self.nice_out.clone(),
                    NICEONLY_OUT_CAPACITY * NICEONLY_STRIDE as usize,
                ),
                ArrayArg::from_raw_parts(self.nice_count.clone(), 1),
                ArrayArg::from_raw_parts(
                    self.plan.low_masks.clone(),
                    if self.plan.cross {
                        2 * self.plan.stride_r as usize
                    } else {
                        2
                    },
                ),
                ArrayArg::from_raw_parts(masks_handle, masks_len),
                self.field_start as u32,
                (self.field_start >> 32) as u32,
                (self.field_start >> 64) as u32,
                (self.field_start >> 96) as u32,
                self.fs_mod_m,
                num_ranges,
                NICEONLY_OUT_CAPACITY as u32,
                lane_shift,
                self.base,
                self.limbs,
                self.chunk_digits,
                self.chunk_div,
                self.wide_chunk,
                self.plan.stride_m,
                self.plan.stride_r,
                stride_chunk_bits(self.plan.stride_m),
                pre_limbs,
                pre_chunk_digits,
                pre_chunk_div,
                self.probe,
                self.plan.cross,
                compact,
                self.dispatch_plane_scoped(compact),
            );
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The GPU histogram bins are u32; a batch must not be able to overflow one.
    #[test]
    fn batch_size_cannot_overflow_a_u32_bin() {
        assert!(CUBECL_BATCH_SIZE < u128::from(u32::MAX));
    }

    /// CPU/CubeCL parity on the detailed path — the same bases and ranges as
    /// `vulkan_matches_cpu_detailed`, so results are directly comparable.
    /// Runs on lavapipe:
    ///
    /// ```text
    /// VK_ICD_FILENAMES=.../lvp_icd.json \
    ///   cargo test -p nice_common --features cubecl -- --ignored --nocapture
    /// ```
    #[test]
    #[ignore = "requires a wgpu device"]
    fn cubecl_matches_cpu_detailed() {
        let ctx = CubeclContext::new_default().expect("CubeCL init");
        for (base, count) in [
            (10u32, 1_000_000u128),
            (40, 2_000_000),
            (62, 200_000),
            (80, 100_000),
        ] {
            let start = crate::base_range::get_base_range_u128(base)
                .unwrap()
                .unwrap()
                .range_start;
            let range = FieldSize::new(start, start + count);
            let gpu = process_range_detailed_cubecl(&ctx, &range, base).expect("cubecl run");
            let cpu = process_range_detailed(&range, base);
            assert_eq!(
                gpu.distribution, cpu.distribution,
                "base {base}: distribution mismatch"
            );
            assert_eq!(
                gpu.nice_numbers, cpu.nice_numbers,
                "base {base}: near-miss mismatch"
            );
            println!("base {base}: {count} candidates match the CPU exactly");
        }
    }

    /// CPU/CubeCL parity through the native HIP runtime, on real AMD silicon.
    #[test]
    #[cfg(feature = "cubecl-hip")]
    #[ignore = "requires an AMD device with ROCm"]
    fn cubecl_hip_matches_cpu_detailed() {
        let ctx = CubeclContext::new_hip(0).expect("CubeCL HIP init");
        for (base, count) in [
            (10u32, 1_000_000u128),
            (40, 2_000_000),
            (62, 200_000),
            (80, 100_000),
        ] {
            let start = crate::base_range::get_base_range_u128(base)
                .unwrap()
                .unwrap()
                .range_start;
            let range = FieldSize::new(start, start + count);
            let gpu = process_range_detailed_cubecl(&ctx, &range, base).expect("cubecl-hip run");
            let cpu = process_range_detailed(&range, base);
            assert_eq!(
                gpu.distribution, cpu.distribution,
                "base {base}: distribution mismatch"
            );
            assert_eq!(
                gpu.nice_numbers, cpu.nice_numbers,
                "base {base}: near-miss mismatch"
            );
            println!("base {base}: {count} candidates match the CPU exactly (HIP runtime)");
        }
    }

    /// Niceonly parity through the native HIP runtime.
    #[test]
    #[cfg(feature = "cubecl-hip")]
    #[ignore = "requires an AMD device with ROCm"]
    fn cubecl_hip_matches_cpu_niceonly() {
        let ctx = CubeclContext::new_hip(0).expect("CubeCL HIP init");
        for base in [10u32, 12, 25, 40, 45, 62, 80] {
            let Ok(Some(base_range)) = crate::base_range::get_base_range_u128(base) else {
                continue;
            };
            let start = base_range.range_start;
            let end = (start + 5_000_000).min(base_range.range_end);
            let range = FieldSize::new(start, end);

            let table = StrideTable::new(base, GPU_LSD_K);
            let mut cpu = process_range_niceonly(&range, base, &table).nice_numbers;
            cpu.sort_by_key(|n| n.number);
            let gpu = process_range_niceonly_cubecl(&ctx, &range, base).expect("cubecl-hip run");

            assert_eq!(cpu, gpu.nice_numbers, "base {base}: niceonly mismatch");
            println!(
                "base {base}: [{start}, {end}) agrees, {} nice (HIP runtime)",
                gpu.nice_numbers.len()
            );
        }
    }

    /// CPU/CubeCL parity through the native CUDA runtime, on real silicon.
    #[test]
    #[cfg(feature = "cubecl-cuda")]
    #[ignore = "requires an NVIDIA device"]
    fn cubecl_cuda_matches_cpu_detailed() {
        let ctx = CubeclContext::new_cuda(0).expect("CubeCL CUDA init");
        for (base, count) in [
            (10u32, 1_000_000u128),
            (40, 2_000_000),
            (62, 200_000),
            (80, 100_000),
        ] {
            let start = crate::base_range::get_base_range_u128(base)
                .unwrap()
                .unwrap()
                .range_start;
            let range = FieldSize::new(start, start + count);
            let gpu = process_range_detailed_cubecl(&ctx, &range, base).expect("cubecl-cuda run");
            let cpu = process_range_detailed(&range, base);
            assert_eq!(
                gpu.distribution, cpu.distribution,
                "base {base}: distribution mismatch"
            );
            assert_eq!(
                gpu.nice_numbers, cpu.nice_numbers,
                "base {base}: near-miss mismatch"
            );
            println!("base {base}: {count} candidates match the CPU exactly (CUDA runtime)");
        }
    }

    /// The placeholder is only ever reached when the driver cannot answer.
    /// Runs everywhere: an out-of-range ordinal fails before any driver call,
    /// so this pins the fallback contract without needing NVIDIA silicon.
    #[test]
    #[cfg(feature = "cubecl-cuda")]
    fn cuda_device_name_falls_back_when_the_driver_cannot_answer() {
        assert_eq!(
            cuda_device_name(usize::MAX),
            format!("cubecl-cuda device {}", usize::MAX),
        );
    }

    /// On real silicon the reported model must be the driver's name, not the
    /// placeholder — a placeholder is what made every v3.4.3 detailed report
    /// unmatchable for the estimator. Asserting "not the placeholder" rather
    /// than a specific string keeps this runnable on any NVIDIA card.
    #[test]
    #[cfg(feature = "cubecl-cuda")]
    #[ignore = "requires an NVIDIA device"]
    fn cubecl_cuda_reports_the_real_device_name() {
        let name = cuda_device_name(0);
        assert_ne!(
            name, "cubecl-cuda device 0",
            "driver did not name the device; reports would be unattributable"
        );
        assert!(!name.trim().is_empty(), "empty device name");
        // The context must carry the same string the helper resolved, since
        // that is what reaches `hardware.gpu_model`.
        let ctx = CubeclContext::new_cuda(0).expect("CubeCL CUDA init");
        assert_eq!(ctx.device_name(), name);
        assert_eq!(ctx.backend_name(), "cubecl-cuda");
        println!("cubecl-cuda device 0 reports as {name:?}");
    }

    /// The known solution: 69 is nice in base 10.
    #[test]
    #[ignore = "requires a wgpu device"]
    fn cubecl_finds_69_in_base_10() {
        let ctx = CubeclContext::new_default().expect("CubeCL init");
        let range = FieldSize::new(47, 100);
        let results = process_range_detailed_cubecl(&ctx, &range, 10).expect("cubecl run");
        let hit = results
            .nice_numbers
            .iter()
            .find(|n| n.number == 69)
            .expect("69 not found in base 10");
        assert_eq!(hit.num_uniques, 10);
    }

    /// CPU/CubeCL parity on the niceonly path, over the same range with the
    /// same stride table — the same bases and rationale as
    /// `vulkan_matches_cpu_niceonly`. The GPU checks a *superset* of the CPU's
    /// candidates (its MSD floor is coarser), so the nice-number sets must
    /// still be identical.
    /// Throughput of the continuous pipeline on a fixed run of fields on
    /// the default wgpu device, so two configurations can be compared on
    /// identical work. Env as in production (`NICE_GPU_FIELDS_IN_FLIGHT`,
    /// `NICE_GPU_MSD_FLOOR`); `NICE_TEST_FIELDS` (required: it is also the
    /// opt-in, since the parity workflow runs this module's ignored tests on
    /// lavapipe) fields of 1e13 in base 54 from the Anvil region.
    #[test]
    #[ignore = "requires a wgpu device; prints throughput"]
    #[allow(clippy::cast_precision_loss)]
    fn pipeline_throughput_fixed_fields() {
        use crate::gpu_niceonly::{NiceonlyStarted, fields_in_flight, msd_floor_in_use};
        // The parity workflow runs every ignored test in this module on a
        // software rasterizer; a throughput run there is hours of nothing.
        // Only run when asked for by name.
        let Ok(n) = std::env::var("NICE_TEST_FIELDS") else {
            eprintln!("skipping: set NICE_TEST_FIELDS to run the throughput harness");
            return;
        };
        let ctx = CubeclContext::new_default().expect("CubeCL init");
        let base = 54;
        let n: usize = n.parse().expect("NICE_TEST_FIELDS must be a count");
        let start: u128 = 2_778_136_280_153_679_229;
        let size: u128 = 10_000_000_000_000;
        let fields: Vec<FieldSize> = (0..n as u128)
            .map(|i| FieldSize::new(start + i * size, start + (i + 1) * size))
            .collect();
        let warm = FieldSize::new(start - size, start);
        if let NiceonlyStarted::Queued = begin_niceonly_cubecl(&ctx, &warm, base).unwrap() {
            finish_niceonly_cubecl(&ctx).unwrap();
        }
        let lookahead = fields_in_flight().saturating_sub(1);
        let t = std::time::Instant::now();
        let mut queued = 0usize;
        let mut found = 0usize;
        for f in &fields {
            if let NiceonlyStarted::Queued = begin_niceonly_cubecl(&ctx, f, base).unwrap() {
                queued += 1;
            }
            while queued > lookahead {
                found += finish_niceonly_cubecl(&ctx).unwrap().0.nice_numbers.len();
                queued -= 1;
            }
        }
        while queued > 0 {
            found += finish_niceonly_cubecl(&ctx).unwrap().0.nice_numbers.len();
            queued -= 1;
        }
        let secs = t.elapsed().as_secs_f64();
        eprintln!(
            "THROUGHPUT fields_in_flight={} floor_now={} fields={n} secs={secs:.2} rate={:.3e} n/s found={found}",
            fields_in_flight(),
            msd_floor_in_use(),
            (n as f64) * (size as f64) / secs
        );
    }

    #[test]
    #[ignore = "requires a wgpu device"]
    fn cubecl_matches_cpu_niceonly() {
        let ctx = CubeclContext::new_default().expect("CubeCL init");
        for base in [10u32, 12, 25, 40, 45, 62, 80] {
            let Ok(Some(base_range)) = crate::base_range::get_base_range_u128(base) else {
                continue;
            };
            let start = base_range.range_start;
            let end = (start + 5_000_000).min(base_range.range_end);
            let range = FieldSize::new(start, end);

            let table = StrideTable::new(base, GPU_LSD_K);
            let mut cpu = process_range_niceonly(&range, base, &table).nice_numbers;
            cpu.sort_by_key(|n| n.number);
            let gpu = process_range_niceonly_cubecl(&ctx, &range, base).expect("cubecl run");

            assert_eq!(cpu, gpu.nice_numbers, "base {base}: niceonly mismatch");
            assert!(
                gpu.distribution.is_empty(),
                "base {base}: niceonly must report no distribution"
            );
            println!(
                "base {base}: [{start}, {end}) agrees, {} nice",
                gpu.nice_numbers.len()
            );
        }
    }

    /// The known solution, through the niceonly path: the whole pipeline (MSD
    /// filter, descriptors, on-device residue reconstruction, prefilter) has
    /// to line up for 69 to come back.
    #[test]
    #[ignore = "requires a wgpu device"]
    fn cubecl_niceonly_finds_69_in_base_10() {
        let ctx = CubeclContext::new_default().expect("CubeCL init");
        let base_range = crate::base_range::get_base_range_u128(10).unwrap().unwrap();
        let range = FieldSize::new(base_range.range_start, base_range.range_end);
        let results = process_range_niceonly_cubecl(&ctx, &range, 10).expect("cubecl run");
        assert_eq!(
            results
                .nice_numbers
                .iter()
                .map(|n| n.number)
                .collect::<Vec<_>>(),
            vec![69]
        );
    }

    /// The prefilter's own output, checked against an independent host mirror.
    ///
    /// Nice numbers are astronomically rare, so an over-rejecting prefilter
    /// would agree with the CPU on every range the parity test can afford —
    /// rejecting everything is not a hypothetical, it is the bug the CUDA
    /// path shipped in v3.2.14. The probe build reports every candidate the
    /// prefilter *passes*, at every lane width, and the mirror here computes
    /// the expected survivors by the definition (`x^k mod d^p` in u128) rather
    /// than by the kernel's chunk arithmetic, so the two cannot share a bug.
    /// Do the lowest `digits` base-`base` digits of n² and n³, computed
    /// mod `chunk_div^limbs`, contain no duplicate? Host mirror of the
    /// kernel's low-digit prefilter, shared by the probe tests.
    fn prefilter_mirror(n: u128, base: u32, pre: &VulkanPrefilterParams) -> bool {
        let modulus = u128::from(pre.chunk_div).pow(pre.limbs);
        let m = n % modulus;
        let sq = (m * m) % modulus;
        let cu = (sq * m) % modulus;
        let mut seen = 0u128;
        let mut dup = false;
        for mut v in [sq, cu] {
            for _ in 0..pre.digits {
                let d = (v % u128::from(base)) as u32;
                v /= u128::from(base);
                let bit = 1u128 << d;
                dup |= seen & bit != 0;
                seen |= bit;
            }
        }
        !dup
    }

    /// The cross-end filter's device semantics against a host mirror, with
    /// certificates that actually fire: word packing of both the low-mask
    /// table and the range certificates, residue-to-mask indexing, and the
    /// nonzero-intersection skip — in both the compacted and plain
    /// specializations. Probe mode reports the exact survivor set, so an
    /// over-rejection (the failure ordinary parity misses when a range holds
    /// no nice number) shows up as a missing survivor here.
    #[test]
    #[ignore = "requires a wgpu device"]
    fn cubecl_cross_filter_survivors_match_the_host_mirror() {
        let ctx = CubeclContext::new_default().expect("CubeCL init");
        #[allow(irrefutable_let_patterns)]
        let CubeclContext::Wgpu { client, .. } = &ctx else {
            unreachable!("new_default returns the wgpu variant");
        };
        // b40: prefilter base, digits reach 39 (bits in both mask words).
        // b50: no prefilter, so probe reports pure cross-filter survivors;
        // digits reach 49.
        for (base, mask) in [
            (40u32, (1u64 << 5) | (1u64 << 39)),
            (50, (1u64 << 3) | (1u64 << 45)),
        ] {
            let pre = vulkan_prefilter_params(base);
            let table = StrideTable::new(base, GPU_LSD_K);
            let start = crate::base_range::get_base_range_u128(base)
                .unwrap()
                .unwrap()
                .range_start;
            // Small enough that even the no-prefilter base's probe survivors
            // fit the output buffer with wide margin (probe reports every
            // survivor, not just nice numbers).
            let len: u32 = 400_000;

            // Host mirror: every stride candidate whose residue's exact low
            // digits miss the certificate and (where present) whose low
            // digits pass the prefilter.
            let end = start + u128::from(len);
            let (mut n, mut idx) = table.first_valid_at_or_after(start);
            let mut want = Vec::new();
            let mut cross_rejected = 0u32;
            while n < end {
                if table.low_digit_masks[idx] & mask != 0 {
                    cross_rejected += 1;
                } else if pre.is_none_or(|p| prefilter_mirror(n, base, &p)) {
                    want.push(n);
                }
                n += u128::from(table.gap_table[idx]);
                idx = (idx + 1) % table.gap_table.len();
            }
            assert!(
                cross_rejected > 1000,
                "base {base}: certificate {mask:#x} rejected too little to test"
            );
            assert!(
                want.len() < NICEONLY_OUT_CAPACITY / 2,
                "base {base}: {} probe survivors would risk the output buffer; shrink the window",
                want.len()
            );

            // The plane-scoped queue needs a real subgroup barrier: llvmpipe
            // mishandles OpControlBarrier at subgroup scope inside
            // plane-divergent control flow (2 of 283 survivors; exact with
            // the barriers removed), so it is only covered on hardware.
            let plane_sync = client
                .properties()
                .features
                .plane
                .contains(cubecl::ir::features::Plane::Sync)
                && !is_software_rasterizer(&ctx.device_name());
            let variants: &[(bool, bool)] = if plane_sync {
                &[(true, false), (true, true), (false, false)]
            } else {
                &[(true, false), (false, false)]
            };
            for &(forced_compact, forced_plane) in variants {
                let mut run =
                    CubeclNiceonlyRun::new(client, base, start, false).expect("probe run");
                run.probe = true;
                run.compact_override = Some(forced_compact);
                run.plane_override = Some(forced_plane);
                run.launch(0, &[0], &[len], &[mask]).expect("dispatch");
                let mut got: Vec<u128> = run
                    .finish()
                    .expect("results")
                    .iter()
                    .map(|n| n.number)
                    .collect();
                got.sort_unstable();
                assert_eq!(
                    got, want,
                    "base {base} compact={forced_compact} plane={forced_plane}: \
                     cross-filtered survivor set mismatch"
                );
            }
        }
    }

    #[test]
    #[ignore = "requires a wgpu device"]
    fn cubecl_prefilter_survivors_match_the_host_mirror() {
        use crate::gpu_niceonly::MAX_LANES_PER_RANGE;

        let ctx = CubeclContext::new_default().expect("CubeCL init");
        #[allow(irrefutable_let_patterns)]
        let CubeclContext::Wgpu { client, .. } = &ctx else {
            unreachable!("new_default returns the wgpu variant");
        };
        // Base 40 is the live prefilter base; 30 and 34 exercise the same
        // kernel at other chunk/limb constants.
        for base in [30u32, 34, 40] {
            let pre = vulkan_prefilter_params(base).expect("base has a prefilter");
            let table = StrideTable::new(base, GPU_LSD_K);
            let start = crate::base_range::get_base_range_u128(base)
                .unwrap()
                .unwrap()
                .range_start;
            let len: u32 = 5_000_000;

            // Every lane width, over the identical range: the tiling is pure
            // index arithmetic, so a width the host never happens to choose is
            // exactly where an off-by-one would hide.
            let mut per_width = Vec::new();
            for shift in 0..=MAX_LANES_PER_RANGE.ilog2() {
                let mut run =
                    CubeclNiceonlyRun::new(client, base, start, false).expect("probe run");
                assert!(
                    run.plan.prefilter.is_some(),
                    "base {base}: no prefilter params"
                );
                run.probe = true;
                run.lane_shift_override = Some(shift);
                run.launch(0, &[0], &[len], &[0]).expect("dispatch");
                per_width.push(
                    run.finish()
                        .expect("results")
                        .iter()
                        .map(|n| n.number)
                        .collect::<Vec<u128>>(),
                );
            }
            // Every stride candidate in the range, filtered by the mirror.
            let end = start + u128::from(len);
            let (mut n, mut idx) = table.first_valid_at_or_after(start);
            let mut want = Vec::new();
            let mut candidates = 0u32;
            while n < end {
                candidates += 1;
                if prefilter_mirror(n, base, &pre) {
                    want.push(n);
                }
                n += u128::from(table.gap_table[idx]);
                idx = (idx + 1) % table.gap_table.len();
            }

            // Each lane width against the CPU mirror, not merely against
            // each other: if the tiling drops or duplicates candidates at one
            // width, comparing widths only says they disagree, while this says
            // which one is wrong.
            for (shift, got) in per_width.iter().enumerate() {
                assert_eq!(
                    got,
                    &want,
                    "base {base}: {} lanes disagree with the CPU mirror \
                     ({} survivors vs {})",
                    1 << shift,
                    got.len(),
                    want.len()
                );
            }
            assert!(!want.is_empty(), "base {base}: the mirror passed nothing");
            assert!(
                want.len() < candidates as usize,
                "base {base}: the prefilter rejected nothing"
            );
            #[allow(clippy::cast_precision_loss)]
            {
                println!(
                    "base {base}: {} of {candidates} candidates survive ({:.2}%), device agrees",
                    want.len(),
                    100.0 * want.len() as f64 / f64::from(candidates)
                );
            }
        }
    }

    /// Many batches in one field must not lose the device. Without the
    /// per-batch flush in `detailed_impl`, `CubeCL` packs persistent
    /// grid-stride dispatches into one submission until it reaches
    /// `CUBECL_WGPU_MAX_TASKS` (32), and the driver's watchdog resets the GPU
    /// long before that — ~2 s on Linux/RADV, the same 2 s TDR on Windows.
    ///
    /// 40 batches is deliberately over that cap so the test also covers the
    /// aggregated path, but it is not the interesting size: a real field is
    /// `DETAILED_SEARCH_MAX_FIELD_SIZE` = 20 batches, under the cap and still
    /// over the watchdog at every base past ~46.
    ///
    /// Skipped on software rasterizers: CI runs the ignored tests on lavapipe,
    /// where 2e9 candidates would take hours and no watchdog is involved.
    #[test]
    #[ignore = "requires a wgpu device"]
    fn cubecl_detailed_survives_more_batches_than_the_task_cap() {
        let ctx = CubeclContext::new_default().expect("CubeCL init");
        let device = ctx.device_name();
        if is_software_rasterizer(&device) {
            println!("skipping on software rasterizer: {device}");
            return;
        }

        let base = 40u32;
        let batches = 40u128;
        let count = batches * CUBECL_BATCH_SIZE;
        let start = crate::base_range::get_base_range_u128(base)
            .unwrap()
            .unwrap()
            .range_start;
        let range = FieldSize::new(start, start + count);
        let results = process_range_detailed_cubecl(&ctx, &range, base).expect("cubecl run");

        // Non-vacuous: every candidate lands in exactly one bin, so the
        // histogram must account for the whole field. A submission the driver
        // killed would take its batch's counts with it.
        let counted: u128 = results.distribution.iter().map(|d| d.count).sum();
        assert_eq!(
            counted, count,
            "{batches} batches on {device}: histogram covers {counted} of {count} candidates"
        );
        println!("{batches} batches ({count} candidates) survived on {device}");
    }

    /// CPU/CubeCL niceonly parity through the native CUDA runtime, on real
    /// silicon — the wide-chunk digit scan's only niceonly exercise.
    #[test]
    #[cfg(feature = "cubecl-cuda")]
    #[ignore = "requires an NVIDIA device"]
    fn cubecl_cuda_matches_cpu_niceonly() {
        let ctx = CubeclContext::new_cuda(0).expect("CubeCL CUDA init");
        for base in [10u32, 12, 25, 40, 45, 62, 80] {
            let Ok(Some(base_range)) = crate::base_range::get_base_range_u128(base) else {
                continue;
            };
            let start = base_range.range_start;
            let end = (start + 5_000_000).min(base_range.range_end);
            let range = FieldSize::new(start, end);

            let table = StrideTable::new(base, GPU_LSD_K);
            let mut cpu = process_range_niceonly(&range, base, &table).nice_numbers;
            cpu.sort_by_key(|n| n.number);
            let gpu = process_range_niceonly_cubecl(&ctx, &range, base).expect("cubecl-cuda run");

            assert_eq!(cpu, gpu.nice_numbers, "base {base}: niceonly mismatch");
            println!(
                "base {base}: [{start}, {end}) agrees, {} nice (CUDA runtime)",
                gpu.nice_numbers.len()
            );
        }
    }
}
