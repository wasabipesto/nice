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
    VulkanPrefilterParams, chunk_constants_u16, gpu_supports_base, n_limbs,
    vulkan_prefilter_params,
};
use crate::gpu_niceonly::{
    GPU_LSD_K, MAX_STRIDE_MODULUS, RangeSink, lane_shift_for, report_field, residue_empty_result,
    run_range_pipeline, stride_chunk_bits,
};
use crate::number_stats::get_near_miss_cutoff;
use crate::stride_filter::StrideTable;
use crate::{FieldResults, FieldSize, NiceNumberSimple, UniquesDistributionSimple};
use anyhow::{Context as _, Result, ensure};
use cubecl::prelude::*;
use log::{debug, warn};
use std::sync::OnceLock;
use std::time::Instant;

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
#[allow(
    clippy::too_many_arguments,
    clippy::too_many_lines,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    clippy::cast_lossless,
    clippy::used_underscore_binding
)]
fn detailed_kernel(
    hist: &Array<Atomic<u32>>,
    miss_count: &Array<Atomic<u32>>,
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

    // Scratch for the digit scan: cu_limbs u32 words.
    let mut sv = Array::<u32>::new(cu_limbs as usize);

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
                    sv[i as usize] = sq[i as usize];
                }
                #[unroll]
                for i in sq_limbs..cu_limbs {
                    sv[i as usize] = 0u32;
                }
            } else {
                #[unroll]
                for i in 0..cu_limbs {
                    sv[i as usize] = cu[i as usize];
                }
            }

            // top_limb
            let mut top: i32 = comptime!(cu_limbs as i32 - 1).runtime();
            while top >= 0 {
                if sv[top as usize] != 0u32 {
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
                        let cur = (rem64 << 32u64) | u64::cast_from(sv[i as usize]);
                        sv[i as usize] = u32::cast_from(cur / u64::cast_from(chunk_div));
                        rem64 = cur % u64::cast_from(chunk_div);
                        i -= 1;
                    }
                    rem = u32::cast_from(rem64); // rem < chunk_div < 2^31
                } else {
                    let mut i: i32 = top;
                    while i >= 0 {
                        let vi = sv[i as usize];
                        let c1 = (rem << 16u32) | (vi >> 16u32);
                        let q1 = c1 / chunk_div;
                        let c2 = ((c1 % chunk_div) << 16u32) | (vi & 0xFFFFu32);
                        let q2 = c2 / chunk_div;
                        rem = c2 % chunk_div;
                        sv[i as usize] = (q1 << 16u32) | q2;
                        i -= 1;
                    }
                }
                while top >= 0 {
                    if sv[top as usize] != 0u32 {
                        break;
                    }
                    top -= 1;
                }

                let mut chunk = rem;
                if top >= 0 {
                    // Interior chunk: all chunk_digits digits, zeros included.
                    #[unroll]
                    for _k in 0..chunk_digits {
                        let d = chunk % base;
                        chunk /= base;
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
                        let d = chunk % base;
                        chunk /= base;
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
#[allow(
    clippy::too_many_arguments,
    clippy::too_many_lines,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    clippy::cast_lossless,
    clippy::used_underscore_binding,
    clippy::many_single_char_names,
    clippy::similar_names
)]
fn niceonly_kernel(
    residues: &Array<u32>,
    range_offsets: &Array<u32>, // lo, hi pairs
    range_lens: &Array<u32>,
    nice_out: &mut Array<u32>,
    nice_count: &Array<Atomic<u32>>,
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
) {
    let sq_limbs = comptime!(2 * limbs);
    let cu_limbs = comptime!(3 * limbs);
    let two_masks = comptime!(base > 64);
    let has_prefilter = comptime!(pre_limbs > 0);
    let offset_chunks = comptime!(64 / offset_chunk_bits);
    let per_word = comptime!(32 / offset_chunk_bits);
    let offset_chunk_mask = comptime!((1u32 << offset_chunk_bits) - 1);

    let lanes = 1u32 << lane_shift;
    let lane = ABSOLUTE_POS_X & (lanes - 1u32);
    let nwarps = (CUBE_COUNT_X * CUBE_DIM_X) >> lane_shift;
    let fs_lo = (u64::cast_from(fs1) << 32u64) | u64::cast_from(fs0);
    let fs_hi = (u64::cast_from(fs3) << 32u64) | u64::cast_from(fs2);

    // Scratch for the digit scan: cu_limbs u32 words.
    let mut sv = Array::<u32>::new(cu_limbs as usize);

    let mut r = ABSOLUTE_POS_X >> lane_shift;
    while r < num_ranges {
        let off_lo = range_offsets[(2u32 * r) as usize];
        let off_hi = range_offsets[(2u32 * r + 1u32) as usize];
        let offset = (u64::cast_from(off_hi) << 32u64) | u64::cast_from(off_lo);

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
            let word = if comptime!(k < per_word) { off_hi } else { off_lo };
            let shift = comptime!(32 - offset_chunk_bits - (k % per_word) * offset_chunk_bits);
            acc = ((acc << offset_chunk_bits) | ((word >> shift) & offset_chunk_mask)) % stride_m;
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

            // --- candidate_is_nice(n_lo, n_hi), inlined ----------------------
            let mut ok = true;

            // Low-digit modular prefilter: are the lowest
            // `pre_limbs * pre_chunk_digits` digits of n² and n³ all distinct?
            // Fixed-length and branch-free — lanes only save work when their
            // whole group is killed, and this kills ~98% of candidates. Held
            // as digit-chunks of base `pre_chunk_div < 2^16` so the truncated
            // multiplies stay in u32 (`x^k mod b^p == (x mod b^p)^k mod b^p`).
            if has_prefilter {
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
                // kernel — the cube multiply is usually dead work).
                let mut pass: u32 = 0u32;
                while pass < 2u32 && ok {
                    if pass == 0u32 {
                        #[unroll]
                        for i in 0..sq_limbs {
                            sv[i as usize] = sq[i as usize];
                        }
                        #[unroll]
                        for i in sq_limbs..cu_limbs {
                            sv[i as usize] = 0u32;
                        }
                    } else {
                        // cu = sq * n, computed only now.
                        let mut cu = Array::<u32>::new(cu_limbs as usize);
                        #[unroll]
                        for i in 0..cu_limbs {
                            cu[i as usize] = 0u32;
                        }
                        #[unroll]
                        for i in 0..sq_limbs {
                            let mut carry = 0u64;
                            #[unroll]
                            for jj in 0..limbs {
                                let k = comptime!(i + jj);
                                let t = u64::cast_from(sq[i as usize])
                                    * u64::cast_from(nl[jj as usize])
                                    + u64::cast_from(cu[k as usize])
                                    + carry;
                                cu[k as usize] = u32::cast_from(t);
                                carry = t >> 32u64;
                            }
                            cu[comptime!(i + limbs) as usize] = u32::cast_from(carry);
                        }
                        #[unroll]
                        for i in 0..cu_limbs {
                            sv[i as usize] = cu[i as usize];
                        }
                    }

                    let mut top: i32 = comptime!(cu_limbs as i32 - 1).runtime();
                    while top >= 0 {
                        if sv[top as usize] != 0u32 {
                            break;
                        }
                        top -= 1;
                    }

                    // Chunked radix scan, destroying sv; same two comptime
                    // flavors as the detailed kernel. After each chunk's
                    // digits, a set duplicate bit ends the candidate.
                    while top >= 0 && ok {
                        let mut rem = 0u32;
                        if wide_chunk {
                            let mut rem64 = 0u64;
                            let mut i: i32 = top;
                            while i >= 0 {
                                let cur = (rem64 << 32u64) | u64::cast_from(sv[i as usize]);
                                sv[i as usize] = u32::cast_from(cur / u64::cast_from(chunk_div));
                                rem64 = cur % u64::cast_from(chunk_div);
                                i -= 1;
                            }
                            rem = u32::cast_from(rem64); // rem < chunk_div < 2^31
                        } else {
                            let mut i: i32 = top;
                            while i >= 0 {
                                let vi = sv[i as usize];
                                let c1 = (rem << 16u32) | (vi >> 16u32);
                                let q1 = c1 / chunk_div;
                                let c2 = ((c1 % chunk_div) << 16u32) | (vi & 0xFFFFu32);
                                let q2 = c2 / chunk_div;
                                rem = c2 % chunk_div;
                                sv[i as usize] = (q1 << 16u32) | q2;
                                i -= 1;
                            }
                        }
                        while top >= 0 {
                            if sv[top as usize] != 0u32 {
                                break;
                            }
                            top -= 1;
                        }

                        let mut chunk = rem;
                        let mut dup = 0u64;
                        if top >= 0 {
                            // Interior chunk: all chunk_digits digits, zeros
                            // included.
                            #[unroll]
                            for _k in 0..chunk_digits {
                                let d = chunk % base;
                                chunk /= base;
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
                                let d = chunk % base;
                                chunk /= base;
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
                    pass += 1u32;
                }
            }

            if ok {
                let mut u =
                    u32::cast_from(m0).count_ones() + u32::cast_from(m0 >> 32u64).count_ones();
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
            // --- end candidate_is_nice ---------------------------------------

            g += lanes;
        }
        r += nwarps;
    }
}

// ============================================================================
// Host side
// ============================================================================

/// One initialized `CubeCL` device: wgpu everywhere, or the native CUDA
/// runtime when built with `cubecl-cuda` (the meaningful NVIDIA comparison,
/// since it exercises `CubeCL`'s CUDA codegen against the hand kernels).
pub enum CubeclContext {
    Wgpu {
        client: cubecl::prelude::ComputeClient<cubecl::wgpu::WgpuRuntime>,
        device_name: String,
    },
    #[cfg(feature = "cubecl-cuda")]
    Cuda {
        client: cubecl::prelude::ComputeClient<cubecl::cuda::CudaRuntime>,
        device_name: String,
    },
}

impl CubeclContext {
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
        static DEFAULT: OnceLock<(
            cubecl::prelude::ComputeClient<cubecl::wgpu::WgpuRuntime>,
            String,
        )> = OnceLock::new();
        let (client, device_name) = DEFAULT.get_or_init(|| {
            let device = cubecl::wgpu::WgpuDevice::default();
            let setup = cubecl::wgpu::init_setup::<cubecl::wgpu::AutoGraphicsApi>(
                &device,
                cubecl::wgpu::RuntimeOptions::default(),
            );
            let device_name = setup.adapter.get_info().name;
            let client = cubecl::wgpu::WgpuRuntime::client(&device);
            (client, device_name)
        });
        Ok(Self::Wgpu {
            client: client.clone(),
            device_name: device_name.clone(),
        })
    }

    /// Initialize `CubeCL`'s native CUDA runtime on `device_index`.
    ///
    /// # Errors
    /// Returns an error if CUDA is unavailable.
    #[cfg(feature = "cubecl-cuda")]
    pub fn new_cuda(device_index: usize) -> Result<Self> {
        let device = cubecl::cuda::CudaDevice::new(device_index);
        let client = cubecl::cuda::CudaRuntime::client(&device);
        Ok(Self::Cuda {
            client,
            device_name: format!("cubecl-cuda device {device_index}"),
        })
    }

    /// The adapter/device name, for reports.
    #[must_use]
    pub fn device_name(&self) -> String {
        match self {
            Self::Wgpu { device_name, .. } => device_name.clone(),
            #[cfg(feature = "cubecl-cuda")]
            Self::Cuda { device_name, .. } => device_name.clone(),
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
    if !gpu_supports_base(base) {
        warn!("base {base} not supported on GPU, falling back to CPU for this field");
        return Ok(process_range_detailed(range, base));
    }

    match ctx {
        CubeclContext::Wgpu { client, .. } => detailed_impl(client, range, base, false),
        #[cfg(feature = "cubecl-cuda")]
        CubeclContext::Cuda { client, .. } => detailed_impl(client, range, base, true),
    }
}

/// Runtime-generic body of [`process_range_detailed_cubecl`].
fn detailed_impl<R: cubecl::prelude::Runtime>(
    client: &cubecl::prelude::ComputeClient<R>,
    range: &FieldSize,
    base: u32,
    wide_chunk: bool,
) -> Result<FieldResults> {
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

    for batch in range.chunks(CUBECL_BATCH_SIZE) {
        // A fresh zeroed histogram per batch; drained below so a u32 bin
        // cannot overflow (mirrors DetailedRun::drain_histogram).
        let hist_handle = client.create(cubecl::bytes::Bytes::from_elems(vec![0u32; hist_bins]));
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

        let bytes = client
            .read_one(hist_handle)
            .map_err(|e| anyhow::anyhow!("histogram read failed: {e:?}"))?;
        let bins = u32::from_bytes(&bytes);
        for (acc, &bin) in histogram.iter_mut().zip(bins.iter()) {
            *acc += u128::from(bin);
        }
    }

    // Near misses.
    let bytes = client
        .read_one(miss_count_handle.clone())
        .map_err(|e| anyhow::anyhow!("miss count read failed: {e:?}"))?;
    let miss_total = u32::from_bytes(&bytes)[0] as usize;
    ensure!(
        miss_total <= NEAR_MISS_CAPACITY,
        "near-miss buffer overflow: {miss_total} > {NEAR_MISS_CAPACITY}"
    );
    let bytes = client
        .read_one(miss_data_handle)
        .map_err(|e| anyhow::anyhow!("miss data read failed: {e:?}"))?;
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
pub fn process_range_niceonly_cubecl(
    ctx: &CubeclContext,
    range: &FieldSize,
    base: u32,
) -> Result<FieldResults> {
    if let Some(empty) = residue_empty_result(base) {
        return Ok(empty);
    }
    if !gpu_supports_base(base) {
        warn!("base {base} not supported on GPU, falling back to CPU for this field");
        let table = StrideTable::new(base, GPU_LSD_K);
        return Ok(process_range_niceonly(range, base, &table));
    }

    match ctx {
        CubeclContext::Wgpu { client, .. } => niceonly_impl(client, range, base, false),
        #[cfg(feature = "cubecl-cuda")]
        CubeclContext::Cuda { client, .. } => niceonly_impl(client, range, base, true),
    }
}

/// Runtime-generic body of [`process_range_niceonly_cubecl`].
fn niceonly_impl<R: cubecl::prelude::Runtime>(
    client: &cubecl::prelude::ComputeClient<R>,
    range: &FieldSize,
    base: u32,
    wide_chunk: bool,
) -> Result<FieldResults> {
    let mut run = CubeclNiceonlyRun::new(client, base, range.start(), wide_chunk)?;
    let stats = run_range_pipeline(&mut run, range, base)?;
    let nice_numbers = run.finish()?;
    debug!(
        "CubeCL niceonly pipeline: {} ranges in {} dispatches, M={}, R={}, found {}",
        stats.num_ranges,
        stats.launches,
        run.stride_m,
        run.stride_r,
        nice_numbers.len()
    );
    report_field("CubeCL", base, range, &stats);

    Ok(FieldResults {
        distribution: Vec::new(),
        nice_numbers,
    })
}

/// The `CubeCL` end of the niceonly range pipeline: per-base constants, the
/// device residue table, and the output buffers, kept across every dispatch of
/// a field.
struct CubeclNiceonlyRun<'a, R: cubecl::prelude::Runtime> {
    client: &'a cubecl::prelude::ComputeClient<R>,
    residues: cubecl::server::Handle,
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
    stride_m: u32,
    stride_r: u32,
    prefilter: Option<VulkanPrefilterParams>,
    /// Report prefilter survivors instead of nice numbers (device tests only).
    probe: bool,
    /// Pin the lane tiling instead of sizing it per dispatch (device tests).
    lane_shift_override: Option<u32>,
}

impl<'a, R: cubecl::prelude::Runtime> CubeclNiceonlyRun<'a, R> {
    /// # Errors
    /// Returns an error for an unconfigurable base. Residue-empty bases must
    /// be short-circuited by the caller before any stride table is built.
    fn new(
        client: &'a cubecl::prelude::ComputeClient<R>,
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

        let residues = client.create(cubecl::bytes::Bytes::from_elems(table.valid_residues));
        let nice_out = client.create(cubecl::bytes::Bytes::from_elems(vec![
            0u32;
            NICEONLY_OUT_CAPACITY
                * NICEONLY_STRIDE as usize
        ]));
        let nice_count = client.create(cubecl::bytes::Bytes::from_elems(vec![0u32; 1]));

        #[allow(clippy::cast_possible_truncation)]
        let fs_mod_m = (field_start % u128::from(stride_m)) as u32;
        Ok(Self {
            client,
            residues,
            nice_out,
            nice_count,
            field_start,
            fs_mod_m,
            base,
            limbs,
            chunk_digits,
            chunk_div,
            wide_chunk,
            stride_m,
            stride_r,
            prefilter: vulkan_prefilter_params(base),
            probe: false,
            lane_shift_override: None,
        })
    }

    /// Collect the nice numbers found across the whole field.
    ///
    /// The blocking reads double as the device wait: a dispatch still in
    /// flight has simply not written its hits yet, and the failure mode is
    /// *silently missing solutions* — so the read path orders after every
    /// launch on the client's queue.
    ///
    /// # Errors
    /// Returns an error if the kernel tried to write more hits than the buffer
    /// holds — which, given how rare nice numbers are, means a kernel bug
    /// rather than a genuine flood.
    fn finish(&self) -> Result<Vec<NiceNumberSimple>> {
        let bytes = self
            .client
            .read_one(self.nice_count.clone())
            .map_err(|e| anyhow::anyhow!("nice count read failed: {e:?}"))?;
        let written = u32::from_bytes(&bytes)[0] as usize;
        ensure!(
            written <= NICEONLY_OUT_CAPACITY,
            "niceonly output buffer overflow: {written} > {NICEONLY_OUT_CAPACITY} \
             (this strongly suggests a kernel bug)"
        );
        let bytes = self
            .client
            .read_one(self.nice_out.clone())
            .map_err(|e| anyhow::anyhow!("nice out read failed: {e:?}"))?;
        let words = u32::from_bytes(&bytes);
        let mut hits: Vec<NiceNumberSimple> = (0..written)
            .map(|i| {
                let o = i * NICEONLY_STRIDE as usize;
                let lo = u128::from(words[o]) | (u128::from(words[o + 1]) << 32);
                let hi = u128::from(words[o + 2]) | (u128::from(words[o + 3]) << 32);
                NiceNumberSimple {
                    number: (hi << 64) | lo,
                    num_uniques: self.base,
                }
            })
            .collect();
        hits.sort_by_key(|n| n.number);
        Ok(hits)
    }
}

impl<R: cubecl::prelude::Runtime> RangeSink for CubeclNiceonlyRun<'_, R> {
    fn launch(&mut self, offsets: &[u64], lens: &[u32]) -> Result<()> {
        // Pack offsets as lo/hi u32 pairs; buffers are created per dispatch
        // and sized to the batch (CubeCL pools the allocations).
        #[allow(clippy::cast_possible_truncation)]
        let pairs: Vec<u32> = offsets
            .iter()
            .flat_map(|&o| [o as u32, (o >> 32) as u32])
            .collect();
        let offsets_handle = self.client.create(cubecl::bytes::Bytes::from_elems(pairs));
        let lens_handle = self
            .client
            .create(cubecl::bytes::Bytes::from_elems(lens.to_vec()));

        // Tile the dispatch to this batch's ranges; batches are homogeneous
        // enough for the mean to be a good summary, because the MSD recursion
        // bounds range length by the floor.
        let mean_len =
            lens.iter().map(|&l| u64::from(l)).sum::<u64>() / lens.len().max(1) as u64;
        let lane_shift = self.lane_shift_override.unwrap_or_else(|| {
            lane_shift_for(
                offsets.len() as u64,
                mean_len,
                self.stride_m,
                self.stride_r,
            )
        });
        let num_ranges = u32::try_from(offsets.len()).unwrap_or(u32::MAX);
        let threads = u64::from(num_ranges) << lane_shift;
        #[allow(clippy::cast_possible_truncation)]
        let cubes = threads
            .div_ceil(u64::from(WORKGROUP_SIZE))
            .clamp(1, u64::from(MAX_CUBES)) as u32;

        let (pre_limbs, pre_chunk_digits, pre_chunk_div) = self
            .prefilter
            .map_or((0, 0, 0), |p| (p.limbs, p.chunk_digits, p.chunk_div));

        #[allow(clippy::cast_possible_truncation)]
        unsafe {
            niceonly_kernel::launch_unchecked::<R>(
                self.client,
                CubeCount::Static(cubes, 1, 1),
                CubeDim::new_1d(WORKGROUP_SIZE),
                ArrayArg::from_raw_parts(self.residues.clone(), self.stride_r as usize),
                ArrayArg::from_raw_parts(offsets_handle, offsets.len() * 2),
                ArrayArg::from_raw_parts(lens_handle, lens.len()),
                ArrayArg::from_raw_parts(
                    self.nice_out.clone(),
                    NICEONLY_OUT_CAPACITY * NICEONLY_STRIDE as usize,
                ),
                ArrayArg::from_raw_parts(self.nice_count.clone(), 1),
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
                self.stride_m,
                self.stride_r,
                stride_chunk_bits(self.stride_m),
                pre_limbs,
                pre_chunk_digits,
                pre_chunk_div,
                self.probe,
            );
        }
        Ok(())
    }

    fn sync(&mut self) -> Result<()> {
        cubecl::future::block_on(self.client.sync())
            .map_err(|e| anyhow::anyhow!("device sync failed: {e:?}"))
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
            assert_eq!(gpu.distribution, cpu.distribution, "base {base}: distribution mismatch");
            assert_eq!(gpu.nice_numbers, cpu.nice_numbers, "base {base}: near-miss mismatch");
            println!("base {base}: {count} candidates match the CPU exactly (CUDA runtime)");
        }
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
    #[test]
    #[ignore = "requires a wgpu device"]
    fn cubecl_prefilter_survivors_match_the_host_mirror() {
        use crate::gpu_niceonly::MAX_LANES_PER_RANGE;

        /// Do the lowest `digits` base-`base` digits of n² and n³, computed
        /// mod `chunk_div^limbs`, contain no duplicate?
        fn mirror(n: u128, base: u32, pre: &VulkanPrefilterParams) -> bool {
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
                assert!(run.prefilter.is_some(), "base {base}: no prefilter params");
                run.probe = true;
                run.lane_shift_override = Some(shift);
                run.launch(&[0], &[len]).expect("dispatch");
                run.sync().expect("sync");
                per_width.push(
                    run.finish()
                        .expect("results")
                        .iter()
                        .map(|n| n.number)
                        .collect::<Vec<u128>>(),
                );
            }
            for (shift, got) in per_width.iter().enumerate() {
                assert_eq!(
                    got,
                    &per_width[0],
                    "base {base}: {} lanes disagree with 1 lane",
                    1 << shift
                );
            }
            let got = per_width.swap_remove(0);

            // Every stride candidate in the range, filtered by the mirror.
            let end = start + u128::from(len);
            let (mut n, mut idx) = table.first_valid_at_or_after(start);
            let mut want = Vec::new();
            let mut candidates = 0u32;
            while n < end {
                candidates += 1;
                if mirror(n, base, &pre) {
                    want.push(n);
                }
                n += u128::from(table.gap_table[idx]);
                idx = (idx + 1) % table.gap_table.len();
            }

            assert_eq!(got, want, "base {base}: prefilter survivors differ");
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
