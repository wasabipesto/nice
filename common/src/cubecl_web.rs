//! Browser-portable `CubeCL` detailed kernels, with **no 64-bit integers
//! anywhere**: WebGPU has no `shader-int64`, so the wgpu backend of a
//! browser rejects any kernel naming a u64.
//!
//! Two kernels, as on the native path:
//!
//! - [`detailed_fd_kernel_u32`], the finite-difference kernel of
//!   [`crate::cubecl_detailed_fd`] in u32 arithmetic, for every candidate
//!   inside a base's [`fd_plan`] interval (bases 10-64). It shares the seed
//!   ([`fd_seed_split16`]) and the digit-array steps with the native kernel;
//!   only its prologue, mask and near-miss record differ, below.
//! - [`detailed_kernel_u32`], the per-number kernel, for bases above 64, for
//!   any part of a range outside the plan's interval, and as the reference
//!   the finite-difference kernel is tested against.
//!
//! The per-number kernel's arithmetic differs from the native one in these
//! ways, and nothing else:
//!
//! - Multi-precision values are held as 16-bit halves in u32 words, and the
//!   schoolbook multiplies accumulate 16x16 products in u32 (max term
//!   `(2^16-1)^2 + 2*(2^16-1) < 2^32`, checked by
//!   [`tests::half_limb_accumulation_cannot_overflow`]).
//! - The digit masks are u32 quads instead of u64 pairs.
//! - The digit scan is always the split16 flavor (u32 divisions by a sub-2^16
//!   comptime constant), which is what the native kernel already uses on
//!   every wgpu target.
//!
//! The finite-difference kernel's differences from the native one:
//!
//! - The run start, count, offset and stride are u32 (the host keeps
//!   `count + stride` below 2^32), and n is built as four u32 words by a
//!   carry chain, which is the form [`fd_seed_split16`] takes.
//! - The digit mask is a u32 pair ([`fd_mask_u32`]) instead of one u64.
//! - A near miss's n is rebuilt over four words, on the rare path.
//!
//! Apple's Metal compiler miscompiles an unsigned division by a constant whose
//! dividend is a right shift by half the word (see `fd_to_digits` in
//! [`crate::cubecl_detailed_fd`]), and every browser on a Mac compiles WGSL to
//! Metal. In the finite-difference kernel only the seed divides by a constant
//! that is not a power of two, and its dividends are never such a shift; the
//! per-number kernel's scan keeps its remainder in a run-time loop, so its
//! dividends are never a bare shift either.
//!
//! Correctness is anchored the same way as the native kernels: exact CPU
//! parity on lavapipe (which drives these kernels through the identical
//! naga/WGSL path a browser uses, minus the browser).

#![cfg(feature = "cubecl")]
#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_lossless,
    clippy::used_underscore_binding
)]

use crate::client_process::process_range_detailed;
use crate::cubecl_backend::{CubeclContext, HIST_COPIES, MISS_STRIDE, WORKGROUP_SIZE};
use crate::cubecl_detailed_fd::{
    FD_RUN_LEN, NO_CARRY, fd_add_const_top, fd_add_top, fd_plan, fd_ripple, fd_seed_split16,
};
use crate::gpu_config::{chunk_constants_u16, gpu_supports_base, n_limbs};
use crate::number_stats::get_near_miss_cutoff;
use crate::{FieldResults, FieldSize, NiceNumberSimple, UniquesDistributionSimple};
use anyhow::{Context as _, Result, ensure};
use cubecl::prelude::*;
use log::{debug, warn};
use web_time::Instant;

/// Upper bound on cubes per dispatch; threads grid-stride past it.
pub const MAX_CUBES_WEB: u32 = 4096;

/// Candidates per dispatch. Much smaller than the native 50M: browser tabs
/// sit behind the same ~2s GPU watchdogs as native (and stricter compositor
/// deadlines), and the u32-only arithmetic is several times slower per
/// candidate, so dispatches are sized for slow integrated GPUs. Each batch is
/// flushed as its own submission, exactly like the native backend.
pub const CUBECL_WEB_BATCH_SIZE: u128 = 32_000_000;

/// Near-miss records held on the device per field.
///
/// Deliberately far smaller than the native backend's: this buffer is the
/// client's largest GPU allocation, and a browser tab is given a small
/// fraction of the card's memory. 131072 records is 2.6 MB.
///
/// Sizing it is about *low* bases, not production ones. Near misses are
/// candidates above 0.9 x base unique digits, which at base 40 and up is
/// almost nothing (a real field yields a handful), but at base 10 is over
/// half of all candidates. The browser only ever receives fields for the
/// bases actually being searched, so the small buffer is right for the
/// workload; the parity tests keep their low-base ranges short to match,
/// and an overflow is reported rather than silently truncated.
///
/// A browser gets a far smaller figure than the native tests. LibreWolf 149
/// on an RX 9070 XT — a card with 16 GB — refused a 4 MB buffer outright, so
/// the browser's largest allocation is kept in the hundreds of kilobytes. At
/// 8192 records that is 160 KB, still far more near misses than a real field
/// produces at the bases the browser is given. The native value stays large
/// enough for the parity tests' low-base ranges, which is the only place
/// dense near misses occur.
#[cfg(target_family = "wasm")]
const NEAR_MISS_CAPACITY_WEB: usize = 1 << 13;
#[cfg(not(target_family = "wasm"))]
const NEAR_MISS_CAPACITY_WEB: usize = 1 << 17;

/// The near-miss capacity this build compiled with, for the client's
/// startup diagnostics.
#[must_use]
pub fn near_miss_capacity() -> usize {
    NEAR_MISS_CAPACITY_WEB
}

/// The u32-only detailed kernel. See the module docs for how it differs from
/// the native one; the structure (grid-stride, shared histogram copies,
/// shared scan scratch, near-miss records) is the same, kept line-comparable.
#[cube(launch_unchecked)]
// The single-character and lookalike names deliberately match the native
// kernel and the generated WGSL, so the three review side by side.
#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
fn detailed_kernel_u32(
    hist: &mut Array<Atomic<u32>>,
    miss_count: &mut Array<Atomic<u32>>,
    miss_data: &mut Array<u32>,
    s0: u32,
    s1: u32,
    s2: u32,
    s3: u32,
    count: u32,
    miss_cap: u32,
    #[comptime] base: u32,
    #[comptime] limbs: u32,
    #[comptime] chunk_digits: u32,
    #[comptime] chunk_div: u32,
    #[comptime] cutoff: u32,
) {
    let hist_bins = comptime!(base + 1);
    // Half-limb (16-bit digit) counts: n, n^2, n^3.
    let n_halves = comptime!(2 * limbs);
    let sq_halves = comptime!(4 * limbs);
    let cu_halves = comptime!(6 * limbs);
    // u32-word counts for the scan scratch.
    let sq_words = comptime!(2 * limbs);
    let cu_words = comptime!(3 * limbs);
    let four_masks = comptime!(base > 64);

    let hist_s = SharedMemory::<Atomic<u32>>::new(comptime!((HIST_COPIES * (base + 1)) as usize));

    // Zero the workgroup histogram.
    let mut zi = UNIT_POS_X;
    while zi < HIST_COPIES * hist_bins {
        hist_s[zi as usize].store(0u32);
        zi += CUBE_DIM_X;
    }
    sync_cube();

    // Shared scan scratch, one u32 word per thread per cu limb, odd stride —
    // same rationale as the native kernel (register-promoted arrays become
    // compare/select chains; local memory costs a round trip).
    let sv_pad = comptime!(cu_words | 1);
    let mut sv_s = SharedMemory::<u32>::new(comptime!((WORKGROUP_SIZE * (cu_words | 1)) as usize));
    let svb = UNIT_POS_X * sv_pad;

    let copy = (UNIT_POS_X >> 5u32) % HIST_COPIES;
    let stride = CUBE_COUNT_X * CUBE_DIM_X;

    let mut idx = ABSOLUTE_POS_X;
    while idx < count {
        // n = start + idx over u32 words with carries.
        let mut nw = Array::<u32>::new(4usize);
        nw[0] = s0 + idx;
        let mut carry = u32::cast_from(nw[0] < s0);
        nw[1] = s1 + carry;
        carry = u32::cast_from(nw[1] < carry);
        nw[2] = s2 + carry;
        carry = u32::cast_from(nw[2] < carry);
        nw[3] = s3 + carry;

        // Unpack n into 16-bit halves for the multiplies.
        let mut nh = Array::<u32>::new(n_halves as usize);
        #[unroll]
        for i in 0..n_halves {
            let word = comptime!(i / 2);
            let shift = comptime!((i % 2) * 16);
            nh[i as usize] = (nw[word as usize] >> shift) & 0xFFFFu32;
        }

        // sq = n * n over halves; every term fits u32.
        let mut sq = Array::<u32>::new(sq_halves as usize);
        #[unroll]
        for i in 0..sq_halves {
            sq[i as usize] = 0u32;
        }
        #[unroll]
        for i in 0..n_halves {
            let mut c = 0u32;
            #[unroll]
            for j in 0..n_halves {
                let k = comptime!(i + j);
                let t = nh[i as usize] * nh[j as usize] + sq[k as usize] + c;
                sq[k as usize] = t & 0xFFFFu32;
                c = t >> 16u32;
            }
            sq[comptime!(i + n_halves) as usize] = c;
        }

        // Digit masks: u32 quads (the native kernel's u64 pair, split).
        let mut m0a = 0u32;
        let mut m0b = 0u32;
        let mut m1a = 0u32;
        let mut m1b = 0u32;

        // Scan n^2, then n^3, both in full — detailed histograms every
        // candidate's complete unique count, so unlike niceonly there is no
        // early duplicate exit anywhere.
        #[unroll]
        for i in 0..sq_words {
            let lo = comptime!(2 * i);
            let hi = comptime!(2 * i + 1);
            sv_s[(svb + i) as usize] = sq[lo as usize] | (sq[hi as usize] << 16u32);
        }
        let mut top: i32 = comptime!(sq_words as i32 - 1).runtime();
        while top >= 0 {
            if sv_s[(svb + u32::cast_from(top)) as usize] != 0u32 {
                break;
            }
            top -= 1;
        }
        while top >= 0 {
            let mut rem = 0u32;
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
                    if four_masks {
                        if d < 32u32 {
                            m0a |= 1u32 << d;
                        } else if d < 64u32 {
                            m0b |= 1u32 << (d - 32u32);
                        } else if d < 96u32 {
                            m1a |= 1u32 << (d - 64u32);
                        } else {
                            m1b |= 1u32 << (d - 96u32);
                        }
                    } else if d < 32u32 {
                        m0a |= 1u32 << d;
                    } else {
                        m0b |= 1u32 << (d - 32u32);
                    }
                }
            } else {
                // Most significant chunk: digits until zero.
                while chunk != 0u32 {
                    let cq = chunk / base;
                    let d = chunk - cq * base;
                    chunk = cq;
                    if four_masks {
                        if d < 32u32 {
                            m0a |= 1u32 << d;
                        } else if d < 64u32 {
                            m0b |= 1u32 << (d - 32u32);
                        } else if d < 96u32 {
                            m1a |= 1u32 << (d - 64u32);
                        } else {
                            m1b |= 1u32 << (d - 96u32);
                        }
                    } else if d < 32u32 {
                        m0a |= 1u32 << d;
                    } else {
                        m0b |= 1u32 << (d - 32u32);
                    }
                }
            }
        }

        // cu = sq * n over halves, packed straight into the scratch words.
        let mut cu = Array::<u32>::new(cu_halves as usize);
        #[unroll]
        for i in 0..cu_halves {
            cu[i as usize] = 0u32;
        }
        #[unroll]
        for i in 0..sq_halves {
            let mut c = 0u32;
            #[unroll]
            for j in 0..n_halves {
                let k = comptime!(i + j);
                let t = sq[i as usize] * nh[j as usize] + cu[k as usize] + c;
                cu[k as usize] = t & 0xFFFFu32;
                c = t >> 16u32;
            }
            cu[comptime!(i + n_halves) as usize] = c;
        }
        #[unroll]
        for i in 0..cu_words {
            let lo = comptime!(2 * i);
            let hi = comptime!(2 * i + 1);
            sv_s[(svb + i) as usize] = cu[lo as usize] | (cu[hi as usize] << 16u32);
        }
        let mut topc: i32 = comptime!(cu_words as i32 - 1).runtime();
        while topc >= 0 {
            if sv_s[(svb + u32::cast_from(topc)) as usize] != 0u32 {
                break;
            }
            topc -= 1;
        }
        while topc >= 0 {
            let mut rem = 0u32;
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
            while topc >= 0 {
                if sv_s[(svb + u32::cast_from(topc)) as usize] != 0u32 {
                    break;
                }
                topc -= 1;
            }

            let mut chunk = rem;
            if topc >= 0 {
                // Interior chunk: all chunk_digits digits, zeros included.
                #[unroll]
                for _k in 0..chunk_digits {
                    let cq = chunk / base;
                    let d = chunk - cq * base;
                    chunk = cq;
                    if four_masks {
                        if d < 32u32 {
                            m0a |= 1u32 << d;
                        } else if d < 64u32 {
                            m0b |= 1u32 << (d - 32u32);
                        } else if d < 96u32 {
                            m1a |= 1u32 << (d - 64u32);
                        } else {
                            m1b |= 1u32 << (d - 96u32);
                        }
                    } else if d < 32u32 {
                        m0a |= 1u32 << d;
                    } else {
                        m0b |= 1u32 << (d - 32u32);
                    }
                }
            } else {
                // Most significant chunk: digits until zero.
                while chunk != 0u32 {
                    let cq = chunk / base;
                    let d = chunk - cq * base;
                    chunk = cq;
                    if four_masks {
                        if d < 32u32 {
                            m0a |= 1u32 << d;
                        } else if d < 64u32 {
                            m0b |= 1u32 << (d - 32u32);
                        } else if d < 96u32 {
                            m1a |= 1u32 << (d - 64u32);
                        } else {
                            m1b |= 1u32 << (d - 96u32);
                        }
                    } else if d < 32u32 {
                        m0a |= 1u32 << d;
                    } else {
                        m0b |= 1u32 << (d - 32u32);
                    }
                }
            }
        }

        let u = u32::count_ones(m0a)
            + u32::count_ones(m0b)
            + u32::count_ones(m1a)
            + u32::count_ones(m1b);

        hist_s[(copy * hist_bins + u) as usize].fetch_add(1u32);
        if u > cutoff {
            let pos = miss_count[0].fetch_add(1u32);
            if pos < miss_cap {
                let o = MISS_STRIDE * pos;
                miss_data[o as usize] = nw[0];
                miss_data[(o + 1u32) as usize] = nw[1];
                miss_data[(o + 2u32) as usize] = nw[2];
                miss_data[(o + 3u32) as usize] = nw[3];
                miss_data[(o + 4u32) as usize] = u;
            }
        }

        idx += stride;
    }

    sync_cube();
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

/// OR `1 << d` into the mask (`lo`, `hi`) for packed digits `from..ndig` of
/// `x`: the native kernel's u64 mask as a u32 pair. Branch-free: `sel` is all
/// ones for a digit below 32 and zero for one in 32..64 (the kernel serves
/// bases up to 64), and the shift amount is masked to 0..31 explicitly.
#[cube]
fn fd_mask_u32(
    x: &Array<u32>,
    lo: u32,
    hi: u32,
    #[comptime] from: u32,
    #[comptime] ndig: u32,
) -> (u32, u32) {
    let mut mask_lo = lo;
    let mut mask_hi = hi;
    #[unroll]
    for j in from..ndig {
        let w = comptime!(j / 4);
        let sh = comptime!(8 * (j % 4));
        let d = if comptime!(j % 4 == 3) {
            x[w as usize] >> 24u32
        } else {
            (x[w as usize] >> sh) & 0xFFu32
        };
        let bit = 1u32 << (d & 31u32);
        let sel = (d >> 5u32) - 1u32;
        mask_lo |= bit & sel;
        mask_hi |= bit & !sel;
    }
    (mask_lo, mask_hi)
}

/// The finite-difference detailed kernel in u32 arithmetic: the native
/// `detailed_fd_kernel` of [`crate::cubecl_detailed_fd`] with every u64
/// replaced (see the module docs). Thread `t` takes the run of `run_len`
/// consecutive candidates from `start + t * run_len` (grid-stride over runs),
/// seeds its digit arrays from the run's first n with [`fd_seed_split16`] and
/// steps through the run by digit-array addition. Every candidate must lie
/// where n² has `ls` digits and n³ `lc` (see [`fd_plan`]), and the host keeps
/// `count + CUBE_COUNT_X * CUBE_DIM_X * run_len` below 2^32.
///
/// Apple's Metal compiler: this kernel's only divisions by a constant that
/// is not a power of two are the seed's, and those never divide a bare
/// `x >> 16` (see `fd_to_digits`). Keep it that way: the prologue, the mask,
/// the steps and the near-miss record must not divide.
#[cube(launch_unchecked)]
#[allow(
    clippy::too_many_arguments,
    clippy::too_many_lines,
    clippy::similar_names,
    clippy::collapsible_if
)]
fn detailed_fd_kernel_u32(
    hist: &mut Array<Atomic<u32>>,
    miss_count: &mut Array<Atomic<u32>>,
    miss_data: &mut Array<u32>,
    s0: u32,
    s1: u32,
    s2: u32,
    s3: u32,
    count: u32,
    run_len: u32,
    miss_cap: u32,
    #[comptime] base: u32,
    #[comptime] limbs: u32,
    #[comptime] chunk_digits: u32,
    #[comptime] chunk_div: u32,
    #[comptime] cutoff: u32,
    #[comptime] ls: u32,
    #[comptime] lc: u32,
    #[comptime] lds: u32,
    #[comptime] ldc: u32,
    #[comptime] lddc: u32,
) {
    let hist_bins = comptime!(base + 1);
    // Words of S, C, dS, dC, ddC, and the fast words of S and C, exactly as in
    // the native kernel.
    let ws = comptime!(ls.div_ceil(4));
    let wc = comptime!(lc.div_ceil(4));
    let wds = comptime!(lds.div_ceil(4));
    let wdc = comptime!(ldc.div_ceil(4));
    let wddc = comptime!(lddc.div_ceil(4));
    let ks = comptime!((lds + 1).div_ceil(4).min(ws));
    let kc = comptime!((ldc + 1).div_ceil(4).min(wc));
    let ks_dig = comptime!((4 * ks).min(ls));
    let kc_dig = comptime!((4 * kc).min(lc));

    let hist_s = SharedMemory::<Atomic<u32>>::new(comptime!((HIST_COPIES * (base + 1)) as usize));
    let mut zi = UNIT_POS_X;
    while zi < HIST_COPIES * hist_bins {
        hist_s[zi as usize].store(0u32);
        zi += CUBE_DIM_X;
    }
    sync_cube();

    let copy = (UNIT_POS_X >> 5u32) % HIST_COPIES;
    let stride = CUBE_COUNT_X * CUBE_DIM_X * run_len;

    let mut off = ABSOLUTE_POS_X * run_len;
    while off < count {
        let mut len = run_len;
        if count - off < run_len {
            len = count - off;
        }
        // The run's first n = start + off, as 32-bit words with carries.
        let mut nw = Array::<u32>::new(4usize);
        nw[0] = s0 + off;
        let mut carry = u32::cast_from(nw[0] < s0);
        nw[1] = s1 + carry;
        carry = u32::cast_from(nw[1] < carry);
        nw[2] = s2 + carry;
        carry = u32::cast_from(nw[2] < carry);
        nw[3] = s3 + carry;

        // --- seed: the digit arrays of n², n³, 2n+1, 3n²+3n+1, 6n+6 --------
        let mut sd = Array::<u32>::new(ws as usize);
        let mut cd = Array::<u32>::new(wc as usize);
        let mut ds = Array::<u32>::new(wds as usize);
        let mut dc = Array::<u32>::new(wdc as usize);
        let mut ddc = Array::<u32>::new(wddc as usize);
        fd_seed_split16(
            &nw,
            &mut sd,
            &mut cd,
            &mut ds,
            &mut dc,
            &mut ddc,
            base,
            limbs,
            chunk_digits,
            chunk_div,
            ls,
            lc,
            lds,
            ldc,
            lddc,
        );

        // --- walk the run ----------------------------------------------------
        // The cached mask of the slow words of S and C, refreshed by the rare
        // branch below whenever a ripple reaches them.
        let ms0 = fd_mask_u32(&sd, 0u32, 0u32, ks_dig, ls);
        let ms1 = fd_mask_u32(&cd, ms0.0, ms0.1, kc_dig, lc);
        let mut ms_lo = ms1.0;
        let mut ms_hi = ms1.1;
        let mut i = 0u32;
        while i < len {
            let m0 = fd_mask_u32(&sd, ms_lo, ms_hi, 0u32, ks_dig);
            let m1 = fd_mask_u32(&cd, m0.0, m0.1, 0u32, kc_dig);
            let u = u32::count_ones(m1.0) + u32::count_ones(m1.1);
            hist_s[(copy * hist_bins + u) as usize].fetch_add(1u32);
            if u > cutoff {
                let pos = miss_count[0].fetch_add(1u32);
                if pos < miss_cap {
                    // n + i = start + (off + i), with off + i < count.
                    let x0 = s0 + (off + i);
                    let mut c = u32::cast_from(x0 < s0);
                    let x1 = s1 + c;
                    c = u32::cast_from(x1 < c);
                    let x2 = s2 + c;
                    c = u32::cast_from(x2 < c);
                    let x3 = s3 + c;
                    let o = MISS_STRIDE * pos;
                    miss_data[o as usize] = x0;
                    miss_data[(o + 1u32) as usize] = x1;
                    miss_data[(o + 2u32) as usize] = x2;
                    miss_data[(o + 3u32) as usize] = x3;
                    miss_data[(o + 4u32) as usize] = u;
                }
            }

            // Step n -> n+1 as in the native kernel: the five adds without
            // their ripples into the slow words, then one branch for all five.
            let ts = fd_add_top(&mut sd, &ds, wds, ks, base);
            let tds = fd_add_const_top(&mut ds, 2u32, base);
            let tc = fd_add_top(&mut cd, &dc, wdc, kc, base);
            let tdc = fd_add_top(&mut dc, &ddc, wddc, wddc, base);
            let tddc = fd_add_const_top(&mut ddc, 6u32, base);
            let mut all: u32 = 0xFFFF_FFFFu32;
            if comptime!(ks < ws) {
                all &= ts;
            }
            if comptime!(wds > 1) {
                all &= tds;
            }
            if comptime!(kc < wc) {
                all &= tc;
            }
            if comptime!(wddc < wdc) {
                all &= tdc;
            }
            if comptime!(wddc > 1) {
                all &= tddc;
            }
            if all < NO_CARRY {
                let mut refresh: u32 = 0u32;
                if comptime!(ks < ws) {
                    if ts < NO_CARRY {
                        fd_ripple(&mut sd, ks, ws, base);
                        refresh = 1u32;
                    }
                }
                if comptime!(wds > 1) {
                    if tds < NO_CARRY {
                        fd_ripple(&mut ds, 1u32, wds, base);
                    }
                }
                if comptime!(kc < wc) {
                    if tc < NO_CARRY {
                        fd_ripple(&mut cd, kc, wc, base);
                        refresh = 1u32;
                    }
                }
                if comptime!(wddc < wdc) {
                    if tdc < NO_CARRY {
                        fd_ripple(&mut dc, wddc, wdc, base);
                    }
                }
                if comptime!(wddc > 1) {
                    if tddc < NO_CARRY {
                        fd_ripple(&mut ddc, 1u32, wddc, base);
                    }
                }
                if refresh != 0u32 {
                    let r0 = fd_mask_u32(&sd, 0u32, 0u32, ks_dig, ls);
                    let r1 = fd_mask_u32(&cd, r0.0, r0.1, kc_dig, lc);
                    ms_lo = r1.0;
                    ms_hi = r1.1;
                }
            }
            i += 1u32;
        }

        off += stride;
    }

    sync_cube();
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

/// Words of one [`fd_seed_probe_kernel_u32`] record, and where each array
/// starts in it: S at 0, C at 7, dS at 17, dC at 21, ddC at 28 (the widest
/// arrays, at base 64, fill them). Words past an array's top stay 0.
#[doc(hidden)]
pub const FD_SEED_STRIDE: u32 = 32;
#[doc(hidden)]
pub const FD_SEED_OFFSETS: [u32; 5] = [0, 7, 17, 21, 28];

/// [`fd_seed_split16`] for `count` consecutive n from `start`, one n per
/// thread, with n built from u32 words exactly as [`detailed_fd_kernel_u32`]
/// builds a run's first n; the five digit arrays are written out as one
/// [`FD_SEED_STRIDE`]-word record. A histogram hides most seed errors (a
/// wrong digit string has nearly the random-digit distribution), so this is
/// how the seed is checked word for word on a device, browsers included.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments, clippy::similar_names)]
fn fd_seed_probe_kernel_u32(
    out: &mut Array<u32>,
    s0: u32,
    s1: u32,
    s2: u32,
    s3: u32,
    count: u32,
    #[comptime] base: u32,
    #[comptime] limbs: u32,
    #[comptime] chunk_digits: u32,
    #[comptime] chunk_div: u32,
    #[comptime] ls: u32,
    #[comptime] lc: u32,
    #[comptime] lds: u32,
    #[comptime] ldc: u32,
    #[comptime] lddc: u32,
) {
    let t = ABSOLUTE_POS_X;
    if t < count {
        let mut nw = Array::<u32>::new(4usize);
        nw[0] = s0 + t;
        let mut carry = u32::cast_from(nw[0] < s0);
        nw[1] = s1 + carry;
        carry = u32::cast_from(nw[1] < carry);
        nw[2] = s2 + carry;
        carry = u32::cast_from(nw[2] < carry);
        nw[3] = s3 + carry;
        let mut sd = Array::<u32>::new(comptime!(ls.div_ceil(4)) as usize);
        let mut cd = Array::<u32>::new(comptime!(lc.div_ceil(4)) as usize);
        let mut ds = Array::<u32>::new(comptime!(lds.div_ceil(4)) as usize);
        let mut dc = Array::<u32>::new(comptime!(ldc.div_ceil(4)) as usize);
        let mut ddc = Array::<u32>::new(comptime!(lddc.div_ceil(4)) as usize);
        fd_seed_split16(
            &nw,
            &mut sd,
            &mut cd,
            &mut ds,
            &mut dc,
            &mut ddc,
            base,
            limbs,
            chunk_digits,
            chunk_div,
            ls,
            lc,
            lds,
            ldc,
            lddc,
        );
        let o = t * FD_SEED_STRIDE;
        #[unroll]
        for i in 0..comptime!(ls.div_ceil(4)) {
            out[(o + i) as usize] = sd[i as usize];
        }
        #[unroll]
        for i in 0..comptime!(lc.div_ceil(4)) {
            out[(o + 7 + i) as usize] = cd[i as usize];
        }
        #[unroll]
        for i in 0..comptime!(lds.div_ceil(4)) {
            out[(o + 17 + i) as usize] = ds[i as usize];
        }
        #[unroll]
        for i in 0..comptime!(ldc.div_ceil(4)) {
            out[(o + 21 + i) as usize] = dc[i as usize];
        }
        #[unroll]
        for i in 0..comptime!(lddc.div_ceil(4)) {
            out[(o + 28 + i) as usize] = ddc[i as usize];
        }
    }
}

// ============================================================================
// Host side
// ============================================================================

/// Browser-portable implementation of `process_range_detailed`: the u32-only
/// kernels over the context's wgpu client, finite differences inside the
/// base's [`fd_plan`] interval and the per-number kernel elsewhere. The
/// native kernels are faster where they run; this exists for devices whose
/// wgpu backend has no `shader-int64` — browsers foremost, but it runs
/// identically on native wgpu, which is how it is parity-tested without one.
///
/// **Range semantics**: half-open [`range_start`, `range_end`).
///
/// # Errors
/// Returns an error on any device failure, if the near-miss buffer
/// overflows, or on a non-wgpu context (the u32 kernel targets WebGPU; the
/// CUDA runtime always has u64 and the native kernel).
pub async fn process_range_detailed_web_async(
    ctx: &CubeclContext,
    range: &FieldSize,
    base: u32,
) -> Result<FieldResults> {
    process_range_detailed_web_kernel(ctx, range, base, WebKernel::DEFAULT).await
}

/// Which kernel the browser path runs on the candidates inside a base's
/// [`fd_plan`] interval. Everything outside it, and every base without a
/// plan (above 64), always goes to the per-number kernel. Public, but only
/// for tests and the wasm client's test hook; production uses
/// [`WebKernel::DEFAULT`].
#[doc(hidden)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WebKernel {
    /// The per-number kernel for everything: the reference the
    /// finite-difference kernel is tested against.
    Division,
    /// [`detailed_fd_kernel_u32`] with `run_len` candidates per thread and
    /// at most `batch` (no more than [`CUBECL_WEB_BATCH_SIZE`]) per dispatch.
    FiniteDifference { run_len: u32, batch: u128 },
}

impl WebKernel {
    /// The finite-difference kernel at its production settings: the native
    /// run length, and the web batch.
    pub const DEFAULT: Self = Self::FiniteDifference {
        run_len: FD_RUN_LEN,
        batch: CUBECL_WEB_BATCH_SIZE,
    };
}

/// Cubes for a finite-difference dispatch of `count` candidates in runs of
/// `run_len`, one run per thread where the cube cap allows, and the grid
/// stride in candidates that results. The kernel's offsets are u32, so
/// `count + stride` must stay below 2^32; the caller checks it.
fn fd_launch_shape(count: u32, run_len: u32) -> (u32, u64) {
    let runs = count.div_ceil(run_len);
    let cubes = runs.div_ceil(WORKGROUP_SIZE).clamp(1, MAX_CUBES_WEB);
    let stride = u64::from(cubes) * u64::from(WORKGROUP_SIZE) * u64::from(run_len);
    (cubes, stride)
}

/// As [`process_range_detailed_web_async`] with the kernel chosen.
///
/// # Errors
/// As [`process_range_detailed_web_async`]; also on a run length of 0, a
/// batch outside 1..=[`CUBECL_WEB_BATCH_SIZE`], or a run shape whose
/// offsets would not fit u32.
///
/// # Panics
/// Panics if a web batch exceeds u32 candidates, which the batch check
/// makes impossible.
#[doc(hidden)]
#[allow(clippy::too_many_lines)]
pub async fn process_range_detailed_web_kernel(
    ctx: &CubeclContext,
    range: &FieldSize,
    base: u32,
    kernel: WebKernel,
) -> Result<FieldResults> {
    /// Batches between blocking histogram drains; bins are u32 and each
    /// candidate increments exactly one, so the bound is checked below.
    const DRAIN_INTERVAL: usize = 64;
    const _: () = assert!((DRAIN_INTERVAL as u128) * CUBECL_WEB_BATCH_SIZE < u32::MAX as u128);

    if !gpu_supports_base(base) {
        warn!("base {base} not supported on GPU, falling back to CPU for this field");
        return Ok(process_range_detailed(range, base));
    }
    #[allow(irrefutable_let_patterns)] // refutable only with cubecl-cuda on
    let CubeclContext::Wgpu { client, .. } = ctx else {
        anyhow::bail!("the u32 web kernel runs on the wgpu runtime only");
    };

    let (plan, run_len, batch_size) = match kernel {
        WebKernel::FiniteDifference { run_len, batch } => (fd_plan(base), run_len, batch),
        WebKernel::Division => (None, FD_RUN_LEN, CUBECL_WEB_BATCH_SIZE),
    };
    ensure!(
        run_len > 0 && (1..=CUBECL_WEB_BATCH_SIZE).contains(&batch_size),
        "web detailed run length {run_len} or batch {batch_size} out of range"
    );

    let start_time = Instant::now();
    let limbs = n_limbs(base).with_context(|| format!("base {base} has no u128 range"))?;
    let (chunk_digits, chunk_div) = chunk_constants_u16(base);
    let cutoff = get_near_miss_cutoff(base);
    let hist_bins = (base + 1) as usize;

    let miss_count_handle = client.create(cubecl::bytes::Bytes::from_elems(vec![0u32; 1]));
    let miss_data_handle =
        client.empty(NEAR_MISS_CAPACITY_WEB * MISS_STRIDE as usize * core::mem::size_of::<u32>());

    let mut histogram = vec![0u128; hist_bins];
    // Small batches keep single dispatches far under GPU watchdogs; the
    // drain interval bounds u32 bins exactly like the native host loop.
    let mut hist_handle = client.create(cubecl::bytes::Bytes::from_elems(vec![0u32; hist_bins]));
    let mut undrained = 0usize;

    // The per-number kernel over [start, end), at most one batch.
    let launch_division = |hist: &cubecl::server::Handle, start: u128, end: u128| {
        let count = u32::try_from(end - start).expect("web batch fits u32");
        let cubes = count.div_ceil(WORKGROUP_SIZE).clamp(1, MAX_CUBES_WEB);
        unsafe {
            detailed_kernel_u32::launch_unchecked::<cubecl::wgpu::WgpuRuntime>(
                client,
                CubeCount::Static(cubes, 1, 1),
                CubeDim::new_1d(WORKGROUP_SIZE),
                ArrayArg::from_raw_parts(hist.clone(), hist_bins),
                ArrayArg::from_raw_parts(miss_count_handle.clone(), 1),
                ArrayArg::from_raw_parts(
                    miss_data_handle.clone(),
                    NEAR_MISS_CAPACITY_WEB * MISS_STRIDE as usize,
                ),
                start as u32,
                (start >> 32) as u32,
                (start >> 64) as u32,
                (start >> 96) as u32,
                count,
                NEAR_MISS_CAPACITY_WEB as u32,
                base,
                limbs,
                chunk_digits,
                chunk_div,
                cutoff,
            );
        }
    };

    for batch in range.chunks(batch_size) {
        // The finite-difference kernel takes the part of the batch inside its
        // plan's interval (all of it, for a field in the base's legal range);
        // the per-number kernel takes anything outside. Both write the same
        // histogram and near-miss buffers, so the read below is unchanged.
        let (a, z) = (batch.start(), batch.end());
        let (lo, hi) = plan.map_or((z, z), |p| {
            let lo = a.max(p.reg_lo).min(z);
            (lo, z.min(p.reg_hi).max(lo))
        });
        if let Some(p) = plan.filter(|_| hi > lo) {
            let count = u32::try_from(hi - lo).expect("web batch fits u32");
            let (cubes, stride) = fd_launch_shape(count, run_len);
            ensure!(
                u32::try_from(u64::from(count) + stride).is_ok(),
                "web run length {run_len} over {count} candidates overflows the u32 offsets"
            );
            unsafe {
                detailed_fd_kernel_u32::launch_unchecked::<cubecl::wgpu::WgpuRuntime>(
                    client,
                    CubeCount::Static(cubes, 1, 1),
                    CubeDim::new_1d(WORKGROUP_SIZE),
                    ArrayArg::from_raw_parts(hist_handle.clone(), hist_bins),
                    ArrayArg::from_raw_parts(miss_count_handle.clone(), 1),
                    ArrayArg::from_raw_parts(
                        miss_data_handle.clone(),
                        NEAR_MISS_CAPACITY_WEB * MISS_STRIDE as usize,
                    ),
                    lo as u32,
                    (lo >> 32) as u32,
                    (lo >> 64) as u32,
                    (lo >> 96) as u32,
                    count,
                    run_len,
                    NEAR_MISS_CAPACITY_WEB as u32,
                    base,
                    limbs,
                    chunk_digits,
                    chunk_div,
                    cutoff,
                    p.ls,
                    p.lc,
                    p.lds,
                    p.ldc,
                    p.lddc,
                );
            }
        }
        for (start, end) in [(a, lo), (hi, z)] {
            if end > start {
                launch_division(&hist_handle, start, end);
            }
        }
        // One submission per batch: same watchdog rationale as the native
        // backend, and doubly so in a browser tab.
        client
            .flush()
            .map_err(|e| anyhow::anyhow!("stream flush failed: {e:?}"))?;

        undrained += 1;
        if undrained == DRAIN_INTERVAL {
            crate::cubecl_backend::drain(client, hist_handle, &mut histogram).await?;
            hist_handle = client.create(cubecl::bytes::Bytes::from_elems(vec![0u32; hist_bins]));
            undrained = 0;
        }
    }
    // One round trip for the histogram and both near-miss buffers. Every
    // GPU->CPU read costs a queue drain and, in a browser, a task tick before
    // the mapped memory is readable; this tail runs once per slice, so three
    // separate reads paid that latency three times over for data that is
    // ready at the same moment.
    let mut reads = client
        .read_async(vec![hist_handle, miss_count_handle, miss_data_handle])
        .await
        .map_err(|e| anyhow::anyhow!("result read failed: {e:?}"))?;
    let miss_bytes = reads.pop().expect("miss data read");
    let count_bytes = reads.pop().expect("miss count read");
    let hist_bytes = reads.pop().expect("histogram read");
    for (acc, &bin) in histogram
        .iter_mut()
        .zip(u32::from_bytes(&hist_bytes).iter())
    {
        *acc += u128::from(bin);
    }

    // Conservation: every candidate lands in exactly one bin.
    let counted: u128 = histogram.iter().sum();
    ensure!(
        counted == range.size(),
        "GPU histogram counted {counted} of {} candidates; the device dropped work \
         (was the device reset by the browser or driver watchdog?)",
        range.size()
    );

    let miss_total = u32::from_bytes(&count_bytes)[0] as usize;
    ensure!(
        miss_total <= NEAR_MISS_CAPACITY_WEB,
        "near-miss buffer overflow: {miss_total} > {NEAR_MISS_CAPACITY_WEB}"
    );
    let words = u32::from_bytes(&miss_bytes);
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
            "CubeCL web detailed b{base} ({kernel:?}): {:.2e} numbers in {secs:.2}s ({:.2e} n/s), {} near-misses",
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

/// Sync wrapper over [`process_range_detailed_web_async`] for native tests.
///
/// # Errors
/// As the async form.
pub fn process_range_detailed_web(
    ctx: &CubeclContext,
    range: &FieldSize,
    base: u32,
) -> Result<FieldResults> {
    cubecl::future::block_on(process_range_detailed_web_async(ctx, range, base))
}

/// The finite-difference seed for `count` consecutive n from `start` at
/// `base`, as [`fd_seed_probe_kernel_u32`] records ([`FD_SEED_STRIDE`] words
/// each). A test hook, for the native tests and the browser parity check.
///
/// # Errors
/// On a device failure, a non-wgpu context, a base without a plan, `count`
/// outside 1..=65536, or n leaving the plan's interval.
#[doc(hidden)]
pub async fn fd_seed_probe_web_async(
    ctx: &CubeclContext,
    base: u32,
    start: u128,
    count: u32,
) -> Result<Vec<u32>> {
    #[allow(irrefutable_let_patterns)] // refutable only with cubecl-cuda on
    let CubeclContext::Wgpu { client, .. } = ctx else {
        anyhow::bail!("the u32 web kernel runs on the wgpu runtime only");
    };
    let p = fd_plan(base).with_context(|| format!("base {base} has no finite-difference plan"))?;
    ensure!(
        (1..=1 << 16).contains(&count)
            && start >= p.reg_lo
            && start + u128::from(count) <= p.reg_hi,
        "seed probe [{start}, +{count}) outside b{base}'s interval or too long"
    );
    let limbs = n_limbs(base).with_context(|| format!("base {base} has no u128 range"))?;
    let (chunk_digits, chunk_div) = chunk_constants_u16(base);
    let len = (count * FD_SEED_STRIDE) as usize;
    let h = client.create(cubecl::bytes::Bytes::from_elems(vec![0u32; len]));
    unsafe {
        fd_seed_probe_kernel_u32::launch_unchecked::<cubecl::wgpu::WgpuRuntime>(
            client,
            CubeCount::Static(count.div_ceil(64), 1, 1),
            CubeDim::new_1d(64),
            ArrayArg::from_raw_parts(h.clone(), len),
            start as u32,
            (start >> 32) as u32,
            (start >> 64) as u32,
            (start >> 96) as u32,
            count,
            base,
            limbs,
            chunk_digits,
            chunk_div,
            p.ls,
            p.lc,
            p.lds,
            p.ldc,
            p.lddc,
        );
    }
    let bytes = client
        .read_async(vec![h])
        .await
        .map_err(|e| anyhow::anyhow!("seed probe read failed: {e:?}"))?
        .remove(0);
    Ok(u32::from_bytes(&bytes).to_vec())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cubecl_detailed_fd::FD_MAX_BASE;
    use crate::cubecl_detailed_fd::tests::{
        SEED_FIELDS, SEED_STRIDE, cpu_detailed_parallel, seed_expected, windows,
    };

    /// Run shapes the finite-difference tests cover, as the native suite's
    /// with the web batch: the default; a run length that leaves a partial
    /// run in every window; one number per run, so every n is seeded; and
    /// many short batches, each ending in a partial run.
    const WEB_SHAPES: [(u32, u128); 4] = [
        (FD_RUN_LEN, CUBECL_WEB_BATCH_SIZE),
        (61, CUBECL_WEB_BATCH_SIZE),
        (1, CUBECL_WEB_BATCH_SIZE),
        (1000, 333_333),
    ];

    fn web_run(
        ctx: &CubeclContext,
        range: &FieldSize,
        base: u32,
        kernel: WebKernel,
    ) -> FieldResults {
        cubecl::future::block_on(process_range_detailed_web_kernel(ctx, range, base, kernel))
            .expect("web kernel run")
    }

    /// The web kernel against `cpu` over `range`, once per run shape.
    fn assert_web_shapes(
        ctx: &CubeclContext,
        range: &FieldSize,
        base: u32,
        cpu: &FieldResults,
        what: &str,
        shapes: &[(u32, u128)],
    ) {
        for &(run_len, batch) in shapes {
            let gpu = web_run(
                ctx,
                range,
                base,
                WebKernel::FiniteDifference { run_len, batch },
            );
            assert_eq!(
                gpu.distribution, cpu.distribution,
                "b{base} {what}, run {run_len}, batch {batch}: distribution"
            );
            assert_eq!(
                gpu.nice_numbers, cpu.nice_numbers,
                "b{base} {what}, run {run_len}, batch {batch}: near-misses"
            );
        }
    }

    /// The finite-difference launch never overflows its u32 offsets at any
    /// run shape the tests or the browser hook use, and a full default batch
    /// is one run per thread.
    #[test]
    fn web_fd_launch_bounds() {
        for run_len in [1u32, 2, 61, 64, 127, 128, 129, 256, 1000, 4096] {
            for count in [
                1u32,
                2,
                255,
                256,
                257,
                32_767,
                1 << 20,
                31_999_999,
                32_000_000,
            ] {
                let (cubes, stride) = fd_launch_shape(count, run_len);
                assert!((1..=MAX_CUBES_WEB).contains(&cubes));
                assert!(
                    u32::try_from(u64::from(count) + stride).is_ok(),
                    "run {run_len}, count {count}: stride {stride}"
                );
            }
        }
        let batch = u32::try_from(CUBECL_WEB_BATCH_SIZE).unwrap();
        let (cubes, stride) = fd_launch_shape(batch, FD_RUN_LEN);
        assert_eq!(cubes, 977);
        assert!(stride >= u64::from(batch));
    }

    /// The finite-difference web kernel against the CPU on the windows the
    /// browser parity check uses:
    ///
    /// - b40 and b50: the native suite's windows (range start and end,
    ///   straddling the start so each batch splits between the kernels, eight
    ///   seeded, every digit-gain point of an addend), every run shape;
    /// - bases 10-25 in full, every run shape (the windows with near-misses);
    /// - every other base 27-64: the same windows and the middle of the range,
    ///   at the default run length and at 61;
    /// - partial runs at the b40 and b57 range starts;
    /// - the b49 and b40 near-miss windows.
    ///
    /// Bases above 64 have no plan and run the per-number kernel, which
    /// `web_kernel_matches_cpu_detailed` covers.
    #[test]
    #[ignore = "requires a wgpu device"]
    fn web_fd_matches_cpu_detailed() {
        let ctx = CubeclContext::new_default().expect("CubeCL init");
        let start = |base: u32| fd_plan(base).unwrap().reg_lo;
        let mut checked = 0usize;
        for base in [40u32, 50] {
            for (range, what) in windows(base, 100_000) {
                let cpu = cpu_detailed_parallel(&range, base);
                assert_web_shapes(&ctx, &range, base, &cpu, &what, &WEB_SHAPES);
                checked += 1;
            }
            println!("b{base}: every window exact at every run shape");
        }
        let mut misses = 0;
        for base in 10..=25 {
            let Ok(Some(range)) = crate::base_range::get_base_range_u128(base) else {
                continue;
            };
            let cpu = cpu_detailed_parallel(&range, base);
            assert_web_shapes(&ctx, &range, base, &cpu, "full range", &WEB_SHAPES);
            misses += cpu.nice_numbers.len();
            checked += 1;
            println!(
                "b{base} full range ({} numbers): exact, {} near-misses",
                range.size(),
                cpu.nice_numbers.len()
            );
        }
        assert!(misses > 0, "no near-misses compared");
        let two = [
            (FD_RUN_LEN, CUBECL_WEB_BATCH_SIZE),
            (61, CUBECL_WEB_BATCH_SIZE),
        ];
        for base in 27..=FD_MAX_BASE {
            if base == 40 || base == 50 {
                continue;
            }
            let Some(p) = fd_plan(base) else { continue };
            let mid = p.reg_lo + (p.reg_hi - p.reg_lo) / 2;
            let mut ws = windows(base, 100_000);
            ws.push((FieldSize::new(mid - 50_000, mid + 50_000), "middle".into()));
            for (range, what) in &ws {
                let cpu = cpu_detailed_parallel(range, base);
                assert_web_shapes(&ctx, range, base, &cpu, what, &two);
            }
            checked += ws.len();
            println!("b{base}: {} windows exact", ws.len());
        }
        for base in [40u32, 57] {
            let rs = start(base);
            for len in [1u128, 2, 61, 127, 128, 129, 255, 257, 100_001] {
                let range = FieldSize::new(rs, rs + len);
                let cpu = process_range_detailed(&range, base);
                assert_web_shapes(&ctx, &range, base, &cpu, &format!("length {len}"), &two);
                checked += 1;
            }
            println!("b{base}: partial runs exact");
        }
        for (base, at) in [(49u32, 20_363_742_218_601_559u128), (40, 1_917_398_732_423)] {
            let range = FieldSize::new(at - 50_000, at + 50_000);
            let cpu = cpu_detailed_parallel(&range, base);
            assert!(
                cpu.nice_numbers.iter().any(|n| n.number == at),
                "b{base}: {at} is a near miss"
            );
            assert_web_shapes(&ctx, &range, base, &cpu, "near-miss window", &WEB_SHAPES);
            checked += 1;
            println!("b{base} near-miss window around {at}: exact");
        }
        println!("{checked} windows exact");
    }

    /// The web seed, one n at a time: at every base with a plan, the five
    /// digit arrays of each of 1024 n at the interval's start, middle and
    /// end, word for word against the CPU (as the native seed test does for
    /// the native kernel's n-to-words step).
    #[test]
    #[ignore = "requires a wgpu device"]
    fn web_fd_seed_matches_cpu() {
        const COUNT: u32 = 1024;
        assert_eq!(FD_SEED_STRIDE, SEED_STRIDE);
        assert_eq!(FD_SEED_OFFSETS, SEED_FIELDS.map(|f| f.1));
        let ctx = CubeclContext::new_default().expect("CubeCL init");
        let mut bad = Vec::new();
        let mut windows = 0;
        for base in 10..=FD_MAX_BASE {
            let Some(p) = fd_plan(base) else { continue };
            let mid = p.reg_lo + (p.reg_hi - p.reg_lo) / 2;
            let span = u128::from(COUNT).min(p.reg_hi - p.reg_lo);
            let count = span as u32;
            for start in [p.reg_lo, mid.min(p.reg_hi - span), p.reg_hi - span] {
                let rec =
                    cubecl::future::block_on(fd_seed_probe_web_async(&ctx, base, start, count))
                        .expect("seed probe");
                let stride = FD_SEED_STRIDE as usize;
                let wrong: Vec<u128> = (0..count as usize)
                    .filter(|&t| {
                        rec[t * stride..(t + 1) * stride]
                            != seed_expected(start + t as u128, base, &p)[..]
                    })
                    .map(|t| start + t as u128)
                    .collect();
                windows += 1;
                if wrong.is_empty() {
                    println!("b{base} [{start}, +{count}): exact");
                } else {
                    println!(
                        "b{base} [{start}, +{count}): {} n wrong, first {}",
                        wrong.len(),
                        wrong[0]
                    );
                    bad.push(base);
                }
            }
        }
        bad.dedup();
        assert!(bad.is_empty(), "seed digit arrays wrong at bases {bad:?}");
        assert_eq!(windows, 132, "44 bases x 3 windows");
    }

    /// The u32 half-limb multiply's accumulator bound: product + acc + carry
    /// must fit u32 for any 16-bit operands.
    #[test]
    fn half_limb_accumulation_cannot_overflow() {
        let max = u64::from(u16::MAX);
        assert!(u32::try_from(max * max + max + max).is_ok());
    }

    /// A drain interval of web batches must not overflow a u32 bin. Must
    /// track `DRAIN_INTERVAL` in `process_range_detailed_web`, which the
    /// compile-time assert beside it pins; this is the readable statement of
    /// the same bound.
    #[test]
    fn web_batch_drain_interval_cannot_overflow_a_bin() {
        assert!(64 * CUBECL_WEB_BATCH_SIZE < u128::from(u32::MAX));
    }

    /// CPU/web parity across limb widths and both mask layouts — the same
    /// bases and ranges as the native suites, plus base 97 (the widest
    /// supported base: 4 n-limbs, four-mask digit set) — through production's
    /// kernel choice (finite differences to base 64) and through the
    /// per-number kernel alone.
    ///
    /// Runs on lavapipe, which drives the identical naga/WGSL path a browser
    /// uses.
    #[test]
    #[ignore = "requires a wgpu device"]
    fn web_kernel_matches_cpu_detailed() {
        let ctx = CubeclContext::new_default().expect("CubeCL init");
        // Base 10's range is short because over half its candidates are
        // near misses: 1e6 candidates there would overflow the device
        // buffer, which is sized for real browser fields (base 40+).
        for (base, count) in [
            (10u32, 100_000u128),
            (40, 2_000_000),
            (62, 200_000),
            (80, 100_000),
            (97, 50_000),
        ] {
            let Ok(Some(base_range)) = crate::base_range::get_base_range_u128(base) else {
                continue;
            };
            let start = base_range.range_start;
            let range = FieldSize::new(start, start + count);
            let cpu = process_range_detailed(&range, base);
            for kernel in [WebKernel::DEFAULT, WebKernel::Division] {
                let gpu = web_run(&ctx, &range, base, kernel);
                assert_eq!(
                    gpu.distribution, cpu.distribution,
                    "base {base} {kernel:?}: distribution mismatch"
                );
                assert_eq!(
                    gpu.nice_numbers, cpu.nice_numbers,
                    "base {base} {kernel:?}: near-miss mismatch"
                );
            }
            println!("base {base}: {count} candidates match the CPU exactly (both u32 kernels)");
        }
    }

    /// The near miss from the base-49 field that exposed the browser's
    /// precision bug. All three implementations agree it is 20363742218601559
    /// with 45 unique digits; the browser submitted 20363742218601560, because
    /// that value is above 2^53 and had been through a double on the
    /// JavaScript side. The kernels were never wrong — this test is what
    /// established that, and it stays as the coverage that was missing.
    ///
    /// Every other near-miss assertion in this crate is either at base 10
    /// (`web_kernel_finds_69_in_base_10`, one u32 limb) or over a range too
    /// short to contain a near miss at all, so the emission path had never
    /// been exercised at two limbs. This runs the CPU, the u32 web kernel and
    /// the native kernel over the same candidates and compares all three.
    #[test]
    #[ignore = "requires a wgpu device"]
    fn web_kernel_near_miss_matches_cpu_at_base_49() {
        let ctx = CubeclContext::new_default().expect("CubeCL init");
        let hit = 20_363_742_218_601_560u128;
        for (label, range) in [
            ("single candidate", FieldSize::new(hit, hit + 1)),
            ("small window", FieldSize::new(hit - 512, hit + 512)),
        ] {
            let cpu = process_range_detailed(&range, 49);
            let web = process_range_detailed_web(&ctx, &range, 49).expect("web kernel run");
            let native = crate::cubecl_backend::process_range_detailed_cubecl(&ctx, &range, 49)
                .expect("native kernel run");
            println!("--- {label} ---");
            println!("cpu    nice_numbers: {:?}", cpu.nice_numbers);
            println!("web    nice_numbers: {:?}", web.nice_numbers);
            println!("native nice_numbers: {:?}", native.nice_numbers);
            assert_eq!(
                web.distribution, cpu.distribution,
                "{label}: web distribution mismatch"
            );
            assert_eq!(
                native.distribution, cpu.distribution,
                "{label}: native distribution mismatch"
            );
            assert_eq!(
                web.nice_numbers, cpu.nice_numbers,
                "{label}: web near-miss mismatch"
            );
            assert_eq!(
                native.nice_numbers, cpu.nice_numbers,
                "{label}: native near-miss mismatch"
            );
        }
    }

    /// The known solution through the u32 kernel: 69 is nice in base 10.
    #[test]
    #[ignore = "requires a wgpu device"]
    fn web_kernel_finds_69_in_base_10() {
        let ctx = CubeclContext::new_default().expect("CubeCL init");
        let range = FieldSize::new(47, 100);
        let results = process_range_detailed_web(&ctx, &range, 10).expect("web kernel run");
        let hit = results
            .nice_numbers
            .iter()
            .find(|n| n.number == 69)
            .expect("69 not found in base 10");
        assert_eq!(hit.num_uniques, 10);
    }
}
