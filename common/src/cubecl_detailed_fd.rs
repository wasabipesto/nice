//! Finite-difference detailed kernel for `CubeCL`: the unique-digit histogram
//! without a division per number.
//!
//! [`crate::cubecl_backend`]'s per-number kernel squares and cubes every n and
//! divides both results down to digits. Over a run of consecutive n the
//! division is unnecessary: with dS = 2n+1, dC = 3n²+3n+1 and ddC = 6n+6,
//! stepping n to n+1 is S += dS, dS += 2, C += dC, dC += ddC, ddC += 6. Here
//! each thread takes a run of [`FD_RUN_LEN`] consecutive n, seeds the five
//! values once from the run's first n (limb products and a division-by-
//! constant chunk split as in the per-number kernel; see [`fd_seed`]), and
//! keeps them as base-b digit arrays in registers, four digits to a
//! u32 (one byte lane each, little-endian). Each step is five digit-array
//! additions, each four digits per native add: adding 256 - b to every byte of
//! the addend makes a lane overflow into the next exactly when its digit sum
//! reaches b, and the lanes that did not overflow give the bias back
//! (`fd_add_top`). Words of S and C above the addend's width change only when
//! a carry ripples through four digits of b-1, so their part of the digit mask
//! is cached per run and one rare branch per step does every such ripple and
//! the cache refresh. The mask build, histogram and near-miss records are the
//! per-number kernel's.
//!
//! Exactness: the seed is exact integer arithmetic, and every step is an exact
//! base-b addition, so after k steps the arrays hold exactly (n+k)², (n+k)³ and
//! the three differences. The biased add needs every lane below b (b ≤ 128),
//! which its own output keeps true. The kernel is specialised to digit
//! lengths: n² has exactly `ls` digits and n³ exactly `lc` for every n in a
//! base's legal range (their digit counts sum to the base and neither can
//! fall, so neither can rise), so the mask covers comptime positions and
//! leading zeros never enter it; [`fd_plan`] computes that interval exactly,
//! and the host sends any part of a batch outside it to the per-number kernel.
//! The addends' widths are taken at the interval's top, where they are
//! largest, so no addend can outgrow its array.
//!
//! Portability: the kernel needs what the per-number kernel needs (u64 add,
//! multiply and shift; a u64 division only in the CUDA and HIP seed, the
//! split16 division on wgpu). It is exact on CUDA, on Vulkan (lavapipe,
//! through naga's WGSL and through `CubeCL`'s SPIR-V compiler) and on an
//! Apple M4 (macOS 26.3) through Metal, whose compiler miscompiled the
//! seed's first form (see [`fd_to_digits`]).
//!
//! The wgpu seed, [`fd_seed_split16`], and the step helpers use u32 only, so
//! that a kernel without u64, as on WebGPU in the browser, can share them;
//! they are `pub(crate)` for that.

#![cfg(feature = "cubecl")]
// The cube macro evaluates comptime! expressions host-side, where fn-level
// allows do not reach; these lints are all macro-expansion artifacts.
#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_lossless,
    clippy::used_underscore_binding
)]

use crate::cubecl_backend::{HIST_COPIES, MISS_STRIDE};
use crate::gpu_config::gpu_supports_base;
use cubecl::prelude::*;
use malachite::base::num::arithmetic::traits::{CeilingRoot, DivAssignRem, Pow};
use malachite::natural::Natural;

/// Consecutive candidates per thread. The seed costs about 5 % of a run's
/// work at 128; measured on an RTX 3080, throughput is flat (within 3 %) from
/// 64 to 512 and drops below 64 (the seed) and at 2048 (too few threads to
/// fill the device at one batch per dispatch).
pub const FD_RUN_LEN: u32 = 128;

/// Highest base the kernel serves: the digit mask is one u64. (The biased add
/// itself holds to base 128.)
pub const FD_MAX_BASE: u32 = 64;

/// Bit 31 of a word's raw biased sum is clear exactly when its top lane
/// carried out of the word.
pub(crate) const NO_CARRY: u32 = 0x8000_0000;

/// Biased add `x += y` over words `0..wy`, then the carry alone into words
/// `wy..kd`. Nothing is added above `kd`: returns the last word's raw sum,
/// whose bit 31 is clear exactly when it carried out, for the caller's rare
/// ripple.
#[cube]
pub(crate) fn fd_add_top(
    x: &mut Array<u32>,
    y: &Array<u32>,
    #[comptime] wy: u32,
    #[comptime] kd: u32,
    #[comptime] base: u32,
) -> u32 {
    let bias = comptime!((256 - base) * 0x0101_0101u32);
    let fix = comptime!(256 - base);
    let mut cin: u32 = 0u32;
    let mut t: u32 = 0u32;
    #[unroll]
    for w in 0..wy {
        // A lane that carried out holds its digit sum minus b, below b, so
        // bit 7 is clear; one that did not holds at least 256 - b, bit 7 set.
        t = x[w as usize] + y[w as usize] + bias + cin;
        cin = (t >> 31u32) ^ 1u32;
        x[w as usize] = t - ((t >> 7u32) & 0x0101_0101u32) * fix;
    }
    #[unroll]
    for w in wy..kd {
        t = x[w as usize] + bias + cin;
        cin = (t >> 31u32) ^ 1u32;
        x[w as usize] = t - ((t >> 7u32) & 0x0101_0101u32) * fix;
    }
    t
}

/// Biased add of a one-digit constant into word 0; returns the raw sum as
/// [`fd_add_top`] does.
#[cube]
pub(crate) fn fd_add_const_top(
    x: &mut Array<u32>,
    #[comptime] c: u32,
    #[comptime] base: u32,
) -> u32 {
    let bias = comptime!((256 - base) * 0x0101_0101u32);
    let fix = comptime!(256 - base);
    let t = x[0] + comptime!(bias + c);
    x[0] = t - ((t >> 7u32) & 0x0101_0101u32) * fix;
    t
}

/// Add a carry of 1 into word `from` and ripple it up to word `wx - 1`.
#[cube]
pub(crate) fn fd_ripple(
    x: &mut Array<u32>,
    #[comptime] from: u32,
    #[comptime] wx: u32,
    #[comptime] base: u32,
) {
    let bias = comptime!((256 - base) * 0x0101_0101u32);
    let fix = comptime!(256 - base);
    let mut cin = 1u32;
    #[unroll]
    for w in from..wx {
        let t = x[w as usize] + bias + cin;
        cin = (t >> 31u32) ^ 1u32;
        x[w as usize] = t - ((t >> 7u32) & 0x0101_0101u32) * fix;
    }
}

/// Convert the little-endian limbs `v` (destroyed) into `ndig` packed
/// base-b digits in `out` (pre-zeroed), `chunk_digits` digits per long
/// division by `chunk_div` = b^`chunk_digits`. With `wide_chunk` the limbs
/// are 32-bit and each step divides a limb pair in u64 (a multiply-high
/// under nvcc and hipcc), as in the per-number kernel; otherwise (split16)
/// they are 16-bit, one to a u32 (at least two), and each step divides
/// `rem << 16 | limb` in u32, with `chunk_div` below 2^16, for targets
/// without a fast u64 division.
///
/// In split16 the first step of each division takes the top two limbs as
/// one u32 (the remainder is still 0 there, so this is the same long
/// division). Dividing the top limb alone would divide, in the first chunk,
/// a value that is a bare `t >> 16` (the top limb of a product or sum), and
/// on an Apple M4 the Metal compiler miscompiles a constant division whose
/// dividend is a right shift by half its width (see the
/// `metal_half_shift_div_repro` test): the seed came out wrong at b23,
/// b25, b27-b30 and b49-b55 there. The pair is `t` itself, and no dividend
/// below is a bare shift.
#[cube]
#[allow(clippy::too_many_arguments)]
pub(crate) fn fd_to_digits(
    v: &mut Array<u32>,
    out: &mut Array<u32>,
    #[comptime] nlimbs: u32,
    #[comptime] ndig: u32,
    #[comptime] base: u32,
    #[comptime] chunk_digits: u32,
    #[comptime] chunk_div: u32,
    #[comptime] wide_chunk: bool,
) {
    let nchunks = comptime!(ndig.div_ceil(chunk_digits));
    #[unroll]
    for k in 0..nchunks {
        let mut chunk = if wide_chunk {
            let mut rem64 = 0u64;
            #[unroll]
            for ii in 0..nlimbs {
                let i = comptime!(nlimbs - 1 - ii);
                let cur = (rem64 << 32u64) | u64::cast_from(v[i as usize]);
                let q = cur / u64::cast_from(chunk_div);
                v[i as usize] = u32::cast_from(q);
                rem64 = cur - q * u64::cast_from(chunk_div);
            }
            u32::cast_from(rem64)
        } else {
            let top = comptime!(nlimbs - 1);
            let next = comptime!(nlimbs - 2);
            let c = (v[top as usize] << 16u32) | v[next as usize];
            let q = c / chunk_div;
            let mut rem = c - q * chunk_div;
            v[top as usize] = q >> 16u32;
            v[next as usize] = q & 0xFFFFu32;
            #[unroll]
            for ii in 2..nlimbs {
                let i = comptime!(nlimbs - 1 - ii);
                let c = (rem << 16u32) | v[i as usize];
                let q = c / chunk_div;
                rem = c - q * chunk_div;
                v[i as usize] = q;
            }
            rem
        };
        #[unroll]
        for i in 0..chunk_digits {
            let j = comptime!(k * chunk_digits + i);
            if comptime!(j < ndig) {
                let cq = chunk / base;
                let d = chunk - cq * base;
                chunk = cq;
                let w = comptime!(j / 4);
                let sh = comptime!(8 * (j % 4));
                out[w as usize] |= d << sh;
            }
        }
    }
}

/// Zero the words of `x` that hold `ndig` packed digits.
#[cube]
fn fd_clear(x: &mut Array<u32>, #[comptime] ndig: u32) {
    #[unroll]
    for i in 0..comptime!(ndig.div_ceil(4)) {
        x[i as usize] = 0u32;
    }
}

/// The run's seed: the packed base-b digits of n² into `sd`, n³ into `cd`,
/// 2n+1 into `ds`, 3n²+3n+1 into `dc` and 6n+6 into `ddc`, for n =
/// `n_hi`·2^64 + `n_lo`.
///
/// The wide flavor (CUDA, HIP) multiplies 32-bit limbs in u64. The split16
/// flavor (wgpu) splits n into 32-bit words and runs [`fd_seed_split16`].
#[cube]
#[allow(
    clippy::too_many_arguments,
    clippy::too_many_lines,
    clippy::similar_names
)]
fn fd_seed(
    n_lo: u64,
    n_hi: u64,
    sd: &mut Array<u32>,
    cd: &mut Array<u32>,
    ds: &mut Array<u32>,
    dc: &mut Array<u32>,
    ddc: &mut Array<u32>,
    #[comptime] base: u32,
    #[comptime] limbs: u32,
    #[comptime] chunk_digits: u32,
    #[comptime] chunk_div: u32,
    #[comptime] wide_chunk: bool,
    #[comptime] ls: u32,
    #[comptime] lc: u32,
    #[comptime] lds: u32,
    #[comptime] ldc: u32,
    #[comptime] lddc: u32,
) {
    if comptime!(wide_chunk) {
        let sq_limbs = comptime!(2 * limbs);
        let cu_limbs = comptime!(3 * limbs);
        fd_clear(sd, ls);
        fd_clear(cd, lc);
        fd_clear(ds, lds);
        fd_clear(dc, ldc);
        fd_clear(ddc, lddc);
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
        // 2n+1 and 6n+6 (limbs + 1 words), 3n²+3n+1 (2·limbs + 1 words).
        let mut v_ds = Array::<u32>::new(comptime!(limbs + 1) as usize);
        let mut v_ddc = Array::<u32>::new(comptime!(limbs + 1) as usize);
        let mut c_ds = 1u64;
        let mut c_ddc = 6u64;
        #[unroll]
        for i in 0..comptime!(limbs + 1) {
            if comptime!(i < limbs) {
                c_ds += 2u64 * u64::cast_from(nl[i as usize]);
                c_ddc += 6u64 * u64::cast_from(nl[i as usize]);
            }
            v_ds[i as usize] = u32::cast_from(c_ds);
            v_ddc[i as usize] = u32::cast_from(c_ddc);
            c_ds >>= 32u64;
            c_ddc >>= 32u64;
        }
        let mut v_dc = Array::<u32>::new(comptime!(sq_limbs + 1) as usize);
        let mut c_dc = 1u64;
        #[unroll]
        for i in 0..comptime!(sq_limbs + 1) {
            if comptime!(i < sq_limbs) {
                c_dc += 3u64 * u64::cast_from(sq[i as usize]);
            }
            if comptime!(i < limbs) {
                c_dc += 3u64 * u64::cast_from(nl[i as usize]);
            }
            v_dc[i as usize] = u32::cast_from(c_dc);
            c_dc >>= 32u64;
        }

        fd_to_digits(
            &mut sq,
            sd,
            sq_limbs,
            ls,
            base,
            chunk_digits,
            chunk_div,
            wide_chunk,
        );
        fd_to_digits(
            &mut cu,
            cd,
            cu_limbs,
            lc,
            base,
            chunk_digits,
            chunk_div,
            wide_chunk,
        );
        fd_to_digits(
            &mut v_ds,
            ds,
            comptime!(limbs + 1),
            lds,
            base,
            chunk_digits,
            chunk_div,
            wide_chunk,
        );
        fd_to_digits(
            &mut v_dc,
            dc,
            comptime!(sq_limbs + 1),
            ldc,
            base,
            chunk_digits,
            chunk_div,
            wide_chunk,
        );
        fd_to_digits(
            &mut v_ddc,
            ddc,
            comptime!(limbs + 1),
            lddc,
            base,
            chunk_digits,
            chunk_div,
            wide_chunk,
        );
    } else {
        let mut nw = Array::<u32>::new(limbs as usize);
        #[unroll]
        for i in 0..limbs {
            let shift = comptime!(((i & 1) * 32) as u64);
            if comptime!(i < 2) {
                nw[i as usize] = u32::cast_from(n_lo >> shift);
            } else {
                nw[i as usize] = u32::cast_from(n_hi >> shift);
            }
        }
        fd_seed_split16(
            &nw,
            sd,
            cd,
            ds,
            dc,
            ddc,
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
    }
}

/// [`fd_seed`]'s split16 flavor, in u32 arithmetic only: the five digit
/// arrays for the n whose 32-bit words, little-endian, are `nw[0..limbs]`.
///
/// It works in 16-bit limbs, one to a u32, so that [`fd_to_digits`] can take
/// a value's top 32 bits as one dividend: with 32-bit limbs the top limb of
/// n² or n³ is the high half of a u64 sum or product, and dividing that by a
/// constant is the Apple M4 Metal miscompile `fd_to_digits` describes. Every
/// product here is of two 16-bit limbs and every sum stays below 2^32 (a
/// limb product plus two values below 2^16 is at most 2^32 - 1).
#[cube]
#[allow(
    clippy::too_many_arguments,
    clippy::too_many_lines,
    clippy::similar_names
)]
pub(crate) fn fd_seed_split16(
    nw: &Array<u32>,
    sd: &mut Array<u32>,
    cd: &mut Array<u32>,
    ds: &mut Array<u32>,
    dc: &mut Array<u32>,
    ddc: &mut Array<u32>,
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
    fd_clear(sd, ls);
    fd_clear(cd, lc);
    fd_clear(ds, lds);
    fd_clear(dc, ldc);
    fd_clear(ddc, lddc);
    let h = comptime!(2 * limbs);
    let mut nh = Array::<u32>::new(h as usize);
    #[unroll]
    for i in 0..limbs {
        let w = nw[i as usize];
        nh[comptime!(2 * i) as usize] = w & 0xFFFFu32;
        nh[comptime!(2 * i + 1) as usize] = w >> 16u32;
    }
    let mut sq = Array::<u32>::new(comptime!(2 * h) as usize);
    #[unroll]
    for i in 0..comptime!(2 * h) {
        sq[i as usize] = 0u32;
    }
    #[unroll]
    for i in 0..h {
        let mut carry = 0u32;
        #[unroll]
        for j in 0..h {
            let k = comptime!(i + j);
            let t = nh[i as usize] * nh[j as usize] + sq[k as usize] + carry;
            sq[k as usize] = t & 0xFFFFu32;
            carry = t >> 16u32;
        }
        sq[comptime!(i + h) as usize] = carry;
    }
    let mut cu = Array::<u32>::new(comptime!(3 * h) as usize);
    #[unroll]
    for i in 0..comptime!(3 * h) {
        cu[i as usize] = 0u32;
    }
    #[unroll]
    for i in 0..comptime!(2 * h) {
        let mut carry = 0u32;
        #[unroll]
        for j in 0..h {
            let k = comptime!(i + j);
            let t = sq[i as usize] * nh[j as usize] + cu[k as usize] + carry;
            cu[k as usize] = t & 0xFFFFu32;
            carry = t >> 16u32;
        }
        cu[comptime!(i + h) as usize] = carry;
    }
    // 2n+1 and 6n+6 (h + 1 limbs), 3n²+3n+1 (2h + 1 limbs).
    let mut v_ds = Array::<u32>::new(comptime!(h + 1) as usize);
    let mut v_ddc = Array::<u32>::new(comptime!(h + 1) as usize);
    let mut c_ds = 1u32;
    let mut c_ddc = 6u32;
    #[unroll]
    for i in 0..comptime!(h + 1) {
        if comptime!(i < h) {
            c_ds += 2u32 * nh[i as usize];
            c_ddc += 6u32 * nh[i as usize];
        }
        v_ds[i as usize] = c_ds & 0xFFFFu32;
        v_ddc[i as usize] = c_ddc & 0xFFFFu32;
        c_ds >>= 16u32;
        c_ddc >>= 16u32;
    }
    let mut v_dc = Array::<u32>::new(comptime!(2 * h + 1) as usize);
    let mut c_dc = 1u32;
    #[unroll]
    for i in 0..comptime!(2 * h + 1) {
        if comptime!(i < 2 * h) {
            c_dc += 3u32 * sq[i as usize];
        }
        if comptime!(i < h) {
            c_dc += 3u32 * nh[i as usize];
        }
        v_dc[i as usize] = c_dc & 0xFFFFu32;
        c_dc >>= 16u32;
    }
    fd_to_digits(
        &mut sq,
        sd,
        comptime!(2 * h),
        ls,
        base,
        chunk_digits,
        chunk_div,
        false,
    );
    fd_to_digits(
        &mut cu,
        cd,
        comptime!(3 * h),
        lc,
        base,
        chunk_digits,
        chunk_div,
        false,
    );
    fd_to_digits(
        &mut v_ds,
        ds,
        comptime!(h + 1),
        lds,
        base,
        chunk_digits,
        chunk_div,
        false,
    );
    fd_to_digits(
        &mut v_dc,
        dc,
        comptime!(2 * h + 1),
        ldc,
        base,
        chunk_digits,
        chunk_div,
        false,
    );
    fd_to_digits(
        &mut v_ddc,
        ddc,
        comptime!(h + 1),
        lddc,
        base,
        chunk_digits,
        chunk_div,
        false,
    );
}

/// OR `1 << d` into `m` for packed digits `from..ndig` of `x`.
#[cube]
fn fd_mask(x: &Array<u32>, m: u64, #[comptime] from: u32, #[comptime] ndig: u32) -> u64 {
    let mut mm = m;
    #[unroll]
    for j in from..ndig {
        let w = comptime!(j / 4);
        let sh = comptime!(8 * (j % 4));
        let d = if comptime!(j % 4 == 3) {
            x[w as usize] >> 24u32
        } else {
            (x[w as usize] >> sh) & 0xFFu32
        };
        mm |= 1u64 << u64::cast_from(d);
    }
    mm
}

/// The finite-difference detailed kernel: thread `t` takes the run of
/// `run_len` consecutive candidates from `start + t * run_len` (grid-stride
/// over runs), seeds its digit arrays from the run's first n and steps
/// through the run by digit-array addition. Every candidate must lie where n²
/// has `ls` digits and n³ `lc` (see [`fd_plan`]).
#[cube(launch_unchecked)]
#[allow(
    clippy::too_many_arguments,
    clippy::too_many_lines,
    clippy::similar_names,
    clippy::collapsible_if
)]
pub(crate) fn detailed_fd_kernel(
    hist: &mut Array<Atomic<u32>>,
    miss_count: &mut Array<Atomic<u32>>,
    miss_data: &mut Array<u32>,
    s0: u32,
    s1: u32,
    s2: u32,
    s3: u32,
    cnt_lo: u32,
    cnt_hi: u32,
    run_len: u32,
    miss_cap: u32,
    #[comptime] base: u32,
    #[comptime] limbs: u32,
    #[comptime] chunk_digits: u32,
    #[comptime] chunk_div: u32,
    #[comptime] wide_chunk: bool,
    #[comptime] cutoff: u32,
    #[comptime] ls: u32,
    #[comptime] lc: u32,
    #[comptime] lds: u32,
    #[comptime] ldc: u32,
    #[comptime] lddc: u32,
) {
    let hist_bins = comptime!(base + 1);
    // Words of S, C, dS, dC, ddC.
    let ws = comptime!(ls.div_ceil(4));
    let wc = comptime!(lc.div_ceil(4));
    let wds = comptime!(lds.div_ceil(4));
    let wdc = comptime!(ldc.div_ceil(4));
    let wddc = comptime!(lddc.div_ceil(4));
    // Words of S and C that change on most steps: the addend's words and the
    // word holding the digit just above the addend's top, which takes its
    // carry. A word above them changes only when a carry ripples through
    // every digit between, all b-1: at most once per b^4 steps or so.
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

    let start_lo = (u64::cast_from(s1) << 32u64) | u64::cast_from(s0);
    let start_hi = (u64::cast_from(s3) << 32u64) | u64::cast_from(s2);
    let count = (u64::cast_from(cnt_hi) << 32u64) | u64::cast_from(cnt_lo);
    let copy = (UNIT_POS_X >> 5u32) % HIST_COPIES;
    let stride =
        u64::cast_from(CUBE_COUNT_X) * u64::cast_from(CUBE_DIM_X) * u64::cast_from(run_len);

    let mut off = u64::cast_from(ABSOLUTE_POS_X) * u64::cast_from(run_len);
    while off < count {
        let mut len = run_len;
        if count - off < u64::cast_from(run_len) {
            len = u32::cast_from(count - off);
        }
        let n_lo = start_lo + off;
        let mut n_hi = start_hi;
        if n_lo < start_lo {
            n_hi += 1u64;
        }

        // --- seed: the digit arrays of n², n³, 2n+1, 3n²+3n+1, 6n+6 --------
        let mut sd = Array::<u32>::new(ws as usize);
        let mut cd = Array::<u32>::new(wc as usize);
        let mut ds = Array::<u32>::new(wds as usize);
        let mut dc = Array::<u32>::new(wdc as usize);
        let mut ddc = Array::<u32>::new(wddc as usize);
        fd_seed(
            n_lo,
            n_hi,
            &mut sd,
            &mut cd,
            &mut ds,
            &mut dc,
            &mut ddc,
            base,
            limbs,
            chunk_digits,
            chunk_div,
            wide_chunk,
            ls,
            lc,
            lds,
            ldc,
            lddc,
        );

        // --- walk the run ----------------------------------------------------
        // The cached mask of the slow words of S and C (digits from ks_dig and
        // kc_dig up). It must be refreshed whenever a ripple reaches them,
        // which the rare branch below does; everything else reads only the
        // fast words.
        let mut ms = fd_mask(&sd, 0u64, ks_dig, ls);
        ms = fd_mask(&cd, ms, kc_dig, lc);
        let mut i = 0u32;
        while i < len {
            let mut m = fd_mask(&sd, ms, 0u32, ks_dig);
            m = fd_mask(&cd, m, 0u32, kc_dig);
            let u = u32::cast_from(m).count_ones() + u32::cast_from(m >> 32u64).count_ones();
            hist_s[(copy * hist_bins + u) as usize].fetch_add(1u32);
            if u > cutoff {
                let pos = miss_count[0].fetch_add(1u32);
                if pos < miss_cap {
                    let x_lo = n_lo + u64::cast_from(i);
                    let mut x_hi = n_hi;
                    if x_lo < n_lo {
                        x_hi += 1u64;
                    }
                    let o = MISS_STRIDE * pos;
                    miss_data[o as usize] = u32::cast_from(x_lo);
                    miss_data[(o + 1u32) as usize] = u32::cast_from(x_lo >> 32u64);
                    miss_data[(o + 2u32) as usize] = u32::cast_from(x_hi);
                    miss_data[(o + 3u32) as usize] = u32::cast_from(x_hi >> 32u64);
                    miss_data[(o + 4u32) as usize] = u;
                }
            }

            // Step n -> n+1: the five adds without their ripples into the
            // slow words, then one branch for all five. Each add returns the
            // raw sum of the last word it touched, and a word carried out iff
            // bit 31 of that sum is clear, so the AND of the sums that can
            // carry into a further word is below NO_CARRY iff any did. An add
            // whose last word is its array's top word has nowhere to carry
            // (the array is wide enough for the interval), so it is left out.
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
                // A ripple changed a slow word of S or C: its cached mask is
                // stale.
                if refresh != 0u32 {
                    ms = fd_mask(&sd, 0u64, ks_dig, ls);
                    ms = fd_mask(&cd, ms, kc_dig, lc);
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

/// The digit lengths the kernel is specialised to at one base: n² has `ls`
/// digits and n³ `lc` for every n in `[reg_lo, reg_hi)`, and the addends
/// 2n+1, 3n²+3n+1 and 6n+6 fit in `lds`, `ldc` and `lddc` digits up to
/// `reg_hi`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FdPlan {
    pub ls: u32,
    pub lc: u32,
    pub lds: u32,
    pub ldc: u32,
    pub lddc: u32,
    pub reg_lo: u128,
    pub reg_hi: u128,
}

/// Number of base-`base` digits of `x` (0 for 0).
fn natural_digits(x: &Natural, base: u32) -> u32 {
    let mut y = x.clone();
    let mut c = 0;
    while y != 0u32 {
        y.div_assign_rem(Natural::from(base));
        c += 1;
    }
    c
}

/// The finite-difference plan for `base`, or `None` where the kernel does not
/// apply: no u128 range, or above [`FD_MAX_BASE`].
#[must_use]
#[allow(clippy::similar_names)]
pub fn fd_plan(base: u32) -> Option<FdPlan> {
    if base > FD_MAX_BASE || !gpu_supports_base(base) {
        return None;
    }
    let range = crate::base_range::get_base_range_u128(base).ok()??;
    let b = Natural::from(base);
    let n0 = Natural::from(range.range_start);
    let ls = natural_digits(&(&n0 * &n0), base);
    let lc = natural_digits(&(&n0 * &n0 * &n0), base);
    // n² has ls digits for ceil(sqrt(b^(ls-1))) <= n < ceil(sqrt(b^ls)), and
    // likewise n³.
    let lo_s = b.clone().pow(u64::from(ls - 1)).ceiling_root(2);
    let lo_c = b.clone().pow(u64::from(lc - 1)).ceiling_root(3);
    let hi_s = b.clone().pow(u64::from(ls)).ceiling_root(2);
    let hi_c = b.pow(u64::from(lc)).ceiling_root(3);
    let reg_lo = u128::try_from(&lo_s.max(lo_c)).ok()?;
    let hi = hi_s.min(hi_c);
    let reg_hi = u128::try_from(&hi).ok()?;
    // The addends after the interval's last step, the largest they get.
    let lds = natural_digits(&(&hi * Natural::from(2u32) + Natural::from(1u32)), base);
    let ldc = natural_digits(
        &(&hi * &hi * Natural::from(3u32) + &hi * Natural::from(3u32) + Natural::from(1u32)),
        base,
    );
    let lddc = natural_digits(&(&hi * Natural::from(6u32) + Natural::from(6u32)), base);
    Some(FdPlan {
        ls,
        lc,
        lds,
        ldc,
        lddc,
        reg_lo,
        reg_hi,
    })
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::client_process::process_range_detailed;
    use crate::cubecl_backend::{
        CUBECL_BATCH_SIZE, CubeclContext, DetailedKernel, process_range_detailed_cubecl_kernel,
    };
    use crate::{FieldResults, FieldSize};

    /// The kernel's interval is exactly the legal range at every base it
    /// serves, so production fields never leave it, and nothing above
    /// [`FD_MAX_BASE`] gets a plan.
    #[test]
    fn plan_covers_exactly_the_legal_range() {
        let mut served = 0;
        for base in 10..=FD_MAX_BASE {
            let Ok(Some(range)) = crate::base_range::get_base_range_u128(base) else {
                assert!(fd_plan(base).is_none(), "b{base}: plan without a range");
                continue;
            };
            let plan = fd_plan(base).unwrap_or_else(|| panic!("b{base}: no plan"));
            assert_eq!(
                (plan.reg_lo, plan.reg_hi),
                (range.range_start, range.range_end),
                "b{base}: regular interval"
            );
            assert_eq!(plan.ls + plan.lc, base, "b{base}: digit lengths");
            served += 1;
        }
        assert_eq!(served, 44, "bases 10-64 with a range");
        for base in FD_MAX_BASE + 1..=128 {
            assert!(fd_plan(base).is_none(), "b{base}");
        }
    }

    /// `process_range_detailed` over scoped threads, merged.
    pub(crate) fn cpu_detailed_parallel(range: &FieldSize, base: u32) -> FieldResults {
        let threads =
            std::thread::available_parallelism().map_or(8, std::num::NonZero::get) as u128;
        let step = range.size().div_ceil(threads).max(1);
        let parts: Vec<FieldResults> = std::thread::scope(|sc| {
            let hs: Vec<_> = (0..threads)
                .filter_map(|t| {
                    let a = range.start() + t * step;
                    let z = (a + step).min(range.end());
                    (a < z).then(|| {
                        sc.spawn(move || process_range_detailed(&FieldSize::new(a, z), base))
                    })
                })
                .collect();
            hs.into_iter().map(|h| h.join().unwrap()).collect()
        });
        let mut distribution = parts[0].distribution.clone();
        let mut nice_numbers = Vec::new();
        for (k, p) in parts.iter().enumerate() {
            if k > 0 {
                for (acc, d) in distribution.iter_mut().zip(&p.distribution) {
                    acc.count += d.count;
                }
            }
            nice_numbers.extend(p.nice_numbers.iter().cloned());
        }
        nice_numbers.sort_by_key(|n| n.number);
        FieldResults {
            distribution,
            nice_numbers,
        }
    }

    /// Run shapes the exactness tests cover: the default; a run length that
    /// leaves a partial run in every window; one number per run, so every n
    /// is seeded by division; and many short batches, each ending in a
    /// partial run.
    const SHAPES: [(u32, u128); 4] = [
        (FD_RUN_LEN, CUBECL_BATCH_SIZE),
        (61, CUBECL_BATCH_SIZE),
        (1, CUBECL_BATCH_SIZE),
        (1000, 333_333),
    ];

    fn run(
        ctx: &CubeclContext,
        range: &FieldSize,
        base: u32,
        kernel: DetailedKernel,
    ) -> FieldResults {
        cubecl::future::block_on(process_range_detailed_cubecl_kernel(
            ctx, range, base, kernel,
        ))
        .expect("cubecl detailed run")
    }

    fn assert_shapes_match(
        ctx: &CubeclContext,
        range: &FieldSize,
        base: u32,
        cpu: &FieldResults,
        what: &str,
    ) {
        for (run_len, batch) in SHAPES {
            let gpu = run(
                ctx,
                range,
                base,
                DetailedKernel::FiniteDifference { run_len, batch },
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

    /// Windows of `w` numbers at b40 and b50: the range start and end, one
    /// straddling the range start (n³ gains a digit there, so each batch
    /// splits between the two kernels), eight seeded random ones, and one
    /// centred on every point inside the range where an addend gains a
    /// digit.
    pub(crate) fn windows(base: u32, width: u128) -> Vec<(FieldSize, String)> {
        let legal = crate::base_range::get_base_range_u128(base)
            .unwrap()
            .unwrap();
        let (rs, re) = (legal.range_start, legal.range_end);
        let mut out = vec![
            (
                FieldSize::new(rs, rs + 2 * width),
                "range start".to_string(),
            ),
            (FieldSize::new(re - 2 * width, re), "range end".into()),
            (
                FieldSize::new(rs - width, rs + width),
                "straddling the range start".into(),
            ),
        ];
        let mut state = 0x9E37_79B9_7F4A_7C15u64 ^ u64::from(base);
        for i in 0..8 {
            state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = state;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^= z >> 31;
            let off = (u128::from(z) * (re - rs - width)) >> 64;
            out.push((
                FieldSize::new(rs + off, rs + off + width),
                format!("random {i}"),
            ));
        }
        let b128 = u128::from(base);
        for k in 1..=30u32 {
            let Some(bk) = b128.checked_pow(k) else { break };
            let g_ds = bk.saturating_sub(1).div_ceil(2);
            let g_ddc = bk.saturating_sub(6).div_ceil(6);
            // The smallest n with 3n² + 3n + 1 >= b^k.
            let (mut lo, mut hi) = (0u128, 1u128 << 64);
            while lo < hi {
                let mid = lo.midpoint(hi);
                let value = mid
                    .checked_mul(mid)
                    .and_then(|q| q.checked_mul(3))
                    .and_then(|q| q.checked_add(3 * mid + 1));
                if value.is_none_or(|v| v >= bk) {
                    hi = mid;
                } else {
                    lo = mid + 1;
                }
            }
            for (name, g) in [("dS", g_ds), ("ddC", g_ddc), ("dC", lo)] {
                if g > rs + width && g + width < re {
                    out.push((
                        FieldSize::new(g - width / 2, g + width / 2),
                        format!("{name} gains digit {k} at {g}"),
                    ));
                }
            }
        }
        out
    }

    /// The whole range of every base from 10 to `max_base` that has one,
    /// every run shape: the windows with near-misses. Returns how many.
    fn small_bases(ctx: &CubeclContext, max_base: u32) -> usize {
        let mut misses = 0;
        for base in 10..=max_base {
            let Ok(Some(range)) = crate::base_range::get_base_range_u128(base) else {
                continue;
            };
            let cpu = cpu_detailed_parallel(&range, base);
            assert_shapes_match(ctx, &range, base, &cpu, "full range");
            misses += cpu.nice_numbers.len();
            println!(
                "b{base} full range ({} numbers): exact, {} near-misses",
                range.size(),
                cpu.nice_numbers.len()
            );
        }
        misses
    }

    /// The finite-difference kernel against the CPU through the native CUDA
    /// runtime: at b40 and b50 every window of [`windows`] at 1e6 numbers
    /// under every run shape, the 2e8-number benchmark window whole against
    /// both the CPU and the per-number kernel, and the full range of every
    /// base 10-28 (76 million numbers).
    #[test]
    #[cfg(feature = "cubecl-cuda")]
    #[ignore = "requires an NVIDIA device"]
    fn cubecl_cuda_fd_matches_cpu_detailed() {
        let ctx = CubeclContext::new_cuda(0).expect("CubeCL CUDA init");
        for base in [40u32, 50] {
            for (range, what) in windows(base, 1_000_000) {
                let cpu = cpu_detailed_parallel(&range, base);
                assert_shapes_match(&ctx, &range, base, &cpu, &what);
                println!("b{base} {what}: [{}, {}) exact", range.start(), range.end());
            }
            let rs = crate::base_range::get_base_range_u128(base)
                .unwrap()
                .unwrap()
                .range_start;
            let range = FieldSize::new(rs, rs + 200_000_000);
            let cpu = cpu_detailed_parallel(&range, base);
            let fd = run(&ctx, &range, base, DetailedKernel::DEFAULT);
            let div = run(&ctx, &range, base, DetailedKernel::Division);
            assert_eq!(
                fd.distribution, cpu.distribution,
                "b{base} benchmark window"
            );
            assert_eq!(
                fd.nice_numbers, cpu.nice_numbers,
                "b{base} benchmark window"
            );
            assert_eq!(
                div.distribution, cpu.distribution,
                "b{base} benchmark window"
            );
            assert_eq!(
                div.nice_numbers, cpu.nice_numbers,
                "b{base} benchmark window"
            );
            println!("b{base} benchmark window [{rs}, +2e8): finite differences = division = CPU");
        }
        let misses = small_bases(&ctx, 28);
        assert!(misses > 0, "no near-misses compared");
        println!("bases 10-28: {misses} near-misses");
    }

    /// As [`cubecl_cuda_fd_matches_cpu_detailed`] on the default wgpu
    /// device, sized for a software rasterizer: windows of 1e5 numbers and
    /// bases 10-24 in full.
    #[test]
    #[ignore = "requires a wgpu device"]
    fn cubecl_fd_matches_cpu_detailed() {
        let ctx = CubeclContext::new_default().expect("CubeCL init");
        for base in [40u32, 50] {
            for (range, what) in windows(base, 100_000) {
                let cpu = cpu_detailed_parallel(&range, base);
                assert_shapes_match(&ctx, &range, base, &cpu, &what);
                println!("b{base} {what}: [{}, {}) exact", range.start(), range.end());
            }
        }
        let misses = small_bases(&ctx, 24);
        assert!(misses > 0, "no near-misses compared");
        println!("bases 10-24: {misses} near-misses");
    }

    /// Words of one [`fd_seed_probe_kernel`] record: S at 0, C at 7, dS at
    /// 17, dC at 21, ddC at 28 (the widest arrays, at b64, fill them).
    pub(crate) const SEED_STRIDE: u32 = 32;
    pub(crate) const SEED_FIELDS: [(&str, u32); 5] =
        [("S", 0), ("C", 7), ("dS", 17), ("dC", 21), ("ddC", 28)];

    /// [`fd_seed`] for `count` consecutive n from `start`, one n per thread,
    /// its five digit arrays written out as a [`SEED_STRIDE`]-word record.
    #[cube(launch_unchecked)]
    #[allow(clippy::too_many_arguments, clippy::similar_names)]
    fn fd_seed_probe_kernel(
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
        #[comptime] wide_chunk: bool,
        #[comptime] ls: u32,
        #[comptime] lc: u32,
        #[comptime] lds: u32,
        #[comptime] ldc: u32,
        #[comptime] lddc: u32,
    ) {
        let t = ABSOLUTE_POS_X;
        if t < count {
            let start_lo = (u64::cast_from(s1) << 32u64) | u64::cast_from(s0);
            let n_lo = start_lo + u64::cast_from(t);
            let mut n_hi = (u64::cast_from(s3) << 32u64) | u64::cast_from(s2);
            if n_lo < start_lo {
                n_hi += 1u64;
            }
            let mut sd = Array::<u32>::new(comptime!(ls.div_ceil(4)) as usize);
            let mut cd = Array::<u32>::new(comptime!(lc.div_ceil(4)) as usize);
            let mut ds = Array::<u32>::new(comptime!(lds.div_ceil(4)) as usize);
            let mut dc = Array::<u32>::new(comptime!(ldc.div_ceil(4)) as usize);
            let mut ddc = Array::<u32>::new(comptime!(lddc.div_ceil(4)) as usize);
            fd_seed(
                n_lo,
                n_hi,
                &mut sd,
                &mut cd,
                &mut ds,
                &mut dc,
                &mut ddc,
                base,
                limbs,
                chunk_digits,
                chunk_div,
                wide_chunk,
                ls,
                lc,
                lds,
                ldc,
                lddc,
            );
            let o = t * SEED_STRIDE;
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

    /// `ndig` base-`base` digits of `x`, packed four to a u32 from `at`.
    fn pack_digits(rec: &mut [u32], at: u32, x: &Natural, base: u32, ndig: u32) {
        let mut y = x.clone();
        let b = Natural::from(base);
        for j in 0..ndig {
            let d = u32::try_from(&y.div_assign_rem(&b)).unwrap();
            rec[(at + j / 4) as usize] |= d << (8 * (j % 4));
        }
    }

    /// The record [`fd_seed_probe_kernel`] should write for `n`.
    pub(crate) fn seed_expected(n: u128, base: u32, p: &FdPlan) -> Vec<u32> {
        let mut rec = vec![0u32; SEED_STRIDE as usize];
        let nn = Natural::from(n);
        let sq = &nn * &nn;
        let one = Natural::from(1u32);
        pack_digits(&mut rec, 0, &sq, base, p.ls);
        pack_digits(&mut rec, 7, &(&sq * &nn), base, p.lc);
        pack_digits(
            &mut rec,
            17,
            &(&nn * Natural::from(2u32) + &one),
            base,
            p.lds,
        );
        let dc = &sq * Natural::from(3u32) + &nn * Natural::from(3u32) + &one;
        pack_digits(&mut rec, 21, &dc, base, p.ldc);
        pack_digits(
            &mut rec,
            28,
            &(&nn * Natural::from(6u32) + Natural::from(6u32)),
            base,
            p.lddc,
        );
        rec
    }

    /// Run [`fd_seed_probe_kernel`] on `count` n from `start` with the chunk
    /// flavor production picks for this client.
    fn run_seed_probe<R: Runtime>(
        client: &ComputeClient<R>,
        base: u32,
        start: u128,
        count: u32,
    ) -> Vec<u32> {
        let p = fd_plan(base).unwrap();
        let limbs = crate::gpu_config::n_limbs(base).unwrap();
        let wide_chunk = crate::cubecl_backend::wide_chunk_for(client);
        let (chunk_digits, chunk_div) = if wide_chunk {
            crate::gpu_config::chunk_constants(base)
        } else {
            crate::gpu_config::chunk_constants_u16(base)
        };
        let len = (count * SEED_STRIDE) as usize;
        let h = client.create(cubecl::bytes::Bytes::from_elems(vec![0xDEAD_BEEFu32; len]));
        unsafe {
            fd_seed_probe_kernel::launch_unchecked::<R>(
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
                wide_chunk,
                p.ls,
                p.lc,
                p.lds,
                p.ldc,
                p.lddc,
            );
        }
        let mut rec = u32::from_bytes(&client.read_one(h).expect("seed probe readback")).to_vec();
        // Words past each array's top are never written: blank them.
        for t in 0..count as usize {
            let r = &mut rec[t * SEED_STRIDE as usize..(t + 1) * SEED_STRIDE as usize];
            for (k, (_, at)) in SEED_FIELDS.iter().enumerate() {
                let nd = [p.ls, p.lc, p.lds, p.ldc, p.lddc][k];
                let top = SEED_FIELDS.get(k + 1).map_or(SEED_STRIDE, |f| f.1);
                for w in at + nd.div_ceil(4)..top {
                    r[w as usize] = 0;
                }
            }
        }
        rec
    }

    /// The seed, one n at a time: at every base the kernel serves, the five
    /// digit arrays of each of 1024 n at the range start, the middle and the
    /// end, word for word against the CPU. Run length 1 and histograms hide
    /// most seed errors (a wrong digit string has nearly the random-digit
    /// distribution); this sees every digit. Before the split16 seed took
    /// its top two limbs together, it caught the Apple M4 Metal miscompile
    /// of [`metal_half_shift_div_repro`] at b23, b25, b27-b30, b49, b50 and
    /// b52-b55 there.
    #[test]
    #[ignore = "requires a wgpu device"]
    fn cubecl_fd_seed_matches_cpu() {
        const COUNT: u32 = 1024;
        let ctx = CubeclContext::new_default().expect("CubeCL init");
        #[allow(irrefutable_let_patterns)] // refutable only with cubecl-cuda on
        let CubeclContext::Wgpu { client, .. } = &ctx else {
            unreachable!("new_default is wgpu");
        };
        let mut bad = Vec::new();
        for base in 10..=FD_MAX_BASE {
            let Some(p) = fd_plan(base) else { continue };
            let mid = p.reg_lo + (p.reg_hi - p.reg_lo) / 2;
            let span = u128::from(COUNT).min(p.reg_hi - p.reg_lo);
            for start in [p.reg_lo, mid.min(p.reg_hi - span), p.reg_hi - span] {
                let count = span as u32;
                let rec = run_seed_probe(client, base, start, count);
                let mut wrong = [0u32; 5];
                let mut first = None;
                for t in 0..count {
                    let n = start + u128::from(t);
                    let want = seed_expected(n, base, &p);
                    let got = &rec[(t * SEED_STRIDE) as usize..((t + 1) * SEED_STRIDE) as usize];
                    for (k, (name, at)) in SEED_FIELDS.iter().enumerate() {
                        let top = SEED_FIELDS.get(k + 1).map_or(SEED_STRIDE, |f| f.1) as usize;
                        let (g, w) = (&got[*at as usize..top], &want[*at as usize..top]);
                        if g != w {
                            wrong[k] += 1;
                            if first.is_none() {
                                let word = g.iter().zip(w).position(|(a, b)| a != b).unwrap();
                                let lane = (0..4usize)
                                    .find(|l| (g[word] ^ w[word]) >> (8 * l) & 0xFF != 0);
                                first = Some(format!(
                                    "n={n} {name}: first wrong digit {} (got {:08x?}, want {:08x?})",
                                    4 * word + lane.unwrap_or(0),
                                    g,
                                    w
                                ));
                            }
                        }
                    }
                }
                if let Some(f) = first {
                    println!(
                        "b{base} [{start}, +{count}): wrong n per array (S C dS dC ddC) {wrong:?}; {f}"
                    );
                    bad.push(base);
                } else {
                    println!("b{base} [{start}, +{count}): exact");
                }
            }
        }
        bad.dedup();
        assert!(bad.is_empty(), "seed digit arrays wrong at bases {bad:?}");
    }

    /// Minimal reproducer of the Apple Metal miscompile behind the seed
    /// failure, kept for an upstream report: an unsigned division (or
    /// remainder) by a non-power-of-two constant whose dividend is a logical
    /// right shift by half its width. One thread per input pair (a, b);
    /// variant 0 stores `(a >> 16) / k`, 1 `(a >> 16) % k`, 2
    /// `u32((u64(a) * u64(b)) >> 32) / k`; the controls are 3 `(a >> 15) / k`,
    /// 4 `(a & 0xFFFF) / k` and 5 the top-pair form `fd_to_digits` uses,
    /// `((a >> 16) << 16 | (b & 0xFFFF)) / k`.
    ///
    /// Seen on an Apple M4 (macOS 26.3 25D125, wgpu 29.0.4 / naga 29.0.4,
    /// adapter "Apple M4 (Metal, wgpu<wgsl>)"), 2026-10-31: variants 0-2 are
    /// wrong for every k tried but 32768 (3 to 21952), the controls exact.
    /// naga's MSL for variant 0 is `naga_div(l_2_ >> 16u, 3u)` with
    /// `naga_div(lhs, rhs) = lhs / metal::select(rhs, 1u, rhs == 0u)`, a
    /// correct translation. Run as hand-written MSL through Metal directly
    /// (no wgpu, no naga), `naga_div(x >> 16u, 3u)`, `(x >> 16u) % 3u` and
    /// `uint(ulong(a) * ulong(b) >> 32) / 3u` are wrong in the same way
    /// (`naga_div(3u >> 16u, 3u)` comes out 1, as if the low half were
    /// divided), while a plain `(x >> 16u) / 3u`, shifts by 8, 15, 17 or 24,
    /// a mask, an OR and a power-of-two divisor are exact: Apple's GPU
    /// compiler. The kernel this broke was exact on lavapipe, CUDA and the
    /// macOS CI runner's paravirtual device (macOS 26.6).
    ///
    /// Gated on `NICE_METAL_DIV_REPRO=1` because CI runs every ignored test
    /// and this one fails on real Apple silicon by design.
    #[test]
    #[ignore = "Apple Metal compiler reproducer: set NICE_METAL_DIV_REPRO=1"]
    fn metal_half_shift_div_repro() {
        if std::env::var_os("NICE_METAL_DIV_REPRO").is_none() {
            println!("skipped: set NICE_METAL_DIV_REPRO=1 to run");
            return;
        }
        let ctx = CubeclContext::new_default().expect("CubeCL init");
        #[allow(irrefutable_let_patterns)] // refutable only with cubecl-cuda on
        let CubeclContext::Wgpu { client, .. } = &ctx else {
            unreachable!("new_default is wgpu");
        };
        // Small words first (their high half is 0), then xorshift words.
        let mut z = 0x9E37_79B9u32;
        let inp: Vec<u32> = (0..512)
            .chain((0..1536).map(|_| {
                z ^= z << 13;
                z ^= z >> 17;
                z ^= z << 5;
                z
            }))
            .collect();
        let pairs = (inp.len() / 2) as u32;
        let mut failures = Vec::new();
        for variant in 0..6u32 {
            for k in [3u32, 1000, 2500, 12167, 21952, 32768] {
                let hin = client.create(cubecl::bytes::Bytes::from_elems(inp.clone()));
                let hout =
                    client.create(cubecl::bytes::Bytes::from_elems(vec![0u32; pairs as usize]));
                unsafe {
                    half_shift_div_repro_kernel::launch_unchecked::<cubecl::wgpu::WgpuRuntime>(
                        client,
                        CubeCount::Static(pairs.div_ceil(64), 1, 1),
                        CubeDim::new_1d(64),
                        ArrayArg::from_raw_parts(hin, inp.len()),
                        ArrayArg::from_raw_parts(hout.clone(), pairs as usize),
                        pairs,
                        k,
                        variant,
                    );
                }
                let got = u32::from_bytes(&client.read_one(hout).unwrap()).to_vec();
                let mut wrong = Vec::new();
                for (t, &g) in got.iter().enumerate() {
                    let (a, b) = (inp[2 * t], inp[2 * t + 1]);
                    let want = match variant {
                        0 => (a >> 16) / k,
                        1 => (a >> 16) % k,
                        2 => ((u64::from(a) * u64::from(b)) >> 32) as u32 / k,
                        3 => (a >> 15) / k,
                        4 => (a & 0xFFFF) / k,
                        _ => (((a >> 16) << 16) | (b & 0xFFFF)) / k,
                    };
                    if g != want {
                        wrong.push((a, b, g, want));
                    }
                }
                println!(
                    "variant {variant} k={k}: {} of {pairs} wrong{}",
                    wrong.len(),
                    wrong.first().map_or(String::new(), |(a, b, g, w)| format!(
                        "; e.g. a={a} b={b}: got {g}, want {w}"
                    ))
                );
                if !wrong.is_empty() {
                    failures.push((variant, k));
                }
            }
        }
        assert!(
            failures.is_empty(),
            "wrong results for (variant, k) {failures:?}"
        );
    }

    /// See [`metal_half_shift_div_repro`].
    #[cube(launch_unchecked)]
    fn half_shift_div_repro_kernel(
        inp: &Array<u32>,
        out: &mut Array<u32>,
        pairs: u32,
        #[comptime] k: u32,
        #[comptime] variant: u32,
    ) {
        let t = ABSOLUTE_POS_X;
        if t < pairs {
            let a = inp[(2 * t) as usize];
            let b = inp[(2 * t + 1) as usize];
            if comptime!(variant == 0) {
                out[t as usize] = (a >> 16u32) / k;
            } else if comptime!(variant == 1) {
                out[t as usize] = (a >> 16u32) % k;
            } else if comptime!(variant == 2) {
                let hi = u32::cast_from((u64::cast_from(a) * u64::cast_from(b)) >> 32u64);
                out[t as usize] = hi / k;
            } else if comptime!(variant == 3) {
                out[t as usize] = (a >> 15u32) / k;
            } else if comptime!(variant == 4) {
                out[t as usize] = (a & 0xFFFFu32) / k;
            } else {
                out[t as usize] = (((a >> 16u32) << 16u32) | (b & 0xFFFFu32)) / k;
            }
        }
    }
}
