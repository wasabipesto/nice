//! Browser-portable `CubeCL` detailed kernel: the same algorithm as
//! [`crate::cubecl_backend`]'s detailed kernel with **no 64-bit integers
//! anywhere** — WebGPU has no `shader-int64`, so the wgpu backend of a
//! browser rejects any kernel naming a u64.
//!
//! The arithmetic differences from the native kernel, and nothing else:
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
//! Correctness is anchored the same way as the native kernels: exact CPU
//! parity on lavapipe (which drives this kernel through the identical
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

// ============================================================================
// Host side
// ============================================================================

/// Browser-portable implementation of `process_range_detailed`: the u32-only
/// kernel over the context's wgpu client. The native kernels are faster where
/// they run; this exists for devices whose wgpu backend has no
/// `shader-int64` — browsers foremost, but it runs identically on native
/// wgpu, which is how it is parity-tested without one.
///
/// **Range semantics**: half-open [`range_start`, `range_end`).
///
/// # Errors
/// Returns an error on any device failure, if the near-miss buffer
/// overflows, or on a non-wgpu context (the u32 kernel targets WebGPU; the
/// CUDA runtime always has u64 and the native kernel).
///
/// # Panics
/// Panics if a web batch exceeds u32 candidates, which
/// `CUBECL_WEB_BATCH_SIZE` makes impossible.
#[allow(clippy::too_many_lines)]
pub async fn process_range_detailed_web_async(
    ctx: &CubeclContext,
    range: &FieldSize,
    base: u32,
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

    for batch in range.chunks(CUBECL_WEB_BATCH_SIZE) {
        let start = batch.start();
        let count = u32::try_from(batch.size()).expect("web batch fits u32");
        let cubes = count.div_ceil(WORKGROUP_SIZE).clamp(1, MAX_CUBES_WEB);

        unsafe {
            detailed_kernel_u32::launch_unchecked::<cubecl::wgpu::WgpuRuntime>(
                client,
                CubeCount::Static(cubes, 1, 1),
                CubeDim::new_1d(WORKGROUP_SIZE),
                ArrayArg::from_raw_parts(hist_handle.clone(), hist_bins),
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
            "CubeCL web detailed b{base}: {:.2e} numbers in {secs:.2}s ({:.2e} n/s), {} near-misses",
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

#[cfg(test)]
mod tests {
    use super::*;

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

    /// CPU/web-kernel parity across limb widths and both mask layouts —
    /// the same bases and ranges as the native suites, plus base 97 (the
    /// widest supported base: 4 n-limbs, four-mask digit set).
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
            let gpu = process_range_detailed_web(&ctx, &range, base).expect("web kernel run");
            let cpu = process_range_detailed(&range, base);
            assert_eq!(
                gpu.distribution, cpu.distribution,
                "base {base}: distribution mismatch"
            );
            assert_eq!(
                gpu.nice_numbers, cpu.nice_numbers,
                "base {base}: near-miss mismatch"
            );
            println!("base {base}: {count} candidates match the CPU exactly (u32 kernel)");
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
