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
//! Niceonly is deliberately not ported yet — this file exists to measure
//! detailed-mode performance and auditability against the hand-written
//! backends before committing to the full surface.

#![cfg(feature = "cubecl")]
// The cube macro evaluates comptime! expressions host-side, where fn-level
// allows do not reach; these lints are all macro-expansion artifacts.
#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_lossless,
    clippy::used_underscore_binding
)]

use crate::client_process::process_range_detailed;
use crate::gpu_config::{chunk_constants_u16, gpu_supports_base, n_limbs};
use crate::number_stats::get_near_miss_cutoff;
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

// ============================================================================
// Host side
// ============================================================================

/// One initialized `CubeCL` device: wgpu everywhere, or the native CUDA
/// runtime when built with `cubecl-cuda` (the meaningful NVIDIA comparison,
/// since it exercises CubeCL's CUDA codegen against the hand kernels).
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
}
