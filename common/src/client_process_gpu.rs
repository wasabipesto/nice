//! GPU-accelerated implementation of nice number checking using CUDA.
//!
//! This module provides GPU implementations of the hot loop operations for finding
//! square-cube pandigitals. It uses CUDA through the `cudarc` crate and requires
//! an NVIDIA GPU with CUDA support.
//!
//! The GPU kernels are compiled at runtime using NVRTC (NVIDIA Runtime Compiler),
//! which means the CUDA toolkit must be installed on the system.

#![cfg(feature = "gpu")]

use super::*;
use anyhow::{Context as _, Result};
use cudarc::driver::{CudaContext, CudaFunction, CudaStream, LaunchConfig, PushKernelArg};
use cudarc::nvrtc::compile_ptx;
use std::sync::Arc;

/// GPU context and compiled kernels.
/// This struct manages the CUDA device and compiled kernel functions.
#[allow(dead_code)]
pub struct GpuContext {
    _device: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    count_kernel: CudaFunction,
    nice_kernel: CudaFunction,
    filter_kernel: CudaFunction,
}

impl GpuContext {
    /// Initialize GPU context and compile kernels.
    ///
    /// # Arguments
    /// * `device_ordinal` - Which GPU to use (0 for first GPU, 1 for second, etc.)
    ///
    /// # Returns
    /// A GpuContext ready for processing, or an error if initialization fails.
    ///
    /// # Example
    /// ```no_run
    /// # #[cfg(feature = "gpu")]
    /// # {
    /// use nice_common::client_process_gpu::GpuContext;
    /// let ctx = GpuContext::new(0).expect("Failed to initialize GPU");
    /// # }
    /// ```
    pub fn new(device_ordinal: usize) -> Result<Self> {
        // Initialize CUDA context
        let device = CudaContext::new(device_ordinal)?;

        // Get default stream
        let stream = device.default_stream();

        // Load CUDA kernel source
        let kernel_src = include_str!("cuda/nice_kernels.cu");

        // Compile kernels using NVRTC
        let ptx = compile_ptx(kernel_src).context("Failed to compile CUDA kernels")?;

        // Load compiled module
        let module = device.load_module(ptx)?;

        // Get function handles
        let count_kernel = module.load_function("count_unique_digits_kernel")?;
        let nice_kernel = module.load_function("check_is_nice_kernel")?;
        let filter_kernel = module.load_function("filter_by_residue_kernel")?;

        Ok(GpuContext {
            _device: device,
            stream,
            count_kernel,
            nice_kernel,
            filter_kernel,
        })
    }
}

/// Convert u128 numbers to separate lo/hi u64 arrays for GPU transfer.
fn split_u128_vec(numbers: &[u128]) -> (Vec<u64>, Vec<u64>) {
    let mut lo = Vec::with_capacity(numbers.len());
    let mut hi = Vec::with_capacity(numbers.len());

    for &num in numbers {
        lo.push(num as u64);
        hi.push((num >> 64) as u64);
    }

    (lo, hi)
}

/// GPU implementation of process_range_detailed.
///
/// Processes a range of numbers on the GPU, calculating statistics on the niceness
/// of each number. This is the GPU equivalent of `client_process::process_range_detailed`.
///
/// # Arguments
/// * `ctx` - GPU context with compiled kernels
/// * `range_start` - First number to check (inclusive)
/// * `range_end` - Last number to check (exclusive)
/// * `base` - Number base to use for pandigital checking
///
/// # Returns
/// FieldResults containing distribution statistics and nice numbers found
pub fn process_range_detailed_gpu(
    ctx: &GpuContext,
    range_start: u128,
    range_end: u128,
    base: u32,
) -> Result<FieldResults> {
    let nice_list_cutoff = number_stats::get_near_miss_cutoff(base);
    let range_size = (range_end - range_start) as usize;

    // Generate the range of numbers to check
    let numbers: Vec<u128> = (range_start..range_end).collect();

    // Split u128 into lo/hi components
    let (numbers_lo, numbers_hi) = split_u128_vec(&numbers);

    // Transfer to GPU
    let d_numbers_lo = ctx.stream.clone_htod(&numbers_lo)?;
    let d_numbers_hi = ctx.stream.clone_htod(&numbers_hi)?;
    let mut d_unique_counts = ctx.stream.alloc_zeros::<u32>(range_size)?;

    // Launch kernel
    let cfg = LaunchConfig {
        grid_dim: (range_size.div_ceil(256) as u32, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };

    // Launch kernel using builder pattern
    let mut launch_args = ctx.stream.launch_builder(&ctx.count_kernel);
    launch_args.arg(&d_numbers_lo);
    launch_args.arg(&d_numbers_hi);
    launch_args.arg(&mut d_unique_counts);
    launch_args.arg(&base);
    launch_args.arg(&range_size);
    unsafe {
        launch_args.launch(cfg)?;
    }

    // Copy results back
    let unique_counts = ctx.stream.clone_dtoh(&d_unique_counts)?;

    // Aggregate results (same as CPU version)
    let mut unique_distribution_map: HashMap<u32, u128> = (1..=base).map(|i| (i, 0u128)).collect();
    let mut nice_numbers: Vec<NiceNumberSimple> = Vec::new();

    for (i, &num_uniques) in unique_counts.iter().enumerate() {
        *unique_distribution_map.entry(num_uniques).or_insert(0) += 1;

        if num_uniques > nice_list_cutoff {
            nice_numbers.push(NiceNumberSimple {
                number: range_start + i as u128,
                num_uniques,
            });
        }
    }

    // Convert distribution map to sorted Vec
    let mut distribution: Vec<UniquesDistributionSimple> = unique_distribution_map
        .into_iter()
        .map(|(num_uniques, count)| UniquesDistributionSimple { num_uniques, count })
        .collect();
    distribution.sort_by_key(|d| d.num_uniques);

    Ok(FieldResults {
        distribution,
        nice_numbers,
    })
}

/// GPU implementation of process_range_niceonly.
///
/// Processes a range looking only for 100% nice numbers. This is much faster than
/// the detailed version because it uses early-exit optimizations. This is the GPU
/// equivalent of `client_process::process_range_niceonly`.
///
/// # Arguments
/// * `ctx` - GPU context with compiled kernels
/// * `range_start` - First number to check (inclusive)
/// * `range_end` - Last number to check (exclusive)
/// * `base` - Number base to use for pandigital checking
///
/// # Returns
/// FieldResults containing only the nice numbers found (distribution is empty)
pub fn process_range_niceonly_gpu(
    ctx: &GpuContext,
    range_start: u128,
    range_end: u128,
    base: u32,
) -> Result<FieldResults> {
    let base_u128_minusone = base as u128 - 1;
    let residue_filter = residue_filter::get_residue_filter_u128(&base);

    // First, apply residue filter on CPU to reduce GPU workload
    // (The filter typically eliminates 70-90% of candidates)
    let candidates: Vec<u128> = (range_start..range_end)
        .filter(|num| residue_filter.contains(&(num % base_u128_minusone)))
        .collect();

    let candidate_count = candidates.len();
    if candidate_count == 0 {
        return Ok(FieldResults {
            distribution: Vec::new(),
            nice_numbers: Vec::new(),
        });
    }

    // Split u128 into lo/hi components
    let (numbers_lo, numbers_hi) = split_u128_vec(&candidates);

    // Transfer to GPU
    let d_numbers_lo = ctx.stream.clone_htod(&numbers_lo)?;
    let d_numbers_hi = ctx.stream.clone_htod(&numbers_hi)?;
    let mut d_is_nice = ctx.stream.alloc_zeros::<u8>(candidate_count)?;

    // Launch kernel
    let cfg = LaunchConfig {
        grid_dim: (candidate_count.div_ceil(256) as u32, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };

    // Launch kernel using builder pattern
    let mut launch_args = ctx.stream.launch_builder(&ctx.nice_kernel);
    launch_args.arg(&d_numbers_lo);
    launch_args.arg(&d_numbers_hi);
    launch_args.arg(&mut d_is_nice);
    launch_args.arg(&base);
    launch_args.arg(&candidate_count);
    unsafe {
        launch_args.launch(cfg)?;
    }

    // Copy results back
    let is_nice = ctx.stream.clone_dtoh(&d_is_nice)?;

    // Collect nice numbers
    let nice_numbers: Vec<NiceNumberSimple> = candidates
        .iter()
        .zip(is_nice.iter())
        .filter(|(_, nice)| **nice == 1)
        .map(|(number, _)| NiceNumberSimple {
            number: *number,
            num_uniques: base,
        })
        .collect();

    Ok(FieldResults {
        distribution: Vec::new(),
        nice_numbers,
    })
}

/// Process a field using GPU acceleration (detailed mode).
///
/// This is a convenience wrapper that matches the signature of
/// `client_process::process_detailed`.
pub fn process_detailed_gpu(
    ctx: &GpuContext,
    claim_data: &DataToClient,
    username: &String,
) -> Result<DataToServer> {
    let results = process_range_detailed_gpu(
        ctx,
        claim_data.range_start,
        claim_data.range_end,
        claim_data.base,
    )?;

    Ok(DataToServer {
        claim_id: claim_data.claim_id,
        username: username.to_owned(),
        client_version: CLIENT_VERSION.to_string(),
        unique_distribution: Some(results.distribution),
        nice_numbers: results.nice_numbers,
    })
}

/// Process a field using GPU acceleration (niceonly mode).
///
/// This is a convenience wrapper that matches the signature of
/// `client_process::process_niceonly`.
pub fn process_niceonly_gpu(
    ctx: &GpuContext,
    claim_data: &DataToClient,
    username: &String,
) -> Result<DataToServer> {
    let results = process_range_niceonly_gpu(
        ctx,
        claim_data.range_start,
        claim_data.range_end,
        claim_data.base,
    )?;

    Ok(DataToServer {
        claim_id: claim_data.claim_id,
        username: username.to_owned(),
        client_version: CLIENT_VERSION.to_string(),
        unique_distribution: None,
        nice_numbers: results.nice_numbers,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::client_process::*;

    /// Helper to initialize GPU context for tests.
    /// Returns None if GPU is not available (tests will be skipped).
    fn try_init_gpu() -> Option<GpuContext> {
        GpuContext::new(0).ok()
    }

    /// Recombine lo/hi u64 arrays back to u128.
    fn combine_u64_to_u128(lo: &[u64], hi: &[u64]) -> Vec<u128> {
        lo.iter()
            .zip(hi.iter())
            .map(|(&l, &h)| ((h as u128) << 64) | (l as u128))
            .collect()
    }

    #[test]
    fn test_gpu_matches_cpu_detailed_small() {
        let ctx = match try_init_gpu() {
            Some(c) => c,
            None => {
                println!("GPU not available, skipping test");
                return;
            }
        };

        let range_start = 1_000_000u128;
        let range_end = 1_001_000u128;
        let base = 10u32;

        let cpu_result = process_range_detailed(range_start, range_end, base);
        let gpu_result = process_range_detailed_gpu(&ctx, range_start, range_end, base)
            .expect("GPU processing failed");

        // Check that distributions match
        assert_eq!(
            cpu_result.distribution, gpu_result.distribution,
            "Distribution mismatch between CPU and GPU"
        );

        // Check that nice numbers match
        assert_eq!(
            cpu_result.nice_numbers.len(),
            gpu_result.nice_numbers.len(),
            "Different number of nice numbers found"
        );

        for (cpu_nice, gpu_nice) in cpu_result
            .nice_numbers
            .iter()
            .zip(gpu_result.nice_numbers.iter())
        {
            assert_eq!(cpu_nice, gpu_nice, "Nice number mismatch");
        }
    }

    #[test]
    fn test_gpu_matches_cpu_niceonly_small() {
        let ctx = match try_init_gpu() {
            Some(c) => c,
            None => {
                println!("GPU not available, skipping test");
                return;
            }
        };

        let range_start = 1_000_000u128;
        let range_end = 1_010_000u128;
        let base = 10u32;

        let cpu_result = process_range_niceonly(range_start, range_end, base);
        let gpu_result = process_range_niceonly_gpu(&ctx, range_start, range_end, base)
            .expect("GPU processing failed");

        // Sort both results for comparison (order might differ)
        let mut cpu_nice = cpu_result.nice_numbers;
        let mut gpu_nice = gpu_result.nice_numbers;
        cpu_nice.sort_by_key(|n| n.number);
        gpu_nice.sort_by_key(|n| n.number);

        assert_eq!(
            cpu_nice.len(),
            gpu_nice.len(),
            "Different number of nice numbers found"
        );

        for (cpu, gpu) in cpu_nice.iter().zip(gpu_nice.iter()) {
            assert_eq!(cpu, gpu, "Nice number mismatch");
        }
    }

    #[test]
    fn test_gpu_base_40_range() {
        let ctx = match try_init_gpu() {
            Some(c) => c,
            None => {
                println!("GPU not available, skipping test");
                return;
            }
        };

        // Test with a base 40 range (more realistic for the actual problem)
        let range_start = 2_000_000_000_000u128;
        let range_end = 2_000_100_000u128;
        let base = 40u32;

        let cpu_result = process_range_niceonly(range_start, range_end, base);
        let gpu_result = process_range_niceonly_gpu(&ctx, range_start, range_end, base)
            .expect("GPU processing failed");

        // Sort for comparison
        let mut cpu_nice = cpu_result.nice_numbers;
        let mut gpu_nice = gpu_result.nice_numbers;
        cpu_nice.sort_by_key(|n| n.number);
        gpu_nice.sort_by_key(|n| n.number);

        assert_eq!(cpu_nice, gpu_nice, "Results differ for base 40");
    }

    #[test]
    fn test_split_combine_u128() {
        let numbers = vec![0u128, 1u128, 12345u128, u64::MAX as u128, u128::MAX];
        let (lo, hi) = split_u128_vec(&numbers);
        let recombined = combine_u64_to_u128(&lo, &hi);
        assert_eq!(numbers, recombined);
    }

    #[test]
    fn test_gpu_context_creation() {
        // Just test that we can create a context
        // If no GPU is available, this should return an error
        match GpuContext::new(0) {
            Ok(_ctx) => {
                println!("GPU context created successfully");
            }
            Err(e) => {
                println!("Expected failure without GPU: {:?}", e);
            }
        }
    }
}
