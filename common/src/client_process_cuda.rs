//! GPU-accelerated implementation of nice number checking using CUDA.
//!
//! This module offloads the hot loops to the GPU while keeping the same
//! filter cascade and result semantics as the CPU path:
//!
//! - **Niceonly**: the CPU runs the real MSD prefix filter (parallelized
//!   across cores) with a coarser recursion floor than the CPU client (see
//!   [`AdaptiveFloor`]), then ships only compact *range descriptors*
//!   to the GPU (~12 bytes per surviving range). The GPU reconstructs the
//!   stride filter's candidates on-device from the residue table — the g-th
//!   valid candidate at or after a range start is
//!   `B0 + (g/R)*M + residues[g%R]` — and runs the early-exit nice check.
//!   No per-candidate data ever crosses the bus. The GPU checks a superset
//!   of the CPU path's candidates (coarser pruning is still sound), so the
//!   nice numbers found are identical.
//! - **Detailed**: each GPU thread derives its own `n = start + idx`, so
//!   there is no input transfer at all. Unique-digit counts accumulate in an
//!   on-device histogram; only the histogram and the (rare) near-miss list
//!   come back.
//!
//! Kernels are compiled at runtime with NVRTC, **once per (base, mode)**,
//! with all base-dependent values injected as preprocessor defines. This is
//! the GPU analog of the CPU's const-generic dispatch: the compiler
//! strength-reduces every division by the base (or the stride modulus) into
//! multiply-high sequences, for *every* base — not just a hardcoded list.
//! Compiled modules are cached in the context.
//!
//! The GPU path supports every base with a valid u128 search range (through
//! ~b97): kernel buffers are sized per base at JIT time from `N_LIMBS`, so
//! there is no 256-bit ceiling like the CPU's `U256` fast path, and bases
//! above 64 use a two-word digit mask. Bases with no u128 range fall back to
//! the CPU implementation with a logged warning.

#![cfg(feature = "cuda")]
#![allow(clippy::cast_possible_truncation)]

use crate::client_process::{process_range_detailed, process_range_niceonly};
use crate::gpu_config::{
    MAX_GPU_DIGIT_MASK_BASE, chunk_constants, gpu_supports_base, prefilter_params,
};
use crate::gpu_niceonly::{RangeSink, report_field, run_range_pipeline};
use crate::{
    CLIENT_VERSION, DataToClient, DataToServer, FieldResults, FieldSize, NiceNumberSimple,
    UniquesDistributionSimple,
};
use crate::{base_range, number_stats, residue_filter, stride_filter};
use anyhow::{Context as _, Result, bail, ensure};
use cudarc::driver::{
    CudaContext as DriverContext, CudaFunction, CudaSlice, CudaStream, LaunchConfig, PushKernelArg,
};
use cudarc::nvrtc::{CompileOptions, Ptx, compile_ptx_with_opts};
use log::{debug, warn};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::Instant;

/// The MSD/descriptor pipeline these two constants belong to now lives in
/// [`crate::gpu_niceonly`], which the Vulkan backend shares. Re-exported so the
/// CUDA path's public surface is unchanged.
pub use crate::gpu_niceonly::{GPU_LSD_K, PROCESSING_CHUNK_SIZE};

/// Numbers processed per detailed-mode kernel launch. Larger batches amortize
/// launch overhead; since the detailed kernel takes no input arrays, batch
/// size costs no memory or transfer bandwidth.
pub const CUDA_BATCH_SIZE: usize = 50_000_000;

/// Threads per block. Must match `BLOCK_THREADS` in `nice_kernels.cu` (the
/// detailed kernel's shared-memory histogram is sized from it).
const BLOCK_THREADS: u32 = 256;

/// Capacity of the niceonly output buffer (in nice numbers) per field.
/// Genuinely nice numbers are astronomically rare; this is pure headroom.
const NICE_OUT_CAPACITY: usize = 1 << 16;

/// Capacity of the detailed-mode near-miss buffer per field.
const NEAR_MISS_CAPACITY: usize = 1 << 20;

/// Maximum MSD-valid ranges per niceonly kernel launch.
const RANGES_PER_LAUNCH: usize = 1 << 22;

/// Compiled niceonly kernel plus the per-base stride data living on-device.
struct NiceonlyPlan {
    base: u32,
    func: CudaFunction,
    residues: CudaSlice<u32>,
    modulus: u32,
    num_residues: u32,
}

/// GPU context: CUDA device handle plus caches of per-base compiled kernels.
pub struct CudaContext {
    device: Arc<DriverContext>,
    stream: Arc<CudaStream>,
    niceonly_plans: Mutex<HashMap<u32, Arc<NiceonlyPlan>>>,
    detailed_kernels: Mutex<HashMap<u32, CudaFunction>>,
}

impl CudaContext {
    /// Initialize the GPU context and verify that NVRTC compilation works.
    ///
    /// Kernels themselves are compiled lazily, once per (base, mode), when
    /// the first field for that base arrives.
    ///
    /// # Arguments
    /// * `device_ordinal` - Which GPU to use (0 for first GPU, etc.)
    ///
    /// # Errors
    /// Returns an error if the CUDA context cannot be initialized or if a
    /// smoke-test NVRTC compilation fails (e.g. missing NVRTC library).
    pub fn new(device_ordinal: usize) -> Result<Self> {
        let device = DriverContext::new(device_ordinal)
            .with_context(|| format!("initializing CUDA device {device_ordinal}"))?;
        let stream = device.default_stream();

        // Smoke-test NVRTC + module loading now so a broken install fails at
        // startup with a clear error instead of mid-run on the first field.
        let smoke_start = Instant::now();
        let defines = detailed_defines(10).context("building smoke-test kernel config")?;
        let ptx = compile_kernel_ptx(&defines).context(
            "NVRTC smoke-test compilation failed (is the CUDA toolkit's NVRTC available?)",
        )?;
        let module = device
            .load_module(ptx)
            .context("loading smoke-test module")?;
        module
            .load_function("detailed_kernel")
            .context("resolving smoke-test kernel")?;
        debug!(
            "GPU init: NVRTC smoke test passed in {:.2}s",
            smoke_start.elapsed().as_secs_f64()
        );

        Ok(CudaContext {
            device,
            stream,
            niceonly_plans: Mutex::new(HashMap::new()),
            detailed_kernels: Mutex::new(HashMap::new()),
        })
    }

    /// Get or build the compiled niceonly kernel + device residue table for a base.
    fn niceonly_plan(&self, base: u32) -> Result<Arc<NiceonlyPlan>> {
        if let Some(plan) = self.niceonly_plans.lock().unwrap().get(&base) {
            return Ok(plan.clone());
        }

        let build_start = Instant::now();
        let (defines, table) = niceonly_defines(base)?;
        let modulus = table.modulus as u32;
        let residues_host: Vec<u32> = table.valid_residues.clone();
        let num_residues = residues_host.len() as u32;

        let ptx = compile_kernel_ptx(&defines)
            .with_context(|| format!("compiling niceonly kernel for base {base}"))?;
        let module = self.device.load_module(ptx)?;
        let func = module.load_function("niceonly_ranges_kernel")?;
        let residues = self.stream.clone_htod(&residues_host)?;

        debug!(
            "GPU niceonly plan for base {base}: M={modulus}, R={num_residues}, built in {:.2}s",
            build_start.elapsed().as_secs_f64()
        );

        let plan = Arc::new(NiceonlyPlan {
            base,
            func,
            residues,
            modulus,
            num_residues,
        });
        self.niceonly_plans
            .lock()
            .unwrap()
            .insert(base, plan.clone());
        Ok(plan)
    }

    /// Get or build the compiled detailed kernel for a base.
    fn detailed_kernel(&self, base: u32) -> Result<CudaFunction> {
        if let Some(func) = self.detailed_kernels.lock().unwrap().get(&base) {
            return Ok(func.clone());
        }

        let build_start = Instant::now();
        let defines = detailed_defines(base)?;
        let ptx = compile_kernel_ptx(&defines)
            .with_context(|| format!("compiling detailed kernel for base {base}"))?;
        let module = self.device.load_module(ptx)?;
        let func = module.load_function("detailed_kernel")?;
        debug!(
            "GPU detailed kernel for base {base} built in {:.2}s",
            build_start.elapsed().as_secs_f64()
        );

        self.detailed_kernels
            .lock()
            .unwrap()
            .insert(base, func.clone());
        Ok(func)
    }
}

/// Defines shared by both kernels for a base: `BASE`, `N_LIMBS`,
/// `CHUNK_DIGITS`, `CHUNK_DIV`. Fails for bases the GPU cannot handle
/// (see [`gpu_supports_base`]).
fn common_defines(base: u32) -> Result<Vec<String>> {
    ensure!(
        base <= MAX_GPU_DIGIT_MASK_BASE,
        "base {base} exceeds the GPU digit mask limit {MAX_GPU_DIGIT_MASK_BASE}"
    );
    let range = base_range::get_base_range_u128(base)
        .context("computing base range")?
        .with_context(|| format!("base {base} has no valid u128 search range"))?;
    let n_max = range.range_end - 1;
    let n_bits = 128 - n_max.leading_zeros();
    let n_limbs = n_bits.div_ceil(32).max(1);
    let (chunk_digits, chunk_div) = chunk_constants(base);
    Ok(vec![
        format!("BASE={base}"),
        format!("N_LIMBS={n_limbs}"),
        format!("CHUNK_DIGITS={chunk_digits}"),
        format!("CHUNK_DIV={chunk_div}u"),
    ])
}

fn detailed_defines(base: u32) -> Result<Vec<String>> {
    let mut defines = common_defines(base)?;
    defines.push("DETAILED".to_string());
    defines.push(format!(
        "NEAR_MISS_CUTOFF={}",
        number_stats::get_near_miss_cutoff(base)
    ));
    Ok(defines)
}

/// Full define set for the niceonly kernel (common + stride + prefilter),
/// along with the stride table whose residues the caller uploads. Requires
/// no GPU or NVRTC, so tests can exercise kernel configuration for every
/// base without hardware.
fn niceonly_defines(base: u32) -> Result<(Vec<String>, stride_filter::StrideTable)> {
    let table = stride_filter::StrideTable::new(base, GPU_LSD_K);
    ensure!(
        !table.valid_residues.is_empty(),
        "no valid stride residues for base {base} (residue-empty base?)"
    );
    ensure!(
        table.modulus <= u128::from(u32::MAX),
        "stride modulus {} exceeds u32 for base {base}",
        table.modulus
    );
    let modulus = table.modulus as u32;
    let num_residues = table.valid_residues.len() as u32;
    let pow64_mod_m = ((1u128 << 64) % table.modulus) as u32;

    let mut defines = common_defines(base)?;
    defines.push("NICEONLY".to_string());
    defines.push(format!("STRIDE_M={modulus}u"));
    defines.push(format!("STRIDE_R={num_residues}u"));
    defines.push(format!("POW64_MOD_M={pow64_mod_m}u"));
    if let Some(pre) = prefilter_params(base) {
        defines.push("PREFILTER".to_string());
        defines.push(format!("PRE_DIGITS={}", pre.digits));
        defines.push(format!("PRE_MOD={}ull", pre.modulus));
        defines.push(format!("POW64_MOD_PRE={}ull", pre.pow64_mod));
    } else {
        debug!("modular prefilter disabled for base {base}");
    }
    Ok((defines, table))
}

/// Compile the embedded CUDA source with the given `-D` defines via NVRTC.
fn compile_kernel_ptx(defines: &[String]) -> Result<Ptx> {
    let kernel_src = include_str!("cuda/nice_kernels.cu");
    let opts = CompileOptions {
        options: defines
            .iter()
            .map(|d| format!("--define-macro={d}"))
            .collect(),
        ..Default::default()
    };
    compile_ptx_with_opts(kernel_src, opts)
        .map_err(|e| anyhow::anyhow!("NVRTC compilation failed: {e:?}"))
}

/// Split a u128 into (lo, hi) u64 halves for kernel arguments.
fn split_u128(num: u128) -> (u64, u64) {
    (num as u64, (num >> 64) as u64)
}

fn combine_u64(lo: u64, hi: u64) -> u128 {
    (u128::from(hi) << 64) | u128::from(lo)
}

// ============================================================================
// Niceonly
// ============================================================================

/// GPU implementation of `process_range_niceonly`.
///
/// Runs the MSD prefix filter on the CPU (all cores), then checks the
/// surviving ranges' stride-valid candidates on the GPU. Produces the exact
/// same nice-number set as the CPU path.
///
/// **Range semantics**: half-open [`range_start`, `range_end`).
///
/// # Errors
/// Returns an error on any CUDA failure or if the output buffer overflows.
pub fn process_range_niceonly_cuda(
    ctx: &CudaContext,
    range: &FieldSize,
    base: u32,
) -> Result<FieldResults> {
    if !gpu_supports_base(base) {
        warn!("base {base} not supported on GPU, falling back to CPU for this field");
        let stride_table = stride_filter::StrideTable::new(base, GPU_LSD_K);
        return Ok(process_range_niceonly(range, base, &stride_table));
    }
    if residue_filter::get_residue_filter_u128(&base).is_empty() {
        debug!("base {base} is residue-empty; no candidates to check");
        return Ok(FieldResults {
            distribution: Vec::new(),
            nice_numbers: Vec::new(),
        });
    }

    // Build (or fetch cached) the compiled kernel and device residue table up
    // front, so the timings below reflect per-field work only.
    let plan = ctx.niceonly_plan(base)?;

    let mut launcher = NiceonlyLauncher::new(ctx, &plan, range)?;
    let stats = run_range_pipeline(&mut launcher, range, base)?;
    let nice_numbers = launcher.finish()?;
    debug!(
        "GPU niceonly pipeline: {} ranges in {} launches, M={}, R={}, found {}",
        stats.num_ranges,
        stats.launches,
        plan.modulus,
        plan.num_residues,
        nice_numbers.len()
    );
    report_field("GPU", base, range, &stats);

    Ok(FieldResults {
        distribution: Vec::new(),
        nice_numbers,
    })
}

/// Holds the per-field output buffers and issues asynchronous niceonly
/// kernel launches over batches of range descriptors.
struct NiceonlyLauncher<'a> {
    ctx: &'a CudaContext,
    plan: &'a NiceonlyPlan,
    field_start_lo: u64,
    field_start_hi: u64,
    d_nice_out: CudaSlice<u64>,
    d_nice_count: CudaSlice<u32>,
}

impl<'a> NiceonlyLauncher<'a> {
    fn new(ctx: &'a CudaContext, plan: &'a NiceonlyPlan, range: &FieldSize) -> Result<Self> {
        let (field_start_lo, field_start_hi) = split_u128(range.start());
        Ok(NiceonlyLauncher {
            ctx,
            plan,
            field_start_lo,
            field_start_hi,
            d_nice_out: ctx.stream.alloc_zeros::<u64>(2 * NICE_OUT_CAPACITY)?,
            d_nice_count: ctx.stream.alloc_zeros::<u32>(1)?,
        })
    }

    /// Synchronize and collect the found nice numbers.
    fn finish(self) -> Result<Vec<NiceNumberSimple>> {
        let nice_count = self.ctx.stream.clone_dtoh(&self.d_nice_count)?[0] as usize;
        if nice_count > NICE_OUT_CAPACITY {
            bail!(
                "niceonly output buffer overflow: {nice_count} > {NICE_OUT_CAPACITY} \
                 (this strongly suggests a kernel bug)"
            );
        }
        let mut nice_numbers = Vec::with_capacity(nice_count);
        if nice_count > 0 {
            let out = self.ctx.stream.clone_dtoh(&self.d_nice_out)?;
            for i in 0..nice_count {
                nice_numbers.push(NiceNumberSimple {
                    number: combine_u64(out[2 * i], out[2 * i + 1]),
                    num_uniques: self.plan.base,
                });
            }
            nice_numbers.sort_by_key(|n| n.number);
        }
        Ok(nice_numbers)
    }
}

impl RangeSink for NiceonlyLauncher<'_> {
    /// Upload a batch of range descriptors and launch the kernel on them.
    /// Launches are asynchronous on the stream; results accumulate in the
    /// shared output buffers until [`NiceonlyLauncher::finish`].
    fn launch(&mut self, offsets: &[u64], lens: &[u32]) -> Result<()> {
        let nice_capacity = NICE_OUT_CAPACITY as u32;
        for (batch_offsets, batch_lens) in offsets
            .chunks(RANGES_PER_LAUNCH)
            .zip(lens.chunks(RANGES_PER_LAUNCH))
        {
            let d_offsets = self.ctx.stream.clone_htod(batch_offsets)?;
            let d_lens = self.ctx.stream.clone_htod(batch_lens)?;
            let num_ranges = batch_offsets.len() as u32;

            // One warp per range.
            let total_threads = u64::from(num_ranges) * 32;
            let grid_blocks = total_threads.div_ceil(u64::from(BLOCK_THREADS)) as u32;
            let cfg = LaunchConfig {
                grid_dim: (grid_blocks, 1, 1),
                block_dim: (BLOCK_THREADS, 1, 1),
                shared_mem_bytes: 0,
            };

            let mut launch_args = self.ctx.stream.launch_builder(&self.plan.func);
            launch_args.arg(&self.field_start_lo);
            launch_args.arg(&self.field_start_hi);
            launch_args.arg(&d_offsets);
            launch_args.arg(&d_lens);
            launch_args.arg(&num_ranges);
            launch_args.arg(&self.plan.residues);
            launch_args.arg(&self.d_nice_out);
            launch_args.arg(&mut self.d_nice_count);
            launch_args.arg(&nice_capacity);
            unsafe {
                launch_args.launch(cfg)?;
            }
        }
        Ok(())
    }

    /// Launches are asynchronous, so the pipeline's `total_secs` would
    /// otherwise stop before the device had done the work.
    fn sync(&mut self) -> Result<()> {
        self.ctx.stream.synchronize()?;
        Ok(())
    }
}

// ============================================================================
// Detailed
// ============================================================================

/// GPU implementation of `process_range_detailed`.
///
/// Each GPU thread derives its own candidate (no input transfer); the
/// distribution accumulates in an on-device histogram and only near-miss
/// numbers come back individually.
///
/// **Range semantics**: half-open [`range_start`, `range_end`).
///
/// # Errors
/// Returns an error on any CUDA failure or if the near-miss buffer overflows.
pub fn process_range_detailed_cuda(
    ctx: &CudaContext,
    range: &FieldSize,
    base: u32,
) -> Result<FieldResults> {
    if !gpu_supports_base(base) {
        warn!("base {base} not supported on GPU, falling back to CPU for this field");
        return Ok(process_range_detailed(range, base));
    }

    let start_time = Instant::now();
    let func = ctx.detailed_kernel(base)?;

    let hist_bins = (base + 1) as usize;
    let d_hist = ctx.stream.alloc_zeros::<u64>(hist_bins)?;
    let d_miss_out = ctx.stream.alloc_zeros::<u64>(2 * NEAR_MISS_CAPACITY)?;
    let mut d_miss_uniques = ctx.stream.alloc_zeros::<u32>(NEAR_MISS_CAPACITY)?;
    let mut d_miss_count = ctx.stream.alloc_zeros::<u32>(1)?;
    let miss_capacity = NEAR_MISS_CAPACITY as u32;

    for batch in range.chunks(CUDA_BATCH_SIZE as u128) {
        let (start_lo, start_hi) = split_u128(batch.start());
        let count = batch.size() as u64;

        // Grid-stride: cap the grid and let threads loop.
        let grid_blocks = count.div_ceil(u64::from(BLOCK_THREADS)).min(65_536) as u32;
        let cfg = LaunchConfig {
            grid_dim: (grid_blocks, 1, 1),
            block_dim: (BLOCK_THREADS, 1, 1),
            shared_mem_bytes: 0,
        };

        let mut launch_args = ctx.stream.launch_builder(&func);
        launch_args.arg(&start_lo);
        launch_args.arg(&start_hi);
        launch_args.arg(&count);
        launch_args.arg(&d_hist);
        launch_args.arg(&d_miss_out);
        launch_args.arg(&mut d_miss_uniques);
        launch_args.arg(&mut d_miss_count);
        launch_args.arg(&miss_capacity);
        unsafe {
            launch_args.launch(cfg)?;
        }
    }

    let histogram = ctx.stream.clone_dtoh(&d_hist)?;
    let miss_count = ctx.stream.clone_dtoh(&d_miss_count)?[0] as usize;
    if miss_count > NEAR_MISS_CAPACITY {
        bail!("near-miss buffer overflow: {miss_count} > {NEAR_MISS_CAPACITY}");
    }
    let mut nice_numbers = Vec::with_capacity(miss_count);
    if miss_count > 0 {
        let out = ctx.stream.clone_dtoh(&d_miss_out)?;
        let uniques = ctx.stream.clone_dtoh(&d_miss_uniques)?;
        for i in 0..miss_count {
            nice_numbers.push(NiceNumberSimple {
                number: combine_u64(out[2 * i], out[2 * i + 1]),
                num_uniques: uniques[i],
            });
        }
        nice_numbers.sort_by_key(|n| n.number);
    }

    let distribution: Vec<UniquesDistributionSimple> = (1..=base)
        .map(|i| UniquesDistributionSimple {
            num_uniques: i,
            count: u128::from(histogram[i as usize]),
        })
        .collect();

    #[allow(clippy::cast_precision_loss)]
    {
        let secs = start_time.elapsed().as_secs_f64();
        debug!(
            "GPU detailed b{base}: {:.2e} numbers in {secs:.2}s ({:.2e} n/s), {miss_count} near-misses",
            range.size() as f64,
            range.size() as f64 / secs,
        );
    }

    Ok(FieldResults {
        distribution,
        nice_numbers,
    })
}

// ============================================================================
// Convenience wrappers (same signatures as the CPU process_* functions)
// ============================================================================

/// Process a field using GPU acceleration (detailed mode).
///
/// # Errors
/// Returns an error on any CUDA failure.
pub fn process_detailed_cuda(
    ctx: &CudaContext,
    claim_data: &DataToClient,
    username: &String,
) -> Result<DataToServer> {
    let results = process_range_detailed_cuda(ctx, &claim_data.into(), claim_data.base)?;

    Ok(DataToServer {
        claim_id: claim_data.claim_id,
        username: username.to_owned(),
        client_version: CLIENT_VERSION.to_string(),
        unique_distribution: Some(results.distribution),
        nice_numbers: results.nice_numbers,
        telemetry: None,
    })
}

/// Process a field using GPU acceleration (niceonly mode).
///
/// # Errors
/// Returns an error on any CUDA failure.
pub fn process_niceonly_cuda(
    ctx: &CudaContext,
    claim_data: &DataToClient,
    username: &String,
) -> Result<DataToServer> {
    let results = process_range_niceonly_cuda(ctx, &claim_data.into(), claim_data.base)?;

    Ok(DataToServer {
        claim_id: claim_data.claim_id,
        username: username.to_owned(),
        client_version: CLIENT_VERSION.to_string(),
        unique_distribution: None,
        nice_numbers: results.nice_numbers,
        telemetry: None,
    })
}

// ============================================================================
// Tests
// ============================================================================
//
// The GPU-requiring tests are #[ignore]d and meant for the A100. The rest are
// CPU-side mirrors of the kernel's algorithms — they exercise the *exact*
// index math and digit-extraction logic the kernel uses, against the trusted
// CPU implementations, so kernel logic bugs are caught without a GPU.

// The mirror functions intentionally transliterate the kernel's C, casts and
// all, so the cast lints are relaxed for this module.
#[cfg(test)]
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss
)]
mod tests {
    use super::*;
    use crate::client_process;
    use crate::gpu_config::PrefilterParams;
    use crate::stride_filter::StrideTable;

    /// Bases used for CPU-side mirror tests: a mix of small, u64-range,
    /// u128-range, two-mask (>64), and beyond-U256 (>68) regimes.
    const MIRROR_TEST_BASES: [u32; 8] = [10, 40, 45, 57, 62, 68, 70, 94];

    fn try_init_cuda() -> Option<CudaContext> {
        CudaContext::new(0).ok()
    }

    // ------------------------------------------------------------------
    // Mirror of the kernel's stride candidate enumeration (niceonly)
    // ------------------------------------------------------------------

    /// Rust mirror of `mod_m` in `nice_kernels.cu`.
    fn mirror_mod_m(n: u128, modulus: u32, pow64_mod_m: u32) -> u32 {
        let (n_lo, n_hi) = split_u128(n);
        let hi_mod = (n_hi % u64::from(modulus)) as u32;
        let lo_mod = (n_lo % u64::from(modulus)) as u32;
        let t = u64::from(hi_mod) * u64::from(pow64_mod_m) + u64::from(lo_mod);
        (t % u64::from(modulus)) as u32
    }

    /// Rust mirror of the candidate loop in `niceonly_ranges_kernel`,
    /// with all 32 lanes' indices merged (g increments by 1).
    fn mirror_kernel_candidates(range: &FieldSize, table: &StrideTable) -> Vec<u128> {
        let modulus = table.modulus as u32;
        let pow64_mod_m = ((1u128 << 64) % table.modulus) as u32;
        let residues: Vec<u32> = table.valid_residues.clone();
        let r_count = residues.len() as u32;

        let m = mirror_mod_m(range.start(), modulus, pow64_mod_m);
        let b0 = range.start() - u128::from(m);
        let idx0 = residues.partition_point(|&r| r < m) as u32;

        let mut out = Vec::new();
        let mut g = idx0;
        loop {
            let cycle = g / r_count;
            let j = g - cycle * r_count;
            let add = u64::from(cycle) * u64::from(modulus) + u64::from(residues[j as usize]);
            let n = b0 + u128::from(add);
            if n >= range.end() {
                break;
            }
            out.push(n);
            g += 1;
        }
        out
    }

    /// Candidates via the trusted CPU stride table iteration.
    fn cpu_candidates(range: &FieldSize, table: &StrideTable) -> Vec<u128> {
        let mut out = Vec::new();
        let (mut n, mut idx) = table.first_valid_at_or_after(range.start());
        while n < range.end() {
            out.push(n);
            n += u128::from(table.gap_table[idx]);
            idx = (idx + 1) % table.gap_table.len();
        }
        out
    }

    #[test_log::test]
    fn kernel_candidate_enumeration_matches_stride_table() {
        for base in MIRROR_TEST_BASES {
            let Ok(Some(base_range)) = base_range::get_base_range_u128(base) else {
                continue;
            };
            let table = StrideTable::new(base, GPU_LSD_K);
            if table.valid_residues.is_empty() {
                continue;
            }

            // Sub-ranges probing: field start, mid-range, modulus wraparound,
            // and small/odd sizes.
            let modulus = table.modulus;
            // A start whose residue lands strictly past the last valid
            // residue, forcing the kernel's lower_bound to return R (the
            // next-cycle wraparound case).
            let past_last = {
                let m_target = u128::from(table.valid_residues.last().unwrap() + 1);
                let cycle_base = base_range.range_start - (base_range.range_start % modulus);
                let mut s = cycle_base + m_target.min(modulus - 1);
                if s < base_range.range_start {
                    s += modulus;
                }
                s
            };
            let starts = [
                base_range.range_start,
                base_range.range_start + 1,
                base_range.range_start + modulus - 1,
                base_range.range_start + modulus * 7 + modulus / 2,
                base_range.range_start + (base_range.range_end - base_range.range_start) / 2,
                past_last,
            ];
            for start in starts {
                for size in [1u128, 250, 1999, 3 * modulus + 17] {
                    let end = (start + size).min(base_range.range_end);
                    if start >= end {
                        continue;
                    }
                    let range = FieldSize::new(start, end);
                    assert_eq!(
                        mirror_kernel_candidates(&range, &table),
                        cpu_candidates(&range, &table),
                        "candidate mismatch: base {base} range [{start}, {end})"
                    );
                }
            }
        }
    }

    #[test_log::test]
    fn mirror_mod_m_matches_direct() {
        for base in MIRROR_TEST_BASES {
            let table = StrideTable::new(base, GPU_LSD_K);
            if table.valid_residues.is_empty() {
                continue;
            }
            let modulus = table.modulus as u32;
            let pow64_mod_m = ((1u128 << 64) % table.modulus) as u32;
            // Deterministic pseudo-random u128 samples.
            let mut x: u128 = 0x1234_5678_9abc_def0_1122_3344_5566_7788;
            for _ in 0..1000 {
                x = x.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
                assert_eq!(
                    u128::from(mirror_mod_m(x, modulus, pow64_mod_m)),
                    x % table.modulus,
                    "mod_m mismatch for base {base}, n={x}"
                );
            }
        }
    }

    // ------------------------------------------------------------------
    // Mirror of the kernel's digit extraction
    // ------------------------------------------------------------------

    /// Rust mirror of `mul_limbs` in `nice_kernels.cu`.
    fn mirror_mul_limbs(a: &[u32], b: &[u32]) -> Vec<u32> {
        let mut r = vec![0u32; a.len() + b.len()];
        for i in 0..a.len() {
            let mut carry = 0u64;
            for j in 0..b.len() {
                let cur = u64::from(a[i]) * u64::from(b[j]) + u64::from(r[i + j]) + carry;
                r[i + j] = cur as u32;
                carry = cur >> 32;
            }
            r[i + b.len()] = carry as u32;
        }
        r
    }

    /// Rust mirror of `scan_digits` in `nice_kernels.cu`. Returns the digits in
    /// extraction order, or None if `stop_on_dup` hit a duplicate (with the
    /// digits seen so far tracked in `seen`).
    fn mirror_scan_digits(
        v: &mut [u32],
        chunk_digits: u32,
        chunk_div: u32,
        base: u32,
        seen: &mut [bool; 128],
        stop_on_dup: bool,
        digits_out: &mut Vec<u32>,
    ) -> bool {
        let mut top = v.len() as i32 - 1;
        while top >= 0 && v[top as usize] == 0 {
            top -= 1;
        }
        while top >= 0 {
            let mut rem = 0u32;
            for i in (0..=top).rev() {
                let cur = (u64::from(rem) << 32) | u64::from(v[i as usize]);
                let q = cur / u64::from(chunk_div);
                rem = (cur - q * u64::from(chunk_div)) as u32;
                v[i as usize] = q as u32;
            }
            while top >= 0 && v[top as usize] == 0 {
                top -= 1;
            }
            let mut chunk = rem;
            if top >= 0 {
                for _ in 0..chunk_digits {
                    let d = chunk % base;
                    chunk /= base;
                    digits_out.push(d);
                    if stop_on_dup && seen[d as usize] {
                        return false;
                    }
                    seen[d as usize] = true;
                }
            } else {
                while chunk != 0 {
                    let d = chunk % base;
                    chunk /= base;
                    digits_out.push(d);
                    if stop_on_dup && seen[d as usize] {
                        return false;
                    }
                    seen[d as usize] = true;
                }
            }
        }
        true
    }

    /// Rust mirror of `square_and_cube` + both scans, computing `num_uniques`
    /// the way the detailed kernel does.
    fn mirror_num_unique_digits(n: u128, base: u32) -> u32 {
        let (chunk_digits, chunk_div) = chunk_constants(base);
        let n_bits = 128 - n.leading_zeros();
        let n_limbs = (n_bits.div_ceil(32).max(1)) as usize;
        let n32: Vec<u32> = (0..n_limbs).map(|i| (n >> (32 * i)) as u32).collect();

        let mut sq = mirror_mul_limbs(&n32, &n32);
        let mut cu = mirror_mul_limbs(&sq, &n32);

        let mut seen = [false; 128];
        let mut digits = Vec::new();
        mirror_scan_digits(
            &mut sq,
            chunk_digits,
            chunk_div,
            base,
            &mut seen,
            false,
            &mut digits,
        );
        mirror_scan_digits(
            &mut cu,
            chunk_digits,
            chunk_div,
            base,
            &mut seen,
            false,
            &mut digits,
        );
        seen.iter().filter(|&&s| s).count() as u32
    }

    /// Rust mirror of `reduce_pre` in `nice_kernels.cu`.
    fn mirror_reduce_pre(mut hi: u64, mut lo: u64, modulus: u64, pow64_mod: u64) -> u64 {
        while hi != 0 {
            let p_lo = hi.wrapping_mul(pow64_mod);
            let p_hi = ((u128::from(hi) * u128::from(pow64_mod)) >> 64) as u64;
            lo = lo.wrapping_add(p_lo);
            hi = p_hi + u64::from(lo < p_lo);
        }
        lo % modulus
    }

    /// Rust mirror of `prefilter_low_digits` in `nice_kernels.cu`.
    fn mirror_prefilter(n: u128, base: u32, pre: &PrefilterParams) -> bool {
        let mulhi = |a: u64, b: u64| ((u128::from(a) * u128::from(b)) >> 64) as u64;
        let mulmod = |a: u64, b: u64| {
            mirror_reduce_pre(mulhi(a, b), a.wrapping_mul(b), pre.modulus, pre.pow64_mod)
        };
        let (n_lo, n_hi) = split_u128(n);
        let nm = mirror_reduce_pre(n_hi, n_lo, pre.modulus, pre.pow64_mod);
        let mut sq = mulmod(nm, nm);
        let mut cu = mulmod(sq, nm);

        let mut seen = [false; 128];
        let mut dup = false;
        for _ in 0..pre.digits {
            let d = (sq % u64::from(base)) as usize;
            sq /= u64::from(base);
            dup |= seen[d];
            seen[d] = true;
        }
        for _ in 0..pre.digits {
            let d = (cu % u64::from(base)) as usize;
            cu /= u64::from(base);
            dup |= seen[d];
            seen[d] = true;
        }
        !dup
    }

    #[test_log::test]
    fn prefilter_modular_arithmetic_matches_direct() {
        for base in MIRROR_TEST_BASES {
            let Some(pre) = prefilter_params(base) else {
                continue;
            };
            let Ok(Some(base_range)) = base_range::get_base_range_u128(base) else {
                continue;
            };
            let m = u128::from(pre.modulus);
            assert_eq!(u128::from(pre.pow64_mod), (1u128 << 64) % m);
            assert_eq!(m, u128::from(base).pow(pre.digits));

            let span = base_range.range_end - base_range.range_start;
            let mut x: u128 = 0x0123_4567_89ab_cdef_0f1e_2d3c_4b5a_6978;
            for i in 0..500u128 {
                x = x.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(i);
                let n = base_range.range_start + (x % span);
                // reduce_pre(n) == n mod m
                let (n_lo, n_hi) = split_u128(n);
                let nm = mirror_reduce_pre(n_hi, n_lo, pre.modulus, pre.pow64_mod);
                assert_eq!(u128::from(nm), n % m, "reduce_pre mismatch b{base} n={n}");
                // and the mulmod chain reproduces n^2 mod m, n^3 mod m
                let sq_direct = (n % m) * (n % m) % m;
                let cu_direct = sq_direct * (n % m) % m;
                let mulhi = |a: u64, b: u64| ((u128::from(a) * u128::from(b)) >> 64) as u64;
                let sq = mirror_reduce_pre(
                    mulhi(nm, nm),
                    nm.wrapping_mul(nm),
                    pre.modulus,
                    pre.pow64_mod,
                );
                let cu = mirror_reduce_pre(
                    mulhi(sq, nm),
                    sq.wrapping_mul(nm),
                    pre.modulus,
                    pre.pow64_mod,
                );
                assert_eq!(u128::from(sq), sq_direct, "sq mismatch b{base} n={n}");
                assert_eq!(u128::from(cu), cu_direct, "cu mismatch b{base} n={n}");
            }
        }
    }

    /// Sample count for `prefilter_is_sound_and_selective`.
    const SAMPLES: u32 = 2000;

    #[test_log::test]
    fn prefilter_is_sound_and_selective() {
        for base in MIRROR_TEST_BASES {
            let Some(pre) = prefilter_params(base) else {
                continue;
            };
            let Ok(Some(base_range)) = base_range::get_base_range_u128(base) else {
                continue;
            };
            let span = base_range.range_end - base_range.range_start;
            let mut x: u128 = 0xdead_beef_cafe_f00d_0d15_ea5e_feed_face;
            let mut rejected = 0u32;
            for i in 0..SAMPLES {
                x = x
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(u128::from(i));
                let n = base_range.range_start + (x % span);
                if mirror_prefilter(n, base, &pre) {
                    continue;
                }
                rejected += 1;
                // Soundness: a rejected candidate must not be nice.
                assert!(
                    !client_process::get_is_nice(n, base),
                    "prefilter rejected a nice number: b{base} n={n}"
                );
            }
            // Selectivity sanity: expected kill rates are 76-99% for these
            // bases; require at least half so a broken filter can't silently
            // pass everything.
            assert!(
                rejected * 2 > SAMPLES,
                "prefilter suspiciously weak at b{base}: {rejected}/{SAMPLES}"
            );
        }
    }

    #[test_log::test]
    fn prefilter_guard_disables_small_bases() {
        // b10's n^2 has ~4 digits, far below PRE_DIGITS — the digit-count
        // guard must disable the prefilter or it would extract phantom zeros.
        assert!(prefilter_params(10).is_none());
        // Low bases where the prefilter pays (survival ~1%) keep it.
        for base in [30, 40] {
            assert!(
                prefilter_params(base).is_some(),
                "expected prefilter at b{base}"
            );
        }
        // Above the profitability threshold it is compiled out even though
        // it would be sound: at 4%+ lane survival most warps run the full
        // check anyway (GPU_PREFILTER_MAX_BASE, g1-verdict.md).
        for base in [42, 52, 62, 68] {
            assert!(
                prefilter_params(base).is_none(),
                "expected prefilter disabled at b{base}"
            );
        }
    }

    /// Regression for the v3.2.14 phantom-zero bug: an `#ifndef PREFILTER`
    /// fallback in the kernel source force-enabled the prefilter (with
    /// base-40 constants) on the bases where the host deliberately omits it,
    /// so the GPU silently rejected every candidate on b10-25. The define
    /// must come only from the host or the standalone syntax-check block.
    #[test_log::test]
    fn prefilter_has_no_ifndef_fallback() {
        let kernel_src = include_str!("cuda/nice_kernels.cu");
        assert!(
            !kernel_src.contains("#ifndef PREFILTER"),
            "PREFILTER must not have an #ifndef fallback; the host omits the \
             define deliberately for bases with too-short n^2/n^3"
        );
        // And the host must keep omitting it where the guard says so.
        for base in [10u32, 12, 25] {
            let (defines, _) = niceonly_defines(base).unwrap();
            assert!(
                !defines.iter().any(|d| d.starts_with("PREFILTER")),
                "b{base}: host emitted PREFILTER despite disabled guard"
            );
        }
    }

    #[test_log::test]
    fn mirror_digit_extraction_matches_cpu() {
        for base in MIRROR_TEST_BASES {
            let Ok(Some(base_range)) = base_range::get_base_range_u128(base) else {
                continue;
            };
            // Deterministic samples across the base's range.
            let span = base_range.range_end - base_range.range_start;
            let mut x: u128 = 0x9e37_79b9_7f4a_7c15_f39c_c060_5ced_c834;
            for i in 0..200u128 {
                x = x.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(i);
                let n = base_range.range_start + (x % span);
                assert_eq!(
                    mirror_num_unique_digits(n, base),
                    client_process::get_num_unique_digits(n, base),
                    "unique digit count mismatch: base {base}, n={n}"
                );
            }
        }
    }

    #[test_log::test]
    fn mirror_digit_extraction_known_nice() {
        // 69 is nice in base 10: 69² = 4761, 69³ = 328509.
        assert_eq!(mirror_num_unique_digits(69, 10), 10);
        assert!(client_process::get_is_nice(69, 10));
    }

    // ------------------------------------------------------------------
    // NVRTC compile tests: need libnvrtc but NO GPU device, so they run
    // on any machine with the CUDA runtime libraries installed (e.g. inside
    // the nvidia/cuda docker image). Skipped gracefully when NVRTC is absent.
    // ------------------------------------------------------------------

    #[test_log::test]
    fn nvrtc_compiles_kernels_for_all_supported_bases() {
        // Probe with a trivial program first: if THIS fails, the library is
        // missing and we skip; any later failure is a real kernel bug.
        // cudarc panics (rather than erroring) when libnvrtc can't be
        // loaded, so the probe runs under catch_unwind.
        let probe = std::panic::catch_unwind(|| {
            compile_ptx_with_opts(
                "extern \"C\" __global__ void probe() {}",
                CompileOptions::default(),
            )
        });
        if !matches!(probe, Ok(Ok(_))) {
            println!("NVRTC not available, skipping compile test");
            return;
        }

        for base in 10..=MAX_GPU_DIGIT_MASK_BASE {
            if !gpu_supports_base(base) {
                continue;
            }
            let defines = detailed_defines(base).unwrap();
            compile_kernel_ptx(&defines)
                .unwrap_or_else(|e| panic!("detailed kernel failed to compile for b{base}: {e:?}"));
            if !residue_filter::get_residue_filter_u128(&base).is_empty() {
                let (defines, _table) = niceonly_defines(base).unwrap();
                compile_kernel_ptx(&defines).unwrap_or_else(|e| {
                    panic!("niceonly kernel failed to compile for b{base}: {e:?}")
                });
            }
        }
    }

    // ------------------------------------------------------------------
    // GPU integration tests (run on a CUDA machine with --ignored)
    // ------------------------------------------------------------------

    #[test_log::test]
    #[ignore = "requires GPU"]
    fn gpu_kernels_compile_for_all_supported_bases() {
        let Some(ctx) = try_init_cuda() else {
            println!("GPU not available, skipping test");
            return;
        };
        for base in 2..=MAX_GPU_DIGIT_MASK_BASE {
            if !gpu_supports_base(base) {
                continue;
            }
            ctx.detailed_kernel(base)
                .unwrap_or_else(|e| panic!("detailed kernel failed for base {base}: {e:?}"));
            if !residue_filter::get_residue_filter_u128(&base).is_empty() {
                ctx.niceonly_plan(base)
                    .unwrap_or_else(|e| panic!("niceonly kernel failed for base {base}: {e:?}"));
            }
        }
    }

    #[test_log::test]
    #[ignore = "requires GPU"]
    fn gpu_matches_cpu_detailed_small() {
        let Some(ctx) = try_init_cuda() else {
            println!("GPU not available, skipping test");
            return;
        };

        for (base, start, size) in [
            (10u32, 1_000_000u128, 10_000u128),
            (40, 2_000_000_000_000, 100_000),
        ] {
            let range = FieldSize::new(start, start + size);
            let cpu = process_range_detailed(&range, base);
            let gpu = process_range_detailed_cuda(&ctx, &range, base).expect("GPU failed");

            assert_eq!(
                cpu.distribution, gpu.distribution,
                "distribution mismatch at base {base}"
            );
            assert_eq!(
                cpu.nice_numbers, gpu.nice_numbers,
                "near-miss mismatch at base {base}"
            );
        }
    }

    #[test_log::test]
    #[ignore = "requires GPU"]
    fn gpu_matches_cpu_niceonly() {
        let Some(ctx) = try_init_cuda() else {
            println!("GPU not available, skipping test");
            return;
        };

        // 10, 12, and 25 run with the prefilter host-disabled (regression for
        // the v3.2.14 phantom-zero bug, where the GPU missed every nice
        // number on such bases); the rest run the full prefilter path.
        for base in [10u32, 12, 25, 40, 45, 62] {
            let Ok(Some(base_range)) = base_range::get_base_range_u128(base) else {
                continue;
            };
            let start = base_range.range_start;
            let end = (start + 5_000_000).min(base_range.range_end);
            let range = FieldSize::new(start, end);

            let stride_table = StrideTable::new(base, GPU_LSD_K);
            let cpu = process_range_niceonly(&range, base, &stride_table);
            let gpu = process_range_niceonly_cuda(&ctx, &range, base).expect("GPU failed");

            let mut cpu_nice = cpu.nice_numbers;
            cpu_nice.sort_by_key(|n| n.number);
            assert_eq!(
                cpu_nice, gpu.nice_numbers,
                "niceonly mismatch at base {base}"
            );
        }
    }

    #[test_log::test]
    fn test_split_combine_u128() {
        for num in [0u128, 1, 12345, u128::from(u64::MAX), u128::MAX] {
            let (lo, hi) = split_u128(num);
            assert_eq!(combine_u64(lo, hi), num);
        }
    }
}
