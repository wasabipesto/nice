//! GPU-accelerated implementation of nice number checking using CUDA.
//!
//! This module offloads the hot loops to the GPU while keeping the same
//! filter cascade and result semantics as the CPU path:
//!
//! - **Niceonly**: the CPU runs the real MSD prefix filter
//!   (`msd_prefix_filter::get_valid_ranges`, parallelized across cores) at
//!   production chunking, then ships only compact *range descriptors* to the
//!   GPU (~12 bytes per surviving range). The GPU reconstructs the stride
//!   filter's candidates on-device from the residue table — the g-th valid
//!   candidate at or after a range start is `B0 + (g/R)*M + residues[g%R]` —
//!   and runs the early-exit nice check. No per-candidate data ever crosses
//!   the bus, and the candidate set is identical to the CPU path's.
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
//! The GPU path supports bases up to `MAX_BASE_FOR_FIXED_WIDTH_U256` (68),
//! where n³ still fits in 256 bits. Higher bases (or bases with no valid
//! u128 range) fall back to the CPU implementation with a logged warning.

#![cfg(feature = "gpu")]
#![allow(clippy::cast_possible_truncation)]

use crate::client_process::{
    MAX_BASE_FOR_FIXED_WIDTH_U256, process_range_detailed, process_range_niceonly,
};
use crate::{
    CLIENT_VERSION, DataToClient, DataToServer, FieldResults, FieldSize, NiceNumberSimple,
    PROCESSING_CHUNK_SIZE, UniquesDistributionSimple,
};
use crate::{base_range, msd_prefix_filter, number_stats, residue_filter, stride_filter};
use anyhow::{Context as _, Result, bail, ensure};
use cudarc::driver::{
    CudaContext, CudaFunction, CudaSlice, CudaStream, LaunchConfig, PushKernelArg,
};
use cudarc::nvrtc::{CompileOptions, Ptx, compile_ptx_with_opts};
use log::{debug, info, warn};
use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

/// Numbers processed per detailed-mode kernel launch. Larger batches amortize
/// launch overhead; since the detailed kernel takes no input arrays, batch
/// size costs no memory or transfer bandwidth.
pub const GPU_BATCH_SIZE: usize = 50_000_000;

/// LSD filter depth for the stride table, matching the CPU client's
/// `DEFAULT_LSD_K_VALUE` so GPU and CPU check the identical candidate set.
const GPU_LSD_K: u32 = 2;

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
pub struct GpuContext {
    device: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    niceonly_plans: Mutex<HashMap<u32, Arc<NiceonlyPlan>>>,
    detailed_kernels: Mutex<HashMap<u32, CudaFunction>>,
}

impl GpuContext {
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
        let device = CudaContext::new(device_ordinal)
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

        Ok(GpuContext {
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
        let residues_host: Vec<u32> = table.valid_residues.iter().map(|&r| r as u32).collect();
        let num_residues = residues_host.len() as u32;
        let pow64_mod_m = ((1u128 << 64) % table.modulus) as u32;

        let mut defines = common_defines(base)?;
        defines.push("NICEONLY".to_string());
        defines.push(format!("STRIDE_M={modulus}u"));
        defines.push(format!("STRIDE_R={num_residues}u"));
        defines.push(format!("POW64_MOD_M={pow64_mod_m}u"));

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
        base <= MAX_BASE_FOR_FIXED_WIDTH_U256,
        "base {base} exceeds GPU max base {MAX_BASE_FOR_FIXED_WIDTH_U256} (n³ overflows 256 bits)"
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

/// Largest (e, base^e) with base^e < 2^31. The kernel splits n² and n³ into
/// chunks of `e` digits by dividing by `base^e`, then peels single digits
/// from each u32 chunk — all divisions by compile-time constants.
fn chunk_constants(base: u32) -> (u32, u32) {
    let mut e = 0u32;
    let mut div = 1u64;
    while div * u64::from(base) < (1 << 31) {
        div *= u64::from(base);
        e += 1;
    }
    (e, div as u32)
}

/// Whether the GPU path can process this base natively. Bases outside this
/// fall back to the CPU implementation.
fn gpu_supports_base(base: u32) -> bool {
    base <= MAX_BASE_FOR_FIXED_WIDTH_U256
        && matches!(base_range::get_base_range_u128(base), Ok(Some(_)))
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
pub fn process_range_niceonly_gpu(
    ctx: &GpuContext,
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
    // front, so the phase timings below reflect per-field work only.
    let plan = ctx.niceonly_plan(base)?;

    // Phase 1: MSD prefix filter on CPU, parallel over production-size chunks.
    // Chunking at PROCESSING_CHUNK_SIZE matters: it lets the recursion reach
    // full pruning depth, exactly like the CPU client.
    let msd_start = Instant::now();
    let (offsets, lens) = msd_filter_parallel(range, base)?;
    let msd_secs = msd_start.elapsed().as_secs_f64();
    let valid_numbers: u64 = lens.iter().map(|&l| u64::from(l)).sum();

    // Phase 2: check surviving ranges on the GPU.
    let gpu_start = Instant::now();
    let mut nice_numbers: Vec<NiceNumberSimple> = Vec::new();
    if !offsets.is_empty() {
        nice_numbers = launch_niceonly(ctx, &plan, range, &offsets, &lens)?;
    }
    let gpu_secs = gpu_start.elapsed().as_secs_f64();

    #[allow(clippy::cast_precision_loss)]
    {
        let total_secs = msd_secs + gpu_secs;
        info!(
            "GPU niceonly b{base}: msd {msd_secs:.2}s -> {} ranges ({:.2}% of field), gpu {gpu_secs:.2}s, {:.2e} n/s overall",
            offsets.len(),
            100.0 * valid_numbers as f64 / range.size() as f64,
            range.size() as f64 / total_secs,
        );
    }

    Ok(FieldResults {
        distribution: Vec::new(),
        nice_numbers,
    })
}

/// Run `get_valid_ranges` over the field's chunks on all available cores.
/// Returns the surviving ranges as (offset from field start, length) pairs.
fn msd_filter_parallel(range: &FieldSize, base: u32) -> Result<(Vec<u64>, Vec<u32>)> {
    let chunks = range.chunks(PROCESSING_CHUNK_SIZE);
    let num_threads = std::thread::available_parallelism()
        .map(std::num::NonZeroUsize::get)
        .unwrap_or(4)
        .min(chunks.len().max(1));

    let next_chunk = AtomicUsize::new(0);
    let collected: Mutex<(Vec<u64>, Vec<u32>)> = Mutex::new((Vec::new(), Vec::new()));
    let error: Mutex<Option<anyhow::Error>> = Mutex::new(None);

    std::thread::scope(|scope| {
        for _ in 0..num_threads {
            scope.spawn(|| {
                let mut local_offsets: Vec<u64> = Vec::new();
                let mut local_lens: Vec<u32> = Vec::new();
                loop {
                    let i = next_chunk.fetch_add(1, Ordering::Relaxed);
                    let Some(chunk) = chunks.get(i) else { break };
                    for sub in msd_prefix_filter::get_valid_ranges(*chunk, base) {
                        let offset = u64::try_from(sub.start() - range.start());
                        let len = u32::try_from(sub.size());
                        if let (Ok(offset), Ok(len)) = (offset, len) {
                            local_offsets.push(offset);
                            local_lens.push(len);
                        } else {
                            *error.lock().unwrap() = Some(anyhow::anyhow!(
                                "valid range doesn't fit descriptor: start {} size {}",
                                sub.start(),
                                sub.size()
                            ));
                            return;
                        }
                    }
                }
                let mut guard = collected.lock().unwrap();
                guard.0.extend(local_offsets);
                guard.1.extend(local_lens);
            });
        }
    });

    if let Some(e) = error.into_inner().unwrap() {
        return Err(e);
    }
    Ok(collected.into_inner().unwrap())
}

/// Upload range descriptors and run the niceonly kernel over them.
fn launch_niceonly(
    ctx: &GpuContext,
    plan: &NiceonlyPlan,
    range: &FieldSize,
    offsets: &[u64],
    lens: &[u32],
) -> Result<Vec<NiceNumberSimple>> {
    let (field_start_lo, field_start_hi) = split_u128(range.start());

    let nice_capacity = NICE_OUT_CAPACITY as u32;
    let d_nice_out = ctx.stream.alloc_zeros::<u64>(2 * NICE_OUT_CAPACITY)?;
    let mut d_nice_count = ctx.stream.alloc_zeros::<u32>(1)?;

    for (batch_offsets, batch_lens) in offsets
        .chunks(RANGES_PER_LAUNCH)
        .zip(lens.chunks(RANGES_PER_LAUNCH))
    {
        let d_offsets = ctx.stream.clone_htod(batch_offsets)?;
        let d_lens = ctx.stream.clone_htod(batch_lens)?;
        let num_ranges = batch_offsets.len() as u32;

        // One warp per range.
        let total_threads = u64::from(num_ranges) * 32;
        let grid_blocks = total_threads.div_ceil(u64::from(BLOCK_THREADS)) as u32;
        let cfg = LaunchConfig {
            grid_dim: (grid_blocks, 1, 1),
            block_dim: (BLOCK_THREADS, 1, 1),
            shared_mem_bytes: 0,
        };

        let mut launch_args = ctx.stream.launch_builder(&plan.func);
        launch_args.arg(&field_start_lo);
        launch_args.arg(&field_start_hi);
        launch_args.arg(&d_offsets);
        launch_args.arg(&d_lens);
        launch_args.arg(&num_ranges);
        launch_args.arg(&plan.residues);
        launch_args.arg(&d_nice_out);
        launch_args.arg(&mut d_nice_count);
        launch_args.arg(&nice_capacity);
        unsafe {
            launch_args.launch(cfg)?;
        }
    }

    let nice_count = ctx.stream.clone_dtoh(&d_nice_count)?[0] as usize;
    if nice_count > NICE_OUT_CAPACITY {
        bail!(
            "niceonly output buffer overflow: {nice_count} > {NICE_OUT_CAPACITY} \
             (this strongly suggests a kernel bug)"
        );
    }
    let mut nice_numbers = Vec::with_capacity(nice_count);
    if nice_count > 0 {
        let out = ctx.stream.clone_dtoh(&d_nice_out)?;
        for i in 0..nice_count {
            nice_numbers.push(NiceNumberSimple {
                number: combine_u64(out[2 * i], out[2 * i + 1]),
                num_uniques: plan.base,
            });
        }
        nice_numbers.sort_by_key(|n| n.number);
    }
    debug!(
        "GPU niceonly launch: {} ranges, M={}, R={}, found {}",
        offsets.len(),
        plan.modulus,
        plan.num_residues,
        nice_count
    );
    Ok(nice_numbers)
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
pub fn process_range_detailed_gpu(
    ctx: &GpuContext,
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

    for batch in range.chunks(GPU_BATCH_SIZE as u128) {
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
pub fn process_detailed_gpu(
    ctx: &GpuContext,
    claim_data: &DataToClient,
    username: &String,
) -> Result<DataToServer> {
    let results = process_range_detailed_gpu(ctx, &claim_data.into(), claim_data.base)?;

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
/// # Errors
/// Returns an error on any CUDA failure.
pub fn process_niceonly_gpu(
    ctx: &GpuContext,
    claim_data: &DataToClient,
    username: &String,
) -> Result<DataToServer> {
    let results = process_range_niceonly_gpu(ctx, &claim_data.into(), claim_data.base)?;

    Ok(DataToServer {
        claim_id: claim_data.claim_id,
        username: username.to_owned(),
        client_version: CLIENT_VERSION.to_string(),
        unique_distribution: None,
        nice_numbers: results.nice_numbers,
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
    use crate::stride_filter::StrideTable;

    /// Bases used for CPU-side mirror tests: a mix of small, u64-range,
    /// u128-range, and two-mask (>64) regimes.
    const MIRROR_TEST_BASES: [u32; 6] = [10, 40, 45, 57, 62, 68];

    fn try_init_gpu() -> Option<GpuContext> {
        GpuContext::new(0).ok()
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
        let residues: Vec<u32> = table.valid_residues.iter().map(|&r| r as u32).collect();
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
            n += table.gap_table[idx];
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
                let m_target = table.valid_residues.last().unwrap() + 1;
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

    #[test_log::test]
    fn chunk_constants_are_maximal() {
        for base in 2..=MAX_BASE_FOR_FIXED_WIDTH_U256 {
            let (e, div) = chunk_constants(base);
            assert!(e >= 1);
            assert_eq!(u64::from(div), u64::from(base).pow(e));
            assert!(u64::from(div) < (1 << 31), "base {base}: div too large");
            assert!(
                u64::from(div) * u64::from(base) >= (1 << 31),
                "base {base}: e not maximal"
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
    // GPU integration tests (run on a CUDA machine with --ignored)
    // ------------------------------------------------------------------

    #[test_log::test]
    #[ignore = "requires GPU"]
    fn gpu_kernels_compile_for_all_supported_bases() {
        let Some(ctx) = try_init_gpu() else {
            println!("GPU not available, skipping test");
            return;
        };
        for base in 2..=MAX_BASE_FOR_FIXED_WIDTH_U256 {
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
        let Some(ctx) = try_init_gpu() else {
            println!("GPU not available, skipping test");
            return;
        };

        for (base, start, size) in [
            (10u32, 1_000_000u128, 10_000u128),
            (40, 2_000_000_000_000, 100_000),
        ] {
            let range = FieldSize::new(start, start + size);
            let cpu = process_range_detailed(&range, base);
            let gpu = process_range_detailed_gpu(&ctx, &range, base).expect("GPU failed");

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
        let Some(ctx) = try_init_gpu() else {
            println!("GPU not available, skipping test");
            return;
        };

        for base in [10u32, 40, 45, 62] {
            let Ok(Some(base_range)) = base_range::get_base_range_u128(base) else {
                continue;
            };
            let start = base_range.range_start;
            let end = (start + 5_000_000).min(base_range.range_end);
            let range = FieldSize::new(start, end);

            let stride_table = StrideTable::new(base, GPU_LSD_K);
            let cpu = process_range_niceonly(&range, base, &stride_table);
            let gpu = process_range_niceonly_gpu(&ctx, &range, base).expect("GPU failed");

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
