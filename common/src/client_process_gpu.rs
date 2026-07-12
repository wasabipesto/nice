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

#![cfg(feature = "gpu")]
#![allow(clippy::cast_possible_truncation)]

use crate::client_process::{process_range_detailed, process_range_niceonly};
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

/// Minimum MSD recursion floor (matches the CPU client's default).
/// Below this the GPU receives virtually the same candidates as the CPU would
/// check itself, so there is no point going lower.
const MSD_FLOOR_MIN: f64 = 250.0;

/// Maximum useful MSD recursion floor. Beyond ~64 000 the survival rate
/// saturates around 23 % (b52 measurement), so larger values buy nothing.
/// Measured on a real b52 production field (per 1e12 numbers, single core):
///
/// | floor  | CPU time | surviving |
/// |--------|----------|-----------|
/// | 250    | 350 s    | 2.3 %     |
/// | 4 000  | 50 s     | 15.2 %    |
/// | 16 000 | 15 s     | 19.0 %    |
/// | 64 000 | 4.8 s    | 22.6 %    |
const MSD_FLOOR_MAX: f64 = 256_000.0;

/// Adaptive MSD recursion floor for the niceonly GPU pipeline.
///
/// Goal: keep `msd_time ≈ gpu_tail_time` so the overlapped pipeline is
/// balanced.  The floor is seeded from the CPU count (fewer cores → coarser
/// floor, because MSD is the bottleneck) and then nudged ≤ 1.5× per field
/// toward that balance.  Setting `NICE_GPU_MSD_FLOOR` in the environment
/// pins the floor and disables adaptation.
struct AdaptiveFloor {
    floor: f64,
    /// Fields remaining in warmup (skip adaptation); `u32::MAX` = permanently
    /// fixed via env-var override.
    warmup: u32,
}

/// Fields to observe before adapting, so NVRTC and JIT one-time costs don't
/// skew the first measurement.
const ADAPT_WARMUP: u32 = 3;

/// Maximum multiplicative step per field in either direction.
const ADAPT_MAX_STEP: f64 = 1.5;

/// Ignore a phase if it took less than this many seconds — the measurement
/// noise would dominate the ratio.
const ADAPT_MIN_SECS: f64 = 0.002;

/// Floor value calibrated for 32 cores. Derived value for N cores:
/// `ADAPT_BASE_CORE_PRODUCT / N`, clamped to `[MSD_FLOOR_MIN, MSD_FLOOR_MAX]`.
const ADAPT_BASE_CORE_PRODUCT: f64 = 512_000.0;

impl AdaptiveFloor {
    fn current(&self) -> u128 {
        self.floor as u128
    }

    fn update(&mut self, msd_secs: f64, total_secs: f64) {
        if self.warmup == u32::MAX {
            return;
        }
        if self.warmup > 0 {
            self.warmup -= 1;
            return;
        }
        let gpu_tail = (total_secs - msd_secs).max(0.0);
        let ratio = if gpu_tail < ADAPT_MIN_SECS {
            ADAPT_MAX_STEP
        } else if msd_secs < ADAPT_MIN_SECS {
            1.0 / ADAPT_MAX_STEP
        } else {
            msd_secs / gpu_tail
        };
        let factor = ratio.clamp(1.0 / ADAPT_MAX_STEP, ADAPT_MAX_STEP);
        let new_floor = (self.floor * factor).clamp(MSD_FLOOR_MIN, MSD_FLOOR_MAX);
        if (new_floor - self.floor).abs() > self.floor * 0.05 {
            info!(
                "GPU MSD floor: {:.0} → {:.0} (msd {:.3}s, gpu_tail {:.3}s)",
                self.floor, new_floor, msd_secs, gpu_tail,
            );
        }
        self.floor = new_floor;
    }
}

static ADAPTIVE_FLOOR: std::sync::OnceLock<Mutex<AdaptiveFloor>> = std::sync::OnceLock::new();

fn adaptive_floor() -> &'static Mutex<AdaptiveFloor> {
    ADAPTIVE_FLOOR.get_or_init(|| {
        if let Ok(v) = std::env::var("NICE_GPU_MSD_FLOOR") {
            match v.parse::<f64>() {
                Ok(f) if f >= 1.0 => {
                    info!("GPU MSD floor fixed at {f:.0} via NICE_GPU_MSD_FLOOR");
                    return Mutex::new(AdaptiveFloor {
                        floor: f,
                        warmup: u32::MAX,
                    });
                }
                _ => warn!("ignoring invalid NICE_GPU_MSD_FLOOR '{v}'; using adaptive floor"),
            }
        }
        let cpu_count = std::thread::available_parallelism()
            .map(std::num::NonZeroUsize::get)
            .unwrap_or(32) as f64;
        let seed = (ADAPT_BASE_CORE_PRODUCT / cpu_count).clamp(MSD_FLOOR_MIN, MSD_FLOOR_MAX);
        info!("GPU MSD floor: adaptive, seed {seed:.0} ({cpu_count:.0} logical cores)");
        Mutex::new(AdaptiveFloor {
            floor: seed,
            warmup: ADAPT_WARMUP,
        })
    })
}

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
        let (defines, table) = niceonly_defines(base)?;
        let modulus = table.modulus as u32;
        let residues_host: Vec<u32> = table.valid_residues.iter().map(|&r| r as u32).collect();
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

/// Highest base the GPU digit mask can represent (two u64 words). The
/// kernel's u32-limb arithmetic is width-generic (buffers are sized from
/// `N_LIMBS` at JIT time), so unlike the CPU's `U256` path there is no
/// 256-bit ceiling; in practice the u128 candidate representation caps
/// usable bases around 97 via `get_base_range_u128`.
const MAX_GPU_DIGIT_MASK_BASE: u32 = 128;

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

/// Parameters for the niceonly kernel's modular prefilter.
struct PrefilterParams {
    /// Digits checked per value (the lowest `digits` of n² and of n³).
    digits: u32,
    /// `base^digits`, at most 2^48.
    modulus: u64,
    /// `2^64 mod modulus`.
    pow64_mod: u64,
}

/// Compute the prefilter parameters for a base, or None when the prefilter
/// must stay disabled.
///
/// The prefilter checks the lowest p digits of n² and n³ using
/// `x mod b^p` arithmetic in u64. Constraints:
/// - `b^p <= 2^48` keeps every intermediate product reducible with a couple
///   of multiply-high steps;
/// - n² and n³ must each be guaranteed at least p digits across the base's
///   whole range, or the digit loop would extract phantom leading zeros and
///   could falsely reject a nice number. Verified with a conservative
///   log-based lower bound on the digit counts at the range start.
fn prefilter_params(base: u32) -> Option<PrefilterParams> {
    let mut digits = 0u32;
    let mut modulus = 1u64;
    while modulus <= (1u64 << 48) / u64::from(base) {
        modulus *= u64::from(base);
        digits += 1;
    }

    let range = base_range::get_base_range_u128(base).ok()??;
    #[allow(clippy::cast_precision_loss)]
    let ln_n_min = (range.range_start as f64).ln();
    let ln_base = f64::from(base).ln();
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let digit_lower_bound =
        |power: f64| ((power * ln_n_min / ln_base).floor() - 1.0).max(0.0) as u32;
    let sq_digits_min = digit_lower_bound(2.0);
    let cu_digits_min = digit_lower_bound(3.0);
    if digits < 4 || sq_digits_min < digits || cu_digits_min < digits {
        return None;
    }

    let pow64_mod = ((1u128 << 64) % u128::from(modulus)) as u64;
    Some(PrefilterParams {
        digits,
        modulus,
        pow64_mod,
    })
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
/// fall back to the CPU implementation. Unlike the CPU fast path (capped at
/// `MAX_BASE_FOR_FIXED_WIDTH_U256` = 68 by its 256-bit type), the GPU's
/// limb-generic arithmetic handles every base with a valid u128 range.
///
/// Bases below 10 are excluded: their search ranges are trivially small
/// (b5's is two numbers), and `get_base_range_u128` panics outright on
/// degenerate ones like b4 — not worth guarding for on the GPU path.
fn gpu_supports_base(base: u32) -> bool {
    (10..=MAX_GPU_DIGIT_MASK_BASE).contains(&base)
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
    // front, so the timings below reflect per-field work only.
    let plan = ctx.niceonly_plan(base)?;

    let (nice_numbers, stats) = run_niceonly_pipeline(ctx, &plan, range, base)?;

    #[allow(clippy::cast_precision_loss)]
    {
        info!(
            "GPU niceonly b{base}: msd {:.3}s -> {} ranges ({:.2}% of field), gpu tail {:.3}s, total {:.3}s, {:.2e} n/s overall",
            stats.msd_secs,
            stats.num_ranges,
            100.0 * stats.valid_numbers as f64 / range.size() as f64,
            (stats.total_secs - stats.msd_secs).max(0.0),
            stats.total_secs,
            range.size() as f64 / stats.total_secs,
        );
    }
    update_msd_floor(stats.msd_secs, stats.total_secs);

    Ok(FieldResults {
        distribution: Vec::new(),
        nice_numbers,
    })
}

fn gpu_msd_floor() -> u128 {
    adaptive_floor().lock().unwrap().current()
}

fn update_msd_floor(msd_secs: f64, total_secs: f64) {
    adaptive_floor()
        .lock()
        .unwrap()
        .update(msd_secs, total_secs);
}

/// Per-field statistics from the overlapped niceonly pipeline.
struct NiceonlyStats {
    /// Wall time until the MSD workers finished (launches overlap with this).
    msd_secs: f64,
    total_secs: f64,
    num_ranges: usize,
    valid_numbers: u64,
    launches: u32,
}

/// Ranges buffered before each kernel launch. Big enough to amortize launch
/// and upload overhead, small enough that launches start while the MSD
/// workers are still producing.
const LAUNCH_BATCH_RANGES: usize = 1 << 16;

/// Run the niceonly field: MSD workers stream surviving-range descriptors
/// through a channel while the main thread batches them into asynchronous
/// kernel launches, so the CPU filter and the GPU checks overlap instead of
/// running as sequential phases.
fn run_niceonly_pipeline(
    ctx: &GpuContext,
    plan: &NiceonlyPlan,
    range: &FieldSize,
    base: u32,
) -> Result<(Vec<NiceNumberSimple>, NiceonlyStats)> {
    let start_time = Instant::now();
    let chunks = range.chunks(PROCESSING_CHUNK_SIZE);
    let floor = gpu_msd_floor();
    let num_threads = std::thread::available_parallelism()
        .map(std::num::NonZeroUsize::get)
        .unwrap_or(4)
        .min(chunks.len().max(1));

    let mut launcher = NiceonlyLauncher::new(ctx, plan, range)?;

    let next_chunk = AtomicUsize::new(0);
    let worker_error: Mutex<Option<anyhow::Error>> = Mutex::new(None);
    let (tx, rx) = std::sync::mpsc::channel::<(Vec<u64>, Vec<u32>)>();

    let mut stats = NiceonlyStats {
        msd_secs: 0.0,
        total_secs: 0.0,
        num_ranges: 0,
        valid_numbers: 0,
        launches: 0,
    };
    let mut buf_offsets: Vec<u64> = Vec::new();
    let mut buf_lens: Vec<u32> = Vec::new();
    let mut launch_error: Option<anyhow::Error> = None;

    std::thread::scope(|scope| {
        let chunks = &chunks;
        let next_chunk = &next_chunk;
        let worker_error = &worker_error;
        for _ in 0..num_threads {
            let tx = tx.clone();
            scope.spawn(move || {
                loop {
                    let i = next_chunk.fetch_add(1, Ordering::Relaxed);
                    let Some(chunk) = chunks.get(i) else { break };
                    let mut offsets: Vec<u64> = Vec::new();
                    let mut lens: Vec<u32> = Vec::new();
                    for sub in msd_prefix_filter::get_valid_ranges_recursive(
                        *chunk,
                        base,
                        0,
                        msd_prefix_filter::MSD_RECURSIVE_MAX_DEPTH,
                        floor,
                        msd_prefix_filter::MSD_RECURSIVE_SUBDIVISION_FACTOR,
                    ) {
                        let offset = u64::try_from(sub.start() - range.start());
                        let len = u32::try_from(sub.size());
                        if let (Ok(offset), Ok(len)) = (offset, len) {
                            offsets.push(offset);
                            lens.push(len);
                        } else {
                            *worker_error.lock().unwrap() = Some(anyhow::anyhow!(
                                "valid range doesn't fit descriptor: start {} size {}",
                                sub.start(),
                                sub.size()
                            ));
                            return;
                        }
                    }
                    if !offsets.is_empty() {
                        // The receiver may be gone if a launch failed; the
                        // remaining chunks are then discarded.
                        let _ = tx.send((offsets, lens));
                    }
                }
            });
        }
        // The consumer runs on this thread while the workers produce. The
        // clone of `tx` held by each worker keeps the channel open; dropping
        // ours lets `recv` disconnect once they all finish.
        drop(tx);

        while let Ok((offsets, lens)) = rx.recv() {
            stats.num_ranges += offsets.len();
            stats.valid_numbers += lens.iter().map(|&l| u64::from(l)).sum::<u64>();
            buf_offsets.extend_from_slice(&offsets);
            buf_lens.extend_from_slice(&lens);
            if buf_offsets.len() >= LAUNCH_BATCH_RANGES {
                if let Err(e) = launcher.launch(&buf_offsets, &buf_lens) {
                    launch_error = Some(e);
                    break;
                }
                stats.launches += 1;
                buf_offsets.clear();
                buf_lens.clear();
            }
        }
        // Workers are done (or the launch failed); either way this marks the
        // end of the CPU-side phase.
        stats.msd_secs = start_time.elapsed().as_secs_f64();
    });

    if let Some(e) = launch_error {
        return Err(e);
    }
    if let Some(e) = worker_error.into_inner().unwrap() {
        return Err(e);
    }
    if !buf_offsets.is_empty() {
        launcher.launch(&buf_offsets, &buf_lens)?;
        stats.launches += 1;
    }

    let nice_numbers = launcher.finish()?;
    stats.total_secs = start_time.elapsed().as_secs_f64();
    debug!(
        "GPU niceonly pipeline: {} ranges in {} launches, M={}, R={}, found {}",
        stats.num_ranges,
        stats.launches,
        plan.modulus,
        plan.num_residues,
        nice_numbers.len()
    );
    Ok((nice_numbers, stats))
}

/// Holds the per-field output buffers and issues asynchronous niceonly
/// kernel launches over batches of range descriptors.
struct NiceonlyLauncher<'a> {
    ctx: &'a GpuContext,
    plan: &'a NiceonlyPlan,
    field_start_lo: u64,
    field_start_hi: u64,
    d_nice_out: CudaSlice<u64>,
    d_nice_count: CudaSlice<u32>,
}

impl<'a> NiceonlyLauncher<'a> {
    fn new(ctx: &'a GpuContext, plan: &'a NiceonlyPlan, range: &FieldSize) -> Result<Self> {
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

    /// Upload a batch of range descriptors and launch the kernel on them.
    /// Launches are asynchronous on the stream; results accumulate in the
    /// shared output buffers until [`Self::finish`].
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
    /// u128-range, two-mask (>64), and beyond-U256 (>68) regimes.
    const MIRROR_TEST_BASES: [u32; 8] = [10, 40, 45, 57, 62, 68, 70, 94];

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

    #[test_log::test]
    fn prefilter_is_sound_and_selective() {
        const SAMPLES: u32 = 2000;
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
        // Frontier bases must have it enabled.
        for base in [40, 52, 62, 68] {
            assert!(
                prefilter_params(base).is_some(),
                "expected prefilter at b{base}"
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
    fn chunk_constants_are_maximal() {
        for base in 2..=MAX_GPU_DIGIT_MASK_BASE {
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
        let Some(ctx) = try_init_gpu() else {
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
