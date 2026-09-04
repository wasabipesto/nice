//! A simple CLI for the nice library.

#![warn(clippy::all, clippy::pedantic)]
#![allow(clippy::too_many_lines)]

extern crate nice_common;
use nice_common::client_api_async::{
    Client, get_field_from_server_async, get_validation_data_from_server_async,
    submit_field_to_server_async,
};
use nice_common::client_process::{process_range_detailed, process_range_niceonly};
use nice_common::stride_filter;
use nice_common::{
    CLIENT_REQUEST_TIMEOUT_SECS, CLIENT_VERSION, DataToClient, DataToServer, FieldResults,
    FieldSize, SearchMode, UniquesDistributionSimple, ValidationData,
};

// k=3 removes 15-22% of stride candidates versus k=2 at production bases
// (fewer suffixes survive the all-different check on 3+3 fixed low digits),
// and the u32 residue/gap representation keeps the larger table cheap.
const DEFAULT_LSD_K_VALUE: u32 = 3;

/// Defaults for the prefetch buffer.
/// The buffer is sized in seconds rather than fields so that it adapts to how
/// fast this machine actually is: a client taking at least this long per field
/// prefetches a single one, while a client finishing a field in a fraction of
/// this buffers enough to ride out a slow claim.
const DEFAULT_PREFETCH_SECONDS: f64 = 2.0;
const DEFAULT_PREFETCH_MAX: usize = 16;
const DEFAULT_PREFETCH_CONCURRENCY: usize = 4;

/// Cap on result submissions in flight at once.
/// Submissions are small and were measured to cost far less than a claim, but an
/// unbounded queue would quietly absorb a server outage instead of surfacing it.
const MAX_SUBMITS_IN_FLIGHT: usize = 8;

mod bench;

#[cfg(feature = "cuda")]
use nice_common::client_process_cuda::{
    CUDA_BATCH_SIZE, CudaContext, begin_niceonly_cuda, finish_niceonly_cuda,
    process_range_detailed_cuda, process_range_niceonly_cuda,
};
#[cfg(feature = "vulkan")]
use nice_common::client_process_vulkan::{
    VULKAN_BATCH_SIZE, process_range_detailed_vulkan, process_range_niceonly_vulkan,
};
#[cfg(feature = "cubecl")]
use nice_common::cubecl_backend::{
    CUBECL_BATCH_SIZE, CubeclContext, begin_niceonly_cubecl, finish_niceonly_cubecl,
    process_range_detailed_cubecl, process_range_niceonly_cubecl,
};
#[cfg(feature = "vulkan")]
use nice_common::vulkan::VulkanContext;

/// Which GPU backend to drive.
///
/// Every backend `dlopen`s its driver at runtime, so a single binary can carry
/// all of them and require none at build time. `Auto` tries CUDA first,
/// leaving behaviour on an NVIDIA machine exactly as it was.
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
enum GpuBackend {
    /// Fastest measured order for the mode: detailed tries `cubecl-cuda`,
    /// `cubecl`, CUDA, then Vulkan; niceonly tries CUDA, `cubecl`, then
    /// Vulkan. See `init_gpu` for the numbers behind the ordering.
    Auto,
    /// NVIDIA only; requires the CUDA toolkit at runtime for NVRTC.
    Cuda,
    /// Any Vulkan 1.2 device with `shaderInt64` (AMD, Intel, NVIDIA,
    /// llvmpipe). Experimental: only present in builds with the `vulkan`
    /// feature, which the `gpu` umbrella no longer includes.
    Vulkan,
    /// `CubeCL` over wgpu: kernels written in Rust, JIT-specialized per base.
    Cubecl,
    /// `CubeCL` over its native CUDA runtime (needs the `cubecl-cuda`
    /// feature and, like `cuda`, the CUDA toolkit at runtime for NVRTC).
    CubeclCuda,
    /// `CubeCL` over its native HIP runtime (AMD; needs the `cubecl-hip`
    /// feature, which links against `ROCm` at build time). Never chosen by
    /// `auto`.
    CubeclHip,
}

/// An initialized GPU backend.
///
/// In a build with no backend features this enum has no variants, so it is
/// uninhabited and `GpuCtx` below is provably always `None` — the same trick
/// the single-backend version used, extended to two.
///
/// The variants differ a lot in size, which does not matter: a process holds
/// exactly one of these, behind an `Arc`, for its whole run.
#[allow(clippy::large_enum_variant)]
enum GpuHandle {
    #[cfg(feature = "cuda")]
    Cuda(CudaContext),
    #[cfg(feature = "vulkan")]
    Vulkan(VulkanContext),
    #[cfg(feature = "cubecl")]
    Cubecl(CubeclContext),
}

impl GpuHandle {
    /// The name of the device this backend is actually running on, for the
    /// benchmark report and submission telemetry.
    ///
    /// Asking the live handle rather than the CLI flag is what keeps the two
    /// backends apart: on a box with both, `--gpu-backend vulkan` (or a CUDA
    /// init that failed over to Vulkan) would otherwise be reported under the
    /// CUDA device's name. Vulkan recorded its name at init; the CUDA handle
    /// does not carry one, so that arm asks the driver again by device index.
    // Both lints fire only in configurations this function degenerates in: a
    // Vulkan-only build reaches no `None` arm, and a build with no backend at
    // all reaches neither `self` nor `device`.
    #[allow(clippy::unnecessary_wraps, clippy::unused_self)]
    fn device_name(&self, device: usize) -> Option<String> {
        let _ = device; // only the CUDA arm needs it, and it may be absent
        // With no backend feature the enum is uninhabited, but `self` is a
        // reference and references are always considered inhabited, so an
        // empty match does not type-check. Same `#[cfg]` split as
        // `process_field_sync`.
        #[cfg(any(feature = "cuda", feature = "vulkan", feature = "cubecl"))]
        {
            match self {
                #[cfg(feature = "cuda")]
                GpuHandle::Cuda(_) => cudarc::driver::CudaContext::new(device)
                    .ok()
                    .and_then(|d| d.name().ok()),
                #[cfg(feature = "vulkan")]
                GpuHandle::Vulkan(ctx) => Some(ctx.device_name.clone()),
                #[cfg(feature = "cubecl")]
                GpuHandle::Cubecl(ctx) => Some(ctx.device_name()),
            }
        }
        #[cfg(not(any(feature = "cuda", feature = "vulkan", feature = "cubecl")))]
        {
            None
        }
    }

    /// The backend actually processing fields, as its `--gpu-backend` value —
    /// asked of the live handle for the same reason as `device_name`: with
    /// `auto` and multiple compiled backends, the CLI flag does not say which
    /// one won. Three backends at very different throughputs can share one
    /// device name, so telemetry without this cannot be compared.
    // Same degenerate-configuration lints as `device_name`.
    #[allow(clippy::unnecessary_wraps, clippy::unused_self)]
    fn backend_name(&self) -> Option<&'static str> {
        #[cfg(any(feature = "cuda", feature = "vulkan", feature = "cubecl"))]
        {
            match self {
                #[cfg(feature = "cuda")]
                GpuHandle::Cuda(_) => Some("cuda"),
                #[cfg(feature = "vulkan")]
                GpuHandle::Vulkan(_) => Some("vulkan"),
                #[cfg(feature = "cubecl")]
                GpuHandle::Cubecl(ctx) => Some(ctx.backend_name()),
            }
        }
        #[cfg(not(any(feature = "cuda", feature = "vulkan", feature = "cubecl")))]
        {
            None
        }
    }
}

/// The GPU context, if this build has one and the user asked for it.
///
/// Naming the type in every build means everything that only passes the context
/// along can do so without a `#[cfg]`, leaving the split to
/// `process_field_sync` — the one place that actually looks inside it.
type GpuCtx = Option<Arc<GpuHandle>>;

extern crate serde_json;
use anyhow::{Result, anyhow};
use clap::builder::FalseyValueParser;
use clap::parser::ValueSource;
use clap::{ArgMatches, CommandFactory, FromArgMatches, Parser, ValueEnum};
use env_logger::Env;
use log::{LevelFilter, debug, error, info, warn};
use rayon::prelude::*;
use simple_tqdm::ParTqdm;
use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::task::JoinSet;

#[derive(Debug, Clone, Copy, ValueEnum)]
enum LogLevel {
    Off,
    Error,
    Warn,
    Info,
    Debug,
    Trace,
}

impl From<LogLevel> for LevelFilter {
    fn from(level: LogLevel) -> Self {
        match level {
            LogLevel::Off => LevelFilter::Off,
            LogLevel::Error => LevelFilter::Error,
            LogLevel::Warn => LevelFilter::Warn,
            LogLevel::Info => LevelFilter::Info,
            LogLevel::Debug => LevelFilter::Debug,
            LogLevel::Trace => LevelFilter::Trace,
        }
    }
}

#[derive(Parser, Debug, Clone)]
#[command(author, version, about, long_about = None)]
#[command(propagate_version = true)]
#[allow(clippy::struct_excessive_bools)]
pub struct Cli {
    /// The checkout mode to use
    #[arg(value_enum, default_value = "detailed", env = "NICE_MODE")]
    mode: SearchMode,

    /// The base API URL to connect to
    #[arg(
        long,
        default_value = "https://api.nicenumbers.net",
        env = "NICE_API_BASE"
    )]
    api_base: String,

    /// If an API call encounters a retryable error, retry with exponential backoff this many times
    #[arg(long, default_value_t = 10, env = "NICE_API_MAX_RETRIES")]
    api_max_retries: u32,

    /// The username to send alongside your contribution
    #[arg(short, long, default_value = "anonymous", env = "NICE_USERNAME")]
    username: String,

    /// Run indefinitely with the current settings
    #[arg(short, long, env = "NICE_REPEAT", value_parser = FalseyValueParser::new())]
    repeat: bool,

    /// Hide the progress bar
    #[arg(short, long, env = "NICE_NO_PROGRESS", value_parser = FalseyValueParser::new())]
    no_progress: bool,

    /// Run parallel with this many threads
    #[arg(short, long, default_value_t = 4, env = "NICE_THREADS")]
    threads: usize,

    /// Keep roughly this many seconds of work claimed ahead of the processor.
    /// Set to 0 to force the old single-field prefetch.
    #[arg(long, default_value_t = DEFAULT_PREFETCH_SECONDS, env = "NICE_PREFETCH_SECONDS")]
    prefetch_seconds: f64,

    /// Never hold more than this many claimed fields at once.
    #[arg(long, default_value_t = DEFAULT_PREFETCH_MAX, env = "NICE_PREFETCH_MAX")]
    prefetch_max: usize,

    /// Allow this many claim requests to be in flight at once.
    #[arg(long, default_value_t = DEFAULT_PREFETCH_CONCURRENCY, env = "NICE_PREFETCH_CONCURRENCY")]
    prefetch_concurrency: usize,

    /// Run an offline benchmark sweep and print a detailed report.
    /// Implied by the other --benchmark-* options.
    #[arg(short, long, env = "NICE_BENCHMARK", value_parser = FalseyValueParser::new())]
    benchmark: bool,

    /// Approximate time budget for the benchmark sweep, in seconds
    #[arg(long, default_value_t = 10.0, env = "NICE_BENCHMARK_SECS")]
    benchmark_secs: f64,

    /// Upload benchmark results without prompting
    #[arg(long, env = "NICE_BENCHMARK_UPLOAD", value_parser = FalseyValueParser::new())]
    benchmark_upload: bool,

    /// Print the benchmark report as machine-readable JSON instead of the
    /// table; everything else (progress, upload chatter) moves to stderr so
    /// stdout is exactly one JSON document
    #[arg(long, env = "NICE_BENCHMARK_JSON", value_parser = FalseyValueParser::new())]
    benchmark_json: bool,

    /// Attach hardware/config telemetry to each submission
    #[arg(long, env = "NICE_TELEMETRY", value_parser = FalseyValueParser::new())]
    telemetry: bool,

    /// Validate results against the server before submitting
    #[arg(long, env = "NICE_VALIDATE", value_parser = FalseyValueParser::new())]
    validate: bool,

    /// Use GPU acceleration (requires a build with the gpu feature).
    /// Implied by the other --gpu-* options.
    #[arg(long, env = "NICE_GPU", value_parser = FalseyValueParser::new())]
    gpu: bool,

    /// GPU device to use (0 for first GPU, 1 for second, etc.)
    #[arg(long, default_value_t = 0, env = "NICE_GPU_DEVICE")]
    gpu_device: usize,

    /// Which GPU backend to use with --gpu
    #[arg(long, value_enum, default_value_t = GpuBackend::Auto, env = "NICE_GPU_BACKEND")]
    gpu_backend: GpuBackend,

    /// Which wgpu adapter the `CubeCL` backend uses, in `CubeCL`'s device
    /// spelling: `DiscreteGpu(0)`, `IntegratedGpu(1)`, `Cpu`, ... Unset picks
    /// the best adapter. This exists because --gpu-device indexes a
    /// per-backend namespace (CUDA ordinals != Vulkan ordinals != wgpu
    /// adapters), so on a multi-GPU box no single number is right for every
    /// backend; the chosen adapter and its graphics API are always logged.
    #[arg(long, env = "NICE_GPU_WGPU_DEVICE")]
    gpu_wgpu_device: Option<String>,

    /// Set the log level (overrides `RUST_LOG` environment variable)
    #[arg(short, long, value_enum, env = "NICE_LOG_LEVEL")]
    log_level: Option<LogLevel>,
}

/// Which backends this build carries, for error messages.
fn compiled_backends() -> String {
    let mut have = Vec::new();
    if cfg!(feature = "cuda") {
        have.push("cuda");
    }
    if cfg!(feature = "vulkan") {
        have.push("vulkan");
    }
    if cfg!(feature = "cubecl") {
        have.push("cubecl");
    }
    if cfg!(feature = "cubecl-cuda") {
        have.push("cubecl-cuda");
    }
    if cfg!(feature = "cubecl-hip") {
        have.push("cubecl-hip");
    }
    if have.is_empty() {
        "none".to_string()
    } else {
        have.join(", ")
    }
}

/// Try to bring up CUDA, turning `cudarc`'s panics into errors.
///
/// `cudarc` panics rather than returning when the CUDA shared library is absent
/// (`panic_no_lib_found`, cudarc/src/lib.rs) — which is the ordinary case on any
/// machine with no NVIDIA driver, i.e. precisely the case `--gpu-backend auto`
/// exists to fall through. Upstream catches the same panic for the same reason
/// in `nvrtc_compiles_kernels_for_all_supported_bases`. See `guarded_init` for
/// how the panic is contained and what that relies on.
#[cfg(feature = "cuda")]
fn try_init_cuda(device: usize) -> Result<CudaContext> {
    guarded_init("the CUDA driver library could not be loaded", || {
        CudaContext::new(device)
    })
}

/// Run a GPU backend initializer that may panic instead of returning an error.
///
/// Several of them do. cudarc panics when it cannot dlopen libcuda. cubecl-wgpu
/// panics when no adapter serves the requested backend, which is ordinary
/// inside a CUDA container with no Vulkan ICD. cubecl's CUDA runtime brings its
/// own worker thread down and then unwraps the resulting `RecvError` on ours,
/// which is what an NVIDIA-less machine sees. Any of those unwinding out of the
/// backend chain kills the process, so `auto` never reaches the backend that
/// would have worked — the whole point of having a chain.
///
/// The panic hook is silenced for the duration so a routine fallback does not
/// print somebody else's twenty-line message. GPU init runs before any of our
/// worker threads start, so swapping the global hook here is safe.
///
/// Note this relies on unwinding; under `panic = "abort"` the process would die
/// as before. The workspace does not set that.
#[cfg(any(feature = "cuda", feature = "cubecl"))]
fn guarded_init<T>(
    unknown_cause: &str,
    init: impl FnOnce() -> Result<T> + std::panic::UnwindSafe,
) -> Result<T> {
    let previous = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {}));
    let caught = std::panic::catch_unwind(init);
    std::panic::set_hook(previous);

    match caught {
        Ok(result) => result,
        Err(payload) => {
            let msg = payload
                .downcast_ref::<String>()
                .map(String::as_str)
                .or_else(|| payload.downcast_ref::<&str>().copied())
                .unwrap_or(unknown_cause);
            Err(anyhow!("{msg}"))
        }
    }
}

/// Bring up the GPU backend the user asked for.
///
/// `Auto`'s order is per mode, from the measured tables in the `CubeCL`
/// evaluation (RTX 4060, RX 9070 XT, Apple M4, plus review runs on Intel
/// iGPU and an RTX A1000):
///
/// - **Detailed**: `cubecl-cuda` → `cubecl` → `cuda` → `vulkan`. The
///   `CubeCL` family wins detailed on every vendor tested (1.03-1.07x over hand-CUDA
///   on NVIDIA, 1.2-2.6x over hand-Vulkan elsewhere), and a failed
///   `cubecl-cuda` init (no CUDA toolkit) falls through to `cubecl`, which
///   needs only a driver — so NVIDIA-without-toolkit gets wgpu speed instead
///   of the hand-WGSL fallback.
/// - **Niceonly**: `cuda` → `cubecl` → `vulkan`. Hand-CUDA keeps a slim edge
///   at the b50+ bases where long-run wall time concentrates; everywhere
///   without a toolkit, `CubeCL` is the best available (wins RADV b50+ and
///   NVIDIA-over-wgpu outright, runs out of the box on Apple).
///
/// An **explicitly named** backend that fails to initialize is fatal rather
/// than falling back: for a distributed compute client, silently dropping to
/// a much slower path is a worse outcome than stopping and saying so.
#[allow(unused_variables)]
fn init_gpu(cli: &Cli) -> GpuCtx {
    if !cli.gpu {
        return None;
    }
    let want = cli.gpu_backend;
    let detailed = cli.mode == SearchMode::Detailed;

    // Mesa prints "<driver> is not a conformant Vulkan implementation" straight
    // to stderr — driver chatter our log filter can't reach. Silence it through
    // Mesa's own knob at normal verbosity; an explicit user setting (either
    // way) or debug logging leaves it alone.
    // Safety: init_gpu runs before any worker thread starts (same argument as
    // the CUBECL_WGPU_DEFAULT_DEVICE set below).
    if !log::log_enabled!(log::Level::Debug)
        && std::env::var_os("MESA_VK_IGNORE_CONFORMANCE_WARNING").is_none()
    {
        unsafe { std::env::set_var("MESA_VK_IGNORE_CONFORMANCE_WARNING", "true") };
    }

    // The CubeCL wgpu runtime reads its device selection from this env var
    // (typed adapter namespaces; --gpu-device's flat ordinal cannot name
    // them). Translating our flag here keeps one parser — CubeCL's own.
    #[cfg(feature = "cubecl")]
    if let Some(spec) = &cli.gpu_wgpu_device {
        // Safety: init_gpu runs before any worker thread starts (same
        // argument as try_init_cuda's panic-hook swap).
        unsafe { std::env::set_var("CUBECL_WGPU_DEFAULT_DEVICE", spec) };
        info!("CubeCL wgpu adapter pinned to {spec}");
    }

    // Detailed auto leads with the CubeCL CUDA runtime; a clean init failure
    // (validated by its smoke kernel) falls through to the wgpu runtime.
    #[cfg(feature = "cubecl-cuda")]
    if want == GpuBackend::CubeclCuda || (want == GpuBackend::Auto && detailed) {
        let attempt = guarded_init("the CubeCL CUDA runtime could not be started", || {
            CubeclContext::new_cuda(cli.gpu_device)
        });
        match attempt {
            Ok(ctx) => {
                info!(
                    "GPU initialized: CubeCL CUDA device {}, batch size {}",
                    cli.gpu_device, CUBECL_BATCH_SIZE
                );
                return Some(Arc::new(GpuHandle::Cubecl(ctx)));
            }
            Err(e) if want == GpuBackend::CubeclCuda => {
                error!("Failed to initialize CubeCL CUDA runtime: {e:?}");
                std::process::exit(1);
            }
            Err(e) => {
                info!("CubeCL CUDA unavailable; trying the next backend");
                debug!("  CubeCL CUDA init failed: {e:#}");
            }
        }
    }

    #[cfg(feature = "cuda")]
    if want == GpuBackend::Cuda || (want == GpuBackend::Auto && !detailed) {
        match try_init_cuda(cli.gpu_device) {
            Ok(ctx) => {
                info!(
                    "GPU initialized: CUDA device {}, batch size {}",
                    cli.gpu_device, CUDA_BATCH_SIZE
                );
                if let Ok(device) = cudarc::driver::CudaContext::new(cli.gpu_device)
                    && let Ok(name) = device.name()
                {
                    info!("  GPU: {name}");
                }
                return Some(Arc::new(GpuHandle::Cuda(ctx)));
            }
            Err(e) => {
                if want == GpuBackend::Cuda {
                    error!(
                        "Failed to initialize CUDA on device {}: {e:#}",
                        cli.gpu_device
                    );
                    eprintln!("Troubleshooting:");
                    eprintln!("1. Ensure NVIDIA GPU drivers are installed");
                    eprintln!("2. Verify CUDA toolkit is installed (nvcc --version)");
                    eprintln!("3. Check that GPU {} exists (nvidia-smi)", cli.gpu_device);
                    eprintln!("4. Try a different device with --gpu-device <N>");
                    if cfg!(feature = "vulkan") {
                        eprintln!("5. Or use the Vulkan backend: --gpu-backend vulkan");
                    }
                    std::process::exit(1);
                }
                info!("CUDA unavailable; trying the next backend");
                debug!("  CUDA init failed: {e:#}");
            }
        }
    }

    // Explicit only: the HIP runtime is an evaluation arm, not part of auto.
    #[cfg(feature = "cubecl-hip")]
    if want == GpuBackend::CubeclHip {
        let attempt = guarded_init("the CubeCL HIP runtime could not be started", || {
            CubeclContext::new_hip(cli.gpu_device)
        });
        match attempt {
            Ok(ctx) => {
                info!(
                    "GPU initialized: CubeCL HIP device {}, batch size {}",
                    cli.gpu_device, CUBECL_BATCH_SIZE
                );
                return Some(Arc::new(GpuHandle::Cubecl(ctx)));
            }
            Err(e) => {
                error!("Failed to initialize CubeCL HIP runtime: {e:?}");
                std::process::exit(1);
            }
        }
    }

    #[cfg(feature = "cubecl")]
    if matches!(want, GpuBackend::Auto | GpuBackend::Cubecl) {
        let attempt = guarded_init(
            "no adapter available for the requested wgpu backend",
            CubeclContext::new_default,
        );
        match attempt {
            Ok(ctx) => {
                info!(
                    "GPU initialized: CubeCL wgpu device ({}), batch size {}",
                    ctx.device_name(),
                    CUBECL_BATCH_SIZE
                );
                return Some(Arc::new(GpuHandle::Cubecl(ctx)));
            }
            Err(e) if want == GpuBackend::Cubecl => {
                error!("Failed to initialize CubeCL wgpu runtime: {e:?}");
                std::process::exit(1);
            }
            Err(e) => {
                info!("CubeCL unavailable; trying the next backend");
                debug!("  CubeCL init failed: {e:#}");
            }
        }
    }

    // Detailed auto's third stop: hand-CUDA, for the corner where both
    // CubeCL runtimes failed but NVRTC works (e.g. a CUDA container with no
    // Vulkan ICD and a CubeCL regression).
    #[cfg(feature = "cuda")]
    if want == GpuBackend::Auto && detailed {
        if let Ok(ctx) = try_init_cuda(cli.gpu_device) {
            info!(
                "GPU initialized: CUDA device {}, batch size {}",
                cli.gpu_device, CUDA_BATCH_SIZE
            );
            return Some(Arc::new(GpuHandle::Cuda(ctx)));
        }
        info!("CUDA unavailable; trying Vulkan");
    }

    #[cfg(feature = "vulkan")]
    if matches!(want, GpuBackend::Auto | GpuBackend::Vulkan) {
        match VulkanContext::new(cli.gpu_device) {
            Ok(ctx) => {
                info!(
                    "GPU initialized: Vulkan device {} ({}), batch size {}",
                    cli.gpu_device, ctx.device_name, VULKAN_BATCH_SIZE
                );
                return Some(Arc::new(GpuHandle::Vulkan(ctx)));
            }
            Err(e) => {
                error!(
                    "Failed to initialize Vulkan on device {}: {e:?}",
                    cli.gpu_device
                );
                eprintln!("Troubleshooting:");
                eprintln!("1. Ensure a Vulkan driver is installed (try vulkaninfo)");
                eprintln!("2. The device must support shaderInt64");
                eprintln!("3. Try a different device with --gpu-device <N>");
                std::process::exit(1);
            }
        }
    }

    error!(
        "No usable GPU backend for --gpu-backend {want:?}; this build has: {}",
        compiled_backends()
    );
    std::process::exit(1);
}

/// Process a field synchronously (`CPU` or `GPU`).
/// This is wrapped in `spawn_blocking` when called from async context.
///
/// `stride_table` lets a caller reuse a prebuilt table across many calls for
/// the same base (the benchmark sweep times many small windows, where a
/// per-call table build would dominate); `None` builds one for this field,
/// which is negligible at production field sizes.
fn process_field_sync(
    claim_data: &DataToClient,
    cli: &Cli,
    gpu: &GpuCtx,
    stride_table: Option<&Arc<stride_filter::StrideTable>>,
) -> Vec<FieldResults> {
    let mode = cli.mode;
    if cli.gpu {
        // GPU processing path
        #[cfg(any(feature = "cuda", feature = "vulkan", feature = "cubecl"))]
        {
            let handle = gpu.as_ref().expect("GPU context failed to initialize");
            let range: FieldSize = claim_data.into();

            let gpu_results = match &**handle {
                #[cfg(feature = "cuda")]
                GpuHandle::Cuda(ctx) => match mode {
                    SearchMode::Detailed => {
                        process_range_detailed_cuda(ctx, &range, claim_data.base)
                    }
                    SearchMode::Niceonly => {
                        process_range_niceonly_cuda(ctx, &range, claim_data.base)
                    }
                },
                #[cfg(feature = "vulkan")]
                GpuHandle::Vulkan(ctx) => match mode {
                    SearchMode::Detailed => {
                        process_range_detailed_vulkan(ctx, &range, claim_data.base)
                    }
                    SearchMode::Niceonly => {
                        process_range_niceonly_vulkan(ctx, &range, claim_data.base)
                    }
                },
                #[cfg(feature = "cubecl")]
                GpuHandle::Cubecl(ctx) => match mode {
                    SearchMode::Detailed => {
                        process_range_detailed_cubecl(ctx, &range, claim_data.base)
                    }
                    SearchMode::Niceonly => {
                        process_range_niceonly_cubecl(ctx, &range, claim_data.base)
                    }
                },
            };

            match gpu_results {
                Ok(result) => vec![result],
                Err(e) => {
                    error!("GPU processing error: {e:?}");
                    std::process::exit(1);
                }
            }
        }
        #[cfg(not(any(feature = "cuda", feature = "vulkan", feature = "cubecl")))]
        {
            let _ = gpu; // there is no context to look at in this build
            error!("GPU support not compiled in");
            std::process::exit(1);
        }
    } else {
        // CPU processing path
        let range: FieldSize = claim_data.into();

        // Scale the processing chunk size with the field size
        let chunk_default_size: u128 = 1_000_000;
        let target_max_chunks: u128 = 100_000;

        let chunk_multiple = range
            .size()
            .div_ceil(chunk_default_size * target_max_chunks)
            .clamp(1, 1_000);
        let chunk_size = chunk_default_size * chunk_multiple;

        let chunks = range.chunks(chunk_size);

        // Precompute stride table once for Niceonly mode to avoid redundant computation
        // in each chunk. The table is wrapped in Arc for thread-safe sharing across
        // parallel chunk processing.
        let stride_table_opt = if mode == SearchMode::Niceonly {
            Some(match stride_table {
                Some(table) => Arc::clone(table),
                None => Arc::new(stride_filter::StrideTable::new(
                    claim_data.base,
                    DEFAULT_LSD_K_VALUE,
                )),
            })
        } else {
            None
        };

        // Configure TQDM
        #[allow(
            clippy::cast_precision_loss,
            clippy::cast_sign_loss,
            clippy::cast_possible_truncation
        )]
        let chunk_scale = (chunk_size as f32).log10() as u32;
        let tqdm_config = simple_tqdm::Config::new()
            .with_unit(format!("e{chunk_scale}"))
            .with_disable(cli.no_progress);

        // Process each chunk and gather the results
        chunks
            .par_iter()
            .tqdm_config(tqdm_config)
            .map(|chunk| match mode {
                SearchMode::Detailed => process_range_detailed(chunk, claim_data.base),
                SearchMode::Niceonly => {
                    let stride_table = stride_table_opt
                        .as_ref()
                        .expect("Stride table not initialized");
                    process_range_niceonly(chunk, claim_data.base, stride_table)
                }
            })
            .collect()
    }
}

/// A field that has been started: either finished on the spot, or queued in
/// a GPU pipeline that returns fields in the order they were begun.
enum FieldStage {
    Done(Vec<FieldResults>),
    // Only the GPU niceonly path queues; a CPU-only build never constructs it.
    #[cfg_attr(
        not(any(feature = "cuda", feature = "vulkan", feature = "cubecl")),
        allow(dead_code)
    )]
    Queued,
}

/// Start a field. GPU niceonly fields go into the backend's continuous
/// pipeline and return `Queued`; everything else is processed here and now.
///
/// Splitting start from finish is what lets the caller keep one field ahead
/// of the device: the next field's MSD filtering runs while the device drains
/// the current one. See `nice_common::gpu_niceonly::NiceonlyPipeline`.
fn begin_field_sync(claim_data: &DataToClient, cli: &Cli, gpu: &GpuCtx) -> FieldStage {
    #[cfg(any(feature = "cuda", feature = "vulkan", feature = "cubecl"))]
    if cli.gpu && cli.mode == SearchMode::Niceonly {
        use nice_common::gpu_niceonly::NiceonlyStarted;
        let handle = gpu.as_ref().expect("GPU context failed to initialize");
        let range: FieldSize = claim_data.into();
        let started = match &**handle {
            #[cfg(feature = "cuda")]
            GpuHandle::Cuda(ctx) => begin_niceonly_cuda(ctx, &range, claim_data.base),
            #[cfg(feature = "cubecl")]
            GpuHandle::Cubecl(ctx) => begin_niceonly_cubecl(ctx, &range, claim_data.base),
            // The Vulkan sink cannot leave the calling thread, so it has no
            // pipeline: the field is done by the time this returns.
            #[cfg(feature = "vulkan")]
            GpuHandle::Vulkan(ctx) => process_range_niceonly_vulkan(ctx, &range, claim_data.base)
                .map(NiceonlyStarted::Immediate),
        };
        return match started {
            Ok(NiceonlyStarted::Immediate(results)) => FieldStage::Done(vec![results]),
            Ok(NiceonlyStarted::Queued) => FieldStage::Queued,
            Err(e) => {
                error!("GPU processing error: {e:?}");
                std::process::exit(1);
            }
        };
    }
    FieldStage::Done(process_field_sync(claim_data, cli, gpu, None))
}

/// Per-field pipeline telemetry from the GPU niceonly path, already as the
/// JSON object that goes into the submission; `None` on every other path.
type PipelineTelemetry = Option<serde_json::Value>;

/// Finish a field begun with `begin_field_sync`: for a queued one, wait for
/// the GPU pipeline's next result (fields come back in order).
fn finish_field_sync(
    stage: FieldStage,
    cli: &Cli,
    gpu: &GpuCtx,
) -> (Vec<FieldResults>, PipelineTelemetry) {
    match stage {
        FieldStage::Done(results) => (results, None),
        FieldStage::Queued => {
            #[cfg(any(feature = "cuda", feature = "vulkan", feature = "cubecl"))]
            {
                let _ = cli;
                let handle = gpu.as_ref().expect("GPU context failed to initialize");
                let finished = match &**handle {
                    #[cfg(feature = "cuda")]
                    GpuHandle::Cuda(ctx) => finish_niceonly_cuda(ctx),
                    #[cfg(feature = "cubecl")]
                    GpuHandle::Cubecl(ctx) => finish_niceonly_cubecl(ctx),
                    #[cfg(feature = "vulkan")]
                    GpuHandle::Vulkan(_) => unreachable!("Vulkan fields are never queued"),
                };
                match finished {
                    Ok((results, stats)) => (vec![results], Some(stats.telemetry_json())),
                    Err(e) => {
                        error!("GPU processing error: {e:?}");
                        std::process::exit(1);
                    }
                }
            }
            #[cfg(not(any(feature = "cuda", feature = "vulkan", feature = "cubecl")))]
            {
                let _ = (cli, gpu);
                unreachable!("no GPU backend compiled in, so nothing is ever queued")
            }
        }
    }
}

/// Start one field on the blocking pool. Returns the claim back alongside
/// its stage, since the blocking task has to take ownership of it.
async fn begin_field(
    claim_data: DataToClient,
    cli: &Arc<Cli>,
    gpu: &GpuCtx,
) -> (DataToClient, FieldStage) {
    let cli = Arc::clone(cli);
    let gpu = gpu.clone();
    tokio::task::spawn_blocking(move || {
        let stage = begin_field_sync(&claim_data, &cli, &gpu);
        (claim_data, stage)
    })
    .await
    .expect("Processing task panicked")
}

/// Finish one field on the blocking pool.
async fn finish_field(
    stage: FieldStage,
    cli: &Arc<Cli>,
    gpu: &GpuCtx,
) -> (Vec<FieldResults>, PipelineTelemetry) {
    let cli = Arc::clone(cli);
    let gpu = gpu.clone();
    tokio::task::spawn_blocking(move || finish_field_sync(stage, &cli, &gpu))
        .await
        .expect("Processing task panicked")
}

/// Start and finish one field on the blocking pool: the field-serial form.
///
/// Returns the claim back alongside its results, since the blocking task has to
/// take ownership of it, plus how long the processing took.
async fn process_field(
    claim_data: DataToClient,
    cli: &Arc<Cli>,
    gpu: &GpuCtx,
) -> (DataToClient, Vec<FieldResults>, Duration) {
    let start_time = Instant::now();
    let (claim_data, stage) = begin_field(claim_data, cli, gpu).await;
    let (results, _) = finish_field(stage, cli, gpu).await;
    (claim_data, results, start_time.elapsed())
}

/// How many fields to keep started ahead of the one being finished. Only the
/// GPU niceonly pipeline overlaps fields; everything else finishes a field
/// in `begin`, so lookahead would just hold claims.
fn field_lookahead(cli: &Cli) -> usize {
    #[cfg(any(feature = "cuda", feature = "vulkan", feature = "cubecl"))]
    {
        if cli.gpu && cli.mode == SearchMode::Niceonly && cli.repeat {
            return nice_common::gpu_niceonly::fields_in_flight().saturating_sub(1);
        }
    }
    let _ = cli;
    0
}

/// A field started but not yet finished and submitted.
struct InFlightField {
    claim: DataToClient,
    stage: FieldStage,
    begun_at: Instant,
}

/// Report the processing rate, for the runs where the progress bar isn't
/// already showing it.
#[allow(clippy::cast_precision_loss)]
fn log_field_rate(claim_data: &DataToClient, elapsed: Duration, cli: &Cli) {
    if !cli.no_progress && !cli.gpu {
        return;
    }
    let range_size = claim_data.range_size as f64;
    let seconds = elapsed.as_secs_f64();
    info!(
        "✓ Processed {range_size:.2e} numbers in {seconds:.2}s ({:.2e} numbers/sec)",
        range_size / seconds
    );
}

/// Send one field's results to the server and log whatever it says back.
async fn submit_results(
    client: &Client,
    api_base: &str,
    submit_data: DataToServer,
    max_retries: u32,
) -> Result<()> {
    let response = submit_field_to_server_async(client, api_base, submit_data, max_retries).await?;
    match response.text().await {
        Ok(msg) => debug!("Server response: {msg}"),
        Err(e) => error!("Server returned success but an error occurred: {e}"),
    }
    Ok(())
}

/// Compile results from multiple chunks into a single `DataToServer`.
fn compile_results(
    results: Vec<FieldResults>,
    claim_data: &DataToClient,
    username: &str,
    mode: SearchMode,
    telemetry: Option<serde_json::Value>,
) -> DataToServer {
    // Take the pieces out of the results rather than cloning them back out.
    // Niceonly runs always report an empty distribution, so summing it costs
    // nothing there even though the total is discarded below.
    let mut nice_numbers = Vec::new();
    let mut dist_map: HashMap<u32, u128> = HashMap::new();
    for result in results {
        nice_numbers.extend(result.nice_numbers);
        for dist in result.distribution {
            *dist_map.entry(dist.num_uniques).or_insert(0) += dist.count;
        }
    }

    let unique_distribution = if mode == SearchMode::Niceonly {
        None
    } else {
        // Convert the counts back into a formatted, sorted list
        let mut distribution: Vec<UniquesDistributionSimple> = dist_map
            .into_iter()
            .map(|(num_uniques, count)| UniquesDistributionSimple { num_uniques, count })
            .collect();
        distribution.sort_by_key(|d| d.num_uniques);
        Some(distribution)
    };

    let submit_data = DataToServer {
        claim_id: claim_data.claim_id,
        username: username.to_string(),
        client_version: CLIENT_VERSION.to_string(),
        unique_distribution,
        nice_numbers,
        telemetry,
    };
    debug!(
        "Submit Data: {}",
        serde_json::to_string(&submit_data).unwrap()
    );
    submit_data
}

/// Validate results against expected `ValidationData`.
fn validate_results(
    submit_data: &DataToServer,
    validation_data: &ValidationData,
    mode: SearchMode,
) -> bool {
    let mut validation_passed = true;

    // Compare nice numbers
    let mut our_numbers = submit_data.nice_numbers.clone();
    let mut server_numbers = validation_data.nice_numbers.clone();
    our_numbers.sort_by_key(|n| n.number);
    server_numbers.sort_by_key(|n| n.number);

    if our_numbers != server_numbers {
        error!("VALIDATION FAILED: Semi-nice numbers don't match!");
        validation_passed = false;
    }

    // Compare distribution (only for detailed mode)
    if mode == SearchMode::Detailed
        && let Some(ref our_dist) = submit_data.unique_distribution
    {
        let mut our_dist_sorted = our_dist.clone();
        let mut server_dist_sorted = validation_data.unique_distribution.clone();
        our_dist_sorted.sort_by_key(|d| d.num_uniques);
        server_dist_sorted.sort_by_key(|d| d.num_uniques);

        if our_dist_sorted != server_dist_sorted {
            error!("VALIDATION FAILED: Distribution doesn't match!");
            validation_passed = false;
        }
    }

    validation_passed
}

/// Check this client against a field the server has already accepted.
///
/// Nothing is claimed and nothing is submitted, so there is no pipeline here;
/// each round is one field, start to finish.
async fn run_validation(cli: &Arc<Cli>, client: &Client, gpu: &GpuCtx) -> Result<()> {
    loop {
        let validation_data =
            get_validation_data_from_server_async(client, &cli.api_base, cli.api_max_retries)
                .await?;
        let claim_data = DataToClient {
            claim_id: 0,
            base: validation_data.base,
            range_start: validation_data.range_start,
            range_end: validation_data.range_end,
            range_size: validation_data.range_size,
        };

        info!("Beginning validation: {}", validation_data.field_id);
        debug!(
            "Claim Data: {}",
            serde_json::to_string(&claim_data).unwrap()
        );

        let (claim_data, results, elapsed) = process_field(claim_data, cli, gpu).await;
        log_field_rate(&claim_data, elapsed, cli);

        let submit_data = compile_results(results, &claim_data, &cli.username, cli.mode, None);

        if validate_results(&submit_data, &validation_data, cli.mode) {
            println!();
            println!("Validation passed! Results match the canonical submission.");
        } else {
            println!();
            println!("Validation failed! Results do not match the canonical submission.");
            println!("  Our submission data: {submit_data:?}");
            println!("  Canonical submission: {validation_data:?}");
            std::process::exit(1);
        }

        if !cli.repeat {
            break;
        }
    }
    Ok(())
}

/// How many claims to keep on hand — buffered or in flight — counting the one
/// about to be processed.
///
/// `field_process_seconds` is how long this machine takes to process one field,
/// smoothed over recent fields; `None` before the first one completes.
///
/// Sized in seconds of work rather than in fields so that it adapts to the
/// machine. Any client averaging at least `prefetch_seconds` per field gets 2,
/// which is exactly the historical behavior of one field processing and one
/// prefetched; only a client faster than that buffers more, and then only enough
/// to ride out a slow claim without stalling the processor.
fn prefetch_target(
    prefetch_seconds: f64,
    prefetch_max: usize,
    field_process_seconds: Option<f64>,
) -> usize {
    let floor = 2;
    let ceiling = prefetch_max.max(floor);
    if prefetch_seconds <= 0.0 {
        return floor;
    }
    match field_process_seconds {
        // Nothing processed yet, so we have no idea how fast this box is.
        None => floor,
        Some(per_field) if per_field <= 0.0 => ceiling,
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        Some(per_field) => {
            // `as usize` saturates rather than wrapping, so a tiny time is safe.
            let fields = (prefetch_seconds / per_field).ceil() as usize;
            fields.saturating_add(1).clamp(floor, ceiling)
        }
    }
}

/// Move every claim that has already arrived out of the join set and into the
/// buffer. Non-blocking.
fn collect_ready_claims(
    fetches: &mut JoinSet<Result<DataToClient>>,
    buffer: &mut VecDeque<DataToClient>,
) -> Result<()> {
    while let Some(joined) = fetches.try_join_next() {
        buffer.push_back(joined.expect("Fetch task panicked")?);
    }
    Ok(())
}

/// Top the claim requests back up to the target, subject to the concurrency cap.
fn request_claims(
    fetches: &mut JoinSet<Result<DataToClient>>,
    buffered: usize,
    target: usize,
    cli: &Cli,
    client: &Client,
) {
    let concurrency = cli.prefetch_concurrency.max(1);
    while buffered + fetches.len() < target && fetches.len() < concurrency {
        let mode = cli.mode;
        let api_base = cli.api_base.clone();
        let api_num_retries = cli.api_max_retries;
        let client = client.clone();
        fetches.spawn(async move {
            get_field_from_server_async(&client, &mode, &api_base, api_num_retries).await
        });
    }
}

/// Drive the claim/process/submit pipeline until it stops or fails.
///
/// The processor is fed from a buffer of claims that is refilled by several
/// concurrent requests.
///
/// The buffer and the two join sets are borrowed rather than owned so that an
/// early return from here still leaves the caller holding them: dropping a
/// `JoinSet` aborts its tasks, which on the submit side would throw away results
/// that have already been computed.
async fn run_pipelined_fields(
    cli: &Arc<Cli>,
    client: &Client,
    gpu: &GpuCtx,
    buffer: &mut VecDeque<DataToClient>,
    fetches: &mut JoinSet<Result<DataToClient>>,
    submits: &mut JoinSet<Result<()>>,
    in_flight: &mut VecDeque<InFlightField>,
) -> Result<()> {
    // Exponentially weighted moving average of how long one field takes to
    // process, this is what the target buffer depth is derived from.
    let mut field_process_ewma: Option<f64> = None;

    // Hardware/config context is constant for the process; collect it once
    // and stamp each submission with it plus the per-field timing.
    let telemetry_base = cli.telemetry.then(|| bench::telemetry_base(cli, gpu));

    // Fields started ahead of the one being finished; see `field_lookahead`.
    let lookahead = field_lookahead(cli);
    // When the previous field finished: with fields overlapping, a field's
    // own start-to-finish span double-counts time shared with its neighbour,
    // so its rate is measured as the time since the previous result instead.
    let mut last_finished: Option<Instant> = None;

    loop {
        // Without --repeat there is exactly one field to do, so claim exactly
        // one: anything extra would be claimed, never processed, and left for
        // the server to time out.
        let target = if cli.repeat {
            prefetch_target(cli.prefetch_seconds, cli.prefetch_max, field_process_ewma)
        } else {
            1
        };

        // Take delivery of anything that landed while we were processing, then
        // put the requests back up to depth.
        collect_ready_claims(fetches, buffer)?;
        request_claims(fetches, buffer.len(), target, cli, client);

        // With an empty buffer there is nothing to do but wait for a claim.
        while buffer.is_empty() {
            let Some(joined) = fetches.join_next().await else {
                return Err(anyhow!("no claims buffered and no requests in flight"));
            };
            buffer.push_back(joined.expect("Fetch task panicked")?);
            collect_ready_claims(fetches, buffer)?;
            request_claims(fetches, buffer.len(), target, cli, client);
        }

        let claim_data = buffer.pop_front().expect("buffer was just checked");
        info!(
            "Acquired claim:  {}, Base {}",
            claim_data.claim_id, claim_data.base
        );
        debug!(
            "Claim Data: {}",
            serde_json::to_string(&claim_data).unwrap()
        );
        debug!(
            "Prefetch: {} buffered, {} in flight, target {target}",
            buffer.len(),
            fetches.len()
        );

        // Start the field. The claim requests already spawned keep making
        // progress on the runtime while this is awaited; so does the device,
        // on the field(s) started before it.
        let begun_at = Instant::now();
        let (claim_data, stage) = begin_field(claim_data, cli, gpu).await;
        in_flight.push_back(InFlightField {
            claim: claim_data,
            stage,
            begun_at,
        });

        // Finish the oldest field(s) beyond the lookahead — all of them on
        // the last round.
        while in_flight.len() > lookahead || (!cli.repeat && !in_flight.is_empty()) {
            let field = in_flight.pop_front().expect("checked non-empty");
            let (results, pipeline) = finish_field(field.stage, cli, gpu).await;
            let now = Instant::now();
            let measured_from = last_finished.map_or(field.begun_at, |t| t.max(field.begun_at));
            let elapsed = now.duration_since(measured_from);
            last_finished = Some(now);
            complete_field(
                field.claim,
                results,
                elapsed,
                pipeline,
                cli,
                client,
                submits,
                telemetry_base.as_ref(),
                &mut field_process_ewma,
            )
            .await?;
        }

        if !cli.repeat {
            break;
        }
    }
    Ok(())
}

/// Log, account, and submit one finished field.
#[allow(clippy::too_many_arguments)]
async fn complete_field(
    claim_data: DataToClient,
    results: Vec<FieldResults>,
    elapsed: Duration,
    pipeline: PipelineTelemetry,
    cli: &Arc<Cli>,
    client: &Client,
    submits: &mut JoinSet<Result<()>>,
    telemetry_base: Option<&serde_json::Value>,
    field_process_ewma: &mut Option<f64>,
) -> Result<()> {
    log_field_rate(&claim_data, elapsed, cli);

    // Feed the buffer sizer.
    let elapsed_secs = elapsed.as_secs_f64();
    *field_process_ewma = Some(match *field_process_ewma {
        Some(previous) => 0.7 * previous + 0.3 * elapsed_secs,
        None => elapsed_secs,
    });

    // Compile results for submission
    let telemetry =
        telemetry_base.map(|base| bench::field_telemetry(base, elapsed_secs, pipeline.as_ref()));
    let submit_data = compile_results(results, &claim_data, &cli.username, cli.mode, telemetry);

    // Submit without blocking the next field on the round trip.
    //
    // Hand this field off before inspecting any earlier submission: a
    // failure there returns from this function, and results not yet spawned
    // would go with it. Once spawned, the caller's drain will see it
    // through.
    submits.spawn({
        let api_base = cli.api_base.clone();
        let api_num_retries = cli.api_max_retries;
        let client = client.clone();
        async move { submit_results(&client, &api_base, submit_data, api_num_retries).await }
    });

    // Reap whatever has finished, so a failure is still reported promptly.
    while let Some(joined) = submits.try_join_next() {
        joined.expect("Submit task panicked")?;
    }
    // Then wait if we are running too far ahead of the server.
    while submits.len() >= MAX_SUBMITS_IN_FLIGHT {
        let joined = submits
            .join_next()
            .await
            .expect("join set is not empty here");
        joined.expect("Submit task panicked")?;
    }
    Ok(())
}

/// Wait for every outstanding submission to finish, then report the first
/// failure among them.
///
/// Every task is joined even after one fails: the alternative is to return
/// early and drop the `JoinSet`, which aborts the rest and discards fields that
/// were already processed.
async fn drain_submits(submits: &mut JoinSet<Result<()>>) -> Result<()> {
    let mut first_error: Option<anyhow::Error> = None;
    while let Some(joined) = submits.join_next().await {
        if let Err(e) = joined.expect("Submit task panicked") {
            match first_error {
                None => first_error = Some(e),
                Some(_) => error!("Further submission failure: {e}"),
            }
        }
    }
    match first_error {
        Some(e) => Err(e),
        None => Ok(()),
    }
}

/// Run in pipelined mode: overlap API calls with processing.
///
/// Owns the pipeline state so that however `run_pipelined_fields` ends, the
/// submissions still in flight are finished rather than aborted and the claims
/// left behind are reported.
async fn run_pipelined_loop(cli: &Arc<Cli>, client: &Client, gpu: &GpuCtx) -> Result<()> {
    let mut buffer: VecDeque<DataToClient> = VecDeque::new();
    let mut fetches: JoinSet<Result<DataToClient>> = JoinSet::new();
    let mut submits: JoinSet<Result<()>> = JoinSet::new();
    let mut in_flight: VecDeque<InFlightField> = VecDeque::new();

    let outcome = run_pipelined_fields(
        cli,
        client,
        gpu,
        &mut buffer,
        &mut fetches,
        &mut submits,
        &mut in_flight,
    )
    .await;

    // Work already done outranks the error that interrupted it.
    let drained = drain_submits(&mut submits).await;

    // Claims we never got to stay claimed on the server until the window
    // expires, so account for them instead of letting them disappear quietly.
    let abandoned = buffer.len() + fetches.len() + in_flight.len();
    if abandoned > 0 {
        warn!(
            "Abandoning up to {abandoned} claimed field(s) ({} buffered, {} in flight, {} started on the device).",
            buffer.len(),
            fetches.len(),
            in_flight.len()
        );
    }

    match (outcome, drained) {
        // Report what stopped the pipeline, not what the drain then ran into.
        (Err(e), drain) => {
            if let Err(drain_error) = drain {
                error!("Submission also failed while draining: {drain_error}");
            }
            Err(e)
        }
        (Ok(()), drain) => drain,
    }
}

/// Whether the user set any of these arguments themselves — on the command
/// line or through its `NICE_*` environment variable, which counts the same
/// because docker and fleet deployments configure the client through the
/// environment. A value that clap filled in from a default does not count.
fn user_set(matches: &ArgMatches, ids: &[&str]) -> bool {
    ids.iter().any(|id| {
        matches!(
            matches.value_source(id),
            Some(ValueSource::CommandLine | ValueSource::EnvVariable)
        )
    })
}

#[tokio::main]
async fn main() -> Result<()> {
    // Parse command line arguments. Shared behind an `Arc` because every field
    // handed to the blocking pool needs an owned copy of the settings.
    let matches = Cli::command().get_matches();
    let mut cli = Cli::from_arg_matches(&matches).expect("clap already validated the matches");

    // Reaching for a sub-option is asking for the feature: --gpu-backend
    // without --gpu (or NICE_BENCHMARK_SECS without -b) should work, not
    // silently do nothing. Bool sub-options imply only when truthy, so
    // NICE_BENCHMARK_UPLOAD=false doesn't switch the benchmark on.
    let gpu_implied =
        !cli.gpu && user_set(&matches, &["gpu_device", "gpu_backend", "gpu_wgpu_device"]);
    if gpu_implied {
        cli.gpu = true;
    }
    let benchmark_implied = !cli.benchmark
        && (user_set(&matches, &["benchmark_secs"]) || cli.benchmark_upload || cli.benchmark_json);
    if benchmark_implied {
        cli.benchmark = true;
    }
    let cli = Arc::new(cli);

    // Set up logger. The default filter quiets the wgpu stack's adapter and
    // server chatter at info level; RUST_LOG or --log-level replaces the
    // whole filter, so `-l debug` still shows everything.
    let mut builder = env_logger::Builder::from_env(Env::default().default_filter_or(
        "info,wgpu_core=warn,wgpu_hal=warn,naga=warn,cubecl_wgpu=warn,cubecl_runtime=warn",
    ));
    if let Some(level) = cli.log_level {
        builder.filter_level(level.into());
    }
    builder.init();

    if gpu_implied {
        debug!("--gpu implied by an explicit --gpu-* option");
    }
    if benchmark_implied {
        debug!("--benchmark implied by an explicit --benchmark-* option");
    }

    // Check for GPU support
    if cli.gpu && !(cfg!(feature = "cuda") || cfg!(feature = "vulkan") || cfg!(feature = "cubecl"))
    {
        error!(
            "This build carries no GPU backends. Use a GPU release binary \
             (or the `-gpu` docker tag), or rebuild with --features gpu."
        );
        std::process::exit(1);
    }

    if cli.validate && cli.mode == SearchMode::Niceonly {
        error!("Configuration not supported: Validation && Niceonly");
        std::process::exit(1);
    }

    #[allow(unused_mut)]
    let mut cpu_or_gpu = format!("CPU with {} threads", cli.threads);

    #[cfg(any(feature = "cuda", feature = "vulkan", feature = "cubecl"))]
    if cli.gpu {
        cpu_or_gpu = format!("GPU device {}", cli.gpu_device);
    };

    info!(
        "Nice Client v{} started in {} mode, using {}.",
        CLIENT_VERSION, cli.mode, cpu_or_gpu
    );
    if cli.validate {
        debug!("Validating correctness by checking against accepted field.");
    }
    if cli.repeat && !cli.validate && !cli.benchmark {
        debug!("Pipeline mode enabled: overlapping API calls with processing.");
    }
    debug!("CLI Inputs: {cli:?}");

    // Initialize GPU context if requested
    let gpu_ctx: GpuCtx = init_gpu(&cli);

    // Configure Rayon for CPU processing
    if !cli.gpu {
        rayon::ThreadPoolBuilder::new()
            .num_threads(cli.threads)
            .build_global()
            .unwrap();
    }

    // Create a shared HTTP client with proper timeout for connection reuse
    let http_client = Client::builder()
        .timeout(Duration::from_secs(CLIENT_REQUEST_TIMEOUT_SECS))
        .build()
        .expect("Failed to create HTTP client");

    // Choose execution mode based on flags. Each mode repeats on its own, so
    // that the pipelined one can keep its claim buffer full across fields.
    if cli.validate {
        run_validation(&cli, &http_client, &gpu_ctx).await?;
    } else if cli.benchmark {
        bench::run_benchmark_sweep(&cli, &gpu_ctx, &http_client).await;
    } else {
        run_pipelined_loop(&cli, &http_client, &gpu_ctx).await?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{DEFAULT_PREFETCH_MAX, DEFAULT_PREFETCH_SECONDS, drain_submits, prefetch_target};
    use anyhow::{Result, anyhow};
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use tokio::task::JoinSet;

    // Deliberately not built through `Cli::parse_from`: clap reads the `env`
    // attributes from the real process environment, so a developer with
    // NICE_PREFETCH_SECONDS exported would see these fail, and a malformed value
    // would make clap exit the test binary outright. Testing the plain function
    // against the named defaults keeps the assertions about the shipped
    // behavior without making them hostage to the shell they run in.

    #[test]
    fn slow_clients_keep_the_historical_depth() {
        // Any client at or above `prefetch_seconds` per field must behave exactly
        // as it always has: one field being processed, one prefetched. The
        // boundary case is the interesting one, hence DEFAULT_PREFETCH_SECONDS
        // itself in the list.
        for per_field in [660.0, 11.0, DEFAULT_PREFETCH_SECONDS] {
            assert_eq!(
                prefetch_target(
                    DEFAULT_PREFETCH_SECONDS,
                    DEFAULT_PREFETCH_MAX,
                    Some(per_field)
                ),
                2,
                "a {per_field}s field should not deepen the buffer"
            );
        }
        // And before anything has been processed we do not guess.
        assert_eq!(
            prefetch_target(DEFAULT_PREFETCH_SECONDS, DEFAULT_PREFETCH_MAX, None),
            2
        );
    }

    #[test]
    fn fast_clients_buffer_the_configured_seconds() {
        // A quarter-second field needs eight buffered to cover two seconds,
        // plus the one being processed.
        assert_eq!(prefetch_target(2.0, DEFAULT_PREFETCH_MAX, Some(0.25)), 9);
        assert_eq!(prefetch_target(2.0, DEFAULT_PREFETCH_MAX, Some(1.0)), 3);
        // Doubling the window doubles the fields held, cap permitting.
        assert_eq!(prefetch_target(4.0, 64, Some(0.25)), 17);
    }

    #[test]
    fn the_cap_and_the_opt_out_both_hold() {
        assert_eq!(
            prefetch_target(2.0, DEFAULT_PREFETCH_MAX, Some(0.0001)),
            DEFAULT_PREFETCH_MAX
        );
        assert_eq!(
            prefetch_target(2.0, DEFAULT_PREFETCH_MAX, Some(0.0)),
            DEFAULT_PREFETCH_MAX
        );

        // Zero seconds is the opt-out: back to a single prefetched field.
        assert_eq!(prefetch_target(0.0, DEFAULT_PREFETCH_MAX, Some(0.25)), 2);

        // A nonsensical cap must still leave room for one prefetched field.
        assert_eq!(prefetch_target(2.0, 1, Some(0.25)), 2);
        assert_eq!(prefetch_target(2.0, 0, Some(0.25)), 2);
    }

    #[tokio::test]
    async fn draining_finishes_every_submission_despite_a_failure() {
        // The point of the drain: returning early on the first failure would
        // drop the JoinSet, and dropping a JoinSet aborts the tasks still in it.
        // Those tasks are fields that have already been processed, so losing
        // them means the work is done again by somebody else.
        let completed = Arc::new(AtomicUsize::new(0));
        let mut submits: JoinSet<Result<()>> = JoinSet::new();

        // The failure lands first; the successes are still in progress behind it.
        submits.spawn(async { Err(anyhow!("server rejected the submission")) });
        for _ in 0..4 {
            let completed = Arc::clone(&completed);
            submits.spawn(async move {
                for _ in 0..8 {
                    tokio::task::yield_now().await;
                }
                completed.fetch_add(1, Ordering::SeqCst);
                Ok(())
            });
        }

        let result = drain_submits(&mut submits).await;

        assert!(result.is_err(), "the failure still has to be reported");
        assert_eq!(
            completed.load(Ordering::SeqCst),
            4,
            "every queued submission must run to completion despite the failure"
        );
        assert_eq!(submits.len(), 0, "nothing may be left unjoined");
    }

    #[tokio::test]
    async fn draining_a_healthy_pipeline_succeeds() {
        let mut submits: JoinSet<Result<()>> = JoinSet::new();
        for _ in 0..3 {
            submits.spawn(async { Ok(()) });
        }
        assert!(drain_submits(&mut submits).await.is_ok());
    }

    #[test]
    fn the_shipped_defaults_are_what_the_other_tests_assume() {
        // If these move, the "slow clients are unaffected" guarantee above is
        // no longer a statement about what actually ships.
        assert!((DEFAULT_PREFETCH_SECONDS - 2.0).abs() < f64::EPSILON);
        assert_eq!(DEFAULT_PREFETCH_MAX, 16);
    }

    #[test]
    fn sub_options_imply_their_umbrella() {
        // Positive cases only: an exported NICE_* variable can only add an
        // explicit source, never remove one, so these hold in any shell. The
        // negative case (nothing passed → nothing implied) would be hostage
        // to the environment, per the note at the top of this module.
        use clap::CommandFactory;
        let m = super::Cli::command().get_matches_from(["nice_client", "--gpu-device", "1"]);
        assert!(super::user_set(
            &m,
            &["gpu_device", "gpu_backend", "gpu_wgpu_device"]
        ));
        let m = super::Cli::command().get_matches_from(["nice_client", "--gpu-backend", "cubecl"]);
        assert!(super::user_set(
            &m,
            &["gpu_device", "gpu_backend", "gpu_wgpu_device"]
        ));
        let m = super::Cli::command().get_matches_from(["nice_client", "--benchmark-secs", "5"]);
        assert!(super::user_set(&m, &["benchmark_secs"]));
        // Passing the default value explicitly still counts as asking for it.
        let m = super::Cli::command().get_matches_from(["nice_client", "--gpu-device", "0"]);
        assert!(super::user_set(
            &m,
            &["gpu_device", "gpu_backend", "gpu_wgpu_device"]
        ));
    }

    /// `cudarc` panics rather than returning when the CUDA shared library is
    /// missing, which took the whole client down before `--gpu-backend auto`
    /// could fall through to Vulkan. Whatever this machine has installed,
    /// initialization must *return* — Ok on a CUDA box, Err on one without —
    /// and never unwind past here.
    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_init_returns_instead_of_panicking() {
        // The result depends on the host; only the absence of a panic matters.
        let _ = super::try_init_cuda(0);
    }
}
