//! A simple CLI for the nice library.

#![warn(clippy::all, clippy::pedantic)]
#![allow(clippy::too_many_lines)]

extern crate nice_common;
use nice_common::benchmark::{BenchmarkMode, get_benchmark_field};
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

const DEFAULT_LSD_K_VALUE: u32 = 2;

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

#[cfg(feature = "gpu")]
use nice_common::client_process_gpu::{
    GPU_BATCH_SIZE, GpuContext, process_range_detailed_gpu, process_range_niceonly_gpu,
};
#[cfg(feature = "vulkan")]
use nice_common::client_process_vulkan::{
    VULKAN_BATCH_SIZE, process_range_detailed_vulkan, process_range_niceonly_vulkan,
};
#[cfg(feature = "vulkan")]
use nice_common::vulkan::VulkanContext;

/// Which GPU backend to drive.
///
/// Both backends `dlopen` their driver at runtime, so a single binary can carry
/// both and require neither at build time. `Auto` tries CUDA first, leaving
/// behaviour on an NVIDIA machine exactly as it was.
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
enum GpuBackend {
    /// CUDA if available, otherwise Vulkan.
    Auto,
    /// NVIDIA only; requires the CUDA toolkit at runtime for NVRTC.
    Cuda,
    /// Any Vulkan 1.2 device with `shaderInt64` (AMD, Intel, NVIDIA, llvmpipe).
    Vulkan,
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
    #[cfg(feature = "gpu")]
    Cuda(GpuContext),
    #[cfg(feature = "vulkan")]
    Vulkan(VulkanContext),
}

/// The GPU context, if this build has one and the user asked for it.
///
/// Naming the type in every build means everything that only passes the context
/// along can do so without a `#[cfg]`, leaving the split to
/// `process_field_sync` — the one place that actually looks inside it.
type GpuCtx = Option<Arc<GpuHandle>>;

extern crate serde_json;
use anyhow::{Result, anyhow};
use clap::{Parser, ValueEnum};
use env_logger::Env;
use log::{LevelFilter, debug, error, info, warn};
use rayon::prelude::*;
use simple_tqdm::ParTqdm;
use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use std::time::Duration;
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
    #[arg(short, long, env = "NICE_REPEAT")]
    repeat: bool,

    /// Hide the progress bar
    #[arg(short, long, env = "NICE_NO_PROGRESS")]
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

    /// Run an offline benchmark
    #[arg(short, long, env = "NICE_BENCHMARK")]
    benchmark: Option<BenchmarkMode>,

    /// Validate results against the server before submitting
    #[arg(long, env = "NICE_VALIDATE")]
    validate: bool,

    /// Use GPU acceleration (requires gpu feature)
    #[arg(long, env = "NICE_GPU")]
    gpu: bool,

    /// GPU device to use (0 for first GPU, 1 for second, etc.)
    #[arg(long, default_value_t = 0, env = "NICE_GPU_DEVICE")]
    gpu_device: usize,

    /// Which GPU backend to use with --gpu
    #[arg(long, value_enum, default_value_t = GpuBackend::Auto, env = "NICE_GPU_BACKEND")]
    gpu_backend: GpuBackend,

    /// Set the log level (overrides `RUST_LOG` environment variable)
    #[arg(short, long, value_enum, env = "NICE_LOG_LEVEL")]
    log_level: Option<LogLevel>,
}

/// Which backends this build carries, for error messages.
fn compiled_backends() -> &'static str {
    match (cfg!(feature = "gpu"), cfg!(feature = "vulkan")) {
        (true, true) => "cuda, vulkan",
        (true, false) => "cuda",
        (false, true) => "vulkan",
        (false, false) => "none",
    }
}

/// Try to bring up CUDA, turning `cudarc`'s panics into errors.
///
/// `cudarc` panics rather than returning when the CUDA shared library is absent
/// (`panic_no_lib_found`, cudarc/src/lib.rs) — which is the ordinary case on any
/// machine with no NVIDIA driver, i.e. precisely the case `--gpu-backend auto`
/// exists to fall through. Upstream catches the same panic for the same reason
/// in `nvrtc_compiles_kernels_for_all_supported_bases`.
///
/// The panic hook is silenced for the duration so a routine fallback does not
/// print cudarc's 20-line message. GPU init runs before any worker threads
/// start, so swapping the global hook here is safe.
///
/// Note this relies on unwinding; under `panic = "abort"` the process would die
/// as before. The workspace does not set that.
#[cfg(feature = "gpu")]
fn try_init_cuda(device: usize) -> Result<GpuContext> {
    let previous = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {}));
    let caught = std::panic::catch_unwind(|| GpuContext::new(device));
    std::panic::set_hook(previous);

    match caught {
        Ok(result) => result,
        Err(payload) => {
            let msg = payload
                .downcast_ref::<String>()
                .map(String::as_str)
                .or_else(|| payload.downcast_ref::<&str>().copied())
                .unwrap_or("the CUDA driver library could not be loaded");
            Err(anyhow!("{msg}"))
        }
    }
}

/// Bring up the GPU backend the user asked for.
///
/// `Auto` tries CUDA first and falls back to Vulkan, so an NVIDIA machine
/// behaves exactly as it did before this existed. An **explicitly named**
/// backend that fails to initialize is fatal rather than falling back: for a
/// distributed compute client, silently dropping to a much slower path is a
/// worse outcome than stopping and saying so.
#[allow(unused_variables)]
fn init_gpu(cli: &Cli) -> GpuCtx {
    if !cli.gpu {
        return None;
    }
    let want = cli.gpu_backend;

    #[cfg(feature = "gpu")]
    if matches!(want, GpuBackend::Auto | GpuBackend::Cuda) {
        match try_init_cuda(cli.gpu_device) {
            Ok(ctx) => {
                info!(
                    "GPU initialized: CUDA device {}, batch size {}",
                    cli.gpu_device, GPU_BATCH_SIZE
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
                info!("CUDA unavailable; trying Vulkan");
                debug!("  CUDA init failed: {e:#}");
            }
        }
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
fn process_field_sync(claim_data: &DataToClient, cli: &Cli, gpu: &GpuCtx) -> Vec<FieldResults> {
    let mode = cli.mode;
    if cli.gpu {
        // GPU processing path
        #[cfg(any(feature = "gpu", feature = "vulkan"))]
        {
            let handle = gpu.as_ref().expect("GPU context failed to initialize");
            let range: FieldSize = claim_data.into();

            let gpu_results = match &**handle {
                #[cfg(feature = "gpu")]
                GpuHandle::Cuda(ctx) => match mode {
                    SearchMode::Detailed => {
                        process_range_detailed_gpu(ctx, &range, claim_data.base)
                    }
                    SearchMode::Niceonly => {
                        process_range_niceonly_gpu(ctx, &range, claim_data.base)
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
            };

            match gpu_results {
                Ok(result) => vec![result],
                Err(e) => {
                    error!("GPU processing error: {e:?}");
                    std::process::exit(1);
                }
            }
        }
        #[cfg(not(any(feature = "gpu", feature = "vulkan")))]
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
            Some(Arc::new(stride_filter::StrideTable::new(
                claim_data.base,
                DEFAULT_LSD_K_VALUE,
            )))
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

/// Process one field on the blocking pool.
///
/// Returns the claim back alongside its results, since the blocking task has to
/// take ownership of it, plus how long the processing took.
async fn process_field(
    claim_data: DataToClient,
    cli: &Arc<Cli>,
    gpu: &GpuCtx,
) -> (DataToClient, Vec<FieldResults>, Duration) {
    let cli = Arc::clone(cli);
    let gpu = gpu.clone();
    let start_time = std::time::Instant::now();

    tokio::task::spawn_blocking(move || {
        let results = process_field_sync(&claim_data, &cli, &gpu);
        (claim_data, results, start_time.elapsed())
    })
    .await
    .expect("Processing task panicked")
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

        let submit_data = compile_results(results, &claim_data, &cli.username, cli.mode);

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

/// Process a fixed offline field to measure this machine.
///
/// Never contacts the server, hence no HTTP client and no `Result`.
async fn run_benchmark(cli: &Arc<Cli>, benchmark: BenchmarkMode, gpu: &GpuCtx) {
    loop {
        let claim_data = get_benchmark_field(benchmark);

        info!("Beginning benchmark:  {benchmark}");
        debug!(
            "Claim Data: {}",
            serde_json::to_string(&claim_data).unwrap()
        );

        let (claim_data, results, elapsed) = process_field(claim_data, cli, gpu).await;
        log_field_rate(&claim_data, elapsed, cli);

        // Compiled but never sent; the debug trace it emits is the point.
        compile_results(results, &claim_data, &cli.username, cli.mode);

        if !cli.repeat {
            break;
        }
    }
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
) -> Result<()> {
    // Exponentially weighted moving average of how long one field takes to
    // process, this is what the target buffer depth is derived from.
    let mut field_process_ewma: Option<f64> = None;

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

        // Process the field. The claim requests already spawned keep making
        // progress on the runtime while this is awaited.
        let (claim_data, results, elapsed) = process_field(claim_data, cli, gpu).await;
        log_field_rate(&claim_data, elapsed, cli);

        // Feed the buffer sizer.
        let elapsed_secs = elapsed.as_secs_f64();
        field_process_ewma = Some(match field_process_ewma {
            Some(previous) => 0.7 * previous + 0.3 * elapsed_secs,
            None => elapsed_secs,
        });

        // Compile results for submission
        let submit_data = compile_results(results, &claim_data, &cli.username, cli.mode);

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

        if !cli.repeat {
            break;
        }
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

    let outcome =
        run_pipelined_fields(cli, client, gpu, &mut buffer, &mut fetches, &mut submits).await;

    // Work already done outranks the error that interrupted it.
    let drained = drain_submits(&mut submits).await;

    // Claims we never got to stay claimed on the server until the window
    // expires, so account for them instead of letting them disappear quietly.
    let abandoned = buffer.len() + fetches.len();
    if abandoned > 0 {
        warn!(
            "Abandoning up to {abandoned} claimed field(s) ({} buffered, {} in flight).",
            buffer.len(),
            fetches.len()
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

#[tokio::main]
async fn main() -> Result<()> {
    // Parse command line arguments. Shared behind an `Arc` because every field
    // handed to the blocking pool needs an owned copy of the settings.
    let cli = Arc::new(Cli::parse());

    // Set up logger
    let mut builder = env_logger::Builder::from_env(Env::default().default_filter_or("info"));
    if let Some(level) = cli.log_level {
        builder.filter_level(level.into());
    }
    builder.init();

    // Check for GPU support
    if cli.gpu && !(cfg!(feature = "gpu") || cfg!(feature = "vulkan")) {
        error!(
            "Error: GPU support not enabled. Rebuild with --features gpu (CUDA) \
             and/or --features vulkan"
        );
        std::process::exit(1);
    }

    if cli.validate && cli.mode == SearchMode::Niceonly {
        error!("Configuration not supported: Validation && Niceonly");
        std::process::exit(1);
    }

    #[allow(unused_mut)]
    let mut cpu_or_gpu = format!("CPU with {} threads", cli.threads);

    #[cfg(any(feature = "gpu", feature = "vulkan"))]
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
    if cli.repeat && !cli.validate && cli.benchmark.is_none() {
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
    } else if let Some(benchmark) = cli.benchmark {
        run_benchmark(&cli, benchmark, &gpu_ctx).await;
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

    /// `cudarc` panics rather than returning when the CUDA shared library is
    /// missing, which took the whole client down before `--gpu-backend auto`
    /// could fall through to Vulkan. Whatever this machine has installed,
    /// initialization must *return* — Ok on a CUDA box, Err on one without —
    /// and never unwind past here.
    #[cfg(feature = "gpu")]
    #[test]
    fn cuda_init_returns_instead_of_panicking() {
        // The result depends on the host; only the absence of a panic matters.
        let _ = super::try_init_cuda(0);
    }
}
