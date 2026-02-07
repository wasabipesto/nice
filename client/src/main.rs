//! A simple CLI for the nice library.

#![warn(clippy::all, clippy::pedantic)]
#![allow(clippy::too_many_lines)]

extern crate nice_common;
use nice_common::benchmark::{BenchmarkMode, get_benchmark_field};
use nice_common::client_api::{
    get_field_from_server, get_field_from_server_async, get_validation_data_from_server,
    submit_field_to_server_async,
};
use nice_common::client_process::{process_range_detailed, process_range_niceonly};
use nice_common::{
    CLIENT_VERSION, DataToClient, DataToServer, FieldResults, PROCESSING_CHUNK_SIZE, SearchMode,
    UniquesDistributionSimple, ValidationData,
};

#[cfg(feature = "gpu")]
use nice_common::client_process_gpu::{
    GPU_BATCH_SIZE, GpuContext, process_range_detailed_gpu, process_range_niceonly_gpu,
};
#[cfg(feature = "gpu")]
use std::sync::Arc;

extern crate serde_json;
use clap::{Parser, ValueEnum};
use env_logger::Env;
use log::{LevelFilter, debug, error, info};
use rayon::prelude::*;
use simple_tqdm::ParTqdm;
use std::collections::HashMap;

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

    /// Run an offline benchmark
    #[arg(short, long, env = "NICE_BENCHMARK")]
    benchmark: Option<BenchmarkMode>,

    /// Validate results against the server before submitting
    #[arg(long, env = "NICE_VALIDATE")]
    validate: bool,

    /// Use GPU acceleration (requires gpu feature)
    #[arg(long, env = "NICE_GPU")]
    gpu: bool,

    /// CUDA device to use for GPU processing (0 for first GPU, 1 for second, etc.)
    #[arg(long, default_value_t = 0, env = "NICE_GPU_DEVICE")]
    gpu_device: usize,

    /// Set the log level (overrides `RUST_LOG` environment variable)
    #[arg(short, long, value_enum, env = "NICE_LOG_LEVEL")]
    log_level: Option<LogLevel>,
}

/// Break up the range into chunks, returning the start and end of each.
fn chunked_ranges(range_start: u128, range_end: u128, chunk_size: usize) -> Vec<(u128, u128)> {
    let mut chunks = Vec::new();
    let mut start = range_start;

    while start < range_end {
        let end = (start + chunk_size as u128).min(range_end);
        chunks.push((start, end));
        start = end;
    }

    chunks
}

/// Process a field synchronously (`CPU` or `GPU`).
/// This is wrapped in `spawn_blocking` when called from async context.
fn process_field_sync(
    claim_data: &DataToClient,
    mode: SearchMode,
    cli: &Cli,
    #[cfg(feature = "gpu")] gpu_ctx: Option<&Arc<GpuContext>>,
) -> Vec<FieldResults> {
    if cli.gpu {
        // GPU processing path
        #[cfg(feature = "gpu")]
        {
            let gpu_ctx = gpu_ctx.expect("GPU context failed to initialize");

            let gpu_results = match mode {
                SearchMode::Detailed => process_range_detailed_gpu(
                    gpu_ctx,
                    claim_data.range_start,
                    claim_data.range_end,
                    claim_data.base,
                ),
                SearchMode::Niceonly => process_range_niceonly_gpu(
                    gpu_ctx,
                    claim_data.range_start,
                    claim_data.range_end,
                    claim_data.base,
                ),
            };

            match gpu_results {
                Ok(result) => vec![result],
                Err(e) => {
                    error!("GPU processing error: {e:?}");
                    std::process::exit(1);
                }
            }
        }
        #[cfg(not(feature = "gpu"))]
        {
            error!("GPU support not compiled in");
            std::process::exit(1);
        }
    } else {
        // CPU processing path
        let chunk_size = PROCESSING_CHUNK_SIZE;
        let chunks = chunked_ranges(claim_data.range_start, claim_data.range_end, chunk_size);

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
            .map(|(start, end)| match mode {
                SearchMode::Detailed => process_range_detailed(*start, *end, claim_data.base),
                SearchMode::Niceonly => process_range_niceonly(*start, *end, claim_data.base),
            })
            .collect()
    }
}

/// Compile results from multiple chunks into a single `DataToServer`.
#[allow(clippy::needless_pass_by_value)]
fn compile_results(
    results: Vec<FieldResults>,
    claim_data: &DataToClient,
    username: &str,
    mode: SearchMode,
) -> DataToServer {
    let nice_numbers = results
        .iter()
        .flat_map(|result| result.nice_numbers.clone())
        .collect();

    let unique_distribution = if mode == SearchMode::Niceonly {
        None
    } else {
        // Flatten all distribution sets from the results
        let result_distributions: Vec<UniquesDistributionSimple> = results
            .iter()
            .flat_map(|result| result.distribution.clone())
            .collect();

        // Collect the counts into a map
        let mut dist_map: HashMap<u32, u128> = HashMap::new();
        for dist in result_distributions {
            *dist_map.entry(dist.num_uniques).or_insert(0) += dist.count;
        }

        // Convert the counts back into a formatted, sorted list
        let mut distribution: Vec<UniquesDistributionSimple> = dist_map
            .into_iter()
            .map(|(num_uniques, count)| UniquesDistributionSimple { num_uniques, count })
            .collect();
        distribution.sort_by_key(|d| d.num_uniques);
        Some(distribution)
    };

    DataToServer {
        claim_id: claim_data.claim_id,
        username: username.to_string(),
        client_version: CLIENT_VERSION.to_string(),
        unique_distribution,
        nice_numbers,
    }
}

/// Validate results against expected `ValidationData`.
#[allow(clippy::needless_pass_by_value)]
fn validate_results(
    submit_data: &DataToServer,
    validation_data: ValidationData,
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

/// Run a single iteration in non-pipelined mode (validation or benchmark).
async fn run_single_iteration(
    cli: &Cli,
    #[cfg(feature = "gpu")] gpu_ctx: Option<&Arc<GpuContext>>,
) {
    // Get the field (synchronously for validation/benchmark)
    let (claim_data, validation_data_opt) = if cli.validate {
        let validation_data = get_validation_data_from_server(&cli.api_base);
        let claim_data = DataToClient {
            claim_id: 0,
            base: validation_data.base,
            range_start: validation_data.range_start,
            range_end: validation_data.range_end,
            range_size: validation_data.range_size,
        };
        (claim_data, Some(validation_data))
    } else if let Some(benchmark) = cli.benchmark {
        (get_benchmark_field(benchmark), None)
    } else {
        (get_field_from_server(&cli.mode, &cli.api_base), None)
    };

    // Show claim details
    if let Some(ref validation_data) = validation_data_opt {
        info!("Beginning validation: {}", validation_data.field_id);
    } else if let Some(benchmark) = cli.benchmark {
        info!("Beginning benchmark:  {benchmark}");
    } else {
        info!(
            "Acquired claim:  {}, Base {}",
            claim_data.claim_id, claim_data.base
        );
    }
    debug!(
        "Claim Data: {}",
        serde_json::to_string(&claim_data).unwrap()
    );

    let start_time = std::time::Instant::now();

    // Process the field
    let results = tokio::task::spawn_blocking({
        let claim_data = claim_data.clone();
        let mode = cli.mode;
        let cli_clone = cli.clone();
        #[cfg(feature = "gpu")]
        let gpu_ctx_clone = gpu_ctx.cloned();
        move || {
            #[cfg(feature = "gpu")]
            {
                process_field_sync(&claim_data, mode, &cli_clone, gpu_ctx_clone.as_ref())
            }
            #[cfg(not(feature = "gpu"))]
            {
                process_field_sync(&claim_data, mode, &cli_clone)
            }
        }
    })
    .await
    .expect("Processing task panicked");

    let elapsed = start_time.elapsed();

    // Print performance stats if progress bar is disabled
    #[allow(clippy::cast_precision_loss)]
    if cli.no_progress || cli.gpu {
        let range_size = claim_data.range_size;
        let numbers_per_sec = range_size as f64 / elapsed.as_secs_f64();
        info!(
            "✓ Processed {:.2e} numbers in {:.2}s ({:.2e} numbers/sec)",
            range_size as f64,
            elapsed.as_secs_f64(),
            numbers_per_sec
        );
    }

    // Compile results
    let submit_data = compile_results(results, &claim_data, &cli.username, cli.mode);

    debug!(
        "Submit Data: {}",
        serde_json::to_string(&submit_data).unwrap()
    );

    // Handle validation or submission
    if cli.validate {
        let validation_data = validation_data_opt.expect("Validation data not found");
        let validation_passed = validate_results(&submit_data, validation_data.clone(), cli.mode);

        if validation_passed {
            println!();
            println!("Validation passed! Results match the canoncical submission.");
        } else {
            println!();
            println!("Validation failed! Results do not match the canoncical submission.");
            println!("  Our submission data: {submit_data:?}");
            println!("  Canoncical submission: {validation_data:?}");
            std::process::exit(1);
        }
    } else if cli.benchmark.is_none() {
        let response = submit_field_to_server_async(&cli.api_base, submit_data).await;
        match response.text().await {
            Ok(msg) => {
                debug!("Server response: {msg}");
            }
            Err(e) => error!("Server returned success but an error occured: {e}"),
        }
    }
}

/// Run in pipelined mode: overlap API calls with processing.
async fn run_pipelined_loop(cli: &Cli, #[cfg(feature = "gpu")] gpu_ctx: Option<&Arc<GpuContext>>) {
    // State for the pipeline
    let mut pending_submit: Option<DataToServer> = None;
    let mut current_claim: Option<DataToClient> = None;
    let mut next_claim: Option<DataToClient> = None;

    loop {
        // Stage 1: Start fetching the next field if we don't have one
        let fetch_next = if next_claim.is_none() {
            Some(tokio::spawn({
                let mode = cli.mode;
                let api_base = cli.api_base.clone();
                async move { get_field_from_server_async(&mode, &api_base).await }
            }))
        } else {
            None
        };

        // Stage 2: If we have a current claim, process it
        let process_current = if let Some(claim_data) = current_claim.take() {
            info!(
                "Acquired claim:  {}, Base {}",
                claim_data.claim_id, claim_data.base
            );
            debug!(
                "Claim Data: {}",
                serde_json::to_string(&claim_data).unwrap()
            );

            let start_time = std::time::Instant::now();

            Some(tokio::task::spawn_blocking({
                let claim_data = claim_data.clone();
                let mode = cli.mode;
                let cli_clone = cli.clone();
                #[cfg(feature = "gpu")]
                let gpu_ctx_clone = gpu_ctx.cloned();
                move || {
                    let results = {
                        #[cfg(feature = "gpu")]
                        {
                            process_field_sync(
                                &claim_data,
                                mode,
                                &cli_clone,
                                gpu_ctx_clone.as_ref(),
                            )
                        }
                        #[cfg(not(feature = "gpu"))]
                        {
                            process_field_sync(&claim_data, mode, &cli_clone)
                        }
                    };
                    (claim_data, results, start_time.elapsed())
                }
            }))
        } else {
            None
        };

        // Stage 3: Submit previous results if we have any
        let submit_previous = pending_submit.take().map(|submit_data| {
            tokio::spawn({
                let api_base = cli.api_base.clone();
                async move {
                    let response = submit_field_to_server_async(&api_base, submit_data).await;
                    match response.text().await {
                        Ok(msg) => {
                            debug!("Server response: {msg}");
                        }
                        Err(e) => error!("Server returned success but an error occured: {e}"),
                    }
                }
            })
        });

        // Wait for all concurrent operations to complete
        if let Some(fetch_task) = fetch_next {
            next_claim = Some(fetch_task.await.expect("Fetch task panicked"));
        }

        if let Some(process_task) = process_current {
            let (claim_data, results, elapsed) =
                process_task.await.expect("Processing task panicked");

            // Print performance stats if progress bar is disabled
            #[allow(clippy::cast_precision_loss)]
            if cli.no_progress || cli.gpu {
                let range_size = claim_data.range_end - claim_data.range_start;
                let numbers_per_sec = range_size as f64 / elapsed.as_secs_f64();
                info!(
                    "✓ Processed {:.2e} numbers in {:.2}s ({:.2e} numbers/sec)",
                    range_size as f64,
                    elapsed.as_secs_f64(),
                    numbers_per_sec
                );
            }

            // Compile results for submission
            let submit_data = compile_results(results, &claim_data, &cli.username, cli.mode);

            debug!(
                "Submit Data: {}",
                serde_json::to_string(&submit_data).unwrap()
            );

            pending_submit = Some(submit_data);
        }

        if let Some(submit_task) = submit_previous {
            submit_task.await.expect("Submit task panicked");
        }

        // Move the pipeline forward
        current_claim = next_claim.take();

        // If we don't have a current claim and we're not repeating, we're done
        if current_claim.is_none() && !cli.repeat {
            break;
        }
    }

    // Submit any remaining results
    if let Some(submit_data) = pending_submit {
        let response = submit_field_to_server_async(&cli.api_base, submit_data).await;
        match response.text().await {
            Ok(msg) => {
                debug!("Server response: {msg}");
            }
            Err(e) => error!("Server returned success but an error occured: {e}"),
        }
    }
}

#[tokio::main]
async fn main() {
    // Parse command line arguments
    let cli = Cli::parse();

    // Set up logger
    let mut builder = env_logger::Builder::from_env(Env::default().default_filter_or("info"));
    if let Some(level) = cli.log_level {
        builder.filter_level(level.into());
    }
    builder.init();

    // Check for GPU support
    if cli.gpu && !cfg!(feature = "gpu") {
        error!("Error: GPU support not enabled. Rebuild with --features gpu");
        std::process::exit(1);
    }

    if cli.validate && cli.mode == SearchMode::Niceonly {
        error!("Configuration not supported: Validation && Niceonly");
        std::process::exit(1);
    }

    #[allow(unused_mut)]
    let mut cpu_or_gpu = format!("CPU with {} threads", cli.threads);

    #[cfg(feature = "gpu")]
    if cli.gpu {
        cpu_or_gpu = format!(
            "GPU device {} and batch size {}",
            cli.gpu_device, GPU_BATCH_SIZE
        );
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
    #[cfg(feature = "gpu")]
    let gpu_ctx = if cli.gpu {
        match GpuContext::new(cli.gpu_device) {
            Ok(ctx) => {
                info!("GPU initialized successfully on device {}", cli.gpu_device);
                // Try to get GPU name if possible
                if let Ok(device) = cudarc::driver::CudaContext::new(cli.gpu_device)
                    && let Ok(name) = device.name()
                {
                    info!("  GPU: {name}");
                }
                Some(Arc::new(ctx))
            }
            Err(e) => {
                error!(
                    "Failed to initialize GPU on device {}: {:?}",
                    cli.gpu_device, e
                );
                eprintln!("");
                eprintln!("Troubleshooting:");
                eprintln!("1. Ensure NVIDIA GPU drivers are installed");
                eprintln!("2. Verify CUDA toolkit is installed (nvcc --version)");
                eprintln!("3. Check that GPU {} exists (nvidia-smi)", cli.gpu_device);
                eprintln!("4. Try a different device with --gpu-device <N>");
                std::process::exit(1);
            }
        }
    } else {
        None
    };

    // Configure Rayon for CPU processing
    if !cli.gpu {
        rayon::ThreadPoolBuilder::new()
            .num_threads(cli.threads)
            .build_global()
            .unwrap();
    }

    // Choose execution mode based on flags
    if cli.validate || cli.benchmark.is_some() {
        // Validation and benchmark modes don't support pipelining
        loop {
            #[cfg(feature = "gpu")]
            {
                run_single_iteration(&cli, gpu_ctx.as_ref()).await;
            }
            #[cfg(not(feature = "gpu"))]
            {
                run_single_iteration(&cli).await;
            }

            if !cli.repeat {
                break;
            }
        }
    } else {
        // Normal mode: use pipelining for repeat mode, simple mode otherwise
        if cli.repeat {
            #[cfg(feature = "gpu")]
            {
                run_pipelined_loop(&cli, gpu_ctx.as_ref()).await;
            }
            #[cfg(not(feature = "gpu"))]
            {
                run_pipelined_loop(&cli).await;
            }
        } else {
            #[cfg(feature = "gpu")]
            {
                run_single_iteration(&cli, gpu_ctx.as_ref()).await;
            }
            #[cfg(not(feature = "gpu"))]
            {
                run_single_iteration(&cli).await;
            }
        }
    }
}
