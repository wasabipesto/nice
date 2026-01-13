//! GPU-accelerated client for distributed search of square-cube pandigitals.
//!
//! This client uses CUDA to accelerate the search for nice numbers on NVIDIA GPUs.
//! It requires an A100 or other CUDA-capable GPU and the CUDA toolkit to be installed.

#![warn(clippy::all, clippy::pedantic)]

extern crate nice_common;
use nice_common::benchmark::{BenchmarkMode, get_benchmark_field};
use nice_common::client_api::{get_field_from_server, submit_field_to_server};
use nice_common::client_process_gpu::GpuContext;
use nice_common::{CLIENT_VERSION, DataToClient, DataToServer, SearchMode};

extern crate serde_json;
use clap::Parser;

#[derive(Parser)]
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

    /// Suppress all output
    #[arg(short, long, env = "NICE_QUIET")]
    quiet: bool,

    /// Show additional output
    #[arg(short, long, env = "NICE_VERBOSE")]
    verbose: bool,

    /// CUDA device to use (0 for first GPU, 1 for second, etc.)
    #[arg(short, long, default_value_t = 0, env = "NICE_GPU_DEVICE")]
    device: usize,

    /// Run an offline benchmark
    #[arg(short, long, env = "NICE_BENCHMARK")]
    benchmark: Option<BenchmarkMode>,

    /// Batch size for GPU processing (number of ranges to process per kernel launch)
    #[arg(long, default_value_t = 10_000_000, env = "NICE_BATCH_SIZE")]
    batch_size: usize,
}

fn main() {
    // Parse command line arguments
    let cli = Cli::parse();

    // Initialize GPU context
    // This compiles the CUDA kernels and sets up the device
    let gpu_ctx = match GpuContext::new(cli.device) {
        Ok(ctx) => {
            if !cli.quiet {
                println!("✓ GPU initialized successfully on device {}", cli.device);
                // Try to get GPU name if possible
                if let Ok(device) = cudarc::driver::CudaContext::new(cli.device)
                    && let Ok(name) = device.name()
                {
                    println!("  GPU: {name}");
                }
            }
            ctx
        }
        Err(e) => {
            eprintln!("Failed to initialize GPU on device {}: {:?}", cli.device, e);
            eprintln!("\nTroubleshooting:");
            eprintln!("1. Ensure NVIDIA GPU drivers are installed");
            eprintln!("2. Verify CUDA toolkit is installed (nvcc --version)");
            eprintln!("3. Check that GPU {} exists (nvidia-smi)", cli.device);
            eprintln!("4. Try a different device with --device <N>");
            std::process::exit(1);
        }
    };

    // Repeat indefinitely if requested, otherwise run once
    if cli.repeat {
        if !cli.quiet {
            println!("Running in repeat mode (Ctrl+C to stop)");
        }
        loop {
            if let Err(e) = process_one_field(&cli, &gpu_ctx) {
                eprintln!("Error processing field: {e:?}");
                if !cli.repeat {
                    std::process::exit(1);
                }
                // In repeat mode, continue to next field
            }
        }
    } else if let Err(e) = process_one_field(&cli, &gpu_ctx) {
        eprintln!("Error processing field: {e:?}");
        std::process::exit(1);
    }
}

/// Process a single field from the server
fn process_one_field(cli: &Cli, gpu_ctx: &GpuContext) -> Result<(), Box<dyn std::error::Error>> {
    // Get work from server or use benchmark
    let claim_data = if let Some(benchmark) = cli.benchmark {
        get_benchmark_field(benchmark)
    } else {
        get_field_from_server(&cli.mode, &cli.api_base)
    };

    // Print debug info
    if cli.benchmark.is_some() {
        println!("Beginning GPU benchmark: {:?}", cli.benchmark.unwrap());
    } else if cli.verbose {
        println!("Claim Data: {}", serde_json::to_string_pretty(&claim_data)?);
    } else if !cli.quiet {
        println!("Acquired claim: {}", claim_data.claim_id);
    }

    // Show what we're processing
    #[allow(clippy::cast_precision_loss)]
    if cli.verbose {
        let range_size = claim_data.range_end - claim_data.range_start;
        println!(
            "Processing range: {} to {} (size: {:.2e}, base: {})",
            claim_data.range_start, claim_data.range_end, range_size as f64, claim_data.base
        );
    }

    // Record start time for benchmarking
    let start_time = std::time::Instant::now();

    // Process on GPU based on mode
    let results = match cli.mode {
        SearchMode::Detailed => {
            if !cli.quiet {
                println!("Mode: Detailed (calculating full statistics)");
            }
            process_detailed_gpu(gpu_ctx, &claim_data, &cli.username)?
        }
        SearchMode::Niceonly => {
            if !cli.quiet {
                println!("Mode: Nice-only (optimized for speed)");
            }
            process_niceonly_gpu(gpu_ctx, &claim_data, &cli.username)?
        }
    };

    let elapsed = start_time.elapsed();

    // Print performance stats
    #[allow(clippy::cast_precision_loss)]
    if !cli.quiet {
        let range_size = claim_data.range_end - claim_data.range_start;
        let numbers_per_sec = range_size as f64 / elapsed.as_secs_f64();
        println!(
            "✓ Processed {:.2e} numbers in {:.2}s ({:.2e} numbers/sec)",
            range_size as f64,
            elapsed.as_secs_f64(),
            numbers_per_sec
        );

        if !results.nice_numbers.is_empty() {
            println!("  Found {} nice numbers!", results.nice_numbers.len());
        }
    }

    // Print verbose results
    if cli.verbose {
        println!("Submit Data: {}", serde_json::to_string_pretty(&results)?);
    }

    // Submit results if not a benchmark
    if cli.benchmark.is_none() {
        if cli.verbose {
            println!("Submitting results to server...");
        }

        let response = submit_field_to_server(&cli.api_base, results);
        match response.text() {
            Ok(msg) => {
                if !cli.quiet {
                    println!("Server response: {msg}");
                }
            }
            Err(e) => {
                eprintln!("Server returned success but an error occurred: {e}");
            }
        }
    } else if !cli.quiet {
        println!("Benchmark complete (results not submitted)");
    }

    Ok(())
}

/// Process a field in detailed mode using GPU
fn process_detailed_gpu(
    gpu_ctx: &GpuContext,
    claim_data: &DataToClient,
    username: &str,
) -> Result<DataToServer, Box<dyn std::error::Error>> {
    use nice_common::client_process_gpu::process_range_detailed_gpu;

    let results = process_range_detailed_gpu(
        gpu_ctx,
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

/// Process a field in nice-only mode using GPU
fn process_niceonly_gpu(
    gpu_ctx: &GpuContext,
    claim_data: &DataToClient,
    username: &str,
) -> Result<DataToServer, Box<dyn std::error::Error>> {
    use nice_common::client_process_gpu::process_range_niceonly_gpu;

    let results = process_range_niceonly_gpu(
        gpu_ctx,
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
