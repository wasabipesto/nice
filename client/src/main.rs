//! A simple CLI for the nice library.

#![warn(clippy::all, clippy::pedantic)]
#![allow(clippy::too_many_lines)]

extern crate nice_common;
use nice_common::benchmark::{BenchmarkMode, get_benchmark_field};
use nice_common::client_api::{
    get_field_from_server, get_validation_data_from_server, submit_field_to_server,
};
use nice_common::client_process::{process_range_detailed, process_range_niceonly};
use nice_common::{
    CLIENT_VERSION, DataToClient, DataToServer, FieldResults, PROCESSING_CHUNK_SIZE, SearchMode,
    UniquesDistributionSimple,
};

extern crate serde_json;
use clap::Parser;
use rayon::prelude::*;
use simple_tqdm::ParTqdm;
use std::collections::HashMap;

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

    /// Run parallel with this many threads
    #[arg(short, long, default_value_t = 4, env = "NICE_THREADS")]
    threads: usize,

    /// Run an offline benchmark
    #[arg(short, long, env = "NICE_BENCHMARK")]
    benchmark: Option<BenchmarkMode>,

    /// Validate results against the server before submitting
    #[arg(long, env = "NICE_VALIDATE")]
    validate: bool,
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

fn main() {
    // Parse command line arguments
    let cli = Cli::parse();

    // Configure Rayon
    // This must be done outside the loop
    rayon::ThreadPoolBuilder::new()
        .num_threads(cli.threads)
        .build_global()
        .unwrap();

    if cli.validate && cli.mode == SearchMode::Niceonly {
        eprintln!("Configuration not supported: Validation && Niceonly");
    }

    loop {
        // Check whether to query the server for a search range or use the benchmark
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

        // Print some debug info
        if validation_data_opt.is_some() {
            println!(
                "Beginning validation: {:?}",
                validation_data_opt.clone().unwrap().field_id
            );
        } else if cli.benchmark.is_some() {
            println!("Beginning benchmark:  {:?}", cli.benchmark.unwrap());
        } else if cli.verbose {
            println!(
                "Claim Data: {}",
                serde_json::to_string_pretty(&claim_data).unwrap()
            );
        } else if !cli.quiet {
            println!("Acquired claim:  {}", claim_data.claim_id);
        }

        // Break up the range into chunks
        let chunk_size = 100 * PROCESSING_CHUNK_SIZE;
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
            .with_disable(cli.quiet);

        // Process each chunk and gather the results
        let results: Vec<FieldResults> = chunks
            .par_iter()
            .tqdm_config(tqdm_config)
            .map(|(start, end)| match cli.mode {
                SearchMode::Detailed => process_range_detailed(*start, *end, claim_data.base),
                SearchMode::Niceonly => process_range_niceonly(*start, *end, claim_data.base),
            })
            .collect();

        // Compile results from all chunks
        let nice_numbers = results
            .iter()
            .flat_map(|result| result.nice_numbers.clone())
            .collect();
        let unique_distribution = if cli.mode == SearchMode::Niceonly {
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

        // Assemble the data package to submit to the server
        let submit_data = DataToServer {
            claim_id: claim_data.claim_id,
            username: cli.username.clone(),
            client_version: CLIENT_VERSION.to_string(),
            unique_distribution,
            nice_numbers,
        };

        // Print some debug info
        if cli.verbose {
            println!(
                "Submit Data: {}",
                serde_json::to_string_pretty(&submit_data).unwrap()
            );
        }

        // Validate results if requested
        if cli.validate {
            // Check if our results match the server's expected results
            let validation_data = validation_data_opt.expect("Validation data not found");
            let mut validation_passed = true;

            // Compare nice numbers
            let mut our_numbers = submit_data.nice_numbers.clone();
            let mut server_numbers = validation_data.nice_numbers.clone();
            our_numbers.sort_by_key(|n| n.number);
            server_numbers.sort_by_key(|n| n.number);

            if our_numbers != server_numbers {
                println!("VALIDATION FAILED: Semi-nice numbers don't match!");
                validation_passed = false;
            }

            // Compare distribution (only for detailed mode)
            if cli.mode == SearchMode::Detailed
                && let Some(ref our_dist) = submit_data.unique_distribution
            {
                let mut our_dist_sorted = our_dist.clone();
                let mut server_dist_sorted = validation_data.unique_distribution.clone();
                our_dist_sorted.sort_by_key(|d| d.num_uniques);
                server_dist_sorted.sort_by_key(|d| d.num_uniques);

                if our_dist_sorted != server_dist_sorted {
                    println!("VALIDATION FAILED: Distribution doesn't match!");
                    validation_passed = false;
                }
            }

            if validation_passed {
                println!("Validation passed! Results match the canoncical submission.");
            } else {
                println!("Validation failed! Results do not match the canoncical submission.");
                println!("  Our submission data: {submit_data:?}");
                println!("  Canoncical submission: {validation_data:?}");
            }
        } else if cli.benchmark.is_none() {
            // Submit the results if it's not a benchmark
            let response = submit_field_to_server(&cli.api_base, submit_data);
            match response.text() {
                Ok(msg) => {
                    if !cli.quiet {
                        println!("Server response: {msg}");
                    }
                }
                Err(e) => println!("Server returned success but an error occured: {e}"),
            }
        }

        if !cli.repeat {
            break;
        }
    }
}
