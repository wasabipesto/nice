//! A simple CLI for the nice library.

#![warn(clippy::all, clippy::pedantic)]

extern crate nice_common;
use nice_common::benchmark::{get_benchmark_field, BenchmarkMode};
use nice_common::client_api::{get_field_from_server, submit_field_to_server};
use nice_common::client_process::{process_range_detailed, process_range_niceonly};
use nice_common::{
    DataToServer, FieldResults, SearchMode, UniquesDistributionSimple, CLIENT_VERSION,
    PROCESSING_CHUNK_SIZE,
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
    #[arg(value_enum, default_value = "detailed")]
    mode: SearchMode,

    /// The base API URL to connect to
    #[arg(long, default_value = "https://api.nicenumbers.net")]
    api_base: String,

    /// The username to send alongside your contribution
    #[arg(short, long, default_value = "anonymous")]
    username: String,

    /// Run indefinitely with the current settings
    #[arg(short, long)]
    repeat: bool,

    /// Suppress all output
    #[arg(short, long)]
    quiet: bool,

    /// Show additional output
    #[arg(short, long)]
    verbose: bool,

    /// Run parallel with this many threads
    #[arg(short, long, default_value_t = 4)]
    threads: usize,

    /// Run an offline benchmark
    #[arg(short, long)]
    benchmark: Option<BenchmarkMode>,
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

    // Repeat indefinitely if requested
    // Otherwise, run once
    if cli.repeat {
        loop {
            submian(&cli);
        }
    } else {
        submian(&cli);
    }
}

fn submian(cli: &Cli) {
    // Check whether to query the server for a search range or use the benchmark
    let claim_data = if let Some(benchmark) = cli.benchmark {
        get_benchmark_field(benchmark)
    } else {
        get_field_from_server(&cli.mode, &cli.api_base)
    };

    // Print some debug info
    if cli.benchmark.is_some() {
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

    // Submit the results if it's not a benchmark
    if cli.benchmark.is_none() {
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
}
