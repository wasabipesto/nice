//! A simple CLI for the nice library.

extern crate nice_common;
use nice_common::benchmark::{get_benchmark_field, BenchmarkMode};
use nice_common::client_api::get_field_from_server;
use nice_common::client_api::submit_field_to_server;
use nice_common::client_process::process_detailed;
use nice_common::client_process::process_niceonly;
use nice_common::{DataToServer, SearchMode};

extern crate serde_json;
use clap::Parser;
use std::time::Instant;

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
#[command(propagate_version = true)]
pub struct Cli {
    /// The checkout mode to use
    #[arg(value_enum, default_value = "detailed")]
    mode: SearchMode,

    /// The base API URL to connect to
    #[arg(long, default_value = "https://nicenumbers.net/api")]
    api_base: String,

    /// The username to send alongside your contribution
    #[arg(short, long, default_value = "anonymous")]
    username: String,

    /// Suppress some output
    #[arg(short, long)]
    quiet: bool,

    /// Show additional output
    #[arg(short, long)]
    verbose: bool,

    /// Run an offline benchmark
    #[arg(short, long)]
    benchmark: Option<BenchmarkMode>,
}

fn main() {
    // parse args from command line
    let cli = Cli::parse();

    // check whether to query the server for a search range or use the benchmark
    let claim_data = if let Some(benchmark) = cli.benchmark {
        get_benchmark_field(benchmark)
    } else {
        get_field_from_server(&cli.mode, &cli.api_base, &cli.username)
    };

    // print some debug info
    if !cli.quiet {
        println!(
            "Claim Data: {}",
            serde_json::to_string_pretty(&claim_data).unwrap()
        );
    }

    // start the timer for benchmarking
    let before = Instant::now();

    // process range & compile results
    let submit_data: DataToServer = match cli.mode {
        SearchMode::Detailed => process_detailed(&claim_data, &cli.username),
        SearchMode::Niceonly => process_niceonly(&claim_data, &cli.username),
    };

    // stop the benchmarking timer
    let elapsed_seconds = before.elapsed().as_secs_f64();

    // print some debug info
    if !cli.quiet {
        println!(
            "Submit Data: {}",
            serde_json::to_string_pretty(&submit_data).unwrap()
        );
    }

    // print the benchmarking results
    if cli.benchmark.is_some() || cli.verbose {
        println!("Elapsed time:    {:.3?}", before.elapsed());
        println!(
            "Processing rate: {:.3e} numbers/second",
            claim_data.range_size as f64 / elapsed_seconds
        );
    }

    // submit the results
    if cli.benchmark.is_none() {
        submit_field_to_server(&cli.api_base, submit_data)
    }
}
