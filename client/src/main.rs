//! A simple CLI for the nice library.

extern crate nice_common;
use nice_common::client_api::get_field_benchmark;
use nice_common::client_api::get_field_from_server;
use nice_common::client_api::submit_field_to_server;
use nice_common::client_process::process_detailed;
use nice_common::client_process::process_niceonly;
use nice_common::{FieldSubmit, SearchMode};

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

    /// Run an offline benchmark [default: base 40, range 100000]
    #[arg(long)]
    benchmark: bool,

    /// Process the range in parallel, improving speed
    #[arg(long)]
    parallel: bool,
}

fn main() {
    // parse args from command line
    let cli = Cli::parse();

    // check whether to query the server for a search range or use the benchmark
    let claim_data = if cli.benchmark {
        get_field_benchmark()
    } else {
        get_field_from_server(&cli.mode, &cli.api_base, &cli.username)
    };

    // print some debug info
    // TODO: implement a pretty print for claim/submit data
    if !cli.quiet {
        println!("{:?}", claim_data);
    }

    // start the timer for benchmarking
    let before = Instant::now();

    // process range & compile results
    let submit_data: FieldSubmit = match cli.mode {
        SearchMode::Detailed => process_detailed(&claim_data, cli.parallel),
        SearchMode::Niceonly => process_niceonly(&claim_data, cli.parallel),
    };

    // stop the benchmarking timer
    let elapsed_seconds = before.elapsed().as_secs_f64();

    // print some debug info
    if !cli.quiet {
        println!("{:?}", submit_data);
    }

    // print the benchmarking results
    if cli.benchmark || cli.verbose {
        println!("Elapsed time: {:.3?}", before.elapsed());
        println!(
            "Hash rate:    {:.3e}",
            f64::try_from(&claim_data.search_range).unwrap() / elapsed_seconds
        );
    }

    // print some debug info
    if !cli.benchmark {
        submit_field_to_server(&cli.mode, &cli.api_base, submit_data)
    }
}
