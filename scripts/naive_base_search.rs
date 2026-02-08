#!/usr/bin/env rust-script
//! ```cargo
//! [dependencies]
//! nice_common = { path = "../common" }
//! clap = { version = "4.5", features = ["env", "derive"] }
//! env_logger = { version = "0.11" }
//! rayon = { version = "1.11" }
//! log = { version = "0.4" }
//! simple-tqdm = {version = "0.2", features = ["rayon"]}
//! ```

use clap::Parser;
use log::{info, warn};
use rayon::prelude::*;
use simple_tqdm::ParTqdm;

use nice_common::base_range::get_base_range_u128;
use nice_common::client_process::process_range_niceonly;
use nice_common::stride_filter;
use nice_common::{FieldResults, FieldSize, PROCESSING_CHUNK_SIZE};
use std::sync::Arc;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
#[command(propagate_version = true)]
#[allow(clippy::struct_excessive_bools)]
pub struct Cli {
    /// The base to search
    #[arg(short, long)]
    base: u32,
}

/// Break up the range into chunks, returning FieldSize for each.
fn chunked_ranges(range_start: u128, range_end: u128, chunk_size: usize) -> Vec<FieldSize> {
    let mut chunks = Vec::new();
    let mut start = range_start;

    while start < range_end {
        let end = (start + chunk_size as u128).min(range_end);
        chunks.push(FieldSize::new(start, end));
        start = end;
    }

    chunks
}

fn main() {
    // Parse command line arguments
    let cli = Cli::parse();

    // Initialize logger from environment variables (RUST_LOG)
    env_logger::init();

    let base = cli.base;
    info!("=== Starting Nice Number Search ===");
    info!("Base: {}", base);

    // Get the valid range of numbers we should search within for this base
    let base_range = match get_base_range_u128(base) {
        Ok(Some(range)) => range,
        Ok(None) => {
            warn!("No base range defined for base {}", base);
            return;
        }
        Err(e) => {
            warn!("Error getting base range: {}", e);
            return;
        }
    };

    // Break up the range into chunks
    let chunk_size = 100 * PROCESSING_CHUNK_SIZE;
    let chunks = chunked_ranges(base_range.range_start, base_range.range_end, chunk_size);

    // Precompute stride table once and share across all chunks
    let k = 1; // Number of digits for multi-digit LSD filter
    let stride_table = Arc::new(stride_filter::StrideTable::new(base, k));

    // Configure TQDM
    let chunk_scale = (chunk_size as f32).log10() as u32;
    let tqdm_config = simple_tqdm::Config::new().with_unit(format!("e{chunk_scale}"));

    // Process each chunk and gather the results
    let results: Vec<FieldResults> = chunks
        .par_iter()
        .tqdm_config(tqdm_config)
        .map(|chunk| process_range_niceonly(chunk, base, &stride_table))
        .collect();

    // Compile results from all chunks
    let nice_numbers: Vec<_> = results
        .iter()
        .flat_map(|result| result.nice_numbers.clone())
        .collect();

    println!();
    println!();
    println!("╔════════════════════════════════════════╗");
    println!("║  Nice Number Search Results (Base {})  ║", base);
    println!("╚════════════════════════════════════════╝");
    println!();

    if nice_numbers.is_empty() {
        println!("  No nice numbers found in the search range.");
    } else {
        println!("  Found {} nice number(s):\n", nice_numbers.len());
        for (index, number) in nice_numbers.into_iter().enumerate() {
            println!("    {}. {}", index + 1, number.number);
        }
    }
    println!();
}
