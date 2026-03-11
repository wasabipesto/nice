use anyhow::Result;
use clap::Parser;
use nice_common::base_range::get_base_range_u128;
use nice_common::msd_prefix_filter::get_filter_effectiveness;
use rand::Rng;
use rand::distr::Distribution;
use rand::distr::weighted::WeightedIndex;
use rayon::prelude::*;
use serde::Serialize;
use std::fs::OpenOptions;
use std::io::Write;

#[derive(Parser)]
#[command(name = "msd_filter_effectiveness")]
#[command(about = "Measure the effectiveness of MSD filters")]
struct Args {
    /// Lowest base to evaluate
    #[arg(long, default_value = "20")]
    min: u32,

    /// Highest base to evaluate
    #[arg(long, default_value = "97")]
    max: u32,

    /// JSONL file for output
    #[arg(long, default_value = "output/msd_filter_samples.jsonl")]
    output_file: String,

    /// Batch size for parallel processing
    #[arg(long, default_value = "100000")]
    batch_size: usize,
}

#[derive(Serialize)]
struct SampleResult {
    base: u32,
    num_start: u128,
    effectiveness: f64,
}

fn main() -> Result<()> {
    let args = Args::parse();

    // Build a list of bases with their ranges and weights
    let mut bases_with_ranges: Vec<(u32, u128, u128)> = Vec::new();

    for base in args.min..=args.max {
        if let Some(range) = get_base_range_u128(base)? {
            bases_with_ranges.push((base, range.start(), range.end()));
        }
    }

    if bases_with_ranges.is_empty() {
        anyhow::bail!("No valid bases found in the specified range");
    }

    // Calculate weights based on range sizes (convert to f64 for WeightedIndex)
    let weights: Vec<f64> = bases_with_ranges
        .iter()
        .map(|(_, start, end)| (*end - *start) as f64)
        .collect();

    // Create weighted distribution for sampling bases
    let weighted_index = WeightedIndex::new(&weights)?;

    // Create output directory if it doesn't exist
    if let Some(parent) = std::path::Path::new(&args.output_file).parent() {
        std::fs::create_dir_all(parent)?;
    }

    println!("Starting MSD filter effectiveness sampling...");
    println!("Bases: {} to {}", args.min, args.max);
    println!("Output file: {}", args.output_file);
    println!("Valid bases found: {}", bases_with_ranges.len());
    println!("Batch size: {}", args.batch_size);

    // Loop indefinitely
    loop {
        // Clone data for use in parallel closure
        let bases_clone = bases_with_ranges.clone();
        let weighted_clone = weighted_index.clone();

        // Process a batch of samples in parallel
        let results: Vec<SampleResult> = (0..args.batch_size)
            .into_par_iter()
            .map(|_| {
                let mut rng = rand::rng();

                // Sample a base weighted by range size
                let idx = weighted_clone.sample(&mut rng);
                let (base, range_start, range_end) = bases_clone[idx];

                // Sample a random number within the range
                let range_size = range_end - range_start;
                let offset: u128 = rng.random_range(0..range_size);
                let num_start = range_start + offset;

                // Calculate effectiveness
                let effectiveness = get_filter_effectiveness(num_start, base);

                SampleResult {
                    base,
                    num_start,
                    effectiveness,
                }
            })
            .collect();

        // Write batch results to file
        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&args.output_file)?;

        for result in &results {
            let json_line = serde_json::to_string(result)?;
            writeln!(file, "{}", json_line)?;
        }

        // Print progress
        match results.last() {
            Some(last) => println!(
                "Processed batch of {} samples, last item was B{} {:.2}%",
                results.len(),
                last.base,
                last.effectiveness * 100.0
            ),
            None => println!("Processed batch but no results!"),
        };
    }
}
