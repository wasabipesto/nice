use anyhow::Result;
use clap::Parser;
use nice_common::base_range::get_base_range_u128;
use nice_common::msd_prefix_filter::get_filter_effectiveness;
use rand::Rng;
use rand::distr::Distribution;
use rand::distr::weighted::WeightedIndex;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::OpenOptions;
use std::io::Write;
use std::path::Path;
use std::sync::{Arc, Mutex};

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

    /// JSON file for aggregated output
    #[arg(long, default_value = "output/aggregated_stats.json")]
    output_file: String,

    /// Batch size for parallel processing
    #[arg(long, default_value = "100000")]
    batch_size: usize,

    /// Number of chunks per base
    #[arg(long, default_value = "1000")]
    num_chunks: usize,

    /// Minimum samples per chunk - ensures under-sampled chunks get at least one sample per batch
    #[arg(long, default_value = "1")]
    min_samples_per_chunk: u64,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
struct ChunkStats {
    chunk_index: usize,
    chunk_start: f64,
    chunk_end: f64,
    sum: f64,
    count: u64,
}

type AggregatedStats = HashMap<u32, Vec<ChunkStats>>;

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
    if let Some(parent) = Path::new(&args.output_file).parent() {
        std::fs::create_dir_all(parent)?;
    }

    // Load existing aggregated stats or initialize new ones
    let aggregated_stats =
        load_or_initialize_stats(&args.output_file, &bases_with_ranges, args.num_chunks)?;
    let stats = Arc::new(Mutex::new(aggregated_stats));

    println!("Starting MSD filter effectiveness sampling...");
    println!("Bases: {} to {}", args.min, args.max);
    println!("Output file: {}", args.output_file);
    println!("Valid bases found: {}", bases_with_ranges.len());
    println!("Batch size: {}", args.batch_size);
    println!("Chunks per base: {}", args.num_chunks);

    // Loop indefinitely
    loop {
        // First, generate samples for under-sampled chunks if needed
        let mut forced_samples: Vec<(u32, usize, f64)> = Vec::new();

        if args.min_samples_per_chunk > 0 {
            let stats_guard = stats.lock().unwrap();

            for (base, range_start, range_end) in &bases_with_ranges {
                if let Some(chunks) = stats_guard.get(base) {
                    let chunk_size = (*range_end - *range_start) as f64 / args.num_chunks as f64;

                    for (chunk_idx, chunk) in chunks.iter().enumerate() {
                        if chunk.count < args.min_samples_per_chunk {
                            // Generate a sample for this under-sampled chunk
                            let mut rng = rand::rng();

                            // Pick a random number within this chunk's range
                            let chunk_start_offset = chunk_idx as f64 * chunk_size;
                            let offset_in_chunk: u128 = rng.random_range(0..chunk_size as u128);
                            let offset = chunk_start_offset as u128 + offset_in_chunk;
                            let num_start = *range_start + offset;

                            // Calculate effectiveness
                            let effectiveness = get_filter_effectiveness(num_start, *base);

                            forced_samples.push((*base, chunk_idx, effectiveness));
                        }
                    }
                }
            }
        }

        // Clone data for use in parallel closure
        let bases_clone = bases_with_ranges.clone();
        let weighted_clone = weighted_index.clone();
        let stats_clone = Arc::clone(&stats);
        let num_chunks = args.num_chunks;

        // Calculate how many regular samples we need (batch_size minus forced samples)
        let regular_sample_count = args.batch_size.saturating_sub(forced_samples.len());

        // Process regular samples in parallel
        let mut batch_updates: Vec<(u32, usize, f64)> = (0..regular_sample_count)
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

                // Calculate chunk index
                let chunk_size = (range_end - range_start) as f64 / num_chunks as f64;
                let chunk_idx = ((offset as f64 / chunk_size).floor() as usize).min(num_chunks - 1);

                (base, chunk_idx, effectiveness)
            })
            .collect();

        // Add forced samples to batch updates
        batch_updates.extend(forced_samples);

        // Update aggregated stats
        {
            let mut stats_guard = stats_clone.lock().unwrap();
            for (base, chunk_idx, effectiveness) in batch_updates {
                if let Some(chunks) = stats_guard.get_mut(&base) {
                    chunks[chunk_idx].sum += effectiveness;
                    chunks[chunk_idx].count += 1;
                }
            }
        }

        // Save aggregated stats to disk
        {
            let stats_guard = stats.lock().unwrap();
            save_stats(&args.output_file, &stats_guard)?;
        }

        // Print progress
        let total_samples: u64 = stats
            .lock()
            .unwrap()
            .values()
            .flat_map(|chunks| chunks.iter().map(|c| c.count))
            .sum();

        println!(
            "Processed batch of {} samples. Total samples: {}",
            args.batch_size, total_samples
        );
    }
}

fn load_or_initialize_stats(
    output_file: &str,
    bases_with_ranges: &[(u32, u128, u128)],
    num_chunks: usize,
) -> Result<AggregatedStats> {
    // Try to load existing stats
    if Path::new(output_file).exists() {
        println!("Loading existing aggregated stats from {}", output_file);
        let content = std::fs::read_to_string(output_file)?;
        let stats: AggregatedStats = serde_json::from_str(&content)?;

        // Validate that all bases are present
        let mut valid = true;
        for (base, _, _) in bases_with_ranges {
            if !stats.contains_key(base) {
                println!(
                    "Warning: Base {} missing from existing stats, reinitializing",
                    base
                );
                valid = false;
                break;
            }
        }

        if valid {
            println!("Successfully loaded existing stats");
            return Ok(stats);
        }
    }

    // Initialize new stats
    println!("Initializing new aggregated stats");
    let mut stats = AggregatedStats::new();

    for (base, range_start, range_end) in bases_with_ranges {
        let chunk_size = (*range_end - *range_start) as f64 / num_chunks as f64;
        let mut chunks = Vec::with_capacity(num_chunks);

        for i in 0..num_chunks {
            let chunk_start = *range_start as f64 + i as f64 * chunk_size;
            let chunk_end = *range_start as f64 + (i + 1) as f64 * chunk_size;

            chunks.push(ChunkStats {
                chunk_index: i,
                chunk_start,
                chunk_end,
                sum: 0.0,
                count: 0,
            });
        }

        stats.insert(*base, chunks);
    }

    Ok(stats)
}

fn save_stats(output_file: &str, stats: &AggregatedStats) -> Result<()> {
    // Write to a temporary file first, then atomically rename to prevent data loss
    // if the process is interrupted during the write
    let temp_file = format!("{}.tmp", output_file);

    let mut file = OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(true)
        .open(&temp_file)?;

    let json = serde_json::to_string_pretty(stats)?;
    write!(file, "{}", json)?;

    // Ensure all data is flushed to disk before renaming
    file.sync_all()?;

    // Atomically replace the old file with the new one
    std::fs::rename(&temp_file, output_file)?;

    Ok(())
}
