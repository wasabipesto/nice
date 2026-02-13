//! MSD Filter Effectiveness Experiment with SQLite Caching
//!
//! This tool computes the effectiveness of the MSD (Most Significant Digit) filter
//! across different bases, with caching support for resumability and progressive refinement.

mod compute;
mod db;

use compute::flush_write_buffer;

use anyhow::{Context, Result};
use clap::Parser;
use rayon::prelude::*;
use std::path::PathBuf;
use std::time::Instant;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Base to compute (or start of range if --base-end is specified)
    #[arg(short, long, default_value_t = 10)]
    base: u32,

    /// End of base range (exclusive). If specified, computes all bases from --base to this value.
    #[arg(long)]
    base_end: Option<u32>,

    /// Maximum recursion depth
    #[arg(short, long, default_value_t = 20)]
    max_depth: u32,

    /// Minimum range size before stopping recursion
    #[arg(short = 'r', long, default_value_t = 10_000)]
    min_range_size: u128,

    /// Subdivision factor (how many parts to split each range into)
    #[arg(short, long, default_value_t = 2)]
    subdivision_factor: usize,

    /// Path to SQLite database file
    #[arg(short, long, default_value = "msd_cache.db")]
    db_path: PathBuf,

    /// Use parallel processing for multiple bases
    #[arg(short, long)]
    parallel: bool,

    /// Show statistics for cached data instead of computing
    #[arg(long)]
    stats: bool,

    /// Export cached data for a base to JSON
    #[arg(long)]
    export: Option<u32>,

    /// Export summary of all cached bases to a single JSON file
    #[arg(long)]
    export_all: bool,

    /// Output path for export-all (default: msd_filter_results.json)
    #[arg(long, default_value = "cache/msd_filter_results.json")]
    export_all_path: PathBuf,

    /// Clear cache for a specific base
    #[arg(long)]
    clear_cache: Option<u32>,

    /// Verbose output
    #[arg(short, long)]
    verbose: bool,
}

fn main() -> Result<()> {
    let args = Args::parse();

    // Initialize database
    println!("Initializing database at: {}", args.db_path.display());
    let pool = db::init_db(&args.db_path).context("Failed to initialize database")?;
    println!("Database initialized successfully.\n");

    // Handle special modes
    if args.stats {
        return show_stats(&pool, args.base, args.base_end);
    }

    if let Some(export_base) = args.export {
        return export_base_data(&pool, export_base);
    }

    if args.export_all {
        return export_all_bases(&pool, &args.export_all_path);
    }

    if let Some(clear_base) = args.clear_cache {
        return clear_cache(&pool, clear_base);
    }

    // Determine base range
    let base_start = args.base;
    let base_end = args.base_end.unwrap_or(args.base + 1);

    if base_start >= base_end {
        anyhow::bail!("Invalid base range: {} to {}", base_start, base_end);
    }

    let bases: Vec<u32> = (base_start..base_end).collect();

    println!("Computing MSD filter effectiveness");
    println!("===================================");
    println!(
        "Bases: {} to {} ({} total)",
        base_start,
        base_end - 1,
        bases.len()
    );
    println!("Max depth: {}", args.max_depth);
    println!("Min range size: {}", args.min_range_size);
    println!("Subdivision factor: {}", args.subdivision_factor);
    println!("Parallel: {}", args.parallel);
    println!();

    let overall_start = Instant::now();

    if args.parallel && bases.len() > 1 {
        // Parallel processing
        println!("Processing bases in parallel...\n");

        bases.par_iter().for_each(|&base| {
            match compute_and_report_base(
                &pool,
                base,
                args.max_depth,
                args.min_range_size,
                args.subdivision_factor,
                args.verbose,
            ) {
                Ok(_) => {
                    // Flush any remaining buffered writes for this thread
                    if let Err(e) = flush_write_buffer(&pool) {
                        eprintln!("Error flushing write buffer for base {}: {}", base, e);
                    }
                }
                Err(e) => eprintln!("Error processing base {}: {}", base, e),
            }
        });
    } else {
        // Sequential processing
        for base in bases {
            compute_and_report_base(
                &pool,
                base,
                args.max_depth,
                args.min_range_size,
                args.subdivision_factor,
                args.verbose,
            )?;
            // Flush any remaining buffered writes
            flush_write_buffer(&pool)?;
        }
    }

    let overall_elapsed = overall_start.elapsed();
    println!("\n===================================");
    println!("Total time: {:.2}s", overall_elapsed.as_secs_f64());
    println!("Database: {}", args.db_path.display());

    Ok(())
}

fn compute_and_report_base(
    pool: &db::DbPool,
    base: u32,
    max_depth: u32,
    min_range_size: u128,
    subdivision_factor: usize,
    verbose: bool,
) -> Result<()> {
    let base_range = match nice_common::base_range::get_base_range_u128(base)? {
        Some(range) => range,
        None => {
            if verbose {
                println!("Base {}: No valid range (skipped)", base);
            }
            return Ok(());
        }
    };

    let total_size = base_range.size();

    if verbose {
        println!("Base {}: Starting computation...", base);
        println!(
            "  Range: [{}, {}) → {:.3e} numbers",
            base_range.start(),
            base_range.end(),
            total_size as f64
        );
    }

    let start = Instant::now();

    let valid_size =
        compute::compute_base(pool, base, max_depth, min_range_size, subdivision_factor)?;

    let elapsed = start.elapsed();
    let filtered_size = total_size - valid_size;
    let filtered_pct = (filtered_size as f64 / total_size as f64) * 100.0;

    println!(
        "Base {:3}: {:.3e} → {:.3e} ({:.2}% filtered) in {:.2}s",
        base,
        total_size as f64,
        valid_size as f64,
        filtered_pct,
        elapsed.as_secs_f64()
    );

    Ok(())
}

fn show_stats(pool: &db::DbPool, base_start: u32, base_end: Option<u32>) -> Result<()> {
    let base_end = base_end.unwrap_or(base_start + 1);

    println!("Cached Data Statistics");
    println!("======================\n");

    for base in base_start..base_end {
        if let Some(stats) = db::get_base_stats(pool, base)? {
            println!("Base {}:", base);
            println!("  Cached ranges: {}", stats.num_cached_ranges);
            println!("  Total valid size: {:.3e}", stats.total_valid_size as f64);
            println!("  Average depth: {:.2}", stats.avg_depth);
            println!("  Max depth: {}", stats.max_depth);
            println!();
        } else {
            println!("Base {}: No cached data\n", base);
        }
    }

    Ok(())
}

fn export_base_data(pool: &db::DbPool, base: u32) -> Result<()> {
    println!("Exporting cached data for base {}...", base);

    let ranges = db::get_all_ranges_for_base(pool, base)?;

    if ranges.is_empty() {
        println!("No cached data found for base {}", base);
        return Ok(());
    }

    let output_path = format!("msd_cache_base_{}.json", base);
    let file = std::fs::File::create(&output_path).context("Failed to create output file")?;

    serde_json::to_writer_pretty(file, &ranges).context("Failed to write JSON")?;

    println!("Exported {} ranges to {}", ranges.len(), output_path);

    // Also compute summary statistics from the top-level cached entry
    let base_range = nice_common::base_range::get_base_range_u128(base)?.context("Invalid base")?;
    let total_size = base_range.size();

    // Find the cached entry that matches the full base range
    let valid_size = ranges
        .iter()
        .find(|r| r.range_start == base_range.start() && r.range_end == base_range.end())
        .map(|r| r.valid_size)
        .unwrap_or(0);

    let filtered_pct = if valid_size > 0 {
        ((total_size - valid_size) as f64 / total_size as f64) * 100.0
    } else {
        0.0
    };

    println!("\nSummary:");
    println!("  Total ranges cached: {}", ranges.len());
    println!("  Total size: {:.3e}", total_size as f64);
    println!("  Valid size: {:.3e}", valid_size as f64);
    println!("  Filtered: {:.2}%", filtered_pct);

    Ok(())
}

fn export_all_bases(pool: &db::DbPool, output_path: &PathBuf) -> Result<()> {
    use serde::Serialize;

    #[derive(Serialize)]
    struct MsdFilterResult {
        base: u32,
        total_size: u128,
        valid_size: u128,
        filtered_size: u128,
        filtered_pct: f64,
        max_depth: u32,
        num_cached_ranges: u64,
    }

    println!("Exporting all cached bases to {}...", output_path.display());

    // Get all unique bases from the database
    let conn = pool.get().context("Failed to get connection from pool")?;
    let mut stmt = conn
        .prepare("SELECT DISTINCT base FROM msd_cache ORDER BY base")
        .context("Failed to prepare query")?;

    let bases: Vec<u32> = stmt
        .query_map([], |row| row.get(0))
        .context("Failed to query bases")?
        .collect::<Result<Vec<_>, _>>()
        .context("Failed to collect bases")?;

    drop(stmt);
    drop(conn);

    if bases.is_empty() {
        println!("No cached data found");
        return Ok(());
    }

    println!("Found {} bases with cached data", bases.len());

    let mut results = Vec::new();

    for base in bases {
        let base_range = match nice_common::base_range::get_base_range_u128(base)? {
            Some(range) => range,
            None => continue,
        };

        let total_size = base_range.size();

        // Get the cached entry that matches the full base range
        let ranges = db::get_all_ranges_for_base(pool, base)?;

        let valid_size = ranges
            .iter()
            .find(|r| r.range_start == base_range.start() && r.range_end == base_range.end())
            .map(|r| r.valid_size)
            .unwrap_or(0);

        if valid_size == 0 {
            continue; // Skip bases without top-level cache entry
        }

        let max_depth = ranges.iter().map(|r| r.max_depth).max().unwrap_or(0);
        let filtered_size = total_size - valid_size;
        let filtered_pct = (filtered_size as f64 / total_size as f64) * 100.0;

        results.push(MsdFilterResult {
            base,
            total_size,
            valid_size,
            filtered_size,
            filtered_pct,
            max_depth,
            num_cached_ranges: ranges.len() as u64,
        });
    }

    let file = std::fs::File::create(output_path).context("Failed to create output file")?;
    serde_json::to_writer_pretty(file, &results).context("Failed to write JSON")?;

    println!(
        "Successfully exported {} bases to {}",
        results.len(),
        output_path.display()
    );
    println!("\nSummary:");
    println!("  Bases exported: {}", results.len());
    if let Some(min_depth) = results.iter().map(|r| r.max_depth).min() {
        println!("  Min depth: {}", min_depth);
    }
    if let Some(max_depth) = results.iter().map(|r| r.max_depth).max() {
        println!("  Max depth: {}", max_depth);
    }

    Ok(())
}

fn clear_cache(pool: &db::DbPool, base: u32) -> Result<()> {
    println!("Clearing cache for base {}...", base);

    let affected = db::clear_base_cache(pool, base)?;

    println!("Cleared {} cached entries for base {}", affected, base);

    Ok(())
}
