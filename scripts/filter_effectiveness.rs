#!/usr/bin/env rust-script
//! ```cargo
//! [dependencies]
//! nice_common = { path = "../common" }
//! serde = { version = "1.0", features = ["derive"] }
//! serde_json = "1.0"
//! sha2 = "0.10"
//! ```

use nice_common::base_range::get_base_range_u128;
use nice_common::lsd_filter::get_valid_lsds;
use nice_common::msd_prefix_filter::has_duplicate_msd_prefix;
use nice_common::residue_filter::get_residue_filter;
use nice_common::FieldSize;
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::fs::{self, File};
use std::io::{Read, Write};
use std::path::PathBuf;

fn get_cache_hash(base: u32, max_depth: u32, min_size: u128, subdivision_factor: usize) -> String {
    // Create a hash of the arguments
    let mut hasher = Sha256::new();
    hasher.update(base.to_le_bytes());
    hasher.update(max_depth.to_le_bytes());
    hasher.update(min_size.to_le_bytes());
    hasher.update(subdivision_factor.to_le_bytes());
    let hash = hasher.finalize();
    format!("{:x}", hash)
}

pub fn get_valid_ranges_size_recursive(
    range: FieldSize,
    base: u32,
    current_depth: u32,
    max_depth: u32,
    min_range_size: u128,
    subdivision_factor: usize,
) -> u128 {
    // Check if range is too small or we've hit max depth
    if current_depth >= max_depth {
        return range.size();
    }
    if range.size() <= min_range_size {
        return range.size();
    }

    // Check if the entire range can be skipped
    if has_duplicate_msd_prefix(range, base) {
        return 0;
    }

    // Check if subdivision would be worthwhile
    // If the range is not much larger than min_range_size, don't bother subdividing
    if range.size() < min_range_size * (subdivision_factor as u128) {
        return range.size();
    }

    // Subdivide the range and recursively check each part
    let chunk_size = range.size() / (subdivision_factor as u128);
    let mut total_size = 0u128;

    for i in 0..subdivision_factor {
        let sub_start = range.start() + (i as u128) * chunk_size;
        let sub_end = if i == subdivision_factor - 1 {
            range.end() // Last chunk gets any remainder
        } else {
            sub_start + chunk_size
        };
        let sub_range = FieldSize::new(sub_start, sub_end);

        if sub_start < sub_end {
            let sub_size = get_valid_ranges_size_recursive(
                sub_range,
                base,
                current_depth + 1,
                max_depth,
                min_range_size,
                subdivision_factor,
            );
            total_size += sub_size;
        }
    }

    total_size
}

fn get_msd_filtered_valid_range_cached(
    base_range: FieldSize,
    base: u32,
    max_depth: u32,
    min_size: u128,
    subdivision_factor: usize,
) -> u128 {
    let cache_path = PathBuf::from("cache/msd_cache.json");
    let cache_hash = get_cache_hash(base, max_depth, min_size, subdivision_factor);

    // Load existing cache or create new one
    let mut cache: HashMap<String, u128> = if let Ok(mut file) = File::open(&cache_path) {
        let mut contents = String::new();
        if file.read_to_string(&mut contents).is_ok() {
            serde_json::from_str(&contents).unwrap_or_default()
        } else {
            HashMap::new()
        }
    } else {
        HashMap::new()
    };

    // Check if we have a cached result
    if let Some(&size) = cache.get(&cache_hash) {
        return size;
    }

    // Compute the result
    let size = get_valid_ranges_size_recursive(
        base_range,
        base,
        0,
        max_depth,
        min_size,
        subdivision_factor,
    );

    // Update cache and save
    cache.insert(cache_hash, size);

    if let Some(parent) = cache_path.parent() {
        let _ = fs::create_dir_all(parent);
    }

    if let Ok(json) = serde_json::to_string_pretty(&cache) {
        if let Ok(mut file) = File::create(&cache_path) {
            let _ = file.write_all(json.as_bytes());
        }
    }

    size
}

#[derive(Debug, Serialize)]
#[allow(dead_code)]
struct FilterStats {
    base: u32,
    total_numbers: u128,
    lsd_raw_pct: f64,
    lsd_marginal_pct: f64,
    lsd_remaining: u128,
    residue_raw_pct: f64,
    residue_marginal_pct: f64,
    residue_remaining: u128,
    msd_raw_pct: f64,
    msd_marginal_pct: f64,
    msd_remaining: u128,
    total_eliminated_pct: f64,
    reduction_factor: f64,
}

fn main() {
    let mut all_stats = Vec::new();

    println!("Filter Effectiveness Analysis");
    println!("===========================");
    println!();

    for base in 10..=100 {
        let base_range = match get_base_range_u128(base) {
            Ok(Some(range)) => range,
            Ok(None) => continue,
            Err(_) => continue,
        };

        let total_numbers = base_range.size();

        println!("BASE {}", base);
        println!("-----------------------");
        println!(
            "Total range: [{:.3e}, {:.3e}) → {:.3e} numbers",
            base_range.start() as f64,
            base_range.end() as f64,
            total_numbers as f64
        );
        println!();

        // Track how many numbers pass each filter
        let mut remaining = total_numbers;

        // Filter 1: LSD Filter
        let valid_lsds = get_valid_lsds(&base);
        let lsd_pass_count = (total_numbers as f64 * valid_lsds.len() as f64 / base as f64) as u128;
        let lsd_raw_eliminated = total_numbers - lsd_pass_count;
        let lsd_marginal_eliminated = remaining - lsd_pass_count;

        println!("  1. LSD Filter (k=1):");
        println!(
            "     Valid LSDs: {} out of {} ({:.1}% pass)",
            valid_lsds.len(),
            base,
            (valid_lsds.len() as f64 / base as f64) * 100.0
        );
        println!(
            "     Raw efficacy:      {:.3e} eliminated ({:.2}% of original)",
            lsd_raw_eliminated as f64,
            (lsd_raw_eliminated as f64 / total_numbers as f64) * 100.0
        );
        println!(
            "     Marginal efficacy: {:.3e} eliminated ({:.2}% of remaining)",
            lsd_marginal_eliminated as f64,
            (lsd_marginal_eliminated as f64 / remaining as f64) * 100.0
        );
        println!("     Remaining: {:.3e}", lsd_pass_count as f64);
        println!();

        remaining = lsd_pass_count;

        // Filter 2: Residue Filter
        let valid_residues = get_residue_filter(&base);
        let residue_pass_ratio = valid_residues.len() as f64 / (base - 1) as f64;
        let residue_pass_count = (lsd_pass_count as f64 * residue_pass_ratio) as u128;
        let residue_raw_eliminated =
            total_numbers - (total_numbers as f64 * residue_pass_ratio) as u128;
        let residue_marginal_eliminated = remaining - residue_pass_count;

        println!("  2. Residue Filter:");
        println!(
            "     Valid residues: {} out of {} ({:.1}% pass)",
            valid_residues.len(),
            base - 1,
            residue_pass_ratio * 100.0
        );
        println!(
            "     Raw efficacy:      {:.3e} eliminated ({:.2}% of original)",
            residue_raw_eliminated as f64,
            (residue_raw_eliminated as f64 / total_numbers as f64) * 100.0
        );
        println!(
            "     Marginal efficacy: {:.3e} eliminated ({:.2}% of remaining)",
            residue_marginal_eliminated as f64,
            (residue_marginal_eliminated as f64 / remaining as f64) * 100.0
        );
        println!("     Remaining: {:.3e}", residue_pass_count as f64);
        println!();

        remaining = residue_pass_count;

        // Filter 3: MSD Prefix Filter
        let max_depth = 50;
        let min_size = 10_000;
        let subdivision_factor = 2;
        let filtered_valid_range = get_msd_filtered_valid_range_cached(
            base_range,
            base,
            max_depth,
            min_size,
            subdivision_factor,
        );

        let msd_raw_eliminated = total_numbers - filtered_valid_range;

        // Assume raw elimination is independent of other filters, estimate marginal efficacy
        let right_hand_eff = remaining as f64 / total_numbers as f64;
        let msd_marginal_eliminated = (right_hand_eff * msd_raw_eliminated as f64) as u128;
        assert!(msd_marginal_eliminated <= remaining);
        let msd_pass_count = remaining - msd_marginal_eliminated;

        println!("  3. MSD Prefix Filter (Depth {max_depth}):");
        println!(
            "     Raw efficacy:      {:.3e} eliminated ({:.2}% of original)",
            msd_raw_eliminated as f64,
            (msd_raw_eliminated as f64 / total_numbers as f64) * 100.0
        );
        println!(
            "     Marginal efficacy: {:.3e} eliminated ({:.2}% of remaining)",
            msd_marginal_eliminated as f64,
            (msd_marginal_eliminated as f64 / remaining as f64) * 100.0
        );
        println!("     Remaining: {:.3e}", msd_pass_count as f64);
        println!();

        remaining = msd_pass_count;

        // Summary
        let total_eliminated = total_numbers - remaining;
        println!();
        println!("  COMBINED SUMMARY:");
        println!(
            "     Total eliminated:  {:.3e} ({:.2}% of original)",
            total_eliminated as f64,
            (total_eliminated as f64 / total_numbers as f64) * 100.0
        );
        println!(
            "     Final remaining:   {:.3e} ({:.2}% of original)",
            remaining as f64,
            (remaining as f64 / total_numbers as f64) * 100.0
        );
        println!(
            "     Reduction factor:  {:.2}x",
            total_numbers as f64 / remaining as f64
        );
        println!();
        println!("===========================");
        println!();

        // Store stats for summary table
        // Calculate marginal percentages based on what was available before each filter
        let lsd_before = total_numbers;
        let residue_before = lsd_pass_count;
        let msd_before = residue_pass_count;

        all_stats.push(FilterStats {
            base,
            total_numbers,
            lsd_raw_pct: (lsd_raw_eliminated as f64 / total_numbers as f64) * 100.0,
            lsd_marginal_pct: (lsd_marginal_eliminated as f64 / lsd_before as f64) * 100.0,
            lsd_remaining: lsd_pass_count,
            residue_raw_pct: (residue_raw_eliminated as f64 / total_numbers as f64) * 100.0,
            residue_marginal_pct: (residue_marginal_eliminated as f64 / residue_before as f64)
                * 100.0,
            residue_remaining: residue_pass_count,
            msd_raw_pct: (msd_raw_eliminated as f64 / total_numbers as f64) * 100.0,
            msd_marginal_pct: (msd_marginal_eliminated as f64 / msd_before as f64) * 100.0,
            msd_remaining: msd_pass_count,
            total_eliminated_pct: (total_eliminated as f64 / total_numbers as f64) * 100.0,
            reduction_factor: total_numbers as f64 / remaining as f64,
        });
    }

    // Print summary table
    println!();
    println!("SUMMARY TABLE");
    println!("===========================================================================================================");
    println!();
    println!(
        "{:<6} {:>12} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10}",
        "Base",
        "Total",
        "LSD Raw%",
        "LSD Marg%",
        "Res Raw%",
        "Res Marg%",
        "MSD Raw%",
        "MSD Marg%",
        "Total%",
        "Factor"
    );
    println!(
        "{:<6} {:>12} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10}",
        "", "Numbers", "", "", "", "", "", "", "Elim", ""
    );
    println!("------------------------------------------------------------------------------------------------------------");

    for stats in &all_stats {
        println!("{:<6} {:>12.2e} {:>10.2} {:>10.2} {:>10.2} {:>10.2} {:>10.2} {:>10.2} {:>10.2} {:>10.2}",
            stats.base,
            stats.total_numbers as f64,
            stats.lsd_raw_pct,
            stats.lsd_marginal_pct,
            stats.residue_raw_pct,
            stats.residue_marginal_pct,
            stats.msd_raw_pct,
            stats.msd_marginal_pct,
            stats.total_eliminated_pct,
            stats.reduction_factor,
        );
    }

    println!("===========================================================================================================");
    println!();
    println!("Legend:");
    println!("  Raw%   = % of total range eliminated by filter alone");
    println!("  Marg%  = % of remaining candidates eliminated by filter");
    println!("  Total% Elim = % of total range eliminated by all filters combined");
    println!("  Factor     = Reduction factor (original size / final size)");
    println!();

    // Save data to JSON file
    let json_output = serde_json::to_string_pretty(&all_stats).expect("Failed to serialize data");
    let mut file =
        File::create("output/filter_effectiveness.json").expect("Failed to create JSON file");
    file.write_all(json_output.as_bytes())
        .expect("Failed to write JSON file");
    println!("Data saved to filter_effectiveness.json");
}
