#!/usr/bin/env rust-script
//! ```cargo
//! [dependencies]
//! nice_common = { path = "../common" }
//! serde = { version = "1.0", features = ["derive"] }
//! serde_json = "1.0"
//! ```

use nice_common::base_range::get_base_range_u128;
use nice_common::lsd_filter::get_valid_lsds;
use nice_common::residue_filter::get_residue_filter;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::File;
use std::io::{Read, Write};

#[derive(Debug, Deserialize, Clone)]
struct ChunkStats {
    chunk_index: usize,
    chunk_start: f64,
    chunk_end: f64,
    sum: f64,
    count: u64,
}

type AggregatedStats = HashMap<u32, Vec<ChunkStats>>;

fn load_msd_stats() -> Result<AggregatedStats, String> {
    let path = "output/aggregated_stats.json";
    let mut file = File::open(path).map_err(|e| format!("Failed to open {}: {}", path, e))?;
    let mut contents = String::new();
    file.read_to_string(&mut contents)
        .map_err(|e| format!("Failed to read {}: {}", path, e))?;
    serde_json::from_str(&contents).map_err(|e| format!("Failed to parse {}: {}", path, e))
}

fn get_msd_effectiveness(msd_stats: &AggregatedStats, base: u32) -> Option<f64> {
    msd_stats.get(&base).map(|chunks| {
        let total_count: u64 = chunks.iter().map(|c| c.count).sum();
        if total_count == 0 {
            return 0.0;
        }
        let total_sum: f64 = chunks.iter().map(|c| c.sum).sum();
        1.0 - total_sum / total_count as f64
    })
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
    // Load MSD filter effectiveness data from sampling
    let msd_stats = match load_msd_stats() {
        Ok(stats) => {
            println!(
                "Loaded MSD filter effectiveness data for {} bases",
                stats.len()
            );
            stats
        }
        Err(e) => {
            eprintln!("Error: {}", e);
            eprintln!(
                "Please run the msd_filter_effectiveness sampler first to generate the data."
            );
            return;
        }
    };

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

        // Filter 3: MSD Prefix Filter (from sampled data)
        let msd_effectiveness = match get_msd_effectiveness(&msd_stats, base) {
            Some(eff) => eff,
            None => {
                println!(
                    "  3. MSD Prefix Filter: No sampled data available for base {}",
                    base
                );
                println!();
                continue;
            }
        };

        // msd_effectiveness is the fraction that PASS the filter
        let msd_pass_count = (remaining as f64 * msd_effectiveness) as u128;
        let msd_marginal_eliminated = remaining - msd_pass_count;
        let msd_raw_eliminated = (total_numbers as f64 * (1.0 - msd_effectiveness)) as u128;

        println!("  3. MSD Prefix Filter (Sampled):");
        println!(
            "     Pass rate:         {:.2}% (from sampling)",
            msd_effectiveness * 100.0
        );
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
