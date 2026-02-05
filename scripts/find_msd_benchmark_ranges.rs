#!/usr/bin/env rust-script
//! ```cargo
//! [dependencies]
//! nice_common = { path = "../common" }
//! clap = { version = "4.5", features = ["env", "derive"] }
//! env_logger = { version = "0.11" }
//! log = { version = "0.4" }
//! malachite = { version = "0.9" }
//! malachite-nz = { version = "0.9", features = ["enable_serde"] }
//! ```

use clap::Parser;
use log::info;
use malachite::base::num::arithmetic::traits::Pow;
use malachite::base::num::conversion::traits::Digits;
use malachite::natural::Natural;

use nice_common::base_range::get_base_range_u128;
use nice_common::msd_prefix_filter::{get_valid_ranges_recursive, has_duplicate_msd_prefix};
use nice_common::{
    MSD_RECURSIVE_MAX_DEPTH, MSD_RECURSIVE_MIN_RANGE_SIZE, MSD_RECURSIVE_SUBDIVISION_FACTOR,
};

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
pub struct Cli {
    /// Number of sample points to test across the range
    #[arg(short, long, default_value = "100")]
    samples: usize,

    /// Target range size (default 1e11)
    #[arg(short, long, default_value = "100000000000")]
    target_size: u128,

    /// Maximum allowed subdivision effectiveness for worst case
    /// Ranges with higher elimination rates will be rejected as worst-case candidates
    #[arg(long, default_value = "0.50")]
    max_subdivision_tolerance: f64,
}

struct BenchmarkRange {
    start: u128,
    end: u128,
    size: u128,
    filter_effective: bool,
    subdivision_effectiveness: f64, // 0.0 = no elimination, 1.0 = complete elimination
    ranges_after_subdivision: usize,
}

impl BenchmarkRange {
    fn new(start: u128, size: u128, base: u32) -> Self {
        let end = start + size;
        let filter_effective = has_duplicate_msd_prefix(start, end, base);

        // Test subdivision effectiveness
        let valid_ranges = get_valid_ranges_recursive(
            start,
            end,
            base,
            0,
            //MSD_RECURSIVE_MAX_DEPTH,
            20,
            MSD_RECURSIVE_MIN_RANGE_SIZE,
            MSD_RECURSIVE_SUBDIVISION_FACTOR,
        );

        let ranges_after_subdivision = valid_ranges.len();
        let total_valid_size: u128 = valid_ranges.iter().map(|(s, e)| e - s).sum();
        let subdivision_effectiveness = 1.0 - (total_valid_size as f64 / size as f64);

        Self {
            start,
            end,
            size,
            filter_effective,
            subdivision_effectiveness,
            ranges_after_subdivision,
        }
    }

    fn width_order(&self) -> u32 {
        (self.size as f64).log10() as u32
    }
}

fn analyze_prefix_info(range_start: u128, range_end: u128, base: u32) -> String {
    let range_start_square = Natural::from(range_start).pow(2).to_digits_asc(&base);
    let range_end_square = Natural::from(range_end - 1).pow(2).to_digits_asc(&base);
    let range_start_cube = Natural::from(range_start).pow(3).to_digits_asc(&base);
    let range_end_cube = Natural::from(range_end - 1).pow(3).to_digits_asc(&base);

    let square_prefix = find_common_msd_prefix(&range_start_square, &range_end_square);
    let cube_prefix = find_common_msd_prefix(&range_start_cube, &range_end_cube);

    let has_dup_square = has_duplicate_digits(&square_prefix);
    let has_dup_cube = has_duplicate_digits(&cube_prefix);
    let has_overlap = has_overlapping_digits(&square_prefix, &cube_prefix);

    let mut reasons = Vec::new();
    if has_dup_square {
        reasons.push(format!(
            "square prefix has duplicates (len={})",
            square_prefix.len()
        ));
    }
    if has_dup_cube {
        reasons.push(format!(
            "cube prefix has duplicates (len={})",
            cube_prefix.len()
        ));
    }
    if has_overlap {
        reasons.push(format!(
            "square/cube prefixes overlap (lens={}/{})",
            square_prefix.len(),
            cube_prefix.len()
        ));
    }

    if reasons.is_empty() {
        format!(
            "no early exit (square prefix len={}, cube prefix len={})",
            square_prefix.len(),
            cube_prefix.len()
        )
    } else {
        reasons.join(", ")
    }
}

// Helper functions from msd_prefix_filter (duplicated to analyze)
fn find_common_msd_prefix(digits1: &[u32], digits2: &[u32]) -> Vec<u32> {
    let len1 = digits1.len();
    let len2 = digits2.len();
    let mut common_prefix = Vec::new();

    let min_len = len1.min(len2);
    for i in 0..min_len {
        let idx1 = len1 - 1 - i;
        let idx2 = len2 - 1 - i;
        if digits1[idx1] == digits2[idx2] {
            common_prefix.push(digits1[idx1]);
        } else {
            break;
        }
    }

    common_prefix
}

fn has_duplicate_digits(digits: &[u32]) -> bool {
    let mut seen = vec![false; 256];
    for &digit in digits {
        if digit < 256 {
            if seen[digit as usize] {
                return true;
            }
            seen[digit as usize] = true;
        }
    }
    false
}

fn has_overlapping_digits(digits1: &[u32], digits2: &[u32]) -> bool {
    let mut seen = vec![false; 256];
    for &digit in digits1 {
        if digit < 256 {
            seen[digit as usize] = true;
        }
    }
    for &digit in digits2 {
        if digit < 256 && seen[digit as usize] {
            return true;
        }
    }
    false
}

fn main() {
    let cli = Cli::parse();
    env_logger::init();

    let base = 50u32;
    info!("=== MSD Prefix Filter Benchmark Range Finder ===");
    info!("Base: {}", base);
    info!(
        "Target range size: {} (1e{})",
        cli.target_size,
        cli.target_size.ilog10()
    );
    info!("Sample points: {}", cli.samples);
    info!(
        "Max subdivision tolerance for worst case: {:.1}%",
        cli.max_subdivision_tolerance * 100.0
    );

    let base_range = match get_base_range_u128(base) {
        Ok(Some(range)) => range,
        Ok(None) => {
            eprintln!("No base range defined for base {}", base);
            return;
        }
        Err(e) => {
            eprintln!("Error getting base range: {}", e);
            return;
        }
    };

    println!("\nBase {} valid range:", base);
    println!("  Start: {}", base_range.range_start);
    println!("  End:   {}", base_range.range_end);
    println!(
        "  Size:  {} (1e{})",
        base_range.range_size,
        base_range.range_size.ilog10()
    );
    println!();

    let mut best_effective: Option<BenchmarkRange> = None;
    let mut best_ineffective: Option<BenchmarkRange> = None;

    // Try different range sizes, starting from largest
    let target_sizes = vec![cli.target_size, cli.target_size / 10, cli.target_size / 100];

    for &target_size in &target_sizes {
        println!(
            "\n=== Testing ranges of size {} (1e{}) ===\n",
            target_size,
            target_size.ilog10()
        );

        let step_size = (base_range.range_size - target_size) / cli.samples as u128;

        for i in 0..cli.samples {
            let range_start = base_range.range_start + (i as u128 * step_size);

            // Make sure we don't exceed the valid range
            if range_start + target_size > base_range.range_end {
                break;
            }

            let benchmark = BenchmarkRange::new(range_start, target_size, base);

            // Check if this is a better example
            if benchmark.filter_effective {
                let is_better = best_effective
                    .as_ref()
                    .map_or(true, |b| benchmark.size > b.size);
                if is_better {
                    let info = analyze_prefix_info(benchmark.start, benchmark.end, base);
                    println!(
                        "✓ NEW BEST EFFECTIVE: [{}, {}) size=1e{} - {}",
                        benchmark.start,
                        benchmark.end,
                        benchmark.width_order(),
                        info
                    );
                    best_effective = Some(benchmark);
                }
            } else {
                // Only consider ranges that meet the subdivision resistance criteria
                if benchmark.subdivision_effectiveness <= cli.max_subdivision_tolerance {
                    // For ineffective ranges, prefer ones that also resist subdivision
                    let is_better = best_ineffective.as_ref().map_or(true, |b| {
                        // First priority: larger size
                        if benchmark.size != b.size {
                            benchmark.size > b.size
                        } else {
                            // Second priority: lower subdivision effectiveness (more resistant)
                            benchmark.subdivision_effectiveness < b.subdivision_effectiveness
                        }
                    });
                    if is_better {
                        let info = analyze_prefix_info(benchmark.start, benchmark.end, base);
                        println!(
                        "✗ NEW BEST INEFFECTIVE: [{}, {}) size=1e{} - {} | subdivision: {:.1}% eliminated, {} ranges remain",
                        benchmark.start,
                        benchmark.end,
                        benchmark.width_order(),
                        info,
                        benchmark.subdivision_effectiveness * 100.0,
                        benchmark.ranges_after_subdivision
                    );
                        best_ineffective = Some(benchmark);
                    }
                }
            }

            // Progress indicator every 1%
            let percent_complete = (i + 1) as f64 / cli.samples as f64 * 100.0;
            if percent_complete.fract() == 0.0 {
                println!(
                    "  ... tested {}/{} ({:.1}%) sample points",
                    i + 1,
                    cli.samples,
                    percent_complete
                );
            }
        }

        // If we found both types at this size, we're done
        if let (Some(ref eff), Some(ref ineff)) = (&best_effective, &best_ineffective) {
            if eff.size == target_size && ineff.size == target_size {
                println!(
                    "\n✓ Found both range types at target size 1e{}",
                    target_size.ilog10()
                );
                break;
            }
        }
    }

    // Print final results
    println!("\n");
    println!("╔═══════════════════════════════════════════════════════════════════════╗");
    println!("║                   MSD FILTER BENCHMARK RESULTS                        ║");
    println!("╚═══════════════════════════════════════════════════════════════════════╝");
    println!();

    if let Some(ref eff) = best_effective {
        println!("BEST CASE (Filter is EFFECTIVE - range can be skipped):");
        println!("  Range: [{}, {})", eff.start, eff.end);
        println!("  Size:  {} (1e{})", eff.size, eff.width_order());
        println!("  Why:   {}", analyze_prefix_info(eff.start, eff.end, base));
        println!();
    } else {
        println!("BEST CASE: No effective range found");
        println!();
    }

    if let Some(ref ineff) = best_ineffective {
        println!("WORST CASE (Filter is INEFFECTIVE - range must be processed):");
        println!("  Range: [{}, {})", ineff.start, ineff.end);
        println!("  Size:  {} (1e{})", ineff.size, ineff.width_order());
        println!(
            "  Why:   {}",
            analyze_prefix_info(ineff.start, ineff.end, base)
        );
        println!(
            "  Subdivision: {:.1}% eliminated by recursive filter",
            ineff.subdivision_effectiveness * 100.0
        );
        println!(
            "  Ranges after subdivision: {}",
            ineff.ranges_after_subdivision
        );
        println!(
            "  Net processing required: {:.1}% of original range",
            (1.0 - ineff.subdivision_effectiveness) * 100.0
        );
        println!();
    } else {
        println!(
            "WORST CASE: No ineffective range found that meets subdivision resistance criteria"
        );
        println!(
            "  (Try increasing --max-subdivision-tolerance above {:.1}%)",
            cli.max_subdivision_tolerance * 100.0
        );
        println!();
    }

    println!("Copy these ranges into your benchmark configuration:");
    println!();
    if let Some(ref eff) = best_effective {
        println!("  // Best case: filter eliminates entire range");
        println!("  let best_case_start = {};", eff.start);
        println!("  let best_case_end = {};", eff.end);
        println!();
    }
    if let Some(ref ineff) = best_ineffective {
        println!("  // Worst case: filter cannot help, full processing needed");
        println!("  let worst_case_start = {};", ineff.start);
        println!("  let worst_case_end = {};", ineff.end);

        println!("  // Worse case filter effectiveness at varying depth:");
        for depth in 0..21 {
            let valid_ranges = get_valid_ranges_recursive(
                ineff.start,
                ineff.end,
                base,
                0,
                depth,
                MSD_RECURSIVE_MIN_RANGE_SIZE,
                MSD_RECURSIVE_SUBDIVISION_FACTOR,
            );
            let ranges_after_subdivision = valid_ranges.len();
            let total_valid_size: u128 = valid_ranges.iter().map(|(s, e)| e - s).sum();
            let subdivision_effectiveness = 1.0 - (total_valid_size as f64 / ineff.size as f64);
            let curstring = if MSD_RECURSIVE_MAX_DEPTH == depth {
                "(current setting)"
            } else {
                ""
            };

            println!(
                "  // Depth {}: {} ranges, {:.1}% effective {}",
                depth,
                ranges_after_subdivision,
                subdivision_effectiveness * 100.0,
                curstring
            );
        }
        println!();
    }
}
