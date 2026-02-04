//! Nice Number Radix Tree Search Algorithm
//!
//! This program searches for "nice numbers" in a given base using a backtracking algorithm
//! with pruning
//!
//! The algorithm builds candidate numbers digit-by-digit from least significant to most
//! significant, pruning branches early when digit collisions are detected.

use clap::Parser;
use log::{debug, info, trace, warn};
use malachite::base::num::arithmetic::traits::{DivRem, Pow};
use malachite::natural::Natural;
use rayon::prelude::*;
use std::sync::Arc;

use nice_common::FieldSize;
use nice_common::base_range::get_base_range_u128;
use nice_common::client_process::get_is_nice;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
#[command(propagate_version = true)]
#[allow(clippy::struct_excessive_bools)]
pub struct Cli {
    /// The base to search
    #[arg(short, long)]
    base: u32,
}

/// A bitmask to track which digits (0 through base-1) have been used.
/// Each bit position represents whether a particular digit has appeared.
type DigitMask = u128;

/// Statistics tracking for search tree exploration.
#[derive(Debug, Default, Clone)]
struct SearchStats {
    /// Total nodes explored in the search tree
    nodes_explored: u64,

    /// Nodes pruned due to collision detection
    nodes_pruned: u64,

    /// Candidates tested with get_is_nice()
    candidates_tested: u64,

    /// Candidates skipped due to range filtering
    candidates_skipped_range: u64,

    /// Candidates skipped due to leading zeros
    candidates_skipped_leading_zeros: u64,
}

impl SearchStats {
    fn new() -> Self {
        Self::default()
    }

    fn log_summary(&self, worker_id: u32) {
        let total_nodes = self.nodes_explored + self.nodes_pruned;
        let prune_rate = if total_nodes > 0 {
            (self.nodes_pruned as f64 / total_nodes as f64) * 100.0
        } else {
            0.0
        };

        debug!(
            "Worker {} stats: explored={}, pruned={} ({:.1}%), tested={}, skipped_range={}, skipped_zeros={}",
            worker_id,
            self.nodes_explored,
            self.nodes_pruned,
            prune_rate,
            self.candidates_tested,
            self.candidates_skipped_range,
            self.candidates_skipped_leading_zeros
        );
    }
}

/// Configuration and state for searching nice numbers in a specific base.
struct NiceNumberSearcher {
    /// The numeric base we're working in (e.g., 10 for decimal, 40 for base-40)
    base: u32,

    /// Minimum number of digits in candidate numbers to test
    min_candidate_digits: u32,

    /// Maximum number of digits in candidate numbers to test
    max_candidate_digits: u32,

    /// Precomputed powers of the base: [base^0, base^1, base^2, ...]
    /// Used for efficient digit extraction and number construction
    base_powers: Vec<u128>,

    /// Minimum valid value in the search range
    range_start: u128,

    /// Maximum valid value in the search range
    range_end: u128,
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

    info!(
        "Search range: {} to {}",
        base_range.range_start, base_range.range_end
    );

    // Calculate actual digit bounds from the range endpoints
    let min_digits = count_digits_in_base(base_range.range_start, base);
    let max_digits = count_digits_in_base(base_range.range_end, base);
    info!(
        "Candidate numbers will have {} to {} digits",
        min_digits, max_digits
    );

    let searcher = Arc::new(NiceNumberSearcher::new(
        base,
        min_digits,
        max_digits,
        base_range.range_start,
        base_range.range_end,
    ));

    info!(
        "Starting parallel search across {} initial branches...",
        base
    );

    // Parallelize the search by starting each CPU core with a different least significant digit
    // This distributes the workload evenly across available cores
    let results: Vec<(Vec<u128>, SearchStats)> = (0..base)
        .into_par_iter()
        .map(|least_significant_digit| {
            debug!("Worker {} starting...", least_significant_digit);
            let mut candidates_found = Vec::new();
            let mut stats = SearchStats::new();

            searcher.search_with_backtracking(
                0,                               // Start at digit position 0 (least significant)
                least_significant_digit as u128, // Initial candidate value
                0,                               // No digits used yet
                &mut candidates_found,
                &mut stats,
            );

            stats.log_summary(least_significant_digit);
            debug!(
                "Worker {} complete: found {} nice numbers",
                least_significant_digit,
                candidates_found.len()
            );
            (candidates_found, stats)
        })
        .collect();

    // Separate candidates and stats
    let (all_candidates, all_stats): (Vec<Vec<u128>>, Vec<SearchStats>) =
        results.into_iter().unzip();
    let all_candidates: Vec<u128> = all_candidates.into_iter().flatten().collect();

    // Aggregate statistics
    let total_stats = all_stats.iter().fold(SearchStats::new(), |mut acc, stats| {
        acc.nodes_explored += stats.nodes_explored;
        acc.nodes_pruned += stats.nodes_pruned;
        acc.candidates_tested += stats.candidates_tested;
        acc.candidates_skipped_range += stats.candidates_skipped_range;
        acc.candidates_skipped_leading_zeros += stats.candidates_skipped_leading_zeros;
        acc
    });

    info!(
        "Parallel search complete. Found {} total candidates (may include duplicates)",
        all_candidates.len()
    );

    // Log aggregate statistics
    let total_nodes = total_stats.nodes_explored + total_stats.nodes_pruned;
    let prune_rate = if total_nodes > 0 {
        (total_stats.nodes_pruned as f64 / total_nodes as f64) * 100.0
    } else {
        0.0
    };
    info!("=== Search Tree Statistics ===");
    info!("  Total nodes visited: {}", total_nodes);
    info!("  Nodes explored: {}", total_stats.nodes_explored);
    info!(
        "  Nodes pruned: {} ({:.1}%)",
        total_stats.nodes_pruned, prune_rate
    );
    info!("  Candidates tested: {}", total_stats.candidates_tested);
    info!(
        "  Candidates skipped (range): {}",
        total_stats.candidates_skipped_range
    );
    info!(
        "  Candidates skipped (leading zeros): {}",
        total_stats.candidates_skipped_leading_zeros
    );
    info!(
        "  Pruning efficiency: {:.1}% of branches eliminated early",
        prune_rate
    );

    // Deduplicate and filter results to the valid range
    let nice_numbers = deduplicate_and_filter(all_candidates, &base_range);

    // Display results
    print_results(base, &nice_numbers);
}

impl NiceNumberSearcher {
    /// Creates a new searcher with precomputed base powers for efficiency.
    fn new(
        base: u32,
        min_candidate_digits: u32,
        max_candidate_digits: u32,
        range_start: u128,
        range_end: u128,
    ) -> Self {
        debug!(
            "Initializing searcher with base={}, digits range=[{}, {}], value range=[{}, {}]",
            base, min_candidate_digits, max_candidate_digits, range_start, range_end
        );

        // Precompute powers: [base^0, base^1, base^2, ..., base^max_digits]
        let mut base_powers = vec![1u128];
        for exponent in 1..=max_candidate_digits {
            let next_power = base_powers.last().unwrap() * (base as u128);
            base_powers.push(next_power);
            trace!("{}^{} = {}", base, exponent, next_power);
        }

        Self {
            base,
            min_candidate_digits,
            max_candidate_digits,
            base_powers,
            range_start,
            range_end,
        }
    }

    /// Recursively searches for nice numbers using backtracking with early pruning.
    ///
    /// # Algorithm
    /// - Builds candidate numbers digit-by-digit from least significant to most significant
    /// - At each position, extracts the corresponding digit from n² and n³
    /// - Prunes branches where digit collisions would occur
    /// - Tests complete candidates when minimum length is reached
    ///
    /// # Arguments
    /// - `digit_position`: Current digit position being constructed (0 = least significant)
    /// - `current_candidate`: The number built so far
    /// - `used_digits_mask`: Bitmask tracking which digits have appeared in n² or n³
    /// - `results`: Accumulator for nice numbers found
    /// - `stats`: Statistics tracker for this search branch
    fn search_with_backtracking(
        &self,
        digit_position: u32,
        current_candidate: u128,
        used_digits_mask: DigitMask,
        results: &mut Vec<u128>,
        stats: &mut SearchStats,
    ) {
        stats.nodes_explored += 1;
        // Step 1: Compute n² and n³ once for this recursion level
        let n_natural = Natural::from(current_candidate);
        let n_cubed = (&n_natural).pow(3);
        let n_squared = n_natural.pow(2);

        // Extract the digit at this position from n² and n³
        // These digits are "locked in" - they won't change as we add higher-order digits
        let square_digit = self.extract_digit_from_power(&n_squared, digit_position);
        let cube_digit = self.extract_digit_from_power(&n_cubed, digit_position);

        trace!(
            "Evaluating node - Position {}: candidate={}, square_digit={}, cube_digit={}",
            digit_position, current_candidate, square_digit, cube_digit
        );

        // Step 2: Early pruning - check for digit collisions
        if self.has_digit_collision(square_digit, cube_digit, used_digits_mask) {
            stats.nodes_pruned += 1;
            trace!(
                "✗ Pruned at position {} (collision detected: sq={}, cu={}, mask={:b})",
                digit_position, square_digit, cube_digit, used_digits_mask
            );
            return;
        }

        // Update the mask with the newly used digits
        let updated_mask = used_digits_mask | (1 << square_digit) | (1 << cube_digit);

        // Step 3: Check if this is a complete candidate worth testing
        // Only test when we FIRST reach a valid length to avoid retesting with leading zeros
        let current_digit_count = digit_position + 1;
        if current_digit_count >= self.min_candidate_digits
            && current_digit_count <= self.max_candidate_digits
        {
            // Calculate the actual number of significant digits (without leading zeros)
            let actual_digit_count = count_digits_in_base(current_candidate, self.base);

            // Only test if the actual digit count matches where we are in the recursion
            // This prevents testing "047" when we've already tested "47"
            if actual_digit_count == current_digit_count {
                // Early range check - skip candidates outside valid range
                if current_candidate < self.range_start || current_candidate > self.range_end {
                    stats.candidates_skipped_range += 1;
                    trace!(
                        "Skipping candidate {} (outside range [{}, {}])",
                        current_candidate, self.range_start, self.range_end
                    );
                } else {
                    stats.candidates_tested += 1;
                    trace!(
                        "Testing candidate {} (digits={})",
                        current_candidate, current_digit_count
                    );

                    // Use the definitive test to check if this is truly a nice number
                    if get_is_nice(current_candidate, self.base) {
                        info!("✓ NICE NUMBER FOUND: n={}", current_candidate);
                        results.push(current_candidate);
                    }
                }
            } else {
                stats.candidates_skipped_leading_zeros += 1;
                trace!(
                    "Skipping candidate {} (has {} actual digits, but at position {})",
                    current_candidate, actual_digit_count, current_digit_count
                );
            }
        }

        // Step 4: Recurse to build longer candidates (if within bounds)
        if current_digit_count < self.max_candidate_digits {
            // Try all possible digits for the next higher-order position
            for next_digit in 0..self.base {
                let next_candidate =
                    self.add_digit_at_position(current_candidate, next_digit, digit_position + 1);

                self.search_with_backtracking(
                    digit_position + 1,
                    next_candidate,
                    updated_mask,
                    results,
                    stats,
                );
            }
        }
    }

    /// Extracts a specific digit from an already-computed power at the given position.
    ///
    /// # Arguments
    /// - `n_power`: The precomputed power (n² or n³)
    /// - `position`: Which digit to extract (0 = least significant)
    ///
    /// # Returns
    /// The digit value at that position
    fn extract_digit_from_power(&self, n_power: &Natural, position: u32) -> u128 {
        let base_natural = Natural::from(self.base);
        let divisor = Natural::from(self.base_powers[position as usize]);

        // Formula: digit = (n_power / base^position) mod base
        // Use div_rem to compute both quotient and remainder in one operation
        let (quotient, _remainder) = n_power.div_rem(&divisor);
        let digit_natural = quotient % base_natural;

        u128::try_from(&digit_natural).expect("Digit should fit in u128")
    }

    /// Constructs a number by adding a digit at a specific position.
    ///
    /// # Arguments
    /// - `current_number`: The number so far
    /// - `digit`: The digit to add (0 to base-1)
    /// - `position`: Where to place the digit (0 = least significant)
    ///
    /// # Returns
    /// The new number with the digit added
    fn add_digit_at_position(&self, current_number: u128, digit: u32, position: u32) -> u128 {
        current_number + (digit as u128 * self.base_powers[position as usize])
    }

    /// Checks if adding these digits would create a collision.
    ///
    /// A collision occurs if:
    /// 1. The square digit equals the cube digit (same digit in both)
    /// 2. The square digit has already been used
    /// 3. The cube digit has already been used
    ///
    /// # Arguments
    /// - `square_digit`: Digit from n² at current position
    /// - `cube_digit`: Digit from n³ at current position
    /// - `used_mask`: Bitmask of already-used digits
    ///
    /// # Returns
    /// `true` if there's a collision, `false` otherwise
    fn has_digit_collision(
        &self,
        square_digit: u128,
        cube_digit: u128,
        used_mask: DigitMask,
    ) -> bool {
        // Check if square and cube use the same digit
        if square_digit == cube_digit {
            return true;
        }

        // Check if square digit was already used
        if (used_mask & (1 << square_digit)) != 0 {
            return true;
        }

        // Check if cube digit was already used
        if (used_mask & (1 << cube_digit)) != 0 {
            return true;
        }

        false
    }
}

/// Counts the number of digits in a number when represented in a given base.
///
/// # Arguments
/// - `n`: The number to count digits in
/// - `base`: The numeric base
///
/// # Returns
/// The number of digits (minimum 1 for n=0)
fn count_digits_in_base(n: u128, base: u32) -> u32 {
    if n == 0 {
        return 1;
    }

    let mut count = 0;
    let mut temp = n;
    let base_u128 = base as u128;

    while temp > 0 {
        temp /= base_u128;
        count += 1;
    }

    trace!("Number {} has {} digits in base {}", n, count, base);

    count
}

/// Removes duplicates from candidates and filters to the valid range.
///
/// # Arguments
/// - `candidates`: Raw candidate numbers (may contain duplicates)
/// - `base_range`: The valid range for this base
///
/// # Returns
/// A sorted vector of unique nice numbers within the valid range
fn deduplicate_and_filter(candidates: Vec<u128>, base_range: &FieldSize) -> Vec<u128> {
    use std::collections::HashSet;

    let initial_count = candidates.len();

    // Convert to HashSet to remove duplicates
    let unique_candidates: HashSet<u128> = candidates.into_iter().collect();
    debug!(
        "Removed {} duplicates",
        initial_count - unique_candidates.len()
    );

    // Filter to valid range and sort
    let mut filtered: Vec<u128> = unique_candidates
        .into_iter()
        .filter(|&n| n >= base_range.range_start && n <= base_range.range_end)
        .collect();

    filtered.sort_unstable();

    debug!(
        "After filtering to range: {} nice numbers remain",
        filtered.len()
    );

    filtered
}

/// Displays the search results in a user-friendly format.
///
/// # Arguments
/// - `base`: The base that was searched
/// - `nice_numbers`: The nice numbers found (sorted)
fn print_results(base: u32, nice_numbers: &[u128]) {
    println!();
    println!("╔════════════════════════════════════════╗");
    println!("║  Nice Number Search Results (Base {})  ║", base);
    println!("╚════════════════════════════════════════╝");
    println!();

    if nice_numbers.is_empty() {
        println!("  No nice numbers found in the search range.");
    } else {
        println!("  Found {} nice number(s):\n", nice_numbers.len());
        for (index, &number) in nice_numbers.iter().enumerate() {
            println!("    {}. {}", index + 1, number);
        }
    }

    println!();
    info!(
        "Search complete. {} nice numbers found.",
        nice_numbers.len()
    );
}
