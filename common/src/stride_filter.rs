//! Stride-based iteration using the Chinese Remainder Theorem (CRT).
//!
//! Instead of iterating through every integer and filtering, we use CRT to combine
//! multiple filters into a single modulus. Currently combines:
//! - Residue filter (mod b-1)
//! - Multi-digit LSD filter (mod b^k)
//! - Alternating sum filter (mod b+1)
//!
//! We precompute which residues mod M are valid, then iterate by jumping directly from
//! one valid candidate to the next using a gap table. This has zero filter overhead
//! per candidate - we simply never visit invalid candidates.

use crate::client_process::get_is_nice;
use crate::{FieldSize, NiceNumberSimple, alternating_sum_filter, lsd_filter, residue_filter};
use log::trace;
use std::collections::HashSet;

/// A precomputed stride table for efficient CRT-based iteration.
///
/// This table combines multiple filters using the Chinese Remainder Theorem:
/// - Residue filter (mod b-1)
/// - Multi-digit LSD filter (mod b^k)
/// - Alternating sum filter (mod b+1)
///
/// Instead of checking filters for each candidate, we can jump directly from one
/// valid candidate to the next.
pub struct StrideTable {
    /// The combined modulus: M = lcm((b-1) × b^k, b+1)
    pub modulus: u128,
    /// Sorted list of valid residues mod M
    pub valid_residues: Vec<u128>,
    /// Gap from each valid residue to the next: `gap_table[i] = valid_residues[i+1] - valid_residues[i]`
    /// The last entry wraps around: `gap_table[last] = M - valid_residues[last] + valid_residues[0]`
    pub gap_table: Vec<u128>,
}

impl StrideTable {
    /// Create a new stride table for the given base and k-digit LSD filter.
    ///
    /// # Arguments
    /// - `base`: The numeric base
    /// - `k`: Number of least significant digits to check (from multi-digit LSD filter)
    ///
    /// # Panics
    /// Panics if base^k overflows u128
    #[must_use]
    pub fn new(base: u32, k: u32) -> Self {
        let b_minus_1 = u128::from(base - 1);
        let b_k = u128::from(base).pow(k);
        let b_plus_1 = u128::from(base) + 1;

        // Compute the combined modulus using LCM
        // gcd(b-1, b^k) = 1 (coprime)
        // gcd(b+1, b-1) = gcd(2, b-1) which is 1 or 2
        // gcd(b+1, b^k) depends on gcd(b+1, b) = gcd(1, b) = 1
        let m1 = b_minus_1 * b_k; // (b-1) × b^k
        let gcd_m1_bplus1 = gcd(m1, b_plus_1);
        let modulus = m1 * b_plus_1 / gcd_m1_bplus1; // lcm(m1, b+1)

        // Get the residue filter valid set (mod b-1)
        let residue_set: HashSet<u128> = residue_filter::get_residue_filter_u128(&base)
            .into_iter()
            .collect();

        // Get the multi-digit LSD filter bitmap (mod b^k)
        let lsd_bitmap = lsd_filter::get_valid_multi_lsd_bitmap(base, k);

        // Get the alternating sum filter valid set (mod b+1)
        let alt_sum_set: HashSet<u128> = alternating_sum_filter::get_alternating_sum_filter(&base)
            .into_iter()
            .collect();

        // Find all residues r mod M that satisfy all three filters
        let mut valid_residues = Vec::new();
        for r in 0..modulus {
            let passes_residue = residue_set.contains(&(r % b_minus_1));
            let passes_lsd = lsd_bitmap[(r % b_k) as usize];
            let passes_alt_sum = alt_sum_set.contains(&(r % b_plus_1));

            if passes_residue && passes_lsd && passes_alt_sum {
                valid_residues.push(r);
            }
        }

        // Compute gaps between consecutive valid residues
        let mut gap_table = Vec::with_capacity(valid_residues.len());
        for i in 0..valid_residues.len() {
            let next_gap = if i + 1 < valid_residues.len() {
                valid_residues[i + 1] - valid_residues[i]
            } else {
                // Wraparound: distance from last valid residue back to first
                modulus - valid_residues[i] + valid_residues[0]
            };
            gap_table.push(next_gap);
        }

        #[allow(clippy::cast_precision_loss)]
        {
            trace!(
                "Stride table for base {base} k={k}: modulus={modulus}, {} valid residues ({:.2}% pass rate) [residue + LSD + alt_sum filters]",
                valid_residues.len(),
                100.0 * valid_residues.len() as f64 / modulus as f64
            );
        }

        StrideTable {
            modulus,
            valid_residues,
            gap_table,
        }
    }

    /// Find the first valid candidate >= start and return `(candidate, gap_index)`.
    ///
    /// # Arguments
    /// - `start`: The starting value
    ///
    /// # Returns
    /// A tuple of `(first_valid_n, gap_index)` where:
    /// - `first_valid_n` is the smallest n >= start with n % M in `valid_residues`
    /// - `gap_index` is the index in `valid_residues`/`gap_table` for this residue
    #[must_use]
    pub fn first_valid_at_or_after(&self, start: u128) -> (u128, usize) {
        let r = start % self.modulus;

        // Binary search for the first valid residue >= r
        let idx = match self.valid_residues.binary_search(&r) {
            Ok(i) => i, // Exact match
            Err(i) => {
                if i < self.valid_residues.len() {
                    i // First residue > r
                } else {
                    0 // Wrapped around, use first residue
                }
            }
        };

        let target_r = self.valid_residues[idx];
        let n = if target_r >= r {
            // Same cycle: just advance to target_r
            start + (target_r - r)
        } else {
            // Next cycle: wrap around the modulus
            start + (self.modulus - r + target_r)
        };

        (n, idx)
    }

    /// Iterate over all valid candidates in the range, applying `get_is_nice` to each.
    ///
    /// This is the core stride-based iteration function. Instead of checking every
    /// integer in the range, we jump directly from one valid candidate to the next
    /// using the precomputed gap table.
    ///
    /// # Arguments
    /// - `range`: The range to process
    /// - `base`: The numeric base
    ///
    /// # Returns
    /// A vector of nice numbers found in the range
    #[must_use]
    pub fn iterate_range(&self, range: &FieldSize, base: u32) -> Vec<NiceNumberSimple> {
        let mut results = Vec::new();
        let (mut n, mut idx) = self.first_valid_at_or_after(range.start());

        while n < range.end() {
            if get_is_nice(n, base) {
                results.push(NiceNumberSimple {
                    number: n,
                    num_uniques: base,
                });
            }
            n += self.gap_table[idx];
            idx = (idx + 1) % self.gap_table.len();
        }

        results
    }
}

/// Compute the greatest common divisor using Euclid's algorithm.
fn gcd(mut a: u128, mut b: u128) -> u128 {
    while b != 0 {
        let temp = b;
        b = a % b;
        a = temp;
    }
    a
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test_log::test]
    fn test_stride_table_base10_k1() {
        let table = StrideTable::new(10, 1);

        // Base 10: (b-1) = 9, b^1 = 10, b+1 = 11
        // M = lcm(90, 11) = 990 (since gcd(90, 11) = 1)
        assert_eq!(table.modulus, 990);

        // Should have valid residues combining both filters
        assert!(!table.valid_residues.is_empty());
        assert_eq!(table.valid_residues.len(), table.gap_table.len());

        // Verify gap table covers full cycle
        let total_gap: u128 = table.gap_table.iter().sum();
        assert_eq!(total_gap, table.modulus);
    }

    #[test_log::test]
    fn test_stride_table_base40_k2() {
        let table = StrideTable::new(40, 2);

        // Base 40: (b-1) = 39, b^2 = 1600, b+1 = 41
        // M1 = 39 × 1600 = 62400
        // M = lcm(62400, 41) = 2558400 (since gcd(62400, 41) = 1)
        assert_eq!(table.modulus, 2_558_400);

        // Should filter significantly
        assert!(table.valid_residues.len() < (table.modulus as usize));

        // Verify properties
        assert_eq!(table.valid_residues.len(), table.gap_table.len());
        let total_gap: u128 = table.gap_table.iter().sum();
        assert_eq!(total_gap, table.modulus);
    }

    #[test_log::test]
    fn test_first_valid_at_or_after() {
        let table = StrideTable::new(10, 1);

        // Start at 0 should find first valid
        let (n, idx) = table.first_valid_at_or_after(0);
        assert_eq!(n, table.valid_residues[idx]);

        // Start at a valid residue should return it
        let first_valid = table.valid_residues[0];
        let (n, idx) = table.first_valid_at_or_after(first_valid);
        assert_eq!(n, first_valid);
        assert_eq!(idx, 0);

        // Start beyond modulus should wrap correctly
        let (n, idx) = table.first_valid_at_or_after(table.modulus + 5);
        assert!(n >= table.modulus + 5);
        assert_eq!(n % table.modulus, table.valid_residues[idx]);
    }

    #[test_log::test]
    fn test_stride_iteration_finds_known_nice() {
        // Base 10: 69 is a known nice number
        let table = StrideTable::new(10, 1);

        let range = FieldSize::new(60, 80);
        let results = table.iterate_range(&range, 10);

        // Should find 69
        assert!(results.iter().any(|r| r.number == 69));
    }

    #[test_log::test]
    fn test_gap_table_properties() {
        let table = StrideTable::new(10, 2);

        // All gaps should be positive
        for gap in &table.gap_table {
            assert!(*gap > 0, "Gap should be positive");
        }

        // Sum of gaps should equal modulus (complete cycle)
        let total: u128 = table.gap_table.iter().sum();
        assert_eq!(total, table.modulus);

        // Valid residues should be sorted
        for i in 1..table.valid_residues.len() {
            assert!(
                table.valid_residues[i] > table.valid_residues[i - 1],
                "Valid residues should be sorted"
            );
        }
    }

    #[test_log::test]
    fn test_gcd() {
        assert_eq!(gcd(10, 5), 5);
        assert_eq!(gcd(90, 11), 1);
        assert_eq!(gcd(100, 50), 50);
        assert_eq!(gcd(17, 19), 1);
        assert_eq!(gcd(62400, 41), 1);
    }

    #[test_log::test]
    fn test_alternating_sum_filter_integration() {
        // Test that alternating sum filter is properly integrated
        let base = 10u32;
        let table = StrideTable::new(base, 1);

        // The modulus should be lcm((b-1)×b^k, b+1) = lcm(90, 11) = 990
        assert_eq!(table.modulus, 990);

        // Verify that the number of valid residues is reduced compared to
        // just residue + LSD filters (which would have modulus 90)
        let just_residue_lsd = {
            let b_minus_1 = 9u128;
            let b_k = 10u128;
            let m = b_minus_1 * b_k;
            let residue_set: HashSet<u128> = residue_filter::get_residue_filter_u128(&base)
                .into_iter()
                .collect();
            let lsd_bitmap = lsd_filter::get_valid_multi_lsd_bitmap(base, 1);
            let mut count = 0;
            for r in 0..m {
                if residue_set.contains(&(r % b_minus_1)) && lsd_bitmap[(r % b_k) as usize] {
                    count += 1;
                }
            }
            count
        };

        // With alternating sum filter, we should have fewer or equal valid residues
        // (scaled by the modulus ratio)
        let expected_max = just_residue_lsd * (table.modulus / 90);
        assert!(
            table.valid_residues.len() as u128 <= expected_max,
            "Alternating sum filter should reduce or maintain the valid residue count"
        );
    }
}
