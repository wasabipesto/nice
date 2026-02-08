//! Stride-based iteration using the Chinese Remainder Theorem (CRT).
//!
//! This module implements Filter B from `NOVEL_FILTERS.md`: instead of iterating through
//! every integer and filtering, we use CRT to combine the residue filter (mod b-1) and
//! the multi-digit LSD filter (mod b^k) into a single modulus M = (b-1) × b^k.
//!
//! We precompute which residues mod M are valid, then iterate by jumping directly from
//! one valid candidate to the next using a gap table. This has **zero filter overhead**
//! per candidate—we simply never visit invalid candidates.
//!
//! Key advantage: This changes from filter-based iteration to generator-based iteration,
//! eliminating all modulo operations and lookups in the hot loop.

use crate::client_process::get_is_nice;
use crate::{FieldResults, FieldSize, NiceNumberSimple, lsd_filter, residue_filter};
use log::trace;

/// A precomputed stride table for efficient CRT-based iteration.
///
/// This table combines the residue filter (mod b-1) and multi-digit LSD filter (mod b^k)
/// into a single modulus using the Chinese Remainder Theorem. Instead of checking filters
/// for each candidate, we can jump directly from one valid candidate to the next.
pub struct StrideTable {
    /// The combined modulus: M = (b-1) × b^k
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
        let modulus = b_minus_1 * b_k; // CRT: gcd(b-1, b^k) = 1

        // Get the residue filter valid set (mod b-1)
        let residue_set = residue_filter::get_residue_filter_u128(&base);

        // Get the multi-digit LSD filter bitmap (mod b^k)
        let lsd_bitmap = lsd_filter::get_valid_multi_lsd_bitmap(base, k);

        // Find all residues r mod M that satisfy both filters
        let mut valid_residues = Vec::new();
        for r in 0..modulus {
            let passes_residue = residue_set.contains(&(r % b_minus_1));
            let passes_lsd = lsd_bitmap[(r % b_k) as usize];
            if passes_residue && passes_lsd {
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
                "Stride table for base {base} k={k}: modulus={modulus}, {} valid residues ({:.2}% pass rate)",
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

/// Process a range using stride-based iteration (Filter B).
///
/// This function combines MSD prefix filtering with CRT stride-based iteration.
/// It first uses the MSD filter to identify valid sub-ranges, then uses stride
/// iteration within each sub-range to efficiently visit only candidates that
/// pass both the residue filter and multi-digit LSD filter.
///
/// # Arguments
/// - `range`: The range to process
/// - `base`: The numeric base
/// - `k`: Number of least significant digits to check (recommended: use `lsd_filter::get_recommended_k`)
///
/// # Returns
/// Field results containing only the nice numbers found
#[must_use]
pub fn process_range_with_stride(range: &FieldSize, base: u32, k: u32) -> FieldResults {
    use crate::msd_prefix_filter;

    // Use MSD prefix filter to get valid sub-ranges
    let valid_ranges = msd_prefix_filter::get_valid_ranges(*range, base);

    // Build the stride table (this is cached in practice)
    let stride_table = StrideTable::new(base, k);

    // Iterate through each valid sub-range using stride iteration
    let mut nice_list = Vec::new();
    for sub_range in valid_ranges {
        let sub_results = stride_table.iterate_range(&sub_range, base);
        nice_list.extend(sub_results);
    }

    FieldResults {
        distribution: Vec::new(),
        nice_numbers: nice_list,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test_log::test]
    fn test_stride_table_base10_k1() {
        let table = StrideTable::new(10, 1);

        // Base 10: (b-1) = 9, b^1 = 10, M = 90
        assert_eq!(table.modulus, 90);

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

        // Base 40: (b-1) = 39, b^2 = 1600, M = 62400
        assert_eq!(table.modulus, 62_400);

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
    fn test_stride_vs_linear_consistency_base10() {
        // Compare stride iteration against linear iteration for base 10
        use crate::client_process::process_range_niceonly;

        let range = FieldSize::new(0, 1000);
        let base = 10;

        // Get results using current implementation
        let linear_results = process_range_niceonly(&range, base);

        // Get results using stride iteration
        let stride_results = process_range_with_stride(&range, base, 1);

        // Should find the same numbers
        assert_eq!(
            linear_results.nice_numbers.len(),
            stride_results.nice_numbers.len(),
            "Different number of results"
        );

        for nice in &linear_results.nice_numbers {
            assert!(
                stride_results
                    .nice_numbers
                    .iter()
                    .any(|r| r.number == nice.number),
                "Stride iteration missed number {}",
                nice.number
            );
        }
    }

    #[test_log::test]
    fn test_stride_vs_linear_consistency_base40() {
        // Compare stride iteration against linear iteration for base 40
        use crate::client_process::process_range_niceonly;

        let base = 40;
        // Use a small range around the starting point for base 40
        let range = FieldSize::new(1_999_999_999_900, 1_999_999_999_950);

        // Get results using current implementation
        let linear_results = process_range_niceonly(&range, base);

        // Get results using stride iteration
        let stride_results = process_range_with_stride(&range, base, 2);

        // Should find the same numbers
        assert_eq!(
            linear_results.nice_numbers.len(),
            stride_results.nice_numbers.len(),
            "Different number of results"
        );

        for nice in &linear_results.nice_numbers {
            assert!(
                stride_results
                    .nice_numbers
                    .iter()
                    .any(|r| r.number == nice.number),
                "Stride iteration missed number {}",
                nice.number
            );
        }
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
}
