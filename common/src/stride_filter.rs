//! Stride-based iteration using the Chinese Remainder Theorem (CRT).
//!
//! Instead of iterating through every integer and filtering, we use CRT to combine
//! multiple filters into a single modulus:
//! - Residue filter (mod b-1)
//! - Multi-digit LSD filter (mod b^k)
//! - Alternating sum filter (mod b+1) when applicable
//!
//! We precompute which residues mod M are valid, then iterate by jumping directly from
//! one valid candidate to the next using a gap table. This has zero filter overhead
//! per candidate - we simply never visit invalid candidates.

use crate::alternating_sum_filter;
use crate::client_process::get_is_nice;
use crate::{FieldSize, NiceNumberSimple, lsd_filter, residue_filter};
use log::trace;

/// A precomputed stride table for efficient CRT-based iteration.
///
/// This table combines the residue filter (mod b-1), multi-digit LSD filter (mod b^k),
/// and optionally the alternating sum filter (mod b+1) into a single modulus using
/// the Chinese Remainder Theorem. Instead of checking filters for each candidate,
/// we can jump directly from one valid candidate to the next.
pub struct StrideTable {
    /// The combined modulus: M = (b-1) × b^k × (b+1 if alternating sum applies)
    pub modulus: u128,
    /// Sorted list of valid residues mod M
    pub valid_residues: Vec<u128>,
    /// Gap from each valid residue to the next: `gap_table[i] = valid_residues[i+1] - valid_residues[i]`
    /// The last entry wraps around: `gap_table[last] = M - valid_residues[last] + valid_residues[0]`
    pub gap_table: Vec<u128>,
    /// Whether the alternating sum filter was applied
    pub uses_alternating_sum: bool,
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
        let b_plus_1 = u128::from(base + 1);

        // Check if alternating sum filter applies
        let uses_alternating_sum = alternating_sum_filter::is_filter_applicable(&base);

        // Compute combined modulus based on which filters apply
        let modulus = if uses_alternating_sum {
            // CRT: gcd(b-1, b^k) = 1, gcd(b-1, b+1) = 1, gcd(b^k, b+1) = 1
            b_minus_1 * b_k * b_plus_1
        } else {
            b_minus_1 * b_k
        };

        // Get the residue filter valid set (mod b-1)
        let residue_set = residue_filter::get_residue_filter_u128(&base);

        // Get the multi-digit LSD filter bitmap (mod b^k)
        let lsd_bitmap = lsd_filter::get_valid_multi_lsd_bitmap(base, k);

        // Get the alternating sum filter valid set (mod b+1) if applicable
        let alternating_set = if uses_alternating_sum {
            alternating_sum_filter::get_valid_residues_u128(&base)
        } else {
            Vec::new()
        };

        // Find all residues r mod M that satisfy all applicable filters
        let mut valid_residues = Vec::new();
        for r in 0..modulus {
            let passes_residue = residue_set.contains(&(r % b_minus_1));
            let passes_lsd = lsd_bitmap[(r % b_k) as usize];

            if !passes_residue || !passes_lsd {
                continue;
            }

            // Check alternating sum filter if applicable
            if uses_alternating_sum {
                let r_mod_b_plus_1 = r % b_plus_1;
                if !alternating_set.contains(&r_mod_b_plus_1) {
                    continue;
                }
            }

            valid_residues.push(r);
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
            let filter_info = if uses_alternating_sum {
                format!("(includes alternating sum filter mod {b_plus_1})")
            } else {
                String::new()
            };
            trace!(
                "Stride table for base {base} k={k}: modulus={modulus}, {} valid residues ({:.2}% pass rate) {filter_info}",
                valid_residues.len(),
                100.0 * valid_residues.len() as f64 / modulus as f64,
            );
        }

        StrideTable {
            modulus,
            valid_residues,
            gap_table,
            uses_alternating_sum,
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test_log::test]
    fn test_stride_table_base10_k1() {
        let table = StrideTable::new(10, 1);

        // Base 10: (b-1) = 9, b^1 = 10, (b+1) = 11 (applies)
        // M = 9 × 10 × 11 = 990
        assert_eq!(table.modulus, 990);
        assert!(table.uses_alternating_sum);

        // Should have valid residues combining all three filters
        assert!(!table.valid_residues.is_empty());
        assert_eq!(table.valid_residues.len(), table.gap_table.len());

        // Verify gap table covers full cycle
        let total_gap: u128 = table.gap_table.iter().sum();
        assert_eq!(total_gap, table.modulus);
    }

    #[test_log::test]
    fn test_stride_table_base40_k2() {
        let table = StrideTable::new(40, 2);

        // Base 40: (b-1) = 39, b^2 = 1600
        // Alternating sum filter doesn't apply (40 ≡ 0 mod 4)
        // M = 39 × 1600 = 62,400
        assert_eq!(table.modulus, 62_400);
        assert!(!table.uses_alternating_sum);

        // Should filter significantly
        assert!(table.valid_residues.len() < (table.modulus as usize));

        // Verify properties
        assert_eq!(table.valid_residues.len(), table.gap_table.len());
        let total_gap: u128 = table.gap_table.iter().sum();
        assert_eq!(total_gap, table.modulus);
    }

    #[test_log::test]
    fn test_stride_table_base50_k1() {
        let table = StrideTable::new(50, 1);

        // Base 50: (b-1) = 49, b^1 = 50, (b+1) = 51 (applies)
        // M = 49 × 50 × 51 = 124,950
        assert_eq!(table.modulus, 124_950);
        assert!(table.uses_alternating_sum);

        // Should have valid residues
        assert!(!table.valid_residues.is_empty());
        assert_eq!(table.valid_residues.len(), table.gap_table.len());

        // Verify gap table covers full cycle
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
    fn test_alternating_sum_filter_effectiveness() {
        // Compare base 10 (with alternating sum) vs base 40 (without)
        let table_10 = StrideTable::new(10, 1);
        let table_40 = StrideTable::new(40, 1);

        // Base 10 should have additional filtering from alternating sum
        let pass_rate_10 = table_10.valid_residues.len() as f64 / table_10.modulus as f64;

        // Base 40 should not have alternating sum filter
        assert!(!table_40.uses_alternating_sum);

        // Verify that the alternating sum filter is actually reducing candidates
        // For base 10, we expect roughly half the residues to pass the alternating sum filter
        // So the pass rate should be lower than it would be without it
        assert!(
            pass_rate_10 < 0.5,
            "Base 10 pass rate should be < 50% with all filters"
        );
    }

    #[test_log::test]
    fn test_crt_compatibility_verification() {
        // Verify that all moduli are coprime for CRT to work
        for base in &[10, 40, 50, 60, 70, 80, 90, 100] {
            // gcd(b-1, b) = 1 always
            assert_eq!(
                gcd(base - 1, *base),
                1,
                "gcd(b-1, b) should be 1 for base {}",
                base
            );

            // gcd(b-1, b+1) = gcd(b-1, 2) = 1 for odd b-1, which is true for even bases
            if *base % 2 == 0 {
                assert_eq!(
                    gcd(base - 1, base + 1),
                    1,
                    "gcd(b-1, b+1) should be 1 for even base {}",
                    base
                );
            }

            // gcd(b, b+1) = 1 always
            assert_eq!(
                gcd(*base, base + 1),
                1,
                "gcd(b, b+1) should be 1 for base {}",
                base
            );
        }
    }

    // Helper function for GCD
    fn gcd(a: u32, b: u32) -> u32 {
        if b == 0 { a } else { gcd(b, a % b) }
    }

    #[test_log::test]
    fn test_valid_residues_pass_all_filters() {
        let table = StrideTable::new(10, 1);
        let base = 10u32;
        let b_minus_1 = u128::from(base - 1);
        let b_k = u128::from(base).pow(1);
        let b_plus_1 = u128::from(base + 1);

        let residue_set = residue_filter::get_residue_filter_u128(&base);
        let lsd_bitmap = lsd_filter::get_valid_multi_lsd_bitmap(base, 1);
        let alternating_set = alternating_sum_filter::get_valid_residues_u128(&base);

        // Every valid residue should pass all filters
        for &r in &table.valid_residues {
            assert!(
                residue_set.contains(&(r % b_minus_1)),
                "Residue {} should pass residue filter",
                r
            );
            assert!(
                lsd_bitmap[(r % b_k) as usize],
                "Residue {} should pass LSD filter",
                r
            );
            assert!(
                alternating_set.contains(&(r % b_plus_1)),
                "Residue {} should pass alternating sum filter",
                r
            );
        }
    }
}
