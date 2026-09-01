//! Stride-based iteration using the Chinese Remainder Theorem (CRT).
//!
//! Instead of iterating through every integer and filtering, we use CRT to combine
//! the residue filter (mod b-1) and the multi-digit LSD filter (mod b^k) into a single
//! modulus M = (b-1) × b^k.
//!
//! We precompute which residues mod M are valid, then iterate by jumping directly from
//! one valid candidate to the next using a gap table. This has zero filter overhead
//! per candidate - we simply never visit invalid candidates.

use crate::client_process::get_is_nice_with_known_lsd;
use crate::{FieldSize, NiceNumberSimple, lsd_filter, residue_filter};
use log::trace;

/// A precomputed stride table for efficient CRT-based iteration.
///
/// This table combines the residue filter (mod b-1) and multi-digit LSD filter (mod b^k)
/// into a single modulus using the Chinese Remainder Theorem. Instead of checking filters
/// for each candidate, we can jump directly from one valid candidate to the next.
///
/// Residues and gaps are stored as `u32`. This caps the modulus at `u32::MAX`,
/// which any base ≤ 256 with k ≤ 3 satisfies, and keeps the table compact:
/// at k=3 the residue count reaches several hundred thousand entries, where
/// 32-byte-per-entry storage would blow past cache while 8 bytes stays cheap
/// (entry lookups binary-search the residues; iteration streams only gaps).
pub struct StrideTable {
    /// The combined modulus: M = (b-1) × b^k
    pub modulus: u128,
    /// The number of low digits fixed by each residue (the LSD filter depth)
    pub k: u32,
    /// Sorted list of valid residues mod M
    pub valid_residues: Vec<u32>,
    /// Gap from each valid residue to the next: `gap_table[i] = valid_residues[i+1] - valid_residues[i]`
    /// The last entry wraps around: `gap_table[last] = M - valid_residues[last] + valid_residues[0]`
    pub gap_table: Vec<u32>,
    /// Per-residue bitmask of the 2k fixed low digits of n² and n³ (bit d set
    /// = digit d appears). All 2k digits are pairwise distinct by
    /// construction (the LSD filter rejected everything else), so the nice
    /// check can seed its duplicate indicator from this mask and skip
    /// re-extracting the low digits. Empty when base > 64 (digits would not
    /// fit a u64 mask); iteration then falls back to the unseeded check.
    pub low_digit_masks: Vec<u64>,
}

impl StrideTable {
    /// Create a new stride table for the given base and k-digit LSD filter.
    ///
    /// # Arguments
    /// - `base`: The numeric base
    /// - `k`: Number of least significant digits to check (from multi-digit LSD filter)
    ///
    /// # Panics
    /// Panics if base^k overflows u32 or (base-1) × base^k overflows u32
    #[must_use]
    pub fn new(base: u32, k: u32) -> Self {
        let b_minus_1 = base - 1;
        let b_k = base.checked_pow(k).expect("base^k must fit in u32");
        let modulus = b_minus_1
            .checked_mul(b_k)
            .expect("(base-1) * base^k must fit in u32"); // CRT: gcd(b-1, b^k) = 1

        // Get the residue filter valid set (mod b-1) as a direct-index table
        let residue_set = residue_filter::get_residue_filter(&base);
        let mut residue_ok = vec![false; b_minus_1 as usize];
        for r in residue_set {
            residue_ok[r as usize] = true;
        }

        // Get the multi-digit LSD filter bitmap (mod b^k)
        let lsd_bitmap = lsd_filter::get_valid_multi_lsd_bitmap(base, k);

        // Find all residues r mod M that satisfy both filters
        let mut valid_residues = Vec::new();
        for r in 0..modulus {
            let passes_residue = residue_ok[(r % b_minus_1) as usize];
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

        // Precompute the fixed low digits of n² and n³ for each residue so
        // the nice check can skip re-extracting them (see `low_digit_masks`).
        let low_digit_masks = if base <= 64 {
            let b_k_u64 = u64::from(b_k);
            valid_residues
                .iter()
                .map(|&r| {
                    let suffix = u64::from(r % b_k);
                    let mut mask: u64 = 0;
                    for value in [
                        suffix * suffix % b_k_u64,
                        suffix * suffix % b_k_u64 * suffix % b_k_u64,
                    ] {
                        let mut v = value;
                        for _ in 0..k {
                            mask |= 1 << (v % u64::from(base));
                            v /= u64::from(base);
                        }
                    }
                    mask
                })
                .collect()
        } else {
            Vec::new()
        };

        #[allow(clippy::cast_precision_loss)]
        {
            trace!(
                "Stride table for base {base} k={k}: modulus={modulus}, {} valid residues ({:.2}% pass rate)",
                valid_residues.len(),
                100.0 * valid_residues.len() as f64 / f64::from(modulus)
            );
        }

        StrideTable {
            modulus: u128::from(modulus),
            k,
            valid_residues,
            gap_table,
            low_digit_masks,
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
        // The modulus fits in u32 (enforced at construction), so the residue does too.
        #[allow(clippy::cast_possible_truncation)]
        let r = (start % self.modulus) as u32;

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

        let target_r = u128::from(self.valid_residues[idx]);
        let r = u128::from(r);
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
        self.iterate_range_masked(range, base, 0)
    }

    /// [`StrideTable::iterate_range`] with the cross-end residue filter:
    /// `high_mask` holds digits the MSD analysis proved occupy some output
    /// position `>= k` for every `n` in this range
    /// (`msd_prefix_filter::MsdAnalysis`). A residue whose exact low-digit
    /// mask intersects it would repeat a digit across two distinct
    /// positions, so its candidates are skipped without a nice check —
    /// one AND on a mask this loop already loads.
    ///
    /// Sound only when `high_mask` excludes positions below `k` (which
    /// `analyze_range(_, _, k)` guarantees); pass 0 to disable.
    #[must_use]
    pub fn iterate_range_masked(
        &self,
        range: &FieldSize,
        base: u32,
        high_mask: u64,
    ) -> Vec<NiceNumberSimple> {
        let mut results = Vec::new();
        let (mut n, mut idx) = self.first_valid_at_or_after(range.start());

        // Seed the nice check with each residue's known low digits when
        // masks are available (base ≤ 64). `get_is_nice_with_known_lsd`
        // itself falls back to the plain check for unspecialized bases.
        // (For bases above 64 the mask table is empty and `high_mask` is
        // always 0 — the analysis never emits mask bits there.)
        let masks = &self.low_digit_masks;

        while n < range.end() {
            let is_nice = if masks.is_empty() {
                crate::client_process::get_is_nice(n, base)
            } else if masks[idx] & high_mask != 0 {
                false
            } else {
                get_is_nice_with_known_lsd(n, base, self.k, masks[idx])
            };
            if is_nice {
                results.push(NiceNumberSimple {
                    number: n,
                    num_uniques: base,
                });
            }
            n += u128::from(self.gap_table[idx]);
            idx += 1;
            if idx == self.gap_table.len() {
                idx = 0;
            }
        }

        results
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::base_range::get_base_range_u128;
    use crate::client_process::get_is_nice;

    #[test_log::test]
    fn test_stride_table_base10_k1() {
        let table = StrideTable::new(10, 1);

        // Base 10: (b-1) = 9, b^1 = 10, M = 90
        assert_eq!(table.modulus, 90);

        // Should have valid residues combining both filters
        assert!(!table.valid_residues.is_empty());
        assert_eq!(table.valid_residues.len(), table.gap_table.len());

        // Verify gap table covers full cycle
        let total_gap: u128 = table.gap_table.iter().map(|&g| u128::from(g)).sum();
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
        let total_gap: u128 = table.gap_table.iter().map(|&g| u128::from(g)).sum();
        assert_eq!(total_gap, table.modulus);
    }

    #[test_log::test]
    fn test_first_valid_at_or_after() {
        let table = StrideTable::new(10, 1);

        // Start at 0 should find first valid
        let (n, idx) = table.first_valid_at_or_after(0);
        assert_eq!(n, u128::from(table.valid_residues[idx]));

        // Start at a valid residue should return it
        let first_valid = u128::from(table.valid_residues[0]);
        let (n, idx) = table.first_valid_at_or_after(first_valid);
        assert_eq!(n, first_valid);
        assert_eq!(idx, 0);

        // Start beyond modulus should wrap correctly
        let (n, idx) = table.first_valid_at_or_after(table.modulus + 5);
        assert!(n >= table.modulus + 5);
        assert_eq!(n % table.modulus, u128::from(table.valid_residues[idx]));
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
    fn test_k3_candidates_subset_of_k2() {
        // Deeper LSD filtering must only remove candidates, never add them:
        // all 3+3 fixed low digits distinct implies the low 2+2 subset is
        // distinct, so every k=3 candidate must also be a k=2 candidate.
        for base in [10u32, 40] {
            let t2 = StrideTable::new(base, 2);
            let t3 = StrideTable::new(base, 3);
            let start = 1_000_000u128;
            let range = FieldSize::new(start, start + 200_000);

            let collect = |t: &StrideTable| {
                let mut out = Vec::new();
                let (mut n, mut idx) = t.first_valid_at_or_after(range.start());
                while n < range.end() {
                    out.push(n);
                    n += u128::from(t.gap_table[idx]);
                    idx = (idx + 1) % t.gap_table.len();
                }
                out
            };
            let c2 = collect(&t2);
            let c3 = collect(&t3);
            assert!(c3.len() < c2.len(), "base {base}: k=3 should filter more");
            let c2set: std::collections::HashSet<u128> = c2.into_iter().collect();
            for n in c3 {
                assert!(c2set.contains(&n), "base {base}: {n} in k=3 but not k=2");
            }
        }
    }

    #[test_log::test]
    fn test_k3_finds_known_nice() {
        // 69 must survive the k=3 table in base 10
        let table = StrideTable::new(10, 3);
        let range = FieldSize::new(60, 80);
        let results = table.iterate_range(&range, 10);
        assert!(results.iter().any(|r| r.number == 69));
    }

    #[test_log::test]
    fn test_seeded_nice_check_matches_plain() {
        // The seeded fast path must agree with the plain check for every
        // stride candidate. Walk real candidates inside each base's search
        // range and compare both paths.
        for base in [40u32, 50, 52, 60, 64] {
            let table = StrideTable::new(base, 3);
            assert!(!table.low_digit_masks.is_empty());
            let range = get_base_range_u128(base).unwrap().unwrap();
            let start = range.start() + (range.end() - range.start()) / 3;
            let (mut n, mut idx) = table.first_valid_at_or_after(start);
            for _ in 0..5_000 {
                let plain = get_is_nice(n, base);
                let seeded =
                    get_is_nice_with_known_lsd(n, base, table.k, table.low_digit_masks[idx]);
                assert_eq!(plain, seeded, "base {base}: mismatch at n={n}");
                n += u128::from(table.gap_table[idx]);
                idx = (idx + 1) % table.gap_table.len();
            }
        }
    }

    #[test_log::test]
    fn test_low_digit_masks_have_2k_bits() {
        // Every mask covers exactly 2k distinct digits (k from the square
        // suffix, k from the cube suffix, all pairwise distinct).
        for (base, k) in [(10u32, 2u32), (40, 3), (50, 3)] {
            let table = StrideTable::new(base, k);
            for (&r, &mask) in table.valid_residues.iter().zip(&table.low_digit_masks) {
                assert_eq!(
                    mask.count_ones(),
                    2 * k,
                    "base {base} k={k} residue {r}: expected {} distinct digits",
                    2 * k
                );
            }
        }
    }

    #[test_log::test]
    fn test_k3_large_base_construction() {
        // Largest production-adjacent table: base 60, k=3.
        // M = 59 * 60^3 = 12,744,000 and all residues/gaps must fit u32.
        let table = StrideTable::new(60, 3);
        assert_eq!(table.modulus, 12_744_000);
        assert!(!table.valid_residues.is_empty());
        let total_gap: u128 = table.gap_table.iter().map(|&g| u128::from(g)).sum();
        assert_eq!(total_gap, table.modulus);
    }

    #[test_log::test]
    fn test_gap_table_properties() {
        let table = StrideTable::new(10, 2);

        // All gaps should be positive
        for gap in &table.gap_table {
            assert!(*gap > 0, "Gap should be positive");
        }

        // Sum of gaps should equal modulus (complete cycle)
        let total: u128 = table.gap_table.iter().map(|&g| u128::from(g)).sum();
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
