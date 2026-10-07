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
use crate::{FieldSize, NiceNumberSimple, affine_filter, lsd_filter, residue_filter};
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
    /// Survivors of that test then go through the affine middle-digit
    /// filter ([`affine_filter`]): the next `k` digits of each power
    /// (positions `k..2k`), computed from `n mod b^{2k}` with word
    /// arithmetic, are tested against the union of both masks before the
    /// full check runs.
    ///
    /// `high_mask` must be the certificate of an analysed range that
    /// contains `range` and has at least two numbers (what
    /// `msd_prefix_filter`'s recursion hands out: it only analyses ranges
    /// larger than its floor), with positions below `k` excluded (which
    /// `analyze_range(_, _, k)` guarantees); pass 0 to disable. The affine
    /// filter additionally needs the certificate to hold no digit from
    /// positions `k..2k`. That holds whenever the range starts at or above
    /// `b^{2k-1}`: two consecutive squares there already differ by more
    /// than `b^{2k-1}`, so none of those positions can be the same across
    /// the analysed range. The filter is only enabled for such ranges, which
    /// covers every legal range of every supported base (`b^5` is at most
    /// 1.1e9; legal ranges start above 1e12).
    #[must_use]
    pub fn iterate_range_masked(
        &self,
        range: &FieldSize,
        base: u32,
        high_mask: u64,
    ) -> Vec<NiceNumberSimple> {
        let use_affine = affine_filter::supports(base, self.k)
            && range.start() >= u128::from(base).pow(2 * self.k - 1);
        self.iterate_range_impl(range, base, high_mask, use_affine)
    }

    /// [`StrideTable::iterate_range_masked`] without the affine middle-digit
    /// filter (the stride table and the cross-end test still apply): the
    /// reference path for parity tests and A/B measurements.
    #[must_use]
    pub fn iterate_range_masked_without_affine(
        &self,
        range: &FieldSize,
        base: u32,
        high_mask: u64,
    ) -> Vec<NiceNumberSimple> {
        self.iterate_range_impl(range, base, high_mask, false)
    }

    #[inline]
    fn iterate_range_impl(
        &self,
        range: &FieldSize,
        base: u32,
        high_mask: u64,
        use_affine: bool,
    ) -> Vec<NiceNumberSimple> {
        self.walk_masked(range, base, high_mask, use_affine, |n, low| {
            get_is_nice_with_known_lsd(n, base, self.k, low)
        })
    }

    /// The masked walk with the seeded nice check injected: `check(n, low)`
    /// runs on exactly the candidates that pass the cross-end test and, with
    /// `use_affine`, the affine filter. Production passes the seeded nice
    /// check; tests pass a recorder to see which candidates got through.
    #[inline]
    fn walk_masked(
        &self,
        range: &FieldSize,
        base: u32,
        high_mask: u64,
        use_affine: bool,
        mut check: impl FnMut(u128, u64) -> bool,
    ) -> Vec<NiceNumberSimple> {
        let mut results = Vec::new();
        let (mut n, mut idx) = self.first_valid_at_or_after(range.start());

        // Seed the nice check with each residue's known low digits when
        // masks are available (base ≤ 64). `get_is_nice_with_known_lsd`
        // itself falls back to the plain check for unspecialized bases.
        // (For bases above 64 the mask table is empty and `high_mask` is
        // always 0 — the analysis never emits mask bits there.)
        let masks = &self.low_digit_masks;

        // `n mod b^{2k}`, tracked incrementally for the affine filter only:
        // the modulus fits u64 whenever the filter is supported (b ≤ 64,
        // k = 3), and every gap is at most the stride modulus
        // `(b-1)·b^k < b^{2k}`, so one conditional subtraction keeps it
        // reduced. Without the filter it is neither needed nor updated.
        let b2k: u64 = if use_affine {
            u64::from(base).pow(2 * self.k)
        } else {
            1
        };
        #[allow(clippy::cast_possible_truncation)]
        let mut nmod = (n % u128::from(b2k)) as u64;

        while n < range.end() {
            let is_nice = if masks.is_empty() {
                crate::client_process::get_is_nice(n, base)
            } else {
                let low = masks[idx];
                let rejected = low & high_mask != 0
                    || (use_affine && !affine_filter::survives(base, nmod, low | high_mask));
                !rejected && check(n, low)
            };
            if is_nice {
                results.push(NiceNumberSimple {
                    number: n,
                    num_uniques: base,
                });
            }
            let gap = self.gap_table[idx];
            n += u128::from(gap);
            if use_affine {
                nmod += u64::from(gap);
                if nmod >= b2k {
                    nmod -= b2k;
                }
            }
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

    /// Base-`base` digits of `n²` and `n³`, least significant first.
    fn power_digits(n: u128, base: u32) -> (Vec<u32>, Vec<u32>) {
        let mut sq = crate::fixed_width::U256::mul_u128_u128(n, n);
        let mut cu = sq.mul_u128_truncating(n);
        let (mut sqd, mut cud) = (Vec::new(), Vec::new());
        while !sq.is_zero() {
            sqd.push(sq.div_assign_rem_u32(base));
        }
        while !cu.is_zero() {
            cud.push(cu.div_assign_rem_u32(base));
        }
        (sqd, cud)
    }

    /// The masked walk with the affine filter lets through exactly the
    /// cross-end survivors whose *true* middle digits (positions 3..5 of
    /// `n²` and `n³`, from wide arithmetic) are pairwise distinct and miss
    /// the residue's low digits and the range certificate. Checked on every
    /// candidate of real MSD leaves, over windows long enough to wrap the
    /// gap table, so a drift in the walk's incremental `n mod b^6` shows up
    /// as a disagreement rather than as an empty nice set on both sides.
    #[test_log::test]
    fn affine_walk_decisions_match_true_digits() {
        // The benchmark's MSD-weak windows (`bench_defs`): production
        // regions with plenty of cross-end survivors.
        for (base, start) in [
            (40u32, 5_007_828_088_304u128),
            (50, 73_940_161_512_353_211),
            (52, 407_887_399_136_188_818),
        ] {
            assert!(affine_filter::supports(base, 3));
            let table = StrideTable::new(base, 3);
            // Longer than the gap table's period M = (b-1)·b³, so every leaf
            // sequence crosses the table's wrap at least once.
            let len = table.modulus + 1_000_000;
            let range = FieldSize::new(start, start + len);
            let (mut crossed, mut passed) = (0usize, 0usize);
            for (leaf, high) in crate::msd_prefix_filter::get_valid_ranges_masked(range, base, 3) {
                let mut cross = Vec::new();
                let _ = table.walk_masked(&leaf, base, high, false, |n, low| {
                    cross.push((n, low));
                    false
                });
                let mut through = Vec::new();
                let _ = table.walk_masked(&leaf, base, high, true, |n, _| {
                    through.push(n);
                    false
                });
                let mut it = through.iter().peekable();
                for (n, low) in cross {
                    let (sq, cu) = power_digits(n, base);
                    let mut seen = low | high;
                    let mut fresh = true;
                    for &d in sq[3..6].iter().chain(&cu[3..6]) {
                        fresh &= seen & (1u64 << d) == 0;
                        seen |= 1u64 << d;
                    }
                    let got = it.next_if_eq(&&n).is_some();
                    assert_eq!(got, fresh, "b{base} n={n}: walk and true digits disagree");
                    crossed += 1;
                    passed += usize::from(got);
                }
                assert!(
                    it.next().is_none(),
                    "b{base}: filtered walk visited extra candidates"
                );
            }
            // Not vacuous: plenty of cross-end survivors, most of them killed.
            assert!(
                crossed > 5_000,
                "b{base}: only {crossed} cross-end survivors"
            );
            assert!(
                passed * 10 < crossed,
                "b{base}: {passed} of {crossed} passed the affine filter"
            );
        }
    }

    #[test_log::test]
    fn test_stride_table_base10_k1() {
        let table = StrideTable::new(10, 1);

        // Base 10: (b-1) = 9, b^1 = 10, M = 90
        assert_eq!(table.modulus, 90);

        // Should have valid residues combining both filters
        assert!(!table.valid_residues.is_empty(), "no valid residues");
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
            assert!(!table.low_digit_masks.is_empty(), "b{base}: no masks");
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
        assert!(!table.valid_residues.is_empty(), "no valid residues");
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
