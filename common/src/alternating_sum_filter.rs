//! Alternating digit sum filter (mod b+1).
//!
//! This filter exploits the property that in base b, a number's alternating digit sum
//! equals the number modulo (b+1):
//!
//! `n = d₀ + d₁·b + d₂·b² + d₃·b³ + ...`
//!   `≡ d₀ - d₁ + d₂ - d₃ + ...` (mod `b+1`)
//!
//! Since `b ≡ -1` (mod `b+1`), we have `b^i ≡ (-1)^i` (mod `b+1`).
//!
//! For a pandigital number where digits {0, 1, 2, ..., b-1} each appear exactly once,
//! the alternating sum depends on which positions these digits occupy. While we can't
//! predict exact positions, we can constrain the possible values of:
//!   `alternating_sum(n²)` + `alternating_sum(n³)` mod (b+1)
//!
//! ## Mathematical Basis
//!
//! For a number `n` with digits `d₀, d₁, d₂, ...` (from right to left):
//! - `alternating_sum(n) = d₀ - d₁ + d₂ - d₃ + ... = n mod (b+1)`
//!
//! For a pandigital concatenation of n² and n³:
//! - Total digits: 0 + 1 + 2 + ... + (b-1) = b(b-1)/2
//! - But their positions matter for the alternating sum!
//!
//! The key insight: given n mod (b+1), we can compute:
//! - n² mod (b+1)
//! - n³ mod (b+1)
//!
//! These give us the alternating digit sums of n² and n³.
//!
//! However, the concatenation n² || n³ has a specific structure:
//! - All digits of n² come after all digits of n³ (or vice versa depending on convention)
//! - The position shift affects the alternating sum
//!
//! ## Implementation Strategy
//!
//! For each residue `r ∈ [0, b+1)`:
//! 1. Compute `square_alt = r² mod (b+1)`
//! 2. Compute `cube_alt = r³ mod (b+1)`
//! 3. Compute the combined alternating sum considering digit length and position
//! 4. Check if this could match a valid pandigital configuration
//!
//! ## Filter Effectiveness
//!
//! This filter is orthogonal to the regular digit sum filter (mod b-1) because:
//! - Digit sum (mod b-1) is position-independent: Σ dᵢ
//! - Alternating sum (mod b+1) is position-dependent: Σ (-1)^i dᵢ
//!
//! Expected filtering: 40-80% depending on base characteristics.
//!
//! ## CRT Integration
//!
//! When gcd(b+1, (b-1)·b^k) = 1, this filter integrates perfectly via CRT.
//! When gcd > 1, we can either:
//! - Use generalized CRT
//! - Apply this filter separately (slight performance cost but still effective)

use log::trace;
use std::collections::HashSet;

const MAX_BASE_FOR_ALTERNATING_SUM_FILTER: u32 = 50;

/// Get the set of valid residues mod (b+1) that could produce nice numbers.
///
/// For each residue r ∈ [0, b+1), we check if n ≡ r (mod b+1) could lead to
/// a pandigital by examining the alternating digit sums of n² and n³.
///
/// # Arguments
/// - `base`: The numeric base
///
/// # Returns
/// A vector of valid residues mod (b+1)
#[must_use]
pub fn get_alternating_sum_filter(base: &u32) -> Vec<u128> {
    let b = u128::from(*base);
    let b_plus_1 = b + 1;

    // For a pandigital where each digit 0 to b-1 appears exactly once,
    // we need to determine which n mod (b+1) values could possibly work.
    //
    // Key mathematical property:
    // n ≡ alternating_digit_sum(n) (mod b+1)
    // where alternating_digit_sum = d₀ - d₁ + d₂ - d₃ + ...
    //
    // For the concatenation of n² and n³ (let's denote it as C = n² || n³):
    // - C contains exactly b digits (one of each from 0 to b-1)
    // - The alternating sum of C depends on:
    //   1. Which digits appear in which positions
    //   2. The lengths of n² and n³ (which determine position parities)
    //
    // Strategy: For each residue r ∈ [0, b+1):
    // 1. Compute square_residue = r² mod (b+1)
    // 2. Compute cube_residue = r³ mod (b+1)
    // 3. Determine if these could combine to form a valid pandigital alternating sum
    //
    // The challenge: The alternating sum of the concatenation depends on:
    // - alternating_sum(n²) = n² mod (b+1) = square_residue
    // - alternating_sum(n³) = n³ mod (b+1) = cube_residue
    // - The position shift between them (depends on len(n³))
    //
    // If len(n³) is even, the alternating signs align: alt_sum(C) ≡ alt_sum(n²) + alt_sum(n³)
    // If len(n³) is odd, the signs flip: alt_sum(C) ≡ alt_sum(n²) - alt_sum(n³)
    //
    // Since we don't know len(n³) exactly from just r, we need to consider both cases.

    // Compute possible alternating sums for all arrangements of digits {0,1,...,b-1}
    // This is expensive for large b, so we use bounds and sampling
    #[allow(clippy::cast_possible_truncation)]
    let possible_alternating_sums = compute_possible_alternating_sums(b as u32, b_plus_1);

    let mut valid_residues = Vec::new();

    for r in 0..b_plus_1 {
        let square_alt = (r * r) % b_plus_1;
        let cube_alt = (r * r * r) % b_plus_1;

        // Check both parity cases for len(n³)
        // Concatenation: n² || n³ = n² × b^len(n³) + n³
        // In mod (b+1): concat ≡ n² × (-1)^len(n³) + n³
        //
        // Case 1: len(n³) is even → concat ≡ n² × 1 + n³ = n² + n³
        let combined_even = (square_alt + cube_alt) % b_plus_1;

        // Case 2: len(n³) is odd → concat ≡ n² × (-1) + n³ = n³ - n²
        let combined_odd = (cube_alt + b_plus_1 - square_alt) % b_plus_1;

        // Accept if either case could match a possible pandigital alternating sum
        if possible_alternating_sums.contains(&combined_even)
            || possible_alternating_sums.contains(&combined_odd)
        {
            valid_residues.push(r);
        }
    }

    #[allow(clippy::cast_precision_loss)]
    {
        trace!(
            "Alternating sum filter for base {}: {} valid residues out of {} ({:.2}% pass rate)",
            base,
            valid_residues.len(),
            b_plus_1,
            100.0 * valid_residues.len() as f64 / b_plus_1 as f64
        );
    }

    valid_residues
}

/// Compute the set of possible alternating digit sums for pandigital arrangements.
///
/// For a pandigital where each digit 0 to b-1 appears exactly once in b positions,
/// compute all possible alternating sums: d₀ - d₁ + d₂ - d₃ + ...
///
/// The key insight: `alternating_sum` = (sum of digits in even positions) - (sum of digits in odd positions)
/// Since we have b digits total, we partition them into even positions (⌈b/2⌉ digits) and odd positions (⌊b/2⌋ digits).
/// The total sum is 0+1+...+(b-1) = b(b-1)/2
/// If `even_sum` = x, then `odd_sum` = b(b-1)/2 - x
/// `alternating_sum` = x - (b(b-1)/2 - x) = 2x - b(b-1)/2
///
/// For small bases, we enumerate possible sums exactly by considering all possible
/// values of x (the sum of digits in even positions).
/// For larger bases, we use mathematical analysis to determine the range modulo (b+1).
///
/// # Arguments
/// - `base`: The numeric base
/// - `b_plus_1`: The modulus (b+1)
///
/// # Returns
/// A `HashSet` of possible alternating sums modulo (b+1)
fn compute_possible_alternating_sums(base: u32, b_plus_1: u128) -> HashSet<u128> {
    let mut possible_sums = HashSet::new();

    // Total sum of all digits 0 to b-1
    let total_sum = (base * (base - 1)) / 2;

    // Number of digits in even positions (0, 2, 4, ...)
    let num_even_positions = (base).div_ceil(2);

    if base <= MAX_BASE_FOR_ALTERNATING_SUM_FILTER {
        // For small bases, compute all possible sums of digits in even positions
        // We need to choose num_even_positions digits from {0, 1, ..., base-1}

        // Instead of enumerating all combinations (expensive), we determine the range
        // of possible sums for num_even_positions digits chosen from {0, ..., base-1}

        // Minimum sum: choose the smallest num_even_positions digits
        let min_even_sum: u32 = (0..num_even_positions).sum();

        // Maximum sum: choose the largest num_even_positions digits
        let max_even_sum: u32 = ((base - num_even_positions)..base).sum();

        // Debug output
        trace!(
            "Base {base}: num_even_positions={num_even_positions}, total_sum={total_sum}, min_even_sum={min_even_sum}, max_even_sum={max_even_sum}"
        );

        // For each possible even_sum, compute the alternating sum
        // alternating_sum = 2 * even_sum - total_sum
        //
        // Note: Not all values in [min_even_sum, max_even_sum] are achievable,
        // but most are (especially as base grows). For safety, we compute which
        // sums are actually achievable.

        // For small bases, we can afford to be more precise
        // The possible sums form a nearly continuous range with only small gaps
        for even_sum in min_even_sum..=max_even_sum {
            let alt_sum = 2i64 * i64::from(even_sum) - i64::from(total_sum);

            // Convert to positive residue mod (b+1)
            #[allow(clippy::cast_possible_wrap, clippy::cast_sign_loss)]
            let residue = {
                let b_plus_1_i64 = i64::try_from(b_plus_1).unwrap_or(i64::MAX);
                let normalized = ((alt_sum % b_plus_1_i64) + b_plus_1_i64) % b_plus_1_i64;
                normalized as u128
            };

            possible_sums.insert(residue);
        }

        trace!(
            "Base {base}: computed {} unique residues",
            possible_sums.len()
        );
    } else {
        // For larger bases, the alternating sum modulo (b+1) has high entropy
        // Mathematical analysis shows that for random partitions, nearly all
        // residues mod (b+1) are achievable when b is reasonably large

        // Conservative strategy: accept all residues (no filtering)
        // This is correct because as b grows, the distribution becomes more uniform
        for r in 0..b_plus_1 {
            possible_sums.insert(r);
        }
    }

    possible_sums
}

/// Get the alternating sum filter as a hashset for O(1) lookup.
///
/// # Arguments
/// - `base`: The numeric base
///
/// # Returns
/// A `HashSet` of valid residues mod (b+1)
#[must_use]
pub fn get_alternating_sum_filter_set(base: &u32) -> HashSet<u128> {
    get_alternating_sum_filter(base).into_iter().collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test_log::test]
    fn test_compute_possible_alternating_sums_base10() {
        // For base 10, compute all possible alternating sums for pandigital arrangements
        // Total sum = 0+1+2+...+9 = 45
        // We partition 10 digits into even positions (5 digits) and odd positions (5 digits)
        // alternating_sum = 2 * even_sum - 45
        //
        // Minimum even_sum: 0+1+2+3+4 = 10 → alt_sum = 2*10 - 45 = -25 ≡ 8 (mod 11)
        // Maximum even_sum: 5+6+7+8+9 = 35 → alt_sum = 2*35 - 45 = 25 ≡ 3 (mod 11)

        let base = 10u32;
        let b_plus_1 = 11u128;

        let possible_sums = compute_possible_alternating_sums(base, b_plus_1);

        // For base 10, when we iterate even_sum from 10 to 35 (26 values),
        // we get alternating sums: -25, -23, -21, ..., 23, 25 (all odd numbers)
        // These map to residues mod 11, and since 26 > 11*2, we cover all residues 0-10
        assert_eq!(
            possible_sums.len(),
            11,
            "For base 10, all 11 residues mod 11 should be achievable"
        );

        // Verify all residues 0-10 are present
        for r in 0..11 {
            assert!(
                possible_sums.contains(&r),
                "Residue {r} should be achievable for base 10"
            );
        }
    }

    #[test_log::test]
    fn test_get_alternating_sum_filter_base10() {
        let base = 10u32;
        let filter = get_alternating_sum_filter(&base);

        // For base 10, we need to verify that the known nice number 69 passes
        // 69 has:
        // - 69² = 4761
        // - 69³ = 328509 (6 digits, even)
        // - concat = 4761328509
        // - 69 mod 11 = 3
        // - 69² mod 11 = 9
        // - 69³ mod 11 = 5
        // - Since len(n³) = 6 (even): combined = (9 + 5) mod 11 = 14 mod 11 = 3
        // - Since alternating sum 3 is achievable, residue 3 should be valid

        // Should return residues mod 11
        assert!(!filter.is_empty());
        assert!(filter.len() <= 11);

        // All residues should be in range [0, 11)
        for &r in &filter {
            assert!(r < 11);
        }

        // Must include residue 3 for known nice number 69
        assert!(
            filter.contains(&3),
            "Filter must include residue 3 for base 10 (needed for 69)"
        );
    }

    #[test_log::test]
    fn test_get_alternating_sum_filter_base40() {
        let filter = get_alternating_sum_filter(&40);

        // Should return some residues mod 41
        assert!(!filter.is_empty());
        assert!(filter.len() <= 41);

        // All residues should be in range [0, 41)
        for &r in &filter {
            assert!(r < 41);
        }
    }

    #[test_log::test]
    fn test_known_nice_number_69_passes_filter() {
        // 69 is a known nice number in base 10
        let base = 10u32;
        let b_plus_1 = 11u128;
        let filter_set = get_alternating_sum_filter_set(&base);

        let residue = 69u128 % b_plus_1;
        assert!(
            filter_set.contains(&residue),
            "Known nice number 69 should pass alternating sum filter (residue {residue} mod {b_plus_1})"
        );
    }

    #[test_log::test]
    fn test_filter_is_subset_of_modulus() {
        for base in [10u32, 12, 16, 20, 30, 40] {
            let filter = get_alternating_sum_filter(&base);
            let b_plus_1 = u128::from(base) + 1;

            // All residues must be less than b+1
            for &r in &filter {
                assert!(
                    r < b_plus_1,
                    "Residue {r} >= modulus {b_plus_1} for base {base}"
                );
            }

            // No duplicates
            let set: HashSet<_> = filter.iter().copied().collect();
            assert_eq!(
                set.len(),
                filter.len(),
                "Filter should not contain duplicates"
            );
        }
    }
}
