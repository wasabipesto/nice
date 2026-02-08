//! A module for filtering numbers by least significant digit (LSD).
//!
//! This filter uses a lightweight backtracking approach to determine which least significant
//! digits can possibly produce nice numbers in a given base. It explores only the first level
//! of the search tree (one node per possible LSD) to check if that digit leads to immediate
//! collision in n² and n³.
//!
//! The filter works across all bases because the LSD of a number completely determines the LSD
//! of its square and cube (via modular arithmetic: if n ≡ d (mod b), then n² ≡ d² (mod b)).
//!
//! At low bases this filter is quite effective (filters out up to 60% of candidates) but its
//! effectiveness is sporadic and diminishes somewhat at higher bases. I experimented with
//! searching deeper in the tree but it didn't improve the results significantly.
//!
//! ## How It Works
//!
//! For each possible LSD (0 to base-1):
//! 1. Compute the LSD of n² and n³ for that starting digit
//! 2. Check if the square and cube have the same LSD (which would create a duplicate)
//! 3. If no collision, the LSD is valid
//!
//! ## Example for Base 10
//!
//! The filter checks each digit and accepts those where `square_lsd` ≠ `cube_lsd`:
//! - LSD=0: 0²=0, 0³=0 → collision (both 0) ✗
//! - LSD=1: 1²=1, 1³=1 → collision (both 1) ✗
//! - LSD=2: 2²=4, 2³=8 → LSDs are 4 and 8, no collision ✓
//! - LSD=3: 3²=9, 3³=27 → LSDs are 9 and 7, no collision ✓
//! - LSD=4: 4²=16, 4³=64 → LSDs are 6 and 4, no collision ✓
//! - LSD=5: 5²=25, 5³=125 → collision (both 5) ✗
//! - LSD=6: 6²=36, 6³=216 → collision (both 6) ✗
//! - LSD=7: 7²=49, 7³=343 → LSDs are 9 and 3, no collision ✓
//! - LSD=8: 8²=64, 8³=512 → LSDs are 4 and 2, no collision ✓
//! - LSD=9: 9²=81, 9³=729 → LSDs are 1 and 9, no collision ✓
//!
//! Result: Valid LSDs for base 10 are {2, 3, 4, 7, 8, 9}, filtering out 40% of candidates.
//!
//! This eliminates a significant portion of the search space with minimal computation.
//!
//! ## Multi-Digit LSD Filter
//!
//! The multi-digit LSD filter extends the single-digit approach to check the last k digits
//! (mod b^k) instead of just the last digit. This is more effective because it catches
//! collisions that occur in the second, third, etc. positions from the right.
//!
//! For example, in base 10 with k=2, instead of just checking if the last digit of n² and n³
//! collide, we check if ANY digit in the last 2 digits of n² collides with ANY digit in the
//! last 2 digits of n³.

use log::trace;
use malachite::base::num::arithmetic::traits::Pow;
use malachite::natural::Natural;
use std::collections::HashSet;

/// Get a list of valid least significant digits for a base.
///
/// Returns a vector of LSD values (0 to base-1) that could potentially
/// produce nice numbers. Numbers with other LSDs are guaranteed to fail
/// and can be skipped.
///
/// # Arguments
/// - `base`: The numeric base
///
/// # Returns
/// A vector of valid LSD values
#[must_use]
pub fn get_valid_lsds(base: &u32) -> Vec<u32> {
    (0..*base).filter(|&lsd| is_valid_lsd(lsd, *base)).collect()
}

/// Get a list of valid least significant digits as u128 for easier filtering.
///
/// # Arguments
/// - `base`: The numeric base
///
/// # Returns
/// A vector of valid LSD values as u128
#[must_use]
pub fn get_valid_lsds_u128(base: &u32) -> Vec<u128> {
    get_valid_lsds(base).into_iter().map(u128::from).collect()
}

/// Check if a specific LSD can potentially produce a nice number.
///
/// This is done by computing the LSD of n² and n³ for a single-digit number
/// and checking if they are the same (which would create a guaranteed duplicate
/// in the output).
///
/// # Arguments
/// - `lsd`: The least significant digit to test
/// - `base`: The numeric base
///
/// # Returns
/// `true` if this LSD could produce a nice number, `false` if it definitely cannot
fn is_valid_lsd(lsd: u32, base: u32) -> bool {
    // The candidate number is just the LSD itself (e.g., 0, 1, 2, ...)
    let n = Natural::from(lsd);
    let base_natural = Natural::from(base);

    // Compute n² and n³
    let n_squared = (&n).pow(2);
    let n_cubed = n.pow(3);

    // Extract the least significant digit (position 0) of n² and n³
    let square_lsd = u32::try_from(&(n_squared % &base_natural)).expect("LSD should fit in u32");
    let cube_lsd = u32::try_from(&(n_cubed % &base_natural)).expect("LSD should fit in u32");

    trace!(
        "Testing LSD {} in base {} - Square LSD: {}, Cube LSD: {}, Collision: {}",
        lsd,
        base,
        square_lsd,
        cube_lsd,
        square_lsd == cube_lsd
    );

    // Check for collision: if square and cube have the same LSD, it's invalid.
    // This would create a guaranteed duplicate in the combined digits of n² and n³.
    // Returns `true` if this LSD could produce a nice number, `false` if it definitely cannot
    square_lsd != cube_lsd
}

/// Extract all digits from a number in the given base.
///
/// # Arguments
/// - `value`: The number to extract digits from
/// - `base`: The numeric base
/// - `num_digits`: Maximum number of digits to extract (from least significant)
///
/// # Returns
/// A `HashSet` containing all unique digits that appear in the number
fn extract_digits(value: u128, base: u32, num_digits: u32) -> HashSet<u32> {
    let mut digits = HashSet::new();
    let mut remaining = value;
    let base_u128 = u128::from(base);

    for _ in 0..num_digits {
        #[allow(clippy::cast_possible_truncation)]
        let digit = (remaining % base_u128) as u32;
        digits.insert(digit);
        remaining /= base_u128;
        if remaining == 0 {
            break;
        }
    }

    digits
}

/// Get a bitmap of valid k-digit suffixes for multi-digit LSD filtering.
///
/// This is Filter A from the Novel Filters document. Instead of checking only the last
/// digit, it checks the last k digits (mod b^k) and validates that no digit collision
/// occurs between the k-digit suffixes of n² and n³.
///
/// # Arguments
/// - `base`: The numeric base
/// - `k`: Number of digits to check (typically 2 or 3)
///
/// # Returns
/// A `Vec<bool>` where `bitmap[i]` is true if suffix i is valid. This allows fast O(1)
/// lookups using direct array indexing.
///
/// # Example
/// For base 10, k=2:
/// - Check all suffixes 00-99
/// - For suffix=12: compute 12²=144 (last 2 digits: 44) and 12³=1728 (last 2 digits: 28)
/// - Extract digits from "44": {4} and from "28": {2, 8}
/// - Since {4} and {2, 8} are disjoint (no shared digits), suffix 12 is valid
///
/// # Panics
/// Panics if base^k would overflow u32
#[must_use]
pub fn get_valid_multi_lsd_bitmap(base: u32, k: u32) -> Vec<bool> {
    // Calculate modulus = base^k
    let modulus = base.checked_pow(k).expect("base^k must fit in u32");
    let modulus_u128 = u128::from(modulus);

    trace!(
        "Computing multi-digit LSD filter for base {base} with k={k} digits (modulus={modulus})"
    );

    // Build a bitmap for fast O(1) lookup via direct indexing
    let mut bitmap = vec![false; modulus as usize];
    let mut valid_count = 0;

    for suffix in 0..modulus {
        let suffix_u128 = u128::from(suffix);

        // Compute n² mod b^k and n³ mod b^k
        let sq = suffix_u128.pow(2) % modulus_u128;
        let cb = suffix_u128.pow(3) % modulus_u128;

        // Extract all digits from the k-digit representations
        let sq_digits = extract_digits(sq, base, k);
        let cb_digits = extract_digits(cb, base, k);

        // Valid if no digit appears in both square and cube
        let is_valid = sq_digits.is_disjoint(&cb_digits);

        if is_valid {
            bitmap[suffix as usize] = true;
            valid_count += 1;
        } else {
            trace!(
                "  Suffix {:0width$} rejected: sq={:0width$} (digits {:?}), cb={:0width$} (digits {:?})",
                suffix,
                sq,
                sq_digits,
                cb,
                cb_digits,
                width = k as usize
            );
        }
    }

    #[allow(clippy::cast_precision_loss)]
    let filter_rate = 100.0 * (1.0 - f64::from(valid_count) / f64::from(modulus));
    trace!(
        "Multi-digit LSD filter: {valid_count}/{modulus} suffixes valid ({filter_rate:.1}% filtered)"
    );

    bitmap
}

/// Get the recommended k value for multi-digit LSD filtering based on base.
///
/// # Arguments
/// - `base`: The numeric base
///
/// # Returns
/// Recommended number of digits to check
#[must_use]
pub fn get_recommended_k(_base: u32) -> u32 {
    // if base >= 30 { 2 } else { 3 };
    // Higher k-values reduced performance instead of increasing it, locking k to 1.
    1
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test_log::test]
    fn test_get_valid_lsds_base10() {
        let valid_lsds = get_valid_lsds(&10);
        assert_eq!(valid_lsds, vec![2, 3, 4, 7, 8, 9]);
    }

    #[test_log::test]
    fn test_known_nice_number_69_passes_filter() {
        // 69 is a KNOWN nice number in base 10:
        // 69² = 4761
        // 69³ = 328509
        // Combined digits: 4,7,6,1,3,2,8,5,0,9 = all 10 digits (pandigital!)
        //
        // 69 ends in 9. For all numbers ending in 9:
        // - Their squares always end in 1 (since 9² = 81)
        // - Their cubes always end in 9 (since 9³ = 729)
        // - This means digits 1 and 9 appear in the output (no collision!)
        //
        // The filter correctly accepts LSD=9 because square_lsd (1) != cube_lsd (9).

        let base = 10u32;
        let lsd_filter = get_valid_lsds_u128(&base);

        let sixty_nine_lsd = 69u128 % u128::from(base);
        assert_eq!(sixty_nine_lsd, 9, "69 ends in 9");

        // The filter correctly includes 9
        assert!(
            lsd_filter.contains(&9),
            "LSD 9 should pass filter - 69 is a known nice number!"
        );

        // This means 69 will be checked in process_range_niceonly
    }

    #[test_log::test]
    fn test_lsd_filter_allows_valid_candidates() {
        // Test that numbers ending in valid LSDs pass the filter
        let base = 10u32;
        let lsd_filter = get_valid_lsds_u128(&base);

        // Numbers ending in 2, 3, 4, 7, 8, 9 should pass
        assert!(lsd_filter.contains(&(12u128 % u128::from(base))));
        assert!(lsd_filter.contains(&(23u128 % u128::from(base))));
        assert!(lsd_filter.contains(&(44u128 % u128::from(base))));
        assert!(lsd_filter.contains(&(47u128 % u128::from(base))));
        assert!(lsd_filter.contains(&(98u128 % u128::from(base))));
        assert!(lsd_filter.contains(&(99u128 % u128::from(base))));

        // Numbers ending in 0, 1, 5, 6 should be filtered
        assert!(!lsd_filter.contains(&(10u128 % u128::from(base))));
        assert!(!lsd_filter.contains(&(21u128 % u128::from(base))));
        assert!(!lsd_filter.contains(&(55u128 % u128::from(base))));
        assert!(!lsd_filter.contains(&(66u128 % u128::from(base))));
    }

    #[test_log::test]
    fn test_get_valid_lsds_u128() {
        let valid = get_valid_lsds_u128(&10);

        // Should return u128 values matching the u32 version
        assert_eq!(valid, vec![2u128, 3u128, 4u128, 7u128, 8u128, 9u128]);
        assert!(valid.iter().all(|&x| x < 10));
    }

    #[test_log::test]
    fn test_get_valid_lsds_base40() {
        let valid = get_valid_lsds(&40);

        // Should have some valid LSDs
        assert!(!valid.is_empty());

        // Should filter out at least some LSDs (not all can be valid)
        assert!(valid.len() < 40);

        // At minimum, 0 and 1 should always be filtered
        assert!(!valid.contains(&0), "0 should always be filtered");
        assert!(!valid.contains(&1), "1 should always be filtered");

        // All returned LSDs should be in valid range
        assert!(valid.iter().all(|&lsd| lsd < 40));

        // Should be sorted (since we're iterating 0..base)
        let mut sorted = valid.clone();
        sorted.sort_unstable();
        assert_eq!(valid, sorted);
    }

    #[test_log::test]
    fn test_various_bases() {
        // Test that the filter works for various bases
        for base in [10u32, 12, 16, 20, 40, 50] {
            let valid = get_valid_lsds(&base);

            // Should return some valid LSDs
            assert!(!valid.is_empty(), "Base {base} should have some valid LSDs");

            // Should filter out at least some LSDs (0 and 1 minimum)
            assert!(
                valid.len() < base as usize,
                "Base {base} should filter at least some LSDs"
            );

            // 0 and 1 should always be filtered
            assert!(!valid.contains(&0), "Base {base} should filter 0");
            assert!(!valid.contains(&1), "Base {base} should filter 1");

            // All returned LSDs should be in valid range
            assert!(
                valid.iter().all(|&lsd| lsd < base),
                "Base {base} has LSD out of range"
            );

            // Should be sorted (since we're iterating 0..base)
            let mut sorted = valid.clone();
            sorted.sort_unstable();
            assert_eq!(valid, sorted, "Base {base} LSDs not sorted");
        }
    }

    #[test_log::test]
    fn test_filter_effectiveness() {
        // Verify the filter actually reduces the search space significantly
        let base10_valid = get_valid_lsds(&10);

        #[allow(clippy::cast_precision_loss)]
        let base10_filtered_pct = (10 - base10_valid.len()) as f32 / 10.0 * 100.0;
        assert!(
            base10_filtered_pct >= 30.0,
            "Base 10 should filter at least 30% of candidates, got {base10_filtered_pct:.1}%"
        );

        // Test other bases have reasonable filtering
        for base in [12u32, 20, 30, 40] {
            let valid = get_valid_lsds(&base);
            let filtered_count = base as usize - valid.len();
            assert!(
                filtered_count >= 2,
                "Base {base} should filter at least 2 LSDs, filtered {filtered_count}"
            );
        }
    }

    #[test_log::test]
    fn test_lsd_filter_integration() {
        // Simulate how this would be used in `process_niceonly`
        let base = 10u32;
        let lsd_filter = get_valid_lsds_u128(&base);

        // Check various numbers
        let test_numbers = vec![47u128, 69u128, 100u128, 123u128, 182u128, 188u128];
        let filtered: Vec<u128> = test_numbers
            .into_iter()
            .filter(|num| lsd_filter.contains(&(num % u128::from(base))))
            .collect();

        // 47 ends in 7 (valid), 69 ends in 9 (valid), 100 ends in 0 (filtered),
        // 123 ends in 3 (valid), 182 ends in 2 (valid), 188 ends in 8 (valid)
        assert_eq!(filtered, vec![47u128, 69u128, 123u128, 182u128, 188u128]);
        assert_eq!(filtered.len(), 5);
    }

    #[test_log::test]
    fn test_idempotent_lsds_correctly_filtered() {
        // Test that idempotent LSDs (where x², x³ both end in x) are correctly filtered
        // These create guaranteed collisions in the output
        let base = 10;

        // In base 10, the idempotent LSDs are: 0, 1, 5, 6
        // 0² = 0, 0³ = 0 (both end in 0)
        // 1² = 1, 1³ = 1 (both end in 1)
        // 5² = 25, 5³ = 125 (both end in 5)
        // 6² = 36, 6³ = 216 (both end in 6)

        for idempotent in [0u32, 1, 5, 6] {
            assert!(
                !is_valid_lsd(idempotent, base),
                "Idempotent LSD {idempotent} correctly filtered (square_lsd == cube_lsd)"
            );
        }
    }

    #[test_log::test]
    fn test_get_valid_lsds_base12() {
        // Test base 12 (duodecimal)
        // Valid LSDs: 2, 3, 5, 7, 8, 11
        // Filtered: 0, 1, 4, 6, 9, 10 (50% filtered)
        let valid_lsds = get_valid_lsds(&12);
        assert_eq!(valid_lsds, vec![2, 3, 5, 7, 8, 11]);

        // Verify specific collision cases:
        // LSD=0: 0²=0, 0³=0 → both 0 (collision)
        assert!(!is_valid_lsd(0, 12));
        // LSD=1: 1²=1, 1³=1 → both 1 (collision)
        assert!(!is_valid_lsd(1, 12));
        // LSD=4: 4²=16₁₀=14₁₂, 4³=64₁₀=54₁₂ → both end in 4 (collision)
        assert!(!is_valid_lsd(4, 12));
        // LSD=6: 6²=36₁₀=30₁₂, 6³=216₁₀=160₁₂ → both end in 0 (collision)
        assert!(!is_valid_lsd(6, 12));
        // LSD=9: 9²=81₁₀=69₁₂, 9³=729₁₀=509₁₂ → both end in 9 (collision)
        assert!(!is_valid_lsd(9, 12));
        // LSD=10: 10²=100₁₀=84₁₂, 10³=1000₁₀=6B4₁₂ → both end in 4 (collision)
        assert!(!is_valid_lsd(10, 12));

        // Verify valid cases:
        // LSD=2: 2²=4, 2³=8 → 4 and 8 (no collision)
        assert!(is_valid_lsd(2, 12));
        // LSD=3: 3²=9, 3³=27₁₀=23₁₂ → 9 and 3 (no collision)
        assert!(is_valid_lsd(3, 12));
        // LSD=11: 11²=121₁₀=A1₁₂, 11³=1331₁₀=927₁₂ → 1 and 7 (no collision)
        assert!(is_valid_lsd(11, 12));
    }

    #[test_log::test]
    fn test_get_valid_lsds_base16() {
        // Test base 16 (hexadecimal)
        // Valid LSDs: 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15
        // Filtered: 0, 1, 4, 8, 12 (31.25% filtered)
        let valid_lsds = get_valid_lsds(&16);
        assert_eq!(valid_lsds, vec![2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15]);

        // Verify specific collision cases:
        // LSD=0: both end in 0
        assert!(!is_valid_lsd(0, 16));
        // LSD=1: both end in 1
        assert!(!is_valid_lsd(1, 16));
        // LSD=4: 4²=10₁₆, 4³=40₁₆ → both end in 0 (collision)
        assert!(!is_valid_lsd(4, 16));
        // LSD=8: 8²=40₁₆, 8³=200₁₆ → both end in 0 (collision)
        assert!(!is_valid_lsd(8, 16));
        // LSD=12 (C): C²=90₁₆, C³=6C0₁₆ → both end in 0 (collision)
        assert!(!is_valid_lsd(12, 16));

        // Verify valid cases:
        // LSD=2: 2²=4, 2³=8 → 4 and 8 (no collision)
        assert!(is_valid_lsd(2, 16));
        // LSD=3: 3²=9, 3³=1B₁₆ → 9 and B (no collision)
        assert!(is_valid_lsd(3, 16));
        // LSD=15 (F): F²=E1₁₆, F³=D2F₁₆ → 1 and F (no collision)
        assert!(is_valid_lsd(15, 16));
    }

    #[test_log::test]
    fn test_extract_digits_base10() {
        // Test extracting digits from various numbers in base 10
        assert_eq!(extract_digits(0, 10, 2), HashSet::from([0]));
        assert_eq!(extract_digits(5, 10, 2), HashSet::from([5]));
        assert_eq!(extract_digits(12, 10, 2), HashSet::from([1, 2]));
        assert_eq!(extract_digits(44, 10, 2), HashSet::from([4]));
        assert_eq!(extract_digits(28, 10, 2), HashSet::from([2, 8]));
        assert_eq!(extract_digits(123, 10, 3), HashSet::from([1, 2, 3]));
        assert_eq!(extract_digits(111, 10, 3), HashSet::from([1]));
    }

    #[test_log::test]
    fn test_extract_digits_base16() {
        // Test extracting digits from various numbers in base 16
        assert_eq!(extract_digits(0x0, 16, 2), HashSet::from([0]));
        assert_eq!(extract_digits(0xF, 16, 2), HashSet::from([15]));
        assert_eq!(extract_digits(0x1A, 16, 2), HashSet::from([1, 10]));
        assert_eq!(extract_digits(0xFF, 16, 2), HashSet::from([15]));
        assert_eq!(extract_digits(0xAB, 16, 2), HashSet::from([10, 11]));
    }

    #[test_log::test]
    fn test_get_valid_multi_lsd_bitmap_base10_k1() {
        // With k=1, should match single-digit LSD filter
        let multi_lsd_bitmap = get_valid_multi_lsd_bitmap(10, 1);
        let single_lsd = get_valid_lsds_u128(&10);

        // Check that all single-digit valid LSDs are marked as valid in the bitmap
        for &valid_lsd in &single_lsd {
            assert!(
                multi_lsd_bitmap[valid_lsd as usize],
                "Single-LSD valid digit {valid_lsd} should be valid in multi-digit bitmap"
            );
        }

        // Check that invalid single-digit LSDs are not in the bitmap
        for lsd in 0..10u128 {
            if single_lsd.contains(&lsd) {
                assert!(multi_lsd_bitmap[lsd as usize]);
            } else {
                assert!(!multi_lsd_bitmap[lsd as usize]);
            }
        }
    }

    #[test_log::test]
    fn test_get_valid_multi_lsd_bitmap_base10_k2() {
        // Test multi-digit LSD filter with k=2 for base 10
        let bitmap = get_valid_multi_lsd_bitmap(10, 2);

        // Should have 100 entries (0-99)
        assert_eq!(
            bitmap.len(),
            100,
            "Bitmap should have 100 entries for base 10, k=2"
        );

        let valid_count = bitmap.iter().filter(|&&v| v).count();
        assert!(valid_count < 100, "k=2 should filter some candidates");
        assert!(valid_count > 0, "k=2 should have some valid suffixes");

        // Check specific cases:
        // 00: 00²=00, 00³=00 → both have digit 0 (collision)
        assert!(!bitmap[0]);

        // 01: 01²=01, 01³=01 → both have digits 0,1 (collision)
        assert!(!bitmap[1]);

        // 12: 12²=144 (last 2: 44={4}), 12³=1728 (last 2: 28={2,8}) → disjoint
        assert!(
            bitmap[12],
            "12 should be valid: sq digits {{4}} ∩ cb digits {{2,8}} = ∅"
        );

        // 69: 69²=4761 (last 2: 61={6,1}), 69³=328509 (last 2: 09={0,9}) → disjoint
        assert!(bitmap[69], "69 should be valid (known nice number)");
    }

    #[test_log::test]
    fn test_get_valid_multi_lsd_bitmap_base10_k3() {
        // Test multi-digit LSD filter with k=3 for base 10
        let bitmap = get_valid_multi_lsd_bitmap(10, 3);

        // Should have 1000 entries (0-999)
        assert_eq!(
            bitmap.len(),
            1000,
            "Bitmap should have 1000 entries for base 10, k=3"
        );

        let valid_count = bitmap.iter().filter(|&&v| v).count();
        assert!(valid_count < 1000, "k=3 should filter some candidates");
        assert!(valid_count > 0, "k=3 should have some valid suffixes");

        // 069: 69²=4761 (last 3: 761={7,6,1}), 69³=328509 (last 3: 509={5,0,9}) → disjoint
        assert!(bitmap[69], "069 should be valid (known nice number)");
    }

    #[test_log::test]
    fn test_multi_lsd_filter_more_effective() {
        // Verify that multi-digit filter is more effective than single-digit
        let base = 10;

        let k1_bitmap = get_valid_multi_lsd_bitmap(base, 1);
        let k2_bitmap = get_valid_multi_lsd_bitmap(base, 2);

        let k1_valid_count = k1_bitmap.iter().filter(|&&v| v).count();
        let k2_valid_count = k2_bitmap.iter().filter(|&&v| v).count();

        #[allow(clippy::cast_precision_loss)]
        let k1_rate = k1_valid_count as f64 / f64::from(base);
        #[allow(clippy::cast_precision_loss)]
        let k2_rate = k2_valid_count as f64 / f64::from(base.pow(2));

        // k=2 should have a lower pass rate (more filtering)
        assert!(
            k2_rate < k1_rate,
            "Multi-digit filter (k=2) should be more effective: k1={k1_rate:.3}, k2={k2_rate:.3}"
        );
    }

    #[test_log::test]
    fn test_multi_lsd_includes_single_lsd_valid_numbers() {
        // Numbers that pass single-digit LSD should be considered by multi-digit
        // (though multi-digit may filter more aggressively)
        let base = 10;
        let k = 2;

        let single_lsd_valid = get_valid_lsds_u128(&base);
        let multi_lsd_bitmap = get_valid_multi_lsd_bitmap(base, k);

        // For any number ending in a valid single-digit LSD, at least some
        // k-digit suffixes ending in that digit should be valid in multi-digit filter
        for &valid_lsd in &single_lsd_valid {
            let has_valid_suffix = (0..base.pow(k - 1)).any(|prefix| {
                let suffix = (u128::from(prefix) * u128::from(base) + valid_lsd) as usize;
                multi_lsd_bitmap[suffix]
            });

            assert!(
                has_valid_suffix,
                "Multi-digit filter should have at least one suffix ending in single-LSD-valid digit {valid_lsd}"
            );
        }
    }
}
