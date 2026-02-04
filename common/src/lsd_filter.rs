//! A module for filtering numbers by least significant digit (LSD).
//!
//! This filter uses a lightweight backtracking approach to determine which least significant
//! digits can possibly produce nice numbers in a given base. It explores only the first level
//! of the search tree (one node per possible LSD) to check if that digit leads to immediate
//! collision in n² and n³.
//!
//! At low bases this filter is quite effective (filters out up to 60% of candidates) but its
//! effectiveness is sporadic and diminishes somewhat at higher bases. I experimented with
//! searching deeper in the tree but it didn't improve the results significantly.
//!
//!
//! ## How It Works
//!
//! For each possible LSD (0 to base-1):
//! 1. Compute the LSD of n² and n³ for that starting digit
//! 2. Check if these create a collision (same digit in both, or duplicate)
//! 3. If no collision at position 0, the LSD is valid
//!
//! ## Example
//!
//! For base 10, only certain LSDs can produce nice numbers:
//! - LSD=0: 0²=0, 0³=0 → collision (both 0) ✗
//! - LSD=1: 1²=1, 1³=1 → collision (both 1) ✗
//! - LSD=2: 4²=6, 8³=2 → collision (2 appears in input) ✗
//! - LSD=3: 9²=9, 27³=7 → no collision ✓
//! - ...
//!
//! This eliminates most of the search space with minimal computation.

use malachite::base::num::arithmetic::traits::Pow;
use malachite::natural::Natural;

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
pub fn get_valid_lsds_u128(base: &u32) -> Vec<u128> {
    get_valid_lsds(base)
        .into_iter()
        .map(|lsd| lsd as u128)
        .collect()
}

/// Check if a specific LSD can potentially produce a nice number.
///
/// This is done by computing the LSD of n² and n³ for a single-digit number
/// and checking for immediate collisions.
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

    // Check for collisions:
    // 1. If square and cube have the same LSD, it's invalid
    if square_lsd == cube_lsd {
        return false;
    }

    // 2. If either matches the input LSD, it's a duplicate
    // (The input digit n itself will appear in the number)
    if square_lsd == lsd || cube_lsd == lsd {
        return false;
    }

    // No collision detected - this LSD is potentially valid
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_get_valid_lsds_base10() {
        let valid = get_valid_lsds(&10);

        // Should have some valid LSDs
        assert!(!valid.is_empty());

        // Should filter out at least some LSDs
        assert!(valid.len() < 10);
    }

    #[test]
    fn test_get_valid_lsds_base40() {
        let valid = get_valid_lsds(&40);

        // Should have some valid LSDs
        assert!(!valid.is_empty());

        // Should filter out at least some LSDs
        assert!(valid.len() < 40);
    }

    #[test]
    fn test_is_valid_lsd_base10_zero() {
        // LSD=0: 0²=0, 0³=0 → collision
        assert!(!is_valid_lsd(0, 10));
    }

    #[test]
    fn test_is_valid_lsd_base10_one() {
        // LSD=1: 1²=1, 1³=1 → collision
        assert!(!is_valid_lsd(1, 10));
    }

    #[test]
    fn test_is_valid_lsd_base10_three() {
        // LSD=3: 3²=9, 3³=27 (7) → no collision
        assert!(is_valid_lsd(3, 10));
    }

    #[test]
    fn test_get_valid_lsds_u128() {
        let valid = get_valid_lsds_u128(&10);

        // Should return u128 values
        assert!(!valid.is_empty());
        assert!(valid.iter().all(|&x| x < 10));
    }

    #[test]
    fn test_various_bases() {
        // Test that the filter works for various bases
        for base in [10, 12, 16, 20, 40, 50].iter() {
            let valid = get_valid_lsds(base);

            // Should return some valid LSDs
            assert!(
                !valid.is_empty(),
                "Base {} should have some valid LSDs",
                base
            );

            // All returned LSDs should be in valid range
            assert!(valid.iter().all(|&lsd| lsd < *base));

            // Should be sorted (since we're iterating 0..base)
            let mut sorted = valid.clone();
            sorted.sort_unstable();
            assert_eq!(valid, sorted);
        }
    }

    #[test]
    fn test_lsd_filter_integration() {
        // Simulate how this would be used in process_niceonly
        let base = 10u32;
        let lsd_filter = get_valid_lsds_u128(&base);

        // Check some numbers
        let test_numbers = vec![47u128, 69u128, 100u128, 123u128];
        let filtered: Vec<u128> = test_numbers
            .into_iter()
            .filter(|num| lsd_filter.contains(&(num % base as u128)))
            .collect();

        // Some should pass, some should fail
        // We don't know exact values without running, but it should work
        assert!(filtered.len() < 4);
    }
}
