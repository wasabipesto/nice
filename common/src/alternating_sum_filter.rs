//! Alternating sum filter based on parity constraints.
//!
//! This module implements a theorem about square-cube pandigital numbers:
//! For base b where b = 5ℓ and b(b-1)/2 is odd, a square-cube pandigital pair
//! requires n²(1+n) mod (b+1) to have the same parity as b(b-1)/2.
//!
//! # Mathematical Background
//!
//! In base b, the alternating digit sum equals the number mod (b+1):
//! - n = d₀ + d₁·b + d₂·b² + ... ≡ d₀ - d₁ + d₂ - ... (mod b+1)
//!
//! For a pandigital concatenation n² || n³ with b = 5ℓ total digits:
//! - If ℓ is even, positions split evenly between even and odd
//! - The alternating sum A₂ + A₃ = n²(1+n) mod (b+1)
//! - Parity constraint: A₂ + A₃ ≡ b(b-1)/2 mod 2
//!
//! # Applicability
//!
//! The filter applies when b(b-1)/2 is odd, which occurs when b ≡ 1 or 2 mod 4.
//! For even bases (which b = 5ℓ always is), this means b ≡ 2 mod 4.
//!
//! Examples:
//! - Base 10: 10·9/2 = 45 (odd) ✓ - Filter applies
//! - Base 40: 40·39/2 = 780 (even) ✗ - Filter doesn't apply
//! - Base 50: 50·49/2 = 1225 (odd) ✓ - Filter applies
//! - Base 70: 70·69/2 = 2415 (odd) ✓ - Filter applies
//!
//! # CRT Compatibility
//!
//! For even bases: gcd(b+1, b-1) = gcd(b+1, 2) = 1 and gcd(b+1, b^k) = 1.
//! This filter is fully CRT-compatible with existing residue and LSD filters.

/// Check if the alternating sum filter applies to a given base.
///
/// The filter applies when b(b-1)/2 is odd, which for even bases means b ≡ 2 mod 4.
///
/// # Arguments
/// - `base`: The numeric base
///
/// # Returns
/// `true` if the filter can eliminate candidates, `false` otherwise
#[must_use]
pub fn is_filter_applicable(base: &u32) -> bool {
    // b(b-1)/2 is odd when b ≡ 1 or 2 mod 4
    // For even bases (b = 5ℓ), this means b ≡ 2 mod 4
    matches!(base % 4, 1 | 2)
}

/// Get the target parity for the alternating sum.
///
/// Returns b(b-1)/2 mod 2, which determines whether valid residues must have
/// odd or even alternating sums.
///
/// # Arguments
/// - `base`: The numeric base
///
/// # Returns
/// `true` if the target parity is odd, `false` if even
#[must_use]
pub fn get_target_parity(base: &u32) -> bool {
    // b(b-1)/2 mod 2
    // For b ≡ 1 or 2 mod 4: odd (true)
    // For b ≡ 0 or 3 mod 4: even (false)
    is_filter_applicable(base)
}

/// Get a list of valid residues mod (b+1) for the alternating sum filter.
///
/// Returns all residues r ∈ [0, b+1) where r²(1+r) mod (b+1) has the correct parity.
/// If the filter doesn't apply to this base, returns an empty vector.
///
/// # Arguments
/// - `base`: The numeric base
///
/// # Returns
/// Vector of valid residues mod (b+1), or empty if filter doesn't apply
#[must_use]
pub fn get_valid_residues(base: &u32) -> Vec<u32> {
    if !is_filter_applicable(base) {
        return Vec::new();
    }

    let modulus = base + 1;
    let target_parity = get_target_parity(base);

    (0..modulus)
        .filter(|&r| {
            let r_u128 = u128::from(r);
            let r_squared = r_u128 * r_u128;
            let r_cubed = r_squared * r_u128;
            let alternating_sum = (r_squared + r_cubed) % u128::from(modulus);
            let parity = alternating_sum % 2 == 1;
            parity == target_parity
        })
        .collect()
}

/// Get a list of valid residues mod (b+1) as u128 for easier processing.
///
/// # Arguments
/// - `base`: The numeric base
///
/// # Returns
/// Vector of valid residues mod (b+1) as u128, or empty if filter doesn't apply
#[must_use]
pub fn get_valid_residues_u128(base: &u32) -> Vec<u128> {
    get_valid_residues(base)
        .iter()
        .map(|&r| u128::from(r))
        .collect()
}

/// Check if a specific number passes the alternating sum filter.
///
/// # Arguments
/// - `n`: The number to check
/// - `base`: The numeric base
///
/// # Returns
/// `true` if the number passes the filter or the filter doesn't apply
#[must_use]
pub fn passes_filter(n: u128, base: u32) -> bool {
    if !is_filter_applicable(&base) {
        return true;
    }

    let modulus = u128::from(base + 1);
    let r = n % modulus;
    let r_squared = r * r;
    let r_cubed = r_squared * r;
    let alternating_sum = (r_squared + r_cubed) % modulus;
    let parity = alternating_sum % 2 == 1;
    let target_parity = get_target_parity(&base);

    parity == target_parity
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test_log::test]
    fn test_is_filter_applicable() {
        // b ≡ 1 or 2 mod 4 should apply
        assert!(is_filter_applicable(&10)); // 10 ≡ 2 mod 4
        assert!(is_filter_applicable(&50)); // 50 ≡ 2 mod 4
        assert!(is_filter_applicable(&70)); // 70 ≡ 2 mod 4
        assert!(is_filter_applicable(&90)); // 90 ≡ 2 mod 4

        // b ≡ 0 or 3 mod 4 should not apply
        assert!(!is_filter_applicable(&40)); // 40 ≡ 0 mod 4
        assert!(!is_filter_applicable(&60)); // 60 ≡ 0 mod 4
        assert!(!is_filter_applicable(&80)); // 80 ≡ 0 mod 4
        assert!(!is_filter_applicable(&100)); // 100 ≡ 0 mod 4
    }

    #[test_log::test]
    fn test_get_target_parity() {
        // For applicable bases, target parity is odd
        assert!(get_target_parity(&10)); // 45 is odd
        assert!(get_target_parity(&50)); // 1225 is odd
        assert!(get_target_parity(&70)); // 2415 is odd

        // For non-applicable bases, target parity is even
        assert!(!get_target_parity(&40)); // 780 is even
        assert!(!get_target_parity(&60)); // 1770 is even
    }

    #[test_log::test]
    fn test_get_valid_residues_base10() {
        let residues = get_valid_residues(&10);

        // Filter applies to base 10
        assert!(!residues.is_empty());

        // All residues should be < 11 (b+1)
        assert!(residues.iter().all(|&r| r < 11));

        // Should filter roughly half the residues
        let expected_count = 11 / 2;
        assert!(
            residues.len() >= expected_count - 1 && residues.len() <= expected_count + 1,
            "Expected approximately {} residues, got {}",
            expected_count,
            residues.len()
        );
    }

    #[test_log::test]
    fn test_get_valid_residues_base40() {
        let residues = get_valid_residues(&40);

        // Filter doesn't apply to base 40
        assert!(residues.is_empty());
    }

    #[test_log::test]
    fn test_get_valid_residues_base50() {
        let residues = get_valid_residues(&50);

        // Filter applies to base 50
        assert!(!residues.is_empty());

        // All residues should be < 51 (b+1)
        assert!(residues.iter().all(|&r| r < 51));

        // Should filter roughly half the residues (allowing wider range)
        let expected_count = 51 / 2;
        assert!(
            residues.len() >= expected_count - 10 && residues.len() <= expected_count + 10,
            "Expected approximately {} residues, got {}",
            expected_count,
            residues.len()
        );
    }

    #[test_log::test]
    fn test_passes_filter_base10() {
        // Known nice number 69 should pass
        assert!(passes_filter(69, 10));

        // Check a few numbers
        for n in 0..100 {
            let residues = get_valid_residues(&10);
            let modulus = 11u128;
            let r = n % modulus;
            let should_pass = residues.contains(&(r as u32));
            assert_eq!(
                passes_filter(n, 10),
                should_pass,
                "Mismatch for n={}, residue={}",
                n,
                r
            );
        }
    }

    #[test_log::test]
    fn test_passes_filter_base40() {
        // Filter doesn't apply to base 40, so all numbers should pass
        assert!(passes_filter(0, 40));
        assert!(passes_filter(100, 40));
        assert!(passes_filter(1000, 40));
    }

    #[test_log::test]
    fn test_filtering_rate() {
        // Test that the filter eliminates approximately half the candidates
        let test_bases = vec![10, 50, 70, 90, 110];

        for base in test_bases {
            let residues = get_valid_residues(&base);
            let modulus = base + 1;
            let pass_rate = residues.len() as f64 / modulus as f64;

            // Should filter roughly half (allowing 25-75% range)
            // The actual rate varies by base depending on the structure of r²(1+r) mod (b+1)
            assert!(
                pass_rate > 0.25 && pass_rate < 0.75,
                "Base {}: pass rate {:.2}% outside expected range (25-75%)",
                base,
                pass_rate * 100.0
            );
        }
    }

    #[test_log::test]
    fn test_residues_are_sorted() {
        for base in &[10, 50, 70, 90, 110] {
            let residues = get_valid_residues(base);
            for i in 1..residues.len() {
                assert!(
                    residues[i] > residues[i - 1],
                    "Residues for base {} not sorted at index {}",
                    base,
                    i
                );
            }
        }
    }

    #[test_log::test]
    fn test_crt_compatibility() {
        // Verify gcd(b+1, b-1) = 1 for even bases
        for base in &[10, 40, 50, 60, 70, 80, 90, 100] {
            let b_plus_1 = base + 1;
            let b_minus_1 = base - 1;
            let gcd = gcd(b_plus_1, b_minus_1);
            assert_eq!(
                gcd, 1,
                "Base {}: gcd(b+1, b-1) = gcd({}, {}) = {} != 1",
                base, b_plus_1, b_minus_1, gcd
            );
        }

        // Verify gcd(b+1, b^k) = 1 for even bases
        for base in &[10, 40, 50, 60, 70, 80, 90, 100] {
            let b_plus_1 = base + 1;
            let b_squared = base * base;
            let gcd = gcd(b_plus_1, b_squared);
            assert_eq!(
                gcd, 1,
                "Base {}: gcd(b+1, b^2) = gcd({}, {}) = {} != 1",
                base, b_plus_1, b_squared, gcd
            );
        }
    }

    // Helper function for GCD
    fn gcd(a: u32, b: u32) -> u32 {
        if b == 0 { a } else { gcd(b, a % b) }
    }

    #[test_log::test]
    fn test_known_nice_numbers_pass() {
        // Base 10: 69 is a known nice number
        assert!(passes_filter(69, 10));

        // Test a range around 69
        for n in 60..80 {
            let residues = get_valid_residues(&10);
            let r = (n % 11) as u32;
            if residues.contains(&r) {
                assert!(
                    passes_filter(n, 10),
                    "n={} with residue {} should pass",
                    n,
                    r
                );
            } else {
                assert!(
                    !passes_filter(n, 10),
                    "n={} with residue {} should fail",
                    n,
                    r
                );
            }
        }
    }
}
