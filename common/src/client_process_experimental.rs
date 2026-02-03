//! An experimental version of the client nice-only process.
//! Compare to `crate::client_process::process_niceonly` as the reference
//! implementation.
//!
//! # Optimization Strategy
//!
//! The main source of processing time for the reference client is converting
//! each square and cube to the base representation and checking for unique digits.
//!
//! This experiment implements a **common MSD prefix pre-check optimization**:
//! Before processing the entire range, we check if all numbers in the range
//! can be eliminated based on their most significant digits (MSD).
//!
//! ## How It Works
//!
//! 1. Convert `range_start²`, `range_end²`, `range_start³`, and `range_end³` to base digits
//!    - **IMPORTANT**: `to_digits_asc` returns digits in ascending order (LSD first, MSD last)
//!    - For 10,004,569 in base 10: returns [9,6,5,4,0,0,0,1] not [1,0,0,0,4,5,6,9]
//!    - We work backwards from the end of vectors to examine most significant digits
//! 2. Find the longest common MSD prefix shared by all squares in the range
//! 3. Find the longest common MSD prefix shared by all cubes in the range
//! 4. Check three early-exit conditions:
//!    - If the square MSD prefix contains duplicate digits → all numbers invalid
//!    - If the cube MSD prefix contains duplicate digits → all numbers invalid
//!    - If square and cube MSD prefixes share any digits → all numbers invalid
//! 5. If any condition triggers, return empty results immediately
//! 6. Otherwise, fall back to the reference implementation
//!
//! ## Example
//!
//! For range 3163-3164 in base 10:
//! - 3163² = 10,004,569
//!   - `to_digits_asc(&10)` returns [9,6,5,4,0,0,0,1] (LSD at index 0, MSD at index 7)
//!   - Most significant digits (reading from end): 1, 0, 0, ...
//! - 3164² = 10,010,896
//!   - `to_digits_asc(&10)` returns [6,9,8,0,1,0,0,1] (LSD at index 0, MSD at index 7)
//!   - Most significant digits (reading from end): 1, 0, 0, ...
//! - Common MSD prefix: [1,0,0] which contains duplicate 0s
//! - **Early exit**: All squares in this range have duplicate 0s in their MSD
//!
//! ## Performance Characteristics
//!
//! - **Best case**: O(log n) when early exit triggers (just 4 conversions + prefix check)
//! - **Worst case**: O(n) when no early exit (falls back to reference implementation)
//! - **Overhead**: Minimal - just 4 extra `Natural::pow()` and digit conversions
//! - **Win rate**: Depends on the distribution of ranges; higher bases and larger
//!   ranges are more likely to have common MSD prefix patterns
//!
//! ## When This Helps
//!
//! This optimization is most effective when:
//! - The range is relatively small compared to the magnitude of numbers
//! - Numbers in the range share many most significant digits
//! - The shared digits contain duplicates or overlaps between squares and cubes
//!
//! ## Limitations
//!
//! - Only checks the common MSD prefix; can't detect patterns in middle/trailing digits
//! - Requires computing 4 extra exponentiations (though these are cheap for endpoints)
//! - Empty prefixes (no common MSD) provide no benefit
//!
//! ## Key Implementation Detail
//!
//! Because `to_digits_asc` returns [LSD, ..., MSD], we must work from the END of the
//! digit vectors backwards to find common most significant digits. The helper function
//! `find_common_msd_prefix` handles this by comparing digits from the end of both vectors.

use super::*;

/// Find the longest common prefix of the most significant digits.
///
/// Since `to_digits_asc` returns digits in ascending order (least-to-most significant),
/// we need to work from the END of the vectors to examine the most significant digits.
///
/// For example, if `to_digits_asc(&10)` returns [9,6,5,4,0,0,0,1] for 10,004,569,
/// the most significant digits are at the end: [1,0,0,0,...].
///
/// Returns the common prefix in MSD-first order [MSD, ..., LSD].
fn find_common_msd_prefix(digits1: &[u32], digits2: &[u32]) -> Vec<u32> {
    let len1 = digits1.len();
    let len2 = digits2.len();
    let mut common_prefix = Vec::new();

    // Work backwards from the end (most significant digits)
    let min_len = len1.min(len2);
    for i in 0..min_len {
        let idx1 = len1 - 1 - i;
        let idx2 = len2 - 1 - i;
        if digits1[idx1] == digits2[idx2] {
            common_prefix.push(digits1[idx1]);
        } else {
            break;
        }
    }

    // Already in MSD-first order (we pushed from end backwards)
    common_prefix
}

/// Check if a sequence of digits contains any duplicates.
fn has_duplicate_digits(digits: &[u32]) -> bool {
    let mut seen = vec![false; 256]; // Support bases up to 256
    for &digit in digits {
        if digit < 256 {
            if seen[digit as usize] {
                return true;
            }
            seen[digit as usize] = true;
        }
    }
    false
}

/// Check if two digit sequences share any common digits.
fn has_overlapping_digits(digits1: &[u32], digits2: &[u32]) -> bool {
    let mut seen = vec![false; 256];
    for &digit in digits1 {
        if digit < 256 {
            seen[digit as usize] = true;
        }
    }
    for &digit in digits2 {
        if digit < 256 && seen[digit as usize] {
            return true;
        }
    }
    false
}

/// Process a field by looking for completely nice numbers.
/// Implements a quick pre-check optimization before falling back to the reference implementation.
pub fn process_range_niceonly(range_start: u128, range_end: u128, base: u32) -> FieldResults {
    // Convert range boundaries to digit representations
    let range_start_square = Natural::from(range_start).pow(2).to_digits_asc(&base);
    let range_start_cube = Natural::from(range_start).pow(3).to_digits_asc(&base);
    let range_end_square = Natural::from(range_end).pow(2).to_digits_asc(&base);
    let range_end_cube = Natural::from(range_end).pow(3).to_digits_asc(&base);

    // Quick pre-check: Find common prefixes of most significant digits
    let square_prefix = find_common_msd_prefix(&range_start_square, &range_end_square);
    let cube_prefix = find_common_msd_prefix(&range_start_cube, &range_end_cube);

    // If the common prefix has duplicate digits, all numbers in range are invalid
    if has_duplicate_digits(&square_prefix) {
        /*
        println!(
            "Early exit: All squares share prefix {:?} with duplicates",
            square_prefix
        );
        */
        return FieldResults {
            distribution: Vec::new(),
            nice_numbers: Vec::new(),
        };
    }

    if has_duplicate_digits(&cube_prefix) {
        /*
        println!(
            "Early exit: All cubes share prefix {:?} with duplicates",
            cube_prefix
        );
        */
        return FieldResults {
            distribution: Vec::new(),
            nice_numbers: Vec::new(),
        };
    }

    // If the square and cube prefixes overlap, all numbers in range are invalid
    if has_overlapping_digits(&square_prefix, &cube_prefix) {
        /*
        println!(
            "Early exit: Square prefix {:?} and cube prefix {:?} overlap",
            square_prefix, cube_prefix
        );
        */
        return FieldResults {
            distribution: Vec::new(),
            nice_numbers: Vec::new(),
        };
    }

    // No early exit possible, fall back to the reference implementation
    println!(
        "No early exit: square prefix {:?}, cube prefix {:?}",
        square_prefix, cube_prefix
    );
    crate::client_process::process_range_niceonly(range_start, range_end, base)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_find_common_msd_prefix() {
        // Simulate to_digits_asc format: [LSD, ..., MSD]
        // 12345 in base 10 = [5, 4, 3, 2, 1]
        // 12367 in base 10 = [7, 6, 3, 2, 1]
        // Common MSD prefix: [1, 2, 3]
        let digits1 = vec![5, 4, 3, 2, 1];
        let digits2 = vec![7, 6, 3, 2, 1];
        assert_eq!(find_common_msd_prefix(&digits1, &digits2), vec![1, 2, 3]);

        // 5512 = [2, 1, 5, 5]
        // 5598 = [8, 9, 5, 5]
        // Common MSD prefix: [5, 5]
        let digits1 = vec![2, 1, 5, 5];
        let digits2 = vec![8, 9, 5, 5];
        assert_eq!(find_common_msd_prefix(&digits1, &digits2), vec![5, 5]);

        // 123 = [3, 2, 1]
        // 456 = [6, 5, 4]
        // No common MSD prefix
        let digits1 = vec![3, 2, 1];
        let digits2 = vec![6, 5, 4];
        assert_eq!(
            find_common_msd_prefix(&digits1, &digits2),
            Vec::<u32>::new()
        );

        // Identical numbers
        let digits1 = vec![9, 8, 7];
        let digits2 = vec![9, 8, 7];
        assert_eq!(find_common_msd_prefix(&digits1, &digits2), vec![7, 8, 9]);

        // Different lengths
        // 10000 = [0, 0, 0, 0, 1]
        // 10100 = [0, 0, 1, 0, 1]
        // Common MSD prefix: [1, 0]
        let digits1 = vec![0, 0, 0, 0, 1];
        let digits2 = vec![0, 0, 1, 0, 1];
        assert_eq!(find_common_msd_prefix(&digits1, &digits2), vec![1, 0]);
    }

    #[test]
    fn test_has_duplicate_digits() {
        assert!(!has_duplicate_digits(&[1, 2, 3, 4]));
        assert!(has_duplicate_digits(&[1, 2, 1, 4]));
        assert!(has_duplicate_digits(&[5, 5]));
        assert!(!has_duplicate_digits(&[]));
        assert!(!has_duplicate_digits(&[1]));
        assert!(has_duplicate_digits(&[7, 7, 1, 2, 3]));
    }

    #[test]
    fn test_has_overlapping_digits() {
        assert!(!has_overlapping_digits(&[1, 2, 3], &[4, 5, 6]));
        assert!(has_overlapping_digits(&[1, 2, 3], &[3, 4, 5]));
        assert!(has_overlapping_digits(&[1, 2, 3], &[1, 2, 3]));
        assert!(!has_overlapping_digits(&[], &[1, 2, 3]));
        assert!(!has_overlapping_digits(&[1, 2, 3], &[]));
        assert!(has_overlapping_digits(&[7], &[7]));
    }

    #[test]
    fn test_digit_order_verification() {
        // Verify that to_digits_asc returns LSD first
        let num = Natural::from(10004569u32);
        let digits = num.to_digits_asc(&10u32);
        // 10,004,569 should be [9,6,5,4,0,0,0,1] in ascending order
        assert_eq!(digits[0], 9); // least significant digit
        assert_eq!(digits[7], 1); // most significant digit

        // Test our MSD prefix finder
        let digits1 = Natural::from(10004569u32).to_digits_asc(&10u32);
        let digits2 = Natural::from(10010896u32).to_digits_asc(&10u32);
        let msd_prefix = find_common_msd_prefix(&digits1, &digits2);
        // Both start with 1,0,0,... in normal notation
        assert_eq!(msd_prefix, vec![1, 0, 0]);
        // This prefix has duplicate 0s
        assert!(has_duplicate_digits(&msd_prefix));
    }

    #[test]
    fn test_early_exit_demonstration() {
        // This test demonstrates the early exit optimization
        // Range: 3163-3164, base 10
        // 3163² = 10,004,569 → to_digits_asc: [9,6,5,4,0,0,0,1]
        // 3164² = 10,010,896 → to_digits_asc: [6,9,8,0,1,0,0,1]
        // Common MSD prefix: [1,0,0] which has duplicate 0s

        let range_start = 3163; // 3163² = 10,004,569
        let range_end = 3164; // 3164² = 10,010,896
        let base = 10;
        let result = process_range_niceonly(range_start, range_end, base);

        // Should return empty because squares share MSD prefix [1,0,0] with duplicate 0s
        assert_eq!(result.nice_numbers, Vec::new());
    }

    #[test]
    fn process_niceonly_b10() {
        let input = DataToClient {
            claim_id: 0,
            base: 10,
            range_start: 47,
            range_end: 100,
            range_size: 53,
        };
        let result = FieldResults {
            distribution: Vec::new(),
            nice_numbers: Vec::from([NiceNumberSimple {
                number: 69,
                num_uniques: 10,
            }]),
        };
        assert_eq!(
            process_range_niceonly(input.range_start, input.range_end, input.base),
            result
        );
    }

    #[test]
    fn process_niceonly_b40() {
        let input = DataToClient {
            claim_id: 0,
            base: 40,
            range_start: 916284264916,
            range_end: 916284264916 + 10000,
            range_size: 10000,
        };
        let result = FieldResults {
            distribution: Vec::new(),
            nice_numbers: Vec::new(),
        };
        assert_eq!(
            process_range_niceonly(input.range_start, input.range_end, input.base),
            result
        );
    }

    #[test]
    fn process_niceonly_b80() {
        let input = DataToClient {
            claim_id: 0,
            base: 80,
            range_start: 653245554420798943087177909799,
            range_end: 653245554420798943087177909799 + 10000,
            range_size: 10000,
        };
        let result = FieldResults {
            distribution: Vec::new(),
            nice_numbers: Vec::new(),
        };
        assert_eq!(
            process_range_niceonly(input.range_start, input.range_end, input.base),
            result
        );
    }
}
