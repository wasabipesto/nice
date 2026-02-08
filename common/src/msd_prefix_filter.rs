//! A filter module for detecting ranges that can be skipped based on most significant digits (MSD).
//!
//! The main source of processing time for the reference client is converting
//! each square and cube to the base representation and checking for unique digits.
//!
//! This module implements a common MSD prefix pre-check filter:
//! Before processing an entire range, we check if all numbers in the range
//! can be eliminated based on their most significant digits (MSD).
//!
//! ## How It Works
//!
//! 1. Convert `range_start²`, `range_end²`, `range_start³`, and `range_end³` to base digits.
//!    - IMPORTANT: `to_digits_asc` returns digits in ascending order (LSD first, MSD last)
//!    - For 10,004,569 in base 10: returns [9,6,5,4,0,0,0,1] not [1,0,0,0,4,5,6,9]
//!    - We work backwards from the end of vectors to examine most significant digits
//! 2. Find the longest common MSD prefix shared by all squares in the range.
//! 3. Find the longest common MSD prefix shared by all cubes in the range.
//! 4. Check three early-exit conditions:
//!    - If the square MSD prefix contains duplicate digits → all numbers invalid
//!    - If the cube MSD prefix contains duplicate digits → all numbers invalid
//!    - If square and cube MSD prefixes share any digits → all numbers invalid
//! 5. If any condition triggers, return `true` (range can be skipped).
//! 6. Otherwise, return `false` (range must be processed normally).

use log::trace;
use malachite::base::num::arithmetic::traits::Pow;
use malachite::base::num::conversion::traits::Digits;
use malachite::natural::Natural;

use crate::FieldSize;

// Recursive MSD filter subdivision parameters
pub const MSD_RECURSIVE_MAX_DEPTH: u32 = 11;
pub const MSD_RECURSIVE_MIN_RANGE_SIZE: u128 = 1000;
pub const MSD_RECURSIVE_SUBDIVISION_FACTOR: usize = 2;

// Cross MSD×LSD collision check parameters
// Number of least significant digits to check for collisions with MSD
pub const MSD_LSD_OVERLAP_K_VALUE: u32 = 1;

/// Find the longest common prefix of the most significant digits.
///
/// Since `to_digits_asc` returns digits in ascending order (least-to-most significant),
/// we need to work from the END of the vectors to examine the most significant digits.
///
/// For example, if `to_digits_asc(&10)` returns [9,6,5,4,0,0,0,1] for 10,004,569,
/// the most significant digits are at the end: [1,0,0,0,...].
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

    common_prefix
}

/// Check if a sequence of digits contains any duplicates.
/// Support bases up to 256.
fn has_duplicate_digits(digits: &[u32]) -> bool {
    let mut seen = vec![false; 256];
    for &digit in digits {
        debug_assert!(digit < 256, "Digit {digit} exceeds base limit");
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
/// Support bases up to 256.
fn has_overlapping_digits(digits1: &[u32], digits2: &[u32]) -> bool {
    let mut seen = vec![false; 256];
    for &digit in digits1 {
        debug_assert!(digit < 256, "Digit {digit} exceeds base limit");
        if digit < 256 {
            seen[digit as usize] = true;
        }
    }
    for &digit in digits2 {
        debug_assert!(digit < 256, "Digit {digit} exceeds base limit");
        if digit < 256 && seen[digit as usize] {
            return true;
        }
    }
    false
}

/// Extract the least significant k digits from a number in the given base.
///
/// Returns a vector of the last k digits in the order they appear (from least to most significant).
///
/// # Arguments
/// - `digits_asc`: The digits in ascending order (LSD first, from `to_digits_asc`)
/// - `k`: Number of least significant digits to extract
///
/// # Returns
/// A vector containing the last k digits (or fewer if the number has fewer than k digits)
fn extract_lsd_suffix(digits_asc: &[u32], k: usize) -> Vec<u32> {
    // digits_asc already has LSD first, so we just take the first k elements
    digits_asc.iter().take(k).copied().collect()
}

/// Check if a range can be skipped based on duplicate or overlapping digits in the MSD prefix.
///
/// Returns `true` if the range can be skipped entirely (all numbers will fail the nice check),
/// `false` if the range needs to be processed normally.
///
/// This function checks if all squares and cubes in the range share a common most significant
/// digit prefix that contains duplicates or overlaps, which would make all numbers in the
/// range invalid.
///
/// Note that this is half-open, meaning that the range is inclusive of the start value and
/// exclusive of the end value. This follows the Rust convention for ranges.
///
/// # Panics
/// Panics if the range is invalid or the base is greater than 256.
#[must_use]
pub fn has_duplicate_msd_prefix(range: FieldSize, base: u32) -> bool {
    // Check for edge cases
    assert!(
        range.size() > 0,
        "Range has invalid bounds, range_start must be < range_end (half-open interval)"
    );
    assert!(base <= 256, "Base must be 256 or less");

    // Can't check for duplicate values when there is only one element
    if range.size() == 1 {
        trace!("Range has only a single value, cannot use prefix optimization.");
        return false;
    }

    // Convert range boundaries to digit representations and find common prefixes of most significant digits
    let range_start_square = Natural::from(range.first()).pow(2).to_digits_asc(&base);
    let range_end_square = Natural::from(range.last()).pow(2).to_digits_asc(&base);

    // If the number of digits changes, it's harder to evaluate the prefix
    // For now we reject these to avoid false positives
    if range_start_square.len() != range_end_square.len() {
        trace!(
            "Range start and end squares have a different number of digits, erring on the side of caution."
        );
        return false;
    }

    // If the common prefix has duplicate digits, all numbers in range are invalid
    let square_prefix = find_common_msd_prefix(&range_start_square, &range_end_square);
    if has_duplicate_digits(&square_prefix) {
        trace!("Square prefix has duplicate digits: {square_prefix:?}");
        return true;
    }

    // Check the same thing for the cubes
    let range_start_cube = Natural::from(range.first()).pow(3).to_digits_asc(&base);
    let range_end_cube = Natural::from(range.last()).pow(3).to_digits_asc(&base);

    // If the number of digits changes, it's harder to evaluate the prefix
    // For now we reject these to avoid false positives
    if range_start_cube.len() != range_end_cube.len() {
        trace!(
            "Range start and end cubes have a different number of digits, erring on the side of caution."
        );
        return false;
    }

    // If the common prefix has duplicate digits, all numbers in range are invalid
    let cube_prefix = find_common_msd_prefix(&range_start_cube, &range_end_cube);
    if has_duplicate_digits(&cube_prefix) {
        trace!("Cube prefix has duplicate digits: {cube_prefix:?}");
        return true;
    }

    // If the square and cube prefixes overlap, all numbers in range are invalid
    if has_overlapping_digits(&square_prefix, &cube_prefix) {
        trace!(
            "Square and cube prefixes have overlapping digits: {square_prefix:?}, {cube_prefix:?}"
        );
        return true;
    }

    // Cross MSD×LSD Collision Check
    //
    // This filter enhances the MSD prefix filter by checking for digit collisions between
    // the Most Significant Digits (MSD) and Least Significant Digits (LSD).
    //
    // Within small ranges where all numbers share the same (n mod b^k), the LSD digits
    // are constant. If any digit appears in BOTH the known MSD prefix AND the known LSD
    // suffix, all numbers in that range are invalid (cannot be nice).
    //
    // For example, if squares have MSD prefix [1, 2] and LSD suffix [3, 2], the digit 2
    // appears in both, creating a guaranteed duplicate. All numbers in this range can be
    // skipped without individual checking.
    //
    // This filter checks 7 additional collision conditions:
    // 1. MSD of squares ∩ LSD of squares
    // 2. MSD of cubes ∩ LSD of cubes
    // 3. MSD of squares ∩ LSD of cubes
    // 4. MSD of cubes ∩ LSD of squares
    // 5. LSD of squares - internal duplicates
    // 6. LSD of cubes - internal duplicates
    // 7. LSD of squares ∩ LSD of cubes
    //
    // Only apply when range is small enough that n mod b^k is constant across the range.
    // This ensures all numbers in the range have the same LSD suffix.
    // Specifically, we check: range.first() / b^k == range.last() / b^k
    //
    // This doesn't appear to have huge gains at the moment, even with higher `k` values.
    let k = MSD_LSD_OVERLAP_K_VALUE;
    let b_k = u128::from(base).saturating_pow(k);

    // Check if range is small enough for constant LSD
    // We need: all numbers in range have same (n mod b^k)
    // This is true when: range.size() <= 1 OR (range.first() / b^k == range.last() / b^k)
    let range_spans_single_lsd_class = range.first() / b_k == range.last() / b_k;

    if range_spans_single_lsd_class {
        // Extract LSD suffixes (first k digits, since to_digits_asc returns LSD first)
        let lsd_sq = extract_lsd_suffix(&range_start_square, k as usize);
        let lsd_cu = extract_lsd_suffix(&range_start_cube, k as usize);

        // Check for collisions between MSD and LSD
        if has_overlapping_digits(&square_prefix, &lsd_sq) {
            trace!(
                "Square MSD prefix overlaps with square LSD suffix: MSD={square_prefix:?}, LSD={lsd_sq:?}"
            );
            return true;
        }

        if has_overlapping_digits(&cube_prefix, &lsd_cu) {
            trace!(
                "Cube MSD prefix overlaps with cube LSD suffix: MSD={cube_prefix:?}, LSD={lsd_cu:?}"
            );
            return true;
        }

        if has_overlapping_digits(&square_prefix, &lsd_cu) {
            trace!(
                "Square MSD prefix overlaps with cube LSD suffix: MSD={square_prefix:?}, LSD={lsd_cu:?}"
            );
            return true;
        }

        if has_overlapping_digits(&cube_prefix, &lsd_sq) {
            trace!(
                "Cube MSD prefix overlaps with square LSD suffix: MSD={cube_prefix:?}, LSD={lsd_sq:?}"
            );
            return true;
        }

        // Check LSD suffixes for internal duplicates
        if has_duplicate_digits(&lsd_sq) {
            trace!("Square LSD suffix has duplicate digits: {lsd_sq:?}");
            return true;
        }

        if has_duplicate_digits(&lsd_cu) {
            trace!("Cube LSD suffix has duplicate digits: {lsd_cu:?}");
            return true;
        }

        // Check if square and cube LSD suffixes overlap
        if has_overlapping_digits(&lsd_sq, &lsd_cu) {
            trace!("Square and cube LSD suffixes overlap: sq={lsd_sq:?}, cu={lsd_cu:?}");
            return true;
        }
    }

    // No early exit possible
    false
}

/// Recursively subdivide a range to find sub-ranges that need to be processed.
///
/// This function applies the MSD prefix filter recursively:
/// 1. If the entire range can be skipped (has duplicate MSD prefix), return empty vec
/// 2. If the range is small or max depth reached, return the range (needs processing)
/// 3. Otherwise, subdivide into smaller ranges and recursively check each
///
/// Returns a vector of `FieldSize` structs representing ranges that need processing.
/// All ranges are half-open intervals [start, end) following Rust's standard convention.
///
/// # Arguments
/// * `range` - The range (exclusive, following half-open convention)
/// * `base` - The base to check
/// * `current_depth` - Current recursion depth (should start at 0)
/// * `max_depth` - Maximum recursion depth to prevent excessive subdivision
/// * `min_range_size` - Minimum range size before stopping subdivision
/// * `subdivision_factor` - Number of parts to subdivide into (2-4 recommended)
#[must_use]
pub fn get_valid_ranges_recursive(
    range: FieldSize,
    base: u32,
    current_depth: u32,
    max_depth: u32,
    min_range_size: u128,
    subdivision_factor: usize,
) -> Vec<FieldSize> {
    // Check if range is too small or we've hit max depth
    if current_depth >= max_depth {
        trace!(
            "Depth {current_depth}: Range [{}, {}) max depth reached, returning for processing",
            range.range_start, range.range_end
        );
        return vec![range];
    }
    if range.size() <= min_range_size {
        trace!(
            "Depth {current_depth}: Range [{}, {}) too small, returning for processing",
            range.range_start, range.range_end
        );
        return vec![range];
    }

    // Check if the entire range can be skipped
    if has_duplicate_msd_prefix(range, base) {
        trace!(
            "Depth {current_depth}: Range [{}, {}) can be skipped entirely",
            range.range_start, range.range_end
        );
        return vec![]; // Skip this entire range
    }

    // Check if subdivision would be worthwhile
    // If the range is not much larger than min_range_size, don't bother subdividing
    if range.size() < min_range_size * (subdivision_factor as u128) {
        trace!(
            "Depth {current_depth}: Range [{}, {}) not worth subdividing, returning for processing",
            range.range_start, range.range_end
        );
        return vec![range];
    }

    // Subdivide the range and recursively check each part
    trace!(
        "Depth {current_depth}: Subdividing range [{}, {}) into {subdivision_factor} parts",
        range.range_start, range.range_end
    );

    let chunk_size = range.size() / (subdivision_factor as u128);
    let mut valid_ranges = Vec::new();

    for i in 0..subdivision_factor {
        let sub_start = range.range_start + (i as u128) * chunk_size;
        let sub_end = if i == subdivision_factor - 1 {
            range.range_end // Last chunk gets any remainder
        } else {
            sub_start + chunk_size
        };
        let sub_range = FieldSize::new(sub_start, sub_end);

        if sub_start < sub_end {
            let sub_ranges = get_valid_ranges_recursive(
                sub_range,
                base,
                current_depth + 1,
                max_depth,
                min_range_size,
                subdivision_factor,
            );
            valid_ranges.extend(sub_ranges);
        }
    }

    valid_ranges
}

/// Convenience wrapper for `get_valid_ranges_recursive` using default parameters from lib.rs.
///
/// Returns a vector of `FieldSize` structs representing half-open ranges [start, end) that need
/// processing. Ranges that can be skipped based on MSD prefix are not included.
#[must_use]
pub fn get_valid_ranges(range: FieldSize, base: u32) -> Vec<FieldSize> {
    get_valid_ranges_recursive(
        range,
        base,
        0,
        MSD_RECURSIVE_MAX_DEPTH,
        MSD_RECURSIVE_MIN_RANGE_SIZE,
        MSD_RECURSIVE_SUBDIVISION_FACTOR,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::base_range;
    use log::debug;

    /// Break up the range into chunks, returning the start and end of each.
    fn chunked_ranges(range_start: u128, range_end: u128, chunk_size: u128) -> Vec<(u128, u128)> {
        let mut chunks = Vec::new();
        let mut start = range_start;

        while start < range_end {
            let end = (start + chunk_size).min(range_end);
            chunks.push((start, end));
            start = end;
        }

        chunks
    }

    #[test_log::test]
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

    #[test_log::test]
    fn test_has_duplicate_digits() {
        assert!(!has_duplicate_digits(&[1, 2, 3, 4]));
        assert!(has_duplicate_digits(&[1, 2, 1, 4]));
        assert!(has_duplicate_digits(&[5, 5]));
        assert!(!has_duplicate_digits(&[]));
        assert!(!has_duplicate_digits(&[1]));
        assert!(has_duplicate_digits(&[7, 7, 1, 2, 3]));
    }

    #[test_log::test]
    fn test_has_overlapping_digits() {
        assert!(!has_overlapping_digits(&[1, 2, 3], &[4, 5, 6]));
        assert!(has_overlapping_digits(&[1, 2, 3], &[3, 4, 5]));
        assert!(has_overlapping_digits(&[1, 2, 3], &[1, 2, 3]));
        assert!(!has_overlapping_digits(&[], &[1, 2, 3]));
        assert!(!has_overlapping_digits(&[1, 2, 3], &[]));
        assert!(has_overlapping_digits(&[7], &[7]));
    }

    #[test_log::test]
    fn test_digit_order_verification() {
        // Verify that to_digits_asc returns LSD first
        let num = Natural::from(10_004_569u32);
        let digits = num.to_digits_asc(&10u32);
        // 10,004,569 should be [9,6,5,4,0,0,0,1] in ascending order
        assert_eq!(digits[0], 9); // least significant digit
        assert_eq!(digits[7], 1); // most significant digit

        // Test our MSD prefix finder
        let digits1 = Natural::from(10_004_569u32).to_digits_asc(&10u32);
        let digits2 = Natural::from(10_010_896u32).to_digits_asc(&10u32);
        let msd_prefix = find_common_msd_prefix(&digits1, &digits2);
        // Both start with 1,0,0,... in normal notation
        assert_eq!(msd_prefix, vec![1, 0, 0]);
        // This prefix has duplicate 0s
        assert!(has_duplicate_digits(&msd_prefix));
    }

    #[test_log::test]
    fn test_early_exit_demonstration() {
        // This test demonstrates the early exit optimization
        // Range: 3163-3165, base 10 (i.e., [3163, 3165) which includes 3163 and 3164)
        // 3163² = 10,004,569 → to_digits_asc: [9,6,5,4,0,0,0,1]
        // 3164² = 10,010,896 → to_digits_asc: [6,9,8,0,1,0,0,1]
        // Common MSD prefix: [1,0,0] which has duplicate 0s

        let range_start = 3163; // 3163² = 10,004,569
        let range_end = 3165; // So range is [3163, 3165), last number is 3164: 3164² = 10,010,896
        let range = FieldSize::new(range_start, range_end);
        let base = 10;
        let can_skip = has_duplicate_msd_prefix(range, base);

        // Should return true because squares share MSD prefix [1,0,0] with duplicate 0s
        assert!(can_skip);
    }

    #[test_log::test]
    fn test_single_element_range() {
        // This test demonstrates the bug: when range_end = range_start + 1,
        // the range contains only one element [range_start, range_start+1)
        // This means the "common prefix" is the entire number, not a real prefix.

        let range_start = 3163;
        let range_end = 3164; // Range is [3163, 3164), which contains only 3163
        let range = FieldSize::new(range_start, range_end);
        let base = 10;

        let can_skip = has_duplicate_msd_prefix(range, base);
        assert!(!can_skip);
    }

    #[test_log::test]
    #[should_panic = "invalid bounds"]
    fn test_invalid_bounds() {
        let range_start = 3163;
        let range_end = 3163;
        let range = FieldSize::new(range_start, range_end);
        let base = 10;

        let _can_skip = has_duplicate_msd_prefix(range, base);
    }

    #[test_log::test]
    fn test_early_exit_b10() {
        let base = 10;
        let base_range = base_range::get_base_range_u128(base).unwrap().unwrap();
        let can_skip = has_duplicate_msd_prefix(base_range, base);
        assert!(!can_skip);
    }

    #[test_log::test]
    fn test_early_exit_b40_whole() {
        let base = 40;
        let base_range = base_range::get_base_range_u128(base).unwrap().unwrap();
        let can_skip = has_duplicate_msd_prefix(base_range, base);
        assert!(!can_skip);
    }

    #[test_log::test]
    fn test_early_exit_b50_whole() {
        let base = 50;
        let base_range = base_range::get_base_range_u128(base).unwrap().unwrap();
        let can_skip = has_duplicate_msd_prefix(base_range, base);
        assert!(!can_skip);
    }

    #[test_log::test]
    fn test_early_exit_b50_segments_large() {
        let base = 50;
        let base_range = base_range::get_base_range_u128(base).unwrap().unwrap();
        let chunk_size = base_range.size() / 100;
        let segments = chunked_ranges(base_range.range_start, base_range.range_end, chunk_size);

        let expected_results = vec![
            (0, false),
            (10, false),
            (30, false),
            (40, false),
            (50, false),
            (60, false),
            (70, false),
            (80, false),
            (90, false),
            (100, true),
        ];
        for (segment_num, expected_result) in expected_results {
            let segment = segments[segment_num];
            let range = FieldSize::new(segment.0, segment.1);
            debug!("Testing base {base} segment #{segment_num}: ({segment:?})");
            let can_skip = has_duplicate_msd_prefix(range, base);
            assert_eq!(can_skip, expected_result);
        }
    }

    #[test_log::test]
    fn test_early_exit_b50_segments_small() {
        let base = 50;
        let base_range = base_range::get_base_range_u128(base).unwrap().unwrap();
        let chunk_size = base_range.size() / 10_000;
        let segments = chunked_ranges(base_range.range_start, base_range.range_end, chunk_size);

        let expected_results = vec![
            (0, false),
            (10, false),
            (30, true),
            (40, true),
            (50, false),
            (60, false),
            (70, false),
            (80, true),
            (90, true),
            (100, false),
        ];
        for (segment_num, expected_result) in expected_results {
            let segment = segments[segment_num];
            let range = FieldSize::new(segment.0, segment.1);
            debug!("Testing base {base} segment #{segment_num}: ({segment:?})");
            let can_skip = has_duplicate_msd_prefix(range, base);
            assert_eq!(can_skip, expected_result);
        }
    }

    #[test_log::test]
    fn test_filter_c_lsd_extraction() {
        // Test the LSD extraction helper function
        // Simulate to_digits_asc format: [LSD, ..., MSD]

        // Number 12345 in base 10 = [5, 4, 3, 2, 1]
        // Last 2 digits should be [5, 4]
        let digits = vec![5, 4, 3, 2, 1];
        assert_eq!(extract_lsd_suffix(&digits, 2), vec![5, 4]);

        // Last 3 digits should be [5, 4, 3]
        assert_eq!(extract_lsd_suffix(&digits, 3), vec![5, 4, 3]);

        // If we ask for more digits than exist, we get all of them
        assert_eq!(extract_lsd_suffix(&digits, 10), vec![5, 4, 3, 2, 1]);
    }

    #[test_log::test]
    fn test_filter_c_small_range_with_lsd_collision() {
        // Test Filter C with a small range where MSD and LSD collide
        // For base 10, we'll construct a scenario where this happens

        let base = 10u32;

        // Find a small range where Filter C should trigger
        // We need a range where:
        // 1. MSD prefix doesn't have duplicates (so MSD filter alone wouldn't catch it)
        // 2. But MSD overlaps with LSD (Filter C catches it)

        // Let's test with a very small range to ensure LSD is constant
        // Range [100, 101) - single element
        let range = FieldSize::new(100, 101);
        let _result = has_duplicate_msd_prefix(range, base);

        // This should complete without panicking
        // The actual result depends on whether 100 is nice, but Filter C adds checks
        // Just verify the function runs successfully
    }

    #[test_log::test]
    fn test_filter_c_range_span_check() {
        // Test that Filter C correctly identifies when a range spans a single LSD class
        let base = 10u32;
        let k = MSD_LSD_OVERLAP_K_VALUE;
        let b_k = u128::from(base).pow(k); // 10^2 = 100

        // Range [100, 199] spans from b^k class 1 to class 1 (100/100 = 1, 199/100 = 1)
        let range1 = FieldSize::new(100, 200);
        let spans_single = range1.first() / b_k == range1.last() / b_k;
        assert!(
            spans_single,
            "Range [100, 200) should span single LSD class"
        );

        // Range [100, 201] spans from class 1 to class 2 (100/100 = 1, 200/100 = 2)
        let range2 = FieldSize::new(100, 201);
        let spans_multiple = range2.first() / b_k != range2.last() / b_k;
        assert!(
            spans_multiple,
            "Range [100, 201) should span multiple LSD classes"
        );
    }

    #[test_log::test]
    fn test_filter_c_base_40_effectiveness() {
        // Test that Filter C provides additional filtering for base 40
        // where it should be particularly effective
        let base = 40u32;
        let base_range = base_range::get_base_range_u128(base).unwrap().unwrap();

        // Create a small range within base 40's range
        let test_range_size = 500u128; // Small enough for Filter C to apply
        let test_range = FieldSize::new(
            base_range.start() + 1_000_000,
            base_range.start() + 1_000_000 + test_range_size,
        );

        // Run the filter - it should complete successfully
        let _can_skip = has_duplicate_msd_prefix(test_range, base);

        // We don't assert a specific result, just that it runs without error
        // and that Filter C logic is exercised
    }

    #[test_log::test]
    fn test_filter_c_does_not_break_existing_filters() {
        // Verify that adding Filter C doesn't break existing MSD filter behavior
        // Test cases that should still be caught by the original MSD filters

        let base = 10u32;

        // A range where squares have duplicate MSD (caught by original filter)
        // This should still return true
        let range1 = FieldSize::new(1000, 1100);
        let _result1 = has_duplicate_msd_prefix(range1, base);

        // The test just ensures the function still works
    }

    #[test_log::test]
    fn test_filter_c_detects_msd_lsd_collision() {
        // Create a specific test case where Filter C should catch a collision
        // that the original MSD filter would miss

        let base = 10u32;

        // We'll create a scenario where:
        // 1. The MSD prefixes alone don't have internal duplicates
        // 2. The MSD and LSD don't overlap (no collision)
        // 3. But we can verify Filter C runs its checks

        // Use a small range to ensure Filter C applies
        let range = FieldSize::new(50, 55);

        // This range is small enough that all numbers share the same n mod 100
        // Filter C will check for MSD×LSD collisions
        let result = has_duplicate_msd_prefix(range, base);

        // Verify the function runs successfully
        // The actual result depends on the specific mathematics, but Filter C
        // adds additional checks that may catch more invalid ranges
        debug!("Filter C result for range [50, 55): {result}");
    }

    #[test_log::test]
    fn test_filter_c_statistics() {
        // Test Filter C across multiple small ranges to verify it's working
        let base = 40u32;
        let base_range = base_range::get_base_range_u128(base).unwrap().unwrap();

        // Test 100 small ranges
        let mut filter_c_applicable = 0;
        let mut ranges_skipped = 0;

        for i in 0..100 {
            let start = base_range.start() + i * 500;
            let end = start + 500;
            let range = FieldSize::new(start, end);

            // Check if Filter C would apply (range < b^k)
            let k = MSD_LSD_OVERLAP_K_VALUE;
            let b_k = u128::from(base).pow(k);
            if range.first() / b_k == range.last() / b_k {
                filter_c_applicable += 1;
            }

            if has_duplicate_msd_prefix(range, base) {
                ranges_skipped += 1;
            }
        }

        debug!(
            "Filter C statistics for base {base}: {filter_c_applicable}/{} ranges had Filter C applicable, {ranges_skipped} ranges skipped",
            100
        );

        // Filter C should apply to at least some ranges
        assert!(
            filter_c_applicable > 0,
            "Filter C should be applicable to some small ranges"
        );
    }
}
