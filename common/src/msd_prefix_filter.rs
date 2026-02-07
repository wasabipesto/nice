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

use super::*;
use log::trace;

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
        debug_assert!(digit < 256, "Digit {} exceeds base limit", digit);
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
        debug_assert!(digit < 256, "Digit {} exceeds base limit", digit);
        if digit < 256 {
            seen[digit as usize] = true;
        }
    }
    for &digit in digits2 {
        debug_assert!(digit < 256, "Digit {} exceeds base limit", digit);
        if digit < 256 && seen[digit as usize] {
            return true;
        }
    }
    false
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
pub fn has_duplicate_msd_prefix(range: FieldSize, base: u32) -> bool {
    // Check for edge cases
    assert!(
        range.range_start < range.range_end,
        "Range has invalid bounds, range_start must be < range_end (half-open interval)"
    );
    assert!(base <= 256, "Base must be 256 or less");

    // Can't check for duplicate values when there is only one element
    if range.range_size == 1 {
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

    // No early exit possible
    trace!("No early exit possible. Prefixes: {square_prefix:?}, {cube_prefix:?}");
    false
}

/// Recursively subdivide a range to find sub-ranges that need to be processed.
///
/// This function applies the MSD prefix filter recursively:
/// 1. If the entire range can be skipped (has duplicate MSD prefix), return empty vec
/// 2. If the range is small or max depth reached, return the range (needs processing)
/// 3. Otherwise, subdivide into smaller ranges and recursively check each
///
/// Returns a vector of FieldSize structs representing ranges that need processing.
/// All ranges are half-open intervals [start, end) following Rust's standard convention.
///
/// # Arguments
/// * `range` - The range (exclusive, following half-open convention)
/// * `base` - The base to check
/// * `current_depth` - Current recursion depth (should start at 0)
/// * `max_depth` - Maximum recursion depth to prevent excessive subdivision
/// * `min_range_size` - Minimum range size before stopping subdivision
/// * `subdivision_factor` - Number of parts to subdivide into (2-4 recommended)
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
    if range.range_size <= min_range_size {
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
    if range.range_size < min_range_size * (subdivision_factor as u128) {
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

    let chunk_size = range.range_size / (subdivision_factor as u128);
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

/// Convenience wrapper for get_valid_ranges_recursive using default parameters from lib.rs.
///
/// Returns a vector of FieldSize structs representing half-open ranges [start, end) that need
/// processing. Ranges that can be skipped based on MSD prefix are not included.
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
    #[should_panic]
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
        let chunk_size = base_range.range_size / 100;
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
        let chunk_size = base_range.range_size / 10_000;
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
}
