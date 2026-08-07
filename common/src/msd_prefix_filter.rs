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

#![allow(clippy::inline_always)]

use log::trace;
use malachite::base::num::arithmetic::traits::Pow;
use malachite::base::num::conversion::traits::Digits;
use malachite::natural::Natural;

use crate::FieldSize;
use crate::fixed_width::U256;

// Maximum digit count for n³ in any specialized base.
// For b40: log_40(40^24) = 24 digits.
// For b50: log_50(50^30) = 30 digits.
// For b60: log_60(60^36) = 36 digits.
// For b62: k=12, b%5=2 → n³ has 3k+1 = 37 digits.
// For b64: k=12, b%5=4 → n³ has 3k+2 = 38 digits.
// 38 covers all specialized bases ≤ 64.
const MAX_FW_DIGITS: usize = 38;

/// Stack-resident digit sequence used by the fixed-width MSD path. Stores
/// digits in LSD-first order (matching malachite's `to_digits_asc`) so the
/// existing `find_common_msd_prefix` / `has_duplicate_digits` helpers work
/// unchanged on the slice `&buf[..len]`.
#[derive(Copy, Clone)]
struct FwDigits {
    buf: [u32; MAX_FW_DIGITS],
    len: usize,
}

impl FwDigits {
    #[inline(always)]
    fn as_slice(&self) -> &[u32] {
        &self.buf[..self.len]
    }
}

/// Extract base-`BASE` digits of `n` (LSD first) into a stack array.
/// Used by the const-generic fixed-width MSD path. With `BASE` known at
/// compile time, `% BASE` and `/ BASE` lower to multiply-by-magic.
// `(n % base_u128) as u32` is bounded by `BASE - 1 < 2^32`. Hot path:
// `inline(always)` matches the convention in `fixed_width.rs`.
#[allow(clippy::cast_possible_truncation)]
#[inline(always)]
fn extract_digits_u128_const<const BASE: u32>(mut n: u128) -> FwDigits {
    let base_u128 = u128::from(BASE);
    let mut buf = [0u32; MAX_FW_DIGITS];
    let mut len = 0;
    if n == 0 {
        return FwDigits { buf, len: 1 };
    }
    while n != 0 {
        debug_assert!(len < MAX_FW_DIGITS);
        buf[len] = (n % base_u128) as u32;
        n /= base_u128;
        len += 1;
    }
    FwDigits { buf, len }
}

/// Extract base-`BASE` digits of a U256 (LSD first) into a stack array.
/// `div_assign_rem_u32_const` mutates the input limbs; we work on a copy.
#[inline(always)]
fn extract_digits_u256_const<const BASE: u32>(mut n: U256) -> FwDigits {
    let mut buf = [0u32; MAX_FW_DIGITS];
    let mut len = 0;
    if n.is_zero() {
        return FwDigits { buf, len: 1 };
    }
    while !n.is_zero() {
        debug_assert!(len < MAX_FW_DIGITS);
        buf[len] = n.div_assign_rem_u32_const::<BASE>();
        len += 1;
    }
    FwDigits { buf, len }
}

/// Maximum number of constrained output positions the Hall check tracks:
/// every digit position of both powers, plus slack.
const HALL_MAX_POSITIONS: usize = 2 * MAX_FW_DIGITS + 2;

/// Append the digit domains of one power's constrained output positions.
///
/// `xd`/`yd` are the LSD-first digit arrays of `low^p` and `high^p` for a
/// contiguous range `[low, high]`; they must have equal length. For output
/// position `j`, every value of `n^p` in between has its digit-`j` quotient
/// `q = floor(n^p / b^j)` inside `[u_j, v_j]` where `u_j = floor(low^p / b^j)`
/// and `v_j = floor(high^p / b^j)` (the power is monotone). The digit at
/// position `j` is `q mod b`, so it lies in the cyclic residue interval
/// starting at `u_j mod b = xd[j]` of size `v_j - u_j + 1` — a conservative
/// superset of the digits that actually occur.
///
/// Walking positions from most significant down, the interval width follows
/// `diff_{j-1} = diff_j * b + (yd[j-1] - xd[j-1])` with `diff_top` seeded by
/// the leading digits, so no wide arithmetic is needed. Once
/// `diff >= base - 1` the domain covers all digits and every lower position
/// is unconstrained (the width only grows as `j` decreases).
///
/// A `diff == 0` position is a singleton — exactly a digit of the classic
/// common MSD prefix — so this generalizes the previous prefix extraction.
#[inline]
fn collect_power_domains(base: u32, xd: &[u32], yd: &[u32], doms: &mut [u64], count: &mut usize) {
    debug_assert_eq!(xd.len(), yd.len());
    debug_assert!(base <= 64);
    let mut diff: i64 = 0;
    for j in (0..xd.len()).rev() {
        if *count == doms.len() {
            // Out of slots (only possible for very long digit arrays outside
            // the fixed-width path). Dropping the remaining positions just
            // loses constraints, which is sound.
            return;
        }
        diff = diff * i64::from(base) + (i64::from(yd[j]) - i64::from(xd[j]));
        debug_assert!(diff >= 0, "endpoint digits imply high < low");
        if diff >= i64::from(base) - 1 {
            return;
        }
        // Cyclic interval of `diff + 1` residues starting at xd[j].
        // diff is bounded by base² here, so the cast is lossless.
        #[allow(clippy::cast_sign_loss, clippy::cast_possible_truncation)]
        let size = (diff as u32) + 1;
        let lo = xd[j];
        #[allow(clippy::cast_possible_truncation)]
        let mask: u64 = if lo + size <= base {
            (((1u128 << size) - 1) << lo) as u64
        } else {
            // Interval wraps past digit base-1 back to 0.
            let wrapped = lo + size - base;
            ((((1u128 << (base - lo)) - 1) << lo) | ((1u128 << wrapped) - 1)) as u64
        };
        doms[*count] = mask;
        *count += 1;
    }
}

/// Try to give position `i` a digit from its domain, recursively evicting
/// current owners along an augmenting path (Kuhn's matching algorithm).
fn hall_augment(i: usize, doms: &[u64], visited: &mut u64, owner: &mut [usize; 64]) -> bool {
    let mut cand = doms[i] & !*visited;
    while cand != 0 {
        let d = cand.trailing_zeros() as usize;
        *visited |= 1u64 << d;
        if owner[d] == usize::MAX || hall_augment(owner[d], doms, visited, owner) {
            owner[d] = i;
            return true;
        }
        cand = doms[i] & !*visited;
    }
    false
}

/// Can every constrained position be assigned a distinct digit from its
/// domain? By Hall's theorem this fails exactly when some set of positions
/// collectively offers fewer digits than positions — which makes a nice
/// number in the range impossible, since a nice number's actual digits are
/// one such distinct assignment.
fn has_distinct_assignment(doms: &[u64]) -> bool {
    let m = doms.len();
    if m <= 1 {
        return true;
    }
    // Fast global check: m positions need at least m distinct digits.
    let mut union: u64 = 0;
    for &d in doms {
        union |= d;
    }
    if (union.count_ones() as usize) < m {
        return false;
    }
    let mut owner = [usize::MAX; 64];
    for i in 0..m {
        let mut visited: u64 = 0;
        if !hall_augment(i, doms, &mut visited, &mut owner) {
            return false;
        }
    }
    true
}

/// Interval digit-domain analysis (Hall check) given pre-extracted endpoint
/// digit arrays. Factored out so both u128 and U256 paths share identical
/// post-extraction logic.
///
/// This subsumes the previous common-MSD-prefix duplicate/overlap checks:
/// prefix digits are exactly the singleton domains, and pairwise
/// duplicate/overlap failures are two-position Hall violations. The interval
/// domains additionally catch collective violations among near-fixed
/// positions (e.g. three positions that only have two digits between them)
/// that no pairwise test can see.
///
/// A power whose endpoint digit counts differ contributes no domains (its
/// positions are unconstrained), matching the previous bail-out behavior.
///
/// NOTE (2026-08 theory review): an unsound cross MSD×LSD collision check
/// was removed from this spot; low-digit filtering is handled soundly by
/// the stride table (`lsd_filter`), whose fixed low digits are per-candidate
/// facts rather than per-range ones.
#[inline(always)]
fn analyze_msd_prefix<const BASE: u32>(
    start_sq_d: &FwDigits,
    end_sq_d: &FwDigits,
    start_cu_d: &FwDigits,
    end_cu_d: &FwDigits,
) -> bool {
    const { assert!(BASE <= 64, "u64 digit-domain masks can't index past bit 63") };
    let mut doms = [0u64; HALL_MAX_POSITIONS];
    let mut m = 0usize;
    if start_sq_d.len == end_sq_d.len {
        collect_power_domains(
            BASE,
            start_sq_d.as_slice(),
            end_sq_d.as_slice(),
            &mut doms,
            &mut m,
        );
    }
    if start_cu_d.len == end_cu_d.len {
        collect_power_domains(
            BASE,
            start_cu_d.as_slice(),
            end_cu_d.as_slice(),
            &mut doms,
            &mut m,
        );
    }
    if m == 0 {
        return false;
    }
    !has_distinct_assignment(&doms[..m])
}

/// Specialized const-generic MSD prefix check for bases where `n³` fits
/// in `u128` (b40 only — see #16). Replaces 4 per-call
/// `Natural::pow().to_digits_asc()` invocations (heap alloc + multi-limb
/// arithmetic) with stack-resident u128 division by a const-known base.
///
/// SAFETY (correctness): for b40 the max valid candidate is `40^8 - 1`
/// and `(40^8 - 1)³ < 40^24 ≈ 1.76e38 < u128::MAX = 3.40e38`, so the
/// cube fits with 50% margin. The `FwDigits` buffer has 38 slots,
/// enough for any specialized base ≤ 64's cube digit count (b64 needs 38).
#[inline]
fn has_duplicate_msd_prefix_u128_const<const BASE: u32>(range: FieldSize) -> bool {
    if range.size() == 1 {
        return false;
    }

    let first = range.first();
    let last = range.last();

    let start_sq = first * first;
    let end_sq = last * last;
    let start_cu = start_sq * first;
    let end_cu = end_sq * last;

    let start_sq_d = extract_digits_u128_const::<BASE>(start_sq);
    let end_sq_d = extract_digits_u128_const::<BASE>(end_sq);
    let start_cu_d = extract_digits_u128_const::<BASE>(start_cu);
    let end_cu_d = extract_digits_u128_const::<BASE>(end_cu);

    analyze_msd_prefix::<BASE>(&start_sq_d, &end_sq_d, &start_cu_d, &end_cu_d)
}

/// Specialized const-generic MSD prefix check for bases where `n³` fits in
/// U256 (i.e., bases > 40 with n ≤ 60-ish). Same structure as the u128 path
/// but uses the U256 fixed-width arithmetic from `fixed_width.rs`.
///
/// Per #16: the original malachite path's heap allocations are the dominant
/// cost on xlarge-niceonly-t1 for b40; the same pattern applies for b50 and
/// other production bases (msd-ineff workload, etc.).
#[inline]
fn has_duplicate_msd_prefix_u256_const<const BASE: u32>(range: FieldSize) -> bool {
    if range.size() == 1 {
        return false;
    }

    let first = range.first();
    let last = range.last();

    // For bases ≤ 60, n² fits in u128 (verified empirically: b60 max n² < 2^144,
    // wait — actually b60 max n ≈ 2.18e21 ≈ 2^71 → n² ≈ 2^142 doesn't fit u128).
    // Use U256 throughout.
    let start_sq = U256::mul_u128_u128(first, first);
    let end_sq = U256::mul_u128_u128(last, last);
    let start_cu = start_sq.mul_u128_truncating(first);
    let end_cu = end_sq.mul_u128_truncating(last);

    let start_sq_d = extract_digits_u256_const::<BASE>(start_sq);
    let end_sq_d = extract_digits_u256_const::<BASE>(end_sq);
    let start_cu_d = extract_digits_u256_const::<BASE>(start_cu);
    let end_cu_d = extract_digits_u256_const::<BASE>(end_cu);

    analyze_msd_prefix::<BASE>(&start_sq_d, &end_sq_d, &start_cu_d, &end_cu_d)
}

// Recursive MSD filter subdivision parameters for the binary search.
// These are tuned to try and find the natural bounds of MSD shifts without wasting too much
// time when they are naturally chaotic.
//
// The floor trades MSD recursion time against extra candidates for the
// stride table: halving it roughly doubles the number of endpoint
// digit-extractions while the deepest levels skip only slivers. With the
// k=3 stride table and the seeded nice check keeping per-candidate cost
// low, 1000 benchmarks 10-25% faster end-to-end than 250 across bases
// 40-52 in both MSD-strong and MSD-weak regions.
pub const MSD_RECURSIVE_MAX_DEPTH: u32 = 22;
pub const MSD_RECURSIVE_MIN_RANGE_SIZE: u128 = 1000;
pub const MSD_RECURSIVE_SUBDIVISION_FACTOR: usize = 2;

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
#[allow(clippy::too_many_lines)]
#[must_use]
pub fn has_duplicate_msd_prefix(range: FieldSize, base: u32) -> bool {
    // Check for edge cases
    assert!(
        range.size() > 0,
        "Range has invalid bounds, range_start must be < range_end (half-open interval)"
    );
    assert!(base <= 256, "Base must be 256 or less");

    // Stack-resident fixed-width MSD path: bypasses malachite's
    // 4 per-call `Natural::pow().to_digits_asc()` invocations (each a
    // heap alloc + multi-limb arithmetic). Profile evidence puts the
    // malachite/heap work at ~17% of total cycles on xlarge benchmark.
    //
    // b40 fits in u128 (max n³ < 1.77e38 < u128::MAX = 3.40e38). All
    // other production bases overflow u128 and need U256.
    match base {
        40 => return has_duplicate_msd_prefix_u128_const::<40>(range),
        42 => return has_duplicate_msd_prefix_u256_const::<42>(range),
        43 => return has_duplicate_msd_prefix_u256_const::<43>(range),
        44 => return has_duplicate_msd_prefix_u256_const::<44>(range),
        45 => return has_duplicate_msd_prefix_u256_const::<45>(range),
        47 => return has_duplicate_msd_prefix_u256_const::<47>(range),
        48 => return has_duplicate_msd_prefix_u256_const::<48>(range),
        49 => return has_duplicate_msd_prefix_u256_const::<49>(range),
        50 => return has_duplicate_msd_prefix_u256_const::<50>(range),
        52 => return has_duplicate_msd_prefix_u256_const::<52>(range),
        53 => return has_duplicate_msd_prefix_u256_const::<53>(range),
        54 => return has_duplicate_msd_prefix_u256_const::<54>(range),
        55 => return has_duplicate_msd_prefix_u256_const::<55>(range),
        57 => return has_duplicate_msd_prefix_u256_const::<57>(range),
        58 => return has_duplicate_msd_prefix_u256_const::<58>(range),
        59 => return has_duplicate_msd_prefix_u256_const::<59>(range),
        60 => return has_duplicate_msd_prefix_u256_const::<60>(range),
        62 => return has_duplicate_msd_prefix_u256_const::<62>(range),
        64 => return has_duplicate_msd_prefix_u256_const::<64>(range),
        _ => {}
    }

    // Can't check for duplicate values when there is only one element
    if range.size() == 1 {
        trace!("Range has only a single value, cannot use prefix optimization.");
        return false;
    }

    // Interval digit-domain (Hall) analysis for unspecialized bases that
    // still fit u64 digit masks — mirrors analyze_msd_prefix so the
    // fixed-width and malachite paths stay behaviorally identical.
    if base <= 64 {
        let s_sq = Natural::from(range.first()).pow(2).to_digits_asc(&base);
        let e_sq = Natural::from(range.last()).pow(2).to_digits_asc(&base);
        let s_cu = Natural::from(range.first()).pow(3).to_digits_asc(&base);
        let e_cu = Natural::from(range.last()).pow(3).to_digits_asc(&base);
        let mut doms = [0u64; HALL_MAX_POSITIONS];
        let mut m = 0usize;
        if s_sq.len() == e_sq.len() {
            collect_power_domains(base, &s_sq, &e_sq, &mut doms, &mut m);
        }
        if s_cu.len() == e_cu.len() {
            collect_power_domains(base, &s_cu, &e_cu, &mut doms, &mut m);
        }
        if m == 0 {
            return false;
        }
        return !has_distinct_assignment(&doms[..m]);
    }

    // Bases above 64 don't fit u64 digit masks; keep the classic
    // common-MSD-prefix duplicate/overlap analysis for them.

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

    // NOTE (2026-08 theory review): a "cross MSD×LSD collision check" used to
    // live here, gated on `range.first() / b^k == range.last() / b^k`. That
    // condition only means the range fits inside one quotient block of b^k;
    // the residues n mod b^k still vary across the range, so the low digits
    // of n² and n³ are NOT constant and using the range start's low digits
    // for the whole range was unsound (it could skip ranges containing nice
    // numbers, e.g. [68, 70) in base 10 which contains 69). The check has
    // been removed. Sound low-digit filtering is done per-candidate by the
    // stride table (lsd_filter).

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

    /// Reference implementation that always uses the malachite path.
    /// Used in `test_fixed_width_matches_malachite_*` to verify the
    /// fixed-width const-generic dispatch path produces identical
    /// results across all specialized bases.
    fn malachite_msd_reference(range: FieldSize, base: u32) -> bool {
        if range.size() == 1 {
            return false;
        }
        let s_sq = Natural::from(range.first()).pow(2).to_digits_asc(&base);
        let e_sq = Natural::from(range.last()).pow(2).to_digits_asc(&base);
        let s_cu = Natural::from(range.first()).pow(3).to_digits_asc(&base);
        let e_cu = Natural::from(range.last()).pow(3).to_digits_asc(&base);
        let mut doms = [0u64; HALL_MAX_POSITIONS];
        let mut m = 0usize;
        if s_sq.len() == e_sq.len() {
            collect_power_domains(base, &s_sq, &e_sq, &mut doms, &mut m);
        }
        if s_cu.len() == e_cu.len() {
            collect_power_domains(base, &s_cu, &e_cu, &mut doms, &mut m);
        }
        if m == 0 {
            return false;
        }
        !has_distinct_assignment(&doms[..m])
    }

    /// Cross-check: every (base, range) sample must produce the same
    /// answer through the dispatched path (which routes to the fixed-width
    /// const-generic variant for specialized bases) as through a fresh
    /// malachite computation. Catches any const-power overflow,
    /// digit-extraction off-by-one, or analyze-step deviation in the
    /// u128/U256 paths added by #16.
    #[test_log::test]
    fn test_fixed_width_msd_matches_malachite_all_bases() {
        // 50 samples per base × 17 bases = 850 cross-checks. Keep
        // sample count modest so the test stays sub-second.
        let bases: &[u32] = &[
            40, 42, 43, 44, 45, 47, 48, 49, 50, 52, 53, 54, 55, 57, 58, 59, 60, 62, 64,
        ];
        let mut state: u128 = 0x1234_5678_9abc_def0_cafe_babe_dead_beef;
        let mut rng = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            state
        };
        for &base in bases {
            let r = base_range::get_base_range_u128(base)
                .unwrap_or_else(|_| panic!("b{base} range error"))
                .unwrap_or_else(|| panic!("b{base} no range"));
            let n_low = r.start();
            let n_high = r.end();
            let span = n_high - n_low;
            for _ in 0..50 {
                let s = n_low + (rng() % (span / 2));
                let sz = (rng() % (span / 100).max(1)) + 1;
                let e = (s + sz).min(n_high);
                if e <= s {
                    continue;
                }
                let range = FieldSize::new(s, e);
                let want = malachite_msd_reference(range, base);
                let got = has_duplicate_msd_prefix(range, base);
                assert_eq!(
                    want, got,
                    "b{base} disagrees on [{s}, {e}): malachite={want}, fixed_width={got}"
                );
            }
        }
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
    fn test_collect_power_domains() {
        // 68² = 4624, 69² = 4761 in base 10 (LSD-first digit arrays).
        // Top digit: fixed 4 (singleton). Next: quotients 46..47 → {6,7}.
        // Next: quotients 462..476 span ≥ 10 values → unconstrained, stop.
        let xd = [4u32, 2, 6, 4];
        let yd = [1u32, 6, 7, 4];
        let mut doms = [0u64; HALL_MAX_POSITIONS];
        let mut m = 0;
        collect_power_domains(10, &xd, &yd, &mut doms, &mut m);
        assert_eq!(&doms[..m], &[1 << 4, (1 << 6) | (1 << 7)]);

        // Wraparound: 193..207 → digits [3,9,1] vs [7,0,2].
        // Top: {1,2}. Next: quotients 19..20 → digits {9,0} (wraps).
        let xd = [3u32, 9, 1];
        let yd = [7u32, 0, 2];
        let mut doms = [0u64; HALL_MAX_POSITIONS];
        let mut m = 0;
        collect_power_domains(10, &xd, &yd, &mut doms, &mut m);
        assert_eq!(&doms[..m], &[(1 << 1) | (1 << 2), (1 << 9) | (1 << 0)]);
    }

    #[test_log::test]
    fn test_has_distinct_assignment() {
        // Two positions, two digits: fine.
        assert!(has_distinct_assignment(&[0b01, 0b10]));
        // Duplicate singletons (the classic prefix-duplicate case): fail.
        assert!(!has_distinct_assignment(&[0b100, 0b100]));
        // Three positions sharing two digits: no pairwise conflict between
        // distinct masks, but collectively impossible.
        assert!(!has_distinct_assignment(&[0b11, 0b11, 0b11]));
        // Four positions over three digits, every pair satisfiable: fail.
        assert!(!has_distinct_assignment(&[0b011, 0b011, 0b110, 0b101]));
        // Augmenting path required: position order forces reassignment.
        assert!(has_distinct_assignment(&[0b01, 0b11, 0b110]));
    }

    #[test_log::test]
    fn test_hall_rejects_more_than_prefix_checks() {
        // Regression pin for the interval-domain upgrade: this b42 range has
        // no duplicate or overlap among its fixed common-prefix digits, but
        // the constrained near-fixed positions collectively lack enough
        // distinct digits, so the range is skippable only via the Hall check.
        let range = FieldSize::new(12_283_591_331_194, 12_297_719_016_486);
        assert!(has_duplicate_msd_prefix(range, 42));
    }

    #[test_log::test]
    fn test_range_containing_nice_number_not_skipped() {
        // Regression test for the unsound cross MSD×LSD check (removed in the
        // 2026-08 theory review). The old check skipped [68, 70) in base 10
        // because 68 and 69 share a b² quotient block and 68's low square
        // digits collide with the square MSD prefix — but the low digits are
        // not constant across the range, and 69 (a nice number!) is inside.
        let range = FieldSize::new(68, 70);
        assert!(
            !has_duplicate_msd_prefix(range, 10),
            "[68, 70) contains the nice number 69 and must not be skipped"
        );
    }

    #[test_log::test]
    fn test_msd_filter_never_skips_nice_numbers_small_bases() {
        // Soundness: for every nice number found by brute force in small
        // bases, no range containing it may be reported as skippable —
        // regardless of the range's size or alignment.
        use crate::client_process::get_is_nice;

        for base in 4u32..=16 {
            let Ok(Some(base_range)) = base_range::get_base_range_u128(base) else {
                continue;
            };
            let nice: Vec<u128> = (base_range.start()..base_range.end())
                .filter(|&n| get_is_nice(n, base))
                .collect();
            debug!("base {base}: nice numbers {nice:?}");
            for &n in &nice {
                for size in [2u128, 3, 5, 10, 50, 251, 1000] {
                    for offset in 0..size.min(8) {
                        let start = n.saturating_sub(offset).max(base_range.start());
                        let end = (start + size).min(base_range.end());
                        if start >= end || n < start || n >= end {
                            continue;
                        }
                        let range = FieldSize::new(start, end);
                        assert!(
                            !has_duplicate_msd_prefix(range, base),
                            "base {base}: range [{start}, {end}) containing nice number {n} was skipped"
                        );
                    }
                }
            }
        }
    }

    #[test_log::test]
    fn test_get_valid_ranges_covers_all_nice_numbers_small_bases() {
        // Integration-level soundness: running the production recursion over
        // the full base range must leave every nice number inside some
        // returned "must process" sub-range.
        use crate::client_process::get_is_nice;

        for base in 4u32..=16 {
            let Ok(Some(base_range)) = base_range::get_base_range_u128(base) else {
                continue;
            };
            let nice: Vec<u128> = (base_range.start()..base_range.end())
                .filter(|&n| get_is_nice(n, base))
                .collect();
            let ranges = get_valid_ranges(base_range, base);
            for &n in &nice {
                assert!(
                    ranges.iter().any(|r| r.start() <= n && n < r.end()),
                    "base {base}: nice number {n} not covered by get_valid_ranges output"
                );
            }
        }
    }
}
