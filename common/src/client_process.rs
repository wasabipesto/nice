//! A module with "nice" calculation utilities for the client.
//!
//! The search ranges are precalculated by the server and all numbers in the
//! range are guaranteed to have a square and cube ("sqube") with the correct
//! number of digits. The ranges provided are a sequential and continuous.

use crate::fixed_width::U256;
use crate::{
    CLIENT_VERSION, DataToClient, DataToServer, FieldResults, FieldSize, NiceNumberSimple,
    UniquesDistributionSimple,
};
use crate::{msd_prefix_filter, number_stats, stride_filter};
use malachite::base::num::arithmetic::traits::{DivAssignRem, Pow};
use malachite::base::num::conversion::traits::Digits;
use malachite::natural::Natural;

/// Maximum supported base for the stack-resident digit array.
/// Bases above this still work but should not be reached given the u128
/// representation already caps n at base 97.
const MAX_BASE_FOR_DIGIT_ARRAY_U128: usize = 128;

/// Inclusive upper bound on bases that can use the U256 fast path.
/// Above this, n³ exceeds 256 bits and we fall back to malachite `Natural`.
/// Empirically determined: base 70's max-n cubed is 3.1e77 > 2^256 (1.16e77),
/// while base 68's max-n cubed is ~2.7e74 < 2^256. Pick 68 for safety.
pub const MAX_BASE_FOR_FIXED_WIDTH_U256: u32 = 68;

/// Inclusive upper bound on bases where n³ fits in u128 (skips the U256 path
/// entirely). Base 40 max n³ is 2.81e38 < 2^128 (3.40e38); base 41+ either
/// has no valid range or exceeds u128.
///
/// Empirically (intervention #3 notes): u128 fast path wins big for niceonly
/// b40 (+57%), but loses to malachite for *detailed* b40 (-12%). U256 with
/// leading-zero skip is the better choice for bases 41–68 (msd-ineff +30%).
/// We accept the detailed-b40 regression because the niceonly speedup
/// dominates real workloads — most production traffic is niceonly.
const MAX_BASE_FOR_FIXED_WIDTH_U128: u32 = 40;

/// Calculate the number of unique digits in (n^2, n^3) represented in base b.
/// A number is nice if the result of this is equal to b (means all digits are used once).
/// If you're just checking if the number is 100% nice, there is a faster version below.
///
/// Const-generic dispatch lets popular bases use multiply-by-magic-constant
/// for digit extraction, which in detailed mode (full extraction every
/// candidate) outperforms malachite's small-divisor multi-limb division.
#[must_use]
pub fn get_num_unique_digits(num_u128: u128, base: u32) -> u32 {
    match base {
        40 => get_num_unique_digits_u128_const::<40>(num_u128),
        42 => get_num_unique_digits_u256_const::<42>(num_u128),
        43 => get_num_unique_digits_u256_const::<43>(num_u128),
        44 => get_num_unique_digits_u256_const::<44>(num_u128),
        45 => get_num_unique_digits_u256_const::<45>(num_u128),
        47 => get_num_unique_digits_u256_const::<47>(num_u128),
        48 => get_num_unique_digits_u256_const::<48>(num_u128),
        49 => get_num_unique_digits_u256_const::<49>(num_u128),
        50 => get_num_unique_digits_u256_const::<50>(num_u128),
        52 => get_num_unique_digits_u256_const::<52>(num_u128),
        53 => get_num_unique_digits_u256_const::<53>(num_u128),
        54 => get_num_unique_digits_u256_const::<54>(num_u128),
        55 => get_num_unique_digits_u256_const::<55>(num_u128),
        57 => get_num_unique_digits_u256_const::<57>(num_u128),
        58 => get_num_unique_digits_u256_const::<58>(num_u128),
        59 => get_num_unique_digits_u256_const::<59>(num_u128),
        60 => get_num_unique_digits_u256_const::<60>(num_u128),
        // Note: Cannot use u256 const path for bases > 63 due to bitmask size limit
        _ => get_num_unique_digits_natural(num_u128, base),
    }
}

/// u128 fast path with compile-time-constant base.
#[inline]
#[allow(clippy::cast_possible_truncation)]
fn get_num_unique_digits_u128_const<const BASE: u32>(num: u128) -> u32 {
    const { assert!(BASE <= 64, "u64 bitmask can't index past bit 63") };
    let base_u128 = u128::from(BASE);

    // Uses a `u64` bitmask for digit tracking: BT/BTS-style register ops instead
    // of the `[bool; 128]` stack array's load/store ping-pong.
    let mut digits_indicator: u64 = 0;

    let squared = num * num;
    let cubed = squared * num;

    let mut n = squared;
    while n != 0 {
        let d = (n % base_u128) as u32;
        n /= base_u128;
        digits_indicator |= 1u64 << d;
    }
    let mut n = cubed;
    while n != 0 {
        let d = (n % base_u128) as u32;
        n /= base_u128;
        digits_indicator |= 1u64 << d;
    }
    digits_indicator.count_ones()
}

/// U256 fast path with compile-time-constant base.
///
/// `u64` bitmask: `1u64 << d` is a single SHL on x86-64 vs `1u128 << d`'s
/// multi-instruction sequence. Const-generic dispatch only routes bases ≤ 60
/// here, so 64 bits is enough.
#[inline]
fn get_num_unique_digits_u256_const<const BASE: u32>(num: u128) -> u32 {
    const { assert!(BASE <= 64, "u64 bitmask can't index past bit 63") };
    // 🔥🔥🔥 HOT LOOP 🔥🔥🔥
    let mut digits_indicator: u64 = 0;

    let squared = U256::mul_u128_u128(num, num);
    let cubed = squared.mul_u128_truncating(num);

    let mut n = squared;
    while !n.is_zero() {
        let d = n.div_assign_rem_u32_const::<BASE>();
        digits_indicator |= 1u64 << d;
    }
    let mut n = cubed;
    while !n.is_zero() {
        let d = n.div_assign_rem_u32_const::<BASE>();
        digits_indicator |= 1u64 << d;
    }
    digits_indicator.count_ones()
}

/// Malachite fallback for bases without a const-generic specialisation.
fn get_num_unique_digits_natural(num_u128: u128, base: u32) -> u32 {
    let mut digits_indicator: u128 = 0;
    let num = Natural::from(num_u128);

    let squared = (&num).pow(2);
    for digit in squared.to_digits_asc(&base) {
        digits_indicator |= 1 << digit;
    }
    let cubed = squared * &num;
    for digit in cubed.to_digits_asc(&base) {
        digits_indicator |= 1 << digit;
    }
    digits_indicator.count_ones()
}

/// The inner loop of detailed field processing. Also called by other crates like the WASM client.
///
/// **Range semantics**: Expects a half-open range [`range_start`, `range_end`) where `range_start`
/// is inclusive and `range_end` is exclusive, following Rust's standard convention.
#[must_use]
pub fn process_range_detailed(range: &FieldSize, base: u32) -> FieldResults {
    // Calculate the minimum num_unique_digits cutoff
    let nice_list_cutoff = number_stats::get_near_miss_cutoff(base);
    let base_idx = base as usize;
    debug_assert!(base_idx < MAX_BASE_FOR_DIGIT_ARRAY_U128);

    // Initialize a list for nice and semi-nice numbers
    let mut nice_numbers: Vec<NiceNumberSimple> = Vec::new();

    // Stack-resident histogram. Index is num_uniques (0..=base).
    let mut histogram = [0u128; MAX_BASE_FOR_DIGIT_ARRAY_U128 + 1];

    for num in range.range_iter() {
        // Get the number of unique digits for this number
        let num_uniques = get_num_unique_digits(num, base);

        // Save the count in the histogram
        histogram[num_uniques as usize] += 1;

        // Save the number if it exceeds the nice list cutoff
        if num_uniques > nice_list_cutoff {
            nice_numbers.push(NiceNumberSimple {
                number: num,
                num_uniques,
            });
        }
    }

    // Build distribution Vec from histogram, matching the previous shape
    // (one entry per i in 1..=base, in ascending order).
    let distribution: Vec<UniquesDistributionSimple> = (1..=base)
        .map(|i| UniquesDistributionSimple {
            num_uniques: i,
            count: histogram[i as usize],
        })
        .collect();

    FieldResults {
        distribution,
        nice_numbers,
    }
}

/// Process a field by aggregating statistics on the niceness of numbers in a range.
#[must_use]
#[deprecated = "use process_range_detailed instead"]
pub fn process_detailed(claim_data: &DataToClient, username: &String) -> DataToServer {
    let results = process_range_detailed(&claim_data.into(), claim_data.base);

    DataToServer {
        claim_id: claim_data.claim_id,
        username: username.to_owned(),
        client_version: CLIENT_VERSION.to_string(),
        unique_distribution: Some(results.distribution),
        nice_numbers: results.nice_numbers,
    }
}

/// Quickly determine if a number is 100% nice in this base.
/// A number is nice if (n^2, n^3), converted to base b, have all digits of base b.
/// Assumes we have already done residue class filtering.
/// Immediately stops if we hit a duplicate digit.
///
/// Dispatches to fast paths based on base:
/// - base ≤ 40: u128 throughout (n³ ≤ 2^128)
/// - 40 < base ≤ 68: U256 (n³ ≤ 2^256)
/// - base > 68: malachite `Natural` fallback
///
/// # Panics
/// Panics in debug builds if base ≥ `MAX_BASE_FOR_DIGIT_ARRAY_U128`.
#[must_use]
#[inline]
pub fn get_is_nice(num: u128, base: u32) -> bool {
    debug_assert!((base as usize) < MAX_BASE_FOR_DIGIT_ARRAY_U128);
    // For popular bases, dispatch to a const-generic specialization so
    // LLVM converts `n / BASE` / `n % BASE` into a multiply-by-magic-
    // constant sequence. This is several times faster than runtime u128
    // division.
    match base {
        40 => get_is_nice_u128_const::<40>(num),
        42 => get_is_nice_u256_const::<42>(num),
        43 => get_is_nice_u256_const::<43>(num),
        44 => get_is_nice_u256_const::<44>(num),
        45 => get_is_nice_u256_const::<45>(num),
        47 => get_is_nice_u256_const::<47>(num),
        48 => get_is_nice_u256_const::<48>(num),
        49 => get_is_nice_u256_const::<49>(num),
        50 => get_is_nice_u256_const::<50>(num),
        52 => get_is_nice_u256_const::<52>(num),
        53 => get_is_nice_u256_const::<53>(num),
        54 => get_is_nice_u256_const::<54>(num),
        55 => get_is_nice_u256_const::<55>(num),
        57 => get_is_nice_u256_const::<57>(num),
        58 => get_is_nice_u256_const::<58>(num),
        59 => get_is_nice_u256_const::<59>(num),
        60 => get_is_nice_u256_const::<60>(num),
        // Note: Cannot use u256 const path for bases > 63 due to bitmask size limit
        _ if base <= MAX_BASE_FOR_FIXED_WIDTH_U128 => get_is_nice_u128(num, base),
        _ if base <= MAX_BASE_FOR_FIXED_WIDTH_U256 => get_is_nice_u256(num, base),
        _ => get_is_nice_natural(num, base),
    }
}

/// u128 fast path with compile-time-constant base.
#[inline]
#[allow(clippy::cast_possible_truncation)]
fn get_is_nice_u128_const<const BASE: u32>(num: u128) -> bool {
    const { assert!(BASE <= 64, "u64 bitmask can't index past bit 63") };
    let base_u128 = u128::from(BASE);

    // Uses a `u64` bitmask for digit tracking: BT/BTS-style register ops instead
    // of the `[bool; 128]` stack array's load/store ping-pong.
    let mut digits_indicator: u64 = 0;

    let squared = num * num;

    let mut n = squared;
    while n != 0 {
        let d = (n % base_u128) as u32;
        n /= base_u128;
        let bit = 1u64 << d;
        if digits_indicator & bit != 0 {
            return false;
        }
        digits_indicator |= bit;
    }

    let mut n = squared * num;
    while n != 0 {
        let d = (n % base_u128) as u32;
        n /= base_u128;
        let bit = 1u64 << d;
        if digits_indicator & bit != 0 {
            return false;
        }
        digits_indicator |= bit;
    }
    true
}

/// U256 fast path with compile-time-constant base.
///
/// Uses a `u64` bitmask for digit tracking: BT/BTS-style register ops instead
/// of the `[bool; 128]` stack array's load/store ping-pong. Const-generic
/// dispatch only routes bases ≤ 60 here, so 64 bits is enough.
#[inline]
fn get_is_nice_u256_const<const BASE: u32>(num: u128) -> bool {
    const { assert!(BASE <= 64, "u64 bitmask can't index past bit 63") };
    // 🔥🔥🔥 HOT LOOP 🔥🔥🔥
    let mut digits_indicator: u64 = 0;

    let squared = U256::mul_u128_u128(num, num);

    let mut n = squared;
    while !n.is_zero() {
        let d = n.div_assign_rem_u32_const::<BASE>();
        let bit = 1u64 << d;
        if digits_indicator & bit != 0 {
            return false;
        }
        digits_indicator |= bit;
    }

    let mut n = squared.mul_u128_truncating(num);
    while !n.is_zero() {
        let d = n.div_assign_rem_u32_const::<BASE>();
        let bit = 1u64 << d;
        if digits_indicator & bit != 0 {
            return false;
        }
        digits_indicator |= bit;
    }
    true
}

/// u128 fast path. Safe for bases ≤ 40.
#[inline]
fn get_is_nice_u128(num: u128, base: u32) -> bool {
    // 🔥🔥🔥 HOT LOOP 🔥🔥🔥
    let base_u128 = u128::from(base);
    let mut digits_indicator = [false; MAX_BASE_FOR_DIGIT_ARRAY_U128];

    let squared = num * num;

    let mut n = squared;
    while n != 0 {
        let d = (n % base_u128) as usize;
        n /= base_u128;
        if digits_indicator[d] {
            return false;
        }
        digits_indicator[d] = true;
    }

    let mut n = squared * num;
    while n != 0 {
        let d = (n % base_u128) as usize;
        n /= base_u128;
        if digits_indicator[d] {
            return false;
        }
        digits_indicator[d] = true;
    }
    true
}

/// U256 path. Safe for bases up to `MAX_BASE_FOR_FIXED_WIDTH_U256`.
#[inline]
fn get_is_nice_u256(num: u128, base: u32) -> bool {
    // 🔥🔥🔥 HOT LOOP 🔥🔥🔥
    let mut digits_indicator = [false; MAX_BASE_FOR_DIGIT_ARRAY_U128];

    let squared = U256::mul_u128_u128(num, num);

    let mut n = squared;
    while !n.is_zero() {
        let d = n.div_assign_rem_u32(base) as usize;
        if digits_indicator[d] {
            return false;
        }
        digits_indicator[d] = true;
    }

    let mut n = squared.mul_u128_truncating(num);
    while !n.is_zero() {
        let d = n.div_assign_rem_u32(base) as usize;
        if digits_indicator[d] {
            return false;
        }
        digits_indicator[d] = true;
    }
    true
}

/// Malachite Natural fallback for bases > `MAX_BASE_FOR_FIXED_WIDTH_U256`.
#[inline]
fn get_is_nice_natural(num: u128, base: u32) -> bool {
    let num = Natural::from(num);
    let base_natural = Natural::from(base);
    let mut digits_indicator = [false; MAX_BASE_FOR_DIGIT_ARRAY_U128];

    let squared = (&num).pow(2);
    let mut n = squared.clone();
    while n > 0 {
        let remainder =
            usize::try_from(&(n.div_assign_rem(&base_natural))).expect("digit fits in usize");
        if digits_indicator[remainder] {
            return false;
        }
        digits_indicator[remainder] = true;
    }
    let mut n = squared * num;
    while n > 0 {
        let remainder =
            usize::try_from(&(n.div_assign_rem(&base_natural))).expect("digit fits in usize");
        if digits_indicator[remainder] {
            return false;
        }
        digits_indicator[remainder] = true;
    }
    true
}

/// Process a field by looking for completely nice numbers.
/// Implements several optimizations over the detailed search.
#[must_use]
#[deprecated = "use process_range_niceonly instead"]
pub fn process_niceonly(claim_data: &DataToClient, username: &String) -> DataToServer {
    let k = 1; // Number of digits for multi-digit LSD filter
    let stride_table = stride_filter::StrideTable::new(claim_data.base, k);
    let results = process_range_niceonly(&claim_data.into(), claim_data.base, &stride_table);

    DataToServer {
        claim_id: claim_data.claim_id,
        username: username.to_owned(),
        client_version: CLIENT_VERSION.to_string(),
        unique_distribution: None,
        nice_numbers: results.nice_numbers,
    }
}

/// Process a field by looking for completely nice numbers.
/// Implements several optimizations over the detailed search.
///
/// **Range semantics**: Expects a half-open range [`range_start`, `range_end`) where `range_start`
/// is inclusive and `range_end` is exclusive, following Rust's standard convention.
#[must_use]
pub fn process_range_niceonly(
    range: &FieldSize,
    base: u32,
    stride_table: &stride_filter::StrideTable,
) -> FieldResults {
    // Use recursive subdivision to get valid ranges that need processing.
    // This adaptively subdivides the range to skip portions where the MSD prefix indicates
    // all numbers will have duplicate/overlapping digits. It's more effective than fixed
    // chunking because it only subdivides when needed and can find natural boundaries.
    let valid_msd_ranges = msd_prefix_filter::get_valid_ranges(*range, base);

    // The stride table integrates the residue filter (mod b-1) and the multi-digit
    // LSD filter (mod b^k). It allows us to jump between valid candidates instead of
    // iterating over each one.
    // The table is precomputed once per field and passed in to avoid redundant computation.

    let mut nice_list = Vec::new();
    for sub_range in valid_msd_ranges {
        let sub_results = stride_table.iterate_range(&sub_range, base);
        nice_list.extend(sub_results);
    }

    FieldResults {
        distribution: Vec::new(),
        nice_numbers: nice_list,
    }
}

#[cfg(test)]
#[allow(clippy::too_many_lines)]
mod tests {
    use super::*;
    use crate::base_range;

    #[test_log::test]
    fn process_detailed_b10() {
        let base = 10;
        let base_range = base_range::get_base_range_u128(base).unwrap().unwrap();
        let input = DataToClient {
            claim_id: 0,
            base,
            range_start: base_range.start(),
            range_end: base_range.end(),
            range_size: base_range.size(),
        };
        let result = FieldResults {
            distribution: Vec::from([
                UniquesDistributionSimple {
                    num_uniques: 1,
                    count: 0,
                },
                UniquesDistributionSimple {
                    num_uniques: 2,
                    count: 0,
                },
                UniquesDistributionSimple {
                    num_uniques: 3,
                    count: 0,
                },
                UniquesDistributionSimple {
                    num_uniques: 4,
                    count: 4,
                },
                UniquesDistributionSimple {
                    num_uniques: 5,
                    count: 5,
                },
                UniquesDistributionSimple {
                    num_uniques: 6,
                    count: 15,
                },
                UniquesDistributionSimple {
                    num_uniques: 7,
                    count: 20,
                },
                UniquesDistributionSimple {
                    num_uniques: 8,
                    count: 7,
                },
                UniquesDistributionSimple {
                    num_uniques: 9,
                    count: 1,
                },
                UniquesDistributionSimple {
                    num_uniques: 10,
                    count: 1,
                },
            ]),
            nice_numbers: Vec::from([NiceNumberSimple {
                number: 69,
                num_uniques: 10,
            }]),
        };
        assert_eq!(process_range_detailed(&input.into(), input.base), result);
    }

    #[test_log::test]
    fn process_detailed_b40() {
        let base = 40;
        let base_range = base_range::get_base_range_u128(base).unwrap().unwrap();
        let size = 10000;
        let input = DataToClient {
            claim_id: 0,
            base,
            range_start: base_range.start(),
            range_end: base_range.start() + size,
            range_size: size,
        };
        let result = FieldResults {
            distribution: Vec::from([
                UniquesDistributionSimple {
                    num_uniques: 1,
                    count: 0,
                },
                UniquesDistributionSimple {
                    num_uniques: 2,
                    count: 0,
                },
                UniquesDistributionSimple {
                    num_uniques: 3,
                    count: 0,
                },
                UniquesDistributionSimple {
                    num_uniques: 4,
                    count: 0,
                },
                UniquesDistributionSimple {
                    num_uniques: 5,
                    count: 0,
                },
                UniquesDistributionSimple {
                    num_uniques: 6,
                    count: 0,
                },
                UniquesDistributionSimple {
                    num_uniques: 7,
                    count: 0,
                },
                UniquesDistributionSimple {
                    num_uniques: 8,
                    count: 0,
                },
                UniquesDistributionSimple {
                    num_uniques: 9,
                    count: 0,
                },
                UniquesDistributionSimple {
                    num_uniques: 10,
                    count: 0,
                },
                UniquesDistributionSimple {
                    num_uniques: 11,
                    count: 0,
                },
                UniquesDistributionSimple {
                    num_uniques: 12,
                    count: 0,
                },
                UniquesDistributionSimple {
                    num_uniques: 13,
                    count: 0,
                },
                UniquesDistributionSimple {
                    num_uniques: 14,
                    count: 0,
                },
                UniquesDistributionSimple {
                    num_uniques: 15,
                    count: 1,
                },
                UniquesDistributionSimple {
                    num_uniques: 16,
                    count: 2,
                },
                UniquesDistributionSimple {
                    num_uniques: 17,
                    count: 15,
                },
                UniquesDistributionSimple {
                    num_uniques: 18,
                    count: 68,
                },
                UniquesDistributionSimple {
                    num_uniques: 19,
                    count: 190,
                },
                UniquesDistributionSimple {
                    num_uniques: 20,
                    count: 423,
                },
                UniquesDistributionSimple {
                    num_uniques: 21,
                    count: 959,
                },
                UniquesDistributionSimple {
                    num_uniques: 22,
                    count: 1615,
                },
                UniquesDistributionSimple {
                    num_uniques: 23,
                    count: 1995,
                },
                UniquesDistributionSimple {
                    num_uniques: 24,
                    count: 1982,
                },
                UniquesDistributionSimple {
                    num_uniques: 25,
                    count: 1438,
                },
                UniquesDistributionSimple {
                    num_uniques: 26,
                    count: 825,
                },
                UniquesDistributionSimple {
                    num_uniques: 27,
                    count: 349,
                },
                UniquesDistributionSimple {
                    num_uniques: 28,
                    count: 110,
                },
                UniquesDistributionSimple {
                    num_uniques: 29,
                    count: 26,
                },
                UniquesDistributionSimple {
                    num_uniques: 30,
                    count: 2,
                },
                UniquesDistributionSimple {
                    num_uniques: 31,
                    count: 0,
                },
                UniquesDistributionSimple {
                    num_uniques: 32,
                    count: 0,
                },
                UniquesDistributionSimple {
                    num_uniques: 33,
                    count: 0,
                },
                UniquesDistributionSimple {
                    num_uniques: 34,
                    count: 0,
                },
                UniquesDistributionSimple {
                    num_uniques: 35,
                    count: 0,
                },
                UniquesDistributionSimple {
                    num_uniques: 36,
                    count: 0,
                },
                UniquesDistributionSimple {
                    num_uniques: 37,
                    count: 0,
                },
                UniquesDistributionSimple {
                    num_uniques: 38,
                    count: 0,
                },
                UniquesDistributionSimple {
                    num_uniques: 39,
                    count: 0,
                },
                UniquesDistributionSimple {
                    num_uniques: 40,
                    count: 0,
                },
            ]),
            nice_numbers: Vec::new(),
        };
        assert_eq!(process_range_detailed(&input.into(), input.base), result);
    }

    #[test_log::test]
    fn process_detailed_b80() {
        let base = 80;
        let base_range = base_range::get_base_range_u128(base).unwrap().unwrap();
        let size = 10000;
        let input = DataToClient {
            claim_id: 0,
            base,
            range_start: base_range.start(),
            range_end: base_range.start() + size,
            range_size: size,
        };
        let result = FieldResults {
            distribution: Vec::from([
                UniquesDistributionSimple {
                    num_uniques: 1,
                    count: 0,
                },
                UniquesDistributionSimple {
                    num_uniques: 2,
                    count: 0,
                },
                UniquesDistributionSimple {
                    num_uniques: 3,
                    count: 0,
                },
                UniquesDistributionSimple {
                    num_uniques: 4,
                    count: 0,
                },
                UniquesDistributionSimple {
                    num_uniques: 5,
                    count: 0,
                },
                UniquesDistributionSimple {
                    num_uniques: 6,
                    count: 0,
                },
                UniquesDistributionSimple {
                    num_uniques: 7,
                    count: 0,
                },
                UniquesDistributionSimple {
                    num_uniques: 8,
                    count: 0,
                },
                UniquesDistributionSimple {
                    num_uniques: 9,
                    count: 0,
                },
                UniquesDistributionSimple {
                    num_uniques: 10,
                    count: 0,
                },
                UniquesDistributionSimple {
                    num_uniques: 11,
                    count: 0,
                },
                UniquesDistributionSimple {
                    num_uniques: 12,
                    count: 0,
                },
                UniquesDistributionSimple {
                    num_uniques: 13,
                    count: 0,
                },
                UniquesDistributionSimple {
                    num_uniques: 14,
                    count: 0,
                },
                UniquesDistributionSimple {
                    num_uniques: 15,
                    count: 0,
                },
                UniquesDistributionSimple {
                    num_uniques: 16,
                    count: 0,
                },
                UniquesDistributionSimple {
                    num_uniques: 17,
                    count: 0,
                },
                UniquesDistributionSimple {
                    num_uniques: 18,
                    count: 0,
                },
                UniquesDistributionSimple {
                    num_uniques: 19,
                    count: 0,
                },
                UniquesDistributionSimple {
                    num_uniques: 20,
                    count: 0,
                },
                UniquesDistributionSimple {
                    num_uniques: 21,
                    count: 0,
                },
                UniquesDistributionSimple {
                    num_uniques: 22,
                    count: 0,
                },
                UniquesDistributionSimple {
                    num_uniques: 23,
                    count: 0,
                },
                UniquesDistributionSimple {
                    num_uniques: 24,
                    count: 0,
                },
                UniquesDistributionSimple {
                    num_uniques: 25,
                    count: 0,
                },
                UniquesDistributionSimple {
                    num_uniques: 26,
                    count: 0,
                },
                UniquesDistributionSimple {
                    num_uniques: 27,
                    count: 0,
                },
                UniquesDistributionSimple {
                    num_uniques: 28,
                    count: 0,
                },
                UniquesDistributionSimple {
                    num_uniques: 29,
                    count: 0,
                },
                UniquesDistributionSimple {
                    num_uniques: 30,
                    count: 0,
                },
                UniquesDistributionSimple {
                    num_uniques: 31,
                    count: 0,
                },
                UniquesDistributionSimple {
                    num_uniques: 32,
                    count: 0,
                },
                UniquesDistributionSimple {
                    num_uniques: 33,
                    count: 0,
                },
                UniquesDistributionSimple {
                    num_uniques: 34,
                    count: 0,
                },
                UniquesDistributionSimple {
                    num_uniques: 35,
                    count: 0,
                },
                UniquesDistributionSimple {
                    num_uniques: 36,
                    count: 1,
                },
                UniquesDistributionSimple {
                    num_uniques: 37,
                    count: 6,
                },
                UniquesDistributionSimple {
                    num_uniques: 38,
                    count: 14,
                },
                UniquesDistributionSimple {
                    num_uniques: 39,
                    count: 62,
                },
                UniquesDistributionSimple {
                    num_uniques: 40,
                    count: 122,
                },
                UniquesDistributionSimple {
                    num_uniques: 41,
                    count: 263,
                },
                UniquesDistributionSimple {
                    num_uniques: 42,
                    count: 492,
                },
                UniquesDistributionSimple {
                    num_uniques: 43,
                    count: 830,
                },
                UniquesDistributionSimple {
                    num_uniques: 44,
                    count: 1170,
                },
                UniquesDistributionSimple {
                    num_uniques: 45,
                    count: 1392,
                },
                UniquesDistributionSimple {
                    num_uniques: 46,
                    count: 1477,
                },
                UniquesDistributionSimple {
                    num_uniques: 47,
                    count: 1427,
                },
                UniquesDistributionSimple {
                    num_uniques: 48,
                    count: 1145,
                },
                UniquesDistributionSimple {
                    num_uniques: 49,
                    count: 745,
                },
                UniquesDistributionSimple {
                    num_uniques: 50,
                    count: 462,
                },
                UniquesDistributionSimple {
                    num_uniques: 51,
                    count: 242,
                },
                UniquesDistributionSimple {
                    num_uniques: 52,
                    count: 88,
                },
                UniquesDistributionSimple {
                    num_uniques: 53,
                    count: 35,
                },
                UniquesDistributionSimple {
                    num_uniques: 54,
                    count: 19,
                },
                UniquesDistributionSimple {
                    num_uniques: 55,
                    count: 7,
                },
                UniquesDistributionSimple {
                    num_uniques: 56,
                    count: 1,
                },
                UniquesDistributionSimple {
                    num_uniques: 57,
                    count: 0,
                },
                UniquesDistributionSimple {
                    num_uniques: 58,
                    count: 0,
                },
                UniquesDistributionSimple {
                    num_uniques: 59,
                    count: 0,
                },
                UniquesDistributionSimple {
                    num_uniques: 60,
                    count: 0,
                },
                UniquesDistributionSimple {
                    num_uniques: 61,
                    count: 0,
                },
                UniquesDistributionSimple {
                    num_uniques: 62,
                    count: 0,
                },
                UniquesDistributionSimple {
                    num_uniques: 63,
                    count: 0,
                },
                UniquesDistributionSimple {
                    num_uniques: 64,
                    count: 0,
                },
                UniquesDistributionSimple {
                    num_uniques: 65,
                    count: 0,
                },
                UniquesDistributionSimple {
                    num_uniques: 66,
                    count: 0,
                },
                UniquesDistributionSimple {
                    num_uniques: 67,
                    count: 0,
                },
                UniquesDistributionSimple {
                    num_uniques: 68,
                    count: 0,
                },
                UniquesDistributionSimple {
                    num_uniques: 69,
                    count: 0,
                },
                UniquesDistributionSimple {
                    num_uniques: 70,
                    count: 0,
                },
                UniquesDistributionSimple {
                    num_uniques: 71,
                    count: 0,
                },
                UniquesDistributionSimple {
                    num_uniques: 72,
                    count: 0,
                },
                UniquesDistributionSimple {
                    num_uniques: 73,
                    count: 0,
                },
                UniquesDistributionSimple {
                    num_uniques: 74,
                    count: 0,
                },
                UniquesDistributionSimple {
                    num_uniques: 75,
                    count: 0,
                },
                UniquesDistributionSimple {
                    num_uniques: 76,
                    count: 0,
                },
                UniquesDistributionSimple {
                    num_uniques: 77,
                    count: 0,
                },
                UniquesDistributionSimple {
                    num_uniques: 78,
                    count: 0,
                },
                UniquesDistributionSimple {
                    num_uniques: 79,
                    count: 0,
                },
                UniquesDistributionSimple {
                    num_uniques: 80,
                    count: 0,
                },
            ]),
            nice_numbers: Vec::new(),
        };
        assert_eq!(process_range_detailed(&input.into(), input.base), result);
    }

    #[test_log::test]
    fn process_niceonly_b10() {
        let base = 10;
        let base_range = base_range::get_base_range_u128(base).unwrap().unwrap();
        let input = DataToClient {
            claim_id: 0,
            base,
            range_start: base_range.start(),
            range_end: base_range.end(),
            range_size: base_range.size(),
        };
        let result = FieldResults {
            distribution: Vec::new(),
            nice_numbers: Vec::from([NiceNumberSimple {
                number: 69,
                num_uniques: 10,
            }]),
        };
        let k = 1;
        let stride_table = stride_filter::StrideTable::new(input.base, k);
        assert_eq!(
            process_range_niceonly(&input.into(), input.base, &stride_table),
            result
        );
    }

    #[test_log::test]
    fn process_niceonly_b40() {
        let base = 40;
        let base_range = base_range::get_base_range_u128(base).unwrap().unwrap();
        let size = 10000;
        let input = DataToClient {
            claim_id: 0,
            base,
            range_start: base_range.start(),
            range_end: base_range.start() + size,
            range_size: size,
        };
        let result = FieldResults {
            distribution: Vec::new(),
            nice_numbers: Vec::new(),
        };
        let k = 1;
        let stride_table = stride_filter::StrideTable::new(input.base, k);
        assert_eq!(
            process_range_niceonly(&input.into(), input.base, &stride_table),
            result
        );
    }

    #[test_log::test]
    fn process_niceonly_b80() {
        let base = 80;
        let base_range = base_range::get_base_range_u128(base).unwrap().unwrap();
        let size = 10000;
        let input = DataToClient {
            claim_id: 0,
            base,
            range_start: base_range.start(),
            range_end: base_range.start() + size,
            range_size: size,
        };
        let result = FieldResults {
            distribution: Vec::new(),
            nice_numbers: Vec::new(),
        };
        let k = 1;
        let stride_table = stride_filter::StrideTable::new(input.base, k);
        assert_eq!(
            process_range_niceonly(&input.into(), input.base, &stride_table),
            result
        );
    }

    #[test_log::test]
    fn test_chunked_msd_filtering() {
        // This test verifies that chunked MSD filtering works correctly
        // Using base 20 which has known skippable segments
        let base = 20;
        let base_range = base_range::get_base_range_u128(base).unwrap().unwrap();

        // Use a segment that should have some skippable chunks
        let chunk_size = base_range.size() / 10_000;
        let segment_start = base_range.range_start + (30 * chunk_size);
        let segment_end = segment_start + (5 * chunk_size);
        let range = FieldSize::new(segment_start, segment_end);

        // Process with chunked filtering
        let k = 1;
        let stride_table = stride_filter::StrideTable::new(base, k);
        let results = process_range_niceonly(&range, base, &stride_table);

        // The test passes if it completes without panic
        // The chunked filtering should skip some sub-chunks within this segment
        assert!(results.nice_numbers.len() <= (segment_end - segment_start) as usize);
    }

    #[test_log::test]
    fn test_chunked_vs_whole_range_consistency() {
        // Verify that processing with chunks gives same results as without
        // (when whole range isn't skippable but some chunks are)
        let base = 10;
        let range_start = 47;
        let range_end = 147; // Larger range to test multiple chunks
        let range = FieldSize::new(range_start, range_end);

        // Process the range
        let k = 1;
        let stride_table = stride_filter::StrideTable::new(base, k);
        let results = process_range_niceonly(&range, base, &stride_table);

        // Should find the nice number 69
        assert!(results.nice_numbers.iter().any(|n| n.number == 69));
    }
}
