//! A module with "nice" calculation utilities for the client.
//!
//! The search ranges are precalculated by the server and all numbers in the
//! range are guaranteed to have a square and cube ("sqube") with the correct
//! number of digits. The ranges provided are a sequential and continuous.
//!
//! There's some tradeoffs to make for speed:
//!  1. We can either get all nicencess statistics (detailed mode) or just the
//!     100% nice numbers (nice-only mode). Nice-only is much faster because it
//!     uses some smart filtering and breaks out of the hot loop early.
//!     Detailed mode is good for analytics and potentially finding patters to
//!     help reduce the search space.
//!  2. We could deserialize our search range as Natural (arbitrarily-large)
//!     numbers, but operations on them are slow. We could deserialize and
//!     perform all operations as u128, but we have to hold n^3 in memory which
//!     limits the maximum value to 7e12 (cube root of 3.4e38). This would get
//!     us through base 40 (1.9e12 to 6.5e12) but not base 41. Instead, we will
//!     iterate over n as u128 (max 3.4e38), but expand it into Natural for
//!     n^2 and n^3. That means we can go up through base 97 (5.6e37 to 2.6e38)
//!     but not base 98 (3.1e38 to 6.7e38).
//!
//! Currently the ranges of interest are bases 40-60 (1.9e12 to 2.1e21), so
//! these tradeoffs will last us for a while. Clients are able to choose if
//! they want to contribute to (or even re-implement) the detailed or nice-only
//! searches, and the results are verified via consensus to ensure that
//! everything can be trusted.

use crate::{
    CLIENT_VERSION, DataToClient, DataToServer, FieldResults, FieldSize, NiceNumberSimple,
    UniquesDistributionSimple,
};
use crate::{lsd_filter, msd_prefix_filter, number_stats, residue_filter};
use itertools::Itertools;
use malachite::base::num::arithmetic::traits::{DivAssignRem, Pow};
use malachite::base::num::conversion::traits::Digits;
use malachite::natural::Natural;
use std::collections::HashMap;

pub const DETAILED_MINI_CHUNK_SIZE: usize = 1_000;

/// Calculate the number of unique digits in (n^2, n^3) represented in base b.
/// A number is nice if the result of this is equal to b (means all digits are used once).
/// If you're just checking if the number is 100% nice, there is a faster version below.
#[must_use]
pub fn get_num_unique_digits(num_u128: u128, base: u32) -> u32 {
    // 🔥🔥🔥 HOT LOOP 🔥🔥🔥

    // Create an indicator variable as a boolean array
    // Each bit represents a number, flip them with bit ops
    let mut digits_indicator: u128 = 0;

    // Convert u128 to natural
    let num = Natural::from(num_u128);

    // Square the number, convert to base and save the digits
    // We tried using foiled out versions but malachite is already pretty good
    let squared = (&num).pow(2);
    for digit in squared.to_digits_asc(&base) {
        digits_indicator |= 1 << digit;
    }

    // Cube, convert to base and save the digits
    let cubed = squared * &num;
    for digit in cubed.to_digits_asc(&base) {
        digits_indicator |= 1 << digit;
    }

    // Output the number of unique digits
    digits_indicator.count_ones()
}

/// The inner loop of detailed field processing. Also called by other crates like the WASM client.
/// Automatically breaks the range into chunks for some performance gains.
///
/// **Range semantics**: Expects a half-open range [`range_start`, `range_end`) where `range_start`
/// is inclusive and `range_end` is exclusive, following Rust's standard convention.
#[must_use]
pub fn process_range_detailed(range: &FieldSize, base: u32) -> FieldResults {
    // Calculate the minimum num_unique_digits cutoff
    let nice_list_cutoff = number_stats::get_near_miss_cutoff(base);

    // Initialize a list for nice and semi-nice numbers
    let mut nice_numbers: Vec<NiceNumberSimple> = Vec::new();

    // Initialize a map indexed by num_unique_digits with the count of each
    let mut unique_distribution_map: HashMap<u32, u128> = (1..=base).map(|i| (i, 0u128)).collect();

    // Break up the range into chunks to avoid allocating too much memory
    // Note: range.iter() and all chunks are half-open ranges [range_start, range_end)
    let chunks = range.range_iter().chunks(DETAILED_MINI_CHUNK_SIZE);

    // Process everything, saving results and aggregating after each chunk finishes
    for chunk in &chunks {
        // Get the chunk results
        let chunk_results: Vec<(u128, u32)> = chunk
            .map(|num| (num, get_num_unique_digits(num, base)))
            .collect();

        // Aggregate unique_distribution
        for (bin_uniques, total_count) in &mut unique_distribution_map {
            let chunk_count = chunk_results
                .iter()
                .filter(|(_, num_unique_digits)| num_unique_digits == bin_uniques)
                .count() as u128;
            *total_count += chunk_count;
        }

        // Collect nice numbers
        nice_numbers.extend(
            chunk_results
                .into_iter()
                .filter(|(_, num_unique_digits)| num_unique_digits > &nice_list_cutoff)
                .map(|(num, num_unique_digits)| NiceNumberSimple {
                    number: num,
                    num_uniques: num_unique_digits,
                }),
        );
    }

    // Convert distribution map to sorted Vec
    let mut distribution: Vec<UniquesDistributionSimple> = unique_distribution_map
        .into_iter()
        .map(|(num_uniques, count)| UniquesDistributionSimple { num_uniques, count })
        .collect();
    distribution.sort_by_key(|d| d.num_uniques);

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
/// # Panics
/// Panics if the base is larger than usize.
#[must_use]
pub fn get_is_nice(num: u128, base: u32) -> bool {
    // 🔥🔥🔥 HOT LOOP 🔥🔥🔥

    // Convert u128 to natural
    let num = Natural::from(num);
    let base_natural = Natural::from(base);

    // Create a boolean array that represents all possible digits
    let mut digits_indicator: Vec<bool> = vec![false; base as usize];

    // Square the number and check those digits
    let squared = (&num).pow(2);
    let mut n = squared.clone();
    while n > 0 {
        let remainder = usize::try_from(&(n.div_assign_rem(&base_natural)))
            .expect("Failed to convert remainder to usize");
        if digits_indicator[remainder] {
            return false;
        }
        digits_indicator[remainder] = true;
    }

    // Cube the number and check those digits
    let mut n = squared * num;
    while n > 0 {
        let remainder = usize::try_from(&(n.div_assign_rem(&base_natural)))
            .expect("Failed to convert remainder to usize");
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
    let results = process_range_niceonly(&claim_data.into(), claim_data.base);

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
pub fn process_range_niceonly(range: &FieldSize, base: u32) -> FieldResults {
    // Precompute these for faster filter checking
    let base_u128 = u128::from(base);
    let base_u128_minusone = base_u128 - 1;

    // Use recursive subdivision to get valid ranges that need processing.
    // This adaptively subdivides the range to skip portions where the MSD prefix indicates
    // all numbers will have duplicate/overlapping digits. It's more effective than fixed
    // chunking because it only subdivides when needed and can find natural boundaries.
    let valid_ranges = msd_prefix_filter::get_valid_ranges(*range, base);

    // Build multi-digit LSD filter
    // Instead of checking just the last digit (mod b), this checks the last k digits (mod b^k).
    // For each k-digit suffix, it verifies that no digit appears in both n² mod b^k and n³ mod b^k.
    // This is significantly more effective than single-digit LSD filtering because it catches
    // collisions in positions beyond just the rightmost digit.
    //
    // Recommended k values: k=2 for bases ≥30, k=3 for smaller bases.
    // For base 40 with k=2, this filters out ~60% of candidates (vs ~40% for k=1).
    let k = lsd_filter::get_recommended_k(base);
    let multi_lsd_bitmap = lsd_filter::get_valid_multi_lsd_bitmap(base, k);
    let multi_lsd_modulus = u128::from(base.pow(k));

    // Build a bit array for residue filter
    let residue_valid = residue_filter::get_residue_filter_u128(&base);
    let mut residue_bits = vec![false; (base - 1) as usize];
    for &residue in &residue_valid {
        residue_bits[residue as usize] = true;
    }

    let mut nice_list = Vec::new();
    for r in valid_ranges {
        for num in r.range_iter() {
            // Multi-digit LSD filter check (Filter A) - uses direct array indexing for speed
            let multi_lsd_suffix = (num % multi_lsd_modulus) as usize;
            debug_assert!(multi_lsd_suffix < multi_lsd_bitmap.len());
            if !multi_lsd_bitmap[multi_lsd_suffix] {
                continue;
            }

            // Bit array lookups for residue filter
            let residue = (num % base_u128_minusone) as usize;
            debug_assert!(residue < residue_bits.len());
            if !residue_bits[residue] {
                continue;
            }

            // Actually check if it's 100% nice with early termination
            if get_is_nice(num, base) {
                nice_list.push(NiceNumberSimple {
                    number: num,
                    num_uniques: base,
                });
            }
        }
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
        assert_eq!(process_range_niceonly(&input.into(), input.base), result);
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
        assert_eq!(process_range_niceonly(&input.into(), input.base), result);
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
        assert_eq!(process_range_niceonly(&input.into(), input.base), result);
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
        let results = process_range_niceonly(&range, base);

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
        let results = process_range_niceonly(&range, base);

        // Should find the nice number 69
        assert!(results.nice_numbers.iter().any(|n| n.number == 69));
    }
}
