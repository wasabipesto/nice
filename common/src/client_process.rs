//! A module with "nice" calculation utilities.
//! We will iterate over n as u128 (max 3.4e38), but expand it into Natural for n^2 and n^3.
//! That means we can go up through base 97 (5.6e37 to 2.6e38) but not base 98 (3.1e38 to 6.7e38).

use super::*;

/// Calculate the number of unique digits in (n^2, n^3) represented in base b.
/// A number is nice if the result of this is equal to b (means all digits are used once).
/// If you're just checking if the number is 100% nice, there is a faster version below.
pub fn get_num_unique_digits(num_u128: u128, base: u32) -> u32 {
    // 🔥🔥🔥 HOT LOOP 🔥🔥🔥

    // create a boolean array that represents all possible digits
    // tested allocating this outside of the loop and it didn't have any effect
    let mut digits_indicator: Vec<bool> = vec![false; base as usize];

    // convert u128 to natural
    let num = Natural::from(num_u128);

    // square the number, convert to base and save the digits
    // tried using foiled out versions but malachite is already pretty good
    let squared = (&num).pow(2);
    for digit in squared.to_digits_asc(&base) {
        digits_indicator[digit as usize] = true;
    }

    // cube, convert to base and save the digits
    let cubed = squared * &num;
    for digit in cubed.to_digits_asc(&base) {
        digits_indicator[digit as usize] = true;
    }

    // output the number of unique digits
    let mut num_unique_digits = 0;
    for digit in digits_indicator {
        if digit {
            num_unique_digits += 1
        }
    }

    num_unique_digits
}

/// Process a field by aggregating statistics on the niceness of numbers in a range.
pub fn process_detailed(claim_data: &DataToClient, username: &String) -> DataToServer {
    // get the basic parameters
    let base = claim_data.base;
    let range_start = claim_data.range_start;
    let range_end = claim_data.range_end;

    // get the minimum cutoff (90% of the base)
    let nice_list_cutoff = (base as f32 * NEAR_MISS_CUTOFF_PERCENT) as u32;

    // init the output maps
    let mut unique_distribution: HashMap<u32, u128> = (1..=base).map(|i| (i, 0u128)).collect();
    let mut nice_numbers: HashMap<u128, u32> = HashMap::new();

    // process the range and collect num_uniques for each item in the range
    (range_start..range_end).for_each(|num| {
        // 🔥🔥🔥 HOT LOOP 🔥🔥🔥

        // get the number of uniques
        let num_unique_digits = get_num_unique_digits(num, base);

        // TODO: Break this up into chunks
        // There's a tradeoff between allocating a bunch of memory upfront and
        // aggregating stats (distribution, top nice numbers) later, or updating
        // the stats as we go. Doing it for every number just wastes cycles but
        // allocating everything at once takes, uh, a bit too much memory.

        // increment the correct bin in the distribution
        *unique_distribution.entry(num_unique_digits).or_insert(0) += 1;

        // save if the number is sufficiently nice
        if num_unique_digits > nice_list_cutoff {
            nice_numbers.insert(num, num_unique_digits);
        }
    });

    let mut submit_distribution: Vec<UniquesDistributionSimple> = unique_distribution
        .into_iter()
        .map(|(num_uniques, count)| UniquesDistributionSimple { num_uniques, count })
        .collect();
    submit_distribution.sort_by_key(|d| d.num_uniques);
    let submit_numbers = nice_numbers
        .into_iter()
        .map(|(number, num_uniques)| NiceNumberSimple {
            number,
            num_uniques,
        })
        .collect();

    DataToServer {
        claim_id: claim_data.claim_id,
        username: username.to_owned(),
        client_version: CLIENT_VERSION.to_string(),
        unique_distribution: Some(submit_distribution),
        nice_numbers: submit_numbers,
    }
}

/// Quickly determine if a number is 100% nice in this base.
/// A number is nice if (n^2, n^3), converted to base b, have all digits of base b.
/// Assumes we have already done residue class filtering.
/// Immediately stops if we hit a duplicate digit.
pub fn get_is_nice(num: u128, base: u32) -> bool {
    // 🔥🔥🔥 HOT LOOP 🔥🔥🔥

    // convert u128 to natural
    let num = Natural::from(num);
    let base_natural = Natural::from(base);

    // create a boolean array that represents all possible digits
    let mut digits_indicator: Vec<bool> = vec![false; base as usize];

    // square the number and check those digits
    let squared = (&num).pow(2);
    let mut n = squared.clone();
    while n > 0 {
        let remainder = usize::try_from(&(n.div_assign_rem(&base_natural))).unwrap();
        if digits_indicator[remainder] {
            return false;
        }
        digits_indicator[remainder] = true;
    }

    // cube the number and check those digits
    let mut n = squared * num;
    while n > 0 {
        let remainder = usize::try_from(&(n.div_assign_rem(&base_natural))).unwrap();
        if digits_indicator[remainder] {
            return false;
        }
        digits_indicator[remainder] = true;
    }
    true
}

/// Process a field by looking for completely nice numbers.
/// Implements several optimizations over the detailed search.
pub fn process_niceonly(claim_data: &DataToClient, username: &String) -> DataToServer {
    let base = claim_data.base;
    let range_start = claim_data.range_start;
    let range_end = claim_data.range_end;
    let residue_filter = residue_filter::get_residue_filter(&base);

    let nice_list = (range_start..range_end)
        .filter(|num| residue_filter.contains(&((num % (base as u128 - 1)) as u32)))
        .filter(|num| get_is_nice(*num, base))
        .map(|number| NiceNumberSimple {
            number,
            num_uniques: base,
        })
        .collect();

    DataToServer {
        claim_id: claim_data.claim_id,
        username: username.to_owned(),
        client_version: CLIENT_VERSION.to_string(),
        unique_distribution: None,
        nice_numbers: nice_list,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn process_detailed_b10() {
        let username = "anonymous".to_string();
        let claim_data = DataToClient {
            claim_id: 0,
            base: 10,
            range_start: 47,
            range_end: 100,
            range_size: 53,
        };
        let submit_data = DataToServer {
            claim_id: claim_data.claim_id,
            username: username.clone(),
            client_version: CLIENT_VERSION.to_string(),
            unique_distribution: Some(Vec::from([
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
            ])),
            nice_numbers: Vec::from([NiceNumberSimple {
                number: 69,
                num_uniques: 10,
            }]),
        };
        assert_eq!(process_detailed(&claim_data, &username), submit_data);
    }

    #[test]
    fn process_detailed_b40() {
        let username = "anonymous".to_string();
        let claim_data = DataToClient {
            claim_id: 0,
            base: 40,
            range_start: 916284264916,
            range_end: 916284264916 + 10000,
            range_size: 10000,
        };
        let submit_data = DataToServer {
            claim_id: claim_data.claim_id,
            username: username.clone(),
            client_version: CLIENT_VERSION.to_string(),
            unique_distribution: Some(Vec::from([
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
                    count: 1,
                },
                UniquesDistributionSimple {
                    num_uniques: 19,
                    count: 13,
                },
                UniquesDistributionSimple {
                    num_uniques: 20,
                    count: 40,
                },
                UniquesDistributionSimple {
                    num_uniques: 21,
                    count: 176,
                },
                UniquesDistributionSimple {
                    num_uniques: 22,
                    count: 520,
                },
                UniquesDistributionSimple {
                    num_uniques: 23,
                    count: 1046,
                },
                UniquesDistributionSimple {
                    num_uniques: 24,
                    count: 1710,
                },
                UniquesDistributionSimple {
                    num_uniques: 25,
                    count: 2115,
                },
                UniquesDistributionSimple {
                    num_uniques: 26,
                    count: 1947,
                },
                UniquesDistributionSimple {
                    num_uniques: 27,
                    count: 1322,
                },
                UniquesDistributionSimple {
                    num_uniques: 28,
                    count: 728,
                },
                UniquesDistributionSimple {
                    num_uniques: 29,
                    count: 283,
                },
                UniquesDistributionSimple {
                    num_uniques: 30,
                    count: 83,
                },
                UniquesDistributionSimple {
                    num_uniques: 31,
                    count: 13,
                },
                UniquesDistributionSimple {
                    num_uniques: 32,
                    count: 3,
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
            ])),
            nice_numbers: Vec::new(),
        };
        assert_eq!(process_detailed(&claim_data, &username), submit_data);
    }

    #[test]
    fn process_detailed_b80() {
        let username = "anonymous".to_string();
        let claim_data = DataToClient {
            claim_id: 0,
            base: 80,
            range_start: 653245554420798943087177909799,
            range_end: 653245554420798943087177909799 + 10000,
            range_size: 10000,
        };
        let submit_data = DataToServer {
            claim_id: claim_data.claim_id,
            username: username.clone(),
            client_version: CLIENT_VERSION.to_string(),
            unique_distribution: Some(Vec::from([
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
            ])),
            nice_numbers: Vec::new(),
        };
        assert_eq!(process_detailed(&claim_data, &username), submit_data);
    }

    #[test]
    fn process_niceonly_b10() {
        let username = "anonymous".to_string();
        let claim_data = DataToClient {
            claim_id: 0,
            base: 10,
            range_start: 47,
            range_end: 100,
            range_size: 53,
        };
        let submit_data = DataToServer {
            claim_id: claim_data.claim_id,
            username: username.clone(),
            client_version: CLIENT_VERSION.to_string(),
            unique_distribution: None,
            nice_numbers: Vec::from([NiceNumberSimple {
                number: 69,
                num_uniques: 10,
            }]),
        };
        assert_eq!(process_niceonly(&claim_data, &username), submit_data);
    }

    #[test]
    fn process_niceonly_b40() {
        let username = "anonymous".to_string();
        let claim_data = DataToClient {
            claim_id: 0,
            base: 40,
            range_start: 916284264916,
            range_end: 916284264916 + 10000,
            range_size: 10000,
        };
        let submit_data = DataToServer {
            claim_id: claim_data.claim_id,
            username: username.clone(),
            client_version: CLIENT_VERSION.to_string(),
            unique_distribution: None,
            nice_numbers: Vec::new(),
        };
        assert_eq!(process_niceonly(&claim_data, &username), submit_data);
    }

    #[test]
    fn process_niceonly_b80() {
        let username = "anonymous".to_string();
        let claim_data = DataToClient {
            claim_id: 0,
            base: 80,
            range_start: 653245554420798943087177909799,
            range_end: 653245554420798943087177909799 + 10000,
            range_size: 10000,
        };
        let submit_data = DataToServer {
            claim_id: claim_data.claim_id,
            username: username.clone(),
            client_version: CLIENT_VERSION.to_string(),
            unique_distribution: None,
            nice_numbers: Vec::new(),
        };
        assert_eq!(process_niceonly(&claim_data, &username), submit_data);
    }
}
