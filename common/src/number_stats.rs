//! Expand basic numbers with some redundant stats.

use crate::{NEAR_MISS_CUTOFF_PERCENT, NiceNumber, NiceNumberSimple, SubmissionRecord};

pub const SAVE_TOP_N_NUMBERS: usize = 10_000;

/// Get the near-miss cutoff given a base.
/// Uses the crate-level `NEAR_MISS_CUTOFF_PERCENT`.
#[must_use]
#[allow(
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::cast_possible_truncation
)]
pub fn get_near_miss_cutoff(base: u32) -> u32 {
    (base as f32 * NEAR_MISS_CUTOFF_PERCENT).floor() as u32
}

/// Converts a list of `NiceNumberSimple` to `NiceNumber` by adding
/// some redundant information that's helpful for other tools.
#[must_use]
#[allow(clippy::cast_precision_loss)]
pub fn expand_numbers(numbers: &[NiceNumberSimple], base: u32) -> Vec<NiceNumber> {
    let base_f32 = base as f32;
    numbers
        .iter()
        .map(|n| NiceNumber {
            number: n.number,
            num_uniques: n.num_uniques,
            base,
            niceness: n.num_uniques as f32 / base_f32,
        })
        .collect()
}

/// Incrementally collects the top `SAVE_TOP_N_NUMBERS` numbers over batches of
/// submissions.
///
/// The working set is compacted back to the cap whenever it doubles, so peak
/// memory is bounded by 2x the cap no matter how many batches are folded in.
/// Compaction never drops a number that belongs in the final top-N: the top-N
/// of a set is contained in the top-N of every superset.
#[derive(Default)]
pub struct NumbersAccumulator {
    numbers: Vec<NiceNumber>,
}

impl NumbersAccumulator {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a batch of submissions' numbers.
    pub fn fold(&mut self, submissions: &[SubmissionRecord]) {
        for sub in submissions {
            self.numbers.extend(sub.numbers.iter().cloned());
        }
        if self.numbers.len() > SAVE_TOP_N_NUMBERS * 2 {
            self.compact();
        }
    }

    fn compact(&mut self) {
        // Sort by number of uniques and take the top few
        self.numbers.sort_by(|a, b| b.num_uniques.cmp(&a.num_uniques));
        self.numbers.truncate(SAVE_TOP_N_NUMBERS);
    }

    #[must_use]
    pub fn finalize(mut self) -> Vec<NiceNumber> {
        self.compact();
        self.numbers
    }
}

/// Take a bunch of `SubmissionRecords`, which each have their own `NiceNumbers`, and aggregate
/// them all into a single list. Then filters to the top 10k for a sanity check.
#[must_use]
pub fn downsample_numbers(submissions: &[SubmissionRecord]) -> Vec<NiceNumber> {
    let mut acc = NumbersAccumulator::new();
    acc.fold(submissions);
    acc.finalize()
}

/// Removes some information from a list of `NiceNumbers` to make `NiceNumberSimple`.
#[must_use]
pub fn shrink_numbers(numbers: &[NiceNumber]) -> Vec<NiceNumberSimple> {
    numbers
        .iter()
        .map(|n| NiceNumberSimple {
            number: n.number,
            num_uniques: n.num_uniques,
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Folding batches through the accumulator, including across its internal
    /// compaction, must select the same top-N as one sorted pass over
    /// everything.
    #[test]
    fn numbers_accumulator_matches_single_pass() {
        use crate::{SearchMode, SubmissionRecord};
        use chrono::Utc;

        // Enough numbers to force several compactions (threshold is
        // 2 * SAVE_TOP_N_NUMBERS), with distinct num_uniques so the expected
        // top-N is unambiguous.
        let total = SAVE_TOP_N_NUMBERS * 5;
        let make_sub = |ids: std::ops::Range<usize>| SubmissionRecord {
            submission_id: 1,
            claim_id: 1,
            field_id: 1,
            search_mode: SearchMode::Detailed,
            submit_time: Utc::now(),
            elapsed_secs: 1.0,
            username: "test".to_string(),
            user_ip: "test".to_string(),
            client_version: "test".to_string(),
            disqualified: false,
            distribution: None,
            numbers: ids
                .map(|i| NiceNumber {
                    number: i as u128,
                    num_uniques: i as u32,
                    base: 40,
                    niceness: 1.0,
                })
                .collect(),
        };
        // Interleave so high values arrive across different batches.
        let subs: Vec<SubmissionRecord> = (0..5)
            .map(|batch| make_sub(batch * total / 5..(batch + 1) * total / 5))
            .collect();

        let single = downsample_numbers(&subs);

        let mut acc = NumbersAccumulator::new();
        for sub in &subs {
            acc.fold(std::slice::from_ref(sub));
        }
        let folded = acc.finalize();

        assert_eq!(single.len(), SAVE_TOP_N_NUMBERS);
        assert_eq!(folded.len(), SAVE_TOP_N_NUMBERS);
        // num_uniques are all distinct here, so the selected sets must match
        // exactly regardless of tie-breaking.
        let key = |v: &[NiceNumber]| {
            let mut k: Vec<u32> = v.iter().map(|n| n.num_uniques).collect();
            k.sort_unstable();
            k
        };
        assert_eq!(key(&single), key(&folded));
        assert_eq!(
            key(&folded).first().copied(),
            Some((total - SAVE_TOP_N_NUMBERS) as u32)
        );
    }
    use crate::SearchMode;
    use chrono::Utc;

    fn create_test_numbers_simple() -> Vec<NiceNumberSimple> {
        vec![
            NiceNumberSimple {
                number: 123,
                num_uniques: 3,
            },
            NiceNumberSimple {
                number: 456,
                num_uniques: 5,
            },
            NiceNumberSimple {
                number: 789,
                num_uniques: 7,
            },
        ]
    }

    fn create_test_submissions() -> Vec<SubmissionRecord> {
        let numbers1 = vec![
            NiceNumber {
                number: 123,
                num_uniques: 3,
                base: 10,
                niceness: 0.3,
            },
            NiceNumber {
                number: 456,
                num_uniques: 5,
                base: 10,
                niceness: 0.5,
            },
        ];

        let numbers2 = vec![
            NiceNumber {
                number: 789,
                num_uniques: 7,
                base: 10,
                niceness: 0.7,
            },
            NiceNumber {
                number: 999,
                num_uniques: 2,
                base: 10,
                niceness: 0.2,
            },
        ];

        vec![
            SubmissionRecord {
                submission_id: 1,
                claim_id: 1,
                field_id: 1,
                search_mode: SearchMode::Detailed,
                submit_time: Utc::now(),
                elapsed_secs: 10.0,
                username: "test1".to_string(),
                user_ip: "127.0.0.1".to_string(),
                client_version: "1.0.0".to_string(),
                disqualified: false,
                distribution: None,
                numbers: numbers1,
            },
            SubmissionRecord {
                submission_id: 2,
                claim_id: 2,
                field_id: 1,
                search_mode: SearchMode::Detailed,
                submit_time: Utc::now(),
                elapsed_secs: 15.0,
                username: "test2".to_string(),
                user_ip: "127.0.0.1".to_string(),
                client_version: "1.0.0".to_string(),
                disqualified: false,
                distribution: None,
                numbers: numbers2,
            },
        ]
    }

    #[test_log::test]
    #[allow(clippy::float_cmp)]
    fn test_expand_numbers() {
        let simple_numbers = create_test_numbers_simple();
        let base = 10;
        let expanded = expand_numbers(&simple_numbers, base);

        assert_eq!(expanded.len(), 3);

        // Check first number
        assert_eq!(expanded[0].number, 123);
        assert_eq!(expanded[0].num_uniques, 3);
        assert_eq!(expanded[0].base, 10);
        assert_eq!(expanded[0].niceness, 0.3); // 3/10

        // Check second number
        assert_eq!(expanded[1].number, 456);
        assert_eq!(expanded[1].num_uniques, 5);
        assert_eq!(expanded[1].base, 10);
        assert_eq!(expanded[1].niceness, 0.5); // 5/10

        // Check third number
        assert_eq!(expanded[2].number, 789);
        assert_eq!(expanded[2].num_uniques, 7);
        assert_eq!(expanded[2].base, 10);
        assert_eq!(expanded[2].niceness, 0.7); // 7/10
    }

    #[test_log::test]
    #[allow(clippy::float_cmp)]
    fn test_expand_numbers_different_bases() {
        let numbers = vec![NiceNumberSimple {
            number: 100,
            num_uniques: 5,
        }];

        let expanded_base_5 = expand_numbers(&numbers, 5);
        assert_eq!(expanded_base_5[0].niceness, 1.0); // 5/5

        let expanded_base_20 = expand_numbers(&numbers, 20);
        assert_eq!(expanded_base_20[0].niceness, 0.25); // 5/20
    }

    #[test_log::test]
    fn test_expand_numbers_empty() {
        let empty_numbers = vec![];
        let base = 10;
        let expanded = expand_numbers(&empty_numbers, base);

        assert_eq!(expanded.len(), 0);
    }

    #[test_log::test]
    fn test_downsample_numbers() {
        let submissions = create_test_submissions();
        let result = downsample_numbers(&submissions);

        // Should collect all numbers from both submissions
        assert_eq!(result.len(), 4);

        // Numbers should be sorted by number value by descending num_uniques
        assert!(result[0].num_uniques >= result[1].num_uniques);
        assert!(result[1].num_uniques >= result[2].num_uniques);
        assert!(result[2].num_uniques >= result[3].num_uniques);

        // Check that all numbers are present
        let numbers: Vec<u128> = result.iter().map(|n| n.number).collect();
        assert!(numbers.contains(&123));
        assert!(numbers.contains(&456));
        assert!(numbers.contains(&789));
        assert!(numbers.contains(&999));
    }

    #[test_log::test]
    fn test_downsample_numbers_large_set() {
        // Create submissions with more than SAVE_TOP_N_NUMBERS
        let mut large_numbers = Vec::new();
        for i in 1..=(SAVE_TOP_N_NUMBERS + 100) {
            large_numbers.push(NiceNumber {
                number: i as u128,
                num_uniques: 3,
                base: 10,
                niceness: 0.3,
            });
        }

        // Add one more that's nicer than the rest
        let nicest_number = (SAVE_TOP_N_NUMBERS + 101) as u128;
        large_numbers.push(NiceNumber {
            number: nicest_number,
            num_uniques: 9,
            base: 10,
            niceness: 0.9,
        });

        let submission = SubmissionRecord {
            submission_id: 1,
            claim_id: 1,
            field_id: 1,
            search_mode: SearchMode::Detailed,
            submit_time: Utc::now(),
            elapsed_secs: 10.0,
            username: "test".to_string(),
            user_ip: "127.0.0.1".to_string(),
            client_version: "1.0.0".to_string(),
            disqualified: false,
            distribution: None,
            numbers: large_numbers,
        };

        let result = downsample_numbers(&[submission]);

        // Should only keep SAVE_TOP_N_NUMBERS
        assert_eq!(result.len(), SAVE_TOP_N_NUMBERS);

        // Nicest saved number is first (sorted descending by num_uniques)
        assert_eq!(result[0].number, nicest_number);
    }

    #[test_log::test]
    fn test_downsample_numbers_empty_submissions() {
        let empty_submissions = vec![];
        let result = downsample_numbers(&empty_submissions);

        assert_eq!(result.len(), 0);
    }

    #[test_log::test]
    fn test_shrink_numbers() {
        let numbers = vec![
            NiceNumber {
                number: 123,
                num_uniques: 3,
                base: 10,
                niceness: 0.3,
            },
            NiceNumber {
                number: 456,
                num_uniques: 5,
                base: 10,
                niceness: 0.5,
            },
        ];

        let shrunk = shrink_numbers(&numbers);

        assert_eq!(shrunk.len(), 2);
        assert_eq!(shrunk[0].number, 123);
        assert_eq!(shrunk[0].num_uniques, 3);
        assert_eq!(shrunk[1].number, 456);
        assert_eq!(shrunk[1].num_uniques, 5);
    }

    #[test_log::test]
    fn test_expand_shrink_roundtrip() {
        let original = create_test_numbers_simple();
        let base = 10;
        let expanded = expand_numbers(&original, base);
        let shrunk = shrink_numbers(&expanded);

        assert_eq!(original, shrunk);
    }
}
