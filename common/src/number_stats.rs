//! Expand basic numbers with some redundant stats.

use super::*;

/// Converts a list of NiceNumberSimple to NiceNumber by adding
/// some redundant information that's helpful for other tools.
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

/// Take a bunch of SubmissionRecords, which each have their own NiceNumbers, and aggregate
/// them all into a single list. Then filters to the top 10k for a sanity check.
pub fn downsample_numbers(submissions: &[SubmissionRecord]) -> Vec<NiceNumber> {
    // collate all numbers
    let mut all_numbers = submissions.iter().fold(Vec::new(), |mut acc, sub| {
        acc.extend(sub.numbers.iter().cloned());
        acc
    });

    // sort and take the top few
    all_numbers.sort_by(|a, b| b.number.cmp(&a.number));
    all_numbers
        .iter()
        .take(SAVE_TOP_N_NUMBERS)
        .cloned()
        .collect()
}

/// Removes some information from a list of NiceNumbers to make NiceNumberSimple.
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

    #[test]
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

    #[test]
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

    #[test]
    fn test_expand_numbers_empty() {
        let empty_numbers = vec![];
        let base = 10;
        let expanded = expand_numbers(&empty_numbers, base);

        assert_eq!(expanded.len(), 0);
    }

    #[test]
    fn test_downsample_numbers() {
        let submissions = create_test_submissions();
        let result = downsample_numbers(&submissions);

        // Should collect all numbers from both submissions
        assert_eq!(result.len(), 4);

        // Numbers should be sorted by number value in descending order
        assert!(result[0].number >= result[1].number);
        assert!(result[1].number >= result[2].number);
        assert!(result[2].number >= result[3].number);

        // Check that all numbers are present
        let numbers: Vec<u128> = result.iter().map(|n| n.number).collect();
        assert!(numbers.contains(&123));
        assert!(numbers.contains(&456));
        assert!(numbers.contains(&789));
        assert!(numbers.contains(&999));
    }

    #[test]
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

        // Should be the highest numbers (sorted descending)
        assert_eq!(result[0].number, (SAVE_TOP_N_NUMBERS + 100) as u128);
        assert_eq!(result[SAVE_TOP_N_NUMBERS - 1].number, 101);
    }

    #[test]
    fn test_downsample_numbers_empty_submissions() {
        let empty_submissions = vec![];
        let result = downsample_numbers(&empty_submissions);

        assert_eq!(result.len(), 0);
    }

    #[test]
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

    #[test]
    fn test_expand_shrink_roundtrip() {
        let original = create_test_numbers_simple();
        let base = 10;
        let expanded = expand_numbers(&original, base);
        let shrunk = shrink_numbers(&expanded);

        assert_eq!(original, shrunk);
    }
}
