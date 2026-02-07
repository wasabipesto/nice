//! Establish field consensus.

use crate::{FieldRecord, SubmissionCandidate, SubmissionRecord};
use crate::{distribution_stats, number_stats};
use anyhow::{Result, anyhow};
use std::collections::HashMap;

/// Given a field and submissions, determine if there is a consensus.
/// If so, update the canon submission ID and field check level.
///
/// # Errors
/// Returns an error if there are no submissions or if there is an issue with the distribution.
pub fn evaluate_consensus(
    field: &FieldRecord,
    submissions: &Vec<SubmissionRecord>,
) -> Result<(Option<SubmissionRecord>, u8)> {
    // If there are no submissions, reset the canon submission and cap the check level
    if submissions.is_empty() {
        return Ok((None, field.check_level.min(1)));
    }
    // If there is one submission, return it
    if submissions.len() == 1
        && let Some(sub) = submissions.first()
    {
        return Ok((Some(sub.clone()), 2));
    }

    // Group submissions by distribution and numbers
    let mut submission_groups: HashMap<SubmissionCandidate, Vec<SubmissionRecord>> = HashMap::new();
    for sub in submissions {
        let sub_distribution = sub.distribution.clone().ok_or_else(|| {
            anyhow!(
                "No distribution found in detailed submission #{}",
                sub.submission_id
            )
        })?;
        let mut distribution = distribution_stats::shrink_distribution(&sub_distribution);
        distribution.sort_by_key(|k| k.num_uniques);
        let mut numbers = number_stats::shrink_numbers(&sub.numbers.clone());
        numbers.sort_by_key(|k| k.number);
        let subcan = SubmissionCandidate {
            distribution,
            numbers,
        };
        submission_groups
            .entry(subcan)
            .or_default()
            .push(sub.clone());
    }

    // Find the group with the highest number of submissions
    // Note this does not handle ties, they are resolved effectively at random
    let majority_group = submission_groups
        .values()
        .max_by_key(|v| v.len())
        .ok_or_else(|| {
            anyhow!("Could not get majority group from submission_groups: {submission_groups:?}.")
        })?
        .clone();

    // Get the first submission inside the agreeing group
    let first_submission = majority_group
        .iter()
        .min_by_key(|sub| sub.submit_time)
        .ok_or_else(|| anyhow!("No submission in majority_group: {majority_group:?}."))?;

    // Determine the check level, cap to u8::MAX (255)
    let check_level_raw = majority_group.len() + 1;
    #[allow(clippy::cast_possible_truncation)]
    let check_level = check_level_raw.min(u8::MAX as usize) as u8;

    Ok((Some(first_submission.clone()), check_level))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{NiceNumberSimple, SearchMode, UniquesDistributionSimple};
    use chrono::Utc;

    fn create_test_field() -> FieldRecord {
        FieldRecord {
            field_id: 1,
            base: 10,
            chunk_id: Some(1),
            range_start: 100,
            range_end: 200,
            range_size: 100,
            last_claim_time: None,
            canon_submission_id: None,
            check_level: 1,
            prioritize: false,
        }
    }

    fn create_test_submission(
        submission_id: u128,
        distribution: &[UniquesDistributionSimple],
        numbers: &[NiceNumberSimple],
    ) -> SubmissionRecord {
        let expanded_distribution = if distribution.is_empty() {
            None
        } else {
            Some(distribution_stats::expand_distribution(distribution, 10))
        };
        let expanded_numbers = number_stats::expand_numbers(numbers, 10);

        SubmissionRecord {
            submission_id,
            claim_id: 1,
            field_id: 1,
            search_mode: SearchMode::Detailed,
            submit_time: Utc::now(),
            elapsed_secs: 10.0,
            username: "test".to_string(),
            user_ip: "127.0.0.1".to_string(),
            client_version: "1.0.0".to_string(),
            disqualified: false,
            distribution: expanded_distribution,
            numbers: expanded_numbers,
        }
    }

    #[test_log::test]
    fn test_evaluate_consensus_no_submissions() {
        let field = create_test_field();
        let submissions = vec![];

        let result = evaluate_consensus(&field, &submissions).unwrap();

        assert_eq!(result.0, None);
        assert_eq!(result.1, 1); // min(field.check_level, 1)
    }

    #[test_log::test]
    fn test_evaluate_consensus_single_submission() {
        let field = create_test_field();
        let distribution = vec![
            UniquesDistributionSimple {
                num_uniques: 1,
                count: 100,
            },
            UniquesDistributionSimple {
                num_uniques: 2,
                count: 50,
            },
        ];
        let numbers = vec![NiceNumberSimple {
            number: 123,
            num_uniques: 3,
        }];
        let submission = create_test_submission(1, &distribution, &numbers);
        let submissions = vec![submission.clone()];

        let result = evaluate_consensus(&field, &submissions).unwrap();

        assert_eq!(result.0, Some(submission));
        assert_eq!(result.1, 2);
    }

    #[test_log::test]
    fn test_evaluate_consensus_multiple_same_submissions() {
        let field = create_test_field();
        let distribution = vec![UniquesDistributionSimple {
            num_uniques: 1,
            count: 100,
        }];
        let numbers = vec![NiceNumberSimple {
            number: 123,
            num_uniques: 3,
        }];

        // Create multiple identical submissions
        let submission_1 = create_test_submission(1, &distribution, &numbers);
        let submission_2 = create_test_submission(2, &distribution, &numbers);
        let submission_3 = create_test_submission(3, &distribution, &numbers);
        let submissions = vec![submission_1, submission_2, submission_3];

        let result = evaluate_consensus(&field, &submissions).unwrap();

        // Should return the first (earliest) submission
        assert_eq!(result.0.unwrap().submission_id, 1);
        assert_eq!(result.1, 4); // 3 submissions + 1
    }

    #[test_log::test]
    fn test_evaluate_consensus_different_submissions() {
        let field = create_test_field();

        // First group (2 submissions)
        let distribution_1 = vec![UniquesDistributionSimple {
            num_uniques: 1,
            count: 100,
        }];
        let numbers_1 = vec![NiceNumberSimple {
            number: 123,
            num_uniques: 3,
        }];

        // Second group (1 submission)
        let distribution_2 = vec![UniquesDistributionSimple {
            num_uniques: 2,
            count: 200,
        }];
        let numbers_2 = vec![NiceNumberSimple {
            number: 456,
            num_uniques: 5,
        }];

        let submission_1 = create_test_submission(1, &distribution_1, &numbers_1);
        let submission_2 = create_test_submission(2, &distribution_1, &numbers_1);
        let submission_3 = create_test_submission(3, &distribution_2, &numbers_2);
        let submissions = vec![submission_1, submission_2, submission_3];

        let result = evaluate_consensus(&field, &submissions).unwrap();

        // Should return from the majority group (first group with 2 submissions)
        assert_eq!(result.0.unwrap().submission_id, 1);
        assert_eq!(result.1, 3); // 2 submissions + 1
    }

    #[test_log::test]
    fn test_evaluate_consensus_check_level_capping() {
        let mut field = create_test_field();
        field.check_level = 5;
        let submissions = vec![];

        let result = evaluate_consensus(&field, &submissions).unwrap();

        // Should cap check_level to 1 when no submissions
        assert_eq!(result.1, 1);
    }

    #[test_log::test]
    fn test_evaluate_consensus_large_group() {
        let field = create_test_field();
        let distribution = vec![UniquesDistributionSimple {
            num_uniques: 1,
            count: 100,
        }];
        let numbers = vec![NiceNumberSimple {
            number: 123,
            num_uniques: 3,
        }];

        // Create many identical submissions (more than u8::MAX)
        let mut submissions = Vec::new();
        for i in 1..=300 {
            submissions.push(create_test_submission(i, &distribution, &numbers));
        }

        let result = evaluate_consensus(&field, &submissions).unwrap();

        // group_size is 300, which is capped to 255 to fit in a u8
        assert_eq!(result.1, 255);
    }

    #[test_log::test]
    fn test_evaluate_consensus_check_level_within_bounds() {
        let field = create_test_field();
        let distribution = vec![UniquesDistributionSimple {
            num_uniques: 1,
            count: 100,
        }];
        let numbers = vec![NiceNumberSimple {
            number: 123,
            num_uniques: 3,
        }];

        // Create 254 identical submissions (to stay within u8 bounds)
        let mut submissions = Vec::new();
        for i in 1..=254 {
            submissions.push(create_test_submission(i, &distribution, &numbers));
        }

        let result = evaluate_consensus(&field, &submissions).unwrap();

        // Check level should be 254 + 1 = 255 (u8::MAX)
        assert_eq!(result.1, 255);
    }

    #[test_log::test]
    fn test_evaluate_consensus_earliest_submission_selected() {
        use std::thread;
        use std::time::Duration;

        let field = create_test_field();
        let distribution = vec![UniquesDistributionSimple {
            num_uniques: 1,
            count: 100,
        }];
        let numbers = vec![NiceNumberSimple {
            number: 123,
            num_uniques: 3,
        }];

        let submission_1 = create_test_submission(1, &distribution, &numbers);
        thread::sleep(Duration::from_millis(10));
        let mut submission_2 = create_test_submission(2, &distribution, &numbers);

        // Make sure `submission_2` has a later timestamp
        submission_2.submit_time = Utc::now();

        let submissions = vec![submission_2, submission_1.clone()]; // Note: out of order

        let result = evaluate_consensus(&field, &submissions).unwrap();

        // Should return the earliest submission by time, not by position in vec
        assert_eq!(result.0.unwrap().submission_id, 1);
    }
}
