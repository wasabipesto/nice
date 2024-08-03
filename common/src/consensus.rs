//! Establish field consensus.

use super::*;

/// Given a field and submissions, determine if there is a consensus.
/// If so, update the canon submission ID and field check level.
pub fn evaluate_consensus(
    field: &FieldRecord,
    submissions: &Vec<SubmissionRecord>,
) -> Result<(Option<SubmissionRecord>, u8), String> {
    // If there are no submissions, reset the canon submission and cap the check level
    if submissions.is_empty() {
        return Ok((None, field.check_level.min(1)));
    }
    // If there is one submission, return it
    if submissions.len() == 1 {
        if let Some(sub) = submissions.first() {
            return Ok((Some(sub.clone()), 2));
        }
    }

    // Group submissions by distribution and numbers
    let mut submission_groups: HashMap<SubmissionCandidate, Vec<SubmissionRecord>> = HashMap::new();
    for sub in submissions {
        let sub_distribution = sub.distribution.clone().ok_or_else(|| {
            format!(
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
            format!(
                "Could not get majority group from submission_groups: {:?}.",
                submission_groups
            )
        })?
        .clone();

    // Get the first submission inside the agreeing group
    let first_submission = majority_group
        .iter()
        .min_by_key(|sub| sub.submit_time)
        .ok_or_else(|| format!("No submission in majority_group: {:?}.", majority_group))?;

    // Determine the check level
    let check_level = (majority_group.len().min(u8::MAX as usize) + 1) as u8;

    Ok((Some(first_submission.clone()), check_level))
}
