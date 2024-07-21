//! Establish field consensus.
// TODO: break out establishing consensus from database connections, then move out of db_util

use super::*;

/// Given a field, get all submissions and determine if there is a consensus.
/// If so, update the canon submission ID and field check level.
pub fn update_consensus(
    conn: &mut PgConnection,
    field: &FieldRecord,
) -> Result<Option<SubmissionRecord>, String> {
    // Get all qualified and detailed submissions for the field
    let submissions = db_util::get_submissions_qualified_detailed_for_field(conn, field.field_id)
        .map_err(|e| e.to_string())?;

    // If there are no submissions, reset the canon submission and check level if necessary
    if submissions.is_empty() {
        if field.canon_submission_id.is_some() || field.check_level > 1 {
            let check_level = field.check_level.min(1);
            println!(
                "Field #{} claimed to be checked (Submission #{:?}, CL{}) but no submissions were found, so it was reset to CL{}.", 
                field.field_id, field.canon_submission_id, field.check_level, check_level
            );
            update_field_canon_and_cl(conn, field.field_id, None, check_level)
                .map_err(|e| e.to_string())?;
        }
        return Ok(None);
    }

    // Group submissions by distribution and numbers
    let mut submission_groups: HashMap<SubmissionCandidate, Vec<SubmissionRecord>> = HashMap::new();
    for sub in &submissions {
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
            .or_insert_with(Vec::new)
            .push(sub.clone());
    }

    // Find the group with the highest number of submissions
    let canon_group = submission_groups
        .values()
        .max_by_key(|v| v.len())
        .ok_or_else(|| "Submission grouping empty.".to_string())?
        .clone();

    // Get the first submission inside the agreeing group
    let first_submission = canon_group
        .iter()
        .min_by_key(|sub| sub.submit_time)
        .ok_or_else(|| "No submissions found in the consensus group.".to_string())?;

    // Determine the check level
    let check_level = (canon_group.len().min(u8::MAX as usize) + 1) as u8;

    // Update the field if necessary
    if field.canon_submission_id != Some(first_submission.submission_id as u32)
        || field.check_level != check_level
    {
        update_field_canon_and_cl(
            conn,
            field.field_id,
            Some(first_submission.submission_id as u32),
            check_level,
        )
        .map_err(|e| e.to_string())?;
    }

    Ok(Some(first_submission.clone()))
}
