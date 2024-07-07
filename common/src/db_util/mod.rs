//! Interfaces between the application code and database.

use super::*;

mod field;

/// Return the lowest field that has not been claimed recently and log the claim.
pub fn claim_next_field() -> Result<(), String> {
    unimplemented!();
}

/// Return a random field that has not been claimed recently and log the claim.
pub fn claim_random_field() -> Result<(), String> {
    unimplemented!();
}

/// Return a specific claim from the log.
pub fn get_claim() -> Result<(), String> {
    unimplemented!();
}

/// Push a new submission to the database.
/// This is assumed to pass some basic validation but it is not considered canon until the consensus is reached.
pub fn insert_submission() -> Result<(), String> {
    unimplemented!();
}

/// Log that scheduled jobs have started.
pub fn log_scheduled_jobs_started() -> Result<(), String> {
    unimplemented!();
}

/// Log that all scheduled jobs are complete.
pub fn log_scheduled_jobs_complete() -> Result<(), String> {
    unimplemented!();
}

/// Get a list of fields with new submissions.
/// Used for consensus.
pub fn get_recently_submitted_fields() -> Result<(), String> {
    unimplemented!();
}

/// Get a list of random fields with new submissions.
/// Used for consensus.
pub fn get_random_fields() -> Result<(), String> {
    unimplemented!();
}

/// Update a field's check level and canon submission.
pub fn update_field_canon() -> Result<(), String> {
    unimplemented!();
}

/// Get a list of fields with new canon submissions.
/// Used for downsampling.
pub fn get_recently_canonized_fields() -> Result<(), String> {
    unimplemented!();
}

/// Update a chunk's calculated statistics.
pub fn update_chunk_stats() -> Result<(), String> {
    unimplemented!();
}

/// Update a base's calculated statistics.
pub fn update_base_stats() -> Result<(), String> {
    unimplemented!();
}

/// Insert a bunch of fields and chunks for processing.
/// Only called by admin scripts.
pub fn insert_fields_and_chunks() -> Result<(), String> {
    unimplemented!();
}
