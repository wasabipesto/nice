//! Interfaces between the application code and database.

use super::*;

use bigdecimal::{BigDecimal, ToPrimitive};
use diesel::prelude::*;
use diesel::table;
use serde_json::Value;

mod base;
mod chunk;
mod claim;
mod conversions;
mod field;
mod submission;

/// Get a single database connection.
pub fn get_database_connection() -> PgConnection {
    dotenv().ok();
    let database_url = env::var("DATABASE_URL").expect("DATABASE_URL must be set");
    PgConnection::establish(&database_url)
        .unwrap_or_else(|_| panic!("Error connecting to {}", database_url))
}

/// Get a base record (base range plus cached stats).
pub fn get_base_by_id(conn: &mut PgConnection, base: u32) -> Result<BaseRecord, String> {
    base::get_base_by_id(conn, base)
}

/// Get all base records.
pub fn get_all_bases(conn: &mut PgConnection) -> Result<Vec<BaseRecord>, String> {
    base::get_all_bases(conn)
}

/// Insert a new base.
/// Only called by admin scripts.
pub fn insert_new_base(
    conn: &mut PgConnection,
    base: u32,
    base_size: FieldSize,
) -> Result<BaseRecord, String> {
    base::insert_base(conn, base, base_size)
}

/// Update a base's calculated statistics.
pub fn update_base_stats() -> Result<(), String> {
    unimplemented!();
}

/// Get a chunk record (range plus cached stats).
pub fn get_chunk_by_id(conn: &mut PgConnection, chunk_id: u32) -> Result<ChunkRecord, String> {
    chunk::get_chunk_by_id(conn, chunk_id)
}

/// Get all chunk records in a certain base.
pub fn get_chunks_in_base(conn: &mut PgConnection, base: u32) -> Result<Vec<ChunkRecord>, String> {
    chunk::get_chunks_in_base(conn, base)
}

/// Insert a bunch of new chunks.
/// Only called by admin scripts.
pub fn insert_new_chunks(
    conn: &mut PgConnection,
    base: u32,
    chunk_sizes: Vec<FieldSize>,
) -> Result<(), String> {
    chunk::insert_chunks(conn, base, chunk_sizes)
}

/// Reassign chunk associations for all fields in a certain base.
pub fn reassign_fields_to_chunks(conn: &mut PgConnection, base: u32) -> Result<(), String> {
    chunk::reassign_fields_to_chunks(conn, base)
}

/// Update a chunk's calculated statistics.
pub fn update_chunk_stats() -> Result<(), String> {
    unimplemented!();
}

/// Get a field record (range plus cached stats).
pub fn get_field_by_id(conn: &mut PgConnection, field_id: u128) -> Result<FieldRecord, String> {
    field::get_field_by_id(conn, field_id)
}

/// Get all field records in a particular base.
/// Could take a while!
pub fn get_fields_in_base(conn: &mut PgConnection, base: u32) -> Result<Vec<FieldRecord>, String> {
    field::get_fields_in_base(conn, base)
}

/// Try to claim a valid field.
/// Returns Ok(None) if no matching fields are found.
pub fn try_claim_field(
    conn: &mut PgConnection,
    claim_strategy: FieldClaimStrategy,
    maximum_timestamp: DateTime<Utc>,
    maximum_check_level: u8,
    maximum_size: u128,
) -> Result<Option<FieldRecord>, String> {
    field::try_claim_field(
        conn,
        claim_strategy,
        maximum_timestamp,
        maximum_check_level,
        maximum_size,
    )
}

/// Insert a bunch of new fields.
/// Only called by admin scripts.
pub fn insert_new_fields(
    conn: &mut PgConnection,
    base: u32,
    field_sizes: Vec<FieldSize>,
) -> Result<(), String> {
    field::insert_fields(conn, base, field_sizes)
}

/// Update a field's check level and canon submission.
pub fn update_field_canon_and_cl(
    conn: &mut PgConnection,
    field_id: u128,
    submission_id: Option<u32>,
    check_level: u8,
) -> Result<(), String> {
    field::update_field_canon_and_cl(conn, field_id, submission_id, check_level)
}

/// Insert a claim with basic information.
pub fn insert_claim(
    conn: &mut PgConnection,
    search_field: &FieldRecord,
    search_mode: SearchMode,
    user_ip: String,
) -> Result<ClaimRecord, String> {
    claim::insert_claim(conn, search_field.field_id, search_mode, user_ip)
}

/// Return a specific claim from the log.
pub fn get_claim_by_id(conn: &mut PgConnection, claim_id: u128) -> Result<ClaimRecord, String> {
    claim::get_claim_by_id(conn, claim_id)
}

/// Push a new submission to the database.
/// This is assumed to pass some basic validation but it is not considered canon until the consensus is reached.
pub fn insert_submission(
    conn: &mut PgConnection,
    claim_record: ClaimRecord,
    submit_data: DataToServer,
    user_ip: String,
    distribution: Option<Vec<UniquesDistribution>>,
    numbers: Vec<NiceNumbersExtended>,
) -> Result<SubmissionRecord, String> {
    submission::insert_submission(
        conn,
        claim_record,
        submit_data,
        user_ip,
        distribution,
        numbers,
    )
}

/// Get all submission records for a particular field.
/// Only returns qualified and detailed submissions.
pub fn get_submissions_qualified_detailed_for_field(
    conn: &mut PgConnection,
    field_id: u128,
) -> Result<Vec<SubmissionRecord>, String> {
    submission::get_submissions_qualified_detailed_for_field(conn, field_id)
}

/// Given a field, get all submissions and update the canon submission ID/check level as necessary.
pub fn update_consensus(
    conn: &mut PgConnection,
    field: &FieldRecord,
) -> Result<(Option<SubmissionRecord>, u8), String> {
    // Get all qualified and detailed submissions for the field
    let submissions = db_util::get_submissions_qualified_detailed_for_field(conn, field.field_id)
        .map_err(|e| e.to_string())?;

    // Establish the consensus
    let (canon_submission, check_level) = consensus::evaluate_consensus(field, &submissions)?;
    match &canon_submission {
        None => {
            if field.canon_submission_id.is_some() || field.check_level > 1 {
                println!(
                "Field #{} claimed to be checked (Submission #{:?}, CL{}) but no submissions were found, so it was reset to CL{}.",
                field.field_id, field.canon_submission_id, field.check_level, check_level
            );
                update_field_canon_and_cl(conn, field.field_id, None, check_level)
                    .map_err(|e| e.to_string())?;
            }
        }
        Some(canon_submission_some) => {
            // Update the field if necessary
            if field.canon_submission_id != Some(canon_submission_some.submission_id as u32)
                || field.check_level != check_level
            {
                update_field_canon_and_cl(
                    conn,
                    field.field_id,
                    Some(canon_submission_some.submission_id as u32),
                    check_level,
                )
                .map_err(|e| e.to_string())?;
            }
        }
    }

    Ok((canon_submission, check_level))
}
