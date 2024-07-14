//! Interfaces between the application code and database.

use super::*;

use bigdecimal::{BigDecimal, ToPrimitive};
use diesel::prelude::*;
use diesel::table;
use serde_json::Value;

mod base;
mod chunk;
mod conversions;
mod field;

/// Get a single database connection.
pub fn get_database_connection() -> PgConnection {
    dotenv().ok();
    let database_url = env::var("DATABASE_URL").expect("DATABASE_URL must be set");
    PgConnection::establish(&database_url)
        .unwrap_or_else(|_| panic!("Error connecting to {}", database_url))
}

/// Return the lowest field that has not been claimed recently and log the claim.
/// TODO: Insert claim row
pub fn claim_field(
    conn: &mut PgConnection,
    claim_strategy: FieldClaimStrategy,
    maximum_check_level: u8,
    maximum_size: u128,
) -> Result<FieldToClient, String> {
    // try to find a field, respecting previous claims
    let maximum_timestamp = Utc::now() - TimeDelta::hours(CLAIM_DURATION_HOURS as i64);
    if let Some(claimed_field) = field::try_claim_field(
        conn,
        claim_strategy,
        maximum_timestamp,
        maximum_check_level,
        maximum_size,
    )? {
        return Ok(FieldToClient {
            claim_id: 0,
            base: claimed_field.base,
            range_start: claimed_field.range_start,
            range_end: claimed_field.range_end,
            range_size: claimed_field.range_size,
        });
    }

    // try again, ignoring all previous claims and grabbing randomly
    let maximum_timestamp = Utc::now();
    if let Some(claimed_field) = field::try_claim_field(
        conn,
        FieldClaimStrategy::Random,
        maximum_timestamp,
        maximum_check_level,
        maximum_size,
    )? {
        return Ok(FieldToClient {
            claim_id: 0,
            base: claimed_field.base,
            range_start: claimed_field.range_start,
            range_end: claimed_field.range_end,
            range_size: claimed_field.range_size,
        });
    }

    Err(format!("Could not find any field with maximum check level {maximum_check_level} and maximum size {maximum_size}!"))
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

/// Get a chunk record (range plus cached stats).
pub fn get_chunk(conn: &mut PgConnection, chunk_id: u32) -> Result<ChunkRecord, String> {
    chunk::get_chunk(conn, chunk_id)
}

/// Update a chunk's calculated statistics.
pub fn update_chunk_stats() -> Result<(), String> {
    unimplemented!();
}

/// Get a base record (base range plus cached stats).
pub fn get_base(conn: &mut PgConnection, base: u32) -> Result<BaseRecord, String> {
    base::get_base(conn, base)
}

/// Update a base's calculated statistics.
pub fn update_base_stats() -> Result<(), String> {
    unimplemented!();
}

/// Insert a bunch of fields and chunks for processing.
/// Only called by admin scripts.
pub fn insert_new_base_and_fields(
    conn: &mut PgConnection,
    base: u32,
    base_size: FieldSize,
    field_sizes: Vec<FieldSize>,
    chunk_sizes: Vec<FieldSize>,
) -> Result<(), String> {
    // insert the base row
    base::insert_base(conn, base, base_size)?;

    // insert each field
    // TODO: Bulk insert fields and chunks
    for size in field_sizes {
        field::insert_field(conn, base, size)?;
    }

    // insert each chunk
    for size in chunk_sizes {
        chunk::insert_chunk(conn, base, size)?;
        // TODO: Assign chunk ID to fields
    }

    Ok(())
}
