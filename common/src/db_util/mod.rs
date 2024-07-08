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

// TODO: Connect foreign keys in sql schema
