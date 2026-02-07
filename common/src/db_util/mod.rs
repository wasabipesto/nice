//! Interfaces between the application code and database.

#![allow(
    clippy::wildcard_imports,
    clippy::missing_panics_doc,
    clippy::missing_errors_doc
)]

use super::*;

use bigdecimal::{BigDecimal, FromPrimitive, ToPrimitive};
use diesel::prelude::*;
use diesel::r2d2::{ConnectionManager, Pool, PooledConnection};
use diesel::table;
use dotenvy::dotenv;
use serde_json::Value;

mod bases;
mod chunks;
mod claims;
mod conversions;
mod fields;
mod submissions;

/// A Diesel Postgres connection pool type.
pub type PgPool = Pool<ConnectionManager<PgConnection>>;

/// A Diesel Postgres pooled connection type.
pub type PgPooledConnection = PooledConnection<ConnectionManager<PgConnection>>;

/// Build a database connection pool.
///
/// Reads:
/// - `DATABASE_URL` (required)
/// - `DATABASE_POOL_SIZE` (optional, defaults to 10)
#[must_use]
pub fn get_database_pool() -> PgPool {
    dotenv().ok();
    let database_url = env::var("DATABASE_URL").expect("DATABASE_URL must be set");

    let pool_size: u32 = env::var("DATABASE_POOL_SIZE")
        .ok()
        .and_then(|v| v.parse::<u32>().ok())
        .filter(|&n| n > 0)
        .unwrap_or(10);

    let manager = ConnectionManager::<PgConnection>::new(database_url);
    Pool::builder()
        .max_size(pool_size)
        .build(manager)
        .expect("Error building database connection pool")
}

/// Get a single pooled database connection.
#[must_use]
pub fn get_pooled_database_connection(pool: &PgPool) -> PgPooledConnection {
    pool.get()
        .expect("Error retrieving database connection from pool")
}

/// Get a single database connection (non-pooled).
#[must_use]
pub fn get_database_connection() -> PgConnection {
    dotenv().ok();
    let database_url = env::var("DATABASE_URL").expect("DATABASE_URL must be set");
    PgConnection::establish(&database_url)
        .unwrap_or_else(|_| panic!("Error connecting to {database_url}"))
}

/// Get a base record (base range plus cached stats).
pub fn get_base_by_id(conn: &mut PgConnection, base: u32) -> Result<BaseRecord, String> {
    bases::get_base_by_id(conn, base)
}

/// Get all base records.
pub fn get_all_bases(conn: &mut PgConnection) -> Result<Vec<BaseRecord>, String> {
    bases::get_all_bases(conn)
}

/// Insert a new base.
/// Only called by admin scripts.
pub fn insert_new_base(
    conn: &mut PgConnection,
    base: u32,
    base_size: FieldSize,
) -> Result<BaseRecord, String> {
    bases::insert_base(conn, base, base_size)
}

/// Update a base's calculated statistics.
pub fn update_base_stats(
    conn: &mut PgConnection,
    updated_base: BaseRecord,
) -> Result<BaseRecord, String> {
    bases::update_base(conn, updated_base.base, updated_base)
}

/// Get a chunk record (range plus cached stats).
pub fn get_chunk_by_id(conn: &mut PgConnection, chunk_id: u32) -> Result<ChunkRecord, String> {
    chunks::get_chunk_by_id(conn, chunk_id)
}

/// Get all chunk records in a certain base.
pub fn get_chunks_in_base(conn: &mut PgConnection, base: u32) -> Result<Vec<ChunkRecord>, String> {
    chunks::get_chunks_in_base(conn, base)
}

/// Insert a bunch of new chunks.
/// Only called by admin scripts.
pub fn insert_new_chunks(
    conn: &mut PgConnection,
    base: u32,
    chunk_sizes: &[FieldSize],
) -> Result<(), String> {
    chunks::insert_chunks(conn, base, chunk_sizes)
}

/// Reassign chunk associations for all fields in a certain base.
pub fn reassign_fields_to_chunks(conn: &mut PgConnection, base: u32) -> Result<(), String> {
    chunks::reassign_fields_to_chunks(conn, base)
}

/// Update a chunk's calculated statistics.
pub fn update_chunk_stats(
    conn: &mut PgConnection,
    updated_chunk: ChunkRecord,
) -> Result<ChunkRecord, String> {
    chunks::update_chunk(conn, updated_chunk.chunk_id, updated_chunk)
}

/// Get a field record (range plus cached stats).
pub fn get_field_by_id(conn: &mut PgConnection, field_id: u128) -> Result<FieldRecord, String> {
    fields::get_field_by_id(conn, field_id)
}

/// Get all field records in a particular base.
/// Could take a while!
pub fn get_fields_in_base(conn: &mut PgConnection, base: u32) -> Result<Vec<FieldRecord>, String> {
    fields::get_fields_in_base(conn, base)
}

/// Get all field records in a particular range
pub fn get_fields_in_range(
    conn: &mut PgConnection,
    range_start: u128,
    range_end: u128,
) -> Result<Vec<FieldRecord>, String> {
    fields::get_fields_in_range(conn, range_start, range_end)
}

/// Get all field records in a particular base that have a detailed submission.
pub fn get_fields_in_base_with_detailed_subs(
    conn: &mut PgConnection,
    base: u32,
) -> Result<Vec<FieldRecord>, String> {
    fields::get_fields_in_base_with_detailed_subs(conn, base)
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
    fields::try_claim_field(
        conn,
        claim_strategy,
        maximum_timestamp,
        maximum_check_level,
        maximum_size,
    )
}

/// Bulk claim multiple fields at once for queue pre-filling.
/// This is much more efficient than calling `try_claim_field` repeatedly.
pub fn bulk_claim_fields(
    conn: &mut PgConnection,
    count: usize,
    maximum_timestamp: DateTime<Utc>,
    maximum_check_level: u8,
    maximum_size: u128,
) -> Result<Vec<FieldRecord>, String> {
    fields::bulk_claim_fields(
        conn,
        count,
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
    field_sizes: &[FieldSize],
) -> Result<(), String> {
    fields::insert_fields(conn, base, field_sizes)
}

/// Update a field's check level and canon submission.
pub fn update_field_canon_and_cl(
    conn: &mut PgConnection,
    field_id: u128,
    submission_id: Option<u32>,
    check_level: u8,
) -> Result<(), String> {
    fields::update_field_canon_and_cl(conn, field_id, submission_id, check_level)
}

/// Insert a claim with basic information.
pub fn insert_claim(
    conn: &mut PgConnection,
    search_field: &FieldRecord,
    search_mode: SearchMode,
    user_ip: String,
) -> Result<ClaimRecord, String> {
    claims::insert_claim(conn, search_field.field_id, search_mode, user_ip)
}

/// Return a specific claim from the log.
pub fn get_claim_by_id(conn: &mut PgConnection, claim_id: u128) -> Result<ClaimRecord, String> {
    claims::get_claim_by_id(conn, claim_id)
}

/// Push a new submission to the database.
/// This is assumed to pass some basic validation but it is not considered canon until the consensus is reached.
pub fn insert_submission(
    conn: &mut PgConnection,
    claim_record: &ClaimRecord,
    submit_data: &DataToServer,
    user_ip: String,
    distribution: Option<Vec<UniquesDistribution>>,
    numbers: Vec<NiceNumber>,
) -> Result<SubmissionRecord, String> {
    submissions::insert_submission(
        conn,
        claim_record,
        submit_data,
        user_ip,
        distribution,
        numbers,
    )
}
/// Gets a pesudo-random field with the minimum check level, then grabs the
/// canonical submission to include the results in the response.
pub fn get_validation_field(conn: &mut PgConnection) -> Result<ValidationData, String> {
    fields::get_validation_field(conn)
}

/// Get all submission records for a particular field.
/// Only returns qualified and detailed submissions.
pub fn get_submissions_qualified_detailed_for_field(
    conn: &mut PgConnection,
    field_id: u128,
) -> Result<Vec<SubmissionRecord>, String> {
    submissions::get_submissions_qualified_detailed_for_field(conn, field_id)
}

/// Get the range that has reached the given check level.
pub fn get_count_checked_by_range(
    conn: &mut PgConnection,
    check_level: u8,
    start: u128,
    end: u128,
) -> Result<u128, String> {
    fields::get_count_checked_by_range(conn, check_level, start, end)
}

/// Get the minimum check level for the range.
pub fn get_minimum_cl_by_range(
    conn: &mut PgConnection,
    start: u128,
    end: u128,
) -> Result<u8, String> {
    fields::get_minimum_cl_by_range(conn, start, end)
}

/// Get all canon submissions in a particular range.
pub fn get_canon_submissions_by_range(
    conn: &mut PgConnection,
    start: u128,
    end: u128,
) -> Result<Vec<SubmissionRecord>, String> {
    submissions::get_canon_submissions_by_range(conn, start, end)
}

/// Get statistics for all chunks in a base in a single query.
/// This is much more efficient than querying each chunk individually.
pub fn get_chunk_stats_batch(
    conn: &mut PgConnection,
    base: u32,
) -> Result<Vec<fields::ChunkStats>, String> {
    fields::get_chunk_stats_batch(conn, base)
}

/// Get all canon submissions for a base with their `chunk_ids` in a single query.
/// This is much more efficient than querying each chunk individually.
pub fn get_canon_submissions_with_chunks_by_base(
    conn: &mut PgConnection,
    base: u32,
) -> Result<Vec<(SubmissionRecord, Option<u32>)>, String> {
    submissions::get_canon_submissions_with_chunks_by_base(conn, base)
}
