//! Interfaces between the application code and database.

use super::*;

use bigdecimal::{BigDecimal, ToPrimitive};
use diesel::prelude::*;
use diesel::table;
use serde_json::Value;
use submissions::get_submission_by_id;

mod bases;
mod chunks;
mod claims;
mod conversions;
mod fields;
mod submissions;

/// Get a single database connection.
pub fn get_database_connection() -> PgConnection {
    dotenv().ok();
    let database_url = env::var("DATABASE_URL").expect("DATABASE_URL must be set");
    PgConnection::establish(&database_url)
        .unwrap_or_else(|_| panic!("Error connecting to {}", database_url))
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
    chunk_sizes: Vec<FieldSize>,
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

/// Insert a bunch of new fields.
/// Only called by admin scripts.
pub fn insert_new_fields(
    conn: &mut PgConnection,
    base: u32,
    field_sizes: Vec<FieldSize>,
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
    claim_record: ClaimRecord,
    submit_data: DataToServer,
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

/// Get all submission records for a particular field.
/// Only returns qualified and detailed submissions.
pub fn get_submissions_qualified_detailed_for_field(
    conn: &mut PgConnection,
    field_id: u128,
) -> Result<Vec<SubmissionRecord>, String> {
    submissions::get_submissions_qualified_detailed_for_field(conn, field_id)
}

/// Get the percent of the range that has reached the given check level.
pub fn get_count_checked_by_range(
    conn: &mut PgConnection,
    check_level: u8,
    range: FieldSize,
) -> Result<u128, String> {
    fields::get_count_checked_by_range(conn, check_level, range)
}

/// Get all canon submissions in a particular range.
pub fn get_canon_submissions_by_range(
    conn: &mut PgConnection,
    range: FieldSize,
) -> Result<Vec<SubmissionRecord>, String> {
    submissions::get_canon_submissions_by_range(conn, range)
}

pub fn do_downsampling(conn: &mut PgConnection) {
    // loop through bases
    let bases = get_all_bases(conn).unwrap();
    for base_rec in bases {
        // get some base data
        let base = base_rec.base;
        let _base_start = base_rec.range_start;
        let _base_end = base_rec.range_end;
        println!("Base {} started...", base);

        // create vec for all fields in the base
        let mut base_fields: Vec<FieldRecord> = Vec::new();
        let mut base_submissions: Vec<SubmissionRecord> = Vec::new();
        let mut end = false;

        // loop thorugh chunks in the base
        let chunks = get_chunks_in_base(conn, base).unwrap();
        for chunk in chunks {
            let chunk_start = chunk.range_start;
            let chunk_end = chunk.range_end;

            // get fields and then check how many have canon submissions
            let mut fields = get_fields_in_range(conn, chunk_start, chunk_end).unwrap();
            let searched_fields: Vec<&FieldRecord> = fields
                .iter()
                .filter(|f| f.canon_submission_id.is_some())
                .collect();
            let chunk_range_searched: u128 = searched_fields.iter().map(|f| f.range_size).sum();
            let chunk_percent_searched =
                chunk_range_searched as f32 / (chunk_end - chunk_start) as f32;

            if chunk_range_searched == 0 {
                println!(
                    "Reached end of searched range (chunk #{} empty). Wrapping up.",
                    chunk.chunk_id
                );
                end = true;
                break;
            }

            let mut submissions: Vec<SubmissionRecord> = searched_fields
                .iter()
                .map(|f| {
                    get_submission_by_id(conn, f.canon_submission_id.unwrap() as u128).unwrap()
                })
                .collect();

            // update chunk record
            let mut updated_chunk = chunk.clone();
            updated_chunk.distribution =
                distribution_stats::downsample_distributions(&submissions, base);
            updated_chunk.numbers = number_stats::downsample_numbers(&submissions);
            // TODO: checked_detailed, checked_niceonly, minimum_cl, niceness_mean, niceness_stdev
            update_chunk_stats(conn, updated_chunk).unwrap();
            println!(
                "Base {}, Chunk #{}: {:.0}% searched",
                base,
                chunk.chunk_id,
                chunk_percent_searched * 100f32
            );

            // save fields and submissions
            base_fields.append(&mut fields);
            base_submissions.append(&mut submissions);

            if chunk_percent_searched < DOWNSAMPLE_CUTOFF_PERCENT {
                println!(
                    "Reached end of searched range (searched < {:.0}%). Wrapping up.",
                    DOWNSAMPLE_CUTOFF_PERCENT * 100f32
                );
                end = true;
                break;
            }
        }

        // TODO: get remaining submissions between final chunk and end of base range

        // update base record
        let mut updated_base = base_rec.clone();
        updated_base.distribution =
            distribution_stats::downsample_distributions(&base_submissions, base);
        updated_base.numbers = number_stats::downsample_numbers(&base_submissions);
        // TODO: checked_detailed, checked_niceonly, minimum_cl, niceness_mean, niceness_stdev
        update_base_stats(conn, updated_base).unwrap();
        println!("Base {} complete.", base,);
        println!();

        if end {
            break;
        }
    }
}
