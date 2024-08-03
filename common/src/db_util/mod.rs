//! Interfaces between the application code and database.

use super::*;

use bigdecimal::{BigDecimal, ToPrimitive};
use diesel::prelude::*;
use diesel::table;
use serde_json::Value;

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

pub fn do_downsampling(conn: &mut PgConnection) {
    // loop through bases
    let bases = get_all_bases(conn).unwrap();
    for base_rec in bases {
        // get some base data
        let base = base_rec.base;
        let base_size = base_rec.range_size;
        print!("Base {}: ", base);

        // get basic stats like how much has been cheked
        let base_checked_niceonly =
            get_count_checked_by_range(conn, 1, base_rec.range_start, base_rec.range_end).unwrap();
        let base_checked_detailed =
            get_count_checked_by_range(conn, 2, base_rec.range_start, base_rec.range_end).unwrap();
        let base_percent_checked_detailed = base_checked_detailed as f32 / base_size as f32;
        let base_minimum_cl =
            get_minimum_cl_by_range(conn, base_rec.range_start, base_rec.range_end).unwrap();
        println!(
            "CL{}, Checked {:.0}%",
            base_minimum_cl,
            base_percent_checked_detailed * 100f32
        );

        // create vec for all fields in the base
        let mut base_submissions: Vec<SubmissionRecord> = Vec::new();

        // loop thorugh chunks in the base
        let chunks = get_chunks_in_base(conn, base).unwrap();
        for chunk in chunks {
            let chunk_size = chunk.range_size;
            print!("Chunk #{}: ", chunk.chunk_id);

            // get basic stats like how much has been cheked
            let checked_niceonly =
                get_count_checked_by_range(conn, 1, chunk.range_start, chunk.range_end).unwrap();
            let checked_detailed =
                get_count_checked_by_range(conn, 2, chunk.range_start, chunk.range_end).unwrap();
            let chunk_percent_checked_detailed = checked_detailed as f32 / chunk_size as f32;
            let minimum_cl =
                get_minimum_cl_by_range(conn, chunk.range_start, chunk.range_end).unwrap();
            print!(
                "CL{}, Checked {:.0}%, ",
                minimum_cl,
                chunk_percent_checked_detailed * 100f32
            );

            // get all submissions for the chunk
            let mut submissions: Vec<SubmissionRecord> =
                get_canon_submissions_by_range(conn, chunk.range_start, chunk.range_end).unwrap();

            // update chunk record
            let mut updated_chunk = chunk.clone();
            updated_chunk.checked_niceonly = checked_niceonly;
            updated_chunk.checked_detailed = checked_detailed;
            updated_chunk.minimum_cl = minimum_cl;
            if chunk_percent_checked_detailed > DOWNSAMPLE_CUTOFF_PERCENT {
                // only update these detailed stats if we have a representative sample
                updated_chunk.distribution =
                    distribution_stats::downsample_distributions(&submissions, base);
                updated_chunk.numbers = number_stats::downsample_numbers(&submissions);
                // TODO: niceness_mean
                // TODO: niceness_stdev
                // print!("Mean {:.2}, StDev {:.2}, ", niceness_mean, niceness_stdev);
            } else {
                // otherwise reset to "no data" default
                updated_chunk.distribution = Vec::new();
                updated_chunk.numbers = Vec::new();
                updated_chunk.niceness_mean = None;
                updated_chunk.niceness_stdev = None;
            }

            // save it
            update_chunk_stats(conn, updated_chunk).unwrap();
            println!("Updated!");

            // save submissions for the base stats
            base_submissions.append(&mut submissions);
        }

        // TODO: get remaining submissions between final chunk and end of base range

        // update base record
        let mut updated_base = base_rec.clone();
        updated_base.checked_niceonly = base_checked_niceonly;
        updated_base.checked_detailed = base_checked_detailed;
        updated_base.minimum_cl = base_minimum_cl;
        if base_percent_checked_detailed > DOWNSAMPLE_CUTOFF_PERCENT {
            // only update these detailed stats if we have a representative sample
            updated_base.distribution =
                distribution_stats::downsample_distributions(&base_submissions, base);
            updated_base.numbers = number_stats::downsample_numbers(&base_submissions);
            // TODO: niceness_mean
            // TODO: niceness_stdev
            // print!("Mean {:.2}, StDev {:.2}, ", niceness_mean, niceness_stdev);
        } else {
            // otherwise reset to "no data" default
            updated_base.distribution = Vec::new();
            updated_base.numbers = Vec::new();
            updated_base.niceness_mean = None;
            updated_base.niceness_stdev = None;
        }

        // save it
        update_base_stats(conn, updated_base).unwrap();
        println!("Base {} complete.", base,);
        println!();
    }
}
