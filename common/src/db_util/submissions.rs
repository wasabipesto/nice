#![allow(dead_code)]

use super::*;

table! {
    submissions (id) {
        id -> BigInt,
        claim_id -> Integer,
        field_id -> Integer,
        search_mode -> Varchar,
        submit_time -> Timestamptz,
        elapsed_secs -> Float,
        username -> Varchar,
        user_ip -> Varchar,
        client_version -> Varchar,
        disqualified -> Bool,
        distribution -> Nullable<Jsonb>,
        numbers -> Jsonb,
        telemetry -> Nullable<Jsonb>,
    }
}

#[derive(Queryable, QueryableByName)]
#[diesel(table_name = submissions)]
struct SubmissionPrivate {
    id: i64,
    claim_id: i32,
    field_id: i32,
    search_mode: String,
    submit_time: DateTime<Utc>,
    elapsed_secs: f32,
    username: String,
    user_ip: String,
    client_version: String,
    disqualified: bool,
    distribution: Option<Value>,
    numbers: Value,
    telemetry: Option<Value>,
}

#[derive(Insertable)]
#[diesel(table_name = submissions)]
struct SubmissionPrivateNew {
    claim_id: i32,
    field_id: i32,
    search_mode: String,
    elapsed_secs: f32,
    username: String,
    user_ip: String,
    client_version: String,
    distribution: Option<Value>,
    numbers: Value,
    telemetry: Option<Value>,
}

fn private_to_public(p: SubmissionPrivate) -> Result<SubmissionRecord> {
    use conversions::*;
    Ok(SubmissionRecord {
        submission_id: i64_to_u128(p.id)?,
        claim_id: i64_to_u128(p.id)?,
        field_id: i32_to_u128(p.field_id)?,
        search_mode: deserialize_searchmode(p.search_mode)?,
        submit_time: p.submit_time,
        elapsed_secs: p.elapsed_secs,
        username: p.username,
        user_ip: p.user_ip,
        client_version: p.client_version,
        disqualified: p.disqualified,
        distribution: deserialize_opt_distribution(p.distribution)?,
        numbers: deserialize_numbers(p.numbers)?,
    })
}

fn public_to_private(p: SubmissionRecord) -> Result<SubmissionPrivate> {
    use conversions::*;
    Ok(SubmissionPrivate {
        id: u128_to_i64(p.submission_id)?,
        claim_id: u128_to_i32(p.claim_id)?,
        field_id: u128_to_i32(p.field_id)?,
        search_mode: serialize_searchmode(p.search_mode),
        submit_time: p.submit_time,
        elapsed_secs: p.elapsed_secs,
        username: p.username,
        user_ip: p.user_ip,
        client_version: p.client_version,
        disqualified: p.disqualified,
        distribution: serialize_opt_distribution(p.distribution)?,
        numbers: serialize_numbers(p.numbers)?,
        telemetry: None,
    })
}

#[allow(clippy::cast_precision_loss)]
fn build_new_row(
    claim_record: &ClaimRecord,
    submit_data: &DataToServer,
    user_ip: String,
    distribution: Option<Vec<UniquesDistribution>>,
    numbers: Vec<NiceNumber>,
) -> Result<SubmissionPrivateNew> {
    use conversions::*;
    Ok(SubmissionPrivateNew {
        claim_id: u128_to_i32(claim_record.claim_id)?,
        field_id: u128_to_i32(claim_record.field_id)?,
        search_mode: serialize_searchmode(claim_record.search_mode),
        elapsed_secs: (Utc::now() - claim_record.claim_time).num_milliseconds() as f32 / 1000f32,
        username: submit_data.username.clone(),
        user_ip,
        client_version: submit_data.client_version.clone(),
        distribution: serialize_opt_distribution(distribution)?,
        numbers: serialize_numbers(numbers)?,
        telemetry: submit_data.telemetry.clone(),
    })
}

pub fn insert_submission(
    conn: &mut PgConnection,
    claim_record: &ClaimRecord,
    submit_data: &DataToServer,
    input_user_ip: String,
    input_distribution: Option<Vec<UniquesDistribution>>,
    input_numbers: Vec<NiceNumber>,
) -> Result<SubmissionRecord> {
    use self::submissions::dsl::*;

    let insert_row = build_new_row(
        claim_record,
        submit_data,
        input_user_ip,
        input_distribution,
        input_numbers,
    )?;

    let result = diesel::insert_into(submissions)
        .values(&insert_row)
        .get_result(conn)
        .map_err(|e| anyhow!("{e}"))?;
    private_to_public(result)
}

pub fn get_submission_by_id(conn: &mut PgConnection, row_id: u128) -> Result<SubmissionRecord> {
    use self::submissions::dsl::*;

    let row_id = conversions::u128_to_i64(row_id)?;

    let result = submissions
        .filter(id.eq(row_id))
        .first::<SubmissionPrivate>(conn)
        .map_err(|e| anyhow!("{e}"))?;
    private_to_public(result)
}

pub fn get_canon_submissions_by_range(
    conn: &mut PgConnection,
    start: u128,
    end: u128,
) -> Result<Vec<SubmissionRecord>> {
    use diesel::sql_query;
    use diesel::sql_types::Numeric;

    let start = conversions::u128_to_bigdec(start)?;
    let end = conversions::u128_to_bigdec(end)?;

    let query = "SELECT s.*
        FROM fields f
        JOIN submissions s ON f.canon_submission_id = s.id
        WHERE f.range_start >= $1
        AND f.range_end <= $2;";

    let items_private: Vec<SubmissionPrivate> = sql_query(query)
        .bind::<Numeric, _>(start)
        .bind::<Numeric, _>(end)
        .load(conn)
        .map_err(|e| anyhow!("{e}"))?;

    items_private
        .into_iter()
        .map(private_to_public)
        .collect::<Result<Vec<SubmissionRecord>>>()
}

pub fn get_submissions_qualified_detailed_for_field(
    conn: &mut PgConnection,
    input_field_id: u128,
) -> Result<Vec<SubmissionRecord>> {
    use self::submissions::dsl::*;

    let input_field_id = conversions::u128_to_i32(input_field_id)?;
    let input_search_mode = conversions::serialize_searchmode(SearchMode::Detailed);
    let input_disqualified = false;

    let items_private: Vec<SubmissionPrivate> = submissions
        .filter(field_id.eq(input_field_id))
        .filter(search_mode.eq(input_search_mode))
        .filter(disqualified.eq(input_disqualified))
        .load(conn)
        .map_err(|e| anyhow!("{e}"))?;

    items_private
        .into_iter()
        .map(private_to_public)
        .collect::<Result<Vec<SubmissionRecord>>>()
}

/// Get the canon submissions for every field in one chunk.
///
/// The jobs binary calls this per chunk and folds the results into
/// accumulators, instead of loading a whole base's canon submissions (3.5M+
/// rows for the largest bases, tens of GB once the jsonb columns are decoded)
/// in a single query. One chunk is 1% of a base, so this bounds peak memory
/// to the largest chunk regardless of how much history a base accumulates.
pub fn get_canon_submissions_for_chunk(
    conn: &mut PgConnection,
    chunk_id: u32,
) -> Result<Vec<SubmissionRecord>> {
    use diesel::sql_query;
    use diesel::sql_types::Integer;

    let chunk_id = conversions::u32_to_i32(chunk_id)?;

    let query = "SELECT s.*
        FROM fields f
        JOIN submissions s ON f.canon_submission_id = s.id
        WHERE f.chunk_id = $1;";

    let items: Vec<SubmissionPrivate> = sql_query(query)
        .bind::<Integer, _>(chunk_id)
        .load(conn)
        .map_err(|e| anyhow!("{e}"))?;

    items.into_iter().map(private_to_public).collect()
}

/// Get the highest submission id, or 0 if the table is empty. Snapshotted at
/// the start of a jobs run so submissions arriving mid-run fall into the next
/// run's window.
pub fn get_max_submission_id(conn: &mut PgConnection) -> Result<i64> {
    use diesel::sql_query;

    #[derive(QueryableByName)]
    struct MaxRow {
        #[diesel(sql_type = diesel::sql_types::BigInt)]
        max_id: i64,
    }

    let row: MaxRow = sql_query("SELECT COALESCE(MAX(id), 0) AS max_id FROM submissions")
        .get_result(conn)
        .map_err(|e| anyhow!("{e}"))?;
    Ok(row.max_id)
}

/// Get the distinct (base, chunk) pairs containing fields that received any
/// submission (either mode) in the id window. These are the chunks whose
/// derived statistics may be stale: niceonly submissions move field
/// `check_level` 0 -> 1 at submit time and detailed ones 1 -> 2, so both modes
/// dirty a chunk even before consensus runs.
pub fn get_chunks_with_new_submissions(
    conn: &mut PgConnection,
    after_id: i64,
    up_to_id: i64,
) -> Result<Vec<(u32, Option<u32>)>> {
    use diesel::sql_query;
    use diesel::sql_types::BigInt;

    #[derive(QueryableByName)]
    struct DirtyChunkRow {
        #[diesel(sql_type = diesel::sql_types::Integer)]
        base_id: i32,
        #[diesel(sql_type = diesel::sql_types::Nullable<diesel::sql_types::Integer>)]
        chunk_id: Option<i32>,
    }

    let query = "SELECT DISTINCT f.base_id, f.chunk_id
        FROM submissions s
        JOIN fields f ON f.id = s.field_id
        WHERE s.id > $1 AND s.id <= $2;";

    let items: Vec<DirtyChunkRow> = sql_query(query)
        .bind::<BigInt, _>(after_id)
        .bind::<BigInt, _>(up_to_id)
        .load(conn)
        .map_err(|e| anyhow!("{e}"))?;

    items
        .into_iter()
        .map(|r| {
            Ok((
                conversions::i32_to_u32(r.base_id)?,
                conversions::opti32_to_optu32(r.chunk_id)?,
            ))
        })
        .collect()
}
