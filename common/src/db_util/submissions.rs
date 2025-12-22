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
}

fn private_to_public(p: SubmissionPrivate) -> Result<SubmissionRecord, String> {
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

fn public_to_private(p: SubmissionRecord) -> Result<SubmissionPrivate, String> {
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
    })
}

fn build_new_row(
    claim_record: ClaimRecord,
    submit_data: DataToServer,
    user_ip: String,
    distribution: Option<Vec<UniquesDistribution>>,
    numbers: Vec<NiceNumber>,
) -> Result<SubmissionPrivateNew, String> {
    use conversions::*;
    Ok(SubmissionPrivateNew {
        claim_id: u128_to_i32(claim_record.claim_id)?,
        field_id: u128_to_i32(claim_record.field_id)?,
        search_mode: serialize_searchmode(claim_record.search_mode),
        elapsed_secs: (Utc::now() - claim_record.claim_time).num_milliseconds() as f32 / 1000f32,
        username: submit_data.username,
        user_ip,
        client_version: submit_data.client_version,
        distribution: serialize_opt_distribution(distribution)?,
        numbers: serialize_numbers(numbers)?,
    })
}

pub fn insert_submission(
    conn: &mut PgConnection,
    claim_record: ClaimRecord,
    submit_data: DataToServer,
    input_user_ip: String,
    input_distribution: Option<Vec<UniquesDistribution>>,
    input_numbers: Vec<NiceNumber>,
) -> Result<SubmissionRecord, String> {
    use self::submissions::dsl::*;

    let insert_row = build_new_row(
        claim_record,
        submit_data,
        input_user_ip,
        input_distribution,
        input_numbers,
    )?;

    diesel::insert_into(submissions)
        .values(&insert_row)
        .get_result(conn)
        .map_err(|err| err.to_string())
        .and_then(private_to_public)
}

pub fn get_submission_by_id(
    conn: &mut PgConnection,
    row_id: u128,
) -> Result<SubmissionRecord, String> {
    use self::submissions::dsl::*;

    let row_id = conversions::u128_to_i64(row_id)?;

    submissions
        .filter(id.eq(row_id))
        .first::<SubmissionPrivate>(conn)
        .map_err(|err| err.to_string())
        .and_then(private_to_public)
}

pub fn get_canon_submissions_by_range(
    conn: &mut PgConnection,
    start: u128,
    end: u128,
) -> Result<Vec<SubmissionRecord>, String> {
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
        .map_err(|err| err.to_string())?;

    items_private
        .into_iter()
        .map(private_to_public)
        .collect::<Result<Vec<SubmissionRecord>, String>>()
}

pub fn get_submissions_qualified_detailed_for_field(
    conn: &mut PgConnection,
    input_field_id: u128,
) -> Result<Vec<SubmissionRecord>, String> {
    use self::submissions::dsl::*;

    let input_field_id = conversions::u128_to_i32(input_field_id)?;
    let input_search_mode = conversions::serialize_searchmode(SearchMode::Detailed);
    let input_disqualified = false;

    let items_private: Vec<SubmissionPrivate> = submissions
        .filter(field_id.eq(input_field_id))
        .filter(search_mode.eq(input_search_mode))
        .filter(disqualified.eq(input_disqualified))
        .load(conn)
        .map_err(|err| err.to_string())?;

    items_private
        .into_iter()
        .map(private_to_public)
        .collect::<Result<Vec<SubmissionRecord>, String>>()
}

/// Struct to hold submission with chunk_id from batch query
#[derive(Debug, QueryableByName)]
pub struct SubmissionWithChunk {
    #[diesel(sql_type = diesel::sql_types::BigInt)]
    pub id: i64,
    #[diesel(sql_type = diesel::sql_types::Integer)]
    pub claim_id: i32,
    #[diesel(sql_type = diesel::sql_types::Integer)]
    pub field_id: i32,
    #[diesel(sql_type = diesel::sql_types::Text)]
    pub search_mode: String,
    #[diesel(sql_type = diesel::sql_types::Timestamptz)]
    pub submit_time: DateTime<Utc>,
    #[diesel(sql_type = diesel::sql_types::Float)]
    pub elapsed_secs: f32,
    #[diesel(sql_type = diesel::sql_types::Text)]
    pub username: String,
    #[diesel(sql_type = diesel::sql_types::Text)]
    pub user_ip: String,
    #[diesel(sql_type = diesel::sql_types::Text)]
    pub client_version: String,
    #[diesel(sql_type = diesel::sql_types::Bool)]
    pub disqualified: bool,
    #[diesel(sql_type = diesel::sql_types::Nullable<diesel::sql_types::Jsonb>)]
    pub distribution: Option<Value>,
    #[diesel(sql_type = diesel::sql_types::Jsonb)]
    pub numbers: Value,
    #[diesel(sql_type = diesel::sql_types::Nullable<diesel::sql_types::Integer>)]
    pub chunk_id: Option<i32>,
}

/// Get all canon submissions for a base with their chunk_ids in a single query.
/// This is much more efficient than querying each chunk individually.
pub fn get_canon_submissions_with_chunks_by_base(
    conn: &mut PgConnection,
    base: u32,
) -> Result<Vec<(SubmissionRecord, Option<u32>)>, String> {
    use diesel::sql_query;
    use diesel::sql_types::Integer;

    let base = conversions::u32_to_i32(base)?;

    let query = "SELECT s.*, f.chunk_id
        FROM fields f
        JOIN submissions s ON f.canon_submission_id = s.id
        WHERE f.base_id = $1;";

    let items: Vec<SubmissionWithChunk> = sql_query(query)
        .bind::<Integer, _>(base)
        .load(conn)
        .map_err(|err| err.to_string())?;

    items
        .into_iter()
        .map(|item| {
            let submission = SubmissionRecord {
                submission_id: conversions::i64_to_u128(item.id)?,
                claim_id: conversions::i32_to_u128(item.claim_id)?,
                field_id: conversions::i32_to_u128(item.field_id)?,
                search_mode: conversions::deserialize_searchmode(item.search_mode)?,
                submit_time: item.submit_time,
                elapsed_secs: item.elapsed_secs,
                username: item.username,
                user_ip: item.user_ip,
                client_version: item.client_version,
                disqualified: item.disqualified,
                distribution: conversions::deserialize_opt_distribution(item.distribution)?,
                numbers: conversions::deserialize_numbers(item.numbers)?,
            };
            let chunk_id = conversions::opti32_to_optu32(item.chunk_id)?;
            Ok((submission, chunk_id))
        })
        .collect::<Result<Vec<_>, String>>()
}
