#![allow(dead_code)]

use super::*;

table! {
    submission (id) {
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

#[derive(Queryable)]
#[diesel(table_name = submission)]
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
#[diesel(table_name = submission)]
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
    distribution: Option<Vec<UniquesDistributionExtended>>,
    numbers: Vec<NiceNumbersExtended>,
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
    input_distribution: Option<Vec<UniquesDistributionExtended>>,
    input_numbers: Vec<NiceNumbersExtended>,
) -> Result<SubmissionRecord, String> {
    use self::submission::dsl::*;

    let insert_row = build_new_row(
        claim_record,
        submit_data,
        input_user_ip,
        input_distribution,
        input_numbers,
    )?;

    diesel::insert_into(submission)
        .values(&insert_row)
        .get_result(conn)
        .map_err(|err| err.to_string())
        .and_then(private_to_public)
}

pub fn get_csubmission_by_id(
    conn: &mut PgConnection,
    row_id: u128,
) -> Result<SubmissionRecord, String> {
    use self::submission::dsl::*;

    let row_id = conversions::u128_to_i64(row_id)?;

    submission
        .filter(id.eq(row_id))
        .first::<SubmissionPrivate>(conn)
        .map_err(|err| err.to_string())
        .and_then(private_to_public)
}
