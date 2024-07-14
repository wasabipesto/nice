#![allow(dead_code)]

use super::*;

table! {
    claim (id) {
        id -> BigInt,
        field_id -> Integer,
        search_mode -> Varchar,
        claim_time -> Timestamptz,
        user_ip -> Varchar,
    }
}

#[derive(Queryable)]
#[diesel(table_name = claim)]
struct ClaimPrivate {
    id: i64,
    field_id: i32,
    search_mode: String,
    claim_time: DateTime<Utc>,
    user_ip: String,
}

#[derive(Insertable)]
#[diesel(table_name = claim)]
struct ClaimPrivateNew {
    field_id: i32,
    search_mode: String,
    user_ip: String,
}

fn private_to_public(p: ClaimPrivate) -> Result<ClaimRecord, String> {
    use conversions::*;
    Ok(ClaimRecord {
        claim_id: i64_to_u128(p.id)?,
        field_id: i32_to_u128(p.field_id)?,
        search_mode: deserialize_searchmode(p.search_mode)?,
        claim_time: p.claim_time,
        user_ip: p.user_ip,
    })
}

fn public_to_private(p: ClaimRecord) -> Result<ClaimPrivate, String> {
    use conversions::*;
    Ok(ClaimPrivate {
        id: u128_to_i64(p.claim_id)?,
        field_id: u128_to_i32(p.field_id)?,
        search_mode: serialize_searchmode(p.search_mode),
        claim_time: p.claim_time,
        user_ip: p.user_ip,
    })
}

fn build_new_row(
    field_id: u128,
    search_mode: SearchMode,
    user_ip: String,
) -> Result<ClaimPrivateNew, String> {
    use conversions::*;
    Ok(ClaimPrivateNew {
        field_id: u128_to_i32(field_id)?,
        search_mode: serialize_searchmode(search_mode),
        user_ip,
    })
}

pub fn insert_claim(
    conn: &mut PgConnection,
    input_field_id: u128,
    input_search_mode: SearchMode,
    input_user_ip: String,
) -> Result<ClaimRecord, String> {
    use self::claim::dsl::*;

    let insert_row = build_new_row(input_field_id, input_search_mode, input_user_ip)?;

    diesel::insert_into(claim)
        .values(&insert_row)
        .get_result(conn)
        .map_err(|err| err.to_string())
        .and_then(private_to_public)
}

pub fn get_claim_by_id(conn: &mut PgConnection, row_id: u128) -> Result<ClaimRecord, String> {
    use self::claim::dsl::*;

    let row_id = conversions::u128_to_i64(row_id)?;

    claim
        .filter(id.eq(row_id))
        .first::<ClaimPrivate>(conn)
        .map_err(|err| err.to_string())
        .and_then(private_to_public)
}
