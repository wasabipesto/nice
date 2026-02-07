#![allow(dead_code)]

use super::*;

table! {
    claims (id) {
        id -> BigInt,
        field_id -> Integer,
        search_mode -> Varchar,
        claim_time -> Timestamptz,
        user_ip -> Varchar,
    }
}

#[derive(Queryable)]
#[diesel(table_name = claims)]
struct ClaimPrivate {
    id: i64,
    field_id: i32,
    search_mode: String,
    claim_time: DateTime<Utc>,
    user_ip: String,
}

#[derive(Insertable)]
#[diesel(table_name = claims)]
struct ClaimPrivateNew {
    field_id: i32,
    search_mode: String,
    user_ip: String,
}

fn private_to_public(p: ClaimPrivate) -> Result<ClaimRecord> {
    use conversions::*;
    Ok(ClaimRecord {
        claim_id: i64_to_u128(p.id)?,
        field_id: i32_to_u128(p.field_id)?,
        search_mode: deserialize_searchmode(p.search_mode)?,
        claim_time: p.claim_time,
        user_ip: p.user_ip,
    })
}

fn public_to_private(p: ClaimRecord) -> Result<ClaimPrivate> {
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
) -> Result<ClaimPrivateNew> {
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
) -> Result<ClaimRecord> {
    use self::claims::dsl::*;

    let insert_row = build_new_row(input_field_id, input_search_mode, input_user_ip)?;

    let result = diesel::insert_into(claims)
        .values(&insert_row)
        .get_result(conn)
        .map_err(|e| anyhow!("{e}"))?;
    private_to_public(result)
}

pub fn get_claim_by_id(conn: &mut PgConnection, row_id: u128) -> Result<ClaimRecord> {
    use self::claims::dsl::*;

    let row_id = conversions::u128_to_i64(row_id)?;

    let result = claims
        .filter(id.eq(row_id))
        .first::<ClaimPrivate>(conn)
        .map_err(|e| anyhow!("{e}"))?;
    private_to_public(result)
}
