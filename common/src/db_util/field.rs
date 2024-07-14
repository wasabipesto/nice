#![allow(dead_code)]

use super::*;

table! {
    field (id) {
        id -> BigInt,
        base_id -> Integer,
        chunk_id -> Nullable<Integer>,
        range_start -> Numeric,
        range_end -> Numeric,
        range_size -> Numeric,
        last_claim_time -> Nullable<Timestamptz>,
        canon_submission_id -> Nullable<Integer>,
        check_level -> Integer,
        prioritize -> Bool,
    }
}

#[derive(Queryable, AsChangeset, QueryableByName)]
#[diesel(table_name = field)]
struct FieldPrivate {
    id: i64,
    base_id: i32,
    chunk_id: Option<i32>,
    range_start: BigDecimal,
    range_end: BigDecimal,
    range_size: BigDecimal,
    last_claim_time: Option<DateTime<Utc>>,
    canon_submission_id: Option<i32>,
    check_level: i32,
    prioritize: bool,
}

#[derive(Insertable)]
#[diesel(table_name = field)]
struct FieldPrivateNew {
    base_id: i32,
    range_start: BigDecimal,
    range_end: BigDecimal,
    range_size: BigDecimal,
}

fn private_to_public(p: FieldPrivate) -> Result<FieldRecord, String> {
    use conversions::*;
    Ok(FieldRecord {
        field_id: i64_to_u128(p.id)?,
        base: i32_to_u32(p.base_id)?,
        chunk_id: opti32_to_optu32(p.chunk_id)?,
        range_start: bigdec_to_u128(p.range_start)?,
        range_end: bigdec_to_u128(p.range_end)?,
        range_size: bigdec_to_u128(p.range_size)?,
        last_claim_time: p.last_claim_time,
        canon_submission_id: opti32_to_optu32(p.canon_submission_id)?,
        check_level: i32_to_u8(p.check_level)?,
        prioritize: p.prioritize,
    })
}

fn public_to_private(p: FieldRecord) -> Result<FieldPrivate, String> {
    use conversions::*;
    Ok(FieldPrivate {
        id: u128_to_i64(p.field_id)?,
        base_id: u32_to_i32(p.base)?,
        chunk_id: optu32_to_opti32(p.chunk_id)?,
        range_start: u128_to_bigdec(p.range_start)?,
        range_end: u128_to_bigdec(p.range_end)?,
        range_size: u128_to_bigdec(p.range_size)?,
        last_claim_time: p.last_claim_time,
        canon_submission_id: optu32_to_opti32(p.canon_submission_id)?,
        check_level: u8_to_i32(p.check_level)?,
        prioritize: p.prioritize,
    })
}

fn build_new_row(base: u32, size: &FieldSize) -> Result<FieldPrivateNew, String> {
    use conversions::*;
    Ok(FieldPrivateNew {
        base_id: u32_to_i32(base)?,
        range_start: u128_to_bigdec(size.range_start)?,
        range_end: u128_to_bigdec(size.range_end)?,
        range_size: u128_to_bigdec(size.range_size)?,
    })
}

pub fn insert_fields(
    conn: &mut PgConnection,
    base: u32,
    sizes: Vec<FieldSize>,
) -> Result<(), String> {
    use self::field::dsl::*;

    let insert_rows: Vec<FieldPrivateNew> = sizes
        .iter()
        .map(|size| build_new_row(base, size).unwrap())
        .collect();

    for chunk in insert_rows.chunks(10000) {
        diesel::insert_into(field)
            .values(chunk)
            .execute(conn)
            .map_err(|err| err.to_string())?;
    }

    Ok(())
}

pub fn get_field_by_id(conn: &mut PgConnection, row_id: u128) -> Result<FieldRecord, String> {
    use self::field::dsl::*;

    let row_id = conversions::u128_to_i64(row_id)?;

    field
        .filter(id.eq(row_id))
        .first::<FieldPrivate>(conn)
        .map_err(|err| err.to_string())
        .and_then(private_to_public)
}

/// Finds the next field that matches the criteria, updates last_claim_time, and returns it.
/// Returns Ok(None) if no matching fields are found.
pub fn try_claim_field(
    conn: &mut PgConnection,
    claim_strategy: FieldClaimStrategy,
    maximum_timestamp: DateTime<Utc>,
    maximum_check_level: u8,
    maximum_size: u128,
) -> Result<Option<FieldRecord>, String> {
    use diesel::sql_query;
    use diesel::sql_types::{Integer, Numeric, Timestamptz};

    let maximum_check_level = conversions::u8_to_i32(maximum_check_level)?;
    let maximum_size = conversions::u128_to_bigdec(maximum_size)?;

    let query = match claim_strategy {
        FieldClaimStrategy::Next => {
            "UPDATE field
            SET last_claim_time = NOW()
            WHERE id = (
                SELECT id FROM field
                WHERE (last_claim_time <= $1 OR last_claim_time IS NULL)
                AND check_level <= $2
                AND range_size <= $3
                ORDER BY id ASC
                LIMIT 1
            )
            RETURNING *;"
        }
        FieldClaimStrategy::Random => {
            "UPDATE field
            SET last_claim_time = NOW()
            WHERE id = (
                SELECT id FROM field
                WHERE (last_claim_time <= $1 OR last_claim_time IS NULL)
                AND check_level <= $2
                AND range_size <= $3
                ORDER BY RANDOM() ASC
                LIMIT 1
            )
            RETURNING *;"
        }
    }
    .to_string();

    sql_query(query)
        .bind::<Timestamptz, _>(maximum_timestamp)
        .bind::<Integer, _>(maximum_check_level)
        .bind::<Numeric, _>(maximum_size)
        .get_result::<FieldPrivate>(conn)
        .optional()
        .map_err(|err| err.to_string())
        .and_then(|opt| opt.map_or(Ok(None), |rec| private_to_public(rec).map(Some)))
}

pub fn update_field(
    conn: &mut PgConnection,
    row_id: u128,
    update_row: FieldRecord,
) -> Result<FieldRecord, String> {
    use self::field::dsl::*;

    let row_id = conversions::u128_to_i64(row_id)?;
    let update_row = public_to_private(update_row)?;

    diesel::update(field.filter(id.eq(row_id)))
        .set(&update_row)
        .get_result(conn)
        .map_err(|err| err.to_string())
        .and_then(private_to_public)
}
