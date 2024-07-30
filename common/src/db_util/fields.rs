#![allow(dead_code)]

use super::*;

table! {
    fields (id) {
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
#[diesel(table_name = fields)]
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
#[diesel(table_name = fields)]
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
    use self::fields::dsl::*;

    let insert_rows: Vec<FieldPrivateNew> = sizes
        .iter()
        .map(|size| build_new_row(base, size).unwrap())
        .collect();

    // chunk it out if there's too many fields
    for chunk in insert_rows.chunks(10000) {
        diesel::insert_into(fields)
            .values(chunk)
            .execute(conn)
            .map_err(|err| err.to_string())?;
    }

    Ok(())
}

pub fn get_field_by_id(conn: &mut PgConnection, row_id: u128) -> Result<FieldRecord, String> {
    use self::fields::dsl::*;

    let row_id = conversions::u128_to_i64(row_id)?;

    fields
        .filter(id.eq(row_id))
        .first::<FieldPrivate>(conn)
        .map_err(|err| err.to_string())
        .and_then(private_to_public)
}

pub fn get_fields_in_base(conn: &mut PgConnection, base: u32) -> Result<Vec<FieldRecord>, String> {
    use self::fields::dsl::*;

    let base = conversions::u32_to_i32(base)?;
    let items_private: Vec<FieldPrivate> = fields
        .filter(base_id.eq(base))
        .order(id.asc())
        .load(conn)
        .map_err(|err| err.to_string())?;

    items_private
        .into_iter()
        .map(private_to_public)
        .collect::<Result<Vec<FieldRecord>, String>>()
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
            "UPDATE fields
            SET last_claim_time = NOW()
            WHERE id = (
                SELECT id FROM fields
                WHERE (last_claim_time <= $1 OR last_claim_time IS NULL)
                AND check_level <= $2
                AND range_size <= $3
                ORDER BY id ASC
                LIMIT 1
            )
            RETURNING *;"
        }
        FieldClaimStrategy::Random => {
            "UPDATE fields
            SET last_claim_time = NOW()
            WHERE id = (
                SELECT id FROM fields
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

pub fn get_count_checked_by_range(
    conn: &mut PgConnection,
    in_check_level: u8,
    range: FieldSize,
) -> Result<u128, String> {
    use self::fields::dsl::*;
    use diesel::dsl::sum;

    let in_check_level = conversions::u8_to_i32(in_check_level)?;
    let in_range_start = conversions::u128_to_bigdec(range.range_start)?;
    let in_range_end = conversions::u128_to_bigdec(range.range_end)?;

    let count = fields
        .select(sum(range_size))
        .filter(check_level.ge(in_check_level))
        .filter(range_start.ge(in_range_start))
        .filter(range_end.le(in_range_end))
        .first::<Option<BigDecimal>>(conn)
        .map_err(|err| err.to_string())?
        .unwrap_or(BigDecimal::from(0u32));

    conversions::bigdec_to_u128(count)
}

pub fn update_field(
    conn: &mut PgConnection,
    row_id: u128,
    update_row: FieldRecord,
) -> Result<FieldRecord, String> {
    use self::fields::dsl::*;

    let row_id = conversions::u128_to_i64(row_id)?;
    let update_row = public_to_private(update_row)?;

    diesel::update(fields.filter(id.eq(row_id)))
        .set(&update_row)
        .get_result(conn)
        .map_err(|err| err.to_string())
        .and_then(private_to_public)
}

pub fn update_field_canon_and_cl(
    conn: &mut PgConnection,
    field_id: u128,
    submission_id: Option<u32>,
    in_check_level: u8,
) -> Result<(), String> {
    use self::fields::dsl::*;

    let field_id = conversions::u128_to_i64(field_id)?;
    let submission_id = conversions::optu32_to_opti32(submission_id)?;
    let in_check_level = conversions::u8_to_i32(in_check_level)?;

    diesel::update(fields)
        .filter(id.eq(field_id))
        .set((
            canon_submission_id.eq(submission_id),
            check_level.eq(in_check_level),
        ))
        .execute(conn)
        .map_err(|err| err.to_string())?;

    Ok(())
}
