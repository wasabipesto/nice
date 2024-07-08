#![allow(dead_code)]

use super::*;

table! {
    chunk (id) {
        id -> Integer,
        base_id -> Integer,
        range_start -> Numeric,
        range_end -> Numeric,
        range_size -> Numeric,
        checked_detailed -> Numeric,
        checked_niceonly -> Numeric,
        minimum_cl -> Integer,
        niceness_mean -> Nullable<Float>,
        niceness_stdev -> Nullable<Float>,
        distribution -> Jsonb,
        numbers -> Jsonb,
    }
}

#[derive(Queryable, AsChangeset)]
#[diesel(table_name = chunk)]
struct ChunkPrivate {
    id: i32,
    base_id: i32,
    range_start: BigDecimal,
    range_end: BigDecimal,
    range_size: BigDecimal,
    checked_detailed: BigDecimal,
    checked_niceonly: BigDecimal,
    minimum_cl: i32,
    niceness_mean: Option<f32>,
    niceness_stdev: Option<f32>,
    distribution: Value,
    numbers: Value,
}

#[derive(Insertable)]
#[diesel(table_name = chunk)]
struct ChunkPrivateNew {
    base_id: i32,
    range_start: BigDecimal,
    range_end: BigDecimal,
    range_size: BigDecimal,
}

fn private_to_public(p: ChunkPrivate) -> Result<ChunkRecord, String> {
    use conversions::*;
    Ok(ChunkRecord {
        chunk_id: i32_to_u32(p.id)?,
        base: i32_to_u32(p.base_id)?,
        range_start: bigdec_to_u128(p.range_start)?,
        range_end: bigdec_to_u128(p.range_end)?,
        range_size: bigdec_to_u128(p.range_size)?,
        checked_detailed: bigdec_to_u128(p.checked_detailed)?,
        checked_niceonly: bigdec_to_u128(p.checked_niceonly)?,
        minimum_cl: i32_to_u8(p.minimum_cl)?,
        niceness_mean: p.niceness_mean,
        niceness_stdev: p.niceness_stdev,
        distribution: deserialize_distribution(p.distribution)?,
        numbers: deserialize_numbers(p.numbers)?,
    })
}

fn public_to_private(p: ChunkRecord) -> Result<ChunkPrivate, String> {
    use conversions::*;
    Ok(ChunkPrivate {
        id: u32_to_i32(p.chunk_id)?,
        base_id: u32_to_i32(p.base)?,
        range_start: u128_to_bigdec(p.range_start)?,
        range_end: u128_to_bigdec(p.range_end)?,
        range_size: u128_to_bigdec(p.range_size)?,
        checked_detailed: u128_to_bigdec(p.checked_detailed)?,
        checked_niceonly: u128_to_bigdec(p.checked_niceonly)?,
        minimum_cl: u8_to_i32(p.minimum_cl)?,
        niceness_mean: p.niceness_mean,
        niceness_stdev: p.niceness_stdev,
        distribution: serialize_distribution(p.distribution)?,
        numbers: serialize_numbers(p.numbers)?,
    })
}

fn build_new_row(base: u32, size: FieldSize) -> Result<ChunkPrivateNew, String> {
    use conversions::*;
    Ok(ChunkPrivateNew {
        base_id: u32_to_i32(base)?,
        range_start: u128_to_bigdec(size.range_start)?,
        range_end: u128_to_bigdec(size.range_end)?,
        range_size: u128_to_bigdec(size.range_size)?,
    })
}

pub fn insert_chunk(
    conn: &mut PgConnection,
    base: u32,
    size: FieldSize,
) -> Result<ChunkRecord, String> {
    use self::chunk::dsl::*;

    let insert_row = build_new_row(base, size)?;

    diesel::insert_into(chunk)
        .values(&insert_row)
        .get_result(conn)
        .map_err(|err| err.to_string())
        .and_then(private_to_public)
}

pub fn get_chunk(conn: &mut PgConnection, row_id: u32) -> Result<ChunkRecord, String> {
    use self::chunk::dsl::*;

    let row_id = conversions::u32_to_i32(row_id)?;

    chunk
        .filter(id.eq(row_id))
        .first::<ChunkPrivate>(conn)
        .map_err(|err| err.to_string())
        .and_then(private_to_public)
}

pub fn update_chunk(
    conn: &mut PgConnection,
    row_id: u32,
    update_row: ChunkRecord,
) -> Result<ChunkRecord, String> {
    use self::chunk::dsl::*;

    let row_id = conversions::u32_to_i32(row_id)?;
    let update_row = public_to_private(update_row)?;

    diesel::update(chunk.filter(id.eq(row_id)))
        .set(&update_row)
        .get_result(conn)
        .map_err(|err| err.to_string())
        .and_then(private_to_public)
}
