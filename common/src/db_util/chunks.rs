#![allow(dead_code)]

use conversions::u32_to_i32;

use super::*;

table! {
    chunks (id) {
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
#[diesel(table_name = chunks)]
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
#[diesel(table_name = chunks)]
struct ChunkPrivateNew {
    base_id: i32,
    range_start: BigDecimal,
    range_end: BigDecimal,
    range_size: BigDecimal,
}

fn private_to_public(p: ChunkPrivate) -> Result<ChunkRecord> {
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

fn public_to_private(p: ChunkRecord) -> Result<ChunkPrivate> {
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

fn build_new_row(base: u32, size: &FieldSize) -> Result<ChunkPrivateNew> {
    use conversions::*;
    Ok(ChunkPrivateNew {
        base_id: u32_to_i32(base)?,
        range_start: u128_to_bigdec(size.range_start)?,
        range_end: u128_to_bigdec(size.range_end)?,
        range_size: u128_to_bigdec(size.size())?,
    })
}

pub fn insert_chunks(conn: &mut PgConnection, base: u32, sizes: &[FieldSize]) -> Result<()> {
    use self::chunks::dsl::*;

    let insert_rows: Vec<ChunkPrivateNew> = sizes
        .iter()
        .map(|size| build_new_row(base, size).unwrap())
        .collect();

    diesel::insert_into(chunks)
        .values(&insert_rows)
        .execute(conn)
        .map_err(|e| anyhow!("{e}"))?;

    Ok(())
}

pub fn get_chunk_by_id(conn: &mut PgConnection, row_id: u32) -> Result<ChunkRecord> {
    use self::chunks::dsl::*;

    let row_id = conversions::u32_to_i32(row_id)?;

    let result = chunks
        .filter(id.eq(row_id))
        .first::<ChunkPrivate>(conn)
        .map_err(|e| anyhow!("{e}"))?;
    private_to_public(result)
}

pub fn get_chunks_in_base(conn: &mut PgConnection, base: u32) -> Result<Vec<ChunkRecord>> {
    use self::chunks::dsl::*;

    let base = u32_to_i32(base)?;
    let items_private: Vec<ChunkPrivate> = chunks
        .filter(base_id.eq(base))
        .order(id.asc())
        .load(conn)
        .map_err(|e| anyhow!("{e}"))?;

    items_private
        .into_iter()
        .map(private_to_public)
        .collect::<Result<Vec<ChunkRecord>>>()
}

pub fn update_chunk(
    conn: &mut PgConnection,
    row_id: u32,
    update_row: ChunkRecord,
) -> Result<ChunkRecord> {
    use self::chunks::dsl::*;

    let row_id = conversions::u32_to_i32(row_id)?;
    let update_row = public_to_private(update_row)?;

    let result = diesel::update(chunks.filter(id.eq(row_id)))
        .set(&update_row)
        .get_result(conn)
        .map_err(|e| anyhow!("{e}"))?;
    private_to_public(result)
}

#[allow(clippy::cast_possible_wrap)]
pub fn reassign_fields_to_chunks(conn: &mut PgConnection, base: u32) -> Result<()> {
    use diesel::sql_types::Integer;

    let query = "
        UPDATE fields f
        SET chunk_id = c.id
        FROM chunks c
        WHERE f.base_id = $1
            AND f.range_start >= c.range_start
            AND f.range_end <= c.range_end;"
        .to_string();

    diesel::sql_query(query)
        .bind::<Integer, _>(base as i32)
        .execute(conn)
        .map_err(|e| anyhow!("{e}"))?;

    Ok(())
}
