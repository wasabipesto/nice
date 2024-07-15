#![allow(dead_code)]

use super::*;

table! {
    base (id) {
        id -> Integer,
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
#[diesel(table_name = base)]
struct BasePrivate {
    id: i32,
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
#[diesel(table_name = base)]
struct BasePrivateNew {
    id: i32,
    range_start: BigDecimal,
    range_end: BigDecimal,
    range_size: BigDecimal,
}

fn private_to_public(p: BasePrivate) -> Result<BaseRecord, String> {
    use conversions::*;
    Ok(BaseRecord {
        base: i32_to_u32(p.id)?,
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

fn public_to_private(p: BaseRecord) -> Result<BasePrivate, String> {
    use conversions::*;
    Ok(BasePrivate {
        id: u32_to_i32(p.base)?,
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

fn build_new_row(base: u32, size: FieldSize) -> Result<BasePrivateNew, String> {
    use conversions::*;
    Ok(BasePrivateNew {
        id: u32_to_i32(base)?,
        range_start: u128_to_bigdec(size.range_start)?,
        range_end: u128_to_bigdec(size.range_end)?,
        range_size: u128_to_bigdec(size.range_size)?,
    })
}

pub fn insert_base(
    conn: &mut PgConnection,
    base_id: u32,
    size: FieldSize,
) -> Result<BaseRecord, String> {
    use self::base::dsl::*;

    let insert_row = build_new_row(base_id, size)?;

    diesel::insert_into(base)
        .values(&insert_row)
        .get_result(conn)
        .map_err(|err| err.to_string())
        .and_then(private_to_public)
}

pub fn get_base_by_id(conn: &mut PgConnection, row_id: u32) -> Result<BaseRecord, String> {
    use self::base::dsl::*;

    let row_id = conversions::u32_to_i32(row_id)?;

    base.filter(id.eq(row_id))
        .first::<BasePrivate>(conn)
        .map_err(|err| err.to_string())
        .and_then(private_to_public)
}

pub fn get_all_bases(conn: &mut PgConnection) -> Result<Vec<BaseRecord>, String> {
    use self::base::dsl::*;

    let bases_private: Vec<BasePrivate> = base.load(conn).map_err(|err| err.to_string())?;
    let mut bases = Vec::new();
    for b in bases_private {
        bases.push(private_to_public(b)?)
    }
    Ok(bases)
}

pub fn update_base(
    conn: &mut PgConnection,
    row_id: u32,
    update_row: BaseRecord,
) -> Result<BaseRecord, String> {
    use self::base::dsl::*;

    let row_id = conversions::u32_to_i32(row_id)?;
    let update_row = public_to_private(update_row)?;

    diesel::update(base.filter(id.eq(row_id)))
        .set(&update_row)
        .get_result(conn)
        .map_err(|err| err.to_string())
        .and_then(private_to_public)
}
