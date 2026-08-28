#![allow(dead_code)]

use super::*;
use anyhow::anyhow;
use rand::RngExt;

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

fn private_to_public(p: FieldPrivate) -> Result<FieldRecord> {
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

fn public_to_private(p: &FieldRecord) -> Result<FieldPrivate> {
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

fn build_new_row(base: u32, size: &FieldSize) -> Result<FieldPrivateNew> {
    use conversions::*;
    Ok(FieldPrivateNew {
        base_id: u32_to_i32(base)?,
        range_start: u128_to_bigdec(size.range_start)?,
        range_end: u128_to_bigdec(size.range_end)?,
        range_size: u128_to_bigdec(size.size())?,
    })
}

pub fn insert_fields(conn: &mut PgConnection, base: u32, sizes: &[FieldSize]) -> Result<()> {
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
            .map_err(|e| anyhow!("{e}"))?;
    }

    Ok(())
}

/// Returns the maximum `fields.id` (as u128). Assumes ids are contiguous and monotonically increasing.
pub fn get_max_field_id(conn: &mut PgConnection) -> Result<u128> {
    use diesel::sql_query;
    use diesel::sql_types::BigInt;

    #[derive(QueryableByName)]
    struct MaxIdRow {
        #[diesel(sql_type = BigInt)]
        max_id: i64,
    }

    let row: MaxIdRow = sql_query("SELECT MAX(id) AS max_id FROM fields;")
        .get_result(conn)
        .map_err(|e| anyhow!("{e}"))?;

    conversions::i64_to_u128(row.max_id)
}

pub fn get_field_by_id(conn: &mut PgConnection, row_id: u128) -> Result<FieldRecord> {
    use self::fields::dsl::*;

    let row_id = conversions::u128_to_i64(row_id)?;

    let result = fields
        .filter(id.eq(row_id))
        .first::<FieldPrivate>(conn)
        .map_err(|e| anyhow!("{e}"))?;
    private_to_public(result)
}

pub fn get_fields_in_base(conn: &mut PgConnection, base: u32) -> Result<Vec<FieldRecord>> {
    use self::fields::dsl::*;

    let base = conversions::u32_to_i32(base)?;
    let items_private: Vec<FieldPrivate> = fields
        .filter(base_id.eq(base))
        .order(id.asc())
        .load(conn)
        .map_err(|e| anyhow!("{e}"))?;

    items_private
        .into_iter()
        .map(private_to_public)
        .collect::<Result<Vec<FieldRecord>>>()
}

pub fn get_fields_in_range(
    conn: &mut PgConnection,
    field_start: u128,
    field_end: u128,
) -> Result<Vec<FieldRecord>> {
    use self::fields::dsl::*;

    let field_start = conversions::u128_to_bigdec(field_start)?;
    let field_end = conversions::u128_to_bigdec(field_end)?;

    let items_private: Vec<FieldPrivate> = fields
        .filter(range_start.ge(field_start))
        .filter(range_end.le(field_end))
        .order(id.asc())
        .load(conn)
        .map_err(|e| anyhow!("{e}"))?;

    items_private
        .into_iter()
        .map(private_to_public)
        .collect::<Result<Vec<FieldRecord>>>()
}

pub fn get_fields_in_base_with_detailed_subs(
    conn: &mut PgConnection,
    base: u32,
) -> Result<Vec<FieldRecord>> {
    use diesel::sql_query;
    use diesel::sql_types::Integer;

    let base = conversions::u32_to_i32(base)?;
    let query = "SELECT DISTINCT ON (f.id) f.*
            FROM fields f
            JOIN submissions s ON f.id = s.field_id
            WHERE f.base_id = $1 AND s.search_mode = 'detailed'
            ORDER BY f.id ASC";

    let items_private: Vec<FieldPrivate> = sql_query(query)
        .bind::<Integer, _>(base)
        .load(conn)
        .map_err(|e| anyhow!("{e}"))?;

    items_private
        .into_iter()
        .map(private_to_public)
        .collect::<Result<Vec<FieldRecord>>>()
}

/// Get the fields that received a new detailed submission in the given
/// submission-id window, across all bases. These are exactly the fields whose
/// consensus needs (re-)evaluating in an incremental jobs run: consensus is a
/// pure function of a field's submissions, so a field with no new submissions
/// cannot change its outcome. Manual edits that create no submission (e.g.
/// disqualifying one) are invisible to this query - that is what the jobs
/// binary's `--full` sweep is for.
pub fn get_fields_with_new_detailed_submissions(
    conn: &mut PgConnection,
    after_id: i64,
    up_to_id: i64,
) -> Result<Vec<FieldRecord>> {
    use diesel::sql_query;
    use diesel::sql_types::BigInt;

    let query = "SELECT DISTINCT ON (f.id) f.*
        FROM submissions s
        JOIN fields f ON f.id = s.field_id
        WHERE s.search_mode = 'detailed' AND s.id > $1 AND s.id <= $2
        ORDER BY f.id ASC";

    let items_private: Vec<FieldPrivate> = sql_query(query)
        .bind::<BigInt, _>(after_id)
        .bind::<BigInt, _>(up_to_id)
        .load(conn)
        .map_err(|e| anyhow!("{e}"))?;

    items_private
        .into_iter()
        .map(private_to_public)
        .collect::<Result<Vec<FieldRecord>>>()
}

/// The `check_level` predicate for a claim query.
///
/// IMPORTANT: the two common bounds are emitted as *literals* rather than
/// `check_level <= $2`, because Postgres can only prove that a query predicate implies a
/// partial index's predicate when the bound is a constant — against a bound parameter it
/// gives up and walks the primary key, filtering as it goes. `check_level = 0` matches
/// the partial index `idx_fields_cl0_id`, which is what makes nice-only claims fast.
///
/// There is deliberately no matching index for `check_level <= 1`: in production that
/// covers 95.6% of `fields`, so the index would be nearly as large as the primary key
/// (~7GB) while excluding almost nothing. The literal is kept anyway because it costs
/// nothing and gives the planner strictly more to work with. Detailed claims avoid
/// needing such an index by scoping to a single chunk instead — `Thin` via
/// `eligible_chunk_cte`, `Next` via `frontier_chunk_cte`.
fn check_level_predicate(maximum_check_level: i32) -> &'static str {
    match maximum_check_level {
        0 => "check_level = 0",
        1 => "check_level <= 1",
        _ => "check_level <= $2",
    }
}

/// Predicate matching fields that are free to claim: never claimed, or claimed longer
/// ago than `$1`.
///
/// Spelled as an explicit `IS NULL` disjunction rather than
/// `COALESCE(last_claim_time, 'epoch'::timestamptz) <= $1`. The two are equivalent for
/// any `$1` after the epoch, which every caller passes, but `COALESCE` over the column
/// is an expression no btree index can answer, so it can only ever be a filter applied
/// after the fact — never an index condition.
///
/// `prefix` is the table qualifier, e.g. `""` or `"f."`.
fn claimable_predicate(prefix: &str) -> String {
    format!("({prefix}last_claim_time IS NULL OR {prefix}last_claim_time <= $1)")
}

/// CTE selecting the frontier chunk: the lowest-id chunk that is both under-explored
/// *and* still has at least one claimable field.
///
/// The `EXISTS` clause is the load-bearing half. Without it this picks the first
/// under-explored chunk whether or not anything in it can be claimed, so once every
/// field in that chunk carries a live claim the whole `Thin` strategy returns nothing
/// and the frontier never advances. That is the steady state under fleet load rather
/// than an edge case: a chunk is 1% of a base, so holding one saturated needs only a
/// few claims per second against the `CLAIM_DURATION_HOURS` expiry. Nor does the chunk
/// exit the under-explored state on its own while that lasts — `checked_detailed` counts
/// *completed* work (`check_level >= 2`) and is written only by the `jobs` binary on a
/// schedule, never by `/submit`.
///
/// The claimability test is one `EXISTS` over `claimable_predicate`'s disjunction, not
/// two `EXISTS` split on it. `EXISTS(A OR B)` and `EXISTS(A) OR EXISTS(B)` are
/// equivalent, and the split form is faster *if* an index covers all four predicate
/// columns, because then each half becomes an index condition. Without such an index the
/// split form is a pessimization, because Postgres evaluates the two in order: on a chunk
/// that has been picked over — every field claimed at least once, some claims now expired
/// — the `IS NULL` half scans the whole chunk to conclude nothing, and only then does the
/// expired half find its row immediately. Measured against production (130M fields):
/// split 78.0ms with the field probe at 48ms, single 35.2ms with the field probe at
/// 0.24ms. The single form short-circuits on the first row matching either branch.
///
/// Parameters: `$1` = maximum claim timestamp, `$2` = maximum check level,
/// `$3` = maximum range size, `$4` = chunk completion cutoff percent.
fn eligible_chunk_cte(check_level_predicate: &str) -> String {
    let claimable = claimable_predicate("f.");
    format!(
        "eligible_chunk AS (
            SELECT c.id
            FROM chunks c
            WHERE CASE
                WHEN $2 = 0 THEN c.checked_niceonly / NULLIF(c.range_size, 0) < $4
                ELSE c.checked_detailed / NULLIF(c.range_size, 0) < $4
            END
              AND EXISTS (
                  SELECT 1
                  FROM fields f
                  WHERE f.chunk_id = c.id
                    AND {claimable}
                    AND f.{check_level_predicate}
                    AND f.range_size <= $3
              )
            ORDER BY c.id ASC
            LIMIT 1
        )"
    )
}

/// CTE selecting the frontier chunk for the `Next` strategy: the lowest-id chunk that
/// still contains a field at or below the requested check level *and* has one free to
/// claim.
///
/// This differs from `eligible_chunk_cte` in its chunk filter, and the difference is the
/// two strategies' semantics. `Thin` explores under-explored chunks, so it filters on
/// the `checked_*` ratio and honors the downsample cutoff. `Next`'s job is the global
/// frontier — the lowest-id claimable field anywhere, including leftovers in chunks the
/// cutoff has retired — so it filters on `minimum_cl`, which is simply "does this chunk
/// contain any field at `check_level <= N`". Chunks partition fields in id order, so the
/// first such chunk with a claimable field contains the globally-first claimable field:
/// scoping to it changes nothing about which field `Next` returns, only what the planner
/// has to read to find it (measured in production: 1,547ms → 32ms, generic plan).
///
/// Trusting `minimum_cl` is sound because it shares a writer with the thing it
/// summarizes: field `check_level`s are advanced only by the `jobs` binary, which
/// recomputes chunk `minimum_cl` in the same run. Between runs the frontier cannot move.
/// Within a run's window a stale `minimum_cl` errs low, which is the safe direction —
/// the `EXISTS` finds nothing in a finished chunk and the CTE steps past it, same as a
/// saturated chunk.
///
/// Parameters: `$1` = maximum claim timestamp, `$2` = maximum check level,
/// `$3` = maximum range size.
fn frontier_chunk_cte(check_level_predicate: &str) -> String {
    let claimable = claimable_predicate("f.");
    format!(
        "frontier_chunk AS (
            SELECT c.id
            FROM chunks c
            WHERE c.minimum_cl <= $2
              AND EXISTS (
                  SELECT 1
                  FROM fields f
                  WHERE f.chunk_id = c.id
                    AND {claimable}
                    AND f.{check_level_predicate}
                    AND f.range_size <= $3
              )
            ORDER BY c.id ASC
            LIMIT 1
        )"
    )
}

/// Finds the next field that matches the criteria, updates `last_claim_time`, and returns it.
/// Returns Ok(None) if no matching fields are found.
#[allow(clippy::too_many_lines)]
pub fn try_claim_field(
    conn: &mut PgConnection,
    claim_strategy: FieldClaimStrategy,
    maximum_timestamp: DateTime<Utc>,
    maximum_check_level: u8,
    maximum_size: u128,
) -> Result<Option<FieldRecord>> {
    use diesel::sql_query;
    use diesel::sql_types::{BigInt, Integer, Numeric, Timestamptz};

    let maximum_check_level = conversions::u8_to_i32(maximum_check_level)?;
    let maximum_size = conversions::u128_to_bigdec(maximum_size)?;
    let maximum_size_clone = maximum_size.clone();

    // Use a single-statement "claim" with row locking to avoid thundering herd / lock contention.
    // `FOR UPDATE SKIP LOCKED` ensures concurrent claimers don't block on the same "next" row.
    let check_level_predicate = check_level_predicate(maximum_check_level);
    let claimable = claimable_predicate("");

    match claim_strategy {
        FieldClaimStrategy::Next => {
            // Get the next available field: the lowest-id claimable one at or below the
            // requested check level.
            //
            // For nice-only claims (cl = 0) the unscoped form is already fast — the
            // `check_level = 0` literal matches the partial index `idx_fields_cl0_id`,
            // which skips completed rows by construction. Keep it.
            //
            // For detailed claims (cl >= 1) no such index exists or is worth having
            // (`check_level <= 1` covers 95.6% of the table in production), so the
            // unscoped form walks the primary key from id 1 and filters every completed
            // field below the frontier — measured at 1.55s and ~1.6GB of buffer reads
            // per claim in production. Scope it to the frontier chunk instead via
            // `frontier_chunk_cte`, which reads only that chunk's fields: 32ms measured,
            // same row returned.
            let query = if maximum_check_level == 0 {
                format!(
                    "WITH candidate AS (
                        SELECT id
                        FROM fields
                        WHERE {claimable}
                          AND {check_level_predicate}
                          AND range_size <= $3
                        ORDER BY id ASC
                        FOR UPDATE SKIP LOCKED
                        LIMIT 1
                    )
                    UPDATE fields f
                    SET last_claim_time = NOW()
                    FROM candidate
                    WHERE f.id = candidate.id
                    RETURNING f.*;"
                )
            } else {
                // `ORDER BY f.chunk_id, f.id` rather than `ORDER BY f.id` is
                // load-bearing, not cosmetic: the two orderings are identical here (the
                // CTE yields exactly one chunk), but a bare `ORDER BY id` lets the
                // planner satisfy the sort by walking the primary key from id 1 with the
                // chunk join as a filter — reintroducing the full 1.3s scan this scoping
                // exists to avoid. Leading with `chunk_id` makes the primary key unable
                // to produce the ordering, so the planner reads the chunk via
                // `idx_fields_chunk_id` and top-1 sorts its ~30k rows. Verified against
                // production under `force_generic_plan`, which is what diesel's prepared
                // statements get.
                let frontier_chunk = frontier_chunk_cte(check_level_predicate);
                let claimable_f = claimable_predicate("f.");
                format!(
                    "WITH {frontier_chunk}, candidate AS (
                        SELECT f.id
                        FROM fields f
                        JOIN frontier_chunk fc ON f.chunk_id = fc.id
                        WHERE {claimable_f}
                          AND f.{check_level_predicate}
                          AND f.range_size <= $3
                        ORDER BY f.chunk_id, f.id ASC
                        FOR UPDATE SKIP LOCKED
                        LIMIT 1
                    )
                    UPDATE fields f
                    SET last_claim_time = NOW()
                    FROM candidate
                    WHERE f.id = candidate.id
                    RETURNING f.*;"
                )
            };

            let result = sql_query(query)
                .bind::<Timestamptz, _>(maximum_timestamp)
                .bind::<Integer, _>(maximum_check_level)
                .bind::<Numeric, _>(maximum_size)
                .get_result::<FieldPrivate>(conn)
                .optional()
                .map_err(|e| anyhow!("{e}"))?;

            match result {
                Some(rec) => private_to_public(rec).map(Some),
                None => Ok(None),
            }
        }
        FieldClaimStrategy::Random => {
            // Pseudorandom strategy: choose a random pivot id and take the next eligible row.
            // If none are found at/after the pivot, wrap around and take the first eligible row.
            //
            // This avoids `ORDER BY RANDOM()`, which requires assigning random values and sorting
            // over the eligible set.
            //
            // Note: Postgres does not allow `FOR UPDATE` with UNION/INTERSECT/EXCEPT, so the
            // wraparound is implemented as a second query if the pivot query finds no rows.
            let query_from_pivot = format!(
                "WITH candidate AS (
                    SELECT id
                    FROM fields
                    WHERE id >= $4
                      AND {claimable}
                      AND {check_level_predicate}
                      AND range_size <= $3
                    ORDER BY id ASC
                    FOR UPDATE SKIP LOCKED
                    LIMIT 1
                )
                UPDATE fields f
                SET last_claim_time = NOW()
                FROM candidate
                WHERE f.id = candidate.id
                RETURNING f.*;"
            );

            let query_wraparound = format!(
                "WITH candidate AS (
                    SELECT id
                    FROM fields
                    WHERE {claimable}
                      AND {check_level_predicate}
                      AND range_size <= $3
                    ORDER BY id ASC
                    FOR UPDATE SKIP LOCKED
                    LIMIT 1
                )
                UPDATE fields f
                SET last_claim_time = NOW()
                FROM candidate
                WHERE f.id = candidate.id
                RETURNING f.*;"
            );

            // Compute a pivot in [1, max_id]. Caller guarantees no id gaps.
            // If max_id is 0 (empty table), use 0 so the pivot branch yields no rows and we wrap.
            let max_id = get_max_field_id(conn)?;
            let pivot: i64 = if max_id == 0 {
                0
            } else {
                let mut rng = rand::rng();
                conversions::u128_to_i64(rng.random_range(1..=max_id)).unwrap_or(0)
            };

            // First attempt: claim from pivot
            let result = sql_query(query_from_pivot)
                .bind::<Timestamptz, _>(maximum_timestamp)
                .bind::<Integer, _>(maximum_check_level)
                .bind::<Numeric, _>(maximum_size)
                .bind::<BigInt, _>(pivot)
                .get_result::<FieldPrivate>(conn)
                .optional()
                .map_err(|e| anyhow!("{e}"))?;

            if let Some(rec) = result {
                return private_to_public(rec).map(Some);
            }

            // Second attempt: wraparound (claim from the beginning)
            let result = sql_query(query_wraparound)
                .bind::<Timestamptz, _>(maximum_timestamp)
                .bind::<Integer, _>(maximum_check_level)
                .bind::<Numeric, _>(maximum_size_clone)
                .get_result::<FieldPrivate>(conn)
                .optional()
                .map_err(|e| anyhow!("{e}"))?;

            match result {
                Some(rec) => private_to_public(rec).map(Some),
                None => Ok(None),
            }
        }
        FieldClaimStrategy::Thin => {
            // First, finds the frontier chunk: the first chunk with less than X% of it
            // checked that still holds a claimable field (see `eligible_chunk_cte`):
            //   When maximum_check_level == 0, use chunk.checked_niceonly
            //   When maximum_check_level >= 1, use chunk.checked_detailed
            // Then find and return a pseudorandom field within that chunk using pivot strategy

            let chunk_completion_cutoff_pct =
                conversions::f32_to_bigdec(DOWNSAMPLE_CUTOFF_PERCENT)?;

            // Single query to get eligible chunk and field ID range within it. The range
            // spans every field in the chunk, not just the claimable ones, so it is only
            // a pivot hint — the wraparound below still finds a claimable field below the
            // pivot when the chunk is nearly exhausted.
            let eligible_chunk = eligible_chunk_cte(check_level_predicate);
            let chunk_info_query = format!(
                "WITH {eligible_chunk}
                SELECT
                    ec.id as chunk_id,
                    MIN(f.id) as min_field_id,
                    MAX(f.id) as max_field_id
                FROM eligible_chunk ec
                JOIN fields f ON f.chunk_id = ec.id
                GROUP BY ec.id"
            );

            #[derive(QueryableByName)]
            #[allow(clippy::items_after_statements, clippy::struct_field_names)]
            struct ChunkInfo {
                #[diesel(sql_type = diesel::sql_types::Integer)]
                chunk_id: i32,
                #[diesel(sql_type = diesel::sql_types::Nullable<BigInt>)]
                min_field_id: Option<i64>,
                #[diesel(sql_type = diesel::sql_types::Nullable<BigInt>)]
                max_field_id: Option<i64>,
            }

            let chunk_info_result: Option<ChunkInfo> = sql_query(chunk_info_query)
                .bind::<Timestamptz, _>(maximum_timestamp)
                .bind::<Integer, _>(maximum_check_level)
                .bind::<Numeric, _>(maximum_size.clone())
                .bind::<Numeric, _>(chunk_completion_cutoff_pct)
                .get_result(conn)
                .optional()
                .map_err(|e| anyhow!("{e}"))?;

            let Some(chunk_info) = chunk_info_result else {
                return Ok(None);
            };

            let (min_id, max_id) = match (chunk_info.min_field_id, chunk_info.max_field_id) {
                (Some(min), Some(max)) if min <= max => (min, max),
                _ => return Ok(None), // Empty chunk
            };

            // Pick a random pivot between min and max field IDs
            let mut rng = rand::rng();
            let pivot = if min_id == max_id {
                min_id
            } else {
                rng.random_range(min_id..=max_id)
            };

            // Attempt to claim from pivot onward
            let query_from_pivot = format!(
                "WITH candidate AS (
                    SELECT id
                    FROM fields
                    WHERE chunk_id = $4
                      AND id >= $5
                      AND {claimable}
                      AND {check_level_predicate}
                      AND range_size <= $3
                    ORDER BY id ASC
                    FOR UPDATE SKIP LOCKED
                    LIMIT 1
                )
                UPDATE fields f
                SET last_claim_time = NOW()
                FROM candidate
                WHERE f.id = candidate.id
                RETURNING f.*;"
            );

            let result = sql_query(query_from_pivot)
                .bind::<Timestamptz, _>(maximum_timestamp)
                .bind::<Integer, _>(maximum_check_level)
                .bind::<Numeric, _>(maximum_size)
                .bind::<Integer, _>(chunk_info.chunk_id)
                .bind::<BigInt, _>(pivot)
                .get_result::<FieldPrivate>(conn)
                .optional()
                .map_err(|e| anyhow!("{e}"))?;

            if let Some(rec) = result {
                return private_to_public(rec).map(Some);
            }

            // Wraparound: try from beginning of chunk
            let query_wraparound = format!(
                "WITH candidate AS (
                    SELECT id
                    FROM fields
                    WHERE chunk_id = $4
                      AND {claimable}
                      AND {check_level_predicate}
                      AND range_size <= $3
                    ORDER BY id ASC
                    FOR UPDATE SKIP LOCKED
                    LIMIT 1
                )
                UPDATE fields f
                SET last_claim_time = NOW()
                FROM candidate
                WHERE f.id = candidate.id
                RETURNING f.*;"
            );

            let result = sql_query(query_wraparound)
                .bind::<Timestamptz, _>(maximum_timestamp)
                .bind::<Integer, _>(maximum_check_level)
                .bind::<Numeric, _>(maximum_size_clone)
                .bind::<Integer, _>(chunk_info.chunk_id)
                .get_result::<FieldPrivate>(conn)
                .optional()
                .map_err(|e| anyhow!("{e}"))?;

            match result {
                Some(rec) => private_to_public(rec).map(Some),
                None => Ok(None),
            }
        }
    }
}

/// Bulk claim multiple fields at once for queue pre-filling.
/// This is much more efficient than calling `try_claim_field` repeatedly.
pub fn bulk_claim_fields(
    conn: &mut PgConnection,
    count: usize,
    maximum_timestamp: DateTime<Utc>,
    maximum_check_level: u8,
    maximum_size: u128,
) -> Result<Vec<FieldRecord>> {
    use diesel::sql_query;
    use diesel::sql_types::{BigInt, Integer, Numeric, Timestamptz};

    let maximum_check_level = conversions::u8_to_i32(maximum_check_level)?;
    let maximum_size = conversions::u128_to_bigdec(maximum_size)?;
    let count_i64 = i64::try_from(count).map_err(|e| anyhow!("{e}"))?;

    let check_level_predicate = check_level_predicate(maximum_check_level);
    let claimable = claimable_predicate("");

    let query = format!(
        "WITH candidates AS (
            SELECT id
            FROM fields
            WHERE {claimable}
              AND {check_level_predicate}
              AND range_size <= $3
            ORDER BY id ASC
            FOR UPDATE SKIP LOCKED
            LIMIT $4
        )
        UPDATE fields f
        SET last_claim_time = NOW()
        FROM candidates
        WHERE f.id = candidates.id
        RETURNING f.*;"
    );

    let results = sql_query(query)
        .bind::<Timestamptz, _>(maximum_timestamp)
        .bind::<Integer, _>(maximum_check_level)
        .bind::<Numeric, _>(maximum_size)
        .bind::<BigInt, _>(count_i64)
        .load::<FieldPrivate>(conn)
        .map_err(|e| anyhow!("{e}"))?;

    results.into_iter().map(private_to_public).collect()
}

/// Bulk claim multiple fields at once from the frontier chunk, using the
/// `Thin` strategy. Mirrors `try_claim_field`'s `Thin` branch but claims up to `count`
/// fields from the chosen chunk in a single statement.
///
/// Only `maximum_check_level >= 1` is supported here; the niceonly (cl=0) bulk path
/// is handled by `bulk_claim_fields`.
pub fn bulk_claim_thin_fields(
    conn: &mut PgConnection,
    count: usize,
    maximum_timestamp: DateTime<Utc>,
    maximum_check_level: u8,
    maximum_size: u128,
) -> Result<Vec<FieldRecord>> {
    use diesel::sql_query;
    use diesel::sql_types::{BigInt, Integer, Numeric, Timestamptz};

    let maximum_check_level = conversions::u8_to_i32(maximum_check_level)?;
    let maximum_size = conversions::u128_to_bigdec(maximum_size)?;
    let count_i64 = i64::try_from(count).map_err(|e| anyhow!("{e}"))?;
    let chunk_completion_cutoff_pct = conversions::f32_to_bigdec(DOWNSAMPLE_CUTOFF_PERCENT)?;

    let check_level_predicate = check_level_predicate(maximum_check_level);
    let claimable = claimable_predicate("f.");

    // Find the frontier chunk, then bulk-claim up to `count` fields within it. We do
    // this in a single statement by joining the eligible chunk back to fields and
    // limiting the candidate set.
    let eligible_chunk = eligible_chunk_cte(check_level_predicate);
    let query = format!(
        "WITH {eligible_chunk}, candidates AS (
            SELECT f.id
            FROM fields f
            JOIN eligible_chunk ec ON f.chunk_id = ec.id
            WHERE {claimable}
              AND f.{check_level_predicate}
              AND f.range_size <= $3
            ORDER BY f.id ASC
            FOR UPDATE SKIP LOCKED
            LIMIT $5
        )
        UPDATE fields f
        SET last_claim_time = NOW()
        FROM candidates
        WHERE f.id = candidates.id
        RETURNING f.*;"
    );

    // Note: bind order is $1=maximum_timestamp, $2=maximum_check_level, $3=maximum_size,
    // $4=chunk_completion_cutoff_pct, $5=count. $1-$4 are fixed by `eligible_chunk_cte`.
    let results = sql_query(query)
        .bind::<Timestamptz, _>(maximum_timestamp)
        .bind::<Integer, _>(maximum_check_level)
        .bind::<Numeric, _>(maximum_size)
        .bind::<Numeric, _>(chunk_completion_cutoff_pct)
        .bind::<BigInt, _>(count_i64)
        .load::<FieldPrivate>(conn)
        .map_err(|e| anyhow!("{e}"))?;

    results.into_iter().map(private_to_public).collect()
}

pub fn get_validation_field(conn: &mut PgConnection) -> Result<ValidationData> {
    use diesel::sql_query;
    use diesel::sql_types::{BigInt, Integer};

    let min_check_level = 2;

    let mut rng = rand::rng();

    // Set pivot between field IDs 10k-50k, which are all base 42-43 and double-checked
    let pivot: i64 = conversions::u128_to_i64(rng.random_range(10_000..50_000)).unwrap_or(1);

    // Try to find a field starting from the pivot
    let query_from_pivot = "SELECT * FROM fields
         WHERE id >= $1
           AND check_level >= $2
           AND canon_submission_id IS NOT NULL
         ORDER BY id ASC
         LIMIT 1";

    let field_result = sql_query(query_from_pivot)
        .bind::<BigInt, _>(pivot)
        .bind::<Integer, _>(min_check_level)
        .get_result::<FieldPrivate>(conn)
        .optional()
        .map_err(|e| anyhow!("{e}"))?;

    // If no field found from pivot, wrap around to the beginning
    let field = if let Some(f) = field_result {
        f
    } else {
        let query_wraparound = "SELECT * FROM fields
             WHERE check_level >= $1
               AND canon_submission_id IS NOT NULL
             ORDER BY id ASC
             LIMIT 1";

        sql_query(query_wraparound)
            .bind::<Integer, _>(min_check_level)
            .get_result::<FieldPrivate>(conn)
            .map_err(|e| anyhow!("{e}"))?
    };

    let field_pub = private_to_public(field)?;

    // Get the canonical submission
    let submission_id = field_pub
        .canon_submission_id
        .ok_or_else(|| anyhow!("Field has no canonical submission"))?;
    let submission = submissions::get_submission_by_id(conn, u128::from(submission_id))?;

    // Convert submission data to simple format for ValidationData
    let unique_distribution = match submission.distribution {
        Some(dist) => distribution_stats::shrink_distribution(&dist),
        None => {
            return Err(anyhow!("Canonical submission has no distribution data"));
        }
    };
    let nice_numbers = number_stats::shrink_numbers(&submission.numbers);

    Ok(ValidationData {
        base: field_pub.base,
        field_id: field_pub.field_id,
        range_start: field_pub.range_start,
        range_end: field_pub.range_end,
        range_size: field_pub.range_size,
        unique_distribution,
        nice_numbers,
    })
}

pub fn get_count_checked_by_range(
    conn: &mut PgConnection,
    in_check_level: u8,
    start: u128,
    end: u128,
) -> Result<u128> {
    use self::fields::dsl::*;
    use diesel::dsl::sum;

    let in_check_level = conversions::u8_to_i32(in_check_level)?;
    let in_range_start = conversions::u128_to_bigdec(start)?;
    let in_range_end = conversions::u128_to_bigdec(end)?;

    let result = fields
        .select(sum(range_size))
        .filter(check_level.ge(in_check_level))
        .filter(range_start.ge(in_range_start))
        .filter(range_end.le(in_range_end))
        .first::<Option<BigDecimal>>(conn)
        .map_err(|e| anyhow!("{e}"))?
        .unwrap_or(BigDecimal::from(0u32));

    conversions::bigdec_to_u128(result)
}

pub fn get_minimum_cl_by_range(conn: &mut PgConnection, start: u128, end: u128) -> Result<u8> {
    use self::fields::dsl::*;
    use diesel::dsl::min;

    let in_range_start = conversions::u128_to_bigdec(start)?;
    let in_range_end = conversions::u128_to_bigdec(end)?;

    let result = fields
        .select(min(check_level))
        .filter(range_start.ge(in_range_start))
        .filter(range_end.le(in_range_end))
        .first::<Option<i32>>(conn)
        .map_err(|e| anyhow!("{e}"))?
        .unwrap_or_default();

    conversions::i32_to_u8(result)
}

pub fn update_field(
    conn: &mut PgConnection,
    row_id: u128,
    update_row: &FieldRecord,
) -> Result<FieldRecord> {
    use self::fields::dsl::*;

    let row_id = conversions::u128_to_i64(row_id)?;
    let update_row = public_to_private(update_row)?;

    let result = diesel::update(fields.filter(id.eq(row_id)))
        .set(&update_row)
        .get_result(conn)
        .map_err(|e| anyhow!("{e}"))?;
    private_to_public(result)
}

pub fn update_field_canon_and_cl(
    conn: &mut PgConnection,
    field_id: u128,
    submission_id: Option<u32>,
    in_check_level: u8,
) -> Result<()> {
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
        .map_err(|e| anyhow!("{e}"))?;

    Ok(())
}

/// Struct to hold chunk statistics from batch query
#[derive(Debug, QueryableByName)]
pub struct ChunkStats {
    #[diesel(sql_type = diesel::sql_types::Integer)]
    pub chunk_id: i32,
    #[diesel(sql_type = diesel::sql_types::Integer)]
    pub minimum_cl: i32,
    #[diesel(sql_type = diesel::sql_types::Numeric)]
    pub checked_niceonly: BigDecimal,
    #[diesel(sql_type = diesel::sql_types::Numeric)]
    pub checked_detailed: BigDecimal,
}

/// Get statistics for all chunks in a base in a single query.
/// This is much more efficient than querying each chunk individually.
pub fn get_chunk_stats_batch(conn: &mut PgConnection, base: u32) -> Result<Vec<ChunkStats>> {
    use diesel::sql_query;
    use diesel::sql_types::Integer;

    let base = conversions::u32_to_i32(base)?;

    let query = "
        SELECT
            chunk_id,
            MIN(check_level) as minimum_cl,
            COALESCE(SUM(CASE WHEN check_level >= 1 THEN range_size ELSE 0 END), 0) as checked_niceonly,
            COALESCE(SUM(CASE WHEN check_level >= 2 THEN range_size ELSE 0 END), 0) as checked_detailed
        FROM fields
        WHERE base_id = $1 AND chunk_id IS NOT NULL
        GROUP BY chunk_id
        ORDER BY chunk_id;
    ";

    sql_query(query)
        .bind::<Integer, _>(base)
        .load(conn)
        .map_err(|e| anyhow!("{e}"))
}

/// Get statistics for a specific set of chunks. The incremental jobs run uses
/// this to recompute only the chunks that received new submissions, instead of
/// aggregating over every field of the base.
pub fn get_chunk_stats_for_chunks(
    conn: &mut PgConnection,
    chunk_ids: &[u32],
) -> Result<Vec<ChunkStats>> {
    use diesel::sql_query;
    use diesel::sql_types::{Array, Integer};

    if chunk_ids.is_empty() {
        return Ok(Vec::new());
    }
    let chunk_ids = chunk_ids
        .iter()
        .map(|&id| conversions::u32_to_i32(id))
        .collect::<Result<Vec<i32>>>()?;

    let query = "
        SELECT
            chunk_id,
            MIN(check_level) as minimum_cl,
            COALESCE(SUM(CASE WHEN check_level >= 1 THEN range_size ELSE 0 END), 0) as checked_niceonly,
            COALESCE(SUM(CASE WHEN check_level >= 2 THEN range_size ELSE 0 END), 0) as checked_detailed
        FROM fields
        WHERE chunk_id = ANY($1)
        GROUP BY chunk_id
        ORDER BY chunk_id;
    ";

    sql_query(query)
        .bind::<Array<Integer>, _>(chunk_ids)
        .load(conn)
        .map_err(|e| anyhow!("{e}"))
}
