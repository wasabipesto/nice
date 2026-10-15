//! The scheduled jobs' state, in one row (`id = 1`) created by
//! `schema/migrations/2026-08-20_job_state.sql`.
//!
//! The watermark is the highest submission id already processed. The jobs
//! binary reads it to scope an incremental run to new submissions, and
//! advances it after a successful run (full or incremental). A run that fails
//! partway leaves the watermark untouched, so the next run redoes that window -
//! all of the work keyed on it is idempotent.
//!
//! The search totals' cursor (`cache_last_submission_id`, added by
//! `schema/migrations/2026-10-15_incremental_search_caches.sql`) is the highest
//! submission id counted into `cache_search_daily_totals`. Counting is not
//! idempotent, so this one is exact instead: it only moves in the same
//! transaction as the totals it covers. See `cache.rs`.

use super::*;
use anyhow::anyhow;

const MISSING_ROW: &str =
    "job_state has no row with id = 1; apply schema/migrations/2026-08-20_job_state.sql";

const MISSING_CACHE_CURSOR: &str = "job_state has no cache_last_submission_id column; apply schema/migrations/2026-10-15_incremental_search_caches.sql";

#[derive(QueryableByName)]
struct WatermarkRow {
    #[diesel(sql_type = diesel::sql_types::BigInt)]
    last_processed_submission_id: i64,
}

#[derive(QueryableByName)]
struct CacheCursorRow {
    #[diesel(sql_type = diesel::sql_types::Nullable<diesel::sql_types::BigInt>)]
    cache_last_submission_id: Option<i64>,
}

/// Get the current watermark. Errors if the migration has not been applied.
pub fn get_watermark(conn: &mut PgConnection) -> Result<i64> {
    use diesel::sql_query;

    let row: Option<WatermarkRow> =
        sql_query("SELECT last_processed_submission_id FROM job_state WHERE id = 1")
            .get_result(conn)
            .optional()
            .map_err(|e| anyhow!("{e}"))?;
    row.map(|r| r.last_processed_submission_id)
        .ok_or_else(|| anyhow!(MISSING_ROW))
}

/// Advance the watermark. Callers should only do this after every side effect
/// of processing submissions up to `id` has been committed.
pub fn set_watermark(conn: &mut PgConnection, id: i64) -> Result<()> {
    use diesel::sql_query;
    use diesel::sql_types::BigInt;

    let updated = sql_query(
        "UPDATE job_state SET last_processed_submission_id = $1, updated_at = NOW() WHERE id = 1",
    )
    .bind::<BigInt, _>(id)
    .execute(conn)
    .map_err(|e| anyhow!("{e}"))?;
    if updated == 1 {
        Ok(())
    } else {
        Err(anyhow!(MISSING_ROW))
    }
}

/// Get the search totals' cursor: the highest submission id counted into
/// `cache_search_daily_totals`, or `None` if the totals have never been built.
///
/// Locks the row until the end of the transaction (`FOR UPDATE`), so a run
/// that reads the cursor in order to advance it holds off any other run until
/// it commits. Errors if the migration has not been applied.
pub fn get_cache_cursor(conn: &mut PgConnection) -> Result<Option<i64>> {
    use diesel::sql_query;

    let row: Option<CacheCursorRow> =
        sql_query("SELECT cache_last_submission_id FROM job_state WHERE id = 1 FOR UPDATE")
            .get_result(conn)
            .optional()
            .map_err(cache_cursor_error)?;
    row.map(|r| r.cache_last_submission_id)
        .ok_or_else(|| anyhow!(MISSING_ROW))
}

/// Set the search totals' cursor. Only call this in the same transaction as
/// the change to `cache_search_daily_totals` that it records.
pub fn set_cache_cursor(conn: &mut PgConnection, id: i64) -> Result<()> {
    use diesel::sql_query;
    use diesel::sql_types::BigInt;

    let updated = sql_query("UPDATE job_state SET cache_last_submission_id = $1 WHERE id = 1")
        .bind::<BigInt, _>(id)
        .execute(conn)
        .map_err(cache_cursor_error)?;
    if updated == 1 {
        Ok(())
    } else {
        Err(anyhow!(MISSING_ROW))
    }
}

/// Name the migration when a query fails because the cursor's column is missing.
fn cache_cursor_error(e: diesel::result::Error) -> anyhow::Error {
    if let diesel::result::Error::DatabaseError(_, info) = &e
        && info.message().contains("cache_last_submission_id")
    {
        return anyhow!(MISSING_CACHE_CURSOR);
    }
    e.into()
}
