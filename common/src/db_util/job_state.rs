//! The scheduled-jobs watermark: the highest submission id already processed.
//!
//! One row (`id = 1`), created by `schema/migrations/2026-08-20_job_state.sql`.
//! The jobs binary reads it to scope an incremental run to new submissions,
//! and advances it after a successful run (full or incremental). A run that
//! fails partway leaves the watermark untouched, so the next run redoes that
//! window - all of the work keyed on it is idempotent.

use super::*;
use anyhow::anyhow;

#[derive(QueryableByName)]
struct WatermarkRow {
    #[diesel(sql_type = diesel::sql_types::BigInt)]
    last_processed_submission_id: i64,
}

/// Get the current watermark. Errors if the migration has not been applied.
pub fn get_watermark(conn: &mut PgConnection) -> Result<i64> {
    use diesel::sql_query;

    let row: Option<WatermarkRow> =
        sql_query("SELECT last_processed_submission_id FROM job_state WHERE id = 1")
            .get_result(conn)
            .optional()
            .map_err(|e| anyhow!("{e}"))?;
    row.map(|r| r.last_processed_submission_id).ok_or_else(|| {
        anyhow!(
            "job_state has no row with id = 1; apply schema/migrations/2026-08-20_job_state.sql"
        )
    })
}

/// Advance the watermark. Callers should only do this after every side effect
/// of processing submissions up to `id` has been committed.
pub fn set_watermark(conn: &mut PgConnection, id: i64) -> Result<()> {
    use diesel::sql_query;
    use diesel::sql_types::BigInt;

    let updated =
        sql_query("UPDATE job_state SET last_processed_submission_id = $1, updated_at = NOW() WHERE id = 1")
            .bind::<BigInt, _>(id)
            .execute(conn)
            .map_err(|e| anyhow!("{e}"))?;
    if updated == 1 {
        Ok(())
    } else {
        Err(anyhow!(
            "job_state has no row with id = 1; apply schema/migrations/2026-08-20_job_state.sql"
        ))
    }
}
