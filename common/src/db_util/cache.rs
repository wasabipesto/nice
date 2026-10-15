//! The website's search caches: the leaderboard (`cache_search_leaderboard`)
//! and the activity chart (`cache_search_rate_daily`), plus the notable numbers
//! chart's points.
//!
//! The first two are derived from `cache_search_daily_totals`: all-time totals
//! of qualified submissions' range sizes per (date, mode, user). Each scheduled
//! run adds only the submissions that arrived since the last run to it, so a
//! run reads those rows and not all of `submissions` and `fields`. Dates are
//! `DATE(submit_time)` in the session time zone, which is UTC: diesel sets it
//! on connect.

use super::*;
use diesel::connection::SimpleConnection;
use diesel::sql_types::BigInt;

/// The most submission ids one transaction adds to the daily totals.
///
/// An hourly run adds about 210k. Only a run after missed runs reaches the cap,
/// and it then adds several windows, each committed with its cursor. With
/// October 2026 statistics the planner joined windows of up to 5M ids through
/// the primary keys, and switched to a full scan of `fields` by 20M. The window
/// transaction also turns hash and merge joins off, which rules that plan out
/// whatever the statistics say.
pub const MAX_CACHE_WINDOW: i64 = 2_000_000;

/// Add the daily totals of the qualified submissions with ids in `($1, $2]`.
///
/// The full rebuild uses this too, so both paths count the same way.
const ADD_TO_TOTALS: &str = "
    INSERT INTO cache_search_daily_totals (date, search_mode, username, total_range)
    SELECT DATE(s.submit_time), s.search_mode, s.username, SUM(f.range_size)
    FROM submissions s
    JOIN fields f ON f.id = s.field_id
    WHERE s.id > $1 AND s.id <= $2 AND NOT s.disqualified
    GROUP BY 1, 2, 3
    ON CONFLICT (date, search_mode, username)
    DO UPDATE SET total_range = cache_search_daily_totals.total_range + EXCLUDED.total_range";

/// Count the submissions that arrived since the last run into the daily totals.
///
/// Adds every qualified submission with an id above the cursor
/// (`job_state.cache_last_submission_id`) and at most `up_to_id`, in windows of
/// at most [`MAX_CACHE_WINDOW`] ids. Each window's totals commit together with
/// the cursor, so every submission is counted exactly once, however a run ends.
/// Returns the windows added as `(after_id, up_to_id)` pairs; none if the cursor
/// is already at or past `up_to_id`.
///
/// Changes to submissions that are already counted, such as disqualifying one,
/// are not seen here. [`rebuild_search_totals`] (`--full`) picks them up.
pub fn update_search_totals(conn: &mut PgConnection, up_to_id: i64) -> Result<Vec<(i64, i64)>> {
    let mut windows = Vec::new();
    while let Some(window) = add_totals_window(conn, up_to_id)? {
        windows.push(window);
    }
    Ok(windows)
}

/// Add the next window to the daily totals and advance the cursor past it, in
/// one transaction. `None` if the cursor has already reached `up_to_id`.
fn add_totals_window(conn: &mut PgConnection, up_to_id: i64) -> Result<Option<(i64, i64)>> {
    conn.transaction::<_, anyhow::Error, _>(|conn| {
        // Locks the cursor until commit: a concurrent run waits here, then
        // reads the cursor this one leaves.
        let after = job_state::get_cache_cursor(conn)?.ok_or_else(|| {
            anyhow!(
                "the search totals have never been built (job_state.cache_last_submission_id is NULL); run `just jobs-full`"
            )
        })?;
        if after >= up_to_id {
            return Ok(None);
        }
        let up_to = up_to_id.min(after.saturating_add(MAX_CACHE_WINDOW));

        // Keep the plan a nested loop over the two primary keys whatever the
        // statistics say. `SET LOCAL` ends with the transaction.
        conn.batch_execute("SET LOCAL enable_hashjoin = off; SET LOCAL enable_mergejoin = off")?;
        diesel::sql_query(ADD_TO_TOTALS)
            .bind::<BigInt, _>(after)
            .bind::<BigInt, _>(up_to)
            .execute(conn)?;
        job_state::set_cache_cursor(conn, up_to)?;
        Ok(Some((after, up_to)))
    })
}

/// Rebuild the daily totals from every qualified submission up to `up_to_id`,
/// and set the cursor to it.
///
/// One full read of `submissions` and `fields`. This is how `--full` picks up
/// changes that create no submission, such as a disqualification.
pub fn rebuild_search_totals(conn: &mut PgConnection, up_to_id: i64) -> Result<()> {
    conn.transaction::<_, anyhow::Error, _>(|conn| {
        // Setting the cursor first takes its lock, so an incremental run waits
        // for the rebuild instead of adding to totals about to be replaced.
        job_state::set_cache_cursor(conn, up_to_id)?;
        diesel::sql_query("DELETE FROM cache_search_daily_totals").execute(conn)?;
        // Submission ids start at 1.
        diesel::sql_query(ADD_TO_TOTALS)
            .bind::<BigInt, _>(0)
            .bind::<BigInt, _>(up_to_id)
            .execute(conn)?;
        Ok(())
    })
}

/// Rebuild the website's search caches from the daily totals, and the notable
/// numbers chart's points, in one transaction.
///
/// Both search caches cover the submissions counted into the totals so far.
pub fn refresh_search_caches(conn: &mut PgConnection) -> Result<()> {
    conn.transaction(|conn| {
        diesel::sql_query("DELETE FROM cache_search_rate_daily").execute(conn)?;

        // The same days as `submit_time >= CURRENT_DATE - INTERVAL '90 days'`.
        diesel::sql_query(
            "INSERT INTO cache_search_rate_daily (date, search_mode, username, total_range)
            SELECT date, search_mode, username, total_range
            FROM cache_search_daily_totals
            WHERE date >= CURRENT_DATE - 90",
        )
        .execute(conn)?;

        diesel::sql_query("DELETE FROM cache_search_leaderboard").execute(conn)?;

        diesel::sql_query(
            "INSERT INTO cache_search_leaderboard (search_mode, username, total_range)
            SELECT search_mode, username, SUM(total_range)
            FROM cache_search_daily_totals
            GROUP BY search_mode, username",
        )
        .execute(conn)?;

        refresh_notable_numbers(conn)?;

        Ok::<_, diesel::result::Error>(())
    })
    .map_err(|e| anyhow!("{e}"))
}

/// Rebuild the plot-ready point set behind the website's notable numbers chart.
///
/// The chart was drawn from `bases.numbers` directly - every base's full top-10k
/// list, ~81k points, of which over 99% land on a pixel another point already
/// covers. This keeps only the visually distinguishable ones, a few hundred rows.
///
/// The thinning is exact in y and in colour: a point's niceness is exactly
/// `num_uniques / base` and its colour is `base - num_uniques`, so both are
/// fixed by `(base, num_uniques)` and only x needs quantizing. It is bucketed at
/// 70 buckets per decade, against the ~61 pixels per decade the chart has, so a
/// dropped point is always within a pixel of the one kept in its place - and it
/// only gets finer as the search reaches larger numbers and the log axis
/// stretches.
///
/// Points within two uniques of a nice number skip bucketing and are all kept.
/// Their key space cannot collide with the bucketed one because `off_by` is
/// constant within a `(base, num_uniques)` group, so a group is either entirely
/// kept or entirely bucketed.
///
/// `ON CONFLICT DO NOTHING` guards the `(base, number)` key: a number should
/// never appear twice in one base's list, but if it ever did, the duplicate is
/// a point drawn on top of an identical one - not worth failing the whole
/// scheduled run over.
fn refresh_notable_numbers(conn: &mut PgConnection) -> Result<(), diesel::result::Error> {
    diesel::sql_query("DELETE FROM cache_notable_numbers").execute(conn)?;

    diesel::sql_query(
        "INSERT INTO cache_notable_numbers (base, number, num_uniques, off_by, niceness)
        SELECT DISTINCT ON (b.id, n.num_uniques, n.bucket)
            b.id, n.number, n.num_uniques, b.id - n.num_uniques, n.niceness
        FROM bases b
        CROSS JOIN LATERAL (
            SELECT
                (e->>'number')::decimal AS number,
                (e->>'num_uniques')::int AS num_uniques,
                (e->>'niceness')::real AS niceness,
                CASE
                    WHEN b.id - (e->>'num_uniques')::int <= 2
                        THEN (e->>'number')::decimal
                    ELSE round(log((e->>'number')::decimal) * 70)
                END AS bucket
            FROM jsonb_array_elements(b.numbers) e
            WHERE (e->>'number')::decimal > 0
        ) n
        ORDER BY b.id, n.num_uniques, n.bucket, n.number
        ON CONFLICT DO NOTHING",
    )
    .execute(conn)?;

    Ok(())
}
