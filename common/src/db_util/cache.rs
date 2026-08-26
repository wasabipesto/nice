use super::*;

pub fn refresh_search_caches(conn: &mut PgConnection) -> Result<()> {
    conn.transaction(|conn| {
        diesel::sql_query("DELETE FROM cache_search_rate_daily").execute(conn)?;

        diesel::sql_query(
            "INSERT INTO cache_search_rate_daily (date, search_mode, username, total_range)
            SELECT
                DATE(s.submit_time) AS date,
                s.search_mode,
                s.username,
                SUM(f.range_size) AS total_range
            FROM submissions s
            JOIN fields f ON s.field_id = f.id
            WHERE s.submit_time >= CURRENT_DATE - INTERVAL '90 days'
              AND s.disqualified = false
            GROUP BY DATE(s.submit_time), s.search_mode, s.username",
        )
        .execute(conn)?;

        diesel::sql_query("DELETE FROM cache_search_leaderboard").execute(conn)?;

        diesel::sql_query(
            "INSERT INTO cache_search_leaderboard (search_mode, username, total_range)
            SELECT
                s.search_mode,
                s.username,
                SUM(f.range_size) AS total_range
            FROM submissions s
            JOIN fields f ON s.field_id = f.id
            WHERE s.disqualified = false
            GROUP BY s.search_mode, s.username",
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
