use super::*;

pub fn refresh_search_caches(conn: &mut PgConnection) -> Result<()> {
    diesel::sql_query("DELETE FROM cache_search_rate_daily")
        .execute(conn)
        .map_err(|e| anyhow!("{e}"))?;

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
    .execute(conn)
    .map_err(|e| anyhow!("{e}"))?;

    diesel::sql_query("DELETE FROM cache_search_leaderboard")
        .execute(conn)
        .map_err(|e| anyhow!("{e}"))?;

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
    .execute(conn)
    .map_err(|e| anyhow!("{e}"))?;

    Ok(())
}
