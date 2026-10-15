//! Integration tests for the incremental search caches against a real PostgreSQL.
//!
//! The leaderboard and activity chart used to be rebuilt every run by two joins over all
//! of `submissions` and `fields`. They are now derived from daily totals that each run
//! only adds its new submissions to. That is right only if every submission is counted
//! exactly once, on the right day, and only while it is qualified. These tests check the
//! incremental path against the two old queries, kept here as the reference.
//!
//! Skipped unless `NICE_TEST_DATABASE_URL` is set. That is deliberately *not*
//! `DATABASE_URL`: these tests truncate tables, so they must never run against a
//! database anyone cares about. Point it at a scratch database with `schema/schema.sql`
//! loaded:
//!
//! ```text
//! NICE_TEST_DATABASE_URL=postgres://nice:nice@localhost:55432/nice \
//!     cargo test -p nice_common --features database --test search_caches
//! ```
//!
//! The steps build on each other's data, so they run in order from one `#[test]`.
//!
//! Requires `--features database`; without it this file compiles to nothing.

#![cfg(feature = "database")]

use diesel::pg::PgConnection;
use diesel::prelude::*;
use diesel::sql_query;
use nice_common::db_util::cache::{
    MAX_CACHE_WINDOW, rebuild_search_totals, refresh_search_caches, update_search_totals,
};
use nice_common::db_util::job_state::get_cache_cursor;

/// A row of `cache_search_rate_daily` or `cache_search_daily_totals`, or of the old query
/// that filled the former. As text, so the comparison is exact, down to the decimals'
/// scale.
#[derive(QueryableByName, Debug, PartialEq, Clone)]
struct DailyRow {
    #[diesel(sql_type = diesel::sql_types::Text)]
    date: String,
    #[diesel(sql_type = diesel::sql_types::Text)]
    search_mode: String,
    #[diesel(sql_type = diesel::sql_types::Text)]
    username: String,
    #[diesel(sql_type = diesel::sql_types::Text)]
    total_range: String,
}

/// A row of `cache_search_leaderboard`, or of the old query that filled it.
#[derive(QueryableByName, Debug, PartialEq, Clone)]
struct LeaderboardRow {
    #[diesel(sql_type = diesel::sql_types::Text)]
    search_mode: String,
    #[diesel(sql_type = diesel::sql_types::Text)]
    username: String,
    #[diesel(sql_type = diesel::sql_types::Text)]
    total_range: String,
}

/// A server setting, as `current_setting` reports it.
#[derive(QueryableByName)]
struct Setting {
    #[diesel(sql_type = diesel::sql_types::Text)]
    value: String,
}

/// Both website tables, as the website reads them.
#[derive(Debug, PartialEq, Clone)]
struct Caches {
    rate: Vec<DailyRow>,
    leaderboard: Vec<LeaderboardRow>,
}

/// The old rate query from `refresh_search_caches`, verbatim but for one change:
/// `submissions` is limited to the ids counted so far, since newer ones are not in the
/// totals yet.
fn old_rate(conn: &mut PgConnection, counted_through: i64) -> Vec<DailyRow> {
    sql_query(format!(
        "SELECT date::text AS date, search_mode, username, total_range::text AS total_range
        FROM (
            SELECT
                DATE(s.submit_time) AS date,
                s.search_mode,
                s.username,
                SUM(f.range_size) AS total_range
            FROM (SELECT * FROM submissions WHERE id <= {counted_through}) s
            JOIN fields f ON s.field_id = f.id
            WHERE s.submit_time >= CURRENT_DATE - INTERVAL '90 days'
              AND s.disqualified = false
            GROUP BY DATE(s.submit_time), s.search_mode, s.username
        ) old
        ORDER BY date, search_mode, username"
    ))
    .load(conn)
    .expect("old rate query")
}

/// The old leaderboard query, with the same one change.
fn old_leaderboard(conn: &mut PgConnection, counted_through: i64) -> Vec<LeaderboardRow> {
    sql_query(format!(
        "SELECT search_mode, username, total_range::text AS total_range
        FROM (
            SELECT
                s.search_mode,
                s.username,
                SUM(f.range_size) AS total_range
            FROM (SELECT * FROM submissions WHERE id <= {counted_through}) s
            JOIN fields f ON s.field_id = f.id
            WHERE s.disqualified = false
            GROUP BY s.search_mode, s.username
        ) old
        ORDER BY search_mode, username"
    ))
    .load(conn)
    .expect("old leaderboard query")
}

fn old_caches(conn: &mut PgConnection, counted_through: i64) -> Caches {
    Caches {
        rate: old_rate(conn, counted_through),
        leaderboard: old_leaderboard(conn, counted_through),
    }
}

fn caches(conn: &mut PgConnection) -> Caches {
    Caches {
        rate: sql_query(
            "SELECT date::text AS date, search_mode, username, total_range::text AS total_range
            FROM cache_search_rate_daily ORDER BY date, search_mode, username",
        )
        .load(conn)
        .expect("read rate cache"),
        leaderboard: sql_query(
            "SELECT search_mode, username, total_range::text AS total_range
            FROM cache_search_leaderboard ORDER BY search_mode, username",
        )
        .load(conn)
        .expect("read leaderboard cache"),
    }
}

fn totals(conn: &mut PgConnection) -> Vec<DailyRow> {
    sql_query(
        "SELECT date::text AS date, search_mode, username, total_range::text AS total_range
        FROM cache_search_daily_totals ORDER BY date, search_mode, username",
    )
    .load(conn)
    .expect("read daily totals")
}

fn cursor(conn: &mut PgConnection) -> Option<i64> {
    get_cache_cursor(conn).expect("read cursor")
}

/// What one scheduled run does to the search caches: count up to `up_to_id`, rebuild.
fn incremental_run(conn: &mut PgConnection, up_to_id: i64) -> Vec<(i64, i64)> {
    let windows = update_search_totals(conn, up_to_id).expect("update totals");
    refresh_search_caches(conn).expect("refresh caches");
    windows
}

/// What a `--full` run does to them.
fn full_run(conn: &mut PgConnection, up_to_id: i64) {
    rebuild_search_totals(conn, up_to_id).expect("rebuild totals");
    refresh_search_caches(conn).expect("refresh caches");
}

/// Field sizes, by field id. The third is far past `u64` and exact only as a decimal,
/// so any rounding on the way through the totals shows up.
const FIELD_SIZES: [&str; 3] = ["1000", "2500", "123456789012345678901234567890"];

/// Submission times, as SQL. The session time zone is UTC (diesel sets it on connect),
/// as in the jobs binary, so `CURRENT_DATE::timestamptz` is the start of today there.
const MIDNIGHT: &str = "CURRENT_DATE::timestamptz";
const JUST_BEFORE_MIDNIGHT: &str = "CURRENT_DATE::timestamptz - INTERVAL '1 microsecond'";
const YESTERDAY_EVENING: &str = "CURRENT_DATE::timestamptz - INTERVAL '3 hours'";
const THIS_MORNING: &str = "CURRENT_DATE::timestamptz + INTERVAL '30 minutes'";
/// The first instant the activity chart includes: the start of the day 90 days ago.
const EDGE: &str = "(CURRENT_DATE - 90)::timestamptz";
const JUST_BEFORE_EDGE: &str = "(CURRENT_DATE - 90)::timestamptz - INTERVAL '1 microsecond'";
const LONG_AGO: &str = "(CURRENT_DATE - 400)::timestamptz + INTERVAL '12 hours'";

/// One submission: id, field id, mode, user, time, and whether it is disqualified.
type Submission = (i64, i64, &'static str, &'static str, &'static str, bool);

fn reset_fixture(conn: &mut PgConnection) {
    sql_query(
        "TRUNCATE submissions, claims, fields, chunks, bases, cache_search_daily_totals,
            cache_search_rate_daily, cache_search_leaderboard
        RESTART IDENTITY CASCADE",
    )
    .execute(conn)
    .expect("truncate fixture tables");
    sql_query(
        "INSERT INTO job_state (id, last_processed_submission_id, cache_last_submission_id)
        VALUES (1, 0, 0)
        ON CONFLICT (id) DO UPDATE SET cache_last_submission_id = 0",
    )
    .execute(conn)
    .expect("reset cursor");

    sql_query("INSERT INTO bases (id, range_start, range_end, range_size) VALUES (40, 0, 1, 1)")
        .execute(conn)
        .expect("insert base");
    for (id, size) in (1..).zip(FIELD_SIZES) {
        sql_query(format!(
            "INSERT INTO fields (id, base_id, range_start, range_end, range_size)
            VALUES ({id}, 40, 0, {size}, {size})"
        ))
        .execute(conn)
        .expect("insert field");
    }
    sql_query(
        "INSERT INTO claims (id, field_id, search_mode, user_ip)
        VALUES (1, 1, 'detailed', '127.0.0.1')",
    )
    .execute(conn)
    .expect("insert claim");
}

/// Insert submissions with explicit ids, so windows and their edges are exact.
fn submit(conn: &mut PgConnection, submissions: &[Submission]) {
    for (id, field, mode, user, when, disqualified) in submissions {
        sql_query(format!(
            "INSERT INTO submissions (id, claim_id, field_id, search_mode, submit_time,
                elapsed_secs, username, user_ip, client_version, disqualified)
            VALUES ({id}, 1, {field}, '{mode}', {when}, 1, '{user}', '127.0.0.1', 'test',
                {disqualified})"
        ))
        .execute(conn)
        .expect("insert submission");
    }
}

/// Three hourly runs: the first before midnight, the second straddling it with two
/// submissions still in flight, the third after it with a straggler for yesterday.
/// Each run's tables must match the old queries over the submissions counted so far.
fn incremental_runs_match_the_old_queries(conn: &mut PgConnection) {
    reset_fixture(conn);

    submit(
        conn,
        &[
            // The leaderboard counts these two; the activity chart does not.
            (1, 1, "detailed", "alice", LONG_AGO, false),
            (2, 2, "detailed", "alice", JUST_BEFORE_EDGE, false),
            // The first day the chart includes, twice on one key in one window.
            (3, 1, "detailed", "alice", EDGE, false),
            (4, 3, "niceonly", "bob", YESTERDAY_EVENING, false),
            (5, 1, "niceonly", "bob", YESTERDAY_EVENING, true),
            (6, 2, "detailed", "carol", JUST_BEFORE_MIDNIGHT, false),
            (7, 1, "detailed", "alice", EDGE, false),
        ],
    );
    assert_eq!(incremental_run(conn, 7), vec![(0, 7)]);
    assert_eq!(cursor(conn), Some(7));
    assert_eq!(caches(conn), old_caches(conn, 7), "run 1");
    assert_eq!(caches(conn).rate.len(), 3, "days 90 and 1 ago, 3 keys");

    submit(
        conn,
        &[
            // Adds to a key the first run created.
            (8, 2, "detailed", "carol", JUST_BEFORE_MIDNIGHT, false),
            // One microsecond later, so a different day.
            (9, 1, "detailed", "carol", MIDNIGHT, false),
            (10, 3, "niceonly", "bob", MIDNIGHT, true),
            (11, 2, "niceonly", "bob", YESTERDAY_EVENING, false),
            // Still being written: within the safety margin, so held back.
            (12, 1, "detailed", "dave", THIS_MORNING, false),
            (13, 2, "detailed", "alice", EDGE, false),
        ],
    );
    assert_eq!(incremental_run(conn, 11), vec![(7, 11)]);
    assert_eq!(caches(conn), old_caches(conn, 11), "run 2");
    assert_ne!(
        old_caches(conn, 11),
        old_caches(conn, 13),
        "the held-back submissions must change the tables once counted, or run 2 shows nothing"
    );

    submit(
        conn,
        &[
            // A straggler: a newer id than today's submissions, but timed yesterday.
            (14, 1, "detailed", "carol", JUST_BEFORE_MIDNIGHT, false),
            (15, 3, "detailed", "dave", THIS_MORNING, false),
            // A late arrival for a day outside the chart: the leaderboard only.
            (16, 1, "niceonly", "erin", JUST_BEFORE_EDGE, false),
        ],
    );
    assert_eq!(
        incremental_run(conn, 16),
        vec![(11, 16)],
        "picks up the two held back"
    );
    assert_eq!(
        caches(conn),
        old_caches(conn, i64::MAX),
        "run 3: everything counted, so the old queries apply unchanged"
    );
}

/// Running again without new submissions, or with a stale boundary, changes nothing.
fn repeated_and_stale_runs_count_nothing_twice(conn: &mut PgConnection) {
    let before = (caches(conn), totals(conn));

    assert!(incremental_run(conn, 16).is_empty(), "nothing new");
    assert!(
        incremental_run(conn, 10).is_empty(),
        "a boundary below the cursor"
    );
    assert_eq!(cursor(conn), Some(16), "the cursor never moves back");
    assert_eq!((caches(conn), totals(conn)), before);
}

/// A run after missed runs takes several windows. Each commits with its cursor, so
/// every submission is counted once, including the ones on window edges.
fn windows_past_the_cap_count_each_submission_once(conn: &mut PgConnection) {
    let start = cursor(conn).expect("cursor set");
    let last = start + 2 * MAX_CACHE_WINDOW + 500_000;
    submit(
        conn,
        &[
            (start + 1, 1, "detailed", "frank", THIS_MORNING, false),
            // The last id of the first window and the first of the second.
            (
                start + MAX_CACHE_WINDOW,
                2,
                "detailed",
                "frank",
                THIS_MORNING,
                false,
            ),
            (
                start + MAX_CACHE_WINDOW + 1,
                3,
                "niceonly",
                "frank",
                YESTERDAY_EVENING,
                false,
            ),
            (last, 1, "detailed", "alice", EDGE, false),
        ],
    );

    assert_eq!(
        incremental_run(conn, last),
        vec![
            (start, start + MAX_CACHE_WINDOW),
            (start + MAX_CACHE_WINDOW, start + 2 * MAX_CACHE_WINDOW),
            (start + 2 * MAX_CACHE_WINDOW, last),
        ]
    );
    assert_eq!(cursor(conn), Some(last));
    assert_eq!(caches(conn), old_caches(conn, i64::MAX));

    // The windows' planner settings ended with their transactions.
    for name in ["enable_hashjoin", "enable_mergejoin"] {
        let setting: Vec<Setting> = sql_query(format!("SELECT current_setting('{name}') AS value"))
            .load(conn)
            .expect("read setting");
        assert_eq!(setting[0].value, "on", "{name}");
    }
}

/// `--full` rebuilds the same totals from scratch, and is how a disqualification of an
/// already-counted submission reaches the caches.
fn full_rebuild_matches_and_applies_disqualifications(conn: &mut PgConnection) {
    let last = cursor(conn).expect("cursor set");
    let before = (caches(conn), totals(conn));
    full_run(conn, last);
    assert_eq!(
        (caches(conn), totals(conn)),
        before,
        "full rebuild == incremental"
    );
    assert_eq!(cursor(conn), Some(last));

    // Disqualify the big field's counted submission. An incremental run cannot see it.
    sql_query("UPDATE submissions SET disqualified = true WHERE id = 15")
        .execute(conn)
        .expect("disqualify");
    assert!(incremental_run(conn, last).is_empty());
    assert_eq!(caches(conn), before.0, "unchanged until --full");
    assert_ne!(caches(conn), old_caches(conn, i64::MAX));

    full_run(conn, last);
    assert_eq!(
        caches(conn),
        old_caches(conn, i64::MAX),
        "--full applies it"
    );

    // A rebuild to an earlier boundary than the cursor (a --full run that started
    // before the cursor last moved) is exact too, and the next run carries on from it.
    full_run(conn, 11);
    assert_eq!(cursor(conn), Some(11));
    assert_eq!(caches(conn), old_caches(conn, 11));
    assert_eq!(incremental_run(conn, last).len(), 3);
    assert_eq!(caches(conn), old_caches(conn, i64::MAX));
}

/// A cursor that was never set must not be read as 0, which would count every
/// submission in history window by window. Only `--full` may build the totals.
fn unbuilt_totals_need_a_full_run(conn: &mut PgConnection) {
    let last = cursor(conn).expect("cursor set");
    sql_query("UPDATE job_state SET cache_last_submission_id = NULL WHERE id = 1")
        .execute(conn)
        .expect("clear cursor");

    let err = update_search_totals(conn, last).expect_err("no cursor");
    assert!(err.to_string().contains("jobs-full"), "{err}");

    full_run(conn, last);
    assert_eq!(cursor(conn), Some(last));
    assert_eq!(caches(conn), old_caches(conn, i64::MAX));
}

/// Deployed before its migration, the jobs binary must say which migration is missing.
fn missing_column_names_the_migration(conn: &mut PgConnection) {
    let result = conn.transaction::<(), diesel::result::Error, _>(|conn| {
        sql_query("ALTER TABLE job_state DROP COLUMN cache_last_submission_id").execute(conn)?;
        let err = get_cache_cursor(conn).expect_err("no column");
        assert!(
            err.to_string()
                .contains("2026-10-15_incremental_search_caches.sql"),
            "{err}"
        );
        Err(diesel::result::Error::RollbackTransaction)
    });
    assert!(matches!(
        result,
        Err(diesel::result::Error::RollbackTransaction)
    ));
    assert!(cursor(conn).is_some(), "the column is back after rollback");
}

#[test]
fn search_caches_against_postgres() {
    let Ok(url) = std::env::var("NICE_TEST_DATABASE_URL") else {
        eprintln!("skipping: NICE_TEST_DATABASE_URL is not set");
        return;
    };
    let mut conn = PgConnection::establish(&url).expect("connect to the test database");

    incremental_runs_match_the_old_queries(&mut conn);
    repeated_and_stale_runs_count_nothing_twice(&mut conn);
    windows_past_the_cap_count_each_submission_once(&mut conn);
    full_rebuild_matches_and_applies_disqualifications(&mut conn);
    unbuilt_totals_need_a_full_run(&mut conn);
    missing_column_names_the_migration(&mut conn);
}
