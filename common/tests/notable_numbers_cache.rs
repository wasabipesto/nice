//! Integration tests for the notable numbers chart cache against a real PostgreSQL.
//!
//! The refresh is hand-written SQL doing a `DISTINCT ON` over a lateral jsonb expansion,
//! none of which the compiler checks. Its correctness is also not obvious by reading:
//! whether a point survives depends on a bucketing rule with a deliberate exception, and
//! getting it wrong silently drops points from the website's chart rather than failing.
//!
//! Skipped unless `NICE_TEST_DATABASE_URL` is set. That is deliberately *not*
//! `DATABASE_URL`: these tests truncate tables, so they must never run against a
//! database anyone cares about. Point it at a scratch database with `schema/schema.sql`
//! loaded:
//!
//! ```text
//! NICE_TEST_DATABASE_URL=postgres://nice:nice@localhost:55432/nice \
//!     cargo test -p nice_common --test notable_numbers_cache
//! ```
//!
//! Requires `--features database`; without it this file compiles to nothing.

#![cfg(feature = "database")]

use diesel::pg::PgConnection;
use diesel::prelude::*;
use diesel::sql_query;
use nice_common::db_util::cache::refresh_search_caches;

#[derive(QueryableByName, Debug, PartialEq)]
struct CachedRow {
    #[diesel(sql_type = diesel::sql_types::Integer)]
    base: i32,
    // As text: `number` is DECIMAL, and reading it as one would pull bigdecimal in
    // as a dev-dependency purely to call `to_string` on it.
    #[diesel(sql_type = diesel::sql_types::Text)]
    number: String,
    #[diesel(sql_type = diesel::sql_types::Integer)]
    num_uniques: i32,
    #[diesel(sql_type = diesel::sql_types::Integer)]
    off_by: i32,
    #[diesel(sql_type = diesel::sql_types::Float)]
    niceness: f32,
}

/// Two numbers land in the same x bucket when their ratio is under ~1.017, since the
/// bucket is `round(log10(number) * 70)`. These exercise both sides of that.
///
/// Every number here is distinct: `(base, number)` is the primary key, because one
/// number has exactly one unique-count within a base.
const SAME_BUCKET_A: u64 = 1_000_000;
const SAME_BUCKET_B: u64 = 1_010_000;
const OTHER_BUCKET: u64 = 2_000_000;
/// A second pair sharing a bucket, one decade up.
const KEPT_PAIR_A: u64 = 10_000_000;
const KEPT_PAIR_B: u64 = 10_100_000;

fn point(number: u64, num_uniques: u32, base: u32) -> String {
    format!(
        r#"{{"number": {number}, "num_uniques": {num_uniques}, "base": {base},
             "niceness": {}}}"#,
        f64::from(num_uniques) / f64::from(base)
    )
}

fn load_fixture(conn: &mut PgConnection, bases: &[(u32, Vec<String>)]) {
    sql_query("TRUNCATE fields, chunks, bases RESTART IDENTITY CASCADE")
        .execute(conn)
        .expect("truncate fixture tables");

    for (base, points) in bases {
        sql_query(format!(
            "INSERT INTO bases (id, range_start, range_end, range_size, numbers)
             VALUES ({base}, 0, 1, 1, '[{}]'::jsonb)",
            points.join(",")
        ))
        .execute(conn)
        .expect("insert base");
    }
}

fn cached(conn: &mut PgConnection) -> Vec<CachedRow> {
    sql_query(
        "SELECT base, number::text AS number, num_uniques, off_by, niceness
         FROM cache_notable_numbers ORDER BY base, number",
    )
    .load(conn)
    .expect("read cache")
}

fn numbers_for(rows: &[CachedRow], base: i32, num_uniques: i32) -> Vec<String> {
    rows.iter()
        .filter(|r| r.base == base && r.num_uniques == num_uniques)
        .map(|r| r.number.clone())
        .collect()
}

#[test]
fn notable_numbers_cache() {
    let Ok(url) = std::env::var("NICE_TEST_DATABASE_URL") else {
        eprintln!("NICE_TEST_DATABASE_URL not set, skipping");
        return;
    };
    let conn = &mut PgConnection::establish(&url).expect("connect to test database");

    load_fixture(
        conn,
        &[
            // The only known nice number: off_by 0, must always survive.
            (10, vec![point(69, 10, 10)]),
            (
                40,
                vec![
                    // off_by 2 - inside the exception, so both survive despite
                    // sharing a bucket.
                    point(KEPT_PAIR_A, 38, 40),
                    point(KEPT_PAIR_B, 38, 40),
                    // off_by 3 - bucketed, so the first two collapse to one.
                    point(SAME_BUCKET_A, 37, 40),
                    point(SAME_BUCKET_B, 37, 40),
                    point(OTHER_BUCKET, 37, 40),
                ],
            ),
            // A base that has been created but never searched.
            (41, vec![]),
        ],
    );

    refresh_search_caches(conn).expect("refresh should succeed");
    let rows = cached(conn);

    assert_eq!(
        numbers_for(&rows, 10, 10),
        vec!["69"],
        "the one known nice number must survive"
    );
    assert_eq!(
        numbers_for(&rows, 40, 38),
        vec![KEPT_PAIR_A.to_string(), KEPT_PAIR_B.to_string()],
        "off_by <= 2 skips bucketing, so both points in the shared bucket survive"
    );
    assert_eq!(
        numbers_for(&rows, 40, 37),
        vec![SAME_BUCKET_A.to_string(), OTHER_BUCKET.to_string()],
        "off_by > 2 is bucketed: the shared bucket keeps its lowest number only"
    );
    assert!(
        rows.iter().all(|r| r.base != 41),
        "a base with no numbers contributes no rows"
    );

    let nice_number = rows.iter().find(|r| r.base == 10).expect("base 10 row");
    assert_eq!(nice_number.off_by, 0, "off_by is base - num_uniques");
    assert!(
        (nice_number.niceness - 1.0).abs() < f32::EPSILON,
        "niceness carries through unchanged"
    );

    // The refresh replaces the table wholesale, so running it again must not
    // duplicate, drop or reorder anything.
    refresh_search_caches(conn).expect("second refresh should succeed");
    assert_eq!(cached(conn), rows, "refresh is idempotent");

    // A base whose points all vanish from the source must lose its rows too.
    load_fixture(conn, &[(10, vec![])]);
    refresh_search_caches(conn).expect("third refresh should succeed");
    assert!(
        cached(conn).is_empty(),
        "rows do not outlive the numbers they came from"
    );
}
