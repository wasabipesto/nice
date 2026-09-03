//! Integration tests for the field claim queries against a real PostgreSQL.
//!
//! The claim path is hand-written SQL with positional bindings, so it is not checked by
//! the compiler at all — a renumbered parameter or a typo'd column only shows up when a
//! client asks for work. These tests execute the real functions against the real schema.
//!
//! They are skipped unless `NICE_TEST_DATABASE_URL` is set. That is deliberately *not*
//! `DATABASE_URL`: every test here truncates the tables and hands out claims, so it must
//! never run against a database anyone cares about. Point it at a scratch database with
//! `schema/schema.sql` loaded:
//!
//! ```text
//! docker run --rm -d --name nice-test-pg -e POSTGRES_PASSWORD=nice \
//!     -e POSTGRES_USER=nice -e POSTGRES_DB=nice -p 55432:5432 postgres:17
//! psql postgres://nice:nice@localhost:55432/nice < schema/schema.sql
//! NICE_TEST_DATABASE_URL=postgres://nice:nice@localhost:55432/nice \
//!     cargo test -p nice_common --test claim_queries
//! ```
//!
//! The claim queries scan the whole `chunks` table (the frontier is global, not
//! per-base), so the tests must run serially against a database they fully own. They
//! share one `#[test]` entry point for that reason.
//!
//! Requires `--features database`, which is where `db_util` and diesel live; without it
//! this file compiles to nothing.

#![cfg(feature = "database")]

use chrono::{DateTime, TimeDelta, Utc};
use diesel::pg::PgConnection;
use diesel::prelude::*;
use diesel::sql_query;
use nice_common::db_util::fields::{
    bulk_claim_fields, bulk_claim_next_fields, bulk_claim_thin_fields, try_claim_field,
};
use nice_common::{FieldClaimStrategy, FieldRecord};

const FIELDS_PER_CHUNK: i64 = 10;
const FIELD_SIZE: u128 = 1_000;
/// Every claim query is called with this, matching `CLAIM_DURATION_HOURS` in the API.
fn claim_cutoff() -> DateTime<Utc> {
    Utc::now() - TimeDelta::hours(1)
}

/// Build three chunks of `FIELDS_PER_CHUNK` fields each, all under-explored
/// (`checked_detailed = 0`) and all unclaimed at `check_level = 1`.
///
/// Chunk ids and field ids ascend together, so "chunk 1" is the frontier.
fn reset_fixture(conn: &mut PgConnection) {
    // TRUNCATE rather than DELETE so the serial ids restart and chunk/field ordering is
    // reproducible across the cases below.
    sql_query("TRUNCATE fields, chunks, bases RESTART IDENTITY CASCADE")
        .execute(conn)
        .expect("truncate fixture tables");

    sql_query(
        "INSERT INTO bases (id, range_start, range_end, range_size)
         VALUES (40, 0, 30000, 30000)",
    )
    .execute(conn)
    .expect("insert base");

    for chunk in 0..3i64 {
        let chunk_start = chunk * FIELDS_PER_CHUNK * FIELD_SIZE as i64;
        let chunk_size = FIELDS_PER_CHUNK * FIELD_SIZE as i64;
        sql_query(format!(
            "INSERT INTO chunks (base_id, range_start, range_end, range_size, checked_detailed)
             VALUES (40, {chunk_start}, {}, {chunk_size}, 0)",
            chunk_start + chunk_size
        ))
        .execute(conn)
        .expect("insert chunk");

        // check_level = 1: nice-only pass done, detailed pass pending. This is what the
        // detailed frontier actually looks like.
        sql_query(format!(
            "INSERT INTO fields (base_id, chunk_id, range_start, range_end, range_size, check_level)
             SELECT 40, {}, {chunk_start} + g * {FIELD_SIZE}, {chunk_start} + (g + 1) * {FIELD_SIZE}, {FIELD_SIZE}, 1
             FROM generate_series(0, {}) AS g",
            chunk + 1,
            FIELDS_PER_CHUNK - 1
        ))
        .execute(conn)
        .expect("insert fields");
    }
}

/// Mark every field in `chunk_id` as claimed just now, i.e. held by another client whose
/// claim has not expired.
fn saturate_chunk(conn: &mut PgConnection, chunk_id: i64) {
    sql_query(format!(
        "UPDATE fields SET last_claim_time = NOW() WHERE chunk_id = {chunk_id}"
    ))
    .execute(conn)
    .expect("saturate chunk");
}

fn chunk_ids_of(fields: &[FieldRecord]) -> Vec<u32> {
    let mut ids: Vec<u32> = fields.iter().filter_map(|f| f.chunk_id).collect();
    ids.sort_unstable();
    ids.dedup();
    ids
}

/// The regression from issue #98: with the frontier chunk fully claimed, the bulk claim
/// must move on to the next under-explored chunk instead of returning nothing.
fn bulk_thin_advances_past_saturated_chunk(conn: &mut PgConnection) {
    reset_fixture(conn);
    saturate_chunk(conn, 1);

    let claimed = bulk_claim_thin_fields(conn, 5, claim_cutoff(), 1, FIELD_SIZE)
        .expect("bulk claim should execute");

    assert_eq!(
        claimed.len(),
        5,
        "expected a full batch from the next chunk"
    );
    assert_eq!(
        chunk_ids_of(&claimed),
        vec![2],
        "claims must come from chunk 2; chunk 1 is fully held by another client"
    );
}

/// Two saturated chunks in a row, to show the advance is not a single-step special case.
fn bulk_thin_skips_multiple_saturated_chunks(conn: &mut PgConnection) {
    reset_fixture(conn);
    saturate_chunk(conn, 1);
    saturate_chunk(conn, 2);

    let claimed = bulk_claim_thin_fields(conn, 5, claim_cutoff(), 1, FIELD_SIZE)
        .expect("bulk claim should execute");

    assert_eq!(chunk_ids_of(&claimed), vec![3]);
}

/// An expired claim is claimable again, so a chunk whose claims have aged out is still
/// the frontier and must not be skipped.
fn bulk_thin_reclaims_expired_fields(conn: &mut PgConnection) {
    reset_fixture(conn);
    sql_query("UPDATE fields SET last_claim_time = NOW() - INTERVAL '2 hours' WHERE chunk_id = 1")
        .execute(conn)
        .expect("expire chunk 1 claims");

    let claimed = bulk_claim_thin_fields(conn, 5, claim_cutoff(), 1, FIELD_SIZE)
        .expect("bulk claim should execute");

    assert_eq!(
        chunk_ids_of(&claimed),
        vec![1],
        "expired claims are free again, so chunk 1 is still the frontier"
    );
}

/// A chunk past the downsample cutoff is not the frontier even when it has free fields.
fn bulk_thin_skips_explored_chunk(conn: &mut PgConnection) {
    reset_fixture(conn);
    // DOWNSAMPLE_CUTOFF_PERCENT is 0.2; half the chunk's range counts as explored.
    sql_query("UPDATE chunks SET checked_detailed = range_size / 2 WHERE id = 1")
        .execute(conn)
        .expect("mark chunk 1 explored");

    let claimed = bulk_claim_thin_fields(conn, 5, claim_cutoff(), 1, FIELD_SIZE)
        .expect("bulk claim should execute");

    assert_eq!(chunk_ids_of(&claimed), vec![2]);
}

/// With every chunk saturated there is genuinely nothing to hand out. Returning nothing
/// is the correct answer — the API turns this into a 503. It must not fall back to
/// re-issuing a field another client holds.
fn bulk_thin_returns_empty_when_everything_is_claimed(conn: &mut PgConnection) {
    reset_fixture(conn);
    for chunk in 1..=3 {
        saturate_chunk(conn, chunk);
    }

    let claimed = bulk_claim_thin_fields(conn, 5, claim_cutoff(), 1, FIELD_SIZE)
        .expect("bulk claim should execute");

    assert!(claimed.is_empty(), "got {} fields", claimed.len());
}

/// `try_claim_field`'s `Thin` branch selects the frontier chunk with the same CTE, so it
/// must advance too — otherwise the API's direct-claim fallback fails where the queue
/// refill would have succeeded.
fn try_claim_thin_advances_past_saturated_chunk(conn: &mut PgConnection) {
    reset_fixture(conn);
    saturate_chunk(conn, 1);

    let claimed = try_claim_field(
        conn,
        FieldClaimStrategy::Thin,
        claim_cutoff(),
        1,
        FIELD_SIZE,
    )
    .expect("thin claim should execute")
    .expect("a field in chunk 2 is free");

    assert_eq!(claimed.chunk_id, Some(2));
}

/// The `Thin` branch picks a random pivot inside the chunk and wraps around if nothing
/// is free above it. With a single free field the wraparound is exercised roughly half
/// the time, so repeat enough to hit both paths.
fn try_claim_thin_finds_the_last_free_field(conn: &mut PgConnection) {
    for _ in 0..20 {
        reset_fixture(conn);
        saturate_chunk(conn, 1);
        // Free exactly one field, in the middle of the chunk.
        sql_query("UPDATE fields SET last_claim_time = NULL WHERE chunk_id = 1 AND id = 5")
            .execute(conn)
            .expect("free one field");

        let claimed = try_claim_field(
            conn,
            FieldClaimStrategy::Thin,
            claim_cutoff(),
            1,
            FIELD_SIZE,
        )
        .expect("thin claim should execute")
        .expect("field 5 is free");

        assert_eq!(
            claimed.field_id, 5,
            "the one free field in the frontier chunk must be found from any pivot"
        );
    }
}

fn try_claim_thin_returns_none_when_everything_is_claimed(conn: &mut PgConnection) {
    reset_fixture(conn);
    for chunk in 1..=3 {
        saturate_chunk(conn, chunk);
    }

    let claimed = try_claim_field(
        conn,
        FieldClaimStrategy::Thin,
        claim_cutoff(),
        1,
        FIELD_SIZE,
    )
    .expect("thin claim should execute");

    assert!(claimed.is_none());
}

/// Mark every field in `chunk_id` completed (`check_level = 2`) and update the chunk's
/// `minimum_cl` to match, the way the `jobs` binary would after processing submissions.
fn complete_chunk(conn: &mut PgConnection, chunk_id: i64) {
    sql_query(format!(
        "UPDATE fields SET check_level = 2 WHERE chunk_id = {chunk_id}"
    ))
    .execute(conn)
    .expect("complete chunk fields");
    sql_query(format!(
        "UPDATE chunks SET minimum_cl = 2 WHERE id = {chunk_id}"
    ))
    .execute(conn)
    .expect("update chunk minimum_cl");
}

/// The detailed `Next` claim is scoped to the frontier chunk via `minimum_cl`, so it
/// must skip chunks whose fields are all completed without reading them.
fn next_advances_past_completed_chunks(conn: &mut PgConnection) {
    reset_fixture(conn);
    complete_chunk(conn, 1);

    let claimed = try_claim_field(
        conn,
        FieldClaimStrategy::Next,
        claim_cutoff(),
        1,
        FIELD_SIZE,
    )
    .expect("next claim should execute")
    .expect("chunk 2 has claimable fields");

    assert_eq!(
        claimed.field_id, 11,
        "the frontier is the first field of the first non-completed chunk"
    );
}

/// A completed chunk followed by a fully-claimed one: the frontier CTE must step past
/// both, for different reasons (`minimum_cl` excludes the first, the `EXISTS` the
/// second).
fn next_skips_completed_then_saturated_chunks(conn: &mut PgConnection) {
    reset_fixture(conn);
    complete_chunk(conn, 1);
    saturate_chunk(conn, 2);

    let claimed = try_claim_field(
        conn,
        FieldClaimStrategy::Next,
        claim_cutoff(),
        1,
        FIELD_SIZE,
    )
    .expect("next claim should execute")
    .expect("chunk 3 has claimable fields");

    assert_eq!(
        claimed.field_id, 21,
        "chunk 3's first field is the frontier"
    );
}

/// `minimum_cl` is maintained by the `jobs` binary, so it can lag the fields within a
/// run's window. A stale value errs low, and the claim must treat that as "advance",
/// not "return the completed work" and not "give up".
fn next_tolerates_stale_minimum_cl(conn: &mut PgConnection) {
    reset_fixture(conn);
    // Fields completed, but the chunk's minimum_cl still says 0 — jobs hasn't run yet.
    sql_query("UPDATE fields SET check_level = 2 WHERE chunk_id = 1")
        .execute(conn)
        .expect("complete chunk 1 fields only");

    let claimed = try_claim_field(
        conn,
        FieldClaimStrategy::Next,
        claim_cutoff(),
        1,
        FIELD_SIZE,
    )
    .expect("next claim should execute")
    .expect("chunk 2 has claimable fields");

    assert_eq!(
        claimed.field_id, 11,
        "a stale minimum_cl must not stall the frontier or resurface completed work"
    );
}

/// The recheck strategy (`Next` with `max_check_level = 2`) exists to revisit completed
/// fields, so a completed chunk is exactly what it should claim from.
fn next_recheck_claims_completed_work(conn: &mut PgConnection) {
    reset_fixture(conn);
    complete_chunk(conn, 1);

    let claimed = try_claim_field(
        conn,
        FieldClaimStrategy::Next,
        claim_cutoff(),
        2,
        FIELD_SIZE,
    )
    .expect("recheck claim should execute")
    .expect("chunk 1's completed fields are claimable at cl<=2");

    assert_eq!(
        claimed.field_id, 1,
        "recheck starts from the lowest completed field, not the cl<=1 frontier"
    );
}

fn next_returns_none_when_everything_is_claimed(conn: &mut PgConnection) {
    reset_fixture(conn);
    for chunk in 1..=3 {
        saturate_chunk(conn, chunk);
    }

    let claimed = try_claim_field(
        conn,
        FieldClaimStrategy::Next,
        claim_cutoff(),
        1,
        FIELD_SIZE,
    )
    .expect("next claim should execute");

    assert!(
        claimed.is_none(),
        "no frontier chunk has a claimable field, so Next must return nothing"
    );
}

/// The non-`Thin` strategies share the `check_level` and claimable predicates, so this
/// is mostly a binding smoke test across the literal (`cl<=1`) and parameterized
/// (`cl<=2`) predicate branches and both `Next` query shapes.
fn unscoped_strategies_execute(conn: &mut PgConnection) {
    reset_fixture(conn);

    let next = try_claim_field(
        conn,
        FieldClaimStrategy::Next,
        claim_cutoff(),
        1,
        FIELD_SIZE,
    )
    .expect("next claim should execute")
    .expect("field 1 is free");
    assert_eq!(next.field_id, 1, "Next takes the lowest free id");

    let random = try_claim_field(
        conn,
        FieldClaimStrategy::Random,
        claim_cutoff(),
        1,
        FIELD_SIZE,
    )
    .expect("random claim should execute");
    assert!(random.is_some());

    // check_level = 2 takes the `check_level <= $2` parameterized branch.
    reset_fixture(conn);
    let recheck = try_claim_field(
        conn,
        FieldClaimStrategy::Next,
        claim_cutoff(),
        2,
        FIELD_SIZE,
    )
    .expect("recheck claim should execute");
    assert!(recheck.is_some());

    // And the nice-only paths, which use the `check_level = 0` literal: the bulk refill
    // and the unscoped `Next` fallback (cl = 0 keeps the partial-index query shape
    // rather than the chunk-scoped one).
    reset_fixture(conn);
    sql_query("UPDATE fields SET check_level = 0")
        .execute(conn)
        .expect("reset check levels");
    let niceonly = bulk_claim_fields(conn, 7, claim_cutoff(), 0, u128::MAX)
        .expect("bulk claim should execute");
    assert_eq!(niceonly.len(), 7);
    let niceonly_next =
        try_claim_field(conn, FieldClaimStrategy::Next, claim_cutoff(), 0, u128::MAX)
            .expect("niceonly next claim should execute")
            .expect("unclaimed cl=0 fields remain");
    assert_eq!(
        niceonly_next.field_id, 8,
        "the unscoped cl=0 Next continues where the bulk claim left off"
    );
}

fn ids_of(fields: &[FieldRecord]) -> Vec<u128> {
    fields.iter().map(|f| f.field_id).collect()
}

/// The bulk `Next` claim is the frontier in order: consecutive batches continue where
/// the last one stopped, exactly as a sequence of single `Next` claims would.
fn bulk_next_returns_the_frontier_in_order(conn: &mut PgConnection) {
    reset_fixture(conn);

    let first = bulk_claim_next_fields(conn, 4, claim_cutoff(), 1, FIELD_SIZE)
        .expect("bulk next should execute");
    assert_eq!(ids_of(&first), vec![1, 2, 3, 4]);

    let second = bulk_claim_next_fields(conn, 4, claim_cutoff(), 1, FIELD_SIZE)
        .expect("bulk next should execute");
    assert_eq!(ids_of(&second), vec![5, 6, 7, 8]);

    // A single claim slots in after the batches, not before them.
    let single = try_claim_field(
        conn,
        FieldClaimStrategy::Next,
        claim_cutoff(),
        1,
        FIELD_SIZE,
    )
    .expect("next claim should execute")
    .expect("field 9 is free");
    assert_eq!(single.field_id, 9);
}

/// A batch never spans chunks: at the end of the frontier chunk it comes up short, and
/// the following batch starts the next chunk. Callers must not read a short batch as
/// "nothing left".
fn bulk_next_batches_do_not_span_chunks(conn: &mut PgConnection) {
    reset_fixture(conn);

    let first = bulk_claim_next_fields(conn, 7, claim_cutoff(), 1, FIELD_SIZE)
        .expect("bulk next should execute");
    assert_eq!(ids_of(&first), vec![1, 2, 3, 4, 5, 6, 7]);

    let short = bulk_claim_next_fields(conn, 7, claim_cutoff(), 1, FIELD_SIZE)
        .expect("bulk next should execute");
    assert_eq!(
        ids_of(&short),
        vec![8, 9, 10],
        "chunk 1 has three fields left"
    );

    let next_chunk = bulk_claim_next_fields(conn, 7, claim_cutoff(), 1, FIELD_SIZE)
        .expect("bulk next should execute");
    assert_eq!(ids_of(&next_chunk), vec![11, 12, 13, 14, 15, 16, 17]);
}

/// Same frontier rules as the single `Next` claim: a completed chunk is skipped via
/// `minimum_cl`, a saturated one via the `EXISTS`.
fn bulk_next_skips_completed_and_saturated_chunks(conn: &mut PgConnection) {
    reset_fixture(conn);
    complete_chunk(conn, 1);
    saturate_chunk(conn, 2);

    let claimed = bulk_claim_next_fields(conn, 5, claim_cutoff(), 1, FIELD_SIZE)
        .expect("bulk next should execute");
    assert_eq!(chunk_ids_of(&claimed), vec![3]);
    assert_eq!(ids_of(&claimed), vec![21, 22, 23, 24, 25]);
}

/// Expired claims are claimable again and keep their place in the frontier.
fn bulk_next_reclaims_expired_fields(conn: &mut PgConnection) {
    reset_fixture(conn);
    sql_query("UPDATE fields SET last_claim_time = NOW() - INTERVAL '2 hours' WHERE chunk_id = 1")
        .execute(conn)
        .expect("expire chunk 1 claims");

    let claimed = bulk_claim_next_fields(conn, 3, claim_cutoff(), 1, FIELD_SIZE)
        .expect("bulk next should execute");
    assert_eq!(ids_of(&claimed), vec![1, 2, 3]);
}

/// The recheck strategy (cl<=2) batches completed work from the lowest id up, like its
/// single-claim counterpart.
fn bulk_next_recheck_claims_completed_work(conn: &mut PgConnection) {
    reset_fixture(conn);
    complete_chunk(conn, 1);

    let claimed = bulk_claim_next_fields(conn, 3, claim_cutoff(), 2, FIELD_SIZE)
        .expect("bulk recheck should execute");
    assert_eq!(ids_of(&claimed), vec![1, 2, 3]);

    // The cl<=1 frontier is untouched by it and still starts at chunk 2.
    let frontier = bulk_claim_next_fields(conn, 1, claim_cutoff(), 1, FIELD_SIZE)
        .expect("bulk next should execute");
    assert_eq!(ids_of(&frontier), vec![11]);
}

/// The niceonly level has its own bulk path; asking this one for it is a caller bug.
fn bulk_next_rejects_the_niceonly_level(conn: &mut PgConnection) {
    reset_fixture(conn);
    assert!(bulk_claim_next_fields(conn, 3, claim_cutoff(), 0, FIELD_SIZE).is_err());
}

/// A claim must not be handed to two clients at once — the property the removed
/// `maximum_timestamp = Utc::now()` fallback violated.
fn claims_are_not_duplicated(conn: &mut PgConnection) {
    reset_fixture(conn);

    let mut seen = Vec::new();
    // 3 chunks x 10 fields, drawn 4 at a time: the last batches must come up short
    // rather than re-issuing anything.
    for _ in 0..10 {
        let batch = bulk_claim_thin_fields(conn, 4, claim_cutoff(), 1, FIELD_SIZE)
            .expect("bulk claim should execute");
        seen.extend(batch.iter().map(|f| f.field_id));
    }

    let mut distinct = seen.clone();
    distinct.sort_unstable();
    distinct.dedup();
    assert_eq!(
        seen.len(),
        distinct.len(),
        "a field was handed out more than once"
    );
    assert_eq!(
        distinct.len() as i64,
        FIELDS_PER_CHUNK * 3,
        "every field should be claimable exactly once"
    );

    // And the same for the bulk Next path, whose batches end short at chunk edges.
    reset_fixture(conn);
    let mut seen = Vec::new();
    for _ in 0..12 {
        let batch = bulk_claim_next_fields(conn, 4, claim_cutoff(), 1, FIELD_SIZE)
            .expect("bulk next should execute");
        seen.extend(batch.iter().map(|f| f.field_id));
    }
    let mut distinct = seen.clone();
    distinct.sort_unstable();
    distinct.dedup();
    assert_eq!(
        seen.len(),
        distinct.len(),
        "bulk Next handed a field out twice"
    );
    assert_eq!(distinct.len() as i64, FIELDS_PER_CHUNK * 3);
    assert_eq!(
        seen, distinct,
        "bulk Next must hand fields out in frontier order"
    );
}

#[test]
fn claim_queries_against_postgres() {
    let Ok(url) = std::env::var("NICE_TEST_DATABASE_URL") else {
        eprintln!("skipping: NICE_TEST_DATABASE_URL is not set");
        return;
    };
    let mut conn = PgConnection::establish(&url).expect("connect to the test database");

    bulk_thin_advances_past_saturated_chunk(&mut conn);
    bulk_thin_skips_multiple_saturated_chunks(&mut conn);
    bulk_thin_reclaims_expired_fields(&mut conn);
    bulk_thin_skips_explored_chunk(&mut conn);
    bulk_thin_returns_empty_when_everything_is_claimed(&mut conn);
    bulk_next_returns_the_frontier_in_order(&mut conn);
    bulk_next_batches_do_not_span_chunks(&mut conn);
    bulk_next_skips_completed_and_saturated_chunks(&mut conn);
    bulk_next_reclaims_expired_fields(&mut conn);
    bulk_next_recheck_claims_completed_work(&mut conn);
    bulk_next_rejects_the_niceonly_level(&mut conn);
    try_claim_thin_advances_past_saturated_chunk(&mut conn);
    try_claim_thin_finds_the_last_free_field(&mut conn);
    try_claim_thin_returns_none_when_everything_is_claimed(&mut conn);
    next_advances_past_completed_chunks(&mut conn);
    next_skips_completed_then_saturated_chunks(&mut conn);
    next_tolerates_stale_minimum_cl(&mut conn);
    next_recheck_claims_completed_work(&mut conn);
    next_returns_none_when_everything_is_claimed(&mut conn);
    unscoped_strategies_execute(&mut conn);
    claims_are_not_duplicated(&mut conn);
}
