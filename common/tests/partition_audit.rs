//! Integration test for `db_util::audit` against a real PostgreSQL.
//!
//! Same harness and gating as `claim_queries.rs`: skipped unless
//! `NICE_TEST_DATABASE_URL` points at a scratch database with
//! `schema/schema.sql` loaded (see that file's header for the docker
//! one-liner). The fixture is truncated on every run.

#![cfg(feature = "database")]

use diesel::pg::PgConnection;
use diesel::prelude::*;
use diesel::sql_query;
use nice_common::BaseRecord;
use nice_common::db_util::audit::{
    audit_base, audit_chunks_partition, audit_fields_partition, audit_fields_within_chunks,
};

const FIELDS_PER_CHUNK: i64 = 10;
const FIELD_SIZE: i64 = 1_000;
const CHUNKS: i64 = 3;
const BASE_START: i64 = 47;

fn base_end() -> i64 {
    BASE_START + CHUNKS * FIELDS_PER_CHUNK * FIELD_SIZE
}

/// One base with three chunks of ten fields each, laid out exactly as the
/// generators would: contiguous, half-open, sizes consistent.
fn reset_fixture(conn: &mut PgConnection) {
    sql_query("TRUNCATE fields, chunks, bases RESTART IDENTITY CASCADE")
        .execute(conn)
        .expect("truncate fixture tables");
    sql_query(format!(
        "INSERT INTO bases (id, range_start, range_end, range_size)
         VALUES (40, {BASE_START}, {}, {})",
        base_end(),
        base_end() - BASE_START
    ))
    .execute(conn)
    .expect("insert base");
    for chunk in 0..CHUNKS {
        let chunk_size = FIELDS_PER_CHUNK * FIELD_SIZE;
        let chunk_start = BASE_START + chunk * chunk_size;
        sql_query(format!(
            "INSERT INTO chunks (base_id, range_start, range_end, range_size)
             VALUES (40, {chunk_start}, {}, {chunk_size})",
            chunk_start + chunk_size
        ))
        .execute(conn)
        .expect("insert chunk");
        sql_query(format!(
            "INSERT INTO fields (base_id, chunk_id, range_start, range_end, range_size)
             SELECT 40, {}, {chunk_start} + g * {FIELD_SIZE}, {chunk_start} + (g + 1) * {FIELD_SIZE}, {FIELD_SIZE}
             FROM generate_series(0, {}) AS g",
            chunk + 1,
            FIELDS_PER_CHUNK - 1
        ))
        .execute(conn)
        .expect("insert fields");
    }
}

fn base_record() -> BaseRecord {
    BaseRecord {
        base: 40,
        range_start: BASE_START as u128,
        range_end: base_end() as u128,
        range_size: (base_end() - BASE_START) as u128,
        checked_detailed: 0,
        checked_niceonly: 0,
        minimum_cl: 0,
        niceness_mean: None,
        niceness_stdev: None,
        distribution: Vec::new(),
        numbers: Vec::new(),
    }
}

fn clean_fixture_passes(conn: &mut PgConnection) {
    reset_fixture(conn);
    let audit = audit_base(conn, &base_record()).expect("audit");
    assert_eq!(audit.problems, Vec::<String>::new(), "{audit:?}");
    assert_eq!(audit.fields.rows, (CHUNKS * FIELDS_PER_CHUNK) as u64);
    assert_eq!(audit.chunks.rows, CHUNKS as u64);
    assert_eq!(audit.fields.total_size, (base_end() - BASE_START) as u128);
}

/// Deleting one interior field leaves a hole: one gap, and the total no
/// longer matches the base size. Nothing else should trip.
fn a_missing_field_is_a_gap(conn: &mut PgConnection) {
    reset_fixture(conn);
    sql_query("DELETE FROM fields WHERE id = 5")
        .execute(conn)
        .expect("delete");
    let fields = audit_fields_partition(conn, 40).expect("audit");
    assert_eq!(fields.gaps, 1, "{fields:?}");
    assert_eq!(fields.overlaps, 0);
    let problems = fields.problems(BASE_START as u128, base_end() as u128);
    assert_eq!(problems.len(), 2, "{problems:?}"); // the gap and the total
}

/// Stretching one field over its neighbour is an overlap; a field nested
/// inside an earlier one (start after, end before) must also count, which
/// is why the scan tracks the running maximum end and not just the previous
/// row's end.
fn overlaps_including_nested_ones(conn: &mut PgConnection) {
    reset_fixture(conn);
    // Field 3 grows to swallow field 4 entirely and half of field 5.
    sql_query(format!(
        "UPDATE fields SET range_end = range_end + {}, range_size = range_size + {}
         WHERE id = 3",
        FIELD_SIZE + FIELD_SIZE / 2,
        FIELD_SIZE + FIELD_SIZE / 2
    ))
    .execute(conn)
    .expect("update");
    let fields = audit_fields_partition(conn, 40).expect("audit");
    assert_eq!(fields.overlaps, 2, "{fields:?}"); // fields 4 and 5
    assert_eq!(fields.gaps, 0);
    assert_eq!(fields.size_mismatches, 0);
}

/// A row whose stored size disagrees with its bounds is caught even when the
/// bounds themselves still tile perfectly.
fn size_mismatch_is_caught(conn: &mut PgConnection) {
    reset_fixture(conn);
    sql_query("UPDATE fields SET range_size = range_size - 1 WHERE id = 7")
        .execute(conn)
        .expect("update");
    let fields = audit_fields_partition(conn, 40).expect("audit");
    assert_eq!(fields.size_mismatches, 1, "{fields:?}");
    assert_eq!(fields.gaps + fields.overlaps, 0);
    let problems = fields.problems(BASE_START as u128, base_end() as u128);
    assert_eq!(problems.len(), 2, "{problems:?}"); // mismatch and the total
}

/// Endpoint checks: the first field starting after the base start, or the
/// last ending before the base end, are not gaps in the window scan and
/// must be reported through the base comparison instead.
fn endpoints_are_checked_against_the_base(conn: &mut PgConnection) {
    reset_fixture(conn);
    sql_query("DELETE FROM fields WHERE id = 1")
        .execute(conn)
        .expect("delete");
    let fields = audit_fields_partition(conn, 40).expect("audit");
    assert_eq!(fields.gaps, 0, "{fields:?}");
    assert_eq!(fields.first_start, Some((BASE_START + FIELD_SIZE) as u128));
    let problems = fields.problems(BASE_START as u128, base_end() as u128);
    assert!(
        problems.iter().any(|p| p.starts_with("first start")),
        "{problems:?}"
    );
}

/// Chunk membership: an unassigned field, one pointed at the wrong chunk
/// (so it is not inside it), and a chunk-level gap.
fn chunk_membership_and_chunk_partition(conn: &mut PgConnection) {
    reset_fixture(conn);
    sql_query("UPDATE fields SET chunk_id = NULL WHERE id = 2")
        .execute(conn)
        .expect("update");
    sql_query("UPDATE fields SET chunk_id = 3 WHERE id = 11")
        .execute(conn)
        .expect("update");
    let membership = audit_fields_within_chunks(conn, 40).expect("audit");
    assert_eq!(membership.unassigned, 1, "{membership:?}");
    assert_eq!(membership.straddling, 1, "{membership:?}");
    assert_eq!(membership.wrong_base, 0);

    // Shrink chunk 2 by one field: a chunk-level gap, and its last field
    // (id 20) now pokes out of it.
    sql_query(format!(
        "UPDATE chunks SET range_end = range_end - {FIELD_SIZE}, range_size = range_size - {FIELD_SIZE}
         WHERE id = 2"
    ))
    .execute(conn)
    .expect("update");
    let chunks = audit_chunks_partition(conn, 40).expect("audit");
    assert_eq!(chunks.rows, 3);
    assert_eq!(chunks.gaps, 1, "{chunks:?}");
    let membership = audit_fields_within_chunks(conn, 40).expect("audit");
    assert_eq!(membership.straddling, 2, "{membership:?}");

    let audit = audit_base(conn, &base_record()).expect("audit");
    assert!(
        audit
            .problems
            .iter()
            .any(|p| p.starts_with("chunks: 1 gap")),
        "{:?}",
        audit.problems
    );
    assert!(
        audit
            .problems
            .iter()
            .any(|p| p.starts_with("fields/chunks: 1 field(s) with no chunk")),
        "{:?}",
        audit.problems
    );
}

/// A base with no rows at all is reported as such rather than passing on an
/// empty sum.
fn empty_base_is_a_problem(conn: &mut PgConnection) {
    reset_fixture(conn);
    sql_query("DELETE FROM fields")
        .execute(conn)
        .expect("delete");
    let fields = audit_fields_partition(conn, 40).expect("audit");
    assert_eq!(fields.rows, 0);
    assert_eq!(
        fields.problems(BASE_START as u128, base_end() as u128),
        vec!["no rows".to_string()]
    );
}

#[test]
fn partition_audit_against_postgres() {
    let Ok(url) = std::env::var("NICE_TEST_DATABASE_URL") else {
        eprintln!("skipping: NICE_TEST_DATABASE_URL is not set");
        return;
    };
    let mut conn = PgConnection::establish(&url).expect("connect to the test database");

    clean_fixture_passes(&mut conn);
    a_missing_field_is_a_gap(&mut conn);
    overlaps_including_nested_ones(&mut conn);
    size_mismatch_is_caught(&mut conn);
    endpoints_are_checked_against_the_base(&mut conn);
    chunk_membership_and_chunk_partition(&mut conn);
    empty_base_is_a_problem(&mut conn);
}
