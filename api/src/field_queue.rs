//! In-memory queue system for pre-claiming fields to reduce database latency.
//!
//! This module provides thread-safe queues that pre-claim fields in bulk,
//! allowing the API to serve field claims with minimal latency (~1ms instead of
//! tens of milliseconds, and without every concurrent request re-running the
//! same frontier query against the database).

use chrono::{TimeDelta, Utc};
use nice_common::db_util::{
    PgPool, PgPooledConnection, fields::bulk_claim_fields, fields::bulk_claim_next_fields,
    fields::bulk_claim_thin_fields, try_get_pooled_database_connection,
};
use nice_common::{CLAIM_DURATION_HOURS, DETAILED_SEARCH_MAX_FIELD_SIZE, FieldRecord};
use std::collections::VecDeque;
use std::sync::Mutex;

/// Configuration for queue refilling behavior
const REFILL_THRESHOLD: usize = 50; // Refill when queue has this many or fewer
const REFILL_AMOUNT: usize = 200; // Claim this many fields when refilling

/// Refill thresholds for the detailed queues. Smaller than the niceonly
/// constants because detailed fields are more expensive to process and we don't
/// want to over-claim: a pre-claimed field that no client asks for within
/// `CLAIM_DURATION_HOURS` is simply claimable again, but until then it is held.
const DETAILED_REFILL_THRESHOLD: usize = 50;
const DETAILED_REFILL_AMOUNT: usize = 100;

/// Refill sizes for the detailed `Next` queues, which serve a fifth of the
/// detailed traffic the thin queue does (15% and 4% of claims against 80%).
/// A batch of 100 is one frontier-chunk query either way; the lower threshold
/// keeps the number of fields held for the rarer strategies proportionate.
const NEXT_REFILL_THRESHOLD: usize = 20;
const NEXT_REFILL_AMOUNT: usize = 100;

/// One pre-claimed queue and its single-flight refill gate.
///
/// Refills are single-flight: a request that observes a low queue only refills
/// if it wins `try_lock` on the gate — every concurrent loser skips straight to
/// popping. Without the gate, every request seeing a low queue launched its own
/// bulk claim: under fleet load that meant 4-6 identical refills landing
/// together (a niceonly queue observed at 1,226 against the 250 one refill can
/// reach), all holding connections at exactly the moment the pool was busiest.
///
/// Refills also run on the connection the requesting handler already holds,
/// rather than checking out a second one. A refilling request used to hold two
/// of the pool's connections for the duration of its bulk claim; with the pool
/// at its default size of 10, the refill herd plus its doubled checkouts was
/// measured saturating the pool for 10-20s stretches about once a minute,
/// fast-failing every other request into 5s-timeout 503s.
struct Preclaimed {
    name: &'static str,
    fields: Mutex<VecDeque<FieldRecord>>,
    /// Held only for the duration of the bulk claim; contenders skip the
    /// refill rather than waiting.
    refill_gate: Mutex<()>,
    threshold: usize,
}

impl Preclaimed {
    fn new(name: &'static str, threshold: usize) -> Self {
        Self {
            name,
            fields: Mutex::new(VecDeque::new()),
            refill_gate: Mutex::new(()),
            threshold,
        }
    }

    fn len(&self) -> usize {
        self.fields.lock().unwrap().len()
    }

    /// Pop the next pre-claimed field, refilling first (single-flight, via
    /// `refill`) if the queue is at or below its threshold.
    ///
    /// Returns `None` when the queue is empty and this request either lost the
    /// refill gate or the refill produced nothing; the caller is expected to
    /// fall back to a direct claim. That fallback is a chunk-scoped single-row
    /// claim costing tens of milliseconds, so brief empty windows while one
    /// refill is in flight are cheap — which is what makes skipping (rather than
    /// waiting on) the gate the right behavior.
    fn claim<E: std::fmt::Display>(
        &self,
        refill: impl FnOnce() -> Result<Vec<FieldRecord>, E>,
    ) -> Option<FieldRecord> {
        if self.len() <= self.threshold {
            // try_lock: winner refills, losers pop whatever is present. A
            // poisoned gate (a previous refill panicked) is treated the same
            // as a contended one — skip, and let the fallback path serve.
            // Re-check under the gate: between observing the low queue and
            // winning the gate, the previous winner may have already
            // refilled — in which case there is nothing to do.
            if let Ok(_guard) = self.refill_gate.try_lock()
                && self.len() <= self.threshold
            {
                self.refill(refill);
            }
        }
        self.fields.lock().unwrap().pop_front()
    }

    /// Run one bulk claim and append its result. Callers must hold the refill
    /// gate (or be a startup prefill, where there is no concurrency to gate).
    fn refill<E: std::fmt::Display>(&self, refill: impl FnOnce() -> Result<Vec<FieldRecord>, E>) {
        match refill() {
            Ok(fields) if fields.is_empty() => {
                tracing::warn!(queue = self.name, "Bulk claim returned no fields");
            }
            Ok(fields) => {
                let mut queue = self.fields.lock().unwrap();
                let count = fields.len();
                queue.extend(fields);
                tracing::debug!(
                    queue = self.name,
                    count = count,
                    queue_size = queue.len(),
                    "Refilled queue"
                );
            }
            Err(e) => {
                tracing::error!(queue = self.name, error = %e, "Failed to refill queue: database error");
            }
        }
    }
}

/// Thread-safe queues of pre-claimed fields, one per claim strategy the API
/// serves from memory.
pub struct FieldQueue {
    /// Pre-claimed `niceonly` fields (`check_level = 0`).
    niceonly: Preclaimed,
    /// Pre-claimed `detailed` fields claimed via the `Thin` strategy
    /// (`check_level = 1`, `range_size <= DETAILED_MAX_FIELD_SIZE`).
    detailed_thin: Preclaimed,
    /// Pre-claimed `detailed` fields in `Next` order at `check_level <= 1`: the
    /// global frontier. Served in frontier order, front to back.
    detailed_next: Preclaimed,
    /// Pre-claimed `detailed` fields in `Next` order at `check_level <= 2`: the
    /// recheck strategy, which revisits completed fields from the lowest id up.
    detailed_recheck: Preclaimed,
    /// Database connection pool, used only by the startup prefills. Claim-path
    /// refills run on the caller's connection instead.
    pool: PgPool,
}

impl FieldQueue {
    /// Create a new field queue with the given database pool.
    pub fn new(pool: PgPool) -> Self {
        Self {
            niceonly: Preclaimed::new("niceonly", REFILL_THRESHOLD),
            detailed_thin: Preclaimed::new("detailed-thin", DETAILED_REFILL_THRESHOLD),
            detailed_next: Preclaimed::new("detailed-next", NEXT_REFILL_THRESHOLD),
            detailed_recheck: Preclaimed::new("detailed-recheck", NEXT_REFILL_THRESHOLD),
            pool,
        }
    }

    fn claim_cutoff() -> chrono::DateTime<Utc> {
        Utc::now() - TimeDelta::hours(CLAIM_DURATION_HOURS)
    }

    /// Try to claim a niceonly field from the queue, refilling first (on
    /// `conn`, single-flight) if the queue is low. `None` means fall back to a
    /// direct claim; see `Preclaimed::claim`.
    pub fn claim_niceonly(&self, conn: &mut PgPooledConnection) -> Option<FieldRecord> {
        self.niceonly
            .claim(|| bulk_claim_fields(conn, REFILL_AMOUNT, Self::claim_cutoff(), 0, u128::MAX))
    }

    /// Try to claim a detailed field (Thin strategy) from the queue, refilling
    /// first (on `conn`, single-flight) if the queue is low. `None` means fall
    /// back to a direct `try_claim_field`, which is chunk-scoped and costs tens
    /// of milliseconds.
    pub fn claim_detailed_thin(&self, conn: &mut PgPooledConnection) -> Option<FieldRecord> {
        self.detailed_thin.claim(|| {
            bulk_claim_thin_fields(
                conn,
                DETAILED_REFILL_AMOUNT,
                Self::claim_cutoff(),
                1,
                DETAILED_SEARCH_MAX_FIELD_SIZE,
            )
        })
    }

    /// Try to claim a detailed field in `Next` order at or below
    /// `max_check_level` (1 = the frontier, 2 = the recheck strategy) from the
    /// matching queue, refilling first (on `conn`, single-flight) if it is low.
    /// `None` means fall back to a direct `Next` claim. Any other check level is
    /// not queued and returns `None` immediately.
    ///
    /// Each queue holds one frontier chunk's worth of fields in id order, so
    /// consumers see the same sequence a direct `Next` claim would produce,
    /// minus interleaving with whatever direct claims land in between.
    pub fn claim_detailed_next(
        &self,
        conn: &mut PgPooledConnection,
        max_check_level: u8,
    ) -> Option<FieldRecord> {
        let queue = match max_check_level {
            1 => &self.detailed_next,
            2 => &self.detailed_recheck,
            _ => return None,
        };
        queue.claim(|| {
            bulk_claim_next_fields(
                conn,
                NEXT_REFILL_AMOUNT,
                Self::claim_cutoff(),
                max_check_level,
                DETAILED_SEARCH_MAX_FIELD_SIZE,
            )
        })
    }

    /// Get the current size of the niceonly queue (for monitoring/debugging).
    #[allow(dead_code)]
    pub fn niceonly_queue_size(&self) -> usize {
        self.niceonly.len()
    }

    /// Get the current size of the detailed-thin queue (for monitoring/debugging).
    pub fn detailed_thin_queue_size(&self) -> usize {
        self.detailed_thin.len()
    }

    /// Current size of the detailed `Next` queue at `check_level <= 1`.
    pub fn detailed_next_queue_size(&self) -> usize {
        self.detailed_next.len()
    }

    /// Current size of the detailed recheck queue (`Next` at `check_level <= 2`).
    pub fn detailed_recheck_queue_size(&self) -> usize {
        self.detailed_recheck.len()
    }

    /// Fill every queue once at startup. Checks out its own pool connection:
    /// startup runs before any request traffic, so the checkout is uncontended.
    pub fn prefill_all(&self) {
        tracing::info!("Pre-filling claim queues on startup");
        let mut conn = match try_get_pooled_database_connection(&self.pool) {
            Ok(conn) => conn,
            Err(e) => {
                tracing::error!(error = %e, "Failed to prefill claim queues: no pool connection");
                return;
            }
        };
        let cutoff = Self::claim_cutoff();
        self.niceonly
            .refill(|| bulk_claim_fields(&mut conn, REFILL_AMOUNT, cutoff, 0, u128::MAX));
        self.detailed_thin.refill(|| {
            bulk_claim_thin_fields(
                &mut conn,
                DETAILED_REFILL_AMOUNT,
                cutoff,
                1,
                DETAILED_SEARCH_MAX_FIELD_SIZE,
            )
        });
        self.detailed_next.refill(|| {
            bulk_claim_next_fields(
                &mut conn,
                NEXT_REFILL_AMOUNT,
                cutoff,
                1,
                DETAILED_SEARCH_MAX_FIELD_SIZE,
            )
        });
        self.detailed_recheck.refill(|| {
            bulk_claim_next_fields(
                &mut conn,
                NEXT_REFILL_AMOUNT,
                cutoff,
                2,
                DETAILED_SEARCH_MAX_FIELD_SIZE,
            )
        });
    }
}

#[cfg(test)]
#[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
mod tests {
    //! Concurrency tests for the refill gates, against a real `PostgreSQL`.
    //!
    //! Skipped unless `NICE_TEST_DATABASE_URL` is set — deliberately not
    //! `DATABASE_URL`, because the fixture truncates the tables. Point it at a
    //! scratch database with `schema/schema.sql` loaded (see
    //! `common/tests/claim_queries.rs` for the setup commands).

    use super::*;
    use diesel::prelude::*;
    use diesel::r2d2::ConnectionManager;
    use diesel::sql_query;
    use std::sync::Barrier;

    const THREADS: usize = 8;

    fn test_pool(url: &str, size: u32) -> PgPool {
        diesel::r2d2::Pool::builder()
            .max_size(size)
            .connection_timeout(std::time::Duration::from_secs(3))
            .build(ConnectionManager::new(url))
            .expect("build test pool")
    }

    /// Seed one base with plenty of claimable fields: 600 at `check_level = 0`
    /// (niceonly) and 3 chunks x 200 at `check_level = 1` (detailed-thin).
    fn reset_fixture(conn: &mut PgPooledConnection) {
        sql_query("TRUNCATE fields, chunks, bases RESTART IDENTITY CASCADE")
            .execute(conn)
            .expect("truncate");
        sql_query("INSERT INTO bases (id, range_start, range_end, range_size) VALUES (40, 0, 2000000, 2000000)")
            .execute(conn)
            .expect("insert base");
        for chunk in 0..3i64 {
            sql_query(format!(
                "INSERT INTO chunks (base_id, range_start, range_end, range_size, checked_detailed)
                 VALUES (40, {}, {}, 200000, 0)",
                chunk * 200_000,
                (chunk + 1) * 200_000
            ))
            .execute(conn)
            .expect("insert chunk");
            sql_query(format!(
                "INSERT INTO fields (base_id, chunk_id, range_start, range_end, range_size, check_level)
                 SELECT 40, {}, {} + g * 1000, {} + (g + 1) * 1000, 1000, 1
                 FROM generate_series(0, 199) AS g",
                chunk + 1,
                chunk * 200_000,
                chunk * 200_000
            ))
            .execute(conn)
            .expect("insert cl1 fields");
        }
        sql_query(
            "INSERT INTO fields (base_id, range_start, range_end, range_size, check_level)
             SELECT 40, 1000000 + g * 1000, 1000000 + (g + 1) * 1000, 1000, 0
             FROM generate_series(0, 599) AS g",
        )
        .execute(conn)
        .expect("insert cl0 fields");
    }

    fn claimed_count(conn: &mut PgPooledConnection, check_level: i32) -> i64 {
        #[derive(QueryableByName)]
        struct Count {
            #[diesel(sql_type = diesel::sql_types::BigInt)]
            n: i64,
        }
        sql_query(format!(
            "SELECT count(*) AS n FROM fields
             WHERE last_claim_time IS NOT NULL AND check_level = {check_level}"
        ))
        .get_result::<Count>(conn)
        .expect("count claimed")
        .n
    }

    /// `THREADS` requests hit an empty queue simultaneously, every one of them
    /// past the refill threshold. Exactly one bulk claim may run: the pre-gate
    /// behavior fired one refill *per request* (a 4-6x herd measured in
    /// production, queue observed at 1,226 against the 250 one refill can
    /// reach), which this asserts against directly — both in the database
    /// (rows stamped) and in the queue depth ceiling.
    ///
    /// The pool is sized so that every thread's request connection together
    /// exhausts it. The refill must still succeed, because it runs on the
    /// caller's connection — a refill that checks out a second connection
    /// (the pre-gate behavior) would time out here and leave the queue empty.
    fn refills_are_single_flight_on_the_callers_connection(url: &str) {
        let pool = test_pool(url, THREADS as u32);
        reset_fixture(&mut pool.get().unwrap());
        let queue = FieldQueue::new(pool.clone());
        let barrier = Barrier::new(THREADS);

        std::thread::scope(|s| {
            for _ in 0..THREADS {
                s.spawn(|| {
                    // Hold this thread's connection across the claim, like a
                    // request handler does. All THREADS connections together
                    // exhaust the pool — and stay held (second barrier) until
                    // every claim has finished, so a refill that tries to
                    // check out an extra connection has none to take.
                    let mut conn = pool.get().expect("per-thread connection");
                    barrier.wait();
                    queue.claim_niceonly(&mut conn);
                    barrier.wait();
                });
            }
        });

        let claimed = claimed_count(&mut pool.get().unwrap(), 0);
        assert_eq!(
            claimed, REFILL_AMOUNT as i64,
            "expected exactly one bulk claim; the refill herd is back (or the \
             refill checked out a second connection and starved)"
        );
        assert!(
            queue.niceonly_queue_size() <= REFILL_AMOUNT,
            "queue depth {} exceeds what a single refill can produce",
            queue.niceonly_queue_size()
        );
    }

    /// Same property for the detailed-thin queue and its separate gate.
    fn detailed_refills_are_single_flight(url: &str) {
        let pool = test_pool(url, THREADS as u32);
        reset_fixture(&mut pool.get().unwrap());
        let queue = FieldQueue::new(pool.clone());
        let barrier = Barrier::new(THREADS);

        std::thread::scope(|s| {
            for _ in 0..THREADS {
                s.spawn(|| {
                    let mut conn = pool.get().expect("per-thread connection");
                    barrier.wait();
                    queue.claim_detailed_thin(&mut conn);
                    barrier.wait();
                });
            }
        });

        let claimed = claimed_count(&mut pool.get().unwrap(), 1);
        assert_eq!(claimed, DETAILED_REFILL_AMOUNT as i64);
        assert!(queue.detailed_thin_queue_size() <= DETAILED_REFILL_AMOUNT);
    }

    /// Same property for the detailed-next queue and its own gate.
    fn next_refills_are_single_flight(url: &str) {
        let pool = test_pool(url, THREADS as u32);
        reset_fixture(&mut pool.get().unwrap());
        let queue = FieldQueue::new(pool.clone());
        let barrier = Barrier::new(THREADS);

        std::thread::scope(|s| {
            for _ in 0..THREADS {
                s.spawn(|| {
                    let mut conn = pool.get().expect("per-thread connection");
                    barrier.wait();
                    queue.claim_detailed_next(&mut conn, 1);
                    barrier.wait();
                });
            }
        });

        assert_eq!(
            claimed_count(&mut pool.get().unwrap(), 1),
            NEXT_REFILL_AMOUNT as i64
        );
        assert!(queue.detailed_next_queue_size() <= NEXT_REFILL_AMOUNT);
    }

    /// The next queue hands out the frontier in id order, the recheck queue
    /// (cl<=2) is a separate queue that does not re-issue what the frontier
    /// queue holds, and unqueued levels are refused without touching the
    /// database.
    fn next_queues_serve_the_frontier_in_order(url: &str) {
        let pool = test_pool(url, 2);
        reset_fixture(&mut pool.get().unwrap());
        let queue = FieldQueue::new(pool.clone());
        let mut conn = pool.get().unwrap();

        let mut ids = Vec::new();
        for _ in 0..5 {
            ids.push(
                queue
                    .claim_detailed_next(&mut conn, 1)
                    .expect("frontier field")
                    .field_id,
            );
        }
        assert_eq!(
            ids,
            vec![1, 2, 3, 4, 5],
            "frontier order, from the lowest id"
        );

        let recheck = queue
            .claim_detailed_next(&mut conn, 2)
            .expect("recheck field");
        assert!(
            recheck.field_id > NEXT_REFILL_AMOUNT as u128,
            "the recheck queue must not re-issue fields the frontier queue holds (got {})",
            recheck.field_id
        );

        assert!(queue.claim_detailed_next(&mut conn, 3).is_none());
        assert_eq!(
            claimed_count(&mut pool.get().unwrap(), 1),
            2 * NEXT_REFILL_AMOUNT as i64,
            "exactly one refill per queue"
        );
    }

    /// The queues still drain and re-refill correctly in the ordinary
    /// sequential case: claims come off in id order, and crossing the
    /// threshold triggers the next single refill.
    fn queues_drain_and_refill_sequentially(url: &str) {
        let pool = test_pool(url, 2);
        reset_fixture(&mut pool.get().unwrap());
        let queue = FieldQueue::new(pool.clone());
        let mut conn = pool.get().unwrap();

        let first = queue.claim_niceonly(&mut conn).expect("refill then pop");
        let second = queue.claim_niceonly(&mut conn).expect("pop");
        assert!(
            second.field_id > first.field_id,
            "queue must preserve claim order"
        );

        // Drain past the threshold: the queue refills itself and keeps serving.
        for _ in 0..(REFILL_AMOUNT + REFILL_THRESHOLD) {
            queue
                .claim_niceonly(&mut conn)
                .expect("queue refills across the threshold");
        }
        assert_eq!(
            claimed_count(&mut pool.get().unwrap(), 0),
            2 * REFILL_AMOUNT as i64
        );
    }

    #[test]
    fn field_queue_against_postgres() {
        let Ok(url) = std::env::var("NICE_TEST_DATABASE_URL") else {
            eprintln!("skipping: NICE_TEST_DATABASE_URL is not set");
            return;
        };

        refills_are_single_flight_on_the_callers_connection(&url);
        detailed_refills_are_single_flight(&url);
        next_refills_are_single_flight(&url);
        next_queues_serve_the_frontier_in_order(&url);
        queues_drain_and_refill_sequentially(&url);
    }
}
