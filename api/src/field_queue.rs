//! In-memory queue system for pre-claiming fields to reduce database latency.
//!
//! This module provides a thread-safe queue that pre-claims fields in bulk,
//! allowing the API to serve field claims with minimal latency (~1ms instead of ~90ms).

use chrono::{TimeDelta, Utc};
use nice_common::db_util::{
    PgPool, PgPooledConnection, fields::bulk_claim_fields, fields::bulk_claim_thin_fields,
    try_get_pooled_database_connection,
};
use nice_common::{CLAIM_DURATION_HOURS, DETAILED_SEARCH_MAX_FIELD_SIZE, FieldRecord};
use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

/// Configuration for queue refilling behavior
const REFILL_THRESHOLD: usize = 50; // Refill when queue has this many or fewer
const REFILL_AMOUNT: usize = 200; // Claim this many fields when refilling

/// Refill thresholds for the detailed-thin queue. Smaller than the niceonly
/// constants because detailed fields are more expensive to process and we don't
/// want to over-claim and starve the rarer detailed strategies (Next/Random/cl=2).
const DETAILED_REFILL_THRESHOLD: usize = 50;
const DETAILED_REFILL_AMOUNT: usize = 100;

/// Thread-safe queue for managing pre-claimed fields.
///
/// Refills are single-flight: each queue has a refill gate, and a request that
/// observes a low queue only refills if it wins `try_lock` on that gate — every
/// concurrent loser skips straight to popping. Without the gate, every request
/// seeing a low queue launched its own bulk claim: under fleet load that meant
/// 4-6 identical refills landing together (a niceonly queue observed at 1,226
/// against the 250 one refill can reach), all holding connections at exactly
/// the moment the pool was busiest.
///
/// Refills also run on the connection the requesting handler already holds,
/// rather than checking out a second one. A refilling request used to hold two
/// of the pool's connections for the duration of its bulk claim; with the pool
/// at its default size of 10, the refill herd plus its doubled checkouts was
/// measured saturating the pool for 10-20s stretches about once a minute,
/// fast-failing every other request into 5s-timeout 503s.
pub struct FieldQueue {
    /// Queue of pre-claimed `niceonly` fields (`check_level = 0`)
    niceonly: Arc<Mutex<VecDeque<FieldRecord>>>,
    /// Queue of pre-claimed `detailed` fields claimed via the `Thin` strategy
    /// (`check_level = 1`, `range_size <= DETAILED_MAX_FIELD_SIZE`).
    detailed_thin: Arc<Mutex<VecDeque<FieldRecord>>>,
    /// Single-flight gate for niceonly refills. Held only for the duration of
    /// the bulk claim; contenders skip the refill rather than waiting.
    niceonly_refill_gate: Mutex<()>,
    /// Single-flight gate for detailed-thin refills.
    detailed_refill_gate: Mutex<()>,
    /// Database connection pool, used only by the startup prefills. Claim-path
    /// refills run on the caller's connection instead.
    pool: PgPool,
}

impl FieldQueue {
    /// Create a new field queue with the given database pool.
    pub fn new(pool: PgPool) -> Self {
        Self {
            niceonly: Arc::new(Mutex::new(VecDeque::new())),
            detailed_thin: Arc::new(Mutex::new(VecDeque::new())),
            niceonly_refill_gate: Mutex::new(()),
            detailed_refill_gate: Mutex::new(()),
            pool,
        }
    }

    /// Try to claim a niceonly field from the queue, refilling first (on
    /// `conn`, single-flight) if the queue is low.
    ///
    /// Returns `None` when the queue is empty and this request either lost the
    /// refill gate or the refill produced nothing; the caller is expected to
    /// fall back to a direct claim. That fallback is an indexed single-row
    /// claim, so brief empty windows while one refill is in flight cost
    /// milliseconds — which is what makes skipping (rather than waiting on)
    /// the gate the right behavior.
    pub fn claim_niceonly(&self, conn: &mut PgPooledConnection) -> Option<FieldRecord> {
        let needs_refill = self.niceonly.lock().unwrap().len() <= REFILL_THRESHOLD;
        if needs_refill {
            // try_lock: winner refills, losers pop whatever is present. A
            // poisoned gate (a previous refill panicked) is treated the same
            // as a contended one — skip, and let the fallback path serve.
            // Re-check under the gate: between observing the low queue and
            // winning the gate, the previous winner may have already
            // refilled — in which case there is nothing to do.
            if let Ok(_guard) = self.niceonly_refill_gate.try_lock()
                && self.niceonly.lock().unwrap().len() <= REFILL_THRESHOLD
            {
                self.refill_niceonly(conn);
            }
        }

        self.niceonly.lock().unwrap().pop_front()
    }

    /// Refill the niceonly queue with pre-claimed fields, using the caller's
    /// connection. Callers must hold the refill gate (or be a startup prefill,
    /// where there is no concurrency to gate).
    fn refill_niceonly(&self, conn: &mut PgPooledConnection) {
        let maximum_timestamp = Utc::now() - TimeDelta::hours(CLAIM_DURATION_HOURS);
        let max_check_level = 0;
        let max_range_size = u128::MAX;

        match bulk_claim_fields(
            conn,
            REFILL_AMOUNT,
            maximum_timestamp,
            max_check_level,
            max_range_size,
        ) {
            Ok(fields) => {
                if fields.is_empty() {
                    tracing::warn!("Bulk claim returned no fields for niceonly queue");
                } else {
                    let mut queue = self.niceonly.lock().unwrap();
                    let count = fields.len();
                    queue.extend(fields);
                    tracing::debug!(
                        count = count,
                        queue_size = queue.len(),
                        "Refilled niceonly queue"
                    );
                }
            }
            Err(e) => {
                tracing::error!(error = %e, "Failed to refill niceonly queue: database error");
            }
        }
    }

    /// Get the current size of the niceonly queue (for monitoring/debugging).
    #[allow(dead_code)]
    pub fn niceonly_queue_size(&self) -> usize {
        self.niceonly.lock().unwrap().len()
    }

    /// Get the current size of the detailed-thin queue (for monitoring/debugging).
    pub fn detailed_thin_queue_size(&self) -> usize {
        self.detailed_thin.lock().unwrap().len()
    }

    /// Force an immediate refill of the niceonly queue (useful for initialization).
    /// Checks out its own pool connection: startup runs before any request
    /// traffic, so the checkout is uncontended.
    pub fn prefill_niceonly(&self) {
        tracing::info!("Pre-filling niceonly queue on startup");
        match try_get_pooled_database_connection(&self.pool) {
            Ok(mut conn) => self.refill_niceonly(&mut conn),
            Err(e) => {
                tracing::error!(error = %e, "Failed to prefill niceonly queue: no pool connection");
            }
        }
    }

    /// Force an immediate refill of the detailed-thin queue (useful for initialization).
    pub fn prefill_detailed_thin(&self) {
        tracing::info!("Pre-filling detailed-thin queue on startup");
        match try_get_pooled_database_connection(&self.pool) {
            Ok(mut conn) => self.refill_detailed_thin(&mut conn),
            Err(e) => {
                tracing::error!(error = %e, "Failed to prefill detailed-thin queue: no pool connection");
            }
        }
    }

    /// Try to claim a detailed field (Thin strategy) from the queue, refilling
    /// first (on `conn`, single-flight) if the queue is low.
    ///
    /// Returns `None` when the queue is empty and this request either lost the
    /// refill gate or the refill produced nothing; the caller is expected to
    /// fall back to a direct `try_claim_field`, which is chunk-scoped and
    /// costs tens of milliseconds.
    pub fn claim_detailed_thin(&self, conn: &mut PgPooledConnection) -> Option<FieldRecord> {
        let needs_refill = self.detailed_thin.lock().unwrap().len() <= DETAILED_REFILL_THRESHOLD;
        if needs_refill {
            // See claim_niceonly on the gate and the re-check under it.
            if let Ok(_guard) = self.detailed_refill_gate.try_lock()
                && self.detailed_thin.lock().unwrap().len() <= DETAILED_REFILL_THRESHOLD
            {
                self.refill_detailed_thin(conn);
            }
        }

        self.detailed_thin.lock().unwrap().pop_front()
    }

    /// Refill the detailed-thin queue with pre-claimed fields, using the
    /// caller's connection. See `refill_niceonly` on gating.
    fn refill_detailed_thin(&self, conn: &mut PgPooledConnection) {
        let maximum_timestamp = Utc::now() - TimeDelta::hours(CLAIM_DURATION_HOURS);
        let max_check_level = 1;
        let max_range_size = DETAILED_SEARCH_MAX_FIELD_SIZE;

        match bulk_claim_thin_fields(
            conn,
            DETAILED_REFILL_AMOUNT,
            maximum_timestamp,
            max_check_level,
            max_range_size,
        ) {
            Ok(fields) => {
                if fields.is_empty() {
                    tracing::warn!("Bulk claim returned no fields for detailed-thin queue");
                } else {
                    let mut queue = self.detailed_thin.lock().unwrap();
                    let count = fields.len();
                    queue.extend(fields);
                    tracing::debug!(
                        count = count,
                        queue_size = queue.len(),
                        "Refilled detailed-thin queue"
                    );
                }
            }
            Err(e) => {
                tracing::error!(
                    error = %e,
                    "Failed to refill detailed-thin queue: database error"
                );
            }
        }
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
        assert!(second.field_id > first.field_id, "queue must preserve claim order");

        // Drain past the threshold: the queue refills itself and keeps serving.
        for _ in 0..(REFILL_AMOUNT + REFILL_THRESHOLD) {
            queue
                .claim_niceonly(&mut conn)
                .expect("queue refills across the threshold");
        }
        assert_eq!(claimed_count(&mut pool.get().unwrap(), 0), 2 * REFILL_AMOUNT as i64);
    }

    #[test]
    fn field_queue_against_postgres() {
        let Ok(url) = std::env::var("NICE_TEST_DATABASE_URL") else {
            eprintln!("skipping: NICE_TEST_DATABASE_URL is not set");
            return;
        };

        refills_are_single_flight_on_the_callers_connection(&url);
        detailed_refills_are_single_flight(&url);
        queues_drain_and_refill_sequentially(&url);
    }
}
