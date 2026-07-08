//! In-memory queue system for pre-claiming fields to reduce database latency.
//!
//! This module provides a thread-safe queue that pre-claims fields in bulk,
//! allowing the API to serve field claims with minimal latency (~1ms instead of ~90ms).

use chrono::{TimeDelta, Utc};
use nice_common::db_util::{
    PgPool, fields::bulk_claim_fields, fields::bulk_claim_thin_fields,
    get_pooled_database_connection,
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
pub struct FieldQueue {
    /// Queue of pre-claimed `niceonly` fields (`check_level = 0`)
    niceonly: Arc<Mutex<VecDeque<FieldRecord>>>,
    /// Queue of pre-claimed `detailed` fields claimed via the `Thin` strategy
    /// (`check_level = 1`, `range_size <= DETAILED_MAX_FIELD_SIZE`).
    detailed_thin: Arc<Mutex<VecDeque<FieldRecord>>>,
    /// Database connection pool for refilling
    pool: PgPool,
}

impl FieldQueue {
    /// Create a new field queue with the given database pool.
    pub fn new(pool: PgPool) -> Self {
        Self {
            niceonly: Arc::new(Mutex::new(VecDeque::new())),
            detailed_thin: Arc::new(Mutex::new(VecDeque::new())),
            pool,
        }
    }

    /// Try to claim a niceonly field from the queue.
    /// If the queue is empty or low, it will attempt to refill first.
    /// Falls back to empty if refill fails.
    pub fn claim_niceonly(&self) -> Option<FieldRecord> {
        // Check if we need to refill
        {
            let queue = self.niceonly.lock().unwrap();
            if queue.len() <= REFILL_THRESHOLD {
                drop(queue); // Release lock before refilling
                self.refill_niceonly();
            }
        }

        // Pop from queue
        let mut queue = self.niceonly.lock().unwrap();
        queue.pop_front()
    }

    /// Refill the niceonly queue with pre-claimed fields.
    /// This is called automatically when the queue drops below the threshold.
    fn refill_niceonly(&self) {
        let pool = self.pool.clone();

        // Perform the bulk claim synchronously
        let mut conn = get_pooled_database_connection(&pool);
        let maximum_timestamp = Utc::now() - TimeDelta::hours(CLAIM_DURATION_HOURS);
        let max_check_level = 0;
        let max_range_size = u128::MAX;

        match bulk_claim_fields(
            &mut conn,
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
    pub fn prefill_niceonly(&self) {
        tracing::info!("Pre-filling niceonly queue on startup");
        self.refill_niceonly();
    }

    /// Force an immediate refill of the detailed-thin queue (useful for initialization).
    pub fn prefill_detailed_thin(&self) {
        tracing::info!("Pre-filling detailed-thin queue on startup");
        self.refill_detailed_thin();
    }

    /// Try to claim a detailed field (Thin strategy) from the queue.
    /// If the queue is empty or low, it will attempt to refill first.
    /// Falls back to `None` if refill fails or yields nothing; the caller is
    /// expected to fall back to a direct `try_claim_field` in that case.
    pub fn claim_detailed_thin(&self) -> Option<FieldRecord> {
        // Check if we need to refill
        {
            let queue = self.detailed_thin.lock().unwrap();
            if queue.len() <= DETAILED_REFILL_THRESHOLD {
                drop(queue); // Release lock before refilling
                self.refill_detailed_thin();
            }
        }

        // Pop from queue
        let mut queue = self.detailed_thin.lock().unwrap();
        queue.pop_front()
    }

    /// Refill the detailed-thin queue with pre-claimed fields.
    /// This is called automatically when the queue drops below the threshold.
    fn refill_detailed_thin(&self) {
        let pool = self.pool.clone();

        // Perform the bulk claim synchronously
        let mut conn = get_pooled_database_connection(&pool);
        let maximum_timestamp = Utc::now() - TimeDelta::hours(CLAIM_DURATION_HOURS);
        let max_check_level = 1;
        let max_range_size = DETAILED_SEARCH_MAX_FIELD_SIZE;

        match bulk_claim_thin_fields(
            &mut conn,
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
