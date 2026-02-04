//! In-memory queue system for pre-claiming fields to reduce database latency.
//!
//! This module provides a thread-safe queue that pre-claims fields in bulk,
//! allowing the API to serve field claims with minimal latency (~1ms instead of ~90ms).

use chrono::{TimeDelta, Utc};
use nice_common::db_util::{PgPool, bulk_claim_fields, get_pooled_database_connection};
use nice_common::{CLAIM_DURATION_HOURS, DEFAULT_FIELD_SIZE, FieldRecord};
use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

/// Configuration for queue refilling behavior
const REFILL_THRESHOLD: usize = 10; // Refill when queue has this many or fewer
const REFILL_AMOUNT: usize = 100; // Claim this many fields when refilling

/// Thread-safe queue for managing pre-claimed fields.
pub struct FieldQueue {
    /// Queue of pre-claimed `niceonly` fields (`check_level = 0`)
    niceonly: Arc<Mutex<VecDeque<FieldRecord>>>,
    /// Database connection pool for refilling
    pool: PgPool,
}

impl FieldQueue {
    /// Create a new field queue with the given database pool.
    pub fn new(pool: PgPool) -> Self {
        Self {
            niceonly: Arc::new(Mutex::new(VecDeque::new())),
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
        let max_range_size = DEFAULT_FIELD_SIZE;

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

    /// Force an immediate refill of the niceonly queue (useful for initialization).
    pub fn prefill_niceonly(&self) {
        tracing::info!("Pre-filling niceonly queue on startup");
        self.refill_niceonly();
    }
}
