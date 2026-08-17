//! In-memory cache of decoded benchmark reports for `/estimate`.
//!
//! The estimator is a pure function of (corpus, request), and the corpus —
//! the most recent `ESTIMATE_SAMPLE_LIMIT` reports — is identical for every
//! request. Before this cache each `/estimate` call re-fetched ~5.5 MB of
//! jsonb and re-decoded ~2000 reports (~55 ms of parse+decode alone, plus
//! the DB round trip, plus a pool connection held for the duration); the
//! fleet controller issues ~100 such calls per tick. Decoding once and
//! serving the corpus from memory drops the handler's cost to the match
//! itself (~0.05 ms) and takes `/estimate` off the connection pool entirely.
//!
//! Freshness: the corpus is refreshed at most once per TTL, and marked stale
//! immediately when `/benchmark` stores a new report so uploads show up in
//! estimates without waiting out the TTL. While one thread refreshes,
//! concurrent requests are served the previous corpus (stale-while-
//! revalidate) rather than queueing; if a refresh fails, the previous corpus
//! is likewise served and the failure logged. Only a cold cache with an
//! unreachable database surfaces an error.

use nice_common::db_util::{
    PgPool, benchmarks::get_recent_benchmarks, try_get_pooled_database_connection,
};
use nice_common::estimator::{BenchmarkSample, decode_sample};
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant};

struct CacheState {
    samples: Arc<Vec<BenchmarkSample>>,
    /// When the corpus was last fetched; `None` means stale (never fetched,
    /// or invalidated by a new upload).
    fetched_at: Option<Instant>,
    /// Whether `samples` has ever been populated from the database. Distinct
    /// from `fetched_at` so an invalidated corpus can still be served while
    /// a refresh is underway or failing.
    loaded: bool,
}

pub struct BenchmarkCache {
    pool: PgPool,
    ttl: Duration,
    sample_limit: i64,
    state: RwLock<CacheState>,
    /// Held by the one thread performing a refresh, so concurrent stale
    /// readers serve the old corpus instead of duplicating the fetch.
    refresh: Mutex<()>,
}

impl BenchmarkCache {
    pub fn new(pool: PgPool, ttl: Duration, sample_limit: i64) -> Self {
        Self {
            pool,
            ttl,
            sample_limit,
            state: RwLock::new(CacheState {
                samples: Arc::new(Vec::new()),
                fetched_at: None,
                loaded: false,
            }),
            refresh: Mutex::new(()),
        }
    }

    /// Get the current corpus, refreshing from the database if stale.
    /// Blocks only when the cache is cold; otherwise the worst case is one
    /// thread paying for a refresh while the rest serve the previous corpus.
    ///
    /// # Errors
    /// Only when the cache has never been populated and the database is
    /// unreachable.
    pub fn get(&self) -> Result<Arc<Vec<BenchmarkSample>>, String> {
        {
            let state = self.state.read().unwrap();
            if state
                .fetched_at
                .is_some_and(|fetched| fetched.elapsed() < self.ttl)
            {
                return Ok(Arc::clone(&state.samples));
            }
        }

        // Stale. Elect one refresher; everyone else serves the old corpus
        // if there is one, or waits for the refresher on a cold cache.
        if let Ok(_guard) = self.refresh.try_lock() {
            let loaded = self.state.read().unwrap().loaded;
            match self.refresh_locked() {
                Ok(samples) => Ok(samples),
                Err(e) if loaded => {
                    // Serve stale, and treat it as fresh for one more TTL so
                    // a database outage costs one refresh attempt (each up
                    // to the pool checkout timeout) per TTL window — not one
                    // per request, which would stall every serial caller.
                    tracing::warn!(error = %e, "benchmark cache refresh failed; serving stale corpus");
                    let mut state = self.state.write().unwrap();
                    state.fetched_at = Some(Instant::now());
                    Ok(Arc::clone(&state.samples))
                }
                Err(e) => Err(e),
            }
        } else {
            {
                let state = self.state.read().unwrap();
                if state.loaded {
                    return Ok(Arc::clone(&state.samples));
                }
            }
            // Cold cache: wait for the in-flight refresh to finish, then
            // take whatever it produced.
            drop(self.refresh.lock().unwrap());
            let state = self.state.read().unwrap();
            if state.loaded {
                Ok(Arc::clone(&state.samples))
            } else {
                Err("benchmark corpus unavailable (initial load failed)".to_string())
            }
        }
    }

    /// Mark the corpus stale so the next `/estimate` refreshes it. Called by
    /// `/benchmark` after storing a new report.
    pub fn invalidate(&self) {
        self.state.write().unwrap().fetched_at = None;
    }

    /// Load the corpus once at startup so the first `/estimate` doesn't pay
    /// for it. A failure is logged, not fatal: the first request will retry.
    pub fn prefill(&self) {
        tracing::info!("Pre-filling benchmark cache on startup");
        let _guard = self.refresh.lock().unwrap();
        if let Err(e) = self.refresh_locked() {
            tracing::warn!(error = %e, "benchmark cache prefill failed; first estimate will retry");
        }
    }

    /// Fetch and decode the corpus, then swap it in. Caller must hold
    /// `self.refresh`. The state lock is only taken for the swap, so readers
    /// are never blocked behind the fetch.
    fn refresh_locked(&self) -> Result<Arc<Vec<BenchmarkSample>>, String> {
        let started = Instant::now();
        let rows = {
            let mut conn = try_get_pooled_database_connection(&self.pool)
                .map_err(|e| format!("pool checkout failed: {e}"))?;
            get_recent_benchmarks(&mut conn, self.sample_limit)
                .map_err(|e| format!("benchmark fetch failed: {e}"))?
        };
        let samples: Arc<Vec<BenchmarkSample>> = Arc::new(
            rows.iter()
                .filter_map(|(version, report)| decode_sample(version, report))
                .collect(),
        );
        tracing::info!(
            rows = rows.len(),
            samples = samples.len(),
            elapsed_ms = started.elapsed().as_millis(),
            "benchmark cache refreshed"
        );
        let mut state = self.state.write().unwrap();
        state.samples = Arc::clone(&samples);
        state.fetched_at = Some(Instant::now());
        state.loaded = true;
        Ok(samples)
    }
}
