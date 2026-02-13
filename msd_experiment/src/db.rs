//! Database module for caching MSD filter computation results.

use anyhow::{Context, Result};
use lru::LruCache;
use r2d2::Pool;
use r2d2_sqlite::SqliteConnectionManager;
use rusqlite::{OptionalExtension, params};
use serde::{Deserialize, Serialize};
use std::cell::RefCell;
use std::num::NonZeroUsize;
use std::path::Path;

/// Cache key for in-memory LRU cache
#[derive(Debug, Clone, Hash, Eq, PartialEq)]
struct CacheKey {
    base: u32,
    range_start: u128,
    range_end: u128,
}

/// In-memory LRU cache for frequently accessed ranges
/// Capacity of 100,000 entries should cover most hot paths (shallow depths, frequently accessed ranges)
const MEMORY_CACHE_CAPACITY: usize = 100_000;

thread_local! {
    static MEMORY_CACHE: RefCell<LruCache<CacheKey, CachedRange>> =
        RefCell::new(LruCache::new(NonZeroUsize::new(MEMORY_CACHE_CAPACITY).unwrap()));
}

/// Represents a cached computation result for a specific range.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CachedRange {
    pub base: u32,
    pub range_start: u128,
    pub range_end: u128,
    pub max_depth: u32,
    pub min_range_size: u128,
    pub subdivision_factor: usize,
    pub valid_size: u128,
}

/// Connection pool type for thread-safe database access.
pub type DbPool = Pool<SqliteConnectionManager>;

/// Initialize the database and create the schema if it doesn't exist.
pub fn init_db<P: AsRef<Path>>(db_path: P) -> Result<DbPool> {
    let manager = SqliteConnectionManager::file(db_path.as_ref());

    // Configure pool for better concurrent access
    let pool = Pool::builder()
        .max_size(32) // Allow up to 32 concurrent connections
        .connection_timeout(std::time::Duration::from_secs(30))
        .build(manager)
        .context("Failed to create connection pool")?;

    // Get a connection to set up the schema
    let conn = pool.get().context("Failed to get connection from pool")?;

    // Enable WAL mode for better concurrent access
    conn.execute_batch(
        "PRAGMA journal_mode=WAL;
         PRAGMA synchronous=NORMAL;
         PRAGMA cache_size=-64000;
         PRAGMA busy_timeout=5000;",
    )
    .context("Failed to set pragmas")?;

    // Create the table if it doesn't exist
    conn.execute(
        "CREATE TABLE IF NOT EXISTS msd_cache (
            base INTEGER NOT NULL,
            range_start TEXT NOT NULL,
            range_end TEXT NOT NULL,
            max_depth INTEGER NOT NULL,
            min_range_size TEXT NOT NULL,
            subdivision_factor INTEGER NOT NULL,
            valid_size TEXT NOT NULL,
            computed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (base, range_start, range_end)
        )",
        [],
    )
    .context("Failed to create msd_cache table")?;

    // Create index for efficient queries
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_base_depth ON msd_cache(base, max_depth)",
        [],
    )
    .context("Failed to create index")?;

    Ok(pool)
}

/// Get a cached result for a specific range.
/// Returns None if not found or if the cached depth is less than requested.
/// Checks in-memory LRU cache first before hitting SQLite.
pub fn get_cached_range(
    pool: &DbPool,
    base: u32,
    range_start: u128,
    range_end: u128,
    required_depth: u32,
) -> Result<Option<CachedRange>> {
    let cache_key = CacheKey {
        base,
        range_start,
        range_end,
    };

    // Check in-memory cache first
    let memory_hit = MEMORY_CACHE.with(|cache| cache.borrow_mut().get(&cache_key).cloned());

    if let Some(cached) = memory_hit {
        if cached.max_depth >= required_depth {
            return Ok(Some(cached));
        }
    }

    // Cache miss - query SQLite
    let conn = pool.get().context("Failed to get connection from pool")?;

    let mut stmt = conn
        .prepare(
            "SELECT base, range_start, range_end, max_depth, min_range_size,
                    subdivision_factor, valid_size
             FROM msd_cache
             WHERE base = ?1 AND range_start = ?2 AND range_end = ?3",
        )
        .context("Failed to prepare SELECT statement")?;

    let range_start_str = range_start.to_string();
    let range_end_str = range_end.to_string();

    let result = stmt
        .query_row(params![base, range_start_str, range_end_str], |row| {
            Ok(CachedRange {
                base: row.get(0)?,
                range_start: row.get::<_, String>(1)?.parse().unwrap(),
                range_end: row.get::<_, String>(2)?.parse().unwrap(),
                max_depth: row.get(3)?,
                min_range_size: row.get::<_, String>(4)?.parse().unwrap(),
                subdivision_factor: row.get(5)?,
                valid_size: row.get::<_, String>(6)?.parse().unwrap(),
            })
        })
        .optional()
        .context("Failed to query cached range")?;

    // Only return if the cached depth is sufficient
    match result {
        Some(cached) if cached.max_depth >= required_depth => {
            // Store in memory cache for future lookups
            MEMORY_CACHE.with(|cache| {
                cache.borrow_mut().put(cache_key, cached.clone());
            });
            Ok(Some(cached))
        }
        _ => Ok(None),
    }
}

/// Store a computed result in the cache.
/// Uses INSERT OR REPLACE to handle updates when computing at greater depth.
pub fn cache_range(
    pool: &DbPool,
    base: u32,
    range_start: u128,
    range_end: u128,
    max_depth: u32,
    min_range_size: u128,
    subdivision_factor: usize,
    valid_size: u128,
) -> Result<()> {
    // Retry logic for handling database busy errors in concurrent access
    let max_retries = 10;
    let mut last_error = None;

    for attempt in 0..max_retries {
        let conn = match pool.get() {
            Ok(c) => c,
            Err(e) => {
                last_error = Some(anyhow::Error::from(e));
                std::thread::sleep(std::time::Duration::from_millis(50 * (attempt + 1)));
                continue;
            }
        };

        match conn.execute(
            "INSERT OR REPLACE INTO msd_cache
             (base, range_start, range_end, max_depth, min_range_size,
              subdivision_factor, valid_size, computed_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, CURRENT_TIMESTAMP)",
            params![
                base,
                range_start.to_string(),
                range_end.to_string(),
                max_depth,
                min_range_size.to_string(),
                subdivision_factor,
                valid_size.to_string(),
            ],
        ) {
            Ok(_) => return Ok(()),
            Err(e) => {
                last_error = Some(anyhow::Error::from(e));
                // Small backoff before retry
                std::thread::sleep(std::time::Duration::from_millis(50 * (attempt + 1)));
            }
        }
    }

    Err(last_error
        .unwrap_or_else(|| anyhow::anyhow!("Failed to insert cached range after retries")))
}

/// Store multiple computed results in the cache using a single transaction.
/// This is much more efficient than calling cache_range multiple times.
/// Also populates the in-memory cache for future reads.
pub fn cache_range_batch(pool: &DbPool, entries: &[CachedRange]) -> Result<()> {
    if entries.is_empty() {
        return Ok(());
    }

    let mut conn = pool.get().context("Failed to get connection from pool")?;

    // Use a transaction for batch insert
    let tx = conn.transaction().context("Failed to start transaction")?;

    {
        let mut stmt = tx
            .prepare(
                "INSERT OR REPLACE INTO msd_cache
                 (base, range_start, range_end, max_depth, min_range_size,
                  subdivision_factor, valid_size, computed_at)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, CURRENT_TIMESTAMP)",
            )
            .context("Failed to prepare batch insert statement")?;

        for entry in entries {
            stmt.execute(params![
                entry.base,
                entry.range_start.to_string(),
                entry.range_end.to_string(),
                entry.max_depth,
                entry.min_range_size.to_string(),
                entry.subdivision_factor,
                entry.valid_size.to_string(),
            ])
            .context("Failed to execute batch insert")?;
        }
    }

    tx.commit().context("Failed to commit batch transaction")?;

    // Populate memory cache with written entries
    MEMORY_CACHE.with(|cache| {
        let mut cache = cache.borrow_mut();
        for entry in entries {
            let cache_key = CacheKey {
                base: entry.base,
                range_start: entry.range_start,
                range_end: entry.range_end,
            };
            cache.put(cache_key, entry.clone());
        }
    });

    Ok(())
}

/// Get statistics for a specific base.
#[derive(Debug)]
pub struct BaseStats {
    pub base: u32,
    pub num_cached_ranges: u64,
    pub total_valid_size: u128,
    pub avg_depth: f64,
    pub max_depth: u32,
}

pub fn get_base_stats(pool: &DbPool, base: u32) -> Result<Option<BaseStats>> {
    let conn = pool.get().context("Failed to get connection from pool")?;

    let mut stmt = conn
        .prepare(
            "SELECT
                COUNT(*) as num_ranges,
                SUM(CAST(valid_size AS INTEGER)) as total_valid,
                AVG(max_depth) as avg_depth,
                MAX(max_depth) as max_depth
             FROM msd_cache
             WHERE base = ?1",
        )
        .context("Failed to prepare stats query")?;

    let result = stmt
        .query_row(params![base], |row| {
            let num_ranges: u64 = row.get(0)?;
            if num_ranges == 0 {
                return Ok(None);
            }

            Ok(Some(BaseStats {
                base,
                num_cached_ranges: num_ranges,
                total_valid_size: row.get::<_, Option<i64>>(1)?.unwrap_or(0) as u128,
                avg_depth: row.get(2)?,
                max_depth: row.get(3)?,
            }))
        })
        .context("Failed to query base stats")?;

    Ok(result)
}

/// Get all cached ranges for a base (for export/visualization).
pub fn get_all_ranges_for_base(pool: &DbPool, base: u32) -> Result<Vec<CachedRange>> {
    let conn = pool.get().context("Failed to get connection from pool")?;

    let mut stmt = conn
        .prepare(
            "SELECT base, range_start, range_end, max_depth, min_range_size,
                    subdivision_factor, valid_size
             FROM msd_cache
             WHERE base = ?1
             ORDER BY range_start",
        )
        .context("Failed to prepare SELECT statement")?;

    let ranges = stmt
        .query_map(params![base], |row| {
            Ok(CachedRange {
                base: row.get(0)?,
                range_start: row.get::<_, String>(1)?.parse().unwrap(),
                range_end: row.get::<_, String>(2)?.parse().unwrap(),
                max_depth: row.get(3)?,
                min_range_size: row.get::<_, String>(4)?.parse().unwrap(),
                subdivision_factor: row.get(5)?,
                valid_size: row.get::<_, String>(6)?.parse().unwrap(),
            })
        })
        .context("Failed to query ranges")?
        .collect::<Result<Vec<_>, _>>()
        .context("Failed to collect ranges")?;

    Ok(ranges)
}

/// Clear all cached data for a specific base.
/// Also clears relevant entries from the in-memory cache.
pub fn clear_base_cache(pool: &DbPool, base: u32) -> Result<usize> {
    let conn = pool.get().context("Failed to get connection from pool")?;

    let affected = conn
        .execute("DELETE FROM msd_cache WHERE base = ?1", params![base])
        .context("Failed to clear base cache")?;

    // Clear entries for this base from memory cache
    MEMORY_CACHE.with(|cache| {
        let mut cache = cache.borrow_mut();
        // Collect keys to remove (LruCache doesn't have retain method)
        let keys_to_remove: Vec<_> = cache
            .iter()
            .filter(|(key, _)| key.base == base)
            .map(|(key, _)| key.clone())
            .collect();

        for key in keys_to_remove {
            cache.pop(&key);
        }
    });

    Ok(affected)
}

/// Get statistics about the in-memory cache (for monitoring/debugging)
pub fn get_memory_cache_stats() -> (usize, usize) {
    MEMORY_CACHE.with(|cache| {
        let cache = cache.borrow();
        (cache.len(), MEMORY_CACHE_CAPACITY)
    })
}
