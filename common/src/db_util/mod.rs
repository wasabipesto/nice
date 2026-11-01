//! Interfaces between the application code and database.

#![allow(
    clippy::wildcard_imports,
    clippy::missing_panics_doc,
    clippy::missing_errors_doc
)]

use super::*;

use anyhow::{Result, anyhow, bail};
use bigdecimal::{BigDecimal, FromPrimitive, ToPrimitive};
use diesel::prelude::*;
use diesel::r2d2::{ConnectionManager, CustomizeConnection, Pool, PooledConnection};
use diesel::table;
use dotenvy::dotenv;
use serde_json::Value;

pub mod bases;
pub mod benchmarks;
pub mod cache;
pub mod chunks;
pub mod claims;
pub mod conversions;
pub mod fields;
pub mod job_state;
pub mod submissions;

/// A Diesel Postgres connection pool type.
pub type PgPool = Pool<ConnectionManager<PgConnection>>;

/// A Diesel Postgres pooled connection type.
pub type PgPooledConnection = PooledConnection<ConnectionManager<PgConnection>>;

/// Build a database connection pool.
///
/// Reads:
/// - `DATABASE_URL` (required)
/// - `DATABASE_POOL_SIZE` (optional, defaults to 30)
/// - `DATABASE_POOL_TIMEOUT_SECS` (optional, defaults to 5)
/// - `DATABASE_STATEMENT_TIMEOUT_SECS` (optional, defaults to 30; 0 disables)
///
/// The checkout timeout is deliberately short: r2d2's default of 30 seconds
/// means one slow endpoint can pin a request worker for half a minute while
/// it waits for a free connection, starving unrelated endpoints. Failing
/// fast lets callers degrade (serve stale, return 503) instead.
///
/// Every pooled connection also gets a `statement_timeout` ([`StatementTimeout`]).
/// Only the API builds this pool; the jobs binary's long statements use their own
/// connections.
#[must_use]
pub fn get_database_pool() -> PgPool {
    dotenv().ok();
    let database_url = env::var("DATABASE_URL").expect("DATABASE_URL must be set");

    let pool_size: u32 = env::var("DATABASE_POOL_SIZE")
        .ok()
        .and_then(|v| v.parse::<u32>().ok())
        .filter(|&n| n > 0)
        .unwrap_or(30);

    let timeout_secs: u64 = env::var("DATABASE_POOL_TIMEOUT_SECS")
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .filter(|&n| n > 0)
        .unwrap_or(5);

    let statement_timeout_secs: u64 = env::var("DATABASE_STATEMENT_TIMEOUT_SECS")
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .unwrap_or(30);

    let manager = ConnectionManager::<PgConnection>::new(database_url);
    Pool::builder()
        .max_size(pool_size)
        .connection_timeout(std::time::Duration::from_secs(timeout_secs))
        .connection_customizer(Box::new(StatementTimeout(statement_timeout_secs)))
        .build(manager)
        .expect("Error building database connection pool")
}

/// Sets `statement_timeout` (in seconds; 0 leaves the server default) on each
/// connection the pool opens.
///
/// Without it a claim statement can run indefinitely while holding the row locks it
/// has already taken. One did for 21 hours in production (2026-11-01): the field it
/// had locked still read as claimable, so every frontier search picked that field's
/// chunk, skipped the locked row and came back empty, stalling detailed `Next`
/// claims; and every jobs run queued behind the lock.
#[derive(Debug, Clone, Copy)]
pub struct StatementTimeout(pub u64);

impl CustomizeConnection<PgConnection, diesel::r2d2::Error> for StatementTimeout {
    fn on_acquire(&self, conn: &mut PgConnection) -> Result<(), diesel::r2d2::Error> {
        if self.0 == 0 {
            return Ok(());
        }
        let millis = self.0.saturating_mul(1000);
        diesel::sql_query(format!("SET statement_timeout = {millis}"))
            .execute(conn)
            .map(|_| ())
            .map_err(diesel::r2d2::Error::QueryError)
    }
}

/// Get a single pooled database connection.
#[must_use]
pub fn get_pooled_database_connection(pool: &PgPool) -> PgPooledConnection {
    pool.get()
        .expect("Error retrieving database connection from pool")
}

/// Get a single pooled database connection, surfacing checkout failure
/// (pool exhausted past the checkout timeout, or the database unreachable)
/// to the caller instead of panicking. API handlers use this to turn pool
/// pressure into a fast 503 rather than a worker-thread pileup.
pub fn try_get_pooled_database_connection(
    pool: &PgPool,
) -> Result<PgPooledConnection, diesel::r2d2::PoolError> {
    pool.get()
}

/// Get a single database connection (non-pooled).
#[must_use]
pub fn get_database_connection() -> PgConnection {
    dotenv().ok();
    let database_url = env::var("DATABASE_URL").expect("DATABASE_URL must be set");
    PgConnection::establish(&database_url)
        .unwrap_or_else(|_| panic!("Error connecting to {database_url}"))
}
