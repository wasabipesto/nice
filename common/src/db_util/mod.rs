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
use diesel::r2d2::{ConnectionManager, Pool, PooledConnection};
use diesel::table;
use dotenvy::dotenv;
use serde_json::Value;

pub mod bases;
pub mod chunks;
pub mod claims;
pub mod conversions;
pub mod fields;
pub mod submissions;

/// A Diesel Postgres connection pool type.
pub type PgPool = Pool<ConnectionManager<PgConnection>>;

/// A Diesel Postgres pooled connection type.
pub type PgPooledConnection = PooledConnection<ConnectionManager<PgConnection>>;

/// Build a database connection pool.
///
/// Reads:
/// - `DATABASE_URL` (required)
/// - `DATABASE_POOL_SIZE` (optional, defaults to 10)
#[must_use]
pub fn get_database_pool() -> PgPool {
    dotenv().ok();
    let database_url = env::var("DATABASE_URL").expect("DATABASE_URL must be set");

    let pool_size: u32 = env::var("DATABASE_POOL_SIZE")
        .ok()
        .and_then(|v| v.parse::<u32>().ok())
        .filter(|&n| n > 0)
        .unwrap_or(10);

    let manager = ConnectionManager::<PgConnection>::new(database_url);
    Pool::builder()
        .max_size(pool_size)
        .build(manager)
        .expect("Error building database connection pool")
}

/// Get a single pooled database connection.
#[must_use]
pub fn get_pooled_database_connection(pool: &PgPool) -> PgPooledConnection {
    pool.get()
        .expect("Error retrieving database connection from pool")
}

/// Get a single database connection (non-pooled).
#[must_use]
pub fn get_database_connection() -> PgConnection {
    dotenv().ok();
    let database_url = env::var("DATABASE_URL").expect("DATABASE_URL must be set");
    PgConnection::establish(&database_url)
        .unwrap_or_else(|_| panic!("Error connecting to {database_url}"))
}
