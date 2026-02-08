//! A library with common utilities for dealing with square-cube pandigitals.

#![warn(clippy::all, clippy::pedantic)]
#![allow(clippy::match_same_arms)]

pub mod base_range;
pub mod benchmark;
#[cfg(any(feature = "openssl-tls", feature = "rustls-tls"))]
pub mod client_api_async;
#[cfg(any(feature = "openssl-tls", feature = "rustls-tls"))]
pub mod client_api_sync;
pub mod client_process;
pub mod client_process_gpu;
pub mod consensus;
#[cfg(feature = "database")]
pub mod db_util;
pub mod distribution_stats;
pub mod generate_chunks;
pub mod generate_fields;
pub mod lsd_filter;
pub mod msd_prefix_filter;
pub mod number_stats;
pub mod residue_filter;
pub mod stride_filter;

use chrono::{DateTime, Utc};
use clap::ValueEnum;
use serde::{Deserialize, Serialize};
use std::env;
use std::fmt;

pub const CLIENT_VERSION: &str = env!("CARGO_PKG_VERSION");
pub const NEAR_MISS_CUTOFF_PERCENT: f32 = 0.9;
pub const DOWNSAMPLE_CUTOFF_PERCENT: f32 = 0.2;
pub const CLAIM_DURATION_HOURS: i64 = 1;
pub const CLIENT_REQUEST_TIMEOUT_SECS: u64 = 5;
pub const DEFAULT_FIELD_SIZE: u128 = 1_000_000_000;
pub const PROCESSING_CHUNK_SIZE: u128 = 1_000_000;

/// Each possible search mode the server and client supports.
#[derive(Debug, Serialize, Deserialize, Copy, Clone, PartialEq, Eq, PartialOrd, Ord, ValueEnum)]
pub enum SearchMode {
    /// Get detailed stats on all numbers, important for long-term analytics.
    Detailed,
    /// Implements optimizations to speed up the search, usually by a factor of around 20.
    /// Does not keep statistics and cannot be quickly verified.
    Niceonly,
}
impl fmt::Display for SearchMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SearchMode::Detailed => write!(f, "Detailed"),
            SearchMode::Niceonly => write!(f, "Nice-only"),
        }
    }
}

/// How we should pick a field when claiming.
#[derive(Debug, Copy, Clone)]
pub enum FieldClaimStrategy {
    /// Simply get the next available field
    Next,
    /// Get a random available field from any chunk
    Random,
    /// Get a random available field in the next chunk with <X% complete
    Thin,
}

/// Data on the bounds of a search range.
/// Could be a base, chunk, field, or something else.
///
/// **Important**: This represents a half-open range [`range_start`, `range_end`),
/// following Rust's standard convention. This means:
/// - `range_start` is inclusive (the first number to check)
/// - `range_end` is exclusive (one past the last number to check)
///
/// Example: `FieldSize { range_start: 100, range_end: 105, range_size: 5 }`
/// represents the numbers [100, 101, 102, 103, 104] (5 numbers total).
#[allow(clippy::struct_field_names)]
#[derive(Debug, Serialize, Deserialize, Copy, Clone, PartialEq)]
pub struct FieldSize {
    range_start: u128,
    range_end: u128,
    range_size: u128,
}

impl FieldSize {
    /// Create a new `FieldSize` with a half-open range [`range_start`, `range_end`).
    ///
    /// # Panics
    /// Panics if `range_start` is greater than or equal to `range_end`.
    #[must_use]
    pub fn new(range_start: u128, range_end: u128) -> Self {
        assert!(
            range_start < range_end,
            "Range has invalid bounds, range_start must be < range_end (half-open interval)"
        );
        FieldSize {
            range_start,
            range_end,
            range_size: range_end - range_start,
        }
    }
    /// Get the first number to check in the range (`range_start`).
    #[must_use]
    pub fn first(&self) -> u128 {
        self.range_start
    }
    /// Get the last number to check in the range (`range_end` - 1).
    #[must_use]
    pub fn last(&self) -> u128 {
        self.range_end - 1
    }
    /// Get the inclusive end of the range (`range_start`).
    #[must_use]
    pub fn start(&self) -> u128 {
        self.range_start
    }
    /// Get the exclusive end of the range (`range_end`).
    #[must_use]
    pub fn end(&self) -> u128 {
        self.range_end
    }
    /// The total number of items in the range.
    #[must_use]
    pub fn size(&self) -> u128 {
        self.range_size
    }
    /// Get an iterator over the numbers in the range [`range_start`, `range_end`).
    #[must_use]
    pub fn range_iter(&self) -> std::ops::Range<u128> {
        self.range_start..self.range_end
    }
    /// Break up the range into chunks of size `chunk_size`.
    /// Each chunk is half-open [`range_start`, `range_end`).
    #[must_use]
    pub fn chunks(&self, chunk_size: u128) -> Vec<FieldSize> {
        let mut chunks = Vec::new();
        let mut start = self.range_start;

        while start < self.range_end {
            let end = (start + chunk_size).min(self.range_end);
            chunks.push(FieldSize::new(start, end));
            start = end;
        }

        chunks
    }
}
impl From<DataToClient> for FieldSize {
    fn from(data: DataToClient) -> Self {
        FieldSize::new(data.range_start, data.range_end)
    }
}
impl From<&DataToClient> for FieldSize {
    fn from(data: &DataToClient) -> Self {
        FieldSize::new(data.range_start, data.range_end)
    }
}

/// Aggregate data on the niceness of all numbers in the range.
#[derive(Debug, Serialize, Deserialize, Clone, PartialEq, Eq, Hash)]
pub struct UniquesDistributionSimple {
    pub num_uniques: u32,
    pub count: u128,
}

/// Extended version with derived stats.
#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
pub struct UniquesDistribution {
    pub num_uniques: u32,
    pub count: u128,
    pub niceness: f32,
    pub density: f32,
}

/// Individual notably nice numbers.
#[derive(Debug, Serialize, Deserialize, Clone, PartialEq, Eq, Hash)]
pub struct NiceNumberSimple {
    pub number: u128,
    pub num_uniques: u32,
}

/// Extended version with derived stats.
#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
pub struct NiceNumber {
    pub number: u128,
    pub num_uniques: u32,
    pub base: u32,
    pub niceness: f32,
}

/// A base record from the database. Used for analytics.
#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
pub struct BaseRecord {
    pub base: u32,
    pub range_start: u128,
    pub range_end: u128,
    pub range_size: u128,
    pub checked_detailed: u128,
    pub checked_niceonly: u128,
    pub minimum_cl: u8,
    pub niceness_mean: Option<f32>,
    pub niceness_stdev: Option<f32>,
    pub distribution: Vec<UniquesDistribution>,
    pub numbers: Vec<NiceNumber>,
}

/// A chunk record from the database. Used for analytics.
#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
pub struct ChunkRecord {
    pub chunk_id: u32,
    pub base: u32,
    pub range_start: u128,
    pub range_end: u128,
    pub range_size: u128,
    pub checked_detailed: u128,
    pub checked_niceonly: u128,
    pub minimum_cl: u8,
    pub niceness_mean: Option<f32>,
    pub niceness_stdev: Option<f32>,
    pub distribution: Vec<UniquesDistribution>,
    pub numbers: Vec<NiceNumber>,
}

/// A field record from the database.
/// Links to a base, a chunk, and a canon submission if any.
///
/// **Range semantics**: This represents a half-open range [`range_start`, `range_end`),
/// following Rust's standard convention where `range_start` is inclusive and `range_end` is exclusive.
#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
pub struct FieldRecord {
    pub field_id: u128,
    pub base: u32,
    pub chunk_id: Option<u32>,
    pub range_start: u128,
    pub range_end: u128,
    pub range_size: u128,
    pub last_claim_time: Option<DateTime<Utc>>,
    pub canon_submission_id: Option<u32>, // u128?
    pub check_level: u8,
    pub prioritize: bool,
}

/// A field sent to the client for processing. Used as input for processing.
/// **Range semantics**: This represents a half-open range [`range_start`, `range_end`).
#[derive(Debug, Serialize, Deserialize, Copy, Clone, PartialEq)]
pub struct DataToClient {
    pub claim_id: u128,
    pub base: u32,
    pub range_start: u128,
    pub range_end: u128,
    pub range_size: u128,
}

/// The compiled results sent to the server after processing.
#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
pub struct DataToServer {
    pub claim_id: u128,
    pub username: String,
    pub client_version: String,
    pub unique_distribution: Option<Vec<UniquesDistributionSimple>>,
    pub nice_numbers: Vec<NiceNumberSimple>,
}

/// Both the field info for processing and the compiled results
/// (for the validation self-check endpoint).
/// **Range semantics**: This represents a half-open range [`range_start`, `range_end`).
#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
pub struct ValidationData {
    pub base: u32,
    pub field_id: u128,
    pub range_start: u128,
    pub range_end: u128,
    pub range_size: u128,
    pub unique_distribution: Vec<UniquesDistributionSimple>,
    pub nice_numbers: Vec<NiceNumberSimple>,
}

/// A basic claim log from the database.
#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
pub struct ClaimRecord {
    pub claim_id: u128,
    pub field_id: u128,
    pub search_mode: SearchMode,
    pub claim_time: DateTime<Utc>,
    pub user_ip: String,
}

/// A validated submission ready to send to the database.
#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
pub struct SubmissionRecord {
    pub submission_id: u128,
    pub claim_id: u128,
    pub field_id: u128,
    pub search_mode: SearchMode,
    pub submit_time: DateTime<Utc>,
    pub elapsed_secs: f32,
    pub username: String,
    pub user_ip: String,
    pub client_version: String,
    pub disqualified: bool,
    pub distribution: Option<Vec<UniquesDistribution>>,
    pub numbers: Vec<NiceNumber>,
}

/// A submission with no metadata, used for consensus hashing.
#[derive(Debug, Serialize, Deserialize, Clone, PartialEq, Eq, Hash)]
pub struct SubmissionCandidate {
    pub distribution: Vec<UniquesDistributionSimple>,
    pub numbers: Vec<NiceNumberSimple>,
}

/// The results from processing a field or a chunk of a field.
#[derive(Debug, Serialize, Clone, PartialEq, Eq)]
pub struct FieldResults {
    pub distribution: Vec<UniquesDistributionSimple>,
    pub nice_numbers: Vec<NiceNumberSimple>,
}
