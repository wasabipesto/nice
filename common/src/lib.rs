//! A library with common utilities for dealing with square-cube pandigitals.

#![allow(clippy::wildcard_imports)]

pub mod base_range;
pub mod benchmark;
#[cfg(any(feature = "openssl-tls", feature = "rustls-tls"))]
pub mod client_api;
pub mod client_process;
pub mod consensus;
#[cfg(feature = "database")]
pub mod db_util;
pub mod distribution_stats;
pub mod generate_chunks;
pub mod generate_fields;
pub mod number_stats;
pub mod residue_filter;

use chrono::{DateTime, Utc};
use clap::ValueEnum;
#[cfg(feature = "database")]
use dotenvy::dotenv;
use itertools::Itertools;
use malachite::base::num::arithmetic::traits::{CeilingRoot, DivAssignRem, FloorRoot, Pow};
use malachite::base::num::conversion::traits::Digits;
use malachite::natural::Natural;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::convert::TryFrom;
use std::env;
use std::ops::Add;

pub const CLIENT_VERSION: &str = env!("CARGO_PKG_VERSION");
pub const NEAR_MISS_CUTOFF_PERCENT: f32 = 0.9;
pub const DOWNSAMPLE_CUTOFF_PERCENT: f32 = 0.2;
pub const CLAIM_DURATION_HOURS: i64 = 1;
pub const DEFAULT_FIELD_SIZE: u128 = 1_000_000_000;
pub const PROCESSING_CHUNK_SIZE: usize = 10_000;
pub const SAVE_TOP_N_NUMBERS: usize = 10000;

/// Each possible search mode the server and client supports.
#[derive(Debug, Serialize, Deserialize, Copy, Clone, PartialEq, Eq, PartialOrd, Ord, ValueEnum)]
pub enum SearchMode {
    /// Get detailed stats on all numbers, important for long-term analytics.
    Detailed,
    /// Implements optimizations to speed up the search, usually by a factor of around 20.
    /// Does not keep statistics and cannot be quickly verified.
    Niceonly,
}

/// Whether we should pick the next or random field when claiming.
#[derive(Debug, Copy, Clone)]
pub enum FieldClaimStrategy {
    Next,
    Random,
}

/// Data on the bounds of a search range.
/// Could be a base, chunk, field, or something else.
#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
pub struct FieldSize {
    pub range_start: u128,
    pub range_end: u128,
    pub range_size: u128,
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
#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
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
