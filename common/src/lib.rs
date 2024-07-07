//! A library with common utilities for dealing with square-cube pandigitals.

pub mod base_range;
pub mod benchmark;
pub mod client_api;
pub mod client_process;
pub mod db_util;
pub mod generate_chunks;
pub mod generate_fields;
pub mod residue_filter;

use clap::ValueEnum;
use malachite::natural::Natural;
use malachite::num::arithmetic::traits::{CeilingRoot, DivAssignRem, FloorRoot, Pow};
use malachite::num::conversion::traits::Digits;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::convert::TryFrom;
use std::env;
use std::ops::Add;

const CLIENT_VERSION: &str = env!("CARGO_PKG_VERSION");
const NEAR_MISS_CUTOFF_PERCENT: f32 = 0.9;

/// Each possible search mode the server and client supports.
#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord, ValueEnum)]
pub enum SearchMode {
    /// Get detailed stats on all numbers, important for long-term analytics.
    Detailed,
    /// Implements optimizations to speed up the search, usually by a factor of around 20.
    /// Does not keep statistics and cannot be quickly verified.
    Niceonly,
}

/// Data on the bounds of a search range.
/// Could be a base, chunk, field, or something else.
#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
pub struct FieldSize {
    pub start: u128,
    pub end: u128,
    pub size: u128,
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
pub struct Base {
    pub base: u32,
    pub search_start: u128,
    pub search_end: u128,
    pub search_range: u128,
}

/// A field returned from the server. Used as input for processing.
#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
pub struct FieldClaim {
    pub id: u32,
    pub username: String,
    pub base: u32,
    pub search_start: u128,
    pub search_end: u128,
    pub search_range: u128,
}

/// Aggregate data on the niceness of all numbers in the range.
#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
pub struct UniquesDistributionSimple {
    pub num_uniques: u32,
    pub count: u128,
}

/// Extended version with derived stats.
#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
pub struct UniquesDistributionExtended {
    pub num_uniques: u32,
    pub count: u128,
    pub niceness: f32,
    pub density: f32,
}

/// Individual, notably nice numbers.
#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
pub struct NiceNumbersSimple {
    pub number: Natural,
    pub num_uniques: u32,
}

/// Extended version with derived stats.
#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
pub struct NiceNumbersExtended {
    pub number: Natural,
    pub num_uniques: u32,
    pub base: u32,
    pub niceness: f32,
}

/// The compiled results sent to the server after processing. Options for both modes.
#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
pub struct FieldSubmit {
    pub id: u32,
    pub username: String,
    pub client_version: String, // TODO: user-agent with repo/version/build
    pub unique_distribution: Option<HashMap<u32, u32>>,
    pub nice_list: HashMap<u128, u32>,
}
