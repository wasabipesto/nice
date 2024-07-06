//! A library with common utilities for dealing with square-cube pandigitals.

pub mod base_range;
pub mod benchmark;
pub mod client_api;
pub mod client_process;
pub mod generate_fields;
pub mod residue_filter;

use clap::ValueEnum;
use malachite::natural::Natural;
use malachite::num::arithmetic::traits::{CeilingRoot, DivAssignRem, FloorRoot, Pow};
use malachite::num::conversion::traits::Digits;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
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

/// Information on searchable fields.
#[derive(Debug, PartialEq)]
pub struct SearchField {
    pub start: Natural,
    pub end: Natural,
    pub size: u128,
}

/// A field returned from the server. Used as input for processing.
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct FieldClaim {
    pub id: u32,
    pub username: String,
    pub base: u32,
    pub search_start: u128,
    pub search_end: u128,
    pub search_range: u128,
}

/// The compiled results sent to the server after processing. Options for both modes.
#[derive(Debug, Serialize, Deserialize, PartialEq)]
pub struct FieldSubmit {
    pub id: u32,
    pub username: String,
    pub client_version: String, // TODO: user-agent with repo/version/build
    pub unique_count: Option<HashMap<u32, u32>>,
    pub near_misses: Option<HashMap<String, u32>>,
    pub nice_list: Option<Vec<String>>,
}
