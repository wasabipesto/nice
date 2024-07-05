//! A library with common utilities for dealing with square-cube pandigitals.

pub mod base_range;
pub mod benchmark;
pub mod client_api;
pub mod client_process;
pub mod residue_filter;

use clap::ValueEnum;
use client_api::deserialize_string_to_u128;
use malachite::natural::Natural;
use malachite::num::arithmetic::traits::{CeilingRoot, DivAssignRem, FloorRoot, Pow};
use malachite::num::basic::traits::Zero;
use malachite::num::conversion::traits::Digits;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::convert::TryFrom;
use std::env;

const CLIENT_VERSION: &str = env!("CARGO_PKG_VERSION");
const MAX_SUPPORTED_BASE_NORMAL: u32 = 97;
const NEAR_MISS_CUTOFF_PERCENT: f32 = 0.9;

/// Each possible search mode the server and client supports.
#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord, ValueEnum)]
pub enum SearchMode {
    Detailed,
    Niceonly,
}

/// A field returned from the server. Used as input for processing.
#[derive(Debug, Deserialize, Clone)]
pub struct FieldClaim {
    pub id: u32,
    pub username: String,
    pub base: u32,
    #[serde(deserialize_with = "deserialize_string_to_u128")]
    pub search_start: u128,
    #[serde(deserialize_with = "deserialize_string_to_u128")]
    pub search_end: u128,
    #[serde(deserialize_with = "deserialize_string_to_u128")]
    pub search_range: u128,
}

/// The compiled results sent to the server after processing. Options for both modes.
#[derive(Debug, Serialize, PartialEq)]
pub struct FieldSubmit {
    pub id: u32,
    pub username: String,
    pub client_version: String, // TODO: user-agent with repo/version/build
    pub unique_count: Option<HashMap<u32, u32>>,
    pub near_misses: Option<HashMap<String, u32>>,
    pub nice_list: Option<Vec<String>>,
}
