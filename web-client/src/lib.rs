//! WebAssembly interface for nice number processing with Web Worker support
//!
//! This module provides a browser-compatible client for the distributed computing
//! project that finds "nice numbers" (square-cube pandigitals).

use wasm_bindgen::prelude::*;

// Use `wee_alloc` as the global allocator for smaller WASM size
#[global_allocator]
static ALLOC: wee_alloc::WeeAlloc = wee_alloc::WeeAlloc::INIT;

// Define the panic hook for better error messages in the browser
#[wasm_bindgen(start)]
pub fn main() {
    console_error_panic_hook::set_once();
}

// Import types from the common library
use malachite::natural::Natural;
use malachite::num::arithmetic::traits::Pow;
use malachite::num::conversion::traits::{Digits, FromStringBase};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_namespace = console)]
    fn log(s: &str);
}

#[derive(Serialize, Deserialize)]
struct NiceNumber {
    number: String,
    num_uniques: u32,
}

#[derive(Serialize, Deserialize)]
struct DistributionEntry {
    num_uniques: u32,
    count: u64,
}

#[derive(Serialize, Deserialize)]
struct ChunkResult {
    nice_numbers: Vec<NiceNumber>,
    distribution_updates: Vec<DistributionEntry>,
    processed_count: u64,
}

/// Process a chunk of numbers and return nice numbers and distribution updates
#[wasm_bindgen]
pub fn process_chunk_wasm(range_start_str: &str, range_end_str: &str, base: u32) -> String {
    let range_start = match Natural::from_string_base(10, range_start_str) {
        Some(n) => n,
        None => return "{}".to_string(),
    };

    let range_end = match Natural::from_string_base(10, range_end_str) {
        Some(n) => n,
        None => return "{}".to_string(),
    };

    let nice_cutoff = (base as f64 * 0.9).floor() as u32;
    let mut nice_numbers = Vec::new();
    let mut distribution = HashMap::new();
    let mut processed_count = 0u64;

    // Initialize distribution map
    for i in 1..=base {
        distribution.insert(i, 0u64);
    }

    let mut current = range_start;
    while current < range_end {
        let num_uniques = get_num_unique_digits(&current, base);

        // Update distribution
        *distribution.entry(num_uniques).or_insert(0) += 1;

        // Check if it's a nice number
        if num_uniques > nice_cutoff {
            nice_numbers.push(NiceNumber {
                number: current.to_string(),
                num_uniques,
            });
        }

        processed_count += 1;
        current += Natural::from(1u32);
    }

    // Convert distribution to sorted vector
    let mut distribution_updates: Vec<DistributionEntry> = distribution
        .into_iter()
        .map(|(num_uniques, count)| DistributionEntry { num_uniques, count })
        .collect();
    distribution_updates.sort_by_key(|entry| entry.num_uniques);

    let result = ChunkResult {
        nice_numbers,
        distribution_updates,
        processed_count,
    };

    match serde_json::to_string(&result) {
        Ok(json) => json,
        Err(_) => "{}".to_string(),
    }
}

/// Internal function to calculate unique digits for a Natural number
fn get_num_unique_digits(num: &Natural, base: u32) -> u32 {
    // Create an indicator variable as a boolean array
    let mut digits_indicator: u128 = 0;

    // Square the number, convert to base and save the digits
    let squared = num.pow(2);
    for digit in squared.to_digits_asc(&base) {
        if digit < 128 {
            digits_indicator |= 1 << digit;
        }
    }

    // Cube, convert to base and save the digits
    let cubed = &squared * num;
    for digit in cubed.to_digits_asc(&base) {
        if digit < 128 {
            digits_indicator |= 1 << digit;
        }
    }

    // Return the number of unique digits
    digits_indicator.count_ones()
}
