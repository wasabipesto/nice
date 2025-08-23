//! WebAssembly interface for nice number processing with Web Worker support
//!
//! This module provides a browser-compatible client for the distributed computing
//! project that finds "nice numbers" (square-cube pandigitals).

use wasm_bindgen::prelude::*;

// Define the panic hook for better error messages in the browser
#[wasm_bindgen(start)]
pub fn main() {
    console_error_panic_hook::set_once();
}

// Import types from the common library
use itertools::Itertools;
use malachite::natural::Natural;
use malachite::num::arithmetic::traits::Pow;
use malachite::num::conversion::traits::Digits;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::str::FromStr;

pub const NEAR_MISS_CUTOFF_PERCENT: f32 = 0.9;

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
    count: u128,
}

#[derive(Serialize, Deserialize)]
struct ChunkResult {
    nice_numbers: Vec<NiceNumber>,
    distribution_updates: Vec<DistributionEntry>,
}

/// Process a chunk of numbers and return nice numbers and distribution updates
#[wasm_bindgen]
pub fn process_chunk_wasm(range_start_str: &str, range_end_str: &str, base: u32) -> String {
    console_error_panic_hook::set_once();

    // get range start and end
    let range_start = u128::from_str(range_start_str).unwrap();
    let range_end = u128::from_str(range_end_str).unwrap();

    // calculate the minimum num_unique_digits cutoff (default 90% of the base)
    let nice_list_cutoff = (base as f32 * NEAR_MISS_CUTOFF_PERCENT) as u32;

    // set up outputs
    let mut nice_numbers = Vec::new();
    let mut unique_distribution_map: HashMap<u32, u128> = (1..=base).map(|i| (i, 0u128)).collect();

    // break up the range into chunks
    let chunk_size: usize = 10_000;
    let chunks = (range_start..range_end).chunks(chunk_size);

    // process everything, saving results and aggregating after each chunk finishes
    for chunk in &chunks {
        // get chunk results
        let chunk_results: Vec<(u128, u32)> = chunk
            .map(|num| (num, get_num_unique_digits(num, base)))
            .collect();

        // aggregate unique_distribution
        for (bin_uniques, total_count) in unique_distribution_map.iter_mut() {
            let chunk_count = chunk_results
                .iter()
                .filter(|(_, num_unique_digits)| num_unique_digits == bin_uniques)
                .count() as u128;
            *total_count += chunk_count;
        }

        // collect nice numbers
        nice_numbers.extend(
            chunk_results
                .into_iter()
                .filter(|(_, num_unique_digits)| num_unique_digits > &nice_list_cutoff)
                .map(|(num, num_unique_digits)| NiceNumber {
                    number: num.to_string(),
                    num_uniques: num_unique_digits,
                }),
        );
    }

    // Convert distribution to sorted vector
    let mut distribution_updates: Vec<DistributionEntry> = unique_distribution_map
        .into_iter()
        .map(|(num_uniques, count)| DistributionEntry { num_uniques, count })
        .collect();
    distribution_updates.sort_by_key(|entry| entry.num_uniques);

    let result = ChunkResult {
        nice_numbers,
        distribution_updates,
    };

    serde_json::to_string(&result).unwrap()
}

/// Internal function to calculate unique digits for a Natural number
fn get_num_unique_digits(num_u128: u128, base: u32) -> u32 {
    // 🔥🔥🔥 HOT LOOP 🔥🔥🔥

    // create an indicator variable as a boolean array
    let mut digits_indicator: Vec<bool> = vec![false; base as usize];

    // convert u128 to natural
    let num = Natural::from(num_u128);

    // square the number, convert to base and save the digits
    // tried using foiled out versions but malachite is already pretty good
    let squared = (&num).pow(2);
    for digit in squared.to_digits_asc(&base) {
        digits_indicator[digit as usize] = true;
    }

    // cube, convert to base and save the digits
    let cubed = squared * &num;
    for digit in cubed.to_digits_asc(&base) {
        digits_indicator[digit as usize] = true;
    }

    // output the number of unique digits
    let mut num_unique_digits = 0;

    for digit in digits_indicator {
        if digit {
            num_unique_digits += 1
        }
    }

    num_unique_digits
}
