//! WebAssembly interface for nice number processing with Web Worker support
//!
//! This module provides a browser-compatible client for the distributed computing
//! project that finds "nice numbers" (square-cube pandigitals).

use nice_common::client_process::process_detailed_unwrapped;
use nice_common::{NiceNumberSimple, UniquesDistributionSimple};
use serde::{Deserialize, Serialize};
use std::str::FromStr;
use wasm_bindgen::prelude::*;

// Define the panic hook for better error messages in the browser
#[wasm_bindgen(start)]
pub fn main() {
    console_error_panic_hook::set_once();
}

#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_namespace = console)]
    fn log(s: &str);
}

#[derive(Serialize, Deserialize)]
struct ChunkResult {
    nice_numbers: Vec<NiceNumberSimple>,
    distribution_updates: Vec<UniquesDistributionSimple>,
}

/// Process a chunk of numbers and return nice numbers and distribution updates
#[wasm_bindgen]
pub fn process_chunk_wasm(range_start_str: &str, range_end_str: &str, base: u32) -> String {
    console_error_panic_hook::set_once();

    // get range start and end
    let range_start = u128::from_str(range_start_str).unwrap();
    let range_end = u128::from_str(range_end_str).unwrap();

    // pass off to common
    let (distribution_updates, nice_numbers) =
        process_detailed_unwrapped(range_start, range_end, base);

    // package up for export
    let result = ChunkResult {
        nice_numbers,
        distribution_updates,
    };
    serde_json::to_string(&result).unwrap()
}
