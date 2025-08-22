//! WebAssembly interface for nice number processing
//!
//! This module provides a browser-compatible client for the distributed computing
//! project that finds "nice numbers" (square-cube pandigitals).

use serde::{Deserialize, Serialize};
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
use std::collections::HashMap;

// Server response format (matches the actual server API)
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ServerDataToClient {
    pub claim_id: u128,
    pub base: u32,
    pub range_start: u128,
    pub range_end: u128,
    pub range_size: u128,
}

// JavaScript-compatible format (all large numbers as strings)
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct DataToClient {
    pub claim_id: String,
    pub base: u32,
    pub range_start: String,
    pub range_end: String,
    pub range_size: String,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct NiceNumberSimple {
    pub number: String,
    pub num_uniques: u32,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct UniquesDistributionSimple {
    pub num_uniques: u32,
    pub count: String,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct DataToServer {
    pub claim_id: u128, // Server expects this as integer
    pub username: String,
    pub client_version: String,
    pub unique_distribution: Option<Vec<UniquesDistributionSimple>>,
    pub nice_numbers: Vec<NiceNumberSimple>,
}

// Constants from the common library
const CLIENT_VERSION: &str = "3.0.0-wasm";
const NEAR_MISS_CUTOFF_PERCENT: f32 = 0.9;

#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_namespace = console)]
    fn log(s: &str);
}

macro_rules! console_log {
    ($($t:tt)*) => (log(&format_args!($($t)*).to_string()))
}

/// Convert server data format to JavaScript-compatible format
fn convert_server_data(server_data: ServerDataToClient) -> DataToClient {
    DataToClient {
        claim_id: server_data.claim_id.to_string(),
        base: server_data.base,
        range_start: server_data.range_start.to_string(),
        range_end: server_data.range_end.to_string(),
        range_size: server_data.range_size.to_string(),
    }
}

/// Calculate the number of unique digits in (n^2, n^3) represented in base b.
fn get_num_unique_digits(num_str: &str, base: u32) -> u32 {
    // Parse the number string as a Natural
    let num = match Natural::from_string_base(10, num_str) {
        Some(n) => n,
        None => return 0,
    };

    // Create an indicator variable as a boolean array
    let mut digits_indicator: u128 = 0;

    // Square the number, convert to base and save the digits
    let squared = (&num).pow(2);
    for digit in squared.to_digits_asc(&base) {
        if digit < 128 {
            digits_indicator |= 1 << digit;
        }
    }

    // Cube, convert to base and save the digits
    let cubed = squared * &num;
    for digit in cubed.to_digits_asc(&base) {
        if digit < 128 {
            digits_indicator |= 1 << digit;
        }
    }

    // Return the number of unique digits
    digits_indicator.count_ones()
}

/// Convert server JSON data to client-compatible format
#[wasm_bindgen]
pub fn convert_server_response(server_json: &str) -> String {
    match serde_json::from_str::<ServerDataToClient>(server_json) {
        Ok(server_data) => {
            let client_data = convert_server_data(server_data);
            match serde_json::to_string(&client_data) {
                Ok(json) => json,
                Err(e) => {
                    console_log!("Error converting to client format: {}", e);
                    "{}".to_string()
                }
            }
        }
        Err(e) => {
            console_log!("Error parsing server response: {}", e);
            "{}".to_string()
        }
    }
}

/// Process a field in "nice-only" mode (faster, only finds 100% nice numbers)
#[wasm_bindgen]
pub fn process_niceonly(claim_data_json: &str, username: &str) -> String {
    let claim_data: DataToClient = match serde_json::from_str(claim_data_json) {
        Ok(data) => data,
        Err(e) => {
            console_log!("Error parsing claim data: {}", e);
            // Return a valid JSON response even on error
            let error_response = DataToServer {
                claim_id: 0,
                username: username.to_string(),
                client_version: CLIENT_VERSION.to_string(),
                unique_distribution: None,
                nice_numbers: Vec::new(),
            };
            return serde_json::to_string(&error_response).unwrap_or("{}".to_string());
        }
    };

    let base = claim_data.base;
    let range_start = match claim_data.range_start.parse::<u128>() {
        Ok(n) => n,
        Err(_) => {
            console_log!("Error parsing range_start: {}", claim_data.range_start);
            let error_response = DataToServer {
                claim_id: 0,
                username: username.to_string(),
                client_version: CLIENT_VERSION.to_string(),
                unique_distribution: None,
                nice_numbers: Vec::new(),
            };
            return serde_json::to_string(&error_response).unwrap_or("{}".to_string());
        }
    };
    let range_end = match claim_data.range_end.parse::<u128>() {
        Ok(n) => n,
        Err(_) => {
            console_log!("Error parsing range_end: {}", claim_data.range_end);
            let error_response = DataToServer {
                claim_id: 0,
                username: username.to_string(),
                client_version: CLIENT_VERSION.to_string(),
                unique_distribution: None,
                nice_numbers: Vec::new(),
            };
            return serde_json::to_string(&error_response).unwrap_or("{}".to_string());
        }
    };

    let claim_id = match claim_data.claim_id.parse::<u128>() {
        Ok(n) => n,
        Err(_) => {
            console_log!("Error parsing claim_id: {}", claim_data.claim_id);
            0
        }
    };

    let mut nice_numbers: Vec<NiceNumberSimple> = Vec::new();

    // Process the range in chunks for better responsiveness
    let chunk_size = 1000;
    let mut current = range_start;

    while current < range_end {
        let chunk_end = std::cmp::min(current + chunk_size, range_end);

        for num in current..chunk_end {
            let num_uniques = get_num_unique_digits(&num.to_string(), base);

            // Only keep 100% nice numbers (all digits used)
            if num_uniques == base {
                nice_numbers.push(NiceNumberSimple {
                    number: num.to_string(),
                    num_uniques,
                });
            }
        }

        current = chunk_end;
    }

    let submit_data = DataToServer {
        claim_id,
        username: username.to_string(),
        client_version: CLIENT_VERSION.to_string(),
        unique_distribution: None,
        nice_numbers,
    };

    match serde_json::to_string(&submit_data) {
        Ok(json) => json,
        Err(e) => {
            console_log!("Error serializing submit data: {}", e);
            let error_response = DataToServer {
                claim_id,
                username: username.to_string(),
                client_version: CLIENT_VERSION.to_string(),
                unique_distribution: None,
                nice_numbers: Vec::new(),
            };
            serde_json::to_string(&error_response).unwrap_or("{}".to_string())
        }
    }
}

/// Process a field in "detailed" mode (slower, collects statistics)
#[wasm_bindgen]
pub fn process_detailed(claim_data_json: &str, username: &str) -> String {
    let claim_data: DataToClient = match serde_json::from_str(claim_data_json) {
        Ok(data) => data,
        Err(e) => {
            console_log!("Error parsing claim data: {}", e);
            let error_response = DataToServer {
                claim_id: 0,
                username: username.to_string(),
                client_version: CLIENT_VERSION.to_string(),
                unique_distribution: Some(Vec::new()),
                nice_numbers: Vec::new(),
            };
            return serde_json::to_string(&error_response).unwrap_or("{}".to_string());
        }
    };

    let base = claim_data.base;
    let range_start = match claim_data.range_start.parse::<u128>() {
        Ok(n) => n,
        Err(_) => {
            console_log!("Error parsing range_start: {}", claim_data.range_start);
            let error_response = DataToServer {
                claim_id: 0,
                username: username.to_string(),
                client_version: CLIENT_VERSION.to_string(),
                unique_distribution: Some(Vec::new()),
                nice_numbers: Vec::new(),
            };
            return serde_json::to_string(&error_response).unwrap_or("{}".to_string());
        }
    };
    let range_end = match claim_data.range_end.parse::<u128>() {
        Ok(n) => n,
        Err(_) => {
            console_log!("Error parsing range_end: {}", claim_data.range_end);
            let error_response = DataToServer {
                claim_id: 0,
                username: username.to_string(),
                client_version: CLIENT_VERSION.to_string(),
                unique_distribution: Some(Vec::new()),
                nice_numbers: Vec::new(),
            };
            return serde_json::to_string(&error_response).unwrap_or("{}".to_string());
        }
    };

    let claim_id = match claim_data.claim_id.parse::<u128>() {
        Ok(n) => n,
        Err(_) => {
            console_log!("Error parsing claim_id: {}", claim_data.claim_id);
            0
        }
    };

    let nice_list_cutoff = (base as f32 * NEAR_MISS_CUTOFF_PERCENT) as u32;
    let mut nice_numbers: Vec<NiceNumberSimple> = Vec::new();
    let mut unique_distribution_map: HashMap<u32, u128> = (1..=base).map(|i| (i, 0u128)).collect();

    // Process the range in chunks
    let chunk_size = 1000;
    let mut current = range_start;

    while current < range_end {
        let chunk_end = std::cmp::min(current + chunk_size, range_end);

        for num in current..chunk_end {
            let num_uniques = get_num_unique_digits(&num.to_string(), base);

            // Update distribution
            if let Some(count) = unique_distribution_map.get_mut(&num_uniques) {
                *count += 1;
            }

            // Collect nice numbers above threshold
            if num_uniques > nice_list_cutoff {
                nice_numbers.push(NiceNumberSimple {
                    number: num.to_string(),
                    num_uniques,
                });
            }
        }

        current = chunk_end;
    }

    let mut submit_distribution: Vec<UniquesDistributionSimple> = unique_distribution_map
        .into_iter()
        .map(|(num_uniques, count)| UniquesDistributionSimple {
            num_uniques,
            count: count.to_string(),
        })
        .collect();
    submit_distribution.sort_by_key(|d| d.num_uniques);

    let submit_data = DataToServer {
        claim_id,
        username: username.to_string(),
        client_version: CLIENT_VERSION.to_string(),
        unique_distribution: Some(submit_distribution),
        nice_numbers,
    };

    match serde_json::to_string(&submit_data) {
        Ok(json) => json,
        Err(e) => {
            console_log!("Error serializing submit data: {}", e);
            let error_response = DataToServer {
                claim_id,
                username: username.to_string(),
                client_version: CLIENT_VERSION.to_string(),
                unique_distribution: Some(Vec::new()),
                nice_numbers: Vec::new(),
            };
            serde_json::to_string(&error_response).unwrap_or("{}".to_string())
        }
    }
}

/// Get a benchmark field for testing (offline mode)
#[wasm_bindgen]
pub fn get_benchmark_field() -> String {
    let benchmark_data = DataToClient {
        claim_id: "0".to_string(),
        base: 10,
        range_start: "1000".to_string(),
        range_end: "2000".to_string(),
        range_size: "1000".to_string(),
    };

    match serde_json::to_string(&benchmark_data) {
        Ok(json) => json,
        Err(_) => "{}".to_string(),
    }
}

/// Simple utility to test WASM loading
#[wasm_bindgen]
pub fn greet(name: &str) {
    console_log!("Hello, {}! WASM module loaded successfully.", name);
}
