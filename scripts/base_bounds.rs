#!/usr/bin/env rust-script
//! ```cargo
//! [dependencies]
//! nice_common = { path = "../common" }
//! serde = { version = "1.0", features = ["derive"] }
//! serde_json = { version = "1.0" }
//! ```

use nice_common::base_range::get_base_range_u128;
use serde::{Deserialize, Serialize};
use serde_json;

#[derive(Debug, Serialize, Deserialize, Copy, Clone, PartialEq)]
pub struct BaseRange {
    base: u32,
    start: u128,
    end: u128,
}

fn main() {
    let mut results = Vec::new();
    for base in 10..=97 {
        match get_base_range_u128(base).unwrap() {
            Some(range) => {
                let base_data = BaseRange {
                    base: base,
                    start: range.start(),
                    end: range.end(),
                };
                results.push(base_data)
            }
            None => {
                continue;
            }
        }
    }
    println!("{}", serde_json::to_string_pretty(&results).unwrap());
}
