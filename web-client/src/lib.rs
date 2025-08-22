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

#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_namespace = console)]
    fn log(s: &str);
}

macro_rules! console_log {
    ($($t:tt)*) => (log(&format_args!($($t)*).to_string()))
}

/// Calculate the number of unique digits in (n^2, n^3) represented in base b.
/// This is the core function used by the Web Worker for computation.
#[wasm_bindgen]
pub fn get_num_unique_digits_wasm(num_str: &str, base: u32) -> u32 {
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

/// Simple utility to test WASM loading in Web Worker
#[wasm_bindgen]
pub fn greet(name: &str) {
    console_log!(
        "Hello, {}! WASM module loaded successfully in Web Worker.",
        name
    );
}
