#!/usr/bin/env rust-script
//! ```cargo
//! [dependencies]
//! nice_common = { path = "../common" }
//! ```

use nice_common::base_range::get_base_range_u128;

fn main() {
    for base in 10..=97 {
        match get_base_range_u128(base).unwrap() {
            Some(base_range) => {
                println!(
                    "Base {base}, Range Start: {start:.2e}, Range End: {end:.2e}, Size: {size:.2e}",
                    start = base_range.start(),
                    end = base_range.end(),
                    size = base_range.size(),
                );
            }
            None => {
                continue;
            }
        }
    }
}
