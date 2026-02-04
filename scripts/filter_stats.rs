#!/usr/bin/env rust-script
//! ```cargo
//! [dependencies]
//! nice_common = { path = "../common" }
//! ```

use nice_common::base_range::get_base_range_u128;
use nice_common::lsd_filter::get_valid_lsds;
use nice_common::residue_filter::get_residue_filter;

fn main() {
    for base in 10..=60 {
        match get_base_range_u128(base).unwrap() {
            Some(base_range) => {
                let lsd_valid = get_valid_lsds(&base).len() as f64;
                let residue_valid = get_residue_filter(&base).len() as f64;
                let base_f64 = base as f64;
                let base_minus_one = (base - 1) as f64;

                let lsd_filter_rate = ((base_f64 - lsd_valid) / base_f64) * 100.0;
                let residue_filter_rate =
                    ((base_minus_one - residue_valid) / base_minus_one) * 100.0;

                println!(
                    "Base {}: LSD filter rate = {:.2}%, Residue filter rate = {:.2}%",
                    base, lsd_filter_rate, residue_filter_rate
                );
            }
            None => {
                continue;
            }
        }
    }
}
