#!/usr/bin/env rust-script
//! ```cargo
//! [dependencies]
//! nice_common = { path = "../common" }
//! ```

use nice_common::base_range::get_base_range_u128;
use nice_common::msd_prefix_filter::MSD_RECURSIVE_MIN_RANGE_SIZE;

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

                // Calculate depth needed to subdivide base range into min_range
                let msd_min_range_size = 10_000;
                let depth = (base_range.size() as f64 / msd_min_range_size as f64)
                    .log2()
                    .ceil() as u32;
                println!(
                    "         Subdivision depth to hit min range {msd_min_range_size:.0e}: 2^{depth}"
                )
            }
            None => {
                continue;
            }
        }
    }
}
