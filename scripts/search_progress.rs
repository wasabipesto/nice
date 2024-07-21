#!/usr/bin/env rust-script
//! ```cargo
//! [dependencies]
//! nice_common = { path = "../common" }
//! ```

use nice_common::{db_util, FieldSize};

fn main() {
    // get db connection
    let mut conn = db_util::get_database_connection();

    // get all bases
    let bases = db_util::get_all_bases(&mut conn).unwrap();

    for b in bases {
        let base = b.base;
        let base_size = FieldSize {
            range_start: b.range_start,
            range_end: b.range_end,
            range_size: b.range_size,
        };
        let complete_count = db_util::get_count_checked_by_range(&mut conn, 2, base_size).unwrap();
        let complete_pct = complete_count as f32 / b.range_size as f32 * 100f32;
        println!(
            "Base {}: {}/{} ({:.2?}%)",
            base, complete_count, b.range_size, complete_pct
        );
    }
}
