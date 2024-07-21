#!/usr/bin/env rust-script
//! ```cargo
//! [dependencies]
//! nice_common = { path = "../common" }
//! ```

use nice_common::db_util;
use nice_common::FieldRecord;

fn main() {
    // get db connection
    let mut conn = db_util::get_database_connection();

    // get all bases
    let bases = db_util::get_all_bases(&mut conn).unwrap();

    for b in bases {
        let base = b.base;

        // get all fields
        let fields_to_check: Vec<FieldRecord> =
            db_util::get_fields_in_base(&mut conn, base).unwrap();

        for field in fields_to_check {
            if let Ok(Some(canon_submission)) = db_util::update_consensus(&mut conn, &field) {
                println!(
                    "Base {} Field #{}: Canon submission is #{}",
                    base, field.field_id, canon_submission.submission_id
                );
            }
        }
    }

    // consensus
    // - only run on new submissions
    // - manual run on all submissions?
}
