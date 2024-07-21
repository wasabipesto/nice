#!/usr/bin/env rust-script
//! ```cargo
//! [dependencies]
//! nice_common = { path = "../common" }
//! ```

use nice_common::db_util;
use nice_common::FieldRecord;

// this takes 30 minutes to process all fields through base 46
// in order to make this sustainable for much larger databases:
// TODO: runs on all fields with submissions within the last 24 hours
// TODO: runs on 10000 random fields

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
            let (canon_submission, check_level) =
                db_util::update_consensus(&mut conn, &field).unwrap();
            if let Some(canon_submission_some) = canon_submission {
                println!(
                    "Base {} Field #{} - Canon submission: #{}, CL{}",
                    base, field.field_id, canon_submission_some.submission_id, check_level
                );
            };
        }
    }
}
