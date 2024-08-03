#!/usr/bin/env rust-script
//! ```cargo
//! [dependencies]
//! nice_common = { path = "../common" }
//! ```

use nice_common::consensus;
use nice_common::db_util;
use nice_common::FieldRecord;
use std::time::Instant;

// this runs at about 20-25 fields per second on a remote machine for single-submission fields
// more complex comparisons drop this to 5-10 fields per second
// takes ~4 minutes total for ~5000 fields as of 8/3/24
// TODO: only run on fields with new submissions
// TODO: get fields to check and their submissions in one operation

fn main() {
    // get db connection
    let mut conn = db_util::get_database_connection();

    // start timer and other process variables
    let timer = Instant::now();
    let mut num_processed = 0u32;

    // get all bases
    let bases = db_util::get_all_bases(&mut conn).unwrap();
    for b in bases {
        let base = b.base;

        // get all fields
        let fields_to_check: Vec<FieldRecord> =
            db_util::get_fields_in_base_with_detailed_subs(&mut conn, base).unwrap();

        for field in fields_to_check {
            // Get all qualified and detailed submissions for the field
            let submissions =
                db_util::get_submissions_qualified_detailed_for_field(&mut conn, field.field_id)
                    .unwrap();

            // Establish the consensus
            let (canon_submission, check_level) =
                consensus::evaluate_consensus(&field, &submissions).unwrap();

            match &canon_submission {
                None => {
                    if field.canon_submission_id.is_some() || field.check_level > 1 {
                        println!(
                        "Field #{} claimed to be checked (Submission #{:?}, CL{}) but no submissions were found, so it was reset to CL{}.",
                        field.field_id, field.canon_submission_id, field.check_level, check_level
                    );
                        db_util::update_field_canon_and_cl(
                            &mut conn,
                            field.field_id,
                            None,
                            check_level,
                        )
                        .unwrap();
                    }
                }
                Some(canon_submission_some) => {
                    // Update the field if necessary
                    if field.canon_submission_id != Some(canon_submission_some.submission_id as u32)
                        || field.check_level != check_level
                    {
                        db_util::update_field_canon_and_cl(
                            &mut conn,
                            field.field_id,
                            Some(canon_submission_some.submission_id as u32),
                            check_level,
                        )
                        .unwrap();
                    }
                    println!(
                        "Base {}: Field #{} - CL{}, Canon Submission: #{}",
                        base, field.field_id, check_level, canon_submission_some.submission_id
                    );
                }
            }

            num_processed += 1;
            if num_processed % 1000 == 0 {
                println!(
                    "Processed {} fields so far at an average of {:.3} fields/second",
                    num_processed,
                    num_processed as f64 / timer.elapsed().as_secs_f64()
                );
            }
        }
    }
    println!(
        "Final: Processed {} fields at an average of {:.3} fields/second",
        num_processed,
        num_processed as f64 / timer.elapsed().as_secs_f64()
    );
}
