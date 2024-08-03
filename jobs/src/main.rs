//! Scheduled jobs for the nice project.

#![warn(clippy::all, clippy::pedantic)]
#![allow(clippy::too_many_lines)]

use nice_common::consensus;
use nice_common::db_util;
use nice_common::distribution_stats;
use nice_common::number_stats;
use nice_common::DOWNSAMPLE_CUTOFF_PERCENT;
use nice_common::{FieldRecord, SubmissionRecord};

fn main() {
    // get db connection
    let mut conn = db_util::get_database_connection();
    println!("Database connection established. Scheduled jobs started.");

    // get all bases
    let bases = db_util::get_all_bases(&mut conn).unwrap();
    for base_record in bases {
        let base = base_record.base;

        println!("=== BASE {base} CONSENSUS ===");

        // get all fields
        // TODO: get fields to check and their submissions in one operation
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
                        "WARNING: Field #{} claimed to be checked (Submission #{:?}, CL{}) but no submissions were found, so it was reset to CL{}.",
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
                #[allow(clippy::cast_possible_truncation)] // TODO: fix submission_id type mismatch
                Some(sub) => {
                    print!(
                        "Field #{}: CL{}, Canon Submission #{}, ",
                        field.field_id, check_level, sub.submission_id
                    );
                    // Update the field if necessary
                    if field.canon_submission_id != Some(sub.submission_id as u32)
                        || field.check_level != check_level
                    {
                        db_util::update_field_canon_and_cl(
                            &mut conn,
                            field.field_id,
                            Some(sub.submission_id as u32),
                            check_level,
                        )
                        .unwrap();
                        println!("Updated!");
                    } else {
                        println!("No change.");
                    }
                }
            }
        }

        println!();
        println!("=== BASE {base} DOWNSAMPLING ===");

        // get basic stats like how much has been cheked
        let base_checked_niceonly = db_util::get_count_checked_by_range(
            &mut conn,
            1,
            base_record.range_start,
            base_record.range_end,
        )
        .unwrap();
        let base_checked_detailed = db_util::get_count_checked_by_range(
            &mut conn,
            2,
            base_record.range_start,
            base_record.range_end,
        )
        .unwrap();

        #[allow(clippy::cast_precision_loss)]
        let base_percent_checked_detailed =
            base_checked_detailed as f32 / base_record.range_size as f32;
        let base_minimum_cl = db_util::get_minimum_cl_by_range(
            &mut conn,
            base_record.range_start,
            base_record.range_end,
        )
        .unwrap();

        // create vec for all fields in the base
        let mut base_submissions: Vec<SubmissionRecord> = Vec::new();

        // loop thorugh chunks in the base
        let chunks = db_util::get_chunks_in_base(&mut conn, base).unwrap();
        for chunk in chunks {
            let chunk_size = chunk.range_size;
            print!("Chunk #{}: ", chunk.chunk_id);

            // get basic stats like how much has been cheked
            let checked_niceonly = db_util::get_count_checked_by_range(
                &mut conn,
                1,
                chunk.range_start,
                chunk.range_end,
            )
            .unwrap();
            let checked_detailed = db_util::get_count_checked_by_range(
                &mut conn,
                2,
                chunk.range_start,
                chunk.range_end,
            )
            .unwrap();

            #[allow(clippy::cast_precision_loss)]
            let chunk_percent_checked_detailed = checked_detailed as f32 / chunk_size as f32;
            let minimum_cl =
                db_util::get_minimum_cl_by_range(&mut conn, chunk.range_start, chunk.range_end)
                    .unwrap();
            print!(
                "CL{}, Checked {:.0}%, ",
                minimum_cl,
                chunk_percent_checked_detailed * 100f32
            );

            // get all submissions for the chunk
            let mut submissions: Vec<SubmissionRecord> = db_util::get_canon_submissions_by_range(
                &mut conn,
                chunk.range_start,
                chunk.range_end,
            )
            .unwrap();

            // update chunk record
            let mut updated_chunk = chunk.clone();
            updated_chunk.checked_niceonly = checked_niceonly;
            updated_chunk.checked_detailed = checked_detailed;
            updated_chunk.minimum_cl = minimum_cl;
            if chunk_percent_checked_detailed > DOWNSAMPLE_CUTOFF_PERCENT {
                // only update these detailed stats if we have a representative sample
                updated_chunk.distribution =
                    distribution_stats::downsample_distributions(&submissions, base);
                updated_chunk.numbers = number_stats::downsample_numbers(&submissions);
                let (niceness_mean, niceness_stdev) =
                    distribution_stats::mean_stdev_from_distribution(&updated_chunk.distribution);
                updated_chunk.niceness_mean = Some(niceness_mean);
                updated_chunk.niceness_stdev = Some(niceness_stdev);
                print!("Mean {niceness_mean:.2}, StDev {niceness_stdev:.2}, ");
            } else {
                // otherwise reset to "no data" default
                updated_chunk.distribution = Vec::new();
                updated_chunk.numbers = Vec::new();
                updated_chunk.niceness_mean = None;
                updated_chunk.niceness_stdev = None;
            }

            // save it
            if chunk == updated_chunk {
                println!("No change.");
            } else {
                db_util::update_chunk_stats(&mut conn, updated_chunk).unwrap();
                println!("Updated!");
            }
            // save submissions for the base stats
            base_submissions.append(&mut submissions);
        }

        // TODO: get remaining submissions between final chunk and end of base range

        print!("Base {base}: ",);
        print!(
            "CL{}, Checked {:.0}%, ",
            base_minimum_cl,
            base_percent_checked_detailed * 100f32
        );

        // update base record
        let mut updated_base = base_record.clone();
        updated_base.checked_niceonly = base_checked_niceonly;
        updated_base.checked_detailed = base_checked_detailed;
        updated_base.minimum_cl = base_minimum_cl;
        if base_percent_checked_detailed > DOWNSAMPLE_CUTOFF_PERCENT {
            // only update these detailed stats if we have a representative sample
            updated_base.distribution =
                distribution_stats::downsample_distributions(&base_submissions, base);
            updated_base.numbers = number_stats::downsample_numbers(&base_submissions);
            let (niceness_mean, niceness_stdev) =
                distribution_stats::mean_stdev_from_distribution(&updated_base.distribution);
            updated_base.niceness_mean = Some(niceness_mean);
            updated_base.niceness_stdev = Some(niceness_stdev);
            print!("Mean {niceness_mean:.2}, StDev {niceness_stdev:.2}, ");
        } else {
            // otherwise reset to "no data" default
            updated_base.distribution = Vec::new();
            updated_base.numbers = Vec::new();
            updated_base.niceness_mean = None;
            updated_base.niceness_stdev = None;
        }

        // save it
        if base_record == updated_base {
            println!("No change.");
        } else {
            db_util::update_base_stats(&mut conn, updated_base).unwrap();
            println!("Updated!");
        }
        println!();
    }
}
