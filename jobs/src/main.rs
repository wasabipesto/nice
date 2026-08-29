//! Scheduled jobs for the nice project.
//!
//! By default this runs *incrementally*: only fields that received a new
//! detailed submission since the last run get their consensus re-evaluated,
//! and only chunks containing fields with any new submission get their
//! statistics recomputed. The boundary is the `job_state` watermark (the
//! highest submission id already processed), advanced at the end of each
//! successful run. Consensus and the derived statistics are pure functions of
//! a field's submissions, so a field with no new submissions cannot change -
//! with one exception:
//!
//! Manual edits that create no submission, such as disqualifying one, are
//! invisible to the watermark. After making one, run the full sweep:
//!
//! ```text
//! just jobs-full        # cargo run -r -p nice_jobs -- --full
//! ```
//!
//! `--full` re-evaluates every field with detailed submissions and recomputes
//! every chunk and base, exactly like the job always did, and then advances
//! the watermark like any other successful run.

#![warn(clippy::all, clippy::pedantic)]
#![allow(clippy::too_many_lines)]

use nice_common::DOWNSAMPLE_CUTOFF_PERCENT;
use nice_common::consensus;
use nice_common::db_util;
use nice_common::distribution_stats::{self, DistributionAccumulator};
use nice_common::number_stats::{self, NumbersAccumulator};
use nice_common::{FieldRecord, SubmissionRecord};
use std::collections::{HashMap, HashSet};
use std::time::Instant;

/// How many submission ids below the snapshot the watermark is held back.
///
/// `MAX(id)` is not a transactionally safe boundary: a submission whose id was
/// allocated before the snapshot can commit after it, and a watermark set at
/// the snapshot would skip it forever. Holding the watermark this many ids
/// back means such stragglers are re-examined next run. Everything keyed on
/// the watermark is idempotent, so the only cost is re-processing a few
/// seconds' worth of submissions twice (~58/s arrive as of writing).
const WATERMARK_SAFETY_MARGIN: i64 = 10_000;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let full = match args.iter().map(String::as_str).collect::<Vec<_>>()[..] {
        [] => false,
        ["--full"] => true,
        _ => {
            eprintln!("usage: nice_jobs [--full]");
            std::process::exit(2);
        }
    };

    let started = Instant::now();

    // get db connection
    let mut conn = db_util::get_database_connection();
    println!("Database connection established. Scheduled jobs started.");

    // Establish the submission window for this run. The watermark is read even
    // in full mode so a missing migration fails here rather than at the end.
    let watermark = db_util::job_state::get_watermark(&mut conn).unwrap();
    let max_submission_id = db_util::submissions::get_max_submission_id(&mut conn).unwrap();
    let window_start = if full { 0 } else { watermark };
    println!(
        "Mode: {}. Processing submissions {} through {}.",
        if full { "FULL SWEEP" } else { "incremental" },
        window_start + 1,
        max_submission_id
    );

    // Which chunks contain fields with new submissions (either mode - both
    // bump field check levels at submit time), and which fields need their
    // consensus (re-)evaluated (new detailed submissions only). In full mode
    // every chunk is a target and the per-base consensus query is used.
    let mut dirty_chunks_by_base: HashMap<u32, HashSet<u32>> = HashMap::new();
    let mut dirty_bases: HashSet<u32> = HashSet::new();
    let mut fields_by_base: HashMap<u32, Vec<FieldRecord>> = HashMap::new();
    if !full {
        for (base, chunk_id) in db_util::submissions::get_chunks_with_new_submissions(
            &mut conn,
            watermark,
            max_submission_id,
        )
        .unwrap()
        {
            dirty_bases.insert(base);
            if let Some(chunk_id) = chunk_id {
                dirty_chunks_by_base
                    .entry(base)
                    .or_default()
                    .insert(chunk_id);
            }
        }
        for field in db_util::fields::get_fields_with_new_detailed_submissions(
            &mut conn,
            watermark,
            max_submission_id,
        )
        .unwrap()
        {
            fields_by_base.entry(field.base).or_default().push(field);
        }
    }

    let mut skipped_bases: Vec<u32> = Vec::new();
    let mut total_fields_evaluated: usize = 0;
    let mut total_fields_updated: usize = 0;
    let mut total_chunks_updated: usize = 0;

    // get all bases
    let bases = db_util::bases::get_all_bases(&mut conn).unwrap();
    for base_record in bases {
        let base = base_record.base;

        if !full && !dirty_bases.contains(&base) {
            skipped_bases.push(base);
            continue;
        }

        println!("=== BASE {base} CONSENSUS ===");

        let fields_to_check: Vec<FieldRecord> = if full {
            db_util::fields::get_fields_in_base_with_detailed_subs(&mut conn, base).unwrap()
        } else {
            fields_by_base.remove(&base).unwrap_or_default()
        };

        let mut fields_evaluated: usize = 0;
        let mut fields_updated: usize = 0;
        for field in fields_to_check {
            fields_evaluated += 1;

            // Get all qualified and detailed submissions for the field
            let submissions = db_util::submissions::get_submissions_qualified_detailed_for_field(
                &mut conn,
                field.field_id,
            )
            .unwrap();

            // Establish the consensus
            let (canon_submission, check_level) =
                consensus::evaluate_consensus(&field, &submissions).unwrap();

            match &canon_submission {
                None => {
                    if field.canon_submission_id.is_some() || field.check_level > 1 {
                        println!(
                            "WARNING: Field #{} claimed to be checked (Submission #{:?}, CL{}) but no submissions were found, so it was reset to CL{}.",
                            field.field_id,
                            field.canon_submission_id,
                            field.check_level,
                            check_level
                        );
                        db_util::fields::update_field_canon_and_cl(
                            &mut conn,
                            field.field_id,
                            None,
                            check_level,
                        )
                        .unwrap();
                        fields_updated += 1;
                    }
                }
                #[allow(clippy::cast_possible_truncation)] // TODO: fix submission_id type mismatch
                Some(sub) => {
                    // Update the field if necessary
                    if field.canon_submission_id != Some(sub.submission_id as u32)
                        || field.check_level != check_level
                    {
                        db_util::fields::update_field_canon_and_cl(
                            &mut conn,
                            field.field_id,
                            Some(sub.submission_id as u32),
                            check_level,
                        )
                        .unwrap();
                        println!(
                            "Field #{}: CL{}, Canon Submission #{}, Updated!",
                            field.field_id, check_level, sub.submission_id
                        );
                        fields_updated += 1;
                    }
                }
            }
        }
        println!("Consensus: {fields_evaluated} fields evaluated, {fields_updated} updated.");
        total_fields_evaluated += fields_evaluated;
        total_fields_updated += fields_updated;

        println!("=== BASE {base} DOWNSAMPLING ===");

        // Get all chunks for the base, and decide which need recomputing.
        let chunks = db_util::chunks::get_chunks_in_base(&mut conn, base).unwrap();
        let target_chunk_ids: HashSet<u32> = if full {
            chunks.iter().map(|c| c.chunk_id).collect()
        } else {
            dirty_chunks_by_base.remove(&base).unwrap_or_default()
        };

        // Recompute field-level statistics for the target chunks only. Chunks
        // without new submissions cannot have changed: field check levels move
        // only at submit time or in the consensus pass above, and both imply a
        // new submission in this run's window.
        let chunk_stats_batch = if full {
            db_util::fields::get_chunk_stats_batch(&mut conn, base).unwrap()
        } else {
            let ids: Vec<u32> = target_chunk_ids.iter().copied().collect();
            db_util::fields::get_chunk_stats_for_chunks(&mut conn, &ids).unwrap()
        };
        let mut chunk_stats_map: HashMap<u32, _> = HashMap::new();
        for stats in chunk_stats_batch {
            #[allow(clippy::cast_sign_loss)]
            let chunk_id = stats.chunk_id as u32;
            chunk_stats_map.insert(chunk_id, stats);
        }

        let mut chunks_updated: usize = 0;
        for chunk in &chunks {
            if !target_chunk_ids.contains(&chunk.chunk_id) {
                continue;
            }
            let chunk_size = chunk.range_size;

            // Use pre-fetched stats
            let (minimum_cl, checked_niceonly, checked_detailed) =
                if let Some(stats) = chunk_stats_map.get(&chunk.chunk_id) {
                    #[allow(clippy::cast_sign_loss, clippy::cast_possible_truncation)]
                    let minimum_cl = stats.minimum_cl as u8;

                    // Convert BigDecimal to u128
                    let checked_niceonly = stats
                        .checked_niceonly
                        .to_string()
                        .parse::<u128>()
                        .unwrap_or(0);
                    let checked_detailed = stats
                        .checked_detailed
                        .to_string()
                        .parse::<u128>()
                        .unwrap_or(0);

                    (minimum_cl, checked_niceonly, checked_detailed)
                } else {
                    // No stats found, chunk has no fields or all at CL0
                    (0, 0, 0)
                };

            #[allow(clippy::cast_precision_loss)]
            let chunk_percent_checked_detailed = checked_detailed as f32 / chunk_size as f32;

            // Update chunk record
            let mut updated_chunk = chunk.clone();
            updated_chunk.checked_niceonly = checked_niceonly;
            updated_chunk.checked_detailed = checked_detailed;
            updated_chunk.minimum_cl = minimum_cl;
            if chunk_percent_checked_detailed > DOWNSAMPLE_CUTOFF_PERCENT {
                // only update these detailed stats if we have a representative
                // sample; the chunk's canon submissions are fetched here (and
                // only here), bounding memory to one chunk at a time
                let submissions: Vec<SubmissionRecord> =
                    db_util::submissions::get_canon_submissions_for_chunk(
                        &mut conn,
                        chunk.chunk_id,
                    )
                    .unwrap();
                updated_chunk.distribution =
                    distribution_stats::downsample_distributions(&submissions, base);
                updated_chunk.numbers = number_stats::downsample_numbers(&submissions);
                let (niceness_mean, niceness_stdev) =
                    distribution_stats::mean_stdev_from_distribution(&updated_chunk.distribution);
                updated_chunk.niceness_mean = Some(niceness_mean);
                updated_chunk.niceness_stdev = Some(niceness_stdev);
            } else {
                // otherwise reset to "no data" default
                updated_chunk.distribution = Vec::new();
                updated_chunk.numbers = Vec::new();
                updated_chunk.niceness_mean = None;
                updated_chunk.niceness_stdev = None;
            }

            // save it
            if *chunk != updated_chunk {
                println!(
                    "Chunk #{}: CL{}, Checked {:.1}%, Updated!",
                    chunk.chunk_id,
                    minimum_cl,
                    chunk_percent_checked_detailed * 100f32
                );
                db_util::chunks::update_chunk(&mut conn, updated_chunk.chunk_id, updated_chunk)
                    .unwrap();
                chunks_updated += 1;
            }
        }
        println!(
            "Downsampling: {} of {} chunks recomputed, {} updated.",
            target_chunk_ids.len(),
            chunks.len(),
            chunks_updated
        );
        total_chunks_updated += chunks_updated;

        // Base-level totals come from the chunk rows just updated - reading
        // ~100 chunk rows instead of re-aggregating over every field.
        let (base_checked_niceonly, base_checked_detailed, base_minimum_cl) =
            db_util::chunks::get_base_totals_from_chunks(&mut conn, base).unwrap();

        #[allow(clippy::cast_precision_loss)]
        let base_percent_checked_detailed =
            base_checked_detailed as f32 / base_record.range_size as f32;

        print!("Base {base}: ");
        print!(
            "CL{}, Checked {:.1}%, ",
            base_minimum_cl,
            base_percent_checked_detailed * 100f32
        );

        // update base record
        let mut updated_base = base_record.clone();
        updated_base.checked_niceonly = base_checked_niceonly;
        updated_base.checked_detailed = base_checked_detailed;
        updated_base.minimum_cl = base_minimum_cl;
        if base_percent_checked_detailed > DOWNSAMPLE_CUTOFF_PERCENT {
            // only update these detailed stats if we have a representative
            // sample. The base-level aggregation streams the base's canon
            // submissions one chunk at a time through accumulators - the same
            // arithmetic as aggregating them all at once, without ever holding
            // more than one chunk's submissions in memory.
            let mut dist_acc = DistributionAccumulator::new(base);
            let mut num_acc = NumbersAccumulator::new();
            for chunk in &chunks {
                let submissions = db_util::submissions::get_canon_submissions_for_chunk(
                    &mut conn,
                    chunk.chunk_id,
                )
                .unwrap();
                dist_acc.fold(&submissions);
                num_acc.fold(&submissions);
            }
            updated_base.distribution = dist_acc.finalize();
            updated_base.numbers = num_acc.finalize();
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
            db_util::bases::update_base(&mut conn, updated_base.base, updated_base).unwrap();
            println!("Updated!");
        }
        println!();
    }

    if !skipped_bases.is_empty() {
        println!(
            "Skipped {} bases with no new submissions: {skipped_bases:?}",
            skipped_bases.len()
        );
    }

    // Advance the watermark, held back by the safety margin (see its docs).
    // This runs only after every consensus and stats update above committed,
    // so a crash anywhere earlier leaves the window to be redone next run.
    let new_watermark = (max_submission_id - WATERMARK_SAFETY_MARGIN).max(watermark);
    db_util::job_state::set_watermark(&mut conn, new_watermark).unwrap();
    println!("Watermark advanced: {watermark} -> {new_watermark}.");

    println!("=== REFRESHING SEARCH CACHES ===");
    db_util::cache::refresh_search_caches(&mut conn).unwrap();
    println!("Search caches refreshed.");

    println!(
        "Done in {:.1}s: {total_fields_evaluated} fields evaluated ({total_fields_updated} updated), {total_chunks_updated} chunks updated.",
        started.elapsed().as_secs_f32()
    );
}
