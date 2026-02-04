//! An API for coordinating the search for square-cube pandigitals.

#![warn(clippy::all, clippy::pedantic)]
#![allow(clippy::too_many_lines)]

#[macro_use]
extern crate rocket;

use chrono::{TimeDelta, Utc};
use nice_common::client_process::get_num_unique_digits;
use nice_common::db_util::{
    PgPool, get_claim_by_id, get_database_pool, get_field_by_id, get_pooled_database_connection,
    get_validation_field, insert_claim, insert_submission, try_claim_field,
    update_field_canon_and_cl,
};
use nice_common::distribution_stats::expand_distribution;
use nice_common::number_stats::{expand_numbers, get_near_miss_cutoff};
use nice_common::{
    CLAIM_DURATION_HOURS, DEFAULT_FIELD_SIZE, DataToClient, DataToServer, FieldClaimStrategy,
    NiceNumber, SearchMode, ValidationData,
};
use rand::Rng;
use rocket::State;
use rocket::http::Status;
use rocket::response::status as rocket_status;
use rocket::serde::json::{Json, Value, json};
use tracing_subscriber::EnvFilter;

mod field_queue;
mod helpers;

use field_queue::FieldQueue;
use helpers::{
    ApiErrorBody, ApiResult, CorsFairing, RequestTimingFairing, bad_request_error, internal_error,
    not_found_error, unprocessable_entity_error,
};

#[get("/claim/validate")]
fn validate(pool: &State<PgPool>) -> ApiResult<ValidationData> {
    // Get database connection from the shared pool
    let mut conn = get_pooled_database_connection(pool);

    // For validation requests, we return a random completed field and the
    // results so the client can perform a self-check.
    let data = get_validation_field(&mut conn).map_err(|e| {
        internal_error(format!(
            "Database error while finding validation field: {e}"
        ))
    })?;
    Ok(Json(data))
}

#[get("/status")]
fn status(queue: &State<FieldQueue>) -> Json<Value> {
    let niceonly_queue_size = queue.niceonly_queue_size();
    Json(json!({
        "status": "ok",
        "niceonly_queue_size": niceonly_queue_size
    }))
}

#[get("/claim/<mode>")]
fn claim(mode: &str, pool: &State<PgPool>, queue: &State<FieldQueue>) -> ApiResult<DataToClient> {
    // Get database connection from the shared pool
    let mut conn = get_pooled_database_connection(pool);

    // Set search mode based on path
    let search_mode = match mode {
        "detailed" => SearchMode::Detailed,
        "niceonly" => SearchMode::Niceonly,
        _ => {
            return Err(not_found_error(
                "The requested resource could not be found. Available resources include /claim/detailed, /claim/niceonly, /claim/validate, and /submit. Visit https://nicenumbers.net for more information.",
            ));
        }
    };

    // Get the user's IP
    // TODO: Actually do this
    let user_ip = "unknown".to_string();

    // Get an RNG thread for random numbers later
    let mut rng = rand::rng();

    let (claim_strategy, max_check_level) = match search_mode {
        SearchMode::Niceonly => {
            // For Niceonly, only ever get the next unchecked field
            (FieldClaimStrategy::Next, 0)
        }
        SearchMode::Detailed => {
            // For Detailed, only ever get the next unchecked field
            match rng.random_range(1..=100) {
                // 60% chance: get random field in a thin chunk
                1..=60 => (FieldClaimStrategy::Thin, 1),
                // 30% chance: get next field in any chunk
                61..=90 => (FieldClaimStrategy::Next, 1),
                // 5% chance: recheck a previously checked field
                91..=96 => (FieldClaimStrategy::Next, 2),
                // 5% chance: get random unchecked field
                _ => (FieldClaimStrategy::Random, 1),
            }
        }
    };

    // This won't affect anything since all fields will be this size or smaller
    // TODO: Implement an "online benchmarking" option for e.g. GH runners that limits this
    let max_range_size = DEFAULT_FIELD_SIZE;

    // Get the field to search based on claim strategy, max check level, etc.
    //
    // For niceonly mode, use the pre-claimed queue for much faster response times.
    // This reduces latency from ~90ms (database query + locking + update) to <1ms (memory access).
    // The queue automatically refills by bulk-claiming fields at once when it drops below a threshold.
    let search_field = if search_mode == SearchMode::Niceonly {
        // Try to get from queue first
        if let Some(queued_field) = queue.claim_niceonly() {
            queued_field
        } else {
            // Queue is empty, fall back to direct database claim
            tracing::warn!("Niceonly queue exhausted, falling back to direct database claim");
            let maximum_timestamp = Utc::now() - TimeDelta::hours(CLAIM_DURATION_HOURS);
            try_claim_field(
                &mut conn,
                FieldClaimStrategy::Next,
                maximum_timestamp,
                0,
                max_range_size,
            )
            .map_err(|e| internal_error(format!("Database error while claiming a field: {e}")))?
            .ok_or_else(|| internal_error("Could not find any niceonly field!".to_string()))?
        }
    } else {
        // For detailed mode, use the original database claim logic
        let maximum_timestamp = Utc::now() - TimeDelta::hours(CLAIM_DURATION_HOURS);
        if let Some(claimed_field) = try_claim_field(
            &mut conn,
            claim_strategy,
            maximum_timestamp,
            max_check_level,
            max_range_size,
        )
        .map_err(|e| internal_error(format!("Database error while claiming a field: {e}")))?
        {
            claimed_field
        } else {
            tracing::info!(
                "Unable to find an unclaimed or expired field, falling back to one that may have been claimed recently."
            );
            let maximum_timestamp = Utc::now();
            let claim_strategy = FieldClaimStrategy::Next;
            try_claim_field(
                &mut conn,
                claim_strategy,
                maximum_timestamp,
                max_check_level,
                max_range_size,
            )
            .map_err(|e| internal_error(format!("Database error while claiming a field: {e}")))?
            .ok_or_else(|| {
                internal_error(format!(
                    "Could not find any field with maximum check level {max_check_level} and maximum size {max_range_size}!"
                ))
            })?
        }
    };

    // Save the claim and get the record
    let claim_record = insert_claim(&mut conn, &search_field, search_mode, user_ip)
        .map_err(|e| internal_error(format!("Database error while inserting claim: {e}")))?;

    // Build the struct to send to the client
    let data_for_client = DataToClient {
        claim_id: claim_record.claim_id,
        base: search_field.base,
        range_start: search_field.range_start,
        range_end: search_field.range_end,
        range_size: search_field.range_size,
    };

    // Log + return to user
    tracing::info!(
        search_mode = ?claim_record.search_mode,
        claim_strategy = ?claim_strategy,
        max_check_level = max_check_level,
        field_id = claim_record.field_id,
        claim_id = claim_record.claim_id,
        "New Claim"
    );
    Ok(Json(data_for_client))
}

#[post("/submit", data = "<data>")]
#[allow(clippy::needless_pass_by_value)]
fn submit(data: Json<DataToServer>, pool: &State<PgPool>) -> ApiResult<Value> {
    // Get database connection from the shared pool
    let mut conn = get_pooled_database_connection(pool);

    // Get submission data from JSON
    let submit_data = DataToServer {
        claim_id: data.claim_id,
        username: data.username.clone(),
        client_version: data.client_version.clone(),
        unique_distribution: data.unique_distribution.clone(),
        nice_numbers: data.nice_numbers.clone(),
    };

    // Get user IP
    // TODO: Actually do this
    let user_ip = "unknown".to_string();

    // Get the associated claim record
    let claim_record = get_claim_by_id(&mut conn, submit_data.claim_id).map_err(|e| {
        bad_request_error(format!("Invalid claim_id {}: {e}", submit_data.claim_id))
    })?;

    // Get the associated field record (to determine the base)
    let field_record = get_field_by_id(&mut conn, claim_record.field_id).map_err(|e| {
        internal_error(format!(
            "Database error while loading field {}: {e}",
            claim_record.field_id
        ))
    })?;
    let base = field_record.base;

    // Expand the nice numbers with some detailed info
    let numbers_expanded = expand_numbers(&submit_data.nice_numbers, base);

    match claim_record.search_mode {
        SearchMode::Niceonly => {
            // No checks for nice-only, honor system
            insert_submission(
                &mut conn,
                &claim_record,
                &submit_data,
                user_ip,
                None,
                numbers_expanded,
            )
            .map_err(|e| {
                internal_error(format!("Database error while inserting submission: {e}"))
            })?;
            // Set CL to 1 if it's 0
            if field_record.check_level == 0 {
                update_field_canon_and_cl(
                    &mut conn,
                    field_record.field_id,
                    field_record.canon_submission_id,
                    1,
                )
                .map_err(|e| internal_error(format!("Database error while updating field: {e}")))?;
            }
        }
        SearchMode::Detailed => {
            // Run through some basic validity tests
            match &submit_data.unique_distribution {
                Some(distribution) => {
                    // Expand the distribution stats
                    let distribution_expanded = expand_distribution(distribution, base);

                    // Check distribution count sums to range_size
                    let dist_total_count = distribution.iter().fold(0, |acc, d| acc + d.count);
                    if dist_total_count != field_record.range_size {
                        return Err(unprocessable_entity_error(format!(
                            "Total distribution count is incorrect (submitted {}, range was {}).",
                            dist_total_count, field_record.range_size
                        )));
                    }

                    // Get the near-miss cutoff
                    let num_uniques_cutoff = get_near_miss_cutoff(base);

                    // Check the count of nice numbers against distribution
                    for d in &distribution_expanded {
                        if d.num_uniques > num_uniques_cutoff {
                            let count_numbers = numbers_expanded
                                .iter()
                                .filter(|n| n.num_uniques == d.num_uniques)
                                .collect::<Vec<&NiceNumber>>()
                                .len();
                            if count_numbers as u128 != d.count {
                                return Err(unprocessable_entity_error(format!(
                                    "Count of nice numbers with {} uniques does not match distribution (submitted {}, distribution claimed {}).",
                                    d.num_uniques, count_numbers, d.count
                                )));
                            }
                        }
                    }

                    // Check the total number of nice numbers
                    let num_total_count = numbers_expanded.len();
                    let dist_total_count_above_cutoff = distribution
                        .iter()
                        .filter(|d| d.num_uniques > num_uniques_cutoff)
                        .fold(0, |acc, d| acc + d.count);
                    if num_total_count as u128 != dist_total_count_above_cutoff {
                        return Err(unprocessable_entity_error(format!(
                            "Count of nice numbers does not match distribution (submitted {num_total_count}, distribution claimed {dist_total_count_above_cutoff})."
                        )));
                    }

                    // Check each nice number provided
                    for n in &numbers_expanded {
                        let calculated_num_uniques = get_num_unique_digits(n.number, base);
                        if calculated_num_uniques != n.num_uniques {
                            return Err(unprocessable_entity_error(format!(
                                "Unique count for {} is incorrect (submitted as {}, server calculated {}).",
                                n.number, n.num_uniques, calculated_num_uniques
                            )));
                        }
                    }

                    // All looks good, save it!
                    insert_submission(
                        &mut conn,
                        &claim_record,
                        &submit_data,
                        user_ip,
                        Some(distribution_expanded),
                        numbers_expanded,
                    )
                    .map_err(|e| {
                        internal_error(format!("Database error while inserting submission: {e}"))
                    })?;
                    // Bump the check level to 2
                    if field_record.check_level < 2 {
                        update_field_canon_and_cl(
                            &mut conn,
                            field_record.field_id,
                            field_record.canon_submission_id,
                            2,
                        )
                        .map_err(|e| {
                            internal_error(format!("Database error while updating field: {e}"))
                        })?;
                    }
                }
                None => {
                    return Err(unprocessable_entity_error(
                        "Unique distribution must be present for detailed searches.",
                    ));
                }
            }
        }
    }

    // Log + respond to user
    tracing::info!(
        search_mode = ?claim_record.search_mode,
        field_id = claim_record.field_id,
        claim_id = claim_record.claim_id,
        username = submit_data.username,
        "New Submission"
    );
    Ok(Json(json!("OK")))
}

#[get("/")]
fn index() -> rocket_status::Custom<Json<ApiErrorBody>> {
    not_found_error(
        "The requested resource could not be found. Available resources include /claim/detailed, /claim/niceonly, /claim/validate, and /submit. Visit https://nicenumbers.net for more information.",
    )
}

#[catch(404)]
fn not_found() -> rocket_status::Custom<Json<ApiErrorBody>> {
    not_found_error(
        "The requested resource could not be found. Available resources include /claim/detailed, /claim/niceonly, /claim/validate, and /submit. Visit https://nicenumbers.net for more information.",
    )
}

#[options("/<_..>")]
fn options_handler() -> Status {
    Status::NoContent
}

#[launch]
fn rocket() -> _ {
    // Initialize structured logging (respects RUST_LOG, defaults to "info")
    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));
    tracing_subscriber::fmt().with_env_filter(filter).init();

    let pool = get_database_pool();

    // Initialize field queue and pre-fill it
    let queue = FieldQueue::new(pool.clone());
    queue.prefill_niceonly();

    rocket::build()
        .attach(CorsFairing)
        .attach(RequestTimingFairing)
        .manage(pool)
        .manage(queue)
        .mount(
            "/",
            routes![claim, validate, submit, status, index, options_handler],
        )
        .register("/", catchers![not_found])
}
