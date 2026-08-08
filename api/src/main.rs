//! An API for coordinating the search for square-cube pandigitals.

#![warn(clippy::all, clippy::pedantic)]
#![allow(clippy::too_many_lines)]

#[macro_use]
extern crate rocket;

use chrono::{TimeDelta, Utc};
use nice_common::client_process::get_num_unique_digits;
use nice_common::db_util::{
    PgPool, benchmarks::get_recent_benchmarks, benchmarks::insert_benchmark,
    claims::get_claim_by_id, claims::insert_claim,
    fields::get_field_by_id, fields::get_validation_field, fields::try_claim_field,
    fields::update_field_canon_and_cl, get_database_pool, get_pooled_database_connection,
    submissions::insert_submission,
};
use nice_common::distribution_stats::expand_distribution;
use nice_common::number_stats::{expand_numbers, get_near_miss_cutoff};
use nice_common::estimator;
use nice_common::{
    BenchmarkToServer, CLAIM_DURATION_HOURS, DETAILED_SEARCH_MAX_FIELD_SIZE, DataToClient,
    DataToServer, EstimateRequest, EstimateResponse, FieldClaimStrategy, NiceNumber,
    ScenarioEstimateResponse, SearchMode, ValidationData,
};

/// Caps on stored metadata blobs. Reports and telemetry beyond these sizes
/// are junk or abuse; the limits are generous multiples of real payloads.
const BENCHMARK_MAX_BYTES: usize = 65_536;
const TELEMETRY_MAX_BYTES: usize = 16_384;

/// Benchmark report schema versions this server accepts.
const BENCHMARK_SCHEMA_VERSIONS: &[u64] = &[1];

/// How many recent benchmark reports the estimator considers. This is a
/// recency window: raising it improves rare-hardware coverage and burst-spam
/// resilience, at the price of per-request fetch+decode work (~2 KB/row —
/// revisit with an in-memory cache or SQL-side filtering if the fleet
/// controller's call rate makes it hot) and, once client versions drift,
/// more stale-version samples pooled into the quantiles.
const ESTIMATE_SAMPLE_LIMIT: i64 = 2000;
use rand::RngExt;
use rocket::State;
use rocket::http::Status;
use rocket::response::status as rocket_status;
use rocket::serde::json::{Json, Value, json};
use rocket_prometheus::PrometheusMetrics;
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

/// Static liveness endpoint for client-side latency measurement. Touches
/// nothing — the point is to measure the network and server front-end alone,
/// unlike `/status` which reads the claim queues.
#[get("/ping")]
fn ping() -> &'static str {
    "pong"
}

#[get("/status")]
fn status(queue: &State<FieldQueue>) -> Json<Value> {
    let niceonly_queue_size = queue.niceonly_queue_size();
    let detailed_thin_queue_size = queue.detailed_thin_queue_size();
    Json(json!({
        "status": "ok",
        "niceonly_queue_size": niceonly_queue_size,
        "detailed_thin_queue_size": detailed_thin_queue_size
    }))
}

/// Helper function that handles the claim logic for both detailed and niceonly modes
fn claim_helper(
    search_mode: SearchMode,
    pool: &State<PgPool>,
    queue: &State<FieldQueue>,
) -> ApiResult<DataToClient> {
    // Get database connection from the shared pool
    let mut conn = get_pooled_database_connection(pool);

    // Get the user's IP
    // TODO: Actually do this
    let user_ip = "unknown".to_string();

    // Get an RNG thread for random numbers later
    let mut rng = rand::rng();

    let (claim_strategy, max_check_level, max_range_size) = match search_mode {
        SearchMode::Niceonly => {
            // For Niceonly, only ever get the next unchecked field
            (FieldClaimStrategy::Next, 0, u128::MAX)
        }
        SearchMode::Detailed => {
            // For Detailed, only ever get the next unchecked field
            match rng.random_range(1..=100) {
                // 80% chance: get random field in the current thin chunk
                1..=80 => (FieldClaimStrategy::Thin, 1, DETAILED_SEARCH_MAX_FIELD_SIZE),
                // 15% chance: get first unchecked field in any chunk
                81..=95 => (FieldClaimStrategy::Next, 1, DETAILED_SEARCH_MAX_FIELD_SIZE),
                // 4% chance: recheck a previously checked field
                96..=99 => (FieldClaimStrategy::Next, 2, DETAILED_SEARCH_MAX_FIELD_SIZE),
                // 1% chance: get random unchecked field
                _ => (
                    FieldClaimStrategy::Random,
                    1,
                    DETAILED_SEARCH_MAX_FIELD_SIZE,
                ),
            }
        }
    };

    // Get the field to search based on claim strategy, max check level, etc.
    //
    // For niceonly mode, use the pre-claimed queue for much faster response times.
    // This reduces latency from ~90ms (database query + locking + update) to <1ms (memory access).
    // The queue automatically refills by bulk-claiming fields at once when it drops below a threshold.
    //
    // For detailed mode, the dominant (80%) `Thin` strategy also consults a pre-claimed queue;
    // the rarer `Next`/`Random`/cl=2 strategies fall back to a direct database claim because they
    // are inherently per-request stateful and not worth bulk-pre-claiming.
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
    } else if claim_strategy == FieldClaimStrategy::Thin {
        // Try the detailed-thin queue first; fall back to a direct claim if it's empty.
        if let Some(queued_field) = queue.claim_detailed_thin() {
            queued_field
        } else {
            tracing::warn!("Detailed-thin queue exhausted, falling back to direct database claim");
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
        }
    } else {
        // For the remaining detailed strategies (Next / Random / cl=2), use the original
        // database claim logic with the two-step fallback.
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
    let claim_record = insert_claim(&mut conn, search_field.field_id, search_mode, user_ip)
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

#[get("/claim/detailed")]
fn claim_detailed(pool: &State<PgPool>, queue: &State<FieldQueue>) -> ApiResult<DataToClient> {
    claim_helper(SearchMode::Detailed, pool, queue)
}

#[get("/claim/niceonly")]
fn claim_niceonly(pool: &State<PgPool>, queue: &State<FieldQueue>) -> ApiResult<DataToClient> {
    claim_helper(SearchMode::Niceonly, pool, queue)
}

#[post("/submit", data = "<data>")]
#[allow(clippy::needless_pass_by_value)]
fn submit(data: Json<DataToServer>, pool: &State<PgPool>) -> ApiResult<Value> {
    // Get database connection from the shared pool
    let mut conn = get_pooled_database_connection(pool);

    // Get submission data from JSON. Oversized telemetry is dropped rather
    // than failing the submission: the search result always outranks its
    // metadata.
    let telemetry = data.telemetry.clone().filter(|t| {
        let len = t.to_string().len();
        let ok = len <= TELEMETRY_MAX_BYTES;
        if !ok {
            tracing::warn!(len, "dropping oversized telemetry payload");
        }
        ok
    });
    let submit_data = DataToServer {
        claim_id: data.claim_id,
        username: data.username.clone(),
        client_version: data.client_version.clone(),
        unique_distribution: data.unique_distribution.clone(),
        nice_numbers: data.nice_numbers.clone(),
        telemetry,
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

/// Accept an uploaded benchmark sweep report and store it for performance
/// analysis. The report is stored as-is in a jsonb column; only the schema
/// version and size are validated here, with quality filtering left to
/// analysis time (matching the trust posture of /submit).
#[post("/benchmark", data = "<data>")]
#[allow(clippy::needless_pass_by_value)]
fn post_benchmark(data: Json<BenchmarkToServer>, pool: &State<PgPool>) -> ApiResult<Value> {
    let schema_version = data.data.get("schema_version").and_then(Value::as_u64);
    if !schema_version.is_some_and(|v| BENCHMARK_SCHEMA_VERSIONS.contains(&v)) {
        return Err(unprocessable_entity_error(format!(
            "Unsupported benchmark schema_version {schema_version:?} (accepted: {BENCHMARK_SCHEMA_VERSIONS:?})."
        )));
    }
    let size = data.data.to_string().len();
    if size > BENCHMARK_MAX_BYTES {
        return Err(unprocessable_entity_error(format!(
            "Benchmark report too large ({size} bytes, cap {BENCHMARK_MAX_BYTES})."
        )));
    }
    let client_version = data
        .data
        .get("client_version")
        .and_then(Value::as_str)
        .unwrap_or("unknown")
        .to_string();

    let mut conn = get_pooled_database_connection(pool);
    let user_ip = "unknown".to_string();
    let benchmark_id = insert_benchmark(
        &mut conn,
        data.username.clone(),
        user_ip,
        client_version,
        data.data.clone(),
    )
    .map_err(|e| internal_error(format!("Database error while inserting benchmark: {e}")))?;

    tracing::info!(benchmark_id, username = data.username, "benchmark stored");
    Ok(Json(json!({
        "message": "Benchmark stored, thanks!",
        "benchmark_id": benchmark_id,
    })))
}

/// Estimate what a hardware configuration will achieve, from recent
/// benchmark uploads. Hierarchical matching; the response names the
/// fallback stage that produced the numbers alongside a confidence percent.
#[post("/estimate", data = "<data>")]
#[allow(clippy::needless_pass_by_value)]
fn post_estimate(data: Json<EstimateRequest>, pool: &State<PgPool>) -> ApiResult<EstimateResponse> {
    let Some(mode) = estimator::mode_string(&data.mode) else {
        return Err(unprocessable_entity_error(format!(
            "Unknown mode {:?} (expected \"niceonly\" or \"detailed\").",
            data.mode
        )));
    };

    let mut conn = get_pooled_database_connection(pool);
    let rows = get_recent_benchmarks(&mut conn, ESTIMATE_SAMPLE_LIMIT)
        .map_err(|e| internal_error(format!("Database error while loading benchmarks: {e}")))?;
    let samples: Vec<estimator::BenchmarkSample> = rows
        .iter()
        .filter_map(|(version, report)| estimator::decode_sample(version, report))
        .collect();

    let input = estimator::EstimateInput {
        gpu: data.gpu,
        mode: mode.to_string(),
        threads: data.threads,
        cpu_model: data.cpu_model.clone(),
        gpu_model: data.gpu_model.clone(),
        base: data.base,
        client_version: data.client_version.clone(),
    };
    let outcome = estimator::estimate(&samples, &input);

    Ok(Json(EstimateResponse {
        prediction_stage: outcome.prediction_stage.to_string(),
        confidence: outcome.confidence,
        samples_used: outcome.samples_used,
        versions_used: outcome.versions_used,
        scenarios: outcome
            .scenarios
            .into_iter()
            .map(|s| ScenarioEstimateResponse {
                key: s.key,
                base: s.base,
                samples: s.samples,
                rate_p25: s.rate_p25,
                rate_p50: s.rate_p50,
                rate_p75: s.rate_p75,
            })
            .collect(),
        blended_rate_p25: outcome.blended_rate_p25,
        blended_rate_p50: outcome.blended_rate_p50,
        blended_rate_p75: outcome.blended_rate_p75,
        notes: outcome.notes,
    }))
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
    queue.prefill_detailed_thin();

    // Initialize Prometheus metrics
    let prometheus = PrometheusMetrics::new();

    rocket::build()
        .attach(prometheus.clone())
        .attach(CorsFairing)
        .attach(RequestTimingFairing)
        .manage(pool)
        .manage(queue)
        .mount(
            "/",
            routes![
                claim_detailed,
                claim_niceonly,
                validate,
                submit,
                status,
                ping,
                post_benchmark,
                post_estimate,
                index,
                options_handler
            ],
        )
        .mount("/metrics", prometheus)
        .register("/", catchers![not_found])
}
