//! An api for coordinating the search for square-cube pandigitals.

#[macro_use]
extern crate rocket;

//use nice_common::benchmark::{get_benchmark_field, BenchmarkMode};
use nice_common::db_util::{claim_field, get_database_connection, log_claim};
use nice_common::{FieldClaimStrategy, FieldToClient, SearchMode, DEFAULT_FIELD_SIZE};
use rocket::serde::json::{json, Value};

// TODO: Define error types (4xx, 5xx) and serialize them properly

#[get("/claim")]
fn claim() -> Result<Value, Value> {
    claim_detailed()
}

#[get("/claim/detailed")]
fn claim_detailed() -> Result<Value, Value> {
    // get database connection
    // TODO: database connection pooling
    let mut conn = get_database_connection();

    // set search mode based on path
    let search_mode = SearchMode::Detailed;

    // get user IP
    // TODO: actually do this
    let user_ip = "unknown".to_string();

    // get lowest valid field
    // TODO: random between next (80%) and random (20%)
    let claim_strategy = FieldClaimStrategy::Next;

    // get CL0 (unchecked) and CL1 (nice only) but not CL2 (detailed)
    // TODO: random between 1 (80%) and 2 (20%)
    let max_check_level = 1;

    // this won't affect anything since all fields will be this size or smaller
    // TODO: implement an "online benchmarking" option for e.g. gh runners that limits this
    let max_range_size = DEFAULT_FIELD_SIZE;

    // get the field to search based on claim strategy, max check level, etc
    let search_field = claim_field(&mut conn, claim_strategy, max_check_level, max_range_size)?;

    // log the claim and get the record
    let claim_record = log_claim(&mut conn, &search_field, search_mode, user_ip)?;

    // build the struct to send to the client
    let data_for_client = FieldToClient {
        claim_id: claim_record.claim_id,
        base: search_field.base,
        range_start: search_field.range_start,
        range_end: search_field.range_end,
        range_size: search_field.range_size,
    };

    // return to user
    Ok(json!(data_for_client))
}

#[get("/claim/niceonly")]
fn claim_niceonly() -> Result<Value, Value> {
    // get database connection
    // TODO: database connection pooling
    let mut conn = get_database_connection();

    // set search mode based on path
    let search_mode = SearchMode::Niceonly;

    // get user IP
    // TODO: actually do this
    let user_ip = "unknown".to_string();

    // get lowest valid field
    // TODO: random between next (80%) and random (20%)
    let claim_strategy = FieldClaimStrategy::Next;

    // get CL0 (unchecked) but not CL1 (nice only) or CL2 (detailed)
    let max_check_level = 0;

    // this won't affect anything since all fields will be this size or smaller
    // TODO: implement an "online benchmarking" option for e.g. gh runners that limits this
    let max_range_size = DEFAULT_FIELD_SIZE;

    // get the field to search based on claim strategy, max check level, etc
    let search_field = claim_field(&mut conn, claim_strategy, max_check_level, max_range_size)?;

    // log the claim and get the record
    let claim_record = log_claim(&mut conn, &search_field, search_mode, user_ip)?;

    // build the struct to send to the client
    let data_for_client = FieldToClient {
        claim_id: claim_record.claim_id,
        base: search_field.base,
        range_start: search_field.range_start,
        range_end: search_field.range_end,
        range_size: search_field.range_size,
    };

    // return to user
    Ok(json!(data_for_client))
}

#[post("/submit")]
fn submit() -> Result<(), Value> {
    Ok(())
}

#[catch(404)]
fn not_found() -> Value {
    json!("The requested resource could not be found.")
}

#[launch]
fn rocket() -> _ {
    rocket::build()
        .mount("/", routes![claim, claim_detailed, claim_niceonly, submit])
        .register("/", catchers![not_found])
}
