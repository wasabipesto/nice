//! An api for coordinating the search for square-cube pandigitals.

#[macro_use]
extern crate rocket;

//use nice_common::benchmark::{get_benchmark_field, BenchmarkMode};
use nice_common::db_util::{
    claim_field, get_claim_by_id, get_database_connection, get_field_by_id, insert_submission,
    log_claim,
};
use nice_common::expand_stats::{expand_distribution, expand_numbers};
use nice_common::{
    FieldClaimStrategy, FieldToClient, FieldToServer, SearchMode, DEFAULT_FIELD_SIZE,
};
use rocket::serde::json::{json, Json, Value};

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

#[post("/submit", data = "<data>")]
fn submit(data: Json<FieldToServer>) -> Result<Value, Value> {
    // get database connection
    // TODO: database connection pooling
    let mut conn = get_database_connection();

    // get submission data out of json container
    let submit_data = FieldToServer {
        claim_id: data.claim_id,
        username: data.username.clone(),
        client_version: data.client_version.clone(),
        unique_distribution: data.unique_distribution.clone(),
        nice_numbers: data.nice_numbers.clone(),
    };

    // get user IP
    // TODO: actually do this
    let user_ip = "unknown".to_string();

    // get claim record
    let claim_record = get_claim_by_id(&mut conn, submit_data.claim_id)?;

    // get field record (for base)
    let field_record = get_field_by_id(&mut conn, claim_record.field_id)?;
    let base = field_record.base;

    // expand nice numbers
    let numbers_expanded = expand_numbers(&submit_data.nice_numbers, base);

    match claim_record.search_mode {
        SearchMode::Niceonly => {
            // no checks, honor system
            insert_submission(
                &mut conn,
                claim_record,
                submit_data,
                user_ip,
                None,
                numbers_expanded,
            )?;
        }
        SearchMode::Detailed => {
            // run through some basic validity tests
            match &submit_data.unique_distribution {
                Some(distribution) => {
                    // expand distribution
                    let distribution_expanded = expand_distribution(distribution, base);

                    // TODO: Check distribution count sums to range_size
                    // TODO: Check count of nice numbers against distribution

                    // check each nice number provided
                    for _num in &numbers_expanded {
                        // TODO: check each nice number server-side
                    }

                    // save it
                    insert_submission(
                        &mut conn,
                        claim_record,
                        submit_data,
                        user_ip,
                        Some(distribution_expanded),
                        numbers_expanded,
                    )?;
                }
                None => {
                    return Err(json!(
                        "Unique distribution must be present for detailed searches."
                    ))
                }
            }
        }
    }

    // respond to user
    Ok(json!("OK"))
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
