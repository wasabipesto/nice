//! An api for coordinating the search for square-cube pandigitals.

#[macro_use]
extern crate rocket;

use nice_common::client_process::get_num_unique_digits;
use nice_common::db_util::{
    claim_field, get_claim_by_id, get_database_connection, get_field_by_id, insert_submission,
    log_claim,
};
use nice_common::expand_stats::{expand_distribution, expand_numbers};
use nice_common::{
    FieldClaimStrategy, FieldToClient, FieldToServer, NiceNumbersExtended, SearchMode,
    DEFAULT_FIELD_SIZE, NEAR_MISS_CUTOFF_PERCENT,
};
use rand::Rng;
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

    // get rng thread
    let mut rng = rand::thread_rng();

    let claim_strategy = if rng.gen_range(0..100) < 80 {
        // 80% chance: get lowest valid field
        FieldClaimStrategy::Next
    } else {
        // 20% chance: get random valid field
        FieldClaimStrategy::Random
    };

    let max_check_level = if rng.gen_range(0..100) < 80 {
        // 80% chance: get CL0 (unchecked) or CL1 (nice only) but not CL2 (detailed) or CL3 (consensus)
        1
    } else {
        // 20% chance: get CL0 (unchecked) or CL1 (nice only) or CL2 (detailed) but not CL3 (consensus)
        2
    };

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

    // get rng thread
    let mut rng = rand::thread_rng();

    let claim_strategy = if rng.gen_range(0..100) < 80 {
        // 80% chance: get lowest valid field
        FieldClaimStrategy::Next
    } else {
        // 20% chance: get random valid field
        FieldClaimStrategy::Random
    };

    let max_check_level = if rng.gen_range(0..100) < 80 {
        // 80% chance: get CL0 (unchecked) or CL1 (nice only) but not CL2 (detailed) or CL3 (consensus)
        1
    } else {
        // 20% chance: get CL0 (unchecked) or CL1 (nice only) or CL2 (detailed) but not CL3 (consensus)
        2
    };

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

                    // check distribution count sums to range_size
                    let dist_total_count = distribution.iter().fold(0, |acc, d| acc + d.count);
                    if dist_total_count != field_record.range_size {
                        return Err(format!(
                            "Total distribution count is incorrect (submitted {}, range was {}).",
                            dist_total_count, field_record.range_size
                        )
                        .into());
                    }

                    // check count of nice numbers against distribution
                    let num_uniques_cutoff = (base as f32 * NEAR_MISS_CUTOFF_PERCENT) as u32;
                    for d in &distribution_expanded {
                        if d.num_uniques > num_uniques_cutoff {
                            let count_numbers = numbers_expanded
                                .iter()
                                .filter(|n| n.num_uniques == d.num_uniques)
                                .collect::<Vec<&NiceNumbersExtended>>()
                                .len();
                            if count_numbers as u128 != d.count {
                                return Err(format!(
                                    "Count of nice numbers with {} uniques does not match distribution (submitted {}, distribution claimed {}).",
                                    d.num_uniques, count_numbers, d.count
                                )
                                .into());
                            }
                        }
                    }

                    // check total number of nice numbers
                    let num_total_count = numbers_expanded.len();
                    let dist_total_count_above_cutoff = distribution
                        .iter()
                        .filter(|d| d.num_uniques > num_uniques_cutoff)
                        .fold(0, |acc, d| acc + d.count);
                    if num_total_count as u128 != dist_total_count_above_cutoff {
                        return Err(format!(
                            "Count of nice numbers does not match distribution (submitted {}, distribution claimed {}).",
                            num_total_count, dist_total_count_above_cutoff
                        )
                        .into());
                    }

                    // check each nice number provided
                    for n in &numbers_expanded {
                        let calculated_num_uniques = get_num_unique_digits(n.number, base);
                        if calculated_num_uniques != n.num_uniques {
                            return Err(format!(
                                "Unique count for {} is incorrect (submitted as {}, sever calculated {}).", n.number, n.num_uniques, calculated_num_uniques
                            ).into());
                        }
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

#[get("/")]
fn index() -> Value {
    not_found()
}

#[catch(404)]
fn not_found() -> Value {
    "The requested resource could not be found. 
    Available resources include /claim and /submit. 
    Visit https://nicenumbers.net for more information."
        .into()
}

#[launch]
fn rocket() -> _ {
    rocket::build()
        .mount(
            "/",
            routes![claim, claim_detailed, claim_niceonly, submit, index],
        )
        .register("/", catchers![not_found])
}
