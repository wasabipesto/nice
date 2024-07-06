//! An api for coordinating the search for square-cube pandigitals.

#[macro_use]
extern crate rocket;

use nice_common::benchmark::{get_benchmark_field, BenchmarkMode};
use rocket::serde::json::{json, Value};

// TODO: Define error types (4xx, 5xx) and serialize them properly

#[get("/claim")]
fn claim() -> Result<Value, Value> {
    claim_detailed()
}

#[get("/claim/detailed")]
fn claim_detailed() -> Result<Value, Value> {
    let claim = get_benchmark_field(BenchmarkMode::Default);
    Ok(json!(claim))
}

#[get("/claim/niceonly")]
fn claim_niceonly() -> Result<Value, Value> {
    let claim = get_benchmark_field(BenchmarkMode::Default);
    Ok(json!(claim))
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
