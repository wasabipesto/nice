//! Some helper functions for the API.

use rocket::fairing::{Fairing, Info, Kind};
use rocket::http::Header;
use rocket::http::Status;
use rocket::request::Request;
use rocket::response::Response;
use rocket::response::status as rocket_status;
use rocket::serde::json::Json;
use rocket::serde::{Deserialize, Serialize};
use std::time::Instant;

#[derive(Clone, Copy)]
pub struct RequestTimingFairing;

#[rocket::async_trait]
impl Fairing for RequestTimingFairing {
    fn info(&self) -> Info {
        Info {
            name: "Request timing",
            kind: Kind::Request | Kind::Response,
        }
    }

    async fn on_request(&self, request: &mut Request<'_>, _data: &mut rocket::Data<'_>) {
        request.local_cache(Instant::now);
    }

    async fn on_response<'r>(&self, request: &'r Request<'_>, response: &mut Response<'r>) {
        let started_at = request.local_cache(Instant::now);
        let elapsed = started_at.elapsed();
        let status = response.status().code;

        tracing::info!(
            method = %request.method(),
            path = %request.uri(),
            status = status,
            elapsed_ms = elapsed.as_millis(),
            "Request Completed"
        );
    }
}

#[derive(Clone, Copy)]
pub struct CorsFairing;

#[rocket::async_trait]
impl Fairing for CorsFairing {
    fn info(&self) -> Info {
        Info {
            name: "CORS",
            kind: Kind::Response,
        }
    }

    async fn on_response<'r>(&self, _request: &'r Request<'_>, response: &mut Response<'r>) {
        response.set_header(Header::new("Access-Control-Allow-Origin", "*"));
        response.set_header(Header::new(
            "Access-Control-Allow-Methods",
            "GET, POST, PUT, DELETE, OPTIONS, PATCH, HEAD",
        ));
        response.set_header(Header::new("Access-Control-Allow-Headers", "*"));
        response.set_header(Header::new("Access-Control-Allow-Credentials", "true"));
        response.set_header(Header::new("Access-Control-Max-Age", "86400"));
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(crate = "rocket::serde")]
#[serde(rename_all = "snake_case")]
pub enum ApiErrorKind {
    NotFound,
    BadRequest,
    Conflict,
    UnprocessableEntity,
    Internal,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(crate = "rocket::serde")]
pub struct ApiErrorBody {
    error: ApiErrorKind,
    message: String,
}

impl ApiErrorBody {
    fn new(error: ApiErrorKind, message: impl Into<String>) -> Self {
        Self {
            error,
            message: message.into(),
        }
    }
}

pub type ApiResult<T> = Result<Json<T>, rocket_status::Custom<Json<ApiErrorBody>>>;

fn api_error(
    status: Status,
    kind: ApiErrorKind,
    message: impl Into<String>,
) -> rocket_status::Custom<Json<ApiErrorBody>> {
    rocket_status::Custom(status, Json(ApiErrorBody::new(kind, message)))
}

pub fn not_found_error(message: impl Into<String>) -> rocket_status::Custom<Json<ApiErrorBody>> {
    api_error(Status::NotFound, ApiErrorKind::NotFound, message)
}

pub fn bad_request_error(message: impl Into<String>) -> rocket_status::Custom<Json<ApiErrorBody>> {
    api_error(Status::BadRequest, ApiErrorKind::BadRequest, message)
}

pub fn unprocessable_entity_error(
    message: impl Into<String>,
) -> rocket_status::Custom<Json<ApiErrorBody>> {
    api_error(
        Status::UnprocessableEntity,
        ApiErrorKind::UnprocessableEntity,
        message,
    )
}

pub fn internal_error(message: impl Into<String>) -> rocket_status::Custom<Json<ApiErrorBody>> {
    api_error(Status::InternalServerError, ApiErrorKind::Internal, message)
}
