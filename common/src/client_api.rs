//! A module with client-server connection utilities.

#![allow(clippy::missing_panics_doc)] // TODO: Replace these panics with Results

use crate::{CLIENT_REQUEST_TIMEOUT_SECS, DataToClient, DataToServer, SearchMode, ValidationData};
use reqwest::blocking::Response;
use std::{thread, time::Duration};

// Reexport tokio and reqwest for async functions
#[cfg(any(feature = "openssl-tls", feature = "rustls-tls"))]
pub use reqwest::Client;
#[cfg(any(feature = "openssl-tls", feature = "rustls-tls"))]
pub use tokio;

/// Helper function to determine if an error is retry-able
/// - `is_timeout()` catches typical network timeouts
/// - `is_connect()` catches typical connection failures
/// - `is_request()` catches DNS resolution failures and other transient request errors
fn is_retryable_error(e: &reqwest::Error) -> bool {
    e.is_timeout() || e.is_connect() || e.is_request()
}

/// Helper function to classify reqwest error types
fn error_type_str(e: &reqwest::Error) -> &'static str {
    if e.is_timeout() {
        "timeout"
    } else if e.is_connect() {
        "connection"
    } else if e.is_request() {
        "request/DNS"
    } else if e.is_body() {
        "body"
    } else if e.is_decode() {
        "decode"
    } else {
        "unknown"
    }
}

/// Generic retry logic for HTTP requests with exponential backoff.
/// Handles both network errors and 5xx server errors.
/// Takes a closure to process the successful response.
fn retry_request<F, P, T>(request_fn: F, process_response: P, max_retries: u32) -> T
where
    F: Fn() -> Result<Response, reqwest::Error>,
    P: Fn(Response) -> T,
{
    let mut attempts = 0;

    loop {
        attempts += 1;

        match request_fn() {
            Ok(response) => {
                // Check if it's a 5xx server error
                if response.status().is_server_error() {
                    if attempts < max_retries {
                        let sleep_secs = 2_u64.pow(attempts.saturating_sub(1));
                        eprintln!(
                            "Server error ({} {}), retrying in {} seconds... (attempt {}/{})",
                            response.status(),
                            response.text().unwrap_or_default(),
                            sleep_secs,
                            attempts,
                            max_retries
                        );
                        thread::sleep(Duration::from_secs(sleep_secs));
                        continue;
                    }
                    panic!(
                        "Server error after {attempts} attempts: {}",
                        response.status()
                    );
                }

                // Process the successful response
                return process_response(response);
            }
            Err(e) => {
                if is_retryable_error(&e) && attempts < max_retries {
                    let sleep_secs = 2_u64.pow(attempts.saturating_sub(1));
                    eprintln!(
                        "Network error ({}), retrying in {} seconds... (attempt {}/{}): {}",
                        error_type_str(&e),
                        sleep_secs,
                        attempts,
                        max_retries,
                        e
                    );
                    thread::sleep(Duration::from_secs(sleep_secs));
                    continue;
                }
                panic!(
                    "Network error ({}) after {attempts} attempts: {e}",
                    error_type_str(&e),
                );
            }
        }
    }
}

/// Request a field from the server and returns the deserialized data.
/// Retries for 5xx errors or network timeouts.
#[must_use]
pub fn get_field_from_server(mode: &SearchMode, api_base: &str, max_retries: u32) -> DataToClient {
    let url = match mode {
        SearchMode::Detailed => format!("{api_base}/claim/detailed"),
        SearchMode::Niceonly => format!("{api_base}/claim/niceonly"),
    };

    retry_request(
        || {
            reqwest::blocking::Client::builder()
                .timeout(Duration::from_secs(CLIENT_REQUEST_TIMEOUT_SECS))
                .build()
                .unwrap()
                .get(&url)
                .send()
        },
        |response| match response.json::<DataToClient>() {
            Ok(data) => data,
            Err(e) => panic!("Error deserializing response: {e}"),
        },
        max_retries,
    )
}

/// Submit field results to the server. Panic if there is an error.
/// Retries for 5xx errors or network timeouts.
#[must_use]
pub fn submit_field_to_server(
    api_base: &str,
    submit_data: &DataToServer,
    max_retries: u32,
) -> Response {
    let url = format!("{api_base}/submit");

    retry_request(
        || {
            reqwest::blocking::Client::builder()
                .timeout(Duration::from_secs(CLIENT_REQUEST_TIMEOUT_SECS))
                .build()
                .unwrap()
                .post(&url)
                .json(&submit_data)
                .send()
        },
        |response| {
            // Check for other client/server errors (4xx, etc.)
            if !response.status().is_success() {
                match response.text() {
                    Ok(msg) => panic!("Server returned an error: {msg}"),
                    Err(e) => panic!("Server returned an error, but another error occurred: {e}"),
                }
            }
            response
        },
        max_retries,
    )
}

/// Request validation data from the server for a specific claim.
/// Returns the deserialized `ValidationData` which includes the expected results.
/// Retries for 5xx errors or network timeouts.
#[must_use]
pub fn get_validation_data_from_server(api_base: &str, max_retries: u32) -> ValidationData {
    let url = format!("{api_base}/claim/validate");

    retry_request(
        || {
            reqwest::blocking::Client::builder()
                .timeout(Duration::from_secs(CLIENT_REQUEST_TIMEOUT_SECS))
                .build()
                .unwrap()
                .get(&url)
                .send()
        },
        |response| match response.json::<ValidationData>() {
            Ok(data) => data,
            Err(e) => panic!("Error deserializing validation response: {e}"),
        },
        max_retries,
    )
}

// ============================================================================
// ASYNC VERSIONS OF API FUNCTIONS
// ============================================================================

/// Helper function to determine if an error is retry-able (async version)
fn is_retryable_error_async(e: &reqwest::Error) -> bool {
    e.is_timeout() || e.is_connect() || e.is_request()
}

/// Helper function to classify reqwest error types (async version)
fn error_type_str_async(e: &reqwest::Error) -> &'static str {
    if e.is_timeout() {
        "timeout"
    } else if e.is_connect() {
        "connection"
    } else if e.is_request() {
        "request/DNS"
    } else if e.is_body() {
        "body"
    } else if e.is_decode() {
        "decode"
    } else {
        "unknown"
    }
}

/// Generic retry logic for async HTTP requests with exponential backoff.
/// Handles both network errors and 5xx server errors.
async fn retry_request_async<F, Fut, P, FutP, T>(
    request_fn: F,
    process_response: P,
    max_retries: u32,
) -> T
where
    F: Fn() -> Fut,
    Fut: std::future::Future<Output = Result<reqwest::Response, reqwest::Error>>,
    P: Fn(reqwest::Response) -> FutP,
    FutP: std::future::Future<Output = T>,
{
    let mut attempts = 0;

    loop {
        attempts += 1;

        match request_fn().await {
            Ok(response) => {
                // Check if it's a 5xx server error
                if response.status().is_server_error() {
                    if attempts < max_retries {
                        let sleep_secs = 2_u64.pow(attempts.saturating_sub(1));
                        eprintln!(
                            "Server error ({} {}), retrying in {} seconds... (attempt {}/{})",
                            response.status(),
                            response.text().await.unwrap_or_default(),
                            sleep_secs,
                            attempts,
                            max_retries
                        );
                        tokio::time::sleep(Duration::from_secs(sleep_secs)).await;
                        continue;
                    }
                    panic!(
                        "Server error after {attempts} attempts: {}",
                        response.status()
                    );
                }

                // Process the successful response
                return process_response(response).await;
            }
            Err(e) => {
                if is_retryable_error_async(&e) && attempts < max_retries {
                    let sleep_secs = 2_u64.pow(attempts.saturating_sub(1));
                    eprintln!(
                        "Network error ({}), retrying in {} seconds... (attempt {}/{}): {}",
                        error_type_str_async(&e),
                        sleep_secs,
                        attempts,
                        max_retries,
                        e
                    );
                    tokio::time::sleep(Duration::from_secs(sleep_secs)).await;
                    continue;
                }
                panic!(
                    "Network error ({}) after {attempts} attempts: {e}",
                    error_type_str_async(&e)
                );
            }
        }
    }
}

/// Async version: Request a field from the server and returns the deserialized data.
/// Retries for 5xx errors or network timeouts.
pub async fn get_field_from_server_async(
    client: &reqwest::Client,
    mode: &SearchMode,
    api_base: &str,
    max_retries: u32,
) -> DataToClient {
    let url = match mode {
        SearchMode::Detailed => format!("{api_base}/claim/detailed"),
        SearchMode::Niceonly => format!("{api_base}/claim/niceonly"),
    };

    retry_request_async(
        || async { client.get(&url).send().await },
        |response| async move {
            match response.json::<DataToClient>().await {
                Ok(data) => data,
                Err(e) => panic!("Error deserializing response: {e}"),
            }
        },
        max_retries,
    )
    .await
}

/// Async version: Submit field results to the server. Panic if there is an error.
/// Retries for 5xx errors or network timeouts.
pub async fn submit_field_to_server_async(
    client: &reqwest::Client,
    api_base: &str,
    submit_data: DataToServer,
    max_retries: u32,
) -> reqwest::Response {
    let url = format!("{api_base}/submit");

    retry_request_async(
        || async { client.post(&url).json(&submit_data).send().await },
        |response| async move {
            // Check for other client/server errors (4xx, etc.)
            if !response.status().is_success() {
                match response.text().await {
                    Ok(msg) => panic!("Server returned an error: {msg}"),
                    Err(e) => panic!("Server returned an error, but another error occurred: {e}"),
                }
            }
            response
        },
        max_retries,
    )
    .await
}

/// Async version: Request validation data from the server for a specific claim.
/// Returns the deserialized `ValidationData` which includes the expected results.
/// Retries for 5xx errors or network timeouts.
pub async fn get_validation_data_from_server_async(
    client: &reqwest::Client,
    api_base: &str,
    max_retries: u32,
) -> ValidationData {
    let url = format!("{api_base}/claim/validate");

    retry_request_async(
        || async { client.get(&url).send().await },
        |response| async move {
            match response.json::<ValidationData>().await {
                Ok(data) => data,
                Err(e) => panic!("Error deserializing validation response: {e}"),
            }
        },
        max_retries,
    )
    .await
}

// TODO: add tests
