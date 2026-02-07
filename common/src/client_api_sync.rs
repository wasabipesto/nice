//! Synchronous client-server connection utilities with proper error handling.

use crate::{CLIENT_REQUEST_TIMEOUT_SECS, DataToClient, DataToServer, SearchMode, ValidationData};
use anyhow::{Context, Result, anyhow};
use log::warn;
use reqwest::blocking::Response;
use std::{thread, time::Duration};

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
fn retry_request<F, P, T>(request_fn: F, process_response: P, max_retries: u32) -> Result<T>
where
    F: Fn() -> Result<Response, reqwest::Error>,
    P: Fn(Response) -> Result<T>,
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
                        let status = response.status();
                        let error_msg = response.text().unwrap_or_default();
                        warn!(
                            "Server error ({status} {error_msg}), retrying in {sleep_secs} seconds... (attempt {attempts}/{max_retries})"
                        );
                        thread::sleep(Duration::from_secs(sleep_secs));
                        continue;
                    }
                    let status = response.status();
                    return Err(anyhow!("Server error after {attempts} attempts: {status}"));
                }

                // Process the successful response
                return process_response(response);
            }
            Err(e) => {
                if is_retryable_error(&e) && attempts < max_retries {
                    let sleep_secs = 2_u64.pow(attempts.saturating_sub(1));
                    warn!(
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
                return Err(anyhow!(
                    "Network error ({}) after {attempts} attempts: {e}",
                    error_type_str(&e)
                ));
            }
        }
    }
}

/// Request a field from the server and returns the deserialized data.
/// Retries for 5xx errors or network timeouts.
///
/// # Errors
///
/// Returns an error if:
/// - The server returns an error after all retry attempts
/// - A network error occurs after all retry attempts
/// - The response cannot be deserialized
///
/// # Panics
///
/// Panics if the HTTP client cannot be built (should be extremely rare).
pub fn get_field_from_server(
    mode: &SearchMode,
    api_base: &str,
    max_retries: u32,
) -> Result<DataToClient> {
    let url = match mode {
        SearchMode::Detailed => format!("{api_base}/claim/detailed"),
        SearchMode::Niceonly => format!("{api_base}/claim/niceonly"),
    };

    retry_request(
        || {
            let client = reqwest::blocking::Client::builder()
                .timeout(Duration::from_secs(CLIENT_REQUEST_TIMEOUT_SECS))
                .build()
                .expect("Failed to build HTTP client");
            client.get(&url).send()
        },
        |response| {
            response
                .json::<DataToClient>()
                .context("Failed to deserialize server response")
        },
        max_retries,
    )
}

/// Submit field results to the server. Returns an error if there is a problem.
/// Retries for 5xx errors or network timeouts.
///
/// # Errors
///
/// Returns an error if:
/// - The server returns an error (4xx, 5xx) after all retry attempts
/// - A network error occurs after all retry attempts
/// - The response indicates a failure
///
/// # Panics
///
/// Panics if the HTTP client cannot be built (should be extremely rare).
pub fn submit_field_to_server(
    api_base: &str,
    submit_data: &DataToServer,
    max_retries: u32,
) -> Result<Response> {
    let url = format!("{api_base}/submit");

    retry_request(
        || {
            let client = reqwest::blocking::Client::builder()
                .timeout(Duration::from_secs(CLIENT_REQUEST_TIMEOUT_SECS))
                .build()
                .expect("Failed to build HTTP client");
            client.post(&url).json(&submit_data).send()
        },
        |response| {
            // Check for other client/server errors (4xx, etc.)
            if !response.status().is_success() {
                let status = response.status();
                let msg = response
                    .text()
                    .unwrap_or_else(|_| "Unknown error".to_string());
                return Err(anyhow!("Server returned an error ({status}): {msg}"));
            }
            Ok(response)
        },
        max_retries,
    )
}

/// Request validation data from the server for a specific claim.
/// Returns the deserialized `ValidationData` which includes the expected results.
/// Retries for 5xx errors or network timeouts.
///
/// # Errors
///
/// Returns an error if:
/// - The server returns an error after all retry attempts
/// - A network error occurs after all retry attempts
/// - The response cannot be deserialized
///
/// # Panics
///
/// Panics if the HTTP client cannot be built (should be extremely rare).
pub fn get_validation_data_from_server(api_base: &str, max_retries: u32) -> Result<ValidationData> {
    let url = format!("{api_base}/claim/validate");

    retry_request(
        || {
            let client = reqwest::blocking::Client::builder()
                .timeout(Duration::from_secs(CLIENT_REQUEST_TIMEOUT_SECS))
                .build()
                .expect("Failed to build HTTP client");
            client.get(&url).send()
        },
        |response| {
            response
                .json::<ValidationData>()
                .context("Failed to deserialize validation response")
        },
        max_retries,
    )
}
