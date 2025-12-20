//! A module with client-server connection utlities.

use super::*;
use reqwest::blocking::Response;
use std::{thread, time::Duration};

/// Request a field from the server and returns the deserialized data.
/// Retries for 5xx errors or network timeouts.
pub fn get_field_from_server(mode: &SearchMode, api_base: &str) -> DataToClient {
    // Build the url
    let url = match mode {
        SearchMode::Detailed => format!("{api_base}/claim/detailed"),
        SearchMode::Niceonly => format!("{api_base}/claim/niceonly"),
    };

    let mut attempts = 0;
    const MAX_ATTEMPTS: u32 = 6;

    loop {
        attempts += 1;

        // Send the request
        let response_result = reqwest::blocking::get(&url);

        match response_result {
            Ok(response) => {
                // Check if it's a 5xx server error
                if response.status().is_server_error() {
                    if attempts < MAX_ATTEMPTS {
                        let sleep_secs = 2_u64.pow(attempts.saturating_sub(1));
                        eprintln!(
                            "Server error ({}), retrying in {} seconds... (attempt {}/{})",
                            response.status(),
                            sleep_secs,
                            attempts,
                            MAX_ATTEMPTS
                        );
                        thread::sleep(Duration::from_secs(sleep_secs));
                        continue;
                    } else {
                        panic!(
                            "Server error after {} attempts: {}",
                            attempts,
                            response.status()
                        );
                    }
                }

                // Try to deserialize the response
                match response.json::<DataToClient>() {
                    Ok(claim_data) => return claim_data,
                    Err(e) => panic!("Error deserializing response: {}", e),
                }
            }
            Err(e) => {
                // Check if it's a timeout or connection error that we should retry
                let should_retry = e.is_timeout() || e.is_connect();

                if should_retry && attempts < MAX_ATTEMPTS {
                    let sleep_secs = 2_u64.pow(attempts.saturating_sub(1));
                    eprintln!(
                        "Network error, retrying in {} seconds... (attempt {}/{}): {}",
                        sleep_secs, attempts, MAX_ATTEMPTS, e
                    );
                    thread::sleep(Duration::from_secs(sleep_secs));
                    continue;
                } else {
                    panic!("Network error after {} attempts: {}", attempts, e);
                }
            }
        }
    }
}

/// Submit field results to the server. Panic if there is an error.
/// Retries for 5xx errors or network timeouts.
pub fn submit_field_to_server(api_base: &str, submit_data: DataToServer) -> Response {
    // Build the url
    let url = format!("{api_base}/submit");

    let mut attempts = 0;
    const MAX_ATTEMPTS: u32 = 6;

    loop {
        attempts += 1;

        // Send the request
        let response_result = reqwest::blocking::Client::new()
            .post(&url)
            .json(&submit_data)
            .send();

        match response_result {
            Ok(response) => {
                // Check if it's a 5xx server error
                if response.status().is_server_error() {
                    if attempts < MAX_ATTEMPTS {
                        let sleep_secs = 2_u64.pow(attempts.saturating_sub(1));
                        eprintln!(
                            "Server error ({}), retrying in {} seconds... (attempt {}/{})",
                            response.status(),
                            sleep_secs,
                            attempts,
                            MAX_ATTEMPTS
                        );
                        thread::sleep(Duration::from_secs(1));
                        continue;
                    } else {
                        // Get error message from server if possible
                        match response.text() {
                            Ok(msg) => {
                                panic!("Server error after {} attempts: {}", MAX_ATTEMPTS, msg)
                            }
                            Err(e) => panic!(
                                "Server error after {} attempts, and error reading response: {}",
                                MAX_ATTEMPTS, e
                            ),
                        }
                    }
                }

                // Check for other client/server errors (4xx, etc.)
                if !response.status().is_success() {
                    match response.text() {
                        Ok(msg) => panic!("Server returned an error: {}", msg),
                        Err(e) => panic!(
                            "Server returned an error, but another error occurred: {}",
                            e
                        ),
                    }
                }

                // Success case
                return response;
            }
            Err(e) => {
                // Check if it's a timeout or connection error that we should retry
                let should_retry = e.is_timeout() || e.is_connect();

                if should_retry && attempts < MAX_ATTEMPTS {
                    let sleep_secs = 2_u64.pow(attempts.saturating_sub(1));
                    eprintln!(
                        "Network error, retrying in {} seconds... (attempt {}/{}): {}",
                        sleep_secs, attempts, MAX_ATTEMPTS, e
                    );
                    thread::sleep(Duration::from_secs(sleep_secs));
                    continue;
                } else {
                    panic!("Network error after {} attempts: {}", attempts, e);
                }
            }
        }
    }
}

// TODO: add tests
