//! A module with client-server connection utilities.

use super::*;
use reqwest::blocking::Response;

/// Request a field from the server. Supplies CLI options as query strings.
pub fn get_field_from_server(mode: &SearchMode, api_base: &str) -> DataToClient {
    // build the url
    let url = match mode {
        SearchMode::Detailed => format!("{api_base}/claim/detailed"),
        SearchMode::Niceonly => format!("{api_base}/claim/niceonly"),
    };

    // send it
    let response = reqwest::blocking::get(url).unwrap_or_else(|e| panic!("Error: {}", e));

    // deserialize and unwrap or panic
    match response.json::<DataToClient>() {
        Ok(claim_data) => claim_data,
        Err(e) => panic!("Error: {}", e),
    }
}

/// Submit field results to the server. Panic if there is an error.
pub fn submit_field_to_server(api_base: &str, submit_data: DataToServer) -> Response {
    // build the url
    let url = format!("{api_base}/submit");

    // send it
    let response = reqwest::blocking::Client::new()
        .post(url)
        .json(&submit_data)
        .send();

    // check for network errors
    match response {
        Ok(response) => {
            // check for server errors
            if response.status().is_success() {
                response
            } else {
                match response.text() {
                    // we probably did something wrong, print anything we got from the server
                    Ok(msg) => panic!("Server returned an error: {}", msg),
                    Err(e) => panic!("Server returned an error, but another error occured: {}", e),
                }
            }
        }
        Err(e) => panic!("Network error: {}", e),
    }
}

// TODO: add tests
