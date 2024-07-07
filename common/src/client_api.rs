//! A module with client-server connection utilities.

use super::*;

/// Request a field from the server. Supplies CLI options as query strings.
pub fn get_field_from_server(mode: &SearchMode, api_base: &str, username: &str) -> FieldClaim {
    // build the url
    // TODO: use an actual url building lib?
    let mut query_url = api_base.to_owned();
    query_url += match mode {
        SearchMode::Detailed => "/claim/detailed",
        SearchMode::Niceonly => "/claim/niceonly",
    };
    query_url += &("?username=".to_owned() + username);

    // send it
    let response = reqwest::blocking::get(&query_url).unwrap_or_else(|e| panic!("Error: {}", e));

    // deserialize and unwrap or panic
    match response.json::<FieldClaim>() {
        Ok(claim_data) => claim_data,
        Err(e) => panic!("Error: {}", e),
    }
}

/// Submit field results to the server. Panic if there is an error.
pub fn submit_field_to_server(mode: &SearchMode, api_base: &str, submit_data: FieldSubmit) {
    // TODO: same route in v6
    let url = match mode {
        SearchMode::Detailed => format!("{}/submit", api_base),
        SearchMode::Niceonly => format!("{}/submit", api_base),
    };

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
                return; // 👍
            }
            match response.text() {
                // we probably did something wrong, print anything we got from the server
                Ok(msg) => panic!("Server returned an error: {}", msg),
                Err(_) => panic!("Server returned an error."),
            }
        }
        Err(e) => panic!("Network error: {}", e),
    }
}
