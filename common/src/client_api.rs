//! A module with client-server connection utilities.

use super::*;

/// Deserialize BigInts from the server that are wrapped in quotes.
pub fn deserialize_string_to_natural<'de, D>(deserializer: D) -> Result<Natural, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let s: String = Deserialize::deserialize(deserializer)?;
    s.trim_matches('"')
        .parse()
        .map_err(|_| serde::de::Error::custom(format!("invalid number: {}", s)))
}

/// Generate a field offline for benchmark testing.
/// TODO: Make small/medium/large benchmarks, move to own utility
pub fn get_field_benchmark() -> FieldClaim {
    let base = BENCHMARK_DEFAULT_BASE;
    let (search_start, range_end) = get_base_range(base);
    let range = Natural::from(BUCNHMARK_DEFAULT_RANGE);
    let search_end = range_end.min(&search_start + &range);
    let search_range = &search_end - &search_start;
    return FieldClaim {
        id: 0,
        username: "benchmark".to_owned(),
        base,
        search_start,
        search_end,
        search_range,
    };
}

/// Build a field request url.
fn get_claim_url(mode: &SearchMode, api_base: &str, username: &str) -> String {
    let mut query_url = api_base.to_owned();
    query_url += &match mode {
        SearchMode::Detailed => "/claim/detailed",
        SearchMode::Niceonly => "/claim/niceonly",
    };
    query_url += &("?username=".to_owned() + &username.to_string());
    query_url
}

/// Request a field from the server. Supplies CLI options as query strings.
pub fn get_field_from_server(mode: &SearchMode, api_base: &str, username: &str) -> FieldClaim {
    let response = reqwest::blocking::get(&get_claim_url(mode, api_base, username))
        .unwrap_or_else(|e| panic!("Error: {}", e));
    match response.json::<FieldClaim>() {
        Ok(claim_data) => claim_data,
        Err(e) => panic!("Error: {}", e),
    }
}

/// Submit field results to the server. Panic if there is an error.
pub fn submit_field_to_server(mode: &SearchMode, api_base: &str, submit_data: FieldSubmit) {
    let url = match mode {
        SearchMode::Detailed => format!("{}/submit/detailed", api_base),
        SearchMode::Niceonly => format!("{}/submit/niceonly", api_base),
    };

    let response = reqwest::blocking::Client::new()
        .post(&url)
        .json(&submit_data)
        .send();
    match response {
        Ok(response) => {
            if response.status().is_success() {
                return; // 👍
            }
            match response.text() {
                Ok(msg) => panic!("Server returned an error: {}", msg),
                Err(_) => panic!("Server returned an error."),
            }
        }
        Err(e) => panic!("Network error: {}", e),
    }
}
