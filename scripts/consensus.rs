#!/usr/bin/env rust-script
//! ```cargo
//! [dependencies]
//! nice_common = { path = "../common" }
//! ```

use nice_common::db_util;

fn main() {
    // get db connection
    let mut conn = db_util::get_database_connection();

    // TODO: register run started

    // consensus
    // - only run on new submissions
    // - manual run on all submissions?
    // get all relevant submissions (matches field, not disqualified, detailed)
    // check there is a majority consensus
    // get the first agreeing submission, set it as canon
    // update field check level

    // register run ended
}
