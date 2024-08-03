#!/usr/bin/env rust-script
//! ```cargo
//! [dependencies]
//! nice_common = { path = "../common" }
//! ```

use nice_common::db_util;

fn main() {
    // get db connection
    let mut conn = db_util::get_database_connection();

    // temporarily in db_util for debugging
    db_util::do_downsampling(&mut conn);
}
