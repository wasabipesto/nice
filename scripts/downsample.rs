#!/usr/bin/env rust-script
//! ```cargo
//! [dependencies]
//! nice_common = { path = "../common" }
//! ```

fn main() {
    // get db connection
    let mut conn = db_util::get_database_connection();

    // register run started

    // downsampling
    // get list of bases
    // for each base, get chunks
    //   get list of chunks
    //   initiate base-level counters
    //   for each chunk, get fields
    //   - check if >10% of range is searched
    //   - downsample distribution
    //   - save all nice numbers
    //   - apply to chunk
    //   downsample base-level distribution
    //   save all nice numbers

    // register run ended
}
