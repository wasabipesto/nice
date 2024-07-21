#!/usr/bin/env rust-script
//! ```cargo
//! [dependencies]
//! nice_common = { path = "../common" }
//! ```

use nice_common::db_util;
use nice_common::SubmissionRecord;

fn main() {
    // get db connection
    let mut conn = db_util::get_database_connection();

    // get all bases
    let bases = db_util::get_all_bases(&mut conn).unwrap();

    for b in bases {
        let base = b.base;

        // get canon submissions
        let canon_submissions: Vec<SubmissionRecord> =
            db_util::get_canon_submissions_in_base(&mut conn, base).unwrap();

        let distributions = canon_submissions.map(|s| s.distribution)
    }

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
