#!/usr/bin/env rust-script
//! ```cargo
//! [dependencies]
//! nice_common = { path = "../common", features = ["database"] }
//! clap = "4.6"
//! read_input = "0.8"
//! ```

use clap::Parser;
use read_input::prelude::*;

fn print_field(i: usize, field: &nice_common::FieldSize) -> () {
    println!("Field #{}:", i + 1);
    println!("  Start: {start} ({start:.2e})", start = field.start());
    println!("  End:   {end} ({end:.2e})", end = field.end());
    println!("  Size:  {size} ({size:.0e})", size = field.size());
}

fn print_chunk(i: usize, chunk: &nice_common::FieldSize) -> () {
    println!("Chunk #{}:", i + 1);
    println!("  Start: {start} ({start:.2e})", start = chunk.start());
    println!("  End:   {end} ({end:.2e})", end = chunk.end());
    println!("  Size:  {size} ({size:.0e})", size = chunk.size());
}

#[derive(Parser)]
pub struct Cli {
    #[arg(short, long)]
    base: u32,
}

fn main() {
    // parse args from command line
    let cli = Cli::parse();
    let base = cli.base;

    let field_size = 1e11 as u128;
    println!("Using field size {:.0e}.", field_size);
    println!();

    let base_range = nice_common::base_range::get_base_range_u128(base)
        .unwrap()
        .expect("Base has no valid range!");
    println!("Base Range:");
    println!("  Base:    {base}");
    println!(
        "  Minimum: {start} ({start:.2e})",
        start = base_range.start()
    );
    println!("  Maximum: {start} ({start:.2e})", start = base_range.end());
    println!();

    let fields = nice_common::generate_fields::break_range_into_fields(
        base_range.start(),
        base_range.end(),
        field_size,
    );

    if fields.len() > 10 {
        for (i, field) in fields.iter().take(5).enumerate() {
            print_field(i, &field)
        }

        println!();
        println!(
            "... {num} ({num:.2e}) fields omitted ...",
            num = fields.len() - 10
        );
        println!();

        for (i, field) in fields.iter().rev().take(5).rev().enumerate() {
            print_field(fields.len() - 5 + i, &field)
        }
    } else {
        for (i, field) in fields.iter().enumerate() {
            print_field(i, &field)
        }
    }
    println!();

    let chunks = nice_common::generate_chunks::group_fields_into_chunks(fields.clone());
    if chunks.len() > 5 {
        for (i, chunk) in chunks.iter().take(5).enumerate() {
            print_chunk(i, &chunk)
        }

        println!();
        println!("... {} chunks omitted ...", chunks.len() - 10);
        println!();

        for (i, chunk) in chunks.iter().rev().take(5).rev().enumerate() {
            print_chunk(chunks.len() - 5 + i, &chunk)
        }
    } else {
        for (i, chunk) in chunks.iter().enumerate() {
            print_chunk(i, &chunk)
        }
    }
    println!();

    print!("Add to database? [y/N] ");
    let confirm_add_to_db = input::<String>().get();
    if !["y", "Y", "ye", "yes"].contains(&confirm_add_to_db.as_str()) {
        return;
    }
    let mut conn = nice_common::db_util::get_database_connection();
    if let Ok(base_data) = nice_common::db_util::bases::get_base_by_id(&mut conn, base) {
        panic!("Base {} already exists: {:?}", base, base_data)
    }

    println!("Inserting base {}...", base);
    nice_common::db_util::bases::insert_base(&mut conn, base, base_range).unwrap();
    println!("Inserting {} fields...", fields.len());
    nice_common::db_util::fields::insert_fields(&mut conn, base, &fields).unwrap();
    println!("Inserting {} chunks...", chunks.len());
    nice_common::db_util::chunks::insert_chunks(&mut conn, base, &chunks).unwrap();
    println!("Updating base {} chunk assignments...", base);
    nice_common::db_util::chunks::reassign_fields_to_chunks(&mut conn, base).unwrap();
    println!("Database updated.")
}
