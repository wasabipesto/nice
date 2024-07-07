#!/usr/bin/env rust-script
//! ```cargo
//! [dependencies]
//! nice_common = { path = "../common" }
//! read_input = "0.8"
//! ```

use read_input::prelude::*;

fn main() {
    println!("Enter a base: ")
    let base = input::<u32>().get();
    let field_size = 1000000000;
    println!("Using default field size {}.", field_size)

    let base_range = nice_common::base_range::get_base_range_natural(base).unwrap();
    println!("Base Range:");
    println!("  Base:    {}", base);
    println!("  Minimum: {}", base_range.0);
    println!("  Maximum: {}", base_range.1);
    println!("");

    let fields = nice_common::generate_fields::break_range_into_fields(
        &base_range.0,
        &base_range.1,
        field_size,
    );
    for (i, field) in fields.iter().enumerate() {
        println!("Field #{}:", i + 1);
        println!("  Start: {}", field.start);
        println!("  End:   {}", field.end);
        println!("  Size:  {}", field.size);
    }
}
