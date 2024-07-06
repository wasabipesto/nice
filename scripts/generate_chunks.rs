#!/usr/bin/env rust-script
//! ```cargo
//! [dependencies]
//! nice_common = { path = "../common" }
//! ```

fn main() {
    let field_size = 1000000000;
    println!("Field size: {}", field_size);

    for base in 10..51 {
        let base_range = nice_common::base_range::get_base_range_natural(base);
        if let Some(range) = base_range {
            let fields = nice_common::generate_fields::break_range_into_fields(
                &range.0, &range.1, field_size,
            );
            let chunks = nice_common::generate_chunks::group_fields_into_chunks(fields.clone());
            println!(
                "Base {}: {} fields, {} chunks, {:.2} f/c",
                base,
                fields.len(),
                chunks.len(),
                fields.len() as f32 / chunks.len() as f32
            );
        }
    }
}
