//! Generate the search fields for processing.

use crate::FieldSize;
use std::ops::Add;

/// Break a base range into smaller, searchable fields.
/// Each field should be `size` in width, with the last one being smaller.
/// If the base range is less than `size` it returns one field.
///
/// **Range semantics**: This function takes an inclusive range [min, max] as input
/// and produces half-open ranges [start, end) as output. Each returned `FieldSize`
/// follows Rust's convention where `range_start` is inclusive and `range_end` is exclusive.
#[must_use]
pub fn break_range_into_fields(min: u128, max: u128, size: u128) -> Vec<FieldSize> {
    // create output vec
    let mut fields = Vec::new();

    // start the field bound counters
    let mut start = min;
    let mut end = min;

    // Walk through base range (half-open ranges: start is inclusive, end is exclusive)
    while end < max {
        // Calculate the end (exclusive) and size for this field
        end = start.add(&size).min(max);

        // Build and push the field (half-open range [start, end))
        fields.push(FieldSize::new(start, end));

        // Bump the start to the previous end (no gap, no overlap in half-open ranges)
        start = end;
    }
    fields
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::base_range;
    use malachite::base::num::arithmetic::traits::DivMod;

    #[test_log::test]
    fn test_break_range_into_fields_b10() {
        let base = 10;
        let size = 1_000_000_000;
        let base_range = base_range::get_base_range_u128(base).unwrap().unwrap();
        let fields = break_range_into_fields(base_range.range_start, base_range.range_end, size);

        // check against known field
        assert_eq!(fields, vec![FieldSize::new(47u128, 100u128)]);
    }

    #[test_log::test]
    fn test_break_range_into_fields_general() {
        for base in [10, 11, 12, 13, 14, 15, 20, 30, 40] {
            for size in [100_000_000, 1_000_000_000, 10_000_000_000] {
                let base_range = base_range::get_base_range_u128(base).unwrap();
                if let Some(range) = base_range {
                    // get the fields
                    let fields = break_range_into_fields(range.range_start, range.range_end, size);

                    // check the start and end are correct
                    assert_eq!(fields.first().unwrap().range_start, range.range_start);
                    assert_eq!(fields.last().unwrap().range_end, range.range_end);

                    // check the sizes are within range
                    for field in &fields {
                        assert!(field.size() <= size);
                    }

                    // check there are the right number of fields
                    let (num_fields_mo, last_field_size) =
                        (range.range_end - range.range_start).div_mod(size);
                    assert_eq!(fields.len(), num_fields_mo as usize + 1);

                    // check the last field is the correct size
                    assert_eq!(fields.last().unwrap().size(), last_field_size);

                    // check the first field is the correct size
                    if fields.len() > 1 {
                        assert_eq!(fields.first().unwrap().size(), size);
                    } else {
                        assert_eq!(fields.first().unwrap().size(), range.size());
                    }

                    // check the fields are in ascending order
                    let mut last_start = 0u128;
                    for field in fields {
                        assert!(field.range_start > last_start);
                        last_start = field.range_start;
                    }
                }
            }
        }
    }
}
