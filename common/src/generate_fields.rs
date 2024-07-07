//! Generate the search fields for processing.

use super::*;

/// Break a base range into smaller, searchable fields.
/// Each field should be `size` in width, with the last one being smaller.
/// If the base range is less than `size` it returns one field.
pub fn break_range_into_fields(min: u128, max: u128, size: u128) -> Vec<FieldSize> {
    // create output vec
    let mut fields = Vec::new();

    // start the field bound counters
    let mut start = min;
    let mut end = min;

    // walk through base range
    while end < max {
        // calculate the end and size
        end = start.add(&size).min(max);

        // build and push the field
        let field = FieldSize {
            start,
            end,
            size: end - start,
        };
        fields.push(field);

        // bump the start
        start = end;
    }
    fields
}

#[cfg(test)]
mod tests {
    use super::*;
    use malachite::num::arithmetic::traits::DivMod;

    #[test]
    fn test_break_range_into_fields_b10() {
        let base = 10;
        let size = 1000000000;
        let base_range = base_range::get_base_range_u128(base).unwrap().unwrap();
        let fields = break_range_into_fields(base_range.0, base_range.1, size);

        // check against known field
        assert_eq!(
            fields,
            vec![FieldSize {
                start: 47u128,
                end: 100u128,
                size: 53u128
            }]
        );
    }

    #[test]
    fn test_break_range_into_fields_general() {
        for base in [10, 11, 12, 13, 14, 15, 20, 30, 40] {
            for size in [100000000, 1000000000, 10000000000] {
                let base_range = base_range::get_base_range_u128(base).unwrap();
                if let Some(range) = base_range {
                    // get the fields
                    let fields = break_range_into_fields(range.0, range.1, size);

                    // check the start and end are correct
                    assert_eq!(fields.first().unwrap().start, range.0);
                    assert_eq!(fields.last().unwrap().end, range.1);

                    // check the sizes are within range
                    for field in &fields {
                        assert!(field.size <= size);
                    }

                    // check there are the right number of fields
                    let (num_fields_mo, last_field_size) =
                        (range.1.clone() - range.0.clone()).div_mod(size);
                    assert_eq!(fields.len(), num_fields_mo as usize + 1);

                    // check the last field is the correct size
                    assert_eq!(fields.last().unwrap().size, last_field_size);

                    // check the first field is the correct size
                    if fields.len() > 1 {
                        assert_eq!(fields.first().unwrap().size, size);
                    } else {
                        let range_size = range.1 - range.0;
                        assert_eq!(fields.first().unwrap().size, range_size);
                    }

                    // check the fields are in ascending order
                    let mut last_start = 0u128;
                    for field in fields {
                        assert!(field.start > last_start);
                        last_start = field.start.clone()
                    }
                }
            }
        }
    }
}
