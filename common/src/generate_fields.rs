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
            range_start: start,
            range_end: end,
            range_size: end - start,
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
        let fields = break_range_into_fields(base_range.range_start, base_range.range_end, size);

        // check against known field
        assert_eq!(
            fields,
            vec![FieldSize {
                range_start: 47u128,
                range_end: 100u128,
                range_size: 53u128
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
                    let fields = break_range_into_fields(range.range_start, range.range_end, size);

                    // check the start and end are correct
                    assert_eq!(fields.first().unwrap().range_start, range.range_start);
                    assert_eq!(fields.last().unwrap().range_end, range.range_end);

                    // check the sizes are within range
                    for field in &fields {
                        assert!(field.range_size <= size);
                    }

                    // check there are the right number of fields
                    let (num_fields_mo, last_field_size) =
                        (range.range_end.clone() - range.range_start.clone()).div_mod(size);
                    assert_eq!(fields.len(), num_fields_mo as usize + 1);

                    // check the last field is the correct size
                    assert_eq!(fields.last().unwrap().range_size, last_field_size);

                    // check the first field is the correct size
                    if fields.len() > 1 {
                        assert_eq!(fields.first().unwrap().range_size, size);
                    } else {
                        assert_eq!(fields.first().unwrap().range_size, range.range_size);
                    }

                    // check the fields are in ascending order
                    let mut last_start = 0u128;
                    for field in fields {
                        assert!(field.range_start > last_start);
                        last_start = field.range_start.clone()
                    }
                }
            }
        }
    }
}
