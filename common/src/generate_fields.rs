//! Generate the search fields for processing.

use super::*;

/// Break a base range into smaller, searchable fields.
/// Each field should be `size` in width, with the last one being smaller.
/// If the base range is less than `size` it returns one field.
pub fn break_range_into_fields(min: &Natural, max: &Natural, size_u128: u128) -> Vec<SearchField> {
    // create output vec
    let mut fields = Vec::new();

    // convert to natural for operation
    let size = Natural::from(size_u128);

    // start the field bound counters
    let mut field_start = min.clone();
    let mut field_end = min.clone();

    // walk through base range
    while &field_end < max {
        // calculate the end and size
        field_end = field_start.clone().add(&size).min(max.clone());

        // size should always fit in u128 since it's clamped to a value that fits in u128
        let field_size = u128::try_from(&(field_end.clone() - field_start.clone())).unwrap();

        // build and push the field
        let field = SearchField {
            start: field_start.clone(),
            end: field_end.clone(),
            size: field_size,
        };
        fields.push(field);

        // bump the start
        field_start = field_end.clone();
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
        let base_range = base_range::get_base_range_natural(base).unwrap();
        let fields = break_range_into_fields(&base_range.0, &base_range.1, size);
        assert_eq!(
            fields,
            vec![SearchField {
                start: Natural::from(47u32),
                end: Natural::from(100u32),
                size: 53
            }]
        );
    }

    #[test]
    fn test_break_range_into_fields_general() {
        for base in [10, 11, 12, 13, 14, 15, 20, 30, 40] {
            for size in [100000000, 1000000000, 10000000000] {
                let base_range = base_range::get_base_range_natural(base);
                if let Some(range) = base_range {
                    // get the fields
                    let fields = break_range_into_fields(&range.0, &range.1, size);

                    // check the start and end are correct
                    assert_eq!(fields.first().unwrap().start, range.0);
                    assert_eq!(fields.last().unwrap().end, range.1);

                    // check the sizes are within range
                    for field in &fields {
                        assert!(field.size <= size);
                    }

                    // check there are the right number of fields
                    let (num_fields_mo, last_field_size) =
                        (range.1.clone() - range.0.clone()).div_mod(Natural::from(size));
                    assert_eq!(fields.len(), usize::try_from(&num_fields_mo).unwrap() + 1);

                    // check the last field is the correct size
                    assert_eq!(
                        fields.last().unwrap().size,
                        u128::try_from(&last_field_size).unwrap()
                    );

                    // check the first field is the correct size
                    if fields.len() > 1 {
                        assert_eq!(fields.first().unwrap().size, size);
                    } else {
                        let range_size = u128::try_from(&(range.1 - range.0)).unwrap();
                        assert_eq!(fields.first().unwrap().size, range_size);
                    }
                }
            }
        }
    }
}
