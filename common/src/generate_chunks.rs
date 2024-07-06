//! Generate larger search ranges ("chunks") for analytics.

use super::*;

const TARGET_NUM_CHUNKS: f32 = 100.0;

/// Takes a list of fields and attempts to group them into equally-sized chunks for analytics.
/// Has a hardcoded target number of chunks, will produce at most that many chunks.
/// If there is less than that many fields, each field gets its own chunk.
/// Destroys the vec passed to it.
pub fn group_fields_into_chunks(fields: Vec<FieldSize>) -> Vec<FieldSize> {
    // convert fields into ring vec in order to pop out the front
    let mut fields = VecDeque::from(fields);

    // create output vec
    let mut chunks = Vec::new();

    // figure out how many fields per chunk
    let num_fields_per_chunk = (fields.len() as f32 / TARGET_NUM_CHUNKS).ceil() as usize;

    // break out each chunk, consuming the fields we use
    while !fields.is_empty() {
        // init vec for chunk fields
        let mut chunk_fields = VecDeque::with_capacity(num_fields_per_chunk);

        // consume the first n fields
        for _ in 0..num_fields_per_chunk {
            match fields.pop_front() {
                Some(field) => chunk_fields.push_back(field),
                None => break,
            }
        }

        // get the start, end, and size from the chunk
        let start = &chunk_fields.front().unwrap().start;
        let end = &chunk_fields.back().unwrap().end;
        let size = u128::try_from(&(end - start)).unwrap();
        chunks.push(FieldSize {
            start: start.clone(),
            end: end.clone(),
            size,
        });
    }

    chunks
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_group_fields_into_chunks_b10() {
        let base = 10;
        let size = 1000000000;
        let base_range = base_range::get_base_range_natural(base).unwrap();
        let fields = generate_fields::break_range_into_fields(&base_range.0, &base_range.1, size);
        let chunks = group_fields_into_chunks(fields.clone());

        // check against known field
        assert_eq!(
            chunks,
            vec![FieldSize {
                start: Natural::from(47u32),
                end: Natural::from(100u32),
                size: 53
            }]
        );

        // check the fields were not affected
        assert_eq!(fields.len(), 1);
    }

    #[test]
    fn test_group_fields_into_chunks_general() {
        for base in [10, 11, 12, 13, 14, 15, 20, 30, 40] {
            for size in [100000000, 1000000000, 10000000000] {
                let base_range = base_range::get_base_range_natural(base);
                if let Some(range) = base_range {
                    // get the fields
                    let fields = generate_fields::break_range_into_fields(&range.0, &range.1, size);
                    let num_fields = fields.len();

                    // get the chunks
                    let chunks = group_fields_into_chunks(fields.clone());

                    // check the start and end are correct
                    assert_eq!(chunks.first().unwrap().start, range.0);
                    assert_eq!(chunks.last().unwrap().end, range.1);

                    // check there are at most 100 chunks
                    assert!(chunks.len() <= TARGET_NUM_CHUNKS as usize);

                    // check the chunks are in ascending order
                    let mut last_start = Natural::from(0u32);
                    for chunk in chunks {
                        assert!(chunk.start > last_start);
                        last_start = chunk.start.clone()
                    }

                    // check the fields were not affected
                    assert_eq!(fields.len(), num_fields);
                }
            }
        }
    }
}
