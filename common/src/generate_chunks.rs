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
        let range_start = chunk_fields.front().unwrap().range_start;
        let range_end = chunk_fields.back().unwrap().range_end;
        let range_size = range_end - range_start;
        chunks.push(FieldSize {
            range_start,
            range_end,
            range_size,
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
        let base_range = base_range::get_base_range_u128(base).unwrap().unwrap();
        let fields = generate_fields::break_range_into_fields(base_range.0, base_range.1, size);
        let chunks = group_fields_into_chunks(fields.clone());

        // check against known field
        assert_eq!(
            chunks,
            vec![FieldSize {
                range_start: 47u128,
                range_end: 100u128,
                range_size: 53u128
            }]
        );

        // check the fields were not affected
        assert_eq!(fields.len(), 1);
    }

    #[test]
    fn test_group_fields_into_chunks_general() {
        for base in [10, 11, 12, 13, 14, 15, 20, 30, 40] {
            for size in [100000000, 1000000000, 10000000000] {
                let base_range = base_range::get_base_range_u128(base).unwrap();
                if let Some(range) = base_range {
                    // get the fields
                    let fields = generate_fields::break_range_into_fields(range.0, range.1, size);
                    let num_fields = fields.len();

                    // get the chunks
                    let chunks = group_fields_into_chunks(fields.clone());

                    // check the start and end are correct
                    assert_eq!(chunks.first().unwrap().range_start, range.0);
                    assert_eq!(chunks.last().unwrap().range_end, range.1);

                    // check there are at most 100 chunks
                    assert!(chunks.len() <= TARGET_NUM_CHUNKS as usize);

                    // check the chunks are in ascending order
                    let mut last_start = 0u128;
                    for chunk in chunks {
                        assert!(chunk.range_start > last_start);
                        last_start = chunk.range_start
                    }

                    // check the fields were not affected
                    assert_eq!(fields.len(), num_fields);
                }
            }
        }
    }
}
