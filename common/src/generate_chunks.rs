//! Generate larger search ranges ("chunks") for analytics.

use crate::FieldSize;
use std::collections::VecDeque;

const TARGET_NUM_CHUNKS: f32 = 100.0;

/// Takes a list of fields and attempts to group them into equally-sized chunks for analytics.
/// Has a hardcoded target number of chunks, will produce at most that many chunks.
/// If there is less than that many fields, each field gets its own chunk.
/// Destroys the vec passed to it.
///
/// **Range semantics**: Input fields are half-open ranges [start, end), and output chunks
/// are also half-open ranges. The function preserves the range boundaries correctly by
/// using the first field's `range_start` and the last field's `range_end` for each chunk.
///
/// # Panics
/// Panics if `fields` is empty.
#[must_use]
pub fn group_fields_into_chunks(fields: Vec<FieldSize>) -> Vec<FieldSize> {
    // Convert fields into ring vec in order to pop out the front
    let mut fields = VecDeque::from(fields);

    // Create output vec
    let mut chunks = Vec::new();

    // Figure out how many fields per chunk
    #[allow(
        clippy::cast_sign_loss,
        clippy::cast_possible_truncation,
        clippy::cast_precision_loss
    )]
    let num_fields_per_chunk = (fields.len() as f32 / TARGET_NUM_CHUNKS).ceil() as usize;

    // Break out each chunk, consuming the fields we use
    while !fields.is_empty() {
        // Initialize vec for chunk fields
        let mut chunk_fields = VecDeque::with_capacity(num_fields_per_chunk);

        // Consume the first n fields
        for _ in 0..num_fields_per_chunk {
            match fields.pop_front() {
                Some(field) => chunk_fields.push_back(field),
                None => break,
            }
        }

        // Get the start and end from the chunk (preserving half-open range semantics)
        // `range_start` is inclusive (from first field), `range_end` is exclusive (from last field)
        let range_start = chunk_fields
            .front()
            .expect("chunk_fields should not be empty")
            .range_start;
        let range_end = chunk_fields
            .back()
            .expect("chunk_fields should not be empty")
            .range_end;
        chunks.push(FieldSize::new(range_start, range_end));
    }

    chunks
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{base_range, generate_fields};

    #[test]
    fn test_group_fields_into_chunks_b10() {
        let base = 10;
        let size = 1_000_000_000;
        let base_range = base_range::get_base_range_u128(base).unwrap().unwrap();
        let fields = generate_fields::break_range_into_fields(
            base_range.range_start,
            base_range.range_end,
            size,
        );
        let chunks = group_fields_into_chunks(fields.clone());

        // check against known field
        assert_eq!(chunks, vec![FieldSize::new(47u128, 100u128)]);

        // check the fields were not affected
        assert_eq!(fields.len(), 1);
    }

    #[test]
    fn test_group_fields_into_chunks_general() {
        for base in [10, 11, 12, 13, 14, 15, 20, 30, 40] {
            for size in [100_000_000, 1_000_000_000, 10_000_000_000] {
                let base_range = base_range::get_base_range_u128(base).unwrap();
                if let Some(range) = base_range {
                    // get the fields
                    let fields = generate_fields::break_range_into_fields(
                        range.range_start,
                        range.range_end,
                        size,
                    );
                    let num_fields = fields.len();

                    // get the chunks
                    let chunks = group_fields_into_chunks(fields.clone());

                    // check the start and end are correct
                    assert_eq!(chunks.first().unwrap().range_start, range.range_start);
                    assert_eq!(chunks.last().unwrap().range_end, range.range_end);

                    // check there are at most 100 chunks
                    #[allow(clippy::cast_precision_loss)]
                    let num_chunks = chunks.len() as f32;
                    assert!(num_chunks <= TARGET_NUM_CHUNKS);

                    // check the chunks are in ascending order
                    let mut last_start = 0u128;
                    for chunk in chunks {
                        assert!(chunk.range_start > last_start);
                        last_start = chunk.range_start;
                    }

                    // check the fields were not affected
                    assert_eq!(fields.len(), num_fields);
                }
            }
        }
    }
}
