//! Expand basic numbers with some redundant stats.

use super::*;

pub fn expand_numbers(numbers: &[NiceNumbersSimple], base: u32) -> Vec<NiceNumbersExtended> {
    let base_f32 = base as f32;
    numbers
        .iter()
        .map(|n| NiceNumbersExtended {
            number: n.number,
            num_uniques: n.num_uniques,
            base,
            niceness: n.num_uniques as f32 / base_f32,
        })
        .collect()
}

pub fn shrink_numbers(numbers: &[NiceNumbersExtended]) -> Vec<NiceNumbersSimple> {
    numbers
        .iter()
        .map(|n| NiceNumbersSimple {
            number: n.number,
            num_uniques: n.num_uniques,
        })
        .collect()
}

// TODO: tests
// TODO: separate out dist/nums
