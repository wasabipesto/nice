//! Expand basic numbers with some redundant stats.

use super::*;

pub fn expand_numbers(numbers: &[NiceNumberSimple], base: u32) -> Vec<NiceNumber> {
    let base_f32 = base as f32;
    numbers
        .iter()
        .map(|n| NiceNumber {
            number: n.number,
            num_uniques: n.num_uniques,
            base,
            niceness: n.num_uniques as f32 / base_f32,
        })
        .collect()
}

pub fn downsample_numbers(submissions: &[SubmissionRecord]) -> Vec<NiceNumber> {
    // collate all numbers
    let mut all_numbers = submissions.iter().fold(Vec::new(), |mut acc, sub| {
        acc.extend(sub.numbers.iter().cloned());
        acc
    });

    // sort and take the top few
    all_numbers.sort_by(|a, b| b.number.cmp(&a.number));
    all_numbers
        .iter()
        .take(SAVE_TOP_N_NUMBERS)
        .cloned()
        .collect()
}

pub fn shrink_numbers(numbers: &[NiceNumber]) -> Vec<NiceNumberSimple> {
    numbers
        .iter()
        .map(|n| NiceNumberSimple {
            number: n.number,
            num_uniques: n.num_uniques,
        })
        .collect()
}

// TODO: tests
