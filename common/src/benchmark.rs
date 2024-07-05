//! A module to generate some basic offline benchmarking ranges.

use super::*;

/// Different benchmark strategies.
#[derive(Debug, Copy, Clone, ValueEnum)]
pub enum BenchmarkType {
    /// The default benchmark range: 1e5 @ base 40.
    Default,
    /// A large benchmark range: 1e7 @ base 40.
    Large,
    /// A very large benchmark range: 1e9 @ base 40.
    /// This is the size of a normal field.
    ExtraLarge,
    /// A benchmark range at a higher range: 1e5 @ base 80.
    HiBase,
}

pub trait Benchmarker {
    fn get_field(&self) -> FieldClaim;
}

/// Generate a field offline for benchmark testing.
impl Benchmarker for BenchmarkType {
    fn get_field(&self) -> FieldClaim {
        let base = match self {
            BenchmarkType::Default => 40,
            BenchmarkType::Large => 40,
            BenchmarkType::ExtraLarge => 40,
            BenchmarkType::HiBase => 80,
        };
        let (search_start, range_end) = get_base_range(base);
        let range: u128 = match self {
            BenchmarkType::Default => 100000,
            BenchmarkType::Large => 10000000,
            BenchmarkType::ExtraLarge => 1000000000,
            BenchmarkType::HiBase => 100000,
        };
        let search_end = range_end.min(&search_start + &Natural::from(range));
        let search_range = &search_end - &search_start;

        FieldClaim {
            id: 0,
            username: "benchmark".to_owned(),
            base,
            search_start,
            search_end,
            search_range,
        }
    }
}
