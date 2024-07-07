//! A module to generate some basic offline benchmarking ranges.

use super::*;

/// Different benchmark strategies.
#[derive(Debug, Copy, Clone, ValueEnum)]
pub enum BenchmarkMode {
    /// The default benchmark range: 1e5 @ base 40.
    Default,
    /// A large benchmark range: 1e7 @ base 40.
    Large,
    /// A very large benchmark range: 1e9 @ base 40.
    /// This is the size of a typical field from the server.
    ExtraLarge,
    /// A benchmark range at a higher range: 1e5 @ base 80.
    HiBase,
}

pub fn get_benchmark_field(mode: BenchmarkMode) -> FieldToClient {
    let base = match mode {
        BenchmarkMode::Default => 40,
        BenchmarkMode::Large => 40,
        BenchmarkMode::ExtraLarge => 40,
        BenchmarkMode::HiBase => 80,
    };
    let (range_start, _) = base_range::get_base_range_u128(base).unwrap().unwrap();
    let range_size = match mode {
        BenchmarkMode::Default => 100000,
        BenchmarkMode::Large => 10000000,
        BenchmarkMode::ExtraLarge => 1000000000,
        BenchmarkMode::HiBase => 100000,
    };

    FieldToClient {
        claim_id: 0,
        base,
        range_start,
        range_end: range_start + range_size,
        range_size,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_get_benchmark_field() {
        get_benchmark_field(BenchmarkMode::Default);
        get_benchmark_field(BenchmarkMode::Large);
        get_benchmark_field(BenchmarkMode::ExtraLarge);
        get_benchmark_field(BenchmarkMode::HiBase);
    }
}
