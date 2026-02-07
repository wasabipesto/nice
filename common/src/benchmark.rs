//! A module to generate some basic offline benchmarking ranges.

use super::*;

/// Different benchmark strategies.
#[derive(Debug, Copy, Clone, ValueEnum)]
pub enum BenchmarkMode {
    /// Checking on base ten with a known nice number.
    BaseTen,
    /// The default benchmark range: 1e6 @ base 40.
    Default,
    /// A large benchmark range: 1e8 @ base 40.
    Large,
    /// A very large benchmark range: 1e9 @ base 40.
    /// This is the size of a typical field from the server.
    ExtraLarge,
    /// A massive range, much larger than any field you would get from the API:
    /// 1e13 @ base 50.
    Massive,
    /// A benchmark range at a higher range: 1e6 @ base 80.
    HiBase,
    /// A range where MSD filtering is quite effective: 1e12 @ base 50.
    MsdEffective,
    /// A range where MSD filtering is ineffective: 1e11 @ base 50.
    MsdIneffective,
}

/// Get a benchmark field for testing and performance evaluation.
///
/// **Range semantics**: Returns a DataToClient with a half-open range [range_start, range_end),
/// where range_start is inclusive, range_end is exclusive, and range_size = range_end - range_start.
pub fn get_benchmark_field(mode: BenchmarkMode) -> DataToClient {
    let base = match mode {
        BenchmarkMode::BaseTen => 10,
        BenchmarkMode::Default => 40,
        BenchmarkMode::Large => 40,
        BenchmarkMode::ExtraLarge => 40,
        BenchmarkMode::Massive => 50,
        BenchmarkMode::HiBase => 80,
        BenchmarkMode::MsdEffective => 50,
        BenchmarkMode::MsdIneffective => 50,
    };
    let base_range = base_range::get_base_range_u128(base).unwrap().unwrap();
    let range_start = match mode {
        BenchmarkMode::MsdEffective => 26507984537059635,
        BenchmarkMode::MsdIneffective => 94760515586064977,
        _ => base_range.range_start,
    };
    let range_size = match mode {
        BenchmarkMode::BaseTen => base_range.range_size,
        BenchmarkMode::Default => 1_000_000,
        BenchmarkMode::Large => 100_000_000,
        BenchmarkMode::ExtraLarge => 1_000_000_000,
        BenchmarkMode::Massive => 1e13 as u128,
        BenchmarkMode::HiBase => 1_000_000_000,
        BenchmarkMode::MsdEffective => 1e12 as u128,
        BenchmarkMode::MsdIneffective => 1e7 as u128,
    };

    // Create a half-open range [range_start, range_end) with range_size elements
    DataToClient {
        claim_id: 0,
        base,
        range_start,
        range_end: range_start + range_size,
        range_size,
    }
}

impl fmt::Display for BenchmarkMode {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{:?}", self)
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
