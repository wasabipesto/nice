//! Benchmark scenario definitions and scoring, shared between clients.
//!
//! The native client's `--benchmark` sweep and the browser client's
//! benchmark suite measure the same fixed windows so their reports are
//! comparable; keeping the definitions (and the score references) in one
//! place is what keeps them comparable. The drivers stay platform-specific
//! — `client/src/bench.rs` for the native sweep, the search page for the
//! browser — but the *work measured* is defined here.
//!
//! A fixed measurement region: both the start and the window length are
//! hardcoded so every machine measures *identical work*; machine speed only
//! changes how many repetitions fit in the scenario's time share.
//! Repetition also solves timer granularity — a machine that clears the
//! window in microseconds simply runs it thousands of times.

use crate::base_range::get_base_range_u128;

/// Version of the benchmark JSON report layout. Bump on breaking changes.
pub const BENCH_SCHEMA_VERSION: u32 = 1;

/// A fixed measurement region (see the module docs).
pub struct ScenarioDef {
    pub key: &'static str,
    pub base: u32,
    /// None = the base range start (a strongly MSD-filtered region).
    pub start: Option<u128>,
    /// Fixed window length for CPU runs; sized so one repetition stays
    /// tractable on very slow devices (a Raspberry Pi class machine should
    /// clear it within roughly a scenario share).
    pub window_cpu: u128,
    /// Fixed window length for GPU runs; sized so one repetition amortizes
    /// launch overhead on data-center class devices.
    pub window_gpu: u128,
    /// Rough character of the region, for human readers of the report.
    pub character: &'static str,
    /// Run with a single thread instead of the configured thread count.
    /// One such scenario per sweep lets analysis decompose full-thread
    /// results into per-core rate × parallel efficiency.
    pub single_thread: bool,
}

impl ScenarioDef {
    /// The region's resolved start position.
    ///
    /// # Panics
    /// If the scenario names a base without a valid range, which would be a
    /// defect in the table below.
    #[must_use]
    pub fn resolved_start(&self) -> u128 {
        self.start.unwrap_or_else(|| {
            get_base_range_u128(self.base)
                .expect("benchmark base must be valid")
                .expect("benchmark base must have a range")
                .start()
        })
    }
}

pub const NICEONLY_SCENARIOS: &[ScenarioDef] = &[
    ScenarioDef {
        key: "b40_msd_strong",
        base: 40,
        start: None,
        window_cpu: 100_000_000,
        window_gpu: 8_000_000_000,
        character: "msd-strong",
        single_thread: false,
    },
    ScenarioDef {
        key: "b40_msd_weak",
        base: 40,
        start: Some(5_007_828_088_304),
        window_cpu: 20_000_000,
        window_gpu: 4_000_000_000,
        character: "msd-weak",
        single_thread: false,
    },
    ScenarioDef {
        key: "b50_residue_dense",
        base: 50,
        start: Some(27_219_467_191_689_038),
        window_cpu: 20_000_000,
        window_gpu: 4_000_000_000,
        character: "residue-dense",
        single_thread: false,
    },
    ScenarioDef {
        key: "b50_msd_weak",
        base: 50,
        start: Some(73_940_161_512_353_211),
        window_cpu: 20_000_000,
        window_gpu: 4_000_000_000,
        character: "msd-weak",
        single_thread: false,
    },
    ScenarioDef {
        key: "b52_msd_weak",
        base: 52,
        start: Some(407_887_399_136_188_818),
        window_cpu: 20_000_000,
        window_gpu: 4_000_000_000,
        character: "msd-weak",
        single_thread: false,
    },
    // Same region and window as b50_msd_weak so the pair decomposes into
    // per-core rate × parallel efficiency. On very slow devices a single
    // repetition of this window may exceed the scenario share; one full
    // repetition is always completed, so the budget is a soft target.
    ScenarioDef {
        key: "b50_msd_weak_1t",
        base: 50,
        start: Some(73_940_161_512_353_211),
        window_cpu: 20_000_000,
        window_gpu: 0,
        character: "msd-weak",
        single_thread: true,
    },
];

pub const DETAILED_SCENARIOS: &[ScenarioDef] = &[
    ScenarioDef {
        key: "b40_detailed",
        base: 40,
        start: None,
        window_cpu: 2_000_000,
        window_gpu: 200_000_000,
        character: "uniform",
        single_thread: false,
    },
    ScenarioDef {
        key: "b50_detailed",
        base: 50,
        start: None,
        window_cpu: 2_000_000,
        window_gpu: 200_000_000,
        character: "uniform",
        single_thread: false,
    },
    ScenarioDef {
        key: "b50_detailed_1t",
        base: 50,
        start: None,
        window_cpu: 1_000_000,
        window_gpu: 0,
        character: "uniform",
        single_thread: true,
    },
];

/// Reference rates (numbers/sec) for the synthetic score, pinned per client
/// version: (scenario key, gpu, reference rate). CPU references were measured
/// on a 4-core `x86_64` dev box, GPU references on an RTX 3060; a score of 1000
/// means "matches the reference machine on the geometric mean".
///
/// The browser suite scores against these same references deliberately: a
/// browser scoring 550 where the native client scores 1000 on the same box
/// is information, not a bug.
///
/// The GPU niceonly references predate the benchmark steering the MSD floor
/// to convergence before each scenario (`gpu_niceonly::benchmark_floor_thaw`);
/// they were measured under the earlier per-field controller and are due for
/// re-pinning at the next reference bump.
pub const SCORE_REFERENCES: &[(&str, bool, f64)] = &[
    ("b40_msd_strong", false, 1.0e12),
    ("b40_msd_weak", false, 1.6e9),
    ("b50_residue_dense", false, 1.1e9),
    ("b50_msd_weak", false, 9.4e8),
    ("b52_msd_weak", false, 3.2e9),
    ("b50_msd_weak_1t", false, 2.0e8),
    ("b40_detailed", false, 1.4e7),
    ("b50_detailed", false, 8.9e6),
    ("b50_detailed_1t", false, 2.2e6),
    ("b40_msd_strong", true, 2.3e11),
    ("b40_msd_weak", true, 1.5e11),
    ("b50_residue_dense", true, 1.3e11),
    ("b50_msd_weak", true, 1.3e11),
    ("b52_msd_weak", true, 1.6e11),
    ("b40_detailed", true, 4.5e9),
    ("b50_detailed", true, 3.2e9),
];

/// Geometric mean of measured rate over reference rate, scaled so the
/// reference machine scores 1000. Scenarios without a pinned reference or
/// that were dropped (rate <= 0) are excluded; `None` if nothing scored.
pub fn compute_score<'a>(
    rates: impl IntoIterator<Item = (&'a str, f64)>,
    gpu: bool,
) -> Option<f64> {
    let mut log_sum = 0.0;
    let mut count = 0usize;
    for (key, rate) in rates {
        if rate <= 0.0 {
            continue;
        }
        let Some((_, _, reference)) = SCORE_REFERENCES
            .iter()
            .find(|(ref_key, is_gpu, _)| *ref_key == key && *is_gpu == gpu)
        else {
            continue;
        };
        log_sum += (rate / reference).ln();
        count += 1;
    }
    #[allow(clippy::cast_precision_loss)]
    (count > 0).then(|| 1000.0 * (log_sum / count as f64).exp())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn score_uses_only_referenced_scenarios() {
        let reference_rate = SCORE_REFERENCES
            .iter()
            .find(|(k, g, _)| *k == "b50_msd_weak" && !g)
            .unwrap()
            .2;
        // Exactly matching the reference on the only scored scenario = 1000;
        // an unknown key contributes nothing.
        let score = compute_score(
            [
                ("b50_msd_weak", reference_rate),
                ("not_a_real_scenario", 1.0),
            ],
            false,
        )
        .unwrap();
        assert!((score - 1000.0).abs() < 1e-6);
        // An unmeasured scenario contributes nothing either.
        assert_eq!(compute_score([("b50_msd_weak", 0.0)], false), None);
    }

    #[test]
    fn all_scenarios_have_cpu_references() {
        // Every CPU scenario must be scoreable, or the score silently thins.
        for def in NICEONLY_SCENARIOS.iter().chain(DETAILED_SCENARIOS) {
            assert!(
                SCORE_REFERENCES
                    .iter()
                    .any(|(k, gpu, _)| k == &def.key && !gpu),
                "missing CPU score reference for {}",
                def.key
            );
        }
    }

    #[test]
    fn scenario_starts_resolve() {
        // `resolved_start` panics on a base with no range; catch a bad table
        // entry here instead of at a user's benchmark run.
        for def in NICEONLY_SCENARIOS.iter().chain(DETAILED_SCENARIOS) {
            let start = def.resolved_start();
            assert!(start > 0, "{} resolved to zero", def.key);
        }
    }
}
