//! Benchmark scenario definitions and scoring, shared between clients.
//!
//! The native client's `--benchmark` sweep and the browser client's
//! benchmark suite measure the same fixed windows so their reports are
//! comparable; keeping the definitions (and the score references) in one
//! place is what keeps them comparable. The drivers stay platform-specific
//! — `client/src/bench.rs` for the native sweep, the search page for the
//! browser — but the *work measured* is defined here.
//!
//! A detailed scenario is a fixed measurement region: both the start and the
//! window length are hardcoded so every machine measures *identical work*;
//! machine speed only changes how many repetitions fit in the scenario's time
//! share. Repetition also solves timer granularity — a machine that clears
//! the window in microseconds simply runs it thousands of times.
//!
//! A nice-only scenario is a fixed production field ([`FieldScenario`]),
//! measured on the route a client would take for it on the device. The
//! overlap join's cost per number keeps falling up to production sizes, so
//! a small window would not say what a field costs; instead the client times
//! a sample of the field's partitions, in [`sample_order`], and scales it to
//! the whole field. Every machine measures the same partitions first, and a
//! faster one covers more of the field in its share.

use crate::base_range::get_base_range_u128;

/// Version of the benchmark JSON report layout. Bump on breaking changes.
pub const BENCH_SCHEMA_VERSION: u32 = 1;

/// A fixed measurement region for detailed mode (see the module docs).
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

/// A nice-only scenario: one production field (see the module docs).
pub struct FieldScenario {
    pub key: &'static str,
    pub base: u32,
    /// The field, a real one from the server's table.
    pub start: u128,
    pub size: u128,
    /// For human readers of the report.
    pub character: &'static str,
    /// Run with a single thread instead of the configured thread count (see
    /// [`ScenarioDef::single_thread`]). CPU only.
    pub single_thread: bool,
    /// The window at the field's start that is timed when the field takes
    /// the stride path, whose cost is linear in size: per CPU thread, and on
    /// a GPU. Both are below the overlap join's minimum field size.
    pub stride_window_cpu: u128,
    pub stride_window_gpu: u128,
}

impl FieldScenario {
    /// The field's end (exclusive).
    #[must_use]
    pub fn end(&self) -> u128 {
        self.start + self.size
    }
}

pub const NICEONLY_FIELDS: &[FieldScenario] = &[
    // Base 57's frontier when the overlap join landed (field 206295804).
    FieldScenario {
        key: "b57_1e14",
        base: 57,
        start: 28_151_599_893_042_801_193,
        size: 100_000_000_000_000,
        character: "b57 field",
        single_thread: false,
        stride_window_cpu: 2_000_000_000,
        stride_window_gpu: 1_000_000_000_000,
    },
    // The same field on one thread, so the pair decomposes into per-core
    // rate × parallel efficiency.
    FieldScenario {
        key: "b57_1e14_1t",
        base: 57,
        start: 28_151_599_893_042_801_193,
        size: 100_000_000_000_000,
        character: "b57 field",
        single_thread: true,
        stride_window_cpu: 2_000_000_000,
        stride_window_gpu: 0,
    },
    // A base-58 field (206813862) of the size the server hands out there,
    // in a region with candidates: half of base 58's fields have none and
    // would measure only the join's setup.
    FieldScenario {
        key: "b58_1e15",
        base: 58,
        start: 102_121_216_587_923_470_200,
        size: 1_000_000_000_000_000,
        character: "b58 field",
        single_thread: false,
        stride_window_cpu: 2_000_000_000,
        stride_window_gpu: 1_000_000_000_000,
    },
];

/// The order a field scenario samples its partitions in: counting with the
/// bits reversed, skipping values past the end. Every prefix is spread
/// evenly over `0..partitions`, so a sample of any length covers the whole
/// partition space, and machines that sample different lengths still share
/// the partitions they measure first.
#[must_use]
pub fn sample_order(partitions: u32) -> Vec<u32> {
    if partitions <= 1 {
        return (0..partitions).collect();
    }
    let bits = u32::BITS - (partitions - 1).leading_zeros();
    (0..1u32 << bits)
        .map(|i| i.reverse_bits() >> (u32::BITS - bits))
        .filter(|&v| v < partitions)
        .collect()
}

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
/// version: (scenario key, gpu, reference rate). A score of 1000 means
/// "matches every reference rate on the geometric mean".
///
/// The references are arbitrary anchors, not a description of any machine:
/// they are re-pinned whenever the maintainers decide the scale has drifted,
/// which changes every score at once. Compare scores within a client version
/// only; the per-scenario rates in the report are what to use across
/// versions.
///
/// The browser suite scores against these same references deliberately: a
/// browser scoring 550 where the native client scores 1000 on the same box
/// is information, not a bug.
pub const SCORE_REFERENCES: &[(&str, bool, f64)] = &[
    ("b57_1e14", false, 1.4e12),
    ("b57_1e14_1t", false, 2.8e11),
    ("b58_1e15", false, 3.1e12),
    ("b40_detailed", false, 4.2e7),
    ("b50_detailed", false, 1.7e7),
    ("b50_detailed_1t", false, 2.8e6),
    ("b57_1e14", true, 2.5e13),
    ("b58_1e15", true, 5.6e13),
    ("b40_detailed", true, 2.4e9),
    ("b50_detailed", true, 1.5e9),
];

/// Geometric mean of measured rate over reference rate, scaled so matching
/// every reference exactly scores 1000. Scenarios without a pinned reference or
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
            .find(|(k, g, _)| *k == "b50_detailed" && !g)
            .unwrap()
            .2;
        // Exactly matching the reference on the only scored scenario = 1000;
        // an unknown key contributes nothing.
        let score = compute_score(
            [
                ("b50_detailed", reference_rate),
                ("not_a_real_scenario", 1.0),
            ],
            false,
        )
        .unwrap();
        assert!((score - 1000.0).abs() < 1e-6);
        // An unmeasured scenario contributes nothing either.
        assert_eq!(compute_score([("b50_detailed", 0.0)], false), None);
    }

    #[test]
    fn every_scenario_has_its_references() {
        // Every scenario scores on the CPU, and every one a GPU runs (all but
        // the single-thread ones) on the GPU too, or the score silently thins.
        let has = |key: &str, gpu: bool| {
            SCORE_REFERENCES
                .iter()
                .any(|(k, g, _)| *k == key && *g == gpu)
        };
        let scenarios: Vec<(&str, bool)> = DETAILED_SCENARIOS
            .iter()
            .map(|d| (d.key, d.single_thread))
            .chain(NICEONLY_FIELDS.iter().map(|d| (d.key, d.single_thread)))
            .collect();
        for &(key, single_thread) in &scenarios {
            assert!(has(key, false), "missing CPU score reference for {key}");
            assert!(
                single_thread || has(key, true),
                "missing GPU score reference for {key}"
            );
        }
        // No reference is left over from a scenario that is gone.
        for (key, _, _) in SCORE_REFERENCES {
            assert!(
                scenarios.iter().any(|(k, _)| k == key),
                "reference for unknown scenario {key}"
            );
        }
    }

    #[test]
    fn scenario_starts_resolve() {
        // `resolved_start` panics on a base with no range; catch a bad table
        // entry here instead of at a user's benchmark run.
        for def in DETAILED_SCENARIOS {
            let start = def.resolved_start();
            assert!(start > 0, "{} resolved to zero", def.key);
        }
    }

    #[test]
    fn field_scenarios_are_production_join_fields() {
        // Each field lies in its base's range and is one the overlap join
        // takes, so the scenario measures the production route.
        for def in NICEONLY_FIELDS {
            let range = get_base_range_u128(def.base).unwrap().unwrap();
            assert!(
                range.start() <= def.start && def.end() <= range.end(),
                "{} is outside base {}",
                def.key,
                def.base
            );
            let field = crate::FieldSize::new(def.start, def.end());
            assert!(
                crate::overlap_join::join_verdict_routed(
                    def.base,
                    &field,
                    crate::overlap_join::RouteOverride::Auto
                )
                .is_ok(),
                "{} does not take the overlap join",
                def.key
            );
            // The stride windows stay below the join's minimum, so they time
            // the stride path wherever the field takes it.
            for w in [def.stride_window_cpu, def.stride_window_gpu] {
                assert!(w < crate::overlap_join::JOIN_MIN_FIELD_SIZE, "{}", def.key);
            }
            // Each `_1t` scenario repeats a multi-thread one's field.
            if let Some(multi) = def.key.strip_suffix("_1t") {
                assert!(def.single_thread);
                let pair = NICEONLY_FIELDS.iter().find(|d| d.key == multi).unwrap();
                assert_eq!(
                    (pair.base, pair.start, pair.size),
                    (def.base, def.start, def.size)
                );
            }
        }
    }

    #[test]
    fn the_sample_order_spreads_every_prefix() {
        for partitions in [1u32, 2, 3, 57 * 57, 58 * 58, 64 * 64, 5000] {
            let order = sample_order(partitions);
            // A permutation of the partitions.
            let mut sorted = order.clone();
            sorted.sort_unstable();
            assert_eq!(sorted, (0..partitions).collect::<Vec<_>>());
            // Every prefix of 8 or more puts a share of its partitions in
            // each eighth of the space that stays near an eighth.
            for len in [8usize, 16, 64, 256]
                .into_iter()
                .filter(|&l| l <= order.len())
            {
                let mut eighths = [0usize; 8];
                for &v in &order[..len] {
                    eighths[usize::try_from(u64::from(v) * 8 / u64::from(partitions)).unwrap()] +=
                        1;
                }
                let want = len / 8;
                for (i, &n) in eighths.iter().enumerate() {
                    assert!(
                        n + 2 >= want && n <= want + 2,
                        "{partitions} partitions, prefix {len}: eighth {i} holds {n}"
                    );
                }
            }
        }
    }
}
