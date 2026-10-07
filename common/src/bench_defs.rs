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

use crate::FieldSize;
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
    /// The field: one on the server's grid of fields for the base
    /// (`range_start + k·size` at the base's field size), or a window of one.
    pub start: u128,
    pub size: u128,
    /// How much of the field, from its start, the CPU times. The CPU's
    /// sample is at least two partitions a thread, and a partition of a 1e16
    /// field takes seconds, so the larger fields time a leading part chosen
    /// to have the field's density.
    pub cpu_size: u128,
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
    /// The part of the field a device times: all of it on a GPU, the
    /// leading `cpu_size` on the CPU.
    #[must_use]
    pub fn measured(&self, gpu: bool) -> FieldSize {
        let size = if gpu { self.size } else { self.cpu_size };
        FieldSize::new(self.start, self.start + size)
    }
}

/// The nice-only fields, at the bases searched next, in their production
/// sizes. The overlap join's work in a field is its top layer, the prefixes
/// the MSD certificates cannot reject: an `msd-weak` field is dense with
/// them, the join's most work per number, and the `msd-strong` one is
/// mostly rejected, so its time is mostly the join's fixed work per
/// partition and the host's setup.
pub const NICEONLY_FIELDS: &[FieldScenario] = &[
    // A dense b58 field (grid index 7680, density 0.45).
    FieldScenario {
        key: "b58_msd_weak",
        base: 58,
        start: 104_400_216_587_923_470_200,
        size: 1_000_000_000_000_000,
        cpu_size: 1_000_000_000_000_000,
        character: "msd-weak",
        single_thread: false,
        stride_window_cpu: 2_000_000_000,
        stride_window_gpu: 1_000_000_000_000,
    },
    // The same field on one thread, so the pair decomposes into per-core
    // rate × parallel efficiency.
    FieldScenario {
        key: "b58_msd_weak_1t",
        base: 58,
        start: 104_400_216_587_923_470_200,
        size: 1_000_000_000_000_000,
        cpu_size: 1_000_000_000_000_000,
        character: "msd-weak",
        single_thread: true,
        stride_window_cpu: 2_000_000_000,
        stride_window_gpu: 0,
    },
    // A b58 field (grid index 1592) whose first 8e14 the MSD certificates
    // reject outright (density 0.056). On the stride path its windows time
    // that rejection, as the old msd-strong windows did.
    FieldScenario {
        key: "b58_msd_strong",
        base: 58,
        start: 98_312_216_587_923_470_200,
        size: 1_000_000_000_000_000,
        cpu_size: 1_000_000_000_000_000,
        character: "msd-strong",
        single_thread: false,
        stride_window_cpu: 2_000_000_000,
        stride_window_gpu: 1_000_000_000_000,
    },
    // A dense b60 field (density 0.41) whose leading part has its density,
    // which the GPU's sample of the first slice and the CPU's first 3e15
    // rely on. It starts 2e13 into grid field 60118, past a pocket the
    // stride path's MSD filter rejects outright, so a stride window at its
    // start times dense work, as `msd_weak` says.
    FieldScenario {
        key: "b60_msd_weak",
        base: 60,
        start: 1_157_209_632_114_824_200_908,
        size: 10_000_000_000_000_000,
        cpu_size: 3_000_000_000_000_000,
        character: "msd-weak",
        single_thread: false,
        stride_window_cpu: 2_000_000_000,
        stride_window_gpu: 1_000_000_000_000,
    },
    // A dense 1e16 window of a b62 field (the last 1e16 of grid index
    // 12468 of 1e17, density 0.34). No 1e17 field there starts with its own
    // density, as a sample of the first slice needs; the window is still
    // several slices on a GPU, so it carries a 1e17 field's per-slice cost.
    FieldScenario {
        key: "b62_msd_weak",
        base: 62,
        start: 4_473_156_762_334_864_396_992,
        size: 10_000_000_000_000_000,
        cpu_size: 3_000_000_000_000_000,
        character: "msd-weak",
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
    let bits = (partitions - 1).bit_width();
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
    ("b58_msd_weak", false, 2.6e12),
    ("b58_msd_weak_1t", false, 4.9e11),
    ("b58_msd_strong", false, 1.5e13),
    ("b60_msd_weak", false, 6.1e12),
    ("b62_msd_weak", false, 7.5e12),
    ("b40_detailed", false, 4.2e7),
    ("b50_detailed", false, 1.7e7),
    ("b50_detailed_1t", false, 2.8e6),
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
        // Every scenario scores on the CPU, and every detailed one a GPU runs
        // (all but the single-thread ones) on the GPU too, or the score
        // silently thins. The nice-only fields' GPU references wait for
        // sweeps with the field time as it now stands.
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
        for &(key, _) in &scenarios {
            assert!(has(key, false), "missing CPU score reference for {key}");
        }
        for def in DETAILED_SCENARIOS {
            assert!(
                def.single_thread || has(def.key, true),
                "missing GPU score reference for {}",
                def.key
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
        // Each field lies in its base's range, and the overlap join takes
        // both what the GPU times and what the CPU times, so the scenario
        // measures the production route.
        for def in NICEONLY_FIELDS {
            let range = get_base_range_u128(def.base).unwrap().unwrap();
            let field = def.measured(true);
            assert!(
                range.start() <= field.start() && field.end() <= range.end(),
                "{} is outside base {}",
                def.key,
                def.base
            );
            assert!(def.cpu_size <= def.size, "{}", def.key);
            for part in [field, def.measured(false)] {
                assert!(
                    crate::overlap_join::join_verdict_routed(
                        def.base,
                        &part,
                        crate::overlap_join::RouteOverride::Auto
                    )
                    .is_ok(),
                    "{} {part:?} does not take the overlap join",
                    def.key
                );
            }
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
                    (pair.base, pair.start, pair.size, pair.cpu_size),
                    (def.base, def.start, def.size, def.cpu_size)
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
