//! Performance estimation from uploaded benchmark reports.
//!
//! Predicts what a hardware configuration will achieve per benchmark
//! scenario, by hierarchical matching against stored reports rather than a
//! fitted model — this is a small-data regime (dozens of hardware combos),
//! where nearest-match with honest spreads beats regression. Every estimate
//! reports which fallback stage produced it alongside a confidence percent,
//! so consumers can gate on how the number was derived, not just its size.
//!
//! Matching stages, best first:
//!
//! GPU configurations: exact (same GPU and CPU model) → same GPU with a
//! similar CPU thread count → same GPU with any CPU → any GPU data at all.
//! GPU nice-only throughput is `min(gpu kernel rate, CPU MSD production)`,
//! so estimates get less trustworthy as the CPU diverges from the samples.
//!
//! CPU configurations: exact (same CPU model and thread count) → same CPU
//! model scaled to the requested thread count → same CPU family scaled →
//! any CPU data at all. Scaling uses the single-thread/multi-thread scenario
//! pair each benchmark carries: rates scale linearly in threads, capped by
//! the single-thread rate times the thread count (perfect scaling), with
//! confidence discounted by extrapolation distance.

use serde_json::Value;

/// One benchmark report decoded for matching.
#[derive(Debug, Clone)]
pub struct BenchmarkSample {
    pub client_version: String,
    pub gpu: bool,
    /// Mode string as reported (`SearchMode`'s Display form).
    pub mode: String,
    pub threads: u32,
    pub cpu_model: Option<String>,
    pub gpu_model: Option<String>,
    pub scenarios: Vec<ScenarioSample>,
}

#[derive(Debug, Clone)]
pub struct ScenarioSample {
    pub key: String,
    pub base: u32,
    pub threads: u32,
    pub rate: f64,
}

/// Decode a stored report (`benchmarks.data`) into a matching sample.
/// Returns `None` for reports that are malformed or from an unknown schema.
#[must_use]
pub fn decode_sample(client_version: &str, data: &Value) -> Option<BenchmarkSample> {
    if data.get("schema_version")?.as_u64()? != 1 {
        return None;
    }
    let config = data.get("config")?;
    let hardware = data.get("hardware")?;
    let scenarios = data
        .get("scenarios")?
        .as_array()?
        .iter()
        .filter_map(|s| {
            let rate = s.get("rate")?.as_f64()?;
            if rate <= 0.0 {
                return None;
            }
            Some(ScenarioSample {
                key: s.get("key")?.as_str()?.to_string(),
                base: u32::try_from(s.get("base")?.as_u64()?).ok()?,
                threads: u32::try_from(s.get("threads")?.as_u64()?).ok()?,
                rate,
            })
        })
        .collect::<Vec<_>>();
    if scenarios.is_empty() {
        return None;
    }
    Some(BenchmarkSample {
        client_version: client_version.to_string(),
        gpu: config.get("gpu")?.as_bool()?,
        mode: config.get("mode")?.as_str()?.to_string(),
        threads: u32::try_from(config.get("threads")?.as_u64()?).ok()?,
        cpu_model: hardware
            .get("cpu_model")
            .and_then(Value::as_str)
            .map(normalize_model),
        gpu_model: hardware
            .get("gpu_model")
            .and_then(Value::as_str)
            .map(normalize_model),
        scenarios,
    })
}

/// What the caller wants an estimate for.
#[derive(Debug, Clone)]
pub struct EstimateInput {
    pub gpu: bool,
    /// Mode string in the report's Display form (see `mode_string`).
    pub mode: String,
    pub threads: Option<u32>,
    pub cpu_model: Option<String>,
    pub gpu_model: Option<String>,
    /// Restrict scenario output and the blended index to one base.
    pub base: Option<u32>,
    pub client_version: Option<String>,
}

/// Map a user-supplied mode string to the Display form stored in reports.
#[must_use]
pub fn mode_string(raw: &str) -> Option<&'static str> {
    match raw.to_lowercase().replace('-', "").as_str() {
        "detailed" => Some("Detailed"),
        "niceonly" => Some("Nice-only"),
        _ => None,
    }
}

#[derive(Debug, Clone)]
pub struct ScenarioEstimate {
    pub key: String,
    pub base: u32,
    pub samples: usize,
    pub rate_p25: f64,
    pub rate_p50: f64,
    pub rate_p75: f64,
}

#[derive(Debug, Clone)]
pub struct EstimateOutcome {
    pub prediction_stage: &'static str,
    /// Rough trust percent; gate on the stage for structure, this for size.
    pub confidence: u32,
    pub samples_used: usize,
    pub versions_used: Vec<String>,
    pub scenarios: Vec<ScenarioEstimate>,
    /// Geometric mean of multi-thread scenario medians (and p25/p75),
    /// as a single ranking index. `None` when nothing matched.
    pub blended_rate_p25: Option<f64>,
    pub blended_rate_p50: Option<f64>,
    pub blended_rate_p75: Option<f64>,
    pub notes: Vec<String>,
}

/// Normalize a hardware model string for matching: lowercase, trademark
/// glyphs and parenthesized marks removed, whitespace collapsed.
#[must_use]
pub fn normalize_model(raw: &str) -> String {
    raw.to_lowercase()
        .replace(['®', '™'], " ")
        .replace("(r)", " ")
        .replace("(tm)", " ")
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

/// A coarse CPU family key: the first two normalized tokens
/// ("amd epyc", "intel xeon", "raspberry pi", ...).
fn family_key(normalized_model: &str) -> String {
    normalized_model
        .split_whitespace()
        .take(2)
        .collect::<Vec<_>>()
        .join(" ")
}

/// Interpolated quantile of a sorted slice.
fn quantile(sorted: &[f64], frac: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    #[allow(clippy::cast_precision_loss)]
    let pos = frac * (sorted.len() - 1) as f64;
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let lo = pos.floor() as usize;
    let hi = (lo + 1).min(sorted.len() - 1);
    let t = pos - pos.floor();
    sorted[lo] * (1.0 - t) + sorted[hi] * t
}

/// The per-sample rate multiplier that scales its multi-thread scenarios to
/// `target` threads. Anchored on the sample's single-thread/multi-thread
/// scenario pair: linear in threads, capped by perfect scaling from the
/// single-thread rate.
fn thread_scale(sample: &BenchmarkSample, target: u32) -> Option<f64> {
    let anchor_multi = sample
        .scenarios
        .iter()
        .find(|s| !s.key.ends_with("_1t") && s.key.starts_with("b50"))?;
    let anchor_single = sample
        .scenarios
        .iter()
        .find(|s| s.key.ends_with("_1t"))?;
    let source = f64::from(sample.threads.max(1));
    let target_f = f64::from(target.max(1));
    let linear = target_f / source;
    // Perfect scaling ceiling: the single-thread rate times the target
    // thread count, expressed as a multiplier on the measured multi rate.
    let ceiling = (anchor_single.rate * target_f) / anchor_multi.rate;
    Some(linear.min(ceiling).max(0.01))
}

struct StageMatch<'a> {
    stage: &'static str,
    confidence: u32,
    /// (sample, rate multiplier applied to its multi-thread scenarios)
    rows: Vec<(&'a BenchmarkSample, f64)>,
    /// Interquartile widening for lower-trust stages.
    spread_widening: f64,
    notes: Vec<String>,
}

#[allow(clippy::too_many_lines, clippy::similar_names)]
fn match_stage<'a>(samples: &'a [BenchmarkSample], input: &EstimateInput) -> StageMatch<'a> {
    let pool: Vec<&BenchmarkSample> = samples
        .iter()
        .filter(|s| s.gpu == input.gpu && s.mode == input.mode)
        .collect();
    if pool.is_empty() {
        return StageMatch {
            stage: "none",
            confidence: 0,
            rows: vec![],
            spread_widening: 1.0,
            notes: vec!["no benchmark data for this mode/device class".to_string()],
        };
    }

    let want_cpu = input.cpu_model.as_deref().map(normalize_model);
    let want_gpu = input.gpu_model.as_deref().map(normalize_model);

    if input.gpu {
        if let Some(want_gpu) = &want_gpu {
            let same_gpu: Vec<&BenchmarkSample> = pool
                .iter()
                .copied()
                .filter(|s| s.gpu_model.as_deref() == Some(want_gpu.as_str()))
                .collect();
            if !same_gpu.is_empty() {
                if let Some(want_cpu) = &want_cpu {
                    let exact: Vec<_> = same_gpu
                        .iter()
                        .copied()
                        .filter(|s| s.cpu_model.as_deref() == Some(want_cpu.as_str()))
                        .map(|s| (s, 1.0))
                        .collect();
                    if !exact.is_empty() {
                        return StageMatch {
                            stage: "exact",
                            confidence: 85,
                            rows: exact,
                            spread_widening: 1.0,
                            notes: vec![],
                        };
                    }
                }
                if let Some(threads) = input.threads {
                    let similar: Vec<_> = same_gpu
                        .iter()
                        .copied()
                        .filter(|s| {
                            let ratio = f64::from(s.threads.max(1)) / f64::from(threads.max(1));
                            (0.5..=2.0).contains(&ratio)
                        })
                        .map(|s| (s, 1.0))
                        .collect();
                    if !similar.is_empty() {
                        return StageMatch {
                            stage: "same-gpu-similar-cpu",
                            confidence: 65,
                            rows: similar,
                            spread_widening: 1.5,
                            notes: vec![
                                "GPU nice-only throughput is bounded by min(GPU kernel, CPU MSD \
                                 production); CPU details differ from samples"
                                    .to_string(),
                            ],
                        };
                    }
                }
                return StageMatch {
                    stage: "same-gpu",
                    confidence: 50,
                    rows: same_gpu.into_iter().map(|s| (s, 1.0)).collect(),
                    spread_widening: 2.0,
                    notes: vec![
                        "samples share the GPU but not the CPU; the CPU side (MSD production) \
                         may bind for nice-only"
                            .to_string(),
                    ],
                };
            }
        }
        return StageMatch {
            stage: "floor",
            confidence: 15,
            rows: pool.into_iter().map(|s| (s, 1.0)).collect(),
            spread_widening: 3.0,
            notes: vec!["no samples for this GPU model; using all GPU data".to_string()],
        };
    }

    // CPU chain.
    if let Some(want_cpu) = &want_cpu {
        let same_cpu: Vec<&BenchmarkSample> = pool
            .iter()
            .copied()
            .filter(|s| s.cpu_model.as_deref() == Some(want_cpu.as_str()))
            .collect();
        if !same_cpu.is_empty() {
            if let Some(threads) = input.threads {
                let exact: Vec<_> = same_cpu
                    .iter()
                    .copied()
                    .filter(|s| s.threads == threads)
                    .map(|s| (s, 1.0))
                    .collect();
                if !exact.is_empty() {
                    return StageMatch {
                        stage: "exact",
                        confidence: 85,
                        rows: exact,
                        spread_widening: 1.0,
                        notes: vec![],
                    };
                }
                return scaled_stage(&same_cpu, threads, "same-cpu-scaled", 60, 1.5);
            }
            return StageMatch {
                stage: "same-cpu-scaled",
                confidence: 45,
                rows: same_cpu.into_iter().map(|s| (s, 1.0)).collect(),
                spread_widening: 1.5,
                notes: vec!["no target thread count given; rates unscaled".to_string()],
            };
        }
        let family = family_key(want_cpu);
        let same_family: Vec<&BenchmarkSample> = pool
            .iter()
            .copied()
            .filter(|s| {
                s.cpu_model
                    .as_deref()
                    .is_some_and(|m| family_key(m) == family)
            })
            .collect();
        if !same_family.is_empty() {
            if let Some(threads) = input.threads {
                return scaled_stage(&same_family, threads, "cpu-family-scaled", 40, 2.0);
            }
            return StageMatch {
                stage: "cpu-family-scaled",
                confidence: 30,
                rows: same_family.into_iter().map(|s| (s, 1.0)).collect(),
                spread_widening: 2.0,
                notes: vec!["no target thread count given; rates unscaled".to_string()],
            };
        }
    }
    let rows: Vec<_> = pool.iter().copied().map(|s| (s, 1.0)).collect();
    StageMatch {
        stage: "floor",
        confidence: 15,
        rows,
        spread_widening: 3.0,
        notes: vec!["no samples for this CPU model or family; using all CPU data".to_string()],
    }
}

/// Build a thread-scaled stage, discounting confidence by how far the
/// requested thread count extrapolates beyond the measured samples.
fn scaled_stage<'a>(
    candidates: &[&'a BenchmarkSample],
    threads: u32,
    stage: &'static str,
    base_confidence: u32,
    widening: f64,
) -> StageMatch<'a> {
    let mut rows = Vec::new();
    let mut max_distance = 0.0f64;
    let mut missing_anchor = false;
    for s in candidates {
        if let Some(scale) = thread_scale(s, threads) {
            let distance = (f64::from(threads.max(1)) / f64::from(s.threads.max(1)))
                .log2()
                .abs();
            max_distance = max_distance.max(distance);
            rows.push((*s, scale));
        } else {
            missing_anchor = true;
        }
    }
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let penalty = (max_distance * 15.0).min(30.0) as u32;
    let mut notes = vec![format!(
        "rates scaled to {threads} threads from measured configurations"
    )];
    if missing_anchor {
        notes.push("some samples lacked a single-thread anchor and were skipped".to_string());
    }
    StageMatch {
        stage,
        confidence: base_confidence.saturating_sub(penalty).max(5),
        rows,
        spread_widening: widening,
        notes,
    }
}

/// Produce the estimate. `samples` should be recent decoded reports; order
/// does not matter.
#[must_use]
pub fn estimate(samples: &[BenchmarkSample], input: &EstimateInput) -> EstimateOutcome {
    let mut matched = match_stage(samples, input);

    // Version awareness: report which client versions back the estimate and
    // discount if the requested version has no representation.
    let mut versions_used: Vec<String> = matched
        .rows
        .iter()
        .map(|(s, _)| s.client_version.clone())
        .collect();
    versions_used.sort();
    versions_used.dedup();
    if let Some(want) = &input.client_version
        && !matched.rows.is_empty()
        && !versions_used.contains(want)
    {
        matched.confidence = matched.confidence.saturating_sub(15).max(5);
        matched.notes.push(format!(
            "no samples from client version {want}; estimates come from {versions_used:?}"
        ));
    }

    // Group scaled rates by scenario key.
    let mut by_key: Vec<(String, u32, Vec<f64>)> = Vec::new();
    for (sample, scale) in &matched.rows {
        for scenario in &sample.scenarios {
            if let Some(base) = input.base
                && scenario.base != base
            {
                continue;
            }
            let effective = if scenario.key.ends_with("_1t") {
                scenario.rate
            } else {
                scenario.rate * scale
            };
            match by_key.iter_mut().find(|(k, _, _)| *k == scenario.key) {
                Some((_, _, rates)) => rates.push(effective),
                None => by_key.push((scenario.key.clone(), scenario.base, vec![effective])),
            }
        }
    }

    let mut scenarios = Vec::new();
    for (key, base, mut rates) in by_key {
        rates.sort_by(f64::total_cmp);
        let p50 = quantile(&rates, 0.5);
        let iqr_low = p50 - quantile(&rates, 0.25);
        let iqr_high = quantile(&rates, 0.75) - p50;
        scenarios.push(ScenarioEstimate {
            key,
            base,
            samples: rates.len(),
            rate_p25: (p50 - iqr_low * matched.spread_widening).max(p50 * 0.05),
            rate_p50: p50,
            rate_p75: p50 + iqr_high * matched.spread_widening,
        });
    }
    scenarios.sort_by(|a, b| a.key.cmp(&b.key));

    // Blended ranking index: geometric mean over multi-thread scenarios.
    let blend = |pick: fn(&ScenarioEstimate) -> f64| -> Option<f64> {
        let logs: Vec<f64> = scenarios
            .iter()
            .filter(|s| !s.key.ends_with("_1t") && pick(s) > 0.0)
            .map(|s| pick(s).ln())
            .collect();
        #[allow(clippy::cast_precision_loss)]
        (!logs.is_empty()).then(|| (logs.iter().sum::<f64>() / logs.len() as f64).exp())
    };

    EstimateOutcome {
        prediction_stage: matched.stage,
        confidence: matched.confidence,
        samples_used: matched.rows.len(),
        versions_used,
        blended_rate_p25: blend(|s| s.rate_p25),
        blended_rate_p50: blend(|s| s.rate_p50),
        blended_rate_p75: blend(|s| s.rate_p75),
        scenarios,
        notes: matched.notes,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn sample(
        gpu: bool,
        cpu_model: &str,
        gpu_model: Option<&str>,
        threads: u32,
        multi_rate: f64,
        single_rate: f64,
    ) -> BenchmarkSample {
        BenchmarkSample {
            client_version: "3.4.0".to_string(),
            gpu,
            mode: "Nice-only".to_string(),
            threads,
            cpu_model: Some(normalize_model(cpu_model)),
            gpu_model: gpu_model.map(normalize_model),
            scenarios: vec![
                ScenarioSample {
                    key: "b50_msd_weak".to_string(),
                    base: 50,
                    threads,
                    rate: multi_rate,
                },
                ScenarioSample {
                    key: "b50_msd_weak_1t".to_string(),
                    base: 50,
                    threads: 1,
                    rate: single_rate,
                },
            ],
        }
    }

    fn input(gpu: bool, cpu: Option<&str>, gpu_model: Option<&str>, threads: Option<u32>) -> EstimateInput {
        EstimateInput {
            gpu,
            mode: "Nice-only".to_string(),
            threads,
            cpu_model: cpu.map(String::from),
            gpu_model: gpu_model.map(String::from),
            base: None,
            client_version: None,
        }
    }

    #[test]
    fn model_normalization_and_family() {
        assert_eq!(
            normalize_model("Intel(R) Xeon(R) CPU E5-2673 v3 @ 2.40GHz"),
            "intel xeon cpu e5-2673 v3 @ 2.40ghz"
        );
        assert_eq!(
            family_key(&normalize_model("AMD EPYC 7763 64-Core Processor")),
            "amd epyc"
        );
    }

    #[test]
    fn exact_cpu_match() {
        let samples = vec![sample(false, "AMD EPYC 7763", None, 8, 2.0e9, 3.0e8)];
        let out = estimate(&samples, &input(false, Some("AMD EPYC 7763"), None, Some(8)));
        assert_eq!(out.prediction_stage, "exact");
        assert_eq!(out.confidence, 85);
        let multi = out.scenarios.iter().find(|s| s.key == "b50_msd_weak").unwrap();
        assert!((multi.rate_p50 - 2.0e9).abs() < 1.0);
    }

    #[test]
    fn same_cpu_scales_by_threads_with_ceiling() {
        // 8 threads measured at 2e9; asking for 16 doubles linearly (3e8
        // single-thread rate x 16 = 4.8e9 ceiling doesn't bind).
        let samples = vec![sample(false, "AMD EPYC 7763", None, 8, 2.0e9, 3.0e8)];
        let out = estimate(&samples, &input(false, Some("AMD EPYC 7763"), None, Some(16)));
        assert_eq!(out.prediction_stage, "same-cpu-scaled");
        let multi = out.scenarios.iter().find(|s| s.key == "b50_msd_weak").unwrap();
        assert!((multi.rate_p50 - 4.0e9).abs() < 1.0);

        // Asking for 64: linear would be 16e9 but the single-thread ceiling
        // (3e8 x 64 = 19.2e9) still doesn't bind; make the ceiling bind with
        // a low single-thread rate.
        let samples = vec![sample(false, "AMD EPYC 7763", None, 8, 2.0e9, 1.0e8)];
        let out = estimate(&samples, &input(false, Some("AMD EPYC 7763"), None, Some(64)));
        let multi = out.scenarios.iter().find(|s| s.key == "b50_msd_weak").unwrap();
        // ceiling: 1e8 x 64 = 6.4e9 < linear 16e9
        assert!((multi.rate_p50 - 6.4e9).abs() < 1.0);
        assert!(out.confidence < 60, "extrapolation must discount confidence");
    }

    #[test]
    fn family_fallback_and_floor() {
        let samples = vec![sample(false, "AMD EPYC 7763", None, 8, 2.0e9, 3.0e8)];
        let out = estimate(&samples, &input(false, Some("AMD EPYC 9654"), None, Some(8)));
        assert_eq!(out.prediction_stage, "cpu-family-scaled");
        let out = estimate(&samples, &input(false, Some("Raspberry Pi 5"), None, Some(4)));
        assert_eq!(out.prediction_stage, "floor");
        assert_eq!(out.confidence, 15);
    }

    #[test]
    fn gpu_chain() {
        let samples = vec![sample(true, "Intel Xeon E5", Some("NVIDIA GeForce RTX 3060"), 4, 1.3e11, 0.0)];
        // exact
        let out = estimate(
            &samples,
            &input(true, Some("Intel Xeon E5"), Some("NVIDIA GeForce RTX 3060"), None),
        );
        assert_eq!(out.prediction_stage, "exact");
        // same gpu, different cpu
        let out = estimate(
            &samples,
            &input(true, Some("AMD Ryzen 9"), Some("NVIDIA GeForce RTX 3060"), None),
        );
        assert_eq!(out.prediction_stage, "same-gpu");
        assert!(out.notes.iter().any(|n| n.contains("MSD")));
        // unknown gpu
        let out = estimate(&samples, &input(true, None, Some("NVIDIA H100"), None));
        assert_eq!(out.prediction_stage, "floor");
        // wrong device class entirely
        let out = estimate(&samples, &input(false, Some("Intel Xeon E5"), None, Some(4)));
        assert_eq!(out.prediction_stage, "none");
        assert_eq!(out.confidence, 0);
    }

    #[test]
    fn version_mismatch_discounts() {
        let samples = vec![sample(false, "AMD EPYC 7763", None, 8, 2.0e9, 3.0e8)];
        let mut req = input(false, Some("AMD EPYC 7763"), None, Some(8));
        req.client_version = Some("9.9.9".to_string());
        let out = estimate(&samples, &req);
        assert_eq!(out.prediction_stage, "exact");
        assert_eq!(out.confidence, 70);
        assert!(out.notes.iter().any(|n| n.contains("9.9.9")));
    }

    #[test]
    fn decode_rejects_bad_schema() {
        assert!(decode_sample("x", &json!({"schema_version": 2})).is_none());
        let good = json!({
            "schema_version": 1,
            "config": {"gpu": false, "mode": "Nice-only", "threads": 4},
            "hardware": {"cpu_model": "AMD EPYC 7763"},
            "scenarios": [
                {"key": "b50_msd_weak", "base": 50, "threads": 4, "rate": 1.0e9}
            ],
        });
        let s = decode_sample("3.4.0", &good).unwrap();
        assert_eq!(s.scenarios.len(), 1);
        assert_eq!(s.cpu_model.as_deref(), Some("amd epyc 7763"));
    }

    #[test]
    fn mode_strings() {
        assert_eq!(mode_string("niceonly"), Some("Nice-only"));
        assert_eq!(mode_string("Nice-Only"), Some("Nice-only"));
        assert_eq!(mode_string("detailed"), Some("Detailed"));
        assert_eq!(mode_string("bogus"), None);
    }
}
