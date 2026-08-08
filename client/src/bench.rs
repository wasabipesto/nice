//! Structured benchmark sweep.
//!
//! Replaces the old single-field benchmark modes with an adaptive sweep that
//! measures the configuration the user would actually run (mode, threads,
//! GPU) across scenarios chosen to span the live search space: bases with
//! different stride densities and window regions where the MSD filter is
//! strong or weak. Each scenario is calibrated with a short run, then sized
//! so the whole sweep fits the `--benchmark-secs` budget.
//!
//! Also measures API latency against the lightweight `/ping` endpoint
//! (spread before and after the sweep), collects hardware and scheduler
//! environment info for cross-correlation, and prints both a human-readable
//! table and a complete machine-readable JSON report. The synthetic
//! `NiceMark` score at the end is for bragging rights only — it is a
//! geometric mean against reference rates pinned per client version and is
//! never used for real analysis.

use crate::{Cli, DEFAULT_LSD_K_VALUE, GpuCtx, process_field_sync};
use log::debug;
use nice_common::base_range::get_base_range_u128;
use nice_common::client_api_async::Client;
use nice_common::stride_filter::StrideTable;
use nice_common::{BenchmarkToServer, CLIENT_VERSION, DataToClient, SearchMode};
use std::io::{IsTerminal, Write};
use serde_json::{Value, json};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

/// Version of the JSON report layout. Bump on breaking changes.
pub const BENCH_SCHEMA_VERSION: u32 = 1;

/// Version of the per-submission telemetry layout. Bump on breaking changes.
pub const TELEMETRY_SCHEMA_VERSION: u32 = 1;

/// Environment variables worth attaching to a report for cross-correlating
/// runs with rented instances or cluster jobs. Allowlist only — nothing
/// account-related is ever collected.
const ENV_ALLOWLIST: &[&str] = &[
    "VAST_CONTAINERLABEL",
    "CONTAINER_ID",
    "SLURM_JOB_ID",
    "SLURM_CLUSTER_NAME",
    "SLURM_JOB_PARTITION",
];

/// A fixed measurement region. Both the start and the window length are
/// hardcoded so every machine measures *identical work*; machine speed only
/// changes how many repetitions fit in the scenario's time share. Repetition
/// also solves timer granularity — a machine that clears the window in
/// microseconds simply runs it thousands of times.
struct ScenarioDef {
    key: &'static str,
    base: u32,
    /// None = the base range start (a strongly MSD-filtered region).
    start: Option<u128>,
    /// Fixed window length for CPU runs; sized so one repetition stays
    /// tractable on very slow devices (a Raspberry Pi class machine should
    /// clear it within roughly a scenario share).
    window_cpu: u128,
    /// Fixed window length for GPU runs; sized so one repetition amortizes
    /// launch overhead on data-center class devices.
    window_gpu: u128,
    /// Rough character of the region, for human readers of the report.
    character: &'static str,
    /// Run with a single thread instead of the configured thread count.
    /// One such scenario per sweep lets analysis decompose full-thread
    /// results into per-core rate × parallel efficiency.
    single_thread: bool,
}

const NICEONLY_SCENARIOS: &[ScenarioDef] = &[
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

const DETAILED_SCENARIOS: &[ScenarioDef] = &[
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
const SCORE_REFERENCES: &[(&str, bool, f64)] = &[
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

/// Prebuilt stride tables per base, so table construction is paid once per
/// base outside the timed windows instead of once per timed call. Recorded
/// build times are themselves useful data on slow devices.
struct TableCache {
    tables: HashMap<u32, Arc<StrideTable>>,
    build_secs: HashMap<u32, f64>,
}

impl TableCache {
    fn get(&mut self, mode: SearchMode, base: u32) -> Option<Arc<StrideTable>> {
        if mode != SearchMode::Niceonly {
            return None;
        }
        if let Some(table) = self.tables.get(&base) {
            return Some(Arc::clone(table));
        }
        let t0 = Instant::now();
        let table = Arc::new(StrideTable::new(base, DEFAULT_LSD_K_VALUE));
        self.build_secs.insert(base, t0.elapsed().as_secs_f64());
        self.tables.insert(base, Arc::clone(&table));
        Some(table)
    }
}

/// Result of one scenario, or the reason it was skipped.
struct ScenarioResult {
    key: &'static str,
    base: u32,
    character: &'static str,
    threads: usize,
    window_start: u128,
    window_size: u128,
    repetitions: u32,
    seconds: f64,
    rate: f64,
    warmup_seconds: f64,
}

/// Run the full sweep and print the report. Never contacts the server except
/// for `/ping` latency samples, which are skipped gracefully offline.
pub async fn run_benchmark_sweep(cli: &Arc<Cli>, gpu: &GpuCtx, client: &Client) {
    println!(
        "Nice benchmark sweep: v{CLIENT_VERSION}, {} mode, {}, {:.1}s budget",
        cli.mode,
        if cli.gpu {
            format!("GPU device {}", cli.gpu_device)
        } else {
            format!("CPU {} threads", cli.threads)
        },
        cli.benchmark_secs,
    );

    let ping_before = ping_samples(client, &cli.api_base, 5).await;

    let sweep_cli = Arc::clone(cli);
    let sweep_gpu = gpu.clone();
    let (results, table_build_secs) =
        tokio::task::spawn_blocking(move || run_sweep(&sweep_cli, &sweep_gpu))
            .await
            .expect("benchmark sweep panicked");

    let ping_after = ping_samples(client, &cli.api_base, 5).await;

    let hardware = collect_hardware(cli, gpu);
    let environment = collect_environment();
    let score = compute_score(&results, cli.gpu);

    print_report(cli, &results, &ping_before, &ping_after, score);

    let report = build_report_json(
        cli,
        &results,
        &ping_before,
        &ping_after,
        &hardware,
        &environment,
        score,
        &table_build_secs,
    );
    println!("\n--- benchmark report (json) ---");
    println!("{}", serde_json::to_string_pretty(&report).unwrap());
    println!();

    match decide_upload(cli.benchmark_upload, std::io::stdin().is_terminal()) {
        UploadDecision::Yes => upload_report(client, cli, &report).await,
        UploadDecision::Prompt => {
            if prompt_yes(&cli.api_base) {
                upload_report(client, cli, &report).await;
            } else {
                println!("Not uploaded.");
            }
        }
        UploadDecision::No => {
            println!("Not uploading (non-interactive; pass --benchmark-upload to upload).");
        }
    }
}

/// The blocking part: calibrate and run every scenario within the budget.
fn run_sweep(cli: &Arc<Cli>, gpu: &GpuCtx) -> (Vec<ScenarioResult>, HashMap<u32, f64>) {
    // The progress bar is noise at benchmark window sizes.
    let mut quiet = (**cli).clone();
    quiet.no_progress = true;
    let quiet = Arc::new(quiet);

    let defs: Vec<&ScenarioDef> = match cli.mode {
        SearchMode::Niceonly => NICEONLY_SCENARIOS,
        SearchMode::Detailed => DETAILED_SCENARIOS,
    }
    .iter()
    // Single-thread scenarios decompose CPU scaling; they mean nothing for
    // the GPU pipeline.
    .filter(|d| !(cli.gpu && d.single_thread))
    .collect();

    #[allow(clippy::cast_precision_loss)]
    let share = cli.benchmark_secs / defs.len() as f64;

    let mut cache = TableCache {
        tables: HashMap::new(),
        build_secs: HashMap::new(),
    };
    let results = defs
        .iter()
        .map(|def| run_scenario(&quiet, gpu, def, share, &mut cache))
        .collect();
    (results, cache.build_secs)
}

/// Run one scenario: warm up, then repeat the fixed window until the
/// scenario's share of the time budget is spent (always at least once).
fn run_scenario(
    cli: &Arc<Cli>,
    gpu: &GpuCtx,
    def: &ScenarioDef,
    share_secs: f64,
    cache: &mut TableCache,
) -> ScenarioResult {
    let threads = if def.single_thread { 1 } else { cli.threads };
    // Multi-thread CPU runs parallelize over 1e6-number chunks, so the window
    // must hold at least a couple of chunks per thread or high-core machines
    // measure their own starvation. This is the one place window length
    // varies by machine: cross-machine comparisons should use the
    // single-thread scenarios (fixed region and length); the multi-thread
    // scenarios measure what this configuration actually achieves.
    let window = if cli.gpu {
        def.window_gpu
    } else if def.single_thread {
        def.window_cpu
    } else {
        def.window_cpu.max(2_000_000 * threads as u128)
    };
    let base_range = get_base_range_u128(def.base)
        .expect("benchmark base must be valid")
        .expect("benchmark base must have a range");
    let start = def.start.unwrap_or_else(|| base_range.start());

    // Build the stride table outside the timed windows.
    let table = cache.get(cli.mode, def.base);

    // One untimed warmup so one-time costs (GPU kernel JIT for this base,
    // thread pool spin-up, cold caches) land outside the measurement. The
    // GPU needs the full window to reach the device path; the CPU warms up
    // on a fraction so slow devices don't pay the window twice.
    let warmup_window = if cli.gpu { window } else { (window / 8).max(1) };
    let warmup_t0 = Instant::now();
    run_window(cli, gpu, def, start, warmup_window, table.as_ref());
    let warmup_seconds = warmup_t0.elapsed().as_secs_f64();

    let scenario_start = Instant::now();
    let mut repetitions = 0u32;
    let mut total_secs = 0.0f64;
    loop {
        total_secs += run_window(cli, gpu, def, start, window, table.as_ref());
        repetitions += 1;
        if scenario_start.elapsed().as_secs_f64() >= share_secs * 0.9 {
            break;
        }
    }

    #[allow(clippy::cast_precision_loss)]
    let rate = approx_f64(window) * f64::from(repetitions) / total_secs.max(1e-4);
    ScenarioResult {
        key: def.key,
        base: def.base,
        character: def.character,
        threads,
        window_start: start,
        window_size: window,
        repetitions,
        seconds: total_secs,
        rate,
        warmup_seconds,
    }
}

/// Process one window through the production path and return elapsed seconds.
fn run_window(
    cli: &Arc<Cli>,
    gpu: &GpuCtx,
    def: &ScenarioDef,
    start: u128,
    window: u128,
    table: Option<&Arc<StrideTable>>,
) -> f64 {
    let claim = DataToClient {
        claim_id: 0,
        base: def.base,
        range_start: start,
        range_end: start + window,
        range_size: window,
    };
    let t0 = Instant::now();
    if def.single_thread {
        // A local one-thread pool overrides the global pool inside `install`.
        rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .expect("single-thread pool")
            .install(|| {
                process_field_sync(&claim, cli, gpu, table);
            });
    } else {
        process_field_sync(&claim, cli, gpu, table);
    }
    t0.elapsed().as_secs_f64()
}

/// Sample `GET /ping` latency. Errors (offline, endpoint not deployed yet)
/// come back as `None` and are reported as skipped.
async fn ping_samples(client: &Client, api_base: &str, n: usize) -> Vec<Option<f64>> {
    let url = format!("{api_base}/ping");
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        if i > 0 {
            tokio::time::sleep(Duration::from_millis(100)).await;
        }
        let t0 = Instant::now();
        let ok = matches!(
            client.get(&url).send().await,
            Ok(resp) if resp.status().is_success()
        );
        out.push(ok.then(|| t0.elapsed().as_secs_f64() * 1000.0));
    }
    out
}

#[allow(clippy::cast_precision_loss)]
fn approx_f64(n: u128) -> f64 {
    n as f64
}

fn median(mut values: Vec<f64>) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    values.sort_by(f64::total_cmp);
    Some(values[values.len() / 2])
}

/// Geometric mean of measured rate over reference rate, scaled so the
/// reference machine scores 1000. Scenarios without a pinned reference or
/// that were dropped are excluded.
fn compute_score(results: &[ScenarioResult], gpu: bool) -> Option<f64> {
    let mut log_sum = 0.0;
    let mut count = 0usize;
    for r in results {
        if r.rate <= 0.0 {
            continue;
        }
        let Some((_, _, reference)) = SCORE_REFERENCES
            .iter()
            .find(|(key, is_gpu, _)| *key == r.key && *is_gpu == gpu)
        else {
            continue;
        };
        log_sum += (r.rate / reference).ln();
        count += 1;
    }
    #[allow(clippy::cast_precision_loss)]
    (count > 0).then(|| 1000.0 * (log_sum / count as f64).exp())
}

fn print_report(
    cli: &Cli,
    results: &[ScenarioResult],
    ping_before: &[Option<f64>],
    ping_after: &[Option<f64>],
    score: Option<f64>,
) {
    println!();
    println!(
        "{:<20} {:>4} {:<14} {:>7} {:>10} {:>6} {:>8} {:>12}",
        "scenario", "base", "character", "threads", "window", "reps", "secs", "numbers/sec"
    );
    for r in results {
        println!(
            "{:<20} {:>4} {:<14} {:>7} {:>10.1e} {:>6} {:>8.3} {:>12.3e}",
            r.key,
            r.base,
            r.character,
            r.threads,
            approx_f64(r.window_size),
            r.repetitions,
            r.seconds,
            r.rate
        );
    }

    let all_pings: Vec<f64> = ping_before
        .iter()
        .chain(ping_after)
        .filter_map(|p| *p)
        .collect();
    let attempted = ping_before.len() + ping_after.len();
    match median(all_pings.clone()) {
        Some(med) => println!(
            "\nAPI latency ({}): median {med:.1} ms over {}/{attempted} samples",
            cli.api_base,
            all_pings.len(),
        ),
        None => println!("\nAPI latency: unavailable ({attempted} attempts failed)"),
    }

    match score {
        Some(s) => println!(
            "\nNiceMark: {s:.0} (v{CLIENT_VERSION}, {} {})",
            cli.mode,
            if cli.gpu { "gpu" } else { "cpu" }
        ),
        None => println!("\nNiceMark: n/a (no scored scenarios completed)"),
    }
}

#[allow(clippy::too_many_arguments)]
fn build_report_json(
    cli: &Cli,
    results: &[ScenarioResult],
    ping_before: &[Option<f64>],
    ping_after: &[Option<f64>],
    hardware: &Value,
    environment: &Value,
    score: Option<f64>,
    table_build_secs: &HashMap<u32, f64>,
) -> Value {
    let scenarios: Vec<Value> = results
        .iter()
        .map(|r| {
            json!({
                "key": r.key,
                "base": r.base,
                "character": r.character,
                "threads": r.threads,
                "window_start": r.window_start.to_string(),
                "window_size": r.window_size.to_string(),
                "repetitions": r.repetitions,
                "seconds": r.seconds,
                "rate": r.rate,
                "warmup_seconds": r.warmup_seconds,
            })
        })
        .collect();

    let all_pings: Vec<f64> = ping_before
        .iter()
        .chain(ping_after)
        .filter_map(|p| *p)
        .collect();

    json!({
        "schema_version": BENCH_SCHEMA_VERSION,
        "client_version": CLIENT_VERSION,
        "timestamp_epoch": SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0),
        "config": {
            "mode": cli.mode.to_string(),
            "gpu": cli.gpu,
            "threads": cli.threads,
            "benchmark_secs": cli.benchmark_secs,
        },
        "hardware": hardware,
        "environment": environment,
        "api_latency": {
            "endpoint": format!("{}/ping", cli.api_base),
            "before_ms": ping_before,
            "after_ms": ping_after,
            "median_ms": median(all_pings),
        },
        "stride_table_build_secs": table_build_secs
            .iter()
            .map(|(base, secs)| (base.to_string(), json!(secs)))
            .collect::<serde_json::Map<String, Value>>(),
        "scenarios": scenarios,
        "score": score,
    })
}

/// The constant part of a submission telemetry payload: hardware, scheduler
/// environment, and client configuration. Collected once per process.
pub fn telemetry_base(cli: &Cli, gpu: &GpuCtx) -> Value {
    json!({
        "schema_version": TELEMETRY_SCHEMA_VERSION,
        "hardware": collect_hardware(cli, gpu),
        "environment": collect_environment(),
        "config": {
            "gpu": cli.gpu,
            "threads": cli.threads,
        },
    })
}

/// Stamp the constant telemetry base with this field's processing time
/// (client-side wall time, unlike the server's claim-to-submit elapsed).
pub fn field_telemetry(base: &Value, processing_secs: f64) -> Value {
    let mut value = base.clone();
    if let Value::Object(map) = &mut value {
        map.insert("processing_secs".to_string(), json!(processing_secs));
    }
    value
}

/// Whether to upload the report: the flag skips the prompt as a yes, a
/// terminal gets asked (default yes), and a non-interactive run without the
/// flag never uploads.
#[derive(Debug, PartialEq)]
enum UploadDecision {
    Yes,
    Prompt,
    No,
}

fn decide_upload(upload_flag: bool, is_tty: bool) -> UploadDecision {
    if upload_flag {
        UploadDecision::Yes
    } else if is_tty {
        UploadDecision::Prompt
    } else {
        UploadDecision::No
    }
}

/// Ask on the terminal, defaulting to yes on an empty answer.
fn prompt_yes(api_base: &str) -> bool {
    print!("Upload results to {api_base}? [Y/n] ");
    let _ = std::io::stdout().flush();
    let mut answer = String::new();
    if std::io::stdin().read_line(&mut answer).is_err() {
        return false;
    }
    let answer = answer.trim().to_lowercase();
    answer.is_empty() || answer == "y" || answer == "yes"
}

/// Send the report to the server. Failures are reported but never fatal —
/// the benchmark already served its local purpose.
async fn upload_report(client: &Client, cli: &Cli, report: &Value) {
    let body = BenchmarkToServer {
        username: cli.username.clone(),
        data: report.clone(),
    };
    let url = format!("{}/benchmark", cli.api_base);
    match client.post(&url).json(&body).send().await {
        Ok(resp) if resp.status().is_success() => {
            let msg = resp
                .json::<Value>()
                .await
                .ok()
                .and_then(|v| v.get("message").and_then(Value::as_str).map(String::from))
                .unwrap_or_else(|| "ok".to_string());
            println!("Upload accepted: {msg}");
        }
        Ok(resp) => println!("Upload rejected ({}).", resp.status()),
        Err(e) => println!("Upload failed: {e}"),
    }
}

/// Collect hardware info. Linux-oriented (`/proc`), with graceful absence
/// elsewhere; every field is optional downstream.
fn collect_hardware(cli: &Cli, gpu: &GpuCtx) -> Value {
    let cpuinfo = std::fs::read_to_string("/proc/cpuinfo").unwrap_or_default();
    let meminfo = std::fs::read_to_string("/proc/meminfo").unwrap_or_default();
    let os_release = std::fs::read_to_string("/etc/os-release").unwrap_or_default();

    json!({
        "cpu_model": parse_cpu_model(&cpuinfo),
        "cpu_threads_available": std::thread::available_parallelism().map(std::num::NonZero::get).ok(),
        "mem_total_kb": parse_meminfo_total_kb(&meminfo),
        "arch": std::env::consts::ARCH,
        "os": std::env::consts::OS,
        "os_pretty": parse_os_pretty(&os_release),
        "gpu_model": gpu_name(cli, gpu),
    })
}

/// The first CPU model string in `/proc/cpuinfo` contents. x86 kernels use
/// `model name`; ARM kernels (e.g. Raspberry Pi) report `Model` or
/// `Hardware` instead.
fn parse_cpu_model(cpuinfo: &str) -> Option<String> {
    for key in ["model name", "Model", "Hardware"] {
        for line in cpuinfo.lines() {
            if let Some(rest) = line.strip_prefix(key)
                && let Some((_, value)) = rest.split_once(':')
            {
                let value = value.trim();
                if !value.is_empty() {
                    return Some(value.to_string());
                }
            }
        }
    }
    None
}

fn parse_meminfo_total_kb(meminfo: &str) -> Option<u64> {
    meminfo
        .lines()
        .find(|l| l.starts_with("MemTotal:"))?
        .split_whitespace()
        .nth(1)?
        .parse()
        .ok()
}

fn parse_os_pretty(os_release: &str) -> Option<String> {
    os_release
        .lines()
        .find(|l| l.starts_with("PRETTY_NAME="))
        .map(|l| l["PRETTY_NAME=".len()..].trim_matches('"').to_string())
}

/// The active GPU's model name, from the initialized backend rather than from
/// the CLI flags — see `GpuHandle::device_name`. `None` on a CPU run, and on a
/// GPU run only if the backend cannot name its device.
fn gpu_name(cli: &Cli, gpu: &GpuCtx) -> Option<String> {
    if !cli.gpu {
        return None;
    }
    gpu.as_ref()
        .and_then(|handle| handle.device_name(cli.gpu_device))
}

/// Scheduler/instance identifiers from the environment allowlist, for
/// cross-correlating benchmark results with rented instances or cluster jobs.
///
/// Falls back to the container init process's environment for keys not in
/// our own: Vast injects its identifiers into PID 1, and a client started
/// from an SSH session (rather than the container entrypoint) doesn't
/// inherit them. Allowlist-only either way — PID 1 also holds credentials.
fn collect_environment() -> Value {
    let init_environ = std::fs::read("/proc/1/environ").unwrap_or_default();
    let init_vars = parse_environ(&init_environ);
    let mut map = serde_json::Map::new();
    for key in ENV_ALLOWLIST {
        let value = std::env::var(key)
            .ok()
            .or_else(|| init_vars.get(*key).cloned());
        if let Some(value) = value
            && !value.is_empty()
        {
            map.insert((*key).to_string(), Value::String(value));
        }
    }
    debug!("environment correlation keys collected: {}", map.len());
    Value::Object(map)
}

/// Parse a NUL-separated `KEY=value` environment block (`/proc/N/environ`).
fn parse_environ(data: &[u8]) -> HashMap<String, String> {
    data.split(|&b| b == 0)
        .filter_map(|entry| {
            let entry = std::str::from_utf8(entry).ok()?;
            let (key, value) = entry.split_once('=')?;
            Some((key.to_string(), value.to_string()))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cpu_model_x86_and_arm() {
        let x86 = "processor\t: 0\nmodel name\t: AMD EPYC 7763 64-Core Processor\n";
        assert_eq!(
            parse_cpu_model(x86).as_deref(),
            Some("AMD EPYC 7763 64-Core Processor")
        );
        let pi = "processor\t: 0\nBogoMIPS\t: 108.00\nModel\t\t: Raspberry Pi 5 Model B Rev 1.0\n";
        assert_eq!(
            parse_cpu_model(pi).as_deref(),
            Some("Raspberry Pi 5 Model B Rev 1.0")
        );
        assert_eq!(parse_cpu_model(""), None);
    }

    #[test]
    fn meminfo_and_os_release() {
        assert_eq!(
            parse_meminfo_total_kb("MemTotal:       16265216 kB\nMemFree: 1 kB\n"),
            Some(16_265_216)
        );
        assert_eq!(
            parse_os_pretty("NAME=\"Debian\"\nPRETTY_NAME=\"Debian GNU/Linux 13 (trixie)\"\n")
                .as_deref(),
            Some("Debian GNU/Linux 13 (trixie)")
        );
    }

    #[test]
    fn upload_decision_matrix() {
        assert_eq!(decide_upload(true, true), UploadDecision::Yes);
        assert_eq!(decide_upload(true, false), UploadDecision::Yes);
        assert_eq!(decide_upload(false, true), UploadDecision::Prompt);
        assert_eq!(decide_upload(false, false), UploadDecision::No);
    }

    #[test]
    fn field_telemetry_stamps_timing() {
        let base = json!({"schema_version": TELEMETRY_SCHEMA_VERSION, "hardware": {}});
        let stamped = field_telemetry(&base, 12.5);
        assert_eq!(stamped["processing_secs"], json!(12.5));
        assert_eq!(stamped["schema_version"], json!(TELEMETRY_SCHEMA_VERSION));
        // The base is not mutated; every field gets a fresh stamp.
        assert!(base.get("processing_secs").is_none());
    }

    #[test]
    fn environ_block_parses() {
        let vars = parse_environ(b"CONTAINER_ID=47102363\0VAST_CONTAINERLABEL=C.47102363\0BAD\0");
        assert_eq!(vars.get("CONTAINER_ID").map(String::as_str), Some("47102363"));
        assert_eq!(
            vars.get("VAST_CONTAINERLABEL").map(String::as_str),
            Some("C.47102363")
        );
        assert!(!vars.contains_key("BAD"));
    }

    #[test]
    fn median_of_samples() {
        assert_eq!(median(vec![]), None);
        assert_eq!(median(vec![3.0]), Some(3.0));
        assert_eq!(median(vec![5.0, 1.0, 3.0]), Some(3.0));
    }

    #[test]
    fn score_uses_only_referenced_scenarios() {
        let reference_rate = SCORE_REFERENCES
            .iter()
            .find(|(k, g, _)| *k == "b50_msd_weak" && !g)
            .unwrap()
            .2;
        let results = vec![
            ScenarioResult {
                key: "b50_msd_weak",
                base: 50,
                character: "msd-weak",
                threads: 4,
                window_start: 0,
                window_size: 1,
                repetitions: 1,
                seconds: 1.0,
                rate: reference_rate,
                warmup_seconds: 0.0,
            },
            ScenarioResult {
                key: "not_a_real_scenario",
                base: 50,
                character: "msd-weak",
                threads: 4,
                window_start: 0,
                window_size: 1,
                repetitions: 1,
                seconds: 1.0,
                rate: 1.0,
                warmup_seconds: 0.0,
            },
        ];
        // Exactly matching the reference on the only scored scenario = 1000.
        let score = compute_score(&results, false).unwrap();
        assert!((score - 1000.0).abs() < 1e-6);
        // An unmeasured scenario contributes nothing.
        let unmeasured = vec![ScenarioResult {
            rate: 0.0,
            ..results.into_iter().next().unwrap()
        }];
        assert_eq!(compute_score(&unmeasured, false), None);
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
}
