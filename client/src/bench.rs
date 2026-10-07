//! Structured benchmark sweep.
//!
//! Replaces the old single-field benchmark modes with an adaptive sweep that
//! measures the configuration the user would actually run (mode, threads,
//! GPU). Nice-only measures production fields on the route a client takes
//! for them on this device (the overlap join where it can run), timing a
//! sample of each field's partitions and scaling it to the whole field.
//! Detailed mode repeats fixed windows. Each scenario gets an equal share of
//! the `--benchmark-secs` budget.
//!
//! Also measures API latency against the lightweight `/ping` endpoint
//! (spread before and after the sweep), collects hardware and scheduler
//! environment info for cross-correlation, and prints both a human-readable
//! table and a complete machine-readable JSON report. The synthetic
//! `NiceMark` score at the end is for bragging rights only — it is a
//! geometric mean against reference rates pinned per client version and is
//! never used for real analysis.

#[cfg(feature = "cubecl")]
use crate::GpuHandle;
use crate::{Cli, DEFAULT_LSD_K_VALUE, GpuCtx, process_field_sync};
use log::{debug, warn};
use nice_common::bench_defs::{
    BENCH_SCHEMA_VERSION, DETAILED_SCENARIOS, FieldScenario, NICEONLY_FIELDS, ScenarioDef,
    compute_score, sample_order,
};
use nice_common::client_api_async::Client;
use nice_common::cpu_join::{CpuJoin, PartitionResult, Scratch, slices_for};
#[cfg(feature = "cubecl")]
use nice_common::cubecl_backend::CubeclContext;
use nice_common::overlap_join::{StrideReason, join_verdict, slice_weight};
use nice_common::stride_filter::StrideTable;
use nice_common::{
    BUILD_SHA, BenchmarkToServer, CLIENT_VERSION, DataToClient, FieldSize, SearchMode,
};
use rayon::prelude::*;
use serde_json::{Value, json};
use std::collections::HashMap;
use std::io::{IsTerminal, Write};
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

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
    /// The region the rate is for: the window, or a nice-only field.
    window_start: u128,
    window_size: u128,
    repetitions: u32,
    seconds: f64,
    rate: f64,
    warmup_seconds: f64,
    /// The MSD floor in force after the measured windows, on a GPU
    /// nice-only field that took the stride path; `None` elsewhere.
    msd_floor: Option<u128>,
    /// How a nice-only field scenario measured its field.
    field: Option<FieldMeasure>,
}

/// How a nice-only field scenario measured its field: on the route a client
/// takes for that field on this device, scaled to the whole field.
struct FieldMeasure {
    /// `join` or `stride`.
    route: &'static str,
    /// Why the field takes the stride path, where it does.
    route_reason: Option<&'static str>,
    /// Seconds the whole field would take.
    field_secs: f64,
    /// The overlap join's sample, on the join route.
    join: Option<JoinMeasure>,
    /// The window timed at the field's start, on the stride route.
    stride_window: Option<u128>,
    /// Why the measurement failed (the rate is then 0).
    error: Option<String>,
}

/// A timed sample of a field's partitions on the overlap join.
struct JoinMeasure {
    slices: usize,
    /// The field's size in first slices
    /// ([`nice_common::overlap_join::slice_weight`]).
    slice_weight: f64,
    /// Partitions per slice, and per launch (GPU).
    partitions: u32,
    slots: Option<usize>,
    /// Partitions timed, all from the first slice.
    sampled: usize,
    /// The first slice's host setup, and the time the sampled partitions
    /// took (the device's wall time on a GPU, the pool's on a CPU).
    setup_secs: f64,
    run_secs: f64,
    /// The time each partition past the sample adds, where a sample also
    /// pays a fixed cost a slice pays only once (GPU; see
    /// [`gpu_join_sample_on`]). Otherwise the sample scales evenly.
    marginal_secs: Option<f64>,
    survivors: u64,
    checked: u64,
    retried_partitions: usize,
    refused_layouts: u32,
    hits: usize,
    /// Whether the host's setup overlaps the device's work (GPU).
    setup_overlaps: bool,
}

impl JoinMeasure {
    /// The first slice's partitions: the sample, and the rest at the
    /// marginal rate where there is one, else at the sample's.
    #[allow(clippy::cast_precision_loss)]
    fn slice_run_secs(&self) -> f64 {
        let (all, done) = (f64::from(self.partitions), self.sampled.max(1) as f64);
        self.marginal_secs.map_or(self.run_secs * all / done, |m| {
            self.run_secs + (all - done).max(0.0) * m
        })
    }

    /// The whole field: the first slice's partitions and its setup, times
    /// the field's size in first slices. A GPU client sets up its next
    /// field (and a field its next slice) on the host while the device runs
    /// the current one, so there setup costs time only where it is the
    /// longer of the two; the CPU sets up a slice before running its
    /// partitions.
    fn field_secs(&self) -> f64 {
        let run = self.slice_run_secs();
        let slice = if self.setup_overlaps {
            self.setup_secs.max(run)
        } else {
            self.setup_secs + run
        };
        self.slice_weight * slice
    }
}

/// What sampling a field on the overlap join came to.
enum JoinOutcome {
    Sampled {
        measure: Box<JoinMeasure>,
        /// Untimed: kernel compilation (GPU).
        warmup_secs: f64,
        /// Wall time of the timed part.
        timed_secs: f64,
    },
    /// The field takes the stride path on this device, for this reason.
    Stride(&'static str),
    /// The device failed (GPU only).
    #[cfg_attr(not(feature = "cubecl"), allow(dead_code))]
    Failed(String),
}

/// Whole batches the GPU sample times at least: a pipeline's worth
/// (`join_plan::BATCHES_IN_FLIGHT`).
#[cfg(feature = "cubecl")]
const MIN_GPU_BATCHES: usize = 4;

/// Partitions per thread the CPU sample times at least.
const MIN_CPU_PER_THREAD: usize = 2;

/// Print a human-facing line: to stdout normally, to stderr under
/// `--benchmark-json`, where stdout is reserved for the JSON document.
macro_rules! say {
    ($cli:expr, $($arg:tt)*) => {
        if $cli.benchmark_json {
            eprintln!($($arg)*);
        } else {
            println!($($arg)*);
        }
    };
}

/// Run the full sweep and print the report. Never contacts the server except
/// for `/ping` latency samples, which are skipped gracefully offline.
pub async fn run_benchmark_sweep(cli: &Arc<Cli>, gpu: &GpuCtx, client: &Client) {
    say!(
        cli,
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
    let score = compute_score(results.iter().map(|r| (r.key, r.rate)), cli.gpu);

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

    if cli.benchmark_json {
        // The one thing on stdout: a single parseable document.
        println!("{}", serde_json::to_string_pretty(&report).unwrap());
    } else {
        print_report(cli, &results, &ping_before, &ping_after, score);
    }

    match decide_upload(cli.benchmark_upload, std::io::stdin().is_terminal()) {
        UploadDecision::Yes => upload_report(client, cli, &report).await,
        UploadDecision::Prompt => {
            if prompt_yes(cli) {
                upload_report(client, cli, &report).await;
            } else {
                say!(cli, "Not uploaded.");
            }
        }
        UploadDecision::No => {
            say!(
                cli,
                "Not uploading (non-interactive; pass --benchmark-upload to upload)."
            );
        }
    }
}

/// The blocking part: calibrate and run every scenario within the budget.
fn run_sweep(cli: &Arc<Cli>, gpu: &GpuCtx) -> (Vec<ScenarioResult>, HashMap<u32, f64>) {
    // The progress bar is noise at benchmark window sizes.
    let mut quiet = (**cli).clone();
    quiet.no_progress = true;
    let quiet = Arc::new(quiet);

    let mut cache = TableCache {
        tables: HashMap::new(),
        build_secs: HashMap::new(),
    };
    // Single-thread scenarios decompose CPU scaling; they mean nothing for
    // the GPU pipeline.
    let results = match cli.mode {
        SearchMode::Niceonly => {
            let defs: Vec<&FieldScenario> = NICEONLY_FIELDS
                .iter()
                .filter(|d| !(cli.gpu && d.single_thread))
                .collect();
            #[allow(clippy::cast_precision_loss)]
            let share = cli.benchmark_secs / defs.len() as f64;
            defs.iter()
                .map(|def| run_field_scenario(&quiet, gpu, def, share, &mut cache))
                .collect()
        }
        SearchMode::Detailed => {
            let defs: Vec<&ScenarioDef> = DETAILED_SCENARIOS
                .iter()
                .filter(|d| !(cli.gpu && d.single_thread))
                .collect();
            #[allow(clippy::cast_precision_loss)]
            let share = cli.benchmark_secs / defs.len() as f64;
            defs.iter()
                .map(|def| run_scenario(&quiet, gpu, def, share, &mut cache))
                .collect()
        }
    };
    (results, cache.build_secs)
}

/// Run one detailed scenario: warm up, then repeat the fixed window until
/// the scenario's share of the time budget is spent (always at least once).
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
    let start = def.resolved_start();
    let w = run_windows(
        cli,
        gpu,
        def.base,
        def.single_thread,
        start,
        window,
        share_secs,
        cache,
    );
    ScenarioResult {
        key: def.key,
        base: def.base,
        character: def.character,
        threads,
        window_start: start,
        window_size: window,
        repetitions: w.repetitions,
        seconds: w.seconds,
        rate: w.rate,
        warmup_seconds: w.warmup_seconds,
        msd_floor: w.msd_floor,
        field: None,
    }
}

/// Run one nice-only field scenario on the route a client takes for the
/// field on this device: a sample of its partitions on the overlap join, or
/// windows at its start on the stride path (whose cost is linear in size),
/// scaled to the whole field either way.
fn run_field_scenario(
    cli: &Arc<Cli>,
    gpu: &GpuCtx,
    def: &FieldScenario,
    share_secs: f64,
    cache: &mut TableCache,
) -> ScenarioResult {
    let threads = if def.single_thread { 1 } else { cli.threads };
    let range = def.measured(cli.gpu);
    let outcome = if cli.gpu {
        gpu_join_sample(gpu, def, &range, share_secs)
    } else {
        cpu_join_sample(def, &range, threads, share_secs)
    };
    let mut result = ScenarioResult {
        key: def.key,
        base: def.base,
        character: def.character,
        threads,
        window_start: range.start(),
        window_size: range.size(),
        repetitions: 0,
        seconds: 0.0,
        rate: 0.0,
        warmup_seconds: 0.0,
        msd_floor: None,
        field: None,
    };
    let size = approx_f64(range.size());
    let measure = match outcome {
        JoinOutcome::Sampled {
            measure,
            warmup_secs,
            timed_secs,
        } => {
            let field_secs = measure.field_secs();
            result.repetitions = 1;
            result.seconds = timed_secs;
            result.warmup_seconds = warmup_secs;
            result.rate = size / field_secs.max(1e-9);
            FieldMeasure {
                route: "join",
                route_reason: None,
                field_secs,
                join: Some(*measure),
                stride_window: None,
                error: None,
            }
        }
        JoinOutcome::Stride(reason) => {
            let window = if cli.gpu {
                def.stride_window_gpu
            } else {
                def.stride_window_cpu * threads as u128
            };
            let w = run_windows(
                cli,
                gpu,
                def.base,
                def.single_thread,
                def.start,
                window,
                share_secs,
                cache,
            );
            result.repetitions = w.repetitions;
            result.seconds = w.seconds;
            result.warmup_seconds = w.warmup_seconds;
            result.msd_floor = w.msd_floor;
            result.rate = w.rate;
            FieldMeasure {
                route: "stride",
                route_reason: Some(reason),
                field_secs: size / w.rate.max(1e-9),
                join: None,
                stride_window: Some(window),
                error: None,
            }
        }
        JoinOutcome::Failed(error) => {
            warn!("benchmark scenario {} failed: {error}", def.key);
            FieldMeasure {
                route: "join",
                route_reason: None,
                field_secs: 0.0,
                join: None,
                stride_window: None,
                error: Some(error),
            }
        }
    };
    result.field = Some(measure);
    result
}

/// The GPU's overlap join, if its backend has one: `CubeCL` alone, or paired
/// with hand-CUDA (`--gpu-backend auto` on NVIDIA).
#[cfg(feature = "cubecl")]
fn join_context(gpu: &GpuCtx) -> Option<&CubeclContext> {
    match &**gpu.as_ref()? {
        GpuHandle::Cubecl(ctx) => Some(ctx),
        #[cfg(all(feature = "cuda", feature = "cubecl-cuda"))]
        GpuHandle::CudaJoin(pair) => Some(&pair.join),
        #[allow(unreachable_patterns)]
        _ => None,
    }
}

/// Sample `def`'s field on the GPU's overlap join (see
/// [`gpu_join_sample_on`]), or say why it takes the stride path there.
fn gpu_join_sample(
    gpu: &GpuCtx,
    def: &FieldScenario,
    range: &FieldSize,
    share_secs: f64,
) -> JoinOutcome {
    #[cfg(feature = "cubecl")]
    if let Some(ctx) = join_context(gpu) {
        return gpu_join_sample_on(ctx, def, range, share_secs);
    }
    let _ = (gpu, share_secs);
    // As `gpu_route::begin_niceonly` decides: the field's own reason if it
    // has one, else the backend's.
    JoinOutcome::Stride(
        join_verdict(def.base, range)
            .err()
            .unwrap_or(StrideReason::NoJoin)
            .label(),
    )
}

/// Time partitions of the field's first slice through the GPU join's driver,
/// in the production layout ([`CubeclContext::join_sample`]). An untimed
/// one-partition sample compiles the kernels (and takes any layout the
/// device refuses); one timed batch then sizes the sample to the share, in
/// whole batches, at least [`MIN_GPU_BATCHES`]. Each sample also pays its own
/// setup, so the batch overstates the time per batch and the sample comes
/// out short of the share rather than past it.
///
/// Each sample's device time also holds a fixed cost before its first batch
/// (25-55 ms on an RTX 3080 or 4090), which a slice pays once. The sizing
/// batch paid it too, so the two samples' difference is the time the
/// further partitions take, and the slice's other partitions are scaled
/// from that rather than from the whole sample's average.
#[cfg(feature = "cubecl")]
fn gpu_join_sample_on(
    ctx: &CubeclContext,
    def: &FieldScenario,
    range: &FieldSize,
    share_secs: f64,
) -> JoinOutcome {
    let sample = |parts: &[u32]| match ctx.join_sample(def.base, range, parts) {
        Ok(Ok(s)) => Ok(s),
        Ok(Err(reason)) => Err(JoinOutcome::Stride(reason.label())),
        Err(e) => Err(JoinOutcome::Failed(format!("{e:#}"))),
    };
    let t0 = Instant::now();
    let warm = match sample(&[0]) {
        Ok(s) => s,
        Err(outcome) => return outcome,
    };
    let warmup_secs = t0.elapsed().as_secs_f64();
    let Ok(partitions) = u32::try_from(warm.partitions) else {
        return JoinOutcome::Failed(format!("{} partitions", warm.partitions));
    };
    let order = sample_order(partitions);
    let timed = Instant::now();
    let batch = match sample(&order[..warm.slots.clamp(1, order.len())]) {
        Ok(s) => s,
        Err(outcome) => return outcome,
    };
    let left = share_secs - timed.elapsed().as_secs_f64();
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let batches = (left / batch.run_secs.max(1e-4)).max(0.0) as usize;
    let count = (batches.max(MIN_GPU_BATCHES) * batch.slots.max(1)).min(order.len());
    let s = match sample(&order[..count]) {
        Ok(s) => s,
        Err(outcome) => return outcome,
    };
    #[allow(clippy::cast_precision_loss)]
    let marginal_secs = (s.sampled > batch.sampled)
        .then(|| (s.run_secs - batch.run_secs) / (s.sampled - batch.sampled) as f64)
        .filter(|&m| m > 0.0);
    JoinOutcome::Sampled {
        measure: Box::new(JoinMeasure {
            slices: s.slices,
            slice_weight: s.slice_weight,
            partitions,
            slots: Some(s.slots),
            sampled: s.sampled,
            setup_secs: s.setup_secs,
            run_secs: s.run_secs,
            marginal_secs,
            survivors: s.survivors,
            checked: s.checked,
            retried_partitions: s.retried_partitions,
            refused_layouts: warm.refused + batch.refused + s.refused,
            hits: s.hits.len(),
            setup_overlaps: true,
        }),
        warmup_secs,
        timed_secs: timed.elapsed().as_secs_f64(),
    }
}

/// Time partitions of the field's first slice on the CPU join, on the
/// client's thread pool as a field runs (one thread for a single-thread
/// scenario): at least [`MIN_CPU_PER_THREAD`] per thread, then one more
/// round sized to fill the share at the rate measured so far. The share
/// includes the slice's setup, which every field pays.
fn cpu_join_sample(
    def: &FieldScenario,
    range: &FieldSize,
    threads: usize,
    share_secs: f64,
) -> JoinOutcome {
    let Some((jp, slices)) = slices_for(def.base, range) else {
        return JoinOutcome::Stride(
            join_verdict(def.base, range)
                .err()
                .unwrap_or(StrideReason::Setup)
                .label(),
        );
    };
    let timed = Instant::now();
    let join = match CpuJoin::new(def.base, &slices[0], jp) {
        Ok(join) => join,
        Err(e) => {
            warn!(
                "overlap join cannot take {} ({e:#}); timing the stride walk",
                def.key
            );
            return JoinOutcome::Stride(StrideReason::Setup.label());
        }
    };
    let setup_secs = timed.elapsed().as_secs_f64();
    let partitions = join.partitions();
    let order = sample_order(partitions);
    let pool = (threads == 1).then(|| {
        rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .expect("single-thread pool")
    });
    let run = |parts: &[u32]| {
        let go = || {
            parts
                .par_iter()
                .map_init(Scratch::default, |scratch, &v| {
                    join.run_partition(v, scratch)
                })
                .reduce(PartitionResult::default, |mut a, b| {
                    a.add(b);
                    a
                })
        };
        pool.as_ref().map_or_else(go, |p| p.install(go))
    };
    let mut found = PartitionResult::default();
    let (mut done, mut run_secs) = (0usize, 0.0f64);
    let mut next = (MIN_CPU_PER_THREAD * threads).clamp(1, order.len());
    while next > 0 {
        let t = Instant::now();
        found.add(run(&order[done..done + next]));
        run_secs += t.elapsed().as_secs_f64();
        done += next;
        let left = share_secs - timed.elapsed().as_secs_f64();
        #[allow(clippy::cast_precision_loss)]
        let per_partition = run_secs / done as f64;
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let fit = (left / per_partition).max(0.0) as usize;
        next = (fit / threads * threads).min(order.len() - done);
    }
    JoinOutcome::Sampled {
        measure: Box::new(JoinMeasure {
            slices: slices.len(),
            slice_weight: slice_weight(&slices),
            partitions,
            slots: None,
            sampled: done,
            setup_secs,
            run_secs,
            marginal_secs: None,
            survivors: found.survivors,
            checked: found.checked,
            retried_partitions: 0,
            refused_layouts: 0,
            hits: found.hits.len(),
            setup_overlaps: false,
        }),
        warmup_secs: 0.0,
        timed_secs: timed.elapsed().as_secs_f64(),
    }
}

/// What timing one region's windows came to.
struct Windows {
    repetitions: u32,
    seconds: f64,
    rate: f64,
    warmup_seconds: f64,
    msd_floor: Option<u128>,
}

/// Time windows of one region: an untimed warm-up, then the window repeated
/// until the share is spent (always at least once).
#[allow(clippy::too_many_arguments)]
fn run_windows(
    cli: &Arc<Cli>,
    gpu: &GpuCtx,
    base: u32,
    single_thread: bool,
    start: u128,
    window: u128,
    share_secs: f64,
    cache: &mut TableCache,
) -> Windows {
    // Build the stride table outside the timed windows.
    let table = cache.get(cli.mode, base);

    // One untimed warmup so one-time costs (GPU kernel JIT for this base,
    // thread pool spin-up, cold caches) land outside the measurement. The
    // GPU needs the full window to reach the device path; the CPU warms up
    // on a fraction so slow devices don't pay the window twice.
    let warmup_window = if cli.gpu { window } else { (window / 8).max(1) };
    let warmup_t0 = Instant::now();
    run_window(
        cli,
        gpu,
        base,
        single_thread,
        start,
        warmup_window,
        table.as_ref(),
    );
    let warmup_seconds = warmup_t0.elapsed().as_secs_f64();

    let scenario_start = Instant::now();
    let mut repetitions = 0u32;
    let mut total_secs = 0.0f64;
    loop {
        total_secs += run_window(cli, gpu, base, single_thread, start, window, table.as_ref());
        repetitions += 1;
        if scenario_start.elapsed().as_secs_f64() >= share_secs * 0.9 {
            break;
        }
    }

    #[allow(clippy::cast_precision_loss)]
    let rate = approx_f64(window) * f64::from(repetitions) / total_secs.max(1e-4);
    Windows {
        repetitions,
        seconds: total_secs,
        rate,
        warmup_seconds,
        msd_floor: msd_floor(cli),
    }
}

/// The MSD floor the GPU nice-only stride pipeline steered to, for the
/// report; `None` elsewhere. The floor is steered as in production, across
/// the warm-up and the windows: holding it for the measurement changed the
/// rate by 3–15% either way on the GPUs measured.
#[cfg(any(feature = "cuda", feature = "cubecl"))]
fn msd_floor(cli: &Cli) -> Option<u128> {
    (cli.gpu && cli.mode == SearchMode::Niceonly).then(nice_common::gpu_niceonly::msd_floor_in_use)
}

#[cfg(not(any(feature = "cuda", feature = "cubecl")))]
fn msd_floor(_cli: &Cli) -> Option<u128> {
    None
}

/// Process one window through the production path and return elapsed seconds.
fn run_window(
    cli: &Arc<Cli>,
    gpu: &GpuCtx,
    base: u32,
    single_thread: bool,
    start: u128,
    window: u128,
    table: Option<&Arc<StrideTable>>,
) -> f64 {
    let claim = DataToClient {
        claim_id: 0,
        base,
        range_start: start,
        range_end: start + window,
        range_size: window,
    };
    let t0 = Instant::now();
    if single_thread {
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

/// The detailed scenarios: fixed windows, repeated.
fn print_window_table(results: &[ScenarioResult]) {
    println!(
        "{:<20} {:>4} {:<14} {:>7} {:>10} {:>6} {:>8} {:>12} {:>8}",
        "scenario",
        "base",
        "character",
        "threads",
        "window",
        "reps",
        "secs",
        "numbers/sec",
        "floor"
    );
    for r in results {
        println!(
            "{:<20} {:>4} {:<14} {:>7} {:>10.1e} {:>6} {:>8.3} {:>12.3e} {:>8}",
            r.key,
            r.base,
            r.character,
            r.threads,
            approx_f64(r.window_size),
            r.repetitions,
            r.seconds,
            r.rate,
            r.msd_floor
                .map_or_else(|| "-".to_string(), |f| f.to_string())
        );
    }
}

/// The nice-only fields: the route each took, what was timed (partitions
/// sampled, or the stride window), and the whole field's time and rate.
fn print_field_table(results: &[ScenarioResult]) {
    let w = results
        .iter()
        .map(|r| r.key.len())
        .chain([8])
        .max()
        .unwrap_or(8);
    println!(
        "{:<w$} {:>4} {:>7} {:<6} {:>12} {:>8} {:>10} {:>12} {:>8}",
        "scenario",
        "base",
        "threads",
        "route",
        "timed",
        "secs",
        "field secs",
        "numbers/sec",
        "floor"
    );
    for r in results {
        let Some(f) = &r.field else { continue };
        let timed = match (&f.join, f.stride_window) {
            (Some(j), _) => format!("{}/{}", j.sampled, j.partitions),
            (None, Some(w)) => format!("{:.0e} window", approx_f64(w)),
            (None, None) => "-".to_string(),
        };
        println!(
            "{:<w$} {:>4} {:>7} {:<6} {:>12} {:>8.3} {:>10.2} {:>12.3e} {:>8}",
            r.key,
            r.base,
            r.threads,
            f.route,
            timed,
            r.seconds,
            f.field_secs,
            r.rate,
            r.msd_floor
                .map_or_else(|| "-".to_string(), |f| f.to_string())
        );
    }
    for r in results {
        let Some(f) = &r.field else { continue };
        if let Some(reason) = f.route_reason {
            println!("{}: the stride path ({reason})", r.key);
        }
        if let Some(error) = &f.error {
            println!("{}: failed: {error}", r.key);
        }
    }
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

fn print_report(
    cli: &Cli,
    results: &[ScenarioResult],
    ping_before: &[Option<f64>],
    ping_after: &[Option<f64>],
    score: Option<f64>,
) {
    println!();
    if results.iter().any(|r| r.field.is_some()) {
        print_field_table(results);
    } else {
        print_window_table(results);
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
            let mut scenario = json!({
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
                "msd_floor": r.msd_floor.map(|f| f.to_string()),
            });
            if let (Some(f), Value::Object(map)) = (&r.field, &mut scenario) {
                map.insert("route".into(), json!(f.route));
                map.insert("route_reason".into(), json!(f.route_reason));
                map.insert("field_secs".into(), json!(f.field_secs));
                map.insert(
                    "stride_window".into(),
                    json!(f.stride_window.map(|w| w.to_string())),
                );
                map.insert(
                    "join".into(),
                    f.join.as_ref().map_or(Value::Null, |j| {
                        json!({
                            "slices": j.slices,
                            "slice_weight": j.slice_weight,
                            "partitions": j.partitions,
                            "slots": j.slots,
                            "sampled": j.sampled,
                            "setup_secs": j.setup_secs,
                            "run_secs": j.run_secs,
                            "marginal_secs": j.marginal_secs,
                            "survivors": j.survivors,
                            "checked": j.checked,
                            "retried_partitions": j.retried_partitions,
                            "refused_layouts": j.refused_layouts,
                            "hits": j.hits,
                        })
                    }),
                );
                map.insert("error".into(), json!(f.error));
            }
            scenario
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
        "build_sha": BUILD_SHA,
        "timestamp_epoch": SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_or(0, |d| d.as_secs()),
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
        "build_sha": BUILD_SHA,
        "hardware": collect_hardware(cli, gpu),
        "environment": collect_environment(),
        "config": {
            "gpu": cli.gpu,
            "threads": cli.threads,
        },
    })
}

/// Stamp the constant telemetry base with this field's processing time
/// (client-side wall time, unlike the server's claim-to-submit elapsed) and,
/// for GPU niceonly fields, the pipeline's per-field accounting: the MSD
/// floor, how long each side waited on the other, and the device's busy
/// time where the backend can measure it. That is what says whether a
/// machine is CPU-bound or GPU-bound, which the hardware fields alone do
/// not.
pub fn field_telemetry(base: &Value, processing_secs: f64, pipeline: Option<&Value>) -> Value {
    let mut value = base.clone();
    if let Value::Object(map) = &mut value {
        map.insert("processing_secs".to_string(), json!(processing_secs));
        if let Some(pipeline) = pipeline {
            map.insert("pipeline".to_string(), pipeline.clone());
        }
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
fn prompt_yes(cli: &Cli) -> bool {
    let api_base = &cli.api_base;
    if cli.benchmark_json {
        eprint!("Upload results to {api_base}? [Y/n] ");
        let _ = std::io::stderr().flush();
    } else {
        print!("Upload results to {api_base}? [Y/n] ");
        let _ = std::io::stdout().flush();
    }
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
            let body = resp.json::<Value>().await.ok();
            let msg = body
                .as_ref()
                .and_then(|v| v.get("message").and_then(Value::as_str))
                .unwrap_or("ok");
            // The id names this run in the stored corpus; print it so a
            // result can be found (or disqualified) later.
            match body
                .as_ref()
                .and_then(|v| v.get("benchmark_id").and_then(Value::as_u64))
            {
                Some(id) => say!(cli, "Upload accepted: {msg} (benchmark id {id})"),
                None => say!(cli, "Upload accepted: {msg}"),
            }
        }
        Ok(resp) => say!(cli, "Upload rejected ({}).", resp.status()),
        Err(e) => say!(cli, "Upload failed: {e}"),
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
        "cpu_simd": nice_common::stride_filter::simd_tier(),
        "mem_total_kb": parse_meminfo_total_kb(&meminfo),
        "arch": std::env::consts::ARCH,
        "os": std::env::consts::OS,
        "os_pretty": parse_os_pretty(&os_release),
        "gpu_model": gpu_name(cli, gpu),
        "gpu_backend": gpu_backend(cli, gpu),
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

/// The backend processing fields, as its `--gpu-backend` value — from the
/// live handle, since `auto` decides at init which compiled backend wins.
fn gpu_backend(cli: &Cli, gpu: &GpuCtx) -> Option<&'static str> {
    if !cli.gpu {
        return None;
    }
    gpu.as_ref().and_then(|handle| handle.backend_name())
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

    fn measure(setup_overlaps: bool) -> JoinMeasure {
        JoinMeasure {
            slices: 2,
            slice_weight: 2.0,
            partitions: 1000,
            slots: None,
            sampled: 100,
            setup_secs: 0.5,
            run_secs: 0.2,
            marginal_secs: None,
            survivors: 0,
            checked: 0,
            retried_partitions: 0,
            refused_layouts: 0,
            hits: 0,
            setup_overlaps,
        }
    }

    #[test]
    fn a_sample_scales_to_the_whole_field() {
        // 100 of 1000 partitions in 0.2 s: 2 s a slice, two slices. The CPU
        // adds each slice's setup; on a GPU the setup hides under the device
        // time, unless it is the longer of the two.
        assert!((measure(false).field_secs() - 2.0 * (0.5 + 2.0)).abs() < 1e-9);
        assert!((measure(true).field_secs() - 2.0 * 2.0).abs() < 1e-9);
        let slow_host = JoinMeasure {
            setup_secs: 3.0,
            ..measure(true)
        };
        assert!((slow_host.field_secs() - 2.0 * 3.0).abs() < 1e-9);
        // A sample of the whole slice is not scaled.
        let whole = JoinMeasure {
            sampled: 1000,
            ..measure(true)
        };
        assert!((whole.field_secs() - 2.0 * 0.5).abs() < 1e-9);
        // A last slice a quarter as wide adds a quarter of a slice.
        let short_last = JoinMeasure {
            slice_weight: 1.25,
            ..measure(true)
        };
        assert!((short_last.field_secs() - 1.25 * 2.0).abs() < 1e-9);
    }

    #[test]
    fn a_fixed_cost_per_sample_is_paid_once_per_slice() {
        // 0.05 s before the first batch, then 1.5 ms a partition: the
        // sample of 100 took 0.2 s. The slice pays the 0.05 s once and its
        // other 900 partitions 1.5 ms each, not 2 ms (the sample's average).
        let m = JoinMeasure {
            marginal_secs: Some(0.0015),
            ..measure(true)
        };
        assert!((m.slice_run_secs() - (0.2 + 900.0 * 0.0015)).abs() < 1e-9);
        assert!((m.field_secs() - 2.0 * 1.55).abs() < 1e-9);
        // A sample of the whole slice is its time either way.
        let whole = JoinMeasure {
            sampled: 1000,
            run_secs: 1.55,
            ..m
        };
        assert!((whole.slice_run_secs() - 1.55).abs() < 1e-9);
    }

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
        let stamped = field_telemetry(&base, 12.5, None);
        assert_eq!(stamped["processing_secs"], json!(12.5));
        assert_eq!(stamped["schema_version"], json!(TELEMETRY_SCHEMA_VERSION));
        // The base is not mutated; every field gets a fresh stamp.
        assert!(base.get("processing_secs").is_none());
    }

    #[test]
    fn environ_block_parses() {
        let vars = parse_environ(b"CONTAINER_ID=47102363\0VAST_CONTAINERLABEL=C.47102363\0BAD\0");
        assert_eq!(
            vars.get("CONTAINER_ID").map(String::as_str),
            Some("47102363")
        );
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
}
