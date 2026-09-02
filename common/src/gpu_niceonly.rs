//! Backend-neutral host pipeline for GPU niceonly fields.
//!
//! Both GPU backends run niceonly the same way: the CPU runs the real MSD
//! prefix filter across all cores with a coarser recursion floor than the CPU
//! client uses (see [`AdaptiveFloor`]), and ships only compact *range
//! descriptors* — 20 bytes per surviving range (offset, length, cross-end
//! certificate mask) — to the device, which
//! reconstructs the stride filter's candidates itself. No per-candidate data
//! ever crosses the bus.
//!
//! Everything in that sentence is independent of CUDA and Vulkan, so it lives
//! here rather than being written twice. The backends supply a [`RangeSink`]:
//! CUDA enqueues asynchronous launches on its stream, Vulkan records and
//! submits a dispatch. This is the same split as [`crate::gpu_config`], which
//! holds the per-base kernel constants for the same reason —
//! [`crate::client_process_cuda`] is `#![cfg(feature = "cuda")]` and unreachable
//! from a Vulkan-only build.

#![cfg(any(feature = "cuda", feature = "vulkan", feature = "cubecl"))]
#![allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]

use crate::{FieldResults, FieldSize, msd_prefix_filter, residue_filter};
use anyhow::Result;
use log::{debug, warn};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Mutex, OnceLock};
use std::time::Instant;

/// Numbers per MSD filter work unit handed to a CPU worker.
pub const PROCESSING_CHUNK_SIZE: u128 = 1_000_000;

/// LSD filter depth for the stride table, matching the CPU client's
/// `DEFAULT_LSD_K_VALUE` so GPU and CPU check the identical candidate set.
///
/// Upstream #88 raised this 2 → 3: the all-different check on 3+3 fixed low
/// digits removes 15-22% of stride candidates at production bases before any
/// per-candidate work, and the u32 residue/gap representation keeps the larger
/// table cheap. Both GPU backends upload the host-built table, so they inherit
/// the reduction without a kernel change — but the k=3 modulus is `b³`, which
/// is what `stride_modulus_fits_the_byte_horner_bound` in the Vulkan codegen
/// now has to hold against.
pub const GPU_LSD_K: u32 = 3;

/// Ranges buffered before each dispatch. Big enough to amortize submission and
/// upload overhead, small enough that dispatches start while the MSD workers
/// are still producing.
pub const LAUNCH_BATCH_RANGES: usize = 1 << 16;

/// Chunks' worth of descriptors allowed to queue between the MSD workers and
/// the consumer thread.
///
/// The consumer's cost per item is what makes this matter, and it differs by
/// backend. A CUDA launch is asynchronous, so that consumer never really
/// blocks and any bound is slack. A Vulkan dispatch blocks on a fence, so with
/// an unbounded channel the workers would race arbitrarily far ahead: a base-52
/// field at floor 250 is ~9e7 surviving ranges, and 20 bytes apiece is nearly two
/// gigabytes of queued descriptors. Bounding the channel keeps the overlap —
/// workers refill the queue while the consumer waits on the device.
///
/// **The unit here is one worker batch, not one launch batch.** Each item a
/// worker sends is a [`WorkerBatch`]: the output of several consecutive
/// chunks, flushed once it holds [`WORKER_BATCH_RANGES`] descriptors or
/// [`WORKER_BATCH_CHUNKS`] chunks' worth. [`LAUNCH_BATCH_RANGES`] is the
/// consumer's flush threshold and never bounds what sits in the channel. A
/// batch is at most `WORKER_BATCH_RANGES` plus one chunk's output — the
/// recursion returns a range whole once it is at or below the floor, so a chunk
/// yields about `PROCESSING_CHUNK_SIZE / floor` ranges, ~4000 at
/// [`MSD_FLOOR_MIN`] — call it 8200 descriptors, 20 bytes apiece. So the cap
/// is about 10 MB of queued descriptors at the worst floor, comfortably below
/// the gigabyte above.
const PIPELINE_DEPTH: usize = 64;

/// Descriptors a worker accumulates before sending one batch to the consumer.
///
/// Workers used to send every chunk's output as its own message. That is one
/// channel operation per chunk, and with many producers hammering a bounded
/// channel the cost is dominated by parking and waking threads rather than by
/// the MSD work: at the no-MSD bypass floor (one descriptor per chunk) the same
/// 1e6 chunks took 0.22 s on one thread, 0.96 s on six and 1.6 s on twelve, and
/// on Anvil's 32-core node a 1e13 field spent 36 s producing 1e7 descriptors
/// the device then checked in well under a second. Batching cuts the message
/// count by two to three orders of magnitude at every floor.
const WORKER_BATCH_RANGES: usize = 4096;

/// Chunks a worker folds into one batch before sending, whatever its size.
///
/// Bounds the latency of a batch at coarse floors, where chunks yield one
/// descriptor each and [`WORKER_BATCH_RANGES`] alone would hold back 4096
/// chunks' worth of device work.
const WORKER_BATCH_CHUNKS: usize = 256;

/// Minimum MSD recursion floor (matches the CPU client's default).
/// Below this the GPU receives virtually the same candidates as the CPU would
/// check itself, so there is no point going lower.
const MSD_FLOOR_MIN: f64 = 250.0;

/// Maximum MSD recursion floor the adaptive controller may reach: half a
/// [`PROCESSING_CHUNK_SIZE`] chunk, i.e. one level of subdivision below the
/// whole-chunk check ([`msd_prefix_filter::MSD_RECURSIVE_SUBDIVISION_FACTOR`]
/// is 2).
///
/// The controller used to be allowed all the way up to one whole chunk, the
/// explicit no-MSD bypass ([`descriptors_for_chunk`] ships every chunk as one
/// descriptor with no endpoint analysis), on the theory that a strong device
/// paired with a weak CPU would want it. In practice the controller reached
/// it on a strong CPU too and stuck there: its only signal is the device
/// *tail* after the workers finish, which is small whenever the device keeps
/// pace, so the floor ratchets up every field until nothing pulls it back.
///
/// The bypass itself is still there for the one configuration it was meant
/// for: pin it explicitly with `NICE_GPU_MSD_FLOOR=1000000`, which bypasses
/// this clamp.
///
/// Survival data (b52, per 1e12, single core):
///
/// | floor  | CPU time | surviving |
/// |--------|----------|-----------|
/// | 250    | 350 s    | 2.3 %     |
/// | 4 000  | 50 s     | 15.2 %    |
/// | 16 000 | 15 s     | 19.0 %    |
/// | 64 000 | 4.8 s    | 22.6 %    |
#[allow(clippy::cast_precision_loss)]
const MSD_FLOOR_MAX: f64 = (PROCESSING_CHUNK_SIZE / 2) as f64;

/// Adaptive MSD recursion floor for the niceonly GPU pipeline.
///
/// Goal: keep `msd_time ≈ gpu_tail_time` so the overlapped pipeline is
/// balanced.  The floor is seeded from the CPU count (fewer cores → coarser
/// floor, because MSD is the bottleneck) and then nudged ≤ 1.5× per field
/// toward that balance.  Setting `NICE_GPU_MSD_FLOOR` in the environment
/// pins the floor and disables adaptation.
struct AdaptiveFloor {
    floor: f64,
    /// Fields remaining in warmup (skip adaptation); `u32::MAX` = permanently
    /// fixed via env-var override.
    warmup: u32,
}

/// Fields to observe before adapting, so shader/kernel JIT one-time costs
/// don't skew the first measurement.
const ADAPT_WARMUP: u32 = 3;

/// Maximum multiplicative step per field in either direction.
const ADAPT_MAX_STEP: f64 = 1.5;

/// Ignore a phase if it took less than this many seconds — the measurement
/// noise would dominate the ratio.
const ADAPT_MIN_SECS: f64 = 0.002;

/// Floor value calibrated for 32 cores. Derived value for N cores:
/// `ADAPT_BASE_CORE_PRODUCT / N`, clamped to `[MSD_FLOOR_MIN, MSD_FLOOR_MAX]`.
const ADAPT_BASE_CORE_PRODUCT: f64 = 512_000.0;

impl AdaptiveFloor {
    fn current(&self) -> u128 {
        self.floor as u128
    }

    fn update(&mut self, msd_secs: f64, total_secs: f64) {
        if self.warmup == u32::MAX {
            return;
        }
        if self.warmup > 0 {
            self.warmup -= 1;
            return;
        }
        // `msd_secs` already has the sink's time subtracted, so this is exactly
        // `NiceonlyStats::device_secs` — the device's share, on either backend.
        let gpu_tail = (total_secs - msd_secs).max(0.0);
        let ratio = if gpu_tail < ADAPT_MIN_SECS {
            ADAPT_MAX_STEP
        } else if msd_secs < ADAPT_MIN_SECS {
            1.0 / ADAPT_MAX_STEP
        } else {
            msd_secs / gpu_tail
        };
        let factor = ratio.clamp(1.0 / ADAPT_MAX_STEP, ADAPT_MAX_STEP);
        let new_floor = (self.floor * factor).clamp(MSD_FLOOR_MIN, MSD_FLOOR_MAX);
        if (new_floor - self.floor).abs() > self.floor * 0.05 {
            debug!(
                "GPU MSD floor: {:.0} → {:.0} (msd {:.3}s, gpu_tail {:.3}s)",
                self.floor, new_floor, msd_secs, gpu_tail,
            );
        }
        self.floor = new_floor;
    }
}

static ADAPTIVE_FLOOR: OnceLock<Mutex<AdaptiveFloor>> = OnceLock::new();

fn adaptive_floor() -> &'static Mutex<AdaptiveFloor> {
    ADAPTIVE_FLOOR.get_or_init(|| {
        if let Ok(v) = std::env::var("NICE_GPU_MSD_FLOOR") {
            match v.parse::<f64>() {
                Ok(f) if f >= 1.0 => {
                    debug!("GPU MSD floor fixed at {f:.0} via NICE_GPU_MSD_FLOOR");
                    return Mutex::new(AdaptiveFloor {
                        floor: f,
                        warmup: u32::MAX,
                    });
                }
                _ => warn!("ignoring invalid NICE_GPU_MSD_FLOOR '{v}'; using adaptive floor"),
            }
        }
        #[allow(clippy::cast_precision_loss)]
        let cpu_count =
            std::thread::available_parallelism().map_or(32, std::num::NonZeroUsize::get) as f64;
        let seed = (ADAPT_BASE_CORE_PRODUCT / cpu_count).clamp(MSD_FLOOR_MIN, MSD_FLOOR_MAX);
        debug!("GPU MSD floor: adaptive, seed {seed:.0} ({cpu_count:.0} logical cores)");
        Mutex::new(AdaptiveFloor {
            floor: seed,
            warmup: ADAPT_WARMUP,
        })
    })
}

fn gpu_msd_floor() -> u128 {
    adaptive_floor().lock().unwrap().current()
}

/// The MSD floor a `--benchmark` run pins: the controller's cap.
///
/// A benchmark has to be comparable across machines and runs, and the
/// adaptive controller is neither — it is process-global state that moves
/// after every field, so with 4-8e9 windows it ratchets hundreds of times
/// inside one sweep and every scenario's rate depends on where the previous
/// ones left it (an A100 scored below an RTX 3060 this way). The cap is where
/// a strong device settles in production, and it puts the benchmark's weight
/// on the device rather than on this box's MSD cores, which is what a GPU
/// benchmark is for.
#[allow(clippy::cast_precision_loss)]
pub const BENCHMARK_MSD_FLOOR: u128 = MSD_FLOOR_MAX as u128;

/// Pin the MSD floor to [`BENCHMARK_MSD_FLOOR`] for the rest of the process.
///
/// Call before any niceonly field is processed. An explicit
/// `NICE_GPU_MSD_FLOOR` still wins, so floor sweeps under `--benchmark`
/// remain possible; that case just initialises the controller as usual. If
/// the controller was already initialised the pin is refused with a warning
/// rather than silently benchmarking a moving floor.
pub fn pin_msd_floor_for_benchmark() {
    if std::env::var_os("NICE_GPU_MSD_FLOOR").is_some() {
        adaptive_floor();
        return;
    }
    #[allow(clippy::cast_precision_loss)]
    let pinned = Mutex::new(AdaptiveFloor {
        floor: BENCHMARK_MSD_FLOOR as f64,
        warmup: u32::MAX,
    });
    if ADAPTIVE_FLOOR.set(pinned).is_err() {
        warn!("GPU MSD floor already in use; benchmark cannot pin it at {BENCHMARK_MSD_FLOOR}");
    } else {
        debug!("GPU MSD floor pinned at {BENCHMARK_MSD_FLOOR} for the benchmark");
    }
}

/// The MSD floor currently in force, for reports. Initialises the controller
/// if nothing has yet.
#[must_use]
pub fn msd_floor_in_use() -> u128 {
    gpu_msd_floor()
}

/// Per-field statistics from the overlapped niceonly pipeline.
pub struct NiceonlyStats {
    /// CPU time in the MSD filter, with time spent inside the sink removed.
    ///
    /// Not simply "wall time until the workers finished". On a backend whose
    /// `launch` returns immediately (CUDA) the two are the same, but Vulkan
    /// blocks on a fence inside `launch`, so that wall time would charge every
    /// dispatch to the MSD phase — and [`AdaptiveFloor`] would then see a
    /// device that costs nothing and raise the floor until the GPU was doing
    /// all the work at the worst possible granularity. Measured: on a 1e13
    /// base-50 field, floor 4000 finishes in 116 s and floor 256 000 in over
    /// ten minutes, so drifting upward is not a small mistake.
    pub msd_secs: f64,
    /// Wall time in `launch` + `sync`, i.e. everything the device cost that the
    /// host could observe.
    pub device_secs: f64,
    /// Wall time until every dispatch had completed on the device.
    pub total_secs: f64,
    pub num_ranges: usize,
    pub valid_numbers: u64,
    pub launches: u32,
}

/// A backend's device-side end of the pipeline.
///
/// The pipeline hands over batches of MSD-surviving range descriptors and, at
/// the end, waits for the device. Collecting the results is deliberately *not*
/// part of this trait: the two backends read their nice numbers out of very
/// different buffers, and doing it after [`run_range_pipeline`] returns keeps
/// the object-safe surface down to the two calls the pipeline actually makes.
pub trait RangeSink {
    /// Dispatch one batch. `offsets` are relative to the field start, `lens`
    /// are candidate counts, and `masks` are the ranges' cross-end
    /// certificates (digits certainly occupying output positions >= k for
    /// every n in the range — see `msd_prefix_filter::MsdAnalysis`); the
    /// three slices are the same length. A backend that has no device-side
    /// mask test yet may ignore `masks` — the filter is an optimization,
    /// never required for correctness.
    ///
    /// # Errors
    /// Returns an error on any device failure.
    fn launch(&mut self, offsets: &[u64], lens: &[u32], masks: &[u64]) -> Result<()>;

    /// Wait for everything launched so far to finish on the device. Called
    /// once, inside the timed region, so `total_secs` covers real device work
    /// on a backend whose launches are asynchronous.
    ///
    /// # Errors
    /// Returns an error on any device failure.
    fn sync(&mut self) -> Result<()> {
        Ok(())
    }
}

/// MSD-filter one chunk into descriptors relative to `field_start`.
///
/// Each surviving range becomes 20 bytes: a u64 offset, a u32 length, and a
/// u64 cross-end certificate mask. That
/// encoding, not the filter, is what bounds a range — a field is at most 1e12
/// numbers so the offset always fits, but a range longer than `u32::MAX` would
/// not, which is why this can fail.
fn descriptors_for_chunk(
    chunk: FieldSize,
    base: u32,
    floor: u128,
    field_start: u128,
) -> Result<(Vec<u64>, Vec<u32>, Vec<u64>)> {
    let mut offsets: Vec<u64> = Vec::new();
    let mut lens: Vec<u32> = Vec::new();
    let mut masks: Vec<u64> = Vec::new();
    if floor >= PROCESSING_CHUNK_SIZE {
        // Explicit no-MSD bypass: the whole chunk as one descriptor, no
        // endpoint analysis and no certificate. The device still applies
        // the stride table; it just checks every stride candidate.
        let offset = u64::try_from(chunk.start() - field_start);
        let len = u32::try_from(chunk.size());
        match (offset, len) {
            (Ok(offset), Ok(len)) => {
                offsets.push(offset);
                lens.push(len);
                masks.push(0);
            }
            _ => anyhow::bail!(
                "chunk doesn't fit descriptor: start {} size {}",
                chunk.start(),
                chunk.size()
            ),
        }
        return Ok((offsets, lens, masks));
    }
    let mut leaves: Vec<(FieldSize, u64)> = Vec::new();
    msd_prefix_filter::get_valid_ranges_recursive_masked(
        chunk,
        &msd_prefix_filter::MaskedRecursion {
            base,
            fixed_lsd_k: GPU_LSD_K as usize,
            max_depth: msd_prefix_filter::MSD_RECURSIVE_MAX_DEPTH,
            min_range_size: floor,
            subdivision_factor: msd_prefix_filter::MSD_RECURSIVE_SUBDIVISION_FACTOR,
        },
        0,
        0,
        &mut leaves,
    );
    for (sub, mask) in leaves {
        let offset = u64::try_from(sub.start() - field_start);
        let len = u32::try_from(sub.size());
        match (offset, len) {
            (Ok(offset), Ok(len)) => {
                offsets.push(offset);
                lens.push(len);
                masks.push(mask);
            }
            _ => anyhow::bail!(
                "valid range doesn't fit descriptor: start {} size {}",
                sub.start(),
                sub.size()
            ),
        }
    }
    Ok((offsets, lens, masks))
}

/// Descriptors one MSD worker has accumulated since its last send.
#[derive(Default)]
struct WorkerBatch {
    offsets: Vec<u64>,
    lens: Vec<u32>,
    masks: Vec<u64>,
    chunks: usize,
}

impl WorkerBatch {
    /// Fold one chunk's descriptors in. Empty chunks still count toward the
    /// chunk bound so a run of rejected chunks cannot delay a pending batch.
    fn absorb(&mut self, offsets: &[u64], lens: &[u32], masks: &[u64]) {
        self.offsets.extend_from_slice(offsets);
        self.lens.extend_from_slice(lens);
        self.masks.extend_from_slice(masks);
        self.chunks += 1;
    }

    fn is_ready(&self) -> bool {
        self.offsets.len() >= WORKER_BATCH_RANGES || self.chunks >= WORKER_BATCH_CHUNKS
    }

    fn is_empty(&self) -> bool {
        self.offsets.is_empty()
    }

    /// Hand the accumulated descriptors over and start a fresh batch.
    fn take(&mut self) -> (Vec<u64>, Vec<u32>, Vec<u64>) {
        let batch = std::mem::take(self);
        (batch.offsets, batch.lens, batch.masks)
    }
}

/// Run one niceonly field: MSD workers stream surviving-range descriptors
/// through a channel while the calling thread batches them into dispatches, so
/// the CPU filter and the device checks overlap instead of running as
/// sequential phases.
///
/// **Range semantics**: half-open [`range_start`, `range_end`).
///
/// # Errors
/// Returns an error if a descriptor does not fit its 12-byte encoding, or on
/// any device failure reported by the sink.
///
/// # Panics
/// Panics if an MSD worker panicked while holding the error slot.
pub fn run_range_pipeline<S: RangeSink>(
    sink: &mut S,
    range: &FieldSize,
    base: u32,
) -> Result<NiceonlyStats> {
    let start_time = Instant::now();
    let chunks = range.chunks(PROCESSING_CHUNK_SIZE);
    let floor = gpu_msd_floor();
    let num_threads = std::thread::available_parallelism()
        .map_or(4, std::num::NonZeroUsize::get)
        .min(chunks.len().max(1));

    let next_chunk = AtomicUsize::new(0);
    let worker_error: Mutex<Option<anyhow::Error>> = Mutex::new(None);
    let (tx, rx) = std::sync::mpsc::sync_channel::<(Vec<u64>, Vec<u32>, Vec<u64>)>(PIPELINE_DEPTH);

    let mut stats = NiceonlyStats {
        msd_secs: 0.0,
        device_secs: 0.0,
        total_secs: 0.0,
        num_ranges: 0,
        valid_numbers: 0,
        launches: 0,
    };
    let mut buf_offsets: Vec<u64> = Vec::new();
    let mut buf_lens: Vec<u32> = Vec::new();
    let mut buf_masks: Vec<u64> = Vec::new();
    let mut launch_error: Option<anyhow::Error> = None;

    std::thread::scope(|scope| {
        let chunks = &chunks;
        let next_chunk = &next_chunk;
        let worker_error = &worker_error;
        for _ in 0..num_threads {
            let tx = tx.clone();
            scope.spawn(move || {
                let mut batch = WorkerBatch::default();
                loop {
                    let i = next_chunk.fetch_add(1, Ordering::Relaxed);
                    let Some(chunk) = chunks.get(i) else { break };
                    match descriptors_for_chunk(*chunk, base, floor, range.start()) {
                        Ok((offsets, lens, masks)) => {
                            batch.absorb(&offsets, &lens, &masks);
                        }
                        Err(e) => {
                            *worker_error.lock().unwrap() = Some(e);
                            return;
                        }
                    }
                    if batch.is_ready() {
                        if batch.is_empty() {
                            // A run of rejected chunks: nothing to send, but
                            // start the chunk count over.
                            batch = WorkerBatch::default();
                        } else if tx.send(batch.take()).is_err() {
                            // A closed channel means the consumer gave up on a
                            // launch error; the remaining chunks are moot.
                            return;
                        }
                    }
                }
                if !batch.is_empty() {
                    // Nothing to do about a closed channel here either.
                    let _ = tx.send(batch.take());
                }
            });
        }
        // The consumer runs on this thread while the workers produce. The
        // clone of `tx` held by each worker keeps the channel open; dropping
        // ours lets `recv` disconnect once they all finish.
        drop(tx);

        while let Ok((offsets, lens, masks)) = rx.recv() {
            stats.num_ranges += offsets.len();
            stats.valid_numbers += lens.iter().map(|&l| u64::from(l)).sum::<u64>();
            buf_offsets.extend_from_slice(&offsets);
            buf_lens.extend_from_slice(&lens);
            buf_masks.extend_from_slice(&masks);
            if buf_offsets.len() >= LAUNCH_BATCH_RANGES {
                let t = Instant::now();
                let outcome = sink.launch(&buf_offsets, &buf_lens, &buf_masks);
                stats.device_secs += t.elapsed().as_secs_f64();
                if let Err(e) = outcome {
                    launch_error = Some(e);
                    break;
                }
                stats.launches += 1;
                buf_offsets.clear();
                buf_lens.clear();
                buf_masks.clear();
            }
        }
        // Break out of the consume loop and the workers are still producing
        // into a *bounded* channel. Dropping the receiver here is what unblocks
        // the ones already parked in `send`; without it the scope below would
        // wait for threads that are waiting for us, forever.
        drop(rx);

        // Workers are done (or the launch failed); either way this marks the
        // end of the CPU-side phase. Time already spent inside the sink is not
        // MSD time, however synchronous that sink happens to be.
        stats.msd_secs = (start_time.elapsed().as_secs_f64() - stats.device_secs).max(0.0);
    });

    if let Some(e) = launch_error {
        return Err(e);
    }
    if let Some(e) = worker_error.into_inner().unwrap() {
        return Err(e);
    }
    let tail = Instant::now();
    if !buf_offsets.is_empty() {
        sink.launch(&buf_offsets, &buf_lens, &buf_masks)?;
        stats.launches += 1;
    }
    sink.sync()?;
    stats.device_secs += tail.elapsed().as_secs_f64();
    stats.total_secs = start_time.elapsed().as_secs_f64();

    debug!(
        "GPU niceonly pipeline b{base}: {} ranges in {} dispatches, {} candidates",
        stats.num_ranges, stats.launches, stats.valid_numbers,
    );
    Ok(stats)
}

/// Log the per-field summary and feed the measurement back into the adaptive
/// MSD floor. Both backends report the same line.
///
/// # Panics
/// Panics if the adaptive-floor mutex was poisoned by an earlier panic.
#[allow(clippy::cast_precision_loss)]
pub fn report_field(backend: &str, base: u32, range: &FieldSize, stats: &NiceonlyStats) {
    debug!(
        "{backend} niceonly b{base}: msd {:.3}s -> {} ranges ({:.2}% of field), gpu {:.3}s, total {:.3}s, {:.2e} n/s overall",
        stats.msd_secs,
        stats.num_ranges,
        100.0 * stats.valid_numbers as f64 / range.size() as f64,
        stats.device_secs,
        stats.total_secs,
        range.size() as f64 / stats.total_secs,
    );
    adaptive_floor()
        .lock()
        .unwrap()
        .update(stats.msd_secs, stats.total_secs);
}

/// The answer for a residue-empty base, if this is one.
///
/// For `b ≡ 3 mod 4` the residue set `R_b` is empty, which means there are
/// provably no solutions — but it also means the stride table does not return
/// so much as panic when indexed
/// (`stride_filter::first_valid_at_or_after` indexes `valid_residues[idx]`).
/// So this has to be checked before any stride table is built.
///
/// The CUDA path has the same guard inside `process_range_niceonly_cuda`; the
/// Vulkan and `CubeCL` paths call this ahead of their CPU fallbacks, so it
/// also covers bases the GPU itself cannot take.
#[must_use]
pub fn residue_empty_result(base: u32) -> Option<FieldResults> {
    if residue_filter::get_residue_filter_u128(&base).is_empty() {
        debug!("base {base} is residue-empty; no candidates to check");
        return Some(FieldResults {
            distribution: Vec::new(),
            nice_numbers: Vec::new(),
        });
    }
    None
}

// ---------------------------------------------------------------------------
// Kernel-shape constants shared by the range-descriptor backends (Vulkan and
// CubeCL). CUDA predates the descriptor pipeline's tiling and keeps its fixed
// one-warp-per-range shape, so only the newer backends read these.
// ---------------------------------------------------------------------------

/// Most threads that may cooperate on one MSD-valid range — CUDA's
/// one-warp-per-range tiling, and the ceiling for [`lane_shift_for`].
///
/// Nothing here is a hardware property: the lanes stride through the range's
/// candidates by index and never communicate, so this is a tiling constant, not
/// a subgroup width. Which is exactly why it does not have to be a constant at
/// all — the kernels take `log2(lanes)` as a launch parameter and the host
/// picks it per dispatch.
pub const MAX_LANES_PER_RANGE: u32 = 32;

/// Candidates each lane should have to work on, which is what
/// [`lane_shift_for`] sizes the tiling to deliver.
///
/// Every lane assigned to a range redundantly repeats that range's setup — the
/// residue reduction and a ~12-iteration binary search over the residue table —
/// before its first candidate. At CUDA's fixed 32 lanes and an MSD floor of 250,
/// a base-40 range holds ~39 candidates, so the tiling buys 32 copies of that
/// setup to share out **1.2 candidates per lane**.
///
/// Measured (b40, 1e12, floor 250, device time): 32 lanes 30.0 s, 16 24.6 s,
/// 8 22.8 s, 4 21.6 s, 2 21.1 s, 1 21.1 s. Monotone, and 32 candidates per lane
/// is what puts a ~39-candidate range on a single lane. Where ranges are long
/// the same sweep is flat inside run-to-run variance (floor 4000: 21.1-21.6 s
/// across every width; floor 32000: 24.9-25.9 s), so this only has to be right
/// at the short-range end.
const TARGET_CANDIDATES_PER_LANE: u64 = 32;

/// Threads a dispatch should have before the tiling starts economizing on them.
///
/// The floor is for small batches — the last of a field, or a field whose whole
/// MSD output is a few thousand ranges. 65536 is measured to saturate this
/// device (at floor 250 a 65536-range batch at one lane apiece is the fastest
/// setting there is), so it is a lower bound rather than a target; on a device
/// with 30x the ALUs the [`MAX_LANES_PER_RANGE`] cap binds first anyway.
const MIN_DISPATCH_THREADS: u64 = 1 << 16;

/// `log2` of the lanes to assign per range, for a dispatch of `num_ranges`
/// ranges averaging `mean_len` numbers, of which `stride_r / stride_m` are
/// candidates.
///
/// Clamped to `[1, MAX_LANES_PER_RANGE]` lanes. Returning a shift rather than a
/// count keeps the kernel's `gid >> shift` / `gid & (lanes - 1)` split exact,
/// so the tiling stays pure index arithmetic at any width.
#[must_use]
pub fn lane_shift_for(num_ranges: u64, mean_len: u64, stride_m: u32, stride_r: u32) -> u32 {
    let candidates = mean_len * u64::from(stride_r) / u64::from(stride_m);
    // Round down to a power of two: 63 candidates' worth of lanes is 4, not 8,
    // because the last lane would otherwise idle through most of the range.
    let by_work =
        (candidates / TARGET_CANDIDATES_PER_LANE).clamp(1, u64::from(MAX_LANES_PER_RANGE));
    // ...but never leave the device short of threads to hide latency behind.
    let by_occupancy = MIN_DISPATCH_THREADS
        .div_ceil(num_ranges.max(1))
        .next_power_of_two()
        .min(u64::from(MAX_LANES_PER_RANGE));
    by_work.max(by_occupancy).ilog2()
}

/// Largest stride modulus the descriptor kernels' residue reduction accepts.
///
/// `n mod M` cannot be computed as a 64-bit division by a constant — that is
/// the one construct RADV/ACO does not strength-reduce (see the Vulkan module
/// docs). Instead the kernel reduces the range's 64-bit *offset* one chunk at
/// a time, `acc = (acc << c | chunk) % M`, with `M` a 32-bit compile-time
/// constant. The running remainder satisfies `acc < M`, so the shift stays
/// inside a u32 exactly while `M <= 2^(32-c)`.
///
/// `c` is picked per base by [`stride_chunk_bits`]. It used to be a fixed 8,
/// on the premise that `M = (b-1)·b^k` with `k = 2` put even base 128 at
/// 127·16384 ≈ 2^21. Upstream #88 raised `k` to 3, which multiplies every
/// modulus by `b`: base 65 reaches 17 576 000 and base 128 reaches
/// 266 338 304, so the fixed byte chunk would have refused every base ≥ 65 —
/// base 80 among them. A 4-bit chunk covers the whole supported range with
/// room to spare, and costs nothing measurable because this reduction runs
/// once per *range descriptor*, not per candidate.
pub const MAX_STRIDE_MODULUS: u128 = 1 << 28;

/// Width in bits of one Horner chunk in the kernels' offset reduction.
///
/// The largest `c` with `M << c` still inside a u32, restricted to widths that
/// divide 64 evenly so the unrolled loop covers the offset exactly. Both the
/// device kernels and the host mirror derive `c` from the same modulus, so
/// they cannot disagree.
#[must_use]
pub fn stride_chunk_bits(stride_m: u32) -> u32 {
    if u128::from(stride_m) <= 1 << 24 {
        8
    } else {
        4
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::sync::mpsc;
    use std::time::Duration;

    /// A sink that records what it was handed instead of touching a device.
    #[derive(Default)]
    struct Recorder {
        batches: Vec<(Vec<u64>, Vec<u32>, Vec<u64>)>,
        synced: bool,
    }

    impl RangeSink for Recorder {
        fn launch(&mut self, offsets: &[u64], lens: &[u32], masks: &[u64]) -> Result<()> {
            self.batches
                .push((offsets.to_vec(), lens.to_vec(), masks.to_vec()));
            Ok(())
        }
        fn sync(&mut self) -> Result<()> {
            self.synced = true;
            Ok(())
        }
    }

    /// A sink that fails the way a device does.
    struct Failing;

    impl RangeSink for Failing {
        fn launch(&mut self, _offsets: &[u64], _lens: &[u32], _masks: &[u64]) -> Result<()> {
            anyhow::bail!("simulated device failure")
        }
    }

    /// A failing sink must make the pipeline *return*, not wedge it.
    ///
    /// The consumer breaks out of its loop on a launch error while the MSD
    /// workers are still producing into a bounded channel. Whoever is parked in
    /// `send` at that moment only wakes when the receiver is dropped, and
    /// `thread::scope` will not return until they do — so getting this wrong is
    /// a deadlock, not a leak. The field has to be big enough to reach a flush
    /// (`LAUNCH_BATCH_RANGES`) and then fill `PIPELINE_DEPTH` worker batches
    /// (`WORKER_BATCH_RANGES` each) behind it, which is why this is not a toy
    /// range.
    ///
    /// Run under a timeout so a regression fails the test instead of hanging
    /// the suite.
    #[test]
    fn a_failing_sink_unblocks_the_workers_instead_of_deadlocking() {
        let (tx, rx) = mpsc::channel();
        std::thread::spawn(move || {
            // Base 40 from its band start: ~2e6 surviving ranges over 5e11 at
            // the seeded floor, i.e. hundreds of worker batches, so the
            // consumer reaches its first flush with far more than
            // `PIPELINE_DEPTH` batches still to come. Base 50 is no good here
            // — MSD prunes its band start to nothing, and nothing is ever
            // launched.
            let base = 40;
            let start = crate::base_range::get_base_range_u128(base)
                .unwrap()
                .unwrap()
                .range_start;
            let field = FieldSize::new(start, start + 500_000_000_000);
            let _ = tx.send(run_range_pipeline(&mut Failing, &field, base).is_err());
        });
        let errored = rx
            .recv_timeout(Duration::from_mins(2))
            .expect("run_range_pipeline did not return: the workers are deadlocked");
        assert!(errored, "the launch failure must reach the caller");
    }

    /// The controller's ratchet must stop short of the no-MSD bypass, however
    /// many fields report a negligible device tail. Only an explicit
    /// `NICE_GPU_MSD_FLOOR` pin may reach it.
    #[test]
    fn adaptive_floor_never_ratchets_into_the_bypass() {
        let mut floor = AdaptiveFloor {
            floor: ADAPT_BASE_CORE_PRODUCT / 32.0,
            warmup: 0,
        };
        for _ in 0..100 {
            // A field where the workers took a while and the device tail was
            // nothing: the strongest possible "raise the floor" signal.
            floor.update(5.0, 5.0);
        }
        #[allow(clippy::cast_precision_loss)]
        let bypass = PROCESSING_CHUNK_SIZE as f64;
        assert!(
            floor.floor < bypass,
            "floor {} reached the bypass",
            floor.floor
        );
        assert!((floor.floor - MSD_FLOOR_MAX).abs() < f64::EPSILON);
        // And the clamp is a real cap, not a coincidence of the step size.
        floor.floor = MSD_FLOOR_MAX * 0.99;
        floor.update(5.0, 5.0);
        assert!((floor.floor - MSD_FLOOR_MAX).abs() < f64::EPSILON);
    }

    /// A worker batch flushes on either bound and never sends an empty one.
    #[test]
    fn worker_batch_flushes_on_either_bound() {
        // Descriptor bound: one big chunk's worth tips it over.
        let mut batch = WorkerBatch::default();
        let big = vec![0u64; WORKER_BATCH_RANGES - 1];
        batch.absorb(&big, &vec![1u32; big.len()], &vec![0u64; big.len()]);
        assert!(!batch.is_ready());
        batch.absorb(&[7], &[1], &[0]);
        assert!(batch.is_ready());
        let (offsets, lens, masks) = batch.take();
        assert_eq!(offsets.len(), WORKER_BATCH_RANGES);
        assert_eq!(lens.len(), WORKER_BATCH_RANGES);
        assert_eq!(masks.len(), WORKER_BATCH_RANGES);
        assert_eq!(*offsets.last().unwrap(), 7);
        // `take` leaves a fresh batch behind.
        assert!(batch.is_empty() && !batch.is_ready());

        // Chunk bound: bypass-style single descriptors, 256 of them.
        let mut batch = WorkerBatch::default();
        for i in 0..WORKER_BATCH_CHUNKS {
            assert!(!batch.is_ready(), "ready after only {i} chunks");
            batch.absorb(&[i as u64], &[1], &[0]);
        }
        assert!(batch.is_ready());
        assert_eq!(batch.take().0.len(), WORKER_BATCH_CHUNKS);

        // Rejected chunks count toward the chunk bound but leave it empty, so
        // the worker resets it instead of sending nothing.
        let mut batch = WorkerBatch::default();
        for _ in 0..WORKER_BATCH_CHUNKS {
            batch.absorb(&[], &[], &[]);
        }
        assert!(batch.is_ready() && batch.is_empty());
    }

    /// At the bypass floor, a chunk becomes exactly one descriptor with no
    /// certificate; below it, descriptors match the masked recursion.
    #[test]
    fn bypass_floor_emits_whole_chunks() {
        let base = 40;
        let range = crate::base_range::get_base_range_u128(base)
            .unwrap()
            .unwrap();
        let start = range.start();
        // Mid-range chunk: the band start is MSD-strong and can reject a
        // whole chunk, which would make the certificate assertions vacuous.
        let mid = start + (range.end() - start) / 2;
        let chunk = FieldSize::new(mid, mid + PROCESSING_CHUNK_SIZE);
        let (offsets, lens, masks) =
            descriptors_for_chunk(chunk, base, PROCESSING_CHUNK_SIZE, start).unwrap();
        assert_eq!(offsets, vec![u64::try_from(mid - start).unwrap()]);
        assert_eq!(lens, vec![PROCESSING_CHUNK_SIZE as u32]);
        assert_eq!(masks, vec![0]);

        // A sub-bypass floor produces the masked recursion's leaves.
        let (offsets, lens, masks) = descriptors_for_chunk(chunk, base, 8000, start).unwrap();
        let leaves = msd_prefix_filter::get_valid_ranges_masked(chunk, base, GPU_LSD_K as usize);
        assert_eq!(offsets.len(), leaves.len());
        assert_eq!(masks.len(), leaves.len());
        for (i, (leaf, mask)) in leaves.iter().enumerate() {
            assert_eq!(u128::from(offsets[i]), leaf.start() - start);
            assert_eq!(u128::from(lens[i]), leaf.size());
            assert_eq!(masks[i], *mask);
        }
        assert!(masks.iter().any(|&m| m != 0), "expected live certificates");
    }

    /// The descriptors the pipeline emits must cover every candidate the CPU
    /// stride iteration would visit. The MSD floor makes the GPU's set a
    /// *superset* (coarser pruning is still sound), so this checks containment
    /// rather than equality — which is exactly the property that makes the two
    /// paths find the same nice numbers.
    #[test]
    fn emitted_ranges_cover_every_stride_candidate() {
        use crate::stride_filter::StrideTable;

        let base = 10;
        let range = crate::base_range::get_base_range_u128(base)
            .unwrap()
            .unwrap();
        let field = FieldSize::new(range.range_start, range.range_end);

        let mut sink = Recorder::default();
        let stats = run_range_pipeline(&mut sink, &field, base).expect("pipeline");
        assert!(sink.synced, "the pipeline must sync the sink");
        assert_eq!(
            stats.num_ranges,
            sink.batches.iter().map(|(o, _, _)| o.len()).sum::<usize>()
        );

        let mut covered: Vec<(u128, u128)> = Vec::new();
        for (offsets, lens, masks) in &sink.batches {
            assert_eq!(offsets.len(), masks.len());
            for (&o, &l) in offsets.iter().zip(lens) {
                let s = field.start() + u128::from(o);
                covered.push((s, s + u128::from(l)));
            }
        }

        let table = StrideTable::new(base, GPU_LSD_K);
        let (mut n, mut idx) = table.first_valid_at_or_after(field.start());
        let mut checked = 0;
        while n < field.end() {
            assert!(
                covered.iter().any(|&(s, e)| n >= s && n < e),
                "candidate {n} is in no emitted range"
            );
            checked += 1;
            n += u128::from(table.gap_table[idx]);
            idx = (idx + 1) % table.gap_table.len();
        }
        assert!(checked > 0, "base {base} produced no candidates to check");
        // 69 is the one nice number in base 10, so it had better be in there.
        assert!(covered.iter().any(|&(s, e)| 69 >= s && 69 < e));
    }
}
