//! Backend-neutral host pipeline for GPU niceonly fields.
//!
//! Every GPU backend runs niceonly the same way: the CPU runs the real MSD
//! prefix filter across all cores with a coarser recursion floor than the CPU
//! client uses (see [`FloorController`]), and ships only compact *range
//! descriptors* — 20 bytes per surviving range (offset, length, cross-end
//! certificate mask) — to the device, which reconstructs the stride filter's
//! candidates itself. No per-candidate data ever crosses the bus.
//!
//! The pipeline is continuous across fields ([`NiceonlyPipeline`]): the MSD
//! workers start on the next field while the device is still draining the
//! previous one, and the device never waits for a field boundary either. The
//! floor is steered by which side is behind — see [`FloorController`] — so
//! neither side idles in steady state. [`run_range_pipeline`] is the one-field
//! synchronous form of the same machinery, for backends whose device handle
//! cannot leave the calling thread and for tests.
//!
//! Everything here is independent of the device API, so it lives here rather
//! than being written per backend. The backends supply a [`RangeSink`]: CUDA
//! enqueues asynchronous launches on its stream, `CubeCL` submits to its
//! client, Vulkan records and submits a dispatch. This is the same split as
//! [`crate::gpu_config`], which holds the per-base kernel constants for the
//! same reason — [`crate::client_process_cuda`] is `#![cfg(feature = "cuda")]`
//! and unreachable from a Vulkan-only build.

#![cfg(any(feature = "cuda", feature = "vulkan", feature = "cubecl"))]
#![allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]

use crate::{FieldResults, FieldSize, NiceNumberSimple, msd_prefix_filter, residue_filter};
use anyhow::{Result, anyhow};
use log::{debug, warn};
use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::mpsc::{Receiver, SyncSender, sync_channel};
use std::sync::{Arc, Condvar, Mutex, OnceLock};
use std::time::{Duration, Instant};

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
/// work units, flushed once it holds [`WORKER_BATCH_RANGES`] descriptors or
/// [`WORKER_BATCH_CHUNKS`] units' worth. [`LAUNCH_BATCH_RANGES`] is the
/// consumer's flush threshold and never bounds what sits in the channel. A
/// unit's output is fed in at most `WORKER_BATCH_RANGES` at a time, so a
/// batch is at most twice that — 8192 descriptors, 20 bytes apiece. So the
/// cap is about 10 MB of queued descriptors at any floor, comfortably below
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

/// Work units a worker folds into one batch before sending, whatever its size.
///
/// Bounds the latency of a batch at coarse floors, where a unit yields one
/// descriptor per chunk and [`WORKER_BATCH_RANGES`] alone would hold back
/// thousands of chunks' worth of device work.
const WORKER_BATCH_CHUNKS: usize = 256;

/// Log2 of the number of chunks one MSD work unit (a *block*) spans.
///
/// The MSD recursion used to start at [`PROCESSING_CHUNK_SIZE`], so a 1e13
/// field paid ten million top-level analyses even where the filter rejects
/// whole swaths at once. Starting at a block of chunks lets one analysis
/// reject 2^k chunks together. Because the recursion halves, a block of
/// exactly 2^k chunks reaches chunk boundaries after k halvings and from
/// there on runs the very same recursion as before — and a rejection of a
/// wider interval implies rejection of every narrower one inside it (fewer
/// fixed leading digits is a weaker premise), while ancestor certificates
/// only ever add digits the chunk-level analysis fixes too. So the leaves,
/// their order and their masks are bit-identical to the chunk-level start;
/// only the work changes ([`msd_blocks`] keeps every block a power of two of
/// chunks for exactly this reason, and the test below checks it).
///
/// Measured with the no-op pipeline on the Anvil base-54 regions (1e12, four
/// cores), chunk start → 64-chunk blocks: floor 500k 0.33 s → 0.23 s and
/// 0.36 s → 0.19 s; floor 250k 0.53 s → 0.44 s and 0.49 s → 0.36 s. Where
/// nothing rejects above the chunk (base 40, 40% survival) it is a wash
/// (0.9-1.0x); at an MSD-strong band start it is 40x. Larger blocks gain
/// little more and cost parallelism on small fields.
const MSD_BLOCK_CHUNKS_LOG2: u32 = 6;

/// The tiling of a field into MSD work units: blocks of `2^k` whole chunks,
/// `k` as large as [`MSD_BLOCK_CHUNKS_LOG2`] allows while still leaving at
/// least `min_blocks` units to spread over the workers. The field's chunk
/// count is rarely a multiple of `2^k`; the remainder is covered by
/// ever-smaller power-of-two blocks, so every block except possibly the very
/// last (a partial chunk) halves down onto chunk boundaries. See
/// [`MSD_BLOCK_CHUNKS_LOG2`] for why that alignment matters.
///
/// Blocks are computed on demand from their index rather than materialised:
/// a 1e13 field is 156 250 of them, and the pipeline keeps two fields open.
#[derive(Clone, Copy, Debug)]
struct BlockTiling {
    start: u128,
    end: u128,
    /// Whole chunks in the field.
    full_chunks: u128,
    /// Chunks per full-size block.
    block_chunks: u128,
    /// Full-size blocks; the tail after them is `full_chunks % block_chunks`
    /// chunks in descending powers of two, then a partial chunk if any.
    n_full: u128,
    /// Total number of blocks.
    len: usize,
}

impl BlockTiling {
    fn new(range: &FieldSize, min_blocks: usize) -> Self {
        let full_chunks = range.size() / PROCESSING_CHUNK_SIZE;
        let mut log2 = MSD_BLOCK_CHUNKS_LOG2;
        while log2 > 0 && (full_chunks >> log2) < min_blocks as u128 {
            log2 -= 1;
        }
        let block_chunks = 1u128 << log2;
        let n_full = full_chunks / block_chunks;
        let tail_chunks = full_chunks % block_chunks;
        let tail_blocks = tail_chunks.count_ones() as usize;
        let partial = usize::from(!range.size().is_multiple_of(PROCESSING_CHUNK_SIZE));
        Self {
            start: range.start(),
            end: range.end(),
            full_chunks,
            block_chunks,
            n_full,
            len: n_full as usize + tail_blocks + partial,
        }
    }

    #[cfg(test)]
    fn len(&self) -> usize {
        self.len
    }

    /// The `i`-th block, or `None` past the end.
    fn get(&self, i: usize) -> Option<FieldSize> {
        if i >= self.len {
            return None;
        }
        let i = i as u128;
        let c = PROCESSING_CHUNK_SIZE;
        if i < self.n_full {
            let s = self.start + i * self.block_chunks * c;
            return Some(FieldSize::new(s, s + self.block_chunks * c));
        }
        // Walk the tail: descending powers of two of the remainder, then the
        // partial chunk.
        let mut chunk_cursor = self.n_full * self.block_chunks;
        let mut remaining = self.full_chunks - chunk_cursor;
        let mut idx = self.n_full;
        while remaining > 0 {
            let take = 1u128 << remaining.ilog2();
            if idx == i {
                let s = self.start + chunk_cursor * c;
                return Some(FieldSize::new(s, s + take * c));
            }
            chunk_cursor += take;
            remaining -= take;
            idx += 1;
        }
        // Partial last chunk.
        Some(FieldSize::new(self.start + chunk_cursor * c, self.end))
    }
}

/// [`BlockTiling`] materialised, for tests.
#[cfg(test)]
fn msd_blocks(range: &FieldSize, min_blocks: usize) -> Vec<FieldSize> {
    let tiling = BlockTiling::new(range, min_blocks);
    (0..tiling.len()).filter_map(|i| tiling.get(i)).collect()
}

/// Minimum MSD recursion floor the controller may reach: a sixteenth of a
/// [`PROCESSING_CHUNK_SIZE`] chunk.
///
/// Below roughly this, a finer floor stops paying: survivors barely
/// decrease (the recursion already stops where the analysis rejects, so
/// leaves are a few tens of thousands of numbers regardless) while the
/// number of descriptors keeps growing, and each range costs the device
/// setup work and the host 20 bytes of traffic. Measured with pinned floors
/// on fixed base-54 fields: a 9070 XT does 6.7e12 n/s at 250-350k, 5.2e12
/// at 125k and 1.7e12 at 60k; an M4 does 6.3e11 at 250k, 7.9e11 at 60k and
/// 4.5e11 at 30k. The wait-balance controller cannot see that cliff — a
/// device that is behind stays behind when the floor drops — and on the M4
/// it steered to 20k and a third of the throughput before this clamp.
/// Every optimum measured (M4 60k, RTX 3060 ~100k, 9070 XT 250k, 4090 and
/// A100 at the cap) is at or above this value.
///
/// An explicit `NICE_GPU_MSD_FLOOR` pin is not clamped.
#[allow(clippy::cast_precision_loss)]
const MSD_FLOOR_MIN: f64 = (PROCESSING_CHUNK_SIZE / 16) as f64;

/// Maximum MSD recursion floor the controller may reach: half a
/// [`PROCESSING_CHUNK_SIZE`] chunk, i.e. one level of subdivision below the
/// whole-chunk check ([`msd_prefix_filter::MSD_RECURSIVE_SUBDIVISION_FACTOR`]
/// is 2).
///
/// One whole chunk is the explicit no-MSD bypass ([`descriptors_for_chunk`]
/// ships every chunk as one descriptor with no endpoint analysis). At the
/// bypass every candidate survives and the device checks the whole field —
/// measured on Anvil (A100 + 32 EPYC cores, base 54, 1e13 fields) at 2.5e11
/// n/s against 1.2-6.3e12 n/s at floors between 100k and 900k — so the
/// controller is not allowed there; pin `NICE_GPU_MSD_FLOOR=1000000` for the
/// one configuration it was meant for (a single-core host with a strong
/// device), which bypasses this clamp.
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

/// Where the controller starts: half the cap. On a many-core host paired with
/// a strong device the balance point measured on Anvil sits right about here
/// (250k), and on a weak host the controller raises it within seconds. Not
/// derived from the core count: a low seed costs whole fields (a 1e13 field
/// at 16k takes 10 s on 32 cores against 1 s at 250k), a high one costs a
/// few seconds of device idling.
const MSD_FLOOR_SEED: f64 = MSD_FLOOR_MAX / 2.0;

/// How often the controller reconsiders the floor.
const FLOOR_ADJUST_INTERVAL: Duration = Duration::from_millis(500);
/// Largest multiplicative step per adjustment, either direction: taken when
/// one side waited the whole interval and the other not at all. The step is
/// proportional to how one-sided the waits were, so near the balance point
/// the controller inches ("57% vs 42%" moves 4%) and a regime change still
/// moves it fast. A fixed step hopped across the balance point instead:
/// measured on an RTX 3060 with 19 cores, 1.25x alternated 82k↔128k every
/// interval with each side waiting >90% in turn, and a reversal-damped
/// variant still hopped 200k↔280k on a 9070 XT for 40 s before settling.
const FLOOR_STEP: f64 = 1.25;
/// Smallest step taken once the controller decides to move at all.
const FLOOR_STEP_MIN: f64 = 1.02;
/// Fraction of an interval one side must spend waiting on the other before
/// the floor moves. Below this the pipeline is called balanced.
const FLOOR_WAIT_THRESHOLD: f64 = 0.15;

/// Steers the MSD recursion floor so that neither the CPU nor the device
/// waits for the other.
///
/// The floor trades CPU work for device work: a finer floor filters harder
/// (more CPU time per number, fewer survivors for the device), a coarser one
/// the reverse. The right setting depends on the host's cores, the device,
/// the base and even the region of the base — an MSD-strong region rejects
/// nearly everything at any floor — so it is steered at run time.
///
/// The signal is *who is waiting*, measured where the two halves meet: the
/// dispatch thread records how long it spends blocked pulling descriptors
/// from the workers (the CPU is behind) and how long it spends blocked in
/// `launch` because the device has all the work it may hold in flight (the
/// device is behind). Every [`FLOOR_ADJUST_INTERVAL`] the floor moves one
/// [`FLOOR_STEP`] toward the side that was waiting, if either waited more
/// than [`FLOOR_WAIT_THRESHOLD`] of the interval; otherwise it holds. No
/// device timing is needed on any backend, and a regime change shows up
/// within an interval rather than a field.
///
/// The earlier controller compared the CPU phase against the device *tail*
/// after the workers finished. Under an overlapped pipeline that tail is one
/// batch whenever the device keeps pace, so it always said "raise", and on
/// Anvil it ratcheted to the bypass and stayed there.
///
/// `NICE_GPU_MSD_FLOOR` pins the floor and disables steering (floor sweeps,
/// benchmarks); see [`pin_msd_floor_for_benchmark`].
pub struct FloorController {
    /// The floor as `f64` bits; workers read it per block, lock-free.
    floor_bits: AtomicU64,
    /// No steering while set: pinned by the environment for the whole
    /// process, or frozen by the benchmark for a measured window.
    pinned: AtomicBool,
    /// Pinned by `NICE_GPU_MSD_FLOOR`: the benchmark's freeze/thaw leave it
    /// alone, so floor sweeps under `--benchmark` still work.
    env_pinned: bool,
    state: Mutex<FloorState>,
}

struct FloorState {
    interval_start: Instant,
    cpu_wait: Duration,
    device_wait: Duration,
}

impl FloorController {
    fn new(floor: f64, pinned: bool) -> Self {
        Self {
            floor_bits: AtomicU64::new(floor.to_bits()),
            pinned: AtomicBool::new(pinned),
            env_pinned: pinned,
            state: Mutex::new(FloorState {
                interval_start: Instant::now(),
                cpu_wait: Duration::ZERO,
                device_wait: Duration::ZERO,
            }),
        }
    }

    /// The floor in force right now.
    pub fn floor(&self) -> u128 {
        f64::from_bits(self.floor_bits.load(Ordering::Relaxed)) as u128
    }

    /// Record time the dispatch thread spent waiting for descriptors (the CPU
    /// side was behind) or blocked handing work to a full device (the device
    /// side was behind), and steer once an interval has elapsed.
    fn observe(&self, cpu_wait: Duration, device_wait: Duration) {
        if self.pinned.load(Ordering::Relaxed) {
            return;
        }
        let mut st = self.state.lock().unwrap();
        st.cpu_wait += cpu_wait;
        st.device_wait += device_wait;
        let elapsed = st.interval_start.elapsed();
        if elapsed < FLOOR_ADJUST_INTERVAL {
            return;
        }
        let cpu_frac = st.cpu_wait.as_secs_f64() / elapsed.as_secs_f64();
        let device_frac = st.device_wait.as_secs_f64() / elapsed.as_secs_f64();
        st.interval_start = Instant::now();
        st.cpu_wait = Duration::ZERO;
        st.device_wait = Duration::ZERO;
        let direction: i8 = if device_frac > FLOOR_WAIT_THRESHOLD && device_frac >= cpu_frac {
            // The device has more than it can take: filter harder.
            -1
        } else {
            // The device is starved: filter less (or hold if nobody waited).
            i8::from(cpu_frac > FLOOR_WAIT_THRESHOLD)
        };
        if direction == 0 {
            return;
        }
        drop(st);
        // Proportional: the more one-sided the waiting, the bigger the step.
        let imbalance = (device_frac - cpu_frac).abs().min(1.0);
        let step = (1.0 + (FLOOR_STEP - 1.0) * imbalance).max(FLOOR_STEP_MIN);
        let floor = f64::from_bits(self.floor_bits.load(Ordering::Relaxed));
        let new_floor = if direction < 0 {
            (floor / step).max(MSD_FLOOR_MIN)
        } else {
            (floor * step).min(MSD_FLOOR_MAX)
        };
        if (new_floor - floor).abs() > f64::EPSILON {
            debug!(
                "GPU MSD floor: {floor:.0} → {new_floor:.0} (cpu waited {:.0}%, device waited {:.0}%, step {step:.3})",
                100.0 * cpu_frac,
                100.0 * device_frac
            );
            self.floor_bits
                .store(new_floor.to_bits(), Ordering::Relaxed);
        }
    }
}

impl FloorController {
    /// Stop steering and hold the current floor. Returns it. A floor pinned
    /// by the environment is unaffected (it is already held).
    fn freeze(&self) -> u128 {
        self.pinned.store(true, Ordering::Relaxed);
        self.floor()
    }

    /// Resume steering from `seed`, with the step and interval reset so the
    /// run does not start with a stale direction. No-op under an
    /// environment pin.
    fn thaw(&self, seed: f64) {
        if self.env_pinned {
            return;
        }
        let mut st = self.state.lock().unwrap();
        st.interval_start = Instant::now();
        st.cpu_wait = Duration::ZERO;
        st.device_wait = Duration::ZERO;
        drop(st);
        self.floor_bits.store(seed.to_bits(), Ordering::Relaxed);
        self.pinned.store(false, Ordering::Relaxed);
    }
}

static FLOOR: OnceLock<FloorController> = OnceLock::new();

/// The process-wide floor controller, initialised on first use: pinned by
/// `NICE_GPU_MSD_FLOOR` if set, otherwise steering from [`MSD_FLOOR_SEED`].
fn floor_controller() -> &'static FloorController {
    FLOOR.get_or_init(|| {
        if let Ok(v) = std::env::var("NICE_GPU_MSD_FLOOR") {
            match v.parse::<f64>() {
                Ok(f) if f >= 1.0 => {
                    debug!("GPU MSD floor fixed at {f:.0} via NICE_GPU_MSD_FLOOR");
                    return FloorController::new(f, true);
                }
                _ => warn!("ignoring invalid NICE_GPU_MSD_FLOOR '{v}'; steering the floor"),
            }
        }
        debug!("GPU MSD floor: steered, seed {MSD_FLOOR_SEED:.0}");
        FloorController::new(MSD_FLOOR_SEED, false)
    })
}

/// Let the benchmark steer a scenario's floor from the production seed:
/// resets the controller and resumes steering. Call before a scenario's
/// warm-up; pair with [`benchmark_floor_freeze`] before its measured
/// windows. An explicit `NICE_GPU_MSD_FLOOR` still wins, so floor sweeps
/// under `--benchmark` remain possible: both calls are then no-ops.
///
/// Why not simply pin: a steered floor is what production runs at, and it
/// differs by machine in both directions (measured: an RTX 4090 with six
/// cores settles at the cap, an RTX 3060 with nineteen near 100k, and the
/// pinned cap undersold the latter by a third). Why not steer through the
/// measurement: the controller moves every half second and the windows are
/// tens of milliseconds, so a moving floor would make the rate depend on
/// where in the controller's cycle the window fell. Steer to convergence
/// first, then hold.
pub fn benchmark_floor_thaw() {
    floor_controller().thaw(MSD_FLOOR_SEED);
}

/// Hold the floor where the warm-up left it for the measured windows, and
/// report it. See [`benchmark_floor_thaw`].
#[must_use]
pub fn benchmark_floor_freeze() -> u128 {
    floor_controller().freeze()
}

/// The MSD floor currently in force, for reports. Initialises the controller
/// if nothing has yet.
#[must_use]
pub fn msd_floor_in_use() -> u128 {
    floor_controller().floor()
}

/// Fields the client keeps open in the pipeline at once: with two, the next
/// field's MSD work overlaps the device's tail on the current one. One is the
/// old field-serial behaviour, for A/B runs. `NICE_GPU_FIELDS_IN_FLIGHT`.
#[must_use]
pub fn fields_in_flight() -> usize {
    static N: OnceLock<usize> = OnceLock::new();
    *N.get_or_init(|| {
        std::env::var("NICE_GPU_FIELDS_IN_FLIGHT")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .filter(|&n| n >= 1)
            .unwrap_or(2)
    })
}

/// Launched batches a backend keeps in flight before it blocks the dispatch
/// thread. This is the device-side queue depth: deep enough that the device
/// never runs dry between batches, shallow enough that a backed-up device is
/// felt as `launch` blocking within a fraction of a second, which is the
/// controller's "device is behind" signal. `NICE_GPU_BATCHES_IN_FLIGHT`.
#[must_use]
pub fn batches_in_flight() -> usize {
    static N: OnceLock<usize> = OnceLock::new();
    *N.get_or_init(|| {
        std::env::var("NICE_GPU_BATCHES_IN_FLIGHT")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .filter(|&n| n >= 1)
            .unwrap_or(16)
    })
}

/// Per-field statistics from the pipeline.
#[derive(Clone, Copy, Debug, Default)]
pub struct NiceonlyStats {
    /// Wall time the MSD workers spent on this field, from the first block
    /// taken to the last finished. Overlaps neighbouring fields' device time.
    pub msd_secs: f64,
    /// Wall time from this field's first launch to its results being ready
    /// on the device. Overlaps the next field's MSD time and, on the device,
    /// its early batches.
    pub device_secs: f64,
    /// Wall time from the field entering the pipeline to its results being
    /// read back. With two fields in flight this exceeds the field's share of
    /// throughput; see the client's rate accounting.
    pub total_secs: f64,
    /// The MSD floor in force when the field was opened; blocks later in the
    /// field may have been filtered at a floor the controller had moved to.
    pub floor: u128,
    pub num_ranges: usize,
    pub valid_numbers: u64,
    pub launches: u32,
    /// Time the dispatch thread spent waiting for descriptors while this was
    /// the oldest open field: the CPU side was behind.
    pub cpu_wait_secs: f64,
    /// Time the dispatch thread spent blocked handing this field's batches to
    /// a full device queue: the device side was behind.
    pub device_wait_secs: f64,
    /// Device time actually spent on this field's batches, where the backend
    /// can measure it (CUDA, from per-batch events); `None` elsewhere.
    pub device_busy_secs: Option<f64>,
}

impl NiceonlyStats {
    /// The per-field pipeline telemetry, as submitted alongside results.
    /// Seconds are raw so consumers can divide by whichever wall time they
    /// mean (the submission's `processing_secs` is the field's share of
    /// throughput); `floor` is a string like the other u128 fields.
    #[must_use]
    pub fn telemetry_json(&self) -> serde_json::Value {
        serde_json::json!({
            "msd_floor": self.floor.to_string(),
            "fields_in_flight": fields_in_flight(),
            "batches_in_flight": batches_in_flight(),
            "msd_secs": self.msd_secs,
            "total_secs": self.total_secs,
            "cpu_wait_secs": self.cpu_wait_secs,
            "device_wait_secs": self.device_wait_secs,
            "device_busy_secs": self.device_busy_secs,
            "num_ranges": self.num_ranges,
            "valid_numbers": self.valid_numbers,
            "launches": self.launches,
        })
    }
}

/// What waiting for a field's device work yields.
pub struct DeviceResult {
    pub nice_numbers: Vec<NiceNumberSimple>,
    /// Device time spent on the field, if the backend measured it.
    pub device_busy_secs: Option<f64>,
}

/// What a backend's `begin` returns: either the field was handled on the
/// spot (a base the device cannot take, or a residue-empty one) or it went
/// into the pipeline and its results come out of the backend's `finish`.
pub enum NiceonlyStarted {
    Immediate(FieldResults),
    Queued,
}

/// The device-side result of a field, not yet waited for: launched, results
/// still on the device. `wait` blocks until they are there and reads them.
pub trait PendingField {
    /// # Errors
    /// Returns the device's error, or an overflowed output buffer.
    fn wait(self: Box<Self>) -> Result<DeviceResult>;
}

/// A backend's device-side end of the pipeline.
///
/// The pipeline opens a field, hands over batches of MSD-surviving range
/// descriptors for it, and closes it. Fields are opened in order, but a
/// field is opened while the previous one may still have batches in flight
/// and is closed only once every one of its batches has been handed over; up
/// to [`fields_in_flight`] fields are open at a time, and a batch always
/// names its field. Closing returns a [`PendingField`] that is waited for on
/// another thread, so it must own whatever the wait needs.
///
/// `launch` is the backpressure point: a backend keeps at most
/// [`batches_in_flight`] launched batches outstanding and blocks in `launch`
/// until the oldest completes. That blocking is what the pipeline measures
/// as "the device is behind"; a backend whose launches are synchronous
/// (Vulkan) blocks naturally.
pub trait RangeSink {
    type Pending: PendingField;

    /// Open a field: allocate its output slot, fetch its base's plan.
    ///
    /// # Errors
    /// Device allocation or kernel build errors.
    fn begin_field(&mut self, seq: u64, base: u32, range: &FieldSize) -> Result<()>;

    /// Hand over one batch of `field`'s descriptors: offsets from the field's
    /// start (`u64`), lengths (`u32`), certificate masks (`u64`), one triple
    /// per range.
    ///
    /// # Errors
    /// Device errors, or a batch for a field that is not open.
    fn launch(&mut self, field: u64, offsets: &[u64], lens: &[u32], masks: &[u64]) -> Result<()>;

    /// Close a field: everything for it has been launched.
    ///
    /// # Errors
    /// Device errors, or a field that is not open.
    fn end_field(&mut self, seq: u64) -> Result<Self::Pending>;
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

/// MSD-filter one block of chunks (see [`BlockTiling`]) into descriptors. At
/// the no-MSD bypass floor this is chunk by chunk, since the bypass's unit is
/// the chunk; below it the recursion starts at the block.
fn descriptors_for_block(
    block: FieldSize,
    base: u32,
    floor: u128,
    field_start: u128,
) -> Result<(Vec<u64>, Vec<u32>, Vec<u64>)> {
    if floor >= PROCESSING_CHUNK_SIZE {
        let mut offsets = Vec::new();
        let mut lens = Vec::new();
        let mut masks = Vec::new();
        for chunk in block.chunks(PROCESSING_CHUNK_SIZE) {
            let (o, l, m) = descriptors_for_chunk(chunk, base, floor, field_start)?;
            offsets.extend(o);
            lens.extend(l);
            masks.extend(m);
        }
        return Ok((offsets, lens, masks));
    }
    descriptors_for_chunk(block, base, floor, field_start)
}

/// One message from an MSD worker to the dispatch thread.
enum Msg {
    /// A field entered the pipeline. Sent by `push` before any worker can see
    /// the field, so it precedes every descriptor for it.
    Begin {
        seq: u64,
        base: u32,
        range: FieldSize,
    },
    /// Descriptors for `field`.
    Ranges {
        field: u64,
        offsets: Vec<u64>,
        lens: Vec<u32>,
        masks: Vec<u64>,
    },
    /// Every descriptor for `seq` has been sent: each worker sends its last
    /// batch for a field before counting itself out, and the last worker out
    /// sends this, so on the channel's FIFO it follows them all.
    End { seq: u64, msd_secs: f64 },
}

/// Descriptors one MSD worker has accumulated since its last send.
#[derive(Default)]
struct WorkerBatch {
    offsets: Vec<u64>,
    lens: Vec<u32>,
    masks: Vec<u64>,
    units: usize,
}

impl WorkerBatch {
    /// Fold one unit's descriptors in. Empty units still count toward the
    /// unit bound so a run of rejected blocks cannot delay a pending batch.
    fn absorb(&mut self, offsets: &[u64], lens: &[u32], masks: &[u64]) {
        self.offsets.extend_from_slice(offsets);
        self.lens.extend_from_slice(lens);
        self.masks.extend_from_slice(masks);
        self.units += 1;
    }

    fn is_ready(&self) -> bool {
        self.offsets.len() >= WORKER_BATCH_RANGES || self.units >= WORKER_BATCH_CHUNKS
    }

    fn is_empty(&self) -> bool {
        self.offsets.is_empty()
    }

    /// Hand the accumulated descriptors over as a message for `field`.
    fn take(&mut self, field: u64) -> Msg {
        let batch = std::mem::take(self);
        Msg::Ranges {
            field,
            offsets: batch.offsets,
            lens: batch.lens,
            masks: batch.masks,
        }
    }

    /// Fold one unit's descriptors in, sending full batches on the way.
    /// A unit can yield far more than one batch's worth at a fine floor, so
    /// this feeds it through in [`WORKER_BATCH_RANGES`] slices and never lets
    /// a message outgrow the `PIPELINE_DEPTH` memory budget. The last slice
    /// is left in the batch for the caller's ready check. `Err` means the
    /// channel is closed.
    fn feed(
        &mut self,
        field: u64,
        offsets: &[u64],
        lens: &[u32],
        masks: &[u64],
        tx: &SyncSender<Msg>,
    ) -> Result<(), ()> {
        if offsets.is_empty() {
            self.absorb(&[], &[], &[]);
            return Ok(());
        }
        let mut slices = offsets
            .chunks(WORKER_BATCH_RANGES)
            .zip(lens.chunks(WORKER_BATCH_RANGES))
            .zip(masks.chunks(WORKER_BATCH_RANGES))
            .peekable();
        while let Some(((o, l), m)) = slices.next() {
            self.absorb(o, l, m);
            if slices.peek().is_some() && self.is_ready() && tx.send(self.take(field)).is_err() {
                return Err(());
            }
        }
        Ok(())
    }

    /// Send whatever is left (a batch that is ready, or the field's last
    /// partial one). `Err` means the channel is closed.
    fn flush(&mut self, field: u64, tx: &SyncSender<Msg>) -> Result<(), ()> {
        if self.is_empty() {
            *self = Self::default();
            return Ok(());
        }
        tx.send(self.take(field)).map_err(|_| ())
    }
}

/// One field's MSD work, shared by the workers.
struct FieldWork {
    seq: u64,
    base: u32,
    range: FieldSize,
    tiling: BlockTiling,
    next_block: AtomicUsize,
    /// Workers that have run out of blocks here and moved on.
    exited: AtomicUsize,
    started: Mutex<Option<Instant>>,
}

/// State shared between the pipeline's threads.
struct Shared {
    fields: Mutex<FieldQueueState>,
    field_added: Condvar,
    /// Set when the pipeline is dropped; workers exit at their next look.
    closed: AtomicBool,
    workers: usize,
}

struct FieldQueueState {
    /// Fields still being filtered, by sequence number. A field leaves once
    /// every worker has exited it.
    open: HashMap<u64, Arc<FieldWork>>,
}

impl Shared {
    /// The field with sequence number `seq`, waiting for it to be pushed.
    /// `None` once the pipeline has been closed.
    fn wait_for_field(&self, seq: u64) -> Option<Arc<FieldWork>> {
        let mut st = self.fields.lock().unwrap();
        loop {
            if let Some(f) = st.open.get(&seq) {
                return Some(f.clone());
            }
            if self.closed.load(Ordering::Acquire) {
                return None;
            }
            st = self.field_added.wait(st).unwrap();
        }
    }
}

/// The MSD worker loop: walk every field in sequence, filtering its blocks
/// into descriptors for the dispatch thread. See [`Msg::End`] for the
/// ordering this maintains.
fn msd_worker(shared: &Shared, tx: &SyncSender<Msg>, error: &Mutex<Option<anyhow::Error>>) {
    let mut seq = 0u64;
    let mut batch = WorkerBatch::default();
    while let Some(work) = shared.wait_for_field(seq) {
        loop {
            let i = work.next_block.fetch_add(1, Ordering::Relaxed);
            let Some(block) = work.tiling.get(i) else {
                break;
            };
            if i == 0 {
                *work.started.lock().unwrap() = Some(Instant::now());
            }
            // The floor is read per block, not per field: the controller
            // steps every half second, and a field on a slow device can take
            // far longer than that. Sampling it once per field would let a
            // whole field's worth of "device behind" pile up unobserved and
            // slam the floor to its minimum for the next one. Every floor is
            // sound, so mixing them within a field only changes the work.
            let floor = floor_controller().floor();
            match descriptors_for_block(block, work.base, floor, work.range.start()) {
                Ok((offsets, lens, masks)) => {
                    if batch.feed(work.seq, &offsets, &lens, &masks, tx).is_err() {
                        return;
                    }
                }
                Err(e) => {
                    *error.lock().unwrap() = Some(e);
                    // Leave the field as if finished so the pipeline reports
                    // the error rather than waiting for descriptors forever.
                    break;
                }
            }
            if batch.is_ready() && batch.flush(work.seq, tx).is_err() {
                return;
            }
        }
        // Out of blocks: send our last batch for this field *before*
        // counting ourselves out, so it precedes the End marker.
        if batch.flush(work.seq, tx).is_err() {
            return;
        }
        if work.exited.fetch_add(1, Ordering::AcqRel) + 1 == shared.workers {
            let msd_secs = work
                .started
                .lock()
                .unwrap()
                .map_or(0.0, |t| t.elapsed().as_secs_f64());
            shared.fields.lock().unwrap().open.remove(&work.seq);
            if tx
                .send(Msg::End {
                    seq: work.seq,
                    msd_secs,
                })
                .is_err()
            {
                return;
            }
        }
        seq += 1;
    }
}

/// A field whose device work has been issued: what the pipeline hands back.
pub struct FieldReady<P> {
    pub seq: u64,
    pub pending: Result<P>,
    pub stats: NiceonlyStats,
    pushed_at: Instant,
}

/// The dispatch side: consumes the workers' messages, batches descriptors
/// into launches, and closes fields as their markers arrive. Generic over
/// the sink so it serves both the threaded pipeline and the one-field
/// synchronous form.
struct Dispatcher<'a, S: RangeSink> {
    sink: &'a mut S,
    rx: &'a Receiver<Msg>,
    controller: &'static FloorController,
    /// Fields opened on the sink, with what has been launched for them.
    open: HashMap<u64, OpenField>,
    buf_offsets: Vec<u64>,
    buf_lens: Vec<u32>,
    buf_masks: Vec<u64>,
    /// The field the launch buffer belongs to; a batch never mixes fields.
    buf_field: Option<u64>,
    first_error: Option<anyhow::Error>,
}

struct OpenField {
    pushed_at: Instant,
    floor: u128,
    first_launch: Option<Instant>,
    num_ranges: usize,
    valid_numbers: u64,
    launches: u32,
    cpu_wait: Duration,
    device_wait: Duration,
}

impl<S: RangeSink> Dispatcher<'_, S> {
    fn flush_launch(&mut self) {
        if self.buf_offsets.is_empty() {
            return;
        }
        let Some(field) = self.buf_field else { return };
        let t = Instant::now();
        let outcome = self
            .sink
            .launch(field, &self.buf_offsets, &self.buf_lens, &self.buf_masks);
        let waited = t.elapsed();
        self.controller.observe(Duration::ZERO, waited);
        if let Some(open) = self.open.get_mut(&field) {
            open.first_launch.get_or_insert(t);
            open.launches += 1;
            open.device_wait += waited;
        }
        if let Err(e) = outcome
            && self.first_error.is_none()
        {
            self.first_error = Some(e);
        }
        self.buf_offsets.clear();
        self.buf_lens.clear();
        self.buf_masks.clear();
    }

    /// Handle one message. Returns the field closed by it, if any.
    fn handle(&mut self, msg: Msg) -> Option<FieldReady<S::Pending>> {
        match msg {
            Msg::Begin { seq, base, range } => {
                if let Err(e) = self.sink.begin_field(seq, base, &range)
                    && self.first_error.is_none()
                {
                    self.first_error = Some(e);
                }
                self.open.insert(
                    seq,
                    OpenField {
                        pushed_at: Instant::now(),
                        floor: floor_controller().floor(),
                        first_launch: None,
                        num_ranges: 0,
                        valid_numbers: 0,
                        launches: 0,
                        cpu_wait: Duration::ZERO,
                        device_wait: Duration::ZERO,
                    },
                );
                None
            }
            Msg::Ranges {
                field,
                offsets,
                lens,
                masks,
            } => {
                if self.buf_field != Some(field) {
                    self.flush_launch();
                    self.buf_field = Some(field);
                }
                if let Some(open) = self.open.get_mut(&field) {
                    open.num_ranges += offsets.len();
                    open.valid_numbers += lens.iter().map(|&l| u64::from(l)).sum::<u64>();
                }
                self.buf_offsets.extend_from_slice(&offsets);
                self.buf_lens.extend_from_slice(&lens);
                self.buf_masks.extend_from_slice(&masks);
                if self.buf_offsets.len() >= LAUNCH_BATCH_RANGES {
                    self.flush_launch();
                }
                None
            }
            Msg::End { seq, msd_secs } => {
                if self.buf_field == Some(seq) {
                    self.flush_launch();
                }
                let open = self.open.remove(&seq)?;
                let pending = match self.first_error.take() {
                    Some(e) => Err(e),
                    None => self.sink.end_field(seq),
                };
                Some(FieldReady {
                    seq,
                    pending,
                    stats: NiceonlyStats {
                        msd_secs,
                        device_secs: 0.0,
                        total_secs: 0.0,
                        floor: open.floor,
                        num_ranges: open.num_ranges,
                        valid_numbers: open.valid_numbers,
                        launches: open.launches,
                        cpu_wait_secs: open.cpu_wait.as_secs_f64(),
                        device_wait_secs: open.device_wait.as_secs_f64(),
                        device_busy_secs: None,
                    },
                    pushed_at: open.pushed_at,
                })
            }
        }
    }

    /// Block for the next message, charging the wait to the CPU side — but
    /// only while a field is open. With nothing open the pipeline is idle for
    /// an outside reason (the client waiting on a claim), and calling that
    /// "the CPU is behind" would ratchet the floor up during every API stall.
    fn recv(&mut self) -> Option<Msg> {
        match self.rx.try_recv() {
            Ok(m) => Some(m),
            Err(std::sync::mpsc::TryRecvError::Disconnected) => None,
            Err(std::sync::mpsc::TryRecvError::Empty) => {
                let t = Instant::now();
                let m = self.rx.recv().ok();
                let waited = t.elapsed();
                if let Some(oldest) = self.open.keys().min().copied() {
                    self.controller.observe(waited, Duration::ZERO);
                    if let Some(open) = self.open.get_mut(&oldest) {
                        open.cpu_wait += waited;
                    }
                }
                m
            }
        }
    }
}

fn new_dispatcher<'a, S: RangeSink>(sink: &'a mut S, rx: &'a Receiver<Msg>) -> Dispatcher<'a, S> {
    Dispatcher {
        sink,
        rx,
        controller: floor_controller(),
        open: HashMap::new(),
        buf_offsets: Vec::new(),
        buf_lens: Vec::new(),
        buf_masks: Vec::new(),
        buf_field: None,
        first_error: None,
    }
}

fn new_shared(workers: usize) -> Shared {
    Shared {
        fields: Mutex::new(FieldQueueState {
            open: HashMap::new(),
        }),
        field_added: Condvar::new(),
        closed: AtomicBool::new(false),
        workers,
    }
}

fn worker_count() -> usize {
    std::thread::available_parallelism().map_or(4, std::num::NonZeroUsize::get)
}

/// Push a field into the shared state, after announcing it on the channel.
fn push_field(
    shared: &Shared,
    tx: &SyncSender<Msg>,
    seq: u64,
    base: u32,
    range: &FieldSize,
) -> Result<()> {
    let work = Arc::new(FieldWork {
        seq,
        base,
        range: *range,
        tiling: BlockTiling::new(range, 2 * shared.workers),
        next_block: AtomicUsize::new(0),
        exited: AtomicUsize::new(0),
        started: Mutex::new(None),
    });
    // The announcement must precede every descriptor for the field on the
    // channel: send it before the workers can see the field.
    tx.send(Msg::Begin {
        seq,
        base,
        range: *range,
    })
    .map_err(|_| anyhow!("niceonly pipeline dispatch thread is gone"))?;
    let mut st = shared.fields.lock().unwrap();
    st.open.insert(seq, work);
    drop(st);
    shared.field_added.notify_all();
    Ok(())
}

/// Finish a closed field: wait for the device, read back, fill in the
/// timings, log the summary line.
fn complete_field<P: PendingField>(
    backend: &str,
    base: u32,
    ready: FieldReady<P>,
    error: &Mutex<Option<anyhow::Error>>,
) -> Result<(NiceonlyStats, Vec<NiceNumberSimple>)> {
    let mut stats = ready.stats;
    let pending = ready.pending?;
    let DeviceResult {
        nice_numbers: results,
        device_busy_secs,
    } = Box::new(pending).wait()?;
    stats.device_busy_secs = device_busy_secs;
    stats.total_secs = ready.pushed_at.elapsed().as_secs_f64();
    // The device span is not directly observable here without device
    // timestamps; report the time from the End marker to results being
    // ready, which is the tail the host actually waited on.
    stats.device_secs = (stats.total_secs - stats.msd_secs).max(0.0);
    if let Some(e) = error.lock().unwrap().take() {
        return Err(e);
    }
    report_field(backend, base, stats);
    Ok((stats, results))
}

/// Run one niceonly field on the calling thread: MSD workers stream
/// descriptors while this thread batches them into launches, then the
/// field's results are waited for and returned. This is the one-field form
/// of [`NiceonlyPipeline`], for a sink that cannot leave the calling thread
/// (Vulkan) and for tests; it has no cross-field overlap.
///
/// **Range semantics**: half-open [`range_start`, `range_end`).
///
/// # Errors
/// Returns an error if a descriptor does not fit its encoding, or on any
/// device failure reported by the sink.
///
/// # Panics
/// Panics if an MSD worker panicked while holding the error slot.
pub fn run_range_pipeline<S: RangeSink>(
    backend: &str,
    sink: &mut S,
    range: &FieldSize,
    base: u32,
) -> Result<(NiceonlyStats, Vec<NiceNumberSimple>)> {
    let shared = new_shared(worker_count());
    let (tx, rx) = sync_channel::<Msg>(PIPELINE_DEPTH);
    let worker_error: Mutex<Option<anyhow::Error>> = Mutex::new(None);
    push_field(&shared, &tx, 0, base, range)?;
    // Closing right away makes the workers exit after this one field.
    shared.closed.store(true, Ordering::Release);

    let ready = std::thread::scope(|scope| {
        let shared = &shared;
        let worker_error = &worker_error;
        for _ in 0..shared.workers {
            let tx = tx.clone();
            scope.spawn(move || msd_worker(shared, &tx, worker_error));
        }
        drop(tx);
        let mut d = new_dispatcher(sink, &rx);
        let mut ready = None;
        while let Some(msg) = d.recv() {
            if let Some(r) = d.handle(msg) {
                ready = Some(r);
            }
        }
        // Workers may still be parked in `send` if a launch failed and we
        // stopped consuming; dropping the receiver wakes them.
        drop(d);
        drop(rx);
        ready
    });
    let ready = ready.ok_or_else(|| anyhow!("pipeline ended without closing the field"))?;
    complete_field(backend, base, ready, &worker_error)
}

/// A continuous niceonly pipeline over one device: fields go in with
/// [`NiceonlyPipeline::push`] and come out, in order, from
/// [`NiceonlyPipeline::next_result`]. The MSD workers move on to the next
/// pushed field the moment they run out of blocks on the current one, and
/// the dispatch thread launches each field's batches as they arrive, so with
/// two fields open the device drains one while the CPU filters the next.
///
/// Threads: [`worker_count`] MSD workers, one dispatch thread owning the
/// sink, and the caller, who waits for results. Dropping the pipeline stops
/// the workers and dispatcher; fields still open are abandoned.
pub struct NiceonlyPipeline<P: PendingField> {
    backend: &'static str,
    shared: Arc<Shared>,
    /// `Option` only so `Drop` can release it before joining the threads:
    /// the dispatcher exits when every sender is gone.
    tx: Option<SyncSender<Msg>>,
    /// Likewise: a dispatcher blocked handing over a result must be released.
    results: Option<Receiver<FieldReady<P>>>,
    worker_error: Arc<Mutex<Option<anyhow::Error>>>,
    next_seq: u64,
    /// `(seq, base)` of fields pushed and not yet returned, in order.
    outstanding: VecDeque<(u64, u32)>,
    threads: Vec<std::thread::JoinHandle<()>>,
}

impl<P: PendingField + Send + 'static> NiceonlyPipeline<P> {
    /// Start the workers and the dispatch thread over `sink`.
    pub fn start<S: RangeSink<Pending = P> + Send + 'static>(
        backend: &'static str,
        mut sink: S,
    ) -> Self {
        let workers = worker_count();
        let shared = Arc::new(new_shared(workers));
        let (tx, rx) = sync_channel::<Msg>(PIPELINE_DEPTH);
        let (results_tx, results) = sync_channel::<FieldReady<P>>(fields_in_flight() + 1);
        let worker_error = Arc::new(Mutex::new(None));
        let mut threads = Vec::with_capacity(workers + 1);
        for _ in 0..workers {
            let shared = shared.clone();
            let tx = tx.clone();
            let worker_error = worker_error.clone();
            threads.push(std::thread::spawn(move || {
                msd_worker(&shared, &tx, &worker_error);
            }));
        }
        threads.push(std::thread::spawn(move || {
            let mut d = new_dispatcher(&mut sink, &rx);
            while let Some(msg) = d.recv() {
                if let Some(ready) = d.handle(msg)
                    && results_tx.send(ready).is_err()
                {
                    // The caller is gone; nothing to deliver to.
                    return;
                }
            }
        }));
        Self {
            backend,
            shared,
            tx: Some(tx),
            results: Some(results),
            worker_error,
            next_seq: 0,
            outstanding: VecDeque::new(),
            threads,
        }
    }

    /// Enter a field. Returns immediately; the workers pick it up as soon as
    /// they finish the fields before it.
    ///
    /// # Errors
    /// Returns an error if the dispatch thread has exited.
    ///
    /// # Panics
    /// Panics if the shared field-queue mutex was poisoned by an earlier panic.
    pub fn push(&mut self, base: u32, range: &FieldSize) -> Result<()> {
        let seq = self.next_seq;
        let tx = self
            .tx
            .as_ref()
            .ok_or_else(|| anyhow!("niceonly pipeline is shut down"))?;
        push_field(&self.shared, tx, seq, base, range)?;
        self.outstanding.push_back((seq, base));
        self.next_seq += 1;
        Ok(())
    }

    /// Fields pushed and not yet returned.
    #[must_use]
    pub fn outstanding(&self) -> usize {
        self.outstanding.len()
    }

    /// Wait for the oldest outstanding field: blocks until its device work is
    /// done and its results are read back.
    ///
    /// # Errors
    /// The field's error (device failure, descriptor overflow), or the
    /// pipeline having stopped.
    ///
    /// # Panics
    /// Panics if an MSD worker's error slot mutex was poisoned by an earlier panic.
    pub fn next_result(&mut self) -> Result<(NiceonlyStats, Vec<NiceNumberSimple>)> {
        let (seq, base) = self
            .outstanding
            .pop_front()
            .ok_or_else(|| anyhow!("no field outstanding in the niceonly pipeline"))?;
        let ready = self
            .results
            .as_ref()
            .ok_or_else(|| anyhow!("niceonly pipeline is shut down"))?
            .recv()
            .map_err(|_| anyhow!("niceonly pipeline dispatch thread is gone"))?;
        debug_assert_eq!(ready.seq, seq);
        complete_field(self.backend, base, ready, &self.worker_error)
    }
}

impl<P: PendingField> Drop for NiceonlyPipeline<P> {
    fn drop(&mut self) {
        // Workers parked waiting for a field exit once they see `closed`.
        self.shared.closed.store(true, Ordering::Release);
        self.shared.field_added.notify_all();
        // The dispatcher exits when the last sender is gone — ours must go
        // before the join, and the workers' go with them. A dispatcher parked
        // handing over a result is released by dropping the receiver.
        drop(self.tx.take());
        drop(self.results.take());
        for t in self.threads.drain(..) {
            let _ = t.join();
        }
    }
}

/// Log the per-field summary. Both backends report the same line.
#[allow(clippy::cast_precision_loss)]
pub fn report_field(backend: &str, base: u32, stats: NiceonlyStats) {
    debug!(
        "{backend} niceonly b{base}: floor {} msd {:.3}s -> {} ranges ({:.3e} numbers), gpu {:.3}s, total {:.3}s, {} launches, waited cpu {:.3}s device {:.3}s, device busy {}",
        stats.floor,
        stats.msd_secs,
        stats.num_ranges,
        stats.valid_numbers as f64,
        stats.device_secs,
        stats.total_secs,
        stats.launches,
        stats.cpu_wait_secs,
        stats.device_wait_secs,
        stats
            .device_busy_secs
            .map_or_else(|| "n/a".to_string(), |b| format!("{b:.3}s")),
    );
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

    /// Nothing to wait for: the test sinks have no device.
    struct NoResults;

    impl PendingField for NoResults {
        fn wait(self: Box<Self>) -> Result<DeviceResult> {
            Ok(DeviceResult {
                nice_numbers: Vec::new(),
                device_busy_secs: None,
            })
        }
    }

    /// A sink that records what it was handed instead of touching a device.
    #[derive(Default)]
    struct Recorder {
        batches: Vec<(Vec<u64>, Vec<u32>, Vec<u64>)>,
        closed: bool,
    }

    impl RangeSink for Recorder {
        type Pending = NoResults;
        fn begin_field(&mut self, _seq: u64, _base: u32, _range: &FieldSize) -> Result<()> {
            Ok(())
        }
        fn launch(
            &mut self,
            _field: u64,
            offsets: &[u64],
            lens: &[u32],
            masks: &[u64],
        ) -> Result<()> {
            self.batches
                .push((offsets.to_vec(), lens.to_vec(), masks.to_vec()));
            Ok(())
        }
        fn end_field(&mut self, _seq: u64) -> Result<Self::Pending> {
            self.closed = true;
            Ok(NoResults)
        }
    }

    /// A sink that fails the way a device does.
    struct Failing;

    impl RangeSink for Failing {
        type Pending = NoResults;
        fn begin_field(&mut self, _seq: u64, _base: u32, _range: &FieldSize) -> Result<()> {
            Ok(())
        }
        fn launch(
            &mut self,
            _field: u64,
            _offsets: &[u64],
            _lens: &[u32],
            _masks: &[u64],
        ) -> Result<()> {
            anyhow::bail!("simulated device failure")
        }
        fn end_field(&mut self, _seq: u64) -> Result<Self::Pending> {
            Ok(NoResults)
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
            let _ = tx.send(run_range_pipeline("test", &mut Failing, &field, base).is_err());
        });
        let errored = rx
            .recv_timeout(Duration::from_mins(2))
            .expect("run_range_pipeline did not return: the workers are deadlocked");
        assert!(errored, "the launch failure must reach the caller");
    }

    /// A sink for the threaded pipeline: records every launch by field,
    /// checks the protocol (a field is opened before its first launch, and
    /// nothing arrives for it after it is closed), and hands the field's
    /// descriptor count back as its "result".
    #[derive(Default)]
    struct MockDevice {
        opened: Vec<u64>,
        closed: Vec<u64>,
        ranges: HashMap<u64, Vec<(u64, u32, u64)>>,
        launches: HashMap<u64, u32>,
    }

    struct MockSink(Arc<Mutex<MockDevice>>);

    struct MockPending {
        device: Arc<Mutex<MockDevice>>,
        seq: u64,
    }

    impl PendingField for MockPending {
        fn wait(self: Box<Self>) -> Result<DeviceResult> {
            // Encode "how many descriptors this field had" as a fake hit.
            let n = self
                .device
                .lock()
                .unwrap()
                .ranges
                .get(&self.seq)
                .map_or(0, Vec::len);
            Ok(DeviceResult {
                nice_numbers: vec![NiceNumberSimple {
                    number: n as u128,
                    num_uniques: self.seq as u32,
                }],
                device_busy_secs: None,
            })
        }
    }

    impl RangeSink for MockSink {
        type Pending = MockPending;
        fn begin_field(&mut self, seq: u64, _base: u32, _range: &FieldSize) -> Result<()> {
            let mut d = self.0.lock().unwrap();
            assert!(!d.opened.contains(&seq), "field {seq} opened twice");
            d.opened.push(seq);
            Ok(())
        }
        fn launch(
            &mut self,
            field: u64,
            offsets: &[u64],
            lens: &[u32],
            masks: &[u64],
        ) -> Result<()> {
            let mut d = self.0.lock().unwrap();
            assert!(
                d.opened.contains(&field),
                "launch for field {field} before it was opened"
            );
            assert!(
                !d.closed.contains(&field),
                "launch for field {field} after it was closed"
            );
            let entry = d.ranges.entry(field).or_default();
            for ((&o, &l), &m) in offsets.iter().zip(lens).zip(masks) {
                entry.push((o, l, m));
            }
            *d.launches.entry(field).or_default() += 1;
            Ok(())
        }
        fn end_field(&mut self, seq: u64) -> Result<Self::Pending> {
            let mut d = self.0.lock().unwrap();
            assert!(d.opened.contains(&seq));
            d.closed.push(seq);
            Ok(MockPending {
                device: self.0.clone(),
                seq,
            })
        }
    }

    /// Windows of base 40 that both keep and reject, for the pipeline tests.
    fn mixed_windows(n: usize) -> Vec<FieldSize> {
        let base = 40;
        let start = crate::base_range::get_base_range_u128(base)
            .unwrap()
            .unwrap()
            .range_start;
        let span = 300 * PROCESSING_CHUNK_SIZE;
        (0u128..2000)
            .map(|i| FieldSize::new(start + i * span, start + (i + 1) * span))
            .filter(|f| {
                let n: usize = f
                    .chunks(PROCESSING_CHUNK_SIZE)
                    .into_iter()
                    .map(|c| {
                        descriptors_for_chunk(c, base, 4000, f.start())
                            .unwrap()
                            .0
                            .len()
                    })
                    .sum();
                n > 0
            })
            .take(n)
            .collect()
    }

    /// The threaded pipeline returns fields in push order, each with exactly
    /// the descriptors the one-field form produces for it, keeps the sink's
    /// open/launch/close protocol, and lets several fields be open at once.
    #[test]
    fn threaded_pipeline_matches_the_one_field_form_field_by_field() {
        let base = 40;
        let fields = mixed_windows(3);
        assert_eq!(fields.len(), 3, "need three mixed windows in base 40");
        let device = Arc::new(Mutex::new(MockDevice::default()));
        let mut pipeline = NiceonlyPipeline::start("test", MockSink(device.clone()));

        // Two open at once, like the client runs it.
        pipeline.push(base, &fields[0]).unwrap();
        pipeline.push(base, &fields[1]).unwrap();
        assert_eq!(pipeline.outstanding(), 2);
        let (stats0, hits0) = pipeline.next_result().unwrap();
        pipeline.push(base, &fields[2]).unwrap();
        let (stats1, hits1) = pipeline.next_result().unwrap();
        let (stats2, hits2) = pipeline.next_result().unwrap();
        assert_eq!(pipeline.outstanding(), 0);
        assert_eq!(
            (
                hits0[0].num_uniques,
                hits1[0].num_uniques,
                hits2[0].num_uniques
            ),
            (0, 1, 2),
            "results must come back in push order"
        );

        let d = device.lock().unwrap();
        assert_eq!(d.opened, vec![0, 1, 2]);
        assert_eq!(d.closed, vec![0, 1, 2]);
        for (seq, (field, stats)) in fields.iter().zip([stats0, stats1, stats2]).enumerate() {
            let seq = seq as u64;
            let mut sink = Recorder::default();
            let (ref_stats, _) = run_range_pipeline("test", &mut sink, field, base).unwrap();
            let mut expected: Vec<(u64, u32, u64)> = sink
                .batches
                .iter()
                .flat_map(|(o, l, m)| {
                    o.iter()
                        .zip(l)
                        .zip(m)
                        .map(|((&o, &l), &m)| (o, l, m))
                        .collect::<Vec<_>>()
                })
                .collect();
            expected.sort_unstable();
            let mut got = d.ranges.get(&seq).cloned().unwrap_or_default();
            got.sort_unstable();
            assert_eq!(
                got, expected,
                "field {seq}: descriptors differ from the one-field form"
            );
            assert_eq!(stats.num_ranges, ref_stats.num_ranges);
            assert_eq!(stats.valid_numbers, ref_stats.valid_numbers);
            assert_eq!(stats.launches, d.launches[&seq]);
            assert!(stats.total_secs > 0.0);
        }
    }

    /// A device failure surfaces from `next_result` for the field it hit,
    /// and neither that nor dropping the pipeline with work outstanding
    /// hangs.
    #[test]
    fn threaded_pipeline_reports_errors_and_drops_cleanly() {
        let (tx, rx) = mpsc::channel();
        std::thread::spawn(move || {
            let base = 40;
            let fields = mixed_windows(2);
            let mut pipeline = NiceonlyPipeline::start("test", Failing);
            pipeline.push(base, &fields[0]).unwrap();
            pipeline.push(base, &fields[1]).unwrap();
            let first = pipeline.next_result();
            // Drop with a field still outstanding.
            drop(pipeline);
            let _ = tx.send(first.is_err());
        });
        let errored = rx
            .recv_timeout(Duration::from_secs(120))
            .expect("the pipeline hung on error or on drop");
        assert!(errored, "the launch failure must reach next_result");
    }

    /// A field the filter rejects entirely still comes back, promptly and
    /// empty: the End marker must not depend on any descriptor having flowed.
    #[test]
    fn threaded_pipeline_returns_fully_rejected_fields() {
        let base = 50;
        // Base 50's band start is MSD-strong: everything is rejected.
        let start = crate::base_range::get_base_range_u128(base)
            .unwrap()
            .unwrap()
            .range_start;
        let field = FieldSize::new(start, start + 100 * PROCESSING_CHUNK_SIZE);
        let device = Arc::new(Mutex::new(MockDevice::default()));
        let mut pipeline = NiceonlyPipeline::start("test", MockSink(device.clone()));
        pipeline.push(base, &field).unwrap();
        let (stats, _) = pipeline.next_result().unwrap();
        assert_eq!(stats.num_ranges, 0);
        assert_eq!(stats.launches, 0);
        let d = device.lock().unwrap();
        assert_eq!(d.opened, vec![0]);
        assert_eq!(d.closed, vec![0]);
    }

    /// The controller moves toward whichever side waited, holds when neither
    /// did, never leaves `[MSD_FLOOR_MIN, MSD_FLOOR_MAX]`, and ignores
    /// everything when pinned.
    #[test]
    #[allow(clippy::cast_precision_loss)]
    fn floor_controller_steers_toward_the_waiting_side() {
        let interval = FLOOR_ADJUST_INTERVAL;
        // Force an adjustment: pretend the interval has elapsed.
        // Pretend exactly one interval (plus a hair) has elapsed, so a wait
        // of `interval` reads as "the whole interval".
        let elapse = |c: &FloorController| {
            c.state.lock().unwrap().interval_start = Instant::now()
                .checked_sub(interval + Duration::from_micros(10))
                .unwrap();
        };
        let c = FloorController::new(100_000.0, false);
        // The device waited (launch blocked) for most of the interval: down.
        elapse(&c);
        c.observe(Duration::ZERO, interval);
        assert!((c.floor() as f64 - 100_000.0 / FLOOR_STEP).abs() < 5.0);
        // The CPU waited (recv blocked) the whole interval: up, full step.
        let before = c.floor() as f64;
        elapse(&c);
        c.observe(interval, Duration::ZERO);
        assert!((c.floor() as f64 - before * FLOOR_STEP).abs() < 5.0);
        // Neither waited enough: hold.
        let before = c.floor();
        elapse(&c);
        c.observe(interval.mul_f64(0.05), interval.mul_f64(0.05));
        assert_eq!(c.floor(), before);
        // Both waited, device a little more: down, but only a little — the
        // step is proportional to the imbalance (0.2 here → 5%).
        elapse(&c);
        c.observe(interval.mul_f64(0.3), interval.mul_f64(0.5));
        let expected = before as f64 / (1.0 + (FLOOR_STEP - 1.0) * 0.2);
        assert!(
            (c.floor() as f64 - expected).abs() < 2.0,
            "{} vs {expected}",
            c.floor()
        );

        // Clamped at both ends, never into the bypass.
        let c = FloorController::new(MSD_FLOOR_MAX * 0.9, false);
        for _ in 0..10 {
            elapse(&c);
            c.observe(interval, Duration::ZERO);
        }
        assert_eq!(c.floor(), MSD_FLOOR_MAX as u128);
        assert!(c.floor() < PROCESSING_CHUNK_SIZE);
        let c = FloorController::new(MSD_FLOOR_MIN * 1.1, false);
        for _ in 0..10 {
            elapse(&c);
            c.observe(Duration::ZERO, interval);
        }
        assert_eq!(c.floor(), MSD_FLOOR_MIN as u128);

        // Pinned: nothing moves.
        let c = FloorController::new(777.0, true);
        elapse(&c);
        c.observe(interval, Duration::ZERO);
        assert_eq!(c.floor(), 777);
        // ...and an environment pin ignores the benchmark's thaw.
        c.thaw(1000.0);
        elapse(&c);
        c.observe(interval, Duration::ZERO);
        assert_eq!(c.floor(), 777);

        // Freeze holds; thaw resumes from the seed with a fresh step.
        let c = FloorController::new(100_000.0, false);
        elapse(&c);
        c.observe(interval, Duration::ZERO);
        let frozen = c.freeze();
        assert!(frozen > 100_000);
        elapse(&c);
        c.observe(interval, Duration::ZERO);
        assert_eq!(c.floor(), frozen);
        c.thaw(50_000.0);
        assert_eq!(c.floor(), 50_000);
        elapse(&c);
        c.observe(interval, Duration::ZERO);
        assert!((c.floor() as f64 - 50_000.0 * FLOOR_STEP).abs() < 5.0);
    }

    /// Blocks are power-of-two chunk counts covering the field exactly, the
    /// remainder in descending powers of two, and small fields still get
    /// enough units to keep the workers busy.
    #[test]
    fn msd_blocks_cover_the_field_in_power_of_two_chunk_counts() {
        let c = PROCESSING_CHUNK_SIZE;
        // 1e7 chunks (a 1e13 field) is exactly 156 250 blocks of 64; a hundred
        // more chunks add 64 + 32 + 4.
        let field = FieldSize::new(1000, 1000 + 10_000_100 * c);
        let blocks = msd_blocks(&field, 64);
        assert_eq!(blocks[0].size(), 64 * c);
        assert_eq!(blocks.len(), 156_250 + 3);
        assert_eq!(blocks[156_250].size(), 64 * c);
        assert_eq!(blocks[156_251].size(), 32 * c);
        assert_eq!(blocks[156_252].size(), 4 * c);
        let mut cursor = field.start();
        for b in &blocks {
            assert_eq!(b.start(), cursor, "blocks must tile the field");
            let chunks = b.size() / c;
            assert!(chunks.is_power_of_two() && b.size() % c == 0);
            cursor = b.end();
        }
        assert_eq!(cursor, field.end());

        // 100 chunks: 64 + 32 + 4.
        let sizes: Vec<u128> = msd_blocks(&FieldSize::new(0, 100 * c), 1)
            .iter()
            .map(|b| b.size() / c)
            .collect();
        assert_eq!(sizes, vec![64, 32, 4]);

        // A partial last chunk is its own (sub-chunk) block.
        let blocks = msd_blocks(&FieldSize::new(0, 3 * c + 500), 1);
        assert_eq!(
            blocks.iter().map(FieldSize::size).collect::<Vec<_>>(),
            vec![2 * c, c, 500]
        );

        // Small field, many workers: the block shrinks so there are at least
        // `min_blocks` units (here 1000 chunks for 64 wanted → 8-chunk blocks).
        let blocks = msd_blocks(&FieldSize::new(0, 1000 * c), 64);
        assert_eq!(blocks[0].size(), 8 * c);
        assert!(blocks.len() >= 64);
    }

    /// The block start must not change what the device is asked to check: the
    /// leaves, their order and their certificate masks must equal the chunk
    /// start's, on a region where the filter both rejects and subdivides.
    #[test]
    fn block_start_yields_the_same_descriptors_as_the_chunk_start() {
        let base = 40;
        let start = crate::base_range::get_base_range_u128(base)
            .unwrap()
            .unwrap()
            .range_start;
        // 300 chunks: blocks of 64+64+64+64+32+8 and a fine floor, so the
        // recursion runs many levels above and below the chunk. The band start
        // is MSD-strong and rejects everything, so walk forward to the first
        // window that both keeps and rejects.
        let floor = 4000;
        let span = 300 * PROCESSING_CHUNK_SIZE;
        let field = (0u128..1000)
            .map(|i| FieldSize::new(start + i * span, start + (i + 1) * span))
            .find(|f| {
                let n: usize = f
                    .chunks(PROCESSING_CHUNK_SIZE)
                    .into_iter()
                    .map(|c| {
                        descriptors_for_chunk(c, base, floor, f.start())
                            .unwrap()
                            .0
                            .len()
                    })
                    .sum();
                n > 0 && n < 300 * (PROCESSING_CHUNK_SIZE / floor) as usize
            })
            .expect("a mixed window inside the first 3e11 of base 40");

        let mut by_chunk = (Vec::new(), Vec::new(), Vec::new());
        for chunk in field.chunks(PROCESSING_CHUNK_SIZE) {
            let (o, l, m) = descriptors_for_chunk(chunk, base, floor, field.start()).unwrap();
            by_chunk.0.extend(o);
            by_chunk.1.extend(l);
            by_chunk.2.extend(m);
        }
        let mut by_block = (Vec::new(), Vec::new(), Vec::new());
        for block in msd_blocks(&field, 1) {
            let (o, l, m) = descriptors_for_block(block, base, floor, field.start()).unwrap();
            by_block.0.extend(o);
            by_block.1.extend(l);
            by_block.2.extend(m);
        }
        assert_eq!(by_block, by_chunk);
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
        let Msg::Ranges {
            offsets,
            lens,
            masks,
            ..
        } = batch.take(0)
        else {
            panic!("take yields a Ranges message")
        };
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
        let Msg::Ranges { offsets, .. } = batch.take(0) else {
            panic!("take yields a Ranges message")
        };
        assert_eq!(offsets.len(), WORKER_BATCH_CHUNKS);

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
        let (stats, _) = run_range_pipeline("test", &mut sink, &field, base).expect("pipeline");
        assert!(sink.closed, "the pipeline must close the field on the sink");
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
