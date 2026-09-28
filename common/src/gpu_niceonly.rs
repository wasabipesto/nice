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
//! floor is steered at run time by measured throughput (or, optionally, by
//! which side is behind) — see [`FloorController`]. [`run_range_pipeline`] is the one-field
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
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, AtomicUsize, Ordering};
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

/// Work units (MSD blocks of `2^MSD_BLOCK_CHUNKS_LOG2` chunks, fewer on small
/// fields; see [`BlockTiling`]) a worker folds into one batch before
/// sending, whatever its size. (The name predates blocks.) A block whose
/// descriptors overflow [`WORKER_BATCH_RANGES`] counts once per batch it
/// spills into.
///
/// Bounds the field span a batch covers, so that at coarse floors or in
/// MSD-strong regions, where a unit yields a descriptor or two,
/// [`WORKER_BATCH_RANGES`] alone cannot hold back thousands of blocks' worth
/// of device work, nor let the workers run that far ahead of the device
/// (see [`LAUNCH_BATCH_UNITS`]). 64 full blocks are 4e9 numbers.
const WORKER_BATCH_CHUNKS: usize = 64;

/// Work units the consumer folds into one launch, whatever its descriptor
/// count.
///
/// [`LAUNCH_BATCH_RANGES`] bounds a launch in surviving ranges; in an
/// MSD-strong region most chunks die on the host, so 65536 survivors can
/// span tens of billions of numbers and the ring of launches in flight then
/// holds most of a claim. Everything that reads the pipeline's pace from the
/// worker side, the wait heuristic's "device is behind" and the measured
/// search's numbers per second, saw the host racing ahead instead of the
/// device's rate: on a live base-57 client the search's per-level readings
/// swung 2.5x within a minute and it settled at the cap. 128 full blocks
/// are 8e9 numbers, a few milliseconds of device time. A launch closes on
/// the message that crosses the bound, so it holds up to 191 units. With
/// `NICE_GPU_BATCHES_IN_FLIGHT` (16) launches outstanding plus the worker
/// channel ([`PIPELINE_DEPTH`] batches of up to 64 units) the workers lead a
/// device-bound pipeline by about 5e11 numbers: under a tick at 5e12 n/s,
/// but a few ticks on the slowest hosts measured (1-2e12 n/s), which the
/// search's settle ticks and its smoothing absorb.
const LAUNCH_BATCH_UNITS: usize = 128;

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

/// Minimum MSD recursion floor the wait heuristic may reach: a sixteenth of a
/// [`PROCESSING_CHUNK_SIZE`] chunk. (The measured search has its own ladder,
/// which goes below this: it can see a finer floor stop paying, so it needs
/// no clamp, and on some weak-device, many-core hosts it settled at 46.9k,
/// 23.4k or 11.7k.)
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
/// A100 at the cap) with this controller was at or above this value.
///
/// An explicit `NICE_GPU_MSD_FLOOR` pin is not clamped.
#[allow(clippy::cast_precision_loss)]
const MSD_FLOOR_MIN: f64 = (PROCESSING_CHUNK_SIZE / 16) as f64;

/// Maximum MSD recursion floor the wait heuristic may reach: half a
/// [`PROCESSING_CHUNK_SIZE`] chunk, i.e. one level of subdivision below the
/// whole-chunk check ([`msd_prefix_filter::MSD_RECURSIVE_SUBDIVISION_FACTOR`]
/// is 2). (The measured search's top level, 750k, analyses each chunk once
/// and ships it whole, which is not the bypass either.)
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

/// Where the wait heuristic starts: half the cap. On a many-core host paired with
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

// ---------------------------------------------------------------------------
// Measured ladder search (`NICE_GPU_FLOOR_CONTROLLER=search`)
// ---------------------------------------------------------------------------
//
// The wait heuristic optimises a proxy (nobody waiting) and cannot see that
// a finer floor has stopped paying. The search measures the objective
// itself: numbers per second through the MSD workers, which the
// span-bounded work in flight ([`LAUNCH_BATCH_UNITS`]) ties to the
// pipeline's throughput on every backend.
//
// Throughput is not comparable across moments, though. A base-57 claim's
// seventh digit changes every 1.95e12 numbers, well under a second at
// production rates, and whether the leading digits already repeat decides
// whether a stretch dies on the host at once or goes to the device: measured
// live, the rate swung 2x within a minute at one floor and 2.5x from one
// claim to the next. So every tick is tagged with the fraction of its span
// that survived the recursion's first analysis, of the whole MSD block (the
// workers' unit; depth 0 of the recursion), which every ladder level
// computes, and levels are compared only within bins of that fraction:
// stretches of like difficulty against stretches of like difficulty.
//
// Per base the search visits each level once (the sweep), holds the best,
// and every [`SEARCH_HOLD`] tries one neighbour, adopting it when the
// stratified comparison says it is faster by more than
// [`SEARCH_HYSTERESIS`]. The benchmark's pinned sweep feeds the same
// accounting, so after `benchmark_floor_thaw` the search starts at the best
// pinned level instead of re-learning it.

/// Levels of the search, coarsest first: level `l` is a floor of three
/// quarters of `chunk >> l`, i.e. 750k, 375k, 187.5k, 93.75k, 46.9k, 23.4k,
/// 11.7k. At each the recursion analyses nodes down to `chunk >> l` and
/// ships the passing ones whole, which does the host work of the
/// power-of-two pin one level finer and hands the device the same surviving
/// volume in half as many descriptors (see [`BENCHMARK_FLOOR_LADDER`]). The
/// top level analyses each chunk once and ships it whole: the host work of
/// the wait heuristic's 500k cap, which ships the chunk's halves instead.
/// It is there for hosts with almost no CPU: on an RTX 3080 with a 1.3-core
/// i3 it is the best pinned floor on five of the benchmark's six windows,
/// 2-12% above the cap. The bottom three levels are below the heuristic's
/// 62.5k clamp (the worse configuration of the level above it) and are for
/// the opposite host, a weak device with many cores: a GTX 1660 Ti with 20
/// Xeon cores, live, read 11.7k 9-20% above 23.4k and held it against every
/// trial, and an RTX 2060 with 16 Ryzen cores read the curve still rising
/// at 23.4k in every sweep. On RTX 3080 hosts 11.7k runs at a third of the
/// peak, which costs their sweep one visit, 2.7 s, once per process and
/// base; 5.9k ran at a fifth of the peak on every host and is not on the
/// ladder. Nor is the no-MSD bypass (floor 1e6): the recursion computes
/// nothing there, so there is no difficulty index, and it measured at a
/// twentieth of the MSD floors even on an A100.
const SEARCH_LEVELS: usize = 7;
/// Accounting tick of the search. Short, so each of the benchmark's 0.75 s
/// pins yields two clean ticks after the settling ones.
const SEARCH_TICK: Duration = Duration::from_millis(150);
/// Ticks discarded after any floor change, while work begun at the old floor
/// drains through the queue and the launches in flight (bounded in field
/// span by [`WORKER_BATCH_CHUNKS`] and [`LAUNCH_BATCH_UNITS`] to well under
/// a tick at any measured rate; two ticks leave a margin for slow devices).
const SEARCH_SETTLE_TICKS: u8 = 2;
/// Clean ticks spent at each level during the sweep, and at a neighbour
/// during a trial: 2.4 s, several periods of the digit structure above, so
/// the samples spread over difficulty bins.
const SEARCH_VISIT_TICKS: u8 = 16;
/// How long the search holds a level before trying a neighbour. A trial
/// spends [`SEARCH_VISIT_TICKS`] at a level that is typically 5-20% slower
/// (live, base 57, every neighbour of the settled level lost by that), so
/// this is the price of staying adaptive: about 0.5% of throughput.
const SEARCH_HOLD: Duration = Duration::from_secs(60);
/// A neighbour must measure faster than the held level by this fraction,
/// within like-difficulty bins, to be adopted.
const SEARCH_HYSTERESIS: f64 = 0.05;
/// Difficulty bins over the block-level (depth-0) survival fraction of a
/// tick's span: the fraction of it whose MSD blocks passed the recursion's
/// first analysis, which every level performs.
const SEARCH_BINS: usize = 6;
const SEARCH_BIN_EDGES: [f64; SEARCH_BINS - 1] = [0.02, 0.1, 0.25, 0.5, 0.75];
/// Ticks a level needs in a bin before that bin counts in a comparison.
const SEARCH_MIN_BIN_TICKS: u32 = 2;
/// Home samples this old still take part in a trial's comparison. A bin's
/// sample is a moving average ([`SEARCH_RATE_ALPHA`], about a second of
/// memory) of home's latest run of ticks in that bin, so what a trial is
/// compared against is home's most recent stretch of each difficulty seen
/// in the last minute.
const SEARCH_COMPARE_WINDOW: Duration = Duration::from_secs(60);
/// A bin's sample this old is replaced by the next tick rather than averaged
/// with it (the pipeline was elsewhere in between).
const SEARCH_BLEND_WITHIN: Duration = Duration::from_secs(2);
/// Weight of a new clean tick in a bin's rate estimate.
const SEARCH_RATE_ALPHA: f64 = 0.25;
/// An idle spell with no field open at least this long restarts the tick
/// when the next field opens; shorter gaps (the benchmark's back-to-back
/// windows, a queued field's hand-over) are part of running.
const SEARCH_IDLE_GAP: Duration = Duration::from_millis(50);
/// A tick longer than this spanned a stall (the pipeline idle between
/// fields, a claim late) and is not a sample. Far above any tick of a
/// running pipeline, whose ticks end at the first dispatcher event 150 ms
/// in (at most one launch later, under a second on the slowest devices).
const SEARCH_MAX_TICK: Duration = Duration::from_secs(2);
/// Levels a thaw needs samples for before it starts from the best of them
/// instead of sweeping: the benchmark pins every level, so anything less
/// means its sweep was cut short, and three still beat re-learning inside
/// its warm-up.
const SEARCH_THAW_MIN_LEVELS: usize = 3;

/// The floor that realises search level `level`.
#[allow(clippy::cast_precision_loss)]
fn search_floor(level: usize) -> f64 {
    (PROCESSING_CHUNK_SIZE >> level) as f64 * 0.75
}

/// The search level whose floor is `floor` (within 1%), if any: a floor set
/// by the benchmark's sweep or the wait heuristic may realise none.
fn search_level(floor: f64) -> Option<usize> {
    (0..SEARCH_LEVELS).find(|&l| (search_floor(l) - floor).abs() <= 0.01 * search_floor(l))
}

/// The difficulty bin of a tick whose span survived the block-level
/// analysis in fraction `phi`.
fn difficulty_bin(phi: f64) -> usize {
    SEARCH_BIN_EDGES.iter().take_while(|&&e| phi >= e).count()
}

#[derive(Clone, Copy, Debug)]
struct BinSample {
    /// Numbers per second through the MSD workers, in this bin.
    rate: f64,
    at: Instant,
    /// Clean ticks folded into `rate` since it was last replaced.
    ticks: u32,
}

#[derive(Clone, Copy, Debug)]
enum SearchPhase {
    /// Visiting the levels from the top; `ticks` clean ticks measured at the
    /// current level so far.
    Sweep { ticks: u8 },
    /// Holding the current level since `since`; `dir` is the neighbour to
    /// try next (+1 finer, -1 coarser).
    Hold { since: Instant, dir: i8 },
    /// Trying the current level against `home`, `ticks` clean ticks in.
    Trial { home: usize, ticks: u8 },
}

/// One base's search state.
#[derive(Clone, Debug)]
struct BaseSearch {
    level: usize,
    bins: [[Option<BinSample>; SEARCH_BINS]; SEARCH_LEVELS],
    phase: SearchPhase,
    /// When the current sweep began: only samples since then decide it.
    sweep_since: Instant,
}

impl BaseSearch {
    fn new(now: Instant) -> Self {
        Self {
            level: 0,
            bins: [[None; SEARCH_BINS]; SEARCH_LEVELS],
            phase: SearchPhase::Sweep { ticks: 0 },
            sweep_since: now,
        }
    }

    fn record(&mut self, level: usize, bin: usize, rate: f64, now: Instant) {
        match &mut self.bins[level][bin] {
            Some(prev) if now.duration_since(prev.at) < SEARCH_BLEND_WITHIN => {
                prev.rate += SEARCH_RATE_ALPHA * (rate - prev.rate);
                prev.at = now;
                prev.ticks += 1;
            }
            slot => {
                *slot = Some(BinSample {
                    rate,
                    at: now,
                    ticks: 1,
                });
            }
        }
    }

    /// Whether `level` has any sample taken since `since`.
    fn sampled_since(&self, level: usize, since: Instant) -> bool {
        self.bins[level].iter().flatten().any(|s| s.at >= since)
    }

    /// Rate of level `a` over level `b` as a geometric mean over the
    /// difficulty bins both were sampled in since `since` (each with at
    /// least [`SEARCH_MIN_BIN_TICKS`] ticks), weighted by the smaller tick
    /// count; `None` when they share no such bin.
    fn compare(&self, a: usize, b: usize, since: Instant) -> Option<f64> {
        let mut sum = 0.0;
        let mut weight = 0.0;
        for bin in 0..SEARCH_BINS {
            let (Some(sa), Some(sb)) = (self.bins[a][bin], self.bins[b][bin]) else {
                continue;
            };
            if sa.at < since || sb.at < since {
                continue;
            }
            let w = sa.ticks.min(sb.ticks);
            if w < SEARCH_MIN_BIN_TICKS || sa.rate <= 0.0 || sb.rate <= 0.0 {
                continue;
            }
            sum += f64::from(w) * (sa.rate / sb.rate).ln();
            weight += f64::from(w);
        }
        (weight > 0.0).then(|| (sum / weight).exp())
    }

    /// The best level among those sampled since `since`, ranked by the
    /// geometric mean of each level's ratio to every other sampled level it
    /// shares a difficulty bin with (no hysteresis here: a sweep or a thaw
    /// picks once, and the hold's trials guard against flapping). A level
    /// that shares a bin with no other is passed over, not favoured: a
    /// stretch of unusual digits seen at one level alone says nothing about
    /// the level (a sweep once crowned the coarsest floor on a burst of easy
    /// digits nobody else had sampled). `None` if fewer than `min_levels`
    /// were sampled, or no two of them are comparable.
    fn pick_best(&self, since: Instant, min_levels: usize) -> Option<usize> {
        let sampled: Vec<usize> = (0..SEARCH_LEVELS)
            .filter(|&l| self.sampled_since(l, since))
            .collect();
        if sampled.len() < min_levels {
            return None;
        }
        let mut best: Option<(usize, f64)> = None;
        for &level in &sampled {
            let ratios: Vec<f64> = sampled
                .iter()
                .filter(|&&other| other != level)
                .filter_map(|&other| self.compare(level, other, since))
                .map(f64::ln)
                .collect();
            if ratios.is_empty() {
                continue;
            }
            #[allow(clippy::cast_precision_loss)]
            let score = ratios.iter().sum::<f64>() / ratios.len() as f64;
            if best.is_none_or(|(_, s)| score > s) {
                best = Some((level, score));
            }
        }
        best.map(|(level, _)| level)
    }

    fn describe(&self, since: Instant) -> String {
        (0..SEARCH_LEVELS)
            .map(|l| {
                self.bins[l]
                    .iter()
                    .enumerate()
                    .filter_map(|(b, s)| s.filter(|s| s.at >= since).map(|s| (b, s)))
                    .max_by_key(|(_, s)| s.ticks)
                    .map_or_else(
                        || "-".to_string(),
                        |(b, s)| format!("{:.2e}@{b}x{}", s.rate, s.ticks),
                    )
            })
            .collect::<Vec<_>>()
            .join(", ")
    }

    /// A clean tick was measured at `level` (already recorded). Returns the
    /// level to be at next and why, if the search has something to say; the
    /// level may be the current one (a decision without a move).
    fn step(&mut self, level: usize, now: Instant) -> Option<(usize, &'static str)> {
        if level != self.level {
            return None;
        }
        match self.phase {
            SearchPhase::Sweep { ticks } => {
                let ticks = ticks + 1;
                if ticks < SEARCH_VISIT_TICKS {
                    self.phase = SearchPhase::Sweep { ticks };
                    return None;
                }
                let next = level + 1;
                if next < SEARCH_LEVELS {
                    self.level = next;
                    self.phase = SearchPhase::Sweep { ticks: 0 };
                    return Some((next, "sweep"));
                }
                let Some(best) = self.pick_best(self.sweep_since, 1) else {
                    // No two levels saw a stretch of like difficulty: the
                    // sweep ranks nothing. Sweep again.
                    self.level = 0;
                    self.phase = SearchPhase::Sweep { ticks: 0 };
                    self.sweep_since = now;
                    return Some((0, "sweep unranked, sweeping again"));
                };
                self.level = best;
                self.phase = SearchPhase::Hold { since: now, dir: 1 };
                Some((best, "sweep done"))
            }
            SearchPhase::Hold { since, dir } => {
                // Try a neighbour once the hold is up and home has a fresh
                // comparison sample (after a return to this base, its last
                // ones may have aged out while it was away).
                let fresh = now.checked_sub(SEARCH_COMPARE_WINDOW).unwrap_or(since);
                if now.duration_since(since) < SEARCH_HOLD
                    || !self.bins[level]
                        .iter()
                        .flatten()
                        .any(|b| b.at >= fresh && b.ticks >= SEARCH_MIN_BIN_TICKS)
                {
                    return None;
                }
                let neighbour = |d: i8| {
                    let t = isize::try_from(level).ok()? + isize::from(d);
                    usize::try_from(t).ok().filter(|&t| t < SEARCH_LEVELS)
                };
                let trial = neighbour(dir).or_else(|| neighbour(-dir))?;
                self.phase = SearchPhase::Trial {
                    home: level,
                    ticks: 0,
                };
                self.level = trial;
                Some((trial, "trial"))
            }
            SearchPhase::Trial { home, ticks } => {
                let ticks = ticks + 1;
                if ticks < SEARCH_VISIT_TICKS {
                    self.phase = SearchPhase::Trial { home, ticks };
                    return None;
                }
                let dir: i8 = if level > home { 1 } else { -1 };
                let since = now
                    .checked_sub(SEARCH_COMPARE_WINDOW)
                    .unwrap_or(self.sweep_since);
                match self.compare(level, home, since) {
                    Some(r) if r > 1.0 + SEARCH_HYSTERESIS => {
                        self.phase = SearchPhase::Hold { since: now, dir };
                        Some((level, "trial won"))
                    }
                    Some(_) => {
                        self.level = home;
                        self.phase = SearchPhase::Hold {
                            since: now,
                            dir: -dir,
                        };
                        Some((home, "trial lost"))
                    }
                    None => {
                        // The neighbour saw no stretch of a difficulty home
                        // was measured at: nothing to compare. Same direction
                        // next time.
                        self.level = home;
                        self.phase = SearchPhase::Hold { since: now, dir };
                        Some((home, "trial void, no comparable stretch"))
                    }
                }
            }
        }
    }
}

/// Process-wide state of the measured search: the accounting tick and one
/// [`BaseSearch`] per base seen.
struct LadderSearch {
    tick_start: Instant,
    floor_at_tick: f64,
    settle: u8,
    /// False while the benchmark holds the floor for a measured window: what
    /// flows then says nothing the sweep has not, and the held level's samples
    /// would otherwise blend into the next scenario's first pin of that level,
    /// which comes well under [`SEARCH_BLEND_WITHIN`] later. Measured: a
    /// scenario frozen at 500k made the next one, on a slower window of the
    /// same base, hold 500k against its own sweep's 17% faster 375k.
    learning: bool,
    /// When the benchmark's current pinned sweep began (learning resumed
    /// after a freeze): a thaw decides from samples since then only.
    epoch: Instant,
    current_base: Option<u32>,
    bases: HashMap<u32, BaseSearch>,
    /// When the dispatcher last closed its only open field, if it has not
    /// opened another since.
    idle_since: Option<Instant>,
}

impl LadderSearch {
    fn new(now: Instant, floor: f64) -> Self {
        Self {
            tick_start: now,
            floor_at_tick: floor,
            settle: SEARCH_SETTLE_TICKS,
            learning: true,
            epoch: now,
            current_base: None,
            bases: HashMap::new(),
            idle_since: None,
        }
    }

    /// The floor changed (by the search, a pin or a thaw): start a fresh tick
    /// and discard the settling ones.
    fn floor_changed(&mut self, now: Instant, floor: f64) {
        self.tick_start = now;
        self.floor_at_tick = floor;
        self.settle = SEARCH_SETTLE_TICKS;
    }

    /// The pipeline was idle (no field open) and resumes: start a fresh tick
    /// now, so the idle time is not charged to the level in force, and
    /// discard the first tick while the new field fills the pipeline.
    fn resume(&mut self, now: Instant, floor: f64) {
        self.tick_start = now;
        self.floor_at_tick = floor;
        self.settle = self.settle.max(1);
    }

    /// The benchmark pinned a floor: learn again, and if this is the first
    /// pin since a freeze, start a new epoch for the thaw's decision.
    fn pinned(&mut self, now: Instant) {
        if !self.learning {
            self.learning = true;
            self.epoch = now;
        }
    }

    /// A block finished MSD for `base`. On a base switch, returns the floor
    /// of the level that base settled on (or the top of the ladder, to sweep,
    /// for a base not seen before).
    fn note_block(&mut self, base: u32, now: Instant) -> Option<f64> {
        if self.current_base == Some(base) {
            return None;
        }
        self.current_base = Some(base);
        let st = self
            .bases
            .entry(base)
            .or_insert_with(|| BaseSearch::new(now));
        Some(search_floor(st.level))
    }

    /// One accounting tick: `numbers` went through the workers since the
    /// last tick, of which `passing` survived the block-level analysis, at
    /// `floor`. Returns the floor to be at and why, if the search has a
    /// decision (the floor may be the current one); never while `pinned`,
    /// when it only learns.
    #[allow(clippy::cast_precision_loss)]
    fn tick(
        &mut self,
        now: Instant,
        floor: f64,
        numbers: u64,
        passing: u64,
        pinned: bool,
    ) -> Option<(f64, String)> {
        let elapsed = now.duration_since(self.tick_start);
        let same_floor = (floor - self.floor_at_tick).abs() <= f64::EPSILON;
        self.tick_start = now;
        self.floor_at_tick = floor;
        if !same_floor {
            self.settle = SEARCH_SETTLE_TICKS;
            return None;
        }
        if elapsed > SEARCH_MAX_TICK {
            // The pipeline stalled inside this tick (no field open, a claim
            // late): the work it counts was done in a fraction of it. Not a
            // sample, and the next one settles.
            self.settle = self.settle.max(1);
            return None;
        }
        if self.settle > 0 {
            self.settle -= 1;
            return None;
        }
        if numbers == 0 || !self.learning {
            // Nothing flowed (between fields, or the device path was idle),
            // or the benchmark is measuring: not a sample of any level.
            return None;
        }
        let st = self.bases.get_mut(&self.current_base?)?;
        let Some(level) = search_level(floor) else {
            // A floor no level realises and nobody pins: go back to this
            // base's level.
            return (!pinned).then(|| (search_floor(st.level), String::from("realign")));
        };
        let bin = difficulty_bin(passing as f64 / numbers as f64);
        st.record(
            level,
            bin,
            numbers as f64 / elapsed.as_secs_f64().max(1e-6),
            now,
        );
        if pinned {
            return None;
        }
        if level != st.level {
            // The floor in force is another level than this base's (a base
            // switch raced a thaw): the tick counts for the level it
            // measured, then go back.
            return Some((search_floor(st.level), String::from("realign")));
        }
        let (target, why) = st.step(level, now)?;
        Some((
            search_floor(target),
            format!(
                "{why}; by level, rate@bin x ticks [{}]",
                st.describe(
                    now.checked_sub(SEARCH_COMPARE_WINDOW)
                        .unwrap_or(st.sweep_since)
                )
            ),
        ))
    }

    /// The benchmark resumes steering: start from the best of the levels its
    /// pinned sweep just measured, else sweep.
    fn on_thaw(&mut self, now: Instant) -> (f64, String) {
        self.learning = true;
        let epoch = self.epoch;
        let Some(st) = self.current_base.and_then(|b| self.bases.get_mut(&b)) else {
            return (search_floor(0), String::from("no base yet"));
        };
        let why = if let Some(best) = st.pick_best(epoch, SEARCH_THAW_MIN_LEVELS) {
            st.level = best;
            st.phase = SearchPhase::Hold { since: now, dir: 1 };
            "best of the pinned levels"
        } else {
            st.level = 0;
            st.phase = SearchPhase::Sweep { ticks: 0 };
            st.sweep_since = now;
            "too few levels measured, sweeping"
        };
        (
            search_floor(st.level),
            format!("{why}; by level, rate@bin x ticks [{}]", st.describe(epoch)),
        )
    }
}

/// Steers the MSD recursion floor.
///
/// The floor trades CPU work for device work: a finer floor filters harder
/// (more CPU time per number, fewer survivors for the device), a coarser one
/// the reverse. The right setting depends on the host's cores, the device,
/// the base and even the region of the base — an MSD-strong region rejects
/// nearly everything at any floor — so it is steered at run time.
///
/// `NICE_GPU_FLOOR_CONTROLLER` selects one of two controllers, both sharing
/// this struct:
///
/// - `search` (default): measures the objective itself, numbers per second
///   through the MSD workers, which the span-bounded work in flight
///   ([`LAUNCH_BATCH_UNITS`]) ties to the device's rate; sweeps the ladder
///   once per base, holds the best, and trials a neighbour every
///   [`SEARCH_HOLD`], judged within stretches of like difficulty. See
///   [`LadderSearch`].
/// - `heuristic`: steers so that neither the CPU nor the device waits for
///   the other. The signal is *who is waiting*, measured where the two
///   halves meet: the dispatch thread records how long it spends blocked
///   pulling descriptors from the workers (the CPU is behind) and how long
///   it spends blocked in `launch` because the device has all the work it
///   may hold in flight (the device is behind). Every
///   [`FLOOR_ADJUST_INTERVAL`] the floor moves one [`FLOOR_STEP`] toward the
///   side that was waiting, if either waited more than
///   [`FLOOR_WAIT_THRESHOLD`] of the interval; otherwise it holds. No device
///   timing is needed on any backend, and a regime change shows up within an
///   interval rather than a field. It optimises a proxy, though, and cannot
///   tell that a finer floor has stopped paying, so it is clamped at
///   [`MSD_FLOOR_MIN`]; on device-bound hosts it sits at that clamp, which is
///   the worse of the two configurations at that level (see
///   [`BENCHMARK_FLOOR_LADDER`]).
///
/// The controller before the heuristic compared the CPU phase against the
/// device *tail* after the workers finished. Under an overlapped pipeline
/// that tail is one batch whenever the device keeps pace, so it always said
/// "raise", and on Anvil it ratcheted to the bypass and stayed there.
///
/// `NICE_GPU_MSD_FLOOR` pins the floor and disables all steering (floor
/// sweeps, benchmarks); see [`benchmark_floor_pin`].
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
    /// `NICE_GPU_FLOOR_CONTROLLER=search`: steer by measured throughput per
    /// level.
    search_enabled: bool,
    search: Mutex<LadderSearch>,
    /// Numbers through the MSD workers since the search's last tick, and
    /// how many of them survived the recursion's block-level analysis.
    numbers_seen: AtomicU64,
    passing_seen: AtomicU64,
    /// The base of the last block the search was told about, so the workers
    /// take the search lock only on a base switch.
    search_base: AtomicU32,
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
            search_enabled: false,
            search: Mutex::new(LadderSearch::new(Instant::now(), floor)),
            numbers_seen: AtomicU64::new(0),
            passing_seen: AtomicU64::new(0),
            search_base: AtomicU32::new(u32::MAX),
        }
    }

    /// Whether the measured search steers this controller.
    #[must_use]
    pub fn search_enabled(&self) -> bool {
        self.search_enabled && !self.env_pinned
    }

    /// Which controller is in charge, for reports: `pinned` under
    /// `NICE_GPU_MSD_FLOOR`, else `search` or `heuristic`.
    #[must_use]
    pub fn name(&self) -> &'static str {
        if self.env_pinned {
            "pinned"
        } else if self.search_enabled {
            "search"
        } else {
            "heuristic"
        }
    }

    /// A worker finished the MSD pass over one block of `base`: `numbers`
    /// long, of which `passing` survived the chunk-level analysis. Feeds the
    /// search's accounting; on a base switch, jumps to where that base last
    /// settled.
    fn observe_msd_block(&self, base: u32, numbers: u64, passing: u64) {
        if !self.search_enabled() {
            return;
        }
        self.numbers_seen.fetch_add(numbers, Ordering::Relaxed);
        self.passing_seen.fetch_add(passing, Ordering::Relaxed);
        if self.search_base.load(Ordering::Relaxed) == base {
            return;
        }
        let now = Instant::now();
        let mut search = self.search.lock().unwrap();
        self.search_base.store(base, Ordering::Relaxed);
        if let Some(floor) = search.note_block(base, now)
            && !self.pinned.load(Ordering::Relaxed)
        {
            debug!("GPU MSD floor: {floor:.0} for base {base} (search: base switch)");
            self.floor_bits.store(floor.to_bits(), Ordering::Relaxed);
            search.floor_changed(now, floor);
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
        if self.search_enabled() {
            self.search_observe();
            return;
        }
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
    /// Hold the floor at `floor` until the next thaw. Returns `false`, and
    /// does nothing, under an environment pin. The search keeps learning at
    /// the pinned level (the benchmark's sweep feeds it).
    #[allow(clippy::cast_precision_loss)]
    fn pin(&self, floor: u128) -> bool {
        if self.env_pinned {
            return false;
        }
        // Under the search lock, which a worker's base switch also takes,
        // so the switch cannot land between the store and the pin.
        let search = self.search_enabled().then(|| self.search.lock().unwrap());
        self.floor_bits
            .store((floor as f64).to_bits(), Ordering::Relaxed);
        self.pinned.store(true, Ordering::Relaxed);
        if let Some(mut search) = search {
            let now = Instant::now();
            search.pinned(now);
            search.floor_changed(now, floor as f64);
        }
        true
    }

    /// The dispatcher closed its last open field.
    fn pipeline_idle(&self) {
        if self.search_enabled() {
            self.search.lock().unwrap().idle_since = Some(Instant::now());
        }
    }

    /// The dispatcher opens a field with none open. If the pipeline sat idle
    /// long enough to matter (a claim late, not the benchmark's back-to-back
    /// windows), restart the search's tick: see [`LadderSearch::resume`].
    /// What the workers counted meanwhile (the tail of the last field) is
    /// dropped with the settling tick.
    fn pipeline_resumed(&self) {
        if !self.search_enabled() {
            return;
        }
        let now = Instant::now();
        let mut search = self.search.lock().unwrap();
        if search
            .idle_since
            .take()
            .is_some_and(|t| now.duration_since(t) >= SEARCH_IDLE_GAP)
        {
            self.numbers_seen.store(0, Ordering::Relaxed);
            self.passing_seen.store(0, Ordering::Relaxed);
            search.resume(now, f64::from_bits(self.floor_bits.load(Ordering::Relaxed)));
        }
    }

    /// The search's accounting: every [`SEARCH_TICK`], attribute the numbers
    /// that went through the workers to the level in force, and move if the
    /// search asks to. Under a pin (the benchmark's sweep or freeze) it only
    /// learns.
    fn search_observe(&self) {
        let now = Instant::now();
        let mut search = self.search.lock().unwrap();
        if now.duration_since(search.tick_start) < SEARCH_TICK {
            return;
        }
        let floor = f64::from_bits(self.floor_bits.load(Ordering::Relaxed));
        let numbers = self.numbers_seen.swap(0, Ordering::Relaxed);
        let passing = self.passing_seen.swap(0, Ordering::Relaxed);
        let pinned = self.pinned.load(Ordering::Relaxed);
        if let Some((new_floor, why)) = search.tick(now, floor, numbers, passing, pinned) {
            if (new_floor - floor).abs() > f64::EPSILON {
                debug!("GPU MSD floor: {floor:.0} → {new_floor:.0} (search: {why})");
                self.floor_bits
                    .store(new_floor.to_bits(), Ordering::Relaxed);
                search.floor_changed(now, new_floor);
            } else {
                debug!("GPU MSD floor: {floor:.0} held (search: {why})");
            }
        }
    }
}

impl FloorController {
    /// Stop steering and hold the current floor. Returns it. A floor pinned
    /// by the environment is unaffected (it is already held).
    fn freeze(&self) -> u128 {
        self.pinned.store(true, Ordering::Relaxed);
        if self.search_enabled()
            && let Ok(mut search) = self.search.lock()
        {
            search.learning = false;
        }
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
        if self.search_enabled() {
            // Under the search lock, which a worker's base switch also takes,
            // so the switch cannot land between the new floor and the unpin.
            let now = Instant::now();
            let mut search = self.search.lock().unwrap();
            let (floor, why) = search.on_thaw(now);
            debug!("GPU MSD floor: thaw → {floor:.0} (search: {why})");
            search.floor_changed(now, floor);
            self.floor_bits.store(floor.to_bits(), Ordering::Relaxed);
            self.pinned.store(false, Ordering::Relaxed);
            return;
        }
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
        let mut controller = FloorController::new(MSD_FLOOR_SEED, false);
        match std::env::var("NICE_GPU_FLOOR_CONTROLLER").as_deref() {
            Ok("heuristic") => {}
            Ok("search") | Err(_) => controller.search_enabled = true,
            Ok(other) => {
                warn!(
                    "ignoring unknown NICE_GPU_FLOOR_CONTROLLER '{other}' (search or heuristic); using the search"
                );
                controller.search_enabled = true;
            }
        }
        debug!(
            "GPU MSD floor: steered by the {} controller, seed {MSD_FLOOR_SEED:.0}",
            controller.name()
        );
        controller
    })
}

/// Let the benchmark steer a scenario's floor: resumes steering, the
/// measured search from the best level of the pinned sweep that preceded it,
/// the wait heuristic from its seed. Call after the scenario's pinned sweep;
/// pair with [`benchmark_floor_freeze`] before its measured windows. An
/// explicit `NICE_GPU_MSD_FLOOR` still wins, so floor sweeps under
/// `--benchmark` remain possible: both calls are then no-ops.
///
/// Why not simply pin: a steered floor is what production runs at, and it
/// differs by machine in both directions (measured: an RTX 4090 with six
/// cores settles at the cap, an RTX 3060 with nineteen near 100k, and the
/// pinned cap undersold the latter by a third). Why not steer through the
/// measurement: the controllers move within a second and the windows are
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

/// Hold the floor at `floor` (the benchmark's pinned sweep). Returns `false`,
/// and does nothing, under an explicit `NICE_GPU_MSD_FLOOR` pin, which the
/// sweep must not disturb. Pair with [`benchmark_floor_thaw`] to steer again.
#[allow(clippy::cast_precision_loss)]
#[must_use]
pub fn benchmark_floor_pin(floor: u128) -> bool {
    floor_controller().pin(floor)
}

/// Floors the benchmark sweeps: the measured search's seven analysed-leaf
/// floors and the power-of-two floors between them (all but 31.25k).
///
/// The recursion halves a chunk until a node is no larger than the floor,
/// and emits a node *without* analysing it when it is no larger than the
/// floor but analyses it and emits it whole when it is larger than the floor
/// yet smaller than twice the floor. So the floor picks one of two
/// configurations per level: at exactly a power-of-two size `P` the `P`
/// nodes ship unanalysed (the halves of their analysed parents); at any
/// floor strictly between `P/2` and `P` the `P` nodes are analysed and the
/// passing ones ship whole. The second does the same host work as the pin at
/// `P/2` and hands the device the same surviving volume in half as many
/// descriptors, and it beat the first at every level on every host measured;
/// the search's ladder is those floors (`search_floor`). The power-of-two
/// floors stay in the sweep because the wait heuristic lands on them (its
/// 500k cap, its 62.5k clamp), so the report shows what they cost.
pub const BENCHMARK_FLOOR_LADDER: usize = 12;

/// See [`BENCHMARK_FLOOR_LADDER`]: 750k, 500k, 375k, 250k, 187.5k, 125k,
/// 93.75k, 62.5k, 46.875k, 23.4k, 15.6k, 11.7k.
#[must_use]
pub fn benchmark_floor_candidates() -> [u128; BENCHMARK_FLOOR_LADDER] {
    [
        750_000, 500_000, 375_000, 250_000, 187_500, 125_000, 93_750, 62_500, 46_875, 23_437,
        15_625, 11_718,
    ]
}

/// The MSD floor currently in force, for reports. Initialises the controller
/// if nothing has yet.
#[must_use]
pub fn msd_floor_in_use() -> u128 {
    floor_controller().floor()
}

/// Which floor controller this process runs (`pinned`, `search` or
/// `heuristic`), for reports. Initialises the controller if nothing has yet.
#[must_use]
pub fn msd_floor_controller() -> &'static str {
    floor_controller().name()
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
/// wait heuristic's "device is behind" signal, and small enough that the
/// work in flight stays near the device's own pace, which the measured
/// search reads. `NICE_GPU_BATCHES_IN_FLIGHT`.
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
/// u64 cross-end certificate mask. That encoding, not the filter, is what
/// bounds a range — fields are 1e13 numbers at most (the CUDA throughput
/// harness uses that size), far inside u64, so the offset always fits, but a
/// range longer than `u32::MAX` would not, which is why this can fail.
fn descriptors_for_chunk(
    chunk: FieldSize,
    base: u32,
    floor: u128,
    field_start: u128,
    profile: &mut msd_prefix_filter::DepthProfile,
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
    msd_prefix_filter::get_valid_ranges_recursive_masked_profiled(
        chunk,
        &msd_prefix_filter::MaskedRecursion {
            base,
            fixed_lsd_k: GPU_LSD_K as usize,
            max_depth: msd_prefix_filter::MSD_RECURSIVE_MAX_DEPTH,
            min_range_size: floor,
            subdivision_factor: msd_prefix_filter::MSD_RECURSIVE_SUBDIVISION_FACTOR,
        },
        &mut leaves,
        profile,
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
#[cfg(test)]
fn descriptors_for_block(
    block: FieldSize,
    base: u32,
    floor: u128,
    field_start: u128,
) -> Result<(Vec<u64>, Vec<u32>, Vec<u64>)> {
    let mut profile = msd_prefix_filter::DepthProfile::default();
    descriptors_for_block_profiled(block, base, floor, field_start, &mut profile)
}

/// Descriptors for one block (the no-MSD bypass goes chunk by chunk), also
/// filling the recursion's depth profile.
fn descriptors_for_block_profiled(
    block: FieldSize,
    base: u32,
    floor: u128,
    field_start: u128,
    profile: &mut msd_prefix_filter::DepthProfile,
) -> Result<(Vec<u64>, Vec<u32>, Vec<u64>)> {
    if floor >= PROCESSING_CHUNK_SIZE {
        let mut offsets = Vec::new();
        let mut lens = Vec::new();
        let mut masks = Vec::new();
        for chunk in block.chunks(PROCESSING_CHUNK_SIZE) {
            let (o, l, m) = descriptors_for_chunk(chunk, base, floor, field_start, profile)?;
            offsets.extend(o);
            lens.extend(l);
            masks.extend(m);
        }
        return Ok((offsets, lens, masks));
    }
    descriptors_for_chunk(block, base, floor, field_start, profile)
}

/// Descriptors for one block, and how many of its numbers survived the
/// recursion's block-level (depth-0) analysis: the measured search's
/// difficulty index.
#[allow(clippy::type_complexity)]
fn msd_block(
    block: FieldSize,
    base: u32,
    floor: u128,
    field_start: u128,
) -> Result<(Vec<u64>, Vec<u32>, Vec<u64>, u64)> {
    let mut profile = msd_prefix_filter::DepthProfile::default();
    let (offsets, lens, masks) =
        descriptors_for_block_profiled(block, base, floor, field_start, &mut profile)?;
    let passing = u64::try_from(profile.passing_volume[0]).unwrap_or(u64::MAX);
    Ok((offsets, lens, masks, passing))
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
    /// Descriptors for `field`, from `units` work units of it.
    Ranges {
        field: u64,
        units: u32,
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
            units: u32::try_from(batch.units).unwrap_or(u32::MAX),
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
            let controller = floor_controller();
            let floor = controller.floor();
            match msd_block(block, work.base, floor, work.range.start()) {
                Ok((offsets, lens, masks, passing)) => {
                    controller.observe_msd_block(
                        work.base,
                        u64::try_from(block.size()).unwrap_or(u64::MAX),
                        passing,
                    );
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
    /// Work units the launch buffer covers, survivors or not.
    buf_units: usize,
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
        self.buf_units = 0;
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
                if self.open.is_empty() {
                    self.controller.pipeline_resumed();
                }
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
                units,
                offsets,
                lens,
                masks,
            } => {
                if self.buf_field != Some(field) {
                    self.flush_launch();
                    self.buf_field = Some(field);
                }
                self.buf_units += units as usize;
                if let Some(open) = self.open.get_mut(&field) {
                    open.num_ranges += offsets.len();
                    open.valid_numbers += lens.iter().map(|&l| u64::from(l)).sum::<u64>();
                }
                self.buf_offsets.extend_from_slice(&offsets);
                self.buf_lens.extend_from_slice(&lens);
                self.buf_masks.extend_from_slice(&masks);
                if self.buf_offsets.len() >= LAUNCH_BATCH_RANGES
                    || self.buf_units >= LAUNCH_BATCH_UNITS
                {
                    self.flush_launch();
                }
                None
            }
            Msg::End { seq, msd_secs } => {
                if self.buf_field == Some(seq) {
                    self.flush_launch();
                }
                let open = self.open.remove(&seq)?;
                if self.open.is_empty() {
                    self.controller.pipeline_idle();
                }
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
        buf_units: 0,
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

    #[test]
    fn search_floors_realise_analysed_leaf_levels() {
        // Level l analyses nodes of chunk >> l and ships the passing ones
        // whole, so the leaf count equals the passing count at that depth
        // and nothing deeper is analysed. An MSD-weak base-57 window keeps
        // every level populated (level 0 is the chunk itself, passing).
        let start = 58_549_892_695_752_322_464u128;
        let chunk = FieldSize::new(start, start + PROCESSING_CHUNK_SIZE);
        let params = |floor: u128| msd_prefix_filter::MaskedRecursion {
            base: 57,
            fixed_lsd_k: GPU_LSD_K as usize,
            max_depth: msd_prefix_filter::MSD_RECURSIVE_MAX_DEPTH,
            min_range_size: floor,
            subdivision_factor: msd_prefix_filter::MSD_RECURSIVE_SUBDIVISION_FACTOR,
        };
        for level in 0..SEARCH_LEVELS {
            assert_eq!(search_level(search_floor(level)), Some(level));
            let depth = level;
            let mut leaves = Vec::new();
            let mut profile = msd_prefix_filter::DepthProfile::default();
            #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
            let floor = search_floor(level) as u128;
            msd_prefix_filter::get_valid_ranges_recursive_masked_profiled(
                chunk,
                &params(floor),
                &mut leaves,
                &mut profile,
            );
            // Halving 1e6 `depth` times leaves two sizes one apart; a leaf is
            // either.
            let leaf_size = PROCESSING_CHUNK_SIZE >> depth;
            assert!(
                leaves
                    .iter()
                    .all(|(r, _)| r.size() == leaf_size || r.size() == leaf_size + 1),
                "level {level}: leaf sizes"
            );
            assert!(profile.passing[depth] > 0, "level {level}: no survivors");
            assert_eq!(
                leaves.len() as u64,
                profile.passing[depth],
                "level {level}: leaf count"
            );
            assert_eq!(
                profile.analyzed[depth + 1],
                0,
                "level {level}: analysed too deep"
            );
            if depth >= 1 {
                assert_eq!(
                    profile.analyzed[depth],
                    2 * profile.passing[depth - 1],
                    "level {level}: fan-out"
                );
            }
        }
        // The power-of-two pins realise no level; the benchmark's truncated
        // pins of 23437.5 and 11718.75 still map to their levels.
        assert_eq!(search_level(500_000.0), None);
        assert_eq!(search_level(250_000.0), None);
        assert_eq!(search_level(62_500.0), None);
        assert_eq!(search_level(750_000.0), Some(0));
        assert_eq!(search_level(375_000.0), Some(1));
        assert_eq!(search_level(23_437.0), Some(5));
        assert_eq!(search_level(11_718.0), Some(6));
        assert_eq!(search_level(15_625.0), None);
        // Every one of the search's floors is in the benchmark's sweep.
        for level in 0..SEARCH_LEVELS {
            #[allow(clippy::cast_precision_loss)]
            let swept = benchmark_floor_candidates()
                .iter()
                .any(|&f| search_level(f as f64) == Some(level));
            assert!(swept, "level {level} not swept by the benchmark");
        }
        // Difficulty bins over the block-level survival fraction.
        assert_eq!(difficulty_bin(0.0), 0);
        assert_eq!(difficulty_bin(0.019), 0);
        assert_eq!(difficulty_bin(0.02), 1);
        assert_eq!(difficulty_bin(0.1), 2);
        assert_eq!(difficulty_bin(0.3), 3);
        assert_eq!(difficulty_bin(0.6), 4);
        assert_eq!(difficulty_bin(0.75), 5);
        assert_eq!(difficulty_bin(1.0), 5);
    }

    /// Synthetic throughput of `level` on a stretch whose block-level
    /// survival is `phi`: the level's base rate, scaled by how easy the
    /// stretch is.
    fn synthetic_rate(base: &[f64; SEARCH_LEVELS], level: usize, phi: f64) -> f64 {
        base[level] * (1.6 - phi)
    }

    /// Drive the search for `ticks` ticks, cycling the stretch difficulty
    /// through `phis`, moving the floor as it asks.
    #[allow(clippy::too_many_arguments)]
    fn drive_search(
        s: &mut LadderSearch,
        now: &mut Instant,
        floor: &mut f64,
        base: &[f64; SEARCH_LEVELS],
        phis: &[f64],
        cursor: &mut usize,
        ticks: usize,
        pinned: bool,
    ) {
        for _ in 0..ticks {
            *now += SEARCH_TICK;
            let phi = phis[*cursor % phis.len()];
            *cursor += 1;
            let rate = search_level(*floor).map_or(0.0, |l| synthetic_rate(base, l, phi));
            #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
            let numbers = (rate * SEARCH_TICK.as_secs_f64()) as u64;
            #[allow(
                clippy::cast_possible_truncation,
                clippy::cast_sign_loss,
                clippy::cast_precision_loss
            )]
            let passing = (phi * numbers as f64) as u64;
            if let Some((new_floor, _)) = s.tick(*now, *floor, numbers, passing, pinned)
                && (new_floor - *floor).abs() > f64::EPSILON
            {
                *floor = new_floor;
                s.floor_changed(*now, new_floor);
            }
        }
    }

    /// Drive one tick at a time until the base's phase satisfies `pred`, or
    /// `max` ticks pass.
    #[allow(clippy::too_many_arguments)]
    fn drive_until(
        s: &mut LadderSearch,
        now: &mut Instant,
        floor: &mut f64,
        base: &[f64; SEARCH_LEVELS],
        phis: &[f64],
        cursor: &mut usize,
        max: usize,
        pred: impl Fn(&SearchPhase) -> bool,
    ) -> bool {
        let b = s.current_base.unwrap();
        for _ in 0..max {
            if pred(&s.bases[&b].phase) {
                return true;
            }
            drive_search(s, now, floor, base, phis, cursor, 1, false);
        }
        pred(&s.bases[&b].phase)
    }

    /// Ticks in one hold plus a full trial, with slack.
    fn hold_cycle_ticks() -> usize {
        usize::try_from(SEARCH_HOLD.as_millis() / SEARCH_TICK.as_millis()).unwrap()
            + usize::from(SEARCH_SETTLE_TICKS)
            + usize::from(SEARCH_VISIT_TICKS)
            + 8
    }

    const PHIS: [f64; 4] = [0.05, 0.3, 0.6, 0.9];

    #[test]
    fn search_sweeps_to_the_peak_then_trials_neighbours() {
        let base = [4.5e11, 5.0e11, 6.0e11, 7.0e11, 6.5e11, 5.0e11, 4.0e11];
        let mut now = Instant::now();
        let mut s = LadderSearch::new(now, search_floor(1));
        let mut floor = s.note_block(40, now).expect("first base switches");
        assert!(
            (floor - search_floor(0)).abs() < 1e-9,
            "a new base sweeps from the top"
        );
        s.floor_changed(now, floor);
        let mut cur = 0usize;
        // Seven levels at (settle + visit) ticks each, plus the moves.
        let sweep_ticks = SEARCH_LEVELS
            * (usize::from(SEARCH_SETTLE_TICKS) + usize::from(SEARCH_VISIT_TICKS))
            + 16;
        drive_search(
            &mut s,
            &mut now,
            &mut floor,
            &base,
            &PHIS,
            &mut cur,
            sweep_ticks,
            false,
        );
        let st = &s.bases[&40];
        assert_eq!(st.level, 3, "sweep settles on the peak");
        assert!((floor - search_floor(3)).abs() < 1e-9);
        assert!(
            matches!(st.phase, SearchPhase::Hold { .. }),
            "{:?}",
            st.phase
        );
        // Each hold ends in a trial of the neighbour in the current direction,
        // compared within like-difficulty bins; it loses and flips the
        // direction: first finer, then coarser. Level stays put.
        for expected_dir in [-1i8, 1] {
            drive_search(
                &mut s,
                &mut now,
                &mut floor,
                &base,
                &PHIS,
                &mut cur,
                hold_cycle_ticks(),
                false,
            );
            let st = &s.bases[&40];
            assert_eq!(st.level, 3, "a losing trial returns home");
            assert!((floor - search_floor(3)).abs() < 1e-9);
            assert!(
                matches!(st.phase, SearchPhase::Hold { dir, .. } if dir == expected_dir),
                "{:?}",
                st.phase
            );
        }
        // The finer neighbour becomes 9% faster: the next trial there wins.
        let mut shifted = base;
        shifted[4] = 7.6e11;
        let mut adopted = false;
        for _ in 0..4 {
            drive_search(
                &mut s,
                &mut now,
                &mut floor,
                &shifted,
                &PHIS,
                &mut cur,
                hold_cycle_ticks(),
                false,
            );
            if s.bases[&40].level == 4 {
                adopted = true;
                break;
            }
        }
        assert!(adopted, "a neighbour beyond the hysteresis is adopted");
        assert!((floor - search_floor(4)).abs() < 1e-9);
    }

    /// Ticks one sweep visit takes: the settle after the move, then the
    /// clean ticks.
    fn visit_ticks() -> usize {
        usize::from(SEARCH_SETTLE_TICKS) + usize::from(SEARCH_VISIT_TICKS)
    }

    #[test]
    fn search_sweep_passes_over_a_level_seen_only_in_an_unshared_bin() {
        // The coarsest level is ten times faster on paper, but its whole
        // visit falls on a stretch of easy digits (bin 0) no other level
        // samples: nothing can be compared with it, so it is not the pick.
        let base = [5.0e12, 5.0e11, 6.0e11, 7.0e11, 6.5e11, 5.0e11, 4.0e11];
        let mut now = Instant::now();
        let mut s = LadderSearch::new(now, search_floor(1));
        let mut floor = s.note_block(40, now).expect("first base switches");
        s.floor_changed(now, floor);
        let mut cur = 0usize;
        drive_search(
            &mut s,
            &mut now,
            &mut floor,
            &base,
            &[0.01],
            &mut cur,
            visit_ticks(),
            false,
        );
        assert!(
            (floor - search_floor(1)).abs() < 1e-9,
            "the first visit is over and the sweep has moved on"
        );
        drive_search(
            &mut s,
            &mut now,
            &mut floor,
            &base,
            &PHIS,
            &mut cur,
            (SEARCH_LEVELS - 1) * visit_ticks() + 8,
            false,
        );
        let st = &s.bases[&40];
        assert!(
            matches!(st.phase, SearchPhase::Hold { .. }),
            "{:?}",
            st.phase
        );
        assert_eq!(st.level, 3, "the peak among the comparable levels");
        assert!((floor - search_floor(3)).abs() < 1e-9);
        assert_eq!(st.pick_best(st.sweep_since, 1), Some(3));
    }

    #[test]
    fn search_resweeps_when_no_two_levels_share_a_bin() {
        let base = [4.5e11, 5.0e11, 6.0e11, 7.0e11, 6.5e11, 5.0e11, 4.0e11];
        let mut now = Instant::now();
        let mut s = LadderSearch::new(now, search_floor(1));
        let mut floor = s.note_block(40, now).expect("first base switches");
        s.floor_changed(now, floor);
        // Six levels each in a bin of their own, the seventh with a single
        // tick in a bin another level holds (a bin sample older than the
        // blend window restarts at one tick, so one tick per bin is what a
        // visit across flickering difficulty leaves): nothing compares.
        {
            let st = s.bases.get_mut(&40).unwrap();
            for (level, &rate) in base.iter().enumerate().take(SEARCH_LEVELS - 1) {
                for _ in 0..SEARCH_MIN_BIN_TICKS {
                    st.record(level, level, rate, now);
                }
            }
            st.record(SEARCH_LEVELS - 1, 0, base[SEARCH_LEVELS - 1], now);
            assert_eq!(st.pick_best(st.sweep_since, 1), None);
            st.level = SEARCH_LEVELS - 1;
            st.phase = SearchPhase::Sweep {
                ticks: SEARCH_VISIT_TICKS - 1,
            };
            now += SEARCH_TICK;
            assert_eq!(
                st.step(SEARCH_LEVELS - 1, now),
                Some((0, "sweep unranked, sweeping again"))
            );
            assert!(matches!(st.phase, SearchPhase::Sweep { ticks: 0 }));
            assert_eq!(st.level, 0, "sweeping again from the top");
            assert_eq!(st.sweep_since, now, "the stale samples are out");
        }
        floor = search_floor(0);
        s.floor_changed(now, floor);
        // Mixed digits from here: the second sweep ranks and settles.
        let mut cur = 0usize;
        drive_search(
            &mut s,
            &mut now,
            &mut floor,
            &base,
            &PHIS,
            &mut cur,
            SEARCH_LEVELS * visit_ticks() + 16,
            false,
        );
        let st = &s.bases[&40];
        assert!(
            matches!(st.phase, SearchPhase::Hold { .. }),
            "{:?}",
            st.phase
        );
        assert_eq!(st.level, 3);
        assert!((floor - search_floor(3)).abs() < 1e-9);
    }

    #[test]
    fn search_compares_levels_only_within_like_difficulty() {
        let now = Instant::now();
        let since = now.checked_sub(Duration::from_secs(1)).unwrap();
        let mut st = BaseSearch::new(now);
        // Level 0 sampled on easy and medium stretches, level 1 on medium and
        // hard: only the medium bin is common, and the ratio comes from it
        // alone, whatever the other bins say.
        for _ in 0..3 {
            st.record(0, 1, 10.0, now);
            st.record(0, 2, 8.0, now);
            st.record(1, 2, 9.0, now);
            st.record(1, 3, 5.0, now);
        }
        let r = st.compare(1, 0, since).expect("one shared bin");
        assert!((r - 9.0 / 8.0).abs() < 1e-9, "{r}");
        // A level sampled only on stretches the other never saw compares to
        // nothing; a bin with a single tick does not count.
        st.record(2, 5, 100.0, now);
        assert_eq!(st.compare(2, 0, since), None);
        st.record(2, 1, 100.0, now);
        assert_eq!(st.compare(2, 0, since), None, "one tick is not a bin");
        st.record(2, 1, 100.0, now);
        assert!(st.compare(2, 0, since).is_some());
        // Samples before `since` do not take part.
        assert_eq!(st.compare(1, 0, now + Duration::from_secs(1)), None);
    }

    #[test]
    fn search_trial_is_judged_within_shared_bins() {
        // Home measured across all stretches, the neighbour only on hard
        // ones. Its raw rate is far lower, but within the shared bin it is
        // what the base rates say, so the trial is judged on that.
        let base = [4.5e11, 5.0e11, 6.0e11, 7.0e11, 6.5e11, 5.0e11, 4.0e11];
        let mut now = Instant::now();
        let mut s = LadderSearch::new(now, search_floor(1));
        let mut floor = s.note_block(40, now).unwrap();
        s.floor_changed(now, floor);
        let mut cur = 0usize;
        let sweep_ticks = SEARCH_LEVELS
            * (usize::from(SEARCH_SETTLE_TICKS) + usize::from(SEARCH_VISIT_TICKS))
            + 16;
        drive_search(
            &mut s,
            &mut now,
            &mut floor,
            &base,
            &PHIS,
            &mut cur,
            sweep_ticks,
            false,
        );
        assert_eq!(s.bases[&40].level, 3);
        let is_trial = |p: &SearchPhase| matches!(p, SearchPhase::Trial { .. });
        assert!(drive_until(
            &mut s,
            &mut now,
            &mut floor,
            &base,
            &PHIS,
            &mut cur,
            hold_cycle_ticks(),
            is_trial
        ));
        assert_eq!(s.bases[&40].level, 4, "trial of the finer neighbour");
        // Hard stretches only during the trial: raw rate 6.5e11 * 0.7 against
        // home's recent mix, yet the shared hard bin says 6.5 < 7.0: lost.
        let hard = [0.9];
        drive_search(
            &mut s,
            &mut now,
            &mut floor,
            &base,
            &hard,
            &mut cur,
            usize::from(SEARCH_SETTLE_TICKS) + usize::from(SEARCH_VISIT_TICKS) + 2,
            false,
        );
        let st = &s.bases[&40];
        assert_eq!(st.level, 3, "judged within the shared bin, the trial lost");
        assert!(
            matches!(st.phase, SearchPhase::Hold { dir: -1, .. }),
            "{:?}",
            st.phase
        );
        // A trial that shares no bin with home is void: back home, direction
        // kept. Let the sweep's samples age out of the comparison window,
        // give home fresh samples on one difficulty only, then feed the trial
        // a difficulty home never saw.
        now += SEARCH_COMPARE_WINDOW + Duration::from_secs(1);
        s.floor_changed(now, floor);
        drive_search(
            &mut s,
            &mut now,
            &mut floor,
            &base,
            &[0.3],
            &mut cur,
            6,
            false,
        );
        assert!(drive_until(
            &mut s,
            &mut now,
            &mut floor,
            &base,
            &[0.3],
            &mut cur,
            60,
            is_trial
        ));
        assert_eq!(s.bases[&40].level, 2, "trial of the coarser neighbour");
        drive_search(
            &mut s,
            &mut now,
            &mut floor,
            &base,
            &[0.0],
            &mut cur,
            usize::from(SEARCH_SETTLE_TICKS) + usize::from(SEARCH_VISIT_TICKS) + 2,
            false,
        );
        let st = &s.bases[&40];
        assert_eq!(st.level, 3, "void trial returns home");
        assert!(
            matches!(st.phase, SearchPhase::Hold { dir: -1, .. }),
            "void trial keeps its direction: {:?}",
            st.phase
        );
    }

    /// A tick that spans a stall is not a sample (and the next settles); a
    /// resume after an idle spell discards the first tick; a floor that is
    /// not the base's level, with nobody pinning it, is steered back.
    #[test]
    fn search_discards_stalls_and_realigns() {
        let mut now = Instant::now();
        let mut s = LadderSearch::new(now, search_floor(1));
        let floor = s.note_block(40, now).unwrap();
        s.floor_changed(now, floor);
        let tick = |s: &mut LadderSearch, now: &mut Instant, floor: f64, dt: Duration| {
            *now += dt;
            s.tick(*now, floor, 1_000_000_000, 500_000_000, false)
        };
        let samples = |s: &LadderSearch| -> Vec<(usize, u32)> {
            s.bases[&40].bins[0]
                .iter()
                .enumerate()
                .filter_map(|(b, x)| x.map(|x| (b, x.ticks)))
                .collect()
        };
        for _ in 0..SEARCH_SETTLE_TICKS {
            assert!(tick(&mut s, &mut now, floor, SEARCH_TICK).is_none());
        }
        assert!(samples(&s).is_empty(), "settling ticks are not samples");
        tick(&mut s, &mut now, floor, SEARCH_TICK);
        let one = samples(&s);
        assert_eq!(one.len(), 1, "a clean tick is a sample");
        // A stalled tick: not recorded, and the next one settles.
        assert!(tick(&mut s, &mut now, floor, SEARCH_MAX_TICK + SEARCH_TICK).is_none());
        tick(&mut s, &mut now, floor, SEARCH_TICK);
        assert_eq!(samples(&s), one, "stall and its settle recorded nothing");
        // After an idle spell the first tick is discarded too.
        s.resume(now, floor);
        tick(&mut s, &mut now, floor, SEARCH_TICK);
        assert_eq!(samples(&s), one, "resume settles");
        // Someone else's floor (a level, or none): back to the base's level
        // once the new floor has settled.
        for other in [search_floor(3), 500_000.0] {
            assert!(tick(&mut s, &mut now, other, SEARCH_TICK).is_none());
            for _ in 0..SEARCH_SETTLE_TICKS {
                assert!(tick(&mut s, &mut now, other, SEARCH_TICK).is_none());
            }
            let (to, why) = tick(&mut s, &mut now, other, SEARCH_TICK).unwrap();
            assert_eq!(why, "realign");
            assert!((to - floor).abs() < 1e-9);
        }
    }

    /// A neighbour that measures faster by less than the hysteresis is not
    /// adopted.
    #[test]
    fn search_trial_needs_to_clear_the_hysteresis() {
        let sweep = [4.5e11, 5.0e11, 6.0e11, 7.0e11, 6.9e11, 5.0e11, 4.0e11];
        let mut now = Instant::now();
        let mut s = LadderSearch::new(now, search_floor(1));
        let mut floor = s.note_block(40, now).unwrap();
        s.floor_changed(now, floor);
        let mut cur = 0usize;
        let sweep_ticks = SEARCH_LEVELS
            * (usize::from(SEARCH_SETTLE_TICKS) + usize::from(SEARCH_VISIT_TICKS))
            + 16;
        drive_search(
            &mut s,
            &mut now,
            &mut floor,
            &sweep,
            &PHIS,
            &mut cur,
            sweep_ticks,
            false,
        );
        assert_eq!(s.bases[&40].level, 3);
        // The finer neighbour is now 3% faster than home: still not enough.
        let close = [4.5e11, 5.0e11, 6.0e11, 7.0e11, 7.21e11, 5.0e11, 4.0e11];
        let is_trial = |p: &SearchPhase| matches!(p, SearchPhase::Trial { .. });
        assert!(drive_until(
            &mut s,
            &mut now,
            &mut floor,
            &close,
            &PHIS,
            &mut cur,
            hold_cycle_ticks(),
            is_trial
        ));
        assert_eq!(s.bases[&40].level, 4);
        drive_search(
            &mut s,
            &mut now,
            &mut floor,
            &close,
            &PHIS,
            &mut cur,
            usize::from(SEARCH_SETTLE_TICKS) + usize::from(SEARCH_VISIT_TICKS) + 2,
            false,
        );
        let st = &s.bases[&40];
        assert_eq!(st.level, 3, "a 3% gain is inside the hysteresis");
        assert!(
            matches!(st.phase, SearchPhase::Hold { dir: -1, .. }),
            "{:?}",
            st.phase
        );
    }

    #[test]
    fn search_learns_under_pins_and_thaws_to_the_best() {
        let base = [3.5e11, 4.0e11, 5.0e11, 5.5e11, 6.0e11, 5.2e11, 4.5e11];
        let mut now = Instant::now();
        let mut s = LadderSearch::new(now, search_floor(1));
        let mut floor = s.note_block(50, now).unwrap();
        s.floor_changed(now, floor);
        let mut cur = 0usize;
        let flat = [0.5];
        // The benchmark's sweep: pin every ladder floor (and the power-of-two
        // ones, which realise no level) for a few ticks; the search may not
        // move.
        for pin in benchmark_floor_candidates() {
            #[allow(clippy::cast_precision_loss)]
            let f = pin as f64;
            floor = f;
            s.pinned(now);
            s.floor_changed(now, f);
            drive_search(
                &mut s, &mut now, &mut floor, &base, &flat, &mut cur, 5, true,
            );
            assert!((floor - f).abs() < 1e-9, "pinned: the search moved");
        }
        let st = &s.bases[&50];
        for level in 0..SEARCH_LEVELS {
            assert!(
                st.sampled_since(level, s.epoch),
                "pinned level {level} sampled"
            );
        }
        assert!(
            matches!(st.phase, SearchPhase::Sweep { ticks: 0 }),
            "pins do not advance the sweep"
        );
        // A frozen measured window at level 0 must not teach the search
        // anything, however fast: the next scenario's sweep decides alone.
        s.learning = false;
        floor = search_floor(0);
        s.floor_changed(now, floor);
        let frozen = [1.0e15; SEARCH_LEVELS];
        drive_search(
            &mut s, &mut now, &mut floor, &frozen, &flat, &mut cur, 6, true,
        );
        let r0 = s.bases[&50].bins[0][difficulty_bin(0.5)].unwrap().rate;
        assert!(r0 < 1.0e12, "frozen window leaked: {r0}");
        let (thawed, _) = s.on_thaw(now);
        assert!(
            (thawed - search_floor(4)).abs() < 1e-9,
            "thaw starts at the best pinned level"
        );
        assert!(matches!(s.bases[&50].phase, SearchPhase::Hold { .. }));
        // Without fresh data the thaw sweeps from the top instead.
        s.note_block(52, now);
        assert!((s.on_thaw(now).0 - search_floor(0)).abs() < 1e-9);
        // A new pinned sweep after a freeze starts an epoch: what was
        // measured before it does not decide the thaw, however good.
        s.note_block(53, now);
        floor = search_floor(5);
        s.floor_changed(now, floor);
        let fast = [1.0e15; SEARCH_LEVELS];
        drive_search(
            &mut s, &mut now, &mut floor, &fast, &flat, &mut cur, 5, true,
        );
        s.learning = false;
        now += Duration::from_secs(5);
        for level in [0usize, 1, 2] {
            floor = search_floor(level);
            s.pinned(now);
            s.floor_changed(now, floor);
            drive_search(
                &mut s, &mut now, &mut floor, &base, &flat, &mut cur, 5, true,
            );
        }
        let (thawed, _) = s.on_thaw(now);
        assert!(
            (thawed - search_floor(2)).abs() < 1e-9,
            "best of the levels pinned since the epoch, not the stale fast one"
        );
        // Two levels are not enough to skip the sweep.
        s.note_block(54, now);
        for level in [1usize, 4] {
            floor = search_floor(level);
            s.floor_changed(now, floor);
            drive_search(
                &mut s, &mut now, &mut floor, &base, &flat, &mut cur, 5, true,
            );
        }
        assert!((s.on_thaw(now).0 - search_floor(0)).abs() < 1e-9);
    }

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
                        descriptors_for_chunk(
                            c,
                            base,
                            4000,
                            f.start(),
                            &mut msd_prefix_filter::DepthProfile::default(),
                        )
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
            .recv_timeout(Duration::from_mins(2))
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

    /// The benchmark's pin holds a floor through any wait signal until the
    /// thaw, which resumes steering from the seed; under an environment pin
    /// it does nothing.
    #[test]
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    fn benchmark_pin_holds_until_thaw_and_yields_to_the_environment() {
        let interval = FLOOR_ADJUST_INTERVAL;
        let c = FloorController::new(MSD_FLOOR_SEED, false);
        assert!(c.pin(93_750));
        assert_eq!(c.floor(), 93_750);
        // A pinned floor ignores the wait signal however one-sided it is.
        c.state.lock().unwrap().interval_start = Instant::now()
            .checked_sub(interval + Duration::from_micros(10))
            .unwrap();
        c.observe(interval, Duration::ZERO);
        assert_eq!(c.floor(), 93_750);
        // Thaw resumes steering from the seed.
        c.thaw(MSD_FLOOR_SEED);
        assert_eq!(c.floor(), MSD_FLOOR_SEED as u128);
        assert!(!c.pinned.load(Ordering::Relaxed));
        // An environment pin is never disturbed.
        let env = FloorController::new(77_000.0, true);
        assert!(!env.pin(93_750));
        assert_eq!(env.floor(), 77_000);
        assert_eq!(env.name(), "pinned");
        assert_eq!(c.name(), "heuristic");
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
                        descriptors_for_chunk(
                            c,
                            base,
                            floor,
                            f.start(),
                            &mut msd_prefix_filter::DepthProfile::default(),
                        )
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
            let (o, l, m) = descriptors_for_chunk(
                chunk,
                base,
                floor,
                field.start(),
                &mut msd_prefix_filter::DepthProfile::default(),
            )
            .unwrap();
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

        // Unit bound: bypass-style single descriptors, one per unit.
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
        let (offsets, lens, masks) = descriptors_for_chunk(
            chunk,
            base,
            PROCESSING_CHUNK_SIZE,
            start,
            &mut msd_prefix_filter::DepthProfile::default(),
        )
        .unwrap();
        assert_eq!(offsets, vec![u64::try_from(mid - start).unwrap()]);
        assert_eq!(lens, vec![PROCESSING_CHUNK_SIZE as u32]);
        assert_eq!(masks, vec![0]);

        // A sub-bypass floor produces the masked recursion's leaves.
        let (offsets, lens, masks) = descriptors_for_chunk(
            chunk,
            base,
            msd_prefix_filter::MSD_RECURSIVE_MIN_RANGE_SIZE,
            start,
            &mut msd_prefix_filter::DepthProfile::default(),
        )
        .unwrap();
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
