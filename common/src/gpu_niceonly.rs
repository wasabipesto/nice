//! Backend-neutral host pipeline for GPU niceonly fields.
//!
//! Both GPU backends run niceonly the same way: the CPU runs the real MSD
//! prefix filter across all cores with a coarser recursion floor than the CPU
//! client uses (see [`AdaptiveFloor`]), and ships only compact *range
//! descriptors* — ~12 bytes per surviving range — to the device, which
//! reconstructs the stride filter's candidates itself. No per-candidate data
//! ever crosses the bus.
//!
//! Everything in that sentence is independent of CUDA and Vulkan, so it lives
//! here rather than being written twice. The backends supply a [`RangeSink`]:
//! CUDA enqueues asynchronous launches on its stream, Vulkan records and
//! submits a dispatch. This is the same split as [`crate::gpu_config`], which
//! holds the per-base kernel constants for the same reason —
//! [`crate::client_process_gpu`] is `#![cfg(feature = "gpu")]` and unreachable
//! from a Vulkan-only build.

#![cfg(any(feature = "gpu", feature = "vulkan"))]
#![allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]

use crate::{FieldSize, msd_prefix_filter};
use anyhow::Result;
use log::{debug, info, warn};
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
/// field at floor 250 is ~9e7 surviving ranges, and 12 bytes apiece is over a
/// gigabyte of queued descriptors. Bounding the channel keeps the overlap —
/// workers refill the queue while the consumer waits on the device.
///
/// **The unit here is one chunk, not one launch batch.** Each item a worker
/// sends is the whole output of one [`PROCESSING_CHUNK_SIZE`] chunk;
/// [`LAUNCH_BATCH_RANGES`] is the consumer's flush threshold and never bounds
/// what sits in the channel. So the cap is `PIPELINE_DEPTH` × the most ranges a
/// chunk can yield — the recursion returns a range whole once it is at or below
/// the floor, so that is about `PROCESSING_CHUNK_SIZE / floor`, i.e. ~4000 at
/// [`MSD_FLOOR_MIN`] and fewer at any coarser floor. Roughly 3 MB of
/// descriptors at the worst floor, comfortably below the gigabyte above.
const PIPELINE_DEPTH: usize = 64;

/// Minimum MSD recursion floor (matches the CPU client's default).
/// Below this the GPU receives virtually the same candidates as the CPU would
/// check itself, so there is no point going lower.
const MSD_FLOOR_MIN: f64 = 250.0;

/// Maximum useful MSD recursion floor. Beyond ~64 000 the survival rate
/// saturates around 23 % (b52 measurement), so larger values buy nothing.
/// Measured on a real b52 production field (per 1e12 numbers, single core):
///
/// | floor  | CPU time | surviving |
/// |--------|----------|-----------|
/// | 250    | 350 s    | 2.3 %     |
/// | 4 000  | 50 s     | 15.2 %    |
/// | 16 000 | 15 s     | 19.0 %    |
/// | 64 000 | 4.8 s    | 22.6 %    |
const MSD_FLOOR_MAX: f64 = 256_000.0;

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
            info!(
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
                    info!("GPU MSD floor fixed at {f:.0} via NICE_GPU_MSD_FLOOR");
                    return Mutex::new(AdaptiveFloor {
                        floor: f,
                        warmup: u32::MAX,
                    });
                }
                _ => warn!("ignoring invalid NICE_GPU_MSD_FLOOR '{v}'; using adaptive floor"),
            }
        }
        #[allow(clippy::cast_precision_loss)]
        let cpu_count = std::thread::available_parallelism()
            .map(std::num::NonZeroUsize::get)
            .unwrap_or(32) as f64;
        let seed = (ADAPT_BASE_CORE_PRODUCT / cpu_count).clamp(MSD_FLOOR_MIN, MSD_FLOOR_MAX);
        info!("GPU MSD floor: adaptive, seed {seed:.0} ({cpu_count:.0} logical cores)");
        Mutex::new(AdaptiveFloor {
            floor: seed,
            warmup: ADAPT_WARMUP,
        })
    })
}

fn gpu_msd_floor() -> u128 {
    adaptive_floor().lock().unwrap().current()
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
    /// are candidate counts; the two slices are the same length.
    ///
    /// # Errors
    /// Returns an error on any device failure.
    fn launch(&mut self, offsets: &[u64], lens: &[u32]) -> Result<()>;

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
/// Each surviving range becomes 12 bytes: a u64 offset and a u32 length. That
/// encoding, not the filter, is what bounds a range — a field is at most 1e12
/// numbers so the offset always fits, but a range longer than `u32::MAX` would
/// not, which is why this can fail.
fn descriptors_for_chunk(
    chunk: FieldSize,
    base: u32,
    floor: u128,
    field_start: u128,
) -> Result<(Vec<u64>, Vec<u32>)> {
    let mut offsets: Vec<u64> = Vec::new();
    let mut lens: Vec<u32> = Vec::new();
    for sub in msd_prefix_filter::get_valid_ranges_recursive(
        chunk,
        base,
        0,
        msd_prefix_filter::MSD_RECURSIVE_MAX_DEPTH,
        floor,
        msd_prefix_filter::MSD_RECURSIVE_SUBDIVISION_FACTOR,
    ) {
        let offset = u64::try_from(sub.start() - field_start);
        let len = u32::try_from(sub.size());
        match (offset, len) {
            (Ok(offset), Ok(len)) => {
                offsets.push(offset);
                lens.push(len);
            }
            _ => anyhow::bail!(
                "valid range doesn't fit descriptor: start {} size {}",
                sub.start(),
                sub.size()
            ),
        }
    }
    Ok((offsets, lens))
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
        .map(std::num::NonZeroUsize::get)
        .unwrap_or(4)
        .min(chunks.len().max(1));

    let next_chunk = AtomicUsize::new(0);
    let worker_error: Mutex<Option<anyhow::Error>> = Mutex::new(None);
    let (tx, rx) = std::sync::mpsc::sync_channel::<(Vec<u64>, Vec<u32>)>(PIPELINE_DEPTH);

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
    let mut launch_error: Option<anyhow::Error> = None;

    std::thread::scope(|scope| {
        let chunks = &chunks;
        let next_chunk = &next_chunk;
        let worker_error = &worker_error;
        for _ in 0..num_threads {
            let tx = tx.clone();
            scope.spawn(move || {
                loop {
                    let i = next_chunk.fetch_add(1, Ordering::Relaxed);
                    let Some(chunk) = chunks.get(i) else { break };
                    match descriptors_for_chunk(*chunk, base, floor, range.start()) {
                        Ok((offsets, lens)) if !offsets.is_empty() => {
                            // A closed channel means the consumer gave up on a
                            // launch error; the remaining chunks are moot.
                            if tx.send((offsets, lens)).is_err() {
                                break;
                            }
                        }
                        Ok(_) => {}
                        Err(e) => {
                            *worker_error.lock().unwrap() = Some(e);
                            return;
                        }
                    }
                }
            });
        }
        // The consumer runs on this thread while the workers produce. The
        // clone of `tx` held by each worker keeps the channel open; dropping
        // ours lets `recv` disconnect once they all finish.
        drop(tx);

        while let Ok((offsets, lens)) = rx.recv() {
            stats.num_ranges += offsets.len();
            stats.valid_numbers += lens.iter().map(|&l| u64::from(l)).sum::<u64>();
            buf_offsets.extend_from_slice(&offsets);
            buf_lens.extend_from_slice(&lens);
            if buf_offsets.len() >= LAUNCH_BATCH_RANGES {
                let t = Instant::now();
                let outcome = sink.launch(&buf_offsets, &buf_lens);
                stats.device_secs += t.elapsed().as_secs_f64();
                if let Err(e) = outcome {
                    launch_error = Some(e);
                    break;
                }
                stats.launches += 1;
                buf_offsets.clear();
                buf_lens.clear();
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
        sink.launch(&buf_offsets, &buf_lens)?;
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
    info!(
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

#[cfg(test)]
mod tests {
    use super::*;

    use std::sync::mpsc;
    use std::time::Duration;

    /// A sink that records what it was handed instead of touching a device.
    #[derive(Default)]
    struct Recorder {
        batches: Vec<(Vec<u64>, Vec<u32>)>,
        synced: bool,
    }

    impl RangeSink for Recorder {
        fn launch(&mut self, offsets: &[u64], lens: &[u32]) -> Result<()> {
            self.batches.push((offsets.to_vec(), lens.to_vec()));
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
        fn launch(&mut self, _offsets: &[u64], _lens: &[u32]) -> Result<()> {
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
    /// (`LAUNCH_BATCH_RANGES`) and then fill `PIPELINE_DEPTH` behind it, which
    /// is why this is not a toy range.
    ///
    /// Run under a timeout so a regression fails the test instead of hanging
    /// the suite.
    #[test]
    fn a_failing_sink_unblocks_the_workers_instead_of_deadlocking() {
        let (tx, rx) = mpsc::channel();
        std::thread::spawn(move || {
            // Base 40 from its band start: ~2e5 surviving ranges over 5e10,
            // so the consumer reaches several flushes with the workers still
            // going. Base 50 is no good here — MSD prunes its band start to
            // nothing, and nothing is ever launched.
            let base = 40;
            let start = crate::base_range::get_base_range_u128(base)
                .unwrap()
                .unwrap()
                .range_start;
            let field = FieldSize::new(start, start + 50_000_000_000);
            let _ = tx.send(run_range_pipeline(&mut Failing, &field, base).is_err());
        });
        let errored = rx
            .recv_timeout(Duration::from_secs(120))
            .expect("run_range_pipeline did not return: the workers are deadlocked");
        assert!(errored, "the launch failure must reach the caller");
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
        let range = crate::base_range::get_base_range_u128(base).unwrap().unwrap();
        let field = FieldSize::new(range.range_start, range.range_end);

        let mut sink = Recorder::default();
        let stats = run_range_pipeline(&mut sink, &field, base).expect("pipeline");
        assert!(sink.synced, "the pipeline must sync the sink");
        assert_eq!(stats.num_ranges, sink.batches.iter().map(|(o, _)| o.len()).sum::<usize>());

        let mut covered: Vec<(u128, u128)> = Vec::new();
        for (offsets, lens) in &sink.batches {
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
