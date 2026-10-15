//! Coarse per-field progress, from the CPU and GPU paths alike.
//!
//! The paths report through a process-wide sink: the client installs one
//! that draws a bar, and everything else (tests, the browser build,
//! `--no-progress`, the benchmark) leaves it empty, which makes every call
//! here a no-op.
//!
//! What a unit means is up to the caller and only needs to be monotone: CPU
//! fields count chunks or join partitions done, GPU detailed fields count
//! launched batches, niceonly fields on the GPU's stride pipeline count MSD
//! blocks filtered on the host, and fields on the GPU's overlap join count
//! partitions the device has run. Fields are identified by their range,
//! since the niceonly pipeline keeps more than one open at a time.

use crate::FieldSize;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, Ordering};

/// Where per-field progress goes. Implementations must tolerate calls from
/// any thread and in any interleaving across fields. Threads that share a
/// field can report its positions out of order.
pub trait ProgressSink: Send + Sync {
    /// A field of `units` units started; each unit stands for about
    /// `numbers_per_unit` numbers.
    fn begin(&self, range: &FieldSize, units: u64, numbers_per_unit: u128);
    /// `done` units of the field are complete (an absolute position).
    fn advance(&self, range: &FieldSize, done: u64);
    /// The field's results are in.
    fn finish(&self, range: &FieldSize);
}

static SINK: OnceLock<Box<dyn ProgressSink>> = OnceLock::new();

/// Install the process-wide sink. Only the first call takes effect.
pub fn install(sink: impl ProgressSink + 'static) {
    let _ = SINK.set(Box::new(sink));
}

pub(crate) fn begin(range: &FieldSize, units: u64, numbers_per_unit: u128) {
    if let Some(s) = SINK.get() {
        s.begin(range, units, numbers_per_unit);
    }
}

pub(crate) fn advance(range: &FieldSize, done: u64) {
    if let Some(s) = SINK.get() {
        s.advance(range, done);
    }
}

pub(crate) fn finish(range: &FieldSize) {
    if let Some(s) = SINK.get() {
        s.finish(range);
    }
}

/// A field's progress reaches the sink in steps of `1/STEPS` of its units
/// (and at its end), however small the units are.
const STEPS: u64 = 1_000;

/// A field's progress for work that owns it start to finish: units count
/// up, from any number of threads, and dropping it (on any exit path)
/// finishes the field.
pub struct FieldProgress {
    range: FieldSize,
    units: u64,
    /// Units done, from every thread (past `units` if work runs again).
    done: AtomicU64,
    /// The sink hears of every `step` units, and of the last.
    step: u64,
}

impl FieldProgress {
    /// Begin a field split into units of `unit` numbers (the last one may be
    /// partial).
    #[must_use]
    pub fn begin(range: &FieldSize, unit: u128) -> Self {
        let units = u64::try_from(range.size().div_ceil(unit)).unwrap_or(u64::MAX);
        Self::with_units(range, units, unit)
    }

    /// Begin a field of `units` units of about `numbers_per_unit` numbers.
    #[must_use]
    pub fn with_units(range: &FieldSize, units: u64, numbers_per_unit: u128) -> Self {
        begin(range, units, numbers_per_unit);
        Self {
            range: *range,
            units,
            done: AtomicU64::new(0),
            step: units.div_ceil(STEPS).max(1),
        }
    }

    /// One more unit done.
    pub fn tick(&self) {
        self.add(1);
    }

    /// `n` more units done. The position stops at the field's units, so
    /// work that runs again (a join batch whose survivor list overflowed)
    /// cannot take the bar past its end.
    pub fn add(&self, n: u64) {
        if let Some(done) = self.count(n) {
            advance(&self.range, done);
        }
    }

    /// Count `n` more units. Returns the position to report if the count
    /// crossed a step or reached the end.
    fn count(&self, n: u64) -> Option<u64> {
        if n == 0 {
            return None;
        }
        let before = self.done.fetch_add(n, Ordering::Relaxed);
        if before >= self.units {
            return None;
        }
        let after = before.saturating_add(n).min(self.units);
        (after == self.units || after / self.step != before / self.step).then_some(after)
    }
}

impl Drop for FieldProgress {
    fn drop(&mut self) {
        finish(&self.range);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ticks_from_eight_threads_end_exactly_at_the_units() {
        let units = 80_000;
        let p = FieldProgress::with_units(&FieldSize::new(0, 1_000), units, 1);
        let reported: Vec<Vec<u64>> = std::thread::scope(|s| {
            let threads: Vec<_> = (0..8)
                .map(|_| {
                    s.spawn(|| {
                        // Each thread's tenth extra tick runs past the end.
                        (0..units / 8 + 10).filter_map(|_| p.count(1)).collect()
                    })
                })
                .collect();
            threads.into_iter().map(|t| t.join().unwrap()).collect()
        });
        let all: Vec<u64> = reported.into_iter().flatten().collect();
        assert_eq!(all.iter().filter(|&&d| d == units).count(), 1, "one end");
        assert_eq!(all.iter().max(), Some(&units));
        // Every step is crossed by exactly one thread's tick.
        assert_eq!(all.len(), 1_000);
    }

    #[test]
    fn a_million_units_reach_the_sink_a_thousand_times() {
        let p = FieldProgress::with_units(&FieldSize::new(0, 1_000_000), 1_000_000, 1);
        let reported: Vec<u64> = (0..1_000_000).filter_map(|_| p.count(1)).collect();
        assert_eq!(reported.len(), 1_000);
        assert!(reported.windows(2).all(|w| w[1] - w[0] == 1_000));
        assert_eq!(reported.last(), Some(&1_000_000));
    }

    #[test]
    fn a_jump_across_steps_is_reported_once() {
        // 10,000 units in steps of 10.
        let p = FieldProgress::with_units(&FieldSize::new(0, 10), 10_000, 1);
        assert_eq!(p.count(25), Some(25), "across two steps");
        assert_eq!(p.count(3), None, "inside a step");
        assert_eq!(p.count(2), Some(30));
        assert_eq!(p.count(0), None);
        assert_eq!(p.count(u64::MAX / 2), Some(10_000), "capped at the end");
        assert_eq!(p.count(1), None, "the end is reported once");
    }

    #[test]
    fn small_fields_report_every_unit() {
        let p = FieldProgress::begin(&FieldSize::new(0, 20_000_000), 1_000_000);
        let reported: Vec<u64> = (0..20).filter_map(|_| p.count(1)).collect();
        assert_eq!(reported, (1..=20).collect::<Vec<u64>>());
    }
}
