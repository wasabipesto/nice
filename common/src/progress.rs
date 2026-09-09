//! Coarse per-field progress from the GPU paths.
//!
//! The CPU path gets a progress bar for free from the rayon iterator it maps
//! over. The GPU paths drive the device from loops and threads inside this
//! crate, so they report through a process-wide sink instead: the client
//! installs one that draws a bar, and everything else (tests, the browser
//! build, `--no-progress`) leaves it empty, which makes every call here a
//! no-op.
//!
//! What a unit means is up to the caller and only needs to be monotone:
//! detailed fields count launched batches, niceonly fields count MSD blocks
//! filtered on the host. Fields are identified by their range, since the
//! niceonly pipeline keeps more than one open at a time.

use crate::FieldSize;
use std::sync::OnceLock;

/// Where per-field progress goes. Implementations must tolerate calls from
/// any thread and in any interleaving across fields.
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

/// A field's progress for a loop that owns it start to finish: ticks count
/// up, and dropping it (on any exit path) finishes the field.
pub(crate) struct FieldProgress {
    range: FieldSize,
    done: u64,
}

impl FieldProgress {
    /// Begin a field split into units of `unit` numbers (the last one may be
    /// partial).
    pub(crate) fn begin(range: &FieldSize, unit: u128) -> Self {
        let units = u64::try_from(range.size().div_ceil(unit)).unwrap_or(u64::MAX);
        begin(range, units, unit);
        Self {
            range: *range,
            done: 0,
        }
    }

    /// One more unit done.
    pub(crate) fn tick(&mut self) {
        self.done += 1;
        advance(&self.range, self.done);
    }
}

impl Drop for FieldProgress {
    fn drop(&mut self) {
        finish(&self.range);
    }
}
