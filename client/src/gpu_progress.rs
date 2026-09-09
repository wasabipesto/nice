//! The GPU progress bars: an indicatif sink for [`nice_common::progress`].
//!
//! The CPU path's bar comes from `simple-tqdm` over the rayon iterator. The
//! GPU paths report through `nice_common::progress` instead, and this is
//! what the client hangs on the other end: one bar per open field, drawn as
//! a group so the niceonly pipeline's overlapping fields each get a line.
//! A finished field's bar is cleared; the "✓ Processed" log line is the
//! record that stays in the scrollback.

use indicatif::{MultiProgress, ProgressBar, ProgressDrawTarget, ProgressState, ProgressStyle};
use nice_common::FieldSize;
use nice_common::progress::ProgressSink;
use std::collections::HashMap;
use std::io::{self, IsTerminal, Write};
use std::sync::Mutex;
use std::time::Duration;

/// The bars for every field currently open on the device.
pub struct GpuProgress {
    multi: MultiProgress,
    bars: Mutex<HashMap<(u128, u128), ProgressBar>>,
}

impl GpuProgress {
    /// `None` when stderr is not a terminal: there is nothing to draw on,
    /// and log lines routed through a hidden group would be dropped.
    pub fn new() -> Option<Self> {
        if !io::stderr().is_terminal() {
            return None;
        }
        Some(Self::with_draw_target(ProgressDrawTarget::stderr()))
    }

    fn with_draw_target(target: ProgressDrawTarget) -> Self {
        Self {
            multi: MultiProgress::with_draw_target(target),
            bars: Mutex::new(HashMap::new()),
        }
    }

    /// A writer that prints above the bars instead of through them, so the
    /// logger can keep writing to stderr without shredding a bar mid-draw.
    pub fn log_writer(&self) -> Box<dyn Write + Send> {
        Box::new(LogWriter {
            multi: self.multi.clone(),
            buf: Vec::new(),
        })
    }
}

/// The same shape as the CPU bar: `simple-tqdm`'s template with the rate in
/// units of `numbers_per_unit`, labelled by its power of ten.
fn style(numbers_per_unit: u128) -> ProgressStyle {
    let unit = format!("e{}", numbers_per_unit.max(1).ilog10());
    ProgressStyle::with_template(
        "{percent}|{wide_bar:.white}| {pos}/{len} [{elapsed}<{eta}, {per_sec}{msg}]",
    )
    .expect("static template")
    .with_key(
        "per_sec",
        move |state: &ProgressState, w: &mut dyn std::fmt::Write| {
            let _ = write!(w, "{:.2}{unit}/s", state.per_sec());
        },
    )
    .progress_chars("█▉▊▋▌▍▎▏ ")
}

impl ProgressSink for GpuProgress {
    fn begin(&self, range: &FieldSize, units: u64, numbers_per_unit: u128) {
        let bar = self
            .multi
            .add(ProgressBar::new(units).with_style(style(numbers_per_unit)));
        // Keep the clock moving while the host waits on the device.
        bar.enable_steady_tick(Duration::from_millis(250));
        self.bars
            .lock()
            .unwrap()
            .insert((range.start(), range.end()), bar);
    }

    fn advance(&self, range: &FieldSize, done: u64) {
        let bars = self.bars.lock().unwrap();
        if let Some(bar) = bars.get(&(range.start(), range.end())) {
            bar.set_position(done);
            if bar.length().is_some_and(|len| done >= len) {
                bar.set_message(", waiting on device");
            }
        }
    }

    fn finish(&self, range: &FieldSize) {
        if let Some(bar) = self
            .bars
            .lock()
            .unwrap()
            .remove(&(range.start(), range.end()))
        {
            bar.finish_and_clear();
            self.multi.remove(&bar);
        }
    }
}

/// Line-buffers the logger's output and prints each line above the bars.
struct LogWriter {
    multi: MultiProgress,
    buf: Vec<u8>,
}

impl Write for LogWriter {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        self.buf.extend_from_slice(bytes);
        while let Some(nl) = self.buf.iter().position(|&b| b == b'\n') {
            let line: Vec<u8> = self.buf.drain(..=nl).collect();
            self.multi
                .println(String::from_utf8_lossy(&line).trim_end_matches('\n'))?;
        }
        Ok(bytes.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        if !self.buf.is_empty() {
            let line = std::mem::take(&mut self.buf);
            self.multi.println(String::from_utf8_lossy(&line))?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn open_fields(p: &GpuProgress) -> usize {
        p.bars.lock().unwrap().len()
    }

    #[test]
    fn bars_track_open_fields_by_range() {
        let p = GpuProgress::with_draw_target(ProgressDrawTarget::hidden());
        let a = FieldSize::new(0, 1_000);
        let b = FieldSize::new(1_000, 3_000);
        p.begin(&a, 10, 100);
        p.begin(&b, 20, 100);
        assert_eq!(open_fields(&p), 2);

        p.advance(&a, 4);
        p.advance(&b, 20);
        {
            let bars = p.bars.lock().unwrap();
            assert_eq!(bars[&(0, 1_000)].position(), 4);
            assert_eq!(bars[&(1_000, 3_000)].position(), 20);
            assert!(bars[&(1_000, 3_000)].message().contains("device"));
            assert!(bars[&(0, 1_000)].message().is_empty());
        }

        p.finish(&a);
        assert_eq!(open_fields(&p), 1);
        // Finishing again, or a field never begun, is harmless.
        p.finish(&a);
        p.advance(&FieldSize::new(5, 6), 1);
        p.finish(&b);
        assert_eq!(open_fields(&p), 0);
    }

    #[test]
    fn log_writer_splits_lines() {
        let p = GpuProgress::with_draw_target(ProgressDrawTarget::hidden());
        let mut w = p.log_writer();
        w.write_all(b"one\ntwo\nthr").unwrap();
        w.write_all(b"ee\n").unwrap();
        w.flush().unwrap();
        // Nothing to observe through a hidden target; the point is that no
        // write path panics or errors on partial lines.
    }
}
