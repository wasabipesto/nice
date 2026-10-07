//! The GPU progress bar: an indicatif sink for [`nice_common::progress`].
//!
//! The CPU path's bar comes from `simple-tqdm` over the rayon iterator. The
//! GPU paths report through `nice_common::progress` instead, and this is
//! what the client hangs on the other end: one bar, for the oldest field
//! still open. The niceonly stride pipeline opens a field before the one
//! ahead of it is done (the host filters one field's blocks after the
//! other's, while the device drains the first), so a bar per open field
//! left one line standing still. Every open field keeps a bar of its own,
//! with its own position, and a clock that starts with its first unit of
//! work, but only the oldest is drawn; when it finishes, the next takes the
//! line. A finished field's bar is cleared;
//! the "✓ Processed" log line is the record that stays in the scrollback.

use indicatif::{MultiProgress, ProgressBar, ProgressDrawTarget, ProgressState, ProgressStyle};
use nice_common::FieldSize;
use nice_common::progress::ProgressSink;
use std::collections::VecDeque;
use std::io::{self, IsTerminal, Write};
use std::sync::Mutex;
use std::time::Duration;

/// The bar for the oldest field open on the device.
pub struct GpuProgress {
    multi: MultiProgress,
    /// Every open field's bar, in the order the fields began; only the
    /// first is drawn.
    fields: Mutex<VecDeque<((u128, u128), ProgressBar)>>,
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
            fields: Mutex::new(VecDeque::new()),
        }
    }

    /// Draw `bar`, the oldest open field's.
    fn show(&self, bar: &ProgressBar) {
        let bar = self.multi.add(bar.clone());
        // Keep the clock moving while the host waits on the device.
        bar.enable_steady_tick(Duration::from_millis(250));
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

/// The same shape as the CPU bar: `simple-tqdm`'s template, with the rate in
/// numbers per second (a unit is `numbers_per_unit` of them, not always a
/// power of ten: a join partition is a field's share).
fn style(numbers_per_unit: u128) -> ProgressStyle {
    #[allow(clippy::cast_precision_loss)]
    let scale = numbers_per_unit.max(1) as f64;
    ProgressStyle::with_template(
        "{percent}|{wide_bar:.white}| {pos}/{len} [{elapsed}<{eta}, {per_sec}{msg}]",
    )
    .expect("static template")
    .with_key(
        "per_sec",
        move |state: &ProgressState, w: &mut dyn std::fmt::Write| {
            let _ = write!(w, "{:.2e}/s", state.per_sec() * scale);
        },
    )
    .progress_chars("█▉▊▋▌▍▎▏ ")
}

/// Once the host is done with a field and the device drains it, the rate
/// would only decay towards zero; the bar shows the time it has waited.
fn waiting_style() -> ProgressStyle {
    ProgressStyle::with_template("{percent}|{wide_bar:.white}| {pos}/{len} [{elapsed}{msg}]")
        .expect("static template")
        .progress_chars("█▉▊▋▌▍▎▏ ")
}

fn key(range: &FieldSize) -> (u128, u128) {
    (range.start(), range.end())
}

impl ProgressSink for GpuProgress {
    fn begin(&self, range: &FieldSize, units: u64, numbers_per_unit: u128) {
        let bar = ProgressBar::with_draw_target(Some(units), ProgressDrawTarget::hidden())
            .with_style(style(numbers_per_unit));
        let mut fields = self.fields.lock().unwrap();
        if fields.is_empty() {
            self.show(&bar);
        }
        fields.push_back((key(range), bar));
    }

    fn advance(&self, range: &FieldSize, done: u64) {
        let fields = self.fields.lock().unwrap();
        if let Some((_, bar)) = fields.iter().find(|(k, _)| *k == key(range)) {
            let started = bar.position() == 0 && done > 0;
            bar.set_position(done);
            if started {
                // The field's work has begun: its clock and rate run from
                // here, not from when it was opened behind another field.
                bar.reset_elapsed();
                bar.reset_eta();
            }
            if bar.length().is_some_and(|len| done >= len) && bar.message().is_empty() {
                bar.set_style(waiting_style());
                bar.set_message(", waiting on device");
            }
        }
    }

    fn finish(&self, range: &FieldSize) {
        let mut fields = self.fields.lock().unwrap();
        let Some(i) = fields.iter().position(|(k, _)| *k == key(range)) else {
            return;
        };
        let Some((_, bar)) = fields.remove(i) else {
            return;
        };
        if i == 0 {
            bar.finish_and_clear();
            self.multi.remove(&bar);
            if let Some((_, next)) = fields.front() {
                self.show(next);
            }
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

    /// The open fields, oldest (the drawn one) first, with their positions.
    fn open_fields(p: &GpuProgress) -> Vec<((u128, u128), u64)> {
        p.fields
            .lock()
            .unwrap()
            .iter()
            .map(|(k, bar)| (*k, bar.position()))
            .collect()
    }

    #[test]
    fn the_oldest_open_field_has_the_bar() {
        let p = GpuProgress::with_draw_target(ProgressDrawTarget::hidden());
        let a = FieldSize::new(0, 1_000);
        let b = FieldSize::new(1_000, 3_000);
        p.begin(&a, 10, 100);
        p.begin(&b, 20, 100);
        assert_eq!(open_fields(&p), [((0, 1_000), 0), ((1_000, 3_000), 0)]);

        // The field behind can move first; its bar keeps the position.
        p.advance(&a, 4);
        p.advance(&b, 20);
        assert_eq!(open_fields(&p), [((0, 1_000), 4), ((1_000, 3_000), 20)]);
        {
            let fields = p.fields.lock().unwrap();
            assert!(fields[1].1.message().contains("device"));
            assert!(
                fields[0].1.message().is_empty(),
                "the oldest is not waiting"
            );
        }

        // When the oldest finishes, the next takes the line where it is.
        p.finish(&a);
        assert_eq!(open_fields(&p), [((1_000, 3_000), 20)]);
        // Finishing again, or a field never begun, is harmless.
        p.finish(&a);
        p.advance(&FieldSize::new(5, 6), 1);
        p.finish(&b);
        assert!(open_fields(&p).is_empty(), "every field finished");
    }

    #[test]
    fn a_later_field_can_finish_first() {
        let p = GpuProgress::with_draw_target(ProgressDrawTarget::hidden());
        let (a, b, c) = (
            FieldSize::new(0, 10),
            FieldSize::new(10, 20),
            FieldSize::new(20, 30),
        );
        for f in [&a, &b, &c] {
            p.begin(f, 5, 2);
        }
        p.finish(&b);
        assert_eq!(open_fields(&p), [((0, 10), 0), ((20, 30), 0)]);
        p.finish(&a);
        assert_eq!(open_fields(&p), [((20, 30), 0)]);
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
