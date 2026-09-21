//! One progress bar, many workers.
//!
//! Long parallel walks over on-disk data (see [`crate::inspect`]) have no
//! natural place to print per-thread progress: four threads interleaving their
//! own percentages is less readable than no output at all. [`ByteProgress`] is
//! the alternative -- a single bar whose total is known up front (bytes to
//! read) and whose workers only ever say "I finished N more bytes".
//!
//! Bytes are the unit on purpose: they are the one measure of work that is
//! known before any parsing happens, so the bar is accurate from the first
//! frame instead of drifting as the shape of a file becomes clear.

use std::io::{IsTerminal, Write};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;
use std::time::{Duration, Instant};

/// Human-readable byte count, so a 2 GB trace does not read as a bare integer.
pub fn human(bytes: u64) -> String {
    const UNITS: [&str; 5] = ["B", "KiB", "MiB", "GiB", "TiB"];
    let mut v = bytes as f64;
    let mut u = 0;
    while v >= 1024.0 && u + 1 < UNITS.len() {
        v /= 1024.0;
        u += 1;
    }
    if u == 0 {
        format!("{bytes} B")
    } else {
        format!("{v:.2} {}", UNITS[u])
    }
}

/// `M:SS`, the only resolution worth showing for a walk measured in seconds.
fn mmss(d: Duration) -> String {
    let s = d.as_secs();
    format!("{}:{:02}", s / 60, s % 60)
}

const BAR_WIDTH: usize = 33;
const MIN_REDRAW: Duration = Duration::from_millis(100);
const LOG_STEP_PERCENT: u64 = 10;

/// A total-known progress bar that any number of threads may advance.
///
/// Advancing is a relaxed atomic add; drawing is behind a mutex, so frames
/// never interleave on stderr and are always rendered from the newest total
/// (a slow worker cannot draw a stale, backwards-looking frame).
///
/// When stderr is not a terminal the bar degrades to one log line every
/// [`LOG_STEP_PERCENT`]: `\r`-redrawn frames are noise in a log file.
pub struct ByteProgress {
    /// Prefix naming the work, e.g. `"inspect"`.
    label: String,
    /// Total bytes this bar represents. Never zero (see [`Self::new`]).
    total: u64,
    done: AtomicU64,
    draw: Mutex<DrawState>,
    /// Whether stderr is a terminal, decided once at construction.
    tty: bool,
    started: Instant,
}

struct DrawState {
    /// Permille last drawn, so an advance that does not move the bar draws
    /// nothing.
    last_permille: u64,
    /// When the last frame was drawn, for the [`MIN_REDRAW`] throttle.
    last_draw: Instant,
    /// Set by [`ByteProgress::finish`]: a late advance from a worker that is
    /// already accounted for must not reopen the line.
    finished: bool,
}

impl ByteProgress {
    /// A bar over `total` bytes of work. A zero total is treated as one byte
    /// so the percentage is always defined; such a bar simply jumps to 100%.
    pub fn new(label: impl Into<String>, total: u64) -> Self {
        Self {
            label: label.into(),
            total: total.max(1),
            done: AtomicU64::new(0),
            draw: Mutex::new(DrawState {
                // 0/1000 is a legitimate first frame, so start out of range to
                // force it.
                last_permille: u64::MAX,
                last_draw: Instant::now(),
                finished: false,
            }),
            tty: std::io::stderr().is_terminal(),
            started: Instant::now(),
        }
    }

    /// Record `bytes` more of the work as done, from any thread.
    pub fn advance(&self, bytes: u64) {
        if bytes == 0 {
            return;
        }
        self.done.fetch_add(bytes, Ordering::Relaxed);
        self.draw(false);
    }

    /// Close the bar: draw a final 100% frame and end the line.
    ///
    /// A total that was never reached is a bug in the caller's accounting
    /// rather than something to show the user, so it is logged at `debug` and
    /// the bar still completes -- a bar stuck at 97% reads as a hang.
    pub fn finish(&self) {
        let done = self.done.load(Ordering::Relaxed);
        if done != self.total {
            log::debug!(
                "progress '{}': {done} of {} byte(s) accounted for",
                self.label,
                self.total
            );
        }
        self.done.store(self.total, Ordering::Relaxed);
        // Forced, unless the last worker's own advance already drew the full
        // bar -- redrawing an identical frame is a duplicate line in a log.
        if self.draw.lock().unwrap().last_permille != 1000 {
            self.draw(true);
        }
        let mut st = self.draw.lock().unwrap();
        if !st.finished {
            st.finished = true;
            if self.tty {
                // nosemgrep: error-handling-ignored-critical-result -- a broken stderr pipe must not kill a finished run
                let _ = writeln!(std::io::stderr());
            }
        }
    }

    /// Draw a frame if it would show something new, or unconditionally when
    /// `force`. The current total is read under the lock so concurrent
    /// advances can only ever draw the bar forwards.
    fn draw(&self, force: bool) {
        let mut st = self.draw.lock().unwrap();
        if st.finished {
            return;
        }
        let done = self.done.load(Ordering::Relaxed).min(self.total);
        let permille = 1000 * done / self.total;
        let stale = st.last_permille == u64::MAX;
        if !force
            && (permille == st.last_permille || (!stale && st.last_draw.elapsed() < MIN_REDRAW))
        {
            return;
        }

        if self.tty {
            // \r plus erase-to-end-of-line: the frame overwrites the previous
            // one, and a shrinking line leaves no tail behind.
            // nosemgrep: error-handling-ignored-critical-result -- a broken stderr pipe must not kill a finished run
            let _ = write!(
                std::io::stderr(),
                "\r\x1b[2K{}",
                frame(&self.label, done, self.total, self.started.elapsed())
            );
            let _ = std::io::stderr().flush();
        } else {
            // Only whole LOG_STEP_PERCENT crossings, and never the same one
            // twice, so a log file gets a fixed number of lines.
            let step = |p: u64| p / (10 * LOG_STEP_PERCENT);
            if stale || step(permille) > step(st.last_permille) {
                log::info!(
                    "{}: {}",
                    self.label,
                    frame_text(done, self.total, self.started.elapsed())
                );
            } else {
                // Nothing drawn, so leave last_draw/last_permille alone: the
                // next crossing is still owed a line.
                return;
            }
        }
        st.last_permille = permille;
        st.last_draw = Instant::now();
    }
}

/// One rendered bar line. Pure, so the layout is testable without a terminal.
fn frame(label: &str, done: u64, total: u64, elapsed: Duration) -> String {
    let filled = (BAR_WIDTH as u64 * done / total.max(1)) as usize;
    format!(
        "{label} [{}{}] {}",
        "#".repeat(filled),
        ".".repeat(BAR_WIDTH - filled),
        frame_text(done, total, elapsed),
    )
}

/// The text half of a frame: percentage, byte counts, elapsed and ETA. Shared
/// with the non-terminal path so both report identical numbers.
fn frame_text(done: u64, total: u64, elapsed: Duration) -> String {
    let total = total.max(1);
    // ETA from the mean rate so far. Withheld until 1% is done, where the rate
    // estimate is still dominated by thread start-up.
    let eta = if done * 100 >= total && done < total {
        let remaining = elapsed.mul_f64((total - done) as f64 / done as f64);
        format!(" eta {}", mmss(remaining))
    } else {
        String::new()
    };
    format!(
        "{:>3}% {}/{} in {}{eta}",
        100 * done / total,
        human(done),
        human(total),
        mmss(elapsed),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn human_scales_units() {
        assert_eq!(human(0), "0 B");
        assert_eq!(human(1023), "1023 B");
        assert_eq!(human(1024), "1.00 KiB");
        assert_eq!(human(50 << 20), "50.00 MiB");
    }

    #[test]
    fn frame_fills_proportionally() {
        let t = Duration::from_secs(0);
        assert!(frame("x", 0, 100, t).contains(&format!("[{}]", ".".repeat(BAR_WIDTH))));
        assert!(frame("x", 100, 100, t).contains(&format!("[{}]", "#".repeat(BAR_WIDTH))));
        let half = frame("x", 50, 100, t);
        assert!(half.starts_with("x ["));
        assert!(half.contains(&"#".repeat(BAR_WIDTH / 2)));
        assert!(half.contains(" 50% "));
    }

    #[test]
    fn eta_appears_only_mid_run() {
        let t = Duration::from_secs(10);
        // Too early to estimate, and nothing left to estimate at the end.
        assert!(!frame_text(0, 1000, t).contains("eta"));
        assert!(!frame_text(1000, 1000, t).contains("eta"));
        // 25% done in 10s => 30s remaining.
        assert!(frame_text(250, 1000, t).contains("eta 0:30"));
    }

    /// A zero total must not divide by zero, and must still complete.
    #[test]
    fn empty_work_completes() {
        let p = ByteProgress::new("empty", 0);
        p.advance(0);
        p.finish();
        assert_eq!(p.done.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn advances_from_many_threads_reach_the_total() {
        let p = ByteProgress::new("threads", 8 * 1024);
        std::thread::scope(|s| {
            for _ in 0..8 {
                s.spawn(|| {
                    for _ in 0..64 {
                        p.advance(16);
                    }
                });
            }
        });
        assert_eq!(p.done.load(Ordering::Relaxed), 8 * 1024);
        p.finish();
    }
}
