//! Gradle-style progress on stderr. On a terminal, a live block is pinned
//! to the bottom of stderr: an overall bar (batch runs only), one line per
//! busy worker naming its file and current stage, and a counters line.
//! Failures scroll above the block as they happen. Off a terminal (CI,
//! `2>file`, the integration tests) it degrades to plain one-line-per-event
//! output; `-q` turns it off entirely. Stdout is never touched, so JSON
//! Lines / table output is identical with or without progress.
//!
//! Hand-rolled with ANSI cursor codes rather than `indicatif`: its current
//! releases (and `console`'s) would need the same edition2021 pinning as
//! `clap`/`indexmap` (see this crate's Cargo.toml), and the layout here is
//! simple enough not to need it.
//!
//! [`pipeline`](crate::pipeline) only sees the [`StageReporter`] trait, and
//! reports compute stages from *inside* each `Resolver::resolve` closure --
//! that closure only runs on a cache miss, so a warm entry never shows up as
//! work, and a task whose stages were all cache hits is counted as cached.

use std::io::{IsTerminal, Write};
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

const TICK: Duration = Duration::from_millis(100);
const BAR_CELLS: usize = 20;
const MAX_NAME_WIDTH: usize = 40;

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Mode {
    /// Live ANSI block on a terminal.
    Rich,
    /// One line per start/failure, plus a summary (batch only).
    Plain,
    Off,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Stage {
    Hashing,
    /// Looking a key up in the cache; whether it then computes is only
    /// known once the next (compute) stage is or isn't reported.
    CheckingCache,
    Decoding,
    Resampling,
    TrackingBeats,
    Chords,
    Structure,
    Key,
}

impl Stage {
    pub fn label(self) -> &'static str {
        match self {
            Stage::Hashing => "hashing",
            Stage::CheckingCache => "checking cache",
            Stage::Decoding => "decoding",
            Stage::Resampling => "resampling",
            Stage::TrackingBeats => "tracking beats",
            Stage::Chords => "estimating chords",
            Stage::Structure => "segmenting",
            Stage::Key => "estimating key",
        }
    }

    /// Whether reporting this stage means real work ran (vs. bookkeeping
    /// that happens on every run, warm cache or not).
    fn is_compute(self) -> bool {
        !matches!(self, Stage::Hashing | Stage::CheckingCache)
    }
}

pub trait StageReporter {
    fn stage(&self, stage: Stage);
}

/// Reporter for callers with no progress display.
pub struct NoProgress;

impl StageReporter for NoProgress {
    fn stage(&self, _stage: Stage) {}
}

pub static NONE: NoProgress = NoProgress;

struct Slot {
    name: String,
    stage: Option<Stage>,
}

#[derive(Default)]
struct State {
    total: usize,
    started: usize,
    done: usize,
    cached: usize,
    failed: usize,
    slots: Vec<Option<Slot>>,
    /// Lines of the block currently on screen (Rich mode only).
    drawn: usize,
}

struct Inner {
    mode: Mode,
    color: bool,
    start: Instant,
    stop: AtomicBool,
    state: Mutex<State>,
}

pub struct Progress {
    inner: Arc<Inner>,
    ticker: Option<JoinHandle<()>>,
}

impl Progress {
    pub fn detect(quiet: bool) -> Mode {
        if quiet {
            return Mode::Off;
        }
        let dumb = std::env::var("TERM").map(|t| t == "dumb").unwrap_or(false);
        if !std::io::stderr().is_terminal() || dumb || !platform::enable_vt() {
            return Mode::Plain;
        }
        Mode::Rich
    }

    pub fn start(total: usize, mode: Mode) -> Progress {
        let inner = Arc::new(Inner {
            mode,
            color: std::env::var_os("NO_COLOR").map_or(true, |v| v.is_empty()),
            start: Instant::now(),
            stop: AtomicBool::new(false),
            state: Mutex::new(State {
                total,
                ..State::default()
            }),
        });
        // The first draw waits one tick, so a warm-cache run that finishes
        // inside it never flashes a block at all.
        let ticker = (mode == Mode::Rich).then(|| {
            let inner = Arc::clone(&inner);
            std::thread::spawn(move || loop {
                std::thread::park_timeout(TICK);
                if inner.stop.load(Ordering::Acquire) {
                    break;
                }
                inner.redraw();
            })
        });
        Progress { inner, ticker }
    }

    /// Claims a worker line for `path` until the returned task is finished
    /// or dropped.
    pub fn task(&self, path: &Path) -> Task<'_> {
        let mut st = self.inner.lock();
        st.started += 1;
        let index = st.started;
        let name = path
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_else(|| path.display().to_string());
        let slot = Slot { name, stage: None };
        let slot_idx = match st.slots.iter().position(Option::is_none) {
            Some(i) => {
                st.slots[i] = Some(slot);
                i
            }
            None => {
                st.slots.push(Some(slot));
                st.slots.len() - 1
            }
        };
        if self.inner.mode == Mode::Plain && st.total > 1 {
            eprintln!("audan: [{index}/{}] {}", st.total, path.display());
        }
        Task {
            inner: &self.inner,
            slot: slot_idx,
            index,
            path: path.display().to_string(),
            computed: AtomicBool::new(false),
            finished: false,
        }
    }

    /// Clears the block and, for a batch, prints a one-line summary. Also
    /// what `Drop` does, so a `?` early return never leaves a stale block.
    pub fn finish(self) {}
}

impl Drop for Progress {
    fn drop(&mut self) {
        if let Some(ticker) = self.ticker.take() {
            self.inner.stop.store(true, Ordering::Release);
            ticker.thread().unpark();
            let _ = ticker.join();
        }
        let mut st = self.inner.lock();
        if self.inner.mode == Mode::Off {
            return;
        }
        let mut buf = String::new();
        clear_block(&mut buf, &mut st);
        if st.total > 1 {
            buf.push_str(&summary(&st, self.inner.start.elapsed()));
            buf.push('\n');
        }
        write_stderr(&buf);
    }
}

pub struct Task<'a> {
    inner: &'a Inner,
    slot: usize,
    index: usize,
    path: String,
    computed: AtomicBool,
    finished: bool,
}

impl Task<'_> {
    /// Frees the worker line and counts the outcome: failed if `error` is
    /// set, cached if no compute stage was ever reported, done otherwise.
    /// Batch failures are printed above the block as they happen.
    pub fn finish(mut self, error: Option<&anyhow::Error>) {
        self.finished = true;
        let mut st = self.inner.lock();
        st.slots[self.slot] = None;
        st.done += 1;
        match error {
            Some(e) => {
                st.failed += 1;
                if st.total > 1 {
                    let msg = match self.inner.mode {
                        Mode::Rich if self.inner.color => {
                            format!("\x1b[31mFAILED\x1b[0m {}: {e:#}", self.path)
                        }
                        Mode::Rich => format!("FAILED {}: {e:#}", self.path),
                        _ => format!(
                            "audan: [{}/{}] {} FAILED: {e:#}",
                            self.index, st.total, self.path
                        ),
                    };
                    self.inner.println(&mut st, &msg);
                }
            }
            None if !self.computed.load(Ordering::Relaxed) => st.cached += 1,
            None => {}
        }
    }
}

impl StageReporter for Task<'_> {
    fn stage(&self, stage: Stage) {
        if stage.is_compute() {
            self.computed.store(true, Ordering::Relaxed);
        }
        if let Some(slot) = self.inner.lock().slots[self.slot].as_mut() {
            slot.stage = Some(stage);
        }
    }
}

impl Drop for Task<'_> {
    fn drop(&mut self) {
        if !self.finished {
            self.inner.lock().slots[self.slot] = None;
        }
    }
}

impl Inner {
    fn lock(&self) -> std::sync::MutexGuard<'_, State> {
        // A panicking worker must not take the display down with it.
        self.state.lock().unwrap_or_else(|e| e.into_inner())
    }

    fn redraw(&self) {
        let mut st = self.lock();
        let mut buf = String::new();
        clear_block(&mut buf, &mut st);
        self.draw_block(&mut buf, &mut st);
        write_stderr(&buf);
    }

    /// Prints `msg` as a permanent line above the block.
    fn println(&self, st: &mut State, msg: &str) {
        if self.mode != Mode::Rich {
            eprintln!("{msg}");
            return;
        }
        let mut buf = String::new();
        let had_block = st.drawn > 0;
        clear_block(&mut buf, st);
        buf.push_str(msg);
        buf.push('\n');
        if had_block {
            self.draw_block(&mut buf, st);
        }
        write_stderr(&buf);
    }

    fn draw_block(&self, buf: &mut String, st: &mut State) {
        let lines = render_block(st, platform::width(), self.start.elapsed());
        for (i, line) in lines.iter().enumerate() {
            if i == 0 && st.total > 1 && self.color {
                buf.push_str(&colorize_bar(line));
            } else {
                buf.push_str(line);
            }
            buf.push('\n');
        }
        st.drawn = lines.len();
    }
}

/// Moves the cursor back to the top of the block and clears to the end of
/// the screen. Every block line ends in `\n`, so the cursor always sits at
/// column 0 just below it.
fn clear_block(buf: &mut String, st: &mut State) {
    if st.drawn > 0 {
        buf.push_str(&format!("\x1b[{}F\x1b[0J", st.drawn));
        st.drawn = 0;
    }
}

fn write_stderr(buf: &str) {
    if buf.is_empty() {
        return;
    }
    let mut err = std::io::stderr().lock();
    let _ = err.write_all(buf.as_bytes());
    let _ = err.flush();
}

/// The block as plain text: bar (batch only), one line per busy worker,
/// counters (batch only). Each line is cut to `width - 1` characters -- a
/// line that wrapped would throw off the cursor-up count in `clear_block`,
/// and some consoles wrap a line that exactly fills the last column.
fn render_block(st: &State, width: usize, elapsed: Duration) -> Vec<String> {
    let batch = st.total > 1;
    let mut lines = Vec::new();

    if batch {
        let filled = (st.done * BAR_CELLS).checked_div(st.total).unwrap_or(0);
        let pct = (st.done * 100).checked_div(st.total).unwrap_or(0);
        lines.push(format!(
            "<{}{}> {pct}% EXECUTING [{}]",
            "=".repeat(filled),
            "-".repeat(BAR_CELLS - filled),
            fmt_elapsed(elapsed)
        ));
    }

    let active: Vec<&Slot> = st.slots.iter().flatten().collect();
    let name_width = active
        .iter()
        .map(|s| s.name.chars().count())
        .max()
        .unwrap_or(0)
        .min(MAX_NAME_WIDTH);
    for slot in active {
        let stage = slot.stage.map_or("starting", Stage::label);
        let mut line = format!("> {}  {stage}", fit(&slot.name, name_width));
        if !batch {
            line.push_str(&format!(" [{}]", fmt_elapsed(elapsed)));
        }
        lines.push(line);
    }

    if batch {
        let mut counters = format!("  {}/{} files", st.done, st.total);
        if st.cached > 0 {
            counters.push_str(&format!(" · {} cached", st.cached));
        }
        if st.failed > 0 {
            counters.push_str(&format!(" · {} failed", st.failed));
        }
        lines.push(counters);
    }

    let max = width.saturating_sub(1).max(10);
    lines.into_iter().map(|l| truncate(&l, max)).collect()
}

fn summary(st: &State, elapsed: Duration) -> String {
    let files = if st.done == st.total {
        format!("{} files", st.total)
    } else {
        format!("{} of {} files", st.done, st.total)
    };
    let mut extra = Vec::new();
    if st.cached > 0 {
        extra.push(format!("{} cached", st.cached));
    }
    if st.failed > 0 {
        extra.push(format!("{} failed", st.failed));
    }
    let extra = if extra.is_empty() {
        String::new()
    } else {
        format!(" ({})", extra.join(", "))
    };
    format!("audan: {files} in {}{extra}", fmt_elapsed(elapsed))
}

/// Gradle's elapsed-time style: `3s`, `1m 12s`, `1h 2m 3s`.
fn fmt_elapsed(d: Duration) -> String {
    let s = d.as_secs();
    match (s / 3600, (s % 3600) / 60, s % 60) {
        (0, 0, s) => format!("{s}s"),
        (0, m, s) => format!("{m}m {s}s"),
        (h, m, s) => format!("{h}h {m}m {s}s"),
    }
}

/// Pads or ellipsizes `s` to exactly `width` characters.
fn fit(s: &str, width: usize) -> String {
    let n = s.chars().count();
    if n <= width {
        format!("{s}{}", " ".repeat(width - n))
    } else {
        let mut out: String = s.chars().take(width.saturating_sub(1)).collect();
        out.push('…');
        out
    }
}

fn truncate(s: &str, max: usize) -> String {
    s.chars().take(max).collect()
}

/// Greens the `=` run of a bar line (`<====--->`), after truncation so the
/// escape codes never count towards the line width.
fn colorize_bar(line: &str) -> String {
    let Some(rest) = line.strip_prefix('<') else {
        return line.to_string();
    };
    let filled = rest.len() - rest.trim_start_matches('=').len();
    if filled == 0 {
        return line.to_string();
    }
    format!("<\x1b[32m{}\x1b[0m{}", &rest[..filled], &rest[filled..])
}

#[cfg(windows)]
mod platform {
    use windows_sys::Win32::Foundation::INVALID_HANDLE_VALUE;
    use windows_sys::Win32::System::Console::{
        GetConsoleMode, GetConsoleScreenBufferInfo, GetStdHandle, SetConsoleMode,
        CONSOLE_SCREEN_BUFFER_INFO, ENABLE_VIRTUAL_TERMINAL_PROCESSING, STD_ERROR_HANDLE,
    };

    /// Windows Terminal has VT processing on already; classic conhost needs
    /// it switched on per process.
    pub fn enable_vt() -> bool {
        // SAFETY: plain Win32 console calls on the process's own stderr
        // handle; `mode` is a valid out-pointer for the call's duration.
        unsafe {
            let h = GetStdHandle(STD_ERROR_HANDLE);
            if h.is_null() || h == INVALID_HANDLE_VALUE {
                return false;
            }
            let mut mode = 0;
            if GetConsoleMode(h, &mut mode) == 0 {
                return false;
            }
            mode & ENABLE_VIRTUAL_TERMINAL_PROCESSING != 0
                || SetConsoleMode(h, mode | ENABLE_VIRTUAL_TERMINAL_PROCESSING) != 0
        }
    }

    pub fn width() -> usize {
        // SAFETY: as above; `info` is a plain-old-data out-struct.
        unsafe {
            let h = GetStdHandle(STD_ERROR_HANDLE);
            let mut info: CONSOLE_SCREEN_BUFFER_INFO = std::mem::zeroed();
            if GetConsoleScreenBufferInfo(h, &mut info) != 0 {
                let w = info.srWindow.Right - info.srWindow.Left + 1;
                if w > 0 {
                    return w as usize;
                }
            }
        }
        super::columns_env()
    }
}

#[cfg(unix)]
mod platform {
    pub fn enable_vt() -> bool {
        true
    }

    pub fn width() -> usize {
        // SAFETY: TIOCGWINSZ fills a `winsize` out-struct for the given fd.
        unsafe {
            let mut ws: libc::winsize = std::mem::zeroed();
            if libc::ioctl(libc::STDERR_FILENO, libc::TIOCGWINSZ, &mut ws) == 0 && ws.ws_col > 0 {
                return ws.ws_col as usize;
            }
        }
        super::columns_env()
    }
}

#[cfg(not(any(windows, unix)))]
mod platform {
    pub fn enable_vt() -> bool {
        false
    }

    pub fn width() -> usize {
        super::columns_env()
    }
}

fn columns_env() -> usize {
    std::env::var("COLUMNS")
        .ok()
        .and_then(|c| c.parse().ok())
        .filter(|&c| c > 0)
        .unwrap_or(80)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn slot(name: &str, stage: Option<Stage>) -> Option<Slot> {
        Some(Slot {
            name: name.into(),
            stage,
        })
    }

    #[test]
    fn formats_elapsed_like_gradle() {
        assert_eq!(fmt_elapsed(Duration::from_millis(400)), "0s");
        assert_eq!(fmt_elapsed(Duration::from_secs(72)), "1m 12s");
        assert_eq!(fmt_elapsed(Duration::from_secs(3723)), "1h 2m 3s");
    }

    #[test]
    fn batch_block_has_bar_workers_and_counters() {
        let st = State {
            total: 4,
            done: 2,
            cached: 1,
            failed: 1,
            slots: vec![
                slot("one.mp3", Some(Stage::Decoding)),
                None,
                slot("battery.flac", None),
            ],
            ..State::default()
        };
        let lines = render_block(&st, 120, Duration::from_secs(72));
        assert_eq!(
            lines,
            vec![
                "<==========----------> 50% EXECUTING [1m 12s]",
                "> one.mp3       decoding",
                "> battery.flac  starting",
                "  2/4 files · 1 cached · 1 failed",
            ]
        );
    }

    #[test]
    fn single_file_block_is_just_the_worker_line() {
        let st = State {
            total: 1,
            slots: vec![slot("song.flac", Some(Stage::TrackingBeats))],
            ..State::default()
        };
        let lines = render_block(&st, 120, Duration::from_secs(3));
        assert_eq!(lines, vec!["> song.flac  tracking beats [3s]"]);
    }

    #[test]
    fn lines_never_reach_the_last_column() {
        let st = State {
            total: 3,
            slots: vec![slot(&"x".repeat(100), Some(Stage::Resampling))],
            ..State::default()
        };
        let lines = render_block(&st, 30, Duration::ZERO);
        assert!(lines.iter().all(|l| l.chars().count() <= 29), "{lines:?}");
        // Long names are ellipsized to MAX_NAME_WIDTH before the line cut.
        let st = State {
            total: 1,
            slots: vec![slot(&"x".repeat(100), Some(Stage::Resampling))],
            ..State::default()
        };
        let line = &render_block(&st, 200, Duration::ZERO)[0];
        assert!(
            line.starts_with(&format!("> {}…  resampling", "x".repeat(39))),
            "{line}"
        );
    }

    #[test]
    fn colorizes_only_the_filled_bar() {
        assert_eq!(colorize_bar("<==--> 50%"), "<\x1b[32m==\x1b[0m--> 50%");
        assert_eq!(colorize_bar("<----> 0%"), "<----> 0%");
    }

    #[test]
    fn tasks_reuse_slots_and_count_outcomes() {
        let progress = Progress::start(3, Mode::Off);
        let a = progress.task(Path::new("a.wav"));
        let b = progress.task(Path::new("b.wav"));
        b.stage(Stage::CheckingCache);
        b.finish(None); // only bookkeeping stages ran: a cache hit
        let c = progress.task(Path::new("c.wav"));
        assert_eq!(c.slot, 1, "c should reuse b's freed slot");
        c.stage(Stage::Decoding);
        c.finish(None);
        a.finish(Some(&anyhow::anyhow!("boom")));

        let st = progress.inner.lock();
        assert_eq!((st.done, st.cached, st.failed), (3, 1, 1));
        assert!(st.slots.iter().all(Option::is_none));
        assert_eq!(
            summary(&st, Duration::from_secs(5)),
            "audan: 3 files in 5s (1 cached, 1 failed)"
        );
    }
}
