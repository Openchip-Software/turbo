//! End-user "wall clock" experience test: runs `turbo run` on a real binary
//! with every combination of the cheap, always-available `Run` flags from
//! `main_interface.rs` (lines 55-96) and reports the runtime of each
//! combination relative to the default (no flags) run.
//!
//! This is NOT a micro-benchmark (see `benches/processing_benchmarks.rs` for
//! that). It measures the thing a user actually experiences: how long
//! `turbo run <binary>` takes to return, for each flag they might reach for.
//!
//! Run with:
//!   cargo nextest run -p turbo --test integration user_performance::flag_combo_runtime_matrix --no-capture -r

use std::fs;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use serde::Serialize;
use turbo::workspace_root;

use crate::harness::TurboTest;

const BIN: &str = "test_bins/flash/flash_full.riscv";
const BIN_ARGS: &[&str] = &["256", "128"];

/// One point in the flag combination matrix.
struct FlagCombo {
    name: &'static str,
    annotate: bool,
    perfetto: bool,
}

const COMBOS: &[FlagCombo] = &[
    FlagCombo {
        name: "default",
        annotate: false,
        perfetto: false,
    },
    FlagCombo {
        name: "annotate",
        annotate: true,
        perfetto: false,
    },
    FlagCombo {
        name: "perfetto",
        annotate: false,
        perfetto: true,
    },
    FlagCombo {
        name: "annotate+perfetto",
        annotate: true,
        perfetto: true,
    },
];

fn run_combo(combo: &FlagCombo) -> Duration {
    let turbo = TurboTest::new(&format!("user_performance_matrix/{}", combo.name), BIN)
        .args(BIN_ARGS)
        .process(|p| {
            p.annotate = combo.annotate;
            p.perfetto = combo.perfetto;
        });

    let start = Instant::now();
    turbo.run();
    start.elapsed()
}

/// A single flag combo's timing, in a form suitable for long-horizon JSON
/// tracking (one record per combo per test run).
#[derive(Serialize)]
struct ComboTiming<'a> {
    flags: &'a str,
    seconds: f64,
    vs_default: f64,
}

/// The full matrix result for one invocation of the test, written to disk as
/// JSON so runtime trends can be tracked across commits/CI runs over time.
#[derive(Serialize)]
struct MatrixReport<'a> {
    timestamp_unix_secs: u64,
    binary: &'a str,
    binary_args: &'a [&'a str],
    combos: Vec<ComboTiming<'a>>,
}

/// Runs `turbo run test_bins/flash/flash_full.riscv 256 128` under every
/// combination of `--annotate` / `--perfetto`,
/// timing each end to end, and reports the runtime of each combination
/// relative to the default (all flags off) so we can see which flags cost the
/// user wall-clock time. The report is printed as a table and also written to
/// disk as JSON for long-horizon performance tracking.
#[test]
fn flag_combo_runtime_matrix() {
    // Surface the pipeline's per-run `[phase-timing]` breakdown (startup / run /
    // enrich-tail, logged by run_perf_workflow_opt) so `--no-capture` shows the
    // biggest wall-clock consumer per combo, not just the end-to-end total.
    let _ =
        env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("turbo=info"))
            .is_test(false)
            .try_init();

    let mut results: Vec<(&str, Duration)> = Vec::with_capacity(COMBOS.len());

    for combo in COMBOS {
        let elapsed = run_combo(combo);
        println!(
            "[user_performance] {:<45} {:>8.3}s",
            combo.name,
            elapsed.as_secs_f64()
        );
        results.push((combo.name, elapsed));
    }

    let default_secs = results
        .iter()
        .find(|(name, _)| *name == "default")
        .expect("default combo missing")
        .1
        .as_secs_f64();

    println!();
    println!(
        "=== turbo run {} {} -- flag combo runtime report ===",
        BIN,
        BIN_ARGS.join(" ")
    );
    println!("{:<45} {:>10} {:>10}", "flags", "time (s)", "vs default");
    let mut combo_timings = Vec::with_capacity(results.len());
    for (name, elapsed) in &results {
        let secs = elapsed.as_secs_f64();
        let ratio = secs / default_secs;
        println!("{:<45} {:>10.3} {:>9.2}x", name, secs, ratio);
        combo_timings.push(ComboTiming {
            flags: name,
            seconds: secs,
            vs_default: ratio,
        });
    }

    assert!(default_secs > 0.0, "default run reported zero elapsed time");

    let _report = MatrixReport {
        timestamp_unix_secs: SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system time before UNIX epoch")
            .as_secs(),
        binary: BIN,
        binary_args: BIN_ARGS,
        combos: combo_timings,
    };

    // Mute per-test run reports for now
    // write_json_report(&report);
}

/// Writes the matrix report to disk as JSON: a `latest.json` snapshot (always
/// overwritten) plus one line appended to `history.jsonl` per test run, so
/// runtime trends can be tracked over the long horizon across many runs.
#[allow(dead_code)]
fn write_json_report(report: &MatrixReport) {
    let dir = workspace_root().join("turbo_data");
    fs::create_dir_all(&dir).expect("failed to create JSON report directory");

    let mut line = serde_json::to_string(report).expect("failed to serialize matrix report");
    line.push('\n');
    use std::io::Write as _;

    let history_file = dir.join("history.jsonl");
    let mut history = fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&history_file)
        .expect("failed to open history.jsonl");
    history
        .write_all(line.as_bytes())
        .expect("failed to append to history.jsonl");

    println!(
        "[user_performance] history JSONL updated in {}",
        history_file.display()
    );
}
