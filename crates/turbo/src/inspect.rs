//! `turbo --inspect`: report what a recorded trace file is made of.
//!
//! All of the accounting lives in [`turbo_tracer_shared::bbt_inspector`]; this
//! module is presentation only. Keeping the split means the numbers a user
//! reads and the numbers a test asserts on come from one implementation.

use std::cell::Cell;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;

use turbo_tracer_shared::inspector::{inspect_trace_file_with_progress, TraceReport};

use crate::progress::ByteProgress;
use crate::workflow::TraceInput;

// The byte formatter lives with the progress bar, which needs the same one;
// re-exported here because every report line below is written in terms of it.
pub use crate::progress::human;

/// Files inspected concurrently: the platform's parallelism minus one, so the
/// machine stays usable while a big spool is walked. Never below one, and never
/// above the file count (see [`inspect_in_parallel`]). Falls back to one thread
/// when the platform will not report a count.
fn max_inspect_threads() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get().saturating_sub(1).max(1))
        .unwrap_or(1)
}

/// Inspect every record log a [`TraceInput`] refers to, one thread per file up
/// to [`max_inspect_threads`], and report what they contain.
///
/// A directory covers every hart in the spool; a named log covers just that
/// hart. Unreadable or malformed files are reported and counted, not fatal, so
/// one bad hart does not hide the others.
///
/// The output is layered by log level, because the per-hart detail is rarely
/// what a user wants: the combined whole-run report goes to `info`, one summary
/// line per hart to `debug`, and each hart's full report to `trace`.
pub fn report_trace_input(input: &TraceInput) -> anyhow::Result<()> {
    let paths = trace_log_paths(input);
    if paths.is_empty() {
        anyhow::bail!("inspect: no turbo_trace_vcpu<N>.bin logs found to inspect");
    }

    let results = inspect_in_parallel(&paths);

    let mut reports = Vec::new();
    let mut failed = 0;
    for (path, result) in paths.iter().zip(results) {
        match result {
            Ok(report) => {
                log::debug!("inspect: {}", summary_line(path, &report));
                for line in report_lines(path, &report) {
                    log::trace!("{line}");
                }
                reports.push(report);
            }
            Err(e) => {
                log::error!("inspect: {e}");
                failed += 1;
            }
        }
    }
    if failed == paths.len() {
        anyhow::bail!("inspect: every trace log failed to parse ({failed})");
    }

    let global = TraceReport::combined(&reports);
    for line in report_lines(
        Path::new(&format!("all {} vCPU trace(s) combined", reports.len())),
        &global,
    ) {
        log::info!("{line}");
    }
    Ok(())
}

/// Inspect `paths` with at most [`max_inspect_threads`] workers, returning one
/// result per path in the same order. Workers claim paths from a shared index
/// so a single huge hart file does not idle the others behind a static split.
///
/// Walking a multi-hundred-MiB spool takes tens of seconds, so the workers
/// share one [`ByteProgress`] over the on-disk size of every file. Per-file
/// announcements stay at `debug`: four threads narrating themselves is exactly
/// what the single bar replaces.
fn inspect_in_parallel(paths: &[PathBuf]) -> Vec<Result<TraceReport, String>> {
    let next = AtomicUsize::new(0);
    let slots: Vec<Mutex<Option<Result<TraceReport, String>>>> =
        paths.iter().map(|_| Mutex::new(None)).collect();
    let workers = paths.len().min(max_inspect_threads());

    // One metadata call per file, kept as the bar's per-file budget: the total
    // and the per-file credit below must come from the same numbers or the bar
    // cannot land exactly on 100%. A file whose size will not stat is worth
    // zero work rather than a guess -- it is about to fail to be read anyway.
    let sizes: Vec<u64> = paths
        .iter()
        .map(|p| p.metadata().map(|m| m.len()).unwrap_or(0))
        .collect();
    let total: u64 = sizes.iter().sum();
    log::info!(
        "inspect: {} file(s), {} over {workers} thread(s)",
        paths.len(),
        human(total),
    );
    let bar = ByteProgress::new("inspect", total);

    std::thread::scope(|s| {
        for _ in 0..workers {
            s.spawn(|| loop {
                let i = next.fetch_add(1, Ordering::Relaxed);
                let Some(path) = paths.get(i) else { break };
                log::debug!(
                    "inspecting {}/{}: {} ({})",
                    i + 1,
                    paths.len(),
                    path.file_name()
                        .unwrap_or(path.as_os_str())
                        .to_string_lossy(),
                    human(sizes[i]),
                );
                // Thread-local tally of what this file reported, so the bytes
                // the walk never reads -- the tail after a `Done` sentinel, or
                // everything after an error -- can be credited once the file is
                // finished. Without that the bar would stop short of its total.
                let credited = Cell::new(0u64);
                let report = inspect_trace_file_with_progress(path, &|n| {
                    credited.set(credited.get() + n);
                    bar.advance(n);
                });
                bar.advance(sizes[i].saturating_sub(credited.get()));
                *slots[i].lock().unwrap() = Some(report);
            });
        }
    });
    bar.finish();

    slots
        .into_iter()
        // Every slot was filled: the index handed each path to exactly one
        // worker, and the scope joined them all.
        .map(|slot| slot.into_inner().unwrap().unwrap())
        .collect()
}

/// One hart on one line: the numbers worth scanning across harts to spot an
/// odd one out (size, compression, work done, completeness).
fn summary_line(path: &Path, r: &TraceReport) -> String {
    format!(
        "{}: {} on disk ({:.1}x, {} batches), {} insn(s) in {} block exec(s), \
         {} distinct block(s), {:.3} disk bytes/insn{}",
        path.file_name()
            .unwrap_or(path.as_os_str())
            .to_string_lossy(),
        human(r.file_bytes),
        r.compression_ratio(),
        r.batches,
        r.retired_insns,
        r.block_records,
        r.distinct_blocks(),
        r.disk_bytes_per_insn(),
        if r.saw_done {
            ""
        } else {
            " -- TRUNCATED (no Done sentinel)"
        },
    )
}

/// The record logs to inspect, sorted so multi-hart output is hart-ordered.
///
/// The directory scan is [`crate::workflow::record_logs_in_dir`], the same one
/// `process` validates its input with -- including the `--vcpu` filter, so the
/// report describes the run the user actually asked for.
fn trace_log_paths(input: &TraceInput) -> Vec<PathBuf> {
    match input {
        TraceInput::RecordedLog { path, .. } => vec![path.clone()],
        TraceInput::RecordedDir { dir, harts } => {
            crate::workflow::record_logs_in_dir(dir, harts.as_deref())
                .into_iter()
                .map(|(_, path)| path)
                .collect()
        }
    }
}

/// One file's report as text lines, in the layout the standalone `bbt-inspect`
/// binary used. Built as lines rather than printed directly so the same report
/// can go to stdout or to the log without a second copy of the formatting.
pub fn report_lines(path: &Path, r: &TraceReport) -> Vec<String> {
    let mut out = Vec::new();
    out.push(format!("{}", path.display()));
    out.push(format!(
        "  on disk         {:>14}  ({} batches, Done sentinel: {})",
        human(r.file_bytes),
        r.batches,
        if r.saw_done {
            "yes"
        } else {
            "NO -- truncated or still being written"
        },
    ));
    out.push(format!(
        "  uncompressed    {:>14}  ({:.1}x compression)",
        human(r.logical_bytes),
        r.compression_ratio(),
    ));
    // Which run wrote this. Reported because the commonest confusing state of
    // an output directory is a mixture: one hart's log left over from an
    // earlier run beside the current one's. `process` refuses that outright;
    // `inspect` is where a user goes to see it.
    if let Some(run_id) = r.run_id {
        out.push(format!("  run             {run_id:>14}"));
    }

    out.push("  size breakdown (uncompressed):".to_string());
    for row in r.size_rows() {
        out.push(format!(
            "    {:<24} {:>14} ({:>5.1}%)  {:>12} item(s)",
            row.label,
            human(row.bytes),
            row.percent,
            row.count,
        ));
    }

    if r.roi_markers != 0 {
        let (distinct, shipped) = r.roi_name_repeats();
        out.push("  roi markers:".to_string());
        out.push(format!(
            "    {} marker(s), {} avg bytes each; {} of that is name strings",
            r.roi_markers,
            r.roi_bytes / r.roi_markers.max(1),
            human(r.roi_name_bytes),
        ));
        if shipped > distinct as u64 {
            out.push(format!(
                "    DUPLICATE: {} name string(s) shipped for only {} distinct name(s) -- \
                 the same text is re-sent {:.0}x on average",
                shipped,
                distinct,
                shipped as f64 / distinct.max(1) as f64,
            ));
        }
    }

    out.push("  dictionary:".to_string());
    out.push(format!(
        "    {} distinct block(s), {} instruction(s) described, {:.1} insn/block",
        r.distinct_blocks(),
        r.dict_insns,
        r.insns_per_block(),
    ));
    if r.dict_redundant_entries != 0 {
        out.push(format!(
            "    DUPLICATE: {} descriptor(s) re-describe an ID already in this file ({})",
            r.dict_redundant_entries,
            human(r.dict_redundant_bytes),
        ));
    } else {
        out.push("    no duplicate descriptors: every block described exactly once".to_string());
    }
    if r.unresolved_block_records != 0 {
        out.push(format!(
            "    WARNING: {} block record(s) name an ID with no descriptor -- the \
             dictionary-before-record invariant is broken",
            r.unresolved_block_records,
        ));
    }

    out.push("  expansion:".to_string());
    out.push(format!(
        "    {} block execution(s) -> {} guest instruction(s) ({:.2} insn/record)",
        r.block_records,
        r.retired_insns,
        r.insns_per_record(),
    ));
    out.push(format!(
        "    {:.3} bytes/insn uncompressed, {:.3} bytes/insn on disk (compressed)",
        r.bytes_per_insn(),
        r.disk_bytes_per_insn(),
    ));

    let top = r.hottest_blocks(5);
    if !top.is_empty() {
        out.push("  hottest blocks (by executions):".to_string());
        for b in top {
            out.push(format!(
                "    id {:<8} {:>12} exec(s) x {:>3} insn = {:>14} insn(s) ({:>5.1}% of stream)",
                b.id,
                b.executions,
                b.insns,
                b.total_insns(),
                r.percent_of_stream(b.total_insns()),
            ));
        }
    }

    if r.truncated_tail != 0 {
        out.push(format!(
            "  note: {} trailing byte(s) after the last complete record",
            r.truncated_tail
        ));
    }
    out
}
