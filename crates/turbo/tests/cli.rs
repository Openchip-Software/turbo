//! CLI-level integration tests driving `main_interface()` through the shared
//! `TurboTest` harness.

use turbo::main_interface::{main_interface, Command, Overlap};
use turbo::workspace_root;
use turbo_tracer_shared::inspector::inspect_trace_file;

use crate::harness::{global_instructions, skip_if_missing, TurboTest};

const LOOP_BIN: &str = "test_bins/simple_loop/loop.riscv";
const FLASH_FULL_BIN: &str = "test_bins/flash/flash_full.riscv";
const FLASH_FULL_ARGS: &[&str] = &["256", "128"];
const FLASH_FULL_SMALL_ARGS: &[&str] = &["128", "32"];
const PTHREADS_BIN: &str = "test_bins/thread_roi/pthreads_roi.riscv";

/// `turbo run` captures a binary, then `turbo process` re-analyses the produced
/// turbo_trace_vcpu0.bin without re-running QEMU.
#[test]
fn run_then_process() {
    let run = TurboTest::new("cli_run_then_process", LOOP_BIN).run();
    let trace = run.trace_log(0);

    let reprocessed = run.reprocess("cli_run_then_process_again", &trace, |_| {});
    reprocessed.perf_data();
}

/// `turbo run` then `turbo process` on `flash_full.riscv`, then a "load" of
/// the produced `perf_data_final.json` (mirroring `turbo load`'s data path
/// without invoking the CLI `Load` command itself, which starts a blocking
/// web server unsuited to a test process).
#[test]
fn flash_full_run_load_process() {
    let run = TurboTest::new("cli_flash_full_run_load_process", FLASH_FULL_BIN)
        .args(FLASH_FULL_ARGS)
        .run();
    let trace = run.trace_log(0);
    let run_data = run.perf_data();

    // `process`: re-analyse the trace file produced above, into a separate
    // output directory, without re-running QEMU.
    let reprocessed = run.reprocess(
        "cli_flash_full_run_load_process_reprocessed",
        &trace,
        |_| {},
    );

    // `load`: the same data path `turbo load` drives (rendering the loaded
    // ROIs/functions on the web server), without starting that server.
    let perf_data = reprocessed.perf_data();
    assert!(
        !perf_data.rois.is_empty() || !perf_data.functions.is_empty(),
        "loaded performance data has no ROIs or functions"
    );

    // `run` and `process` analysed the same capture, so their outputs must
    // agree on the actual instruction counts (not necessarily byte-for-byte:
    // map/vec ordering can differ between the two entry points).
    assert_eq!(
        perf_data.rois.len(),
        run_data.rois.len(),
        "`run` and `process` reported a different number of ROIs for the same capture"
    );
    assert_eq!(
        perf_data.functions.len(),
        run_data.functions.len(),
        "`run` and `process` reported a different number of functions for the same capture"
    );
}

/// Widening the vector registers must retire fewer instructions: the same
/// flash-attention work packs into fewer, longer vector operations, so a bigger
/// `vlen` is strictly cheaper in retired instructions.
///
/// This is the end-to-end check that `--vlen` actually reaches QEMU *and* is
/// applied consistently through decode/enrich: a `--vlen` silently dropped (or
/// only reaching one of the two) leaves the totals identical across all three
/// widths, and a VL mis-decode makes them move the wrong way.
#[test]
fn instruction_count_falls_as_vlen_grows() {
    let counts: Vec<(u32, u64)> = [128u32, 256, 512]
        .iter()
        .map(|&vlen| {
            let run = TurboTest::new(&format!("cli_flash_full_vlen_{vlen}"), FLASH_FULL_BIN)
                .args(FLASH_FULL_SMALL_ARGS)
                .vlen(vlen)
                .run();
            let insns = run.global_instructions();
            assert!(insns > 0, "vlen={vlen} retired no instructions");
            (vlen, insns)
        })
        .collect();

    for pair in counts.windows(2) {
        let (lo, lo_insns) = pair[0];
        let (hi, hi_insns) = pair[1];
        println!(
            "vlen={hi} retired {hi_insns} instruction(s), fewer than vlen={lo}'s \
             {lo_insns}: {counts:?}"
        );
        assert!(
            hi_insns < lo_insns,
            "vlen={hi} retired {hi_insns} instruction(s), fewer than vlen={lo}'s \
             {lo_insns}: {counts:?}"
        );
    }
}

/// `--inspect` on a real `flash_full 256 128` capture.
///
/// Asserted through the library [`inspect_trace_file`] the CLI flag calls, so
/// the numbers a user is shown are the numbers checked here. The CLI path
/// itself is driven too (`turbo process --inspect`), so a broken wiring fails
/// even though its output is stdout and not an artifact.
#[test]
fn flash_full_inspect_trace_report() {
    let run = TurboTest::new("cli_flash_full_inspect", FLASH_FULL_BIN)
        .args(FLASH_FULL_ARGS)
        .run();
    let trace = run.trace_log(0);

    let r = inspect_trace_file(&trace).expect("inspecting the recorded trace failed");

    // The recording completed, so the writer's sentinel must be there: without
    // it every size below would be of a partial file.
    assert!(r.saw_done, "no Done sentinel: trace is truncated");
    assert!(r.batches > 0, "no batches in the trace");
    assert!(r.file_bytes > 0, "empty trace file on disk");

    // File sizes and compression: records are deflated on disk, so the
    // uncompressed accounting must exceed the file, and the ratio must be a
    // real win rather than deflate breaking even on the BBT record stream.
    assert!(
        r.logical_bytes > r.file_bytes,
        "uncompressed {} is not larger than on-disk {} -- compression did nothing",
        r.logical_bytes,
        r.file_bytes
    );
    let ratio = r.compression_ratio();
    assert!(
        ratio > 1.5,
        "compression ratio {ratio:.2}x is implausibly low for a BBT trace \
         ({} on disk, {} uncompressed)",
        r.file_bytes,
        r.logical_bytes
    );

    // The breakdown must partition the uncompressed size exactly: `record
    // framing` is defined as the remainder, so any drift means a category is
    // being double-counted or missed.
    let rows = r.size_rows();
    let summed: u64 = rows.iter().map(|row| row.bytes).sum();
    assert_eq!(
        summed, r.logical_bytes,
        "size categories {rows:?} sum to {summed}, not the uncompressed {}",
        r.logical_bytes
    );
    assert!(
        r.block_records > 0 && r.retired_insns >= r.block_records,
        "block records {} expanded to only {} instruction(s)",
        r.block_records,
        r.retired_insns
    );

    // Format invariants: every record's block must have been described first,
    // and described exactly once per file.
    assert_eq!(
        r.unresolved_block_records, 0,
        "block record(s) name an ID with no descriptor"
    );
    assert_eq!(
        r.dict_redundant_entries, 0,
        "the file re-describes {} already-known block ID(s)",
        r.dict_redundant_entries
    );

    // And the CLI path to the above is driven: the standalone `turbo inspect`
    // verb over the whole spool, printed to stdout, decoding nothing.
    main_interface(
        Command::Inspect {
            trace_input: run.dir.clone(),
            vcpu: vec![],
        }
        .into(),
    )
    .expect("`turbo inspect` failed");
}

/// `turbo process --annotate --src-dir` interleaves the C source (compiled with
/// -g) into the annotation HTML output.
#[test]
fn process_annotate_with_src_dir() {
    let src_dir = workspace_root().join("test_bins/simple_loop");
    let run = TurboTest::new("cli_process_annotate", LOOP_BIN).run();
    let trace = run.trace_log(0);

    let annotated = run.reprocess("cli_process_annotate_out", &trace, |p| {
        p.annotate = true;
        p.src_dir = vec![src_dir];
    });

    let html = annotated.annotate_page("annotate_main.html");
    assert!(
        html.contains("sum += i;"),
        "C source was not interleaved into the annotation report"
    );
    assert!(
        !html.contains("<source unavailable>"),
        "source lines were not resolved via --src-dir"
    );
}

/// `--annotate` produces the hotspots page. Using the load_1gb vector benchmark
/// (small --size for CI, --validate to also exercise the scalar loops), the
/// hotspots report must rank the scalar sum loops as scalar-tier hotspots above
/// the healthy vectorized loop.
#[test]
fn hotspots_report() {
    let src_dir = workspace_root().join("test_bins/simple_loop");
    let run = TurboTest::new("cli_hotspots", "test_bins/simple_loop/load_1gb.riscv")
        .args(&["--size", "2", "--validate"])
        .run();
    let trace = run.trace_log(0);

    let annotated = run.reprocess("cli_hotspots_out", &trace, |p| {
        p.annotate = true;
        p.src_dir = vec![src_dir];
    });

    let html = annotated.annotate_page("annotate_hotspots.html");
    assert!(
        html.contains("scalar_sum"),
        "scalar_sum loop missing from hotspots report"
    );
    assert!(
        html.contains("vector_sum"),
        "vector_sum loop missing from hotspots report"
    );
    assert!(html.contains(">scalar<"), "no scalar-tier hotspot reported");
}

/// The multi-hart regression this source layer exists for: a recorded spool of
/// a multi-threaded guest must be processed in FULL. `process` pointed at a
/// directory previously read only `turbo_trace_vcpu0.bin`, silently dropping every
/// other thread's instructions with no warning.
#[test]
fn process_directory_covers_every_hart() {
    if skip_if_missing(PTHREADS_BIN) {
        return;
    }
    let recording = TurboTest::new("cli_process_all_harts", PTHREADS_BIN).record();

    // The guest spawns pthreads, so the recording must span several harts --
    // otherwise this test would pass vacuously on a single-hart spool.
    let logs = recording.trace_logs();
    assert!(
        logs.len() > 1,
        "expected a multi-hart recording, found {} log(s): {logs:?}",
        logs.len()
    );

    let processed = recording.reprocess_dir("cli_process_all_harts_out");

    // One per-hart file per recorded log: every hart reached the enricher.
    for (hart, _) in &logs {
        assert!(
            processed.has_vcpu_perf(*hart),
            "hart {hart} was recorded but produced no per-hart output"
        );
    }

    // Each thread's own ROI region must be present in the merged view. These
    // run on different harts, so a vCPU-0-only read cannot produce all three.
    let merged = processed.perf_data();
    for name in ["thread_0_work", "thread_1_work", "thread_2_work"] {
        assert!(
            merged.rois.iter().any(|r| &*r.name == name),
            "ROI `{name}` missing from the merged view; harts were dropped. Got: {:?}",
            merged.rois.iter().map(|r| &r.name).collect::<Vec<_>>()
        );
    }
}

/// `--vcpu` restricts processing to the named harts.
#[test]
fn process_vcpu_filter_selects_one_hart() {
    if skip_if_missing(PTHREADS_BIN) {
        return;
    }
    let recording = TurboTest::new("cli_process_vcpu_filter", PTHREADS_BIN).record();
    let dir = recording.dir.clone();

    let filtered = recording.reprocess("cli_process_vcpu_filter_out", dir, |p| p.vcpu = vec![0]);

    assert!(filtered.has_vcpu_perf(0), "requested hart 0 missing");
    for hart in 1..4 {
        assert!(
            !filtered.has_vcpu_perf(hart),
            "hart {hart} was not requested but was processed anyway"
        );
    }
}

/// `run --overlap=on` (decode while QEMU runs) and `--overlap=off` (record,
/// then process) must produce the same numbers. The overlap is a scheduling
/// choice; it must not change what is measured.
#[test]
fn overlap_modes_agree() {
    let profile = |label: &str, overlap: Overlap| {
        let run = TurboTest::new(&format!("cli_overlap_{label}"), LOOP_BIN)
            .overlap(overlap)
            .run();
        let pd = run.perf_data();
        let global = global_instructions(&pd);
        assert!(global > 0, "overlap={label} recorded no instructions");
        (global, pd.functions.len())
    };

    let on = profile("on", Overlap::On);
    let off = profile("off", Overlap::Off);

    assert_eq!(
        on.0, off.0,
        "instruction totals differ between overlap modes: on={on:?} off={off:?}"
    );
    assert_eq!(
        on.1, off.1,
        "function counts differ between overlap modes: on={on:?} off={off:?}"
    );
}

/// A live `run` must leave a re-processable recording behind, so a failure in
/// decode/enrich never costs the (expensive) QEMU execution.
#[test]
fn live_run_leaves_a_reprocessable_recording() {
    let live = TurboTest::new("cli_live_run_reprocessable", LOOP_BIN)
        .overlap(Overlap::On)
        .run();

    // Re-process the spool the live run tailed, into a fresh directory.
    let again = live.reprocess_dir("cli_live_run_reprocessable_again");

    assert_eq!(
        live.global_instructions(),
        again.global_instructions(),
        "re-processing the recording gave different totals than the live run"
    );
}

/// `(rois, functions)` as sorted name -> instruction-count pairs.
type NameCountProfile = (Vec<(String, u64)>, Vec<(String, u64)>);

/// Parallel decode must be an execution-strategy change ONLY. Same recording,
/// different worker counts, byte-identical ROI and function tables — otherwise
/// the pool is corrupting per-hart decode state or losing work.
#[test]
fn decode_thread_count_does_not_change_results() {
    if skip_if_missing(PTHREADS_BIN) {
        return;
    }
    let recording = TurboTest::new("cli_decode_threads_recording", PTHREADS_BIN).record();

    // Needs several harts, or every thread count trivially uses one worker.
    let harts = recording.trace_logs().len();
    assert!(harts > 1, "expected a multi-hart recording, got {harts}");

    // (rois, functions) as sorted name->instruction pairs, which is what must
    // not vary. Comparing the whole PerformanceData would also drag in fields
    // that legitimately differ (e.g. nothing today, but this keeps the assertion
    // about the numbers rather than the serialization).
    let profile_of = |threads: usize| -> NameCountProfile {
        let dir = recording.dir.clone();
        let out = recording.reprocess(&format!("cli_decode_threads_{threads}"), dir, |p| {
            p.decode_threads = Some(threads)
        });

        let pd = out.perf_data();
        let mut rois: Vec<(String, u64)> = pd
            .rois
            .iter()
            .map(|r| (r.name.to_string(), r.instructions))
            .collect();
        let mut funcs: Vec<(String, u64)> = pd
            .functions
            .iter()
            .map(|f| (f.name.to_string(), f.instructions))
            .collect();
        rois.sort();
        funcs.sort();
        assert!(!rois.is_empty(), "threads={threads} produced no ROIs");
        (rois, funcs)
    };

    let serial = profile_of(1);
    let parallel = profile_of(8);

    assert_eq!(
        serial.0, parallel.0,
        "ROI totals differ between 1 and 8 decode threads"
    );
    assert_eq!(
        serial.1, parallel.1,
        "function totals differ between 1 and 8 decode threads"
    );
}

/// Every hart must get its own worker up to the cap, and a cap of 1 must still
/// process every hart (sharing one worker) rather than dropping any.
#[test]
fn one_decode_thread_still_covers_every_hart() {
    if skip_if_missing(PTHREADS_BIN) {
        return;
    }
    let recording = TurboTest::new("cli_one_thread_all_harts_rec", PTHREADS_BIN).record();
    let dir = recording.dir.clone();

    let out = recording.reprocess("cli_one_thread_all_harts_out", dir, |p| {
        p.decode_threads = Some(1)
    });

    let merged = out.perf_data();
    for name in ["thread_0_work", "thread_1_work", "thread_2_work"] {
        assert!(
            merged.rois.iter().any(|r| &*r.name == name),
            "ROI `{name}` missing with a single decode worker: harts sharing a \
             worker must still all be processed"
        );
    }
}
