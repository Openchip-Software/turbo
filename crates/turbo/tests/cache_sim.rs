//! End-to-end tests for the cache model wiring.
//!
//! Each records an `asm_validation` kernel, replays its captured memory
//! footprint through the cache model, and asserts numbers computed from the
//! `.S` source rather than from a previous run.
//!
//! Two kinds of assertion, and the second is the one these tests exist for:
//!
//! * **hit/miss counts** -- what the model concluded.
//! * **call counts** -- how many times the model was *called*. A vector
//!   instruction must cost exactly one call whatever its addressing mode, and
//!   nothing else in the pipeline would notice if a strided op were silently
//!   unrolled into `vl` calls: the hit/miss totals would be identical. The call
//!   count is the only thing that catches it, so every test here asserts it.

use turbo::processing::{simulate_trace_file, CacheConfig, Counts, SimSummary};

use crate::harness::{cpu_config, skip_unless_fixture_cpu, FixtureCpu, TestRun, TurboTest};

/// The geometry the hand-computed expectations below are computed for: 32 KiB
/// 8-way L1, 512 KiB 8-way L2.
///
/// Stated here rather than read from `cpu_config.json`, because these tests
/// assert absolute hit/miss counts derived from the kernels' `.S` sources --
/// editing the shipped config must not silently invalidate them. The one test
/// that *must* track the config is
/// [`the_enricher_agrees_with_the_offline_replay`], which compares the enricher
/// against an offline replay of the same run and so reads the config the
/// enricher was given.
const MODEL: CacheConfig = CacheConfig {
    l1_bytes: 32 * 1024,
    l1_ways: 8,
    l2_bytes: 512 * 1024,
    l2_ways: 8,
};

/// Record a kernel and simulate its trace at `config`.
///
/// `tag` names the output directory, not the kernel: two tests recording the
/// same kernel must not share one, because a run wipes its directory and
/// nextest runs tests in parallel.
fn simulate(tag: &str, kernel: &str, config: CacheConfig) -> (TestRun, SimSummary) {
    let run = TurboTest::new(
        &format!("cache_sim_{tag}"),
        format!("test_bins/asm_validation/bins/{kernel}.riscv"),
    )
    .run();
    let vlenb = u64::from(
        turbo_tracer_shared::TurboMetadata::read_from_workdir(&run.dir)
            .expect("the run must have written its metadata")
            .vlen_bits,
    ) / 8;
    let summary = simulate_trace_file(&run.trace_log(0), &run.bin, vlenb, config)
        .expect("the recorded trace must simulate");
    println!(
        "{kernel}: {} call(s), l1 {}h/{}m, l2 {}h/{}m, {} pc(s)",
        summary.calls,
        summary.counts.l1_hits,
        summary.counts.l1_misses,
        summary.counts.l2_hits,
        summary.counts.l2_misses,
        summary.per_pc.len(),
    );
    for (pc, c) in &summary.per_pc {
        println!(
            "  pc {pc:#x}: l1 {}h/{}m l2 {}h/{}m ({} lines)",
            c.l1_hits,
            c.l1_misses,
            c.l2_hits,
            c.l2_misses,
            c.lines()
        );
    }
    (run, summary)
}

/// The per-PC counts must sum to the total, or attribution is losing accesses.
fn assert_per_pc_sums_to_total(s: &SimSummary) {
    let mut sum = Counts::default();
    for c in s.per_pc.values() {
        sum += *c;
    }
    assert_eq!(sum, s.counts, "per-PC counts must sum to the run total");
}

/// A unit-stride load and store, 100 iterations over the same two buffers.
///
/// Every count is readable from `vload_store_simple.S` and the symbol table:
/// `vl` is 256 at e64, so each op moves 2048 B = 32 lines from a base that never
/// moves. `source_data` is at 0x12000 and `dest_data` 512 B later at 0x12200, so
/// the load (which deliberately over-reads its 512-byte buffer) covers
/// 0x12000..0x12800 and the store 0x12200..0x12a00 -- **24 of their 32 lines are
/// the same**. The store therefore finds most of its footprint already warm,
/// which is a real property of this kernel and not an artefact.
#[test]
fn unit_stride_is_one_call_per_instruction_and_cold_once() {
    if skip_unless_fixture_cpu(
        "unit_stride_is_one_call_per_instruction_and_cold_once",
        FixtureCpu::Vlen,
    ) {
        return;
    }
    let (_run, s) = simulate("unit_stride", "vload_store_simple", MODEL);

    assert_eq!(
        s.calls, 200,
        "2 vector memory instructions x 100 iterations = 200 calls -- NOT 6400, \
         which is what one call per line would give"
    );
    assert_eq!(s.counts.lines(), 6400, "200 calls x 32 lines each");
    assert_eq!(
        s.counts.l1_misses, 64,
        "64 lines, 32x for the load and store each"
    );
    assert_eq!(s.counts.l1_hits, 6400 - 64);
    assert_eq!(
        (s.counts.l2_hits, s.counts.l2_misses),
        (0, 64),
        "every L1 miss here is a cold miss, so it misses the L2 too"
    );
    assert_per_pc_sums_to_total(&s);

    let mut per_pc: Vec<Counts> = s.per_pc.values().copied().collect();
    per_pc.sort_by_key(|c| c.l1_misses);
    assert_eq!(
        (per_pc[0].l1_misses, per_pc[0].l1_hits),
        (32, 3168),
        "the store runs second and is cold on all 32"
    );
    assert_eq!(
        (per_pc[1].l1_misses, per_pc[1].l1_hits),
        (32, 3168),
        "the load runs first and is cold on all 32"
    );
}

/// Indexed ops at `vl` below VLMAX: the model is called once per instruction,
/// and sees exactly the lines the gather touched.
#[test]
fn indexed_ops_are_one_call_each_and_reuse_is_a_hit() {
    if skip_unless_fixture_cpu(
        "indexed_ops_are_one_call_each_and_reuse_is_a_hit",
        FixtureCpu::Vlen,
    ) {
        return;
    }
    let (_run, s) = simulate("indexed", "vindexed_vl_dedup", MODEL);

    assert_eq!(
        s.calls, 200,
        "2 indexed instructions x 100 iterations -- NOT 2100, which is what one \
         call per line would give"
    );
    assert_per_pc_sums_to_total(&s);

    let mut per_pc: Vec<Counts> = s.per_pc.values().copied().collect();
    per_pc.sort_by_key(|c| c.lines());

    // The op whose 20 indices fold into one line: 1 line per execution, and
    // never a miss -- both ops share the base register, the `i*64` op runs first
    // and its index 0 element already brought that very line in. This is the
    // kind of cross-instruction reuse that only a stateful model reports, and it
    // is why the folded op shows 0 misses rather than the 1 it would show alone.
    assert_eq!(per_pc[0].lines(), 100, "1 line x 100 iterations");
    assert_eq!((per_pc[0].l1_misses, per_pc[0].l1_hits), (0, 100));

    // The i*64 op: 20 distinct lines per execution, cold once each.
    assert_eq!(per_pc[1].lines(), 2000, "20 lines x 100 iterations");
    assert_eq!((per_pc[1].l1_misses, per_pc[1].l1_hits), (20, 1980));

    assert_eq!(
        s.counts.l1_misses, 20,
        "20 distinct lines in total: the folded op's line is one of the 20"
    );
}

/// The mixed benchmark kernel: the call count is where a silently unrolled
/// strided op would show up, because its hit/miss totals would be unchanged.
#[test]
fn strided_ops_do_not_unroll_into_one_call_per_element() {
    if skip_unless_fixture_cpu(
        "strided_ops_do_not_unroll_into_one_call_per_element",
        FixtureCpu::Vlen,
    ) {
        return;
    }
    let (_run, s) = simulate("loads", "membench_loads", MODEL);

    assert_eq!(
        s.calls, 60_000,
        "6 vector memory instructions x 10000 iterations. One call per line \
         would be 2_620_000; one call per element for the strided ops alone \
         would be 1_320_000"
    );
    assert_eq!(s.per_pc.len(), 6);
    assert_per_pc_sums_to_total(&s);

    // 2 unit-stride ops of 8 lines, 2 strided and 2 indexed of 64 lines each.
    let mut lines: Vec<u64> = s.per_pc.values().map(|c| c.lines() / 10_000).collect();
    lines.sort_unstable();
    assert_eq!(lines, vec![8, 8, 64, 64, 64, 64]);

    // The working set fits in the 32 KiB L1, so after the first iteration
    // everything hits. It is 64 lines, not 128: the strided and indexed ops
    // reach `base + 63*64 + 8`, i.e. 4096 B of the 8 KiB buffer, and the
    // unit-stride ops' 512 B lie inside that.
    assert_eq!(
        s.counts.l1_misses, 64,
        "64 distinct lines (4096 B of the buffer), each missed exactly once"
    );
}

/// The store path is wired the same as the load path, and reaches the model as
/// stores. Same kernel shape as the loads, so the counts are directly
/// comparable.
#[test]
fn stores_reach_the_model_too() {
    if skip_unless_fixture_cpu("stores_reach_the_model_too", FixtureCpu::Vlen) {
        return;
    }
    let (_run, s) = simulate("stores", "membench_stores", MODEL);

    assert_eq!(
        s.calls, 60_000,
        "as membench_loads: 6 ops x 10000 iterations"
    );
    assert_eq!(
        s.counts.l1_misses, 64,
        "the same 64-line footprint as membench_loads, in dest_data"
    );
    assert_per_pc_sums_to_total(&s);

    // The model cannot tell us which kind it was told (it treats both alike), so
    // check the kind at the replay that feeds it: every access this kernel
    // produces must be a store. Otherwise `AccessKind` could be hardcoded to
    // `Load` and every assertion above would still pass.
    let vlenb = u64::from(
        turbo_tracer_shared::TurboMetadata::read_from_workdir(&_run.dir)
            .expect("metadata")
            .vlen_bits,
    ) / 8;
    let accesses = turbo::processing::replay_trace_file(&_run.trace_log(0), &_run.bin, vlenb)
        .expect("the recorded trace must replay");
    assert!(!accesses.is_empty());
    assert!(
        accesses.iter().all(|a| a.is_store),
        "every access in membench_stores is a store"
    );
}

/// A working set larger than the L1 must keep missing. Without this, every test
/// above would pass on a model that only counted first touches.
#[test]
fn a_working_set_larger_than_the_l1_keeps_missing() {
    if skip_unless_fixture_cpu(
        "a_working_set_larger_than_the_l1_keeps_missing",
        FixtureCpu::Vlen,
    ) {
        return;
    }
    // 1 KiB L1: the kernel's 8 KiB buffer cannot fit, and its 272-line
    // per-iteration footprint cannot either.
    let small = CacheConfig {
        l1_bytes: 1024,
        l1_ways: 8,
        l2_bytes: 64 * 1024,
        l2_ways: 8,
    };
    let (_run, s) = simulate("small_l1", "membench_loads", small);

    assert_eq!(s.calls, 60_000, "the call count is a property of the trace");
    assert!(
        s.counts.l1_misses > 100_000,
        "a 1 KiB L1 must keep missing on an 8 KiB working set, got {} miss(es)",
        s.counts.l1_misses
    );
    // The 64 KiB L2 does hold it, so those misses are mostly L2 hits -- the
    // check that the L1 is chaining into the L2 rather than each level seeing
    // the raw stream.
    assert_eq!(
        s.counts.l2_hits + s.counts.l2_misses,
        s.counts.l1_misses,
        "the L2 sees exactly the L1's misses"
    );
    assert_eq!(
        s.counts.l2_misses, 64,
        "the 64-line working set fits in the 64 KiB L2, so it is cold once"
    );
}

/// A kernel with no vector memory op must not call the model at all.
#[test]
fn a_kernel_without_memory_ops_never_calls_the_model() {
    let (_run, s) = simulate("no_mem_ops", "vfmacc_simple", MODEL);
    assert_eq!(s.calls, 0);
    assert_eq!(s.counts, Counts::default());
    assert!(s.per_pc.is_empty());
}

// ---------------------------------------------------------------------------
// The enricher path: `turbo run --cache-sim`, i.e. the same model driven from
// inside the pipeline rather than from `simulate_trace_file`.
// ---------------------------------------------------------------------------

/// The counts the enricher attributes must equal the ones the offline replay
/// computes for the same trace. Two independent walks of the same data: if the
/// enricher drops a block, feeds blocks out of order, or resets the model
/// between batches, they diverge.
#[test]
fn the_enricher_agrees_with_the_offline_replay() {
    if skip_unless_fixture_cpu(
        "the_enricher_agrees_with_the_offline_replay",
        FixtureCpu::VlenAndCache,
    ) {
        return;
    }
    let run = TurboTest::new(
        "cache_sim_enricher",
        "test_bins/asm_validation/bins/vload_store_simple.riscv",
    )
    .process(|p| p.cache_sim = true)
    .run();

    let pd = run.perf_data();
    // The whole-run totals live on the "Global" ROI, which is the only entry in
    // `rois` for a run with no ROI markers.
    let cache = pd
        .rois
        .iter()
        .find(|r| &*r.name == "Global")
        .expect("the Global ROI")
        .cache;
    assert!(!cache.is_zero(), "--cache-sim must produce counts");

    let vlenb = u64::from(
        turbo_tracer_shared::TurboMetadata::read_from_workdir(&run.dir)
            .expect("metadata")
            .vlen_bits,
    ) / 8;
    // The config the enricher was handed, not `MODEL`: the two walks must be of
    // the same cache, or this test fails for a reason that isn't a bug.
    let config = cpu_config()
        .cache
        .expect("the in-tree cpu_config.json must configure a cache for --cache-sim");
    let offline =
        simulate_trace_file(&run.trace_log(0), &run.bin, vlenb, config).expect("offline replay");

    assert_eq!(
        (
            cache.l1_hits,
            cache.l1_misses,
            cache.l2_hits,
            cache.l2_misses
        ),
        (
            offline.counts.l1_hits,
            offline.counts.l1_misses,
            offline.counts.l2_hits,
            offline.counts.l2_misses
        ),
        "the enricher and the offline replay must agree exactly"
    );

    // And the per-function attribution must add up to the same thing: the
    // kernel's only function is `loop`, so all of it lands there.
    let loop_roi = pd
        .functions
        .iter()
        .find(|r| &*r.name == "loop")
        .expect("ROI 'loop'");
    assert_eq!(
        loop_roi.cache, cache,
        "one function, so it holds every access"
    );
}

/// The per-PC counts reach the annotated source+asm view, and they reconcile
/// with the ROI totals.
///
/// The reconciliation is the real assertion. A vector memory instruction's
/// *first* execution reports its cache counts before the block containing it
/// commits, so its `PcRecord` does not exist yet; those counts are held aside
/// and folded in at the end. Without that, every such PC would silently lose one
/// execution's worth, which is invisible in a total and obvious here.
#[test]
fn per_pc_counts_reach_the_annotated_view_and_reconcile() {
    if skip_unless_fixture_cpu(
        "per_pc_counts_reach_the_annotated_view_and_reconcile",
        FixtureCpu::VlenAndCache,
    ) {
        return;
    }
    let run = TurboTest::new(
        "cache_sim_annotate",
        "test_bins/asm_validation/bins/vindexed_vl_dedup.riscv",
    )
    .process(|p| {
        p.cache_sim = true;
        p.annotate = true;
    })
    .run();

    // `annotate_function_pages` yields page *names*; the kernel's only function
    // is `loop`, which is where both indexed ops live.
    let names = run.annotate_function_pages();
    assert!(
        names.contains(&"annotate_loop.html".to_string()),
        "expected a page for the `loop` function, got {names:?}"
    );
    let loop_page = run.annotate_page("annotate_loop.html");

    // Both indexed ops appear, with the counts the offline replay computes:
    // 20 misses / 1980 hits for the scattered one, 0 / 100 for the folded one.
    assert!(
        loop_page.contains("data-l1h=\"1980\" data-l1m=\"20\""),
        "the i*64 gather's per-PC counts must appear in the annotation"
    );
    assert!(
        loop_page.contains("data-l1h=\"100\" data-l1m=\"0\""),
        "the folded-index gather's per-PC counts must appear too -- and its \
         first execution must not have been dropped, which would show as 99"
    );

    // Per-PC and per-ROI are two independent accumulations of the same events.
    let pd = run.perf_data();
    let global = pd
        .rois
        .iter()
        .find(|r| &*r.name == "Global")
        .expect("the Global ROI");
    assert_eq!(
        (global.cache.l1_hits, global.cache.l1_misses),
        (2080, 20),
        "1980 + 100 hits, 20 + 0 misses -- the same numbers the per-PC \
         attributes carry"
    );
}

/// Without `--cache-sim` nothing is modelled and the JSON is unchanged: the
/// counters must be absent, not zero-and-present, so an unmodelled run cannot be
/// read as a run that had no misses.
#[test]
fn without_the_flag_no_cache_counters_are_produced() {
    let run = TurboTest::new(
        "cache_sim_disabled",
        "test_bins/asm_validation/bins/vload_store_simple.riscv",
    )
    .run();

    let pd = run.perf_data();
    assert!(pd.rois.iter().all(|r| r.cache.is_zero()));
    assert!(pd.functions.iter().all(|r| r.cache.is_zero()));
    let json = std::fs::read_to_string(run.dir.join("perf_data_final.json")).expect("json");
    assert!(
        !json.contains("l1_hits"),
        "an unmodelled run must not serialise cache fields at all"
    );
}

// ---------------------------------------------------------------------------
// One recording, several CPU configurations.
// ---------------------------------------------------------------------------

/// Write a `cpu_config.json` into `target/test_output/` and return its path.
fn cpu_config_file(name: &str, vlen_bits: u32, cache: CacheConfig) -> std::path::PathBuf {
    let path = crate::harness::out_dir(&format!("cpu_config_{name}")).join("cpu_config.json");
    let json = serde_json::json!({
        "vlen": vlen_bits,
        "cache": {
            "l1_bytes": cache.l1_bytes,
            "l1_ways": cache.l1_ways,
            "l2_bytes": cache.l2_bytes,
            "l2_ways": cache.l2_ways,
        }
    });
    std::fs::write(&path, serde_json::to_string_pretty(&json).expect("json")).expect("write");
    path
}

/// The two halves of `cpu_config.json` are not equally re-configurable, and this
/// pins the asymmetry:
///
/// * **cache geometry may differ from the recording.** The trace carries captured
///   memory addresses, so `process --cache-sim` re-simulates them against
///   whatever cache is configured -- which is the point of the replay, and the
///   whole reason to run `process` twice on one recording.
/// * **VLEN may not.** It is not replayed, it is what the recorded instructions
///   *meant*: the guest executed at the VLEN the plugin read from QEMU's `vlenb`
///   CSR, and re-enriching at another would mis-size every vector op. So a
///   `cpu_config.json` claiming a different VLEN is warned about loudly and then
///   ignored, which is what the third reprocess below checks.
#[test]
fn cache_geometry_is_reconfigurable_per_process_but_vlen_is_not() {
    if skip_unless_fixture_cpu(
        "cache_geometry_is_reconfigurable_per_process_but_vlen_is_not",
        FixtureCpu::Vlen,
    ) {
        return;
    }
    let recording = TurboTest::new(
        "cache_sim_recfg_record",
        "test_bins/asm_validation/bins/membench_loads.riscv",
    )
    .record();

    let recorded_vlen = turbo_tracer_shared::TurboMetadata::read_from_workdir(&recording.dir)
        .expect("metadata")
        .vlen_bits;

    let tiny = CacheConfig {
        l1_bytes: 1024,
        l1_ways: 2,
        l2_bytes: 16 * 1024,
        l2_ways: 4,
    };

    // Same recording, three configurations: the shipped cache, a tiny one, and
    // the tiny one paired with a VLEN the run never used.
    let big_run = recording.reprocess_with_cpu_config(
        "cache_sim_recfg_big",
        &recording.dir,
        cpu_config_file("big", recorded_vlen, MODEL),
        |p| p.cache_sim = true,
    );
    let tiny_run = recording.reprocess_with_cpu_config(
        "cache_sim_recfg_tiny",
        &recording.dir,
        cpu_config_file("tiny", recorded_vlen, tiny),
        |p| p.cache_sim = true,
    );
    let wrong_vlen_run = recording.reprocess_with_cpu_config(
        "cache_sim_recfg_wrong_vlen",
        &recording.dir,
        cpu_config_file("wrong_vlen", recorded_vlen / 8, tiny),
        |p| p.cache_sim = true,
    );

    let cache = |r: &TestRun| {
        let pd = r.perf_data();
        let g = pd
            .rois
            .iter()
            .find(|roi| &*roi.name == "Global")
            .expect("the Global ROI");
        (g.instructions, g.cache)
    };
    let (big_insns, big_cache) = cache(&big_run);
    let (tiny_insns, tiny_cache) = cache(&tiny_run);
    let (wrong_insns, wrong_cache) = cache(&wrong_vlen_run);

    // The cache is genuinely re-simulated: a 1 KiB L1 cannot hold the kernel's
    // 8 KiB working set, so it must miss where the 32 KiB one hits.
    assert!(
        tiny_cache.l1_misses > big_cache.l1_misses * 100,
        "a 1 KiB L1 must miss far more than a 32 KiB one over the same \
         addresses, got {} vs {}",
        tiny_cache.l1_misses,
        big_cache.l1_misses
    );
    assert_eq!(
        tiny_cache.l2_hits + tiny_cache.l2_misses,
        tiny_cache.l1_misses,
        "the L2 still sees exactly the L1's misses"
    );

    // VLEN, in contrast, is unaffected by what the config claimed: identical
    // instruction counts, and byte-identical cache counts to the run that
    // configured the recording's own VLEN.
    assert_eq!(
        (big_insns, tiny_insns),
        (wrong_insns, wrong_insns),
        "the recorded VLEN is authoritative, so the instruction count cannot \
         move with the configured one"
    );
    assert_eq!(
        wrong_cache, tiny_cache,
        "a mismatched configured VLEN must be ignored (loudly), not applied: \
         these two differ only in a `vlen` the recording contradicts"
    );
}
