//! Verifies that per-hart (per-vCPU) PerformanceData stays available
//! individually alongside the merged whole-program view, so per-thread
//! imbalance/issues can be diagnosed later instead of only seeing summed
//! totals.

use crate::harness::{skip_if_missing, TurboTest};

/// test_bins/thread_roi/pthreads_roi.c spawns 3 pthreads, each running its
/// own uniquely named ROI region ("thread_<N>_work") with a distinct,
/// deterministic instruction count (2, 4, 6 respectively). This lets us
/// confirm each thread's stats land in a separate per-vcpu file and are not
/// conflated with one another.
#[test]
fn per_vcpu_perf_json_written_per_thread() {
    if skip_if_missing("test_bins/thread_roi/pthreads_roi.riscv") {
        return;
    }
    let run = TurboTest::new("pthreads_roi", "test_bins/thread_roi/pthreads_roi.riscv").run();
    let merged = run.perf_data();

    // The merged, whole-program view should contain all three thread regions
    // summed with the rest of the run into a single global picture.
    for name in ["thread_0_work", "thread_1_work", "thread_2_work"] {
        assert!(
            merged.rois.iter().any(|r| &*r.name == name),
            "expected merged PerformanceData to contain ROI `{name}`"
        );
    }

    // Each hart also gets its own turbo_vcpu_<N>_perf.json, independent of
    // the merged file, containing only that thread's own work.
    let expected_regions = [
        ("thread_0_work", 2u64),
        ("thread_1_work", 4u64),
        ("thread_2_work", 6u64),
    ];

    // How many per-vcpu files each named region appears in. Must end up as
    // exactly 1 each: found at all (the region reached its hart's snapshot),
    // and never more than once (harts don't leak each other's regions).
    let mut region_hart_counts = std::collections::HashMap::new();
    for hart_id in 0..8 {
        let Some(pd) = run.vcpu_perf(hart_id) else {
            continue;
        };

        // Every per-hart snapshot must still carry its own "Global" ROI --
        // it's a whole-program view of that single thread, not just the
        // named regions.
        assert!(
            pd.rois.iter().any(|r| &*r.name == "Global"),
            "hart {hart_id}'s snapshot is missing its own Global ROI"
        );

        for (name, expected_instructions) in expected_regions {
            if let Some(roi) = pd.rois.iter().find(|r| &*r.name == name) {
                assert_eq!(
                    roi.instructions, expected_instructions,
                    "hart {hart_id}'s `{name}` region has unexpected instruction count"
                );
                *region_hart_counts.entry(name).or_insert(0usize) += 1;
            }
        }
    }

    for (name, _) in expected_regions {
        assert_eq!(
            region_hart_counts.get(name).copied().unwrap_or(0),
            1,
            "region `{name}` must appear in exactly one per-vcpu file, counts: {region_hart_counts:?}"
        );
    }
}

/// The same run also writes one flamegraph per thread beside the merged one,
/// plus the `roi_flamegraph_harts.json` index a viewer's thread selector is
/// built from.
///
/// The merged graph folds the three threads' regions into siblings of one
/// synthetic root and is the only view today; the per-thread graphs are what
/// make "which thread did this work" answerable, since merging re-keys by name
/// and would collapse a shared region name across every hart.
#[test]
fn per_thread_flamegraphs_and_index_written() {
    if skip_if_missing("test_bins/thread_roi/pthreads_roi.riscv") {
        return;
    }
    let run = TurboTest::new(
        "pthreads_roi_flame",
        "test_bins/thread_roi/pthreads_roi.riscv",
    )
    .run();

    let index: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(run.dir.join("roi_flamegraph_harts.json"))
            .expect("roi_flamegraph_harts.json must be written for a multi-thread run"),
    )
    .expect("parse the per-thread flamegraph index");
    let harts = index["harts"].as_array().expect("index has a harts array");

    // Every thread that retired anything: the three workers *and* the main
    // thread, which opens no region at all and is nonetheless ~98% of the run.
    assert_eq!(
        harts.len(),
        4,
        "expected one entry per guest thread, got {index:#}"
    );

    let ids: Vec<u64> = harts
        .iter()
        .map(|h| h["hart_id"].as_u64().unwrap())
        .collect();
    let mut sorted = ids.clone();
    sorted.sort_unstable();
    assert_eq!(ids, sorted, "index must be in hart order, got {ids:?}");

    let merged_global = run
        .perf_data()
        .rois
        .iter()
        .find(|r| &*r.name == "Global")
        .expect("merged Global ROI")
        .instructions;

    // Each thread's region and instruction count, as pthreads_roi.c sets them
    // up: thread N runs `thread_N_work` for 2*(N+1) instructions. The main
    // thread has none.
    let mut seen_regions = Vec::new();
    let mut summed = 0u64;
    let mut region_free = 0;
    for hart in harts {
        let hart_id = hart["hart_id"].as_u64().unwrap();
        let json_name = hart["json"].as_str().unwrap();
        assert_eq!(json_name, format!("roi_flamegraph_hart{hart_id}.json"));

        // The index must not name a file that was never written.
        let root: serde_json::Value = serde_json::from_str(
            &std::fs::read_to_string(run.dir.join(json_name)).expect("per-hart graph JSON"),
        )
        .expect("parse per-hart graph");

        // Rooted at *this hart's* retired count, not the run's, so its
        // self-time is that thread's own work outside any region.
        assert_eq!(root["name"], "Global");
        let retired = root["value"].as_u64().unwrap();
        assert_eq!(retired, hart["instructions"].as_u64().unwrap());
        assert_eq!(
            root["self_value"].as_u64().unwrap(),
            hart["self_instructions"].as_u64().unwrap()
        );
        assert!(
            retired < merged_global,
            "a single thread cannot have retired the whole run"
        );
        summed += retired;

        let children = root["children"].as_array().unwrap();
        // The folded companion, for flamegraph.pl.
        let folded =
            std::fs::read_to_string(run.dir.join(format!("roi_flamegraph_hart{hart_id}.folded")))
                .expect("per-hart folded stacks");

        if children.is_empty() {
            // The main thread: no region, so the graph is a bare root and all
            // of its instructions are self-time. Most of the run lives here,
            // which is why it has to be listed at all.
            region_free += 1;
            assert_eq!(root["self_value"].as_u64().unwrap(), retired);
            assert_eq!(folded.trim(), format!("Global {retired}"));
            continue;
        }

        // A worker: exactly one region, and only its own.
        assert_eq!(children.len(), 1, "hart {hart_id} should own one region");
        let region = children[0]["name"].as_str().unwrap().to_string();
        assert!(
            region.starts_with("thread_") && region.ends_with("_work"),
            "hart {hart_id}'s region is `{region}`"
        );
        let n: u64 = region
            .trim_start_matches("thread_")
            .trim_end_matches("_work")
            .parse()
            .unwrap();
        assert_eq!(children[0]["value"].as_u64().unwrap(), 2 * (n + 1));
        assert!(
            folded.contains(&format!("Global;{region}")),
            "hart {hart_id}'s folded stacks are {folded:?}"
        );
        seen_regions.push(region);
    }

    seen_regions.sort();
    assert_eq!(
        seen_regions,
        vec!["thread_0_work", "thread_1_work", "thread_2_work"],
        "each worker's region must appear on exactly one hart"
    );
    assert_eq!(region_free, 1, "exactly one thread entered no region");

    // `Global` is additive across harts, so listing every thread that retired
    // anything makes the index account for the whole run. This is the property
    // that breaks the moment a thread is left out.
    assert_eq!(
        summed, merged_global,
        "per-thread graphs must sum to the merged run total"
    );
}

/// A single-threaded run's output set is unchanged: with one graph the
/// per-thread view *is* the merged view, so writing it would add files to
/// every existing run's output for no information gained.
#[test]
fn single_thread_run_writes_no_per_thread_flamegraphs() {
    if skip_if_missing("test_bins/roi_validation/bins/v2_flamegraph_chain.riscv") {
        return;
    }
    let run = TurboTest::new(
        "roi_single_thread_flame",
        "test_bins/roi_validation/bins/v2_flamegraph_chain.riscv",
    )
    .run();

    assert!(
        run.dir.join("roi_flamegraph.json").exists(),
        "the merged graph must still be written"
    );
    assert!(
        !run.dir.join("roi_flamegraph_harts.json").exists(),
        "no per-thread index for a single-graph run"
    );
    let strays: Vec<String> = std::fs::read_dir(&run.dir)
        .unwrap()
        .filter_map(|e| e.ok())
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .filter(|n| n.starts_with("roi_flamegraph_hart"))
        .collect();
    assert!(strays.is_empty(), "unexpected per-hart graphs: {strays:?}");
}
