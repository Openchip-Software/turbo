use std::fs;

use serde_json::Value;
use turbo::{common::PerformanceData, processing::RoiInfo};

use crate::harness::{skip_if_missing, TestRun, TurboTest};

/// Path of the `roi_validation` guest built from `version`/`binary_name`.
fn roi_binary(version: &str, binary_name: &str) -> String {
    format!("test_bins/roi_validation/bins/{version}_{binary_name}")
}

fn run_roi_validation(version: &str, binary_name: &str) -> PerformanceData {
    let pd = TurboTest::new(
        &format!("roi_{version}_{binary_name}"),
        roi_binary(version, binary_name),
    )
    .run()
    .perf_data();
    dbg!(&pd.rois);
    pd
}

/// [`run_roi_validation`], returning early from the calling test when the guest
/// was not built.
///
/// Every guest here `#include`s the external RAVE headers, which are not
/// vendored in the repo, so `Makefile.tests` skips `roi_validation` entirely --
/// see [`skip_if_missing`]. A macro rather than an `Option`-returning helper
/// because the skip has to `return` from the test itself.
macro_rules! roi_validation {
    ($version:expr, $binary_name:expr) => {{
        let (version, binary_name) = ($version, $binary_name);
        if skip_if_missing(roi_binary(version, binary_name)) {
            return;
        }
        run_roi_validation(version, binary_name)
    }};
}

fn read_flamegraph(run: &TestRun) -> Value {
    let path = run.dir.join("roi_flamegraph.json");
    assert!(
        path.exists(),
        "missing flamegraph artifact: {}",
        path.display()
    );
    let text = fs::read_to_string(&path).expect("read flamegraph json");
    serde_json::from_str(&text).expect("parse flamegraph json")
}

fn sort_by_region_name(rois: &mut [RoiInfo]) {
    rois.sort_unstable_by(|a, b| a.name.cmp(&b.name))
}

/// Filter out the Global ROI from the list for testing purposes
fn filter_global_roi(rois: Vec<RoiInfo>) -> Vec<RoiInfo> {
    rois.into_iter().filter(|r| &*r.name != "Global").collect()
}

macro_rules! roi_common_tests {
    ($mod_name:ident, $version:expr) => {
        mod $mod_name {
            use super::*;

            #[test]
            fn one_region() {
                let pd = roi_validation!($version, "one_region.riscv");
                let rois = filter_global_roi(pd.rois);
                assert_eq!(rois.len(), 1);
                let roi = &rois[0];
                assert_eq!(&*roi.name, "region_1");
                assert_eq!(roi.instructions, 1);
            }

            #[test]
            fn non_overlapping() {
                let pd = roi_validation!($version, "non_overlapping.riscv");
                let mut rois = filter_global_roi(pd.rois);
                sort_by_region_name(&mut rois);
                assert_eq!(rois.len(), 3);
                assert_eq!(&*rois[0].name, "region_1");
                assert_eq!(&*rois[1].name, "region_2");
                assert_eq!(&*rois[2].name, "region_3");

                assert_eq!(rois[0].instructions, 1);
                assert_eq!(rois[1].instructions, 2);
                assert_eq!(rois[2].instructions, 1);
            }

            #[test]
            // Region is opened but never closed, so it still gets tracked
            fn opening_only() {
                let pd = roi_validation!($version, "opening_only.riscv");
                let rois = filter_global_roi(pd.rois);
                // Region exists but is incomplete
                assert_eq!(rois.len(), 1);
                assert_eq!(&*rois[0].name, "region_1");
            }

            #[test]
            // Region close without open doesn't create a region
            fn closing_only() {
                let pd = roi_validation!($version, "closing_only.riscv");
                let rois = filter_global_roi(pd.rois);
                // No region created when closing without opening
                assert_eq!(rois.len(), 0);
            }

            #[test]
            // Visiting the same region multiple times should
            // result in one aggregated region, even when it is called from
            // the recursive regions in the v2 fixture.
            fn repeated() {
                let pd = roi_validation!($version, "repeated.riscv");
                let mut rois = filter_global_roi(pd.rois);
                sort_by_region_name(&mut rois);

                let expected: Vec<(&str, u64)> = if $version == "v2" {
                    vec![
                        ("iterative loop", 15),
                        ("recursion-work", 5),
                        ("recursive_region_1", 12),
                        ("recursive_region_2", 636),
                        ("recursive_region_3", 1260),
                        ("recursive_region_4", 1884),
                        ("recursive_region_5", 2508),
                    ]
                } else {
                    vec![("iterative loop", 10)]
                };
                assert_eq!(rois.len(), expected.len());
                for (roi, (name, instructions)) in rois.iter().zip(expected) {
                    assert_eq!(&*roi.name, name);
                    assert_eq!(roi.instructions, instructions);
                }
            }

            #[test]
            // Multiple regions with the same name should be aggregated
            fn distributed() {
                let pd = roi_validation!($version, "distributed.riscv");
                let rois = filter_global_roi(pd.rois);
                assert_eq!(rois.len(), 1);
                let roi = &rois[0];
                assert_eq!(&*roi.name, "region_1");
                assert_eq!(roi.instructions, 3);
            }

            #[test]
            // Supports all non-whitespace printable ASCII characters plus space
            fn naming() {
                let region_name = "ABCDEFGHIJKLMNOPQRSTUVWXYZ\
                                    abcdefghijklmnopqrstuvwxyz\
                                    0123456789\
                                    ! #$%&'()*+,-./:;<=>?@[\\]^_`{|}~'\"?";
                let pd = roi_validation!($version, "naming.riscv");
                let rois = filter_global_roi(pd.rois);
                assert_eq!(rois.len(), 1);
                assert_eq!(&*rois[0].name, region_name);
            }

            #[test]
            fn restart_trace() {
                let pd = roi_validation!($version, "restart.riscv");
                let rois = filter_global_roi(pd.rois);
                assert_eq!(rois.len(), 1);
                assert_eq!(&*rois[0].name, "region_2");

                assert_eq!(rois[0].instructions, 2);
            }
        }
    };
}

roi_common_tests!(v1_common, "v1");
roi_common_tests!(v2_common, "v2");

mod v2 {

    use super::*;

    #[test]
    fn nested() {
        let pd = roi_validation!("v2", "nested.riscv");
        let mut rois = filter_global_roi(pd.rois);
        sort_by_region_name(&mut rois);
        assert_eq!(rois.len(), 2);
        assert_eq!(&*rois[0].name, "region_1");
        assert_eq!(&*rois[1].name, "region_2");

        assert_eq!(rois[0].instructions, 8);
        assert_eq!(rois[1].instructions, 2);
    }

    #[test]
    fn overlapping() {
        let pd = roi_validation!("v2", "overlapping.riscv");
        let mut rois = filter_global_roi(pd.rois);
        sort_by_region_name(&mut rois);
        dbg!(&rois);
        assert_eq!(rois.len(), 2);
        assert_eq!(&*rois[0].name, "region_1");
        assert_eq!(&*rois[1].name, "region_2");

        assert_eq!(rois[0].instructions, 7);
        assert_eq!(rois[1].instructions, 4);
    }

    #[test]
    // start_trace and stop_trace are ignored by v2
    // In v2, these only affect Paraver
    // Use enable and disable to affect region counters
    fn start_stop() {
        let pd = roi_validation!("v2", "start_stop.riscv");
        let mut rois = filter_global_roi(pd.rois);
        sort_by_region_name(&mut rois);
        assert_eq!(rois.len(), 3);
        assert_eq!(&*rois[0].name, "region_1");
        assert_eq!(&*rois[1].name, "region_2");
        assert_eq!(&*rois[2].name, "region_3");

        assert_eq!(rois[0].instructions, 1);
        assert_eq!(rois[1].instructions, 2);
        assert_eq!(rois[2].instructions, 3);
    }

    #[test]
    fn function_call() {
        let pd = roi_validation!("v2", "function_call.riscv");
        let rois = filter_global_roi(pd.rois);
        assert_eq!(rois.len(), 1);
        assert_eq!(&*rois[0].name, "region_1");

        assert_eq!(rois[0].instructions, 5);
    }

    #[test]
    // A >64-byte region name passed by pointer+length from the *stack* (the
    // Fortran/cloudsc shape). The name has to be captured from guest memory at
    // runtime; there is no ELF fallback for a stack pointer, so a dropped read
    // shows up as a "roi_region_0x<pc>" region.
    fn long_name_stack() {
        let expected = "This is a very very very very very very very very very very very very very very long region name";
        assert!(expected.len() > 64);

        let pd = roi_validation!("v2", "long_name_stack.riscv");
        let rois = filter_global_roi(pd.rois);
        assert_eq!(rois.len(), 1);
        assert_eq!(&*rois[0].name, expected);
        assert_eq!(rois[0].instructions, 1);
    }

    #[test]
    fn enable_disable() {
        let pd = roi_validation!("v2", "enable_disable.riscv");
        let mut rois = filter_global_roi(pd.rois);
        sort_by_region_name(&mut rois);
        assert_eq!(rois.len(), 3);
        assert_eq!(&*rois[0].name, "region_1");
        assert_eq!(&*rois[1].name, "region_3");
        assert_eq!(&*rois[2].name, "region_5");

        assert_eq!(rois[0].instructions, 1);
        assert_eq!(rois[1].instructions, 3);
        assert_eq!(rois[2].instructions, 2);
    }

    #[test]
    fn flamegraph_chain_20_deep() {
        if skip_if_missing("test_bins/roi_validation/bins/v2_flamegraph_chain.riscv") {
            return;
        }
        let run = TurboTest::new(
            "roi_v2_flamegraph_chain",
            "test_bins/roi_validation/bins/v2_flamegraph_chain.riscv",
        )
        .run();

        let root = read_flamegraph(&run);
        assert_eq!(root["name"], "Global");
        assert!(root["self_value"].as_u64().unwrap() > 0);

        let outer_driver = root["children"]
            .as_array()
            .unwrap()
            .iter()
            .find(|child| child["name"] == "outer_driver")
            .expect("outer_driver in flamegraph");
        // Interior nodes must carry the region they represent, not a
        // placeholder: the name-keyed builder minted every *parent* node with a
        // hardcoded region_id of 0, so only nodes first seen as a leaf reported
        // a real id and every interior node of this 20-deep chain read as
        // region 0. `outer_driver` is the first region the binary opens, so it
        // is genuinely region 0 -- assert on `fan_out` and deeper instead.
        let fan_out = outer_driver["children"]
            .as_array()
            .unwrap()
            .iter()
            .find(|child| child["name"] == "fan_out")
            .expect("fan_out in flamegraph");
        assert_eq!(fan_out["call_count"].as_u64().unwrap(), 3);
        assert_ne!(
            fan_out["region_id"].as_u64().unwrap(),
            0,
            "interior node must report its own region_id, not a placeholder"
        );

        let mut current = fan_out["children"]
            .as_array()
            .unwrap()
            .iter()
            .find(|child| child["name"] == "chain_01")
            .expect("chain_01 in flamegraph");

        for depth in 2..=20 {
            let expected = format!("chain_{depth:02}");
            let next = current["children"]
                .as_array()
                .unwrap()
                .iter()
                .find(|child| child["name"] == expected)
                .unwrap_or_else(|| panic!("missing {expected} in flamegraph"));
            assert_ne!(
                next["region_id"].as_u64().unwrap(),
                0,
                "{expected} must report its own region_id"
            );
            current = next;
        }

        let leaf_loop = current["children"]
            .as_array()
            .unwrap()
            .iter()
            .find(|child| child["name"] == "leaf_loop")
            .expect("leaf_loop in flamegraph");
        assert_eq!(leaf_loop["call_count"].as_u64().unwrap(), 9);
    }

    #[test]
    // Region counting toggled *inside* an open region, which `enable_disable`
    // never does: there every toggle lands with an empty region stack.
    //
    // Two properties, both of which the tracker gets wrong today (see the
    // ignore reason): an open region must not accumulate while counting is
    // disabled, and a close marker arriving while disabled must still take
    // effect rather than leaving the region open to the end of the program.
    #[ignore = "known failure, measured on this binary: region_1 counts 5 rather \
                than 2, because an already-open region keeps accumulating while \
                counting is disabled; region_2 counts 907 rather than 2, because \
                its close marker is swallowed while disabled, so it stays open \
                across libc teardown. Un-ignore once the tracker closes/suspends \
                open regions on a gate change."]
    fn disable_inside_region() {
        let pd = roi_validation!("v2", "disable_inside_region.riscv");
        let mut rois = filter_global_roi(pd.rois);
        sort_by_region_name(&mut rois);
        assert_eq!(rois.len(), 2);
        assert_eq!(&*rois[0].name, "region_1");
        assert_eq!(&*rois[1].name, "region_2");

        // region_1 spans 5 instructions, 3 of them inside the disabled window.
        assert_eq!(rois[0].instructions, 2);

        // region_2's close arrives while disabled. If it is honoured, the 3
        // trailing instructions (and all of libc teardown) stay out of the
        // region; if it is swallowed, this reads in the hundreds.
        assert_eq!(rois[1].instructions, 2);
    }

    #[test]
    // This test attempts to capture a race condition that sometimes occurs with large regions.
    // To test for it, run with different logging levels.
    // The different levels deterministically alter the execution time of the
    // decoder, resulting in different warnings due to missed region markers
    //
    // ```
    // cargo test --release --test roi_validation large_trace -- --nocapture 2>&1 | grep -E "WARN|ERROR"
    // RUST_LOG=info cargo test --release --test roi_validation large_trace -- --nocapture 2>&1 | grep -E "WARN|ERROR"
    // RUST_LOG=warn cargo test --release --test roi_validation large_trace -- --nocapture 2>&1 | grep -E "WARN|ERROR"
    // RUST_LOG=debug cargo test --release --test roi_validation large_trace -- --nocapture 2>&1 | grep -E "WARN|ERROR"
    // RUST_LOG=trace cargo test --release --test roi_validation large_trace -- --nocapture 2>&1 | grep -E "WARN|ERROR"
    // ```
    fn large_trace() {
        roi_validation!("v2", "large_trace.riscv");
    }
}

mod v1 {
    use super::*;

    #[test]
    fn manual() {
        let pd = roi_validation!("v1", "manual.riscv");
        let rois = filter_global_roi(pd.rois);
        assert_eq!(rois.len(), 1);
        assert_eq!(&*rois[0].name, "region_1");
        assert_eq!(rois[0].instructions, 1);
    }

    #[test]
    fn start_stop() {
        let pd = roi_validation!("v1", "start_stop.riscv");
        let mut rois = filter_global_roi(pd.rois);
        sort_by_region_name(&mut rois);
        assert_eq!(rois.len(), 2);
        assert_eq!(&*rois[0].name, "region_1");
        assert_eq!(&*rois[1].name, "region_3");

        assert_eq!(rois[0].instructions, 1);
        assert_eq!(rois[1].instructions, 3);
    }

    #[test]
    // Compiler inserts an unnecessary instruction in v1, resulting in a different count
    fn function_call() {
        let pd = roi_validation!("v1", "function_call.riscv");
        let rois = filter_global_roi(pd.rois);
        assert_eq!(rois.len(), 1);
        assert_eq!(&*rois[0].name, "region_1");

        // One higher than the v2 count
        assert_eq!(rois[0].instructions, 6);
    }
}
