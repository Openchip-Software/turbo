//! Assembly validation tests for RVV instruction profiling
//!
//! These tests use hand-crafted assembly files with known, calculable metrics
//! to validate that the profiling tool correctly counts instructions, FLOPs,
//! memory operations, and other HPC-relevant metrics.

use crate::harness::{
    load_expected_metrics, skip_unless_fixture_cpu, validate_metrics, FixtureCpu, TestRun,
    TurboTest,
};

/// Run one `test_bins/asm_validation` binary and assert its per-function
/// metrics match the `expected_metrics.json` fixture entry of the same name.
/// `None` when skipped: every fixture entry assumes the fixture VLEN.
fn run_asm_validation(test_name: &str) -> Option<TestRun> {
    if skip_unless_fixture_cpu(test_name, FixtureCpu::Vlen) {
        return None;
    }
    let metrics = load_expected_metrics();
    let expected = metrics.get(test_name).expect("Expected metrics not found");

    let run = TurboTest::new(
        test_name,
        format!("test_bins/asm_validation/bins/{test_name}.riscv"),
    )
    .run();
    validate_metrics(test_name, &run, expected);
    Some(run)
}

/// Assert the exact per-opcode instruction mix of the `loop` function, on top
/// of the `addi`/`bne` loop counter every one of these kernels runs 100 times.
///
/// Per-function `instruction_mix` is derived from the `PcProfile` at finalize
/// time (`Enricher::apply_deferred_instruction_mix`) rather than bumped per
/// instruction, so it is only populated when annotate is enabled -- which is
/// the CLI default, and hence what `run_asm_validation` runs with.
fn assert_loop_mix(run: &TestRun, body: &[(&str, u64)]) {
    let actual_pd = run.perf_data();
    let roi = actual_pd
        .functions
        .iter()
        .find(|roi| &*roi.name == "loop")
        .expect("ROI 'loop' not found in performance data");

    let expected: std::collections::BTreeMap<String, u64> = body
        .iter()
        .chain(&[("addi", 100), ("bne", 100)])
        .map(|(mnemonic, count)| (mnemonic.to_string(), *count))
        .collect();
    assert_eq!(
        roi.instruction_mix, expected,
        "instruction mix mismatch for ROI 'loop'"
    );
}

// Kernels whose whole assertion is "the counters match the expected_metrics.json
// entry of the same name". Vector arithmetic, memory, and mixed/realistic HPC
// patterns; the fixture says what each one expects.
turbo_tests_named!(run_asm_validation {
    vfmacc_simple,
    vadd_simple,
    vl_variation,
    lmul_stress,
    vfmul_simple,
    compute_intensive,
    vload_store_simple,
    stream_triad,
    daxpy,
    full_reg_ops,
    vzvbb_ops,
    vload_widths,
});

// The kernels below additionally pin the exact per-opcode mnemonics, because
// the mnemonics are what is under test: segmented and indexed load/stores share
// discriminants and must not collapse into one bucket.

#[test]
fn vseg_load_store() {
    let Some(run) = run_asm_validation("vseg_load_store") else {
        return;
    };

    // Four distinct segment mnemonics that all share the VLSEG_V / VSSEG_V
    // discriminants -- they must not collapse into one bucket each.
    assert_loop_mix(
        &run,
        &[
            ("vlseg2e64.v", 100),
            ("vlseg3e64.v", 100),
            ("vlseg2e32.v", 100),
            ("vlseg4e32.v", 100),
            ("vlseg8e8.v", 100),
            ("vsetvli", 300),
            ("vsseg2e32.v", 100),
            ("vsseg2e64.v", 100),
            ("vsseg3e64.v", 100),
            ("vsseg4e32.v", 100),
            ("vsseg8e8.v", 100),
        ],
    );
}

#[test]
fn vseg_strided_indexed() {
    let Some(run) = run_asm_validation("vseg_strided_indexed") else {
        return;
    };

    assert_loop_mix(
        &run,
        &[
            ("vloxseg2ei64.v", 100),
            ("vlsseg2e64.v", 100),
            ("vluxseg2ei64.v", 100),
            ("vsoxseg2ei64.v", 100),
            ("vssseg2e64.v", 100),
            ("vsuxseg2ei64.v", 100),
        ],
    );
}

#[test]
fn vindexed_eew_mismatch() {
    let Some(run) = run_asm_validation("vindexed_eew_mismatch") else {
        return;
    };

    assert_loop_mix(
        &run,
        &[
            ("vloxei64.v", 100),
            ("vluxei32.v", 100),
            ("vluxseg2ei16.v", 100),
            ("vsoxseg2ei16.v", 100),
            ("vsuxei32.v", 100),
        ],
    );
}

#[test]
fn vseg_nf_sweep() {
    let Some(run) = run_asm_validation("vseg_nf_sweep") else {
        return;
    };

    // Every NFIELDS value must render as its own mnemonic.
    assert_loop_mix(
        &run,
        &[
            ("vlseg2e8.v", 100),
            ("vlseg2e8ff.v", 100),
            ("vlseg3e8.v", 100),
            ("vlseg4e8.v", 100),
            ("vlseg5e8.v", 100),
            ("vlseg6e8.v", 100),
            ("vlseg7e8.v", 100),
            ("vlseg8e8.v", 100),
            ("vlsseg2e8.v", 100),
            ("vsseg2e8.v", 100),
            ("vssseg2e8.v", 100),
        ],
    );
}

#[test]
fn vindexed_widths() {
    let Some(run) = run_asm_validation("vindexed_widths") else {
        return;
    };

    assert_loop_mix(
        &run,
        &[
            ("vloxei16.v", 100),
            ("vloxei32.v", 100),
            ("vloxei64.v", 100),
            ("vloxei8.v", 100),
            ("vluxei16.v", 100),
            ("vluxei32.v", 100),
            ("vluxei64.v", 100),
            ("vluxei8.v", 100),
            ("vluxseg3ei8.v", 100),
            ("vsoxei16.v", 100),
            ("vsoxei32.v", 100),
            ("vsoxei64.v", 100),
            ("vsoxei8.v", 100),
            ("vsoxseg3ei32.v", 100),
            ("vsuxei16.v", 100),
            ("vsuxei32.v", 100),
            ("vsuxei64.v", 100),
            ("vsuxei8.v", 100),
        ],
    );
}

/// A fault-only-first load whose `vl` is really trimmed, the store that runs at
/// the trimmed length, and the `vsetvli x0, x0` that inherits it as its AVL.
///
/// The fixture numbers are the whole assertion, and they are the ones that were
/// wrong before `vl*ff` capture: `vle8ff.v` requests 16 elements, faults on
/// element 8 (the guest punches a page hole to guarantee it), and writes
/// `vl = 8`. So the ff moves 8 bytes and each of the two `vse8.v` behind it
/// moves 8 more -- 800 loaded and 1600 stored over the 100 iterations. A tool
/// that does not capture the trim keeps `vl = 16` from the `vsetivli` and reports
/// exactly double for every one of `load_bytes`, `store_bytes`,
/// `vector_elements` and `vector_load_elements`.
///
/// The second store is behind a `vsetvli zero, zero`, whose AVL is the *current*
/// `vl`. The tracer computes that one from its own shadow rather than reading
/// the CSR, so this is the case that pins the trim being folded into the shadow
/// and not merely into the emitted record.
#[test]
fn vleff_trim() {
    let Some(run) = run_asm_validation("vleff_trim") else {
        return;
    };

    assert_loop_mix(
        &run,
        &[
            ("vsetivli", 100),
            ("vle8ff.v", 100),
            ("vse8.v", 200),
            ("vsetvli", 100),
        ],
    );

    // The VL histogram is the direct statement of "the tool believes the guest
    // ran at vl = 8". All three vector ops in the body are e8/m1, so one bucket.
    let roi = run
        .perf_data()
        .functions
        .iter()
        .find(|roi| &*roi.name == "loop")
        .expect("ROI 'loop' not found")
        .clone();
    assert_eq!(
        roi.vl_hist_e8_raw,
        vec![[8, 300]],
        "the trimmed vl = 8 must be the only e8 length seen, over all three vector \
         ops of all 100 iterations -- vl = 16 here means the vle8ff.v trim was missed"
    );
}

/// A spawned guest thread inherits the spawning thread's `vl`, and the tool
/// must compute from the inherited value rather than from zero.
///
/// QEMU linux-user builds a new thread's vCPU with `cpu_copy()` -- a `memcpy`
/// of the whole `CPUArchState` -- and `cpu_clone_regs_child()` then fixes up
/// only `sp` and `a0`, so the child starts at whatever `vl`/`vtype` the parent
/// was running at when it called `clone`. The tracer's per-vCPU `vl` shadow was
/// zeroed at vCPU init on the opposite assumption.
///
/// The fixture's child loop is 100 iterations of `vsetvli zero, zero` plus one
/// `vse8.v`, and nothing in it names a length: that form's AVL is the *current*
/// `vl`, which the plugin supplies from its shadow rather than a CSR read. So
/// the store's byte count is a direct readout of what the tool thinks the thread
/// started at -- 2 bytes per iteration from the inherited `vl = 2`, or 0 from a
/// zeroed shadow. The `vsetvli t1, a5` after the loop is an `R_REGS` site, where
/// the same disagreement used to abort the run outright under the `vl_verify`
/// default.
///
/// Asserted on vCPU 1's own perf JSON, not the merged one, because the merged
/// totals of a multi-hart run are not reproducible run to run.
#[test]
fn clone_vl_inherit() {
    let run = TurboTest::new(
        "clone_vl_inherit",
        "test_bins/asm_validation/bins/clone_vl_inherit.riscv",
    )
    .run();

    let child = run
        .vcpu_perf(1)
        .expect("the guest clones a second thread, so vCPU 1 must have its own perf JSON");
    let roi = child
        .functions
        .iter()
        .find(|roi| &*roi.name == "loop")
        .expect("ROI 'loop' not found in vCPU 1's performance data");

    assert_eq!(
        roi.instruction_mix,
        [
            ("vsetvli", 100),
            ("vse8.v", 100),
            ("addi", 100),
            ("bne", 100)
        ]
        .iter()
        .map(|(m, n)| (m.to_string(), *n))
        .collect::<std::collections::BTreeMap<_, _>>(),
        "the child's loop body is one vsetvli + one vse8.v + the loop counter"
    );

    // The whole point: 2 bytes per iteration, from a vl the child was never
    // told and can only have inherited.
    assert_eq!(
        roi.store_bytes, 200,
        "the vse8.v runs at the inherited vl = 2, so 2 bytes per iteration -- 0 \
         here means the shadow was zeroed at vcpu init and `vsetvli zero, zero` \
         computed vl = 0"
    );
    assert_eq!(roi.vector_elements, 200);
    assert_eq!(roi.load_bytes, 0);
    assert_eq!(
        roi.vl_hist_e8_raw,
        vec![[2, 100]],
        "vl = 2 must be the only length seen in the child"
    );
}
