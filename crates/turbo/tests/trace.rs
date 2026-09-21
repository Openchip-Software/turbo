//! E-Trace integration tests that don't require ROI markers.

use turbo::{common::PerformanceData, processing::RoiInfo};

use crate::harness::{skip_unless_fixture_cpu, smoke, FixtureCpu, TurboTest};

// Guests whose only assertion is that record + decode + enrich completes. Most
// are encoder/decoder regression cases: the binary is the test.
turbo_tests!(smoke {
    loop_10x_trace => "test_bins/basic_branch_asm/loop_10x.riscv";
    vld_vst_trace => "test_bins/basic_branch_asm/vld_vst.riscv";
    branch_minimal_trace => "test_bins/basic_branch_asm/branch_minimal.riscv";
    branch_call_trace => "test_bins/basic_branch_asm/branch_call.riscv";
    branch_complex_trace => "test_bins/basic_branch_asm/branch_complex.riscv";

    /// Reproduces the `UnprocessedBranches` error seen in flash_attn: the
    /// encoder emits a packet with branches + landing address, but the decoder
    /// can't consume branches that occurred before the jalr when it is
    /// positioned at the landing address after it.
    four_branches_jalr_trace => "test_bins/basic_branch_asm/four_branches_jalr.riscv";

    /// Exactly 31 branches followed by jalr -- the `UnexpectedUninferableDiscon`
    /// fix. Before it, the encoder emitted `Branch { count: 31, address: None }`
    /// and then hit the jalr; now it flushes the buffered branches before the
    /// jalr so the packet carries `address: Some(landing)`.
    thirty_one_branches_jalr_trace => "test_bins/basic_branch_asm/thirty_one_branches_jalr.riscv";

    /// The same 31-branch case with the target address loaded up front, so the
    /// jalr immediately follows the 31st branch with nothing in between.
    simple_31_branches_jalr_trace => "test_bins/basic_branch_asm/simple_31_branches_jalr.riscv";

    /// Branches followed by JALR -- previously `UnprocessedBranches(2)`.
    unprocessed_branches_jalr_test => "test_bins/basic_branch_asm/unprocessed_branches_test.riscv";

    /// Branches followed by RET -- previously `UnprocessedBranches(2)`.
    unprocessed_branches_ret_test => "test_bins/basic_branch_asm/unprocessed_branches_ret_test.riscv";

    rust_asm_musl_static =>
        "test_bins/rust_asm_bins/target/riscv64gc-unknown-linux-musl/release/vector_asm",
        &["--simple"];
    rvv_stream_hvh_slow_trace =>
        "test_bins/rust_asm_bins/target/riscv64gc-unknown-linux-musl/release/stream_hvh",
        &["--lmul=1"];
    flash_trace => "test_bins/flash/flash.riscv", &["32", "32"];
    flash_full_trace => "test_bins/flash/flash_full.riscv", &["64", "32"];
});

/// Exact whole-`PerformanceData` counters for a hand-written vector kernel.
///
/// The program is `_start` (scalar setup) falling into `loop` (the vector
/// body), with an ROI region wrapping exactly the loop -- so three of the four
/// expected records are the loop's numbers, and only the deltas are spelled
/// out here.
#[test]
fn internal_flops_test() {
    if skip_unless_fixture_cpu("internal_flops_test", FixtureCpu::Vlen) {
        return;
    }
    let mut lp = RoiInfo::new("loop");
    lp.instructions = 1255;
    lp.float_instructions = 500;
    lp.float_elements = 192_000;
    lp.fmacc_instructions = 250;
    lp.fmacc_elements = 64_000;
    lp.vector_instructions = 750;
    lp.vector_elements = 192_000;
    lp.vector_load_instructions = 250;
    lp.vector_load_elements = 64_000;
    lp.load_bytes = 512_000;
    lp.branches = 250;

    // Setup before the loop: one scalar float instruction, no vector work.
    let mut start = RoiInfo::new("_start");
    start.instructions = 9;
    start.float_instructions = 1;
    start.float_elements = 1;

    // The `flops_test` region spans the loop's instructions, but its float
    // counters also carry `_start`'s single scalar float op.
    let mut region = lp.clone();
    region.name = "flops_test".into();
    region.float_instructions += start.float_instructions;
    region.float_elements += start.float_elements;

    // Global tracks everything, so it is the region plus `_start`'s
    // instructions -- the only counters the region does not already include.
    let mut global = region.clone();
    global.name = "Global".into();
    global.instructions += start.instructions;

    let expected = PerformanceData::with_rois_and_funcs(vec![region, global], vec![start, lp]);

    TurboTest::new("flops_test", "test_bins/basic_branch_asm/flops_test.riscv")
        .run()
        .perf_data()
        .assert_equals(&expected, "flops_test");
}
