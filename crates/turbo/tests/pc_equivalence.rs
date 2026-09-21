//! PC equivalence: the trace decode must reproduce, instruction for
//! instruction, what QEMU's execlog plugin says was executed.
//!
//! `turbo record --reference-traces` runs the guest with the turbo tracer plugin AND
//! the execlog plugin, so `execlog_out.txt` lands beside the trace spool.
//! `turbo process --dump-pcs` then decodes that spool through the production
//! pipeline and writes `turbo_trace_pcs.txt`. Any disagreement is a decode bug.

use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

use crate::harness::{skip_if_missing, skip_unless_execlog, TurboTest};

/// Extracts PC values from `execlog_out.txt`, whose lines are
/// `cycle, PC, encoding, instruction` with the PC as hex in column two.
fn extract_pc_from_execlog<P: AsRef<Path>>(filepath: P) -> std::io::Result<Vec<u64>> {
    read_pcs(filepath, |line| line.split(',').nth(1))
}

/// Extracts PC values from `turbo_trace_pcs.txt`: one hex PC per line.
fn extract_pc_from_trace<P: AsRef<Path>>(filepath: P) -> std::io::Result<Vec<u64>> {
    read_pcs(filepath, |line| Some(line))
}

/// Reads a PC per line, `field` picking the hex column out of each line.
fn read_pcs<P: AsRef<Path>>(
    filepath: P,
    field: fn(&str) -> Option<&str>,
) -> std::io::Result<Vec<u64>> {
    let mut pcs = Vec::new();
    for line in BufReader::new(File::open(filepath)?).lines() {
        let line = line?;
        let Some(pc_hex) = field(line.trim()).map(str::trim) else {
            continue;
        };
        if let Ok(pc) = u64::from_str_radix(pc_hex.trim_start_matches("0x"), 16) {
            pcs.push(pc);
        }
    }
    Ok(pcs)
}

fn do_comparison(test_name: &str, bin: &str, bin_args: &[&str]) {
    if skip_unless_execlog(test_name) || skip_if_missing(bin) {
        return;
    }
    let run = TurboTest::new(test_name, bin)
        .args(bin_args)
        .reference_traces()
        .record();

    // Decode the recording through the production pipeline: same memory-map
    // discovery (shared libs, dynamic linker, VDSO) and same decoder the real
    // `process` path uses, which is exactly what the linker-flavour cases below
    // are testing.
    let trace_pcs_file = run.dump_pcs();

    let mut execlog_pcs = extract_pc_from_execlog(run.dir.join("execlog_out.txt"))
        .expect("Failed to extract PCs from execlog_out.txt");
    // Sorted to match `--dump-pcs`, which sorts because multi-hart streams
    // interleave nondeterministically: only the multiset is well defined.
    execlog_pcs.sort_unstable();

    let trace_pcs = extract_pc_from_trace(&trace_pcs_file)
        .expect("Failed to extract PCs from turbo_trace_pcs.txt");

    let mismatches: Vec<(usize, u64, u64)> = execlog_pcs
        .iter()
        .zip(&trace_pcs)
        .enumerate()
        .filter(|(_, (a, b))| a != b)
        .map(|(i, (&a, &b))| (i, a, b))
        .collect();

    for (i, execlog_pc, trace_pc) in mismatches.iter().take(10) {
        println!("mismatch at {i}: execlog=0x{execlog_pc:08x} trace=0x{trace_pc:08x}");
    }
    if mismatches.len() > 10 {
        println!("... and {} more mismatches", mismatches.len() - 10);
    }

    assert!(
        mismatches.is_empty() && execlog_pcs.len() == trace_pcs.len(),
        "execlog and trace do not match! {} mismatches, lengths: execlog={}, trace={}",
        mismatches.len(),
        execlog_pcs.len(),
        trace_pcs.len()
    );
}

turbo_tests!(do_comparison {
    loop_10x_basic_comparison => "test_bins/basic_branch_asm/loop_10x.riscv";
    branch_minimal_comparison => "test_bins/basic_branch_asm/branch_minimal.riscv";
    branch_call_comparison => "test_bins/basic_branch_asm/branch_call.riscv";
    branch_complex_comparison => "test_bins/basic_branch_asm/branch_complex.riscv";
    thirty_one_branches_jalr_comparison =>
        "test_bins/basic_branch_asm/thirty_one_branches_jalr.riscv";
    simple_31_branches_jalr_comparison =>
        "test_bins/basic_branch_asm/simple_31_branches_jalr.riscv";
    four_branches_jalr_comparison => "test_bins/basic_branch_asm/four_branches_jalr.riscv";
    unprocessed_branches_test_comparison =>
        "test_bins/basic_branch_asm/unprocessed_branches_test.riscv";
    unprocessed_branches_ret_test_comparison =>
        "test_bins/basic_branch_asm/unprocessed_branches_ret_test.riscv";
    branches_before_ecall_test_comparison =>
        "test_bins/basic_branch_asm/branches_before_ecall_test.riscv";
    three_branches_jalr_test_comparison =>
        "test_bins/basic_branch_asm/three_branches_jalr_test.riscv";
    simple_loop_comparison => "test_bins/simple_loop/loop.riscv";

    rust_asm_comparison_musl =>
        "test_bins/rust_asm_bins/target/riscv64gc-unknown-linux-musl/release/vector_asm",
        &["--simple"];
    flash_comparison_slow => "test_bins/flash/flash.riscv", &["32", "32"];

    /// Requires the QEMU execlog plugin race-condition patches that make its
    /// output deterministic (and correct on multi-threaded workloads). CI does
    /// not build & upload those to Artifactory, so run this locally against a
    /// locally built QEMU.
    #[ignore = "requires-qemu-execlog-plugin-CI-upgrade"]
    pthreads_comparison => "test_bins/simple_loop/pthreads.riscv";
});

// Linker-flavour binaries: the same program built static/dynamic with/without
// vDSO (gettimeofday) and against a custom shared library. See
// test_bins/linker_flavours/Makefile. The dynamic flavours find their .so via
// an $ORIGIN rpath, so no LD_LIBRARY_PATH is needed.
//
// Each covers an ELF linking mode the memory-map discovery in turbo_tracer has
// to handle, and fails the comparison if the corresponding module (main binary
// at its real load bias, dynamic linker, run-time-mapped .so) is not resolved
// -- so these are the decode-coverage tests for those modes.
turbo_tests!(do_comparison {
    linker_flavour_static_plain => "test_bins/linker_flavours/static_plain.riscv";
    linker_flavour_static_gtod => "test_bins/linker_flavours/static_gtod.riscv";
    linker_flavour_dyn_gtod => "test_bins/linker_flavours/dyn_gtod.riscv";
    linker_flavour_dyn_addso => "test_bins/linker_flavours/dyn_addso.riscv";
    linker_flavour_dyn_addso_gtod => "test_bins/linker_flavours/dyn_addso_gtod.riscv";

    /// static-PIE: ET_DYN with NO interpreter, loaded at a nonzero bias.
    /// Exercises recovering the main-binary load base (start_code vs link-time
    /// vaddr) with no dynamic linker or NEEDED libs in the picture -- the
    /// sharpest test of PIE-bias handling for the main module.
    linker_flavour_static_pie => "test_bins/linker_flavours/static_pie.riscv";

    /// Non-PIE dynamic: ET_EXEC main at fixed absolute addresses (zero load
    /// bias) while ld.so, libc and the vDSO are mapped elsewhere. The only
    /// dynamic flavour whose main binary is not position-independent.
    linker_flavour_dyn_nopie => "test_bins/linker_flavours/dyn_nopie.riscv";

    /// `-z separate-code`: split R / R-E / R-W PT_LOAD segments with alignment
    /// gaps, so the executable text is not at file offset 0 of the mapping.
    /// Stresses the main-binary code-range/offset derivation.
    linker_flavour_dyn_sepcode => "test_bins/linker_flavours/dyn_sepcode.riscv";

    /// dlopen: libadd.so is mapped mid-run by an explicit dlopen() in main(),
    /// not by the loader at startup. Exercises the plugin's openat/mmap syscall
    /// discovery on the live path -- a run-time load rather than a loader-init
    /// NEEDED map.
    linker_flavour_dyn_dlopen => "test_bins/linker_flavours/dyn_dlopen.riscv";
});
