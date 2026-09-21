//! One kernel, every VLEN the RVV specification allows.
//!
//! `vadd_vlmax.S` is `vadd_simple.S` with its `vsetvli` AVL taken from `x0`, so
//! `vl` is VLMAX rather than a constant the source picked. At e8/m1 that makes
//! `vl == VLEN/8` exactly, which splits this guest's counters cleanly in two:
//!
//! * **fixed at every VLEN** -- 1000 iterations of `vadd.vv`/`addi`/`bnez`
//!   retire whatever VLEN is, because widening the registers does not change
//!   the program.
//! * **linear in VLEN** -- every element counter, and the VL the histogram
//!   records, scale by exactly `VLEN/8`.
//!
//! Asserting both halves at once is what makes the sweep worth running: a VLEN
//! that never reaches QEMU leaves the element counts flat, a VLEN applied at
//! decode but not at enrich (or vice versa) makes them disagree with `vlenb`,
//! and a VLMAX computation that mishandles a width makes exactly one point of
//! the sweep wrong while its neighbours pass.
//!
//! # Range
//!
//! The RVV spec puts VLEN in `[128, 65536]` bits, every power of two, and this
//! sweep covers all of them -- ten runs, from the ISA minimum to the maximum
//! QEMU's `RV_VLEN_MAX` also stops at. 65536 is the interesting end: it is one
//! past the top of a `u16`, so for as long as VLEN-in-bits was a `u16` it could
//! not even be asked for, and the tracer recorded it as a saturated 65535.
//!
//! Only a QEMU built for that range can run all of it: upstream 11.0 stops at
//! 1024, so on that QEMU the widths above it skip rather than fail (see
//! [`skip_unless_qemu_vlen`]).

use std::collections::BTreeMap;

use turbo::source::{VLEN_BITS_MAX, VLEN_BITS_MIN};
use turbo_tracer_shared::TurboMetadata;

use crate::harness::{skip_unless_qemu_vlen, TurboTest};

/// `vadd_vlmax.S`, built by `test_bins/asm_validation/Makefile`.
const VADD_VLMAX_BIN: &str = "test_bins/asm_validation/bins/vadd_vlmax.riscv";

/// The kernel's loop trip count, and hence the count of every opcode in it.
const ITERATIONS: u64 = 1000;

/// Record and process `vadd_vlmax` at `vlen_bits`, and assert every counter of
/// its `loop` function against the value VLEN implies.
///
/// The single parameterised body behind the sweep below: the expected metrics
/// are *computed* from `vlen_bits` rather than read from
/// `expected_metrics.json`, because a fixture file can only state one VLEN's
/// worth of them and the whole point here is how they move with VLEN.
fn vadd_vlmax_at_vlen(vlen_bits: u32) {
    // Bounded by the production constants, not by numbers repeated here: this
    // sweep claims to cover the whole legal range, and that claim is only worth
    // anything if it is the same range the runner enforces.
    assert!(
        vlen_bits.is_power_of_two() && (VLEN_BITS_MIN..=VLEN_BITS_MAX).contains(&vlen_bits),
        "VLEN must be a power of two in [{VLEN_BITS_MIN}, {VLEN_BITS_MAX}] bits, \
         got {vlen_bits}"
    );

    let name = format!("vlen_sweep_vadd_vlmax_{vlen_bits}");
    if skip_unless_qemu_vlen(&name, vlen_bits) {
        return;
    }
    let run = TurboTest::new(&name, VADD_VLMAX_BIN).vlen(vlen_bits).run();

    // QEMU silently clamps a `vlen=` its build cannot honour, and the plugin
    // reports back the `vlenb` CSR the guest actually saw. Check that first:
    // without it, a clamped run would be compared against the counters of a
    // VLEN it never ran at, and the failure would look like a decode bug.
    let recorded = TurboMetadata::read_from_workdir(&run.dir)
        .expect("the recording must carry turbo metadata");
    assert_eq!(
        recorded.vlen_bits, vlen_bits,
        "asked QEMU for vlen={vlen_bits} but the guest ran at vlen={}",
        recorded.vlen_bits
    );

    // vl = VLMAX = VLEN/8 at e8, m1 -- the one number the whole assertion set
    // is derived from.
    let vl = u64::from(vlen_bits) / 8;

    let pd = run.perf_data();
    let loop_roi = pd
        .functions
        .iter()
        .find(|roi| &*roi.name == "loop")
        .unwrap_or_else(|| {
            panic!(
                "function 'loop' not found at vlen={vlen_bits}; got {:?}",
                pd.functions.iter().map(|r| &r.name).collect::<Vec<_>>()
            )
        });
    eprintln!(
        "vlen={vlen_bits} (vl={vl}): {}",
        serde_json::to_string(&loop_roi).expect("ROI must serialize")
    );

    // Invariant under VLEN: the program is the same program.
    assert_eq!(
        loop_roi.instructions,
        3 * ITERATIONS,
        "vlen={vlen_bits}: retired instructions must not depend on VLEN"
    );
    assert_eq!(
        loop_roi.branches, ITERATIONS,
        "vlen={vlen_bits}: branch count must not depend on VLEN"
    );
    assert_eq!(
        loop_roi.vector_instructions, ITERATIONS,
        "vlen={vlen_bits}: one vadd.vv per iteration, at any VLEN"
    );
    let expected_mix: BTreeMap<String, u64> = ["vadd.vv", "addi", "bne"]
        .into_iter()
        .map(|mnemonic| (mnemonic.to_string(), ITERATIONS))
        .collect();
    assert_eq!(
        loop_roi.instruction_mix, expected_mix,
        "vlen={vlen_bits}: instruction mix must not depend on VLEN"
    );

    // Linear in VLEN: `vadd.vv` computes one element per `vl`.
    assert_eq!(
        loop_roi.vector_elements,
        ITERATIONS * vl,
        "vlen={vlen_bits}: {ITERATIONS} vadd.vv at vl={vl} must be {} elements",
        ITERATIONS * vl
    );

    // The VL the enricher attributed, at full resolution: exactly one bucket,
    // at VLMAX, for all 1000 executions -- and under e8/m1 specifically, so a
    // VLMAX derived at the wrong SEW or LMUL cannot land here by accident.
    let expected_hist = vec![[vl, ITERATIONS]];
    assert_eq!(
        loop_roi.vl_hist_e8_raw, expected_hist,
        "vlen={vlen_bits}: e8 VL histogram must be {ITERATIONS} samples at vl={vl}"
    );
    assert_eq!(
        loop_roi.vl_hist_e8_m1_raw, expected_hist,
        "vlen={vlen_bits}: e8/m1 VL histogram must be {ITERATIONS} samples at vl={vl}"
    );
    for (sew, hist) in [
        ("e16", &loop_roi.vl_hist_e16_raw),
        ("e32", &loop_roi.vl_hist_e32_raw),
        ("e64", &loop_roi.vl_hist_e64_raw),
    ] {
        assert!(
            hist.is_empty(),
            "vlen={vlen_bits}: kernel runs entirely at e8, but {sew} recorded {hist:?}"
        );
    }

    // No memory and no floating point anywhere in the kernel, at any VLEN: a
    // VLEN-scaled counter appearing here would mean an integer `vadd.vv` was
    // classified as something it is not.
    for (what, value) in [
        ("load_bytes", loop_roi.load_bytes),
        ("store_bytes", loop_roi.store_bytes),
        ("vector_load_insns", loop_roi.vector_load_instructions),
        ("vector_load_elems", loop_roi.vector_load_elements),
        ("float_instructions", loop_roi.float_instructions),
        ("float_elements", loop_roi.float_elements),
        ("fmacc_instructions", loop_roi.fmacc_instructions),
        ("fmacc_elements", loop_roi.fmacc_elements),
    ] {
        assert_eq!(value, 0, "vlen={vlen_bits}: {what} must be 0, got {value}");
    }
}

/// Declare one test per VLEN, named after the width it runs at.
///
/// A table rather than a loop inside one test: each width is an independent
/// QEMU run, so as separate tests they run in parallel under nextest and a
/// single bad width is named by the failure instead of hiding the widths after
/// it behind an early panic.
macro_rules! vlen_sweep {
    ($($name:ident => $vlen:expr;)*) => { $(
        #[test]
        fn $name() {
            vadd_vlmax_at_vlen($vlen);
        }
    )* };
}

vlen_sweep! {
    vlen_128   => 128;
    vlen_256   => 256;
    vlen_512   => 512;
    vlen_1024  => 1024;
    vlen_2048  => 2048;
    vlen_4096  => 4096;
    vlen_8192  => 8192;
    vlen_16384 => 16384;
    vlen_32768 => 32768;
    vlen_65536 => 65536;
}
