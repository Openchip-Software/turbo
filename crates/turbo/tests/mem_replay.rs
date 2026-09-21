//! End-to-end tests for the captured vector memory footprint.
//!
//! Each one records an `asm_validation/memory` kernel whose addresses are known
//! from its `.S` source, replays the recorded trace through
//! [`replay_trace_file`], and asserts the cache lines that come out.
//!
//! These are the tests that connect the two halves of address capture: the
//! plugin's runtime capture and the offline expansion. A mismatch between them
//! (a mode classified differently on the two sides, a group attributed to the
//! wrong instruction, an index vector read at the wrong width) shows up here as
//! a wrong address, and nowhere else.

use std::collections::{BTreeMap, BTreeSet};

use turbo::processing::{replay_trace_file, MemAccess};
use turbo_tracer_shared::vmem::CACHE_LINE_BYTES;

use crate::harness::{skip_unless_fixture_cpu, FixtureCpu, TestRun, TurboTest};

/// Record one `asm_validation` kernel and replay its vector memory addresses.
///
/// The output directory is named separately from the kernel so these runs never
/// share one with the `asm_validation` test of the same guest -- both wipe it.
fn replay(kernel: &str) -> (TestRun, Vec<MemAccess>) {
    let run = TurboTest::new(
        &format!("mem_replay_{kernel}"),
        format!("test_bins/asm_validation/bins/{kernel}.riscv"),
    )
    .run();
    let vlenb = u64::from(
        turbo_tracer_shared::TurboMetadata::read_from_workdir(&run.dir)
            .expect("the run must have written its metadata")
            .vlen_bits,
    ) / 8;
    let accesses = replay_trace_file(&run.trace_log(0), &run.bin, vlenb)
        .expect("the recorded trace must replay");
    (run, accesses)
}

/// The distinct lines each PC touched, and how many accesses it contributed.
fn by_pc(accesses: &[MemAccess]) -> BTreeMap<u64, (BTreeSet<u64>, usize, bool)> {
    let mut out: BTreeMap<u64, (BTreeSet<u64>, usize, bool)> = BTreeMap::new();
    for a in accesses {
        let e = out
            .entry(a.pc)
            .or_insert_with(|| (BTreeSet::new(), 0, a.is_store));
        e.0.insert(a.line);
        e.1 += 1;
        assert_eq!(e.2, a.is_store, "PC {:#x} is both a load and a store", a.pc);
    }
    out
}

/// Every line in `lines` must be exactly the lines a `len`-byte run from the
/// lowest one covers -- i.e. the footprint is contiguous and of the expected
/// size. This is what turns "some addresses came out" into a real check of the
/// byte count, independently of the statically-computed `load_bytes`.
fn assert_contiguous_span(lines: &BTreeSet<u64>, len: u64, what: &str) {
    let base = *lines.iter().next().expect("at least one line");
    let expected: BTreeSet<u64> = (0..len.div_ceil(CACHE_LINE_BYTES))
        .map(|i| base + i * CACHE_LINE_BYTES)
        .collect();
    assert_eq!(
        lines, &expected,
        "{what}: expected a contiguous {len}-byte footprint from {base:#x}"
    );
}

/// A plain unit-stride load and store: one contiguous `vl * 8` byte run each,
/// from a base that never moves, on every one of the 100 iterations.
#[test]
fn unit_stride_footprint_is_the_two_buffers() {
    if skip_unless_fixture_cpu("unit_stride_footprint_is_the_two_buffers", FixtureCpu::Vlen) {
        return;
    }
    let (run, accesses) = replay("vload_store_simple");

    let pcs = by_pc(&accesses);
    assert_eq!(pcs.len(), 2, "one vle64.v and one vse64.v: {pcs:#x?}");

    // `vl` is 256 at e64 (VLMAX at this VLEN), so each op moves 2048 bytes --
    // the same 32 lines every iteration, since the base register is loop-invariant.
    let (load_pc, (load_lines, load_count, _)) = pcs.iter().find(|(_, v)| !v.2).expect("a load");
    let (_, (store_lines, store_count, _)) = pcs.iter().find(|(_, v)| v.2).expect("a store");

    assert_contiguous_span(load_lines, 2048, "vle64.v");
    assert_contiguous_span(store_lines, 2048, "vse64.v");
    assert_eq!(
        (*load_count, *store_count),
        (100 * 32, 100 * 32),
        "100 iterations x 32 lines per op"
    );
    // Deliberately NOT asserting the two footprints are disjoint: `source_data`
    // is 512 bytes and this kernel reads 2048, so the load genuinely runs off
    // the end of its buffer and into `dest_data`. That over-read is the guest's
    // behaviour, and a footprint trace that hid it would be the wrong trace.
    assert_ne!(
        load_lines.iter().next(),
        store_lines.iter().next(),
        "the load and the store still start at different objects"
    );

    // The statically counted byte total and the replayed footprint are two
    // independent derivations of the same traffic, so they must agree.
    let pd = run.perf_data();
    let roi = pd
        .functions
        .iter()
        .find(|r| &*r.name == "loop")
        .expect("ROI 'loop'");
    assert_eq!(roi.load_bytes, 100 * 2048);
    assert_eq!(roi.store_bytes, 100 * 2048);
    let _ = load_pc;
}

/// Strided and indexed segment ops in one kernel: the strided ones walk 64
/// segments 32 bytes apart (so their footprint is contiguous after all, 2048
/// bytes of it), and the indexed ones gather at `i * 16` for 1024 bytes.
#[test]
fn strided_and_indexed_footprints() {
    if skip_unless_fixture_cpu("strided_and_indexed_footprints", FixtureCpu::Vlen) {
        return;
    }
    let (_run, accesses) = replay("vseg_strided_indexed");

    let pcs = by_pc(&accesses);
    assert_eq!(pcs.len(), 6, "two strided and four indexed ops: {pcs:#x?}");

    let mut spans: Vec<u64> = Vec::new();
    for (pc, (lines, count, _)) in &pcs {
        // Each op's line set repeats identically on every iteration: the base,
        // the stride and the index vector are all loop-invariant here.
        assert_eq!(
            *count,
            100 * lines.len(),
            "PC {pc:#x} must touch the same lines every iteration"
        );
        let base = *lines.iter().next().unwrap();
        let span = *lines.iter().next_back().unwrap() + CACHE_LINE_BYTES - base;
        spans.push(span);
    }
    spans.sort_unstable();
    assert_eq!(
        spans,
        vec![1024, 1024, 1024, 1024, 2048, 2048],
        "four indexed ops over 64 x 16-byte segments at offsets i*16 (1024 B), and \
         two strided ops over 64 x 16-byte segments 32 B apart (2048 B)"
    );

    // Loads read source_data, stores write dest_data, and the two never share a
    // line -- so no address has been attributed to the wrong instruction.
    let loads: BTreeSet<u64> = accesses
        .iter()
        .filter(|a| !a.is_store)
        .map(|a| a.line)
        .collect();
    let stores: BTreeSet<u64> = accesses
        .iter()
        .filter(|a| a.is_store)
        .map(|a| a.line)
        .collect();
    assert!(loads.is_disjoint(&stores));
}

/// Indexed ops whose index width differs from `SEW`: the index vector is all
/// zeros, so every element of every op lands on the base's own line, and the
/// data width (which decides whether a segment spills into the next line) comes
/// from `SEW` rather than the instruction's `width` field.
#[test]
fn indexed_eew_mismatch_footprint() {
    let (_run, accesses) = replay("vindexed_eew_mismatch");

    let pcs = by_pc(&accesses);
    assert_eq!(pcs.len(), 5, "three loads and two stores: {pcs:#x?}");
    assert_eq!(accesses.len(), 500, "5 ops x 100 iterations x 1 line each");

    for (pc, (lines, count, _)) in &pcs {
        assert_eq!(
            lines.len(),
            1,
            "PC {pc:#x}: all indices are zero, so every element reaches one line"
        );
        assert_eq!(*count, 100);
    }

    let loads: BTreeSet<u64> = accesses
        .iter()
        .filter(|a| !a.is_store)
        .map(|a| a.line)
        .collect();
    let stores: BTreeSet<u64> = accesses
        .iter()
        .filter(|a| a.is_store)
        .map(|a| a.line)
        .collect();
    assert_eq!(loads.len(), 1, "all three loads gather from source_data");
    assert_eq!(stores.len(), 1, "both stores scatter into dest_data");
    assert!(loads.is_disjoint(&stores));
}

/// Indexed loads at `vl` below VLMAX: the expansion must stop at `vl`, and a
/// line several elements share must be reported once.
///
/// The other indexed fixtures index with an all-zero vector, which collapses to
/// one line whether or not the expansion honours `vl` -- so neither property is
/// actually pinned by them. Here `vl` is 20 against a VLMAX of 256, one op
/// spreads its 20 elements over 20 distinct lines and the other folds them into
/// a single one.
#[test]
fn indexed_footprint_stops_at_vl_and_dedupes_lines() {
    if skip_unless_fixture_cpu(
        "indexed_footprint_stops_at_vl_and_dedupes_lines",
        FixtureCpu::Vlen,
    ) {
        return;
    }
    let (run, accesses) = replay("vindexed_vl_dedup");

    let pcs = by_pc(&accesses);
    assert_eq!(pcs.len(), 2, "one vluxei64.v and one vloxei64.v: {pcs:#x?}");

    let mut line_counts: Vec<usize> = pcs.values().map(|(lines, ..)| lines.len()).collect();
    line_counts.sort_unstable();
    assert_eq!(
        line_counts,
        vec![1, 20],
        "the i*64 op reaches one line per active element -- 20, not VLMAX and not \
         VLEN/SEW -- and the (i*8)&63 op folds all 20 into the line its base sits in"
    );

    for (pc, (lines, count, is_store)) in &pcs {
        assert!(!is_store, "PC {pc:#x} is a load");
        assert_eq!(
            *count,
            100 * lines.len(),
            "PC {pc:#x} must touch the same lines on each of the 100 iterations"
        );
    }

    // The byte total is 20 elements x 8 bytes for *both* ops, while their line
    // counts differ by 20x: deduplicating lines must not have moved the byte
    // count, and bounding at `vl` must have moved it.
    let pd = run.perf_data();
    let roi = pd
        .functions
        .iter()
        .find(|r| &*r.name == "loop")
        .expect("ROI 'loop'");
    assert_eq!(roi.load_bytes, 100 * 2 * 20 * 8);
    assert_eq!(roi.store_bytes, 0);
}

/// A fault-only-first load that really was trimmed touches only the elements it
/// actually read, and the op behind it runs at the trimmed `vl` too.
///
/// This is the footprint half of `vl*ff` capture. `vleff_trim`'s `vle8ff.v` asks
/// for 16 bytes at 8 bytes before a page hole, so it reads 8 and writes
/// `vl = 8`; the two `vse8.v` behind it then store 8 each. All three are one
/// line each here (the ff's 8 bytes end at a page boundary, the stores' start at
/// one), so the assertion that carries the weight is the byte count: before the
/// trim was captured, the replay derived every footprint from the `vsetivli`'s
/// `vl = 16` and claimed twice the bytes, including 8 bytes inside the unmapped
/// page.
#[test]
fn fault_only_first_footprint_stops_at_the_trimmed_vl() {
    let (run, accesses) = replay("vleff_trim");

    let pcs = by_pc(&accesses);
    assert_eq!(pcs.len(), 3, "one vle8ff.v and two vse8.v: {pcs:#x?}");

    for (pc, (lines, count, _)) in &pcs {
        assert_eq!(
            lines.len(),
            1,
            "PC {pc:#x} moves 8 bytes that do not straddle a line: {lines:#x?}"
        );
        assert_eq!(*count, 100, "PC {pc:#x} runs once per iteration");
    }

    // The ff's own line is the one *below* the hole: its 8 bytes are the last of
    // the mapped page, so the line it touches is the page's last, and nothing in
    // the unmapped page is ever claimed.
    let (ff_pc, (ff_lines, ..)) = pcs
        .iter()
        .find(|(_, (_, _, is_store))| !is_store)
        .expect("the vle8ff.v is a load");
    let ff_line = *ff_lines.iter().next().unwrap();
    assert_eq!(
        ff_line % 4096,
        4096 - CACHE_LINE_BYTES,
        "the vle8ff.v at PC {ff_pc:#x} must touch the last line of the mapped page \
         ({ff_line:#x}), never the hole after it"
    );

    let pd = run.perf_data();
    let roi = pd
        .functions
        .iter()
        .find(|r| &*r.name == "loop")
        .expect("ROI 'loop'");
    assert_eq!(
        roi.load_bytes,
        100 * 8,
        "8 bytes read, not the 16 requested"
    );
    assert_eq!(
        roi.store_bytes,
        100 * 2 * 8,
        "both vse8.v inherit the trimmed vl -- the second through the shadow AVL of \
         the `vsetvli zero, zero` in front of it"
    );
}

/// Whole-register load/store: their footprint is `nf * VLENB` bytes and ignores
/// both `vl` and `SEW`, which is the one mode where taking the element width
/// from the encoding would give the wrong answer.
#[test]
fn whole_register_footprint_ignores_vl_and_sew() {
    if skip_unless_fixture_cpu(
        "whole_register_footprint_ignores_vl_and_sew",
        FixtureCpu::Vlen,
    ) {
        return;
    }
    let (_run, accesses) = replay("full_reg_ops");

    // The kernel is straight-line: vl2re64 (2 registers), vs1r (1), vle64 and
    // vse32 (one VLEN each), all over the same buffer, once.
    let spans: Vec<u64> = by_pc(&accesses)
        .values()
        .map(|(lines, count, _)| {
            assert_eq!(*count, lines.len(), "each op runs exactly once");
            (lines.len() as u64) * CACHE_LINE_BYTES
        })
        .collect();
    let mut spans = spans;
    spans.sort_unstable();
    assert_eq!(
        spans,
        vec![2048, 2048, 2048, 4096],
        "vl2re64.v moves 2 x VLENB = 4096 B; vs1r.v, vle64.v and vse32.v move one \
         VLEN (2048 B) each"
    );
}

/// Every element width, in unit-stride, strided and fault-only-first form, plus
/// the mask pair -- the sweep that catches a per-width footprint error.
#[test]
fn every_element_width_and_mode_replays() {
    if skip_unless_fixture_cpu("every_element_width_and_mode_replays", FixtureCpu::Vlen) {
        return;
    }
    let (_run, accesses) = replay("vload_widths");

    let pcs = by_pc(&accesses);
    // 4 widths x (unit load, unit store, strided load, strided store, ff load)
    // plus vlm.v/vsm.v.
    assert_eq!(pcs.len(), 22, "{pcs:#x?}");

    let mut spans: Vec<u64> = pcs
        .values()
        .map(|(lines, count, _)| {
            assert_eq!(*count, 100 * lines.len(), "loop-invariant bases");
            *lines.iter().next_back().unwrap() + CACHE_LINE_BYTES - *lines.iter().next().unwrap()
        })
        .collect();
    spans.sort_unstable();
    assert_eq!(
        spans,
        vec![
            // 64 B: the vlm.v/vsm.v pair (8 bytes each) and the three e8 ops
            // (64 x 1 byte), all inside one line.
            64, 64, 64, 64, 64, //
            // 128 B: the three e16 ops, 64 x 2 bytes.
            128, 128, 128, //
            // 256 B: the three e32 ops, 64 x 4 bytes.
            256, 256, 256, //
            // 512 B: the three e64 ops (64 x 8 bytes) and all eight strided ops,
            // which walk 64 elements 8 bytes apart whatever their EEW.
            512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512,
        ],
        "each op's footprint is its vl x EEW run, or its 64 x 8-byte stride walk"
    );
}

/// A trace with no vector memory op at all must replay to nothing -- the
/// producer must not be emitting groups for instructions the consumer does not
/// expect, which would desynchronise every later group.
#[test]
fn a_kernel_without_memory_ops_replays_empty() {
    let (_run, accesses) = replay("vfmacc_simple");
    assert!(
        accesses.is_empty(),
        "vfmacc_simple has no vector load or store: {accesses:#x?}"
    );
}
