//! Per-PC execution profile for the source & assembly annotation view.
//!
//! Where [`StatsAccumulator`](super::stats_accumulator::StatsAccumulator) keeps
//! *scope-level* aggregates ([`RoiInfo`](crate::common::RoiInfo)), this
//! module keeps a *per-instruction-address* breakdown: for every unique PC that
//! committed, it records the disassembly, how many times it executed, its
//! instruction class, and (for vector ops) the vector-length range and element
//! count.
//!
//! The store is bounded by the number of unique executed PCs (i.e. by binary
//! size, not trace length), and is opt-in — off unless annotation output is
//! requested.
//!
//! # Correctness invariant
//!
//! The per-PC classification uses the *same* [`riscv_isa`] predicates that
//! [`RoiInfo::update_stats`](crate::common::RoiInfo::update_stats) uses,
//! so the sum of per-PC contributions reconciles with the aggregate counters.
//! In particular `Σ exec_count == RoiInfo.instructions` and
//! `Σ vector_elems == RoiInfo.vector_elements` for the same instruction stream.

use crate::common::Lmul;
use rustc_hash::FxHashMap;
use std::collections::BTreeMap;

/// Instruction-class bit flags.
///
/// A single instruction can belong to several classes at once (e.g. a vector
/// floating-point fused-multiply-add is `VECTOR | FLOAT | FMACC`).
pub mod class {
    /// Control transfer: conditional branch (`beq`, `blt`, …), jump
    /// (`j`/`jal`/`jalr`), call, return, or return-from-trap. All are shown in
    /// the same "control flow" colour.
    pub const BRANCH: u8 = 1 << 0;
    /// Any vector (RVV) instruction.
    pub const VECTOR: u8 = 1 << 1;
    /// Scalar or vector floating-point instruction.
    pub const FLOAT: u8 = 1 << 2;
    /// Fused multiply-accumulate (integer or float).
    pub const FMACC: u8 = 1 << 3;
    /// Load (scalar or vector).
    pub const LOAD: u8 = 1 << 4;
    /// Store (scalar or vector).
    pub const STORE: u8 = 1 << 5;
}

/// Classify an instruction into the [`class`] bitset using the same predicates
/// the aggregate counters use, so the two can never disagree.
#[inline]
pub fn classify(ins: &riscv_isa::Instruction) -> u8 {
    let mut f = 0u8;
    if is_control_flow(ins) {
        f |= class::BRANCH;
    }
    if ins.vector() {
        f |= class::VECTOR;
    }
    if ins.float() {
        f |= class::FLOAT;
    }
    if ins.fma() {
        f |= class::FMACC;
    }
    if ins.load() {
        f |= class::LOAD;
    }
    if ins.store() {
        f |= class::STORE;
    }
    f
}

/// Render a class bitset as human-readable tags (e.g. `"vector,float,fmacc"`).
pub fn class_tags(c: u8) -> String {
    let mut tags: Vec<&str> = Vec::new();
    if c & class::LOAD != 0 {
        tags.push("load");
    }
    if c & class::STORE != 0 {
        tags.push("store");
    }
    if c & class::VECTOR != 0 {
        tags.push("vector");
    }
    if c & class::FLOAT != 0 {
        tags.push("float");
    }
    if c & class::FMACC != 0 {
        tags.push("fmacc");
    }
    if c & class::BRANCH != 0 {
        tags.push("branch");
    }
    if tags.is_empty() {
        tags.push("scalar");
    }
    tags.join(",")
}

/// One executed instruction address and its accumulated statistics.
///
/// `Default` is derived so constructors and test fixtures can name only the
/// fields they care about and let `..Default::default()` cover the rest. Adding
/// a counter here is then a one-line change, not an edit to every literal.
/// Note that the zero value of `vl_min` is NOT its neutral value — accumulating
/// a minimum starts from `u32::MAX` (see [`PcRecord::unexecuted`]).
#[derive(Clone, Debug, Default)]
pub struct PcRecord {
    /// Program counter (instruction address).
    pub pc: u64,
    /// Disassembly text (`ins.to_string()`), captured once.
    pub disasm: String,
    /// Raw little-endian encoding (2 or 4 bytes packed into a u32).
    pub bytes: u32,
    /// Number of times this PC committed.
    pub exec_count: u64,
    /// Instruction-class bitset (see [`class`]).
    pub class: u8,
    /// Sum of VL over executions (matches `RoiInfo::vector_elements` semantics).
    pub vector_elems: u64,
    /// Smallest VL seen (only meaningful for vector instructions).
    pub vl_min: u32,
    /// Largest VL seen (only meaningful for vector instructions).
    pub vl_max: u32,
    /// Last-seen SEW encoding (0=e8, 1=e16, 2=e32, 3=e64); usually stable per PC.
    pub sew: Option<u8>,
    /// Last-seen LMUL.
    pub lmul: Option<Lmul>,
    /// Per-VL execution histogram for this instruction: VL value → number of
    /// executions at that vector length. Empty for scalar instructions. This
    /// is the full distribution (e.g. a strip-mined loop body executes at
    /// VLMAX for the bulk and a shorter VL on the tail).
    pub vl_hist: FxHashMap<u32, u64>,
    /// Index of the resolving function (from the ELF symbol/DWARF table).
    pub func_index: Option<usize>,
    /// Absolute target address of this instruction if it is a
    /// (conditional or inferable-jump) control transfer, for arrow rendering.
    pub branch_target: Option<u64>,
    /// Modelled cache behaviour of this instruction's memory accesses, summed
    /// over its executions. All-zero unless the run modelled a cache, and for
    /// every instruction that is not a vector load or store.
    pub cache: crate::common::CacheCounts,
}

impl PcRecord {
    /// Build a record for an instruction address that has not (yet) been seen
    /// to execute — `exec_count` starts at 0. Used both as the seed for the
    /// first live execution ([`PcProfile::record`]) and to statically fill in
    /// the full ELF-symbol PC range for the annotate report, so a function's
    /// disassembly isn't limited to PCs the trace happened to visit.
    pub fn unexecuted(
        pc: u64,
        ins: &riscv_isa::Instruction,
        ins_bytes: &[u8],
        func_index: Option<usize>,
    ) -> Self {
        let mut arr = [0u8; 4];
        for (i, b) in ins_bytes.iter().take(4).enumerate() {
            arr[i] = *b;
        }
        PcRecord {
            pc,
            disasm: format!("{ins}"),
            bytes: u32::from_le_bytes(arr),
            class: classify(ins),
            // Accumulating a minimum has to start above every possible sample.
            vl_min: u32::MAX,
            func_index,
            branch_target: branch_target_of(ins, pc),
            ..Default::default()
        }
    }
}

/// Whether `ins` is any control transfer (for the "branch"/control-flow class):
/// conditional branch, direct/indirect jump (`j`/`jal`/`jalr`), call, return,
/// or return-from-trap.
///
/// `riscv_isa`'s [`branch()`](riscv_isa::Instruction::branch) is "could branch
/// or jump", i.e. every `B*`, `JAL` and `JALR`; the trap returns are the only
/// control transfers it does not name. Together they are exactly the set of
/// discontinuities a trace decoder distinguishes, bar the traps handled by
/// [`is_run_terminator`].
///
/// This is a *classification* predicate: it drives the `class::BRANCH` bit in
/// [`classify`], i.e. what an annotated listing reports as a control transfer.
/// It deliberately excludes `ecall`/`ebreak`, which are traps rather than
/// branches from a reader's point of view. For deciding where a static run must
/// end, use [`is_run_terminator`] instead -- see its docs for why the two are
/// not the same predicate.
#[inline]
pub(crate) fn is_control_flow(ins: &riscv_isa::Instruction) -> bool {
    use riscv_isa::Instruction as I;
    ins.branch() || matches!(ins, I::SRET | I::MRET)
}

/// Whether a static run must end at `ins`, i.e. whether `ins` can affect trace
/// decode state.
///
/// This is [`is_control_flow`] plus `ecall`/`ebreak`, and the difference is
/// load-bearing. What the tracer's `next_pc` actually branches on is the
/// E-Trace notion of an "uninferable discontinuity" -- an indirect jump, a
/// return-from-trap, or an `ecall`/`ebreak`, i.e. `JALR | SRET | MRET | ECALL |
/// EBREAK` here -- while `is_control_flow` covers only the first two. So an
/// `ecall` sitting mid-run has the run predicting `pc + len` where the tracer
/// would take the *reported* address.
///
/// On QEMU user-mode SMI traces that happens to be harmless: syscalls are
/// emulated inline, so the next retired instruction really is `pc + len` and the
/// run cache's slice compare never fires (`mismatches = 0`). It is not harmless
/// in general -- a trace where the trap is genuinely reported, such as a HW
/// capture covering kernel space, would diverge. And it stops being *detectable*
/// once the slice compare is removed and the decoder becomes the sole authority
/// on which run executed.
///
/// Kept separate from `is_control_flow` rather than widening it, because that
/// predicate also feeds `class::BRANCH` in [`classify`]: widening it re-labels
/// every `ecall` as a branch in the annotate report, which changed the rendered
/// output for each syscall wrapper (`__brk`, `__mmap64`, `munmap`,
/// `__internal_syscall_cancel`, ...) while leaving `perf_data_final.json`
/// untouched. Two different questions, two predicates.
///
/// Spelled as its own flat `matches!` rather than delegating to
/// [`is_control_flow`], because the whole point is that the two variant lists
/// differ by exactly `ECALL`/`EBREAK`: having both visible at both sites is what
/// stops the next reader from "simplifying" one into the other.
#[inline]
pub(crate) fn is_run_terminator(ins: &riscv_isa::Instruction) -> bool {
    use riscv_isa::Instruction as I;
    ins.branch() || matches!(ins, I::SRET | I::MRET | I::ECALL | I::EBREAK)
}

/// Absolute control-transfer target of `ins` at `pc`, if statically known.
///
/// Covers conditional branches and inferable jumps (`j`/`jal`, and `jalr` off
/// `x0`); returns `None` for register-indirect jumps, whose target needs the
/// register file.
///
/// Note that for `jalr rd, off(x0)` the architectural target is `off` absolute,
/// not `pc + off`. Computing `pc + off` matches what the E-Trace tracer this
/// replaced did, and the encoding does not arise in compiler output, so the
/// behaviour is deliberately preserved rather than "fixed" here.
#[inline]
fn branch_target_of(ins: &riscv_isa::Instruction, pc: u64) -> Option<u64> {
    use riscv_isa::Instruction as I;
    let offset = match ins {
        I::BEQ { offset, .. }
        | I::BNE { offset, .. }
        | I::BLT { offset, .. }
        | I::BGE { offset, .. }
        | I::BLTU { offset, .. }
        | I::BGEU { offset, .. }
        | I::JAL { offset, .. }
        | I::JALR { rs1: 0, offset, .. } => *offset,
        _ => return None,
    };
    Some(pc.wrapping_add_signed(offset as i64))
}

/// A global (per-hart) map from PC to its accumulated [`PcRecord`].
pub struct PcProfile {
    enabled: bool,
    records: FxHashMap<u64, PcRecord>,
    /// Cache counts whose PC has no record yet. See
    /// [`flush_pending_cache`](Self::flush_pending_cache); bounded by the number
    /// of distinct vector memory PCs, and empty unless cache modelling is on.
    pending_cache: FxHashMap<u64, crate::common::CacheCounts>,
}

impl PcProfile {
    /// Create a profile; `enabled == false` makes [`record`](Self::record) a
    /// no-op so the hot loop pays nothing when annotation is not requested.
    pub fn new(enabled: bool) -> Self {
        Self {
            enabled,
            records: FxHashMap::default(),
            pending_cache: FxHashMap::default(),
        }
    }

    /// Whether recording is active.
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// Borrow the per-PC records.
    pub fn records(&self) -> &FxHashMap<u64, PcRecord> {
        &self.records
    }

    /// Add one instruction's modelled cache counts to its record.
    ///
    /// Separate from [`record`](Self::record) because the two arrive at
    /// different times: `record` runs as the instruction commits, while cache
    /// counts come from the block's footprint walk, which happens before the
    /// block is committed. They meet in the same [`PcRecord`], keyed by PC.
    ///
    /// A no-op when the profile is disabled, matching `record`, so the cost of
    /// cache modelling does not depend on whether annotation was requested.
    /// Unlike `record` this does *not* create a record for an unseen PC: the
    /// commit that follows will, and creating one here from an address with no
    /// decoded instruction would put a `PcRecord` with an empty disassembly in
    /// the annotate view.
    #[inline]
    pub fn add_cache_counts(&mut self, pc: u64, counts: &crate::common::CacheCounts) {
        if !self.enabled {
            return;
        }
        if let Some(rec) = self.records.get_mut(&pc) {
            rec.cache.add(counts);
        } else {
            self.pending_cache.entry(pc).or_default().add(counts);
        }
    }

    /// Fold in cache counts that arrived before their PC had a record.
    ///
    /// Called once at the end of a hart's processing. The first execution of a
    /// vector memory instruction reports its cache counts *before* the block
    /// containing it commits, so its record does not exist yet; without this,
    /// every instruction would silently lose its first execution's counts, and
    /// the per-PC sums would disagree with the ROI totals by exactly one
    /// execution each.
    pub fn flush_pending_cache(&mut self) {
        for (pc, counts) in std::mem::take(&mut self.pending_cache) {
            if let Some(rec) = self.records.get_mut(&pc) {
                rec.cache.add(&counts);
            }
        }
    }

    /// Record a single committed instruction.
    ///
    /// Must be called at exactly the points where the aggregate counters are
    /// bumped so the sum-vs-aggregate invariant holds.
    ///
    /// `set_vl` is `Some(vl)` only when `ins` is the instruction that just set the
    /// vector length to `vl` (i.e. the value the tracker consumed this step): a
    /// `vsetvl*`, or a fault-only-first load that trimmed it.
    /// The per-VL histogram is attributed to that PC — not to every consuming
    /// vector op — so a loop body shows a single histogram at the instruction
    /// that actually determined the VL.
    #[inline]
    // Every parameter is a distinct piece of the committed instruction's
    // decode/VL state; bundling them into a struct would only move the
    // argument list to the call site.
    #[allow(clippy::too_many_arguments)]
    pub fn record(
        &mut self,
        pc: u64,
        ins: &riscv_isa::Instruction,
        ins_bytes: &[u8],
        func_index: Option<usize>,
        delta: &crate::common::InsnDelta,
        current_lmul: Option<Lmul>,
        set_vl: Option<u32>,
    ) {
        if !self.enabled {
            return;
        }

        // Everything numeric comes from the `InsnDelta` the caller already
        // resolved for the aggregate counters: `delta.vl` is the same
        // "scalar contributes 1, vector contributes VL" quantity this used to
        // recompute, and `delta.elems` is `InsnClass::vector_elements`, which
        // is `vector_elements_for` precomputed per PC. Deriving them a second
        // time here cost an `elem_mult`/whole-register re-classification plus
        // an atomic `SYSTEM_VLEN` load on all 306M instructions of a
        // flash_full run, and risked drifting from the aggregate counters --
        // sharing the delta makes the
        // `Σ vector_elems == RoiInfo::vector_elements` invariant structural.
        let is_vector = delta.flags & crate::common::IC_VECTOR != 0;
        let vl_used = delta.vl as u32;
        let elems = delta.elems;
        let current_sew = delta.sew;

        let rec = self
            .records
            .entry(pc)
            .or_insert_with(|| PcRecord::unexecuted(pc, ins, ins_bytes, func_index));

        rec.exec_count += 1;
        if is_vector {
            // Aggregate element counts stay on the consuming vector op, so the
            // Σ vector_elems == RoiInfo.vector_elements invariant is preserved.
            rec.vector_elems += elems;
            rec.vl_min = rec.vl_min.min(vl_used);
            rec.vl_max = rec.vl_max.max(vl_used);
            rec.sew = current_sew;
            rec.lmul = current_lmul;
        }
        // The VL histogram belongs to the instruction that set the length, not
        // to every vector op that reads it — so the annotation shows one
        // histogram at the controlling instruction rather than a duplicate per
        // row. For a `vl*ff` that is the load itself, which both sets `vl` and
        // consumes it (so it also lands in the `is_vector` arm above).
        if let Some(set) = set_vl {
            *rec.vl_hist.entry(set).or_insert(0) += 1;
            rec.vl_min = rec.vl_min.min(set);
            rec.vl_max = rec.vl_max.max(set);
            rec.sew = current_sew;
            rec.lmul = current_lmul;
        }
    }

    /// Record `n` executions of the instruction at `pc`, all at the same vtype.
    ///
    /// The batched form of [`record`](Self::record) for the enricher's run
    /// fast path, which commits a whole straight-line run at once and so knows
    /// only at finalisation how many times each of the run's PCs executed (see
    /// [`RunCache::distribute_pc_profile`](super::run_cache::RunCache::distribute_pc_profile)).
    ///
    /// Observationally identical to `n` `record` calls with the same `delta`:
    /// the count and the element total scale by `n`, while `vl_min`/`vl_max` and
    /// the last-seen `sew`/`lmul` are idempotent under repetition. There is no
    /// `set_vl` parameter because every instruction that writes `vl` is a run
    /// terminator and so is never inside a batched run -- the VL histogram stays
    /// entirely on the per-instruction path.
    #[inline]
    // Same parameter list as `record` (minus `set_vl`), for the same reason.
    #[allow(clippy::too_many_arguments)]
    pub fn record_n(
        &mut self,
        pc: u64,
        ins: &riscv_isa::Instruction,
        ins_bytes: &[u8],
        func_index: Option<usize>,
        delta: &crate::common::InsnDelta,
        current_lmul: Option<Lmul>,
        n: u64,
    ) {
        if !self.enabled || n == 0 {
            return;
        }
        let is_vector = delta.flags & crate::common::IC_VECTOR != 0;
        let vl_used = delta.vl as u32;
        let elems = delta.elems;
        let current_sew = delta.sew;

        let rec = self
            .records
            .entry(pc)
            .or_insert_with(|| PcRecord::unexecuted(pc, ins, ins_bytes, func_index));

        rec.exec_count += n;
        if is_vector {
            // Aggregate element counts stay on the consuming vector op, so the
            // Σ vector_elems == RoiInfo.vector_elements invariant is preserved.
            rec.vector_elems += elems * n;
            rec.vl_min = rec.vl_min.min(vl_used);
            rec.vl_max = rec.vl_max.max(vl_used);
            rec.sew = current_sew;
            rec.lmul = current_lmul;
        }
    }

    /// Fold another profile's records into this one (used to merge per-hart
    /// profiles before report generation).
    pub fn merge_from(&mut self, other: &PcProfile) {
        for (pc, o) in &other.records {
            match self.records.get_mut(pc) {
                Some(e) => {
                    e.exec_count += o.exec_count;
                    e.vector_elems += o.vector_elems;
                    e.class |= o.class;
                    if o.vl_max > 0 {
                        e.vl_min = e.vl_min.min(o.vl_min);
                        e.vl_max = e.vl_max.max(o.vl_max);
                    }
                    if e.sew.is_none() {
                        e.sew = o.sew;
                    }
                    if e.lmul.is_none() {
                        e.lmul = o.lmul;
                    }
                    for (vl, c) in &o.vl_hist {
                        *e.vl_hist.entry(*vl).or_insert(0) += c;
                    }
                    if e.func_index.is_none() {
                        e.func_index = o.func_index;
                    }
                    if e.branch_target.is_none() {
                        e.branch_target = o.branch_target;
                    }
                }
                None => {
                    self.records.insert(*pc, o.clone());
                }
            }
        }
    }

    /// Instruction mix (mnemonic -> execution count) for one function, derived
    /// from the already-collected per-PC records. This is the only source of
    /// `RoiInfo::instruction_mix`: O(unique PCs in the function) rather than
    /// O(dynamic instructions), which is why the mix is computed once at
    /// finalize time (see `Enricher::apply_deferred_instruction_mix`) instead of
    /// being maintained per instruction commit.
    pub fn instruction_mix_for_function(&self, func_index: usize) -> BTreeMap<String, u64> {
        let mut mix = BTreeMap::new();
        for rec in self.records.values() {
            if rec.func_index == Some(func_index) {
                let mnemonic = rec.disasm.split_whitespace().next().unwrap_or(&rec.disasm);
                *mix.entry(mnemonic.to_string()).or_insert(0) += rec.exec_count;
            }
        }
        mix
    }

    /// Instruction mix across every recorded PC, regardless of function --
    /// used for the global ROI, which has no single function scope.
    pub fn instruction_mix_all(&self) -> BTreeMap<String, u64> {
        let mut mix = BTreeMap::new();
        for rec in self.records.values() {
            let mnemonic = rec.disasm.split_whitespace().next().unwrap_or(&rec.disasm);
            *mix.entry(mnemonic.to_string()).or_insert(0) += rec.exec_count;
        }
        mix
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::RoiInfo;

    fn target() -> riscv_isa::Target {
        use std::str::FromStr;
        riscv_isa::Target::from_str("RV64I").unwrap().full()
    }

    fn decode(bytes: &[u8]) -> riscv_isa::Instruction {
        riscv_isa::decode_le_bytes(bytes, &target()).unwrap().0
    }

    /// Build the `InsnDelta` for `ins` under the given live `vtype` state,
    /// exactly as `Enricher::transform_optimized` does before handing the same
    /// delta to `count_instruction` and `PcProfile::record`. Note that
    /// `RoiInfo::update_stats` -- the other side of the reconciliation
    /// assertions below -- resolves its own contribution internally, so these
    /// tests still compare two independent paths.
    fn delta_for(
        ins: &riscv_isa::Instruction,
        vl: Option<u32>,
        sew: Option<u8>,
        lmul: Option<Lmul>,
    ) -> crate::common::InsnDelta {
        let vlen = crate::common::SYSTEM_VLEN.get_bytes();
        let class = crate::common::InsnClass::from_instruction(ins, vlen);
        crate::common::InsnDelta::new(&class, ins, vl, sew, lmul, vlen)
    }

    #[test]
    fn disabled_is_noop() {
        // delta_for reads SYSTEM_VLEN; nextest gives each test its own process,
        // so set it here rather than relying on a sibling test.
        crate::common::SYSTEM_VLEN.set_bytes(16);

        let mut p = PcProfile::new(false);
        let ins = decode(&[0x13, 0x00, 0x00, 0x00]); // addi x0,x0,0 (nop)
        p.record(
            0x1000,
            &ins,
            &[0x13, 0x00, 0x00, 0x00],
            Some(0),
            &delta_for(&ins, None, None, None),
            None,
            None,
        );
        assert!(p.records().is_empty());
    }

    #[test]
    fn exec_count_reconciles_with_aggregate() {
        // Record a small scalar stream into both a PcProfile and a RoiInfo and
        // assert the per-PC sums equal the aggregate counters.
        let nop = decode(&[0x13, 0x00, 0x00, 0x00]); // addi x0,x0,0
        let bytes = [0x13u8, 0x00, 0x00, 0x00];

        // update_stats reads SYSTEM_VLEN via bytes_moved; set it for the test.
        crate::common::SYSTEM_VLEN.set_bytes(16);

        let mut profile = PcProfile::new(true);
        let mut roi = RoiInfo::new("t".to_string());

        const N: usize = 5;
        for i in 0..N {
            let pc = 0x1000 + (i as u64 % 2) * 4; // two distinct PCs
            profile.record(
                pc,
                &nop,
                &bytes,
                Some(0),
                &delta_for(&nop, None, None, None),
                None,
                None,
            );
            roi.instructions += 1;
            roi.update_stats(&nop, None, None, None);
        }

        let sum_exec: u64 = profile.records().values().map(|r| r.exec_count).sum();
        let sum_elems: u64 = profile.records().values().map(|r| r.vector_elems).sum();
        assert_eq!(sum_exec, roi.instructions);
        assert_eq!(sum_elems, roi.vector_elements);
    }

    /// Whole-register and mask ops must agree between the two element-count
    /// paths, as segment ops do.
    ///
    /// `RoiInfo` reads these from the per-PC cached `InsnClass`, while
    /// `PcProfile` recomputes them via `vector_elements_for`. Both must apply
    /// the spec's `evl` -- `NFIELDS * VLEN / EEW` for whole-register ops
    /// (section 7.9), `ceil(vl/8)` for mask ops (section 7.4) -- rather than
    /// the current `vl`, which these instructions ignore.
    ///
    /// Without this, mutating the whole-register branch of
    /// `vector_elements_for` survives: the asm validation tests run with
    /// annotate off, so no `PcProfile` is built for them.
    #[test]
    fn whole_register_and_mask_agree_across_paths() {
        // VLEN is 16 bytes = 128 bits for this test, so:
        //   vl1r.v    nreg 1, EEW 8  -> evl = 1 * 128 / 8  =  16
        //   vl2re64.v nreg 2, EEW 64 -> evl = 2 * 128 / 64 =   4
        //   vs1r.v    nreg 1, EEW 8  -> evl = 1 * 128 / 8  =  16
        //   vlm.v / vsm.v            -> evl = ceil(vl/8)   =   1  (vl = 4)
        const VL: u32 = 4;
        let cases: [(u32, &str, u64); 5] = [
            (0x0285_0207, "vl1r.v", 16),
            (0x2285_7207, "vl2re64.v", 4),
            (0x0285_0227, "vs1r.v", 16),
            (0x02b5_0207, "vlm.v", 1),
            (0x02b5_0227, "vsm.v", 1),
        ];

        crate::common::SYSTEM_VLEN.set_bytes(16);
        let mut profile = PcProfile::new(true);
        let mut roi = RoiInfo::new("t".to_string());

        for (i, (code, _, _)) in cases.iter().enumerate() {
            let bytes = code.to_le_bytes();
            let ins = decode(&bytes);
            profile.record(
                0x2000 + (i as u64) * 4,
                &ins,
                &bytes,
                Some(0),
                &delta_for(&ins, Some(VL), Some(3), None),
                None,
                None,
            );
            roi.instructions += 1;
            roi.update_stats(&ins, Some(VL), Some(3), None);
        }

        let expected_elems: u64 = cases.iter().map(|(_, _, evl)| evl).sum();
        let sum_elems: u64 = profile.records().values().map(|r| r.vector_elems).sum();
        assert_eq!(
            roi.vector_elements, expected_elems,
            "RoiInfo must use evl, not vl"
        );
        assert_eq!(
            sum_elems, roi.vector_elements,
            "PcProfile and RoiInfo element counts must agree"
        );

        // The mix is a per-PC artifact: `RoiInfo` never accumulates one on the
        // counting path, it is derived from this profile at end of run.
        roi.update_from_histogram();
        assert!(roi.instruction_mix.is_empty());
        let expected: BTreeMap<String, u64> = cases
            .iter()
            .map(|(_, mnemonic, _)| (mnemonic.to_string(), 1u64))
            .collect();
        assert_eq!(profile.instruction_mix_for_function(0), expected);
    }

    /// Segment load/stores must land in per-mnemonic mix buckets and preserve
    /// the element-sum invariant against `RoiInfo`.
    ///
    /// Both are easy to break independently. The mnemonic of a segment op comes
    /// from its `nf`/`eew` *fields*, not from its enum variant, so several of
    /// these share a variant and must still be counted separately -- which the
    /// per-PC keying gives for free, since it renders the disassembly. And each
    /// path applies the `nf` element multiplier itself, so the element counts can
    /// drift apart.
    #[test]
    fn segment_ops_mix_and_elements() {
        // Four mnemonics sharing the VLSEG_V discriminant, plus one that does
        // not, all at v4, (x10). vlseg2e64.v and vlseg2e32.v share a variant
        // *and* an nf, so they pin the eew half of the mnemonic split.
        let cases: [(u32, &str, u64); 5] = [
            (0x2205_7207, "vlseg2e64.v", 2),
            (0x2205_6207, "vlseg2e32.v", 2),
            (0x4205_7207, "vlseg3e64.v", 3),
            (0xe205_0207, "vlseg8e8.v", 8),
            (0x2205_0227, "vsseg2e8.v", 2),
        ];

        crate::common::SYSTEM_VLEN.set_bytes(16);
        let mut profile = PcProfile::new(true);
        let mut roi = RoiInfo::new("t".to_string());
        const VL: u32 = 4;

        for (i, (code, _, _)) in cases.iter().enumerate() {
            let bytes = code.to_le_bytes();
            let ins = decode(&bytes);
            profile.record(
                0x1000 + (i as u64) * 4,
                &ins,
                &bytes,
                Some(0),
                &delta_for(&ins, Some(VL), Some(3), None),
                None,
                None,
            );
            roi.instructions += 1;
            roi.update_stats(&ins, Some(VL), Some(3), None);
        }

        // Element sums must still reconcile with the nf multiplier applied.
        let sum_elems: u64 = profile.records().values().map(|r| r.vector_elems).sum();
        let expected_elems: u64 = cases.iter().map(|(_, _, nf)| nf * VL as u64).sum();
        assert_eq!(sum_elems, roi.vector_elements);
        assert_eq!(roi.vector_elements, expected_elems);

        // The mix is a per-PC artifact: `RoiInfo` never accumulates one on the
        // counting path, it is derived from this profile at end of run.
        roi.update_from_histogram();
        assert!(roi.instruction_mix.is_empty());
        let expected: BTreeMap<String, u64> = cases
            .iter()
            .map(|(_, mnemonic, _)| (mnemonic.to_string(), 1u64))
            .collect();
        assert_eq!(profile.instruction_mix_for_function(0), expected);
    }

    #[test]
    fn classify_flags() {
        let nop = decode(&[0x13, 0x00, 0x00, 0x00]);
        // A plain addi is scalar: no class bits set.
        assert_eq!(classify(&nop), 0);
        assert_eq!(class_tags(0), "scalar");
    }

    #[test]
    fn jal_is_control_flow() {
        // jal x1, 0x10 — a direct jump/call. It is not a conditional branch, but
        // it is control flow and must get the BRANCH (control-flow) class bit.
        let jal = decode(&[0xef, 0x00, 0x00, 0x01]);
        assert!(!jal.conditional_branch(), "jal is not a conditional branch");
        // ... but riscv_isa's wider `branch()` does include it. This pins the
        // distinction `InsnClass::from_instruction` depends on: swapping its
        // `conditional_branch()` for `branch()` would silently redefine the ROI
        // `branches` counter from conditional branches to all control transfers.
        assert!(jal.branch(), "but riscv_isa::branch() does include jal");
        assert_eq!(classify(&jal) & class::BRANCH, class::BRANCH);
        assert!(class_tags(classify(&jal)).contains("branch"));
    }

    #[test]
    fn ecall_terminates_a_run_but_is_not_classified_as_a_branch() {
        // ecall. The tracer reacts to it -- an ecall is an uninferable
        // discontinuity, so `next_pc` takes the *reported* address -- which
        // means a static run must not continue past it, or the run would
        // predict `pc + 4` where the decoder went somewhere else.
        let ecall = decode(&[0x73, 0x00, 0x00, 0x00]);
        assert!(
            matches!(ecall, riscv_isa::Instruction::ECALL),
            "expected an ecall"
        );
        assert!(
            is_run_terminator(&ecall),
            "ecall must end a run: the decoder's next PC is not statically known"
        );

        // But it must NOT be reported as a control transfer in the annotate
        // view. Widening `is_control_flow` to cover it (rather than adding the
        // separate `is_run_terminator`) re-labels every syscall wrapper's ecall
        // as a branch and changes the rendered annotate_*.html.gz output, while
        // leaving perf_data_final.json byte-identical -- which is how easy it is
        // to miss. Two questions, two predicates.
        assert!(
            !is_control_flow(&ecall),
            "ecall is a trap, not a branch: it must not set class::BRANCH"
        );
        assert_eq!(classify(&ecall) & class::BRANCH, 0);

        // The relationship the run cache depends on: every control transfer is
        // also a run terminator, so the terminator rule is a strict superset.
        let jal = decode(&[0xef, 0x00, 0x00, 0x01]);
        assert!(is_control_flow(&jal) && is_run_terminator(&jal));
    }

    #[test]
    fn vl_histogram_lands_on_vsetvl_not_vector_op() {
        // The histogram is attributed to the vsetvl* PC (via set_vl), while a
        // consuming vector op carries only the aggregate element count.
        crate::common::SYSTEM_VLEN.set_bytes(16);
        let nop = decode(&[0x13, 0x00, 0x00, 0x00]);
        let bytes = [0x13u8, 0x00, 0x00, 0x00];

        let mut p = PcProfile::new(true);
        // vsetvl* PC: pretend this instruction set VL=2 four times.
        for _ in 0..4 {
            p.record(
                0x100,
                &nop,
                &bytes,
                Some(0),
                &delta_for(&nop, Some(2), Some(3), None),
                None,
                Some(2),
            );
        }
        // Consuming vector op executes at VL=2, but with no set_vl.
        // (Use the nop opcode purely as a stand-in; is_vector() is false for it,
        // so drive the vector path via a real vector encoding instead.)
        let vadd = decode(&[0x57, 0x80, 0x20, 0x02]); // vadd.vv v0,v0,v1
        assert!(vadd.vector());
        for _ in 0..10 {
            p.record(
                0x200,
                &vadd,
                &bytes,
                Some(0),
                &delta_for(&vadd, Some(2), Some(3), None),
                None,
                None,
            );
        }

        let vset = &p.records()[&0x100];
        let vop = &p.records()[&0x200];
        // Histogram only at the vsetvl PC.
        assert_eq!(vset.vl_hist.get(&2).copied(), Some(4));
        assert!(
            vop.vl_hist.is_empty(),
            "vector op must not carry a histogram"
        );
        // Aggregate element count stays on the vector op (invariant).
        assert_eq!(vop.vector_elems, 20);
        assert_eq!(vset.vector_elems, 0);
    }
}
