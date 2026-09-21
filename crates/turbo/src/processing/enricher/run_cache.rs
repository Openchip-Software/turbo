//! Static run cache: per-*run* aggregates instead of per-instruction deltas.
//!
//! Enrichment is currently per retired instruction, but almost nothing it does
//! depends on dynamic state. Given a start PC, the instruction sequence that
//! follows is fixed by the binary until the first control transfer, and the only
//! dynamic input to each instruction's *counter contribution* is the live vtype
//! (VL/SEW/LMUL) -- which cannot change inside such a sequence, because
//! `vsetvl*` instructions are themselves run terminators.
//!
//! So a maximal straight-line **run** can have its whole counter contribution
//! summed once per entry vtype and then applied with a single
//! [`RoiInfo::add_run_delta`] per execution. On the measured flash_full trace the
//! instruction-weighted mean run length is 15.3 over just 2,310 distinct static
//! runs and 3,541 distinct `(run, vtype)` pairs, so these tables are tiny and
//! the amortisation is large.
//!
//! `Enricher::transform_optimized` consumes this through [`RunCache::lookup`]:
//! one lookup per run start, a slice compare against the trace, then one
//! [`RoiInfo::add_run_delta`] per scope. The per-PC annotation profile is not
//! touched in the loop at all -- the run's execution count is bumped instead,
//! and [`RunCache::distribute_pc_profile`] multiplies the contributions out over
//! the run's PCs once at finalisation.
//!
//! ## The invariant everything rests on
//!
//! A [`RunDelta`] must be *exactly* the sum of the [`InsnDelta`]s that the
//! per-instruction path would have produced. It is therefore built by calling
//! the very same [`InsnDelta::new`] once per instruction in the run and
//! accumulating -- never by re-deriving the element/byte arithmetic -- and
//! `run_delta_matches_per_insn_*` in this module's tests compares a `RoiInfo`
//! fed one `add_run_delta` against one fed the N `apply_delta` calls, field by
//! field, histograms included.

use std::cell::Cell;

use rustc_hash::FxHashMap;

use crate::common::{
    InsnClass, InsnDelta, Lmul, IC_BRANCH, IC_FLOAT, IC_FMA, IC_INT_FMA, IC_LOAD, IC_STORE,
    IC_VECTOR,
};
use crate::processing::instruction_decoder::InstructionDecoder;

use super::pc_profile::is_run_terminator;
use super::vl_tracker::writes_vl;

/// Hard cap on the instructions in one run.
///
/// A run is validated against the trace with a slice compare before its
/// aggregate is applied, so an unbounded straight-line stretch would just move
/// the cost from the counter updates into that compare. 4096 is far above the
/// observed p90 of 31.
pub(crate) const MAX_RUN_LEN: usize = 4096;

/// Hard cap on memoized `(vtype, aggregate)` pairs per run.
///
/// Searched linearly, so it must stay small. The measured worst case on
/// flash_full is 11 distinct vtypes for a single run; 16 keeps that entirely
/// inside the fast path with headroom, and beyond it the caller simply falls
/// back to the per-instruction path (see [`RunCache::lookup`]).
pub(crate) const MAX_MEMO_ENTRIES: usize = 16;

/// The dynamic vtype state an instruction's contribution depends on.
///
/// These are precisely the three values [`InsnDelta::new`] consumes from
/// [`VlTracker`](super::VlTracker) (`current_vl`, `current_sew`, `current_lmul`),
/// copied verbatim including their `Option`-ness -- so "same [`VtypeKey`]"
/// means "same dynamic inputs", and a memo hit is sound rather than merely
/// likely. `vlen_bytes` is not part of the key: it is fixed for a whole run of
/// the tool and folded in when the run is built.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) struct VtypeKey {
    /// Live architectural VL, `None` before any `vsetvl*`.
    pub vl: Option<u32>,
    /// Live SEW in vtype encoding (0..=3 for e8/e16/e32/e64).
    pub sew: Option<u8>,
    /// Live LMUL.
    pub lmul: Option<Lmul>,
}

/// Per-instruction static data, parallel to [`Run::pcs`].
///
/// All three fields come straight out of the decoder's per-PC cache, so
/// building a run costs one decode per instruction *once* in the whole program.
/// Kept per run rather than re-fetched from the decode cache, so that
/// [`RunDelta::build`] and [`RunCache::distribute_pc_profile`] can walk a run
/// without any hash lookups at all.
#[derive(Clone, Copy, Debug)]
pub(crate) struct RunInsn {
    pub ins: riscv_isa::Instruction,
    pub class: InsnClass,
    /// Encoded length in bytes (2 for RVC, 4 otherwise).
    pub len: u8,
}

/// The summed counter contribution of one run executed once at one vtype.
///
/// Field-for-field the aggregate of what [`RoiInfo::apply_delta`] would have
/// added; see [`RoiInfo::add_run_delta`] for the replay side.
#[derive(Clone, Debug, Default, PartialEq)]
pub(crate) struct RunDelta {
    pub instructions: u64,
    pub branches: u64,
    pub vector_instructions: u64,
    pub float_instructions: u64,
    pub fmacc_instructions: u64,
    pub vector_load_instructions: u64,
    pub vector_elements: u64,
    pub float_elements: u64,
    pub fmacc_elements: u64,
    pub vector_load_elements: u64,
    pub load_bytes: u64,
    pub store_bytes: u64,
    /// The run's single VL-histogram bump: `(sew, lmul, vl, count)`.
    ///
    /// One bump suffices because vtype is constant across a run, so every vector
    /// instruction in it records the same `(vl, lmul)` into the same per-SEW
    /// histogram. `None` when the run has no vector instructions, or when the
    /// live SEW is unknown/above e64 -- exactly the cases where `apply_delta`'s
    /// `match` falls through without recording.
    pub vl_hist_bump: Option<(u8, Option<u8>, u64, u64)>,
}

impl RunDelta {
    /// Sum the run's per-instruction [`InsnDelta`]s at one entry vtype.
    ///
    /// Deliberately delegates every per-instruction derivation to
    /// [`InsnDelta::new`]: that keeps `vl`/`elems`/`bytes_moved` -- the indexed
    /// vector-memory SEW correction, the whole-register and mask `evl` rules,
    /// the segment `nf` multiplier -- defined in exactly one place, so the fast
    /// and slow paths cannot drift. Only the *accumulation* (which mirrors
    /// `RoiInfo::apply_delta`'s flag tests) lives here.
    pub(crate) fn build(insns: &[RunInsn], key: VtypeKey, vlen_bytes: u16) -> Self {
        let mut out = RunDelta::default();

        for RunInsn { ins, class, .. } in insns {
            let delta = InsnDelta::new(class, ins, key.vl, key.sew, key.lmul, vlen_bytes);
            let (vl, elems, bytes_moved) = (delta.vl, delta.elems, delta.bytes_moved);
            let flags = delta.flags;

            out.instructions += 1;

            let is_vector = flags & IC_VECTOR != 0;
            if is_vector {
                out.vector_instructions += 1;
                out.vector_elements += elems;
            }
            if flags & IC_BRANCH != 0 {
                out.branches += 1;
            }
            if flags & IC_FLOAT != 0 {
                out.float_instructions += 1;
                out.float_elements += vl;
                if flags & IC_FMA != 0 {
                    out.fmacc_instructions += 1;
                    out.fmacc_elements += vl;
                    // FMA counts 2x for float ops, as in `apply_delta`.
                    out.float_elements += vl;
                }
            }
            if flags & IC_INT_FMA != 0 {
                out.fmacc_elements += vl;
            }
            if flags & IC_LOAD != 0 {
                out.load_bytes += bytes_moved;
                if is_vector {
                    out.vector_load_instructions += 1;
                    out.vector_load_elements += elems;
                }
            }
            if flags & IC_STORE != 0 {
                out.store_bytes += bytes_moved;
            }
        }

        // The histogram is fed the *architectural* VL, never `elems`, and only
        // for the four representable SEWs -- mirroring `apply_delta`'s `match`,
        // whose `_` arm records nothing.
        out.vl_hist_bump = match key.sew {
            Some(sew) if sew <= 3 && out.vector_instructions > 0 => Some((
                sew,
                key.lmul.map(|l| l.to_vlmul()),
                key.vl.unwrap_or(1) as u64,
                out.vector_instructions,
            )),
            _ => None,
        };
        out
    }
}

/// A maximal straight-line run of instructions. Built lazily on first sighting
/// of its start PC, then immutable apart from its memo list.
pub(crate) struct Run {
    /// The exact PC sequence, in order. Contiguous by construction, kept as a
    /// slice so validating the trace against it is one slice compare -- the step
    /// that makes the fast path *sound* in the presence of traps, interrupts or
    /// self-modifying code rather than merely plausible.
    pcs: Box<[u64]>,
    /// Static per-instruction data, parallel to `pcs`.
    insns: Box<[RunInsn]>,
    /// Memoized aggregates by entry vtype. Almost always 1-2 entries, so a
    /// linear scan beats hashing.
    memo: Vec<MemoEntry>,
    /// Function this run is attributed to, as resolved by the enricher for its
    /// start PC on the first application. A run cannot cross a function boundary
    /// without a control transfer, and the enricher additionally refuses any run
    /// whose last PC falls outside the resolved range, so this is a property of
    /// the run rather than of the execution -- which is what lets the per-PC
    /// distribution happen once at the end instead of per instruction.
    ///
    /// A [`Cell`] for the same reason `MemoEntry::execs` is one.
    func_index: Cell<Option<usize>>,
}

/// One memoized `(entry vtype, aggregate)` pair, plus the number of times the
/// run has been executed at that vtype.
pub(crate) struct MemoEntry {
    pub key: VtypeKey,
    pub delta: RunDelta,
    /// Executions at `key`, i.e. the `N` that [`RunCache::distribute_pc_profile`]
    /// multiplies the run's per-PC contributions out by.
    ///
    /// A [`Cell`] so the enricher can count an execution while holding only the
    /// shared borrow it already has on the aggregate it is applying: the
    /// alternative is a second hash lookup per run application, on the very path
    /// this whole module exists to keep short.
    pub execs: Cell<u64>,
}

impl Run {
    /// Number of distinct entry vtypes memoized so far. Tests only: the hot
    /// path reaches the PCs and the aggregate through [`RunHit`], so `Run`
    /// itself needs no other accessors.
    #[cfg(test)]
    fn memo_len(&self) -> usize {
        self.memo.len()
    }

    /// Index of the memo entry for `key`, building it on first sighting.
    ///
    /// `None` means the memo is full: the caller must take the per-instruction
    /// path for this stretch. `builds` counts aggregate constructions, for
    /// diagnostics and for the memo tests.
    fn memo_index(&mut self, key: VtypeKey, vlen_bytes: u16, builds: &mut u64) -> Option<usize> {
        if let Some(i) = self.memo.iter().position(|e| e.key == key) {
            return Some(i);
        }
        if self.memo.len() >= MAX_MEMO_ENTRIES {
            return None;
        }
        *builds += 1;
        self.memo.push(MemoEntry {
            key,
            delta: RunDelta::build(&self.insns, key, vlen_bytes),
            execs: Cell::new(0),
        });
        Some(self.memo.len() - 1)
    }
}

/// Lazily built map from start PC to the run beginning there, for **one**
/// module.
///
/// Keyed by PC alone, so the caller must hold one cache per
/// [`InstructionDecoder`] / module index: the same PC in two modules is two
/// different instructions. Bounded by static code size, not by trace length.
pub(crate) struct RunCache {
    /// `None` records "no run starts here" (the PC is itself a terminator, or
    /// undecodable), so a hopeless start PC is only analysed once.
    runs: FxHashMap<u64, Option<Run>>,
    /// Run-fixed `SYSTEM_VLEN`, folded into every aggregate built here. Held by
    /// the cache rather than passed per lookup so a caller cannot mix VLENs
    /// within one set of memos.
    vlen_bytes: u16,
    runs_built: u64,
    deltas_built: u64,
    /// Last `start_pc` looked up, and a raw handle to its `Run` inside `runs`.
    /// Loop bodies re-hit the same run start over and over, so this lets a
    /// repeated `start_pc` skip the `runs` hashmap probe entirely -- the same
    /// trick `StatsAccumulator` already uses for `func_map`/`last_func_roi`.
    ///
    /// INVARIANT: `last_run` is only dereferenced when `last_start_pc ==
    /// Some(start_pc)` for the incoming `start_pc`, and it is reset to
    /// `None`/null whenever `runs` could have been structurally mutated
    /// (i.e. on every `lookup` that misses the cache, before the potential
    /// `entry().or_insert_with()` reallocates the map). Between two
    /// consecutive `lookup` calls with the same `start_pc` that both hit the
    /// cache, no structural mutation of `runs` occurs, so the pointer stays
    /// valid.
    last_start_pc: Option<u64>,
    last_run: *mut Option<Run>,
}

/// SAFETY: as for `StatsAccumulator`'s `last_func_roi` -- `last_run` is a
/// self-referential memo into this cache's own `runs` map, never shared or
/// aliased, only dereferenced under the invariant documented on the field, and
/// unaffected by moving the owner between threads (the map's heap allocation
/// does not move). NOT `Sync`: the memo is not race-safe.
unsafe impl Send for RunCache {}

/// A validated-run lookup result: the run's PC sequence and its aggregate at the
/// requested vtype.
pub(crate) struct RunHit<'a> {
    /// PCs the trace must match for the aggregate to be applicable.
    pub pcs: &'a [u64],
    /// The memoized aggregate for the requested vtype.
    pub delta: &'a RunDelta,
    /// Execution counter for this `(run, vtype)`; the caller bumps it once the
    /// slice compare has confirmed the run really did execute, and
    /// [`RunCache::distribute_pc_profile`] later multiplies the run's per-PC
    /// contributions out by it.
    pub execs: &'a Cell<u64>,
    /// Function the run is attributed to; the caller stores its resolution here
    /// so the deferred per-PC distribution can attribute records without
    /// re-resolving.
    pub func_index: &'a Cell<Option<usize>>,
}

impl RunCache {
    /// New empty cache for a module decoded at the given (run-fixed) VLEN.
    pub fn new(vlen_bytes: u16) -> Self {
        Self {
            runs: FxHashMap::default(),
            vlen_bytes,
            runs_built: 0,
            deltas_built: 0,
            last_start_pc: None,
            last_run: std::ptr::null_mut(),
        }
    }

    /// The run starting at `start_pc` together with its aggregate at `key`,
    /// building either on first sighting.
    ///
    /// Returns `None` -- "no memo available, use the per-instruction path" --
    /// when no run starts at `start_pc` (it is a `vsetvl*`, an ROI marker, or
    /// undecodable) or when the run's memo is full. Callers must treat `None` as
    /// routine, not exceptional.
    ///
    /// On `Some`, the caller still has to compare [`RunHit::pcs`] against the
    /// trace's next PCs before applying [`RunHit::delta`]: this is a static
    /// prediction of control flow, and only that compare rules out a trap or an
    /// interrupt having diverted execution mid-run.
    pub fn lookup(
        &mut self,
        decoder: &InstructionDecoder,
        start_pc: u64,
        key: VtypeKey,
    ) -> Option<RunHit<'_>> {
        // Fast path: consecutive lookups usually re-hit the same run start
        // (loop bodies), so a repeated `start_pc` reuses the cached `Run`
        // slot handle and skips the `runs` hashmap probe entirely.
        //
        // SAFETY: `last_run` points at an `Option<Run>` slot owned by `runs`.
        // It is dereferenced only when `last_start_pc` matches `start_pc`,
        // which guarantees the pointer was set on a previous call for this
        // same key and that `runs` has not been structurally mutated since
        // (the only mutation site, the `entry().or_insert_with()` below, runs
        // exclusively on the cache-miss branch). The enricher is
        // single-threaded per hart, so there is no aliasing.
        let slot: &mut Option<Run> = if self.last_start_pc == Some(start_pc) {
            unsafe { &mut *self.last_run }
        } else {
            let runs_built = &mut self.runs_built;
            let slot = self.runs.entry(start_pc).or_insert_with(|| {
                *runs_built += 1;
                build_run(decoder, start_pc)
            });
            self.last_start_pc = Some(start_pc);
            self.last_run = slot as *mut Option<Run>;
            slot
        };
        let run = slot.as_mut()?;
        let idx = run.memo_index(key, self.vlen_bytes, &mut self.deltas_built)?;
        Some(RunHit {
            pcs: &run.pcs,
            delta: &run.memo[idx].delta,
            execs: &run.memo[idx].execs,
            func_index: &run.func_index,
        })
    }

    /// Multiply every run's per-PC contributions out into `profile`, once, from
    /// the `(vtype, executions)` pairs the enricher accumulated.
    ///
    /// This is the deferred half of the batched fast path: instead of calling
    /// [`PcProfile::record`](super::pc_profile::PcProfile::record) for each of
    /// the ~262M instructions committed inside runs, the enricher counts run
    /// executions and this walks the *static* table -- a few thousand
    /// `(run, vtype)` pairs of a handful of instructions each -- at finalisation.
    ///
    /// The per-instruction [`InsnDelta`] is rebuilt here with the same
    /// [`InsnDelta::new`] call the per-instruction path makes, so the recorded
    /// values are identical; only the arithmetic is scaled by `N`.
    ///
    /// Drains the counters, so calling it twice adds nothing the second time.
    pub fn distribute_pc_profile(
        &self,
        profile: &mut super::pc_profile::PcProfile,
        decoder: &InstructionDecoder,
    ) {
        for run in self.runs.values().filter_map(|r| r.as_ref()) {
            let func_index = run.func_index.get();
            for entry in &run.memo {
                let n = entry.execs.replace(0);
                if n == 0 {
                    continue;
                }
                for (pc, insn) in run.pcs.iter().zip(run.insns.iter()) {
                    let delta = InsnDelta::new(
                        &insn.class,
                        &insn.ins,
                        entry.key.vl,
                        entry.key.sew,
                        entry.key.lmul,
                        self.vlen_bytes,
                    );
                    let bytes = decoder.encoded_bytes_at(*pc, insn.len).unwrap_or(&[]);
                    profile.record_n(*pc, &insn.ins, bytes, func_index, &delta, entry.key.lmul, n);
                }
            }
        }
    }

    /// Runs analysed (including start PCs found to begin no run).
    pub fn runs_built(&self) -> u64 {
        self.runs_built
    }

    /// `(run, vtype)` aggregates constructed. Grows only on a memo miss, so
    /// tests can assert that a repeat lookup was served from the memo.
    pub fn deltas_built(&self) -> u64 {
        self.deltas_built
    }

    /// The run at `start_pc`, if it has been analysed. Tests only.
    #[cfg(test)]
    fn run_at(&self, start_pc: u64) -> Option<&Run> {
        self.runs.get(&start_pc)?.as_ref()
    }
}

/// Walk forward from `start_pc` collecting the maximal straight-line run.
///
/// Termination rules, in the order they are applied per candidate instruction:
///
/// * **before** the instruction, i.e. it never becomes part of a run, when it is
///   an ROI marker (it changes the active region set), a `vsetvl*` (it changes
///   vtype and consumes from the VL stream in order), undecodable, or outside the
///   module's executable range -- the latter two both surface as a decode error;
/// * **including** the instruction when it is a control transfer, since the next
///   PC is then not statically implied by this one.
///
/// Returns `None` when nothing at all could be collected, i.e. `start_pc` itself
/// hits a "before" rule.
fn build_run(decoder: &InstructionDecoder, start_pc: u64) -> Option<Run> {
    let mut pcs: Vec<u64> = Vec::new();
    let mut insns: Vec<RunInsn> = Vec::new();
    let mut pc = start_pc;

    while pcs.len() < MAX_RUN_LEN {
        let Ok((ins, bytes, class, marker)) = decoder.get_instruction_class_and_marker_at(pc)
        else {
            break;
        };
        if marker.is_some() || writes_vl(&ins) {
            break;
        }

        let len = bytes.len() as u8;
        pcs.push(pc);
        insns.push(RunInsn { ins, class, len });

        if is_run_terminator(&ins) {
            break;
        }
        pc += u64::from(len);
    }

    if pcs.is_empty() {
        return None;
    }
    Some(Run {
        pcs: pcs.into_boxed_slice(),
        insns: insns.into_boxed_slice(),
        memo: Vec::new(),
        func_index: Cell::new(None),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::{RoiInfo, SYSTEM_VLEN};

    const VLEN_BYTES: u16 = 16;
    const TEXT_ADDR: u64 = 0x1000;

    // Hand-assembled RV64 encodings. Every one of these is asserted about
    // indirectly below (element/byte counters via the path-equivalence checks,
    // decodability via the run lengths), so a bad encoding fails loudly rather
    // than silently weakening a test.
    /// `addi x1, x1, 1` -- plain scalar, not a marker (rd != x0).
    const ADDI_X1: u32 = 0x0010_8093;
    /// `jal x1, +0x10` -- control transfer, terminates a run inclusively.
    const JAL: u32 = 0x0100_00ef;
    /// `vsetvli x0, x0, e32, m1` -- terminates a run exclusively.
    const VSETVLI: u32 = 0x0100_7057;
    /// `add x0, x1, x2` -- writes x0, so the decoder reads it as an ROI
    /// begin-region marker; terminates a run exclusively.
    const MARKER_ADD_X0: u32 = 0x0020_8033;

    fn decoder_for(words: &[u32]) -> InstructionDecoder {
        let mut bytes = Vec::with_capacity(words.len() * 4);
        for w in words {
            bytes.extend_from_slice(&w.to_le_bytes());
        }
        InstructionDecoder::from_exec_bytes(TEXT_ADDR, &bytes)
    }

    /// PCs `0..n` of a text block built from 4-byte words.
    fn pcs(n: usize) -> Vec<u64> {
        (0..n).map(|i| TEXT_ADDR + 4 * i as u64).collect()
    }

    fn key(vl: Option<u32>, sew: Option<u8>, lmul: Option<Lmul>) -> VtypeKey {
        VtypeKey { vl, sew, lmul }
    }

    fn cache() -> RunCache {
        SYSTEM_VLEN.set_bytes(VLEN_BYTES);
        RunCache::new(VLEN_BYTES)
    }

    /// A run must swallow its control-transfer terminator and stop there, even
    /// when more decodable instructions follow.
    #[test]
    fn run_ends_including_control_transfer() {
        let mut c = cache();
        let d = decoder_for(&[ADDI_X1, ADDI_X1, JAL, ADDI_X1, ADDI_X1]);
        let hit = c
            .lookup(&d, TEXT_ADDR, key(None, None, None))
            .expect("a run starts at the first addi");
        assert_eq!(hit.pcs, &pcs(3)[..], "run must include the jal, then stop");
        assert_eq!(hit.delta.instructions, 3);
    }

    /// A `vsetvl*` changes vtype, so it must never be inside a run: the run ends
    /// before it, and a lookup *at* it reports "no run".
    #[test]
    fn run_ends_before_vsetvl() {
        let mut c = cache();
        let d = decoder_for(&[ADDI_X1, ADDI_X1, VSETVLI, ADDI_X1]);
        let hit = c
            .lookup(&d, TEXT_ADDR, key(None, None, None))
            .expect("a run starts at the first addi");
        assert_eq!(hit.pcs, &pcs(2)[..], "run must stop before the vsetvli");

        assert!(
            c.lookup(&d, TEXT_ADDR + 8, key(None, None, None)).is_none(),
            "no run may start at a vsetvli"
        );
    }

    /// ROI markers change the active region set, so like `vsetvl*` they stay
    /// outside runs.
    #[test]
    fn run_ends_before_marker() {
        let mut c = cache();
        let d = decoder_for(&[ADDI_X1, MARKER_ADD_X0, ADDI_X1]);
        let hit = c
            .lookup(&d, TEXT_ADDR, key(None, None, None))
            .expect("a run starts at the first addi");
        assert_eq!(hit.pcs, &pcs(1)[..], "run must stop before the marker");
        assert!(
            c.lookup(&d, TEXT_ADDR + 4, key(None, None, None)).is_none(),
            "no run may start at an ROI marker"
        );
    }

    /// An unbroken straight-line stretch is truncated at `MAX_RUN_LEN`, so the
    /// validation slice compare can never dominate.
    #[test]
    fn run_respects_length_cap() {
        let mut c = cache();
        let d = decoder_for(&vec![ADDI_X1; MAX_RUN_LEN + 64]);
        let hit = c
            .lookup(&d, TEXT_ADDR, key(None, None, None))
            .expect("a run starts at the first addi");
        assert_eq!(hit.pcs.len(), MAX_RUN_LEN);
        assert_eq!(hit.delta.instructions, MAX_RUN_LEN as u64);
    }

    /// A run also just ends when the text runs out (or fails to decode).
    #[test]
    fn run_ends_at_end_of_text() {
        let mut c = cache();
        let d = decoder_for(&[ADDI_X1, ADDI_X1]);
        let hit = c.lookup(&d, TEXT_ADDR, key(None, None, None)).unwrap();
        assert_eq!(hit.pcs, &pcs(2)[..]);
        assert!(
            c.lookup(&d, TEXT_ADDR + 0x100, key(None, None, None))
                .is_none(),
            "a PC outside the module has no run"
        );
    }

    // --- equivalence -------------------------------------------------------

    /// A mixed instruction sequence exercising every branch of `apply_delta`:
    /// scalar int, scalar float, vector float FMA, vector *integer* FMA (the
    /// `IC_INT_FMA` path, which bumps `fmacc_elements` without
    /// `fmacc_instructions`), a vector element load/store pair, a segment load
    /// (`elem_mult`), whole-register load/store (`whole_reg_elems`, ignoring
    /// `vl`) and mask load/store (`ceil(vl/8)`).
    fn mixed_insns() -> Vec<riscv_isa::Instruction> {
        use riscv_isa::Instruction as I;
        let decode = |code: u32| {
            use std::str::FromStr;
            let target = riscv_isa::Target::from_str("RV64I").unwrap().full();
            riscv_isa::decode_le_bytes(&code.to_le_bytes(), &target)
                .expect("test encoding must decode")
                .0
        };
        vec![
            I::ADDI {
                rd: 1,
                rs1: 1,
                imm: 1,
            },
            I::FADD_S {
                frd: 1,
                rm: 0,
                frs1: 2,
                frs2: 3,
            },
            I::VADD_VV {
                vd: 1,
                vs1: 2,
                vs2: 3,
                vm: 1,
            },
            I::VFMACC_VV {
                vm: 1,
                vs2: 3,
                vs1: 2,
                vd: 1,
            },
            I::VMACC_VV {
                vm: 1,
                vs2: 3,
                vs1: 2,
                vd: 1,
            },
            I::VLE64_V {
                vm: 1,
                rs1: 10,
                vd: 4,
            },
            I::VSE64_V {
                vm: 1,
                rs1: 10,
                vs3: 4,
            },
            decode(0x2205_7207), // vlseg2e64.v  -- elem_mult = 2
            decode(0x0285_0207), // vl1r.v       -- whole-register evl
            decode(0x0285_0227), // vs1r.v
            decode(0x02b5_0207), // vlm.v        -- mask evl = ceil(vl/8)
            decode(0x02b5_0227), // vsm.v
            // A repeated mnemonic, so the run contains the same opcode twice
            // and any per-opcode bookkeeping is exercised above a count of 1.
            I::VADD_VV {
                vd: 5,
                vs1: 6,
                vs2: 7,
                vm: 1,
            },
        ]
    }

    /// Build a `Run` from instructions directly, without a decoder: the
    /// equivalence property is about the *aggregate*, and hand-assembling this
    /// mix into a text block would test the encodings rather than the sum.
    fn run_of(insns: &[riscv_isa::Instruction]) -> Run {
        let pcs: Vec<u64> = (0..insns.len()).map(|i| TEXT_ADDR + 4 * i as u64).collect();
        let ri: Vec<RunInsn> = insns
            .iter()
            .map(|ins| RunInsn {
                ins: *ins,
                class: InsnClass::from_instruction(ins, VLEN_BYTES),
                len: 4,
            })
            .collect();
        Run {
            pcs: pcs.into_boxed_slice(),
            insns: ri.into_boxed_slice(),
            memo: Vec::new(),
            func_index: Cell::new(None),
        }
    }

    /// Assert that one `add_run_delta` and the N `apply_delta` calls it replaces
    /// leave two `RoiInfo`s in the same state -- counter by counter, plus the
    /// four per-SEW histograms.
    ///
    /// This is the property the whole batching plan rests on: if it ever fails,
    /// the fast path is silently reporting different numbers from the slow one.
    fn assert_paths_agree(k: VtypeKey) {
        SYSTEM_VLEN.set_bytes(VLEN_BYTES);
        let insns = mixed_insns();
        let run = run_of(&insns);
        let delta = RunDelta::build(&run.insns, k, VLEN_BYTES);

        let mut batched = RoiInfo::new("t".to_string());
        let mut per_insn = RoiInfo::new("t".to_string());

        batched.add_run_delta(&delta);
        for ins in &insns {
            let class = InsnClass::from_instruction(ins, VLEN_BYTES);
            let d = InsnDelta::new(&class, ins, k.vl, k.sew, k.lmul, VLEN_BYTES);
            per_insn.instructions += 1;
            per_insn.apply_delta(&d);
        }

        let what = format!("{k:?}");
        assert_eq!(batched.instructions, per_insn.instructions, "{what}");
        assert_eq!(batched.branches, per_insn.branches, "{what}");
        assert_eq!(
            batched.vector_instructions, per_insn.vector_instructions,
            "{what}"
        );
        assert_eq!(batched.vector_elements, per_insn.vector_elements, "{what}");
        assert_eq!(
            batched.float_instructions, per_insn.float_instructions,
            "{what}"
        );
        assert_eq!(batched.float_elements, per_insn.float_elements, "{what}");
        assert_eq!(
            batched.fmacc_instructions, per_insn.fmacc_instructions,
            "{what}"
        );
        assert_eq!(batched.fmacc_elements, per_insn.fmacc_elements, "{what}");
        assert_eq!(
            batched.vector_load_instructions, per_insn.vector_load_instructions,
            "{what}"
        );
        assert_eq!(
            batched.vector_load_elements, per_insn.vector_load_elements,
            "{what}"
        );
        assert_eq!(batched.load_bytes, per_insn.load_bytes, "{what}");
        assert_eq!(batched.store_bytes, per_insn.store_bytes, "{what}");

        // Histograms: compare the finalized sparse exports, which cover both the
        // combined and every per-LMUL series.
        batched.update_from_histogram();
        per_insn.update_from_histogram();
        assert_eq!(batched.vl_hist_e8_raw, per_insn.vl_hist_e8_raw, "{what}");
        assert_eq!(batched.vl_hist_e16_raw, per_insn.vl_hist_e16_raw, "{what}");
        assert_eq!(batched.vl_hist_e32_raw, per_insn.vl_hist_e32_raw, "{what}");
        assert_eq!(batched.vl_hist_e64_raw, per_insn.vl_hist_e64_raw, "{what}");
        assert_eq!(
            batched.vl_hist_e32_m2_raw, per_insn.vl_hist_e32_m2_raw,
            "{what}"
        );
        assert_eq!(
            batched.vl_hist_e64_m1_raw, per_insn.vl_hist_e64_m1_raw,
            "{what}"
        );
    }

    #[test]
    fn run_delta_matches_per_insn_scalar_only_vtype() {
        // No vsetvl* has executed yet: vector ops fall back to vl = 1 and the
        // histograms stay untouched.
        assert_paths_agree(key(None, None, None));
    }

    #[test]
    fn run_delta_matches_per_insn_at_several_vls() {
        for vl in [1u32, 4, 7, 8, 64] {
            assert_paths_agree(key(Some(vl), Some(3), Some(Lmul::M1)));
        }
    }

    #[test]
    fn run_delta_matches_per_insn_across_vtypes() {
        // Two genuinely different vtypes: e64/m1 and e32/m2 route to different
        // histograms *and* change the indexed/element byte accounting.
        assert_paths_agree(key(Some(4), Some(3), Some(Lmul::M1)));
        assert_paths_agree(key(Some(16), Some(2), Some(Lmul::M2)));
        // An SEW above e64 must record no histogram bump on either path.
        assert_paths_agree(key(Some(4), Some(5), Some(Lmul::M1)));
    }

    // --- deferred per-PC distribution --------------------------------------

    /// `distribute_pc_profile` must leave the per-PC profile exactly where N
    /// per-instruction `PcProfile::record` calls would have -- the Phase 3
    /// counterpart of `assert_paths_agree`, and the property the annotation
    /// report's `Σ exec_count == RoiInfo.instructions` reconciliation rests on
    /// once the fast path stops recording per instruction.
    #[test]
    fn distributed_pc_profile_matches_per_insn_records() {
        SYSTEM_VLEN.set_bytes(VLEN_BYTES);
        const N: u64 = 7;
        let k = key(Some(16), Some(2), Some(Lmul::M2));
        let insns = mixed_insns();
        let run = run_of(&insns);

        // Batched: one memo entry, N recorded executions, distributed once.
        let mut cache = cache();
        run.func_index.set(Some(3));
        let mut run = run;
        let mut builds = 0;
        let idx = run.memo_index(k, VLEN_BYTES, &mut builds).unwrap();
        run.memo[idx].execs.set(N);
        cache.runs.insert(TEXT_ADDR, Some(run));
        let mut batched = super::super::pc_profile::PcProfile::new(true);
        // The instructions are synthetic, so no decoder can supply their bytes;
        // `encoded_bytes_at` returning `None` is the same on both paths (the
        // per-instruction call below passes an empty slice too), and `bytes` is
        // only ever used for rendering.
        let empty = decoder_for(&[]);
        cache.distribute_pc_profile(&mut batched, &empty);

        // Per-instruction: N passes over the run.
        let mut per_insn = super::super::pc_profile::PcProfile::new(true);
        for _ in 0..N {
            for (i, ins) in insns.iter().enumerate() {
                let class = InsnClass::from_instruction(ins, VLEN_BYTES);
                let d = InsnDelta::new(&class, ins, k.vl, k.sew, k.lmul, VLEN_BYTES);
                per_insn.record(
                    TEXT_ADDR + 4 * i as u64,
                    ins,
                    &[],
                    Some(3),
                    &d,
                    k.lmul,
                    None,
                );
            }
        }

        assert_eq!(batched.records().len(), per_insn.records().len());
        for (pc, want) in per_insn.records() {
            let got = batched.records().get(pc).expect("pc missing from batched");
            assert_eq!(got.exec_count, want.exec_count, "exec_count at {pc:#x}");
            assert_eq!(
                got.vector_elems, want.vector_elems,
                "vector_elems at {pc:#x}"
            );
            assert_eq!(got.vl_min, want.vl_min, "vl_min at {pc:#x}");
            assert_eq!(got.vl_max, want.vl_max, "vl_max at {pc:#x}");
            assert_eq!(got.sew, want.sew, "sew at {pc:#x}");
            assert_eq!(got.lmul, want.lmul, "lmul at {pc:#x}");
            assert_eq!(got.class, want.class, "class at {pc:#x}");
            assert_eq!(got.func_index, want.func_index, "func_index at {pc:#x}");
            assert!(got.vl_hist.is_empty(), "no vsetvl* can be inside a run");
        }

        // Draining: a second distribution must add nothing.
        cache.distribute_pc_profile(&mut batched, &empty);
        for (pc, want) in per_insn.records() {
            assert_eq!(batched.records()[pc].exec_count, want.exec_count);
        }
    }

    // --- memoization -------------------------------------------------------

    /// A second lookup at the same vtype must be served from the memo, and a
    /// different vtype must add a second entry rather than replacing the first.
    #[test]
    fn memo_reuses_same_vtype_and_grows_per_vtype() {
        let mut c = cache();
        let d = decoder_for(&[ADDI_X1, ADDI_X1, JAL]);
        let k1 = key(Some(4), Some(3), Some(Lmul::M1));
        let k2 = key(Some(8), Some(3), Some(Lmul::M1));

        let first = c.lookup(&d, TEXT_ADDR, k1).unwrap().delta.clone();
        assert_eq!(c.deltas_built(), 1);
        assert_eq!(c.runs_built(), 1);

        let second = c.lookup(&d, TEXT_ADDR, k1).unwrap().delta.clone();
        assert_eq!(
            c.deltas_built(),
            1,
            "a repeat lookup at the same vtype must not rebuild the aggregate"
        );
        assert_eq!(second, first);
        assert_eq!(
            c.runs_built(),
            1,
            "a repeat lookup must not re-analyse the run"
        );

        c.lookup(&d, TEXT_ADDR, k2).unwrap();
        assert_eq!(c.deltas_built(), 2);
        assert_eq!(c.run_at(TEXT_ADDR).unwrap().memo_len(), 2);
    }

    /// Beyond `MAX_MEMO_ENTRIES` distinct vtypes the cache reports "no memo",
    /// which the caller must read as "take the per-instruction path".
    #[test]
    fn memo_cap_falls_back_rather_than_growing() {
        let mut c = cache();
        let d = decoder_for(&[ADDI_X1, JAL]);
        for i in 0..MAX_MEMO_ENTRIES {
            let k = key(Some(i as u32 + 1), Some(3), Some(Lmul::M1));
            assert!(c.lookup(&d, TEXT_ADDR, k).is_some(), "entry {i} must fit");
        }
        let overflow = key(Some(MAX_MEMO_ENTRIES as u32 + 1), Some(3), Some(Lmul::M1));
        assert!(
            c.lookup(&d, TEXT_ADDR, overflow).is_none(),
            "the {MAX_MEMO_ENTRIES}th+1 vtype must fall back to the slow path"
        );
        assert_eq!(c.run_at(TEXT_ADDR).unwrap().memo_len(), MAX_MEMO_ENTRIES);
        // Already-memoized vtypes keep working.
        assert!(c
            .lookup(&d, TEXT_ADDR, key(Some(1), Some(3), Some(Lmul::M1)))
            .is_some());
    }
}
