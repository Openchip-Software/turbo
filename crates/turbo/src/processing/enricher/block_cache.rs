//! Per-BBT-block memo: the run cache, keyed by the producer's block identity
//! instead of by a statically guessed run.
//!
//! [`RunCache`](super::run_cache::RunCache) exists because almost nothing the
//! enricher does per instruction depends on dynamic state, so a straight-line
//! stretch's whole counter contribution can be summed once per entry vtype. It
//! has to *guess* that stretch, though: it walks forward from a start PC through
//! the static ELF and predicts which PCs would retire, then compares the
//! prediction against the trace before applying the aggregate. That compare is
//! what makes it sound — and it is also 3 million slice compares per
//! `flash_full` run, against a prediction the producer already told us.
//!
//! A BBT block record *is* the producer's statement, taken from QEMU's
//! translator, of exactly which contiguous PC run retired. So on that path there
//! is nothing to predict and nothing to validate: this module builds its segments
//! **from the block's own PC list**, memoizes them behind the dense block ID, and
//! the per-execution work collapses to one array index, one linear memo scan
//! (usually hitting entry 0) and one `add_run_delta` per scope. No hash probe, no
//! slice compare, no `ElfResolver` call, no decoder lookup.
//!
//! ## Why guessing was not merely wasteful but wrong-sized
//!
//! Measured on `flash_full`, 675 K of 3.26 M block entries had a statically
//! derived run *longer* than the block, so the run cache declined them and their
//! instructions fell to the per-instruction path. QEMU ends a translation block
//! at things a static walk cannot see: a page crossing, its own instruction cap,
//! a fault-only-first vector load (`vle8ff.v` — it can shorten `vl`), a `vl` CSR
//! read. Enumerating QEMU's TB-ending rules in the consumer would be a guess
//! about a guess; taking the block as given removes the question.
//!
//! ## The invariant everything rests on
//!
//! Identical to `run_cache`'s, and deliberately shares its code: a [`RunDelta`]
//! must be *exactly* the sum of the [`InsnDelta`](crate::common::InsnDelta)s
//! the per-instruction path would have produced, so it is built by
//! [`RunDelta::build`] — one [`InsnDelta::new`](crate::common::InsnDelta::new)
//! per instruction, accumulated — and never by re-deriving the arithmetic.
//!
//! ## Block identity is safer than a start PC
//!
//! `run_cache` keys on `start_pc` and decodes from the static ELF, so
//! self-modifying or JIT-generated code silently reuses a stale decode. The
//! producer interns blocks by `(start_pc, hash)`
//! (`turbo_tracer/src/lib.rs:1262`), so rewritten code gets a *new* dictionary
//! ID and therefore a new entry here.

use std::cell::Cell;

use crate::processing::function_lookup::MultiResolver;
use crate::processing::instruction_decoder::MultiDecoder;

use super::run_cache::{MemoEntry, RunDelta, RunInsn, VtypeKey, MAX_MEMO_ENTRIES, MAX_RUN_LEN};
use super::vl_tracker::writes_vl;

/// One block's memo, built on first sighting of its dictionary ID.
struct BlockInfo {
    /// Module (decoder / resolver) index every PC of the block belongs to.
    module_index: usize,
    /// Function the whole block is attributed to, resolved once. `None` means
    /// every PC of the block resolves to no function (checked at intern time —
    /// otherwise the block falls back).
    func: Option<(usize, Box<str>)>,
    /// The block's PCs, copied once per *static* block so the deferred per-PC
    /// profile distribution does not need the dictionary arena. ~7 `u64` per
    /// block over ten thousand blocks, i.e. the copy this module exists to stop
    /// making per *execution*.
    pcs: Box<[u64]>,
    /// The block in commit order: memoizable stretches, and the individual
    /// instructions that must go through the per-instruction path.
    segments: Vec<Segment>,
}

/// A stretch of the block, in commit order.
enum Segment {
    /// A memoizable stretch: no `vsetvl*` and no ROI marker inside, so the live
    /// vtype and the active-scope set are constant across it.
    Run(RunSeg),
    /// One instruction, at this index within the block, that must commit
    /// individually: an ROI marker (it changes the active region set) or a
    /// `vsetvl*` (it changes vtype and consumes from the `vl` stream in order).
    Single(u16),
}

/// A memoizable stretch of a block.
///
/// Note there is no `pcs` field to compare against the trace: the PCs came *from*
/// the trace's own block record. `off`/`len` index the owning [`BlockInfo::pcs`].
struct RunSeg {
    off: u16,
    len: u16,
    /// Static per-instruction data, parallel to the segment's PCs.
    insns: Box<[RunInsn]>,
    /// Memoized aggregates by entry vtype. Almost always one entry, so a linear
    /// scan beats hashing.
    memo: Vec<MemoEntry>,
}

/// What one segment is, cheaply readable without holding a borrow on the cache.
pub(crate) enum SegmentKind {
    /// Memoizable stretch: `(index within the block, instruction count)`.
    Run { off: u16, len: u16 },
    /// Single instruction at this index within the block.
    Single(u16),
}

/// Memoized aggregates for every BBT block seen so far, indexed by dictionary ID.
///
/// One cache for the whole hart rather than one per module (as `run_cache` needs):
/// a dictionary ID identifies a block in a specific module, so there is no
/// cross-module aliasing to design around.
pub(crate) struct BlockCache {
    /// Dense by dictionary ID. `None` = not yet interned, `Some(None)` = interned
    /// and unusable (see [`BlockCache::intern`]), so a hopeless block is only
    /// analysed once.
    blocks: Vec<Option<Option<BlockInfo>>>,
    /// Run-fixed `SYSTEM_VLEN`, folded into every aggregate built here. Held by
    /// the cache rather than passed per lookup so a caller cannot mix VLENs
    /// within one set of memos.
    vlen_bytes: u16,
    blocks_interned: u64,
    blocks_unusable: u64,
    deltas_built: u64,
}

impl BlockCache {
    pub fn new(vlen_bytes: u16) -> Self {
        Self {
            blocks: Vec::new(),
            vlen_bytes,
            blocks_interned: 0,
            blocks_unusable: 0,
            deltas_built: 0,
        }
    }

    /// The module index of block `id`, interning it on first sighting.
    ///
    /// `None` means the block cannot be memoized and every execution of it must
    /// go through the per-instruction path -- which is *always* correct, just
    /// slower. Callers must treat `None` as routine.
    pub fn module_index(
        &mut self,
        id: u32,
        pcs: &[u64],
        decoders: &MultiDecoder,
        resolver: &MultiResolver,
    ) -> Option<usize> {
        let idx = id as usize;
        if self.blocks.len() <= idx {
            self.blocks.resize_with(idx + 1, || None);
        }
        if self.blocks[idx].is_none() {
            self.blocks_interned += 1;
            let info = self.intern(pcs, decoders, resolver);
            if info.is_none() {
                self.blocks_unusable += 1;
            }
            self.blocks[idx] = Some(info);
        }
        Some(self.blocks[idx].as_ref()?.as_ref()?.module_index)
    }

    /// Number of segments block `id` was split into. Blocks that are one
    /// straight-line stretch -- the overwhelming majority -- have exactly one.
    pub fn segment_count(&self, id: u32) -> usize {
        match self.blocks.get(id as usize) {
            Some(Some(Some(info))) => info.segments.len(),
            _ => 0,
        }
    }

    /// What segment `si` of block `id` is.
    pub fn segment_kind(&self, id: u32, si: usize) -> SegmentKind {
        match &self.blocks[id as usize]
            .as_ref()
            .unwrap()
            .as_ref()
            .unwrap()
            .segments[si]
        {
            Segment::Run(seg) => SegmentKind::Run {
                off: seg.off,
                len: seg.len,
            },
            Segment::Single(off) => SegmentKind::Single(*off),
        }
    }

    /// The aggregate for segment `si` of block `id` at `key`, counting one
    /// execution of it.
    ///
    /// `None` means the segment's memo is full (more than [`MAX_MEMO_ENTRIES`]
    /// distinct entry vtypes), so this execution must take the per-instruction
    /// path. Nothing is counted in that case.
    pub fn apply(
        &mut self,
        id: u32,
        si: usize,
        key: VtypeKey,
    ) -> Option<(&RunDelta, u16, Option<usize>, Option<&str>)> {
        let vlen_bytes = self.vlen_bytes;
        let deltas_built = &mut self.deltas_built;
        let info = self.blocks[id as usize].as_mut()?.as_mut()?;
        let func = info.func.as_ref();
        let Segment::Run(seg) = &mut info.segments[si] else {
            return None;
        };

        let i = match seg.memo.iter().position(|e| e.key == key) {
            Some(i) => i,
            None => {
                if seg.memo.len() >= MAX_MEMO_ENTRIES {
                    return None;
                }
                *deltas_built += 1;
                seg.memo.push(MemoEntry {
                    key,
                    delta: RunDelta::build(&seg.insns, key, vlen_bytes),
                    execs: Cell::new(0),
                });
                seg.memo.len() - 1
            }
        };
        let entry = &seg.memo[i];
        entry.execs.set(entry.execs.get() + 1);
        Some((
            &entry.delta,
            seg.len,
            func.map(|(index, _)| *index),
            func.map(|(_, name)| &**name),
        ))
    }

    /// Split one block into segments, resolving its module and function once.
    ///
    /// Returns `None` -- "this block can never be memoized" -- when any of the
    /// assumptions the aggregate rests on fails to hold for it:
    ///
    /// * a PC is undecodable or outside every module, so its contribution cannot
    ///   be summed at all;
    /// * the block spans two modules, so one decoder/resolver does not cover it;
    /// * the block spans two functions (or a resolved and an unresolved region),
    ///   so attributing the whole aggregate to one function would be wrong. This
    ///   is `run_cache`'s `func_straddle` check, hoisted to intern time: a block
    ///   cannot leave a function without a control transfer, but
    ///   overlapping/misreported ELF symbol ranges are a property of the ELF, not
    ///   something to assume away;
    /// * the block is longer than [`MAX_RUN_LEN`] (a QEMU TB cannot be, but it is
    ///   asserted rather than assumed).
    fn intern(
        &self,
        pcs: &[u64],
        decoders: &MultiDecoder,
        resolver: &MultiResolver,
    ) -> Option<BlockInfo> {
        if pcs.is_empty() || pcs.len() > MAX_RUN_LEN {
            log::warn!(
                "BBT block of {} instructions cannot be memoized (cap {MAX_RUN_LEN}); \
                 every execution of it will take the per-instruction path",
                pcs.len()
            );
            return None;
        }

        let module_index = decoders.find_decoder_index_for(pcs[0])?;
        let decoder = &decoders.dec[module_index];

        // One function for the whole block, by the same rule the
        // per-instruction path applies per PC.
        let func = resolver
            .resolve_in_module(module_index, pcs[0])
            .map(|(name, index, _range)| (index, name.to_owned().into_boxed_str()));
        for &pc in pcs {
            if decoders.find_decoder_index_for(pc) != Some(module_index) {
                return None;
            }
            let here = resolver
                .resolve_in_module(module_index, pc)
                .map(|(_, i, _)| i);
            if here != func.as_ref().map(|(i, _)| *i) {
                return None;
            }
        }

        // Split at the instructions whose dynamic effect a memoized aggregate
        // cannot carry: a `vsetvl*` changes vtype, an ROI marker changes the
        // active-scope set. Everything else -- including a control transfer, which
        // by construction can only be the block's last instruction -- just
        // accumulates, because the PCs are the producer's record of what retired
        // rather than a static prediction of what would.
        let mut segments: Vec<Segment> = Vec::new();
        let mut run_off: u16 = 0;
        let mut insns: Vec<RunInsn> = Vec::new();
        for (i, &pc) in pcs.iter().enumerate() {
            let (ins, bytes, class, marker) =
                decoder.get_instruction_class_and_marker_at(pc).ok()?;
            if marker.is_some() || writes_vl(&ins) {
                if !insns.is_empty() {
                    segments.push(Segment::Run(RunSeg {
                        off: run_off,
                        len: insns.len() as u16,
                        insns: std::mem::take(&mut insns).into_boxed_slice(),
                        memo: Vec::new(),
                    }));
                }
                segments.push(Segment::Single(i as u16));
                run_off = i as u16 + 1;
                continue;
            }
            insns.push(RunInsn {
                ins,
                class,
                len: bytes.len() as u8,
            });
        }
        if !insns.is_empty() {
            segments.push(Segment::Run(RunSeg {
                off: run_off,
                len: insns.len() as u16,
                insns: insns.into_boxed_slice(),
                memo: Vec::new(),
            }));
        }

        Some(BlockInfo {
            module_index,
            func,
            pcs: pcs.to_vec().into_boxed_slice(),
            segments,
        })
    }

    /// Multiply every block's per-PC contributions out into `profile`, once, from
    /// the `(vtype, executions)` pairs accumulated during the run.
    ///
    /// The deferred half of the batched path, exactly as
    /// [`RunCache::distribute_pc_profile`](super::run_cache::RunCache::distribute_pc_profile):
    /// instead of recording per retired instruction, the hot path counts segment
    /// executions and this walks the *static* table at finalisation, rebuilding
    /// each per-instruction delta with the same `InsnDelta::new` call the
    /// per-instruction path makes.
    ///
    /// Drains the counters, so calling it twice adds nothing the second time.
    pub fn distribute_pc_profile(
        &self,
        profile: &mut super::pc_profile::PcProfile,
        decoders: &MultiDecoder,
    ) {
        for info in self.blocks.iter().flatten().flatten() {
            let Some(decoder) = decoders.dec.get(info.module_index) else {
                continue;
            };
            let func_index = info.func.as_ref().map(|(i, _)| *i);
            for segment in &info.segments {
                let Segment::Run(seg) = segment else { continue };
                let pcs = &info.pcs[seg.off as usize..seg.off as usize + seg.len as usize];
                for entry in &seg.memo {
                    let n = entry.execs.replace(0);
                    if n == 0 {
                        continue;
                    }
                    for (pc, insn) in pcs.iter().zip(seg.insns.iter()) {
                        let delta = crate::common::InsnDelta::new(
                            &insn.class,
                            &insn.ins,
                            entry.key.vl,
                            entry.key.sew,
                            entry.key.lmul,
                            self.vlen_bytes,
                        );
                        let bytes = decoder.encoded_bytes_at(*pc, insn.len).unwrap_or(&[]);
                        profile.record_n(
                            *pc,
                            &insn.ins,
                            bytes,
                            func_index,
                            &delta,
                            entry.key.lmul,
                            n,
                        );
                    }
                }
            }
        }
    }

    /// Distinct static blocks analysed.
    pub fn blocks_interned(&self) -> u64 {
        self.blocks_interned
    }

    /// Blocks found unusable, i.e. permanently routed to the per-instruction
    /// path. Should be ~0; see [`BlockCache::intern`] for what causes it.
    pub fn blocks_unusable(&self) -> u64 {
        self.blocks_unusable
    }

    /// `(segment, vtype)` aggregates constructed.
    pub fn deltas_built(&self) -> u64 {
        self.deltas_built
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::SYSTEM_VLEN;
    use crate::processing::instruction_decoder::InstructionDecoder;

    const VLEN_BYTES: u16 = 16;
    const TEXT_ADDR: u64 = 0x1000;

    /// `addi x1, x1, 1` -- plain scalar, not a marker (rd != x0).
    const ADDI_X1: u32 = 0x0010_8093;
    /// `jal x1, +0x10` -- a control transfer, so only ever a block's last insn.
    const JAL: u32 = 0x0100_00ef;
    /// `vsetvli x0, x0, e32, m1` -- splits a block.
    const VSETVLI: u32 = 0x0100_7057;
    /// `add x0, x1, x2` -- writes x0, so the decoder reads it as an ROI
    /// begin-region marker; splits a block.
    const MARKER_ADD_X0: u32 = 0x0020_8033;

    fn decoders_for(words: &[u32]) -> MultiDecoder {
        let mut bytes = Vec::with_capacity(words.len() * 4);
        for w in words {
            bytes.extend_from_slice(&w.to_le_bytes());
        }
        MultiDecoder {
            dec: vec![InstructionDecoder::from_exec_bytes(TEXT_ADDR, &bytes)],
        }
    }

    fn pcs(n: usize) -> Vec<u64> {
        (0..n).map(|i| TEXT_ADDR + 4 * i as u64).collect()
    }

    fn cache() -> BlockCache {
        SYSTEM_VLEN.set_bytes(VLEN_BYTES);
        BlockCache::new(VLEN_BYTES)
    }

    fn key() -> VtypeKey {
        VtypeKey {
            vl: Some(4),
            sew: Some(3),
            lmul: Some(crate::common::Lmul::M1),
        }
    }

    fn segments(c: &BlockCache, id: u32) -> Vec<(bool, u16, u16)> {
        (0..c.segment_count(id))
            .map(|si| match c.segment_kind(id, si) {
                SegmentKind::Run { off, len } => (true, off, len),
                SegmentKind::Single(off) => (false, off, 1),
            })
            .collect()
    }

    /// The ordinary case: a whole block is one memoizable stretch, control
    /// transfer included. A run cache walking the ELF would have to *predict*
    /// this; here the block record states it.
    #[test]
    fn a_straight_line_block_is_one_segment() {
        let mut c = cache();
        let d = decoders_for(&[ADDI_X1, ADDI_X1, JAL]);
        let r = MultiResolver::new(vec![
            crate::processing::function_lookup::ElfResolver::empty(),
        ]);
        assert_eq!(c.module_index(0, &pcs(3), &d, &r), Some(0));
        assert_eq!(segments(&c, 0), vec![(true, 0, 3)]);
        let (delta, len, _, _) = c.apply(0, 0, key()).expect("memo must build");
        assert_eq!(delta.instructions, 3);
        assert_eq!(len, 3);
    }

    /// A block *shorter* than the run the ELF implies -- QEMU cut it at a page
    /// crossing, its instruction cap or a fault-only-first load -- is the case
    /// that makes `run_cache` decline 675K block entries per `flash_full` run.
    /// Here it is simply a 2-instruction block, because that is what retired.
    #[test]
    fn a_block_cut_short_of_the_static_run_is_still_one_segment() {
        let mut c = cache();
        // Six decodable instructions in the module, but the block says two.
        let d = decoders_for(&[ADDI_X1; 6]);
        let r = MultiResolver::new(vec![
            crate::processing::function_lookup::ElfResolver::empty(),
        ]);
        c.module_index(0, &pcs(2), &d, &r).unwrap();
        assert_eq!(segments(&c, 0), vec![(true, 0, 2)]);
        let (delta, len, _, _) = c.apply(0, 0, key()).unwrap();
        assert_eq!(delta.instructions, 2, "only what the block retired");
        assert_eq!(len, 2);
    }

    /// A `vsetvl*` changes vtype and consumes from the `vl` stream in order, so
    /// it must commit individually and split the stretches around it.
    #[test]
    fn a_vsetvl_splits_the_block() {
        let mut c = cache();
        let d = decoders_for(&[ADDI_X1, VSETVLI, ADDI_X1, ADDI_X1]);
        let r = MultiResolver::new(vec![
            crate::processing::function_lookup::ElfResolver::empty(),
        ]);
        c.module_index(0, &pcs(4), &d, &r).unwrap();
        assert_eq!(
            segments(&c, 0),
            vec![(true, 0, 1), (false, 1, 1), (true, 2, 2)]
        );
    }

    /// A ROI marker changes the active-scope set, so likewise.
    #[test]
    fn a_marker_splits_the_block() {
        let mut c = cache();
        let d = decoders_for(&[ADDI_X1, ADDI_X1, MARKER_ADD_X0, ADDI_X1]);
        let r = MultiResolver::new(vec![
            crate::processing::function_lookup::ElfResolver::empty(),
        ]);
        c.module_index(0, &pcs(4), &d, &r).unwrap();
        assert_eq!(
            segments(&c, 0),
            vec![(true, 0, 2), (false, 2, 1), (true, 3, 1)]
        );
    }

    /// A block that is nothing but a `vsetvl*` has no memoizable stretch at all.
    #[test]
    fn a_vsetvl_only_block_has_no_run_segment() {
        let mut c = cache();
        let d = decoders_for(&[VSETVLI]);
        let r = MultiResolver::new(vec![
            crate::processing::function_lookup::ElfResolver::empty(),
        ]);
        c.module_index(0, &pcs(1), &d, &r).unwrap();
        assert_eq!(segments(&c, 0), vec![(false, 0, 1)]);
        assert!(c.apply(0, 0, key()).is_none(), "a Single has no aggregate");
    }

    /// An undecodable PC (here: outside the module) means the block's
    /// contribution cannot be summed, so it is permanently routed to the
    /// per-instruction path rather than approximated.
    #[test]
    fn an_uncovered_block_is_unusable_and_analysed_once() {
        let mut c = cache();
        let d = decoders_for(&[ADDI_X1]);
        let r = MultiResolver::new(vec![
            crate::processing::function_lookup::ElfResolver::empty(),
        ]);
        assert_eq!(c.module_index(0, &[TEXT_ADDR + 0x10_000], &d, &r), None);
        assert_eq!(c.module_index(0, &[TEXT_ADDR + 0x10_000], &d, &r), None);
        assert_eq!(c.blocks_interned(), 1, "a hopeless block is analysed once");
        assert_eq!(c.blocks_unusable(), 1);
    }

    /// A repeat execution at the same vtype must be served from the memo; a
    /// different vtype adds an entry rather than replacing one.
    #[test]
    fn memo_reuses_same_vtype_and_grows_per_vtype() {
        let mut c = cache();
        let d = decoders_for(&[ADDI_X1, JAL]);
        let r = MultiResolver::new(vec![
            crate::processing::function_lookup::ElfResolver::empty(),
        ]);
        c.module_index(0, &pcs(2), &d, &r).unwrap();

        let first = c.apply(0, 0, key()).unwrap().0.clone();
        assert_eq!(c.deltas_built(), 1);
        let second = c.apply(0, 0, key()).unwrap().0.clone();
        assert_eq!(
            c.deltas_built(),
            1,
            "a repeat must not rebuild the aggregate"
        );
        assert_eq!(first, second);

        let other = VtypeKey {
            vl: Some(8),
            ..key()
        };
        c.apply(0, 0, other).unwrap();
        assert_eq!(c.deltas_built(), 2);
    }

    /// Beyond `MAX_MEMO_ENTRIES` distinct vtypes the cache reports "no memo",
    /// which the caller must read as "take the per-instruction path".
    #[test]
    fn memo_cap_falls_back_rather_than_growing() {
        let mut c = cache();
        let d = decoders_for(&[ADDI_X1, JAL]);
        let r = MultiResolver::new(vec![
            crate::processing::function_lookup::ElfResolver::empty(),
        ]);
        c.module_index(0, &pcs(2), &d, &r).unwrap();
        for i in 0..MAX_MEMO_ENTRIES {
            let k = VtypeKey {
                vl: Some(i as u32 + 1),
                ..key()
            };
            assert!(c.apply(0, 0, k).is_some(), "entry {i} must fit");
        }
        let overflow = VtypeKey {
            vl: Some(MAX_MEMO_ENTRIES as u32 + 1),
            ..key()
        };
        assert!(c.apply(0, 0, overflow).is_none());
        // Already-memoized vtypes keep working.
        assert!(c
            .apply(
                0,
                0,
                VtypeKey {
                    vl: Some(1),
                    ..key()
                }
            )
            .is_some());
    }

    /// The Phase-2 counterpart of `run_cache`'s `distributed_pc_profile_matches_
    /// per_insn_records`: the deferred distribution must leave the per-PC profile
    /// exactly where N per-instruction `record` calls would have, which is what
    /// the annotation report's `Σ exec_count == RoiInfo.instructions`
    /// reconciliation rests on.
    #[test]
    fn distributed_pc_profile_matches_per_insn_records() {
        let mut c = cache();
        const N: u64 = 7;
        let k = key();
        let d = decoders_for(&[ADDI_X1, ADDI_X1, JAL]);
        let r = MultiResolver::new(vec![
            crate::processing::function_lookup::ElfResolver::empty(),
        ]);
        c.module_index(0, &pcs(3), &d, &r).unwrap();
        for _ in 0..N {
            c.apply(0, 0, k).unwrap();
        }

        let mut batched = super::super::pc_profile::PcProfile::new(true);
        c.distribute_pc_profile(&mut batched, &d);

        let mut per_insn = super::super::pc_profile::PcProfile::new(true);
        let decoder = &d.dec[0];
        for _ in 0..N {
            for &pc in pcs(3).iter() {
                let (ins, bytes, class, _) =
                    decoder.get_instruction_class_and_marker_at(pc).unwrap();
                let delta =
                    crate::common::InsnDelta::new(&class, &ins, k.vl, k.sew, k.lmul, VLEN_BYTES);
                per_insn.record(pc, &ins, bytes, None, &delta, k.lmul, None);
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
            assert_eq!(got.class, want.class, "class at {pc:#x}");
        }

        // Draining: a second distribution must add nothing.
        c.distribute_pc_profile(&mut batched, &d);
        for (pc, want) in per_insn.records() {
            assert_eq!(batched.records()[pc].exec_count, want.exec_count);
        }
    }
}
