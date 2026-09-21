//! Replaying a trace's captured vector memory addresses as cache-line accesses.
//!
//! This is the consumer half of address capture: the tracer captures what only
//! the running guest knows (the base pointer of every vector load/store, and the
//! index vector of every gather/scatter), and this turns that back into the
//! sequence of cache lines the workload touched — the input a cache simulator
//! needs.
//!
//! # How a group is matched to its instruction
//!
//! Nothing in the trace names the instruction a captured address belongs to, and
//! nothing needs to: address groups follow their block's record in program
//! order, so walking the block's PCs and pulling the next group at each vector
//! memory op pairs them exactly. That the two sides agree about *which*
//! instructions are vector memory ops is enforced rather than assumed — both
//! classify with [`VecMemOp::decode`], and a block whose PCs and groups do not
//! line up is an error here, not a silent misattribution of every later address.
//!
//! # Fidelity
//!
//! * **Masked elements are included.** An op with `vm = 0` skips its masked-off
//!   elements; this reports the lines they would have touched. That makes the
//!   output a superset of the true footprint — a cache model over-counts rather
//!   than missing traffic.
//! * **`vle*ff` reports its requested `vl`.** A fault-trimmed load touches fewer
//!   lines than reported. The trimmed `vl` is not observable through the plugin
//!   API.
//! * **Scalar loads and stores are absent.** They move cache lines too; see the
//!   plan's phase 7.

use rustc_hash::FxHashMap;
use std::path::Path;

use std::sync::Arc;

use anyhow::{bail, Context, Result};

use turbo_tracer_shared::bbt::{bbt_sign_extend_addr, MemGroupKind};
use turbo_tracer_shared::vmem::{
    line_of, lines_of, vsetvl_static_sew_bits, SegmentShape, VecMemOp, CACHE_LINE_BYTES,
};

use crate::processing::cache_model::{AccessKind, Footprint};

use crate::processing::data_processor::MemGroup;
use crate::processing::enricher::Modules;
use crate::processing::{BbtBlock, DataProcessor, ElfMetadata, TraceBatch, TraceFormat};

/// One cache line touched by one instruction.
///
/// A line rather than a byte range because that is the granularity a cache
/// model works in, and because it is the granularity the indexed capture is
/// forced to use anyway (it deduplicates by line inside the tracer, at the only
/// point the index vector exists).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MemAccess {
    /// PC of the instruction that touched it.
    pub pc: u64,
    /// A store rather than a load.
    pub is_store: bool,
    /// Cache-line-aligned address.
    pub line: u64,
}

/// Replays the address groups of a decoded BBT stream.
///
/// Holds the `vl`/`SEW` state the expansion needs, carried across blocks exactly
/// as the guest carries it: a `vsetvl*` is the last instruction of its block, so
/// its `vl` (the block's trailing record) and its `SEW` (its immediate) apply
/// from the *next* block onwards, never to memory ops in its own.
pub struct MemReplay {
    modules: Arc<Modules>,
    /// `VLEN` in bytes, for whole-register ops whose size is `vl`-independent.
    vlenb: u64,
    /// Current architectural vector length.
    vl: u64,
    /// Current `SEW` in bits. Only indexed ops read it, and one always follows a
    /// `vsetvl*`, so the initial value is never used by a well-formed program.
    sew_bits: u16,
    /// Per-PC decode memo: `None` records "looked, not a vector memory op",
    /// which is the common answer and the one worth not re-deriving.
    ops: FxHashMap<u64, Option<VecMemOp>>,
    /// Per-block-id memo of [`BlockPlan`], indexed by `BbtBlock::id` — which is
    /// dense precisely so a consumer can do this behind a plain array index.
    plans: Vec<Option<Arc<BlockPlan>>>,
    /// Per-instruction line set, reused so a replay of a long trace does not
    /// allocate per instruction.
    lines: Vec<u64>,
}

/// Everything about one *static* block that the replay walk would otherwise
/// re-derive on every execution of it.
///
/// Which of a block's PCs are vector memory ops, what op each one is, and what
/// `SEW` the block's trailing `vsetvl*` establishes are all functions of the
/// instruction words alone — so they are properties of the block id, not of this
/// execution of it. Deriving them per execution meant a `find_decoder_for` plus
/// a decode-memo probe **per retired instruction**, making the cost of
/// `--cache-sim` scale with the length of the trace rather than with the number
/// of accesses being modelled. With this, a re-executed block touches only its
/// vector memory instructions.
struct BlockPlan {
    /// `(index into `BbtBlock::pcs`, op)` for each vector memory instruction, in
    /// program order. Indices rather than PCs so the block's own `pcs` slice
    /// stays the single source of truth for the address.
    ops: Vec<(u32, VecMemOp)>,
    /// `SEW` established by the `vsetvl*` ending this block, if it has one with
    /// a statically known `SEW`. Applies from the *next* block onwards.
    pending_sew: Option<u16>,
    /// Whether this block's last instruction is a fault-only-first load, i.e.
    /// whether the block's trailing `vl` was written by that load rather than by
    /// a `vsetvl*`.
    ///
    /// It changes *when* the trailing `vl` is applied. A `vsetvl*`'s takes effect
    /// from the next block; a `vl*ff`'s takes effect for the ff itself, because
    /// the trim means the elements at and past the fault were never accessed and
    /// so are not part of its footprint. That load is necessarily the last entry
    /// in [`BlockPlan::ops`] as well as the last PC, since it ends its block.
    ends_in_ff: bool,
}

impl MemReplay {
    pub fn new(modules: Arc<Modules>, vlenb: u64) -> Self {
        Self {
            modules,
            vlenb,
            vl: 0,
            sew_bits: 8,
            ops: FxHashMap::default(),
            plans: Vec::new(),
            lines: Vec::new(),
        }
    }

    /// The instruction word at `pc`, or `None` if no module covers it.
    ///
    /// Two-byte (compressed) instructions can read four bytes here and pick up
    /// the next instruction's low half; that is harmless, because every
    /// compressed encoding has `insn[1:0] != 0b11` and both vector memory
    /// opcodes and `vsetvl*` require `0b11`, so a compressed instruction can
    /// never be mistaken for either.
    fn word_at(&self, pc: u64) -> Option<u32> {
        let dec = self.modules.decoder.find_decoder_for(pc)?;
        let bytes = dec.encoded_bytes_at(pc, 4)?;
        Some(u32::from_le_bytes(bytes.try_into().ok()?))
    }

    /// Replay one block, calling `on_access` for every cache line its vector
    /// memory instructions touched, in program order.
    ///
    /// The line-level view. [`MemReplay::block_footprints`] is the same walk
    /// without the flattening, and is what a cache model wants; this is
    /// implemented on top of it so the two cannot disagree about which
    /// instructions have footprints or which lines they cover.
    pub fn block(
        &mut self,
        block: &BbtBlock<'_>,
        mut on_access: impl FnMut(MemAccess),
    ) -> Result<()> {
        let mut buf = Vec::new();
        self.block_footprints(block, |pc, kind, fp| {
            buf.clear();
            fp.push_lines(&mut buf);
            buf.sort_unstable();
            buf.dedup();
            for &line in &buf {
                debug_assert_eq!(line % CACHE_LINE_BYTES, 0);
                on_access(MemAccess {
                    pc,
                    is_store: kind == AccessKind::Store,
                    line,
                });
            }
        })
    }

    /// The [`BlockPlan`] for `block`, derived on first sight of its id and
    /// memoized thereafter.
    ///
    /// Returns an `Arc` rather than a borrow so the walk can hold the plan while
    /// still calling `&mut self` methods on the replay; the plan is immutable
    /// once built, and a refcount bump per block is nothing next to the per-PC
    /// decode walk it replaces.
    fn plan_for(&mut self, block: &BbtBlock<'_>) -> Arc<BlockPlan> {
        let id = block.id as usize;
        if self.plans.len() <= id {
            self.plans.resize_with(id + 1, || None);
        }
        if let Some(plan) = &self.plans[id] {
            return Arc::clone(plan);
        }

        let mut ops = Vec::new();
        let mut pending_sew = None;
        for (off, &pc) in block.pcs.iter().enumerate() {
            let word = self.word_at(pc);
            if let Some(sew) = word.and_then(vsetvl_static_sew_bits) {
                // Applies from the next block: a `vsetvl*` ends its block.
                pending_sew = Some(sew);
                continue;
            }
            // The per-PC memo still earns its keep here: the same PC can appear
            // in more than one static block, and this is the only site left that
            // reads it.
            let op = match self.ops.entry(pc) {
                std::collections::hash_map::Entry::Occupied(e) => *e.get(),
                std::collections::hash_map::Entry::Vacant(e) => *e.insert(word.and_then(|w| {
                    // `decode` panics on the reserved `mew` bit, which cannot be
                    // reached from a stream a QEMU that ran the program produced.
                    VecMemOp::decode(w)
                })),
            };
            if let Some(op) = op {
                ops.push((off as u32, op));
            }
        }

        let ends_in_ff = ops
            .last()
            .is_some_and(|&(off, op)| op.writes_vl() && off as usize + 1 == block.pcs.len());
        let plan = Arc::new(BlockPlan {
            ops,
            pending_sew,
            ends_in_ff,
        });
        self.plans[id] = Some(Arc::clone(&plan));
        plan
    }

    /// Replay one block, calling `f` once per vector memory instruction with the
    /// footprint it touched, in program order.
    ///
    /// Per instruction, not per line: a cache model that takes a footprint
    /// resolves a whole instruction in one call, and the value it returns is
    /// then that instruction's own cache behaviour — which is what makes per-PC
    /// attribution free.
    ///
    /// The slice behind [`Footprint::Lines`] borrows this replay's scratch
    /// buffer, so a block full of gathers allocates nothing.
    pub fn block_footprints(
        &mut self,
        block: &BbtBlock<'_>,
        mut f: impl FnMut(u64, AccessKind, Footprint<'_>),
    ) -> Result<()> {
        let mut groups = block.mem;
        let plan = self.plan_for(block);

        for (i, &(off, op)) in plan.ops.iter().enumerate() {
            // A trailing `vl` from a `vl*ff` applies to the ff itself, so it is
            // installed before that op's footprint rather than after the block.
            if plan.ends_in_ff && i + 1 == plan.ops.len() {
                if let Some(vl) = block.vl {
                    self.vl = u64::from(vl);
                }
            }
            let pc = block.pcs[off as usize];
            let Some(group) = groups.next() else {
                bail!(
                    "no captured addresses for the vector memory op at PC {pc:#x}. The \
                     producer emits one group per such instruction, so either the two \
                     sides disagree about what a vector memory op is, or the stream \
                     lost records."
                );
            };
            self.emit_footprint(pc, &op, &group, &mut f)?;
        }

        if groups.next().is_some() {
            bail!(
                "block {} carries more captured address groups than it has vector memory \
                 instructions. Every later group would be attributed to the wrong \
                 instruction. Block PCs and the words this consumer decoded there: {:#x?}",
                block.id,
                block
                    .pcs
                    .iter()
                    .map(|&pc| (pc, self.word_at(pc)))
                    .collect::<Vec<_>>()
            );
        }

        // `vl` and `SEW` both take effect after the `vsetvl*` that ends the
        // block, i.e. from here on -- unless the block ends in a `vl*ff`, whose
        // `vl` the loop above already installed for the ff's own footprint.
        if let Some(vl) = block.vl.filter(|_| !plan.ends_in_ff) {
            self.vl = u64::from(vl);
        }
        if let Some(sew) = plan.pending_sew {
            self.sew_bits = sew;
        }
        Ok(())
    }

    /// Turn one instruction's captured group into the footprint it touched, and
    /// hand it to `f`.
    fn emit_footprint(
        &mut self,
        pc: u64,
        op: &VecMemOp,
        group: &MemGroup<'_>,
        f: &mut impl FnMut(u64, AccessKind, Footprint<'_>),
    ) -> Result<()> {
        if group.kind != op.group_kind() {
            bail!(
                "the vector memory op at PC {pc:#x} needs a {:?} capture, but the trace \
                 carries a {:?} group for it. Producer and consumer classify this \
                 instruction differently.",
                op.group_kind(),
                group.kind
            );
        }

        let kind = if op.is_store {
            AccessKind::Store
        } else {
            AccessKind::Load
        };

        match group.kind {
            // Already a deduplicated line set: the tracer had to expand it while
            // the index vector existed, so it passes straight through.
            MemGroupKind::Lines => {
                // Taken out of `self` so the callback can borrow the slice while
                // `self` stays available to the walk that called us.
                let mut lines = std::mem::take(&mut self.lines);
                lines.clear();
                lines.extend(group.values().map(line_of));
                f(pc, kind, Footprint::Lines(&lines));
                self.lines = lines;
            }
            // Derivable offline, and passed on *as* a shape: a strided op stays
            // one `Strided` footprint rather than becoming `vl` ranges, which is
            // the difference between one call per instruction and one per
            // element.
            MemGroupKind::Base | MemGroupKind::BaseStride => {
                let mut values = group.values();
                let base = values.next().unwrap_or(0);
                let stride = values.next().map_or(0, bbt_sign_extend_addr);
                let segments = op
                    .segments(base, stride, self.vl, self.sew_bits, self.vlenb)
                    .expect(
                        "only an indexed op has no derivable footprint, and it is a Lines group",
                    );
                let fp = match segments.shape() {
                    SegmentShape::Contiguous { addr, len } => Footprint::Range { addr, len },
                    SegmentShape::Strided {
                        base,
                        stride,
                        count,
                        len,
                    } => Footprint::Strided {
                        base,
                        stride,
                        count,
                        len,
                    },
                };
                f(pc, kind, fp);
            }
        }
        Ok(())
    }
}

/// The lines a [`Footprint`] covers, for consumers that want them enumerated.
///
/// Lives here rather than on `Footprint` itself because it is the *consumer's*
/// flattening: the model's own line walk is its business, and a footprint that
/// stayed folded all the way into the model is the point of the type.
trait FootprintLines {
    fn push_lines(&self, out: &mut Vec<u64>);
}

impl FootprintLines for Footprint<'_> {
    fn push_lines(&self, out: &mut Vec<u64>) {
        match *self {
            Footprint::Range { addr, len } => out.extend(lines_of(addr, len)),
            Footprint::Strided {
                base,
                stride,
                count,
                len,
            } => {
                for i in 0..count {
                    let addr = base.wrapping_add((i as i64).wrapping_mul(stride) as u64);
                    out.extend(lines_of(addr, len));
                }
            }
            Footprint::Lines(lines) => out.extend_from_slice(lines),
        }
    }
}

/// Replay one recorded per-hart trace file into its cache-line accesses.
///
/// The end-to-end entry point for "a later component replays the trace and runs
/// a cache simulation on it": give it the `.bin` a run produced and the guest
/// ELF it was recorded from, and it hands back every line access in execution
/// order.
///
/// Returns them in a `Vec`, which bounds this to traces that fit in memory. A
/// streaming form is a straightforward variant of the same loop, and would be
/// the one a simulator over a long run should use.
pub fn replay_trace_file(trace_path: &Path, elf_path: &Path, vlenb: u64) -> Result<Vec<MemAccess>> {
    use turbo_tracer_shared::{
        check_record_log_header, decode_next_record, StreamRecord, RECORD_LOG_HEADER_BYTES,
    };

    let buf = std::fs::read(trace_path)
        .with_context(|| format!("reading trace file {}", trace_path.display()))?;
    check_record_log_header(&buf)
        .map_err(anyhow::Error::msg)
        .with_context(|| format!("{}", trace_path.display()))?
        .with_context(|| {
            format!(
                "{}: too short for a record-log header",
                trace_path.display()
            )
        })?;
    let mut cursor = RECORD_LOG_HEADER_BYTES;

    // The real thing, not `new_uninit`: the replay decodes each PC's
    // instruction word out of the image, so the ELF has to be loaded.
    let metadata = ElfMetadata::new(elf_path.to_path_buf());
    let modules = Arc::new(Modules::from_metadata(&metadata)?);
    let dp = DataProcessor::with_modules(modules.clone(), TraceFormat::Bbt)?;
    let mut decoder = dp.make_decoder();
    let mut replay = MemReplay::new(modules, vlenb);

    let mut out = Vec::new();
    let mut record_decoder = turbo_tracer_shared::RecordLogDecoder::new();
    loop {
        match decode_next_record(&mut record_decoder, &buf, &mut cursor) {
            // A torn tail is what a run still being written looks like; the
            // accesses decoded so far are still valid.
            None | Some(Ok(StreamRecord::Done)) => break,
            Some(Err(e)) => bail!("{}: {e}", trace_path.display()),
            Some(Ok(StreamRecord::Batch { trace_data, .. })) => {
                let mut err = None;
                dp.decode_into(&mut decoder, &trace_data, |batch| {
                    let TraceBatch::Blocks(blocks) = batch else {
                        err.get_or_insert_with(|| {
                            anyhow::anyhow!("BBT decode produced a non-block batch")
                        });
                        return;
                    };
                    for block in blocks {
                        if err.is_some() {
                            return;
                        }
                        match block.map_err(anyhow::Error::from) {
                            Ok(b) => {
                                if let Err(e) = replay.block(&b, |a| out.push(a)) {
                                    err = Some(e);
                                }
                            }
                            Err(e) => err = Some(e),
                        }
                    }
                })?;
                if let Some(e) = err {
                    return Err(e);
                }
            }
        }
    }
    Ok(out)
}
