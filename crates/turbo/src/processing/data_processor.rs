//! Trace decoding: the recorded stream to executed instruction PCs.
//!
//! The QEMU turbo tracer plugin emits BBT (see [`turbo_tracer_shared::bbt`]): one
//! `u32` per executed block plus a dictionary describing each static block once.
//! Decoding is therefore a dictionary lookup rather than a packet decode, and
//! the PCs are handed to the consumer borrowed from the dictionary arena, never
//! materialized per execution. Vector length values travel with the block that
//! produced them.

use anyhow::Result;
use std::sync::Arc;

use crate::processing::enricher::Modules;
use crate::processing::ElfMetadata;
use crate::RerfError;
use turbo_tracer_shared::bbt::{bbt_addr_join, bbt_decode, BbtRecord, MemGroupKind};
/// Wire format of an incoming trace stream.
///
/// One variant today, but the [`DataProcessor`] still dispatches on it: what
/// framing a stream is in is a property of its producer, not of the consumer,
/// and collapsing that away would have to be undone by the next producer that
/// is not the QEMU plugin.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TraceFormat {
    /// BBT: one `u32` per block entry plus a block dictionary, produced by the
    /// QEMU plugin with `format=bbt`. See [`turbo_tracer_shared::bbt`].
    ///
    /// Software tracing only -- BBT needs a dictionary that only a translator can
    /// write, so it can never ingest a recorded hardware trace.
    #[default]
    Bbt,
}

/// Per-hart decode state, in whichever format this run's stream is.
///
/// One variant per format keeps the format choice in `DataProcessor` rather than
/// spread across the message loop: everything upstream (record-log tailing,
/// `TraceData`, the message loop, the enricher) is format-independent and
/// unchanged.
pub enum HartDecoder {
    /// BBT records, expanded from the block dictionary.
    Bbt(BbtDecoder),
}

/// The BBT block dictionary: every described block's PC sequence, in one
/// append-only arena.
///
/// One arena rather than a `Vec<Vec<u64>>` because the dictionary is read
/// millions of times and written ten thousand: a measured workload has 10,328
/// blocks, 785 KB of contiguous `u64`s here, against 10,328 scattered heap allocations plus
/// 248 KB of `Vec` headers before. Resolving a record is then two indexed loads
/// and no dereference chain, and -- the point of the exercise -- the PCs are
/// handed to the consumer *borrowed*, so no block record is ever expanded into a
/// PC array again.
pub struct BbtDict {
    /// Every block's PCs, back to back, in dictionary-ID order.
    pc_arena: Vec<u64>,
    /// Where each block's PCs live in `pc_arena`, indexed by dictionary ID.
    blocks: Vec<BlockSpan>,
}

/// One block's slice of [`BbtDict::pc_arena`]. `u32`/`u16` rather than `usize`
/// so the table stays 8 bytes per block and cache-resident: a QEMU TB is capped
/// far below 65,536 instructions, and the arena is bounded by static code size.
#[derive(Clone, Copy)]
struct BlockSpan {
    off: u32,
    len: u16,
}

impl BbtDict {
    fn new() -> Self {
        Self {
            pc_arena: Vec::new(),
            blocks: Vec::new(),
        }
    }

    /// The PCs block `id` retires, or `None` if no dictionary entry describes it.
    #[inline(always)]
    fn run(&self, id: u32) -> Option<&[u64]> {
        let span = *self.blocks.get(id as usize)?;
        let off = span.off as usize;
        self.pc_arena.get(off..off + span.len as usize)
    }

    /// Expand one dictionary entry into the arena.
    ///
    /// IDs arrive dense and in first-translation order (`turbo_tracer` assigns
    /// `id = bbt_ids.len()` under `&mut self`, and each hart ships strictly
    /// increasing tails of the one shared dictionary log), so this is a push --
    /// but the ordering is asserted rather than assumed, because silently
    /// tolerating a gap would mis-resolve every later ID.
    fn intern(&mut self, desc: &turbo_tracer_shared::bbt::BlockDesc) -> Result<(), RerfError> {
        if desc.insn_sizes.is_empty() {
            return Err(RerfError::trace_decoding_failed(format!(
                "BBT dictionary entry {} describes a block with no instructions",
                desc.id
            )));
        }
        if desc.id as usize != self.blocks.len() {
            return Err(RerfError::trace_decoding_failed(format!(
                "BBT dictionary entry {} arrived out of order: {} entries are already \
                 loaded, so IDs are no longer dense. The producer assigns IDs densely \
                 in first-translation order and ships strictly increasing tails of the \
                 dictionary log, so this means the chunk stream lost or reordered data.",
                desc.id,
                self.blocks.len()
            )));
        }
        let off = u32::try_from(self.pc_arena.len()).map_err(|_| {
            RerfError::trace_decoding_failed(
                "BBT dictionary exceeds 2^32 instructions of distinct static code",
            )
        })?;
        let len = u16::try_from(desc.insn_sizes.len()).map_err(|_| {
            RerfError::trace_decoding_failed(format!(
                "BBT dictionary entry {} describes {} instructions; a QEMU TB cannot \
                 exceed 65,535",
                desc.id,
                desc.insn_sizes.len()
            ))
        })?;
        let mut pc = desc.start_pc;
        self.pc_arena.reserve(desc.insn_sizes.len());
        for &size in &desc.insn_sizes {
            self.pc_arena.push(pc);
            pc += u64::from(size);
        }
        self.blocks.push(BlockSpan { off, len });
        Ok(())
    }
}

/// BBT decode state for one hart: just the block dictionary, accumulated as
/// chunks deliver it.
///
/// This is the whole BBT consumer. Against the E-Trace path -- a packet decoder,
/// a branch-map state machine, per-item `Instruction` copies and a `Result` per
/// item -- resolving a record is a tag test and two indexed loads.
pub struct BbtDecoder {
    dict: BbtDict,
}

impl BbtDecoder {
    pub fn new() -> Self {
        Self {
            dict: BbtDict::new(),
        }
    }
}

impl Default for BbtDecoder {
    fn default() -> Self {
        Self::new()
    }
}

/// One block entry from the record stream, with its PCs borrowed from the
/// dictionary arena.
#[derive(Debug)]
pub struct BbtBlock<'a> {
    /// Dictionary ID, i.e. the identity of this static block. Dense, so a
    /// consumer can memoize per block behind a plain array index.
    pub id: u32,
    /// The PCs this block retires, in order. Borrowed from [`BbtDict::pc_arena`].
    pub pcs: &'a [u64],
    /// The `vl` produced by this block's last instruction, if that instruction
    /// writes `vl` -- a `vsetvl*`, or a fault-only-first vector load that may
    /// have trimmed it.
    ///
    /// The wire format puts a `VL` record immediately *after* the `BLOCK` record
    /// of the block that produced it, but the consumer needs the value while
    /// committing that block's last instruction -- so the trailing record is
    /// peeked here and delivered with its block. Both producers are asserted at
    /// translate time to be the last instruction of their TB, so there is at
    /// most one.
    pub vl: Option<u16>,
    /// The captured address groups of this block's vector memory instructions,
    /// in program order — one per such instruction executed.
    ///
    /// Borrowed from the record stream and decoded on demand: a block with no
    /// vector memory op (the overwhelming majority) costs nothing here. See
    /// [`MemGroups`].
    pub mem: MemGroups<'a>,
}

/// The `MEM` address groups following one `BLOCK` record, as raw record bytes.
///
/// A cursor rather than a `Vec`: the consumer pairs each group with the PC it
/// belongs to while walking the block's instructions, and most blocks have no
/// groups at all.
#[derive(Debug, Clone, Copy)]
pub struct MemGroups<'a> {
    records: &'a [u8],
    pos: usize,
}

/// One vector memory instruction's captured addresses.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MemGroup<'a> {
    pub kind: MemGroupKind,
    /// The group's address values, as raw `ADDR` record pairs. Use
    /// [`MemGroup::values`].
    words: &'a [u8],
}

impl<'a> MemGroup<'a> {
    /// The captured 64-bit values, reassembled from their record pairs.
    pub fn values(&self) -> impl Iterator<Item = u64> + 'a {
        let words = self.words;
        (0..words.len() / 8).map(move |i| {
            let word = |n: usize| u32::from_le_bytes(words[n..n + 4].try_into().unwrap());
            bbt_addr_join(word(i * 8), word(i * 8 + 4))
        })
    }

    /// The first captured value: the base address for every group kind, and the
    /// first line for a `Lines` group.
    pub fn first(&self) -> Option<u64> {
        self.values().next()
    }
}

impl<'a> MemGroups<'a> {
    /// The number of address values a well-formed group header claims, or an
    /// error naming what is wrong with it.
    fn header(word: u32) -> Result<(MemGroupKind, u32), RerfError> {
        let BbtRecord::Mem { kind, len } = bbt_decode(word) else {
            unreachable!("only called on words already decoded as a MEM header");
        };
        let Some(kind) = MemGroupKind::from_u32(kind) else {
            return Err(RerfError::trace_decoding_failed(format!(
                "BBT memory group uses kind {kind}, which this consumer does not know. \
                 The producer is newer than it."
            )));
        };
        if let Some(fixed) = kind.fixed_len() {
            if len != fixed {
                return Err(RerfError::trace_decoding_failed(format!(
                    "BBT memory group of kind {kind:?} carries {len} address value(s), \
                     but that kind is defined as carrying exactly {fixed}."
                )));
            }
        }
        Ok((kind, len))
    }

    /// Byte length of the whole run of groups starting at `pos`, so the block's
    /// groups can be sliced out in one pass and validated in another.
    fn run_length(records: &[u8], mut pos: usize) -> Result<usize, RerfError> {
        let start = pos;
        while let Some(word) = word_at(records, pos) {
            if !matches!(bbt_decode(word), BbtRecord::Mem { .. }) {
                break;
            }
            let (_, len) = Self::header(word)?;
            let end = pos + 4 + 8 * len as usize;
            if end > records.len() {
                return Err(RerfError::trace_decoding_failed(format!(
                    "BBT memory group claims {len} address value(s), but only {} record \
                     byte(s) remain. The chunk stream lost data.",
                    records.len() - pos - 4
                )));
            }
            // Every word of the group's body must be an ADDR record: anything
            // else means the group and the stream disagree about its length,
            // which would silently reattribute addresses to the wrong
            // instruction.
            for off in (pos + 4..end).step_by(4) {
                if !matches!(
                    bbt_decode(word_at(records, off).unwrap()),
                    BbtRecord::Addr(_)
                ) {
                    return Err(RerfError::trace_decoding_failed(format!(
                        "BBT memory group body contains a non-address record at byte {off}."
                    )));
                }
            }
            pos = end;
        }
        Ok(pos - start)
    }
}

impl<'a> Iterator for MemGroups<'a> {
    type Item = MemGroup<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        let word = word_at(self.records, self.pos)?;
        let BbtRecord::Mem { kind, len } = bbt_decode(word) else {
            return None;
        };
        // Both were validated when the run was sliced out, so a malformed group
        // never reaches here.
        let kind = MemGroupKind::from_u32(kind)?;
        let body = self.pos + 4;
        let end = body + 8 * len as usize;
        self.pos = end;
        Some(MemGroup {
            kind,
            words: &self.records[body..end],
        })
    }
}

/// Read one record word at a byte offset, or `None` past the end.
#[inline(always)]
fn word_at(records: &[u8], pos: usize) -> Option<u32> {
    let bytes = records.get(pos..pos + 4)?;
    Some(u32::from_le_bytes(bytes.try_into().unwrap()))
}

/// Iterator over one chunk's BBT records, resolving each block entry against the
/// dictionary.
///
/// The loop body is a `u32` load, a shift-compare on the tag, two indexed loads
/// and a peek at the next word -- no allocation, no hashing, no copying. Errors
/// are yielded rather than swallowed so that a reserved tag or an unknown block
/// ID still fails the run loudly instead of silently shortening the trace.
pub struct BbtBatch<'a> {
    dict: &'a BbtDict,
    records: &'a [u8],
    /// Byte offset of the next record. `records.len()` is a multiple of 4
    /// (checked by `parse_chunk`), so this always lands on a record boundary.
    pos: usize,
}

impl<'a> BbtBatch<'a> {
    fn new(dict: &'a BbtDict, records: &'a [u8]) -> Self {
        Self {
            dict,
            records,
            pos: 0,
        }
    }

    #[inline(always)]
    fn word_at(&self, pos: usize) -> Option<u32> {
        word_at(self.records, pos)
    }
}

impl<'a> Iterator for BbtBatch<'a> {
    type Item = Result<BbtBlock<'a>, RerfError>;

    #[inline(always)]
    fn next(&mut self) -> Option<Self::Item> {
        let word = self.word_at(self.pos)?;
        self.pos += 4;
        let id = match bbt_decode(word) {
            BbtRecord::Block(id) => id,
            BbtRecord::Vl(vl) => {
                return Some(Err(RerfError::trace_decoding_failed(format!(
                    "BBT vl record ({vl}) with no block record in front of it. A vl is \
                     emitted by a `vsetvl*` or a `vl*ff`, either of which is always the \
                     last instruction of its block, so it always trails that block's \
                     record."
                ))));
            }
            BbtRecord::Mem { .. } | BbtRecord::Addr(_) => {
                return Some(Err(RerfError::trace_decoding_failed(
                    "BBT memory address record with no block record in front of it. \
                     Address groups are emitted by an instruction callback, so they \
                     always trail the record of the block containing that instruction."
                        .to_string(),
                )));
            }
            BbtRecord::Reserved(w) => {
                return Some(Err(RerfError::trace_decoding_failed(format!(
                    "BBT record {w:#010x} uses a reserved tag. PARTIAL/SYNC are \
                     specified but not implemented on either side, so a producer \
                     emitting one is newer than this consumer."
                ))));
            }
        };

        let Some(pcs) = self.dict.run(id) else {
            return Some(Err(RerfError::trace_decoding_failed(format!(
                "BBT record names block {id}, which no dictionary entry describes. The \
                 dictionary is written at translate time, before any record can name a \
                 block, so this means the chunk stream lost data."
            ))));
        };

        // The block's memory groups come next: an instruction callback runs
        // after the block callback and before its own instruction, so they sit
        // between this BLOCK record and the block's trailing `vl` (if any).
        let mem_start = self.pos;
        let mem_len = match MemGroups::run_length(self.records, mem_start) {
            Ok(n) => n,
            Err(e) => return Some(Err(e)),
        };
        self.pos += mem_len;
        let mem = MemGroups {
            records: &self.records[mem_start..mem_start + mem_len],
            pos: 0,
        };

        // Peek the trailing `vl`, if this block ends in a `vsetvl*` or a `vl*ff`.
        let mut vl = None;
        if let Some(next) = self.word_at(self.pos) {
            if let BbtRecord::Vl(v) = bbt_decode(next) {
                vl = Some(v as u16);
                self.pos += 4;
                if let Some(after) = self.word_at(self.pos) {
                    if let BbtRecord::Vl(v2) = bbt_decode(after) {
                        return Some(Err(RerfError::trace_decoding_failed(format!(
                            "BBT block {id} is followed by two vl records ({v} then \
                             {v2}). The instructions that write `vl` -- a `vsetvl*` and \
                             a `vl*ff` -- are each asserted at translate time to be the \
                             last instruction of their block, so a block can produce at \
                             most one vl."
                        ))));
                    }
                }
            }
        }

        Some(Ok(BbtBlock { id, pcs, vl, mem }))
    }
}

/// One decoded batch, in whichever shape this run's format produces.
///
/// The two formats are genuinely different objects and neither is converted into
/// the other: E-Trace reconstructs individual PCs from a packet stream and has no
/// notion of a block, while BBT names pre-described blocks and never materializes
/// a PC array at all. Adapting one to the other would mean either re-expanding
/// blocks into a `Vec<u64>` (the 192 MB memcpy this shape exists to delete) or
/// inventing synthetic single-instruction blocks for E-Trace (a branch in the
/// hottest loop, for no gain).
pub enum TraceBatch<'a> {
    /// E-Trace: PCs reconstructed from the packet stream, plus the `vl` values
    /// carried inline as CSR data-trace packets over the same stretch.
    Pcs { pcs: &'a [u64], vls: &'a [u16] },
    /// BBT: the chunk's block records, resolving to borrowed per-block PC runs.
    /// `vl` values travel with the block that produced them.
    Blocks(BbtBatch<'a>),
}

/// DataProcessor caches ELF parsing and binary analysis to avoid redundant work
/// in performance-critical trace processing paths.
pub struct DataProcessor {
    // The decode tables, shared with the enricher so the whole run builds them
    // once. BBT resolves PCs from its own dictionary and so reads nothing here,
    // but a decoder that must fetch instruction bytes to reconstruct a stream
    // needs exactly these, and `with_modules` is how it is handed them.
    #[allow(dead_code)]
    modules: Arc<Modules>,

    // Wire format of the incoming trace stream.
    format: TraceFormat,
}

impl DataProcessor {
    /// Create a new DataProcessor with cached ELF parsing and binary analysis
    ///
    /// # Arguments
    /// * `metadata` - Processor metadata containing ELF path and shared library configuration
    /// * `format` - Wire format of the incoming trace stream.
    ///
    /// Builds a private [`Modules`]; prefer
    /// [`with_modules`](Self::with_modules) alongside the enricher's, so the
    /// decode tables are built once for the whole run.
    pub fn new(metadata: &ElfMetadata, format: TraceFormat) -> Result<Self> {
        Self::with_modules(Arc::new(Modules::from_metadata(metadata)?), format)
    }

    /// Create a DataProcessor over an already-built, shared program image.
    pub fn with_modules(modules: Arc<Modules>, format: TraceFormat) -> Result<Self> {
        Ok(DataProcessor { modules, format })
    }

    /// Process trace data and return a Vec of executed PC values
    ///
    /// # Arguments
    /// * `trace_data` - Raw trace data bytes to decode
    ///
    /// # Returns
    /// A vector of program counter (PC) values in execution order
    pub fn process_trace(&self, trace_data: &[u8]) -> Vec<u64> {
        let mut all_pcs = Vec::new();
        let _ = self.process_trace_channel(trace_data, |batch| match batch {
            TraceBatch::Pcs { pcs, .. } => all_pcs.extend_from_slice(pcs),
            TraceBatch::Blocks(blocks) => {
                for block in blocks {
                    match block {
                        Ok(b) => all_pcs.extend_from_slice(b.pcs),
                        Err(e) => log::error!("BBT decode error: {e}"),
                    }
                }
            }
        });
        all_pcs
    }

    /// Build a fresh per-hart decoder for this run's format.
    pub fn make_decoder(&self) -> HartDecoder {
        match self.format {
            TraceFormat::Bbt => HartDecoder::Bbt(BbtDecoder::new()),
        }
    }

    /// Decode a trace chunk, feeding a caller-owned (persistent) decoder.
    ///
    /// Used by the live path: each record-log batch is one chunk of a single
    /// continuous trace, so the same `decoder` is passed across calls to retain
    /// decode state across batch boundaries.
    pub fn decode_into(
        &self,
        decoder: &mut HartDecoder,
        trace_data: &[u8],
        on_batch: impl FnMut(TraceBatch<'_>),
    ) -> Result<(), RerfError> {
        match decoder {
            HartDecoder::Bbt(dec) => self.decode_bbt_into(dec, trace_data, on_batch),
        }
    }

    /// Decode a self-contained trace blob with a fresh tracer.
    pub fn process_trace_channel(
        &self,
        trace_data: &[u8],
        on_batch: impl FnMut(TraceBatch<'_>),
    ) -> Result<(), RerfError> {
        let mut decoder = self.make_decoder();
        self.decode_into(&mut decoder, trace_data, on_batch)
    }

    /// Hand one BBT chunk's records to the consumer as borrowed per-block PC runs.
    ///
    /// One batch per chunk, and nothing is copied: the chunk *is* the producer's
    /// flush unit (its `vl` records never straddle it, because a `vsetvl*` always
    /// ends its block and the dictionary tail is snapshotted after the records),
    /// and the consumer walks the record bytes directly. The 100K-PC sub-batching
    /// the E-Trace path needs exists only to bound the size of the `Vec<u64>` it
    /// materializes; there is no such `Vec` here.
    fn decode_bbt_into(
        &self,
        dec: &mut BbtDecoder,
        payload: &[u8],
        mut on_batch: impl FnMut(TraceBatch<'_>),
    ) -> Result<(), RerfError> {
        use turbo_tracer_shared::bbt::{decode_block_descs, parse_chunk};

        let (dict, records) = parse_chunk(payload).map_err(RerfError::trace_decoding_failed)?;

        // Dictionary first: this chunk's entries describe every block its records
        // can name (the producer snapshots the dictionary after appending the
        // records, so it can only ever be ahead, never behind).
        for desc in decode_block_descs(dict).map_err(RerfError::trace_decoding_failed)? {
            dec.dict.intern(&desc)?;
        }

        if !records.is_empty() {
            on_batch(TraceBatch::Blocks(BbtBatch::new(&dec.dict, records)));
        }
        Ok(())
    }
}

#[cfg(test)]
mod bbt_tests {
    use super::*;
    use turbo_tracer_shared::bbt::{
        bbt_addr_hi, bbt_addr_lo, bbt_block, bbt_mem, bbt_vl, encode_block_desc, push_mem_group,
        BlockDesc,
    };

    fn dict_of(entries: &[(u32, u64, &[u8])]) -> BbtDict {
        let mut dict = BbtDict::new();
        for &(id, start_pc, sizes) in entries {
            dict.intern(&BlockDesc {
                id,
                start_pc,
                insn_sizes: sizes.to_vec(),
            })
            .expect("entry must intern");
        }
        dict
    }

    fn records(words: &[u32]) -> Vec<u8> {
        words.iter().flat_map(|w| w.to_le_bytes()).collect()
    }

    /// The arena must hand out exactly the PCs the sizes describe, borrowed --
    /// this is the property that replaces `BlockDesc::pcs()`'s `Vec` per record.
    #[test]
    fn arena_resolves_each_block_to_its_own_pcs() {
        let dict = dict_of(&[(0, 0x1000, &[4, 2, 2]), (1, 0x2000, &[4])]);
        assert_eq!(dict.run(0).unwrap(), &[0x1000, 0x1004, 0x1006]);
        assert_eq!(dict.run(1).unwrap(), &[0x2000]);
        assert!(dict.run(2).is_none(), "an undescribed ID must not resolve");
    }

    /// IDs are dense and ascending by construction on the producer side, so a gap
    /// means lost data and must fail loudly rather than mis-resolve every later ID.
    #[test]
    fn out_of_order_dictionary_entry_is_rejected() {
        let mut dict = BbtDict::new();
        let err = dict
            .intern(&BlockDesc {
                id: 3,
                start_pc: 0x1000,
                insn_sizes: vec![4],
            })
            .expect_err("a gap must be rejected");
        assert!(format!("{err}").contains("out of order"), "{err}");
    }

    #[test]
    fn empty_dictionary_entry_is_rejected() {
        let mut dict = BbtDict::new();
        assert!(dict
            .intern(&BlockDesc {
                id: 0,
                start_pc: 0x1000,
                insn_sizes: vec![],
            })
            .is_err());
    }

    /// A block's trailing `vl` record belongs to the `vsetvl*` that ends it, and
    /// the consumer needs it *while* committing that instruction -- so the
    /// iterator must peek it forward onto the block, and must not leave it to be
    /// yielded as a record of its own.
    #[test]
    fn trailing_vl_is_attached_to_its_block() {
        let dict = dict_of(&[(0, 0x1000, &[4, 4]), (1, 0x2000, &[4])]);
        let recs = records(&[bbt_block(0), bbt_vl(8), bbt_block(1), bbt_block(0)]);
        let got: Vec<(u32, Vec<u64>, Option<u16>)> = BbtBatch::new(&dict, &recs)
            .map(|b| b.map(|b| (b.id, b.pcs.to_vec(), b.vl)).unwrap())
            .collect();
        assert_eq!(
            got,
            vec![
                (0, vec![0x1000, 0x1004], Some(8)),
                (1, vec![0x2000], None),
                (0, vec![0x1000, 0x1004], None),
            ]
        );
    }

    /// `vl == 0` is the legitimate `vill` case, so it must arrive as a value and
    /// not be confused with "no vl".
    #[test]
    fn a_zero_vl_is_a_value() {
        let dict = dict_of(&[(0, 0x1000, &[4])]);
        let recs = records(&[bbt_block(0), bbt_vl(0)]);
        let block = BbtBatch::new(&dict, &recs).next().unwrap().unwrap();
        assert_eq!(block.vl, Some(0));
    }

    /// A `vsetvl*` is asserted at translate time to end its block, so two `vl`s
    /// behind one block record means producer and consumer disagree about the
    /// format -- which must fail rather than silently drop one.
    #[test]
    fn two_vls_behind_one_block_is_an_error() {
        let dict = dict_of(&[(0, 0x1000, &[4])]);
        let recs = records(&[bbt_block(0), bbt_vl(8), bbt_vl(4)]);
        let err = BbtBatch::new(&dict, &recs)
            .next()
            .unwrap()
            .expect_err("two vls must be rejected");
        assert!(format!("{err}").contains("two vl records"), "{err}");
    }

    #[test]
    fn a_record_naming_an_undescribed_block_is_an_error() {
        let dict = dict_of(&[(0, 0x1000, &[4])]);
        let recs = records(&[bbt_block(7)]);
        let err = BbtBatch::new(&dict, &recs).next().unwrap().unwrap_err();
        assert!(format!("{err}").contains("no dictionary entry"), "{err}");
    }

    /// Reserved tags (the `PARTIAL`/`SYNC` space) mean the producer is newer than
    /// this consumer. Yielding an error rather than `None` is what keeps that from
    /// silently truncating the trace.
    #[test]
    fn a_reserved_tag_is_an_error_not_an_end_of_stream() {
        let dict = dict_of(&[(0, 0x1000, &[4])]);
        let recs = records(&[(0xC << 28) | 3]);
        let err = BbtBatch::new(&dict, &recs).next().unwrap().unwrap_err();
        assert!(format!("{err}").contains("reserved tag"), "{err}");
    }

    /// A `vl` with no block in front of it cannot happen in a well-formed stream.
    #[test]
    fn a_leading_vl_is_an_error() {
        let dict = dict_of(&[(0, 0x1000, &[4])]);
        let recs = records(&[bbt_vl(8)]);
        let err = BbtBatch::new(&dict, &recs).next().unwrap().unwrap_err();
        assert!(format!("{err}").contains("no block record"), "{err}");
    }

    /// Address groups belong to the block whose instructions produced them, and
    /// several ops in one block must stay separate and in program order -- that
    /// order is the only thing that pairs a group with its PC.
    #[test]
    fn memory_groups_are_attached_to_their_block_in_program_order() {
        let dict = dict_of(&[(0, 0x1000, &[4, 4, 4]), (1, 0x2000, &[4])]);
        let mut recs = records(&[bbt_block(0)]);
        push_mem_group(&mut recs, MemGroupKind::Base, &[0x1_0000]);
        push_mem_group(&mut recs, MemGroupKind::BaseStride, &[0x2_0000, 64]);
        recs.extend(records(&[bbt_vl(8)]));
        recs.extend(records(&[bbt_block(1)]));
        push_mem_group(&mut recs, MemGroupKind::Lines, &[0x3_0000, 0x3_0040]);

        let blocks: Vec<_> = BbtBatch::new(&dict, &recs)
            .map(|b| b.unwrap())
            .map(|b| {
                (
                    b.id,
                    b.vl,
                    b.mem
                        .map(|g| (g.kind, g.values().collect::<Vec<_>>()))
                        .collect::<Vec<_>>(),
                )
            })
            .collect();

        assert_eq!(
            blocks,
            vec![
                (
                    0,
                    Some(8),
                    vec![
                        (MemGroupKind::Base, vec![0x1_0000]),
                        (MemGroupKind::BaseStride, vec![0x2_0000, 64]),
                    ]
                ),
                (
                    1,
                    None,
                    vec![(MemGroupKind::Lines, vec![0x3_0000, 0x3_0040])]
                ),
            ]
        );
    }

    /// A block with no vector memory op yields no groups, and the groups of one
    /// block must never leak into the next.
    #[test]
    fn a_block_without_memory_ops_has_no_groups() {
        let dict = dict_of(&[(0, 0x1000, &[4]), (1, 0x2000, &[4])]);
        let mut recs = records(&[bbt_block(0)]);
        push_mem_group(&mut recs, MemGroupKind::Base, &[0x1_0000]);
        recs.extend(records(&[bbt_block(1)]));

        let counts: Vec<usize> = BbtBatch::new(&dict, &recs)
            .map(|b| b.unwrap().mem.count())
            .collect();
        assert_eq!(counts, vec![1, 0]);
    }

    /// An address record with no group header in front of it means the stream
    /// and the group lengths disagree, which would reattribute every later
    /// address to the wrong instruction.
    #[test]
    fn a_leading_address_record_is_an_error() {
        let dict = dict_of(&[(0, 0x1000, &[4])]);
        let recs = records(&[bbt_addr_lo(0x1000), bbt_addr_hi(0x1000)]);
        let err = BbtBatch::new(&dict, &recs).next().unwrap().unwrap_err();
        assert!(format!("{err}").contains("no block record"), "{err}");
    }

    /// A group whose body is cut short by the end of the records is lost data,
    /// not a shorter group.
    #[test]
    fn a_truncated_memory_group_is_an_error() {
        let dict = dict_of(&[(0, 0x1000, &[4])]);
        let mut recs = records(&[bbt_block(0)]);
        push_mem_group(&mut recs, MemGroupKind::Lines, &[0x1_0000, 0x1_0040]);
        recs.truncate(recs.len() - 4);
        let err = BbtBatch::new(&dict, &recs).next().unwrap().unwrap_err();
        assert!(format!("{err}").contains("record byte(s) remain"), "{err}");
    }

    /// The fixed-size kinds are fixed: a `Base` group carrying two addresses is
    /// a producer/consumer disagreement about the format.
    #[test]
    fn a_fixed_kind_with_the_wrong_length_is_an_error() {
        let dict = dict_of(&[(0, 0x1000, &[4])]);
        let mut recs = records(&[bbt_block(0), bbt_mem(MemGroupKind::Base, 2)]);
        for v in [0x1_0000u64, 0x2_0000] {
            recs.extend(records(&[bbt_addr_lo(v), bbt_addr_hi(v)]));
        }
        let err = BbtBatch::new(&dict, &recs).next().unwrap().unwrap_err();
        assert!(format!("{err}").contains("defined as carrying"), "{err}");
    }

    /// End to end through `decode_bbt_into`: the dictionary section is ingested
    /// and the records resolve against it, with nothing copied out of the arena.
    #[test]
    fn a_chunk_decodes_to_borrowed_block_runs() {
        use turbo_tracer_shared::bbt::encode_chunk;

        // Building the modules' decode tables folds in the run-fixed VLEN, so
        // it has to be set first; nextest gives each test its own process.
        crate::common::SYSTEM_VLEN.set_bytes(16);
        let metadata = ElfMetadata::new_uninit(std::path::PathBuf::from("/bin/true"));
        let dp = DataProcessor::new(&metadata, TraceFormat::Bbt).expect("processor");
        let mut decoder = dp.make_decoder();

        let mut dict = Vec::new();
        encode_block_desc(&mut dict, 0, 0x1000, &[4, 2]);
        let chunk = encode_chunk(&dict, &records(&[bbt_block(0), bbt_vl(16), bbt_block(0)]));

        let mut seen: Vec<(u32, Vec<u64>, Option<u16>)> = Vec::new();
        dp.decode_into(&mut decoder, &chunk, |batch| match batch {
            TraceBatch::Blocks(blocks) => {
                for b in blocks {
                    let b = b.unwrap();
                    seen.push((b.id, b.pcs.to_vec(), b.vl));
                }
            }
            TraceBatch::Pcs { .. } => panic!("BBT must not produce a PC batch"),
        })
        .expect("chunk must decode");

        assert_eq!(
            seen,
            vec![
                (0, vec![0x1000, 0x1004], Some(16)),
                (0, vec![0x1000, 0x1004], None),
            ]
        );
    }
}

#[cfg(test)]
mod thread_safety {
    /// The parallel decode pool shares one `DataProcessor` across worker threads
    /// (`&self` on `decode_into`) and moves one `HartDecoder` + `Enricher` into
    /// each. If any of these stops being `Send`/`Sync`, the pool stops compiling
    /// -- this pins the requirement to a named test instead of a confusing error
    /// inside the scope closure.
    #[test]
    fn decode_types_cross_thread_boundaries() {
        fn assert_send<T: Send>() {}
        fn assert_sync<T: Sync>() {}
        assert_sync::<super::DataProcessor>();
        assert_send::<super::DataProcessor>();
        assert_send::<super::HartDecoder>();
        assert_send::<crate::processing::Enricher>();
    }
}
