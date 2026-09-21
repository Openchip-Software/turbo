//! BBT — the basic-block trace wire format.
//!
//! An alternative to the E-Trace packet stream for *software* tracing. The
//! producer emits one `u32` per block entry and a dictionary describing each
//! distinct block; the consumer expands each record into that block's PCs with a
//! `memcpy`. E-Trace remains the only format that can ingest a recorded hardware
//! trace, so it is not going away.
//!
//! # Why
//!
//! Measured on a workload, the E-Trace *encoder* is the tracer's cost:
//! `TraceEncoder::process_step` 21% of the run, the `riscv-etrace` generator
//! 10%, packet encoding 7.4% — ~39%, against ~7% for the per-block callback
//! dispatch. BBT deletes all of it on both sides: the producer appends 4 bytes
//! and the consumer copies a precomputed PC run, so neither runs a branch-map
//! state machine, a packet encoder, or a packet decoder.
//!
//! # Records
//!
//! One `u32`, little-endian, tagged in its top 4 bits so a `BLOCK` record keeps a
//! fixed 4-byte stride (the common case is ~97% of records):
//!
//! | tag | meaning |
//! |---|---|
//! | `0x0` | [`BBT_TAG_BLOCK`] — this block was *entered*; value is its dictionary ID |
//! | `0xD` | [`BBT_TAG_MEM`] — header of one vector memory op's address group |
//! | `0xE` | [`BBT_TAG_ADDR`] — 28 bits of an address value; see [`MemGroupKind`] |
//! | `0xF` | [`BBT_TAG_VL`] — a `vsetvl*` or a `vl*ff` produced this `vl`; value is the `vl` |
//! | `0x1`..=`0xC` | reserved; see the note on `PARTIAL`/`SYNC` below |
//!
//! A `MEM` group is one `MEM` header followed by `n` address *pairs* (`ADDR`
//! low half, then `ADDR` high half). The groups of a block follow that block's
//! `BLOCK` record in **program order**, because QEMU fires an instruction
//! callback before the instruction and the block callback at block entry. A
//! consumer walking the block's PCs pulls the next group when the PC decodes as
//! a vector memory op — the same discipline `VL` already uses.
//!
//! A `VL` record always *immediately follows* the `BLOCK` record of the block
//! that produced it (after that block's `MEM` groups), because both instructions
//! that write `vl` -- a `vsetvl*` and a fault-only-first vector load
//! (`vl*ff`) -- are always the last instruction in their block, and
//! the plugin asserts that at translate time. So a block carries at most one
//! `VL`, and the invariant that lets a consumer cut a batch safely holds:
//! **cut before a `BLOCK` record, never between a `BLOCK` and its trailing
//! `VL`**, and no batch can end owing a `vl` to an instruction it already
//! emitted.
//!
//! The two differ in *which* callback appends the record. A `vsetvl*`'s `vl` is
//! computed from the instruction, so its own callback emits it; a `vl*ff`'s is
//! only knowable after the fault has (or has not) happened, so the *successor*
//! block's first callback reads the `vl` CSR and appends the record before that
//! block's own `BLOCK` record. The position in the stream is the same either
//! way, which is why the consumer needs no new tag.
//!
//! `vl` fits in 28 bits comfortably: the architectural maximum is `VLEN` (65536
//! elements at `e8,m8` with the widest `VLEN` QEMU supports).
//!
//! `PARTIAL(n)` (a block was entered but only `n` instructions retired, after a
//! fault) and `SYNC(pc)` (discard position, restart at an absolute PC) are
//! specified in the design and have tag space reserved here, but are **not
//! emitted or handled**: producing them needs QEMU's exception hook, and the
//! exposure they cover is no worse than the E-Trace path's (which cannot even
//! represent it). Reserved rather than implemented, deliberately.
//!
//! # Records assert entry, not completion
//!
//! A `BLOCK` record means "this block was entered", not "it ran to completion":
//! QEMU injects the block callback at `PLUGIN_GEN_FROM_TB`, i.e. before the first
//! instruction. There is no hidden intra-block control flow to worry about —
//! everything that leaves a block mid-stream ends the block — so the only
//! exposure is a fault or signal aborting a block part-way, which is what
//! `PARTIAL` is reserved for.

/// Leading magic of every BBT chunk. Present so that pointing the BBT decoder at
/// an E-Trace stream (or vice versa) fails immediately with a clear message
/// rather than producing plausible nonsense.
pub const BBT_CHUNK_MAGIC: [u8; 4] = *b"BBT1";

/// Size of a chunk's header: [`BBT_CHUNK_MAGIC`] plus the two `u32` section
/// lengths written by [`encode_chunk`].
pub const BBT_CHUNK_HEADER_BYTES: u64 = 12;

/// Bit position of a record's 4-bit tag.
pub const BBT_TAG_SHIFT: u32 = 28;
/// Mask of a record's 28-bit value field.
pub const BBT_VALUE_MASK: u32 = (1 << BBT_TAG_SHIFT) - 1;

/// Tag: block entry, value is the dictionary ID. Zero so that a `BLOCK` record
/// *is* its ID, which keeps the producer's append to a plain store.
pub const BBT_TAG_BLOCK: u32 = 0x0;
/// Tag: a `vsetvl*` or a `vl*ff` produced this `vl`.
pub const BBT_TAG_VL: u32 = 0xF;
/// Tag: header of one vector memory op's address group. Value is
/// `kind << 24 | n`, `n` being the number of 64-bit address values that follow.
pub const BBT_TAG_MEM: u32 = 0xD;
/// Tag: one half of an address value. Values travel as a low-half word followed
/// by a high-half word, 28 bits each.
pub const BBT_TAG_ADDR: u32 = 0xE;

/// Bits of an address carried per [`BBT_TAG_ADDR`] word.
pub const BBT_ADDR_HALF_BITS: u32 = BBT_TAG_SHIFT;
/// Addresses (and wrapped strides) must fit in two [`BBT_TAG_ADDR`] words. This
/// covers Sv39/Sv48 guest user space; an Sv57 address trips the assert in
/// [`push_mem_group`] rather than being silently truncated.
pub const BBT_ADDR_BITS: u32 = 2 * BBT_ADDR_HALF_BITS;

/// What the address values of a [`BBT_TAG_MEM`] group mean.
///
/// The split exists because only the indexed forms need their footprint spelled
/// out: for everything else the byte range is an analytic function of `vl`,
/// `SEW`, `nf` and the mode, all of which the consumer already has. Sending one
/// base address instead of a line list is both smaller and exact.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum MemGroupKind {
    /// One value: the base address `x[rs1]`. The consumer derives the range.
    Base = 0,
    /// Two values: base address, then the stride `x[rs2]`, wrapped to
    /// [`BBT_ADDR_BITS`] and to be sign-extended by the consumer.
    BaseStride = 1,
    /// `n` values, each already cache-line aligned: the complete, deduplicated
    /// set of lines this instruction touched. Emitted for indexed
    /// (gather/scatter) ops, whose footprint no offline derivation can recover.
    Lines = 2,
}

impl MemGroupKind {
    /// Decode a group kind, or `None` for a value this consumer does not know.
    pub const fn from_u32(v: u32) -> Option<Self> {
        match v {
            0 => Some(Self::Base),
            1 => Some(Self::BaseStride),
            2 => Some(Self::Lines),
            _ => None,
        }
    }

    /// The fixed number of address values this kind carries, or `None` when it
    /// is variable (`Lines`).
    pub const fn fixed_len(self) -> Option<u32> {
        match self {
            Self::Base => Some(1),
            Self::BaseStride => Some(2),
            Self::Lines => None,
        }
    }
}

/// Bit position of the kind inside a [`BBT_TAG_MEM`] word's 28-bit value.
const BBT_MEM_KIND_SHIFT: u32 = 24;
/// Largest address-value count a `MEM` header can carry.
///
/// A `Lines` group has at most one line per element step, i.e. `vl` of them
/// (65536 at the widest `VLEN`), so 16 bits is ample and the check is an assert
/// rather than a saturation.
pub const BBT_MEM_MAX_LEN: u32 = (1 << BBT_MEM_KIND_SHIFT) - 1;

/// Encode a `MEM` group header.
#[inline(always)]
pub fn bbt_mem(kind: MemGroupKind, len: u32) -> u32 {
    debug_assert!(
        len <= BBT_MEM_MAX_LEN,
        "mem group length {len} exceeds the header's count field"
    );
    (BBT_TAG_MEM << BBT_TAG_SHIFT) | ((kind as u32) << BBT_MEM_KIND_SHIFT) | len
}

/// Encode the low half of an address value.
#[inline(always)]
pub fn bbt_addr_lo(addr: u64) -> u32 {
    (BBT_TAG_ADDR << BBT_TAG_SHIFT) | (addr as u32 & BBT_VALUE_MASK)
}

/// Encode the high half of an address value.
#[inline(always)]
pub fn bbt_addr_hi(addr: u64) -> u32 {
    (BBT_TAG_ADDR << BBT_TAG_SHIFT) | ((addr >> BBT_ADDR_HALF_BITS) as u32 & BBT_VALUE_MASK)
}

/// Reassemble an address from its two halves.
#[inline(always)]
pub fn bbt_addr_join(lo: u32, hi: u32) -> u64 {
    u64::from(lo & BBT_VALUE_MASK) | (u64::from(hi & BBT_VALUE_MASK) << BBT_ADDR_HALF_BITS)
}

/// Sign-extend a stride recovered by [`bbt_addr_join`] from its
/// [`BBT_ADDR_BITS`]-wide wrapped form.
#[inline(always)]
pub fn bbt_sign_extend_addr(v: u64) -> i64 {
    ((v << (64 - BBT_ADDR_BITS)) as i64) >> (64 - BBT_ADDR_BITS)
}

/// Append one whole `MEM` group — header then `values.len()` address pairs — to
/// a record buffer.
///
/// Values are asserted to fit [`BBT_ADDR_BITS`]. A guest address that does not
/// is a real (Sv57) case this format would have to grow a word for; truncating
/// it would produce a plausible-looking wrong line, which is the one outcome a
/// cache-footprint trace must not have.
pub fn push_mem_group(out: &mut Vec<u8>, kind: MemGroupKind, values: &[u64]) {
    assert!(
        values.len() as u32 <= BBT_MEM_MAX_LEN,
        "mem group of {} values exceeds the header's count field",
        values.len()
    );
    out.extend_from_slice(&bbt_mem(kind, values.len() as u32).to_le_bytes());
    for &v in values {
        assert!(
            v >> BBT_ADDR_BITS == 0,
            "address/stride {v:#x} does not fit in {BBT_ADDR_BITS} bits; the BBT \
             address encoding needs widening (Sv57 guest?)"
        );
        out.extend_from_slice(&bbt_addr_lo(v).to_le_bytes());
        out.extend_from_slice(&bbt_addr_hi(v).to_le_bytes());
    }
}

/// Largest dictionary ID the record encoding can carry. IDs are dense and
/// assigned in first-translation order; a measured workload reaches 10 328 distinct blocks,
/// so 268 M is not a practical limit — but it is asserted rather than assumed,
/// since a self-modifying or JIT guest grows the dictionary without bound by
/// design.
pub const BBT_MAX_BLOCK_ID: u32 = BBT_VALUE_MASK;

/// Encode a block-entry record.
#[inline(always)]
pub fn bbt_block(id: u32) -> u32 {
    debug_assert!(
        id <= BBT_MAX_BLOCK_ID,
        "block id {id} exceeds the 28-bit record field"
    );
    (BBT_TAG_BLOCK << BBT_TAG_SHIFT) | id
}

/// Encode a `vl` record. `vl` values wider than 28 bits are impossible for any
/// `VLEN` QEMU supports, and are saturated rather than silently truncated.
#[inline(always)]
pub fn bbt_vl(vl: u64) -> u32 {
    let v = (vl as u32).min(BBT_VALUE_MASK);
    (BBT_TAG_VL << BBT_TAG_SHIFT) | v
}

/// One decoded record.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BbtRecord {
    /// This block was entered. Carries its dictionary ID.
    Block(u32),
    /// A `vsetvl*` or a `vl*ff` produced this `vl`.
    Vl(u32),
    /// Header of a vector memory op's address group: its kind, and how many
    /// 64-bit values (i.e. `2 * len` `Addr` records) follow.
    ///
    /// `kind` is the raw field rather than a [`MemGroupKind`], so an unknown
    /// kind from a newer producer is a decode-time error at the point of use
    /// instead of an unrepresentable record here.
    Mem { kind: u32, len: u32 },
    /// One half of an address value. Only meaningful inside a `Mem` group.
    Addr(u32),
    /// A reserved tag (`PARTIAL`/`SYNC` space). Carries the raw word.
    Reserved(u32),
}

/// Decode one record word.
#[inline(always)]
pub fn bbt_decode(word: u32) -> BbtRecord {
    match word >> BBT_TAG_SHIFT {
        BBT_TAG_BLOCK => BbtRecord::Block(word & BBT_VALUE_MASK),
        BBT_TAG_VL => BbtRecord::Vl(word & BBT_VALUE_MASK),
        BBT_TAG_MEM => BbtRecord::Mem {
            kind: (word & BBT_VALUE_MASK) >> BBT_MEM_KIND_SHIFT,
            len: word & BBT_MEM_MAX_LEN,
        },
        BBT_TAG_ADDR => BbtRecord::Addr(word & BBT_VALUE_MASK),
        _ => BbtRecord::Reserved(word),
    }
}

/// One dictionary entry: everything a consumer needs to expand a `BLOCK` record
/// into the PCs the block retires.
///
/// Instruction *sizes* rather than PCs, because that is what the producer has at
/// translate time and it is 1 byte per instruction instead of 8. `total_bytes`
/// and the block's end PC are derivable, so they are not carried.
///
/// The design also specified `flags` (ends-in-cond-branch / jump / trap) and
/// `static_vl`. Neither is written: `flags` existed for offline *branch*
/// resolution, and this consumer wants the PC sequence, which the sizes already
/// give; `static_vl` was an optimisation for a producer that skipped a callback
/// on constant-`vl` sites, and the producer does not skip it (it still has to
/// emit the record).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BlockDesc {
    /// Dense dictionary ID, matching the `BLOCK` records that name it.
    pub id: u32,
    /// Guest PC of the block's first instruction.
    pub start_pc: u64,
    /// Size in bytes of each instruction in the block, in order (2 or 4 on
    /// RISC-V).
    pub insn_sizes: Vec<u8>,
}

impl BlockDesc {
    /// The PCs this block retires, in order.
    pub fn pcs(&self) -> Vec<u64> {
        // nosemgrep: dos-unbounded-memory-allocation -- len of a resident Vec, bounded by the u16 count decode validates
        let mut pcs = Vec::with_capacity(self.insn_sizes.len());
        let mut pc = self.start_pc;
        for &size in &self.insn_sizes {
            pcs.push(pc);
            pc += u64::from(size);
        }
        pcs
    }
}

/// Append one dictionary entry to `out`: `id: u32`, `start_pc: u64`,
/// `n_insns: u16`, then `n_insns` size bytes, all little-endian.
///
/// A block is written the first time it is interned, which is strictly before
/// any record can name it — so a live-tailing consumer can always resolve an ID
/// it has just seen.
pub fn encode_block_desc(out: &mut Vec<u8>, id: u32, start_pc: u64, insn_sizes: &[u8]) {
    out.extend_from_slice(&id.to_le_bytes());
    out.extend_from_slice(&start_pc.to_le_bytes());
    out.extend_from_slice(&(insn_sizes.len() as u16).to_le_bytes());
    out.extend_from_slice(insn_sizes);
}

/// Decode a run of dictionary entries written by [`encode_block_desc`].
pub fn decode_block_descs(mut bytes: &[u8]) -> Result<Vec<BlockDesc>, String> {
    let mut out = Vec::new();
    while !bytes.is_empty() {
        if bytes.len() < 14 {
            return Err(format!(
                "truncated BBT dictionary entry: {} byte(s) left, need at least 14",
                bytes.len()
            ));
        }
        let id = u32::from_le_bytes(bytes[0..4].try_into().unwrap());
        let start_pc = u64::from_le_bytes(bytes[4..12].try_into().unwrap());
        let n = usize::from(u16::from_le_bytes(bytes[12..14].try_into().unwrap()));
        if bytes.len() < 14 + n {
            return Err(format!(
                "truncated BBT dictionary entry for id {id}: needs {n} size byte(s), \
                 {} left",
                bytes.len() - 14
            ));
        }
        out.push(BlockDesc {
            id,
            start_pc,
            insn_sizes: bytes[14..14 + n].to_vec(),
        });
        bytes = &bytes[14 + n..];
    }
    Ok(out)
}

/// Assemble one chunk: magic, then a length-prefixed dictionary section (the
/// entries interned since the previous chunk), then the record stream.
///
/// The dictionary travels *with* the records rather than in a side file, so the
/// "dictionary entry before any record naming it" guarantee holds for a
/// live-tailing consumer without any cross-file write-ordering discipline.
pub fn encode_chunk(dict: &[u8], records: &[u8]) -> Vec<u8> {
    // nosemgrep: dos-unbounded-memory-allocation -- sum of two resident slice lens
    let mut out = Vec::with_capacity(12 + dict.len() + records.len());
    encode_chunk_into(&mut out, dict, records);
    out
}

/// As [`encode_chunk`], writing into a caller-owned buffer (cleared first).
///
/// A producer flushing every window would otherwise allocate and free a
/// records-sized `Vec` per chunk; reusing one buffer across chunks makes the
/// assembly two `memcpy`s and nothing else.
pub fn encode_chunk_into(out: &mut Vec<u8>, dict: &[u8], records: &[u8]) {
    out.clear();
    out.reserve(12 + dict.len() + records.len());
    out.extend_from_slice(&BBT_CHUNK_MAGIC);
    out.extend_from_slice(&(dict.len() as u32).to_le_bytes());
    out.extend_from_slice(&(records.len() as u32).to_le_bytes());
    out.extend_from_slice(dict);
    out.extend_from_slice(records);
}

/// Split a chunk into its `(dictionary, records)` sections.
pub fn parse_chunk(buf: &[u8]) -> Result<(&[u8], &[u8]), String> {
    if buf.len() < 12 {
        return Err(format!("BBT chunk too short: {} byte(s)", buf.len()));
    }
    if buf[0..4] != BBT_CHUNK_MAGIC {
        return Err(format!(
            "not a BBT chunk: expected magic {:?}, got {:?}. This usually means an \
             E-Trace stream is being decoded as BBT (or the reverse) -- check the \
             producer's `format=` argument against the consumer's.",
            BBT_CHUNK_MAGIC,
            &buf[0..4]
        ));
    }
    let dict_len = u32::from_le_bytes(buf[4..8].try_into().unwrap()) as usize;
    let rec_len = u32::from_le_bytes(buf[8..12].try_into().unwrap()) as usize;
    if buf.len() != 12 + dict_len + rec_len {
        return Err(format!(
            "BBT chunk length mismatch: header says {dict_len} dict + {rec_len} record \
             bytes, but the chunk carries {}",
            buf.len() - 12
        ));
    }
    if !rec_len.is_multiple_of(4) {
        return Err(format!(
            "BBT record section is {rec_len} bytes, not a multiple of 4"
        ));
    }
    Ok((&buf[12..12 + dict_len], &buf[12 + dict_len..]))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn block_records_are_their_own_id() {
        assert_eq!(bbt_block(0), 0);
        assert_eq!(bbt_block(10_328), 10_328);
        assert_eq!(bbt_decode(bbt_block(10_328)), BbtRecord::Block(10_328));
    }

    #[test]
    fn vl_records_round_trip_and_zero_is_a_value() {
        // vl == 0 is legitimate (the `vill` case) and must survive as a value.
        assert_eq!(bbt_decode(bbt_vl(0)), BbtRecord::Vl(0));
        assert_eq!(bbt_decode(bbt_vl(256)), BbtRecord::Vl(256));
        assert_eq!(bbt_decode(bbt_vl(65536)), BbtRecord::Vl(65536));
        // A vl record is never confusable with a block record.
        assert_ne!(bbt_vl(0) >> BBT_TAG_SHIFT, BBT_TAG_BLOCK);
    }

    #[test]
    fn reserved_tags_decode_as_reserved() {
        let partial = (0xC << BBT_TAG_SHIFT) | 3;
        assert_eq!(bbt_decode(partial), BbtRecord::Reserved(partial));
    }

    #[test]
    fn mem_groups_round_trip() {
        let mut out = Vec::new();
        push_mem_group(&mut out, MemGroupKind::Base, &[0x7fff_dead_beef]);
        let words: Vec<u32> = out
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| u32::from_le_bytes(*c))
            .collect();
        assert_eq!(
            bbt_decode(words[0]),
            BbtRecord::Mem {
                kind: MemGroupKind::Base as u32,
                len: 1
            }
        );
        let (BbtRecord::Addr(lo), BbtRecord::Addr(hi)) =
            (bbt_decode(words[1]), bbt_decode(words[2]))
        else {
            panic!("a Base group is a header and one address pair, got {words:#x?}");
        };
        assert_eq!(bbt_addr_join(lo, hi), 0x7fff_dead_beef);
    }

    #[test]
    fn a_full_width_address_survives_the_two_word_split() {
        // The widest value the encoding claims to carry, and the one just past
        // it, which must be rejected rather than truncated.
        let widest = (1u64 << BBT_ADDR_BITS) - 1;
        assert_eq!(
            bbt_addr_join(bbt_addr_lo(widest), bbt_addr_hi(widest)),
            widest
        );
        let mut out = Vec::new();
        assert!(
            std::panic::catch_unwind(move || push_mem_group(
                &mut out,
                MemGroupKind::Base,
                &[1 << BBT_ADDR_BITS]
            ))
            .is_err(),
            "an address wider than the encoding must fail loudly, not truncate"
        );
    }

    #[test]
    fn strides_survive_as_signed_values() {
        // Strides are signed and routinely negative; they travel wrapped to the
        // address width and are sign-extended back by the consumer.
        for stride in [-64i64, -1, 0, 1, 4096] {
            let wrapped = (stride as u64) & ((1 << BBT_ADDR_BITS) - 1);
            let joined = bbt_addr_join(bbt_addr_lo(wrapped), bbt_addr_hi(wrapped));
            assert_eq!(bbt_sign_extend_addr(joined), stride);
        }
    }

    #[test]
    fn a_mem_header_carries_its_kind_and_count() {
        for kind in [
            MemGroupKind::Base,
            MemGroupKind::BaseStride,
            MemGroupKind::Lines,
        ] {
            let w = bbt_mem(kind, 7);
            assert_eq!(
                bbt_decode(w),
                BbtRecord::Mem {
                    kind: kind as u32,
                    len: 7
                }
            );
            assert_eq!(MemGroupKind::from_u32(kind as u32), Some(kind));
        }
        // An unknown kind decodes as a record, but not as a known group: the
        // consumer errors at the point of use rather than guessing.
        assert_eq!(MemGroupKind::from_u32(9), None);
    }

    #[test]
    fn block_desc_expands_to_pcs() {
        let d = BlockDesc {
            id: 7,
            start_pc: 0x1000,
            insn_sizes: vec![4, 2, 2, 4],
        };
        assert_eq!(d.pcs(), vec![0x1000, 0x1004, 0x1006, 0x1008]);
    }

    #[test]
    fn dictionary_round_trips() {
        let mut dict = Vec::new();
        encode_block_desc(&mut dict, 0, 0x1000, &[4, 4]);
        encode_block_desc(&mut dict, 1, 0x2000, &[2]);
        encode_block_desc(&mut dict, 2, 0x3000, &[]);
        let descs = decode_block_descs(&dict).unwrap();
        assert_eq!(descs.len(), 3);
        assert_eq!(descs[0].pcs(), vec![0x1000, 0x1004]);
        assert_eq!(descs[1].pcs(), vec![0x2000]);
        assert!(descs[2].pcs().is_empty());
    }

    #[test]
    fn truncated_dictionary_is_an_error_not_a_silent_drop() {
        let mut dict = Vec::new();
        encode_block_desc(&mut dict, 0, 0x1000, &[4, 4]);
        for cut in 1..dict.len() {
            assert!(
                decode_block_descs(&dict[..cut]).is_err(),
                "a dictionary truncated at {cut} must be rejected"
            );
        }
    }

    #[test]
    fn chunk_round_trips() {
        let mut dict = Vec::new();
        encode_block_desc(&mut dict, 0, 0x1000, &[4]);
        let records: Vec<u8> = [bbt_block(0), bbt_vl(8)]
            .iter()
            .flat_map(|w| w.to_le_bytes())
            .collect();
        let chunk = encode_chunk(&dict, &records);
        let (d, r) = parse_chunk(&chunk).unwrap();
        assert_eq!(d, &dict[..]);
        assert_eq!(r, &records[..]);
    }

    #[test]
    fn an_etrace_stream_is_rejected_as_bbt() {
        // The magic exists so this fails loudly rather than decoding as garbage.
        let err = parse_chunk(&[0u8; 64]).unwrap_err();
        assert!(err.contains("not a BBT chunk"), "{err}");
    }
}
