//! RISC-V instruction decoding and ROI marker detection.
//!
//! ## RAVE vs. ROI: the naming
//!
//! RAVE is the *external*, user-facing C API that application code calls to
//! instrument itself: `rave_begin_region()`, `rave_end_region()`,
//! `rave_event_and_value()`, `rave_name_event()`, and friends (see
//! `test_bins/roi_validation/rave/rave_user_events_v{1,2}.h`). It is an ABI
//! contract owned by
//! someone else; we only consume it.
//!
//! What those calls leave behind in the instruction stream is what *we* call an
//! **ROI marker**, and the regions they delimit are **ROI regions**. Every module
//! downstream of this one speaks only in ROI terms; "RAVE" appears there only
//! when a comment names an actual function of the external API. This module is
//! the boundary: it is where the external encoding is recognised and translated
//! into our internal [`RoiMarker`] vocabulary.
//!
//! ## The encoding
//!
//! ROI markers are encoded as instructions that write to x0 (the zero register,
//! whose result is discarded), so they are architecturally inert and can be left
//! in production builds. The opcode plus the operand source distinguishes them:
//! - immediate-only forms (`li x0, -2`, `lui x0, imm`) carry their payload in the
//!   instruction word and can be resolved by static decode alone;
//! - R-type forms (`add`/`sub`/`or`/`and`/`sll`/`srl` with `rd = x0`) carry their
//!   payload in `rs1`/`rs2`, so they additionally need the runtime register
//!   values supplied by the tracer plugin ([`RoiRtMarker`]).
//!
//! They are used to begin/end named regions, control trace recording, mark
//! parallel execution boundaries, and transmit event metadata.
//!
//! ## Marker detection
//!
//! The [`InstructionDecoder`] decodes instructions and checks each one against
//! the marker encoding patterns above. The [`MultiDecoder`] extends this to
//! support multiple executable segments (main binary + shared libraries).
//!
//! [`RoiRtMarker`]: turbo_tracer_shared::RoiRtMarker

use anyhow::{anyhow, Result};
use riscv_isa::Target;
use std::{str::FromStr, sync::Arc};

/// What one decode-table lookup yields on the enrichment hot path: the decoded
/// instruction, its encoded bytes, its cached [`InsnClass`](crate::common::InsnClass),
/// and the ROI marker it encodes (if any).
pub type InsnLookup<'a> = (
    riscv_isa::Instruction,
    &'a [u8],
    crate::common::InsnClass,
    Option<RoiMarker>,
);

/// ROI marker types that can be detected in instruction stream
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RoiMarker {
    BeginRegion,
    EndRegion,
    RestartTrace,
    StartTrace,
    StopTrace,
    EnableRegions,
    DisableRegions,
    EventAndValue,
    NameEvent,
    EventString,
    ValueString,
    ParallelBarrier,
    ParallelBegin,
    ParallelEnd,
    // Version 1 markers (LUI-based encoding)
    NameDelimiter, // li x0, -1 (used as start/end of name string)
    NameHexDigit,  // lui x0, <hex> (hex digit in name encoding)
}

fn get_riscv_isa_target() -> riscv_isa::Target {
    riscv_isa::Target::from_str("RV64I").unwrap().full()
    // .or_else(|e| {
    //     let isa = "RV64IMAC_Zicsr_Zifencei";
    //     log::debug!("riscv_isa crate does not support V isa: {e:?}; falling back to {isa}");
    //     riscv_isa::Target::from_str(isa).map_err(|_| "Failed to create RISC-V target")
    // })
    // .expect("riscv_isa target strings: the 2nd hard coded target must be accepted.")
}

/// Holds a number of InstructionDecoders, to iter over the main binary, the .so files, etc
#[derive(Debug)]
pub struct MultiDecoder {
    pub dec: Vec<InstructionDecoder>,
}

impl MultiDecoder {
    pub fn find_decoder_for(&self, pc: u64) -> Option<&InstructionDecoder> {
        self.dec
            .iter()
            .find(|d| d.calculate_address_offset(pc).is_ok())
    }

    /// Like [`find_decoder_for`](Self::find_decoder_for), but returns the
    /// index into `dec` instead of the decoder itself. Used to pick the
    /// module-matched [`ElfResolver`](crate::processing::ElfResolver) out of
    /// the resolver list built alongside `dec` by
    /// [`Modules::from_metadata`](crate::processing::enricher::Modules::from_metadata),
    /// which is what keeps the two in the same module order.
    pub fn find_decoder_index_for(&self, pc: u64) -> Option<usize> {
        self.dec
            .iter()
            .position(|d| d.calculate_address_offset(pc).is_ok())
    }

    /// Decode table lookup for the E-Trace tracer's PC reconstruction, which
    /// asks for one address per reconstructed PC and cares only about the size
    /// and the control-flow shape of the instruction.
    ///
    /// `last` is the module index that answered the previous call, tried first
    /// and updated on success: consecutive PCs are nearly always in the same
    /// module, so this turns the module scan into one range check in the
    /// steady state. (The `riscv_etrace` `Multi` combinator this replaced did
    /// the same thing internally.)
    #[inline]
    pub fn decoded_at_memo(
        &self,
        address: u64,
        last: &mut usize,
    ) -> Option<(u8, riscv_isa::Instruction)> {
        if let Some(hit) = self.dec.get(*last).and_then(|d| d.decoded_at(address)) {
            return Some(hit);
        }
        for (i, d) in self.dec.iter().enumerate() {
            if let Some(hit) = d.decoded_at(address) {
                *last = i;
                return Some(hit);
            }
        }
        None
    }
}

/// Everything about one 2-byte-aligned address in a module's executable
/// segment that is a pure function of the instruction encoding (plus the
/// run-fixed `SYSTEM_VLEN`): the decode, its length, its `InsnClass`
/// classification and whether it is an ROI marker.
///
/// All four are computed together, once, when the module's table is built --
/// so the enrichment hot path is a single bounds-checked load, with no
/// `RefCell` borrow, no `Option` unwrapping and no "decoded but not yet
/// classified" second phase. The paired lookup was ~7% of
/// `transform_optimized`'s self time when class and decode lived apart.
///
/// The table is immutable once built, which is what lets every hart share one
/// copy: it used to be filled lazily behind a `RefCell`, forcing a private
/// ~26x-text-size allocation per hart per module.
#[derive(Clone, Copy)]
struct DecodedInsn {
    instruction: riscv_isa::Instruction,
    class: crate::common::InsnClass,
    marker: Option<RoiMarker>,
    /// Encoded length in bytes, taken from the length bits of the first
    /// halfword: 2 for RVC, 4 for a standard encoding, `0` for anything else.
    ///
    /// RV64 as implemented here has only 2- and 4-byte instructions, so a
    /// first byte carrying a 48-bit-or-wider length prefix is not an
    /// instruction at all -- it is data, or the tail of a 4-byte encoding, and
    /// gets `len: 0` like any other non-instruction slot.
    ///
    /// Note this is the *encoded* length, so it is known even when `decoded`
    /// is false — which is what lets the tracer keep advancing the PC across
    /// an instruction `riscv_isa` has no variant for.
    len: u8,
    /// Whether `riscv_isa` produced a real decode for these bytes. When false,
    /// `instruction` is `UNIMP` (control-flow-neutral, so the tracer can carry
    /// on) and the enricher's lookups report an error, as they did when this
    /// table was filled on demand.
    decoded: bool,
}

#[derive(Clone)]
pub struct InstructionDecoder {
    base_addr: u64,
    address: u64,
    data: Arc<[u8]>,
    target: Target,
    elf_data: Arc<[u8]>,
    /// Decode table, indexed by `offset / 2` (RISC-V instructions --
    /// compressed or not -- always start on a 2-byte boundary), built in full
    /// by [`build_table`] at construction time.
    ///
    /// Every slot is decoded independently at its own offset, exactly as the
    /// previous lazy fill did on demand; nothing here follows the instruction
    /// stream, so a slot landing mid-instruction or in inline data simply
    /// records `len == 0` and is never asked for.
    ///
    /// `Arc` rather than `Box` so cloning an `InstructionDecoder` (the
    /// `#[derive(Clone)]` above is used to hand modules to per-hart
    /// enrichers) shares the table instead of copying it.
    table: Arc<[DecodedInsn]>,
}

impl std::fmt::Debug for InstructionDecoder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("InstructionDecoder")
            .field("base_addr", &self.base_addr)
            .field("address", &self.address)
            .field("data.len()", &self.data.len())
            .field("target", &self.target)
            .field("elf_data.len()", &self.elf_data.len())
            .finish()
    }
}

/// Decode every 2-byte-aligned offset of a module's executable segment into a
/// flat, immutable table.
///
/// Each slot is decoded on its own, independently of its neighbours -- the
/// same thing the old lazy fill did per requested address, just done up front
/// for all of them. That keeps the result identical for any address a trace
/// can actually name, while making the table shareable (no interior
/// mutability) and turning the hot path into one indexed load.
///
/// Slots that do not decode -- offsets landing inside a 4-byte instruction, or
/// in constant pools embedded in `.text` -- get `len: 0` and are reported as a
/// decode error if anything ever asks for them.
fn build_table(exec_bytes: &[u8], target: &Target, vlen_bytes: u16) -> Arc<[DecodedInsn]> {
    // One slot per 2-byte-aligned offset (RVC allows 2-byte-aligned
    // instructions, so this is the finest granularity needed).
    let slots = exec_bytes.len().div_ceil(2);
    // nosemgrep: dos-unbounded-memory-allocation -- one slot per 2 bytes of an already-mapped ELF segment
    let mut table = Vec::with_capacity(slots);
    for slot in 0..slots {
        let offset = slot * 2;
        // Length comes from the encoding, not from the decode: `riscv_isa`
        // reports 4 bytes for the 48-bit-and-wider encodings it has no variant
        // for, which would make every following slot's meaning wrong for a
        // caller that steps by length. This is the Base Instruction-Length
        // Encoding of section 1.5 of the ISA manual, restricted to the only
        // two lengths RV64 defines here: 2 for RVC, 4 for a standard encoding.
        // A wider length prefix is by definition not an instruction, so the
        // slot is left empty like any other stretch of inline data.
        let first = exec_bytes[offset];
        let encoded_len: u8 = if first & 0b11 != 0b11 {
            2
        } else if first & 0b1_1100 != 0b1_1100 {
            4
        } else {
            0
        };
        // Not enough bytes left in the segment to hold it: no instruction here.
        let encoded_len = if offset + encoded_len as usize > exec_bytes.len() {
            0
        } else {
            encoded_len
        };
        let undecodable = |len: u8| DecodedInsn {
            instruction: riscv_isa::Instruction::UNIMP,
            class: crate::common::InsnClass::from_instruction(
                &riscv_isa::Instruction::UNIMP,
                vlen_bytes,
            ),
            marker: None,
            len,
            decoded: false,
        };
        let entry = match riscv_isa::decode_le_bytes(&exec_bytes[offset..], target) {
            // `riscv_isa` can return a length that disagrees with the
            // encoding (see above); the encoding wins.
            Some((instruction, len)) if len as u8 == encoded_len => DecodedInsn {
                class: crate::common::InsnClass::from_instruction(&instruction, vlen_bytes),
                marker: detect_roi_marker_bytes(
                    &exec_bytes[offset..(offset + len).min(exec_bytes.len())],
                ),
                instruction,
                len: encoded_len,
                decoded: true,
            },
            // Well-formed bytes `riscv_isa` has no variant for. The size is
            // still known, so the tracer can step over them; the enricher
            // cannot classify them and reports an error.
            _ => undecodable(encoded_len),
        };
        table.push(entry);
    }
    table.into()
}

impl InstructionDecoder {
    /// Build a decoder for one module, classifying against the run-fixed
    /// `SYSTEM_VLEN`.
    ///
    /// # Panics
    /// Panics if `SYSTEM_VLEN` has not been set yet -- the classification
    /// baked into the table depends on it, so building before it is known
    /// would silently mis-size every vector memory operation. Every production
    /// path sets it before the enricher is constructed.
    pub fn new(base_addr: u64, data: &[u8]) -> Result<InstructionDecoder> {
        Self::new_with_vlen(base_addr, data, crate::common::SYSTEM_VLEN.get_bytes())
    }

    /// As [`new`](Self::new), with the VLEN passed explicitly rather than read
    /// from the `SYSTEM_VLEN` global. For callers (benchmarks, tests) that
    /// build a decoder outside a configured run.
    pub fn new_with_vlen(
        base_addr: u64,
        data: &[u8],
        vlen_bytes: u16,
    ) -> Result<InstructionDecoder> {
        let target = get_riscv_isa_target();

        // Parse the ELF to find executable LOAD segments
        let elf = elf::ElfBytes::<elf::endian::LittleEndian>::minimal_parse(data)?;
        let elf_data: Arc<[u8]> = Arc::from(data);

        // Find the first executable LOAD segment
        let phdrs = elf
            .segments()
            .ok_or_else(|| anyhow!("No program headers found"))?;

        let mut exec_segment_addr: Option<u64> = None;
        let mut exec_segment_offset: Option<u64> = None;
        let mut exec_segment_size: Option<u64> = None;

        // Only the first executable LOAD is covered. Every ELF this has been
        // run against has exactly one, but a module with more would leave the
        // rest undecodable -- and since the E-Trace tracer now reads this same
        // table (see `TableBinary`), that would surface as a decode failure
        // rather than as merely missing enrichment. Say so loudly instead of
        // dropping it silently.
        for phdr in phdrs.iter() {
            // Look for LOAD segments with execute permission (PF_X = 0x1)
            if phdr.p_type == elf::abi::PT_LOAD && (phdr.p_flags & elf::abi::PF_X) != 0 {
                if exec_segment_addr.is_some() {
                    log::warn!(
                        "module at base 0x{:x} has more than one executable LOAD segment; \
                         only the first is decoded (vaddr 0x{:x}, size 0x{:x} ignored)",
                        base_addr,
                        phdr.p_vaddr,
                        phdr.p_filesz
                    );
                    continue;
                }
                exec_segment_addr = Some(phdr.p_vaddr);
                exec_segment_offset = Some(phdr.p_offset);
                exec_segment_size = Some(phdr.p_filesz);
            }
        }

        let address =
            exec_segment_addr.ok_or_else(|| anyhow!("No executable LOAD segment found"))?;
        let offset = exec_segment_offset.unwrap() as usize;
        let size = exec_segment_size.unwrap() as usize;

        // Extract the executable segment data
        let exec_bytes: Arc<[u8]> = Arc::from(&data[offset..offset + size]);

        // Detect if this is a VDSO or pre-relocated ELF by checking if the ELF's
        // virtual address matches or is close to the base_addr we're given.
        // VDSOs have their runtime addresses baked into the ELF, so we shouldn't
        // add an offset. Regular shared libraries have low addresses (near 0) and
        // need the base_addr offset applied.
        let actual_base_addr = if address >= base_addr && address < base_addr + 0x10000 {
            // ELF addresses are already at or near the runtime address (VDSO case)
            // Don't apply relocation offset
            0
        } else {
            // Normal case: ELF has low addresses, apply the base_addr offset
            base_addr
        };

        // log::debug!(
        //     "InstructionDecoder range base addr 0x{:x} (input 0x{:x}) address start 0x{:x} to 0x{:x}, runtime range 0x{:x} to 0x{:x}",
        //     actual_base_addr,
        //     base_addr,
        //     address,
        //     address + exec_bytes.len() as u64,
        //     actual_base_addr + address,
        //     actual_base_addr + address + exec_bytes.len() as u64
        // );

        let table = build_table(&exec_bytes, &target, vlen_bytes);

        Ok(InstructionDecoder {
            base_addr: actual_base_addr,
            address,
            data: exec_bytes,
            target,
            elf_data,
            table,
        })
    }

    /// Build a decoder directly over a raw block of instruction bytes mapped at
    /// `text_addr`, bypassing ELF parsing.
    ///
    /// Test-only: unit tests for consumers of the decoder (e.g. the run cache's
    /// termination rules) need a handful of hand-assembled instructions at known
    /// addresses, and wrapping those in a synthetic ELF would add a lot of
    /// machinery that tests nothing. There is no `elf_data`, so any of the
    /// symbol/string helpers on this type will find nothing.
    #[cfg(test)]
    pub(crate) fn from_exec_bytes(text_addr: u64, exec_bytes: &[u8]) -> Self {
        let target = get_riscv_isa_target();
        let table = build_table(exec_bytes, &target, crate::common::SYSTEM_VLEN.get_bytes());
        InstructionDecoder {
            base_addr: 0,
            address: text_addr,
            data: Arc::from(exec_bytes),
            target,
            elf_data: Arc::from(&[][..]),
            table,
        }
    }

    /// Decode the instruction at `address` and return it together with its
    /// bytes, its precomputed [`InsnClass`] classification and whether it is a
    /// ROI marker.
    ///
    /// All of it is a single indexed load out of the module's decode table,
    /// which was fully built (against the run-fixed `SYSTEM_VLEN`) at
    /// construction time. This is the enrichment hot path's only decode entry
    /// point.
    #[inline]
    pub fn get_instruction_class_and_marker_at(
        &self,
        address: u64,
    ) -> core::result::Result<InsnLookup<'_>, &str> {
        let offset = self.calculate_address_offset(address)?;
        let entry = &self.table[offset / 2];
        if !entry.decoded {
            return Err("decoding the instruction bytes failed: objdump the addr for info");
        }
        Ok((
            entry.instruction,
            &self.data[offset..offset + entry.len as usize],
            entry.class,
            entry.marker,
        ))
    }

    /// The decode and encoded length at `address`, or `None` if this module
    /// does not cover it (or the bytes there do not decode).
    ///
    /// The tracer's entry point: unlike the enricher's lookups it wants
    /// neither the raw bytes nor the classification, so it skips the slicing
    /// and returns a miss rather than a message.
    #[inline]
    pub fn decoded_at(&self, address: u64) -> Option<(u8, riscv_isa::Instruction)> {
        let offset = self.calculate_address_offset(address).ok()?;
        let entry = &self.table[offset / 2];
        // `decoded` deliberately not consulted: an encoding `riscv_isa` does
        // not know is still an instruction of known size that the tracer must
        // step over, exactly as the decoder this replaced treated it.
        (entry.len != 0).then_some((entry.len, entry.instruction))
    }

    #[inline]
    pub fn get_instruction_at(
        &self,
        address: u64,
    ) -> core::result::Result<(riscv_isa::Instruction, &[u8]), &str> {
        let offset = self.calculate_address_offset(address)?;
        let entry = &self.table[offset / 2];
        if !entry.decoded {
            return Err("decoding the instruction bytes failed: objdump the addr for info");
        }
        Ok((
            entry.instruction,
            &self.data[offset..offset + entry.len as usize],
        ))
    }

    /// The `len` raw encoding bytes at `address`, without decoding.
    ///
    /// For callers that already know what the instruction *is* (the run cache
    /// keeps the decode) but still need its bytes -- e.g. `PcProfile::record`,
    /// which stores them on a PC's first sighting for disassembly rendering.
    /// Skipping the decode-cache probe is the whole point: in the enricher's
    /// batched fast path this is on the per-instruction path that the run
    /// aggregate exists to keep cheap.
    #[inline]
    pub fn encoded_bytes_at(&self, address: u64, len: u8) -> Option<&[u8]> {
        let offset = self.calculate_address_offset(address).ok()?;
        self.data.get(offset..offset + len as usize)
    }

    // Public for now to allow struct to be used with existing functions
    #[inline]
    pub fn calculate_address_offset(&self, address: u64) -> core::result::Result<usize, &str> {
        // Calculate runtime address of the .text section
        let runtime_text_start = self.base_addr + self.address;

        if address < runtime_text_start {
            return Err("address too low");
        }

        let offset = (address - runtime_text_start) as usize;
        if offset >= self.data.len() {
            return Err("address too high");
        }

        Ok(offset)
    }
}

/// Detect if an instruction is an ROI marker
///
/// ROI markers are instructions that write to x0 (the zero register),
/// which is hardwired to zero and discards all writes. This makes them
/// "invisible" no-ops that can be used for tracing.
///
/// A pure function of the encoding, so it is a free function: [`build_table`]
/// calls it while filling a module's decode table, before any
/// `InstructionDecoder` exists to call it on.
#[inline]
pub fn detect_roi_marker_bytes(insn_bytes: &[u8]) -> Option<RoiMarker> {
    if insn_bytes.len() < 4 {
        return None;
    }

    // Decode as little-endian 32-bit instruction
    let insn_opcode =
        u32::from_le_bytes([insn_bytes[0], insn_bytes[1], insn_bytes[2], insn_bytes[3]]);

    // Extract bit fields
    let dst = (insn_opcode >> 7) & 0x1F;

    // ROI markers must write to x0
    if dst != 0 {
        return None;
    }

    let major = insn_opcode & 0x7F;
    let funct3 = (insn_opcode >> 12) & 0x7;
    let funct6 = (insn_opcode >> 26) & 0x3F;
    let imm = (insn_opcode as i32) >> 20;

    // Check for different ROI marker patterns
    match major {
        0x13 => {
            // I-type instructions (addi x0, x0, imm) -> li x0, imm
            if funct3 == 0 {
                match imm {
                    -1 => Some(RoiMarker::NameDelimiter), // Version 1: string delimiter
                    -2 => Some(RoiMarker::RestartTrace),
                    -3 => Some(RoiMarker::StartTrace),
                    -4 => Some(RoiMarker::StopTrace),
                    -5 => Some(RoiMarker::ParallelBarrier),
                    -6 => Some(RoiMarker::ParallelEnd),
                    -7 => Some(RoiMarker::EnableRegions),
                    -8 => Some(RoiMarker::DisableRegions),
                    _ => None,
                }
            } else {
                None
            }
        }
        0x37 => {
            // U-type: lui x0, imm
            // Version 1 ROI markers use lui x0, <hex> for encoding hex digits
            Some(RoiMarker::NameHexDigit)
        }
        0x33 => {
            // R-type instructions
            match funct3 {
                0x0 => {
                    // ADD or SUB
                    if funct6 == 0x00 {
                        // add x0, rs1, rs2 -> Begin region
                        Some(RoiMarker::BeginRegion)
                    } else if funct6 == 0x10 {
                        // sub x0, rs1, rs2 -> End region
                        Some(RoiMarker::EndRegion)
                    } else {
                        None
                    }
                }
                0x1 => {
                    // SLL - shift left logical
                    if funct6 == 0x00 {
                        // sll x0, rs1, rs2 -> Event string
                        Some(RoiMarker::EventString)
                    } else {
                        None
                    }
                }
                0x4 => {
                    // XOR
                    if funct6 == 0x00 {
                        // xor x0, rs1, rs2 -> Parallel begin
                        Some(RoiMarker::ParallelBegin)
                    } else {
                        None
                    }
                }
                0x5 => {
                    // SRL - shift right logical
                    if funct6 == 0x00 {
                        // srl x0, rs1, rs2 -> Value string
                        Some(RoiMarker::ValueString)
                    } else {
                        None
                    }
                }
                0x6 => {
                    // OR
                    // or x0, rs1, rs2 -> Event and value
                    Some(RoiMarker::EventAndValue)
                }
                0x7 => {
                    // AND
                    // and x0, rs1, rs2 -> Name event/value
                    Some(RoiMarker::NameEvent)
                }
                _ => None,
            }
        }
        _ => None,
    }
}

impl InstructionDecoder {
    /// Extract the nibble value from a LUI x0 instruction (NameHexDigit marker).
    /// The nibble is in bits [15:12] of the immediate (bits [31:12] of instruction).
    /// For `lui x0, N`, the raw immediate upper 20 bits are insn[31:12],
    /// and the nibble value is the lowest 4 bits of that field (insn[15:12]).
    #[inline]
    pub fn extract_lui_nibble(insn_bytes: &[u8]) -> Option<u8> {
        if insn_bytes.len() < 4 {
            return None;
        }
        let insn = u32::from_le_bytes([insn_bytes[0], insn_bytes[1], insn_bytes[2], insn_bytes[3]]);
        // LUI: imm[31:12] is insn[31:12]
        let imm_upper = insn >> 12;
        Some((imm_upper & 0xf) as u8)
    }

    /// Extract string name from ROI region marker by analyzing preceding instructions
    ///
    /// Supports two versions:
    ///
    /// **Version 2** (pointer-based):
    /// ```asm
    /// auipc reg, upper_imm    # Load upper 20 bits + PC
    /// addi  reg, reg, lower   # Add lower 12 bits
    /// li    a5, -1            # Load -1 (may be compressed 2-byte instruction)
    /// add   zero, reg, a5     # ROI marker (or sub for end_region)
    /// ```
    ///
    /// **Version 1** (inline encoding):
    /// ```asm
    /// and   zero, rs1, rs2    # NameEvent marker
    /// li    zero, -1          # Start delimiter
    /// lui   zero, <hex>       # Hex digit (repeated for each char)
    /// li    zero, -1          # End delimiter
    /// ```
    ///
    /// We decode the auipc/addi pair to calculate the string address,
    /// then read the string from the ELF binary.
    pub fn extract_region_name(&self, marker_pc: u64) -> Option<String> {
        // Look back through the instruction stream to find auipc/addi pair
        // We need to scan backwards byte-by-byte since compressed instructions
        // can be 2 or 4 bytes, making fixed-stride lookback unreliable

        // Start looking back from marker_pc - 2 (minimum instruction size)
        // Look back up to 32 bytes to find the pattern
        for byte_offset in (2..=32).step_by(2) {
            let check_pc = marker_pc.checked_sub(byte_offset)?;

            if let Ok(offset) = self.calculate_address_offset(check_pc) {
                if offset + 8 <= self.data.len() {
                    // Try to decode auipc/addi pair
                    let insn1_bytes = &self.data[offset..offset + 4];
                    let insn2_bytes = &self.data[offset + 4..offset + 8];

                    let insn1 = u32::from_le_bytes([
                        insn1_bytes[0],
                        insn1_bytes[1],
                        insn1_bytes[2],
                        insn1_bytes[3],
                    ]);
                    let insn2 = u32::from_le_bytes([
                        insn2_bytes[0],
                        insn2_bytes[1],
                        insn2_bytes[2],
                        insn2_bytes[3],
                    ]);

                    // Check for auipc (opcode 0x17)
                    if (insn1 & 0x7F) == 0x17 {
                        // Check for addi (opcode 0x13, funct3 0x0)
                        if (insn2 & 0x7F) == 0x13 && ((insn2 >> 12) & 0x7) == 0x0 {
                            // Extract destination registers
                            let rd1 = (insn1 >> 7) & 0x1F;
                            let rd2 = (insn2 >> 7) & 0x1F;
                            let rs1 = (insn2 >> 15) & 0x1F;

                            // Verify addi uses same register: addi rd2, rs1, imm where rd2 == rs1 == rd1
                            if rd1 == rd2 && rd2 == rs1 {
                                // Decode auipc: upper 20 bits
                                let upper_imm = (insn1 & 0xFFFFF000) as i32 as i64;

                                // Decode addi: lower 12 bits (sign-extended)
                                let lower_imm = ((insn2 as i32) >> 20) as i64;

                                // Calculate address: PC of auipc + upper_imm + lower_imm
                                let string_addr = (check_pc as i64 + upper_imm + lower_imm) as u64;

                                // Try to read string from ELF
                                if let Some(name) = self.read_string_from_elf(string_addr) {
                                    return Some(name);
                                }
                            }
                        }
                    }
                }
            }
        }

        None
    }

    /// Read a null-terminated string from the ELF binary at the given virtual address.
    /// `vaddr` is a runtime (PIE-relocated) address; we subtract `base_addr` to
    /// recover the ELF-file virtual address before searching sections.
    pub fn read_string_from_elf(&self, vaddr: u64) -> Option<String> {
        use object::{Object, ObjectSection};

        // Convert runtime address back to ELF virtual address
        let vaddr = vaddr.checked_sub(self.base_addr).unwrap_or(vaddr);

        // Parse ELF to find the section containing this address
        let obj_file = object::File::parse(&*self.elf_data).ok()?;

        for section in obj_file.sections() {
            let section_addr = section.address();
            let section_size = section.size();

            if vaddr >= section_addr && vaddr < section_addr + section_size {
                let offset = (vaddr - section_addr) as usize;
                let section_data = section.data().ok()?;

                if offset < section_data.len() {
                    // Read null-terminated string
                    let mut end = offset;
                    while end < section_data.len() && section_data[end] != 0 {
                        end += 1;
                    }

                    if end > offset {
                        let string_bytes = &section_data[offset..end];
                        if let Ok(s) = std::str::from_utf8(string_bytes) {
                            return Some(s.to_string());
                        }
                    }
                }
            }
        }

        None
    }

    /// Read a string of known length from the ELF binary at the given runtime virtual address.
    /// `vaddr` is a runtime (PIE-relocated) address; `len` is the byte count.
    pub fn read_string_from_elf_with_len(&self, vaddr: u64, len: u64) -> Option<String> {
        use object::{Object, ObjectSection};

        let len = len as usize;
        if len == 0 || len > 4096 {
            return None;
        }

        // Convert runtime address back to ELF virtual address
        let elf_vaddr = vaddr.checked_sub(self.base_addr).unwrap_or(vaddr);

        let obj_file = object::File::parse(&*self.elf_data).ok()?;

        for section in obj_file.sections() {
            let section_addr = section.address();
            let section_size = section.size();

            if elf_vaddr >= section_addr && elf_vaddr < section_addr + section_size {
                let offset = (elf_vaddr - section_addr) as usize;
                let section_data = section.data().ok()?;

                if offset + len <= section_data.len() {
                    let string_bytes = &section_data[offset..offset + len];
                    if let Ok(s) = std::str::from_utf8(string_bytes) {
                        return Some(s.to_string());
                    }
                }
            }
        }

        None
    }

    /// Read raw instruction word at given PC address
    /// Returns the 32-bit instruction word, or None if address is invalid
    fn read_instruction(&self, pc: u64) -> Option<u32> {
        let offset = self.calculate_address_offset(pc).ok()?;

        if offset + 4 <= self.data.len() {
            // Read 32-bit instruction in little-endian
            let bytes = &self.data[offset..offset + 4];
            Some(u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
        } else if offset + 2 <= self.data.len() {
            // Might be a 16-bit compressed instruction
            let bytes = &self.data[offset..offset + 2];
            Some(u16::from_le_bytes([bytes[0], bytes[1]]) as u32)
        } else {
            None
        }
    }

    /// Extract event ID from a NameEvent marker (and x0, rs1, rs2)
    /// Returns the value in rs1 which contains the event ID
    pub fn extract_event_id(&self, name_event_pc: u64) -> Option<u64> {
        // The NameEvent marker is: and x0, rs1, rs2
        // We need to extract the value in rs1 at the time of this instruction
        // This requires looking at the instruction that loaded rs1

        let name_event_insn = self.read_instruction(name_event_pc)?;

        // Extract rs1 from the and instruction (event ID is in rs1)
        let rs1 = ((name_event_insn >> 15) & 0x1f) as u8;

        log::trace!(
            "ROI: NameEvent at {:#x}, rs1=x{}, looking for event ID",
            name_event_pc,
            rs1
        );

        // Look backwards for the instruction that loaded rs1
        // Use a larger search window since registers like s1 may be loaded far back
        for offset in (1..=100).step_by(1) {
            let check_pc = name_event_pc.checked_sub(offset * 2)?;

            if let Some(prev_insn) = self.read_instruction(check_pc) {
                let opcode = prev_insn & 0x7f;

                // Check for compressed instruction (li a5, imm)
                if (prev_insn & 0x3) != 0x3 {
                    // This is a 16-bit compressed instruction
                    let rd_compressed = ((prev_insn >> 7) & 0x1f) as u8;

                    if rd_compressed == rs1 {
                        // Check if it's a C.LI instruction (opcode 0b01, funct3 0b010)
                        if (prev_insn & 0x3) == 0x01 && ((prev_insn >> 13) & 0x7) == 0x2 {
                            let imm = ((prev_insn >> 2) & 0x1f) | (((prev_insn >> 12) & 0x1) << 5);
                            // Sign extend 6-bit immediate
                            let value = if (imm & 0x20) != 0 {
                                (imm | 0xffffffc0) as i8 as i64 as u64
                            } else {
                                imm as u64
                            };

                            log::trace!(
                                "ROI: Found C.LI x{}, {} at {:#x} (compressed)",
                                rs1,
                                value as i64,
                                check_pc
                            );
                            return Some(value);
                        }
                    }
                }

                // Check for 32-bit instructions (li/addi)
                if opcode == 0x13 {
                    // I-type: addi rd, rs1, imm
                    let rd = ((prev_insn >> 7) & 0x1f) as u8;
                    let funct3 = (prev_insn >> 12) & 0x7;
                    let rs1_src = ((prev_insn >> 15) & 0x1f) as u8;

                    if rd == rs1 && funct3 == 0 && rs1_src == 0 {
                        // This is li rd, imm (which is addi rd, x0, imm)
                        let imm = (prev_insn as i32) >> 20; // Sign-extend 12-bit immediate
                        let value = imm as i64 as u64;

                        log::trace!("ROI: Found li x{}, {} at {:#x}", rs1, imm, check_pc);
                        return Some(value);
                    }
                }
            }
        }

        log::warn!(
            "ROI: Could not find value for rs1=x{} before NameEvent at {:#x}",
            rs1,
            name_event_pc
        );
        None
    }

    /// Extract event value from a NameEvent marker's rs2 (and x0, rs1, rs2)
    /// Returns the value in rs2 which may contain a sub-event value for hierarchical naming
    /// Returns None if rs2 is -1 or 0 (indicating no value, just an event declaration)
    pub fn extract_event_value(&self, name_event_pc: u64) -> Option<u64> {
        // The NameEvent marker is: and x0, rs1, rs2
        // rs2 may contain a value for rave_name_value(event_id, value, name)
        // If rs2 is -1 or 0, it's just rave_name_event(event_id, name) with no value

        let name_event_insn = self.read_instruction(name_event_pc)?;

        // Extract rs2 from the and instruction
        let rs2 = ((name_event_insn >> 20) & 0x1f) as u8;

        log::trace!(
            "ROI: NameEvent at {:#x}, rs2=x{}, looking for value",
            name_event_pc,
            rs2
        );

        // Look backwards for the instruction that loaded rs2
        for offset in (1..=10).step_by(1) {
            let check_pc = name_event_pc.checked_sub(offset * 2)?;

            if let Some(prev_insn) = self.read_instruction(check_pc) {
                let opcode = prev_insn & 0x7f;

                // Check for compressed instruction (li)
                if (prev_insn & 0x3) != 0x3 {
                    // This is a 16-bit compressed instruction
                    let rd_compressed = ((prev_insn >> 7) & 0x1f) as u8;

                    if rd_compressed == rs2 {
                        // Check if it's a C.LI instruction
                        if (prev_insn & 0x3) == 0x01 && ((prev_insn >> 13) & 0x7) == 0x2 {
                            let imm = ((prev_insn >> 2) & 0x1f) | (((prev_insn >> 12) & 0x1) << 5);
                            // Sign extend 6-bit immediate
                            let value = if (imm & 0x20) != 0 {
                                (imm | 0xffffffc0) as i8 as i64 as u64
                            } else {
                                imm as u64
                            };

                            // -1 means no value (just event declaration)
                            if value == u64::MAX {
                                log::trace!("ROI: rs2 is -1, no value for this NameEvent");
                                return None;
                            }

                            log::trace!(
                                "ROI: Found value={} at {:#x} (C.LI x{})",
                                value,
                                check_pc,
                                rs2
                            );
                            return Some(value);
                        }
                    }
                }

                // Check for 32-bit instructions (li/addi)
                if opcode == 0x13 {
                    // I-type: addi rd, rs1_src, imm
                    let rd = ((prev_insn >> 7) & 0x1f) as u8;
                    let funct3 = (prev_insn >> 12) & 0x7;
                    let rs1_src = ((prev_insn >> 15) & 0x1f) as u8;

                    if rd == rs2 && funct3 == 0 && rs1_src == 0 {
                        // This is li rd, imm (which is addi rd, x0, imm)
                        let imm = (prev_insn as i32) >> 20; // Sign-extend 12-bit immediate
                        let value = imm as i64 as u64;

                        // -1 means no value (just event declaration)
                        if imm == -1 {
                            log::debug!("ROI: rs2 is -1, no value for this NameEvent");
                            return None;
                        }

                        log::trace!("ROI: Found value={} at {:#x} (li x{})", imm, check_pc, rs2);
                        return Some(value);
                    }
                }
            }
        }

        log::warn!(
            "ROI: Could not find value for rs2=x{} before NameEvent at {:#x}",
            rs2,
            name_event_pc
        );
        None
    }

    /// Extract event ID and value from an EventAndValue marker (or x0, rs1, rs2)
    /// Returns (event_id, value) where rs1 contains event_id and rs2 contains value
    pub fn extract_event_and_value(&self, marker_pc: u64) -> Option<(u64, u64)> {
        // The EventAndValue marker is: or x0, rs1, rs2
        // We need to extract the values in rs1 and rs2 at the time of this instruction

        let marker_insn = self.read_instruction(marker_pc)?;

        // Extract rs1 and rs2 from the or instruction
        let rs1 = ((marker_insn >> 15) & 0x1f) as u8;
        let rs2 = ((marker_insn >> 20) & 0x1f) as u8;

        log::trace!(
            "ROI: EventAndValue at {:#x}, rs1=x{}, rs2=x{}, looking for values",
            marker_pc,
            rs1,
            rs2
        );

        // Look backwards for the instructions that loaded rs1 and rs2
        // Use a larger search window since registers like s1 may be loaded far back
        let mut event_id: Option<u64> = None;
        let mut value: Option<u64> = None;

        for offset in (1..=200).step_by(1) {
            let check_pc = marker_pc.checked_sub(offset * 2)?;

            if let Some(prev_insn) = self.read_instruction(check_pc) {
                let opcode = prev_insn & 0x7f;

                // Check for compressed instruction (li)
                if (prev_insn & 0x3) != 0x3 {
                    // This is a 16-bit compressed instruction
                    let rd_compressed = ((prev_insn >> 7) & 0x1f) as u8;

                    // Check if it's a C.LI instruction
                    if (prev_insn & 0x3) == 0x01 && ((prev_insn >> 13) & 0x7) == 0x2 {
                        let imm = ((prev_insn >> 2) & 0x1f) | (((prev_insn >> 12) & 0x1) << 5);
                        // Sign extend 6-bit immediate
                        let val = if (imm & 0x20) != 0 {
                            (imm | 0xffffffc0) as i8 as i64 as u64
                        } else {
                            imm as u64
                        };

                        if rd_compressed == rs1 && event_id.is_none() {
                            log::trace!(
                                "ROI: Found event_id={} at {:#x} (C.LI x{})",
                                val as i64,
                                check_pc,
                                rs1
                            );
                            event_id = Some(val);
                        }
                        if rd_compressed == rs2 && value.is_none() {
                            log::trace!(
                                "ROI: Found value={} at {:#x} (C.LI x{})",
                                val as i64,
                                check_pc,
                                rs2
                            );
                            value = Some(val);
                        }
                    }
                }

                // Check for 32-bit instructions (li/addi)
                if opcode == 0x13 {
                    // I-type: addi rd, rs1_src, imm
                    let rd = ((prev_insn >> 7) & 0x1f) as u8;
                    let funct3 = (prev_insn >> 12) & 0x7;
                    let rs1_src = ((prev_insn >> 15) & 0x1f) as u8;

                    if funct3 == 0 && rs1_src == 0 {
                        // This is li rd, imm (which is addi rd, x0, imm)
                        let imm = (prev_insn as i32) >> 20; // Sign-extend 12-bit immediate
                        let val = imm as i64 as u64;

                        if rd == rs1 && event_id.is_none() {
                            log::trace!(
                                "ROI: Found event_id={} at {:#x} (li x{})",
                                imm,
                                check_pc,
                                rs1
                            );
                            event_id = Some(val);
                        }
                        if rd == rs2 && value.is_none() {
                            log::trace!(
                                "ROI: Found value={} at {:#x} (li x{})",
                                imm,
                                check_pc,
                                rs2
                            );
                            value = Some(val);
                        }
                    }
                }

                // Stop searching once we've found both values
                if event_id.is_some() && value.is_some() {
                    break;
                }
            }
        }

        match (event_id, value) {
            (Some(eid), Some(val)) => {
                log::trace!(
                    "ROI: EventAndValue at {:#x} decoded as event_id={}, value={}",
                    marker_pc,
                    eid,
                    val
                );
                Some((eid, val))
            }
            _ => {
                log::warn!(
                    "ROI: Could not find both values for EventAndValue at {:#x} (event_id={:?}, value={:?})",
                    marker_pc,
                    event_id,
                    value
                );
                None
            }
        }
    }

    /// Extract event ID and value from an EventAndValue marker using execution history
    /// This is needed when the marker is inside a function and registers are loaded from stack
    /// Returns (event_id, value) by looking back through the execution trace
    pub fn extract_event_and_value_from_trace(
        &self,
        marker_pc: u64,
        pc_history: &[u64],
        current_index: usize,
    ) -> Option<(u64, u64)> {
        // The EventAndValue marker is: or x0, rs1, rs2
        // We need to find what values were loaded into rs1 and rs2 by looking at the execution trace

        let marker_insn = self.read_instruction(marker_pc)?;

        // Extract rs1 and rs2 from the or instruction
        let rs1 = ((marker_insn >> 15) & 0x1f) as u8;
        let rs2 = ((marker_insn >> 20) & 0x1f) as u8;

        log::trace!(
            "ROI: EventAndValue at {:#x} (trace index {}), rs1=x{}, rs2=x{}, scanning trace history",
            marker_pc,
            current_index,
            rs1,
            rs2
        );

        let mut event_id: Option<u64> = None;
        let mut value: Option<u64> = None;

        // Look backwards through the execution trace (up to 200 instructions)
        let start_index = current_index.saturating_sub(200);
        for i in (start_index..current_index).rev() {
            let check_pc = pc_history[i];

            if let Some(prev_insn) = self.read_instruction(check_pc) {
                let opcode = prev_insn & 0x7f;

                // Check for compressed instruction (li)
                if (prev_insn & 0x3) != 0x3 {
                    // This is a 16-bit compressed instruction
                    let rd_compressed = ((prev_insn >> 7) & 0x1f) as u8;

                    // Check if it's a C.LI instruction
                    if (prev_insn & 0x3) == 0x01 && ((prev_insn >> 13) & 0x7) == 0x2 {
                        let imm = ((prev_insn >> 2) & 0x1f) | (((prev_insn >> 12) & 0x1) << 5);
                        // Sign extend 6-bit immediate
                        let val = if (imm & 0x20) != 0 {
                            (imm | 0xffffffc0) as i8 as i64 as u64
                        } else {
                            imm as u64
                        };

                        if rd_compressed == rs1 && event_id.is_none() {
                            log::trace!(
                                "ROI: Found event_id={} at trace[{}] PC {:#x} (C.LI x{})",
                                val as i64,
                                i,
                                check_pc,
                                rs1
                            );
                            event_id = Some(val);
                        }
                        if rd_compressed == rs2 && value.is_none() {
                            log::trace!(
                                "ROI: Found value={} at trace[{}] PC {:#x} (C.LI x{})",
                                val as i64,
                                i,
                                check_pc,
                                rs2
                            );
                            value = Some(val);
                        }
                    }
                }

                // Check for 32-bit instructions (li/addi)
                if opcode == 0x13 {
                    // I-type: addi rd, rs1_src, imm
                    let rd = ((prev_insn >> 7) & 0x1f) as u8;
                    let funct3 = (prev_insn >> 12) & 0x7;
                    let rs1_src = ((prev_insn >> 15) & 0x1f) as u8;

                    if funct3 == 0 && rs1_src == 0 {
                        // This is li rd, imm (which is addi rd, x0, imm)
                        let imm = (prev_insn as i32) >> 20; // Sign-extend 12-bit immediate
                        let val = imm as i64 as u64;

                        if rd == rs1 && event_id.is_none() {
                            log::trace!(
                                "ROI: Found event_id={} at trace[{}] PC {:#x} (li x{})",
                                imm,
                                i,
                                check_pc,
                                rs1
                            );
                            event_id = Some(val);
                        }
                        if rd == rs2 && value.is_none() {
                            log::trace!(
                                "ROI: Found value={} at trace[{}] PC {:#x} (li x{})",
                                imm,
                                i,
                                check_pc,
                                rs2
                            );
                            value = Some(val);
                        }
                    }
                }

                // Stop searching once we've found both values
                if event_id.is_some() && value.is_some() {
                    break;
                }
            }
        }

        match (event_id, value) {
            (Some(eid), Some(val)) => {
                log::trace!(
                    "ROI: EventAndValue at {:#x} decoded from trace as event_id={}, value={}",
                    marker_pc,
                    eid,
                    val
                );
                Some((eid, val))
            }
            _ => {
                log::warn!(
                    "ROI: Could not find both values in trace for EventAndValue at {:#x} (event_id={:?}, value={:?})",
                    marker_pc,
                    event_id,
                    value
                );
                None
            }
        }
    }
}
