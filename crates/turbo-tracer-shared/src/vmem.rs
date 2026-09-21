//! Decoding RISC-V vector load/store instructions, and turning one into the set
//! of bytes it touches.
//!
//! # Why this is shared
//!
//! Both sides of the trace need the same answer about the same instruction: the
//! tracer, at translate time, to decide what a callback must read
//! ([`VecMemOp::group_kind`]); the consumer, offline, to expand a captured base
//! address back into a byte range ([`VecMemOp::segments`]). Two independent
//! decoders of one field layout is how the two sides drift apart, so there is
//! one, here.
//!
//! # What is decoded
//!
//! The vector load/store major opcodes are `LOAD-FP` (`0b0000111`) and
//! `STORE-FP` (`0b0100111`), which they share with the scalar `flw`/`fld`/`flq`
//! family. The `width` field separates them: the scalar widths are `0b010`,
//! `0b011` and `0b100`, and everything else in that field is a vector op (the V
//! spec, section 7.3). The remaining fields are `nf` (31:29), `mew` (28), `mop`
//! (27:26), `vm` (25), `lumop`/`rs2`/`vs2` (24:20), `rs1` (19:15) and
//! `vd`/`vs3` (11:7).
//!
//! `mew` (the "extended" width bit, reserved for 128-bit elements) is decoded
//! and rejected rather than ignored: `mew = 1` would change the element width
//! this module reports, and a wrong element width is a wrong footprint.

/// Cache line size, in bytes, that line addresses in the trace are aligned to.
///
/// A single constant rather than a parameter because it is baked into what the
/// producer emits for indexed ops (it deduplicates by line *at capture time*,
/// which is the only place the index vector exists). A simulator for a machine
/// with a different line size can still split these, but not join them, so this
/// is deliberately the smallest line size worth modelling.
pub const CACHE_LINE_BYTES: u64 = 64;

/// The cache line containing `addr`.
#[inline(always)]
pub fn line_of(addr: u64) -> u64 {
    addr & !(CACHE_LINE_BYTES - 1)
}

/// The `SEW` in bits that a `vsetvli`/`vsetivli` sets, from its `vtype`
/// immediate. `None` for `vsetvl` (whose `vtype` is a register) and for anything
/// that is not a `vsetvl*`.
///
/// Lives here, next to the memory decode, because `SEW` is what an *indexed*
/// memory op's data width is: its own `width` field describes its indices
/// instead (V spec 7.6). The tracer needs it to expand a gather at runtime and
/// the consumer needs it to expand everything else offline, so — like
/// [`VecMemOp`] — there is one decoder of it.
pub fn vsetvl_static_sew_bits(insn: u32) -> Option<u16> {
    // OP-V (0b1010111) with funct3 = 0b111 (OPCFG) is the vsetvl* group.
    if insn & 0x7f != 0x57 || (insn >> 12) & 0x7 != 0x7 {
        return None;
    }
    let vtype = if insn >> 31 == 0 {
        // vsetvli: zimm11 = insn[30:20].
        (insn >> 20) & 0x7ff
    } else if insn >> 30 == 0b11 {
        // vsetivli: zimm10 = insn[29:20].
        (insn >> 20) & 0x3ff
    } else {
        return None;
    };
    Some(sew_bits_from_vtype(u64::from(vtype)))
}

/// `SEW` in bits for a `vtype`, i.e. `8 << vsew`.
///
/// Defined for every `vtype`, including illegal (`vill`) ones: those yield
/// `vl == 0`, so the width is only ever multiplied by zero.
#[inline]
pub fn sew_bits_from_vtype(vtype: u64) -> u16 {
    8u16 << ((vtype >> 3) & 0x7)
}

/// How a vector memory instruction forms its addresses.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VecMemMode {
    /// `vle*`/`vse*` and their segment forms: one contiguous run from the base.
    UnitStride,
    /// `vle*ff`: unit-stride, but `vl` may be trimmed by a fault on any element
    /// after the first. Addressing is identical; the trimmed `vl` is captured
    /// like a `vsetvl*`'s (see [`VecMemOp::writes_vl`]), so
    /// it is the `vl` [`VecMemOp::segments`] is given for the ff itself.
    FaultOnlyFirst,
    /// `vl<n>r`/`vs<n>r`: `nf` whole registers, `VLEN/8` bytes each, contiguous
    /// from the base, independent of `vl` and `vtype`.
    WholeRegister,
    /// `vlm`/`vsm`: `ceil(vl/8)` contiguous bytes from the base.
    Mask,
    /// `vlse*`/`vsse*`: `vl` segments, `x[rs2]` bytes apart.
    Strided,
    /// `vluxei*`/`vloxei*`/`vsuxei*`/`vsoxei*`: one segment per element, at
    /// `base + v[vs2][i]`.
    Indexed,
}

/// A decoded vector load or store.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VecMemOp {
    /// `STORE-FP` rather than `LOAD-FP`.
    pub is_store: bool,
    pub mode: VecMemMode,
    /// Base address register.
    pub rs1: u8,
    /// Stride register, for [`VecMemMode::Strided`] only.
    pub rs2: u8,
    /// Index vector register, for [`VecMemMode::Indexed`] only.
    pub vs2: u8,
    /// `NFIELDS`, i.e. the encoded `nf` plus one: fields per segment, or whole
    /// registers transferred. 1 for a non-segment op.
    pub nf: u8,
    /// The `width` field's element width in **bits**. For non-indexed ops this
    /// is the data EEW; for indexed ops it is the *index* EEW and the data
    /// width is `SEW` (V spec 7.6 — the mismatch
    /// `test_bins/asm_validation/memory/vindexed_eew_mismatch.S` exists for).
    pub width_bits: u16,
}

impl VecMemOp {
    /// Decode a 32-bit instruction word, or `None` if it is not a vector
    /// load/store.
    pub fn decode(insn: u32) -> Option<Self> {
        let is_store = match insn & 0x7f {
            0b000_0111 => false,
            0b010_0111 => true,
            _ => return None,
        };

        let width = (insn >> 12) & 0x7;
        let width_bits: u16 = match width {
            0b000 => 8,
            0b101 => 16,
            0b110 => 32,
            0b111 => 64,
            // 010/011/100 are the scalar flw/fld/flq family; 001 is reserved.
            _ => return None,
        };

        let mew = (insn >> 28) & 1;
        assert!(
            mew == 0,
            "instruction {insn:#010x} sets the reserved `mew` bit (128-bit vector \
             elements). Its element width is not what this decoder reports, so its \
             memory footprint would be wrong."
        );

        let nf = ((insn >> 29) & 0x7) as u8 + 1;
        let mop = (insn >> 26) & 0x3;
        let field20 = ((insn >> 20) & 0x1f) as u8;
        let rs1 = ((insn >> 15) & 0x1f) as u8;

        let mode = match mop {
            // Unit-stride: the 24:20 field is `lumop`/`sumop`, selecting the
            // plain, whole-register, mask and fault-only-first variants.
            0b00 => match field20 {
                0b00000 => VecMemMode::UnitStride,
                0b01000 => VecMemMode::WholeRegister,
                0b01011 => VecMemMode::Mask,
                0b10000 if !is_store => VecMemMode::FaultOnlyFirst,
                _ => return None,
            },
            0b10 => VecMemMode::Strided,
            // 01 unordered, 11 ordered: same addressing, and ordering does not
            // change which lines are touched.
            _ => VecMemMode::Indexed,
        };

        Some(Self {
            is_store,
            mode,
            rs1,
            rs2: field20,
            vs2: field20,
            nf,
            width_bits,
        })
    }

    /// Whether this op writes the `vl` CSR, i.e. is a fault-only-first load.
    ///
    /// The `vl*ff` family is the only thing besides a `vsetvl*` that changes
    /// `vl`: a fault on any element after the first trims `vl` to that element's
    /// index (V spec 8.2 "Vector Unit-Stride Fault-Only-First Loads"; QEMU's
    /// `vext_ldff` does exactly `env->vl = i`). So both sides of the trace have
    /// to treat one of these as a `vl` producer -- the tracer emits a `VL`
    /// record for it, the consumer consumes one at it -- and they agree by both
    /// asking this.
    ///
    /// True for the segment forms (`vlseg<nf>e<eew>ff.v`) too: they share the
    /// `lumop` that selects fault-only-first, and they trim `vl` the same way.
    pub const fn writes_vl(&self) -> bool {
        matches!(self.mode, VecMemMode::FaultOnlyFirst)
    }

    /// Which capture this op needs at runtime, i.e. the group its callback
    /// emits.
    pub fn group_kind(&self) -> crate::bbt::MemGroupKind {
        use crate::bbt::MemGroupKind;
        match self.mode {
            VecMemMode::Strided => MemGroupKind::BaseStride,
            VecMemMode::Indexed => MemGroupKind::Lines,
            _ => MemGroupKind::Base,
        }
    }

    /// Bytes transferred per element step: `nf` fields of the *data* element
    /// width. `sew_bits` is only consulted for indexed ops, whose `width` field
    /// describes their indices instead.
    pub fn segment_bytes(&self, sew_bits: u16) -> u64 {
        let data_bits = match self.mode {
            VecMemMode::Indexed => sew_bits,
            _ => self.width_bits,
        };
        u64::from(self.nf) * u64::from(data_bits) / 8
    }

    /// The `(address, length)` byte runs this execution touched, for every mode
    /// whose footprint is derivable offline.
    ///
    /// Returns `None` for [`VecMemMode::Indexed`], whose footprint is carried
    /// explicitly as a line set instead — there is no analytic form, which is
    /// the whole reason that group kind exists.
    ///
    /// `vl` is the architectural vector length in force, `sew_bits` the `SEW`,
    /// `vlenb` the `VLEN` in bytes. For [`VecMemMode::FaultOnlyFirst`] the `vl`
    /// to pass is the *trimmed* one the instruction left behind, not the one
    /// requested: the elements at and past the fault are never accessed, so
    /// they are not part of the footprint. The tracer captures that value (see
    /// [`VecMemOp::writes_vl`]) and the consumer applies it to the ff itself.
    pub fn segments(
        &self,
        base: u64,
        stride: i64,
        vl: u64,
        sew_bits: u16,
        vlenb: u64,
    ) -> Option<Segments> {
        let seg = self.segment_bytes(sew_bits);
        let (count, step, len) = match self.mode {
            VecMemMode::UnitStride | VecMemMode::FaultOnlyFirst => (1, 0, vl * seg),
            VecMemMode::WholeRegister => (1, 0, u64::from(self.nf) * vlenb),
            VecMemMode::Mask => (1, 0, vl.div_ceil(8)),
            VecMemMode::Strided => (vl, stride, seg),
            VecMemMode::Indexed => return None,
        };
        Some(Segments {
            base,
            step,
            len,
            count,
            next: 0,
        })
    }
}

/// The shape of a [`Segments`] run list, without iterating it.
///
/// Exists so a consumer can pass a strided footprint on *as* a stride —
/// `vl` runs described by four numbers — instead of unrolling it into `vl`
/// separate ranges. A cache model that takes a strided access directly is then
/// called once per instruction rather than once per element.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SegmentShape {
    /// One contiguous run.
    Contiguous { addr: u64, len: u64 },
    /// `count` runs of `len` bytes, `stride` apart.
    Strided {
        base: u64,
        stride: i64,
        count: u64,
        len: u64,
    },
}

/// The byte runs of one vector memory execution, as produced by
/// [`VecMemOp::segments`]. An iterator rather than a `Vec` because a strided op
/// at a large `vl` has one run per element and none of them need to be stored.
#[derive(Debug, Clone, Copy)]
pub struct Segments {
    base: u64,
    step: i64,
    len: u64,
    count: u64,
    next: u64,
}

impl Segments {
    /// Describe the run list without walking it.
    ///
    /// A single run reports as [`SegmentShape::Contiguous`] whatever its step,
    /// because a stride is meaningless with nothing to stride between — and a
    /// consumer that special-cases the contiguous form should see it here
    /// rather than have to check `count == 1` itself.
    pub fn shape(&self) -> SegmentShape {
        if self.count == 1 {
            SegmentShape::Contiguous {
                addr: self.base,
                len: self.len,
            }
        } else {
            SegmentShape::Strided {
                base: self.base,
                stride: self.step,
                count: self.count,
                len: self.len,
            }
        }
    }
}

impl Iterator for Segments {
    /// `(address, length in bytes)`.
    type Item = (u64, u64);

    fn next(&mut self) -> Option<Self::Item> {
        if self.next >= self.count {
            return None;
        }
        let addr = self
            .base
            .wrapping_add((self.next as i64).wrapping_mul(self.step) as u64);
        self.next += 1;
        Some((addr, self.len))
    }
}

/// Expand an indexed (gather/scatter) execution into the **sorted, deduplicated**
/// set of cache lines it touched, written into `out` (cleared first).
///
/// This is the one expansion that has to happen inside the tracer, because its
/// input — the index vector's bytes — exists only while the guest is running.
///
/// * `index_bytes` is the raw little-endian content of `v[vs2]` and, when the
///   index EMUL spans a register group, the registers following it.
/// * `index_bits` is the *index* element width (the instruction's `width`
///   field), which need not equal `SEW`.
/// * `segment_bytes` is what one element step moves — `nf * SEW/8`, from
///   [`VecMemOp::segment_bytes`].
///
/// Indices are unsigned and added to the base modulo the address width, per V
/// spec 7.6. Elements masked off by `vm = 0` are *not* excluded: reading `v0`
/// would double the register reads on every masked op, and the result is a
/// superset of the true footprint — an over-count in a cache model, never a
/// missing line.
///
/// Returns `false` if `index_bytes` is too short for `vl` elements, which means
/// the register read came up short and the line set would be silently partial.
pub fn indexed_lines(
    base: u64,
    index_bytes: &[u8],
    index_bits: u16,
    segment_bytes: u64,
    vl: u64,
    out: &mut Vec<u64>,
) -> bool {
    out.clear();
    let stride = usize::from(index_bits) / 8;
    let needed = (vl as usize).saturating_mul(stride);
    if index_bytes.len() < needed {
        return false;
    }
    for i in 0..vl as usize {
        let mut idx = [0u8; 8];
        idx[..stride].copy_from_slice(&index_bytes[i * stride..(i + 1) * stride]);
        let addr = base.wrapping_add(u64::from_le_bytes(idx));
        out.extend(lines_of(addr, segment_bytes));
    }
    out.sort_unstable();
    out.dedup();
    true
}

/// The cache lines a `(address, length)` run covers.
pub fn lines_of(addr: u64, len: u64) -> impl Iterator<Item = u64> {
    let first = line_of(addr);
    let n = if len == 0 {
        0
    } else {
        (line_of(addr + len - 1) - first) / CACHE_LINE_BYTES + 1
    };
    (0..n).map(move |i| first + i * CACHE_LINE_BYTES)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bbt::MemGroupKind;

    /// Assemble a vector load/store word from its fields, so the tests below
    /// state the encoding they mean rather than a hex literal nobody can check.
    fn enc(opcode: u32, nf: u32, mop: u32, field20: u32, rs1: u32, width: u32) -> u32 {
        (nf << 29) | (mop << 26) | (field20 << 20) | (rs1 << 15) | (width << 12) | opcode
    }
    const LOAD_FP: u32 = 0b000_0111;
    const STORE_FP: u32 = 0b010_0111;

    #[test]
    fn vsetvl_immediate_forms_carry_their_sew() {
        // vtype immediate with vsew = 3, i.e. `vsetvli rd, rs1, e64, ...` --
        // the configuration `vindexed_eew_mismatch.S` runs its loop under.
        let vsetvli = (0b011 << 3 << 20) | (5 << 15) | (0b111 << 12) | (6 << 7) | 0x57;
        assert_eq!(vsetvl_static_sew_bits(vsetvli), Some(64));
        // vsetivli sets bits 31:30, and its vtype is one bit narrower.
        let vsetivli = (0b11 << 30) | (0b010 << 3 << 20) | (4 << 15) | (0b111 << 12) | 0x57;
        assert_eq!(vsetvl_static_sew_bits(vsetivli), Some(32));
        // vsetvl takes its vtype from a register, so nothing is knowable here.
        let vsetvl = (0b1000000 << 25) | (0b111 << 12) | 0x57;
        assert_eq!(vsetvl_static_sew_bits(vsetvl), None);
        // Not a vsetvl* at all.
        assert_eq!(vsetvl_static_sew_bits(0x0000_0013), None);
    }

    #[test]
    fn scalar_fp_loads_are_not_vector_ops() {
        // flw/fld/flq share the major opcode and must not be picked up.
        for width in [0b010, 0b011, 0b100] {
            assert_eq!(VecMemOp::decode(enc(LOAD_FP, 0, 0, 0, 5, width)), None);
        }
        // Not a load/store opcode at all.
        assert_eq!(VecMemOp::decode(0x0000_0013), None); // addi
    }

    #[test]
    fn unit_stride_forms_decode_by_lumop() {
        let of = |field20, width| VecMemOp::decode(enc(LOAD_FP, 0, 0b00, field20, 5, width));
        assert_eq!(of(0b00000, 0b111).unwrap().mode, VecMemMode::UnitStride);
        assert_eq!(of(0b01000, 0b000).unwrap().mode, VecMemMode::WholeRegister);
        assert_eq!(of(0b01011, 0b000).unwrap().mode, VecMemMode::Mask);
        assert_eq!(of(0b10000, 0b110).unwrap().mode, VecMemMode::FaultOnlyFirst);
        // A store has no fault-only-first form.
        assert_eq!(
            VecMemOp::decode(enc(STORE_FP, 0, 0b00, 0b10000, 5, 0b110)),
            None
        );
    }

    /// `writes_vl` is what both sides of the trace ask to decide whether a
    /// `VL` record belongs to this instruction, so it must be true for exactly
    /// the fault-only-first forms -- plain and segment -- and nothing else.
    #[test]
    fn only_fault_only_first_writes_vl() {
        // Every EEW of the plain form, and every NFIELDS of the segment form.
        for width in [0b000, 0b101, 0b110, 0b111] {
            for nf in 0..8 {
                let op = VecMemOp::decode(enc(LOAD_FP, nf, 0b00, 0b10000, 5, width)).unwrap();
                assert_eq!(op.mode, VecMemMode::FaultOnlyFirst);
                assert!(
                    op.writes_vl(),
                    "vlseg{}e{}ff must write vl",
                    nf + 1,
                    op.width_bits
                );
            }
        }
        // Nothing else in the family does. `field20` is the stride/index
        // register for the non-unit-stride modes, so 0b10000 is a legal value
        // there and must not be mistaken for `lumop = fault-only-first`.
        let others = [
            enc(LOAD_FP, 0, 0b00, 0b00000, 5, 0b111),  // vle64.v
            enc(LOAD_FP, 0, 0b00, 0b01000, 5, 0b000),  // vl1r.v
            enc(LOAD_FP, 0, 0b00, 0b01011, 5, 0b000),  // vlm.v
            enc(STORE_FP, 0, 0b00, 0b00000, 5, 0b111), // vse64.v
            enc(LOAD_FP, 0, 0b10, 0b10000, 5, 0b111),  // vlse64.v v?, (x5), x16
            enc(LOAD_FP, 0, 0b01, 0b10000, 5, 0b111),  // vluxei64.v v?, (x5), v16
            enc(LOAD_FP, 0, 0b11, 0b10000, 5, 0b111),  // vloxei64.v v?, (x5), v16
            enc(STORE_FP, 0, 0b10, 0b10000, 5, 0b111), // vsse64.v v?, (x5), x16
        ];
        for insn in others {
            let op = VecMemOp::decode(insn).unwrap();
            assert!(
                !op.writes_vl(),
                "{insn:#010x} ({:?}) must not write vl",
                op.mode
            );
        }
    }

    #[test]
    fn width_field_is_the_element_width() {
        for (width, bits) in [(0b000, 8), (0b101, 16), (0b110, 32), (0b111, 64)] {
            let op = VecMemOp::decode(enc(LOAD_FP, 0, 0b00, 0, 5, width)).unwrap();
            assert_eq!(op.width_bits, bits);
        }
    }

    #[test]
    fn nf_is_the_encoded_field_plus_one() {
        // vlseg8e8.v: nf field 7 means eight fields.
        let op = VecMemOp::decode(enc(LOAD_FP, 7, 0b00, 0, 5, 0b000)).unwrap();
        assert_eq!(op.nf, 8);
        // A non-segment op transfers one field per element step.
        let op = VecMemOp::decode(enc(LOAD_FP, 0, 0b00, 0, 5, 0b000)).unwrap();
        assert_eq!(op.nf, 1);
    }

    #[test]
    fn addressing_mode_picks_the_capture() {
        let unit = VecMemOp::decode(enc(LOAD_FP, 0, 0b00, 0, 5, 0b111)).unwrap();
        let strided = VecMemOp::decode(enc(LOAD_FP, 0, 0b10, 9, 5, 0b111)).unwrap();
        let unordered = VecMemOp::decode(enc(LOAD_FP, 0, 0b01, 8, 5, 0b110)).unwrap();
        let ordered = VecMemOp::decode(enc(STORE_FP, 0, 0b11, 8, 5, 0b110)).unwrap();

        assert_eq!(unit.group_kind(), MemGroupKind::Base);
        assert_eq!(strided.group_kind(), MemGroupKind::BaseStride);
        assert_eq!(strided.rs2, 9);
        // Both index orderings touch the same lines, so both capture a line set.
        assert_eq!(unordered.mode, VecMemMode::Indexed);
        assert_eq!(ordered.mode, VecMemMode::Indexed);
        assert_eq!(unordered.group_kind(), MemGroupKind::Lines);
        assert_eq!(unordered.vs2, 8);
        assert!(ordered.is_store);
    }

    #[test]
    fn indexed_data_width_is_sew_not_the_width_field() {
        // vluxseg2ei16.v at SEW=64: indices are 16-bit, data elements 64-bit, so
        // each element step moves nf * 8 = 16 bytes. This is exactly the case
        // `vindexed_eew_mismatch.S` covers.
        let op = VecMemOp::decode(enc(LOAD_FP, 1, 0b01, 8, 5, 0b101)).unwrap();
        assert_eq!(op.width_bits, 16);
        assert_eq!(op.segment_bytes(64), 16);
        // A non-indexed op's data width *is* its width field, whatever SEW says.
        let unit = VecMemOp::decode(enc(LOAD_FP, 1, 0b00, 0, 5, 0b101)).unwrap();
        assert_eq!(unit.segment_bytes(64), 4);
    }

    #[test]
    fn unit_stride_footprint_is_one_contiguous_run() {
        // vlseg2e64.v at vl=64: 2 * 8 * 64 = 1024 contiguous bytes.
        let op = VecMemOp::decode(enc(LOAD_FP, 1, 0b00, 0, 5, 0b111)).unwrap();
        let segs: Vec<_> = op.segments(0x1000, 0, 64, 64, 16).unwrap().collect();
        assert_eq!(segs, vec![(0x1000, 1024)]);
    }

    #[test]
    fn strided_footprint_is_one_run_per_element_and_strides_backwards() {
        // vlsseg2e64.v, vl=3, stride 64: three 16-byte runs, 64 bytes apart.
        let op = VecMemOp::decode(enc(LOAD_FP, 1, 0b10, 9, 5, 0b111)).unwrap();
        let segs: Vec<_> = op.segments(0x1000, 64, 3, 64, 16).unwrap().collect();
        assert_eq!(segs, vec![(0x1000, 16), (0x1040, 16), (0x1080, 16)]);

        // A negative stride is a legal, and used, way to walk an array backwards.
        let segs: Vec<_> = op.segments(0x1000, -64, 3, 64, 16).unwrap().collect();
        assert_eq!(segs, vec![(0x1000, 16), (0x0fc0, 16), (0x0f80, 16)]);
    }

    #[test]
    fn whole_register_and_mask_footprints_ignore_vl() {
        // vl4r.v moves 4 * VLENB bytes whatever vl and vtype say.
        let whole = VecMemOp::decode(enc(LOAD_FP, 3, 0b00, 0b01000, 5, 0b000)).unwrap();
        let segs: Vec<_> = whole.segments(0x2000, 0, 1, 8, 16).unwrap().collect();
        assert_eq!(segs, vec![(0x2000, 64)]);

        // vlm.v moves ceil(vl/8) bytes -- one bit per element, rounded up.
        let mask = VecMemOp::decode(enc(LOAD_FP, 0, 0b00, 0b01011, 5, 0b000)).unwrap();
        let segs: Vec<_> = mask.segments(0x2000, 0, 17, 32, 16).unwrap().collect();
        assert_eq!(segs, vec![(0x2000, 3)]);
    }

    #[test]
    fn indexed_has_no_derivable_footprint() {
        let op = VecMemOp::decode(enc(LOAD_FP, 0, 0b01, 8, 5, 0b110)).unwrap();
        assert!(op.segments(0x1000, 0, 64, 64, 16).is_none());
    }

    #[test]
    fn indexed_expansion_dedups_by_line() {
        // Four 32-bit indices into the same 64-byte line, 8-byte elements: one
        // line, reported once, however many elements land in it.
        let indices: Vec<u8> = [0u32, 8, 16, 24]
            .iter()
            .flat_map(|i| i.to_le_bytes())
            .collect();
        let mut out = Vec::new();
        assert!(indexed_lines(0x1000, &indices, 32, 8, 4, &mut out));
        assert_eq!(out, vec![0x1000]);

        // Spread across three lines, out of order, one of them revisited.
        let indices: Vec<u8> = [128u32, 0, 64, 8]
            .iter()
            .flat_map(|i| i.to_le_bytes())
            .collect();
        assert!(indexed_lines(0x1000, &indices, 32, 8, 4, &mut out));
        assert_eq!(out, vec![0x1000, 0x1040, 0x1080]);
    }

    #[test]
    fn indexed_expansion_covers_a_segment_that_straddles_a_line() {
        // vluxseg2ei16.v at SEW=64: each index reaches nf * 8 = 16 bytes, and an
        // index landing 56 bytes into a line spills into the next one.
        let indices: Vec<u8> = [56u16].iter().flat_map(|i| i.to_le_bytes()).collect();
        let mut out = Vec::new();
        assert!(indexed_lines(0x1000, &indices, 16, 16, 1, &mut out));
        assert_eq!(out, vec![0x1000, 0x1040]);
    }

    #[test]
    fn indexed_expansion_reads_indices_at_their_own_width() {
        // The same bytes read as e16 and as e32 are different index vectors; the
        // width field, not SEW, decides. All-zero indices are what
        // `vindexed_eew_mismatch.S` uses precisely because they survive being
        // reinterpreted at any width.
        let bytes = [0x40u8, 0x00, 0x80, 0x00];
        let mut out = Vec::new();
        // As e16: two indices, 0x0040 and 0x0080.
        assert!(indexed_lines(0x1000, &bytes, 16, 8, 2, &mut out));
        assert_eq!(out, vec![0x1040, 0x1080]);
        // As e32: one index, 0x0080_0040 -- a different address entirely.
        assert!(indexed_lines(0x1000, &bytes, 32, 8, 1, &mut out));
        assert_eq!(out, vec![0x1000 + 0x0080_0040]);
    }

    #[test]
    fn a_short_index_read_is_a_failure_not_a_short_line_set() {
        // Two elements' worth of bytes, four elements asked for: reporting the
        // two lines it could see would silently understate the footprint.
        let mut out = Vec::new();
        assert!(!indexed_lines(0x1000, &[0; 8], 32, 8, 4, &mut out));
    }

    #[test]
    fn a_run_covers_every_line_it_straddles() {
        assert_eq!(lines_of(0x1000, 64).collect::<Vec<_>>(), vec![0x1000]);
        // Unaligned by one byte, so the run spills into the next line.
        assert_eq!(
            lines_of(0x1001, 64).collect::<Vec<_>>(),
            vec![0x1000, 0x1040]
        );
        assert_eq!(
            lines_of(0x1000, 129).collect::<Vec<_>>(),
            vec![0x1000, 0x1040, 0x1080]
        );
        // A zero-length run touches nothing -- `vlm.v` at vl=0, say.
        assert_eq!(lines_of(0x1000, 0).count(), 0);
    }
}
