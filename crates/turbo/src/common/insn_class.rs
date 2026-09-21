//! Per-encoding instruction classification and per-execution deltas.
//!
//! [`InsnClass`] is the VL-independent, once-per-PC classification of an
//! instruction encoding; [`InsnDelta`] resolves one against the live
//! `vtype` state to give the counter contribution of a single execution.
//! The free functions here are the `riscv_isa` predicates both are built on.

use super::vector::Lmul;

/// Memory/compute elements touched per unit of VL: `nf` for a segment
/// load/store, `1` for everything else. Mirror of [`InsnClass::elem_mult`] for
/// callers that have an `Instruction` but no cached class.
pub fn elem_mult(ins: &riscv_isa::Instruction) -> u32 {
    segment_nf_eew(ins).map_or(1, |(nf, _eew)| nf)
}

/// The `(NFIELDS, EEW)` of a whole-register load/store, or `None` otherwise.
///
/// Per the vector spec section 7.9, `nf` on these encodes *how many vector
/// registers to transfer* (1, 2, 4 or 8), and they run at an effective vector
/// length `evl = NFIELDS * VLEN / EEW` "regardless of current settings in
/// vtype and vl". The whole-register stores are encoded as EEW=8.
///
/// EEW is only a rearrangement hint here -- every variant moves the same
/// `NFIELDS * VLEN/8` bytes -- but it does set how many *elements* that is.
pub fn whole_reg_nreg_eew(ins: &riscv_isa::Instruction) -> Option<(u32, u32)> {
    use riscv_isa::Instruction as I;
    Some(match ins {
        I::VL1RE8_V { .. } => (1, 8),
        I::VL1RE16_V { .. } => (1, 16),
        I::VL1RE32_V { .. } => (1, 32),
        I::VL1RE64_V { .. } => (1, 64),
        I::VL2RE8_V { .. } => (2, 8),
        I::VL2RE16_V { .. } => (2, 16),
        I::VL2RE32_V { .. } => (2, 32),
        I::VL2RE64_V { .. } => (2, 64),
        I::VL4RE8_V { .. } => (4, 8),
        I::VL4RE16_V { .. } => (4, 16),
        I::VL4RE32_V { .. } => (4, 32),
        I::VL4RE64_V { .. } => (4, 64),
        I::VL8RE8_V { .. } => (8, 8),
        I::VL8RE16_V { .. } => (8, 16),
        I::VL8RE32_V { .. } => (8, 32),
        I::VL8RE64_V { .. } => (8, 64),
        I::VS1R_V { .. } => (1, 8),
        I::VS2R_V { .. } => (2, 8),
        I::VS4R_V { .. } => (4, 8),
        I::VS8R_V { .. } => (8, 8),
        _ => return None,
    })
}

// Elements moved/processed by one execution of `ins` at vector length `vl`.
//
// Four cases, all from the vector spec:
//
// * whole-register load/store: `evl = NFIELDS * VLEN / EEW`, which ignores
//   `vl` entirely (section 7.9);
// * mask load/store: `evl = ceil(vl/8)` with EEW=8 -- they "operate as byte
//   loads and stores" (section 7.4);
// * segment load/store: `vl` counts *segments* of NFIELDS fields each, so
//   `NFIELDS * vl` elements (section 7.8);
// * everything else: `vl`.
//
// `InsnClass` caches the per-instruction parts of this.

/// Whether `ins` is a mask load/store (`vlm.v` / `vsm.v`).
///
/// Section 7.4: these are encoded like `vle8.v`/`vse8.v` but run at
/// `evl = ceil(vl/8)`, moving that many bytes. Counting them at `vl` elements
/// would put the element count 8x above the byte count.
pub fn is_mask_mem(ins: &riscv_isa::Instruction) -> bool {
    use riscv_isa::Instruction as I;
    matches!(ins, I::VLM_V { .. } | I::VSM_V { .. })
}

/// Whether `ins` is an indexed (gather/scatter) vector load or store.
///
/// These are the ops whose numeric suffix is the **index** element width, not
/// the data width: `vluxei32.v` under `SEW=64` reads 32-bit offsets to move
/// 64-bit data. `Instruction::bytes_moved` has no access to `vtype` and so
/// approximates the data width with the index width, which is only correct
/// when the two happen to match (documented on `bytes_moved` itself).
///
/// We *do* have the live SEW here, so `update_stats_cached` computes the byte
/// count directly for these instead of taking that approximation. The segment
/// forms are included: `nf` comes from [`elem_mult`], which is orthogonal to
/// the width confusion.
pub fn is_indexed_mem(ins: &riscv_isa::Instruction) -> bool {
    use riscv_isa::Instruction as I;
    matches!(
        ins,
        I::VLUXEI8_V { .. }
            | I::VLUXEI16_V { .. }
            | I::VLUXEI32_V { .. }
            | I::VLUXEI64_V { .. }
            | I::VLOXEI8_V { .. }
            | I::VLOXEI16_V { .. }
            | I::VLOXEI32_V { .. }
            | I::VLOXEI64_V { .. }
            | I::VSUXEI8_V { .. }
            | I::VSUXEI16_V { .. }
            | I::VSUXEI32_V { .. }
            | I::VSUXEI64_V { .. }
            | I::VSOXEI8_V { .. }
            | I::VSOXEI16_V { .. }
            | I::VSOXEI32_V { .. }
            | I::VSOXEI64_V { .. }
            | I::VLUXSEG_V { .. }
            | I::VLOXSEG_V { .. }
            | I::VSUXSEG_V { .. }
            | I::VSOXSEG_V { .. }
    )
}

/// The `(nf, eew)` of a segment load/store, or `None` for anything else.
///
/// Unlike `vle64.v`/`vle32.v`/... -- which are one enum variant each -- the
/// segment load/stores are nine variants that carry `nf` and `eew` as *fields*,
/// so a single variant spans a whole mnemonic family (`vlseg2e64.v` through
/// `vlseg8e8.v`). The element count needs those fields rather than just the
/// variant: a segment op transfers `nf` fields per element step, so it touches
/// `nf * vl` elements while VL stays `vl` ([`InsnClass::elem_mult`]).
///
/// Resolved once per unique PC in [`InsnClass::from_instruction`], so this match
/// never runs on the per-execution hot path.
pub fn segment_nf_eew(ins: &riscv_isa::Instruction) -> Option<(u32, u32)> {
    use riscv_isa::Instruction as I;
    match ins {
        I::VLSEG_V { nf, eew, .. }
        | I::VLSEGFF_V { nf, eew, .. }
        | I::VSSEG_V { nf, eew, .. }
        | I::VLSSEG_V { nf, eew, .. }
        | I::VSSSEG_V { nf, eew, .. }
        | I::VLUXSEG_V { nf, eew, .. }
        | I::VLOXSEG_V { nf, eew, .. }
        | I::VSUXSEG_V { nf, eew, .. }
        | I::VSOXSEG_V { nf, eew, .. } => Some((*nf, *eew)),
        _ => None,
    }
}

/// Precomputed, VL-independent classification of a single instruction encoding.
///
/// Every field here is a pure function of the instruction bytes (and the
/// run-fixed `SYSTEM_VLEN`), so it can be computed **once per unique PC** and
/// reused for every dynamic execution of that PC. This turns the per-instruction
/// hot path (`RoiInfo::update_stats_cached`) from ~9 enum-dispatch predicate
/// calls plus an atomic VLEN load into plain integer arithmetic on a cached
/// descriptor.
///
/// The classification is derived from the *same* `riscv_isa` predicates that
/// `update_stats` used directly, so results are identical by construction (see
/// `RoiInfo::update_stats`, which now builds an `InsnClass` and delegates).
#[derive(Clone, Copy, Debug)]
pub struct InsnClass {
    /// Bitset of `IC_*` classification flags.
    pub flags: u16,
    /// VL-independent byte contribution for load/store accounting:
    /// - whole-register vector ops: `vlen_bytes` (matches the historical
    ///   `is_whole_register` override in `update_stats`);
    /// - scalar loads/stores: the fixed transfer width;
    /// - everything else: 0 (unused, since only added on load/store).
    ///
    /// Not valid when `bytes_runtime` is set (VL-dependent vector mem ops).
    pub bytes_const: u32,
    /// When true, the byte count is VL-dependent (vector element/mask
    /// loads/stores that are not whole-register) and must be computed at
    /// runtime via `Instruction::bytes_moved(vlen, vl)`.
    pub bytes_runtime: bool,
    /// Memory/compute elements touched per unit of VL.
    ///
    /// `1` for every instruction except the segment load/stores. VL counts
    /// *segments* for those, and each segment is `nf` fields of `eew` bits, so
    /// the instruction touches `nf * vl` elements while VL itself stays `vl`.
    ///
    /// Multiply the element counters by this, but *not* the VL histograms (they
    /// report architectural VL) and not `bytes_moved` (it already accounts for
    /// `nf`). Doing so puts segment ops on the same footing as the element-wise
    /// vector memory ops, where `bytes == elements * eew/8`.
    ///
    /// Whole-register ops do not go through this multiplier at all -- they run
    /// at their own `evl` and use [`InsnClass::whole_reg_elems`] instead.
    pub elem_mult: u32,
    /// Element count for a whole-register load/store -- `evl = NFIELDS *
    /// VLEN / EEW`, which the spec defines independently of `vtype`/`vl`.
    /// `0` for every other instruction, where the count is `elem_mult * vl`.
    pub whole_reg_elems: u32,
    /// Mask load/store (`vlm.v`/`vsm.v`), which runs at `evl = ceil(vl/8)`
    /// rather than `vl`. See [`is_mask_mem`].
    pub mask_mem: bool,
}

// Classification flag bits for `InsnClass::flags`.
pub const IC_VECTOR: u16 = 1 << 0;
pub const IC_BRANCH: u16 = 1 << 1;
pub const IC_FLOAT: u16 = 1 << 2;
pub const IC_FMA: u16 = 1 << 3;
/// `int_arithmetic() && fma()` -- an integer fused-multiply-accumulate.
pub const IC_INT_FMA: u16 = 1 << 4;
pub const IC_LOAD: u16 = 1 << 5;
pub const IC_STORE: u16 = 1 << 6;
pub const IC_WHOLE_REG: u16 = 1 << 7;
/// Indexed (gather/scatter) vector load/store -- see [`is_indexed_mem`]. Its
/// byte count is computed from the live SEW rather than from `bytes_moved`,
/// whose numeric suffix is the index width.
pub const IC_INDEXED_MEM: u16 = 1 << 8;

impl InsnClass {
    /// Classify an instruction once, reusing the exact `riscv_isa` predicates
    /// that `update_stats` previously called per execution. `vlen_bytes` is the
    /// run-fixed `SYSTEM_VLEN` (folded into `bytes_const` for whole-register and
    /// scalar memory ops so the hot path never touches the atomic).
    pub fn from_instruction(ins: &riscv_isa::Instruction, vlen_bytes: u16) -> Self {
        let is_vector = ins.vector();
        let is_whole = ins.is_whole_register_instruction();
        let is_load = ins.load();
        let is_store = ins.store();
        let is_mem = is_load || is_store;

        let mut flags = 0u16;
        if is_vector {
            flags |= IC_VECTOR;
        }
        // `conditional_branch()`, NOT `branch()`: the latter is riscv_isa's
        // "could branch or jump" and admits JAL/JALR, which would fold every
        // call and return into the ROI branch count. For the wider notion see
        // `pc_profile::is_control_flow`, which drives the annotate listing --
        // the two deliberately disagree about `jal`.
        if ins.conditional_branch() {
            flags |= IC_BRANCH;
        }
        if ins.float() {
            flags |= IC_FLOAT;
        }
        if ins.fma() {
            flags |= IC_FMA;
        }
        if ins.int_arithmetic() && ins.fma() {
            flags |= IC_INT_FMA;
        }
        if is_load {
            flags |= IC_LOAD;
        }
        if is_store {
            flags |= IC_STORE;
        }
        if is_whole {
            flags |= IC_WHOLE_REG;
        }
        if is_indexed_mem(ins) {
            flags |= IC_INDEXED_MEM;
        }

        // VL-dependent only for non-whole-register vector memory ops
        // (element/mask loads/stores). Everything else is a constant.
        let bytes_runtime = is_mem && is_vector && !is_whole;
        let bytes_const = if is_mem && !bytes_runtime {
            // Scalar and whole-register load/store: VL-independent, so vl=1
            // yields the exact constant `bytes_moved` would return for any vl.
            // For whole-register ops that is `NFIELDS * VLEN/8` -- the crate
            // already honours `nf`, so nothing here may override it.
            ins.bytes_moved(vlen_bytes, 1)
        } else {
            0
        };

        // Segment load/stores transfer `nf` fields per element step; every
        // other opcode moves one element per step.
        let elem_mult = elem_mult(ins);

        let whole_reg_elems =
            whole_reg_nreg_eew(ins).map_or(0, |(nreg, eew)| nreg * (vlen_bytes as u32 * 8) / eew);

        InsnClass {
            flags,
            bytes_const,
            bytes_runtime,
            elem_mult,
            whole_reg_elems,
            mask_mem: is_mask_mem(ins),
        }
    }

    /// Memory/compute elements touched by one execution at architectural `vl`.
    ///
    /// The cached equivalent of [`vector_elements_for`] -- same three cases, but
    /// read off the precomputed descriptor instead of re-running
    /// `whole_reg_nreg_eew`/`is_mask_mem`/`elem_mult` over the 690-variant
    /// `Instruction` enum.
    #[inline(always)]
    pub fn vector_elements(self, vl: u64) -> u64 {
        if self.whole_reg_elems != 0 {
            // Whole-register ops run at their own evl, not `vl`.
            self.whole_reg_elems as u64
        } else if self.mask_mem {
            // Mask ops run at evl = ceil(vl/8).
            vl.div_ceil(8)
        } else {
            vl * self.elem_mult as u64
        }
    }

    #[inline(always)]
    fn has(self, bit: u16) -> bool {
        self.flags & bit != 0
    }
}
/// The counter contribution of one *executed* instruction, derived once from
/// its cached [`InsnClass`] plus the live `vtype` state and then applied to
/// every scope it belongs to (see [`RoiInfo::apply_delta`]).
///
/// `InsnClass` is per-PC and VL-independent; this is the per-execution
/// resolution of it against the current VL/SEW/LMUL. It exists purely so that
/// the derivation is not repeated once per scope: `StatsAccumulator` updates
/// the function ROI, the global ROI, and every active ROI region from the
/// same instruction, so a 1-instruction/2-scope trace previously ran the whole
/// derivation twice.
#[derive(Clone, Copy, Debug)]
pub struct InsnDelta {
    /// Copy of [`InsnClass::flags`], so applying a delta needs only one struct.
    pub flags: u16,
    /// Architectural vector length for this execution (1 for scalar ops).
    /// Drives the VL histograms and the float/FMA element counters.
    pub vl: u64,
    /// Memory/compute elements touched -- [`InsnClass::vector_elements`] for
    /// vector ops, 1 for scalar.
    pub elems: u64,
    /// Bytes loaded or stored by this execution.
    pub bytes_moved: u64,
    /// Live SEW encoding (0..=3 for e8/e16/e32/e64), for histogram routing.
    pub sew: Option<u8>,
    /// Live LMUL in `vtype` encoding, for histogram routing.
    pub lmul: Option<u8>,
}

impl InsnDelta {
    /// Resolve `class` against the live `vtype` state. Only the rare VL-dependent
    /// vector-memory case (`IC_BYTES_RUNTIME`) touches `ins` at all.
    #[inline]
    pub fn new(
        class: &InsnClass,
        ins: &riscv_isa::Instruction,
        current_vl: Option<u32>,
        current_sew: Option<u8>,
        current_lmul: Option<Lmul>,
        vlen_bytes: u16,
    ) -> Self {
        let (vl, elems) = if class.has(IC_VECTOR) {
            let vl = current_vl.unwrap_or(1) as u64;
            (vl, class.vector_elements(vl))
        } else {
            (1, 1)
        };

        // Indexed loads/stores move `elem_mult * SEW/8 * vl` bytes. Their
        // mnemonic suffix is the *index* width, so `bytes_moved` -- which
        // cannot see `vtype` -- would substitute that for the data width and
        // be wrong whenever the two differ (e.g. `vluxei32.v` at SEW=64
        // undercounts 2x). We have the live SEW, so compute it directly.
        // `sew` is the vtype encoding: 0..=3 for e8/e16/e32/e64.
        let indexed_bytes = if class.has(IC_INDEXED_MEM) {
            current_sew
                .filter(|sew| *sew <= 3)
                .map(|sew| class.elem_mult as u64 * (1u64 << sew) * vl)
        } else {
            None
        };

        // Register operations aren't affected by VL or SEW.
        // `bytes_const` already folds in the whole-register override and the
        // scalar constant; only VL-dependent vector mem ops need a runtime call.
        let bytes_moved = match indexed_bytes {
            Some(bytes) => bytes,
            // Falls through for an unknown/oversized SEW, keeping the old
            // approximation rather than reporting nothing.
            None if class.bytes_runtime => ins.bytes_moved(vlen_bytes, vl as u32) as u64,
            None => class.bytes_const as u64,
        };

        InsnDelta {
            flags: class.flags,
            vl,
            elems,
            bytes_moved,
            sew: current_sew,
            lmul: current_lmul.map(|l| l.to_vlmul()),
        }
    }
}
