//! Computing a `vsetvl*`'s resulting `vl` from the instruction itself, instead
//! of reading the `vl` CSR back after it has executed.
//!
//! # Why
//!
//! Reading `vl` needs `QEMU_PLUGIN_CB_R_REGS` on the callback that does the
//! read, and `vl` only exists *after* the `vsetvl*` — which always ends its TB
//! ([`crate`]'s translate-time assertion pins that) — so the read had to happen
//! in the *next block's* start callback. That put `R_REGS` on **every** block
//! entry: 116.6 M of them on a measured workload, at ~35–38 ns each of TCG global sync, for
//! 3.56 M `vsetvl*` executions that actually needed it. It was ~4.3 s of a 6.65 s
//! tracing overhead, and it is what this module exists to remove.
//!
//! QEMU instruction callbacks fire *before* the instruction, when `rs1`/`rs2` are
//! still intact as sources. So the plan is: look at the `vsetvl*` at translate
//! time, work out where its AVL comes from, and register a callback that reads
//! only what it must and does the ISA's own arithmetic. Sites whose `vl` is a
//! translate-time constant read nothing at all.
//!
//! # Fidelity
//!
//! [`compute_vl`] and [`vlmax_from_vtype`] are a transcription of QEMU's
//! `HELPER(vsetvl)` (`target/riscv/vector_helper.c`) and `vext_get_vlmax`
//! (`target/riscv/cpu.h`), including the `vill` cases. Two things it cannot see:
//!
//! * the `rvv_vl_half_avl` / `rvv_vsetvl_x0_vill` CPU properties, which the
//!   plugin API cannot query — rejected at init by `audit_cpu_properties`;
//! * `ELEN`, needed for the `sew > elen` `vill` case, also unreadable. Assumed
//!   `>= 64` (true for `-cpu max` and every RVV config this tool is used with),
//!   so only the architecturally-reserved `vsew > 3` is treated as `vill`.
//!
//! Both gaps are also covered at runtime by the computed-vs-read oracle in
//! `lib.rs`, which is why they are acceptable rather than merely documented.

/// `RV_VLEN_MAX` (`target/riscv/cpu.h`), the AVL QEMU substitutes for the
/// `rs1 == x0`, `rd != x0` form of `vsetvli`/`vsetvl` ("set `vl` to `VLMAX`").
/// It is deliberately larger than any representable `VLMAX`, so the
/// `s1 <= vlmax` test in `HELPER(vsetvl)` always falls through to `vl = vlmax` —
/// but it is transcribed as the literal value rather than short-circuited to
/// `vlmax`, so that a future QEMU raising `VLEN` past it behaves the same here as
/// there.
pub const RV_VLEN_MAX: u64 = 65536;

/// Guest XLEN. This plugin is built against `qemu-riscv64`, and XLEN only enters
/// the `vill` decision via which bits of `vtype` are "reserved" and which is
/// `vill` itself — and for `vsetvli`/`vsetivli` the whole `vtype` comes from a
/// 10- or 11-bit immediate, so those bits are unreachable regardless.
const XLEN: u32 = 64;

/// `vext_get_vlmax` (`target/riscv/cpu.h`): the number of elements a `vtype`
/// permits. `lmul` is the *sign-extended* 3-bit `VLMUL` field, so fractional
/// LMUL (`-3..=-1`) widens the shift rather than narrowing it.
#[inline]
fn vext_get_vlmax(vlenb: u64, vsew: u32, lmul: i32) -> u64 {
    let vlen = vlenb << 3;
    let shift = vsew as i32 + 3 - lmul;
    debug_assert!((0..64).contains(&shift), "vlmax shift {shift} out of range");
    vlen >> shift
}

/// `VLMAX` for `vtype`, or `None` if `vtype` is illegal (`vill`), in which case
/// the `vsetvl*` yields `vl == 0`.
///
/// Mirrors the `vill` block of `HELPER(vsetvl)` in order:
/// fractional-LMUL-too-narrow, reserved `VLMUL` encoding, `SEW > ELEN`,
/// `vediv != 0`, reserved `vtype` bits, and the incoming `vill` bit.
pub fn vlmax_from_vtype(vlenb: u64, vtype: u64) -> Option<u64> {
    let vlmul = (vtype & 0x7) as u32;
    let vsew = ((vtype >> 3) & 0x7) as u32;
    let sew = 8u64 << vsew;
    let ediv = (vtype >> 8) & 0x3;
    let vlen = vlenb << 3;

    // vtype bit XLEN-1 is `vill` itself; bits 10..=XLEN-2 are reserved.
    let vill_in = (vtype >> (XLEN - 1)) & 1 != 0;
    let reserved = (vtype >> 10) & ((1u64 << (XLEN - 1 - 10)) - 1);

    if vlmul & 4 != 0 {
        // Fractional LMUL requires VLEN * LMUL >= SEW, i.e.
        // (vlenb << 3) >> (8 - vlmul) >= sew. `vlmul == 4` is reserved.
        if vlmul == 4 || (vlen >> (8 - vlmul)) < sew {
            return None;
        }
    }

    // `sew > elen` is the one QEMU check this cannot reproduce (ELEN is not
    // readable through the plugin API). ELEN is assumed >= 64, so only the
    // architecturally-reserved `vsew` encodings are rejected -- see the module
    // docs, and the runtime oracle that backs this assumption.
    if vsew > 3 || vill_in || ediv != 0 || reserved != 0 {
        return None;
    }

    let lmul = sign_extend_3(vlmul);
    Some(vext_get_vlmax(vlenb, vsew, lmul))
}

/// `sextract32(x, 0, 3)`: 0..=3 stay positive, 4..=7 become -4..=-1.
#[inline]
fn sign_extend_3(v: u32) -> i32 {
    ((v as i32) << 29) >> 29
}

/// The `vl` a `vsetvl*` with this `vlmax` and this AVL produces.
///
/// `vlmax` is `None` for an illegal `vtype`. `vl == 0` is a legitimate result
/// there, not a "missing" value.
#[inline]
pub fn compute_vl(vlmax: Option<u64>, avl: u64) -> u64 {
    match vlmax {
        None => 0,
        Some(vlmax) => {
            if avl <= vlmax {
                avl
            } else {
                vlmax
            }
        }
    }
}

/// Where a particular `vsetvl*` site's AVL and `vtype` come from, and hence what
/// its callback has to read at runtime. Decided once, at translate time, by
/// [`decode_vsetvl`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VlPlan {
    /// `vl` is fully determined here: `vsetivli` (immediate AVL, immediate
    /// `vtype`) and the `rs1 == x0`, `rd != x0` form of `vsetvli` (AVL is
    /// `RV_VLEN_MAX`, so `vl == vlmax`). No callback register read.
    Const(u64),
    /// `vsetvli` with `rs1 == x0` and `rd == x0`: AVL is the *current* `vl`, and
    /// `vl` is not readable without `R_REGS`, so it comes from the plugin's
    /// per-vCPU shadow. `vtype` is still the immediate, hence a known `vlmax`.
    Shadow { vlmax: Option<u64> },
    /// `vsetvli` with a real `rs1`: one `R_REGS` read of `x[rs1]` for the AVL.
    /// This is the only form that needs a register read on this workload class,
    /// and the only one that pays for `R_REGS`.
    Reg { rs1: u8, vlmax: Option<u64> },
    /// `vsetvl`: `vtype` comes from `x[rs2]` too, so nothing can be precomputed.
    /// SEW, LMUL, and VL can all change based on runtime values in this scenario.
    DynVtype { rs1: u8, rs2: u8, rd: u8 },
}

impl VlPlan {
    /// Whether this site's callback must run with `QEMU_PLUGIN_CB_R_REGS`. This
    /// is the count the whole design turns on: it must come out at ~1.79 M
    /// executions rather than the ~116.6 M block entries `R_REGS` used to cover.
    ///
    /// Only the tests read this — `lib.rs` matches on the variant directly, since
    /// each arm needs different callback bodies anyway. It is kept as the single
    /// written-down statement of which forms cost a flush.
    #[allow(dead_code)]
    pub fn needs_regs(&self) -> bool {
        VsetvlForm::from(*self).needs_regs()
    }
}

impl From<VlPlan> for VsetvlForm {
    /// Which form this plan belongs to, for the per-form counter arrays and
    /// the end-of-run report.
    fn from(plan: VlPlan) -> Self {
        match plan {
            VlPlan::Const(_) => VsetvlForm::Const,
            VlPlan::Shadow { .. } => VsetvlForm::Shadow,
            VlPlan::Reg { .. } => VsetvlForm::Reg,
            VlPlan::DynVtype { .. } => VsetvlForm::DynVtype,
        }
    }
}

/// A `vsetvl*` form, i.e. where its `vl` comes from. Its discriminant is the
/// index into the per-form counter arrays (`VSETVL_SITES` / `VSETVL_EXECS` in
/// `lib.rs`), and [`VsetvlForm::name`] is the report label - both live on this
/// one enum so there is no separate index/name table to keep in sync.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(usize)]
pub enum VsetvlForm {
    Const = 0,
    Shadow = 1,
    Reg = 2,
    DynVtype = 3,
}

#[allow(dead_code)] // only `lib.rs` (not built by `cargo test`, see Cargo.toml) uses these
impl VsetvlForm {
    /// All forms, in counter-array order.
    pub const ALL: [VsetvlForm; 4] = [Self::Const, Self::Shadow, Self::Reg, Self::DynVtype];

    /// Label for the end-of-run counter report.
    pub const fn name(self) -> &'static str {
        match self {
            Self::Const => "vl static per PC (vsetivli / vsetvli rs1==x0,rd!=x0)",
            Self::Shadow => "AVL from shadow vl (vsetvli rs1==x0,rd==x0)",
            Self::Reg => "AVL read from rs1  (R_REGS)",
            Self::DynVtype => "vsetvl, vtype from rs2 (R_REGS)",
        }
    }

    /// Whether this form's callback must run with `QEMU_PLUGIN_CB_R_REGS`.
    pub const fn needs_regs(self) -> bool {
        matches!(self, Self::Reg | Self::DynVtype)
    }
}

/// The `SEW` a `vsetvli`/`vsetivli` sets, from its immediate, at translate time.
/// `None` for `vsetvl`, whose `vtype` comes from a register.
///
/// Indexed vector memory ops need `SEW` (their `width` field describes their
/// indices instead), and reading the `vtype` CSR back would put a register read
/// on sites that currently need none — the same trade this module makes for
/// `vl`. The decode itself is
/// [`turbo_tracer_shared::vmem::vsetvl_static_sew_bits`], shared with the
/// consumer that has to agree about it.
pub fn decode_vsetvl_static_sew(bytes: &[u8]) -> Option<u16> {
    let word = u32::from_le_bytes(bytes.get(..4)?.try_into().ok()?);
    turbo_tracer_shared::vmem::vsetvl_static_sew_bits(word)
}

/// Decode a `vsetvl*` from its raw little-endian instruction bytes and decide
/// how its `vl` will be obtained. Returns `None` if `bytes` is not a `vsetvl*`.
///
/// Decoding the encoding rather than parsing QEMU's disassembly text is
/// deliberate: the operand fields are what the arithmetic needs, and the
/// disassembly spells `vtype` as `e64,m1,ta,ma` (or as a bare hex immediate,
/// depending on the binutils version), which is a bad thing to depend on.
pub fn decode_vsetvl(bytes: &[u8], vlenb: u64) -> Option<VlPlan> {
    if bytes.len() < 4 {
        return None; // no compressed vsetvl* encoding exists
    }
    let insn = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);

    // OP-V (0b1010111) with funct3 = 0b111 (OPCFG) is the vsetvl* group.
    if insn & 0x7f != 0x57 || (insn >> 12) & 0x7 != 0x7 {
        return None;
    }

    let rd = ((insn >> 7) & 0x1f) as u8;
    let rs1 = ((insn >> 15) & 0x1f) as u8;

    if insn >> 31 == 0 {
        // vsetvli: zimm11 = insn[30:20] is the whole vtype.
        let vtype = ((insn >> 20) & 0x7ff) as u64;
        let vlmax = vlmax_from_vtype(vlenb, vtype);
        return Some(match (rs1, rd) {
            // `rd == x0 && rs1 == x0`: keep vl, take AVL from the current vl.
            (0, 0) => VlPlan::Shadow { vlmax },
            // `rs1 == x0`: AVL is RV_VLEN_MAX, so vl == vlmax. Fully static.
            (0, _) => VlPlan::Const(compute_vl(vlmax, RV_VLEN_MAX)),
            _ => VlPlan::Reg { rs1, vlmax },
        });
    }

    if insn >> 30 == 0b11 {
        // vsetivli: zimm10 = insn[29:20] is vtype, uimm = insn[19:15] is the AVL.
        let vtype = ((insn >> 20) & 0x3ff) as u64;
        let uimm = ((insn >> 15) & 0x1f) as u64;
        return Some(VlPlan::Const(compute_vl(
            vlmax_from_vtype(vlenb, vtype),
            uimm,
        )));
    }

    if insn >> 25 == 0b1000000 {
        // vsetvl: vtype from x[rs2], AVL from x[rs1] (or the x0 forms).
        let rs2 = ((insn >> 20) & 0x1f) as u8;
        return Some(VlPlan::DynVtype { rs1, rs2, rd });
    }

    None
}
