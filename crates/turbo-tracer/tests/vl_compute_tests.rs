//! Unit tests for the computed-`vl` arithmetic and `vsetvl*` decode
//! (`src/vl_compute.rs`). Follows the same technique as
//! `vector_mem_classify_tests.rs`: the module is pulled in by path, because the
//! `turbo-tracer` crate itself cannot be linked into a normal test binary (it
//! references QEMU plugin FFI symbols that only resolve when QEMU loads the
//! plugin).
//!
//! These pin the transcription of QEMU's `HELPER(vsetvl)` /`vext_get_vlmax`,
//! which is the part of the tracer that replaces a read of ground truth with
//! arithmetic.

#[path = "../src/vl_compute.rs"]
// Pulled in by path, so items the crate's own callers use (rather than these
// tests) have no user in this binary and look dead.
#[allow(dead_code)]
mod vl_compute;

use vl_compute::*;

/// vlen = 16384 bits, the configuration every benchmark here runs with.
const VLENB: u64 = 2048;

/// Assemble a `vsetvli rd, rs1, vtype`.
fn vsetvli(rd: u8, rs1: u8, vtype: u32) -> [u8; 4] {
    let insn = 0x57 | (7 << 12) | ((rd as u32) << 7) | ((rs1 as u32) << 15) | (vtype << 20);
    insn.to_le_bytes()
}

/// Assemble a `vsetivli rd, uimm, vtype`.
fn vsetivli(rd: u8, uimm: u32, vtype: u32) -> [u8; 4] {
    let insn = 0x57 | (7 << 12) | ((rd as u32) << 7) | (uimm << 15) | (vtype << 20) | (0b11 << 30);
    insn.to_le_bytes()
}

/// `vtype` for a given SEW/LMUL, as the assembler would encode `eN,mM`.
fn vtype(vsew: u64, vlmul: u64) -> u64 {
    (vsew << 3) | vlmul
}

#[test]
fn vlmax_matches_vext_get_vlmax() {
    // VLEN=16384. VLMAX = VLEN / SEW * LMUL.
    // e8,m1 -> 16384/8 = 2048
    assert_eq!(vlmax_from_vtype(VLENB, vtype(0, 0)), Some(2048));
    // e64,m1 -> 16384/64 = 256
    assert_eq!(vlmax_from_vtype(VLENB, vtype(3, 0)), Some(256));
    // e64,m8 -> 256*8 = 2048
    assert_eq!(vlmax_from_vtype(VLENB, vtype(3, 3)), Some(2048));
    // e8,m8 -> 2048*8 = 16384
    assert_eq!(vlmax_from_vtype(VLENB, vtype(0, 3)), Some(16384));
    // e32,mf2 (vlmul=7 -> lmul=-1) -> 16384/32/2 = 256
    assert_eq!(vlmax_from_vtype(VLENB, vtype(2, 7)), Some(256));
    // e32,mf8 (vlmul=5 -> lmul=-3) -> 16384/32/8 = 64
    assert_eq!(vlmax_from_vtype(VLENB, vtype(2, 5)), Some(64));
}

#[test]
fn vill_cases_yield_none() {
    // vlmul == 4 is the reserved fractional encoding.
    assert_eq!(vlmax_from_vtype(VLENB, vtype(0, 4)), None);
    // vsew > 3 is reserved (and would exceed any ELEN this tool sees).
    assert_eq!(vlmax_from_vtype(VLENB, vtype(4, 0)), None);
    // vediv != 0.
    assert_eq!(vlmax_from_vtype(VLENB, vtype(0, 0) | (1 << 8)), None);
    // reserved bit 10 set -- reachable from a vsetvli's 11-bit immediate.
    assert_eq!(vlmax_from_vtype(VLENB, vtype(0, 0) | (1 << 10)), None);
    // Fractional LMUL too narrow for SEW: VLEN=8 (vlenb=1), e64, mf8
    // -> (8 >> 3) = 1 < 64.
    assert_eq!(vlmax_from_vtype(1, vtype(3, 5)), None);
    // An illegal vtype yields vl == 0, not a missing value.
    assert_eq!(compute_vl(None, 12345), 0);
}

#[test]
fn avl_is_clamped_to_vlmax() {
    let vlmax = vlmax_from_vtype(VLENB, vtype(3, 0)); // 256
    assert_eq!(compute_vl(vlmax, 100), 100);
    assert_eq!(compute_vl(vlmax, 256), 256);
    assert_eq!(compute_vl(vlmax, 257), 256);
    assert_eq!(compute_vl(vlmax, RV_VLEN_MAX), 256);
    assert_eq!(compute_vl(vlmax, 0), 0);
}

#[test]
fn vsetvli_with_real_rs1_reads_that_register() {
    let plan = decode_vsetvl(&vsetvli(15, 14, vtype(3, 0) as u32), VLENB).unwrap();
    assert_eq!(
        plan,
        VlPlan::Reg {
            rs1: 14,
            vlmax: Some(256)
        }
    );
    assert!(
        plan.needs_regs(),
        "a real rs1 is the one form that pays R_REGS"
    );
}

#[test]
fn vsetvli_rs1_x0_is_static_vlmax() {
    // rd != x0, rs1 == x0: AVL is RV_VLEN_MAX, so vl == vlmax.
    let plan = decode_vsetvl(&vsetvli(15, 0, vtype(3, 0) as u32), VLENB).unwrap();
    assert_eq!(plan, VlPlan::Const(256));
    assert!(!plan.needs_regs());
}

#[test]
fn vsetvli_x0_x0_takes_avl_from_the_shadow() {
    let plan = decode_vsetvl(&vsetvli(0, 0, vtype(3, 0) as u32), VLENB).unwrap();
    assert_eq!(plan, VlPlan::Shadow { vlmax: Some(256) });
    assert!(
        !plan.needs_regs(),
        "the shadow vl exists so this stays NO_REGS"
    );
    // This form preserves vl whenever vlmax still permits it, which is why
    // the shadow is usually a no-op read.
    assert_eq!(compute_vl(Some(256), 100), 100);
}

#[test]
fn vsetivli_is_fully_static() {
    let plan = decode_vsetvl(&vsetivli(15, 8, vtype(3, 0) as u32), VLENB).unwrap();
    assert_eq!(plan, VlPlan::Const(8));
    assert!(!plan.needs_regs());
    // ...including when the immediate AVL exceeds vlmax.
    let plan = decode_vsetvl(&vsetivli(15, 31, vtype(3, 7) as u32), VLENB).unwrap();
    // vlmax here is 128, so the AVL immediate passes through unclamped.
    assert_eq!(plan, VlPlan::Const(31));
}

#[test]
fn vsetvl_needs_both_registers() {
    // vsetvl rd=15, rs1=14, rs2=13: 0b1000000 in insn[31:25].
    let insn: u32 = 0x57 | (7 << 12) | (15 << 7) | (14 << 15) | (13 << 20) | (0x40 << 25);
    let plan = decode_vsetvl(&insn.to_le_bytes(), VLENB).unwrap();
    assert_eq!(
        plan,
        VlPlan::DynVtype {
            rs1: 14,
            rs2: 13,
            rd: 15
        }
    );
    assert!(plan.needs_regs());
}

#[test]
fn non_vsetvl_encodings_are_rejected() {
    // addi a5, a4, 1
    assert!(decode_vsetvl(&0x00170793u32.to_le_bytes(), VLENB).is_none());
    // vadd.vv v1, v2, v3 -- OP-V, but funct3 = 0b000, not OPCFG.
    let vadd: u32 = 0x57 | (1 << 7) | (3 << 20) | (2 << 15);
    assert!(decode_vsetvl(&vadd.to_le_bytes(), VLENB).is_none());
    // A compressed (2-byte) instruction can never be a vsetvl*.
    assert!(decode_vsetvl(&[0x01, 0x00], VLENB).is_none());
}
