//! Vector Length (VL), Selected Element Width (SEW), and LMUL tracking for RISC-V Vector instructions.
//!
//! This module provides a pure data structure that tracks the current VL, SEW, and LMUL values
//! as instructions are processed. It detects the instructions that *write* `vl` --
//! `vsetvli`, `vsetivli`, `vsetvl`, and the fault-only-first vector loads
//! (`vl*ff`, which trim `vl` to the first faulting element) -- and consumes VL
//! values from the provided trace data.

use crate::common::Lmul;

/// Tracks the current Vector Length (VL), Selected Element Width (SEW), and LMUL state.
///
/// This structure maintains state across instruction processing and updates when
/// vector configuration instructions (`vsetvl*`) are encountered.
#[derive(Debug, Clone)]
pub struct VlTracker {
    /// Current vector length value, if set by a vsetvl* instruction
    pub current_vl: Option<u32>,
    /// Current selected element width (encoded as vsew field: 0=e8, 1=e16, 2=e32, 3=e64)
    pub current_sew: Option<u8>,
    /// Current LMUL (encoded as vlmul field bits [2:0]: 0=m1, 1=m2, 2=m4, 3=m8, 5=mf8, 6=mf4, 7=mf2)
    pub current_lmul: Option<Lmul>,
}

impl VlTracker {
    /// Creates a new VL tracker with no initial state.
    pub fn new() -> Self {
        Self {
            current_vl: None,
            current_sew: None,
            current_lmul: None,
        }
    }

    /// Process an instruction and update VL/SEW/LMUL state if it writes `vl`,
    /// i.e. is a `vsetvl*` or a fault-only-first vector load.
    ///
    /// # Arguments
    /// * `ins` - The decoded RISC-V instruction
    /// * `pc` - The program counter (used for warning messages)
    /// * `vl_values` - Slice of VL values captured from the trace
    /// * `vl_index` - Mutable index into the VL values slice (incremented when consumed)
    ///
    /// # Returns
    /// * `Ok(Some(vl))` if the instruction writes `vl` and a value was consumed
    /// * `Ok(None)` if the instruction does not write `vl`
    /// * `Err(_)` if it writes `vl` but the VL stream is exhausted
    ///
    /// Exhaustion is an error, not a warning: `vl` is written into the E-Trace
    /// stream as a CSR data packet *before* the instruction-trace packet that
    /// retires the `vsetvl*`, so by the time the decoder walks past a `vsetvl*`
    /// its `vl` has always been decoded. A missing value therefore means the
    /// producer dropped one, and every downstream vector figure (VL histogram,
    /// bytes moved, cycle estimate) silently uses the *previous* `vsetvl*`'s
    /// `vl` from there on.
    ///
    /// Note `vl == 0` is a legitimate value (the `vill` case), and is returned
    /// as `Ok(Some(0))` rather than treated as missing.
    pub fn process_instruction(
        &mut self,
        ins: &riscv_isa::Instruction,
        pc: u64,
        vl_values: &[u16],
        vl_index: &mut usize,
    ) -> anyhow::Result<Option<u32>> {
        match ins {
            // VSETVLI: Set VL and SEW from immediate value
            // zimm11[10:0] encodes vtypei, where bits [5:3] encode SEW
            riscv_isa::Instruction::VSETVLI { zimm11, .. } => {
                // Extract SEW from vtypei immediate: bits [5:3]
                let vsew = ((zimm11 >> 3) & 0x7) as u8;
                self.current_sew = Some(vsew);
                self.current_lmul = Lmul::from_vlmul((zimm11 & 0x7) as u8);

                self.consume_vl_value(pc, vl_values, vl_index)
            }

            // VSETIVLI: Set VL and SEW from immediate value (VL is also immediate)
            // zimm10[9:0] encodes vtypei, where bits [5:3] encode SEW
            riscv_isa::Instruction::VSETIVLI { zimm10, .. } => {
                // Extract SEW from vtypei immediate: bits [5:3]
                let vsew = ((zimm10 >> 3) & 0x7) as u8;
                self.current_sew = Some(vsew);
                self.current_lmul = Lmul::from_vlmul((zimm10 & 0x7) as u8);

                self.consume_vl_value(pc, vl_values, vl_index)
            }

            // VSETVL: Set VL and SEW from register values
            // SEW comes from rs2 register at runtime - carry forward current_sew
            riscv_isa::Instruction::VSETVL { .. } => self.consume_vl_value(pc, vl_values, vl_index),

            // Fault-only-first loads: `vl` is trimmed to the index of the first
            // element that faults (V spec 8.2), so these write `vl` without
            // touching the rest of `vtype` -- SEW and LMUL carry forward, like
            // `vsetvl`'s. The tracer emits one VL record per execution of one of
            // these, trimmed or not, so the value is consumed unconditionally
            // here.
            //
            // The value takes effect *including* for this instruction: an ff
            // load that trimmed `vl` to n only ever accessed n elements, so n is
            // what its own byte and element counts are computed from. The
            // engine's Step 2 (this call) running before Step 3 (the
            // `InsnDelta`) is what makes that so.
            riscv_isa::Instruction::VLE8FF_V { .. }
            | riscv_isa::Instruction::VLE16FF_V { .. }
            | riscv_isa::Instruction::VLE32FF_V { .. }
            | riscv_isa::Instruction::VLE64FF_V { .. }
            | riscv_isa::Instruction::VLSEGFF_V { .. } => {
                self.consume_vl_value(pc, vl_values, vl_index)
            }

            // Does not write `vl`
            _ => Ok(None),
        }
    }

    /// Consume the next VL value from the slice and update current_vl.
    ///
    /// # Arguments
    /// * `pc` - Program counter for warning messages
    /// * `vl_values` - Slice of VL values from trace
    /// * `vl_index` - Mutable index into vl_values (incremented on success)
    ///
    /// # Returns
    /// * `Ok(Some(vl))` if a value was successfully consumed
    /// * `Err(_)` if no more values are available
    fn consume_vl_value(
        &mut self,
        pc: u64,
        vl_values: &[u16],
        vl_index: &mut usize,
    ) -> anyhow::Result<Option<u32>> {
        if *vl_index < vl_values.len() {
            let vl = vl_values[*vl_index];
            *vl_index += 1;
            let vl_u32 = u32::from(vl);
            self.current_vl = Some(vl_u32);
            Ok(Some(vl_u32))
        } else {
            anyhow::bail!(
                "vl stream underrun at vl-writing instruction, PC {pc:#x}: wanted value \
                 #{} but the trace only carried {}. Every vsetvl* and every vl*ff the \
                 decoder walks past must consume exactly one vl value; a missing one \
                 means the tracer dropped it, and every vector figure after this point \
                 would be computed from a stale vl.",
                *vl_index,
                vl_values.len()
            );
        }
    }
}

/// Whether `ins` writes `vl`, i.e. is one that
/// [`VlTracker::process_instruction`] reacts to.
///
/// This must list exactly the variants that `process_instruction` matches: the
/// run cache uses it to terminate a run *before* any instruction that mutates
/// vtype or consumes from the VL stream, and a run that swallowed one would
/// apply the wrong vtype to everything after it. `vl_writers_match_tracker`
/// in this module's tests pins the two lists together.
///
/// The `vl*ff` entries are why this is no longer called `is_vsetvl`: a
/// fault-only-first load changes `vl` too, and it changes it by an amount only
/// the trace knows, so it has to end a memoized run exactly as a `vsetvl*` does.
#[inline]
pub(crate) fn writes_vl(ins: &riscv_isa::Instruction) -> bool {
    matches!(
        ins,
        riscv_isa::Instruction::VSETVLI { .. }
            | riscv_isa::Instruction::VSETIVLI { .. }
            | riscv_isa::Instruction::VSETVL { .. }
            | riscv_isa::Instruction::VLE8FF_V { .. }
            | riscv_isa::Instruction::VLE16FF_V { .. }
            | riscv_isa::Instruction::VLE32FF_V { .. }
            | riscv_isa::Instruction::VLE64FF_V { .. }
            | riscv_isa::Instruction::VLSEGFF_V { .. }
    )
}

impl Default for VlTracker {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vl_tracker_initial_state() {
        let tracker = VlTracker::new();
        assert_eq!(tracker.current_vl, None);
        assert_eq!(tracker.current_sew, None);
        assert_eq!(tracker.current_lmul, None);
    }

    #[test]
    fn test_vsetvli_updates_sew_and_vl() {
        let mut tracker = VlTracker::new();
        let vl_values = vec![128u16, 64u16];
        let mut vl_index = 0;

        // zimm11 with SEW=2 (e32) encoded in bits [5:3]
        // SEW=2 means bits [5:3] = 010, so zimm11 bit pattern: ...00010...
        let zimm11 = 0b00010000; // SEW field = 2
        let ins = riscv_isa::Instruction::VSETVLI {
            rd: 1,
            rs1: 2,
            zimm11,
        };

        let result = tracker
            .process_instruction(&ins, 0x1000, &vl_values, &mut vl_index)
            .unwrap();

        assert_eq!(result, Some(128));
        assert_eq!(tracker.current_vl, Some(128));
        assert_eq!(tracker.current_sew, Some(2)); // e32
        assert_eq!(tracker.current_lmul, Some(Lmul::M1)); // m1
        assert_eq!(vl_index, 1);
    }

    #[test]
    fn test_vsetivli_updates_sew_and_vl() {
        let mut tracker = VlTracker::new();
        let vl_values = vec![256u16];
        let mut vl_index = 0;

        // zimm10 with SEW=3 (e64) encoded in bits [5:3]
        let zimm10 = 0b00011000; // SEW field = 3
        let ins = riscv_isa::Instruction::VSETIVLI {
            rd: 1,
            uimm: 2,
            zimm10,
        };

        let result = tracker
            .process_instruction(&ins, 0x2000, &vl_values, &mut vl_index)
            .unwrap();

        assert_eq!(result, Some(256));
        assert_eq!(tracker.current_vl, Some(256));
        assert_eq!(tracker.current_sew, Some(3)); // e64
        assert_eq!(tracker.current_lmul, Some(Lmul::M1)); // m1
        assert_eq!(vl_index, 1);
    }

    #[test]
    fn test_vsetvl_preserves_sew() {
        let mut tracker = VlTracker::new();
        tracker.current_sew = Some(1); // e16
        tracker.current_lmul = Some(Lmul::M2); // m2
        let vl_values = vec![64u16];
        let mut vl_index = 0;

        let ins = riscv_isa::Instruction::VSETVL {
            rd: 1,
            rs1: 2,
            rs2: 3,
        };

        let result = tracker
            .process_instruction(&ins, 0x3000, &vl_values, &mut vl_index)
            .unwrap();

        assert_eq!(result, Some(64));
        assert_eq!(tracker.current_vl, Some(64));
        assert_eq!(tracker.current_sew, Some(1)); // Preserved
        assert_eq!(tracker.current_lmul, Some(Lmul::M2)); // Preserved
        assert_eq!(vl_index, 1);
    }

    /// An exhausted VL stream is a hard error now, not a warning: it means the
    /// tracer dropped a value. See `consume_vl_value`.
    #[test]
    fn test_exhausted_vl_values() {
        let mut tracker = VlTracker::new();
        let vl_values = vec![];
        let mut vl_index = 0;

        let ins = riscv_isa::Instruction::VSETVLI {
            rd: 1,
            rs1: 2,
            zimm11: 0,
        };

        let result = tracker.process_instruction(&ins, 0x4000, &vl_values, &mut vl_index);

        assert!(result.is_err(), "an exhausted vl stream must be an error");
        assert_eq!(tracker.current_vl, None);
        assert_eq!(tracker.current_sew, Some(0)); // SEW is set even when VL exhausted
        assert_eq!(tracker.current_lmul, Some(Lmul::M1)); // LMUL is set even when VL exhausted
        assert_eq!(vl_index, 0); // Index should not increment
    }

    #[test]
    fn test_non_vsetvl_instruction() {
        let mut tracker = VlTracker::new();
        tracker.current_vl = Some(128);
        tracker.current_sew = Some(2);
        tracker.current_lmul = Some(Lmul::M8);
        let vl_values = vec![64u16];
        let mut vl_index = 0;

        // Use a non-vsetvl instruction (e.g., VADD)
        let ins = riscv_isa::Instruction::VADD_VV {
            vd: 1,
            vs1: 2,
            vs2: 3,
            vm: 1,
        };

        let result = tracker
            .process_instruction(&ins, 0x5000, &vl_values, &mut vl_index)
            .unwrap();

        assert_eq!(result, None);
        assert_eq!(tracker.current_vl, Some(128)); // Unchanged
        assert_eq!(tracker.current_sew, Some(2)); // Unchanged
        assert_eq!(tracker.current_lmul, Some(Lmul::M8)); // Unchanged
        assert_eq!(vl_index, 0); // No VL consumed
    }

    /// A fault-only-first load consumes a VL value like a `vsetvl*` does, and
    /// leaves SEW/LMUL alone: it writes `vl` and nothing else in `vtype`.
    ///
    /// This is the whole point of `vl*ff` capture. Before it, the tracer emitted
    /// no value for one of these and the tracker consumed none, so a `vl` a
    /// `vl*ff` had trimmed stayed stale until the next `vsetvl*` -- every vector
    /// figure in between computed from a length the guest was not running at.
    #[test]
    fn fault_only_first_loads_consume_a_vl() {
        for ins in [
            riscv_isa::Instruction::VLE8FF_V {
                vm: 1,
                rs1: 10,
                vd: 0,
            },
            riscv_isa::Instruction::VLE16FF_V {
                vm: 1,
                rs1: 10,
                vd: 0,
            },
            riscv_isa::Instruction::VLE32FF_V {
                vm: 1,
                rs1: 10,
                vd: 0,
            },
            riscv_isa::Instruction::VLE64FF_V {
                vm: 1,
                rs1: 10,
                vd: 0,
            },
            riscv_isa::Instruction::VLSEGFF_V {
                nf: 2,
                eew: 8,
                vm: 1,
                rs1: 10,
                vd: 0,
            },
        ] {
            let mut tracker = VlTracker::new();
            // The state a preceding `vsetvli t1, t0, e16, m2` would have left.
            tracker.current_vl = Some(16);
            tracker.current_sew = Some(1);
            tracker.current_lmul = Some(Lmul::M2);
            let mut vl_index = 0;

            let result = tracker
                .process_instruction(&ins, 0x6000, &[8u16], &mut vl_index)
                .unwrap();

            assert_eq!(result, Some(8), "{ins:?} must consume the trimmed vl");
            assert_eq!(tracker.current_vl, Some(8), "{ins:?} must apply it");
            assert_eq!(tracker.current_sew, Some(1), "{ins:?} must not touch SEW");
            assert_eq!(
                tracker.current_lmul,
                Some(Lmul::M2),
                "{ins:?} must not touch LMUL"
            );
            assert_eq!(vl_index, 1);
        }
    }

    /// A non-ff vector load must *not* consume one: the tracer emits a record
    /// only for the instructions that write `vl`, so consuming here would shift
    /// every later value by one.
    #[test]
    fn a_plain_vector_load_consumes_no_vl() {
        let mut tracker = VlTracker::new();
        tracker.current_vl = Some(16);
        let mut vl_index = 0;
        let ins = riscv_isa::Instruction::VLE64_V {
            vm: 1,
            rs1: 10,
            vd: 0,
        };

        let result = tracker
            .process_instruction(&ins, 0x7000, &[8u16], &mut vl_index)
            .unwrap();

        assert_eq!(result, None);
        assert_eq!(tracker.current_vl, Some(16));
        assert_eq!(vl_index, 0);
    }

    /// `writes_vl` must agree with `process_instruction` on every instruction:
    /// the run cache relies on it to spot vtype changes, so a variant handled by
    /// the tracker but missed by the predicate would silently let a run span a
    /// `vsetvl*` or a `vl*ff`.
    #[test]
    fn vl_writers_match_tracker() {
        let vsetvl_family = [
            riscv_isa::Instruction::VSETVLI {
                rd: 1,
                rs1: 2,
                zimm11: 0,
            },
            riscv_isa::Instruction::VSETIVLI {
                rd: 1,
                uimm: 2,
                zimm10: 0,
            },
            riscv_isa::Instruction::VSETVL {
                rd: 1,
                rs1: 2,
                rs2: 3,
            },
            riscv_isa::Instruction::VLE8FF_V {
                vm: 1,
                rs1: 10,
                vd: 0,
            },
            riscv_isa::Instruction::VLE16FF_V {
                vm: 1,
                rs1: 10,
                vd: 0,
            },
            riscv_isa::Instruction::VLE32FF_V {
                vm: 1,
                rs1: 10,
                vd: 0,
            },
            riscv_isa::Instruction::VLE64FF_V {
                vm: 1,
                rs1: 10,
                vd: 0,
            },
            riscv_isa::Instruction::VLSEGFF_V {
                nf: 2,
                eew: 8,
                vm: 1,
                rs1: 10,
                vd: 0,
            },
        ];
        let others = [
            riscv_isa::Instruction::VADD_VV {
                vd: 1,
                vs1: 2,
                vs2: 3,
                vm: 1,
            },
            riscv_isa::Instruction::ADDI {
                rd: 1,
                rs1: 1,
                imm: 1,
            },
            // The non-ff load the ff variants must not be confused with.
            riscv_isa::Instruction::VLE64_V {
                vm: 1,
                rs1: 10,
                vd: 0,
            },
        ];

        for ins in &vsetvl_family {
            let mut tracker = VlTracker::new();
            let mut idx = 0;
            assert!(writes_vl(ins), "{ins:?} must be recognised as writing vl");
            assert_eq!(
                tracker
                    .process_instruction(ins, 0, &[8u16], &mut idx)
                    .unwrap(),
                Some(8),
                "{ins:?} must consume a VL value"
            );
        }
        for ins in &others {
            let mut tracker = VlTracker::new();
            let mut idx = 0;
            assert!(
                !writes_vl(ins),
                "{ins:?} must not be recognised as writing vl"
            );
            assert_eq!(
                tracker
                    .process_instruction(ins, 0, &[8u16], &mut idx)
                    .unwrap(),
                None
            );
        }
    }
}
