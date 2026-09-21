//! RISC-V BBT QEMU Plugin
//!
//! Instruments RISC-V guest execution to produce BBT (basic-block-trace) data:
//! one `u32` record per executed block, naming a dense ID that a shared
//! dictionary (built at translate time) maps back to the block's PC sequence.
//! Control flow is reconstructed offline from the block sequence + dictionary,
//! so this plugin never encodes branch/jump/retirement packets itself.
//!
//! ## Architecture
//!
//! The TB-start callback appends one record word per block entry
//! (`TraceEncoder::record_block_word`) -- computed once at translation time
//! and captured in the block's closure, so nothing at block-entry time
//! derives, branches on, or encodes anything. It is registered `NO_REGS` and
//! is the only per-block callback; there is no per-instruction inline store
//! and no jump/call/first-instruction bookkeeping, because BBT needs none of
//! the "what was the previous instruction" state E-Trace's packet encoding
//! required.
//!
//! The exceptions are the two instructions that write `vl`. A `vsetvl*`
//! callback still runs per site, computing and emitting the instruction's
//! resulting `vl` (`bbt_vl`). Only the sites whose AVL comes from a register
//! carry `QEMU_PLUGIN_CB_R_REGS` -- see [`vl_compute`], which is where the
//! reasoning about that flag lives.
//!
//! A fault-only-first vector load (`vl*ff`) also writes `vl`, and its value
//! *cannot* be computed: it depends on which element first lands on an unmapped
//! page. So each `vl*ff` site marks [`VCPUScoreBoard::ff_pending_pc`] on
//! execution (`NO_REGS`, one scoreboard store) and the *successor* block --
//! statically `ff_pc + 4`, because a `vl*ff` ends its TB just as a `vsetvl*`
//! does -- carries an `R_REGS` callback that reads the `vl` CSR and emits the
//! record. That is the only block in the program that pays `R_REGS`, and only
//! blocks that actually follow a `vl*ff` have one.
//!
//! ## Plugin arguments
//!
//! Besides `workdir`, `run_id`, `debug`:
//!
//! - `flush_interval=<blocks>` (default 10 000, env fallback
//!   `TURBO_TRACE_FLUSH_INTERVAL`): block executions between record-log flushes,
//!   i.e. the size of one batch. Bigger windows compress better (~10% smaller
//!   on disk from 10 000 to 100 000) and amortize the per-batch handoff, but the
//!   compressed record must stay under the consumer's read low-water mark, and
//!   they coarsen the granularity the consumer can pipeline at.
//! - `mem_capture=on|off` (default `on`, env fallback `TURBO_MEM_CAPTURE`):
//!   capture the runtime addresses of
//!   vector loads and stores, so a consumer can replay the workload's cache
//!   footprint.
//! - `vl_verify=abort|warn|off` (default `abort`): what to do when the
//!   computed-vs-read `vl` oracle disagrees with QEMU. See [`check_vl_oracle`].
//! - `compress=0..9` (default 1): deflate level for record-log payloads.
//!   Level 0 does not disable compression so much as make it a no-op -- deflate
//!   emits *stored* blocks, so the file grows to roughly its uncompressed size
//!   while staying a valid stream the ordinary reader decodes. That is the
//!   setting to use when measuring what compression costs the vCPU thread.
//!   Also settable as `TURBO_TRACE_COMPRESS=<0-9>` in the environment, for
//!   sweeping the level through a consumer that does not pass the plugin arg.
//!
//! ## Performance: Lock-Free Per-vCPU Access
//!
//! QEMU guarantees that callbacks receiving a `VCPUIndex` run exclusively on that
//! vCPU's thread. This means per-vCPU data (encoders, register descriptors) does NOT
//! need mutex protection -- we use `VCPUArray<T>` backed by `UnsafeCell` for zero-cost
//! indexed access instead of `Mutex<HashMap<VCPUIndex, T>>`.

use qemu_plugin::{
    register, HasCallbacks, PluginId, Register, Result, TranslationBlock, VCPUIndex,
};
use std::cell::UnsafeCell;
use std::fmt;
use std::mem;
use std::path::PathBuf;
use std::ptr;
use std::sync::atomic::{AtomicPtr, AtomicU64, Ordering};
use std::sync::{Arc, OnceLock};

/// Monotonically increasing counter used to assign a unique file ID to each
/// thread that ever occupies a vCPU slot.  Because QEMU reuses vCPU indices
/// when pthreads exit and new ones start, using the raw vcpu_id would cause
/// multiple threads to share the same `turbo_trace_vcpu<N>.bin` file, leading to
/// concatenated E-Trace streams and decoding artefacts.  Each new thread gets
/// a unique ID here, so every thread's trace lands in its own file.
static NEXT_HART_FILE_ID: AtomicU64 = AtomicU64::new(0);

use turbo_tracer_shared::{RoiRtMarker, RoiRtMarkerKind};

use qemu_plugin::{PluginU64, Scoreboard};

mod trace_encoder;
use trace_encoder::*;

mod vl_compute;
use vl_compute::{
    compute_vl, decode_vsetvl, decode_vsetvl_static_sew, vlmax_from_vtype, VlPlan, VsetvlForm,
    RV_VLEN_MAX,
};

use std::sync::Mutex;
use turbo_tracer_shared::bbt::{
    bbt_block, encode_block_desc, MemGroupKind, BBT_ADDR_BITS, BBT_MAX_BLOCK_ID,
};
use turbo_tracer_shared::vmem::{indexed_lines, sew_bits_from_vtype, VecMemOp};

// ---------------------------------------------------------------------------
// `vsetvl*` / computed-`vl` counters
//
// Every tracer design projection is checked against these, so they ship
// unconditionally rather than behind a debug flag: they are two relaxed
// increments per `vsetvl*` execution (3.56 M on a measured workload), against the 116.6 M
// block callbacks they exist to justify removing work from. The one that
// actually settles whether this design did what it claims is the sum of
// `VSETVL_EXECS` over the forms where `VsetvlForm::needs_regs()` is true --
// the number of `R_REGS` callback invocations, which must read ~1.8 M and
// not ~116 M.
// ---------------------------------------------------------------------------

/// `vsetvl*` sites translated, indexed by [`VsetvlForm`] (as `usize`).
static VSETVL_SITES: [AtomicU64; 4] = [
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
];
/// `vsetvl*` executions, indexed by [`VsetvlForm`] (as `usize`).
static VSETVL_EXECS: [AtomicU64; 4] = [
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
];
/// `vl*ff` sites translated, i.e. fault-only-first loads found at JIT time.
static VLEFF_SITES: AtomicU64 = AtomicU64::new(0);
/// `vl*ff` executions, i.e. `vl` reads owed to a fault-only-first load.
static VLEFF_EXECS: AtomicU64 = AtomicU64::new(0);
/// `vl*ff` executions whose `vl` actually came back *different* from the shadow,
/// i.e. the loads that really were trimmed by a fault.
static VLEFF_TRIMS: AtomicU64 = AtomicU64::new(0);
/// `vl*ff` executions whose successor-block callback never ran, counted when the
/// next one is about to overwrite the pending mark. Non-zero means a `vl` record
/// was owed and never emitted -- see [`TurboTracePlugin::note_ff_successor`].
static VLEFF_LOST: AtomicU64 = AtomicU64::new(0);

/// Computed-vs-read `vl` comparisons actually taken (see [`check_vl_oracle`]).
static VL_ORACLE_SAMPLES: AtomicU64 = AtomicU64::new(0);
/// Comparisons that disagreed. Non-zero means the computed `vl` is wrong.
static VL_ORACLE_MISMATCHES: AtomicU64 = AtomicU64::new(0);

/// What to do when the computed-vs-read `vl` oracle disagrees. Plugin arg
/// `vl_verify=abort|warn|off`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum VlVerify {
    /// Default. A divergence means every `vl` in the trace is suspect, so stop.
    Abort,
    /// Log and keep going -- for investigating a divergence, not for producing
    /// numbers anyone should trust.
    Warn,
    /// Skip the check, and the `vl` CSR read it needs.
    Off,
}

/// The computed-vs-read `vl` oracle, in the cheap form.
///
/// The design proposed sampling the real `vl` on each `vsetvl*`'s successor
/// block for the first K executions of each PC. That inherits Option C's
/// translation-order problem (the successor may be translated before the
/// `vsetvl*` that identifies it) and, worse, cannot be sampled at all: the
/// `R_REGS` flag is fixed per callback registration, so "check the first 16"
/// would still put `R_REGS` on those blocks forever.
///
/// This instead exploits an ordering that costs nothing. A `vsetvl*` site that
/// reads `x[rs1]` already carries `R_REGS`, and its callback runs *before* the
/// instruction -- so at that moment the `vl` CSR still holds the value the
/// *previous* `vsetvl*` produced, which is exactly the value this plugin last
/// computed and wrote to `shadow_vl`. Comparing the two costs one extra
/// register read (~19 ns) at a site that is already paying for the flush, and it
/// validates the computation continuously for the whole run rather than 16 times
/// per PC.
///
/// Coverage, and what it does *not* see:
///
/// * A form that needs no register read (`Const`/`Shadow`) has no `R_REGS`, so
///   it cannot be checked at its own site -- but its result is checked by the
///   next `R_REGS` site to execute, since `shadow_vl` carries it forward. On
///   a measured workload that is 50.1% of `vsetvl*` executions, i.e. continuously.
/// * The last `vsetvl*` before the run ends is never checked.
/// * A program with no `Reg`/`DynVtype` site at all gets no checking. The
///   end-of-run report prints the sample count so that is visible rather than
///   assumed.
/// * `vl` also changes outside a `vsetvl*`, on a fault-only-first vector load
///   (`vl*ff`). That used to make this check fire on any program using one --
///   the shadow simply did not know. It is now captured (see
///   [`VCPUScoreBoard::ff_pending_pc`]), so a mismatch here
///   is once again what it says it is: the `vl` arithmetic diverging from this
///   QEMU's.
/// * A vCPU's *starting* `vl` likewise does not come from a `vsetvl*`. A new
///   guest thread inherits the spawning thread's vector state (QEMU linux-user
///   clones a vCPU with `cpu_copy()`, a `memcpy` of the whole `CPUArchState`),
///   so `on_vcpu_init` seeds `shadow_vl`/`shadow_sew_bits` from the CSRs rather
///   than zeroing them. Zeroed, this check fired on the first `R_REGS`
///   `vsetvl*` of every thread spawned while `vl` was non-zero.
#[inline]
fn check_vl_oracle(data: &mut PerVCPUData, shadow_vl: u64, pc: u64, mode: VlVerify) {
    let Some(real_vl) = data.read_vl() else {
        log::warn!("vl oracle at PC {pc:#x}: could not read the vl CSR; skipping check");
        return;
    };
    VL_ORACLE_SAMPLES.fetch_add(1, Ordering::Relaxed);
    if real_vl == shadow_vl {
        return;
    }
    VL_ORACLE_MISMATCHES.fetch_add(1, Ordering::Relaxed);
    let msg = format!(
        "vl oracle disagreed at vsetvl* PC {pc:#x}: QEMU's vl CSR reads {real_vl}, but \
         the last vl this plugin computed was {shadow_vl}. Either the vl arithmetic in \
         `vl_compute` diverges from this QEMU's `HELPER(vsetvl)`, or a `vl` change \
         this plugin never saw slipped past -- a fault-only-first vector load is \
         the only other instruction that writes `vl`, and it is captured via the \
         successor block, so a `vl*ff` whose successor block \
         was already translated when it was first seen is the case to look at. \
         Every vl in this trace is suspect."
    );
    match mode {
        VlVerify::Abort => panic!("turbo_tracer: {msg} Pass vl_verify=warn to continue anyway."),
        VlVerify::Warn => log::warn!("{msg}"),
        VlVerify::Off => unreachable!("the oracle is not called when it is off"),
    }
}

/// Publish one `vsetvl*` execution: the computed `vl` becomes this vCPU's shadow
/// (so the `x0`/`x0` form can take its AVL from it), and the instruction's
/// retirement plus a `vl` CSR data-trace packet go into the E-Trace stream.
///
/// This is where the `vsetvl*`'s retirement is now recorded. It used to be
/// recorded by the *next* block's start callback, alongside the `vl` read; both
/// move here, so stream order (retirement then `vl`) is unchanged.
#[inline(always)]
fn emit_computed_vl(
    encoders: &VCPUArray<TraceEncoder>,
    shadow: VtypeShadow,
    vcpu_idx: VCPUIndex,
    vl: u64,
    sew_bits: u16,
    form_idx: usize,
) {
    TurboTracePlugin::sb_set(shadow.vl, vcpu_idx, vl);
    // `SEW` rides along with `vl` because the two always change together, and
    // indexed memory ops need both. See `VCPUScoreBoard::shadow_sew_bits`.
    TurboTracePlugin::sb_set(shadow.sew_bits, vcpu_idx, u64::from(sew_bits));
    VSETVL_EXECS[form_idx].fetch_add(1, Ordering::Relaxed);
    // SAFETY: QEMU guarantees vcpu_idx callbacks run only on that vCPU's thread.
    if let Some(encoder) = unsafe { encoders.get_mut(vcpu_idx) } {
        encoder.record_vl(vl);
    }
}

/// The two scoreboard fields every `vsetvl*` writes and the vector memory
/// capture reads: this vCPU's shadow `vl` and `SEW`. Carried as one value
/// because nothing ever needs one without the other, and because both are
/// resolved once per translation.
#[derive(Clone, Copy)]
struct VtypeShadow {
    vl: SbField,
    sew_bits: SbField,
}

/// Read a vector memory op's base (or stride) register.
///
/// Panics rather than substituting a zero: a wrong base address is a wrong set
/// of cache lines, which is worse for a footprint trace than no trace at all.
#[inline]
fn read_base(per_vcpu_data: &VCPUArray<PerVCPUData>, vcpu_idx: VCPUIndex, reg: u8, pc: u64) -> u64 {
    // SAFETY: QEMU guarantees vcpu_idx callbacks run only on that vCPU's thread.
    unsafe { per_vcpu_data.get_mut(vcpu_idx) }
        .and_then(|data| data.read_gp_reg(reg))
        .unwrap_or_else(|| {
            panic!(
                "vector memory op at PC {pc:#x}: could not read x{reg}, so the addresses \
                 it touches are unknown"
            )
        })
}

// ---------------------------------------------------------------------------
// Raw FFI for GLib functions used by fast register reads.
// These are pub(crate) in qemu_plugin, so we declare them ourselves.
// ---------------------------------------------------------------------------
extern "C" {
    fn g_byte_array_new() -> *mut qemu_plugin::sys::GByteArray;
    fn g_byte_array_free(array: *mut qemu_plugin::sys::GByteArray, free_segment: bool) -> *mut u8;
    fn g_byte_array_set_size(
        array: *mut qemu_plugin::sys::GByteArray,
        length: u32,
    ) -> *mut qemu_plugin::sys::GByteArray;
    fn g_array_free(array: *mut qemu_plugin::sys::GArray, free_segment: bool) -> *mut u8;

}

/// RISC-V ABI register names in order x0..x31.
/// QEMU exposes GP registers under these names.
const RISCV_GP_REG_NAMES: [&str; 32] = [
    "zero", "ra", "sp", "gp", "tp", "t0", "t1", "t2", "s0", "s1", "a0", "a1", "a2", "a3", "a4",
    "a5", "a6", "a7", "s2", "s3", "s4", "s5", "s6", "s7", "s8", "s9", "s10", "s11", "t3", "t4",
    "t5", "t6",
];

/// Map an ABI register name (from QEMU disassembly) to its index 0..31.
fn abi_name_to_index(name: &str) -> Option<u8> {
    // First try exact match (e.g. "s0", "a0")
    if let Some(pos) = RISCV_GP_REG_NAMES.iter().position(|&n| n == name) {
        return Some(pos as u8);
    }
    // QEMU exposes register names as "x8/s0", "x10/a0", etc.
    // Try matching the part after '/' (the ABI name).
    if let Some((x_part, abi_part)) = name.split_once('/') {
        if let Some(pos) = RISCV_GP_REG_NAMES.iter().position(|&n| n == abi_part) {
            log::trace!(
                "abi_name_to_index: '{}' matched via ABI suffix '{}' -> index {}",
                name,
                abi_part,
                pos
            );
            return Some(pos as u8);
        }
        // Also try matching the "xN" part as a numeric index.
        if let Some(num_str) = x_part.strip_prefix('x') {
            if let Ok(idx) = num_str.parse::<u8>() {
                if idx < 32 {
                    log::trace!(
                        "abi_name_to_index: '{}' matched via xN prefix -> index {}",
                        name,
                        idx
                    );
                    return Some(idx);
                }
            }
        }
    }
    // QEMU may also use "fp" as alias for s0 (x8) in disassembly
    if name == "fp" {
        return Some(8);
    }
    None
}

/// Map a vector register name (`v0`..`v31`) to its index.
///
/// Matched exactly rather than by prefix: `vl`, `vtype`, `vlenb`, `vstart` and
/// `vxrm` all begin with `v` too, and mistaking one of those for `v0` would feed
/// the index-vector reader four bytes of CSR.
fn vector_reg_name_to_index(name: &str) -> Option<u8> {
    let idx: u8 = name.strip_prefix('v')?.parse().ok()?;
    (idx < 32).then_some(idx)
}

/// Result from scanning QEMU's register list.
struct RawRegisterHandles {
    vl_handle: Option<*mut qemu_plugin::sys::qemu_plugin_register>,
    vtype_handle: Option<*mut qemu_plugin::sys::qemu_plugin_register>,
    vlenb_value: Option<u64>,
    /// GP register handles indexed by register number (0..31).
    /// None entries mean the handle was not found.
    gp_handles: [Option<*mut qemu_plugin::sys::qemu_plugin_register>; 32],
    /// Vector register handles `v0`..`v31`, for reading the index vector of an
    /// indexed (gather/scatter) memory op. `None` when QEMU does not expose
    /// them, which makes indexed footprints uncapturable -- see
    /// `on_translation_block_translate`, which refuses to guess.
    v_handles: [Option<*mut qemu_plugin::sys::qemu_plugin_register>; 32],
}

/// Get raw register handles by name directly from the QEMU FFI.
///
/// This bypasses `RegisterDescriptor` (which is `repr(Rust)` with unpredictable
/// field layout) and reads the `handle` field from the `repr(C)`
/// `qemu_plugin_reg_descriptor` struct where field order is guaranteed.
unsafe fn get_raw_register_handles() -> RawRegisterHandles {
    use std::ffi::CStr;

    let array = qemu_plugin::sys::qemu_plugin_get_registers();
    let len = (*array).len as usize;
    let data = (*array).data as *const qemu_plugin::sys::qemu_plugin_reg_descriptor;
    let descriptors = std::slice::from_raw_parts(data, len);

    let mut result = RawRegisterHandles {
        vl_handle: None,
        vtype_handle: None,
        vlenb_value: None,
        gp_handles: [None; 32],
        v_handles: [None; 32],
    };

    for desc in descriptors {
        let name = CStr::from_ptr(desc.name).to_str().unwrap_or("");
        match name {
            "vl" => result.vl_handle = Some(desc.handle),
            "vtype" => result.vtype_handle = Some(desc.handle),
            "vlenb" => {
                // Read vlenb value now (on_vcpu_init context allows register reads).
                let buf = g_byte_array_new();
                let ret = qemu_plugin::sys::qemu_plugin_read_register(desc.handle, buf);
                if ret {
                    let len = (*buf).len as usize;
                    if len >= 8 {
                        let slice = std::slice::from_raw_parts((*buf).data, 8);
                        let mut bytes = [0u8; 8];
                        bytes.copy_from_slice(slice);
                        result.vlenb_value = Some(u64::from_le_bytes(bytes));
                    } else if len >= 4 {
                        let slice = std::slice::from_raw_parts((*buf).data, 4);
                        let mut bytes = [0u8; 4];
                        bytes.copy_from_slice(slice);
                        result.vlenb_value = Some(u32::from_le_bytes(bytes) as u64);
                    }
                }
                g_byte_array_free(buf, true);
            }
            other => {
                // Check if this is a GP register (x0..x31)
                if let Some(idx) = abi_name_to_index(other) {
                    result.gp_handles[idx as usize] = Some(desc.handle);
                } else if let Some(idx) = vector_reg_name_to_index(other) {
                    result.v_handles[idx as usize] = Some(desc.handle);
                }
            }
        }
    }

    // Free the GArray (but not the strings inside -- QEMU owns them).
    g_array_free(array, true);

    result
}

/// QEMU CPU properties that change `vsetvl*`'s `vl` arithmetic away from what
/// the RISC-V V spec prescribes, and which the plugin API gives no way to query.
///
/// * `rvv_vl_half_avl` makes the AVL-overflow case yield `(AVL+1)>>1` instead of
///   `VLMAX` (`target/riscv/vector_helper.c`).
/// * `rvv_vsetvl_x0_vill` adds a `vill` case when an `x0`/`x0` form would change
///   `vl`.
///
/// Both default `false` (`target/riscv/cpu.c`) and neither is in the ISA, so the
/// plugin computes `vl` per the spec. If a run turns one on, that computation is
/// wrong -- see [`audit_cpu_properties`].
const VL_BREAKING_CPU_PROPS: [&str; 2] = ["rvv_vl_half_avl", "rvv_vsetvl_x0_vill"];

/// Hard-fail at plugin init if the QEMU command line enables a CPU property that
/// invalidates the plugin's computed `vl` (see [`VL_BREAKING_CPU_PROPS`]).
///
/// The plugin API cannot read CPU properties, so the command line QEMU was
/// started with (`/proc/self/cmdline`, since the plugin shares QEMU's process) is
/// the only place to see them. This is belt-and-braces with the runtime
/// computed-vs-read `vl` oracle, which would also catch it; the point of doing it
/// here as well is to turn an obscure numeric divergence deep in a run into a
/// clear message before the guest executes a single instruction.
///
/// A property is only rejected when actually *enabled*: `-cpu max,rvv_vl_half_avl=false`
/// is the default behaviour and is fine.
fn audit_cpu_properties() {
    let Ok(cmdline) = std::fs::read("/proc/self/cmdline") else {
        // No procfs (unexpected on Linux, and QEMU-user only runs there). The
        // runtime oracle still covers this, so warn rather than abort.
        log::warn!("could not read /proc/self/cmdline; skipping -cpu property audit");
        return;
    };
    let args: Vec<&[u8]> = cmdline.split(|&b| b == 0).collect();
    for (i, arg) in args.iter().enumerate() {
        if arg != b"-cpu" {
            continue;
        }
        let Some(spec) = args.get(i + 1).and_then(|a| std::str::from_utf8(a).ok()) else {
            continue;
        };
        for prop in spec.split(',') {
            let (name, value) = prop.split_once('=').unwrap_or((prop, "true"));
            if !VL_BREAKING_CPU_PROPS.contains(&name.trim()) {
                continue;
            }
            if matches!(value.trim(), "false" | "off" | "no" | "0") {
                continue;
            }
            panic!(
                "turbo_tracer: -cpu property `{prop}` is enabled, and it changes how \
                 QEMU derives `vl` from a vsetvl*. The plugin computes `vl` from the \
                 instruction's operands per the RISC-V V spec and the plugin API cannot \
                 read CPU properties, so every recorded `vl` would be wrong. Drop the \
                 property, or use a QEMU that reads `vl` back on the successor block."
            );
        }
    }
}

/// Detect a ROI R-type marker directly from the instruction encoding.
/// Returns `(RoiRtMarkerKind, rs1_index, rs2_index)` if detected.
///
/// ROI markers are RV64I OP (`opcode = 0b0110011`) R-type instructions writing
/// to `x0`, so the hardware discards the result and the encoding is free to
/// carry a marker. The R-type layout is
/// `funct7[31:25] rs2[24:20] rs1[19:15] funct3[14:12] rd[11:7] opcode[6:0]`.
///
/// Decoding from the 4 encoding bytes rather than QEMU's disassembly text keeps
/// this off the string-formatting path (`insn.disas()` allocates and formats)
/// and makes the accepted set exactly the seven `(funct7, funct3)` pairs the
/// marker ABI defines -- no dependence on binutils mnemonic spelling.
fn detect_roi_from_instruction_bytes(insn_bytes: &[u8]) -> Option<(RoiRtMarkerKind, u8, u8)> {
    // Markers are always 32-bit (uncompressed) instructions. A 16-bit RVC
    // encoding cannot be an OP R-type, so a short slice is simply not a marker.
    let raw = u32::from_le_bytes(insn_bytes.get(..4)?.try_into().ok()?);

    // opcode: OP (`6..2=0x0C 1..0=3`). This also rules out RVC (`1..0 != 3`)
    // and the RV64 OP-32 forms (`addw`/`subw`/..., opcode 0x3B), which are not
    // part of the marker ABI.
    if raw & 0x7f != 0x33 {
        return None;
    }
    // rd must be x0, otherwise this is a real computation, not a marker.
    if (raw >> 7) & 0x1f != 0 {
        return None;
    }

    let funct3 = (raw >> 12) & 0x7;
    let funct7 = raw >> 25;
    let kind = match (funct7, funct3) {
        (0, 0b111) => RoiRtMarkerKind::And,
        (0, 0b110) => RoiRtMarkerKind::Or,
        (0, 0b001) => RoiRtMarkerKind::Sll,
        (0, 0b101) => RoiRtMarkerKind::Srl,
        (0, 0b000) => RoiRtMarkerKind::Add,
        (32, 0b000) => RoiRtMarkerKind::Sub,
        (0, 0b100) => RoiRtMarkerKind::Xor,
        // Everything else with rd=x0: `slt`/`sltu`/`sra`, the M extension
        // (funct7=1), and any other funct7. Not marker encodings.
        _ => return None,
    };

    let rs1 = ((raw >> 15) & 0x1f) as u8;
    let rs2 = ((raw >> 20) & 0x1f) as u8;
    Some((kind, rs1, rs2))
}

// RISC-V (asm-generic) syscall numbers used for shared-object mapping discovery.
// RISC-V has no `open`/`mmap2`; the loader uses `openat` + `mmap`.
const SYS_OPENAT: i64 = 56;
const SYS_CLOSE: i64 = 57;
const SYS_MMAP: i64 = 222;

// ---------------------------------------------------------------------------
// Lock-free per-vCPU storage
// ---------------------------------------------------------------------------

/// Maximum number of vCPUs supported. QEMU typically caps at 256 for TCG.
const MAX_VCPUS: usize = 256;

/// Sanity cap on the vDSO image size derived from the program headers read out
/// of guest memory. Used by `TurboTracePlugin::write_vdso_dump`. Linux's RISC-V
/// vDSO is two pages; 1 MiB leaves several orders of magnitude of headroom while
/// keeping a bogus value from turning into a huge reservation.
const MAX_VDSO_BYTES: u64 = 1 << 20;

/// Guest page size. RISC-V (and every other 64-bit target QEMU-user builds this
/// plugin for) uses 4 KiB. The vDSO is mapped page-aligned, which
/// [`vdso_image_size_at`] checks, and it doubles as the chunk size for guest
/// reads, which `qemu_plugin_read_memory_vaddr` caps per call.
const GUEST_PAGE_SIZE: u64 = 4096;

// ---------------------------------------------------------------------------
// vDSO discovery
// ---------------------------------------------------------------------------
//
// The vDSO is the one loaded object nothing outside QEMU can point at: QEMU's
// ELF loader injects it before the guest runs, so no guest syscall observes it
// being mapped, and the image exists only in guest memory -- there is no file on
// disk to name. Yet the guest executes code in it (`__vdso_clock_gettime` and
// friends), and the consumer treats a PC with no module as a fatal
// `AddressNotCovered`, so a guest that calls `gettimeofday` cannot be profiled
// at all unless the plugin hands the consumer this image.
//
// It is found from the guest's own auxiliary vector. QEMU-user's loader puts the
// vDSO base there itself (`NEW_AUX_ENT(AT_SYSINFO_EHDR, vdso_info->load_addr)`
// in `linux-user/elfload.c`), so this is not a heuristic -- it is the same fact
// the loader recorded, read through the frozen SysV process-startup ABI rather
// than through QEMU internals. That matters: the alternative was a private QEMU
// patch exporting `qemu_plugin_get_vdso_info`, which made the plugin fail to
// `dlopen` on any unpatched QEMU, needed carrying across every rebase, and took
// its answer on trust. This route needs no patch and *validates* what it finds
// (see [`vdso_image_size_at`]) before believing it.

/// `AT_SYSINFO_EHDR`: the auxiliary-vector tag whose value is the vDSO's load
/// address.
const AT_SYSINFO_EHDR: u64 = 33;

/// `AT_NULL`: terminates the auxiliary vector.
const AT_NULL: u64 = 0;

/// Upper bound on the words walked while looking for the auxv on the initial
/// stack. The layout is `argc, argv[], NULL, envp[], NULL, auxv[]`, so the
/// distance to the auxv is set by the guest's argument and environment count --
/// a couple of hundred words in practice. The cap exists so a stack that does
/// not have this shape (a guest QEMU set up differently, a bad `sp` read) ends
/// the walk instead of reading guest memory forever.
const AUXV_WALK_MAX_WORDS: u64 = 8192;

/// Read `len` bytes of guest memory at `addr` without touching per-vCPU state.
///
/// Deliberately not [`PerVCPUData::read_guest_bytes`], which is the same read
/// against a preallocated per-vCPU buffer: the vDSO paths run from
/// `on_vcpu_init`, *before* that `PerVCPUData` (and so that buffer) exists for
/// this vCPU. `qemu_plugin_read_memory_vaddr` implicitly uses QEMU's
/// `current_cpu`, which is set there, so a private `GByteArray` is all that is
/// missing.
///
/// All-or-nothing and capped per call by QEMU, so `len` must be <= one page;
/// callers wanting more (the dump) loop.
fn read_guest_exact(addr: u64, len: u64) -> Option<Vec<u8>> {
    if addr == 0 || len == 0 || len > GUEST_PAGE_SIZE {
        return None;
    }
    // SAFETY: the GByteArray is allocated and freed here and nothing else can
    // observe it; `current_cpu` is valid in every context this is called from.
    unsafe {
        let buf = g_byte_array_new();
        (*buf).len = 0;
        let ok = qemu_plugin::sys::qemu_plugin_read_memory_vaddr(addr, buf, len as usize);
        let got = (*buf).len as u64;
        let out = if ok && got == len {
            Some(std::slice::from_raw_parts((*buf).data, got as usize).to_vec())
        } else {
            None
        };
        g_byte_array_free(buf, true);
        out
    }
}

/// One little-endian 64-bit word of guest memory.
fn read_guest_u64(addr: u64) -> Option<u64> {
    let b = read_guest_exact(addr, 8)?;
    Some(u64::from_le_bytes(b.try_into().ok()?))
}

/// Read a register through its raw handle with a private `GByteArray`, for the
/// same reason as [`read_guest_exact`]: `on_vcpu_init` needs `sp` before it has
/// built the `PerVCPUData` whose `read_register_u64` owns the reusable buffer.
/// That one stays as it is -- it is the per-instruction hot path for vector ops
/// and must not allocate.
///
/// # Safety
/// `handle` must be a live handle from `qemu_plugin_get_registers`, read from
/// the vCPU thread it belongs to.
unsafe fn read_register_u64_raw(
    handle: *mut qemu_plugin::sys::qemu_plugin_register,
) -> Option<u64> {
    let buf = g_byte_array_new();
    (*buf).len = 0;
    let ok = qemu_plugin::sys::qemu_plugin_read_register(handle, buf);
    let len = (*buf).len as usize;
    let out = if ok && len >= 8 {
        let mut bytes = [0u8; 8];
        bytes.copy_from_slice(std::slice::from_raw_parts((*buf).data, 8));
        Some(u64::from_le_bytes(bytes))
    } else {
        None
    };
    g_byte_array_free(buf, true);
    out
}

/// Confirm that guest memory at `base` holds a **relocated** RISC-V ELF image
/// and return the size of the region it spans, or `None`.
///
/// This is both where the size comes from and the check that the auxv entry
/// meant what it said. Every other ELF in the address space -- the main binary,
/// `ld.so`, each `.so` -- has an ELF header in memory too, but it is a verbatim
/// copy of the file's, so its `PT_LOAD` `p_vaddr`s are link-time (near zero) and
/// do not equal the address the image is mapped at. The vDSO is the exception:
/// QEMU relocates the image in place when it loads it (`load_elf_vdso` walks a
/// pre-computed relocation list), so its program headers carry *runtime*
/// addresses. `p_vaddr == base` for the executable segment is therefore a
/// property only the vDSO has, and it is the property the consumer relies on
/// anyway -- see the "pre-relocated ELF" handling in `InstructionDecoder::new`.
///
/// The returned size is the span from `base` to the end of the highest
/// `PT_LOAD`, which for QEMU's RISC-V vDSO is 4104 bytes.
fn vdso_image_size_at(base: u64) -> Option<u64> {
    // ELF64 header fields this needs, by offset: e_type(16,u16),
    // e_machine(18,u16), e_phoff(32,u64), e_phentsize(54,u16), e_phnum(56,u16).
    const EHDR_LEN: u64 = 64;
    const PHDR_LEN: u64 = 56;

    if !base.is_multiple_of(GUEST_PAGE_SIZE) {
        // Every ELF image QEMU maps starts on a page boundary, so an
        // AT_SYSINFO_EHDR that is not page-aligned did not come from the ELF
        // loader, and reading a "header" from it would be reading the middle of
        // some mapping.
        return None;
    }
    let ehdr = read_guest_exact(base, EHDR_LEN)?;
    let u16_at = |off: usize| u16::from_le_bytes([ehdr[off], ehdr[off + 1]]);
    let u64_at = |off: usize| u64::from_le_bytes(ehdr[off..off + 8].try_into().unwrap());

    if &ehdr[..4] != b"\x7fELF"
        || ehdr[4] != 2 // ELFCLASS64
        || ehdr[5] != 1 // ELFDATA2LSB
        || u16_at(16) != 3 // ET_DYN: the vDSO is a shared object
        || u16_at(18) != 243
    // EM_RISCV
    {
        return None;
    }

    let phentsize = u64::from(u16_at(54));
    let phnum = u64::from(u16_at(56));
    if phentsize != PHDR_LEN || phnum == 0 || phnum * phentsize > GUEST_PAGE_SIZE {
        return None;
    }
    let phdrs = read_guest_exact(base.checked_add(u64_at(32))?, phnum * phentsize)?;

    let mut end = 0u64;
    let mut saw_relocated_exec = false;
    for i in 0..phnum as usize {
        let ph = &phdrs[i * PHDR_LEN as usize..][..PHDR_LEN as usize];
        let word = |off: usize| u64::from_le_bytes(ph[off..off + 8].try_into().unwrap());
        let p_type = u32::from_le_bytes(ph[0..4].try_into().unwrap());
        let p_flags = u32::from_le_bytes(ph[4..8].try_into().unwrap());
        if p_type != 1 {
            // PT_LOAD
            continue;
        }
        let (p_vaddr, p_memsz) = (word(16), word(40));
        if p_vaddr < base {
            // A link-time vaddr, i.e. an ordinary un-relocated ELF header: this
            // is some other module's, not the vDSO's.
            return None;
        }
        if p_flags & 1 != 0 && p_vaddr == base {
            // PF_X at exactly the mapped base: the relocated vDSO.
            saw_relocated_exec = true;
        }
        end = end.max(p_vaddr.checked_add(p_memsz)?);
    }

    if !saw_relocated_exec {
        return None;
    }
    let size = end.checked_sub(base)?;
    (size > 0 && size <= MAX_VDSO_BYTES).then_some(size)
}

/// The vDSO base from the guest's own auxiliary vector, given a handle for the
/// stack pointer.
///
/// Runs from the first `on_vcpu_init`, i.e. before the guest has executed an
/// instruction, so `sp` still points at the initial stack QEMU built:
/// `argc`, `argv[argc]`, `NULL`, `envp[]`, `NULL`, then `(a_type, a_val)` pairs
/// ending at `AT_NULL`. Word at a time -- the whole structure is a couple of
/// hundred words, walked once per run, and reading it in chunks would need
/// page-crossing logic for no measurable gain.
///
/// Doing this here, rather than on the first vDSO PC the guest executes, is
/// what makes it usable: the live-trace consumer freezes its memory map as soon
/// as the layout looks complete, which is long before a `clock_gettime` call.
/// Discovered at first vCPU init, the vDSO is already in the very first
/// `turbo_metadata.json` write, so every consumer sees it.
fn vdso_base_from_auxv(sp: u64) -> Option<u64> {
    let argc = read_guest_u64(sp)?;
    // argc is a count of pointers on the initial stack; anything huge means
    // this is not the initial stack (or `sp` was misread).
    if argc > AUXV_WALK_MAX_WORDS {
        log::debug!("vdso: auxv walk: implausible argc {argc} at sp {sp:#x}");
        return None;
    }

    // Past argc, argv and its NULL terminator; then skip envp's pointers up to
    // its own NULL. `words` counts from `sp` so one cap covers the whole walk.
    let mut words = 1 + argc + 1;
    loop {
        if words >= AUXV_WALK_MAX_WORDS {
            log::debug!("vdso: auxv walk: no end of envp within {AUXV_WALK_MAX_WORDS} words");
            return None;
        }
        if read_guest_u64(sp + words * 8)? == 0 {
            words += 1;
            break;
        }
        words += 1;
    }

    // The auxv proper: (tag, value) pairs.
    while words + 1 < AUXV_WALK_MAX_WORDS {
        let tag = read_guest_u64(sp + words * 8)?;
        let val = read_guest_u64(sp + (words + 1) * 8)?;
        if tag == AT_NULL {
            log::debug!("vdso: auxv has no AT_SYSINFO_EHDR, so this guest has no vDSO");
            return None;
        }
        if tag == AT_SYSINFO_EHDR {
            return (val != 0).then_some(val);
        }
        words += 2;
    }
    log::debug!("vdso: auxv walk: no AT_NULL within {AUXV_WALK_MAX_WORDS} words");
    None
}

/// Lock-free per-vCPU storage array.
///
/// QEMU guarantees that callbacks tagged with a `VCPUIndex` execute exclusively
/// on that vCPU's thread. This means we can safely hand out `&mut T` for a given
/// vCPU index without any locking, as long as each vCPU only touches its own slot.
///
/// The `on_exit` callback runs after all vCPU threads have stopped, so iterating
/// all slots there is also safe.
struct VCPUArray<T> {
    slots: Box<[UnsafeCell<Option<T>>]>,
}

// SAFETY: VCPUArray is safe to send/share across threads because QEMU's threading
// model guarantees that each VCPUIndex slot is only accessed by its owning thread.
// The UnsafeCell is indexed by VCPUIndex, preventing cross-slot aliasing.
unsafe impl<T: Send> Send for VCPUArray<T> {}
unsafe impl<T: Send> Sync for VCPUArray<T> {}

impl<T> VCPUArray<T> {
    fn new() -> Self {
        // nosemgrep: dos-unbounded-memory-allocation -- MAX_VCPUS is a compile-time const
        let mut slots = Vec::with_capacity(MAX_VCPUS);
        for _ in 0..MAX_VCPUS {
            slots.push(UnsafeCell::new(None));
        }
        Self {
            slots: slots.into_boxed_slice(),
        }
    }

    /// Get a shared reference to the value for `idx`.
    ///
    /// # Safety
    ///
    /// Caller must ensure no mutable reference exists for this slot.
    /// QEMU guarantees this for vCPU-pinned callbacks (each vCPU index is
    /// only accessed by its own thread).
    #[allow(dead_code)]
    #[inline(always)]
    unsafe fn get(&self, idx: VCPUIndex) -> Option<&T> {
        debug_assert!((idx as usize) < self.slots.len(), "VCPUIndex out of bounds");
        (*self.slots[idx as usize].get()).as_ref()
    }

    /// Get a mutable reference to the value for `idx`.
    ///
    /// # Safety
    ///
    /// Caller must ensure exclusive access for this slot.
    /// QEMU guarantees this for vCPU-pinned callbacks.
    #[allow(clippy::mut_from_ref)]
    #[inline(always)]
    unsafe fn get_mut(&self, idx: VCPUIndex) -> Option<&mut T> {
        debug_assert!((idx as usize) < self.slots.len(), "VCPUIndex out of bounds");
        (*self.slots[idx as usize].get()).as_mut()
    }

    /// Insert a value at `idx`.
    ///
    /// # Safety
    ///
    /// Caller must ensure exclusive access for this slot.
    #[inline(always)]
    unsafe fn insert(&self, idx: VCPUIndex, value: T) {
        debug_assert!((idx as usize) < self.slots.len(), "VCPUIndex out of bounds");
        *self.slots[idx as usize].get() = Some(value);
    }

    /// Iterate all populated slots, yielding `(VCPUIndex, &mut T)`.
    ///
    /// # Safety
    ///
    /// Caller must ensure no other references to any slot exist.
    /// Safe to call from `on_exit` after all vCPU threads have stopped.
    unsafe fn iter_mut(&self) -> impl Iterator<Item = (VCPUIndex, &mut T)> {
        self.slots
            .iter()
            .enumerate()
            .filter_map(|(i, cell)| (*cell.get()).as_mut().map(|v| (i as VCPUIndex, v)))
    }
}

// ---------------------------------------------------------------------------
// Scoreboard: per-vCPU state updated via fast inline stores
// ---------------------------------------------------------------------------

/// Per-vCPU execution state stored in QEMU's scoreboard.
///
/// All fields are u64 because QEMU's inline store API operates on u64 values.
/// Fields are updated via `QEMU_PLUGIN_INLINE_STORE_U64` (no callback overhead)
/// and read in the small number of Rust callbacks that need them.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
struct VCPUScoreBoard {
    /// This vCPU's shadow copy of the `vl` CSR: the value the most recent
    /// instruction to write `vl` on this vCPU produced.
    ///
    /// Needed because the `rs1 == x0, rd == x0` form of `vsetvli` takes its AVL
    /// from the *current* `vl`, and `vl` is a TCG global -- so reading it back
    /// from QEMU would need `QEMU_PLUGIN_CB_R_REGS` on that site, which is
    /// exactly the cost this design exists to remove. Every `vsetvl*` writes
    /// this field, and so does the `vl*ff` successor callback, so it is
    /// self-consistent by construction; the only way it can desync is a `vl`
    /// change the plugin never saw, which the computed-vs-read oracle on the
    /// `R_REGS` sites catches (see [`check_vl_oracle`]).
    ///
    /// Its *starting* value is read from the CSR in `on_vcpu_init` rather than
    /// assumed to be zero: a new vCPU is a new guest thread, and QEMU
    /// linux-user clones one with `cpu_copy()`, so it inherits the spawning
    /// thread's `vl` and `vtype`.
    ///
    /// It lives in the scoreboard rather than `PerVCPUData` so a `NO_REGS`
    /// callback can reach it with a single pointer load.
    shadow_vl: u64,

    /// This vCPU's shadow of `SEW`, in **bits**, written by the same `vsetvl*`
    /// callbacks that write [`VCPUScoreBoard::shadow_vl`].
    ///
    /// Needed only by indexed vector memory ops: their `width` field encodes the
    /// *index* element width, so the width of the data each index reaches is
    /// `SEW` (V spec 7.6). Every other addressing mode carries its data width in
    /// the encoding, and the consumer derives the footprint offline.
    shadow_sew_bits: u64,

    /// PC of a fault-only-first vector load that has just executed on this vCPU
    /// and whose resulting `vl` has not been read yet; 0 when none is pending.
    ///
    /// A `vl*ff` trims `vl` to the index of the first element that faults, so
    /// its `vl` exists only *after* it has run -- and it ends its TB, so the
    /// first chance to read the CSR is the successor block. The `vl*ff`'s own
    /// callback stores its PC here (`NO_REGS`, one store), and the successor
    /// block's `R_REGS` callback finds it, reads `vl`, emits the record and
    /// clears the field. Carrying the PC rather than a bare flag makes the
    /// diagnostics name the instruction that owes the value.
    ///
    /// Only the immediate successor clears it, so a `vl*ff` that faults on its
    /// *first* element (a real SIGSEGV, which is architecturally not a trim at
    /// all: QEMU leaves `vl` alone and raises) leaves the mark set until that
    /// successor block is next entered. [`VLEFF_LOST`] counts those.
    ff_pending_pc: u64,
}

// ---------------------------------------------------------------------------
// Per-vCPU non-scoreboard state
// ---------------------------------------------------------------------------

/// Non-primitive per-vCPU state that cannot live in the scoreboard.
/// Uses raw FFI handles and a pre-allocated GByteArray to avoid
/// alloc/free overhead on every register read (~13% CPU saving).
struct PerVCPUData {
    /// Raw handle for VL CSR register (from repr(C) qemu_plugin_reg_descriptor).
    vl_handle: *mut qemu_plugin::sys::qemu_plugin_register,
    /// Raw handle for VTYPE CSR register (from repr(C) qemu_plugin_reg_descriptor).
    vtype_handle: *mut qemu_plugin::sys::qemu_plugin_register,
    /// Raw handles for GP registers x0..x31 (for ROI marker register reads).
    gp_handles: [Option<*mut qemu_plugin::sys::qemu_plugin_register>; 32],
    /// Raw handles for vector registers v0..v31 (for indexed memory ops).
    v_handles: [Option<*mut qemu_plugin::sys::qemu_plugin_register>; 32],
    /// Pre-allocated GByteArray reused for all register reads (avoids alloc/free per read).
    read_buf: *mut qemu_plugin::sys::GByteArray,
    /// Index elements of an indexed memory op, scratch across callbacks so the
    /// hot path allocates nothing.
    index_buf: Vec<u8>,
    /// Cache lines an indexed memory op touched, likewise scratch. Kept sorted
    /// and deduplicated by the expansion, so it is also the emitted order.
    line_buf: Vec<u64>,
}

// SAFETY: PerVCPUData is only accessed by its owning vCPU thread (via VCPUArray).
unsafe impl Send for PerVCPUData {}

impl Drop for PerVCPUData {
    fn drop(&mut self) {
        // SAFETY: allocated in new() just below, and free-ed here.
        unsafe {
            g_byte_array_free(self.read_buf, true);
        }
    }
}

impl PerVCPUData {
    fn new(
        vl_handle: *mut qemu_plugin::sys::qemu_plugin_register,
        vtype_handle: *mut qemu_plugin::sys::qemu_plugin_register,
        gp_handles: [Option<*mut qemu_plugin::sys::qemu_plugin_register>; 32],
        v_handles: [Option<*mut qemu_plugin::sys::qemu_plugin_register>; 32],
    ) -> Self {
        // SAFETY: Allocated here in new, and free in Drop
        let read_buf = unsafe {
            let read_buf = g_byte_array_new();
            // Pre-size to 8 bytes (enough for u64 CSR values).
            g_byte_array_set_size(read_buf, 8);
            read_buf
        };
        Self {
            vl_handle,
            vtype_handle,
            gp_handles,
            v_handles,
            read_buf,
            index_buf: Vec::new(),
            line_buf: Vec::new(),
        }
    }

    /// Read VL CSR directly via FFI, reusing pre-allocated buffer.
    #[inline(always)]
    fn read_vl(&mut self) -> Option<u64> {
        self.read_register_u64(self.vl_handle)
    }

    /// Read VTYPE CSR directly via FFI, reusing pre-allocated buffer.
    #[inline(always)]
    fn read_vtype(&mut self) -> Option<u64> {
        self.read_register_u64(self.vtype_handle)
    }

    /// Read a GP register (x0..x31) by index, reusing pre-allocated buffer.
    #[inline(always)]
    fn read_gp_reg(&mut self, idx: u8) -> Option<u64> {
        if idx == 0 {
            return Some(0); // x0 is always zero
        }
        let handle = self.gp_handles.get(idx as usize).copied().flatten()?;
        self.read_register_u64(handle)
    }

    /// Append the raw contents of vector register `v[idx]` to `out`.
    ///
    /// Returns `false` if the handle is missing or the read failed. Callers must
    /// not proceed on `false`: for an indexed memory op the register *is* the
    /// address list, so a failed read is a missing footprint, not a degraded one.
    fn read_vreg_into(&mut self, idx: u8, out: &mut Vec<u8>) -> bool {
        let Some(handle) = self.v_handles.get(idx as usize).copied().flatten() else {
            return false;
        };
        unsafe {
            (*self.read_buf).len = 0;
            if !qemu_plugin::sys::qemu_plugin_read_register(handle, self.read_buf) {
                return false;
            }
            let buf = &*self.read_buf;
            let len = buf.len as usize;
            if len == 0 {
                return false;
            }
            out.extend_from_slice(std::slice::from_raw_parts(buf.data, len));
        }
        true
    }

    /// Read a V2 ROI name from guest memory at `addr`.
    ///
    /// A non-negative `len` is the exact byte count from the marker ABI.  The
    /// ABI uses `-1` (`u64::MAX` here) for a NUL-terminated string, which is
    /// important for callers such as `rave_begin_region(name)` whose `name`
    /// points at a stack buffer.  The old implementation treated that sentinel
    /// as invalid, leaving `RoiRtMarker::name` empty and forcing the consumer to
    /// look for a stack address in the ELF.
    ///
    /// Names are bounded so a malformed or unterminated guest string cannot make
    /// the callback scan indefinitely.  If the full bound crosses an unmapped
    /// page, progressively smaller reads preserve the useful prefix.
    fn read_guest_string(&mut self, addr: u64, len: u64) -> Option<String> {
        /// Longest ROI name we will copy out of guest memory. Names longer than
        /// this are truncated rather than dropped -- a truncated name is still
        /// far more useful than the `roi_region_0x<pc>` fallback.
        const MAX_ROI_NAME_LEN: u64 = 128;
        // QEMU's RISC-V guest memory is normally 4 KiB paged. The plugin API
        // does not expose the target page size, so this is only a retry hint;
        // the failed-read fallback below still handles other mappings safely.
        const GUEST_PAGE_SIZE: u64 = 4096;

        if addr == 0 || len == 0 {
            return None;
        }
        let nul_terminated = len == u64::MAX;
        let requested_len = if nul_terminated {
            MAX_ROI_NAME_LEN
        } else if len > MAX_ROI_NAME_LEN {
            log::warn!(
                "read_guest_string(): ROI name at {:#x} is {} bytes, truncating to {}",
                addr,
                len,
                MAX_ROI_NAME_LEN
            );
            MAX_ROI_NAME_LEN
        } else {
            len
        };

        // qemu_plugin_read_memory_vaddr() is all-or-nothing. A NUL-terminated
        // string near the end of a mapped page may not have `MAX_ROI_NAME_LEN`
        // readable bytes even though its prefix is valid, so retry with a
        // smaller bound when the first read fails.
        let page_remaining = GUEST_PAGE_SIZE - (addr % GUEST_PAGE_SIZE);
        let mut read_len = requested_len;
        let got = unsafe {
            loop {
                (*self.read_buf).len = 0;
                let ok = qemu_plugin::sys::qemu_plugin_read_memory_vaddr(
                    addr,
                    self.read_buf,
                    read_len as usize,
                );
                if ok {
                    break (*self.read_buf).len as usize;
                }
                if !nul_terminated || read_len == 1 {
                    return None;
                }
                let crosses_page = read_len > page_remaining;
                let next_len = if crosses_page {
                    page_remaining
                } else {
                    (read_len / 2).max(1)
                };
                log::warn!(
                    "read_guest_string(): read of {read_len} byte(s) at {addr:#x} failed; \
                     retrying with {next_len} byte(s){}",
                    if crosses_page {
                        " to stay within the current 4 KiB guest page"
                    } else {
                        ""
                    }
                );
                read_len = next_len;
            }
        };

        if got == 0 {
            return None;
        }
        unsafe {
            let data = std::slice::from_raw_parts((*self.read_buf).data, got);
            let data = if nul_terminated {
                &data[..data
                    .iter()
                    .position(|&byte| byte == 0)
                    .unwrap_or(data.len())]
            } else {
                data
            };
            if data.is_empty() {
                return None;
            }
            Some(String::from_utf8_lossy(data).to_string())
        }
    }

    /// Read up to `len` bytes of guest memory at virtual address `addr`, reusing
    /// the pre-allocated GByteArray. Returns `None` if the read fails (e.g. the
    /// range crosses unmapped memory) or yields nothing. `qemu_plugin_read_memory_vaddr`
    /// is all-or-nothing, so callers that don't know the exact readable length
    /// should retry with a smaller `len`.
    fn read_guest_bytes(&mut self, addr: u64, len: u64) -> Option<Vec<u8>> {
        if addr == 0 || len == 0 || len > 4096 {
            return None;
        }
        unsafe {
            (*self.read_buf).len = 0;
            let ok =
                qemu_plugin::sys::qemu_plugin_read_memory_vaddr(addr, self.read_buf, len as usize);
            if !ok {
                return None;
            }
            let buf = &*self.read_buf;
            let got = buf.len as usize;
            if got == 0 {
                return None;
            }
            Some(std::slice::from_raw_parts(buf.data, got).to_vec())
        }
    }

    #[inline(always)]
    fn read_register_u64(
        &mut self,
        handle: *mut qemu_plugin::sys::qemu_plugin_register,
    ) -> Option<u64> {
        unsafe {
            // Reset buffer length so qemu_plugin_read_register can fill it.
            (*self.read_buf).len = 0;
            let ret = qemu_plugin::sys::qemu_plugin_read_register(handle, self.read_buf);
            if !ret {
                return None;
            }
            let buf = &*self.read_buf;
            let len = buf.len as usize;
            // Support both 4-byte (u32) and 8-byte (u64) register values.
            // Newer QEMU versions (>= 9.x) may return CSRs like VL as 4-byte values.
            if len >= 8 {
                let data = std::slice::from_raw_parts(buf.data, 8);
                let mut bytes = [0u8; 8];
                bytes.copy_from_slice(data);
                Some(u64::from_le_bytes(bytes))
            } else if len >= 4 {
                let data = std::slice::from_raw_parts(buf.data, 4);
                let mut bytes = [0u8; 4];
                bytes.copy_from_slice(data);
                Some(u32::from_le_bytes(bytes) as u64)
            } else {
                log::warn!(
                    "read_register_u64: unexpected buffer length {} (expected 4 or 8)",
                    len
                );
                None
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Main plugin struct
// ---------------------------------------------------------------------------

struct TurboTracePlugin {
    /// Per-vCPU E-Trace encoders (lock-free, indexed by VCPUIndex).
    trace_encoders: Arc<VCPUArray<TraceEncoder>>,
    /// Steps between record-log flushes.
    flush_interval: u64,
    /// Deflate level for record-log payloads; see the `compress=` plugin arg.
    compress_level: u32,
    /// Scoreboard for per-vCPU inline state (initialized once, then read-only).
    scoreboard: Arc<OnceLock<Arc<Scoreboard<'static, VCPUScoreBoard>>>>,
    /// Per-vCPU register descriptors (lock-free, indexed by VCPUIndex).
    per_vcpu_data: Arc<VCPUArray<PerVCPUData>>,
    /// VLENB hardware parameter.
    vlenb: u16,
    /// Working directory for output files.
    workdir: PathBuf,
    /// Opaque per-run identifier (see [`turbo_tracer_shared::RunId`]), stamped
    /// into every `TurboMetadata` write and every record-log header so a
    /// consumer can distinguish this run's files from stale leftovers in the
    /// same work_dir. `None` when the plugin was invoked without a `run_id=`
    /// argument, i.e. by hand rather than by `turbo run`.
    run_id: Option<turbo_tracer_shared::RunId>,
    /// Enable debug file logging.
    debug_enabled: bool,
    /// BBT block interning table: `(start_pc, contents hash) -> dense ID`.
    ///
    /// Keyed on contents as well as PC because QEMU re-translates: 13 839
    /// translations for 10 328 distinct blocks on a six-second measured workload run, so
    /// its translation cache flushed at least once and a per-translation ID would
    /// be unbounded. Hashing the contents (rather than trusting the PC alone)
    /// also keeps a self-modifying or JIT guest correct.
    ///
    /// Lives on `ControlFlow`, which the plugin crate hands out as `&mut self`
    /// from the translate callback, so no locking is needed for the table itself.
    bbt_ids: std::collections::HashMap<(u64, u64), u32>,
    /// Start PCs of every block ever interned, in `bbt_ids` insertion order-free
    /// form: `bbt_ids` is keyed on `(pc, hash)`, and the `vl*ff` handling needs a
    /// lookup by PC alone. See [`TurboTracePlugin::note_ff_successor`].
    bbt_start_pcs: std::collections::HashSet<u64>,
    /// PCs that a fault-only-first vector load falls through to, i.e. the blocks
    /// that must read the `vl` CSR on entry. Filled at translate time, one entry
    /// per `vl*ff` site; a handful of entries even on a vector-heavy guest.
    ff_succ_pcs: std::collections::HashSet<u64>,
    /// Append-only BBT dictionary byte log, shared with every hart's encoder.
    /// Behind a `Mutex` because translation runs concurrently with other vCPUs'
    /// execution callbacks, which read its tail at flush time. Taken 13 839 times
    /// per run on a measured workload (once per translation) plus once per flush window.
    bbt_dict_log: Arc<Mutex<Vec<u8>>>,
    /// Scratch buffer for hashing a block's instruction bytes at translate time,
    /// so interning does not allocate per instruction.
    bbt_hash_buf: Vec<u8>,
    /// What to do when the computed-vs-read `vl` oracle disagrees
    /// ([`check_vl_oracle`]). Plugin arg `vl_verify`.
    vl_verify: VlVerify,
    /// Whether to capture the runtime addresses of vector loads and stores, the
    /// input to the cache-footprint replay. Plugin arg `mem_capture` (default
    /// on); off removes an `R_REGS` callback per vector memory instruction.
    mem_capture: bool,
    /// Main-binary memory-map info (see [`write_meminfo`]), cached once written
    /// so later [`write_metadata`] calls (as libraries load) can include it
    /// without re-querying the QEMU plugin API. `None` until the first vCPU
    /// init.
    meminfo: Option<turbo_tracer_shared::MemMapInfo>,

    // --- Shared-object mapping discovery via guest openat/mmap syscalls ---
    // These are mutated only from on_syscall/on_syscall_return, which the crate
    // dispatches under the global plugin mutex, so plain maps (no locking) are
    // safe here. Not touched by the lock-free per-instruction hot path.
    /// Open file descriptors observed via `openat`: fd -> guest path. Cleared on
    /// `close` so a later reuse of the same fd number is attributed correctly.
    open_fds: std::collections::HashMap<i64, String>,
    /// Per-vCPU `openat` path captured at syscall entry, resolved to its fd on
    /// syscall return.
    pending_openat: std::collections::HashMap<VCPUIndex, String>,
    /// Per-vCPU `(fd, offset)` captured at `mmap` entry, resolved to a base
    /// address on syscall return.
    pending_mmap: std::collections::HashMap<VCPUIndex, (i64, u64)>,
    /// Discovered shared objects: guest path -> runtime load base.
    lib_mappings: std::collections::HashMap<String, u64>,
}

// ---------------------------------------------------------------------------
// Scoreboard field addressing
// ---------------------------------------------------------------------------

/// Cached address of the scoreboard's element array, i.e. of vCPU 0's
/// [`VCPUScoreBoard`].
///
/// `qemu_plugin_u64_get`/`_set` are the documented way to reach a scoreboard
/// field, but each call is a cross-DSO jump into QEMU that re-derives the
/// address from scratch: a `g_assert(vcpu_index < qemu_plugin_num_vcpus())`
/// (itself an out-of-line call), then `g_array_get_element_size()` through
/// glib's PLT, then the multiply. Three calls to load one word, and the TB-start
/// callback needs two loads and a store per block execution, which made that API
/// ~8% of a measured workload run.
///
/// The element size is `size_of::<VCPUScoreBoard>()` and is known here, so all
/// that is actually needed is the array base. QEMU may reallocate the array, but
/// only from `plugin_grow_scoreboards__locked`, which runs with all vCPUs
/// stopped and strictly *before* the plugin's `vcpu_init` callback for the vCPU
/// that triggered the growth (`plugins/core.c`). Refreshing this pointer at the
/// top of every `on_vcpu_init` therefore covers every point at which it can
/// move; the initial allocation is 16 elements, so a single-threaded guest never
/// grows it at all.
static SB_BASE: AtomicPtr<u8> = AtomicPtr::new(ptr::null_mut());

/// The byte offset of one `u64` field within [`VCPUScoreBoard`].
///
/// A plain offset rather than a `qemu_plugin_u64`, so it is trivially `Send` +
/// `Sync` and costs nothing to capture in a per-block closure.
#[derive(Clone, Copy)]
pub(crate) struct SbField(usize);

// ---------------------------------------------------------------------------
// VType debug helper
// ---------------------------------------------------------------------------

#[derive(Clone, Copy)]
struct VType(u64);

impl fmt::Debug for VType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let value = self.0;
        if (value >> 63) & 1 == 1 {
            return write!(f, "VType {{ illegal }}");
        }
        let sew = match (value >> 3) & 7 {
            0 => "e8",
            1 => "e16",
            2 => "e32",
            3 => "e64",
            _ => "reserved",
        };
        let lmul = match value & 7 {
            0 => "1",
            1 => "2",
            2 => "4",
            3 => "8",
            5 => "1/8",
            6 => "1/4",
            7 => "1/2",
            _ => "reserved",
        };
        let ta = if (value >> 6) & 1 == 1 {
            "agnostic"
        } else {
            "undisturbed"
        };
        let ma = if (value >> 7) & 1 == 1 {
            "agnostic"
        } else {
            "undisturbed"
        };
        write!(
            f,
            "VType {{ sew: {}, lmul: {}, tail: {}, mask: {} }}",
            sew, lmul, ta, ma
        )
    }
}

// ---------------------------------------------------------------------------
// Default / Register impls
// ---------------------------------------------------------------------------

impl Default for TurboTracePlugin {
    fn default() -> Self {
        Self {
            trace_encoders: Arc::new(VCPUArray::new()),
            flush_interval: 10_000,
            compress_level: turbo_tracer_shared::DEFAULT_COMPRESSION_LEVEL,
            scoreboard: Arc::new(OnceLock::new()),
            per_vcpu_data: Arc::new(VCPUArray::new()),
            vlenb: 0,
            workdir: PathBuf::from("."),
            run_id: None,
            debug_enabled: false,
            bbt_ids: std::collections::HashMap::new(),
            bbt_start_pcs: std::collections::HashSet::new(),
            ff_succ_pcs: std::collections::HashSet::new(),
            bbt_dict_log: Arc::new(Mutex::new(Vec::new())),
            bbt_hash_buf: Vec::new(),
            vl_verify: VlVerify::Abort,
            mem_capture: true,
            meminfo: None,
            open_fds: std::collections::HashMap::new(),
            pending_openat: std::collections::HashMap::new(),
            pending_mmap: std::collections::HashMap::new(),
            lib_mappings: std::collections::HashMap::new(),
        }
    }
}

impl Register for TurboTracePlugin {
    fn register(
        &mut self,
        _id: PluginId,
        args: &qemu_plugin::install::Args,
        _info: &qemu_plugin::install::Info,
    ) -> Result<()> {
        // The plugin is a .so loaded by QEMU, so nothing else ever initialises a
        // logger for it: without this every `log::` call in the plugin is a
        // silent no-op, including the per-flush trace-size accounting. `try_init`
        // because a second `register` (multi-plugin-instance) must not panic.
        let _ = env_logger::try_init();

        let workdir = args
            .parsed
            .get("workdir")
            .and_then(|v| {
                if let qemu_plugin::install::Value::String(s) = v {
                    Some(PathBuf::from(s))
                } else {
                    None
                }
            })
            .unwrap_or_else(|| PathBuf::from("."));

        let flush_interval = args
            .parsed
            .get("flush_interval")
            .and_then(|v| {
                if let qemu_plugin::install::Value::Integer(n) = v {
                    Some(*n as u64)
                } else {
                    None
                }
            })
            // Env fallback, as for `compress`: `turbo` builds the plugin
            // argument string itself and has no passthrough, so
            // `TURBO_TRACE_FLUSH_INTERVAL=50000 turbo run ...` is how a run
            // sweeps the window size. The plugin arg wins when both are set.
            .or_else(|| {
                std::env::var("TURBO_TRACE_FLUSH_INTERVAL")
                    .ok()
                    .and_then(|s| s.trim().parse::<u64>().ok())
                    .filter(|n| *n > 0)
            })
            .unwrap_or(10_000);

        let compress_level = args
            .parsed
            .get("compress")
            .and_then(|v| {
                if let qemu_plugin::install::Value::Integer(n) = v {
                    Some((*n).clamp(0, 9) as u32)
                } else {
                    None
                }
            })
            // Env fallback so the level can be swept without a consumer that
            // knows how to pass the plugin arg: QEMU inherits the launching
            // process's environment, so `TURBO_TRACE_COMPRESS=9 turbo run ...`
            // reaches the plugin. This is the A/B measurement path; the plugin
            // arg wins when both are set.
            .or_else(|| {
                std::env::var("TURBO_TRACE_COMPRESS")
                    .ok()
                    .and_then(|s| s.trim().parse::<u32>().ok())
                    .map(|n| n.min(9))
            })
            .unwrap_or(turbo_tracer_shared::DEFAULT_COMPRESSION_LEVEL);

        let debug_enabled = args
            .parsed
            .get("debug")
            .and_then(|v| {
                if let qemu_plugin::install::Value::Bool(b) = v {
                    Some(*b)
                } else {
                    None
                }
            })
            .unwrap_or(false);

        // Opaque per-run identifier the host generates and stamps into every
        // TurboMetadata write and record-log header, so a consumer that knows
        // which run it's waiting on can distinguish this run's files from stale
        // leftovers in the same work_dir. Absent (and so skipped by consumers)
        // when the plugin is invoked by hand rather than by `turbo run`.
        //
        // Read from `raw` rather than `parsed` because QEMU types each argument
        // by content -- bool, then `i64`, then string -- and the parsed value
        // of an id is whatever that guessing produced. `RunId::parse` is the
        // one gate: an argument that is not an id is refused here, loudly,
        // rather than half-applied downstream.
        let run_id = args
            .raw
            .iter()
            .find_map(|arg| arg.strip_prefix("run_id="))
            .and_then(|raw| {
                let id = turbo_tracer_shared::RunId::parse(raw);
                if id.is_none() {
                    eprintln!(
                        "turbo_tracer: run_id={raw:?} is not a run id ({} bytes: \"run_\" \
                         + 16 hex digits); this run's output carries no run id, so a \
                         consumer cannot tell it from a stale leftover in the same \
                         work_dir",
                        turbo_tracer_shared::RUN_ID_BYTES
                    );
                }
                id
            });

        // Vector memory address capture. On by default -- it is the only source
        // of the workload's cache footprint -- but it
        // puts an `R_REGS` callback on every vector load/store, so a run that
        // only wants instruction counts can turn it off with `mem_capture=off`.
        let mem_capture = match args.parsed.get("mem_capture") {
            // Env fallback, as for `compress`: `turbo` builds the plugin
            // argument string itself and has no passthrough, so
            // `TURBO_MEM_CAPTURE=off turbo run ...` is how a run declines the
            // capture without a code change. The plugin arg wins when both are
            // set.
            None => !matches!(
                std::env::var("TURBO_MEM_CAPTURE").as_deref().map(str::trim),
                Ok("off" | "false" | "no" | "0")
            ),
            Some(qemu_plugin::install::Value::Bool(b)) => *b,
            Some(qemu_plugin::install::Value::String(s)) => {
                !matches!(s.as_str(), "off" | "false" | "no")
            }
            Some(_) => true,
        };

        println!("turbo_tracer plugin configuration:");
        println!("  workdir: {}", workdir.display());
        println!("  flush_interval: {}", flush_interval);
        println!("  compress: deflate level {}", compress_level);
        println!("  debug: {}", debug_enabled);
        match run_id {
            Some(id) => println!("  run_id: {id}"),
            None => println!("  run_id: (none)"),
        }
        println!("  mem_capture: {}", mem_capture);

        std::fs::create_dir_all(&workdir).map_err(qemu_plugin::error::Error::IoError)?;

        // Before anything else: refuse to run under a -cpu configuration whose
        // `vl` arithmetic the plugin cannot reproduce. See VL_BREAKING_CPU_PROPS.
        audit_cpu_properties();

        // The computed-`vl` oracle is on by default: it is the gate that makes
        // computing `vl` instead of reading it safe, and it is near-free (one
        // extra register read at sites that already carry `R_REGS`).
        let vl_verify = match args.parsed.get("vl_verify") {
            None => VlVerify::Abort,
            Some(qemu_plugin::install::Value::String(s)) => match s.as_str() {
                "abort" | "on" | "true" => VlVerify::Abort,
                "warn" => VlVerify::Warn,
                "off" | "false" => VlVerify::Off,
                other => {
                    log::error!("vl_verify={other} is not one of abort|warn|off; using abort");
                    VlVerify::Abort
                }
            },
            Some(qemu_plugin::install::Value::Bool(false)) => VlVerify::Off,
            Some(_) => VlVerify::Abort,
        };
        println!("  vl_verify: {vl_verify:?}");

        self.workdir = workdir;
        self.run_id = run_id;
        self.flush_interval = flush_interval;
        self.compress_level = compress_level;
        self.debug_enabled = debug_enabled;
        self.vl_verify = vl_verify;
        self.mem_capture = mem_capture;

        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Scoreboard helpers
// ---------------------------------------------------------------------------

impl TurboTracePlugin {
    /// Initialize the scoreboard exactly once (idempotent via OnceLock).
    fn init_scoreboard(&self) {
        self.scoreboard
            .get_or_init(|| Arc::new(Scoreboard::<VCPUScoreBoard>::new()));
    }

    /// Capture the main-binary memory-map info directly from the QEMU plugin API
    /// and cache it for [`write_metadata`], which writes it (with any
    /// shared-object mappings discovered so far) to `TURBO_METADATA_FILE`. This
    /// gives consumers an authoritative main-binary path + code range + entry
    /// point without having to scrape QEMU's `-d page` stderr text, so it works
    /// in any run mode. Called once (guarded by `self.meminfo`) on the first
    /// vCPU init, by which point the guest ELF is fully loaded.
    ///
    /// `sp_handle` is the stack-pointer register handle, which is what lets the
    /// vDSO be found from the guest's auxv (see [`Self::locate_vdso`]); `None`
    /// costs exactly that.
    fn write_meminfo(&mut self, sp_handle: Option<*mut qemu_plugin::sys::qemu_plugin_register>) {
        let binary_path = match qemu_plugin::qemu_plugin_path_to_binary() {
            Ok(Some(p)) => p.display().to_string(),
            other => {
                log::warn!("meminfo: qemu_plugin_path_to_binary unavailable ({other:?})");
                return;
            }
        };
        let vdso = Self::locate_vdso(sp_handle);
        let (vdso_addr, vdso_size) = vdso.unwrap_or((0, 0));
        let info = turbo_tracer_shared::MemMapInfo {
            binary_path,
            start_code: qemu_plugin::qemu_plugin_start_code().unwrap_or(0),
            end_code: qemu_plugin::qemu_plugin_end_code().unwrap_or(0),
            entry_code: qemu_plugin::qemu_plugin_entry_code().unwrap_or(0),
            vdso_addr,
            vdso_size,
        };
        log::info!("meminfo: captured {info:?}");
        self.meminfo = Some(info);

        // The dump goes out BEFORE the metadata that names it, so a consumer
        // polling these files never reads a non-zero `vdso_addr` whose image is
        // not yet on disk.
        if let Some((addr, size)) = vdso {
            self.write_vdso_dump(addr, size);
        }
        self.write_metadata();
    }

    /// Locate the guest vDSO: its base from the auxv, its size from the image's
    /// own program headers (which is also what confirms the base).
    ///
    /// Both halves are needed and neither is a guess -- see the "vDSO discovery"
    /// commentary above. `None` means this guest has no vDSO (a legitimate
    /// outcome: `AT_SYSINFO_EHDR` is absent when QEMU mapped none) or that the
    /// initial stack could not be walked, which is reported at `warn` since
    /// every `clock_gettime` the guest then makes is an uncoverable PC.
    fn locate_vdso(
        sp_handle: Option<*mut qemu_plugin::sys::qemu_plugin_register>,
    ) -> Option<(u64, u64)> {
        // `sp` is read straight from the handle rather than through
        // `PerVCPUData`, which `on_vcpu_init` has not built yet at this point.
        let Some(sp) = sp_handle.and_then(|h| unsafe { read_register_u64_raw(h) }) else {
            log::warn!("vdso: no readable `sp`, so the guest's auxv cannot be walked");
            return None;
        };
        let addr = vdso_base_from_auxv(sp)?;
        let Some(size) = vdso_image_size_at(addr) else {
            log::warn!(
                "vdso: AT_SYSINFO_EHDR says {addr:#x}, but no relocated RISC-V ELF image is \
                 mapped there; ignoring it"
            );
            return None;
        };
        log::info!("vdso: located at {addr:#x} (size {size:#x}) via the guest's AT_SYSINFO_EHDR");
        Some((addr, size))
    }

    /// Dump the guest vDSO region `[addr, addr+size)` to the working directory.
    /// The in-memory vDSO is a full ELF image, so writing its bytes verbatim
    /// gives the consumer a loadable object.
    ///
    /// Reads guest memory with a private `GByteArray` (rather than the per-vCPU
    /// read buffer) so it does not depend on per-vCPU state being initialised —
    /// this runs during `on_vcpu_init` before that state exists, as
    /// [`read_guest_exact`] explains. `qemu_plugin_read_memory_vaddr` is
    /// all-or-nothing and capped per call, so read a page at a time.
    fn write_vdso_dump(&self, addr: u64, size: u64) {
        // `size` is bounded by `vdso_image_size_at` already; re-checked here
        // because this is the one length on the path that comes from guest
        // memory rather than from something already resident, and reserving a
        // bogus one verbatim would abort the guest's own process on the
        // allocation.
        if size > MAX_VDSO_BYTES {
            log::warn!(
                "vdso dump: implausible size of {size} bytes \
                 (cap {MAX_VDSO_BYTES}); skipping the dump"
            );
            return;
        }
        // nosemgrep: dos-unbounded-memory-allocation -- size is bounded by the MAX_VDSO_BYTES check above
        let mut bytes: Vec<u8> = Vec::with_capacity(size as usize);
        // SAFETY: GByteArray is allocated/freed here; qemu_plugin_read_memory_vaddr
        // appends into it and current_cpu is valid during on_vcpu_init.
        unsafe {
            let buf = g_byte_array_new();
            let mut off = 0u64;
            let mut ok_all = true;
            while off < size {
                let chunk = (size - off).min(GUEST_PAGE_SIZE);
                (*buf).len = 0;
                let ok = qemu_plugin::sys::qemu_plugin_read_memory_vaddr(
                    addr + off,
                    buf,
                    chunk as usize,
                );
                let got = (*buf).len as u64;
                if !ok || got == 0 {
                    log::warn!("vdso dump: guest read failed at 0x{:x}", addr + off);
                    ok_all = false;
                    break;
                }
                bytes.extend_from_slice(std::slice::from_raw_parts((*buf).data, got as usize));
                off += got;
                if got < chunk {
                    break;
                }
            }
            g_byte_array_free(buf, true);
            if !ok_all {
                return;
            }
        }
        // nosemgrep: input-validation-path-traversal-2 -- plugin workdir joined with a const file name
        let path = self.workdir.join(turbo_tracer_shared::VDSO_DUMP_FILE);
        // Written via a temporary and renamed, like `TurboMetadata`: the
        // live-trace consumer polls this directory while QEMU runs, and a
        // half-written ELF would not parse -- which it reports as the vDSO
        // being unavailable rather than as a torn read.
        let tmp = path.with_extension("tmp");
        let written = std::fs::write(&tmp, &bytes).and_then(|()| std::fs::rename(&tmp, &path));
        match written {
            Ok(()) => log::info!(
                "vdso dump: wrote {} bytes to {}",
                bytes.len(),
                path.display()
            ),
            Err(e) => log::warn!("vdso dump: write to {} failed: {e}", path.display()),
        }
    }

    /// Read a NUL-terminated guest C string at `ptr` on vCPU `vcpu`. The exact
    /// readable length is unknown, so try progressively smaller reads (a long
    /// path near the end of a mapping would fail an over-long all-or-nothing
    /// read); truncate at the first NUL.
    fn read_guest_cstr(&self, vcpu: VCPUIndex, ptr: u64) -> Option<String> {
        if ptr == 0 {
            return None;
        }
        let pvd = self.per_vcpu_data.clone();
        // SAFETY: on_syscall runs on this vCPU's own thread.
        let d = unsafe { pvd.get_mut(vcpu) }?;
        for &len in &[256u64, 128, 64, 32] {
            if let Some(bytes) = d.read_guest_bytes(ptr, len) {
                let end = bytes.iter().position(|&b| b == 0).unwrap_or(bytes.len());
                return Some(String::from_utf8_lossy(&bytes[..end]).to_string());
            }
        }
        None
    }

    /// True if guest memory at `addr` on vCPU `vcpu` starts with the ELF magic.
    /// Used to keep only real ELF images (filtering out data-file `mmap`s).
    fn guest_is_elf(&self, vcpu: VCPUIndex, addr: u64) -> bool {
        let pvd = self.per_vcpu_data.clone();
        // SAFETY: on_syscall runs on this vCPU's own thread.
        let Some(d) = (unsafe { pvd.get_mut(vcpu) }) else {
            return false;
        };
        matches!(d.read_guest_bytes(addr, 4).as_deref(), Some(b"\x7fELF"))
    }

    /// Write the cached meminfo plus the discovered shared-object mappings to
    /// the working directory as the single `TURBO_METADATA_FILE`. Called once
    /// meminfo is captured, then again each time a new library is discovered.
    fn write_metadata(&self) {
        let libs: Vec<turbo_tracer_shared::LibMapping> = self
            .lib_mappings
            .iter()
            .map(|(path, &base_address)| turbo_tracer_shared::LibMapping {
                path: path.clone(),
                base_address,
            })
            .collect();
        let metadata = turbo_tracer_shared::TurboMetadata {
            run_id: self.run_id,
            // VLEN as QEMU reports it via the `vlenb` CSR, in bits. 0 if the
            // register was not readable (no vector support / handle missing).
            // Widened before scaling: `vlenb` is a `u16` and the RVV maximum
            // VLEN of 65536 bits is 8192 of them, so a `u16` product saturates
            // at 65535 and reports a VLEN that is not a power of two.
            vlen_bits: u32::from(self.vlenb) * 8,
            meminfo: self.meminfo.clone(),
            libs,
        };
        match metadata.write_to_workdir(&self.workdir) {
            Ok(()) => log::info!(
                "metadata: wrote meminfo + {} shared-object mapping(s) to {}",
                metadata.libs.len(),
                self.workdir.display()
            ),
            Err(e) => log::warn!(
                "metadata: failed to write to {}: {e}",
                self.workdir.display()
            ),
        }
    }

    /// Note that the block starting at `succ_pc` must read the `vl` CSR on entry,
    /// because the `vl*ff` at `ff_pc` falls through to it.
    ///
    /// The hazard this checks for is translation order. `succ_pc`'s callback can
    /// only be registered while *it* is being translated, so if QEMU translated
    /// it before it ever translated `ff_pc`, the callback is missing and there is
    /// no way to add it (the plugin API cannot invalidate a TB). Every execution
    /// of that `vl*ff` would then owe a `vl` record the stream never carries, and
    /// the consumer -- which consumes one per `vl*ff` it walks past -- would
    /// mis-attribute every `vl` after the first one.
    ///
    /// In practice it cannot happen: `succ_pc` is `ff_pc + 4`, so reaching it
    /// without executing the `vl*ff` needs a branch that targets the instruction
    /// straight after a fault-only-first load. It is checked rather than assumed
    /// because the failure is silent and total, and it is checked *once* per site
    /// (the `ff_succ_pcs` insert returning false means this was already handled,
    /// which is also what makes a QEMU `tb_flush` and re-translate benign).
    fn note_ff_successor(&mut self, ff_pc: u64, succ_pc: u64) {
        if !self.ff_succ_pcs.insert(succ_pc) {
            return;
        }
        if !self.bbt_start_pcs.contains(&succ_pc) {
            return;
        }
        let msg = format!(
            "the fault-only-first load at PC {ff_pc:#x} falls through to {succ_pc:#x}, but \
             that block was already translated -- so it carries no callback to read the \
             `vl` this load writes, and none can be added now. Every execution of this \
             load will owe the trace a `vl` record it does not carry, which shifts every \
             later `vl` the consumer attributes."
        );
        match self.vl_verify {
            VlVerify::Abort => {
                panic!("turbo_tracer: {msg} Pass vl_verify=warn to continue anyway.")
            }
            VlVerify::Warn | VlVerify::Off => log::warn!("{msg}"),
        }
    }

    /// Intern a translated block for BBT and return its dense dictionary ID.
    ///
    /// A newly-interned block's descriptor is appended to the shared dictionary
    /// log here, at translate time -- strictly before any record can name it, so
    /// a live-tailing consumer can always resolve an ID the moment it sees one.
    fn intern_block(&mut self, start_pc: u64, instructions: &[qemu_plugin::Instruction]) -> u32 {
        // Hash the block's instruction bytes, and collect their sizes for the
        // descriptor. One reused scratch buffer, so this does not allocate per
        // instruction (13 839 translations x ~9.5 instructions on a measured workload).
        self.bbt_hash_buf.clear();
        // nosemgrep: dos-unbounded-memory-allocation -- len of a QEMU-owned slice already in memory
        let mut insn_sizes: Vec<u8> = Vec::with_capacity(instructions.len());
        let mut expect_pc = start_pc;
        for insn in instructions {
            let size = insn.size();
            assert_eq!(
                insn.vaddr(),
                expect_pc,
                "block at {start_pc:#x} is not contiguous: expected an instruction at \
                 {expect_pc:#x}, found one at {:#x}. The BBT dictionary reconstructs a \
                 block's PCs as start_pc plus a running sum of instruction sizes, so a \
                 gap would silently produce wrong PCs.",
                insn.vaddr(),
            );
            expect_pc += size as u64;
            let base = self.bbt_hash_buf.len();
            self.bbt_hash_buf.resize(base + size, 0);
            let got = insn.read_data(&mut self.bbt_hash_buf[base..]);
            debug_assert_eq!(got, size, "short instruction read at translate time");
            insn_sizes.push(
                u8::try_from(size).unwrap_or_else(|_| {
                    panic!("instruction at block {start_pc:#x} is {size} bytes, too wide for the BBT dictionary's per-instruction size byte")
                }),
            );
        }
        let hash = turbo_tracer_shared::fnv1a_hash_64(&self.bbt_hash_buf);

        if let Some(&id) = self.bbt_ids.get(&(start_pc, hash)) {
            return id;
        }
        let id = u32::try_from(self.bbt_ids.len()).ok().filter(|&id| id <= BBT_MAX_BLOCK_ID)
            .unwrap_or_else(|| {
                panic!(
                    "BBT dictionary exceeded {BBT_MAX_BLOCK_ID} blocks. A guest that keeps                      generating new code (self-modifying or JIT) grows this without bound                      by design; it needs the SYNC-and-restart-dictionary handling the                      format reserves space for."
                )
            });
        self.bbt_ids.insert((start_pc, hash), id);
        self.bbt_start_pcs.insert(start_pc);
        {
            let mut log = self.bbt_dict_log.lock().unwrap();
            encode_block_desc(&mut log, id, start_pc, &insn_sizes);
        }
        id
    }

    fn get_plugin_u64(
        scoreboard: &Arc<Scoreboard<'static, VCPUScoreBoard>>,
        offset: usize,
    ) -> PluginU64 {
        let handle_ptr = scoreboard.as_ref() as *const _ as *const usize;
        let handle = unsafe { *handle_ptr };
        qemu_plugin::PluginU64 {
            score: handle as *mut qemu_plugin::sys::qemu_plugin_scoreboard,
            offset,
        }
    }

    /// Cache the scoreboard's element-array base in [`SB_BASE`].
    ///
    /// Must be called before any `sb_get`/`sb_set` and again whenever QEMU could
    /// have reallocated the array -- see [`SB_BASE`] for why `on_vcpu_init` is
    /// the only such point.
    fn refresh_sb_base(scoreboard: &Arc<Scoreboard<'static, VCPUScoreBoard>>) {
        let handle = Self::get_plugin_u64(scoreboard, 0).score;
        // SAFETY: `handle` is the live scoreboard QEMU handed us at
        // `Scoreboard::new`; vCPU 0 always exists by the time any vcpu_init runs.
        let base = unsafe { qemu_plugin::sys::qemu_plugin_scoreboard_find(handle, 0) };
        SB_BASE.store(base.cast::<u8>(), Ordering::Relaxed);
    }

    /// Address of one `u64` field of one vCPU's scoreboard entry.
    #[inline(always)]
    fn sb_addr(field: SbField, vcpu_idx: VCPUIndex) -> *mut u64 {
        let base = SB_BASE.load(Ordering::Relaxed);
        debug_assert!(!base.is_null(), "sb_addr before refresh_sb_base");
        // SAFETY: `base` points at an array of at least `num_vcpus`
        // `VCPUScoreBoard`s (QEMU allocates 16 and grows by doubling), and
        // `field` is an `offset_of!` within that struct, so the result is in
        // bounds and `u64`-aligned.
        unsafe {
            base.add(vcpu_idx as usize * mem::size_of::<VCPUScoreBoard>() + field.0)
                .cast::<u64>()
        }
    }

    /// Read a scoreboard field for a given vCPU.
    #[inline(always)]
    pub(crate) fn sb_get(field: SbField, vcpu_idx: VCPUIndex) -> u64 {
        // SAFETY: only this vCPU's thread touches its own entry, and the
        // inline stores QEMU generates for it are on that same thread.
        unsafe { Self::sb_addr(field, vcpu_idx).read() }
    }

    /// Write a scoreboard field for a given vCPU.
    #[inline(always)]
    pub(crate) fn sb_set(field: SbField, vcpu_idx: VCPUIndex, value: u64) {
        // SAFETY: as `sb_get`.
        unsafe { Self::sb_addr(field, vcpu_idx).write(value) }
    }

    /// Name a scoreboard field by its offset, for later `sb_get`/`sb_set`.
    fn sb_entry(_scoreboard: &Arc<Scoreboard<'static, VCPUScoreBoard>>, offset: usize) -> SbField {
        SbField(offset)
    }
}

// ---------------------------------------------------------------------------
// Callbacks
// ---------------------------------------------------------------------------

impl HasCallbacks for TurboTracePlugin {
    fn on_vcpu_init(&mut self, _id: PluginId, vcpu_id: VCPUIndex) -> Result<()> {
        println!("vcpu init on {}", vcpu_id);
        assert!(
            (vcpu_id as usize) < MAX_VCPUS,
            "VCPUIndex {} exceeds MAX_VCPUS ({})",
            vcpu_id,
            MAX_VCPUS,
        );
        self.init_scoreboard();

        // Cache the scoreboard base for `sb_get`/`sb_set`. QEMU has already
        // grown (and possibly moved) the element array for this vCPU by the
        // time this callback runs, so this is the right, and only necessary,
        // point to (re)resolve it. See `SB_BASE`.
        Self::refresh_sb_base(self.scoreboard.get().expect("init_scoreboard just ran"));

        // Emit the main-binary memory-map info (path + code range + entry) once,
        // sourced directly from the QEMU plugin API. The guest ELF is loaded by
        // this point, so start_code/end_code/entry_code are valid. This also
        // (re)writes TURBO_METADATA_FILE with an empty lib list, so a stale file
        // from a previous run in the same (uncleaned) workdir can never be read:
        // the live-trace path builds its memory map mid-run, so the file must
        // reflect only this run. It is then rewritten incrementally as
        // libraries load (see write_metadata).
        // Find VL, VTYPE, VLENB, and GP register handles via raw FFI.
        // This uses the repr(C) qemu_plugin_reg_descriptor layout (guaranteed
        // field order) instead of repr(Rust) RegisterDescriptor (unpredictable).
        //
        // Read BEFORE the meminfo write below: `vlenb` is stamped into the
        // metadata file so an offline `process` can recover the VLEN this run
        // was recorded with, and that first write happens in `write_meminfo`.
        let reg_handles = unsafe { get_raw_register_handles() };

        if let Some(v) = reg_handles.vlenb_value {
            log::trace!("read vlenb register as {}", v);
            self.vlenb = v.try_into().unwrap_or_else(|e| {
                log::error!("vlenb register value does not fit in u16! value = {e}");
                0
            });
        }

        if self.meminfo.is_none() {
            // `sp` (x2) is passed in for the auxv-based vDSO lookup: this is the
            // only moment the initial stack is still intact, and `PerVCPUData`
            // (the usual way to read a register) is not built until below.
            self.write_meminfo(reg_handles.gp_handles[2]);
        }

        // Log GP register handle availability for ROI marker support
        let gp_found = reg_handles
            .gp_handles
            .iter()
            .filter(|h| h.is_some())
            .count();
        log::trace!(
            "VCPU {}: found {}/32 GP register handles for ROI marker support",
            vcpu_id,
            gp_found
        );

        if let (Some(vl_h), Some(vtype_h)) = (reg_handles.vl_handle, reg_handles.vtype_handle) {
            let vcpu_data =
                PerVCPUData::new(vl_h, vtype_h, reg_handles.gp_handles, reg_handles.v_handles);
            // SAFETY: on_vcpu_init is called on the vCPU's own thread.
            unsafe { self.per_vcpu_data.insert(vcpu_id, vcpu_data) };

            log::trace!(
                "VCPU {} initialized: vl={:?}, vtype={:?}, vlenb={}",
                vcpu_id,
                // SAFETY: We just inserted, and we are on this vCPU's thread.
                unsafe { self.per_vcpu_data.get_mut(vcpu_id) }.and_then(|d| d.read_vl()),
                unsafe { self.per_vcpu_data.get_mut(vcpu_id) }
                    .and_then(|d| d.read_vtype())
                    .map(VType),
                self.vlenb
            );
        } else {
            log::error!(
                "VCPU {}: could not find register handles (vl={}, vtype={}). \
                 VL capture is disabled for this vCPU. All vsetvl* instructions \
                 will produce 'Ran out of VL values' warnings in the enricher. \
                 Check that the plugin API version (plugin-api-vN feature flag) \
                 matches the QEMU binary's plugin API version.",
                vcpu_id,
                reg_handles.vl_handle.is_some(),
                reg_handles.vtype_handle.is_some()
            );
        }

        let debug_log_path = if self.debug_enabled {
            Some(self.workdir.join("turbo_trace_debug.log"))
        } else {
            None
        };

        // Flush the previous encoder for this vcpu slot (if any) before replacing it.
        // QEMU reuses vcpu indices when threads exit and new ones are created.
        // Finishing the old encoder here ensures its trace is flushed to its
        // record log while the writer is still open and the vcpu is quiesced.
        // The new encoder below
        // gets a fresh unique hart_file_id, so each thread's data goes to its own file.
        // SAFETY: on_vcpu_init is called on the vCPU's own thread; no other thread
        // is executing on this vcpu index at this point.
        if let Some(old_encoder) = unsafe { self.trace_encoders.get_mut(vcpu_id) } {
            old_encoder.finish();
        }

        // Each thread that occupies a vCPU slot gets a unique monotonic ID so
        // its trace lands in its own `turbo_trace_vcpu<N>.bin` file, even when
        // QEMU reuses vcpu_id for a second (or later) thread.
        let hart_file_id = NEXT_HART_FILE_ID.fetch_add(1, Ordering::Relaxed);
        match TraceEncoder::new_with_record_log(
            hart_file_id,
            &self.workdir,
            self.flush_interval,
            debug_log_path,
            self.bbt_dict_log.clone(),
            self.compress_level,
            self.run_id,
        ) {
            Ok(encoder) => {
                // SAFETY: on_vcpu_init is called on the vCPU's own thread.
                unsafe { self.trace_encoders.insert(vcpu_id, encoder) };
            }
            Err(e) => {
                log::error!(
                    "Failed to create record-log for hart {hart_file_id} in {}: {e}",
                    self.workdir.display()
                );
            }
        }

        log::debug!(
            "Created TraceEncoder for vCPU {} (hart_id={})",
            vcpu_id,
            vcpu_id
        );

        // Initialise this vCPU's scoreboard row.
        //
        // The `vl`/`SEW` shadow is *seeded from the CSRs*, not zeroed. A new
        // vCPU here is a new guest thread, and QEMU linux-user builds one with
        // `cpu_copy()` -- a `memcpy` of the whole `CPUArchState` -- after which
        // `cpu_clone_regs_child()` fixes up only `sp` and `a0`. So the child
        // inherits the parent's `vl` and `vtype` verbatim, as of the instant it
        // called `clone`, and a thread's vector state at entry is whatever the
        // spawning thread happened to be running at.
        //
        // Zeroing the shadow here was wrong three ways: the `vl` oracle fired on
        // the child's first `R_REGS` `vsetvl*` (real `vl` vs. a zero shadow, on
        // any threaded guest whose spawning thread left `vl` non-zero -- glibc's
        // vector `memcpy` inside `printf` is enough); a `vsetvli x0, x0` on the
        // child took its AVL from that zero and *computed a wrong `vl`* into the
        // trace; and a zeroed `shadow_sew_bits` gave indexed memory ops the
        // wrong element width until the first `vsetvl*`.
        let scoreboard = self
            .scoreboard
            .get()
            .expect("scoreboard must be initialized");
        {
            use std::mem::offset_of;
            // SAFETY: on_vcpu_init runs on this vCPU's own thread, and the
            // per-vCPU data for it was inserted above.
            let inherited = unsafe { self.per_vcpu_data.get_mut(vcpu_id) }
                .and_then(|d| d.read_vl().zip(d.read_vtype()));
            let (vl, sew_bits) = match inherited {
                Some((vl, vtype)) => (vl, u64::from(sew_bits_from_vtype(vtype))),
                None => {
                    // No handles, or the read failed. Zero is the process-start
                    // truth and the best guess available; say so, because on a
                    // spawned thread it can be neither.
                    log::warn!(
                        "vCPU {vcpu_id}: could not read the vl/vtype it starts with; \
                         seeding the shadow with 0. If this vCPU is a spawned guest \
                         thread it inherited its parent's vector state, so the first \
                         vl this plugin computes for it may be wrong."
                    );
                    (0, 0)
                }
            };
            if vl != 0 {
                log::debug!(
                    "vCPU {vcpu_id} starts with an inherited vl={vl} (SEW={sew_bits}); \
                     seeding the shadow with it rather than 0."
                );
            }
            for (offset, value) in [
                (offset_of!(VCPUScoreBoard, shadow_vl), vl),
                (offset_of!(VCPUScoreBoard, shadow_sew_bits), sew_bits),
                // Nothing carries over here: `ff_pending_pc` is the plugin's own
                // "a `vl*ff` has executed and its `vl` is unread" mark, and a
                // vCPU that is only now starting has executed nothing. QEMU
                // reuses vCPU indices as guest threads come and go, so this must
                // be cleared per init and not just once.
                (offset_of!(VCPUScoreBoard, ff_pending_pc), 0),
            ] {
                let entry = Self::get_plugin_u64(scoreboard, offset);
                qemu_plugin::qemu_plugin_u64_set(entry, vcpu_id, value);
            }
        }

        Ok(())
    }

    fn on_translation_block_translate(
        &mut self,
        _id: PluginId,
        tb: TranslationBlock,
    ) -> Result<()> {
        use std::mem::offset_of;

        let tb_pc = tb.vaddr();

        let instructions: Vec<_> = tb.instructions().collect();

        let scoreboard = self
            .scoreboard
            .get()
            .expect("scoreboard must be initialized before translation")
            .clone();

        let first_insn_disas = instructions
            .first()
            .and_then(|i| i.disas().ok())
            .unwrap_or_default();

        // Per-vCPU shadow of the `vl` CSR, written by every `vsetvl*` callback
        // and by the `vl*ff` successor-block callback just below. Resolved here,
        // above both, because both need it.
        let shadow = VtypeShadow {
            vl: Self::sb_entry(&scoreboard, offset_of!(VCPUScoreBoard, shadow_vl)),
            sew_bits: Self::sb_entry(&scoreboard, offset_of!(VCPUScoreBoard, shadow_sew_bits)),
        };
        let ff_pending = Self::sb_entry(&scoreboard, offset_of!(VCPUScoreBoard, ff_pending_pc));

        // ---------------------------------------------------------------
        // `vl*ff` successor-block callback: read the `vl` a fault-only-first
        // load left behind, and emit its record.
        //
        // Registered *before* the TB-start callback below and only on blocks a
        // `vl*ff` falls through to, which matters twice. QEMU runs TB callbacks
        // in registration order, so this one's record lands ahead of this
        // block's own `BLOCK` record -- i.e. it trails the `vl*ff`'s block,
        // exactly where the format puts a `VL` (see `bbt.rs`). And the flush
        // that could cut a batch happens inside `record_block_word`, so the cut
        // is after this record, never between the `vl*ff`'s block and its `vl`.
        //
        // This is the only `R_REGS` callback on a block entry in the whole
        // plugin. It is what the `vsetvl*` work removed from *every* block and
        // it is paid here on `vl*ff` successors alone.
        // ---------------------------------------------------------------
        if self.ff_succ_pcs.contains(&tb_pc) {
            let encoders_ff = self.trace_encoders.clone();
            let per_vcpu_data_ff = self.per_vcpu_data.clone();
            tb.register_execute_callback_flags(
                move |vcpu_idx| {
                    let ff_pc = Self::sb_get(ff_pending, vcpu_idx);
                    if ff_pc == 0 {
                        // Entered from somewhere other than the `vl*ff`: this
                        // block is just an ordinary jump target as well.
                        return;
                    }
                    Self::sb_set(ff_pending, vcpu_idx, 0);
                    // SAFETY: QEMU guarantees vcpu_idx callbacks run only on
                    // that vCPU's thread.
                    let Some(data) = (unsafe { per_vcpu_data_ff.get_mut(vcpu_idx) }) else {
                        return;
                    };
                    let Some(vl) = data.read_vl() else {
                        // The record is owed either way: dropping it would
                        // shift every later `vl` by one on the consumer side,
                        // so emit the shadow and say so.
                        log::warn!(
                            "vl*ff at PC {ff_pc:#x}: could not read the vl CSR it wrote; \
                             recording the untrimmed vl instead"
                        );
                        let stale = Self::sb_get(shadow.vl, vcpu_idx);
                        if let Some(encoder) = unsafe { encoders_ff.get_mut(vcpu_idx) } {
                            encoder.record_vl(stale);
                        }
                        return;
                    };
                    if vl != Self::sb_get(shadow.vl, vcpu_idx) {
                        // A fault really did trim it. `SEW` is untouched: a
                        // `vl*ff` writes `vl` and nothing else in `vtype`.
                        Self::sb_set(shadow.vl, vcpu_idx, vl);
                        VLEFF_TRIMS.fetch_add(1, Ordering::Relaxed);
                    }
                    if let Some(encoder) = unsafe { encoders_ff.get_mut(vcpu_idx) } {
                        encoder.record_vl(vl);
                    }
                },
                qemu_plugin::sys::qemu_plugin_cb_flags::QEMU_PLUGIN_CB_R_REGS,
            );
        }

        // ---------------------------------------------------------------
        // TB-start callback: the entire per-block job is "append this block's
        // record word". No scoreboard read, no branch resolution, no
        // generator, no packet encoder -- and, below, no per-instruction
        // inline store either. The word is computed here, once per
        // translation, and carried in the closure.
        // ---------------------------------------------------------------
        let encoders_tb = self.trace_encoders.clone();

        let word = bbt_block(self.intern_block(tb_pc, &instructions));
        tb.register_execute_callback_flags(
            move |vcpu_idx| {
                // SAFETY: QEMU guarantees vcpu_idx callbacks run only on that
                // vCPU's thread.
                if let Some(encoder) = unsafe { encoders_tb.get_mut(vcpu_idx) } {
                    encoder.record_block_word(word);
                }
            },
            qemu_plugin::sys::qemu_plugin_cb_flags::QEMU_PLUGIN_CB_NO_REGS,
        );

        // ---------------------------------------------------------------
        // Per-instruction processing (translation time)
        // ---------------------------------------------------------------
        // OPT-7: Clone Arcs once before the loop instead of per-instruction.
        let encoders_loop = self.trace_encoders.clone();
        let per_vcpu_data_loop = self.per_vcpu_data.clone();

        for (idx, insn) in instructions.iter().enumerate() {
            let ipc = insn.vaddr();
            let disas = if idx == 0 {
                first_insn_disas.clone()
            } else {
                insn.disas()
                    .expect("qemu disassembly failed - must succeed.")
            };

            let is_vector_set = TraceEncoder::is_vector_set_instruction(&disas);

            // -- Vector configuration instructions (vsetvli, vsetivli, vsetvl) --
            // A `vsetvl*` always ends its TB (`do_vsetvl` -> `lookup_and_goto_ptr`,
            // `DISAS_NORETURN`), so `vl` never exists yet inside the same block and
            // the capture is always done by the *next* block's start callback via
            // `META_NEED_VSETVL`, which this instruction's metadata word already
            // carries. There used to be an "if a next instruction exists in this
            // same TB, capture from it" path here; it never executed on any
            // measured workload (596 of 596 translated `vsetvl*` were last in
            // their TB) and its presence obscured which path really captures `vl`.
            // The invariant is asserted here instead: if a future QEMU ever
            // stopped ending the block at a `vsetvl*`, this instruction's
            // callback would run at a point where the guest's register state is
            // no longer the `vsetvl*`'s inputs. Fail loudly at translate time
            // rather than produce a trace full of wrong `vl` values.
            if is_vector_set {
                assert_eq!(
                    idx,
                    instructions.len() - 1,
                    "vsetvl* at PC {ipc:#x} is instruction {idx} of {} in its TB, \
                     but must be the last (`do_vsetvl` -> `lookup_and_goto_ptr`, \
                     DISAS_NORETURN). A QEMU that no longer does this needs the \
                     `vl` capture path reworked, not this assertion relaxed.",
                    instructions.len(),
                );
                assert!(
                    self.vlenb != 0,
                    "vsetvl* at PC {ipc:#x} but vlenb was never read from QEMU, so no \
                     `vl` can be computed. This means the vlenb register handle was not \
                     found at vCPU init -- check the plugin API version matches QEMU's."
                );

                // Decide, once per site, where this `vsetvl*`'s AVL and `vtype`
                // come from -- from the instruction encoding, not from the
                // disassembly text (see `vl_compute::decode_vsetvl`).
                let bytes = insn.data();
                let plan = decode_vsetvl(&bytes, self.vlenb as u64).unwrap_or_else(|| {
                    panic!(
                        "instruction at PC {ipc:#x} was classified as a vsetvl* from its \
                         disassembly ({disas:?}) but its encoding {bytes:02x?} does not \
                         decode as one. The two classifiers must agree, or a `vl` value \
                         goes missing."
                    )
                });
                // `SEW` for the immediate forms; `vsetvl` overrides it from the
                // runtime `vtype` in its own callback below.
                let static_sew = decode_vsetvl_static_sew(&bytes).unwrap_or(8);
                let form_idx = VsetvlForm::from(plan) as usize;
                VSETVL_SITES[form_idx].fetch_add(1, Ordering::Relaxed);

                let encoders_vs = encoders_loop.clone();
                let vset_pc = ipc;
                let vlenb = self.vlenb as u64;
                let verify = self.vl_verify;

                match plan {
                    // Nothing to read: `vl` was fully determined above. The
                    // callback exists only to put the constant into the stream
                    // (and to keep the shadow current for the `x0`/`x0` form).
                    VlPlan::Const(vl) => {
                        insn.register_execute_callback(move |vcpu_idx| {
                            emit_computed_vl(
                                &encoders_vs,
                                shadow,
                                vcpu_idx,
                                vl,
                                static_sew,
                                form_idx,
                            );
                        });
                    }
                    // AVL is the current `vl`. Reading it from QEMU would need
                    // `R_REGS`; the shadow gives the same value for free. Note
                    // this form *preserves* `vl` whenever `vlmax` still allows
                    // it, so the read is usually a no-op.
                    VlPlan::Shadow { vlmax } => {
                        insn.register_execute_callback(move |vcpu_idx| {
                            let avl = Self::sb_get(shadow.vl, vcpu_idx);
                            emit_computed_vl(
                                &encoders_vs,
                                shadow,
                                vcpu_idx,
                                compute_vl(vlmax, avl),
                                static_sew,
                                form_idx,
                            );
                        });
                    }
                    // The one form that pays for `R_REGS`: one read of `x[rs1]`
                    // for the AVL. `vtype` is the immediate, so `vlmax` was
                    // computed at translate time.
                    VlPlan::Reg { rs1, vlmax } => {
                        let per_vcpu_data_vs = per_vcpu_data_loop.clone();
                        insn.register_execute_callback_flags(
                            move |vcpu_idx| {
                                // SAFETY: QEMU guarantees vcpu_idx callbacks run
                                // only on that vCPU's thread.
                                let Some(data) = (unsafe { per_vcpu_data_vs.get_mut(vcpu_idx) })
                                else {
                                    return;
                                };
                                // The oracle first: at this instant the `vl` CSR
                                // still holds what the *previous* `vsetvl*`
                                // produced, which is what the shadow should say.
                                if verify != VlVerify::Off {
                                    let shadow_vl = Self::sb_get(shadow.vl, vcpu_idx);
                                    check_vl_oracle(data, shadow_vl, vset_pc, verify);
                                }
                                let avl = data.read_gp_reg(rs1).unwrap_or_else(|| {
                                    panic!(
                                        "vsetvl* at PC {vset_pc:#x}: could not read its AVL \
                                         source x{rs1}, so its `vl` cannot be computed"
                                    )
                                });
                                emit_computed_vl(
                                    &encoders_vs,
                                    shadow,
                                    vcpu_idx,
                                    compute_vl(vlmax, avl),
                                    static_sew,
                                    form_idx,
                                );
                            },
                            qemu_plugin::sys::qemu_plugin_cb_flags::QEMU_PLUGIN_CB_R_REGS,
                        );
                    }
                    // `vsetvl`: `vtype` itself is in `x[rs2]`, so `vlmax` has to
                    // be derived at runtime too. Measured at 0 executions on
                    // a measured workload; implemented so the form is never silently wrong.
                    VlPlan::DynVtype { rs1, rs2, rd } => {
                        let per_vcpu_data_vs = per_vcpu_data_loop.clone();
                        insn.register_execute_callback_flags(
                            move |vcpu_idx| {
                                // SAFETY: as above.
                                let Some(data) = (unsafe { per_vcpu_data_vs.get_mut(vcpu_idx) })
                                else {
                                    return;
                                };
                                let shadow_vl = Self::sb_get(shadow.vl, vcpu_idx);
                                if verify != VlVerify::Off {
                                    check_vl_oracle(data, shadow_vl, vset_pc, verify);
                                }
                                fn read(data: &mut PerVCPUData, idx: u8, pc: u64) -> u64 {
                                    data.read_gp_reg(idx).unwrap_or_else(|| {
                                        panic!(
                                            "vsetvl at PC {pc:#x}: could not read x{idx}, so \
                                             its `vl` cannot be computed"
                                        )
                                    })
                                }
                                let vtype = read(data, rs2, vset_pc);
                                // `do_vsetvl`'s AVL selection, same three cases.
                                let avl = match (rs1, rd) {
                                    (0, 0) => shadow_vl,
                                    (0, _) => RV_VLEN_MAX,
                                    _ => read(data, rs1, vset_pc),
                                };
                                emit_computed_vl(
                                    &encoders_vs,
                                    shadow,
                                    vcpu_idx,
                                    compute_vl(vlmax_from_vtype(vlenb, vtype), avl),
                                    sew_bits_from_vtype(vtype),
                                    form_idx,
                                );
                            },
                            qemu_plugin::sys::qemu_plugin_cb_flags::QEMU_PLUGIN_CB_R_REGS,
                        );
                    }
                }
            }

            // -- Vector memory ops: capture the runtime addresses --
            // The cache footprint of the workload is the runtime addresses these
            // touch, and nothing offline can recover them: the base is a GP
            // register and the indices of a gather/scatter are a vector one.
            // What each mode has to read is decided here, once per site, exactly
            // as the `vsetvl*` handling above decides its reads.
            let vec_mem_op = insn
                .data()
                .get(..4)
                .and_then(|b| b.try_into().ok())
                .map(u32::from_le_bytes)
                .and_then(VecMemOp::decode);

            // -- Fault-only-first loads: mark that a `vl` read is owed --
            // Independent of `mem_capture`: this is `vl` fidelity, not the
            // footprint. All this site does is store the PC; the read itself
            // happens in the successor block's callback registered above, since
            // the trimmed `vl` does not exist until the load has run.
            if vec_mem_op.is_some_and(|op| op.writes_vl()) {
                // A `vl*ff` ends its TB (`ldff_trans` -> `gen_update_pc` +
                // `lookup_and_goto_ptr`, `DISAS_NORETURN`), for the same reason
                // a `vsetvl*` does: it may have changed `vl`. That is what makes
                // the successor PC statically `ipc + size` and makes it a block
                // *start*, which is the whole basis of the capture below.
                assert_eq!(
                    idx,
                    instructions.len() - 1,
                    "vl*ff at PC {ipc:#x} is instruction {idx} of {} in its TB, but must \
                     be the last (`ldff_trans` -> `lookup_and_goto_ptr`, DISAS_NORETURN). \
                     If a QEMU stopped ending the block there, `ipc + size` is no longer \
                     the successor block's start PC and the `vl` this load wrote would \
                     never be read. Rework the capture, do not relax this.",
                    instructions.len(),
                );
                let succ_pc = ipc + insn.size() as u64;
                self.note_ff_successor(ipc, succ_pc);
                VLEFF_SITES.fetch_add(1, Ordering::Relaxed);
                insn.register_execute_callback_flags(
                    move |vcpu_idx| {
                        VLEFF_EXECS.fetch_add(1, Ordering::Relaxed);
                        if Self::sb_get(ff_pending, vcpu_idx) != 0 {
                            // The previous mark was never collected, so its `vl`
                            // record was never emitted. See [`VLEFF_LOST`].
                            VLEFF_LOST.fetch_add(1, Ordering::Relaxed);
                        }
                        Self::sb_set(ff_pending, vcpu_idx, ipc);
                    },
                    qemu_plugin::sys::qemu_plugin_cb_flags::QEMU_PLUGIN_CB_NO_REGS,
                );
            }

            if let Some(op) = vec_mem_op.filter(|_| self.mem_capture) {
                let encoders_mem = encoders_loop.clone();
                let per_vcpu_data_mem = per_vcpu_data_loop.clone();
                let mem_pc = ipc;

                match op.group_kind() {
                    // One GP read: the base. Everything else about the footprint
                    // is in the encoding and the `vl` already in the stream.
                    MemGroupKind::Base => {
                        let rs1 = op.rs1;
                        insn.register_execute_callback_flags(
                            move |vcpu_idx| {
                                let base = read_base(&per_vcpu_data_mem, vcpu_idx, rs1, mem_pc);
                                // SAFETY: vcpu_idx callbacks run only on that vCPU's thread.
                                if let Some(encoder) = unsafe { encoders_mem.get_mut(vcpu_idx) } {
                                    encoder.record_mem_group(MemGroupKind::Base, &[base]);
                                }
                            },
                            qemu_plugin::sys::qemu_plugin_cb_flags::QEMU_PLUGIN_CB_R_REGS,
                        );
                    }
                    // Two GP reads: base and stride. The stride is signed and is
                    // wrapped into the address encoding, to be sign-extended by
                    // the consumer.
                    MemGroupKind::BaseStride => {
                        let (rs1, rs2) = (op.rs1, op.rs2);
                        insn.register_execute_callback_flags(
                            move |vcpu_idx| {
                                let base = read_base(&per_vcpu_data_mem, vcpu_idx, rs1, mem_pc);
                                let stride = read_base(&per_vcpu_data_mem, vcpu_idx, rs2, mem_pc)
                                    & ((1 << BBT_ADDR_BITS) - 1);
                                if let Some(encoder) = unsafe { encoders_mem.get_mut(vcpu_idx) } {
                                    encoder.record_mem_group(
                                        MemGroupKind::BaseStride,
                                        &[base, stride],
                                    );
                                }
                            },
                            qemu_plugin::sys::qemu_plugin_cb_flags::QEMU_PLUGIN_CB_R_REGS,
                        );
                    }
                    // Indexed: the footprint is the index vector's contents, so
                    // it is expanded here, at the only point it exists, and the
                    // resulting line set is emitted.
                    MemGroupKind::Lines => {
                        insn.register_execute_callback_flags(
                            move |vcpu_idx| {
                                // SAFETY: as above.
                                let Some(data) = (unsafe { per_vcpu_data_mem.get_mut(vcpu_idx) })
                                else {
                                    return;
                                };
                                let base = data.read_gp_reg(op.rs1).unwrap_or_else(|| {
                                    panic!(
                                        "vector memory op at PC {mem_pc:#x}: could not read its \
                                         base register x{}, so its cache footprint is unknown",
                                        op.rs1
                                    )
                                });
                                let vl = Self::sb_get(shadow.vl, vcpu_idx);
                                let sew_bits = Self::sb_get(shadow.sew_bits, vcpu_idx) as u16;
                                let seg_bytes = op.segment_bytes(sew_bits);

                                // The index EMUL may span a register group, so
                                // read consecutive registers until `vl` indices
                                // are covered. Registers are grouped
                                // consecutively, so `vs2 + k` is the k-th.
                                let index_bytes = vl * u64::from(op.width_bits) / 8;
                                let mut buf = std::mem::take(&mut data.index_buf);
                                buf.clear();
                                let mut vreg = op.vs2;
                                while (buf.len() as u64) < index_bytes {
                                    assert!(
                                        vreg < 32 && data.read_vreg_into(vreg, &mut buf),
                                        "vector memory op at PC {mem_pc:#x}: could not read index \
                                         register v{vreg}. Its addresses are the index vector's \
                                         contents, so without it this instruction's cache \
                                         footprint cannot be recovered at all -- and a partial \
                                         footprint would be indistinguishable from a real one. \
                                         Does this QEMU expose v0..v31 through the plugin API?"
                                    );
                                    vreg += 1;
                                }

                                let mut lines = std::mem::take(&mut data.line_buf);
                                let ok = indexed_lines(
                                    base,
                                    &buf,
                                    op.width_bits,
                                    seg_bytes,
                                    vl,
                                    &mut lines,
                                );
                                assert!(
                                    ok,
                                    "vector memory op at PC {mem_pc:#x}: read {} index byte(s) \
                                     for vl={vl} indices of {} bits",
                                    buf.len(),
                                    op.width_bits
                                );
                                data.index_buf = buf;

                                if let Some(encoder) = unsafe { encoders_mem.get_mut(vcpu_idx) } {
                                    encoder.record_mem_group(MemGroupKind::Lines, &lines);
                                }
                                // SAFETY: as above -- reborrowed to return the
                                // scratch buffer after the encoder borrow ends.
                                if let Some(data) = unsafe { per_vcpu_data_mem.get_mut(vcpu_idx) } {
                                    data.line_buf = lines;
                                }
                            },
                            qemu_plugin::sys::qemu_plugin_cb_flags::QEMU_PLUGIN_CB_R_REGS,
                        );
                    }
                }
            }

            // -- ROI R-type marker detection --
            // Detect R-type instructions writing to x0 (ROI markers) and register
            // execution callbacks to capture the runtime register values (rs1, rs2).
            if let Some((roi_kind, rs1_idx, rs2_idx)) =
                detect_roi_from_instruction_bytes(&insn.data())
            {
                let encoders_roi = encoders_loop.clone();
                let per_vcpu_data_roi = per_vcpu_data_loop.clone();
                let roi_pc = ipc;

                insn.register_execute_callback_flags(
                    move |vcpu_idx| {
                        let rs1_val = unsafe { per_vcpu_data_roi.get_mut(vcpu_idx) }
                            .and_then(|data| data.read_gp_reg(rs1_idx))
                            .unwrap_or_else(|| {
                                log::warn!("ROI: failed to read x{} at PC {:#x}", rs1_idx, roi_pc);
                                0
                            });
                        let rs2_val = unsafe { per_vcpu_data_roi.get_mut(vcpu_idx) }
                            .and_then(|data| data.read_gp_reg(rs2_idx))
                            .unwrap_or_else(|| {
                                log::warn!("ROI: failed to read x{} at PC {:#x}", rs2_idx, roi_pc);
                                0
                            });

                        // For pointer-carrying V2 markers, rs1 is a pointer to
                        // the name and rs2 is its length. Capture the string
                        // from guest memory now, while the vCPU context is live
                        // (the pointer may target the stack, unreadable later).
                        let name = match roi_kind {
                            RoiRtMarkerKind::Add
                            | RoiRtMarkerKind::Sub
                            | RoiRtMarkerKind::Sll
                            | RoiRtMarkerKind::Srl => {
                                unsafe { per_vcpu_data_roi.get_mut(vcpu_idx) }
                                    .and_then(|data| data.read_guest_string(rs1_val, rs2_val))
                            }
                            _ => None,
                        };

                        let marker = RoiRtMarker {
                            pc: roi_pc,
                            kind: roi_kind,
                            rs1_val,
                            rs2_val,
                            name,
                        };

                        if let Some(encoder) = unsafe { encoders_roi.get_mut(vcpu_idx) } {
                            encoder.record_roi_marker(marker);
                        }
                    },
                    qemu_plugin::sys::qemu_plugin_cb_flags::QEMU_PLUGIN_CB_R_REGS,
                );
            }
        }

        Ok(())
    }

    fn on_exit(&mut self, _id: PluginId) -> Result<()> {
        // SAFETY: on_exit is called after all vCPU threads have stopped,
        // so iterating all slots is safe.
        for (vcpu_id, encoder) in unsafe { self.trace_encoders.iter_mut() } {
            encoder.finish();
            println!(
                "E-Trace for vCPU {} finalised: {} steps",
                vcpu_id, encoder.total_steps,
            );
        }

        // `vsetvl*` / computed-`vl` report. Every tracer design projection is
        // checked against these numbers, and the R_REGS
        // execution count is the direct check that this design did what it
        let sites: Vec<u64> = VSETVL_SITES
            .iter()
            .map(|c| c.load(Ordering::Relaxed))
            .collect();
        let execs: Vec<u64> = VSETVL_EXECS
            .iter()
            .map(|c| c.load(Ordering::Relaxed))
            .collect();
        let r_regs_execs: u64 = VsetvlForm::ALL
            .iter()
            .filter(|form| form.needs_regs())
            .map(|form| execs[*form as usize])
            .sum();
        println!(
            "vsetvl*: {} sites translated, {} executions ({} of them with R_REGS)",
            sites.iter().sum::<u64>(),
            execs.iter().sum::<u64>(),
            r_regs_execs,
        );
        for form in VsetvlForm::ALL {
            let i = form as usize;
            if sites[i] != 0 || execs[i] != 0 {
                println!(
                    "    {:<48} {:>8} sites, {:>12} executions",
                    form.name(),
                    sites[i],
                    execs[i],
                );
            }
        }
        let samples = VL_ORACLE_SAMPLES.load(Ordering::Relaxed);
        let mismatches = VL_ORACLE_MISMATCHES.load(Ordering::Relaxed);
        match self.vl_verify {
            VlVerify::Off => println!("    vl oracle: disabled (vl_verify=off)"),
            _ if samples == 0 => println!(
                "    vl oracle: NO SAMPLES TAKEN -- this run has no vsetvl* site that reads \
                 a register, so the computed vl was never checked against QEMU's"
            ),
            _ => println!(
                "    vl oracle: {samples} computed-vs-read comparisons, {mismatches} mismatch(es)"
            ),
        }

        // `vl*ff` report. `lost` is the one number that says the trace is
        // broken: every executed `vl*ff` owes exactly one `vl` record, so a
        // non-zero count means the consumer's `vl` stream is short.
        let ff_sites = VLEFF_SITES.load(Ordering::Relaxed);
        let ff_execs = VLEFF_EXECS.load(Ordering::Relaxed);
        let ff_trims = VLEFF_TRIMS.load(Ordering::Relaxed);
        let ff_lost = VLEFF_LOST.load(Ordering::Relaxed);
        if ff_sites != 0 {
            println!(
                "vl*ff: {ff_sites} sites translated, {ff_execs} executions, \
                 {ff_trims} of them trimmed vl (R_REGS on the successor block)"
            );
        }
        if ff_lost != 0 {
            println!(
                "    WARNING: {ff_lost} vl*ff execution(s) never had their vl read. \
                 Their vl records are missing from the trace."
            );
        }

        // Persist the shared-object mappings discovered from guest syscalls.
        self.write_metadata();

        Ok(())
    }

    /// Observe guest syscall *entry* to discover shared-object load mappings.
    /// We only stash the arguments here; the outcome (fd / mapped address) is
    /// known at syscall return. See [`SYS_OPENAT`]/[`SYS_MMAP`]/[`SYS_CLOSE`].
    fn on_syscall(
        &mut self,
        _id: PluginId,
        vcpu: VCPUIndex,
        num: i64,
        a1: u64,
        a2: u64,
        _a3: u64,
        _a4: u64,
        a5: u64,
        a6: u64,
        _a7: u64,
        _a8: u64,
    ) -> Result<()> {
        match num {
            SYS_OPENAT => {
                // a2 = const char *pathname (a1 = dirfd).
                if let Some(path) = self.read_guest_cstr(vcpu, a2) {
                    self.pending_openat.insert(vcpu, path);
                }
            }
            SYS_MMAP => {
                // a5 = fd, a6 = offset.
                self.pending_mmap.insert(vcpu, (a5 as i64, a6));
            }
            SYS_CLOSE => {
                // a1 = fd. Drop tracking so a later reuse of this fd number is
                // not misattributed to the file that used to hold it.
                self.open_fds.remove(&(a1 as i64));
            }
            _ => {}
        }
        Ok(())
    }

    /// Resolve syscall outcomes stashed by [`on_syscall`]: map fds to paths on
    /// `openat` return, and record a shared object's load base when its
    /// file-offset-0 segment is `mmap`ed (validated by ELF magic).
    fn on_syscall_return(
        &mut self,
        _id: PluginId,
        vcpu: VCPUIndex,
        num: i64,
        ret: i64,
    ) -> Result<()> {
        match num {
            SYS_OPENAT => {
                if let Some(path) = self.pending_openat.remove(&vcpu) {
                    if ret >= 0 {
                        self.open_fds.insert(ret, path);
                    }
                }
            }
            SYS_MMAP => {
                if let Some((fd, offset)) = self.pending_mmap.remove(&vcpu) {
                    // mmap failure returns a small negative errno; a success is
                    // an address. Only the file-offset-0 segment lands at the
                    // module load base.
                    let mmap_failed = (-4095..0).contains(&ret);
                    if !mmap_failed && offset == 0 && fd >= 0 {
                        let base = ret as u64;
                        if let Some(path) = self.open_fds.get(&fd).cloned() {
                            if self.guest_is_elf(vcpu, base) {
                                // A small object can map several segments at
                                // page-aligned file offset 0 (each starting with
                                // the ELF header bytes); the load base is the
                                // lowest such address, so keep the minimum.
                                let changed = match self.lib_mappings.get(&path) {
                                    Some(&existing) => base < existing,
                                    None => true,
                                };
                                if changed {
                                    log::debug!("libmap: {path} @ 0x{base:x}");
                                    self.lib_mappings.insert(path, base);
                                    // Rewrite incrementally: the live-trace path
                                    // reads this mid-run, before on_exit.
                                    self.write_metadata();
                                }
                            }
                        }
                    }
                }
            }
            _ => {}
        }
        Ok(())
    }
}

register!(TurboTracePlugin::default());
