//! Region-of-interest statistics: [`RoiInfo`] and [`RoiExecution`].

use super::insn_class::{
    InsnClass, InsnDelta, IC_BRANCH, IC_FLOAT, IC_FMA, IC_INT_FMA, IC_LOAD, IC_STORE, IC_VECTOR,
};
use super::vector::{Lmul, SYSTEM_VLEN};
use crate::processing::histogram::VectorLengthHistogram;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::{
    collections::BTreeMap,
    ops::{Add, AddAssign},
    sync::Arc,
};

/// Aggregated statistics for a Region of Interest (function)
/// A single dynamic execution of a named ROI region of interest.
///
/// Captured by the [`Enricher`](crate::processing::enricher) as it walks the
/// architectural instruction stream. Region boundaries are identified not by
/// an absolute instruction index (which drifts whenever an engine expands
/// vector/micro-coded instructions into several retired rows) but by the
/// **marker ordinal**: a 0-based count of every ROI marker instruction
/// detected so far in commit order.
///
/// ROI markers are scalar `x0`-writes that are *not* micro-coded, so they
/// retire exactly once, in a stable order, regardless of vector micro-op
/// expansion.
#[derive(Debug, Clone, PartialEq)]
pub struct RoiExecution {
    /// Region name as resolved by the marker handler.
    pub name: String,
    /// Region id (stable per begin-PC / per event-value pair).
    pub region_id: usize,
    /// Ordinal of the begin-marker within the committed marker stream.
    pub begin_marker_ordinal: u64,
    /// PC of the begin-marker instruction.
    pub begin_pc: u64,
    /// Kind of the begin-marker (for cross-stream validation).
    pub begin_marker_kind: crate::processing::RoiMarker,
    /// Ordinal of the end-marker within the committed marker stream.
    pub end_marker_ordinal: u64,
    /// PC of the end-marker instruction.
    pub end_pc: u64,
    /// Kind of the end-marker (for cross-stream validation).
    pub end_marker_kind: crate::processing::RoiMarker,
    pub hart_id: u32,
    /// Retired-instruction counter (global_roi.instructions) snapshotted at begin.
    pub begin_instr: u64,
    /// Retired-instruction counter snapshotted at end.
    pub end_instr: u64,
    /// region_id of the enclosing open region at begin time (None = top level).
    pub parent_region_id: Option<usize>,
    /// Depth in the open-region stack at begin time (0 = top level).
    pub depth: u32,
}

// NOTES:
// - "instruction" means += 1 per instruction
// - "x_op_elements" means "count per scalar value": a VL of 32 f64 values is += 32 (see float_elements example below)
/// Modelled cache behaviour of one scope's vector memory accesses.
///
/// The serialised form of [`crate::processing::cache_model::Counts`], kept as a
/// separate type so `RoiInfo` does not depend on the model's own type being
/// `Serialize` (the intent is that the model is replaced by a real library).
/// Zero unless the run was processed with cache modelling enabled, and skipped
/// entirely when serialising in that case, so an unmodelled run's JSON is
/// byte-identical to what it was before this existed.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct CacheCounts {
    pub l1_hits: u64,
    pub l1_misses: u64,
    /// Only lines that missed the L1 are looked up in the L2, so
    /// `l2_hits + l2_misses == l1_misses`.
    pub l2_hits: u64,
    pub l2_misses: u64,
}

impl CacheCounts {
    /// Nothing was modelled (or nothing missed and nothing hit, which for a run
    /// that executed any vector memory op is the same thing).
    pub fn is_zero(&self) -> bool {
        *self == Self::default()
    }

    /// Line lookups, i.e. `l1_hits + l1_misses`.
    pub fn lines(&self) -> u64 {
        self.l1_hits + self.l1_misses
    }

    /// L1 hit rate, or `None` when nothing was looked up. Deliberately not a
    /// serialised field: a ratio in the JSON invites summing it.
    pub fn l1_hit_rate(&self) -> Option<f64> {
        (self.lines() > 0).then(|| self.l1_hits as f64 / self.lines() as f64)
    }

    pub fn add(&mut self, other: &Self) {
        self.l1_hits += other.l1_hits;
        self.l1_misses += other.l1_misses;
        self.l2_hits += other.l2_hits;
        self.l2_misses += other.l2_misses;
    }
}

impl From<crate::processing::cache_model::Counts> for CacheCounts {
    fn from(c: crate::processing::cache_model::Counts) -> Self {
        Self {
            l1_hits: c.l1_hits,
            l1_misses: c.l1_misses,
            l2_hits: c.l2_hits,
            l2_misses: c.l2_misses,
        }
    }
}

/// Aggregated statistics for one region of interest.
///
/// # Accounting model and its two known divergences from the vector spec
///
/// Element and byte counters are derived from the decoded instruction stream
/// plus tracked `vtype`/`vl` state. That is exact for the addressing modes and
/// widths (see [`vector_elements_for`], [`is_indexed_mem`] and
/// [`whole_reg_nreg_eew`]), but two spec behaviours depend on runtime state
/// that an instruction trace does not carry. Both make counters read *high*;
/// neither is detectable from the trace, so neither is flagged at runtime.
///
/// ## Masking is not modelled
///
/// The spec opens section 7 with: vector loads and stores "only access memory
/// or raise exceptions for active elements", and section 7.8 adds that for
/// segment ops "masking is also applied at the level of whole segments".
///
/// We count all `vl` elements regardless of `v0`, because the mask register's
/// *contents* are architectural state that never appears in the trace. A
/// masked op with half its elements inactive is therefore reported at full
/// width: `load_bytes`/`store_bytes` and the `*_elements` counters all
/// overcount, by exactly the inactive fraction. Unmasked code -- the common
/// case in vectorised kernels, and everything in the assembly validation suite
/// -- is unaffected.
///
/// Fixing this would require either mask-value tracing from the QEMU plugin or
/// full register-value emulation; it is not a decode-side problem.
///
/// ## Fault-only-first VL trimming is not modelled
///
/// Section 7.7: if an element `> 0` of a `vle<eew>ff.v`/`vlseg<nf>e<eew>ff.v`
/// faults, "the vector length vl is reduced to the index of the element that
/// would have raised an exception" -- without executing a `vsetvl*`.
/// Implementations are also permitted to process fewer than `vl` elements and
/// trim `vl` even when no exception is raised.
///
/// The VL tracker only observes `vsetvl*` writes, so after such a trim its
/// `vl` is stale: the faulting load itself and every subsequent vector op are
/// attributed the untrimmed `vl` until the next `vsetvl*`. In practice this
/// requires an actual fault (the fault-only-first ops exist precisely to
/// probe for one), so it has not been observed in any workload profiled so
/// far -- but a `while`-loop idiom built on `vle*ff` would hit it every
/// iteration that terminates.
#[derive(Serialize, Deserialize, Clone, Debug, JsonSchema, Default)]
pub struct RoiInfo {
    /// Human readable string of the function name. May be trimmed if extremely long (..C++)
    ///
    /// `Arc<str>` (not `String`): ROI names are assigned once and never mutated,
    /// but the whole `RoiInfo` is deep-cloned on every live snapshot merge (~259
    /// functions × thousands of batches). Cloning an `Arc<str>` is a refcount
    /// bump — no heap alloc/memcpy — versus a `String` clone per function per
    /// snapshot. Serializes transparently as a string (serde `rc` feature).
    pub name: Arc<str>,

    /// Whether this region was properly closed (end_region seen).
    /// Used to filter out regions that were opened but never closed.
    #[serde(skip)]
    pub completed: bool,

    /// Count of instructions executed
    pub instructions: u64,
    /// Count of branches executed (not hit/miss, just executed)
    pub branches: u64,

    /// Count of Vector Instructions
    pub vector_instructions: u64,
    /// Count of Vector Elements (all types, based on VL).
    ///
    /// "Element" means a memory/compute element, not a VL step. The two
    /// coincide for every instruction except the segment load/stores, where VL
    /// counts segments of `nf` fields each, so the op contributes `nf * VL`.
    /// Architectural VL itself is reported unscaled by `vl_avg_*`/`vl_max_*`.
    ///
    /// Whole-register ops contribute their spec-defined
    /// `evl = NFIELDS * VLEN / EEW` rather than `VL`, which they ignore.
    /// Mask loads/stores contribute `ceil(vl/8)` at EEW=8. With that, every
    /// vector memory op satisfies `bytes == elements * eew/8`.
    ///
    /// Overcounts masked ops, and fault-only-first loads that trim `vl` --
    /// see the accounting caveats on [`RoiInfo`].
    pub vector_elements: u64,

    /// Counts Floating point instructions (scalar and vector included), += 1 per instruction
    pub float_instructions: u64,
    /// Counts Float operations (incremented by 1 for every operation for every element)
    // Eg: += 1   if "fadd" executed (scalar)
    // Eg: += 10   if "vfadd.vv with VL 10" executed
    // Eg: += 20   if "vfmacc.vv with VL 10" executed
    pub float_elements: u64,

    /// Counts Integer Operations counted per element (same as above, but for integer)
    // pub int_elements: u64,

    /// Counts *FUSED* Multiply Accumulate INSTRUCTIONS (both int and float)
    pub fmacc_instructions: u64,
    /// Counts *FUSED* Multiply Accumulate ELEMENTS (both int and float)
    pub fmacc_elements: u64,

    /// Counts bytes loaded. Overcounts masked ops -- see the accounting
    /// caveats on [`RoiInfo`].
    pub load_bytes: u64,
    /// Counts bytes stored. Overcounts masked ops -- see the accounting
    /// caveats on [`RoiInfo`].
    pub store_bytes: u64,

    /// Cache-line hits and misses of this scope's **vector** loads and stores,
    /// from the modelled cache (`--cache-sim`). All zero
    /// when the run did not model a cache, which is the default.
    ///
    /// Vector data-side only: scalar accesses are not captured by the tracer at
    /// all, and instruction fetch never reaches the model. A hit rate computed
    /// from these describes the vector loads and stores of the region, not its
    /// memory behaviour as a machine would see it.
    #[serde(default, skip_serializing_if = "CacheCounts::is_zero")]
    pub cache: CacheCounts,

    /// Counts vector load instructions (at any LMUL, always += 1 per *instruction*)
    pub vector_load_instructions: u64,
    /// Counts vector load elements. Same element convention as
    /// [`RoiInfo::vector_elements`] -- segment loads contribute `nf * VL` --
    /// and the same masking/fault-only-first caveats.
    pub vector_load_elements: u64,

    /// Per-SEW VL histograms
    #[serde(skip)]
    #[serde(default = "default_histogram")]
    pub(crate) vl_histogram_e8: VectorLengthHistogram,

    #[serde(skip)]
    #[serde(default = "default_histogram")]
    pub(crate) vl_histogram_e16: VectorLengthHistogram,

    #[serde(skip)]
    #[serde(default = "default_histogram")]
    pub(crate) vl_histogram_e32: VectorLengthHistogram,

    #[serde(skip)]
    #[serde(default = "default_histogram")]
    pub(crate) vl_histogram_e64: VectorLengthHistogram,

    /// Full-resolution sparse histogram: each entry is [vl, count] for every VL
    /// value that was actually observed.  Serialised only when non-empty so the
    /// JSON stays compact when no vector instructions were executed.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub vl_hist_e8_raw: Vec<[u64; 2]>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub vl_hist_e16_raw: Vec<[u64; 2]>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub vl_hist_e32_raw: Vec<[u64; 2]>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub vl_hist_e64_raw: Vec<[u64; 2]>,

    // Per-SEW, per-LMUL sparse histograms (serialised only when non-empty)
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub vl_hist_e8_m1_raw: Vec<[u64; 2]>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub vl_hist_e8_m2_raw: Vec<[u64; 2]>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub vl_hist_e8_m4_raw: Vec<[u64; 2]>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub vl_hist_e8_m8_raw: Vec<[u64; 2]>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub vl_hist_e8_mf2_raw: Vec<[u64; 2]>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub vl_hist_e8_mf4_raw: Vec<[u64; 2]>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub vl_hist_e8_mf8_raw: Vec<[u64; 2]>,

    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub vl_hist_e16_m1_raw: Vec<[u64; 2]>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub vl_hist_e16_m2_raw: Vec<[u64; 2]>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub vl_hist_e16_m4_raw: Vec<[u64; 2]>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub vl_hist_e16_m8_raw: Vec<[u64; 2]>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub vl_hist_e16_mf2_raw: Vec<[u64; 2]>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub vl_hist_e16_mf4_raw: Vec<[u64; 2]>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub vl_hist_e16_mf8_raw: Vec<[u64; 2]>,

    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub vl_hist_e32_m1_raw: Vec<[u64; 2]>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub vl_hist_e32_m2_raw: Vec<[u64; 2]>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub vl_hist_e32_m4_raw: Vec<[u64; 2]>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub vl_hist_e32_m8_raw: Vec<[u64; 2]>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub vl_hist_e32_mf2_raw: Vec<[u64; 2]>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub vl_hist_e32_mf4_raw: Vec<[u64; 2]>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub vl_hist_e32_mf8_raw: Vec<[u64; 2]>,

    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub vl_hist_e64_m1_raw: Vec<[u64; 2]>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub vl_hist_e64_m2_raw: Vec<[u64; 2]>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub vl_hist_e64_m4_raw: Vec<[u64; 2]>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub vl_hist_e64_m8_raw: Vec<[u64; 2]>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub vl_hist_e64_mf2_raw: Vec<[u64; 2]>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub vl_hist_e64_mf4_raw: Vec<[u64; 2]>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub vl_hist_e64_mf8_raw: Vec<[u64; 2]>,

    /// Instruction mix: mnemonic (e.g. "vfmacc.vv", "vle8.v", "addi") mapped to
    /// the number of times that opcode executed within this ROI. Populated once
    /// at end of run from the per-PC `PcProfile` (see
    /// `set_instruction_mix_from_pc_profile`), so it is only present for scopes
    /// a `PcProfile` covers -- functions and the global scope, with annotate
    /// enabled. Sorted (BTreeMap) for stable, diffable output.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub instruction_mix: BTreeMap<String, u64>,
}

impl PartialEq for RoiInfo {
    fn eq(&self, other: &Self) -> bool {
        self.name == other.name
            && self.instructions == other.instructions
            && self.branches == other.branches
            && self.vector_instructions == other.vector_instructions
            && self.vector_elements == other.vector_elements
            && self.float_instructions == other.float_instructions
            && self.float_elements == other.float_elements
            && self.fmacc_instructions == other.fmacc_instructions
            && self.fmacc_elements == other.fmacc_elements
            && self.load_bytes == other.load_bytes
            && self.store_bytes == other.store_bytes
            && self.vector_load_instructions == other.vector_load_instructions
            && self.vector_load_elements == other.vector_load_elements
            && self.cache == other.cache
    }
}

fn default_histogram() -> VectorLengthHistogram {
    VectorLengthHistogram::new()
}

impl RoiInfo {
    /// Create a new RoiInfo with the given name and zero counters.
    /// Accepts `String`, `&str`, or `Arc<str>` (all `Into<Arc<str>>`).
    pub fn new(name: impl Into<Arc<str>>) -> Self {
        Self {
            name: name.into(),
            ..Default::default()
        }
    }

    /// Populate `instruction_mix` from a per-PC profile at finalize time.
    ///
    /// This is the *only* way `instruction_mix` is ever filled: the mix is
    /// derived once, at end of run, from the `PcProfile`'s per-PC execution
    /// counts (see `Enricher::apply_deferred_instruction_mix`). It is
    /// deliberately not accumulated live -- the per-instruction hashmap probe
    /// that used to do so measured as a very heavy runtime cost, and the per-PC
    /// counts the annotate report already gathers carry the same information.
    pub fn set_instruction_mix_from_pc_profile(&mut self, mix: BTreeMap<String, u64>) {
        self.instruction_mix = mix;
    }

    /// Recalculate intensity based on current float_ops and bytes_moved
    #[inline]
    pub fn recalculate_intensity(&self) -> f32 {
        let bytes_moved = self.load_bytes + self.store_bytes;
        if bytes_moved > 0 {
            self.float_elements as f32 / (bytes_moved as f32)
        } else {
            0_f32
        }
    }

    /// Update ROI statistics based on instruction characteristics
    ///
    /// # Arguments
    /// * `ins` - The instruction to analyze
    /// * `current_vl` - Optional current vector length (for vector instructions)
    /// * `current_sew` - Optional current SEW (0=e8, 1=e16, 2=e32, 3=e64)
    #[inline]
    pub fn update_stats(
        &mut self,
        ins: &riscv_isa::Instruction,
        current_vl: Option<u32>,
        current_sew: Option<u8>,
        current_lmul: Option<Lmul>,
    ) {
        // Single source of truth: build the classification descriptor from the
        // same `riscv_isa` predicates and delegate to the cached hot path. This
        // keeps `update_stats` (used by tests and non-hot callers) semantically
        // identical to `update_stats_cached` by construction.
        let vlen_bytes = SYSTEM_VLEN.get_bytes();
        let class = InsnClass::from_instruction(ins, vlen_bytes);
        self.update_stats_cached(
            &class,
            ins,
            current_vl,
            current_sew,
            current_lmul,
            vlen_bytes,
        );
    }

    /// Hot-path statistics update using a precomputed [`InsnClass`] descriptor.
    ///
    /// This does only integer arithmetic on the cached flags/bytes plus the
    /// histogram record -- no per-execution enum dispatch and no atomic VLEN
    /// load (`vlen_bytes` is hoisted by the caller). `ins` is still required for
    /// the (rare) VL-dependent `bytes_moved` computation on vector element/mask
    /// memory ops.
    ///
    /// Behaviour is bit-identical to the legacy inline `update_stats` body.
    #[inline]
    pub fn update_stats_cached(
        &mut self,
        class: &InsnClass,
        ins: &riscv_isa::Instruction,
        current_vl: Option<u32>,
        current_sew: Option<u8>,
        current_lmul: Option<Lmul>,
        vlen_bytes: u16,
    ) {
        let delta = InsnDelta::new(
            class,
            ins,
            current_vl,
            current_sew,
            current_lmul,
            vlen_bytes,
        );
        self.apply_delta(&delta);
    }

    /// Apply a precomputed per-execution [`InsnDelta`] to this scope's counters.
    ///
    /// Split out of `update_stats_cached` because `count_instruction` applies the
    /// *same* delta to every scope an instruction belongs to (its function, the
    /// global ROI, and each active ROI region) -- 2+ calls per instruction, all
    /// with identical inputs. Deriving `vl`/`elems`/`bytes_moved` once and
    /// replaying only the counter adds here removes that duplicated derivation;
    /// `update_stats_cached` + `count_instruction` together were ~13% of
    /// whole-process samples.
    #[inline]
    pub fn apply_delta(&mut self, delta: &InsnDelta) {
        let (vl, elems, bytes_moved) = (delta.vl, delta.elems, delta.bytes_moved);
        let class_flags = delta.flags;
        // Vector counters increment _elements with += VL
        // For scalar operations, add 1 to all *_elements counters
        // `vl` is the architectural vector length -- it drives the VL
        // histograms and `bytes_moved`. `elems` is how many memory/compute
        // elements the instruction actually touches, which differs only for
        // segment load/stores (`nf` fields per element step); see
        // `InsnClass::elem_mult`.
        let is_vector = class_flags & IC_VECTOR != 0;
        if is_vector {
            self.vector_instructions += 1;
            self.vector_elements += elems;

            // Route into the matching per-SEW histogram. Deliberately `vl`,
            // not `elems`: VL is architectural state and must not be inflated
            // by the segment field count.
            match delta.sew {
                Some(0) => self.vl_histogram_e8.record(vl, delta.lmul),
                Some(1) => self.vl_histogram_e16.record(vl, delta.lmul),
                Some(2) => self.vl_histogram_e32.record(vl, delta.lmul),
                Some(3) => self.vl_histogram_e64.record(vl, delta.lmul),
                _ => {} // unknown / >e64 SEW -- combined only
            }
        }

        if class_flags & IC_BRANCH != 0 {
            self.branches += 1;
        }

        // Float. These use `vl` rather than `elems` because no segment
        // load/store is a float or FMA op, so the two are always equal here --
        // keeping `vl` avoids implying a semantic that is never exercised.
        if class_flags & IC_FLOAT != 0 {
            self.float_instructions += 1;
            self.float_elements += vl;

            if class_flags & IC_FMA != 0 {
                self.fmacc_instructions += 1;
                self.fmacc_elements += vl;
                // FMA counts 2x for float_ops
                self.float_elements += vl;
            }
        }

        if class_flags & IC_INT_FMA != 0 {
            // fma but integer, still count under FMA
            self.fmacc_elements += vl;
        }

        // load bytes, instruction & element counts
        if class_flags & IC_LOAD != 0 {
            // bytes counters
            self.load_bytes += bytes_moved;

            // instruction & elements
            if is_vector {
                self.vector_load_instructions += 1;
                self.vector_load_elements += elems;
            }
        }

        // store bytes only
        if class_flags & IC_STORE != 0 {
            self.store_bytes += bytes_moved;
        }
    }

    /// Apply a whole straight-line run's aggregate contribution in one step.
    ///
    /// The batched counterpart of [`RoiInfo::apply_delta`]: it must leave this
    /// scope in exactly the state the N `apply_delta` calls it replaces would
    /// have (plus the `instructions` bump those callers do themselves, which is
    /// folded in here since a batched caller has no per-instruction loop left to
    /// do it in). That equivalence is not obvious from reading two functions, so
    /// it is pinned by the `run_delta_matches_per_insn_*` tests in
    /// `processing::enricher::run_cache`.
    ///
    /// The structural saving over the per-instruction path is the single
    /// histogram bump -- legal only because vtype is constant across a run, so
    /// every vector instruction in it lands in the same bucket.
    #[inline]
    pub(crate) fn add_run_delta(
        &mut self,
        delta: &crate::processing::enricher::run_cache::RunDelta,
    ) {
        self.instructions += delta.instructions;
        self.branches += delta.branches;
        self.vector_instructions += delta.vector_instructions;
        self.vector_elements += delta.vector_elements;
        self.float_instructions += delta.float_instructions;
        self.float_elements += delta.float_elements;
        self.fmacc_instructions += delta.fmacc_instructions;
        self.fmacc_elements += delta.fmacc_elements;
        self.vector_load_instructions += delta.vector_load_instructions;
        self.vector_load_elements += delta.vector_load_elements;
        self.load_bytes += delta.load_bytes;
        self.store_bytes += delta.store_bytes;

        if let Some((sew, lmul, vl, count)) = delta.vl_hist_bump {
            match sew {
                0 => self.vl_histogram_e8.record_n(vl, lmul, count),
                1 => self.vl_histogram_e16.record_n(vl, lmul, count),
                2 => self.vl_histogram_e32.record_n(vl, lmul, count),
                3 => self.vl_histogram_e64.record_n(vl, lmul, count),
                // Unreachable: the run cache only emits a bump for SEW <= 3,
                // mirroring `apply_delta`'s fall-through for unknown SEW.
                _ => {}
            }
        }
    }

    /// Update sparse histograms from the raw histograms.
    /// Exports combined (all-LMUL) and per-LMUL sparse histograms for each SEW.
    pub fn update_from_histogram(&mut self) {
        // Helper: export per-LMUL sparse data for a given histogram
        fn export_per_lmul(hist: &VectorLengthHistogram) -> [Vec<[u64; 2]>; 7] {
            [
                hist.to_sparse(Some(Lmul::M1.to_vlmul())),
                hist.to_sparse(Some(Lmul::M2.to_vlmul())),
                hist.to_sparse(Some(Lmul::M4.to_vlmul())),
                hist.to_sparse(Some(Lmul::M8.to_vlmul())),
                hist.to_sparse(Some(Lmul::Mf2.to_vlmul())),
                hist.to_sparse(Some(Lmul::Mf4.to_vlmul())),
                hist.to_sparse(Some(Lmul::Mf8.to_vlmul())),
            ]
        }

        if !self.vl_histogram_e8.is_empty() {
            self.vl_hist_e8_raw = self.vl_histogram_e8.to_sparse(None);
            let lmul = export_per_lmul(&self.vl_histogram_e8);
            self.vl_hist_e8_m1_raw = lmul[0].clone();
            self.vl_hist_e8_m2_raw = lmul[1].clone();
            self.vl_hist_e8_m4_raw = lmul[2].clone();
            self.vl_hist_e8_m8_raw = lmul[3].clone();
            self.vl_hist_e8_mf2_raw = lmul[4].clone();
            self.vl_hist_e8_mf4_raw = lmul[5].clone();
            self.vl_hist_e8_mf8_raw = lmul[6].clone();
        }
        if !self.vl_histogram_e16.is_empty() {
            self.vl_hist_e16_raw = self.vl_histogram_e16.to_sparse(None);
            let lmul = export_per_lmul(&self.vl_histogram_e16);
            self.vl_hist_e16_m1_raw = lmul[0].clone();
            self.vl_hist_e16_m2_raw = lmul[1].clone();
            self.vl_hist_e16_m4_raw = lmul[2].clone();
            self.vl_hist_e16_m8_raw = lmul[3].clone();
            self.vl_hist_e16_mf2_raw = lmul[4].clone();
            self.vl_hist_e16_mf4_raw = lmul[5].clone();
            self.vl_hist_e16_mf8_raw = lmul[6].clone();
        }
        if !self.vl_histogram_e32.is_empty() {
            self.vl_hist_e32_raw = self.vl_histogram_e32.to_sparse(None);
            let lmul = export_per_lmul(&self.vl_histogram_e32);
            self.vl_hist_e32_m1_raw = lmul[0].clone();
            self.vl_hist_e32_m2_raw = lmul[1].clone();
            self.vl_hist_e32_m4_raw = lmul[2].clone();
            self.vl_hist_e32_m8_raw = lmul[3].clone();
            self.vl_hist_e32_mf2_raw = lmul[4].clone();
            self.vl_hist_e32_mf4_raw = lmul[5].clone();
            self.vl_hist_e32_mf8_raw = lmul[6].clone();
        }
        if !self.vl_histogram_e64.is_empty() {
            self.vl_hist_e64_raw = self.vl_histogram_e64.to_sparse(None);
            let lmul = export_per_lmul(&self.vl_histogram_e64);
            self.vl_hist_e64_m1_raw = lmul[0].clone();
            self.vl_hist_e64_m2_raw = lmul[1].clone();
            self.vl_hist_e64_m4_raw = lmul[2].clone();
            self.vl_hist_e64_m8_raw = lmul[3].clone();
            self.vl_hist_e64_mf2_raw = lmul[4].clone();
            self.vl_hist_e64_mf4_raw = lmul[5].clone();
            self.vl_hist_e64_mf8_raw = lmul[6].clone();
        }
    }

    /// Merge all statistics from `other` into `self`.
    /// Both entries must have the same name (caller's responsibility).
    pub fn merge_from(&mut self, other: &RoiInfo) {
        self.add_ref(other);
    }

    /// Accumulate `other` into `self` **without cloning** `other`. This is the
    /// real body shared by `merge_from` and the by-value `AddAssign` (which used
    /// to force a full `RoiInfo` clone per merge just to satisfy its by-value
    /// signature). Counters are `Copy`, the histograms already merge by
    /// reference, and the maps are iterated by reference (the representative
    /// the maps are iterated by reference).
    pub fn add_ref(&mut self, other: &RoiInfo) {
        self.instructions += other.instructions;
        self.branches += other.branches;

        self.vector_instructions += other.vector_instructions;
        self.vector_elements += other.vector_elements;

        self.float_instructions += other.float_instructions;
        self.float_elements += other.float_elements;

        self.fmacc_instructions += other.fmacc_instructions;
        self.fmacc_elements += other.fmacc_elements;

        self.load_bytes += other.load_bytes;
        self.store_bytes += other.store_bytes;

        // VPU counters
        self.vector_load_instructions += other.vector_load_instructions;
        self.vector_load_elements += other.vector_load_elements;

        self.cache.add(&other.cache);

        // Merge per-SEW histograms (already by-reference)
        self.vl_histogram_e8.add(&other.vl_histogram_e8);
        self.vl_histogram_e16.add(&other.vl_histogram_e16);
        self.vl_histogram_e32.add(&other.vl_histogram_e32);
        self.vl_histogram_e64.add(&other.vl_histogram_e64);

        // Merge already-finalized instruction_mix maps: set directly from a
        // `PcProfile` at finalize time, this is the only place the per-opcode
        // counts live, so they must be summed explicitly here.
        for (mnemonic, count) in &other.instruction_mix {
            *self.instruction_mix.entry(mnemonic.clone()).or_insert(0) += *count;
        }

        // Recompute per-SEW avg/max from the merged histograms
        self.update_from_histogram();

        self.recalculate_intensity();
    }

    /// Print a summary of all ROIs in a formatted table
    pub fn print_roi_summary(rois: &[RoiInfo]) {
        println!("\n{:-<80}", "");
        println!("ROI Summary");
        println!("{:-<80}", "");
        for roi in rois {
            println!("  ROI: {}", roi.name);
            println!("  Instructions:    {:>12}", roi.instructions);
            println!("  Branches:        {:>12}", roi.branches);
            println!("  Float Insts:     {:>12}", roi.float_instructions);
            println!("  Float Ops:       {:>12}", roi.float_elements);
            println!("  FMA Insts:       {:>12}", roi.fmacc_instructions);
            println!("  FMA Ops:         {:>12}", roi.fmacc_elements);
            println!("  Vector Ops:      {:>12}", roi.vector_instructions);

            // println!("  Memory Insts:    {:>12}", roi.memory_instructions);
            println!("  Load Bytes:      {:>12}", roi.load_bytes);
            println!("  Store Bytes:     {:>12}", roi.store_bytes);
            println!("  FLOPs/Byte:      {:>12.4}", roi.recalculate_intensity());

            // Print per-SEW sample counts
            println!("  {:>6}  {:>8}", "SEW", "samples");
            println!("  {:>6}  {:>8}", "e8", roi.vl_histogram_e8.len());
            println!("  {:>6}  {:>8}", "e16", roi.vl_histogram_e16.len());
            println!("  {:>6}  {:>8}", "e32", roi.vl_histogram_e32.len());
            println!("  {:>6}  {:>8}", "e64", roi.vl_histogram_e64.len());

            // Instruction mix, sorted by count descending (top opcodes first)
            if !roi.instruction_mix.is_empty() {
                println!("  Instruction mix:");
                let mut mix: Vec<(&String, &u64)> = roi.instruction_mix.iter().collect();
                mix.sort_by(|a, b| b.1.cmp(a.1).then(a.0.cmp(b.0)));
                for (mnemonic, count) in mix {
                    println!("    {:<16} {:>12}", mnemonic, count);
                }
            }
            println!();
        }
    }

    /// Assert that this RoiInfo equals another, with detailed error messages for testing
    ///
    /// # Arguments
    /// * `expected` - The expected RoiInfo to compare against
    /// * `test_name` - Name of the test (for error messages)
    ///
    /// # Panics
    /// Panics with a detailed message if any metric doesn't match
    pub fn assert_equals(&self, expected: &RoiInfo, test_name: &str) {
        assert_eq!(
            self.instructions, expected.instructions,
            "Instructions mismatch for {}",
            test_name
        );
        assert_eq!(
            self.branches, expected.branches,
            "Branches mismatch for {}",
            test_name
        );
        assert_eq!(
            self.vector_instructions, expected.vector_instructions,
            "Vector instructions mismatch for {}",
            test_name
        );
        assert_eq!(
            self.vector_elements, expected.vector_elements,
            "Vector elements mismatch for {}",
            test_name
        );
        assert_eq!(
            self.float_instructions, expected.float_instructions,
            "Float instructions mismatch for {}",
            test_name
        );
        assert_eq!(
            self.float_elements, expected.float_elements,
            "Float elements mismatch for {}",
            test_name
        );
        assert_eq!(
            self.fmacc_instructions, expected.fmacc_instructions,
            "FMACC instructions mismatch for {}",
            test_name
        );
        assert_eq!(
            self.fmacc_elements, expected.fmacc_elements,
            "FMACC elements mismatch for {}",
            test_name
        );
        assert_eq!(
            self.load_bytes, expected.load_bytes,
            "load_bytes mismatch for {}",
            test_name
        );
        assert_eq!(
            self.store_bytes, expected.store_bytes,
            "store_bytes mismatch for {}",
            test_name
        );
        assert_eq!(
            self.vector_load_instructions, expected.vector_load_instructions,
            "Vector load instructions mismatch for {}",
            test_name
        );
        assert_eq!(
            self.vector_load_elements, expected.vector_load_elements,
            "Vector load elements mismatch for {}",
            test_name
        );
    }
}

// Allow "Adding" RoiInfos together, to increment the counters
impl Add for RoiInfo {
    type Output = RoiInfo;

    fn add(self, rhs: Self) -> Self::Output {
        let mut new = self.clone();
        new += rhs;
        new
    }
}

impl AddAssign for RoiInfo {
    fn add_assign(&mut self, rhs: Self) {
        // Delegate to the by-reference body; no need to consume `rhs`.
        self.add_ref(&rhs);
    }
}

#[cfg(test)]
mod instruction_mix_tests {
    use super::*;

    /// The mix is a finalize-time artifact: nothing on the counting path fills
    /// it, `set_instruction_mix_from_pc_profile` does, merges sum it, and a
    /// serde roundtrip preserves it.
    #[test]
    fn mix_is_finalize_only_and_merges() {
        let mut a = RoiInfo::new("a".to_string());
        SYSTEM_VLEN.set_bytes(16);
        a.update_stats(
            &riscv_isa::Instruction::ADDI {
                rd: 3,
                rs1: 0,
                imm: 10,
            },
            None,
            None,
            None,
        );
        a.update_from_histogram();
        assert!(
            a.instruction_mix.is_empty(),
            "the counting path must not populate the mix; it is derived at end of run"
        );

        a.set_instruction_mix_from_pc_profile(
            [("addi".to_string(), 2u64), ("bne".to_string(), 1)].into(),
        );

        let mut b = RoiInfo::new("a".to_string());
        b.set_instruction_mix_from_pc_profile(
            [("addi".to_string(), 3u64), ("sub".to_string(), 1)].into(),
        );

        a += b;
        assert_eq!(
            a.instruction_mix,
            [
                ("addi".to_string(), 5u64),
                ("bne".to_string(), 1),
                ("sub".to_string(), 1)
            ]
            .into(),
            "merging must sum per-mnemonic counts"
        );

        let json = serde_json::to_string(&a).expect("serialize");
        // nosemgrep: deserialization-from-untrusted-source -- test-only roundtrip of a value built three lines above
        let mut loaded: RoiInfo = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(loaded.instruction_mix, a.instruction_mix);
        // Re-finalizing a loaded ROI must leave the loaded mix untouched.
        loaded.update_from_histogram();
        assert_eq!(loaded.instruction_mix, a.instruction_mix);
    }

    /// An ROI with no mix omits the field from JSON entirely
    /// (`skip_serializing_if`), which is what keeps region-scoped ROIs -- with
    /// no `PcProfile` to derive from -- out of the output.
    #[test]
    fn empty_mix_is_omitted_from_json() {
        let roi = RoiInfo::new("test".to_string());
        let json = serde_json::to_string(&roi).expect("serialize");
        assert!(!json.contains("\"instruction_mix\""));
    }
}
