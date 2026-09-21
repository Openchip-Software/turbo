//! Main instruction enrichment orchestrator.
//!
//! The [`Enricher`] is the core of per-hart trace processing. It coordinates:
//! - Instruction decoding and ROI marker detection
//! - Vector length state tracking
//! - Region-of-interest stack management
//! - Statistics accumulation (functions, regions, memory operations)
//! - Optional assembly capture to .asm files
//! - Perfetto trace event emission
//!
//! Each hart maintains its own Enricher instance with independent state.
//! Results from all harts are merged at the end of tracing.

use anyhow::Result;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use turbo_tracer_shared::RoiRtMarker;

use crate::common::{PerformanceData, RoiInfo};

/// One open ROI begin: `(begin_instr, flame_node)`, carried from the begin
/// marker to the end marker that closes it -- the region's start of the
/// instruction window, and its node in the live flamegraph call tree.
///
/// Deliberately just these two: this is pushed and popped once per ROI
/// enter/exit, so on a hot ROI loop every extra field is a per-event cost.
/// The begin marker's ordinal, PC and kind (and the region's parent id and
/// depth) used to ride along here for the offline `RoiExecution` records, which
/// the streaming builder replaced -- carrying the region *name* alone cost a
/// heap allocation and a memcpy per region entry.
type OpenRoiStart = (u64, crate::processing::roi_flamegraph::FlameNodeIdx);

/// Open ROI begins keyed by region_id. Each entry is a stack, so
/// nested/recursive entries of the same region pair with their matching end.
///
/// `FxHashMap` rather than `std`'s: the key is a small integer probed twice per
/// ROI enter/exit, so SipHash-ing it is pure overhead.
type OpenRoiStarts = rustc_hash::FxHashMap<usize, Vec<OpenRoiStart>>;

use crate::processing::enricher::SharedTrace;
use crate::processing::function_lookup::{ElfResolver, MultiResolver};
use crate::processing::instruction_decoder::{MultiDecoder, RoiMarker};
use crate::processing::{BbtBlock, ElfMetadata};

/// The side-channel inputs a decoded batch carries into the enricher, alongside
/// the trace itself (PCs and VLs, or BBT blocks).
///
/// These share a delivery rule that is easy to get wrong when they are separate
/// positional parameters: a single producer flush window may decode into several
/// batches, but its side-channels are per-window aggregates that must be
/// consumed exactly ONCE. So they ride with the window's first batch and every
/// later batch of that window gets [`EMPTY`](Self::EMPTY). Grouping them makes
/// that one decision on one value (see [`take_once`](Self::take_once)) rather
/// than a tuple of slices that has to be assembled identically at each site.
///
/// Adding a side-channel is then a field here, not a new parameter on
/// [`Enricher::transform_optimized`], [`Enricher::transform_blocks`] and
/// `begin_batch` plus every caller.
#[derive(Debug, Clone, Copy, Default)]
pub struct BatchSideChannels<'a> {
    /// ROI runtime markers delivered with this batch. Appended to the
    /// enricher's persistent per-PC FIFO, which is consumed across batches.
    pub roi_markers: &'a [RoiRtMarker],
}

impl<'a> BatchSideChannels<'a> {
    /// No side-channels: a later batch of a window whose first batch already
    /// consumed them, or a caller (bench, test) that has none.
    pub const EMPTY: Self = Self { roi_markers: &[] };

    /// Yield `self` the first time and [`EMPTY`](Self::EMPTY) thereafter,
    /// flipping `first` as it goes. This is the once-per-window delivery rule;
    /// call it once per batch.
    pub fn take_once(self, first: &mut bool) -> Self {
        if std::mem::replace(first, false) {
            self
        } else {
            Self::EMPTY
        }
    }
}

use super::marker_handler::{MarkerCtx, MarkerHandler};
use super::pc_profile::PcProfile;
use super::region_tracker::RegionTracker;
use super::stats_accumulator::StatsAccumulator;
use super::trace_emitter::TraceEmitter;
use super::vl_tracker::VlTracker;

/// The program image every hart's enricher reads: one decoder and one symbol
/// resolver per executable module.
///
/// Both are immutable once built and identical for every hart, so they are
/// built once per run and shared. Building them per hart cost a full decode
/// table (~26x the module's `.text` size) and a full DWARF parse per hart --
/// which, at up to 8 decode workers, dominated startup and resident memory.
///
/// Held as a pair rather than two independent handles because their module
/// ordering must stay aligned: `decoder.dec[i]` and resolver module `i` are
/// the same module, and only building them in one loop guarantees it.
pub struct Modules {
    pub decoder: MultiDecoder,
    pub resolver: MultiResolver,
}

impl Modules {
    /// Build one decoder+resolver pair per executable segment (main binary,
    /// each shared library, VDSO).
    ///
    /// Pushing to both lists only when the decoder itself parses successfully
    /// is what keeps them in lockstep, so a PC's decoder index also selects
    /// its resolver. Without per-module resolvers, only the main binary's own
    /// symbols were ever resolved and every shared-library address fell
    /// through to the "Global" catch-all bucket.
    pub fn from_metadata(metadata: &ElfMetadata) -> Result<Self> {
        let mut dec = Vec::new();
        let mut resolvers = Vec::new();
        for (base_addr, data) in metadata.iterate_executable_segments() {
            match crate::processing::instruction_decoder::InstructionDecoder::new(base_addr, &data)
            {
                Ok(d) => {
                    dec.push(d);
                    resolvers.push(ElfResolver::new_at(&data, base_addr).unwrap_or_else(|e| {
                        log::warn!(
                            "Failed to build symbol resolver for module at base 0x{:x}: {}",
                            base_addr,
                            e
                        );
                        ElfResolver::empty()
                    }));
                }
                Err(e) => log::warn!("Skipping invalid segment at 0x{:x}: {}", base_addr, e),
            }
        }
        Ok(Modules {
            decoder: MultiDecoder { dec },
            resolver: MultiResolver::new(resolvers),
        })
    }
}

/// Pipeline for processing trace data into enriched DataFrames
pub struct Enricher {
    /// Program image, shared with every other hart's enricher.
    modules: Arc<Modules>,
    /// Vector length and SEW state tracker
    vl_tracker: VlTracker,
    /// VL values received but not yet consumed.
    ///
    /// VL values arrive batched per record-log batch, but the trace is decoded
    /// as one continuous stream whose PCs no longer partition on those
    /// boundaries (a vsetvl near a flush boundary may be decoded in the next
    /// batch). So VL values are consumed continuously from this queue across
    /// `transform_optimized` calls rather than being matched to a single
    /// batch; unconsumed values are retained for the next call.
    vl_pending: Vec<u16>,
    /// ROI runtime markers received but not yet consumed, keyed by PC.
    ///
    /// Like [`vl_pending`](Self::vl_pending), ROI markers arrive batched per
    /// record-log batch while the trace is decoded as one continuous stream whose
    /// PCs no longer partition on record / 100K-PC-batch boundaries. A single
    /// record-log batch may be decoded into several batches, each calling
    /// `transform_optimized`. Keeping a persistent per-PC FIFO here (rather than
    /// rebuilding it from the full marker list on every call) ensures each marker
    /// is consumed exactly once, in execution order, even when its PC lands in a
    /// later batch than the record boundary that delivered the marker.
    marker_pending: std::collections::HashMap<u64, std::collections::VecDeque<RoiRtMarker>>,
    /// ROI marker processor
    marker_handler: MarkerHandler,
    /// Region stack and event-based region manager
    region_tracker: RegionTracker,
    /// Statistics accumulator (functions, regions, global ROI)
    stats: StatsAccumulator,
    /// Synthetto trace emitter
    trace: TraceEmitter,
    /// Directory the annotation report's `html/` subdirectory is written into
    /// (set via [`with_output_dir`](Self::with_output_dir)).
    output_dir: Option<PathBuf>,
    /// Per-PC execution profile for the source & assembly annotation view
    /// (opt-in; off unless annotation output is requested).
    pc_profile: PcProfile,
    /// Cache model for this hart's vector memory accesses. `None` -- the
    /// default -- makes the whole feature one `is_none` per block, and the
    /// block's captured footprint is never walked.
    ///
    /// One per hart, never shared: a shared level would need a cross-hart access
    /// order, and per-hart trace logs carry none.
    cache_sim: Option<crate::processing::cache_sim::CacheSim>,
    /// The last function [`model_block_cache`](Self::model_block_cache)
    /// resolved: `(address range, name index, name)`.
    ///
    /// `resolve_to_name_index_and_range` is a `BTreeMap` range walk plus a
    /// name-to-index hash lookup, and it already computes the containing range
    /// -- so the next access, which is nearly always in the same function, can
    /// be answered by a pair of comparisons instead. The name is owned because
    /// a borrow of the resolver cannot live in `self`; it is cloned only when
    /// the memo actually misses, i.e. once per function transition.
    cache_func_memo: Option<(std::ops::Range<u64>, usize, String)>,
    /// Counter for generating unique ROI region IDs
    next_region_id: usize,
    /// Open ROI begins keyed by region_id. Each entry is a stack, so
    /// nested/recursive entries of the same region pair with their matching end.
    open_roi_starts: OpenRoiStarts,
    /// Streaming ROI flamegraph builder for this hart, merged across harts at
    /// end of run. Bounded by unique call paths, not by execution count.
    roi_flame_builder: crate::processing::roi_flamegraph::StreamingRoiFlamegraph,
    /// Whether to feed the flamegraph builder at all
    /// (`--no-roi-flamegraph` clears it). See
    /// [`with_roi_flamegraph`](Self::with_roi_flamegraph).
    roi_flamegraph: bool,
    /// Hart index this Enricher instance processes (set via [`with_hart_id`](Self::with_hart_id)).
    hart_id: u32,
    /// Source-file search roots (`--src-dir`) used by the annotation report to
    /// locate sources whose DWARF-encoded build-host path is absent locally.
    source_dirs: Vec<PathBuf>,
    /// Memoized straight-line run aggregates, one cache per entry in
    /// `multi_decoder.dec` (they are keyed by PC alone, and the same PC in two
    /// modules is two different instructions).
    ///
    /// Built lazily on the first `transform_optimized` call rather than in
    /// `new()`: each cache folds in the run-fixed `SYSTEM_VLEN`, which is set
    /// during QEMU-runner construction and so is not reliably known yet when the
    /// `Enricher` is created. Empty means "not built yet".
    run_caches: Vec<super::run_cache::RunCache>,
    /// Memoized per-BBT-block aggregates, indexed by dictionary ID. Unused (and
    /// empty) on the E-Trace path, which has no block identities. Built lazily
    /// alongside `run_caches`, for the same `SYSTEM_VLEN` reason.
    block_cache: super::block_cache::BlockCache,
    /// Fast-path coverage counters, reported once per hart at `debug` level.
    fast_stats: FastPathStats,
}

/// Diagnostics for the batched fast path. `mismatches` is the interesting one:
/// it counts runs whose predicted PC sequence did *not* match the trace, which
/// on a clean QEMU trace should be ~0 -- a nonzero count means either a real
/// trap/interrupt mid-run or a bug in the run model.
#[derive(Default)]
struct FastPathStats {
    /// Instructions committed through the batched path.
    fast_instrs: u64,
    /// Instructions committed one at a time.
    slow_instrs: u64,
    /// Run aggregates applied.
    runs_applied: u64,
    /// Validation slice compares that failed.
    mismatches: u64,
    /// Runs declined because they overran the end of the available PC window
    /// (the E-Trace batch, or the BBT block).
    batch_overruns: u64,
    /// Runs declined because their last PC left the start PC's function range.
    func_straddles: u64,
    /// BBT block entries committed. Zero on the E-Trace path.
    blocks_seen: u64,
    /// BBT block entries that had no memo and were committed one instruction at
    /// a time. Nonzero only when assembly capture is on, or when a block's memo
    /// could not be built at all (see `BlockCache::intern`).
    blocks_unmemoized: u64,
    /// VL values handed to this enricher, over every batch. Reported as trace
    /// metadata; counted here because the BBT path receives them one at a time,
    /// attached to the block that produced them.
    vl_values: u64,
}

/// Per-batch loop state, shared by both entry points.
///
/// These were locals of one monolithic loop. They are grouped here so the fast
/// path and the per-instruction path can be methods on `Enricher` — which is what
/// lets the E-Trace batch and the BBT block iterator drive the *same* code
/// instead of two copies of it — while still carrying state (the cached function
/// range, the active-region list, the VL cursor) across every instruction of the
/// batch. Reused buffers rather than fresh allocations, exactly as before.
struct LoopCtx {
    /// VL values delivered but not yet claimed by a `vsetvl*`, and the cursor
    /// into them. Taken from `Enricher::vl_pending` for the batch and handed back
    /// (minus what was consumed) by `end_batch`.
    vl_pending: Vec<u16>,
    vl_index: usize,
    /// Per-PC FIFO of ROI runtime markers, taken from `Enricher::marker_pending`.
    markers_by_pc: std::collections::HashMap<u64, std::collections::VecDeque<RoiRtMarker>>,
    /// Run-fixed `SYSTEM_VLEN`, read once per batch.
    vlen_bytes: u16,
    /// Function range containing the last resolved PC, reused while it holds.
    cached_func_range: Option<(std::ops::Range<u64>, String, usize)>,
    /// Currently-active ROI region IDs; rebuilt only when `region_ids_dirty`.
    active_region_ids: Vec<usize>,
    /// Region IDs active immediately before the marker being processed.
    pre_marker_region_ids: Vec<usize>,
    /// Whether `active_region_ids` needs rebuilding.
    region_ids_dirty: bool,
}

impl Enricher {
    /// Create a new Enricher with the given ELF metadata, building its own
    /// private [`Modules`].
    ///
    /// Prefer [`with_modules`](Self::with_modules) when more than one enricher
    /// is created for the same program — building `Modules` is the expensive
    /// part of enricher construction, and every hart's copy would be
    /// identical.
    ///
    /// # Arguments
    /// * `metadata` - ELF metadata containing executable segments and symbols
    ///
    /// # Returns
    /// A new Enricher instance ready to process trace data
    pub fn new(metadata: &ElfMetadata) -> Result<Self> {
        Ok(Self::with_modules(Arc::new(Modules::from_metadata(
            metadata,
        )?)))
    }

    /// Create an Enricher over an already-built, shared program image.
    pub fn with_modules(modules: Arc<Modules>) -> Self {
        Enricher {
            modules,
            vl_tracker: VlTracker::new(),
            vl_pending: Vec::new(),
            marker_pending: std::collections::HashMap::new(),
            marker_handler: MarkerHandler::new(),
            region_tracker: RegionTracker::new(),
            stats: StatsAccumulator::new(),
            trace: TraceEmitter::new(),
            output_dir: None,
            pc_profile: PcProfile::new(false), // disabled by default
            cache_sim: None,                   // disabled by default
            cache_func_memo: None,
            next_region_id: 0,
            open_roi_starts: rustc_hash::FxHashMap::default(),
            roi_flame_builder: crate::processing::roi_flamegraph::StreamingRoiFlamegraph::default(),
            roi_flamegraph: true,
            source_dirs: Vec::new(),
            hart_id: 0,
            run_caches: Vec::new(),
            block_cache: super::block_cache::BlockCache::new(0),
            fast_stats: FastPathStats::default(),
        }
    }

    // ==================== Builder Methods ====================

    /// Set the output directory. The annotation report's `html/` subdirectory is
    /// written beneath it; without one, the report is rendered for the HTTP
    /// server but not persisted to disk.
    pub fn with_output_dir(mut self, dir: PathBuf) -> Self {
        self.output_dir = Some(dir);
        self
    }

    /// Enable or disable the per-PC source & assembly annotation profile.
    ///
    /// When enabled, the enricher accumulates a [`PcProfile`] alongside the
    /// aggregate counters; [`build_annotation_report`](Self::build_annotation_report)
    /// turns it into gzipped `annotate_<func>.html.gz` files.
    pub fn with_annotate(mut self, enabled: bool) -> Self {
        self.pc_profile = PcProfile::new(enabled);
        // `instruction_mix` for function/global scopes is derived from this
        // profile at finalize time -- see `apply_deferred_instruction_mix`.
        self
    }

    /// Model a cache over this hart's vector memory accesses.
    ///
    /// `vlenb` is `VLEN` in bytes, which the footprint replay needs for
    /// whole-register ops (whose size is `vl`-independent). Fails only if the
    /// cache configuration is invalid, which is checked once here so the access
    /// path cannot fail.
    pub fn with_cache_sim(
        mut self,
        vlenb: u64,
        config: crate::processing::cache_model::Config,
    ) -> Result<Self> {
        self.cache_sim = Some(crate::processing::cache_sim::CacheSim::new(
            self.modules.clone(),
            vlenb,
            config,
        )?);
        Ok(self)
    }

    /// Enable or disable ROI flamegraph accumulation (default: enabled).
    ///
    /// Aggregation is streaming and bounded by the number of unique ROI call
    /// paths, so it is cheap enough to leave on: one integer-keyed hash probe
    /// per region entry, two integer adds per region exit. This opt-out exists
    /// for `--no-roi-flamegraph`, i.e. for a workload whose ROI markers fire
    /// often enough that even that is worth declining.
    pub fn with_roi_flamegraph(mut self, enabled: bool) -> Self {
        self.roi_flamegraph = enabled;
        self
    }

    /// Set the source-file search roots (`--src-dir`) used to locate sources
    /// for the annotation report when the DWARF build-host path is absent
    /// locally. See [`super::annotate`] for the suffix-matching resolution.
    pub fn with_source_dirs(mut self, dirs: Vec<PathBuf>) -> Self {
        self.source_dirs = dirs;
        self
    }

    /// Attach a shared trace so all harts write into one combined file
    pub fn with_shared_trace(mut self, shared: Arc<Mutex<SharedTrace>>) -> Self {
        self.trace = self.trace.with_shared_trace(shared);
        self
    }

    /// Configure the trace flush threshold (number of packets before flushing)
    pub fn with_trace_flush_threshold(mut self, threshold: usize) -> Self {
        self.trace = self.trace.with_flush_threshold(threshold);
        self
    }

    /// Configure the counter track emit interval (emit every N instructions)
    pub fn with_trace_counter_emit_interval(mut self, interval: u64) -> Self {
        self.trace = self.trace.with_counter_emit_interval(interval);
        self
    }

    /// Set the trace file path and enable periodic flushing
    pub fn set_trace_file_path(&mut self, path: PathBuf) {
        self.trace.set_file_path(path);
    }

    /// Builder: set the hart ID on the underlying TraceEmitter.
    pub fn with_hart_id(mut self, hart_id: u32) -> Self {
        self.trace = self.trace.with_hart_id(hart_id);
        self.hart_id = hart_id;
        self
    }

    // ==================== Main Processing Loop ====================

    /// Process a batch of program counters and vector length values (E-Trace).
    ///
    /// # Arguments
    /// * `pcs` - Array of program counter values
    /// * `vl_values` - Array of vector length values (from vsetvl* instructions)
    /// * `side` - Per-batch side-channels; see [`BatchSideChannels`]
    #[hotpath::measure]
    pub fn transform_optimized(
        &mut self,
        pcs: &[u64],
        vl_values: &[u16],
        side: BatchSideChannels<'_>,
    ) -> Result<()> {
        log::trace!(
            "transform_optimized: processing {} PCs, {} VL values, current_vl={:?}",
            pcs.len(),
            vl_values.len(),
            self.vl_tracker.current_vl
        );

        if pcs.is_empty() {
            log::warn!("transform_optimized called with empty PC array!");
            return Ok(());
        }

        let mut ctx = self.begin_batch(vl_values, side)?;
        self.commit_window(&mut ctx, pcs)?;
        self.end_batch(ctx)
    }

    /// Process one BBT batch: block entries whose PCs are *borrowed* from the
    /// decoder's dictionary arena, memoized per block.
    ///
    /// Nothing is copied out of the dictionary: expanding block records into PC
    /// arrays was 192 MB of `memcpy` per `flash_full` run, and the enricher threw
    /// nearly all of it away — it only read the PCs back to confirm a static
    /// prediction of the very sequence the producer had just told it.
    ///
    /// So this path does not predict. [`BlockCache`](super::block_cache::BlockCache)
    /// splits each block, once, into memoizable stretches plus the individual
    /// instructions whose dynamic effect an aggregate cannot carry (a `vsetvl*`, a
    /// ROI marker), and every later execution of that block is one dense array
    /// index and one `add_run_delta` per scope. The counting is identical to
    /// [`transform_optimized`](Self::transform_optimized)'s: the same
    /// `RunDelta::build`, the same per-instruction path for everything a memo
    /// cannot cover.
    ///
    /// A block's trailing `vl` (see [`BbtBlock::vl`]) is queued *before* the
    /// block's instructions are committed, so the `vsetvl*` -- or the `vl*ff`
    /// -- that ends the block finds it in the pending queue
    /// exactly as an E-Trace batch's inline CSR values would have been.
    #[hotpath::measure]
    pub fn transform_blocks<'a>(
        &mut self,
        blocks: impl Iterator<Item = std::result::Result<BbtBlock<'a>, crate::RerfError>>,
        side: BatchSideChannels<'_>,
    ) -> Result<()> {
        let mut ctx = self.begin_batch(&[], side)?;
        for block in blocks {
            let block = block?;
            if let Some(vl) = block.vl {
                ctx.vl_pending.push(vl);
                self.fast_stats.vl_values += 1;
            }
            self.fast_stats.blocks_seen += 1;
            self.model_block_cache(&block, &ctx)?;
            self.commit_block(&mut ctx, block.id, block.pcs)?;
        }
        self.end_batch(ctx)
    }

    /// Feed this block's captured memory footprint to the cache model, and
    /// attribute what comes back per PC and per active region.
    ///
    /// Runs *before* [`commit_block`](Self::commit_block), for two reasons that
    /// pull the same way: the region stack must be the one in force on entry to
    /// the block (a marker inside it takes effect for later blocks), and the
    /// model must see blocks in execution order, which is exactly the order this
    /// loop has.
    ///
    /// Cache counts deliberately do not travel through `InsnDelta`/`RunDelta`:
    /// they are a function of the model's state at the moment of the access, not
    /// of the instruction, so they cannot be memoized with the rest of a run.
    /// Keeping them on their own path is what lets the run cache stay correct
    /// with cache modelling enabled.
    fn model_block_cache(&mut self, block: &BbtBlock<'_>, ctx: &LoopCtx) -> Result<()> {
        let Some(sim) = self.cache_sim.as_mut() else {
            return Ok(());
        };
        let (stats, pc_profile, resolver, memo) = (
            &mut self.stats,
            &mut self.pc_profile,
            &self.modules.resolver,
            &mut self.cache_func_memo,
        );
        let active = &ctx.active_region_ids;
        sim.block(block, |pc, counts| {
            let counts = crate::common::CacheCounts::from(counts);
            pc_profile.add_cache_counts(pc, &counts);
            // Resolved here rather than reusing the commit path's cached range:
            // this runs before that resolution happens. It gets its own memo,
            // on the range the resolver already returns -- consecutive vector
            // memory ops are nearly always in the same function.
            if !matches!(memo, Some((range, _, _)) if range.contains(&pc)) {
                *memo = resolver
                    .resolve_to_name_index_and_range(pc)
                    .map(|(name, index, range)| (range, index, name.to_string()));
            }
            match memo {
                Some((_, index, name)) => {
                    stats.count_cache(&counts, Some(*index), Some(name), active)
                }
                None => stats.count_cache(&counts, None, None, active),
            }
        })
    }

    /// Commit one BBT block: every memoized segment applied wholesale, everything
    /// else through the per-instruction path.
    fn commit_block(&mut self, ctx: &mut LoopCtx, id: u32, pcs: &[u64]) -> Result<()> {
        // A block whose memo could not be built (undecodable PC, straddled
        // function, ...) routes to the per-instruction path, which is always
        // correct.
        let module_index =
            self.block_cache
                .module_index(id, pcs, &self.modules.decoder, &self.modules.resolver);
        let Some(module_index) = module_index else {
            self.fast_stats.blocks_unmemoized += 1;
            return self.commit_window(ctx, pcs);
        };

        for si in 0..self.block_cache.segment_count(id) {
            let (off, len) = match self.block_cache.segment_kind(id, si) {
                // A `vsetvl*`, a `vl*ff`, or an ROI marker: its dynamic effect
                // is what the per-instruction path exists for.
                super::block_cache::SegmentKind::Single(off) => {
                    self.commit_one(ctx, module_index, pcs, off as usize)?;
                    continue;
                }
                super::block_cache::SegmentKind::Run { off, len } => (off, len),
            };

            // The three values `InsnDelta::new` consumes, re-read per segment
            // because a `Single` between two segments can have changed them.
            let key = super::run_cache::VtypeKey {
                vl: self.vl_tracker.current_vl,
                sew: self.vl_tracker.current_sew,
                lmul: self.vl_tracker.current_lmul,
            };

            // The aggregate is applied to whatever scopes are active, so the
            // cached ID list has to be current first — same rebuild the
            // per-instruction path does at Step 3, on the same dirty flag (only
            // markers change the set, and a marker is never inside a segment).
            if ctx.region_ids_dirty {
                Self::rebuild_active_region_ids(&self.region_tracker, &mut ctx.active_region_ids);
                ctx.region_ids_dirty = false;
            }

            let Some((delta, n, func_index, func_name)) = self.block_cache.apply(id, si, key)
            else {
                // Memo full (more than MAX_MEMO_ENTRIES distinct vtypes for this
                // segment): commit its instructions individually this time.
                for i in off..off + len {
                    self.commit_one(ctx, module_index, pcs, i as usize)?;
                }
                continue;
            };

            // One `add_run_delta` per scope instead of N `apply_delta`s. The whole
            // segment's counters land here *before* the next segment runs, so a
            // marker in a following `Single` still reads a global instruction
            // count covering precisely the instructions committed before it —
            // which is what `record_roi_start`/`record_roi_end` and the `entry.4`
            // re-snapshot use to delimit `[begin, end)` windows.
            self.stats
                .count_run(delta, func_index, func_name, &ctx.active_region_ids);

            // Trace emission stays per instruction in *effect*: `tick_n` emits at
            // exactly the instruction counts N `tick()`s would have, so the
            // Perfetto `.pb` is byte-identical, without running the loop.
            self.trace.tick_n(n as u64, self.stats.global_roi());

            // Note what this deliberately does *not* touch: `vl_pending` /
            // `vl_index` (nothing that writes `vl` is inside a segment, so the VL
            // queue cannot be consumed here) and the marker/trace-region state.
            self.fast_stats.fast_instrs += n as u64;
            self.fast_stats.runs_applied += 1;
        }
        Ok(())
    }

    /// Per-batch preamble shared by both entry points: ROI / VL intake,
    /// lazily built run caches, and the reusable loop buffers.
    fn begin_batch(&mut self, vl_values: &[u16], side: BatchSideChannels<'_>) -> Result<LoopCtx> {
        // Append the newly-delivered ROI runtime markers to the persistent
        // per-PC FIFO. Several distinct markers can share the same PC (e.g. a
        // single RAVE user-API helper like rave_name_value / rave_event_and_value
        // called multiple times), each carrying different rs1/rs2 values, so we
        // keep a FIFO queue per PC and pop them in execution order as the PCs
        // recur in the trace stream.
        //
        // The queue is stored on `self` and consumed across calls: a single
        // datagram's markers are delivered once (with the datagram's first
        // batch) but its PCs may span several batches / calls. Rebuilding
        // the map from the full datagram list on every call — as this previously
        // did — re-consumed already-processed markers from the front, handing the
        // wrong rs1/rs2/name to every marker after the first batch and corrupting
        // region names (spurious "end region without matching begin" warnings).
        let mut markers_by_pc = std::mem::take(&mut self.marker_pending);
        for m in side.roi_markers {
            markers_by_pc.entry(m.pc).or_default().push_back(m.clone());
        }

        // Initialize trace generation on first call
        self.trace.init()?;

        // Append this datagram's VL values to the pending queue and consume the
        // queue continuously: the trace decodes as one stream, so a vsetvl
        // whose VL value arrived in an earlier datagram may only now be decoded.
        // Take the queue into a local so it can be borrowed alongside &mut self.
        self.vl_pending.extend_from_slice(vl_values);
        self.fast_stats.vl_values += vl_values.len() as u64;
        let vl_pending = std::mem::take(&mut self.vl_pending);

        // Hoist the run-fixed system VLEN out of the per-instruction path: it
        // is set once (in QemuRunner construction) before any processing and
        // never changes, so reading it once per batch avoids an atomic load +
        // assert on every instruction. Folded into per-PC `InsnClass`
        // descriptors for bytes_moved accounting.
        let vlen_bytes = crate::common::SYSTEM_VLEN.get_bytes();

        // Build the per-module run caches on first use, now that `vlen_bytes` is
        // known to be the value the whole run will use (it is set during
        // QemuRunner construction, potentially after this Enricher was built, so
        // baking it in at `new()` time could fold a wrong VLEN into every
        // memoized aggregate).
        if self.run_caches.len() != self.modules.decoder.dec.len() {
            self.run_caches = (0..self.modules.decoder.dec.len())
                .map(|_| super::run_cache::RunCache::new(vlen_bytes))
                .collect();
            self.block_cache = super::block_cache::BlockCache::new(vlen_bytes);
        }

        Ok(LoopCtx {
            vl_pending,
            vl_index: 0,
            markers_by_pc,
            vlen_bytes,
            cached_func_range: None,
            active_region_ids: Vec::new(),
            pre_marker_region_ids: Vec::new(),
            // `active_region_ids` needs rebuilding: set on entry (the region
            // stack carries over from the previous batch) and by every marker,
            // which is the only thing that can push or pop a region.
            region_ids_dirty: true,
        })
    }

    /// Per-batch postamble: retain what the next batch must still consume.
    fn end_batch(&mut self, ctx: LoopCtx) -> Result<()> {
        // Retain any VL values whose vsetvl has not been decoded yet; they will
        // be consumed by a subsequent call once those instructions are decoded.
        self.vl_pending = ctx.vl_pending[ctx.vl_index..].to_vec();

        // Retain any ROI markers whose PC has not been decoded yet (e.g. it
        // lands in a later batch of the same datagram); they will be consumed by
        // a subsequent call once those instructions are decoded.
        self.marker_pending = ctx.markers_by_pc;

        self.trace.maybe_flush()?;

        Ok(())
    }

    /// Commit every instruction of one contiguous PC window, in order.
    ///
    /// The window is a whole E-Trace batch on that path and a single BBT block on
    /// the other; the fast path is offered the window's tail, so a run is only
    /// applied when the trace actually available here confirms it.
    fn commit_window(&mut self, ctx: &mut LoopCtx, pcs: &[u64]) -> Result<()> {
        // A `while` rather than a `for` so the batched fast path can advance by a
        // whole run at once.
        let mut i: usize = 0;
        while i < pcs.len() {
            let pc = pcs[i];

            // Get decoder (and its module-matched resolver) for this PC
            let module_index = match self.modules.decoder.find_decoder_index_for(pc) {
                Some(i) => i,
                None => {
                    let msg = format!("enricher multi-decoder has no ELF data for PC 0x{:x?}", pc);
                    log::error!("{}", msg);
                    log::error!("{:?}", self.modules.decoder);
                    return Err(crate::RerfError::address_not_covered(msg).into());
                }
            };

            if let Some(n) = self.try_fast_run(ctx, module_index, &pcs[i..]) {
                i += n;
                continue;
            }

            self.commit_one(ctx, module_index, pcs, i)?;
            i += 1;
        }
        Ok(())
    }

    /// ========== Fast path: one memoized run aggregate ==========
    ///
    /// A run is a maximal straight-line stretch containing no `vsetvl*` and no
    /// ROI marker (see `run_cache`), so across it the *only* inputs to each
    /// instruction's contribution -- the live vtype and the set of active scopes
    /// -- are constant, and the sum of the per-instruction deltas can be memoized
    /// per entry vtype and added to each scope with a single `add_run_delta`.
    ///
    /// Soundness rests on the slice compare below. `lookup` is a purely *static*
    /// prediction of control flow: it says which PCs would follow if nothing
    /// intervened. A trap, an interrupt or self-modifying code can divert
    /// execution mid-run, so the predicted PC sequence is applied only after it
    /// has been confirmed to be exactly what the trace actually retired. On a
    /// mismatch the caller falls through to the per-instruction path, which is
    /// always correct.
    ///
    /// `window` starts at the instruction to commit. Returns the number of
    /// instructions committed, or `None` for "no memo applied, use the
    /// per-instruction path".
    fn try_fast_run(
        &mut self,
        ctx: &mut LoopCtx,
        module_index: usize,
        window: &[u64],
    ) -> Option<usize> {
        let pc = window[0];
        let decoder = &self.modules.decoder.dec[module_index];

        // Function attribution, by the same rule as the slow path: reuse the
        // cached range while it contains the PC, else resolve within the module
        // the decoder lookup above already identified.
        let in_cache = ctx
            .cached_func_range
            .as_ref()
            .map(|(range, _, _)| range.contains(&pc))
            .unwrap_or(false);
        if !in_cache {
            ctx.cached_func_range = self
                .modules
                .resolver
                .resolve_in_module(module_index, pc)
                .map(|(name, index, range)| (range, name.to_owned(), index));
        }

        // The three values `InsnDelta::new` consumes, i.e. exactly what the slow
        // path would pass for every instruction of the run.
        let key = super::run_cache::VtypeKey {
            vl: self.vl_tracker.current_vl,
            sew: self.vl_tracker.current_sew,
            lmul: self.vl_tracker.current_lmul,
        };

        // The aggregate is applied to whatever regions are active, so the cached
        // ID list has to be current first -- same rebuild the slow path does at
        // Step 3, on the same dirty flag (only markers can change the set, and
        // markers are never inside a run).
        if ctx.region_ids_dirty {
            Self::rebuild_active_region_ids(&self.region_tracker, &mut ctx.active_region_ids);
            ctx.region_ids_dirty = false;
        }

        let hit = self.run_caches[module_index].lookup(decoder, pc, key)?;
        let n = hit.pcs.len();

        // The run is attributed wholesale to the function resolved for its start
        // PC, so refuse it if its last PC lies outside that range: a run cannot
        // leave a function without a control transfer, but overlapping /
        // misreported symbol ranges are a property of the ELF, not something to
        // assume away.
        let func_ok = ctx
            .cached_func_range
            .as_ref()
            .is_some_and(|(range, _, _)| range.contains(&hit.pcs[n - 1]));

        if !func_ok {
            self.fast_stats.func_straddles += 1;
            return None;
        }
        if n > window.len() {
            // The run continues past the window (a batch end on the E-Trace path;
            // a block QEMU cut short at a page crossing or its TB cap on the BBT
            // path), so the tail PCs are not available to validate against.
            self.fast_stats.batch_overruns += 1;
            return None;
        }
        if window[..n] != *hit.pcs {
            self.fast_stats.mismatches += 1;
            return None;
        }

        let (func_index_opt, func_name_opt) = match ctx.cached_func_range.as_ref() {
            Some((_, name, index)) => (Some(*index), Some(name.as_str())),
            None => (None, None),
        };

        // One `add_run_delta` per scope instead of N `apply_delta`s. This also
        // keeps `global_roi().instructions` exact for the marker path: a run
        // contains no marker, and the whole run's counters land here *before* the
        // caller advances, so every marker still reads a global instruction count
        // covering precisely the instructions committed before it -- which is
        // what `record_roi_start`/`record_roi_end` and the `entry.4` re-snapshot
        // use to delimit `[begin, end)` windows.
        self.stats.count_run(
            hit.delta,
            func_index_opt,
            func_name_opt,
            &ctx.active_region_ids,
        );

        // Per-PC profile: just count the execution. The run's contributions are
        // multiplied out over its PCs once at finalization by
        // `flush_run_pc_profiles`, which is the whole point -- `PcProfile::record`
        // was being called for 96% of all instructions to add 1 to a hash-map
        // entry whose *static* content the run cache already knows.
        hit.execs.set(hit.execs.get() + 1);
        hit.func_index.set(func_index_opt);

        // Trace emission stays per instruction in *effect*: `tick_n` emits at
        // exactly the instruction counts N `tick()`s would have, so the Perfetto
        // `.pb` is byte-identical, without running the loop.
        self.trace.tick_n(n as u64, self.stats.global_roi());

        // Note what the fast path deliberately does *not* touch: `vl_pending` /
        // `vl_index` (no `vsetvl*` can be inside a run, so the VL queue cannot be
        // consumed here) and the marker/trace-region state.
        self.fast_stats.fast_instrs += n as u64;
        self.fast_stats.runs_applied += 1;
        Some(n)
    }

    /// Per-instruction path: commit exactly one instruction, the one at
    /// `window[index]`.
    ///
    /// `window` is passed whole (rather than just the PC) for the ROI v1
    /// `EventAndValue` back-scan, which recovers a marker's `rs1`/`rs2` by
    /// re-deriving them from preceding `li`s. See
    /// `InstructionDecoder::extract_event_and_value_from_trace`.
    fn commit_one(
        &mut self,
        ctx: &mut LoopCtx,
        module_index: usize,
        window: &[u64],
        index: usize,
    ) -> Result<()> {
        let pc = window[index];
        let vlen_bytes = ctx.vlen_bytes;
        let decoder = &self.modules.decoder.dec[module_index];
        self.fast_stats.slow_instrs += 1;

        // Decode instruction at this PC (with its cached classification and
        // ROI-marker verdict).
        match decoder.get_instruction_class_and_marker_at(pc) {
            Ok((ins, ins_bytes, ins_class, pc_marker)) => {
                // Check if PC is within cached function range
                let in_cache = ctx
                    .cached_func_range
                    .as_ref()
                    .map(|(range, _, _)| range.contains(&pc))
                    .unwrap_or(false);

                if !in_cache {
                    // `resolve_in_module` (not the module-scanning
                    // `resolve_to_name_index_and_range`) since the decoder
                    // lookup above already found the owning module; the
                    // returned index is global (offset-adjusted), so it
                    // stays unique across every module for
                    // `StatsAccumulator`'s function map.
                    match self.modules.resolver.resolve_in_module(module_index, pc) {
                        Some((name, index, range)) => {
                            ctx.cached_func_range = Some((range, name.to_owned(), index));
                        }
                        None => {
                            ctx.cached_func_range = None;
                        }
                    }
                }

                let func_index_opt = ctx.cached_func_range.as_ref().map(|(_, _, index)| *index);
                let func_name_opt = ctx
                    .cached_func_range
                    .as_ref()
                    .map(|(_, name, _)| name.as_str());

                // ========== Step 1: Handle ROI Markers ==========
                if let Some(marker) = pc_marker {
                    // Markers are the only instructions that can change the
                    // active-region set; force a rebuild of the cached ID
                    // list before the next non-marker instruction counts.
                    ctx.region_ids_dirty = true;

                    // Snapshot active region IDs BEFORE processing the marker.
                    // Marker instructions (e.g. begin/end of a nested region)
                    // should still be counted for regions that were already
                    // active before this marker fires.
                    ctx.pre_marker_region_ids.clear();
                    for r in self.region_tracker.all_active_regions().iter() {
                        ctx.pre_marker_region_ids.push(r.region_id);
                    }
                    let pre_marker_event_id = self
                        .region_tracker
                        .current_event_region()
                        .map(|r| r.region_id);

                    let rt_marker = ctx.markers_by_pc.get_mut(&pc).and_then(|q| q.pop_front());
                    let marker_ctx = MarkerCtx {
                        pc,
                        decoder,
                        pcs: window,
                        pc_index: index,
                        rt_marker: rt_marker.as_ref(),
                        ins_bytes,
                    };
                    let events = self.marker_handler.handle_marker(
                        marker,
                        &marker_ctx,
                        &mut self.region_tracker,
                        self.stats.region_roi_map_mut(),
                        &mut self.next_region_id,
                    );

                    // local import for this part of the code only
                    use crate::processing::enricher::region_tracker::RegionKind;
                    use crate::processing::enricher::RegionEvent;

                    // Process region events. Stack and event regions are
                    // recorded identically here: only the trace emitter cares
                    // which kind a region is.
                    for event in &events {
                        match event {
                            RegionEvent::Started {
                                region_id,
                                name,
                                kind,
                            } => {
                                // Resolve this region's call-tree node once,
                                // here, and carry the index to the end marker:
                                // the exit path then costs two integer adds
                                // instead of re-deriving the whole call path
                                // from region names.
                                let flame_node = if self.roi_flamegraph {
                                    let parent = self.region_tracker.parent_flame_node(*kind);
                                    let node =
                                        self.roi_flame_builder.enter(parent, *region_id, name);
                                    // Park it on the tracker's own entry so it
                                    // cannot drift from the region stack that
                                    // defines the call path.
                                    match kind {
                                        RegionKind::Stack => {
                                            self.region_tracker.set_stack_top_flame_node(node)
                                        }
                                        RegionKind::Event => {
                                            self.region_tracker.set_event_flame_node(node)
                                        }
                                    }
                                    node
                                } else {
                                    crate::processing::roi_flamegraph::FLAME_ROOT
                                };
                                Self::record_roi_start(
                                    &mut self.open_roi_starts,
                                    *region_id,
                                    self.stats.global_roi().instructions,
                                    flame_node,
                                );
                                // Region already registered in stats by marker_handler.
                            }
                            RegionEvent::Ended { region_id, .. } => {
                                Self::record_roi_end(
                                    &mut self.open_roi_starts,
                                    &mut self.roi_flame_builder,
                                    *region_id,
                                    self.stats.global_roi().instructions,
                                );
                            }
                        }

                        // Emit trace event for this region event
                        self.trace.handle_region_event(event);
                    }

                    // R-type ROI markers (add/sub/or/and x0, rs1, rs2) are
                    // real CPU instructions that occupy a cycle. They must
                    // be counted for any regions that were already active
                    // *before* this marker was processed (e.g. a nested
                    // begin_region inside an outer region).
                    //
                    // However, the region that is being *started* or *ended*
                    // by this marker must NOT be counted: the begin marker
                    // is not part of the new region.
                    //
                    // The end marker is NOT part of the region being closed.
                    //
                    // In general, think logically: "instructions should be counted if they execute".
                    // - So counting add/or/sub/etc from x0 is no different to other dst registers.
                    // - Think of how hardware counters would operate - and match thier behaviour.
                    let is_rtype_marker = matches!(
                        marker,
                        RoiMarker::BeginRegion
                            | RoiMarker::EndRegion
                            | RoiMarker::EventAndValue
                            | RoiMarker::NameEvent
                            | RoiMarker::EventString
                            | RoiMarker::ValueString
                            | RoiMarker::ParallelBegin
                    );
                    if is_rtype_marker {
                        // Collect region IDs that were started or ended by
                        // this marker so we can exclude them from counting.
                        // A switch emits both an end and a start, so both of
                        // its regions land here.
                        let excluded_ids: Vec<usize> =
                            events.iter().filter_map(RegionEvent::region_id).collect();

                        if let Some(eid) = pre_marker_event_id {
                            ctx.pre_marker_region_ids.push(eid);
                        }
                        ctx.pre_marker_region_ids
                            .retain(|id| !excluded_ids.contains(id));
                        if !ctx.pre_marker_region_ids.is_empty() {
                            let delta = crate::common::InsnDelta::new(
                                &ins_class,
                                &ins,
                                self.vl_tracker.current_vl,
                                self.vl_tracker.current_sew,
                                self.vl_tracker.current_lmul,
                                vlen_bytes,
                            );
                            self.stats.count_instruction(
                                &delta,
                                func_index_opt,
                                func_name_opt,
                                &ctx.pre_marker_region_ids,
                            );
                            // Mirror the same inclusion rule for the per-PC
                            // profile (marker instrs are counted only when a
                            // pre-marker region was active — same as global).
                            self.pc_profile.record(
                                pc,
                                &ins,
                                ins_bytes,
                                func_index_opt,
                                &delta,
                                self.vl_tracker.current_lmul,
                                None, // markers never write vl
                            );
                        }
                    }

                    // A begin marker that opens a *nested* region is itself
                    // attributed (tallied above) to the still-open parent, not
                    // to the region it opens. That tally happens strictly
                    // after `record_roi_start` snapshotted `begin_instr` for
                    // the new region, so the counter bump would otherwise be
                    // (wrongly) included in the new region's own [begin, end)
                    // window. Re-snapshot `begin_instr` now, after any tally
                    // for this marker instruction has been applied, so the
                    // new region's window starts strictly after it.
                    for region_id in events.iter().filter_map(RegionEvent::started_region_id) {
                        if let Some(stack) = self.open_roi_starts.get_mut(&region_id) {
                            if let Some(entry) = stack.last_mut() {
                                entry.0 = self.stats.global_roi().instructions;
                            }
                        }
                    }
                    return Ok(());
                }

                // ========== Step 2: Track VL/SEW ==========
                let vl_value = self.vl_tracker.process_instruction(
                    &ins,
                    pc,
                    &ctx.vl_pending,
                    &mut ctx.vl_index,
                )?;

                // Emit VL counter event if changed
                if let Some(vl) = vl_value {
                    if let Some(sew) = self.vl_tracker.current_sew {
                        // Convert SEW index to actual bit width
                        let sew_bits = match sew {
                            0 => 8,
                            1 => 16,
                            2 => 32,
                            3 => 64,
                            _ => {
                                log::warn!("Invalid SEW value: {}", sew);
                                return Ok(());
                            }
                        };
                        self.trace.emit_vl_counter(sew_bits, vl);
                    }
                }

                // ========== Step 3: Accumulate Statistics ==========
                // Build the list of active region IDs (reusing the
                // pre-allocated buffer) only when the region set can
                // actually have changed. Only marker instructions open or
                // close regions, and every one of them returns early above
                // after setting the dirty flag -- so between two markers
                // this list is invariant, and rebuilding it by walking the
                // tracker on each of the ~306M instructions was pure
                // repetition.
                if ctx.region_ids_dirty {
                    Self::rebuild_active_region_ids(
                        &self.region_tracker,
                        &mut ctx.active_region_ids,
                    );
                    ctx.region_ids_dirty = false;
                }

                // Resolve the per-PC class against the live `vtype` state
                // once per instruction, here: the aggregate counters and
                // the per-PC profile below both consume the exact same
                // contribution, so neither has to re-derive it.
                let delta = crate::common::InsnDelta::new(
                    &ins_class,
                    &ins,
                    self.vl_tracker.current_vl,
                    self.vl_tracker.current_sew,
                    self.vl_tracker.current_lmul,
                    vlen_bytes,
                );

                // Update statistics (single method call handles all scopes)
                self.stats.count_instruction(
                    &delta,
                    func_index_opt,
                    func_name_opt,
                    &ctx.active_region_ids,
                );

                // Record the per-PC profile at the same point the global
                // aggregate is bumped, so Σ exec_count == global instructions.
                self.pc_profile.record(
                    pc,
                    &ins,
                    ins_bytes,
                    func_index_opt,
                    &delta,
                    self.vl_tracker.current_lmul,
                    vl_value, // Some(vl) iff this is the vsetvl* that set it
                );

                // ========== Step 4: Emit Trace Counters ==========
                // Interval-throttled: check inline and skip the call in the
                // overwhelmingly common not-yet-due case.
                if self.trace.should_emit_vpu_counters() {
                    self.trace.emit_vpu_counters(self.stats.global_roi());
                }

                // Increment global instruction counter in trace emitter
                self.trace.tick();
            }
            Err(e) => {
                log::warn!("Failed to decode instruction at PC {:#x}: {:?}", pc, e);
            }
        }

        Ok(())
    }
    /// Refill `out` with the currently-active ROI region IDs (stack regions
    /// plus the current event region), as Step 3 of the main loop does.
    ///
    /// An associated function on the one field it reads so the batched path can
    /// call it while other fields of `self` are mutably borrowed.
    fn rebuild_active_region_ids(region_tracker: &RegionTracker, out: &mut Vec<usize>) {
        out.clear();
        for r in region_tracker.all_active_regions().iter() {
            out.push(r.region_id);
        }
        if let Some(event_region) = region_tracker.current_event_region() {
            out.push(event_region.region_id);
        }
    }

    // ==================== ROI execution tracking ====================

    /// Push a region-start onto the open-region stack for `region_id`.
    ///
    /// Taken as an associated function on the individual fields (rather than
    /// `&mut self`) so it can be called from inside the region-event loop,
    /// which holds an immutable borrow of the event list.
    fn record_roi_start(
        open: &mut OpenRoiStarts,
        region_id: usize,
        begin_instr: u64,
        flame_node: crate::processing::roi_flamegraph::FlameNodeIdx,
    ) {
        open.entry(region_id)
            .or_default()
            .push((begin_instr, flame_node));
    }

    /// Pop the most recent open start for `region_id` and fold its instruction
    /// window into the streaming ROI flamegraph.
    ///
    /// A no-op if no matching start is open (e.g. an end marker seen before any
    /// begin), or if `flame_node` is
    /// [`FLAME_ROOT`](crate::processing::roi_flamegraph::FLAME_ROOT) because
    /// flamegraph tracking is off.
    fn record_roi_end(
        open: &mut OpenRoiStarts,
        builder: &mut crate::processing::roi_flamegraph::StreamingRoiFlamegraph,
        region_id: usize,
        end_instr: u64,
    ) {
        if let Some(stack) = open.get_mut(&region_id) {
            if let Some((begin_instr, flame_node)) = stack.pop() {
                builder.accumulate(flame_node, end_instr.saturating_sub(begin_instr));
            }
        }
    }

    /// Take the streaming ROI flamegraph builder recorded so far, leaving a fresh builder.
    pub fn take_roi_flamegraph_builder(
        &mut self,
    ) -> crate::processing::roi_flamegraph::StreamingRoiFlamegraph {
        std::mem::take(&mut self.roi_flame_builder)
    }

    // ==================== Source & assembly annotation ====================

    /// Whether the per-PC annotation profile is being recorded.
    pub fn annotate_enabled(&self) -> bool {
        self.pc_profile.is_enabled()
    }

    /// Populate `instruction_mix` for every function-scoped and the global
    /// `RoiInfo` from this hart's per-PC profile.
    ///
    /// This is the only thing that ever fills `instruction_mix`: the mix is not
    /// accumulated live (the per-instruction hashmap probe that used to do so
    /// measured as a very heavy runtime cost), it is derived once here from the
    /// per-PC execution counts. No-op when annotate wasn't enabled -- there is
    /// no `PcProfile` to derive from, so those scopes report no mix.
    ///
    /// Call this once, after all instructions for this hart have been
    /// processed, and before `take_pc_profile`/`performance_data` (it reads
    /// the still-populated `pc_profile`).
    pub fn apply_deferred_instruction_mix(&mut self) {
        if !self.pc_profile.is_enabled() {
            return;
        }
        self.flush_run_pc_profiles();
        let pc_profile = &self.pc_profile;
        for (func_index, roi) in self.stats.function_map_mut().iter_mut() {
            roi.set_instruction_mix_from_pc_profile(
                pc_profile.instruction_mix_for_function(*func_index),
            );
        }
        self.stats
            .global_roi_mut()
            .set_instruction_mix_from_pc_profile(pc_profile.instruction_mix_all());
    }

    /// Multiply the batched fast path's run-execution counts out into the per-PC
    /// profile.
    ///
    /// The fast path records only "this run executed once more at this vtype";
    /// this turns those counts into the per-PC records the annotation report
    /// needs, by walking the *static* run tables (a few thousand `(run, vtype)`
    /// pairs) rather than the trace.
    ///
    /// Idempotent -- it drains the counters -- so it can be (and is) called from
    /// both `apply_deferred_instruction_mix` and `take_pc_profile`, whichever
    /// reads the profile first.
    fn flush_run_pc_profiles(&mut self) {
        if !self.pc_profile.is_enabled() {
            return;
        }
        for (module_index, cache) in self.run_caches.iter().enumerate() {
            let Some(decoder) = self.modules.decoder.dec.get(module_index) else {
                continue;
            };
            cache.distribute_pc_profile(&mut self.pc_profile, decoder);
        }
        self.block_cache
            .distribute_pc_profile(&mut self.pc_profile, &self.modules.decoder);
    }

    /// Take this hart's per-PC profile, leaving a fresh (identically-gated) one
    /// behind. Used to merge per-hart profiles before report generation.
    pub fn take_pc_profile(&mut self) -> PcProfile {
        let enabled = self.pc_profile.is_enabled();
        self.flush_run_pc_profiles();
        // Cache counts for a PC's *first* execution arrive before its record
        // exists (the footprint walk runs ahead of the commit), so they are held
        // aside until now.
        self.pc_profile.flush_pending_cache();
        std::mem::replace(&mut self.pc_profile, PcProfile::new(enabled))
    }

    /// Build the source & assembly annotation report, joining `profile` with
    /// this enricher's ELF resolvers (function names + DWARF line table)
    /// across every module — main binary, shared libraries, and VDSO — so
    /// hot functions inside e.g. `libggml`/`libc` get their own annotated
    /// page just like main-binary functions. Source-line interleaving only
    /// applies where `--src-dir` covers the module's DWARF paths (typically
    /// the main binary and any co-located libraries); other modules still
    /// get an asm-only page via the same code path.
    ///
    /// Renders every page exactly once, then (a) writes them to the `html/`
    /// subdirectory of the configured output directory as gzipped
    /// `annotate_*.html.gz` artifacts and (b)
    /// returns the pages gzipped as `(logical_filename, gzip_bytes)` for the
    /// in-memory HTTP store, which serves them with `Content-Encoding: gzip`.
    pub fn build_annotation_report(&self, profile: &PcProfile) -> Vec<(String, Vec<u8>)> {
        let t_start = std::time::Instant::now();
        let pages = super::annotate::render_all(
            &self.modules.resolver,
            profile,
            &self.source_dirs,
            &self.modules.decoder,
            self.output_dir.as_deref(),
        );
        let t_render = t_start.elapsed().as_secs_f64();
        // Compress once and use the same bytes for both consumers: the on-disk
        // artifacts and the in-memory HTTP store (which serves them verbatim
        // with `Content-Encoding: gzip`). Deflating each page twice made
        // `miniz_oxide` 3.5% of whole-process samples.
        let gz: Vec<(String, Vec<u8>)> = super::annotate::gzip_pages(pages);
        let t_gzip = t_start.elapsed().as_secs_f64() - t_render;

        match self.output_dir {
            Some(ref dir) => {
                let out_dir = dir.join("html");
                super::annotate::write_pages_gz(&gz, &out_dir);
                let t_write = t_start.elapsed().as_secs_f64() - t_gzip;

                log::info!(
                    "annotated {} pages, rendered in {:.3}s, gziped in {:.3}s, disk write {:.3}s.",
                    gz.len(),
                    t_render,
                    t_gzip,
                    t_write
                );
            }
            None => log::warn!("annotate: no output dir configured; skipping on-disk report"),
        }

        gz
    }

    // ==================== Public API Methods ====================

    /// Get performance data including all ROIs (functions, regions, global)
    pub fn performance_data(&mut self) -> PerformanceData {
        self.stats.performance_data()
    }

    /// Get all ROI statistics (functions + regions) sorted by instruction count
    pub fn get_sorted_rois(&self) -> Vec<RoiInfo> {
        self.stats.get_sorted_rois()
    }

    /// Flush accumulated trace packets to disk
    pub fn flush_trace(&mut self) -> Result<()> {
        self.trace.flush()
    }

    /// Check if trace packets need flushing and flush if threshold exceeded
    pub fn maybe_flush_trace(&mut self) -> Result<()> {
        self.trace.maybe_flush()
    }

    /// Write the synthetto trace to a file (legacy method, calls flush)
    pub fn write_trace(&mut self, path: &std::path::Path) -> Result<()> {
        self.trace.write(path)
    }

    /// The other half of the `vl`-completeness gate: call once the whole trace
    /// has been decoded, to assert the `vl`
    /// stream was fully consumed.
    ///
    /// `process_instruction` catches a `vsetvl*` with no `vl` behind it; this
    /// catches the opposite, a `vl` with no `vsetvl*` in front of it — which
    /// means the tracer emitted a value the decoder never attributed, so every
    /// `vl` after the extra one was off by a step and the underrun check could
    /// not see it (the counts still balanced).
    pub fn check_vl_stream_exhausted(&self) -> Result<()> {
        if !self.vl_pending.is_empty() {
            anyhow::bail!(
                "vl stream not exhausted at end of trace: {} value(s) left over \
                 ({:?}...). Each one is a vl the tracer emitted that no decoded \
                 vsetvl* claimed, so vl was misattributed from the first surplus \
                 value onwards.",
                self.vl_pending.len(),
                &self.vl_pending[..self.vl_pending.len().min(8)],
            );
        }
        Ok(())
    }

    /// Reset all accumulated statistics and state
    pub fn reset_statistics(&mut self) {
        self.stats.reset();
        self.vl_tracker = VlTracker::new();
        self.vl_pending.clear();
        self.marker_pending.clear();
        self.marker_handler = MarkerHandler::new();
        self.pc_profile = PcProfile::new(self.pc_profile.is_enabled());
        self.next_region_id = 0;
        self.open_roi_starts.clear();
        self.roi_flame_builder =
            crate::processing::roi_flamegraph::StreamingRoiFlamegraph::default();
        // Note: region_tracker is managed by marker_handler, no need to reset
    }

    /// Get the current vector length state
    pub fn current_vl(&self) -> Option<u32> {
        self.vl_tracker.current_vl
    }

    /// Instructions committed by this enricher so far, over every batch and both
    /// paths. The BBT consumer never builds a PC array, so this — not a batch
    /// length — is the instruction count the run report is built from.
    pub fn instructions_committed(&self) -> u64 {
        self.fast_stats.fast_instrs + self.fast_stats.slow_instrs
    }

    /// VL values handed to this enricher so far, over every batch.
    pub fn vl_values_delivered(&self) -> u64 {
        self.fast_stats.vl_values
    }
}

impl Enricher {
    /// Report batched-enrichment coverage for this hart, at `debug` level.
    ///
    /// Called once during finalization (not from `Drop`: the builder methods move
    /// fields out of `self`, which a `Drop` impl would forbid).
    ///
    /// `mismatch` is the number to watch: it counts runs whose statically
    /// predicted PC sequence did not match what the trace retired, and should be
    /// ~0 on a clean trace.
    pub fn log_fast_path_stats(&self) {
        let s = &self.fast_stats;
        if s.fast_instrs == 0 && s.slow_instrs == 0 {
            return;
        }
        let total = s.fast_instrs + s.slow_instrs;
        let runs_built: u64 = self.run_caches.iter().map(|c| c.runs_built()).sum();
        let deltas_built: u64 = self.run_caches.iter().map(|c| c.deltas_built()).sum();
        log::debug!(
            "hart {}: run-cache fast path: {} of {} instructions batched ({:.1}%) \
             over {} run applications (mean {:.2} insns/run); {} instructions on the \
             per-instruction path; mismatch={} overrun={} func_straddle={}; \
             {} static runs analysed, {} (run, vtype) aggregates built",
            self.hart_id,
            s.fast_instrs,
            total,
            100.0 * s.fast_instrs as f64 / total as f64,
            s.runs_applied,
            s.fast_instrs as f64 / s.runs_applied.max(1) as f64,
            s.slow_instrs,
            s.mismatches,
            s.batch_overruns,
            s.func_straddles,
            runs_built,
            deltas_built,
        );
        if s.blocks_seen > 0 {
            log::debug!(
                "hart {}: BBT block memo: {} block entries committed over {} distinct \
                 static blocks ({} unusable), {} (segment, vtype) aggregates built; \
                 {} entries committed without a memo",
                self.hart_id,
                s.blocks_seen,
                self.block_cache.blocks_interned(),
                self.block_cache.blocks_unusable(),
                self.block_cache.deltas_built(),
                s.blocks_unmemoized,
            );
        }
    }
}

#[cfg(test)]
mod roi_execution_memory_tests {
    use super::*;
    use crate::processing::roi_flamegraph::{StreamingRoiFlamegraph, FLAME_ROOT};

    /// Drives `n` complete begin/end ROI marker pairs through the same
    /// `record_roi_start`/`record_roi_end` calls `transform_optimized` makes,
    /// resolving the call-tree node once up front exactly as the `Started` arm
    /// does. Returns the final `(open_roi_starts, builder)` state.
    fn run_n_roi_cycles(n: u64) -> (OpenRoiStarts, StreamingRoiFlamegraph) {
        let mut open = OpenRoiStarts::default();
        let mut builder = StreamingRoiFlamegraph::default();
        const REGION_ID: usize = 0;

        for i in 0..n {
            let flame_node = builder.enter(FLAME_ROOT, REGION_ID, "hot_loop_region");
            Enricher::record_roi_start(&mut open, REGION_ID, i * 100, flame_node);
            Enricher::record_roi_end(&mut open, &mut builder, REGION_ID, i * 100 + 50);
        }

        (open, builder)
    }

    /// This is the regression test for a memory balloon: a long run through a
    /// hot ROI region used to push one
    /// `RoiExecution` per enter/exit into a `Vec` that only the flamegraph
    /// report ever read, so a loop entered millions of times held millions of
    /// records for the run's entire duration. Streaming aggregation is keyed by
    /// call path instead, so one hot region is one node no matter how many times
    /// it is entered -- while still counting every execution.
    #[test]
    fn streaming_flamegraph_stays_bounded_in_hot_loops() {
        const ITERATIONS: u64 = 2_000_000;
        let (open, builder) = run_n_roi_cycles(ITERATIONS);

        assert_eq!(
            builder.node_count(),
            1,
            "one call path must stay one node across {ITERATIONS} cycles"
        );
        assert_eq!(
            builder.executions(),
            ITERATIONS,
            "every completed cycle must still be counted"
        );
        // The open-region stack must also fully drain on each matched
        // begin/end pair -- otherwise the growth would just move from the
        // builder into `open_roi_starts`.
        assert!(
            open.get(&0).is_none_or(|stack| stack.is_empty()),
            "open_roi_starts must drain on every matched begin/end pair"
        );
    }

    /// With flamegraph tracking off the enricher carries `FLAME_ROOT` as the
    /// node index, which `accumulate` must ignore rather than fold into the
    /// first real node.
    #[test]
    fn flame_root_node_index_accumulates_nothing() {
        let mut open = OpenRoiStarts::default();
        let mut builder = StreamingRoiFlamegraph::default();

        for i in 0..1_000u64 {
            Enricher::record_roi_start(&mut open, 0, i * 100, FLAME_ROOT);
            Enricher::record_roi_end(&mut open, &mut builder, 0, i * 100 + 50);
        }

        assert!(builder.is_empty());
        assert!(!builder.has_executions());
    }

    /// An end marker with no matching begin must not pop somebody else's frame
    /// or record a bogus execution.
    #[test]
    fn unmatched_end_marker_records_nothing() {
        let mut open = OpenRoiStarts::default();
        let mut builder = StreamingRoiFlamegraph::default();

        Enricher::record_roi_end(&mut open, &mut builder, 7, 500);

        assert!(!builder.has_executions());
    }
}
