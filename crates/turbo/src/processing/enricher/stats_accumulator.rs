//! Statistics accumulation for ROI (Region of Interest) analysis.
//!
//! Every retired instruction is attributed to several scopes at once: the
//! function containing it, each currently-active ROI region (regions nest, so
//! there may be more than one), and the global ROI covering the whole run. This
//! module owns those per-scope counters and the final roll-up.
//!
//! - Track instruction counts and characteristics per scope
//! - Accumulate statistics for functions and ROI regions
//! - Build final PerformanceData output with merged and sorted ROIs
//!
//! Regions here are identified by the integer ids handed out by
//! [`RegionTracker`](super::region_tracker::RegionTracker); resolving markers to
//! those ids is [`marker_handler`](super::marker_handler)'s job.

use crate::common::{InsnDelta, PerformanceData, RoiInfo, SYSTEM_VLEN};
use rustc_hash::FxHashMap;

/// Manages accumulation of ROI statistics across functions, ROI regions, and global scope.
///
/// The StatsAccumulator owns the maps for function-level and region-level statistics,
/// and provides a unified interface for counting instructions and building performance data.
pub struct StatsAccumulator {
    /// Persistent ROI statistics indexed by function index (FxHash: ~4x faster for integer keys)
    func_map: FxHashMap<usize, RoiInfo>,
    /// ROI region ROI statistics indexed by region ID (FxHash: integer keys)
    region_roi_map: FxHashMap<usize, RoiInfo>,
    /// Global ROI tracking all instructions regardless of function or region
    global_roi: RoiInfo,
    /// Last `func_index` counted, and a raw handle to its `RoiInfo` inside
    /// `func_map`. Consecutive instructions almost always share a function, so
    /// this lets a repeated `func_index` skip the `func_map` hashmap probe.
    ///
    /// INVARIANT: `last_func_roi` is only dereferenced when
    /// `last_func_index == Some(index)` for the incoming `index`, and it is
    /// reset to `None`/null on any *structural* mutation of `func_map`
    /// (`reset`, `function_map_mut`, `ensure_function`). Between two
    /// consecutive `count_instruction` calls with the same `func_index`, no
    /// structural mutation of `func_map` occurs, so the pointer stays valid.
    last_func_index: Option<usize>,
    last_func_roi: *mut RoiInfo,
}

/// SAFETY: `last_func_roi` is a self-referential memo pointing into this
/// struct's OWN `func_map`. It is never handed out, never aliased by another
/// `StatsAccumulator`, and never dereferenced except under the invariant
/// documented on the field. Moving a `StatsAccumulator` to another thread moves
/// the `func_map` handle but not its heap allocation, so the pointee stays at
/// the same address and the pointer stays valid.
///
/// This is what lets the parallel decode pool own one accumulator per hart on
/// one worker thread each. The type is deliberately NOT `Sync`: two threads
/// touching the same accumulator would race on the memo.
unsafe impl Send for StatsAccumulator {}

impl StatsAccumulator {
    /// Create a new StatsAccumulator with empty maps and a global ROI.
    pub fn new() -> Self {
        StatsAccumulator {
            func_map: FxHashMap::default(),
            region_roi_map: FxHashMap::default(),
            global_roi: RoiInfo::new("Global".to_string()),
            last_func_index: None,
            last_func_roi: std::ptr::null_mut(),
        }
    }

    /// Ensure an ROI region exists in the map.
    ///
    /// If the region doesn't exist, it will be created with the given name.
    /// If it already exists, this is a no-op.
    ///
    /// # Arguments
    /// * `region_id` - Unique identifier for the region (typically the BeginRegion PC)
    /// * `name` - Human-readable name for the region
    #[cfg(test)]
    pub fn ensure_region(&mut self, region_id: usize, name: String) {
        self.region_roi_map
            .entry(region_id)
            .or_insert_with(|| RoiInfo::new(name));
    }

    /// Ensure a function exists in the map.
    ///
    /// If the function doesn't exist, it will be created with the given name.
    /// If it already exists, this is a no-op.
    ///
    /// # Arguments
    /// * `func_index` - Unique index for the function from the ELF symbol table
    /// * `name` - Human-readable name for the function
    #[cfg(test)]
    pub fn ensure_function(&mut self, func_index: usize, name: String) {
        // Structural mutation of func_map may reallocate: invalidate the cache.
        self.last_func_index = None;
        self.last_func_roi = std::ptr::null_mut();
        self.func_map
            .entry(func_index)
            .or_insert_with(|| RoiInfo::new(name));
    }

    /// Count one instruction for a function, all active ROI regions, and the global ROI.
    ///
    /// This is the core counting method that updates statistics across all relevant scopes:
    /// - The function containing the instruction (if known)
    /// - All currently active ROI regions (can be nested)
    /// - The global ROI (always)
    ///
    /// # Arguments
    /// * `ins` - The decoded RISC-V instruction
    /// * `func_index` - Optional function index (from ELF symbol table)
    /// * `func_name` - Optional function name (for creating new entries)
    /// * `active_region_ids` - Slice of active ROI region IDs (stack-based regions)
    /// * `current_vl` - Current vector length (for vector instruction statistics)
    /// * `current_sew` - Current SEW (0=e8, 1=e16, 2=e32, 3=e64)
    #[hotpath::measure]
    #[inline]
    #[allow(clippy::too_many_arguments)]
    pub fn count_instruction(
        &mut self,
        delta: &InsnDelta,
        func_index: Option<usize>,
        func_name: Option<&str>,
        active_region_ids: &[usize],
    ) {
        // `delta` -- the per-PC class resolved against the live `vtype` state --
        // is built by the caller and shared with the per-PC profile, since both
        // apply the identical contribution.

        // Update function statistics (single lookup per map)
        if let Some(index) = func_index {
            if let Some(name) = func_name {
                // Fast path: consecutive instructions share a function, so a
                // repeated `func_index` reuses the cached `RoiInfo` handle and
                // skips the `func_map` hashmap probe entirely.
                //
                // SAFETY: `last_func_roi` points at a `RoiInfo` owned by
                // `func_map`. It is dereferenced only when `last_func_index`
                // matches `index`, which guarantees the pointer was set on a
                // previous call for this same key. `func_map` is not mutated
                // structurally between two such calls (the global/roi updates
                // below touch different maps), and every structural mutation of
                // `func_map` resets `last_func_index` to `None`. The enricher
                // is single-threaded per hart, so there is no aliasing.
                let roi: &mut RoiInfo = if self.last_func_index == Some(index) {
                    unsafe { &mut *self.last_func_roi }
                } else {
                    let roi = self
                        .func_map
                        .entry(index)
                        .or_insert_with(|| RoiInfo::new(name.to_string()));
                    self.last_func_index = Some(index);
                    self.last_func_roi = roi as *mut RoiInfo;
                    roi
                };
                roi.instructions += 1;
                roi.apply_delta(delta);
            }
        }

        // Update global ROI (always tracked)
        self.global_roi.instructions += 1;
        self.global_roi.apply_delta(delta);

        // Update active ROI regions (single lookup each)
        for &region_id in active_region_ids {
            if let Some(roi) = self.region_roi_map.get_mut(&region_id) {
                roi.instructions += 1;
                roi.apply_delta(delta);
            }
        }
    }

    /// Count a whole straight-line run at once, for the same three scopes as
    /// [`count_instruction`](Self::count_instruction).
    ///
    /// The [`RunDelta`](crate::processing::enricher::run_cache::RunDelta) is the
    /// exact sum of the run's per-instruction [`InsnDelta`]s (see `run_cache`),
    /// so this is observationally identical to N `count_instruction` calls --
    /// only much cheaper, since each scope is touched once instead of N times.
    ///
    /// Note `add_run_delta` bumps `instructions` itself (unlike `apply_delta`),
    /// so there is deliberately no `+= 1`-style adjustment here.
    pub(crate) fn count_run(
        &mut self,
        delta: &crate::processing::enricher::run_cache::RunDelta,
        func_index: Option<usize>,
        func_name: Option<&str>,
        active_region_ids: &[usize],
    ) {
        if let (Some(index), Some(name)) = (func_index, func_name) {
            // SAFETY: identical argument to `count_instruction`'s -- the pointer
            // is only dereferenced when `last_func_index` matches, it was set by
            // an earlier call for the same key, `func_map` is not structurally
            // mutated in between (the global/roi updates below touch other
            // maps), and every structural mutation clears `last_func_index`.
            let roi: &mut RoiInfo = if self.last_func_index == Some(index) {
                unsafe { &mut *self.last_func_roi }
            } else {
                let roi = self
                    .func_map
                    .entry(index)
                    .or_insert_with(|| RoiInfo::new(name.to_string()));
                self.last_func_index = Some(index);
                self.last_func_roi = roi as *mut RoiInfo;
                roi
            };
            roi.add_run_delta(delta);
        }

        self.global_roi.add_run_delta(delta);

        for &region_id in active_region_ids {
            if let Some(roi) = self.region_roi_map.get_mut(&region_id) {
                roi.add_run_delta(delta);
            }
        }
    }

    /// Attribute one instruction's modelled cache behaviour to its function,
    /// every active region, and the global ROI.
    ///
    /// Shaped like [`count_run`](Self::count_run) — the same three scopes, in
    /// the same order — but kept separate rather than folded into `InsnDelta`
    /// or `RunDelta`: cache counts are a function of the *model's state* at the
    /// moment of the access, not of the instruction, so they cannot be memoized
    /// with the rest of a run. That distinction is the reason the run cache
    /// stays correct with cache modelling on.
    pub(crate) fn count_cache(
        &mut self,
        counts: &crate::common::CacheCounts,
        func_index: Option<usize>,
        func_name: Option<&str>,
        active_region_ids: &[usize],
    ) {
        if let (Some(index), Some(name)) = (func_index, func_name) {
            // Deliberately NOT using the `last_func_roi` memo: this is called
            // once per vector memory instruction rather than once per
            // instruction, so the probe is rare, and taking the raw-pointer
            // path would mean maintaining its invariant from a second site.
            // Only a *structural* mutation invalidates the memo, so probe first
            // and take the inserting path only when the key is genuinely new.
            // Clearing it unconditionally meant that turning cache modelling on
            // slowed the *commit* path down as well: `model_block_cache` runs
            // immediately before `commit_block` for the same block, so every
            // block holding a vector memory op handed `count_run` a cold memo.
            match self.func_map.get_mut(&index) {
                Some(roi) => roi.cache.add(counts),
                None => {
                    self.func_map
                        .entry(index)
                        .or_insert_with(|| RoiInfo::new(name.to_string()))
                        .cache
                        .add(counts);
                    self.last_func_index = None;
                    self.last_func_roi = std::ptr::null_mut();
                }
            }
        }

        self.global_roi.cache.add(counts);

        for &region_id in active_region_ids {
            if let Some(roi) = self.region_roi_map.get_mut(&region_id) {
                roi.cache.add(counts);
            }
        }
    }

    /// Get mutable reference to an ROI for external updates.
    ///
    /// This allows external code to directly modify ROI statistics.
    ///
    /// # Arguments
    /// * `region_id` - The region ID to look up
    ///
    /// # Returns
    /// An optional mutable reference to the RoiInfo if the region exists
    #[cfg(test)]
    pub fn get_region_roi_mut(&mut self, region_id: usize) -> Option<&mut RoiInfo> {
        self.region_roi_map.get_mut(&region_id)
    }

    /// Materialize the serialization-facing views (sparse VL histograms, and
    /// the rendered instruction mix) from the raw accumulators.
    ///
    /// Only the *raw* histograms are updated on the counting path; the sparse
    /// exports are derived state that nothing reads until an `RoiInfo` is
    /// cloned out of here (`performance_data`, and through it every live
    /// snapshot and the final JSON). Deriving them per batch instead meant
    /// ~270 functions x 4 SEWs x 8 LMUL views rebuilt on each of the ~5100
    /// batches of a single run -- over a million `to_sparse` rebuilds whose
    /// results were all overwritten by the next batch.
    ///
    /// `RoiInfo::add_ref` re-derives them after any merge, so a snapshot that
    /// merges harts stays correct regardless of when this last ran.
    fn materialize_derived(&mut self) {
        for roi in self.func_map.values_mut() {
            roi.update_from_histogram();
        }
        for roi in self.region_roi_map.values_mut() {
            roi.update_from_histogram();
        }
        self.global_roi.update_from_histogram();
    }

    /// Build the final PerformanceData output.
    ///
    /// This merges ROIs with the same name (which can happen when the same
    /// named region is entered from different call sites), and returns both
    /// function-level and region-level statistics.
    ///
    /// # Returns
    /// A PerformanceData struct containing merged ROIs and function statistics
    pub fn performance_data(&mut self) -> PerformanceData {
        // Bring the derived (sparse-histogram / instruction-mix) views up to
        // date exactly once, here at the point they are about to be read out,
        // rather than after every batch.
        self.materialize_derived();

        // Combine function ROIs, adding global ROI if no functions were found
        let mut all_funcs: Vec<RoiInfo> = self.func_map.values().cloned().collect();
        if all_funcs.is_empty() {
            // Add global ROI if it has any instructions
            if self.global_roi.instructions > 0 {
                all_funcs.push(self.global_roi.clone());
            } else {
                log::warn!("no functions found, and global region is empty.");
            }
        }

        // Fold entries that share the same region name into one ROI.
        // This happens when the same named region is entered from two distinct
        // call sites (two different BeginRegion PCs), producing two map keys.
        let mut merged_rois: Vec<RoiInfo> = Vec::new();
        for roi in self.region_roi_map.values() {
            if let Some(existing) = merged_rois.iter_mut().find(|r| r.name == roi.name) {
                existing.merge_from(roi);
            } else {
                merged_rois.push(roi.clone());
            }
        }

        // always show a global ROI
        merged_rois.push(self.global_roi.clone());

        PerformanceData {
            rois: merged_rois,
            functions: all_funcs,
            // trace info only available after trace has been decoded
            trace_info: None,
            vlen_bits: SYSTEM_VLEN
                .is_set()
                .then(|| SYSTEM_VLEN.get_bytes() as u32 * 8),
        }
    }

    /// Get a reference to the global ROI.
    ///
    /// The global ROI tracks all instructions regardless of function or region.
    pub fn global_roi(&self) -> &RoiInfo {
        &self.global_roi
    }

    /// Get a reference to the function ROI statistics map.
    pub fn function_map(&self) -> &FxHashMap<usize, RoiInfo> {
        &self.func_map
    }

    /// Get a mutable reference to the function ROI statistics map. Used by
    /// `Enricher::apply_deferred_instruction_mix` to populate `instruction_mix`
    /// from a `PcProfile` once, at finalize time.
    pub fn function_map_mut(&mut self) -> &mut FxHashMap<usize, RoiInfo> {
        // Caller may mutate func_map structurally: invalidate the cache.
        self.last_func_index = None;
        self.last_func_roi = std::ptr::null_mut();
        &mut self.func_map
    }

    /// Get a mutable reference to the global ROI. See `function_map_mut`.
    pub fn global_roi_mut(&mut self) -> &mut RoiInfo {
        &mut self.global_roi
    }

    /// Get a mutable reference to the ROI region ROI map.
    pub fn region_roi_map_mut(&mut self) -> &mut FxHashMap<usize, RoiInfo> {
        &mut self.region_roi_map
    }

    /// Get the name of an ROI region by its ID.
    pub fn get_region_name(&self, region_id: usize) -> String {
        self.region_roi_map
            .get(&region_id)
            .map(|roi| roi.name.to_string())
            .unwrap_or_else(|| format!("region_0x{:x}", region_id))
    }

    /// Get all ROIs (functions and ROI regions) sorted by instruction count.
    ///
    /// This is useful for quick analysis and debugging. The returned vector
    /// contains cloned RoiInfo objects sorted by instruction count descending.
    ///
    /// # Returns
    /// A vector of RoiInfo clones sorted by instruction count
    pub fn get_sorted_rois(&self) -> Vec<RoiInfo> {
        let mut rois: Vec<RoiInfo> = Vec::new();

        // Collect function ROIs
        for roi in self.func_map.values() {
            rois.push(roi.clone());
        }

        // Collect ROI region ROIs
        for roi in self.region_roi_map.values() {
            rois.push(roi.clone());
        }

        // Sort by instruction count descending
        rois.sort_by_key(|r| std::cmp::Reverse(r.instructions));

        rois
    }

    /// Reset all statistics to initial state.
    ///
    /// This clears all accumulated statistics, including:
    /// - Function ROI map
    /// - ROI region ROI map
    /// - Global ROI
    pub fn reset(&mut self) {
        self.func_map.clear();
        self.region_roi_map.clear();
        self.global_roi = RoiInfo::new("Global".to_string());
        // Invalidate the per-function cache: func_map was cleared.
        self.last_func_index = None;
        self.last_func_roi = std::ptr::null_mut();
    }
}

impl Default for StatsAccumulator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_accumulator() {
        let acc = StatsAccumulator::new();
        assert_eq!(acc.func_map.len(), 0);
        assert_eq!(acc.region_roi_map.len(), 0);
        assert_eq!(acc.global_roi.instructions, 0);
        assert_eq!(&*acc.global_roi.name, "Global");
    }

    #[test]
    fn test_ensure_function() {
        let mut acc = StatsAccumulator::new();
        acc.ensure_function(1, "test_func".to_string());
        assert_eq!(acc.func_map.len(), 1);
        assert!(acc.func_map.contains_key(&1));

        // Ensure idempotent
        acc.ensure_function(1, "test_func".to_string());
        assert_eq!(acc.func_map.len(), 1);
    }

    #[test]
    fn test_ensure_region() {
        let mut acc = StatsAccumulator::new();
        acc.ensure_region(100, "roi_region_1".to_string());
        assert_eq!(acc.region_roi_map.len(), 1);
        assert!(acc.region_roi_map.contains_key(&100));

        // Ensure idempotent
        acc.ensure_region(100, "roi_region_1".to_string());
        assert_eq!(acc.region_roi_map.len(), 1);
    }

    #[test]
    fn test_reset() {
        let mut acc = StatsAccumulator::new();
        acc.ensure_function(1, "func1".to_string());
        acc.ensure_region(100, "region1".to_string());

        acc.reset();

        assert_eq!(acc.func_map.len(), 0);
        assert_eq!(acc.region_roi_map.len(), 0);
        assert_eq!(acc.global_roi.instructions, 0);
        assert_eq!(&*acc.global_roi.name, "Global");
    }

    #[test]
    fn test_get_region_roi_mut() {
        let mut acc = StatsAccumulator::new();
        acc.ensure_region(100, "region1".to_string());

        let roi = acc.get_region_roi_mut(100);
        assert!(roi.is_some());

        let roi = acc.get_region_roi_mut(999);
        assert!(roi.is_none());
    }
}
