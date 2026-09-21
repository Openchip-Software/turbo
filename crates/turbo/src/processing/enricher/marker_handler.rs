//! ROI marker processing and name-map ownership.
//!
//! [`InstructionDecoder`] says *that* an instruction is an ROI marker and of
//! which kind; this module says what it *means*. It takes a [`RoiMarker`] plus
//! the runtime register values captured for it, resolves the region or event it
//! refers to (including the V1 nibble-at-a-time name state machine and the
//! static-name fallback when no runtime marker is available), and delegates the
//! resulting begin/end/switch to [`RegionTracker`].
//!
//! It also owns the event- and value-name maps: names arrive out of band, via
//! their own markers, and are looked up later when an event/value pair needs a
//! human-readable region name.
//!
//! See [`instruction_decoder`](crate::processing::instruction_decoder) for the
//! RAVE-user-API vs. internal-ROI-marker naming split.

use std::collections::HashMap;
use std::hash::Hash;

use crate::common::RoiInfo;
use crate::processing::instruction_decoder::{InstructionDecoder, RoiMarker};
use turbo_tracer_shared::RoiRtMarker;

use super::region_tracker::{RegionEvent, RegionTracker};

/// Per-marker context: everything needed to resolve one detected marker.
///
/// Bundled into a struct because the individual handlers otherwise take a
/// long tail of positional arguments that are identical for every marker.
pub struct MarkerCtx<'a> {
    /// Program counter where the marker was detected.
    pub pc: u64,
    /// Instruction decoder for extracting names and values from the binary.
    pub decoder: &'a InstructionDecoder,
    /// Retired-PC window the enricher is currently committing.
    pub pcs: &'a [u64],
    /// Index of `pc` within `pcs`.
    pub pc_index: usize,
    /// Runtime marker values captured by the plugin, when available.
    pub rt_marker: Option<&'a RoiRtMarker>,
    /// Raw bytes of the marker instruction.
    pub ins_bytes: &'a [u8],
}

/// Mark every ROI referenced by a region-closing event as completed.
///
/// Covers every closing shape uniformly: stack regions, event regions closed
/// outright, and the event region displaced by a switch all report the same
/// [`RegionEvent::Ended`].
fn mark_completed(events: &[RegionEvent], roi_map: &mut rustc_hash::FxHashMap<usize, RoiInfo>) {
    for id in events.iter().filter_map(RegionEvent::ended_region_id) {
        if let Some(roi) = roi_map.get_mut(&id) {
            roi.completed = true;
        }
    }
}

/// Look up (or allocate) the stable region ID for `key` and ensure an ROI
/// entry exists for it, so repeated firings accumulate into one ROI.
fn intern_region<K: Eq + Hash>(
    ids: &mut HashMap<K, usize>,
    key: K,
    name: &str,
    next_region_id: &mut usize,
    roi_map: &mut rustc_hash::FxHashMap<usize, RoiInfo>,
) -> usize {
    let region_id = *ids.entry(key).or_insert_with(|| {
        let id = *next_region_id;
        *next_region_id += 1;
        id
    });
    roi_map
        .entry(region_id)
        .or_insert_with(|| RoiInfo::new(name.to_string()));
    region_id
}

/// Handles ROI marker processing and maintains event/value name mappings.
pub struct MarkerHandler {
    /// Map of event ID to event name (populated by NameEvent markers)
    event_name_map: HashMap<u64, String>,
    /// Map of (event_id, value) to value name (populated by NameEvent markers with values)
    value_name_map: HashMap<(u64, u64), String>,
    /// Stable region ID map for EventAndValue markers: (event_id, value) -> region_id.
    /// This ensures the same event+value combination always maps to the same ROI entry
    /// across repeated invocations, rather than allocating a fresh ID each time.
    event_value_region_ids: HashMap<(u64, u64), usize>,
    /// Stable region ID map for BeginRegion markers: region name -> region_id.
    /// Region identity is keyed on the (runtime-resolved) name rather than the
    /// begin-marker PC.  This is required when the begin/end markers are wrapped
    /// in a shared function (e.g. the Fortran `rave_begin_region` helper), where
    /// every named region shares a single begin PC and would otherwise collapse
    /// into one ROI.  It also naturally folds repeated / distributed entries of
    /// the same named region into a single ROI entry.
    region_name_ids: HashMap<String, usize>,
    /// V1 state machine: pending event_id from a NameEvent AND instruction
    v1_pending_event_id: Option<u64>,
    /// V1 state machine: pending value_id (None means rave_name_event, Some means rave_name_value)
    v1_pending_value_id: Option<u64>,
    /// V1 state machine: are we inside a name encoding sequence (between -1 sentinels)?
    v1_in_name: bool,
    /// V1 state machine: accumulated nibbles for current name
    v1_nibbles: Vec<u8>,
}

impl MarkerHandler {
    /// Creates a new MarkerHandler with empty name maps.
    pub fn new() -> Self {
        Self {
            event_name_map: HashMap::new(),
            value_name_map: HashMap::new(),
            event_value_region_ids: HashMap::new(),
            region_name_ids: HashMap::new(),
            v1_pending_event_id: None,
            v1_pending_value_id: None,
            v1_in_name: false,
            v1_nibbles: Vec::new(),
        }
    }

    /// Process a detected ROI marker. Updates internal name maps and delegates
    /// region operations to the provided RegionTracker.
    ///
    /// Returns the list of RegionEvents produced by the marker.
    ///
    /// # Arguments
    /// * `marker` - The ROI marker type detected
    /// * `ctx` - Per-marker context (PC, decoder, PC window, runtime values)
    /// * `region_tracker` - Region tracker to delegate region operations to
    /// * `region_roi_map` - Map of region IDs to ROI info (for creating new regions)
    /// * `next_region_id` - Allocator for stable region IDs (one per interned region)
    pub fn handle_marker(
        &mut self,
        marker: RoiMarker,
        ctx: &MarkerCtx<'_>,
        region_tracker: &mut RegionTracker,
        region_roi_map: &mut rustc_hash::FxHashMap<usize, RoiInfo>,
        next_region_id: &mut usize,
    ) -> Vec<RegionEvent> {
        let pc = ctx.pc;
        match marker {
            RoiMarker::BeginRegion => {
                self.handle_begin_region(ctx, region_tracker, region_roi_map, next_region_id)
            }
            RoiMarker::EndRegion => self.handle_end_region(ctx, region_tracker, region_roi_map),
            RoiMarker::StartTrace => {
                log::trace!("ROI: Start trace (Perfetto output) at PC {:#x}", pc);
                region_tracker.enable_trace();
                Vec::new()
            }
            RoiMarker::StopTrace => {
                log::trace!("ROI: Stop trace (Perfetto output) at PC {:#x}", pc);
                region_tracker.disable_trace();
                Vec::new()
            }
            RoiMarker::RestartTrace => {
                self.handle_restart_trace(pc, region_tracker, region_roi_map)
            }
            RoiMarker::EnableRegions => {
                log::trace!("ROI: Enable region counting at PC {:#x}", pc);
                region_tracker.enable_regions();
                Vec::new()
            }
            RoiMarker::DisableRegions => {
                log::trace!("ROI: Disable region counting at PC {:#x}", pc);
                region_tracker.disable_regions();
                Vec::new()
            }
            RoiMarker::NameEvent => self.handle_name_event(ctx),
            RoiMarker::EventAndValue => {
                self.handle_event_and_value(ctx, region_tracker, region_roi_map, next_region_id)
            }
            RoiMarker::NameDelimiter => {
                self.handle_v1_name_delimiter(pc);
                Vec::new()
            }
            RoiMarker::NameHexDigit => {
                self.handle_v1_name_hex_digit(pc, ctx.ins_bytes);
                Vec::new()
            }
            _ => {
                log::trace!("ROI: Detected marker {:?} at PC {:#x}", marker, pc);
                Vec::new()
            }
        }
    }

    /// Resets the marker handler state (clears name maps).
    #[cfg(test)]
    pub fn reset(&mut self) {
        self.event_name_map.clear();
        self.value_name_map.clear();
        self.v1_pending_event_id = None;
        self.v1_pending_value_id = None;
        self.v1_in_name = false;
        self.v1_nibbles.clear();
    }

    /// V1 state machine: handle a NameDelimiter (li x0, -1).
    /// This is a start or end sentinel for the inline nibble-encoded name.
    fn handle_v1_name_delimiter(&mut self, pc: u64) {
        if !self.v1_in_name {
            // Start sentinel: begin accumulating nibbles
            if self.v1_pending_event_id.is_some() {
                self.v1_in_name = true;
                self.v1_nibbles.clear();
            }
        } else {
            // End sentinel: reconstruct the name from accumulated nibbles
            self.v1_in_name = false;
            let name = self.reconstruct_v1_name();
            let nibbles = self.v1_nibbles.len();

            if !name.is_empty() {
                let event_id = self.v1_pending_event_id.take().unwrap_or(0);
                // One line per name registered, emitted at the end sentinel. The
                // per-nibble steps that built the name are an implementation
                // detail of this decode; the nibble count is enough to recognise
                // a malformed one.
                if let Some(value_id) = self.v1_pending_value_id.take() {
                    log::debug!(
                        "ROI V1: NameValue id={event_id} value={value_id} name='{name}' \
                         ({nibbles} nibbles) at PC {pc:#x}"
                    );
                    self.value_name_map.insert((event_id, value_id), name);
                } else {
                    log::debug!(
                        "ROI V1: NameEvent id={event_id} name='{name}' \
                         ({nibbles} nibbles) at PC {pc:#x}"
                    );
                    self.event_name_map.insert(event_id, name);
                }
            } else {
                log::warn!("ROI marker with empty name found at {pc}");
            }
            self.v1_nibbles.clear();
        }
    }

    /// V1 state machine: handle a NameHexDigit (lui x0, N).
    /// Accumulates nibble values while inside a name sequence.
    fn handle_v1_name_hex_digit(&mut self, _pc: u64, ins_bytes: &[u8]) {
        if self.v1_in_name {
            if let Some(nibble) = InstructionDecoder::extract_lui_nibble(ins_bytes) {
                self.v1_nibbles.push(nibble);
            }
        }
    }

    /// Reconstruct a string from V1 nibble encoding.
    ///
    /// The V1 C macro emits nibbles LSB-first per character:
    ///   for (tmp = char; tmp > 0; tmp >>= 4) write_hex(tmp & 0xf)
    ///
    /// For printable ASCII (0x20..0x7E), characters are always 2 nibbles.
    /// We group nibbles in pairs: char = low | (high << 4).
    fn reconstruct_v1_name(&self) -> String {
        let mut chars: Vec<u8> = Vec::new();
        let mut i = 0;
        while i + 1 < self.v1_nibbles.len() {
            let low = self.v1_nibbles[i] as u32;
            let high = self.v1_nibbles[i + 1] as u32;
            let ch = low | (high << 4);
            if ch > 0 && ch <= 0x7f {
                chars.push(ch as u8);
            }
            i += 2;
        }
        // Handle trailing single nibble (for chars 0x01..0x0F)
        if i < self.v1_nibbles.len() {
            let ch = self.v1_nibbles[i];
            if ch > 0 {
                chars.push(ch);
            }
        }
        String::from_utf8_lossy(&chars).to_string()
    }

    /// Resolve the name of a Begin/End region marker.
    ///
    /// Prefers the name captured from guest memory at runtime (works even when
    /// the name pointer targets the stack, e.g. Fortran).  Otherwise falls back
    /// to reading the string from the ELF (`rs1_val` is a pointer to the name,
    /// `rs2_val` is either the length or -1 for null-terminated), and finally to
    /// statically decoding the name from the binary at `pc`.
    fn resolve_region_name(
        decoder: &InstructionDecoder,
        pc: u64,
        rt_marker: Option<&RoiRtMarker>,
    ) -> Option<String> {
        if let Some(rt) = rt_marker {
            rt.name
                .clone()
                .filter(|s| !s.is_empty())
                .or_else(|| {
                    if rt.rs2_val != u64::MAX && rt.rs2_val != 0 {
                        decoder.read_string_from_elf_with_len(rt.rs1_val, rt.rs2_val)
                    } else {
                        decoder.read_string_from_elf(rt.rs1_val)
                    }
                })
                .or_else(|| decoder.extract_region_name(pc))
        } else {
            decoder.extract_region_name(pc)
        }
    }

    /// Handle BeginRegion marker: starts a new stack-based region.
    fn handle_begin_region(
        &mut self,
        ctx: &MarkerCtx<'_>,
        region_tracker: &mut RegionTracker,
        region_roi_map: &mut rustc_hash::FxHashMap<usize, RoiInfo>,
        next_region_id: &mut usize,
    ) -> Vec<RegionEvent> {
        let (pc, decoder, rt_marker) = (ctx.pc, ctx.decoder, ctx.rt_marker);
        if !region_tracker.regions_enabled() {
            log::trace!(
                "ROI: Begin region ignored (regions disabled) at PC {:#x}",
                pc
            );
            return Vec::new();
        }

        if let Some(rt) = rt_marker {
            log::trace!(
                "ROI begin_region runtime: rs1(ptr)={:#x} rs2(len)={:#x}",
                rt.rs1_val,
                rt.rs2_val
            );
        }

        // Resolve the region name fresh on every invocation.  The name must not
        // be cached by PC: when begin/end markers are wrapped in a shared
        // function (Fortran), all begins share a single PC but denote different
        // regions, distinguished only by their runtime name.
        let region_name = Self::resolve_region_name(decoder, pc, rt_marker).unwrap_or_else(|| {
            log::warn!("ROI marker at PC 0x{pc:x} was not resolved correctly");
            format!("roi_region_0x{:x}", pc)
        });

        // Region identity is keyed on the name, so distinct named regions get
        // distinct ROIs (even from a shared begin PC) and repeated / distributed
        // entries of the same name fold into a single ROI.
        let region_id = intern_region(
            &mut self.region_name_ids,
            region_name.clone(),
            &region_name,
            next_region_id,
            region_roi_map,
        );

        log::trace!(
            "ROI: Begin region '{}' (id={}) at PC {:#x}",
            region_name,
            region_id,
            pc
        );

        // Delegate to region tracker
        region_tracker.begin_region(region_id, region_name)
    }

    /// Handle EndRegion marker: ends the most recent stack-based region.
    fn handle_end_region(
        &mut self,
        ctx: &MarkerCtx<'_>,
        region_tracker: &mut RegionTracker,
        region_roi_map: &mut rustc_hash::FxHashMap<usize, RoiInfo>,
    ) -> Vec<RegionEvent> {
        let (pc, decoder, rt_marker) = (ctx.pc, ctx.decoder, ctx.rt_marker);
        if let Some(rt) = rt_marker {
            log::trace!(
                "ROI end_region runtime: rs1(ptr)={:#x} rs2(len)={:#x}",
                rt.rs1_val,
                rt.rs2_val
            );
        }

        if !region_tracker.regions_enabled() {
            log::trace!("ROI: End region ignored (regions disabled) at PC {:#x}", pc);
            return Vec::new();
        }

        // Try to resolve the region name so we can end the correct region
        // (supports overlapping / out-of-order end_region calls).
        let resolved_name = Self::resolve_region_name(decoder, pc, rt_marker);

        let events = if let Some(ref name) = resolved_name {
            region_tracker.end_region_by_name(name)
        } else {
            region_tracker.end_region()
        };

        if events.is_empty() {
            log::warn!("ROI: End region at PC {:#x} without matching begin", pc);
        } else {
            for event in &events {
                if let RegionEvent::Ended {
                    region_id, name, ..
                } = event
                {
                    log::trace!(
                        "ROI: End region '{}' (id={}) at PC {:#x}",
                        name,
                        region_id,
                        pc
                    );
                }
            }
            mark_completed(&events, region_roi_map);
        }

        events
    }

    /// Handle RestartTrace marker: resets state and enables both trace and regions.
    fn handle_restart_trace(
        &mut self,
        pc: u64,
        region_tracker: &mut RegionTracker,
        region_roi_map: &mut rustc_hash::FxHashMap<usize, RoiInfo>,
    ) -> Vec<RegionEvent> {
        log::trace!("ROI: Restart trace at PC {:#x}", pc);
        // Clear accumulated metrics and counters, but preserve name declarations.
        // The spec says "erase all traced metrics and counters", not name mappings.
        region_roi_map.clear();
        self.event_value_region_ids.clear();
        region_tracker.clear_all();
        region_tracker.enable_trace();
        region_tracker.enable_regions();
        // clear_all() drops open regions wholesale rather than closing them, so
        // there is nothing to report: the ROI metrics they fed have been erased.
        Vec::new()
    }

    /// Handle NameEvent marker: declares an event name for later use by EventAndValue markers.
    ///
    /// This is a Version 1 ROI marker that does NOT start a region.
    fn handle_name_event(&mut self, ctx: &MarkerCtx<'_>) -> Vec<RegionEvent> {
        let (pc, decoder, rt_marker) = (ctx.pc, ctx.decoder, ctx.rt_marker);
        // Extract event ID from runtime marker or static decoding
        let event_id = if let Some(rt) = rt_marker {
            Some(rt.rs1_val)
        } else {
            decoder.extract_event_id(pc)
        };

        // For value: if rt_marker present and rs2_val != u64::MAX (-1) then it's a value.
        // rs2_val == u64::MAX means rave_name_event (rs2 held -1, no value).
        // rs2_val == 0 IS valid: it means rave_name_value(x, 0, name).
        let value = if let Some(rt) = rt_marker {
            let v = rt.rs2_val;
            if v != u64::MAX {
                Some(v)
            } else {
                None
            }
        } else {
            decoder.extract_event_value(pc)
        };

        if let Some(event_id) = event_id {
            // Set up V1 state machine: the name will be decoded from
            // subsequent NameDelimiter / NameHexDigit instructions in the
            // execution trace.  Store pending ids so handle_v1_name_delimiter
            // can finalize the mapping once the end sentinel is reached.
            self.v1_pending_event_id = Some(event_id);
            self.v1_pending_value_id = value;
        } else {
            log::warn!("ROI: NameEvent at PC {:#x} could not extract event ID", pc);
        }

        Vec::new()
    }

    /// Resolve the (event_id, value) pair carried by an EventAndValue marker.
    ///
    /// Prefers the runtime values captured by the plugin; falls back to static
    /// extraction from the binary / retired-PC window for ROI v1.
    fn resolve_event_and_value(ctx: &MarkerCtx<'_>) -> Option<(u64, u64)> {
        if let Some(rt) = ctx.rt_marker {
            return Some((rt.rs1_val, rt.rs2_val));
        }

        // ROI v1 fallback. `pcs` is the retired-PC window the enricher is
        // currently committing — a whole batch on the E-Trace path, but a
        // single basic block on the BBT path, where PCs are borrowed from the
        // block dictionary and no wider window exists (deliberately: there is
        // no PC ring buffer for ROI v1 markers).
        log::warn!(
            "ROI v1 EventAndValue at {:#x}: no plugin-supplied rt_marker, so \
             rs1/rs2 are being recovered by scanning back over only the {} retired \
             PC(s) available here. The rt_marker path is the supported one; if this \
             misresolves, fix the producer, not the window.",
            ctx.pc,
            ctx.pc_index + 1
        );
        ctx.decoder
            .extract_event_and_value_from_trace(ctx.pc, ctx.pcs, ctx.pc_index)
            .or_else(|| ctx.decoder.extract_event_and_value(ctx.pc))
    }

    /// Region name for an (event_id, value) pair: the value-specific name if
    /// one was declared, else the event name plus the value, else "unknown".
    fn event_region_name(&self, event_id: u64, value: u64) -> String {
        if let Some(value_name) = self.value_name_map.get(&(event_id, value)) {
            // We have a specific name for this event+value combination
            value_name.to_string()
        } else if let Some(event_name) = self.event_name_map.get(&event_id) {
            // We only have the event name, include the value
            format!("{} [value={}]", event_name, value)
        } else {
            // Unknown event
            format!("unknown [id={}, value={}]", event_id, value)
        }
    }

    /// Handle EventAndValue marker: switches to a new event-based region.
    ///
    /// Version 2 ROI marker (or x0, rs1, rs2) where rs1 contains event ID
    /// and rs2 contains the value.
    fn handle_event_and_value(
        &mut self,
        ctx: &MarkerCtx<'_>,
        region_tracker: &mut RegionTracker,
        region_roi_map: &mut rustc_hash::FxHashMap<usize, RoiInfo>,
        next_region_id: &mut usize,
    ) -> Vec<RegionEvent> {
        let pc = ctx.pc;
        let Some((event_id, value)) = Self::resolve_event_and_value(ctx) else {
            log::warn!(
                "ROI: EventAndValue at PC {:#x} could not extract event/value",
                pc
            );
            return Vec::new();
        };

        // V1 convention: value=0 means "end the current region" rather than
        // starting a new one.  Close whatever event region is active and return.
        //
        // INTENTIONAL ORDERING: this end path runs *before* the
        // regions_enabled()/trace_enabled() guard below, so a value-0 close is
        // honoured even while regions are disabled.  Do not "tidy" the guard up
        // to the top of the function — that would leak an open event region.
        if value == 0 {
            log::trace!(
                "ROI: EventAndValue id={} value=0 -> ending current event region at PC {:#x}",
                event_id,
                pc
            );
            let events = region_tracker.end_event_region();
            mark_completed(&events, region_roi_map);
            return events;
        }

        let region_name = self.event_region_name(event_id, value);

        log::trace!(
            "ROI: EventAndValue id={} value={} region='{}' at PC {:#x}",
            event_id,
            value,
            region_name,
            pc
        );

        if !region_tracker.regions_enabled() || !region_tracker.trace_enabled() {
            return Vec::new();
        }

        // Look up or allocate a stable region ID for this (event_id, value)
        // combination so that repeated firings accumulate into one ROI entry.
        let region_id = intern_region(
            &mut self.event_value_region_ids,
            (event_id, value),
            &region_name,
            next_region_id,
            region_roi_map,
        );

        log::trace!(
            "ROI: Switching to event region '{}' (id={}) at PC {:#x}",
            region_name,
            region_id,
            pc
        );

        // Delegate to region tracker: close the previous event region, then open
        // the new one.
        let mut events = region_tracker.end_event_region();
        events.extend(region_tracker.start_event_region(region_id, region_name));
        mark_completed(&events, region_roi_map);
        events
    }
}

impl Default for MarkerHandler {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_marker_handler_new() {
        let handler = MarkerHandler::new();
        assert!(handler.event_name_map.is_empty());
        assert!(handler.value_name_map.is_empty());
    }

    #[test]
    fn test_marker_handler_reset() {
        let mut handler = MarkerHandler::new();
        handler.event_name_map.insert(1, "test".to_string());
        handler
            .value_name_map
            .insert((1, 2), "test_value".to_string());

        handler.reset();
        assert!(handler.event_name_map.is_empty());
        assert!(handler.value_name_map.is_empty());
    }

    /// A decoder over a real committed RISC-V ELF. The trace-enable and
    /// region-enable markers never consult the decoder, but `MarkerCtx` holds
    /// one, and `InstructionDecoder::new_with_vlen` parses its input as an ELF,
    /// so an empty slice cannot stand in.
    fn test_decoder() -> InstructionDecoder {
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../test_bins/flash/flash.riscv"
        );
        let data = std::fs::read(path).expect("read test ELF");
        InstructionDecoder::new_with_vlen(0, &data, 16).expect("decoder over test ELF")
    }

    /// Minimal context for markers that need nothing but the PC.
    fn test_ctx(decoder: &InstructionDecoder) -> MarkerCtx<'_> {
        MarkerCtx {
            pc: 0x1000,
            decoder,
            pcs: &[],
            pc_index: 0,
            rt_marker: None,
            ins_bytes: &[],
        }
    }

    #[test]
    fn test_start_stop_trace() {
        let mut handler = MarkerHandler::new();
        let mut tracker = RegionTracker::new();
        let decoder = test_decoder();
        let ctx = test_ctx(&decoder);
        let mut roi_map = rustc_hash::FxHashMap::default();
        let mut next_region_id = 0usize;

        let events = handler.handle_marker(
            RoiMarker::StartTrace,
            &ctx,
            &mut tracker,
            &mut roi_map,
            &mut next_region_id,
        );
        // Gates emit no events; the state query is the observable effect.
        assert!(events.is_empty());
        assert!(tracker.trace_enabled());

        let events = handler.handle_marker(
            RoiMarker::StopTrace,
            &ctx,
            &mut tracker,
            &mut roi_map,
            &mut next_region_id,
        );
        assert!(events.is_empty());
        assert!(!tracker.trace_enabled());
    }

    #[test]
    fn test_enable_disable_regions() {
        let mut handler = MarkerHandler::new();
        let mut tracker = RegionTracker::new();
        let decoder = test_decoder();
        let ctx = test_ctx(&decoder);
        let mut roi_map = rustc_hash::FxHashMap::default();
        let mut next_region_id = 0usize;

        let events = handler.handle_marker(
            RoiMarker::EnableRegions,
            &ctx,
            &mut tracker,
            &mut roi_map,
            &mut next_region_id,
        );
        assert!(events.is_empty());
        assert!(tracker.regions_enabled());

        let events = handler.handle_marker(
            RoiMarker::DisableRegions,
            &ctx,
            &mut tracker,
            &mut roi_map,
            &mut next_region_id,
        );
        assert!(events.is_empty());
        assert!(!tracker.regions_enabled());
    }

    #[test]
    fn test_restart_trace() {
        let mut handler = MarkerHandler::new();
        let mut tracker = RegionTracker::new();
        let mut region_roi_map = rustc_hash::FxHashMap::default();

        // Pre-populate some state
        handler.event_name_map.insert(1, "test".to_string());
        handler.value_name_map.insert((1, 2), "val".to_string());
        region_roi_map.insert(0, RoiInfo::new("roi".to_string()));

        let events = handler.handle_restart_trace(0x1000, &mut tracker, &mut region_roi_map);
        // Should enable both trace and regions, without emitting events
        assert!(events.is_empty());
        assert!(tracker.trace_enabled());
        assert!(tracker.regions_enabled());
        // Should have cleared ROI data but preserved name declarations
        assert!(
            !handler.event_name_map.is_empty(),
            "name declarations should survive restart"
        );
        assert!(
            !handler.value_name_map.is_empty(),
            "value name declarations should survive restart"
        );
        assert!(
            region_roi_map.is_empty(),
            "ROI metrics should be cleared on restart"
        );
    }
}
