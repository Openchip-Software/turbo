//! Region tracking module for managing ROI region stack and state.
//!
//! This module owns the region stack, event region state, and enable/disable flags.
//! It provides a clean interface for region lifecycle management and state queries.

use crate::processing::roi_flamegraph::{FlameNodeIdx, FLAME_ROOT};

/// Represents an active ROI region on the stack or as the current event region.
#[derive(Debug, Clone)]
pub struct ActiveRoiRegion {
    /// Unique identifier for this region instance
    pub region_id: usize,
    /// Name of the region (extracted from binary or generated)
    pub name: String,
    /// This region's node in the enricher's live ROI flamegraph call tree.
    ///
    /// The tracker itself never reads it; it lives here so that the node index
    /// cannot drift out of sync with the region stack that defines the call
    /// path. Starts as [`FLAME_ROOT`] and is filled in by the enricher via
    /// [`RegionTracker::set_stack_top_flame_node`] /
    /// [`RegionTracker::set_event_flame_node`] as soon as the region's `Started`
    /// event is processed. Stays [`FLAME_ROOT`] when flamegraph tracking is off.
    pub flame_node: FlameNodeIdx,
}

/// Which flavour of region a lifecycle event refers to.
///
/// The two behave identically for statistics; they differ only in how the trace
/// emitter tracks them (stack regions get a Perfetto track per region instance,
/// event regions share one track per name so repeated firings stack up).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RegionKind {
    /// Stack-based region, from Begin/EndRegion markers.
    Stack,
    /// Event-based region, from EventAndValue markers.
    Event,
}

/// Events emitted by the region tracker to communicate state changes.
///
/// A region switch is reported as an [`Ended`](RegionEvent::Ended) followed by a
/// [`Started`](RegionEvent::Started) in the returned vector, rather than as a
/// compound event: consumers iterate the vector in order, so they get the same
/// sequencing without every one of them having to decompose a switch by hand.
///
/// Gate changes (`enable_trace` and friends) deliberately emit nothing: they
/// alter what *later* events the tracker produces, which is the whole of their
/// observable effect. Consumers that need the current gate state read
/// [`RegionTracker::trace_enabled`] / [`RegionTracker::regions_enabled`].
#[derive(Debug, Clone)]
pub enum RegionEvent {
    /// A region became active.
    Started {
        region_id: usize,
        name: String,
        kind: RegionKind,
    },
    /// A region stopped being active.
    Ended {
        region_id: usize,
        name: String,
        kind: RegionKind,
    },
}

impl RegionEvent {
    /// The region this event started, if it started one.
    pub fn started_region_id(&self) -> Option<usize> {
        match self {
            RegionEvent::Started { region_id, .. } => Some(*region_id),
            _ => None,
        }
    }

    /// The region this event ended, if it ended one.
    pub fn ended_region_id(&self) -> Option<usize> {
        match self {
            RegionEvent::Ended { region_id, .. } => Some(*region_id),
            _ => None,
        }
    }

    /// The region this event started or ended, if any.
    pub fn region_id(&self) -> Option<usize> {
        self.started_region_id().or_else(|| self.ended_region_id())
    }
}

/// Manages the ROI region stack and state.
pub struct RegionTracker {
    /// Stack of active ROI regions (for nested regions)
    region_stack: Vec<ActiveRoiRegion>,
    /// Currently active event-based region (from EventAndValue markers)
    current_event_region: Option<ActiveRoiRegion>,
    /// Whether ROI trace output (Perfetto) is currently enabled
    trace_enabled: bool,
    /// Whether ROI region counting is currently enabled
    regions_enabled: bool,
}

impl RegionTracker {
    /// Creates a new region tracker with default state.
    pub fn new() -> Self {
        Self {
            region_stack: Vec::new(),
            current_event_region: None,
            trace_enabled: true,
            regions_enabled: true,
        }
    }

    /// Begins a new region by pushing it onto the stack.
    ///
    /// Returns events describing the state change.
    pub fn begin_region(&mut self, region_id: usize, name: String) -> Vec<RegionEvent> {
        if !self.regions_enabled {
            return Vec::new();
        }

        self.region_stack.push(ActiveRoiRegion {
            region_id,
            name: name.clone(),
            flame_node: FLAME_ROOT,
        });

        vec![RegionEvent::Started {
            region_id,
            name,
            kind: RegionKind::Stack,
        }]
    }

    /// Ends the most recent region by popping it from the stack.
    ///
    /// Returns events describing the state change, or an empty vector if the stack is empty.
    pub fn end_region(&mut self) -> Vec<RegionEvent> {
        if !self.regions_enabled {
            return Vec::new();
        }

        match self.region_stack.pop() {
            Some(region) => vec![RegionEvent::Ended {
                region_id: region.region_id,
                name: region.name,
                kind: RegionKind::Stack,
            }],
            None => Vec::new(),
        }
    }

    /// Ends a region identified by name, removing it from anywhere in the stack.
    ///
    /// This supports overlapping (non-nested) regions where a region may be
    /// closed out of LIFO order.  If no region with the given name is found on
    /// the stack, nothing is popped: blindly popping the top would corrupt the
    /// nesting of unrelated open regions (e.g. closing an outer region because
    /// an inner name was mismatched).  The unmatched close is logged and ignored.
    pub fn end_region_by_name(&mut self, name: &str) -> Vec<RegionEvent> {
        if !self.regions_enabled {
            return Vec::new();
        }

        // Search from top of stack for a region with the matching name
        match self.region_stack.iter().rposition(|r| r.name == name) {
            Some(idx) => {
                let region = self.region_stack.remove(idx);
                vec![RegionEvent::Ended {
                    region_id: region.region_id,
                    name: region.name,
                    kind: RegionKind::Stack,
                }]
            }
            None => {
                // No matching open region: do NOT pop an unrelated region, as that
                // would corrupt nesting. Leave the stack untouched.
                log::warn!(
                    "ROI: end_region_by_name('{}') has no matching open region on stack; ignoring",
                    name
                );
                Vec::new()
            }
        }
    }

    /// Ends the current event region without starting a new one.
    /// Used when EventAndValue has value=0 (V1 "end region" convention).
    /// Returns events describing the state change.
    pub fn end_event_region(&mut self) -> Vec<RegionEvent> {
        if !self.regions_enabled || !self.trace_enabled {
            return Vec::new();
        }

        match self.current_event_region.take() {
            Some(old) => vec![RegionEvent::Ended {
                region_id: old.region_id,
                name: old.name,
                kind: RegionKind::Event,
            }],
            None => Vec::new(),
        }
    }

    /// Starts a new current event region.
    ///
    /// Any previously active event region is dropped without an Ended event;
    /// callers that need one should call [`Self::end_event_region`] first.
    pub fn start_event_region(&mut self, region_id: usize, name: String) -> Vec<RegionEvent> {
        if !self.regions_enabled || !self.trace_enabled {
            return Vec::new();
        }

        self.current_event_region = Some(ActiveRoiRegion {
            region_id,
            name: name.clone(),
            flame_node: FLAME_ROOT,
        });

        vec![RegionEvent::Started {
            region_id,
            name,
            kind: RegionKind::Event,
        }]
    }

    /// Enables trace output.
    pub fn enable_trace(&mut self) {
        self.trace_enabled = true;
    }

    /// Disables trace output.
    pub fn disable_trace(&mut self) {
        self.trace_enabled = false;
    }

    /// Enables region counting.
    pub fn enable_regions(&mut self) {
        self.regions_enabled = true;
    }

    /// Disables region counting.
    pub fn disable_regions(&mut self) {
        self.regions_enabled = false;
    }

    /// Returns an iterator over the IDs of all currently active regions.
    ///
    /// This includes both stack-based regions and the current event region.
    #[cfg(test)]
    pub fn active_region_ids(&self) -> impl Iterator<Item = usize> + '_ {
        self.region_stack
            .iter()
            .map(|r| r.region_id)
            .chain(self.current_event_region.iter().map(|r| r.region_id))
    }

    /// Returns true if any region (stack or event) is currently active.
    #[cfg(test)]
    pub fn is_any_region_active(&self) -> bool {
        !self.region_stack.is_empty() || self.current_event_region.is_some()
    }

    /// Returns true if region counting is enabled.
    pub fn regions_enabled(&self) -> bool {
        self.regions_enabled
    }

    /// Returns true if trace output is enabled.
    pub fn trace_enabled(&self) -> bool {
        self.trace_enabled
    }

    /// Returns a reference to the current event region, if any.
    pub fn current_event_region(&self) -> Option<&ActiveRoiRegion> {
        self.current_event_region.as_ref()
    }

    /// Returns all active regions (the stack-based region stack).
    pub fn all_active_regions(&self) -> &[ActiveRoiRegion] {
        &self.region_stack
    }

    /// The flamegraph call-path node that a *just-started* region of `kind`
    /// hangs off, or [`FLAME_ROOT`] if it is top-level.
    ///
    /// The two kinds differ because only [`RegionKind::Stack`] regions live on
    /// `region_stack`: by the time the enricher sees the `Started` event, a
    /// stack region is already the top of the stack, so its parent is the entry
    /// *below* it. An event region is held in `current_event_region` and never
    /// pushed, so its parent is the innermost stack region -- the top. Treating
    /// both the same way silently dropped an event region's innermost parent.
    pub fn parent_flame_node(&self, kind: RegionKind) -> FlameNodeIdx {
        let skip_own_frame = match kind {
            RegionKind::Stack => 1,
            RegionKind::Event => 0,
        };
        let parent_idx = self.region_stack.len().checked_sub(skip_own_frame + 1);
        parent_idx
            .and_then(|i| self.region_stack.get(i))
            .map(|r| r.flame_node)
            .unwrap_or(FLAME_ROOT)
    }

    /// Record the flamegraph node of the region on top of the stack, i.e. the
    /// one a [`RegionKind::Stack`] `Started` event just pushed.
    pub fn set_stack_top_flame_node(&mut self, node: FlameNodeIdx) {
        if let Some(top) = self.region_stack.last_mut() {
            top.flame_node = node;
        }
    }

    /// Record the flamegraph node of the current event region, i.e. the one a
    /// [`RegionKind::Event`] `Started` event just installed.
    pub fn set_event_flame_node(&mut self, node: FlameNodeIdx) {
        if let Some(r) = self.current_event_region.as_mut() {
            r.flame_node = node;
        }
    }

    /// Clears all region state: the region stack and current event region.
    pub fn clear_all(&mut self) {
        self.region_stack.clear();
        self.current_event_region = None;
    }

    /// Gets an active region by ID from the stack.
    ///
    /// # Arguments
    /// * `region_id` - The region ID to look up
    ///
    /// # Returns
    /// A reference to the region if found on the stack
    pub fn get_active_region(&self, region_id: usize) -> Option<&ActiveRoiRegion> {
        self.region_stack.iter().find(|r| r.region_id == region_id)
    }
}

impl Default for RegionTracker {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_begin_end_region() {
        let mut tracker = RegionTracker::new();
        tracker.enable_regions();

        let events = tracker.begin_region(1, "region1".to_string());
        assert_eq!(events.len(), 1);
        assert!(tracker.is_any_region_active());
        assert_eq!(tracker.all_active_regions().len(), 1);

        let events = tracker.end_region();
        assert_eq!(events.len(), 1);
        assert!(!tracker.is_any_region_active());
        assert_eq!(tracker.all_active_regions().len(), 0);
    }

    #[test]
    fn test_nested_regions() {
        let mut tracker = RegionTracker::new();
        tracker.enable_regions();

        tracker.begin_region(1, "outer".to_string());
        tracker.begin_region(2, "inner".to_string());

        assert_eq!(tracker.all_active_regions().len(), 2);
        let ids: Vec<usize> = tracker.active_region_ids().collect();
        assert_eq!(ids, vec![1, 2]);

        tracker.end_region();
        assert_eq!(tracker.all_active_regions().len(), 1);

        tracker.end_region();
        assert_eq!(tracker.all_active_regions().len(), 0);
    }

    #[test]
    fn test_nested_end_by_name_lifo() {
        // Nested regions closed in LIFO order by name (the Fortran nesting case).
        let mut tracker = RegionTracker::new();
        tracker.enable_regions();

        tracker.begin_region(1, "Initializations".to_string());
        tracker.begin_region(2, "Initialize A".to_string());
        assert_eq!(tracker.all_active_regions().len(), 2);

        // End inner by name: must remove exactly "Initialize A", leaving outer.
        let events = tracker.end_region_by_name("Initialize A");
        assert_eq!(events.len(), 1);
        match &events[0] {
            RegionEvent::Ended {
                region_id,
                name,
                kind,
            } => {
                assert_eq!(*region_id, 2);
                assert_eq!(name, "Initialize A");
                assert_eq!(*kind, RegionKind::Stack);
            }
            other => panic!("expected Ended, got {:?}", other),
        }
        assert_eq!(tracker.all_active_regions().len(), 1);
        assert_eq!(tracker.all_active_regions()[0].name, "Initializations");

        let events = tracker.end_region_by_name("Initializations");
        assert_eq!(events.len(), 1);
        assert_eq!(tracker.all_active_regions().len(), 0);
    }

    #[test]
    fn test_end_by_name_out_of_order() {
        // Overlapping regions: close the OUTER first by name. The inner region
        // must remain on the stack (removed from the middle, not the top).
        let mut tracker = RegionTracker::new();
        tracker.enable_regions();

        tracker.begin_region(1, "outer".to_string());
        tracker.begin_region(2, "inner".to_string());

        let events = tracker.end_region_by_name("outer");
        assert_eq!(events.len(), 1);
        match &events[0] {
            RegionEvent::Ended {
                region_id, name, ..
            } => {
                assert_eq!(*region_id, 1);
                assert_eq!(name, "outer");
            }
            other => panic!("expected Ended, got {:?}", other),
        }
        // Only "inner" is left.
        assert_eq!(tracker.all_active_regions().len(), 1);
        assert_eq!(tracker.all_active_regions()[0].name, "inner");
    }

    #[test]
    fn test_end_by_name_mismatch_does_not_pop() {
        // Regression guard: ending an unknown name must NOT pop an unrelated
        // region off the top of the stack (which would corrupt nesting).
        let mut tracker = RegionTracker::new();
        tracker.enable_regions();

        tracker.begin_region(1, "outer".to_string());
        tracker.begin_region(2, "inner".to_string());

        let events = tracker.end_region_by_name("does_not_exist");
        assert!(events.is_empty(), "mismatched close must emit no events");

        // Both regions remain, in their original order.
        assert_eq!(tracker.all_active_regions().len(), 2);
        assert_eq!(tracker.all_active_regions()[0].name, "outer");
        assert_eq!(tracker.all_active_regions()[1].name, "inner");
    }

    #[test]
    fn test_event_region_switch() {
        let mut tracker = RegionTracker::new();
        tracker.enable_regions();

        // No previous event region: the end is a no-op, so start only.
        assert!(tracker.end_event_region().is_empty());
        let events = tracker.start_event_region(1, "event1".to_string());
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].started_region_id(), Some(1));
        assert!(tracker.is_any_region_active());

        // Switching closes the first, then opens the second.
        let mut events = tracker.end_event_region();
        events.extend(tracker.start_event_region(2, "event2".to_string()));
        assert_eq!(events.len(), 2);
        assert_eq!(events[0].ended_region_id(), Some(1));
        assert_eq!(events[1].started_region_id(), Some(2));
    }

    #[test]
    fn test_end_event_region() {
        let mut tracker = RegionTracker::new();

        tracker.start_event_region(1, "event1".to_string());
        let events = tracker.end_event_region();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].ended_region_id(), Some(1));
        assert!(!tracker.is_any_region_active());

        // Nothing open: no event.
        assert!(tracker.end_event_region().is_empty());
    }

    #[test]
    fn test_regions_disabled() {
        let mut tracker = RegionTracker::new();
        // Disable regions (they're enabled by default)
        tracker.disable_regions();

        let events = tracker.begin_region(1, "region1".to_string());
        assert_eq!(events.len(), 0);
        assert!(!tracker.is_any_region_active());
    }

    #[test]
    fn test_clear_all() {
        let mut tracker = RegionTracker::new();
        // Regions are enabled by default

        tracker.begin_region(1, "region1".to_string());
        tracker.begin_region(2, "region2".to_string());
        tracker.start_event_region(3, "event".to_string());
        assert!(tracker.is_any_region_active());

        tracker.clear_all();
        assert!(!tracker.is_any_region_active());
    }

    #[test]
    fn test_enable_disable_flags() {
        let mut tracker = RegionTracker::new();
        // Default state is enabled
        assert!(tracker.trace_enabled());
        assert!(tracker.regions_enabled());

        tracker.disable_trace();
        assert!(!tracker.trace_enabled());

        tracker.disable_regions();
        assert!(!tracker.regions_enabled());

        tracker.enable_trace();
        assert!(tracker.trace_enabled());

        tracker.enable_regions();
        assert!(tracker.regions_enabled());
    }

    /// A just-started *stack* region is already the top of the stack when the
    /// enricher processes its `Started` event, so its flamegraph parent is the
    /// entry below it.
    #[test]
    fn parent_flame_node_of_stack_region_skips_its_own_frame() {
        let mut tracker = RegionTracker::new();
        tracker.enable_regions();

        tracker.begin_region(1, "outer".to_string());
        tracker.set_stack_top_flame_node(7);
        // Top-level region: nothing encloses it.
        assert_eq!(tracker.parent_flame_node(RegionKind::Stack), FLAME_ROOT);

        tracker.begin_region(2, "inner".to_string());
        assert_eq!(tracker.parent_flame_node(RegionKind::Stack), 7);
        tracker.set_stack_top_flame_node(9);

        tracker.begin_region(3, "innermost".to_string());
        assert_eq!(tracker.parent_flame_node(RegionKind::Stack), 9);
    }

    /// An *event* region is never pushed onto the region stack -- it lives in
    /// `current_event_region` -- so its flamegraph parent is the innermost
    /// stack region, i.e. the top of the stack rather than the entry below it.
    /// Treating both kinds the same way dropped an event region's innermost
    /// parent, mis-attributing it one level too high in the call tree.
    #[test]
    fn parent_flame_node_of_event_region_is_the_stack_top() {
        let mut tracker = RegionTracker::new();
        tracker.enable_regions();

        // No enclosing stack region yet.
        assert_eq!(tracker.parent_flame_node(RegionKind::Event), FLAME_ROOT);

        tracker.begin_region(1, "outer".to_string());
        tracker.set_stack_top_flame_node(7);
        tracker.begin_region(2, "inner".to_string());
        tracker.set_stack_top_flame_node(9);

        // `inner` (node 9) is the innermost open region, so it is the parent.
        assert_eq!(tracker.parent_flame_node(RegionKind::Event), 9);

        tracker.start_event_region(3, "an_event".to_string());
        tracker.set_event_flame_node(11);
        assert_eq!(
            tracker.current_event_region().map(|r| r.flame_node),
            Some(11)
        );
        // Starting the event region must not have disturbed the stack, so a
        // nested stack region still parents under `inner`.
        assert_eq!(tracker.parent_flame_node(RegionKind::Event), 9);
    }
}
