//! Enricher module - coordinates instruction processing, ROI marker handling, and trace generation
//!
//! This module orchestrates the following sub-components:
//! - `vl_tracker`: Vector length and SEW state tracking
//! - `region_tracker`: ROI region stack management and event-based regions
//! - `marker_handler`: ROI marker detection and processing
//! - `stats_accumulator`: ROI statistics accumulation (functions, regions, global)
//! - `trace_emitter`: Synthetto trace generation and flushing
//! - `run_cache`: memoized per-run counter aggregates (batched enrichment)

pub(crate) mod annotate;
pub(crate) mod block_cache;
mod engine;
mod hotspots;
mod marker_handler;
mod pc_profile;
mod region_tracker;
pub(crate) mod run_cache;
mod stats_accumulator;
mod trace_emitter;
mod vl_tracker;

pub use engine::BatchSideChannels;
pub use engine::Enricher;
pub use engine::Modules;
pub use marker_handler::MarkerHandler;
pub use pc_profile::PcProfile;
pub use region_tracker::RegionEvent;
pub use stats_accumulator::StatsAccumulator;
pub use trace_emitter::SharedTrace;
pub use trace_emitter::TraceEmitter;
pub use vl_tracker::VlTracker;
