//! Synthetto trace generation for Perfetto visualization
//!
//! This module manages all aspects of generating Perfetto-compatible trace files
//! using the Synthetto library. It provides a clean interface for:
//!
//! - **Trace initialization**: Process, thread, and counter track creation
//! - **Region tracking**: Slice begin/end events for ROI regions
//! - **Counter updates**: VL (vector length) and VPU metrics with throttling
//! - **Incremental flushing**: Memory-efficient streaming to disk
//!
//! # Architecture
//!
//! The TraceEmitter owns all Synthetto state and provides a high-level API
//! that abstracts the low-level Perfetto protocol details. It manages:
//!
//! - Global instruction counter (timeline)
//! - Region tracking (stack-based and event-based)
//! - Counter tracks (VL per SEW, VPU metrics)
//! - Packet buffering and flushing
//!
//! # Usage
//!
//! ```no_run
//! use std::path::PathBuf;
//! # use turbo::processing::enricher::TraceEmitter;
//! # use turbo::common::RoiInfo;
//!
//! let mut emitter = TraceEmitter::new()
//!     .with_flush_threshold(100_000)
//!     .with_counter_emit_interval(1000);
//!
//! // Initialize trace generation
//! emitter.init().unwrap();
//! emitter.set_file_path(PathBuf::from("trace.perfetto"));
//!
//! // Process instructions
//! emitter.tick();  // Increment instruction counter
//!
//! // Emit VL counter update
//! emitter.emit_vl_counter(32, 16);  // SEW=32, VL=16
//!
//! // Emit VPU counters (throttled)
//! let roi = RoiInfo::new("region".to_string());
//! emitter.emit_vpu_counters(&roi);
//!
//! // Periodic flush check
//! emitter.maybe_flush().unwrap();
//!
//! // Final flush
//! emitter.flush().unwrap();
//! ```
//!
//! # Counter Throttling
//!
//! To prevent memory exhaustion on long traces, counter updates are throttled
//! by the `counter_emit_interval` setting (default: 1000 instructions).
//! The actual counter values remain per-instruction accurate; only the
//! trace emission frequency is reduced.
//!
//! # Incremental Flushing
//!
//! Packets are buffered in memory until the `flush_threshold` is reached
//! (default: 100,000 packets), then flushed to disk. The first flush writes
//! descriptor packets plus data; subsequent flushes append data only.

use anyhow::Result;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use synthetto::{CounterTrackUnit, Synthetto, TracePacket};

use super::region_tracker::{RegionEvent, RegionKind};
use crate::common::RoiInfo;

// ============================================================
// SharedTrace - single output file shared across all harts
// ============================================================

/// Shared state owned by one `Arc<Mutex<SharedTrace>>` and referenced by every per-hart
/// `TraceEmitter`. Holds the single `Synthetto` instance, the one `Process` descriptor,
/// and the pooled packet buffer that all harts drain into on flush.
pub struct SharedTrace {
    /// Single Synthetto instance that owns the UUID namespace
    pub synthetto: Synthetto,
    /// Single process descriptor (pid=1, name="traced_program")
    pub trace_process: synthetto::Process,
    /// Output file path for the combined trace
    trace_file_path: Option<PathBuf>,
    /// Whether descriptor packets have been written at least once
    trace_descriptors_written: bool,
    /// Pool of packets from all harts, flushed together
    pending_packets: Vec<TracePacket>,
    /// Number of pending packets before triggering a flush
    trace_flush_threshold: usize,
}

impl SharedTrace {
    /// Create a new `SharedTrace` with pid=1 process "traced_program".
    pub fn new() -> Self {
        let mut synthetto = Synthetto::new();
        let trace_process = synthetto.new_process(
            1,
            "traced_program".to_string(),
            vec!["traced_program".to_string()],
            Some(0),
        );
        Self {
            synthetto,
            trace_process,
            trace_file_path: None,
            trace_descriptors_written: false,
            pending_packets: Vec::new(),
            trace_flush_threshold: 100_000,
        }
    }

    /// Register a new hart thread and return its descriptor.
    ///
    /// This avoids split-borrow issues by encapsulating the call inside `SharedTrace`.
    pub fn new_hart_thread(&mut self, hart_id: i32, name: String) -> synthetto::Thread {
        // SAFETY: self.trace_process and self.synthetto are both owned by self;
        // we need to borrow both simultaneously. We do so by going through a raw
        // pointer for trace_process to avoid the split-borrow limitation.
        let process_ptr: *const synthetto::Process = &self.trace_process;
        // SAFETY: process_ptr is valid for the duration of this call and
        // synthetto.new_thread only reads from the process reference.
        let process_ref: &synthetto::Process = unsafe { &*process_ptr };
        self.synthetto.new_thread(process_ref, hart_id, name)
    }

    /// Create a new thread track on `thread` inside the shared Synthetto instance.
    pub fn new_thread_track(
        &mut self,
        name: String,
        thread: &synthetto::Thread,
    ) -> synthetto::Track<synthetto::Thread, synthetto::EventTrack> {
        self.synthetto.new_thread_track(name, thread)
    }

    /// Create a new thread counter track on `thread` inside the shared Synthetto instance.
    pub fn new_thread_counter_track(
        &mut self,
        name: String,
        unit: synthetto::CounterTrackUnit,
        thread: &synthetto::Thread,
    ) -> synthetto::Track<synthetto::Global, synthetto::CounterTrack> {
        self.synthetto
            .new_thread_counter_track(name, unit, 1, false, thread)
    }

    /// Set the output file path.
    pub fn set_file_path(&mut self, path: PathBuf) {
        self.trace_file_path = Some(path);
        self.trace_descriptors_written = false;
    }

    /// Set the flush threshold.
    pub fn set_flush_threshold(&mut self, threshold: usize) {
        self.trace_flush_threshold = threshold;
    }

    /// Encode and write/append all pending packets to disk, then drain them.
    pub fn write_to_disk(&mut self) -> Result<()> {
        if self.pending_packets.is_empty() {
            return Ok(());
        }

        let Some(ref path) = self.trace_file_path else {
            return Ok(());
        };

        // Always prepend descriptors so lazily-created tracks are visible in Perfetto.
        let mut all_packets = self.synthetto.complete_descriptor_trace();
        let descriptor_count = all_packets.len();
        all_packets.append(&mut self.pending_packets);
        let packet_count = all_packets.len() - descriptor_count;
        let encoded = synthetto::encode_trace(all_packets);

        if !self.trace_descriptors_written {
            std::fs::write(path, &encoded)?;
            self.trace_descriptors_written = true;
            log::debug!(
                "SharedTrace: wrote descriptors + {} packets to {:?}",
                packet_count,
                path
            );
        } else {
            use std::io::Write;
            let mut file = std::fs::OpenOptions::new().append(true).open(path)?;
            file.write_all(&encoded)?;
            log::debug!(
                "SharedTrace: appended {} descriptors + {} packets to {:?}",
                descriptor_count,
                packet_count,
                path
            );
        }

        Ok(())
    }
}

impl Default for SharedTrace {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================
// TraceEmitter - per-hart emitter
// ============================================================

/// Manages Synthetto trace generation for performance visualization
///
/// This module owns all Synthetto state and handles:
/// - Initialization of trace process, thread, and counter tracks
/// - Region event tracking (slice begin/end)
/// - Counter updates with throttling
/// - Incremental flushing to disk
pub struct TraceEmitter {
    /// Optional shared state (set when using multi-hart single-file mode)
    shared: Option<Arc<Mutex<SharedTrace>>>,

    /// Synthetto trace generator instance (used only when `shared` is None)
    synthetto: Option<Synthetto>,

    /// Accumulated trace packets ready to be flushed
    trace_packets: Vec<TracePacket>,

    /// Process descriptor for the trace (used only when `shared` is None)
    trace_process: Option<synthetto::Process>,

    /// Main thread descriptor
    trace_thread: Option<synthetto::Thread>,

    /// Counter tracks for VL (vector length) per SEW (selected element width)
    trace_vl_counter_e8: Option<synthetto::Track<synthetto::Global, synthetto::CounterTrack>>,
    trace_vl_counter_e16: Option<synthetto::Track<synthetto::Global, synthetto::CounterTrack>>,
    trace_vl_counter_e32: Option<synthetto::Track<synthetto::Global, synthetto::CounterTrack>>,
    trace_vl_counter_e64: Option<synthetto::Track<synthetto::Global, synthetto::CounterTrack>>,

    /// VPU (Vector Processing Unit) counter tracks
    trace_vector_elements: Option<synthetto::Track<synthetto::Global, synthetto::CounterTrack>>,
    trace_vector_load_elements:
        Option<synthetto::Track<synthetto::Global, synthetto::CounterTrack>>,

    /// Tracks for stack-based ROI regions, keyed by region_id
    trace_region_tracks: HashMap<usize, synthetto::Track<synthetto::Thread, synthetto::EventTrack>>,

    /// Tracks for event-based regions, keyed by region name
    trace_event_region_tracks:
        HashMap<String, synthetto::Track<synthetto::Thread, synthetto::EventTrack>>,

    /// Hart ID for the traced hardware thread (hart_id == thread_id in RISC-V)
    hart_id: u32,

    /// Global instruction counter (timeline)
    global_instruction_count: u64,

    /// Stack of active region spans: (region_id, start_instruction_count)
    trace_active_spans: Vec<(usize, u64)>,

    /// Output file path for trace
    trace_file_path: Option<PathBuf>,

    /// Whether descriptor packets have been written
    trace_descriptors_written: bool,

    /// Number of packets to accumulate before flushing to disk
    trace_flush_threshold: usize,

    /// Emit counter track events every N instructions (0 = every instruction)
    trace_counter_emit_interval: u64,

    /// Instruction count at last counter emit (for throttling)
    trace_last_counter_emit: u64,
}

impl TraceEmitter {
    /// Create a new TraceEmitter with default settings
    pub fn new() -> Self {
        Self {
            shared: None,
            synthetto: None,
            trace_packets: Vec::new(),
            trace_process: None,
            trace_thread: None,
            trace_vl_counter_e8: None,
            trace_vl_counter_e16: None,
            trace_vl_counter_e32: None,
            trace_vl_counter_e64: None,
            trace_vector_elements: None,
            trace_vector_load_elements: None,
            trace_region_tracks: HashMap::new(),
            trace_event_region_tracks: HashMap::new(),
            hart_id: 0,
            global_instruction_count: 0,
            trace_active_spans: Vec::new(),
            trace_file_path: None,
            trace_descriptors_written: false,
            trace_flush_threshold: 100_000,
            trace_counter_emit_interval: 1000,
            trace_last_counter_emit: 0,
        }
    }

    /// Attach this emitter to a `SharedTrace`.
    ///
    /// Must be called before `init()`. When a shared trace is set, `init()` registers
    /// this hart's thread inside the shared `Synthetto` instance and all packets are
    /// pooled into the shared buffer on flush.
    pub fn with_shared_trace(mut self, shared: Arc<Mutex<SharedTrace>>) -> Self {
        self.shared = Some(shared);
        self
    }

    /// Initialize Synthetto trace generation
    ///
    /// Creates the process, thread, and all counter tracks.
    /// Safe to call multiple times - only initializes once.
    pub fn init(&mut self) -> Result<()> {
        // Already initialized?
        if self.trace_thread.is_some() {
            return Ok(());
        }

        // No sink configured (neither `--perfetto`'s shared trace nor a
        // standalone output path) - skip creating Synthetto state entirely.
        // Leaving every track `Option` as `None` makes `handle_region_event`,
        // `emit_vl_counter`, and `emit_vpu_counters` no-ops, since they all
        // guard on these fields being `Some`. Without this check, tracing
        // state (and the per-instruction `trace_packets` buffer it feeds)
        // was built and grown for the entire run even when no trace was
        // ever requested, which was the dominant cause of OOM on long runs.
        if self.shared.is_none() && self.trace_file_path.is_none() {
            return Ok(());
        }

        if let Some(ref shared_arc) = self.shared.clone() {
            // === Shared-trace path: register this hart's thread in the shared Synthetto ===
            let mut shared = shared_arc.lock().unwrap();
            let thread_name = format!("hart_{}", self.hart_id);
            let thread = shared.new_hart_thread(self.hart_id as i32, thread_name);

            let vl_counter_e8 = shared.new_thread_counter_track(
                "VL e8".to_string(),
                CounterTrackUnit::Custom("elements".to_string()),
                &thread,
            );
            let vl_counter_e16 = shared.new_thread_counter_track(
                "VL e16".to_string(),
                CounterTrackUnit::Custom("elements".to_string()),
                &thread,
            );
            let vl_counter_e32 = shared.new_thread_counter_track(
                "VL e32".to_string(),
                CounterTrackUnit::Custom("elements".to_string()),
                &thread,
            );
            let vl_counter_e64 = shared.new_thread_counter_track(
                "VL e64".to_string(),
                CounterTrackUnit::Custom("elements".to_string()),
                &thread,
            );
            let vector_elements = shared.new_thread_counter_track(
                "Vector Elements".to_string(),
                CounterTrackUnit::Custom("elements".to_string()),
                &thread,
            );
            let vector_load_elements = shared.new_thread_counter_track(
                "Vector Load Elements".to_string(),
                CounterTrackUnit::Custom("elements".to_string()),
                &thread,
            );

            drop(shared); // release lock before storing

            self.trace_thread = Some(thread);
            self.trace_vl_counter_e8 = Some(vl_counter_e8);
            self.trace_vl_counter_e16 = Some(vl_counter_e16);
            self.trace_vl_counter_e32 = Some(vl_counter_e32);
            self.trace_vl_counter_e64 = Some(vl_counter_e64);
            self.trace_vector_elements = Some(vector_elements);
            self.trace_vector_load_elements = Some(vector_load_elements);
        } else {
            // === Standalone path (unit tests / no shared trace) ===
            let mut synthetto = Synthetto::new();

            // Use a fixed synthetic PID of 1 to represent the traced program.
            const TRACED_PROGRAM_PID: i32 = 1;
            let process = synthetto.new_process(
                TRACED_PROGRAM_PID,
                "traced_program".to_string(),
                vec!["traced_program".to_string()],
                Some(0),
            );

            let thread_name = format!("hart_{}", self.hart_id);
            let thread = synthetto.new_thread(&process, self.hart_id as i32, thread_name);

            let vl_counter_e8 = synthetto.new_thread_counter_track(
                "VL e8".to_string(),
                CounterTrackUnit::Custom("elements".to_string()),
                1,
                false,
                &thread,
            );
            let vl_counter_e16 = synthetto.new_thread_counter_track(
                "VL e16".to_string(),
                CounterTrackUnit::Custom("elements".to_string()),
                1,
                false,
                &thread,
            );
            let vl_counter_e32 = synthetto.new_thread_counter_track(
                "VL e32".to_string(),
                CounterTrackUnit::Custom("elements".to_string()),
                1,
                false,
                &thread,
            );
            let vl_counter_e64 = synthetto.new_thread_counter_track(
                "VL e64".to_string(),
                CounterTrackUnit::Custom("elements".to_string()),
                1,
                false,
                &thread,
            );
            let vector_elements = synthetto.new_thread_counter_track(
                "Vector Elements".to_string(),
                CounterTrackUnit::Custom("elements".to_string()),
                1,
                false,
                &thread,
            );
            let vector_load_elements = synthetto.new_thread_counter_track(
                "Vector Load Elements".to_string(),
                CounterTrackUnit::Custom("elements".to_string()),
                1,
                false,
                &thread,
            );

            self.synthetto = Some(synthetto);
            self.trace_process = Some(process);
            self.trace_thread = Some(thread);
            self.trace_vl_counter_e8 = Some(vl_counter_e8);
            self.trace_vl_counter_e16 = Some(vl_counter_e16);
            self.trace_vl_counter_e32 = Some(vl_counter_e32);
            self.trace_vl_counter_e64 = Some(vl_counter_e64);
            self.trace_vector_elements = Some(vector_elements);
            self.trace_vector_load_elements = Some(vector_load_elements);
        }

        Ok(())
    }

    /// Handle region lifecycle events and emit trace slices
    pub fn handle_region_event(&mut self, event: &RegionEvent) {
        // Guard: only proceed if initialized
        if self.trace_thread.is_none() {
            return;
        }

        match event {
            RegionEvent::Started {
                region_id,
                name,
                kind,
            } => {
                // Create the track on first sight. Stack regions get one track
                // per region instance; event regions share a track per name so
                // repeated firings of the same event stack up on one row.
                let track = match kind {
                    RegionKind::Stack => {
                        if !self.trace_region_tracks.contains_key(region_id) {
                            if let Some(track) = self.new_thread_track(name.clone()) {
                                self.trace_region_tracks.insert(*region_id, track);
                            }
                        }
                        self.trace_region_tracks.get(region_id)
                    }
                    RegionKind::Event => {
                        if !self.trace_event_region_tracks.contains_key(name) {
                            if let Some(track) = self.new_thread_track(name.clone()) {
                                self.trace_event_region_tracks.insert(name.clone(), track);
                            }
                        }
                        self.trace_event_region_tracks.get(name)
                    }
                };

                if let Some(track) = track {
                    let packet =
                        track.slice_begin_evt(self.global_instruction_count, Some(name.clone()));
                    self.trace_packets.push(packet);
                    self.trace_active_spans
                        .push((*region_id, self.global_instruction_count));
                }
            }

            RegionEvent::Ended {
                region_id,
                name,
                kind,
            } => {
                let track = match kind {
                    RegionKind::Stack => self.trace_region_tracks.get(region_id),
                    RegionKind::Event => self.trace_event_region_tracks.get(name),
                };

                if let Some(track) = track {
                    let packet = track.slice_end_evt(self.global_instruction_count);
                    self.trace_packets.push(packet);
                    self.trace_active_spans.retain(|(id, _)| id != region_id);
                }
            }
        }
    }

    /// Create a new thread-scoped event track, using either the shared or standalone Synthetto.
    fn new_thread_track(
        &mut self,
        name: String,
    ) -> Option<synthetto::Track<synthetto::Thread, synthetto::EventTrack>> {
        let thread = self.trace_thread.as_ref()?;
        if let Some(ref shared_arc) = self.shared.clone() {
            let mut shared = shared_arc.lock().unwrap();
            Some(shared.new_thread_track(name, thread))
        } else {
            self.synthetto
                .as_mut()
                .map(|synthetto| synthetto.new_thread_track(name, thread))
        }
    }

    /// Emit a VL (vector length) counter update for a specific SEW
    pub fn emit_vl_counter(&mut self, sew: u8, vl: u32) {
        let counter_track = match sew {
            8 => self.trace_vl_counter_e8.as_ref(),
            16 => self.trace_vl_counter_e16.as_ref(),
            32 => self.trace_vl_counter_e32.as_ref(),
            64 => self.trace_vl_counter_e64.as_ref(),
            _ => {
                log::warn!("Unknown SEW value for VL counter: {}", sew);
                return;
            }
        };

        if let Some(track) = counter_track {
            let packet = track.int_counter_evt(self.global_instruction_count, vl as i64);
            self.trace_packets.push(packet);
        }
    }

    /// Whether the next [`emit_vpu_counters`](Self::emit_vpu_counters) would
    /// actually emit anything.
    ///
    /// Exposed so the enricher's per-instruction path can skip the call
    /// entirely: at the default interval this is false for all but one
    /// instruction in thousands, yet the (non-inlined, `&RoiInfo`-taking) call
    /// itself was ~2.3% of samples just to return immediately.
    #[inline(always)]
    pub fn should_emit_vpu_counters(&self) -> bool {
        self.trace_counter_emit_interval == 0
            || (self.global_instruction_count - self.trace_last_counter_emit)
                >= self.trace_counter_emit_interval
    }

    /// Emit throttled VPU counter updates
    ///
    /// Only emits if the counter emit interval has been reached.
    /// The actual counter values remain per-instruction accurate;
    /// only the trace emission frequency is reduced.
    pub fn emit_vpu_counters(&mut self, global_roi: &RoiInfo) {
        if !self.should_emit_vpu_counters() {
            return;
        }

        // Emit vector elements counter
        if let Some(track) = self.trace_vector_elements.as_ref() {
            let packet = track.int_counter_evt(
                self.global_instruction_count,
                global_roi.vector_elements as i64,
            );
            self.trace_packets.push(packet);
        }

        // Emit vector load elements counter
        if let Some(track) = self.trace_vector_load_elements.as_ref() {
            let packet = track.int_counter_evt(
                self.global_instruction_count,
                global_roi.vector_load_elements as i64,
            );
            self.trace_packets.push(packet);
        }

        // Update last emit timestamp
        self.trace_last_counter_emit = self.global_instruction_count;
    }

    /// Increment the instruction counter (timeline)
    pub fn tick(&mut self) {
        self.global_instruction_count += 1;
    }

    /// Advance the instruction counter by `n`, emitting counter samples at
    /// **exactly** the instruction counts that `n` repetitions of
    /// [`should_emit_vpu_counters`](Self::should_emit_vpu_counters) +
    /// [`emit_vpu_counters`](Self::emit_vpu_counters) + [`tick`](Self::tick)
    /// would have.
    ///
    /// The enricher's batched path commits a whole run's counters in one
    /// `add_run_delta` and then had to run `for _ in 0..n { tick() }` purely so
    /// the Perfetto timeline kept its sample points -- 23.4M iterations, each with
    /// an interval test, on a `flash_full` run. The sample points are a closed
    /// form of `(global_instruction_count, trace_last_counter_emit,
    /// trace_counter_emit_interval)`, so they can be hit directly. The emitted
    /// *values* are unchanged too: `global_roi` is already the post-run snapshot
    /// at every one of those points, exactly as in the loop.
    ///
    /// `interval == 0` means "emit every instruction"; it is not a configuration
    /// any measured run uses, so it just walks the counts.
    pub fn tick_n(&mut self, n: u64, global_roi: &RoiInfo) {
        if n == 0 {
            return;
        }
        let end = self.global_instruction_count + n;

        if self.trace_counter_emit_interval == 0 {
            while self.global_instruction_count < end {
                self.emit_vpu_counters(global_roi);
                self.global_instruction_count += 1;
            }
            return;
        }

        // First due point at or after the current count, then one per interval.
        // `emit_vpu_counters` sets `trace_last_counter_emit` to the count it
        // emitted at, so the stride is the interval from there on.
        let interval = self.trace_counter_emit_interval;
        let mut at = self
            .global_instruction_count
            .max(self.trace_last_counter_emit.saturating_add(interval));
        while at < end {
            self.global_instruction_count = at;
            self.emit_vpu_counters(global_roi);
            at += interval;
        }
        self.global_instruction_count = end;
    }

    /// Set the output file path for trace
    pub fn set_file_path(&mut self, path: PathBuf) {
        self.trace_file_path = Some(path);
        self.trace_descriptors_written = false;
    }

    /// Flush accumulated trace packets to disk (or into the shared pool).
    pub fn flush(&mut self) -> Result<()> {
        if self.trace_packets.is_empty() {
            return Ok(());
        }

        if let Some(ref shared_arc) = self.shared {
            // Move per-hart packets into the shared pool, then maybe write to disk.
            let mut shared = shared_arc.lock().unwrap();
            shared.pending_packets.append(&mut self.trace_packets);
            if shared.pending_packets.len() >= shared.trace_flush_threshold {
                shared.write_to_disk()?;
            }
            return Ok(());
        }

        // Standalone path (no shared trace).
        let Some(ref path) = self.trace_file_path else {
            return Ok(());
        };

        let Some(ref synthetto) = self.synthetto else {
            log::warn!("Cannot flush trace: synthetto not initialized");
            return Ok(());
        };

        // Always prepend descriptor packets so that lazily-created region
        // tracks are known to Perfetto. Duplicate descriptors are harmless.
        let mut all_packets = synthetto.complete_descriptor_trace();
        let descriptor_count = all_packets.len();
        all_packets.append(&mut self.trace_packets);
        let packet_count = all_packets.len() - descriptor_count;
        let encoded = synthetto::encode_trace(all_packets);

        if !self.trace_descriptors_written {
            std::fs::write(path, &encoded)?;
            self.trace_descriptors_written = true;
            log::debug!(
                "Flushed trace descriptors + {} packets to {:?}",
                packet_count,
                path
            );
        } else {
            use std::io::Write;
            let mut file = std::fs::OpenOptions::new().append(true).open(path)?;
            file.write_all(&encoded)?;
            log::debug!(
                "Flushed {} descriptors + {} packets to {:?}",
                descriptor_count,
                packet_count,
                path
            );
        }

        Ok(())
    }

    /// Check if trace packets need flushing and flush if threshold exceeded.
    pub fn maybe_flush(&mut self) -> Result<()> {
        if self.trace_packets.len() >= self.trace_flush_threshold {
            self.flush()?;
        }
        Ok(())
    }

    /// Write the synthetto trace to a file (legacy compatibility)
    ///
    /// This sets the path if not already set and flushes remaining packets.
    pub fn write(&mut self, path: &Path) -> Result<()> {
        // Set the path if not already set
        if self.trace_file_path.is_none() {
            self.set_file_path(path.to_path_buf());
        }

        // Flush any remaining packets
        self.flush()?;

        if self.trace_descriptors_written {
            log::info!("Wrote Perfetto trace to {:?}", path);
        } else {
            log::warn!("No trace data was written");
        }

        Ok(())
    }

    /// Set the flush threshold (number of packets before automatic flush)
    pub fn with_flush_threshold(mut self, threshold: usize) -> Self {
        self.trace_flush_threshold = threshold;
        self
    }

    /// Set the counter emit interval (emit every N instructions)
    ///
    /// Setting this to 0 will emit on every instruction (not recommended for long traces).
    /// The default is 1000 instructions.
    ///
    /// The actual counter values remain per-instruction accurate; only the trace emission
    /// frequency is reduced to prevent OOM errors on long traces.
    pub fn with_counter_emit_interval(mut self, interval: u64) -> Self {
        self.trace_counter_emit_interval = interval;
        self
    }

    /// Set the hart ID for the traced hardware thread
    ///
    /// In RISC-V, a hart (hardware thread) maps directly to a thread_id.
    /// This must be called before `init()` to take effect.
    pub fn with_hart_id(mut self, hart_id: u32) -> Self {
        self.hart_id = hart_id;
        self
    }

    /// Set the hart ID directly (mutation version, effective before `init()` is called).
    pub fn set_hart_id(&mut self, hart_id: u32) {
        log::debug!("TraceEmitter: hart_id set to {}", hart_id);
        self.hart_id = hart_id;
    }

    /// Get the current instruction count
    #[cfg(test)]
    pub fn instruction_count(&self) -> u64 {
        self.global_instruction_count
    }

    /// Check if trace is initialized
    #[cfg(test)]
    pub fn is_initialized(&self) -> bool {
        self.synthetto.is_some() || self.trace_thread.is_some()
    }

    /// Get the number of pending packets
    #[cfg(test)]
    pub fn pending_packets(&self) -> usize {
        self.trace_packets.len()
    }
}

impl Default for TraceEmitter {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_trace_emitter_creation() {
        let emitter = TraceEmitter::new();
        assert!(!emitter.is_initialized());
        assert_eq!(emitter.instruction_count(), 0);
        assert_eq!(emitter.pending_packets(), 0);
    }

    #[test]
    fn test_trace_emitter_initialization() {
        let mut emitter = TraceEmitter::new();
        // A sink (standalone output path here, or `with_shared_trace` for the
        // `--perfetto` path) must be configured, otherwise `init()` is a
        // deliberate no-op (see the "avoid creating traces in memory when
        // disabled" fix) and `is_initialized()` stays false.
        emitter.set_file_path(PathBuf::from(
            "/tmp/test_trace_emitter_initialization.pftrace",
        ));
        assert!(emitter.init().is_ok());
        assert!(emitter.is_initialized());

        // Second init should be no-op
        assert!(emitter.init().is_ok());
    }

    #[test]
    fn test_instruction_counter() {
        let mut emitter = TraceEmitter::new();
        assert_eq!(emitter.instruction_count(), 0);

        emitter.tick();
        assert_eq!(emitter.instruction_count(), 1);

        emitter.tick();
        emitter.tick();
        assert_eq!(emitter.instruction_count(), 3);
    }

    /// `tick_n` exists purely to avoid running the loop, so it must be
    /// indistinguishable from it: the same final instruction count and the same
    /// counter-emission instruction counts, for every interval. If this fails,
    /// the batched enricher path silently moves the Perfetto sample points and the
    /// `.pb` output stops being byte-identical.
    #[test]
    fn tick_n_emits_at_the_same_instruction_counts_as_n_ticks() {
        let roi = RoiInfo::new("t".to_string());

        // The emission points are `trace_last_counter_emit` updates, which are
        // observable without a configured sink -- `emit_vpu_counters` records the
        // count it emitted at whether or not any track exists.
        fn looped(interval: u64, start: u64, n: u64, roi: &RoiInfo) -> (u64, Vec<u64>) {
            let mut e = TraceEmitter::new().with_counter_emit_interval(interval);
            e.global_instruction_count = start;
            let mut points = Vec::new();
            for _ in 0..n {
                if e.should_emit_vpu_counters() {
                    points.push(e.global_instruction_count);
                    e.emit_vpu_counters(roi);
                }
                e.tick();
            }
            (e.global_instruction_count, points)
        }

        for interval in [0u64, 1, 7, 1000] {
            for (start, n) in [(0u64, 1u64), (0, 1000), (0, 2500), (5, 9), (999, 3), (0, 0)] {
                let (want_count, want_points) = looped(interval, start, n, &roi);

                let mut e = TraceEmitter::new().with_counter_emit_interval(interval);
                e.global_instruction_count = start;
                let mut got_points = Vec::new();
                // Same observation, via the emitted-at watermark: record the
                // watermark before and after and reconstruct nothing -- instead
                // drive one `tick_n` and compare the watermark and count, then
                // re-derive the points by stepping one at a time through a clone
                // of the closed form.
                let before = e.trace_last_counter_emit;
                e.tick_n(n, &roi);
                assert_eq!(
                    e.global_instruction_count, want_count,
                    "interval={interval} start={start} n={n}: instruction count"
                );
                if let Some(&last) = want_points.last() {
                    assert_eq!(
                        e.trace_last_counter_emit, last,
                        "interval={interval} start={start} n={n}: last emission point"
                    );
                } else {
                    assert_eq!(
                        e.trace_last_counter_emit, before,
                        "interval={interval} start={start} n={n}: must not emit"
                    );
                }

                // And the full point list, by replaying `tick_n` one instruction
                // at a time (`tick_n(1)` must equal one loop iteration).
                let mut e1 = TraceEmitter::new().with_counter_emit_interval(interval);
                e1.global_instruction_count = start;
                for _ in 0..n {
                    let at = e1.global_instruction_count;
                    // The same predicate the loop tests, evaluated at the same
                    // instruction count: `tick_n(1)` must emit exactly when
                    // `tick()` preceded by the guard would have.
                    if e1.should_emit_vpu_counters() {
                        got_points.push(at);
                    }
                    e1.tick_n(1, &roi);
                }
                assert_eq!(
                    got_points, want_points,
                    "interval={interval} start={start} n={n}: emission points"
                );
                assert_eq!(e1.global_instruction_count, want_count);
            }
        }
    }

    #[test]
    fn test_builder_pattern() {
        let emitter = TraceEmitter::new()
            .with_flush_threshold(50_000)
            .with_counter_emit_interval(500);

        assert_eq!(emitter.trace_flush_threshold, 50_000);
        assert_eq!(emitter.trace_counter_emit_interval, 500);
    }
}
