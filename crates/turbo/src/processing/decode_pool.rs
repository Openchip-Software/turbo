//! Parallel decode+enrich: one worker thread per hart, bounded pool.
//!
//! Decode+enrich is ~94% of a run's wall clock (`[phase-timing]` on a measured workload:
//! `run=9.1s, of which enrich=8.6s`), and it is the one stage that was still
//! single-threaded. Harts are the natural axis: each has its own record log, its
//! own `HartDecoder` (decode state is a sequential stream per hart) and its own
//! `Enricher`, with no cross-hart state except the Perfetto packet pool.
//!
//! ## Shape
//!
//! The caller keeps polling the source on its own thread and hands each decoded
//! `Batch` to this pool, keyed by hart. Reading and framing a record log is a
//! file read plus a `postcard` header walk — a few percent of the work — so
//! funnelling it through one thread costs little, and keeps the whole source
//! layer (discovery, the memory-map handshake, `--vcpu` filtering) exactly as it
//! is. What gets parallelised is the expensive part.
//!
//! Per-hart message ORDER is preserved: a hart's jobs go down one channel to one
//! worker, which processes them in order. That is not an optimisation detail — a
//! hart's trace is one continuous qualification stream, so decoding its chunks
//! out of order would produce garbage.
//!
//! ## Bounds
//!
//! Channels are bounded, so a fast reader cannot outrun the workers and turn a
//! 1.2 GB recording into 1.2 GB of queued jobs. Blocking the reader is harmless:
//! the QEMU plugin writes the spool to disk independently, so there is no
//! backpressure path to the guest.
//!
//! Threads are capped at [`MAX_DECODE_THREADS`]. Beyond that, harts share
//! workers round-robin — each still gets its own decoder and enricher, they just
//! take turns on a thread. That also caps memory, which matters: each hart's
//! enricher carries its own decode cache (~73 MiB/hart measured on a workload).

use std::collections::HashMap;
use std::sync::Mutex;
use std::time::{Duration, Instant};

use std::sync::mpsc::{sync_channel, Receiver, SyncSender};
use turbo_tracer_shared::RoiRtMarker;

use crate::common::PerformanceData;
use crate::processing::enricher::BatchSideChannels;
use crate::processing::{self, DataProcessor, Enricher};
use crate::RerfError;

/// Hard cap on decode+enrich worker threads.
pub const MAX_DECODE_THREADS: usize = 8;

/// Jobs queued per worker before the reader blocks.
///
/// Small on purpose: each job owns its trace bytes (a flush window, ~40 KB on
/// BBT), so this bounds queued memory to a few MB per worker while still
/// absorbing the jitter between the reader and a worker mid-batch.
const JOB_QUEUE_DEPTH: usize = 32;

/// One hart's flush window, moved to whichever worker owns that hart.
pub struct DecodeJob {
    pub hart_id: u32,
    pub trace_data: Vec<u8>,
    pub roi_markers: Vec<RoiRtMarker>,
}

/// What a worker hands back once its channel closes.
pub struct WorkerOutput {
    /// This worker's harts, for the caller's finalization (annotate merge, vl
    /// completeness gate, per-hart and merged `PerformanceData`). Returned rather
    /// than finalized here so all of that logic stays in one place, unaware that
    /// enrichment was threaded.
    pub enrichers: HashMap<u32, Enricher>,
    pub insn_trace_bytes: u64,
    pub vl_bytes: u64,
    pub instruction_count: u64,
    /// CPU time this worker spent inside decode+enrich. Summed across workers
    /// this is CPU time, NOT wall clock — the two diverge once there is more
    /// than one worker, which is the whole point.
    pub busy: Duration,
    /// Every PC this worker decoded, in stream order per hart. Only populated
    /// under `WorkerContext::dump_pcs`; empty otherwise, so the normal path
    /// pays nothing.
    pub pcs: Vec<u64>,
}

/// Builds a fresh `Enricher` per hart. A struct rather than a closure so workers
/// can share one by reference across threads.
///
/// Each hart needs its own enricher: the `TraceEmitter` is initialized once with
/// the hart id, and instruction counters / region stacks are per hart.
pub struct EnricherFactory {
    /// The decoders and symbol resolvers, built once and shared by every hart
    /// — see [`Modules`](processing::enricher::Modules). This used to be
    /// rebuilt inside each `Enricher`, duplicating a full decode table and
    /// DWARF parse per worker.
    pub modules: std::sync::Arc<processing::enricher::Modules>,
    pub workdir: Option<std::path::PathBuf>,
    pub source_dirs: Vec<std::path::PathBuf>,
    pub shared_trace: Option<std::sync::Arc<Mutex<processing::enricher::SharedTrace>>>,
    pub annotate: bool,
    /// `(VLEN in bytes, cache config)` when `--cache-sim` asked for a modelled
    /// cache. Each hart's enricher gets its own model built from this: private
    /// per hart, never shared. `None` disables the feature entirely.
    pub cache_sim: Option<(u64, crate::processing::cache_model::Config)>,
    /// Feed each hart's enricher's streaming ROI flamegraph builder
    /// (`--no-roi-flamegraph` clears it). Aggregation is bounded and cheap, so
    /// this is on by default; see `Enricher::with_roi_flamegraph`.
    pub roi_flamegraph: bool,
    pub perfetto: bool,
    pub trace_flush_threshold: usize,
    pub trace_counter_interval: u64,
}

impl EnricherFactory {
    pub fn build(&self, hart_id: u32) -> Enricher {
        let mut e = Enricher::with_modules(self.modules.clone());
        if let Some(ref wd) = self.workdir {
            e = e.with_output_dir(wd.clone());
        }
        // Enable the per-PC source/asm annotation profile that produces the
        // static annotate_*.html report (independent --annotate opt-in).
        e = e.with_annotate(self.annotate);
        if let Some((vlenb, config)) = self.cache_sim {
            // The config was validated once before the pool was built, so the
            // only way this fails here is a programming error.
            e = e
                .with_cache_sim(vlenb, config)
                .expect("the cache configuration was validated before the decode pool was built");
        }
        e = e.with_roi_flamegraph(self.roi_flamegraph);
        e = e.with_source_dirs(self.source_dirs.clone());
        e = e.with_hart_id(hart_id);
        if self.perfetto {
            if let Some(ref arc) = self.shared_trace {
                e = e.with_shared_trace(arc.clone());
            }
            e = e.with_trace_flush_threshold(self.trace_flush_threshold);
            e = e.with_trace_counter_emit_interval(self.trace_counter_interval);
        }
        e
    }
}

/// A worker's published view of its own harts, for the live web UI.
///
/// Workers cannot build the cross-hart snapshot themselves — that would need
/// every enricher, which is precisely what is distributed. So each publishes its
/// own harts' merged `PerformanceData` here and the reader thread combines the
/// slots. Each slot is written by exactly one worker, so the lock is uncontended
/// except against the reader's periodic read.
pub type SnapshotSlots = Vec<Mutex<Option<PerformanceData>>>;

/// Everything a worker needs that is shared and read-only.
pub struct WorkerContext<'a> {
    pub data_processor: &'a DataProcessor,
    pub factory: &'a EnricherFactory,
    pub slots: &'a SnapshotSlots,
    /// Emit live snapshots at all (false under `--no-server`: nothing subscribes).
    pub emit_snapshots: bool,
    /// Minimum gap between a worker's snapshot publications.
    pub snapshot_interval: Duration,
    /// Measurement mode: decode, count instructions, enrich nothing.
    pub decode_only: bool,
    /// Dev cross-check mode (hidden `--dump-pcs`): like `decode_only`, but also
    /// keep every decoded PC so the caller can write them out for comparison
    /// against an independent trace (execlog / Paraver).
    pub dump_pcs: bool,
}

/// Create the channel pair for one worker.
pub fn job_channel() -> (SyncSender<DecodeJob>, Receiver<DecodeJob>) {
    sync_channel(JOB_QUEUE_DEPTH)
}

/// Decode+enrich every job for this worker's harts, in order, until the channel
/// closes.
///
/// Returns on the first decode/enrich error, which drops the receiver and so
/// makes the reader's next `send` fail — that is how the error stops the
/// pipeline without a separate cancellation flag.
pub fn run_worker(
    worker_index: usize,
    jobs: Receiver<DecodeJob>,
    ctx: &WorkerContext<'_>,
) -> Result<WorkerOutput, RerfError> {
    // One persistent decoder per hart: each flush window is a chunk of a single
    // continuous qualification stream, so decode state must carry across them.
    let mut decoders: HashMap<u32, processing::HartDecoder> = HashMap::new();
    let mut enrichers: HashMap<u32, Enricher> = HashMap::new();

    let mut out = WorkerOutput {
        enrichers: HashMap::new(),
        insn_trace_bytes: 0,
        vl_bytes: 0,
        instruction_count: 0,
        busy: Duration::ZERO,
        pcs: Vec::new(),
    };
    let mut last_snapshot = Instant::now();

    for job in jobs {
        let DecodeJob {
            hart_id,
            trace_data,
            roi_markers,
        } = job;
        if trace_data.is_empty() {
            continue;
        }

        let t0 = Instant::now();
        let decoder = decoders
            .entry(hart_id)
            .or_insert_with(|| ctx.data_processor.make_decoder());

        if ctx.decode_only || ctx.dump_pcs {
            // Measurement / cross-check harness. The decode call is IDENTICAL to
            // the enriching one below -- same batching, same materialization --
            // and only the consumer differs, so later speedups are judged
            // against today's real decode path rather than a trimmed version of
            // it, and dumped PCs are exactly the PCs enrichment would have seen.
            let keep = ctx.dump_pcs;
            let mut insns: u64 = 0;
            let mut decode_err: Option<RerfError> = None;
            ctx.data_processor
                .decode_into(decoder, &trace_data, |batch| match batch {
                    processing::TraceBatch::Pcs { pcs, .. } => {
                        insns += pcs.len() as u64;
                        if keep {
                            out.pcs.extend_from_slice(pcs);
                        }
                    }
                    processing::TraceBatch::Blocks(blocks) => {
                        for block in blocks {
                            match block {
                                Ok(b) => {
                                    insns += b.pcs.len() as u64;
                                    if keep {
                                        out.pcs.extend_from_slice(b.pcs);
                                    }
                                }
                                Err(e) => {
                                    if decode_err.is_none() {
                                        decode_err = Some(e);
                                    }
                                }
                            }
                        }
                    }
                })?;
            if let Some(e) = decode_err {
                return Err(e);
            }
            out.insn_trace_bytes += trace_data.len() as u64;
            out.instruction_count += insns;
            out.busy += t0.elapsed();
            continue;
        }

        // A single flush window may decode into several batches, but its
        // side-channels are per-window aggregates that must reach the enricher
        // exactly ONCE. `take_once` hands them to the FIRST batch and gives every
        // later batch of the same window an empty set. See `BatchSideChannels`.
        let window_side = BatchSideChannels {
            roi_markers: &roi_markers,
        };
        let mut first_batch = true;
        let mut enrich_err: Option<anyhow::Error> = None;
        let mut msg_instructions: u64 = 0;
        let mut msg_vls: u64 = 0;

        let enricher = enrichers.entry(hart_id).or_insert_with(|| {
            log::info!(
                "worker {}: creating Enricher for hart_id={}",
                worker_index,
                hart_id
            );
            ctx.factory.build(hart_id)
        });

        ctx.data_processor
            .decode_into(decoder, &trace_data, |batch| {
                // `decode_into` cannot be aborted mid-stream, so once a batch
                // fails, skip the rest without consuming this window's once-only
                // markers; the error is returned just below.
                if enrich_err.is_some() {
                    return;
                }
                let side = window_side.take_once(&mut first_batch);

                // Instruction and VL counts come from the enricher's own
                // counters rather than the batch: on the BBT path there is no PC
                // array whose length could be read -- not materializing one is
                // the whole point of that format.
                let insns_before = enricher.instructions_committed();
                let vls_before = enricher.vl_values_delivered();
                let enriched = match batch {
                    processing::TraceBatch::Pcs { pcs, vls } => {
                        enricher.transform_optimized(pcs, vls, side)
                    }
                    processing::TraceBatch::Blocks(blocks) => {
                        enricher.transform_blocks(blocks, side)
                    }
                };
                msg_instructions += enricher.instructions_committed() - insns_before;
                msg_vls += enricher.vl_values_delivered() - vls_before;
                if let Err(e) = enriched {
                    enrich_err = Some(e);
                }
            })?;

        if let Some(e) = enrich_err {
            return Err(RerfError::trace_decoding_failed(format!(
                "enrichment failed for hart {}: {}",
                hart_id, e
            )));
        }

        out.insn_trace_bytes += trace_data.len() as u64;
        out.vl_bytes += msg_vls;
        out.instruction_count += msg_instructions;

        // Publish this worker's harts for the live UI, throttled. The merge
        // clones every function's RoiInfo and histograms, so doing it per job
        // would spend most of a live run rebuilding data nobody looked at.
        if ctx.emit_snapshots && last_snapshot.elapsed() >= ctx.snapshot_interval {
            last_snapshot = Instant::now();
            let pd = crate::workflow::merge_performance_data(enrichers.values_mut());
            if let Ok(mut slot) = ctx.slots[worker_index].lock() {
                *slot = Some(pd);
            }
        }
        out.busy += t0.elapsed();
    }

    out.enrichers = enrichers;
    Ok(out)
}

/// Assigns harts to workers, spawning one thread per hart up to the cap.
///
/// Assignment is by discovery order, not `hart_id % n`: a guest whose harts are
/// numbered sparsely (or that creates 3 harts on an 8-worker pool) would
/// otherwise pile several onto one worker while others sat idle.
pub struct HartRouter {
    max_workers: usize,
    assignment: HashMap<u32, usize>,
    /// Harts assigned to each worker so far, used to pick the emptiest.
    load: Vec<usize>,
}

impl HartRouter {
    pub fn new(max_workers: usize) -> Self {
        Self {
            max_workers: max_workers.max(1),
            assignment: HashMap::new(),
            load: Vec::new(),
        }
    }

    /// Worker index for `hart_id`, and whether that worker is new (the caller
    /// must spawn it).
    pub fn route(&mut self, hart_id: u32) -> (usize, bool) {
        if let Some(&idx) = self.assignment.get(&hart_id) {
            return (idx, false);
        }
        if self.load.len() < self.max_workers {
            let idx = self.load.len();
            self.load.push(1);
            self.assignment.insert(hart_id, idx);
            return (idx, true);
        }
        // Pool is full: give this hart to the worker with the fewest harts.
        let idx = self
            .load
            .iter()
            .enumerate()
            .min_by_key(|(_, n)| **n)
            .map(|(i, _)| i)
            .unwrap_or(0);
        self.load[idx] += 1;
        self.assignment.insert(hart_id, idx);
        log::info!(
            "hart {} shares decode worker {} (pool of {} is full)",
            hart_id,
            idx,
            self.max_workers
        );
        (idx, false)
    }

    pub fn worker_count(&self) -> usize {
        self.load.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn routes_each_new_hart_to_its_own_worker_until_the_cap() {
        let mut r = HartRouter::new(4);
        assert_eq!(r.route(0), (0, true), "first hart spawns worker 0");
        assert_eq!(r.route(1), (1, true));
        assert_eq!(r.route(2), (2, true));
        assert_eq!(r.route(3), (3, true));
        assert_eq!(r.worker_count(), 4);
    }

    #[test]
    fn repeat_harts_stay_on_their_worker() {
        let mut r = HartRouter::new(4);
        let (a, spawned) = r.route(7);
        assert!(spawned);
        let (b, spawned_again) = r.route(7);
        assert_eq!(a, b, "a hart must not migrate: decode state lives there");
        assert!(!spawned_again);
    }

    /// Beyond the cap, harts share workers -- and must share the *emptiest* one,
    /// so a sparse hart numbering cannot pile up on one thread.
    #[test]
    fn beyond_the_cap_harts_go_to_the_emptiest_worker() {
        let mut r = HartRouter::new(2);
        assert_eq!(r.route(10), (0, true));
        assert_eq!(r.route(20), (1, true));
        // Both workers hold 1 hart; the next goes to worker 0 (first minimum).
        assert_eq!(r.route(30), (0, false));
        // Worker 0 now holds 2, worker 1 holds 1, so the next goes to worker 1.
        assert_eq!(r.route(40), (1, false));
        assert_eq!(r.route(50), (0, false));
        assert_eq!(r.worker_count(), 2, "never exceeds the cap");
    }

    /// `hart_id % n` would send harts 0 and 8 to the same worker while leaving
    /// others idle; discovery-order assignment must not.
    #[test]
    fn sparse_hart_ids_still_spread_across_workers() {
        let mut r = HartRouter::new(8);
        let mut seen = std::collections::HashSet::new();
        for hart in [0u32, 8, 16, 24] {
            let (idx, _) = r.route(hart);
            assert!(seen.insert(idx), "hart {hart} collided on worker {idx}");
        }
    }

    #[test]
    fn a_zero_cap_is_clamped_to_one_worker() {
        let mut r = HartRouter::new(0);
        assert_eq!(r.route(0), (0, true));
        assert_eq!(r.route(1), (0, false));
        assert_eq!(r.worker_count(), 1);
    }
}
