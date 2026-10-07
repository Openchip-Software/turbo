use std::{
    path::{Path, PathBuf},
    sync::{Arc, Mutex},
    thread,
    time::Instant,
};

use anyhow::anyhow;
use schemars::schema_for;

use crate::error::RerfError;
use crate::{
    common::{self, PerformanceData},
    config::{ConfigFile, EnvConfig},
    processing::{self, enricher::SharedTrace, DataProcessor, ElfMetadata, Enricher, TraceFormat},
    report::print_summary,
    server, source,
    source::{QemuRunner, VLEN_BITS_MAX, VLEN_BITS_MIN},
};
use source::{TraceData, TraceSource};

/// How long to sleep when the source has no new data yet.
///
/// Only reached when the consumer has caught up with QEMU, so it costs nothing
/// when enrichment is the bottleneck (the common case) and keeps the poll from
/// busy-spinning a core when it is not.
const SOURCE_POLL_INTERVAL: std::time::Duration = std::time::Duration::from_millis(2);

/// Minimum wall-clock gap between live `PerformanceData` snapshots pushed to the
/// HTTP server. See the throttle at its use site for why this is not per-batch.
const LIVE_SNAPSHOT_INTERVAL: std::time::Duration = std::time::Duration::from_millis(100);

/// Minimum wall-clock gap between trace-routing throughput lines.
const ROUTE_LOG_INTERVAL: std::time::Duration = std::time::Duration::from_secs(2);

/// Aggregates per-message routing volume into a periodic throughput line.
///
/// Routing happens once per trace message (thousands per run, tens of thousands
/// on long runs), so logging each one buries every other line in the run. The
/// per-message facts that actually matter in aggregate are the byte volume, the
/// rate, and whether any message ids were skipped — a gap means dropped trace
/// data, which is a correctness signal worth surfacing on its own.
struct RoutedTally {
    /// Messages and bytes since the last emitted line.
    msgs: u64,
    bytes: u64,
    /// Messages and bytes over the whole run.
    total_msgs: u64,
    total_bytes: u64,
    /// Highest message id seen, to detect skipped ids.
    last_id: Option<u32>,
    gaps: u64,
    window_start: Instant,
}

impl RoutedTally {
    fn new() -> Self {
        Self {
            msgs: 0,
            bytes: 0,
            total_msgs: 0,
            total_bytes: 0,
            last_id: None,
            gaps: 0,
            window_start: Instant::now(),
        }
    }

    fn record(&mut self, message_id: u32, len: usize) {
        // Ids arrive in order per hart but interleave across harts, so only a
        // backwards-or-equal id is unambiguously wrong; count forward jumps as
        // gaps and let the operator correlate with the hart count.
        if let Some(prev) = self.last_id {
            if message_id > prev + 1 {
                self.gaps += u64::from(message_id - prev - 1);
            }
        }
        if self.last_id.is_none_or(|prev| message_id > prev) {
            self.last_id = Some(message_id);
        }
        self.msgs += 1;
        self.bytes += len as u64;
        self.total_msgs += 1;
        self.total_bytes += len as u64;
    }

    /// Emit a throughput line if the window has elapsed and anything was routed.
    fn log_if_due(&mut self) {
        let elapsed = self.window_start.elapsed();
        if self.msgs == 0 || elapsed < ROUTE_LOG_INTERVAL {
            return;
        }
        log::debug!(
            "routed {} trace messages ({:.1} MiB, {:.1} MiB/s), {} total",
            self.msgs,
            self.bytes as f64 / (1024.0 * 1024.0),
            self.bytes as f64 / (1024.0 * 1024.0) / elapsed.as_secs_f64(),
            self.total_msgs
        );
        self.msgs = 0;
        self.bytes = 0;
        self.window_start = Instant::now();
    }

    /// Emit the run total once routing is done.
    fn log_total(&self) {
        log::debug!(
            "routing done: {} trace messages, {:.1} MiB",
            self.total_msgs,
            self.total_bytes as f64 / (1024.0 * 1024.0)
        );
        if self.gaps > 0 {
            log::warn!(
                "{} trace message id(s) never arrived at the router: trace data was dropped",
                self.gaps
            );
        }
    }
}

/// Output file written by the hidden `--dump-pcs` cross-check mode.
pub const TRACE_PCS_FILE: &str = "turbo_trace_pcs.txt";

/// Write one hex PC per line. The format `--dump-pcs` produces, and what the
/// three-way PC comparison reads back.
fn write_pcs_to_text(pcs: &[u64], path: &std::path::Path) -> Result<(), RerfError> {
    use std::io::Write as _;
    let file = std::fs::File::create(path).map_err(|e| {
        RerfError::path_processing_failed(format!("creating {}: {e}", path.display()))
    })?;
    let mut out = std::io::BufWriter::new(file);
    for pc in pcs {
        writeln!(out, "0x{pc:x}").map_err(|e| {
            RerfError::path_processing_failed(format!("writing {}: {e}", path.display()))
        })?;
    }
    out.flush().map_err(|e| {
        RerfError::path_processing_failed(format!("flushing {}: {e}", path.display()))
    })?;
    Ok(())
}

/// Retired instructions on the synthetic `Global` ROI row, which is what a
/// flamegraph's root node is scaled by. Zero if the row is absent, which only
/// happens for a run that enriched nothing.
fn global_roi_instructions(pd: &PerformanceData) -> u64 {
    pd.rois
        .iter()
        .find(|r| &*r.name == "Global")
        .map(|r| r.instructions)
        .unwrap_or(0)
}

/// Write one generated report file, logging rather than failing: an unwritable
/// side file is not worth discarding a completed profiling run over.
fn save_report(path: std::path::PathBuf, contents: &str, what: &str) {
    match std::fs::write(&path, contents) {
        Ok(()) => log::info!("Saved {} to {:?}", what, path),
        Err(e) => log::warn!("Failed to write {} to {:?}: {}", what, path, e),
    }
}

/// Write one flamegraph per guest thread beside the merged one, plus the
/// `roi_flamegraph_harts.json` index a viewer builds its thread selector from.
///
/// Only for runs with more than one thread: with a single graph the per-thread
/// view is the merged view, and emitting it would make every single-threaded
/// run's output set differ from today's for no information gained.
///
/// No per-hart HTML. `to_html` inlines the widget's whole CSS and JS, so N
/// standalone pages means N copies of them; the interactive per-thread view is
/// the dashboard's, driven off this JSON.
fn write_per_hart_flamegraphs(
    wd: &std::path::Path,
    per_hart_flame: &std::collections::HashMap<u32, processing::roi_flamegraph::RoiFlameNode>,
) {
    if per_hart_flame.len() < 2 {
        return;
    }

    // Sorted, so the index and the log line are in hart order rather than
    // `HashMap` order -- otherwise the selector reshuffles between runs.
    let mut hart_ids: Vec<u32> = per_hart_flame.keys().copied().collect();
    hart_ids.sort_unstable();

    // nosemgrep: dos-unbounded-memory-allocation -- len of a resident Vec
    let mut harts = Vec::with_capacity(hart_ids.len());
    for hart_id in hart_ids {
        let root = &per_hart_flame[&hart_id];
        let json_name = processing::roi_flamegraph::hart_json_name(hart_id);

        save_report(
            // nosemgrep: input-validation-path-traversal-2 -- file name generated from a u32 hart id
            wd.join(processing::roi_flamegraph::hart_folded_name(hart_id)),
            &processing::roi_flamegraph::to_folded_stacks(root),
            &format!("hart {hart_id} ROI flamegraph folded stacks"),
        );
        match processing::roi_flamegraph::to_json(root) {
            Ok(json) => save_report(
                // nosemgrep: input-validation-path-traversal-2 -- file name generated from a u32 hart id
                wd.join(&json_name),
                &json,
                &format!("hart {hart_id} ROI flamegraph JSON"),
            ),
            // Skipped in the index too: an entry pointing at a file that was
            // never written would leave the selector with a dead tab.
            Err(e) => {
                log::warn!("Failed to serialize hart {hart_id} ROI flamegraph to JSON: {e}");
                continue;
            }
        }

        harts.push(processing::roi_flamegraph::HartFlamegraphEntry {
            hart_id,
            json: json_name,
            instructions: root.inclusive_instructions,
            self_instructions: root.self_instructions,
            call_paths: root.node_count(),
        });
    }

    let index = processing::roi_flamegraph::HartFlamegraphIndex { harts };
    match serde_json::to_string(&index) {
        Ok(json) => save_report(
            wd.join("roi_flamegraph_harts.json"),
            &json,
            "per-thread ROI flamegraph index",
        ),
        Err(e) => log::warn!("Failed to serialize the per-thread ROI flamegraph index: {e}"),
    }
}

/// Merge performance data from multiple per-hart Enrichers into one PerformanceData.
/// ROIs and functions with the same name are accumulated together.
pub(crate) fn merge_performance_data<'a>(
    enrichers: impl Iterator<Item = &'a mut Enricher>,
) -> PerformanceData {
    merge_perf_data(enrichers.map(|e| e.performance_data()))
}

/// Merge already-extracted `PerformanceData` values, accumulating ROIs and
/// functions that share a name.
///
/// Split out from [`merge_performance_data`] because the live snapshot path can
/// no longer reach every `Enricher`: they are distributed across decode workers,
/// so each worker publishes its own merged view and this combines those.
pub(crate) fn merge_perf_data(parts: impl Iterator<Item = PerformanceData>) -> PerformanceData {
    let mut merged = PerformanceData::default();
    // Name -> index maps so dedup across harts is O(1) per entry instead of the
    // previous O(n^2) linear `find` over the growing merged Vec. With ~259
    // functions and one hart that linear scan was ~33K string compares *per
    // snapshot*; the common single-hart case now just moves each ROI in once.
    let mut roi_idx: std::collections::HashMap<Arc<str>, usize> = std::collections::HashMap::new();
    let mut func_idx: std::collections::HashMap<Arc<str>, usize> = std::collections::HashMap::new();
    for pd in parts {
        merged.vlen_bits = merged.vlen_bits.or(pd.vlen_bits);
        // Merge ROIs (ROI regions)
        for roi in pd.rois {
            match roi_idx.get(&roi.name) {
                Some(&i) => merged.rois[i].merge_from(&roi),
                None => {
                    roi_idx.insert(roi.name.clone(), merged.rois.len());
                    merged.rois.push(roi);
                }
            }
        }
        // Merge functions
        for func in pd.functions {
            match func_idx.get(&func.name) {
                Some(&i) => merged.functions[i].merge_from(&func),
                None => {
                    func_idx.insert(func.name.clone(), merged.functions.len());
                    merged.functions.push(func);
                }
            }
        }
    }
    merged
}

/// Combine every decode worker's published view and push it to the HTTP server.
///
/// Slots a worker has not filled yet are simply absent, so the UI shows partial
/// results early rather than nothing until every hart has reported.
///
/// Returns `false` once nothing is listening any more, which is the caller's
/// signal to stop calling this: the merge is a deep clone of every hart's ROI
/// table, and paying for it per batch with no receiver on the other end is pure
/// waste on the decode critical path.
fn publish_live_snapshot(
    slots: &processing::decode_pool::SnapshotSlots,
    sender: &tokio::sync::watch::Sender<PerformanceData>,
) -> bool {
    // Checked BEFORE the merge, not just via the `send` result, so the wasted
    // clone never happens in the first place (`--no-server` drops the receiver
    // immediately).
    if sender.receiver_count() == 0 {
        return false;
    }
    let parts: Vec<PerformanceData> = slots
        .iter()
        .filter_map(|slot| slot.lock().ok().and_then(|g| g.clone()))
        .collect();
    if parts.is_empty() {
        return true;
    }
    if sender.send(merge_perf_data(parts.into_iter())).is_err() {
        log::debug!("live snapshot: no receivers left, stopping live publishes");
        return false;
    }
    true
}

/// Load the program image, reporting WHICH ELF could not be loaded and why.
///
/// Returned, not `expect`ed: every cause is user input reaching a library that
/// only knows it was handed a path. A file that is not an ELF, a truncated one,
/// or one whose sections cannot be read all arrive here, and used to abort the
/// process with `init must be ok` and a byte range for a message.
fn initialize_elf_metadata(metadata: &mut ElfMetadata, binary: &Path) -> Result<(), RerfError> {
    metadata.initialize().map_err(|e| {
        RerfError::metadata_init_failed(format!(
            "could not load the program image {}: {e:#}; `process` needs the same \
             ELF the trace was recorded from, and it must be readable and complete",
            binary.display()
        ))
    })
}

// The full set of workflow knobs, each independently set by the CLI layer.
#[allow(clippy::too_many_arguments)]
pub fn run_perf_workflow_opt(
    binary: PathBuf,
    mut source: impl TraceSource,
    workdir: Option<PathBuf>,
    disable_server: bool,
    annotate: bool,
    // Build the streaming ROI call-tree flamegraph and write
    // `roi_flamegraph.{folded,json,html}`. On by default; `--no-roi-flamegraph`
    // clears it for workloads whose ROI markers fire often enough that even a
    // bounded per-region cost is unwelcome.
    roi_flamegraph: bool,
    // `(VLEN in bytes, cache config)` when a modelled cache was requested
    // (`--cache-sim`); `None` leaves every enricher's model unbuilt.
    cache_sim: Option<(u64, processing::cache_model::Config)>,
    perfetto: bool,
    trace_flush_threshold: usize,
    trace_counter_interval: u64,
    summary: Option<String>,
    trace_format: TraceFormat,
    source_dirs: Vec<PathBuf>,
    // Cap on parallel decode+enrich worker threads; `None` uses
    // `processing::decode_pool::MAX_DECODE_THREADS`.
    decode_threads: Option<usize>,
    // Measurement harness (hidden `--decode-only`): run the decode path with a
    // consumer that does nothing, so `decode_into` is timed directly instead of
    // being inferred by subtracting an enrich-inclusive run. Produces no outputs.
    decode_only: bool,
    // Dev cross-check harness (hidden `--dump-pcs`): decode and write every PC
    // to `turbo_trace_pcs.txt` in the output dir, for comparison against an
    // independent trace of the same run. Like `--decode-only`, produces no
    // reports.
    dump_pcs: bool,
    // `--strict-trace`: refuse to report a run whose trace does not cover the
    // whole execution (a hart that never reached its `Done` sentinel, or one
    // excluded by `--vcpu`). Off by default: the surviving prefix of a killed
    // guest is real data, and the trace-integrity block says it is partial.
    strict_trace: bool,
    // The recorded logs whose byte breakdown is logged at debug level once they
    // are complete. Reported from in here rather than by the caller because on
    // the live path the caller only regains control after the server has been
    // joined -- i.e. after the user has closed the web UI, which can be minutes
    // later or never. `turbo inspect` prints the same report on demand.
    inspect: TraceInput,
) -> Result<(), RerfError> {
    if decode_only {
        log::warn!(
            "--decode-only: measurement mode. The trace is decoded but NOT enriched: \
             no perf_data_final.json, no annotate_*.html, no Perfetto trace, no ROIs. \
             Do not use this mode to produce results."
        );
    }

    // Wall-clock phase timing. The source polls the per-hart trace .bin spool
    // live (see LiveQemuSource), so QEMU execution overlaps decode + enrichment.
    // The measured spans are: `startup` = time until the memory map
    // arrives (plugin emits it on first vCPU init, so this is small); `run` = the
    // overlapped QEMU-produce + decode + inline-enrich critical path, of which
    // `enrich` is the part spent inside the enricher; `enrich-tail` = the
    // flush/annotate/merge finalization after the stream ends. Logging the split
    // makes the biggest wall-clock consumer obvious.
    let t_start = Instant::now();

    // Make workdir absolute to avoid permission issues when the server thread has
    // a different working directory. std::path::absolute() works even if the dir
    // doesn't exist yet.
    let workdir = workdir.map(|wd| std::path::absolute(&wd).unwrap_or(wd));

    // Messages pulled from the source, drained fully on each pass. The source is
    // polled inline (no thread, no channel), so this Vec is the only buffer
    // between it and decode+enrich.
    let mut pending: Vec<TraceData> = Vec::new();
    // Cleared once the source reports itself exhausted.
    let mut source_live = true;

    let perfdata_sender = tokio::sync::watch::Sender::new(PerformanceData::default());
    let perfdata_reciever = perfdata_sender.subscribe();

    // let roi_tx_this_thread = roi_tx.clone();

    // Shared store of rendered annotation pages. The enrichment thread fills it
    // once the report is ready; the server serves it under /annotate/{file}.
    let annotate_pages: server::AnnotatePages = std::default::Default::default();

    // Only start server thread if not disabled
    let server_jh = if !disable_server {
        let mut server = server::HttpServer::new(perfdata_reciever);
        server = server.annotate_pages(annotate_pages.clone());
        // Tell the server whether an annotation report is coming, so it shows
        // the auto-refreshing "pending" page only while an enabled report is
        // still rendering — not a misleading infinite wait when annotate is off.
        server = server.annotate_enabled(annotate);
        server = server.roi_flamegraph_enabled(roi_flamegraph);
        if let Some(ref wd) = workdir {
            // Serve the output dir under /out/ so the UI can open annotate_*.html.
            server = server.output_path(wd.clone());
        }
        Some(
            thread::Builder::new()
                .name("http-server".to_string())
                .spawn(move || {
                    if let Err(e) = server.run() {
                        log::error!("Server thread failed: {}", e);
                    }
                })
                .expect("Failed to spawn server thread"),
        )
    } else {
        log::debug!("server not started, as --no-server was passed.");
        None
    };

    log::debug!("waiting for memory map or shared/static libs message from the source");

    // Wait for MemoryMapReady to arrive, processing it immediately
    let mut metadata = ElfMetadata::new_uninit(&binary);
    let mut metadata_ready = false;

    // Process messages until the memory map (or the legacy metadata sentinel)
    // arrives. Any trace data that somehow precedes it is dropped, as before.
    while !metadata_ready {
        if pending.is_empty() {
            if !source_live {
                return Err(RerfError::channel_closed(
                    "Source ended before providing a memory map. This likely means that QEMU failed to run."
                ));
            }
            source_live = source.poll(&mut pending)?;
            if pending.is_empty() {
                if source_live {
                    thread::sleep(SOURCE_POLL_INTERVAL);
                }
                continue;
            }
        }

        // Consume from the FRONT one message at a time rather than draining the
        // whole Vec: the map may be followed by trace data in the same poll, and
        // a `drain(..)` that breaks early would throw the rest away. Only ever a
        // couple of iterations, since this loop stops at the first map.
        while !pending.is_empty() {
            let msg = pending.remove(0);
            match msg {
                TraceData::MemoryMapReady(memory_map) => {
                    log::info!(
                        "Received MemoryMapReady: main base 0x{:x}, {} object(s)",
                        memory_map.main_binary_base,
                        memory_map.objects.len(),
                    );

                    metadata.set_main_base_addr(memory_map.main_binary_base);
                    for obj in &memory_map.objects {
                        metadata.add_shared_lib(obj.base_address, &obj.path)?;
                    }
                    initialize_elf_metadata(&mut metadata, &binary)?;
                    metadata_ready = true;
                }
                TraceData::NoMemoryMap => {
                    log::debug!("No memory map for this trace; using the main binary alone");
                    initialize_elf_metadata(&mut metadata, &binary)?;
                    metadata_ready = true;
                }
                TraceData::Trace {
                    trace_data,
                    message_id,
                    ..
                } => {
                    if !trace_data.is_empty() {
                        // Sources always emit the map (or NoMemoryMap) first, so
                        // this cannot be decoded yet; drop it.
                        log::warn!(
                            "Received trace data before MemoryMap (message #{}), this should not happen in streaming mode",
                            message_id
                        );
                    }
                }
            }
            if metadata_ready {
                break;
            }
        }
    }

    if !metadata_ready {
        return Err(RerfError::metadata_init_failed(
            "Failed to initialize metadata",
        ));
    }

    // MemoryMapReady arrives early (plugin emits it on first vCPU init), so this
    // bounds only the startup latency before decode can begin.
    let t_startup = t_start.elapsed();
    log::info!(
        "Metadata initialized, starting streaming trace processing (startup: {:.3}s)",
        t_startup.as_secs_f64()
    );

    // Create shared trace state (single file, one process, one thread per hart)
    let shared_trace_arc: Option<Arc<Mutex<SharedTrace>>> = if perfetto {
        if let Some(ref wd) = workdir {
            let trace_path = wd.join("perfetto.trace");
            log::info!("Perfetto trace -> {:?}", trace_path);
            let mut st = SharedTrace::new();
            st.set_file_path(trace_path);
            st.set_flush_threshold(trace_flush_threshold);
            Some(Arc::new(Mutex::new(st)))
        } else {
            None
        }
    } else {
        None
    };

    // Builds a fresh Enricher per hart. A struct rather than a closure so the
    // decode workers can share one by reference across threads.
    // Decode tables and symbol resolvers: expensive to build, identical for
    // every hart, immutable once built — so build them here, once, and hand
    // every worker's enricher the same Arc. The E-Trace tracers read the same
    // tables (see `TableBinary`), so this is the run's only decode of the
    // program image.
    let modules = Arc::new(
        processing::enricher::Modules::from_metadata(&metadata).map_err(|e| {
            // Reachable from input, not just from a bug: the decode tables and
            // symbol resolvers are built out of whatever ELF the user named.
            RerfError::metadata_init_failed(format!(
                "could not build decode tables and symbol resolvers from {}: {e}",
                binary.display()
            ))
        })?,
    );

    let factory = processing::decode_pool::EnricherFactory {
        modules: modules.clone(),
        workdir: workdir.clone(),
        source_dirs: source_dirs.clone(),
        shared_trace: shared_trace_arc.clone(),
        annotate,
        cache_sim,
        roi_flamegraph,
        perfetto,
        trace_flush_threshold,
        trace_counter_interval,
    };

    let data_processor = DataProcessor::with_modules(modules, trace_format).map_err(|e| {
        RerfError::metadata_init_failed(format!(
            "could not build the {trace_format:?} trace decoder for {}: {e}",
            binary.display()
        ))
    })?;

    let enricher_perfdata_sender = perfdata_sender.clone();
    // Only the HTTP server consumes the live PerformanceData snapshots. With
    // `--no-server` nothing subscribes, so building one is pure waste; and even
    // with a server, they are throttled (see LIVE_SNAPSHOT_INTERVAL). The merge
    // clones every function's RoiInfo + histograms, so per-batch emission spent
    // most of a live run rebuilding data nobody looked at, while a browser cannot
    // show more than a few updates a second and the watch channel keeps only the
    // latest value.
    let emit_live_snapshots = !disable_server;
    let enricher_perfetto = perfetto;
    let enricher_annotate_pages = annotate_pages.clone();

    // Deferred finalization: flush, annotate report and final merge. Runs after
    // the decode loop drains, so it is charged to `enrich-tail`.
    let finalize_enrichers = |enrichers: &mut std::collections::HashMap<u32, Enricher>| {
        for (hart_id, enricher) in enrichers.iter_mut() {
            // Hard gate, not a warning: a leftover `vl` means every `vl` after
            // the surplus one was attributed to the wrong `vsetvl*`, so the
            // whole vector side of this run's numbers is wrong.
            //
            // Returned rather than panicked: the gate is an invariant of the
            // *encoder*, but a damaged or truncated trace violates it too, and
            // that is bad input rather than a turbo bug.
            enricher
                .check_vl_stream_exhausted()
                .map_err(|e| RerfError::trace_decoding_failed(format!("hart {hart_id}: {e:#}")))?;
            enricher.log_fast_path_stats();
        }

        if enricher_perfetto {
            // Push each hart's remaining packets into the shared pool.
            for (hart_id, enricher) in enrichers.iter_mut() {
                log::debug!("Flushing trace packets for hart_id={}", hart_id);
                if let Err(e) = enricher.flush_trace() {
                    log::error!("could not flush trace for hart {}: {:?}", hart_id, e);
                }
            }
            // The final write_to_disk() is called in run_perf_workflow_opt after join.
        }

        // Generate the static source & assembly annotation report (opt-in).
        // Merge every hart's per-PC profile, then render it via any one
        // enricher's ELF resolver (all harts share the same binary).
        if enrichers.values().any(|e| e.annotate_enabled()) {
            // Derive instruction_mix for function/global scopes from each
            // hart's still-populated PcProfile before take_pc_profile()
            // empties it -- this is the only place the mix is ever computed.
            for e in enrichers.values_mut() {
                e.apply_deferred_instruction_mix();
            }
            let mut merged = processing::PcProfile::new(true);
            for e in enrichers.values_mut() {
                merged.merge_from(&e.take_pc_profile());
            }
            if let Some(any) = enrichers.values().next() {
                // Render once: write gzipped .html.gz files to disk (shareable
                // artifacts) and publish the gzipped pages into the shared store
                // for the HTTP router (served with Content-Encoding: gzip).
                let pages = any.build_annotation_report(&merged);
                if let Ok(mut store) = enricher_annotate_pages.write() {
                    store.extend(pages);
                }
            }
        }

        // Snapshot each hart's own PerformanceData before the final merge, so
        // per-thread ROI/function stats remain available individually (e.g.
        // to spot per-thread imbalance) instead of only as a summed total.
        let mut per_hart_pd: std::collections::HashMap<u32, PerformanceData> =
            std::collections::HashMap::new();
        for (&hart_id, enricher) in enrichers.iter_mut() {
            per_hart_pd.insert(hart_id, enricher.performance_data());
        }

        // Return merged performance data across all harts, plus every recorded
        // ROI execution.
        let pd = merge_performance_data(enrichers.values_mut());
        let mut combined_flame_builder =
            processing::roi_flamegraph::StreamingRoiFlamegraph::default();
        // Finalise each hart's own call tree on the way past, before its
        // builder is folded into the combined one. `finish` takes `&self` and
        // the builder is already owned here, so keeping the per-thread tree
        // costs one call and no clone.
        //
        // Worth keeping because the merge re-keys by name, which erases
        // per-thread structure by construction: four threads' executions of
        // one region become a single node, so a 2x imbalance between them is
        // invisible in the merged graph.
        //
        // Rooted at *that hart's* retired count, not the run total -- passing
        // the run total would make every per-hart root's self-time the
        // difference between the run and one thread, which is meaningless.
        let mut per_hart_flame: std::collections::HashMap<
            u32,
            processing::roi_flamegraph::RoiFlameNode,
        > = std::collections::HashMap::new();
        for (&hart_id, e) in enrichers.iter_mut() {
            let builder = e.take_roi_flamegraph_builder();
            let hart_global = per_hart_pd
                .get(&hart_id)
                .map(global_roi_instructions)
                .unwrap_or(0);
            // Every hart that retired anything, whether or not it ever entered
            // a region. Not the merged graph's `has_executions` gate: a thread
            // that ran entirely outside every region is not a gap in the
            // per-thread picture, it *is* part of the picture. pthreads_roi's
            // main thread retires 98% of the run and enters nothing, and
            // dropping it would leave the per-thread view accounting for 2% of
            // the program. Its graph is a bare root, and that root's self-time
            // is the number worth seeing.
            //
            // Only a hart that retired nothing at all is skipped, since a
            // graph scaled to zero instructions has nothing to draw.
            if hart_global > 0 {
                per_hart_flame.insert(hart_id, builder.finish(hart_global));
            }
            combined_flame_builder.merge(builder);
        }
        Ok((pd, combined_flame_builder, per_hart_pd, per_hart_flame))
    };

    // ===================== Parallel decode + enrich =====================
    //
    // The source is polled on THIS thread and each hart's flush windows are
    // routed to a worker that owns that hart's decoder and enricher. Reading and
    // framing a record log is cheap (a file read plus a postcard header walk);
    // decode+enrich is ~94% of the run, and that is what runs in parallel.
    //
    // Per-hart ordering is preserved by construction: one hart, one channel, one
    // worker, processed in order. A hart's trace is one continuous qualification
    // stream, so this is a correctness requirement, not a tuning choice.
    let worker_cap = decode_threads
        .unwrap_or(processing::decode_pool::MAX_DECODE_THREADS)
        .clamp(1, processing::decode_pool::MAX_DECODE_THREADS);

    let snapshot_slots: processing::decode_pool::SnapshotSlots =
        (0..worker_cap).map(|_| Mutex::new(None)).collect();

    let mut insn_trace_bytes: u64 = 0;
    let mut vl_bytes: u64 = 0;
    let mut instruction_count: u64 = 0;
    // Summed across workers this is decode+enrich CPU time, which exceeds the
    // wall clock once more than one worker is busy. `t_run` remains the wall
    // clock, so the ratio of the two is the achieved decode parallelism.
    let mut t_enrich_cpu = std::time::Duration::ZERO;
    let mut enrichers: std::collections::HashMap<u32, Enricher> = std::collections::HashMap::new();
    let mut worker_count = 0usize;
    // Only ever non-empty under `--dump-pcs`.
    let mut dumped_pcs: Vec<u64> = Vec::new();

    let ctx = processing::decode_pool::WorkerContext {
        data_processor: &data_processor,
        factory: &factory,
        slots: &snapshot_slots,
        emit_snapshots: emit_live_snapshots,
        snapshot_interval: LIVE_SNAPSHOT_INTERVAL,
        decode_only,
        dump_pcs,
    };

    // `&ctx` rather than `ctx`: every worker shares the same read-only context,
    // and a shared reference is `Copy` so each spawn can take its own.
    let ctx = &ctx;

    // `thread::scope` so workers can borrow `data_processor`, `factory` and the
    // snapshot slots directly instead of everything being wrapped in an Arc.
    let pool_result: Result<(), RerfError> = thread::scope(|scope| {
        let mut router = processing::decode_pool::HartRouter::new(worker_cap);
        let mut senders: Vec<
            Option<std::sync::mpsc::SyncSender<processing::decode_pool::DecodeJob>>,
        > = Vec::new();
        let mut handles: Vec<
            thread::ScopedJoinHandle<'_, Result<processing::decode_pool::WorkerOutput, RerfError>>,
        > = Vec::new();
        let mut last_publish = Instant::now();
        // Cleared the first time a publish finds no receiver (the server thread
        // exited, or there never was one), so the merge is not re-attempted on
        // every subsequent window.
        let mut publish_live = emit_live_snapshots;
        // Routing is per-message on the hot path (thousands of messages), so the
        // volume is reported as a periodic rate rather than a line per message.
        let mut routed = RoutedTally::new();
        // Set when a worker has died: its channel is gone, so the real error is
        // waiting at the join rather than in the `send` failure.
        let mut worker_died = false;

        'outer: loop {
            if pending.is_empty() {
                if !source_live {
                    break;
                }
                source_live = source.poll(&mut pending)?;
                if pending.is_empty() {
                    if source_live {
                        thread::sleep(SOURCE_POLL_INTERVAL);
                    }
                    // Refresh the live view while idling, so a slow guest still
                    // shows progress between flush windows.
                    if publish_live && last_publish.elapsed() >= LIVE_SNAPSHOT_INTERVAL {
                        last_publish = Instant::now();
                        publish_live =
                            publish_live_snapshot(&snapshot_slots, &enricher_perfdata_sender);
                    }
                    continue;
                }
            }

            for data in pending.drain(..) {
                match data {
                    TraceData::Trace {
                        trace_data,
                        message_id,
                        roi_markers,
                        hart_id,
                    } => {
                        if trace_data.is_empty() {
                            continue;
                        }
                        routed.record(message_id, trace_data.len());

                        let (idx, spawn_needed) = router.route(hart_id);
                        if spawn_needed {
                            let (tx, rx) = processing::decode_pool::job_channel();
                            debug_assert_eq!(idx, senders.len());
                            senders.push(Some(tx));
                            handles.push(
                                thread::Builder::new()
                                    .name(format!("decode-{idx}"))
                                    .spawn_scoped(scope, move || {
                                        processing::decode_pool::run_worker(idx, rx, ctx)
                                    })
                                    .expect("failed to spawn decode worker"),
                            );
                            log::info!("spawned decode worker {} for hart {}", idx, hart_id);
                        }

                        let job = processing::decode_pool::DecodeJob {
                            hart_id,
                            trace_data,
                            roi_markers,
                        };
                        // Blocks when the worker is behind: that is the intended
                        // backpressure, and it cannot stall the guest (QEMU
                        // writes the spool to disk independently of us).
                        if let Some(tx) = senders[idx].as_ref() {
                            if tx.send(job).is_err() {
                                // Worker gone => it returned an error. Stop
                                // feeding and let the join surface the cause.
                                worker_died = true;
                                break 'outer;
                            }
                        }
                    }
                    TraceData::MemoryMapReady(_) | TraceData::NoMemoryMap => {
                        log::debug!("Ignoring duplicate memory-map message");
                    }
                }
            }

            if publish_live && last_publish.elapsed() >= LIVE_SNAPSHOT_INTERVAL {
                last_publish = Instant::now();
                publish_live = publish_live_snapshot(&snapshot_slots, &enricher_perfdata_sender);
            }

            routed.log_if_due();
        }

        routed.log_total();

        // Close every channel so the workers finish their queues and return.
        for s in senders.iter_mut() {
            *s = None;
        }

        worker_count = handles.len();
        let mut first_err: Option<RerfError> = None;
        for (idx, h) in handles.into_iter().enumerate() {
            match h.join() {
                Ok(Ok(out)) => {
                    insn_trace_bytes += out.insn_trace_bytes;
                    vl_bytes += out.vl_bytes;
                    instruction_count += out.instruction_count;
                    t_enrich_cpu += out.busy;
                    // Hart ids are disjoint across workers (one worker owns a
                    // hart for the whole run), so this cannot overwrite.
                    enrichers.extend(out.enrichers);
                    dumped_pcs.extend_from_slice(&out.pcs);
                }
                Ok(Err(e)) => {
                    log::error!("decode worker {} failed: {}", idx, e);
                    if first_err.is_none() {
                        first_err = Some(e);
                    }
                }
                Err(_) => {
                    log::error!("decode worker {} panicked", idx);
                    if first_err.is_none() {
                        first_err = Some(RerfError::channel_closed(format!(
                            "decode worker {idx} panicked"
                        )));
                    }
                }
            }
        }
        match first_err {
            Some(e) => Err(e),
            None => {
                if worker_died {
                    // A send failed but every worker returned Ok: the only way
                    // that happens is a worker exiting early without an error,
                    // which would silently drop trace data.
                    return Err(RerfError::channel_closed(
                        "a decode worker stopped accepting work without reporting an error",
                    ));
                }
                Ok(())
            }
        }
    });
    pool_result?;

    // A run that decoded nothing has no profile to report, and the empty
    // `perf_data_final.json` that used to be written for one is the most
    // misleading artifact turbo can produce: zero instructions, zero ROIs,
    // exit 0 -- identical to a successful profile of a program that did
    // nothing, which is not a program anyone traces. Every way to get here is
    // a broken input (a log holding no decodable record, a `--vcpu` set
    // pointing at an empty hart, a guest that never reached the tracer), so
    // say so instead of writing the file.
    if instruction_count == 0 {
        return Err(RerfError::trace_decoding_failed(
            "no instructions were decoded from this trace, so there is nothing to \
             report; the record log(s) hold no decodable trace data -- check that the \
             recording completed and that the requested vCPU actually ran"
                .to_string(),
        ));
    }

    log::info!(
        "decode pool: {} worker(s) over {} hart(s), {:.3}s CPU in decode+enrich",
        worker_count,
        enrichers.len(),
        t_enrich_cpu.as_secs_f64()
    );

    log::debug!("trace decoding completed, finalizing enrichers now.");

    // The overlapped QEMU-produce + spool-drain + decode + enrich critical path
    // (this thread runs concurrently with QEMU).
    let t_run = t_start.elapsed() - t_startup;

    // Measurement harness: report and stop before finalization. Nothing downstream
    // of the decode loop has anything to work with (no enricher was ever built),
    // and excluding the tail keeps the printed figure decode-only. Printed to
    // stdout, not logged, so it is greppable without RUST_LOG being set.
    if dump_pcs {
        // Sorted: with several harts the streams interleave nondeterministically,
        // so only the multiset of executed PCs is well defined -- which is also
        // all a cross-check against an independent trace can compare.
        dumped_pcs.sort_unstable();
        let path = workdir
            .clone()
            .unwrap_or_else(|| PathBuf::from("."))
            .join(TRACE_PCS_FILE);
        write_pcs_to_text(&dumped_pcs, &path)?;
        log::info!(
            "--dump-pcs: wrote {} PCs to {}",
            dumped_pcs.len(),
            path.display()
        );
    }

    if decode_only || dump_pcs {
        println!(
            "decode-only: {} instructions in {:.3}s",
            instruction_count,
            t_run.as_secs_f64()
        );

        return Ok(());
    }

    // Collected from the drained source: which harts were read, how far, and
    // whether each reached its `Done` sentinel. Recorded in the report so a
    // partially recovered run is distinguishable from a whole one by a caller
    // reading the JSON, not only by a human reading the log.
    let trace_integrity = source.integrity();
    let trace_complete = trace_integrity.is_complete();

    let t_enrich_join = Instant::now();
    let (mut final_pd, combined_flame_builder, per_hart_pd, per_hart_flame) =
        finalize_enrichers(&mut enrichers)?;
    let t_enrich_tail = t_enrich_join.elapsed();
    final_pd.add_trace_metadata(
        insn_trace_bytes,
        vl_bytes,
        instruction_count,
        trace_integrity,
    );

    // `--strict-trace`: a cut-short recording is reported by default, because
    // the surviving prefix is real data and a killed guest is a normal reason
    // to have one. Under this flag it is an error instead, for a caller (CI,
    // validation) that must only ever accept whole runs.
    if strict_trace && !trace_complete {
        return Err(RerfError::trace_decoding_failed(
            "--strict-trace: this trace does not cover the whole run (see the trace \
             integrity warning above); refusing to report it"
                .to_string(),
        ));
    }

    if combined_flame_builder.has_executions() {
        log::info!(
            "QEMU enricher recorded {} ROI execution(s) over {} call-tree node(s)",
            combined_flame_builder.executions(),
            combined_flame_builder.node_count()
        );
    }

    // Final flush: drain any remaining packets from all harts into the single trace file.
    if let Some(ref arc) = shared_trace_arc {
        match arc.lock().unwrap().write_to_disk() {
            Ok(()) => log::info!("Wrote Perfetto trace"),
            Err(e) => log::error!("Failed to write Perfetto trace: {:?}", e),
        }
    }

    perfdata_sender
        .send(final_pd.clone())
        .expect("sending Performancedata (after trace metadata addition) must succeed");

    // drop the roi tx for consumers (eg server etc) to stop
    drop(perfdata_sender);

    // TODO: push the metrics here into the PerformanceData struct.
    if let Some(wd) = workdir {
        let new = wd.join("perf_data_final.json");
        final_pd.save(new);

        // Also persist each hart's own PerformanceData separately, so
        // per-thread ROI/function stats can be inspected in isolation later
        // (e.g. to detect per-thread work imbalance) instead of only as a
        // summed total.
        for (hart_id, hart_pd) in per_hart_pd.iter() {
            // nosemgrep: input-validation-path-traversal-2 -- file name generated from a u32 hart id
            let hart_path = wd.join(format!("turbo_vcpu_{}_perf.json", hart_id));
            hart_pd.save(hart_path);
        }

        // Generate and save JSON schema for PerformanceData
        let schema = schema_for!(common::PerformanceData);
        let schema_path = wd.join("perf_data_schema.json");
        if let Ok(schema_json) = serde_json::to_string_pretty(&schema) {
            if let Err(e) = std::fs::write(&schema_path, schema_json) {
                log::warn!("Failed to write schema to {:?}: {}", schema_path, e);
            } else {
                log::info!("Saved JSON schema to {:?}", schema_path);
            }
        } else {
            log::warn!("Failed to serialize schema to JSON");
        }

        // Gated on completed executions rather than on node count: a region
        // that is entered and never closed creates a node but produces no
        // measurements, and a flamegraph of nothing but the Global root is not
        // worth writing.
        if combined_flame_builder.has_executions() {
            let root = combined_flame_builder.finish(global_roi_instructions(&final_pd));

            save_report(
                wd.join("roi_flamegraph.folded"),
                &processing::roi_flamegraph::to_folded_stacks(&root),
                "ROI flamegraph folded stacks",
            );
            match processing::roi_flamegraph::to_json(&root) {
                Ok(json) => {
                    save_report(wd.join("roi_flamegraph.json"), &json, "ROI flamegraph JSON")
                }
                Err(e) => log::warn!("Failed to serialize ROI flamegraph to JSON: {}", e),
            }
            match processing::roi_flamegraph::to_html(&root) {
                Ok(html) => {
                    save_report(wd.join("roi_flamegraph.html"), &html, "ROI flamegraph HTML")
                }
                Err(e) => log::warn!("Failed to render ROI flamegraph HTML: {}", e),
            }

            write_per_hart_flamegraphs(&wd, &per_hart_flame);
        }
    }

    if let Some(ref s) = summary {
        print_summary(&final_pd, s);
    }

    // Before the server join below: that call blocks until the user closes the
    // web UI, so anything after it is output the user cannot see while the run
    // is still interesting.
    // Gated on debug logging actually being on: the report re-reads and
    // inflates every record log, which is real work to do for output nobody
    // will see. The gate must name the target the report itself logs under,
    // not this module, or `RUST_LOG=turbo::inspect=debug` skips the work it
    // just asked for.
    if log::log_enabled!(target: "turbo::inspect", log::Level::Trace) {
        if let Err(e) = crate::inspect::report_trace_input(&inspect) {
            log::warn!("{e}");
        }
    }

    if let Some(server_jh) = server_jh {
        log::info!(
            "processing done, waiting for server at http://127.0.0.1:{} to close",
            server::DEFAULT_PORT
        );
        join_or_panic(server_jh, "Server")?;
    } else {
        log::info!("Server was disabled, skipping server join");
    }

    // Phase breakdown of the user-visible wall clock. With live spool tailing,
    // QEMU overlaps decode+enrichment, so `run` is the overlapped critical path
    // (bounded by whichever of QEMU-produce vs decode+enrich-consume is slower)
    // and `enrich-tail` is the enrichment/annotate work left after decode stops.
    //
    // NOTE: `run` is NOT "QEMU time", and `enrich-cpu` is NOT wall clock. The
    // former is the overlapped critical path; the latter is decode+enrich CPU
    // time SUMMED over the worker threads, so with N busy workers it can be up
    // to N times `run`. Their ratio is the decode parallelism actually achieved:
    // enrich-cpu/run ~= 1 means the pool never got going (one hart, or the reader
    // is the limit), ~= N means N workers stayed busy.
    let t_total = t_start.elapsed();
    log::info!(
        "[phase-timing] total={:.3}s | startup={:.3}s ({:.0}%) | run={:.3}s ({:.0}%) \
         [enrich-cpu={:.3}s = {:.2}x run] | enrich-tail={:.3}s ({:.0}%)",
        t_total.as_secs_f64(),
        t_startup.as_secs_f64(),
        100.0 * t_startup.as_secs_f64() / t_total.as_secs_f64(),
        t_run.as_secs_f64(),
        100.0 * t_run.as_secs_f64() / t_total.as_secs_f64(),
        t_enrich_cpu.as_secs_f64(),
        t_enrich_cpu.as_secs_f64() / t_run.as_secs_f64().max(1e-9),
        t_enrich_tail.as_secs_f64(),
        100.0 * t_enrich_tail.as_secs_f64() / t_total.as_secs_f64(),
    );

    log::info!("all done!");
    Ok(())
}

fn join_or_panic(handle: thread::JoinHandle<()>, thread_name: &str) -> Result<(), RerfError> {
    match handle.join() {
        Ok(_) => {
            log::info!("{thread_name} thread completed successfully");
            Ok(())
        }
        Err(e) => {
            log::error!("{thread_name} thread panicked!");

            // Try to extract panic information
            let msg = if let Some(s) = e.downcast_ref::<&str>() {
                log::error!("Panic message (str): {}", s);
                format!("{thread_name} thread panicked: {}", s)
            } else if let Some(s) = e.downcast_ref::<String>() {
                log::error!("Panic message (String): {}", s);
                format!("{thread_name} thread panicked: {}", s)
            } else {
                log::error!("Panic message: <unable to extract details>");
                log::error!("Panic type: {:?}", std::any::type_name_of_val(&e));
                format!("{thread_name} thread panicked with unrecognized error type")
            };
            Err(RerfError::channel_closed(msg))
        }
    }
}

/// Options for [`run_with_live_qemu`].
#[derive(Default)]
pub struct RunWithQemuOptions {
    pub bin_work_dir: Option<PathBuf>,
    pub no_server: bool,
    pub env_config: Option<EnvConfig>,
    pub bin_args: Vec<String>,
    pub roi_flamegraph: bool,
    pub annotate: bool,
    /// Model an L1/L2 cache over the captured vector memory addresses
    /// (`--cache-sim`). `None` -- the default -- models nothing.
    pub cache_sim: Option<processing::cache_model::Config>,
    pub perfetto: bool,
    pub vlen_bits: Option<u32>,
    pub trace_flush_threshold: usize,
    pub trace_counter_interval: u64,
    pub summary: Option<String>,
    pub source_dirs: Vec<PathBuf>,
    /// Restrict processing to these harts (`--vcpu`); `None` means every hart
    /// the guest creates.
    pub hart_filter: Option<Vec<u32>>,
    /// Cap on parallel decode+enrich threads (`--decode-threads`).
    pub decode_threads: Option<usize>,
    /// Also record the independent execlog/Paraver reference traces
    /// (`--reference-traces`).
    pub reference_traces: bool,
    /// Environment variables to set for the *guest* program (via QEMU's `-E`),
    /// e.g. `LD_LIBRARY_PATH` so a dynamically-linked guest finds its `.so`.
    pub guest_env: Vec<(String, String)>,
    /// `--strict-trace`: fail instead of reporting a trace that does not cover
    /// the whole run.
    pub strict_trace: bool,
}

/// Build the `QemuRunner` shared by the live-decode and record-only paths.
///
/// With `reference_traces` the execlog plugin is enabled alongside the trace
/// one, so the same guest execution also produces `execlog_out.txt` and
/// `qemutrace.prv`: two independently produced records of the executed PCs,
/// which is what makes a decode cross-check possible (see `--dump-pcs`). Off by
/// default because execlog writes one text line per instruction.
fn build_qemu_runner(
    output_dir: &std::path::Path,
    bin_work_dir: Option<PathBuf>,
    env_config: Option<EnvConfig>,
    reference_traces: bool,
    vlen_bits: Option<u32>,
    guest_env: Vec<(String, String)>,
) -> QemuRunner {
    let mut qemu_runner = if let Some(config) = env_config {
        QemuRunner::from(config)
    } else {
        let runner = QemuRunner::default();
        log_builtin_env_defaults(&runner);
        runner
    }
    .tracer(true)
    .execlog(reference_traces)
    .with_work_dir(output_dir.to_path_buf());

    if let Some(bin_w_d) = bin_work_dir {
        qemu_runner = qemu_runner.with_bin_work_dir(bin_w_d);
    }
    for (name, value) in guest_env {
        qemu_runner = qemu_runner.with_guest_env(name, value);
    }
    // Already resolved by the CLI layer as `--vlen` -> `cpu_config.json`'s
    // `vlen` -> `None`; `None` leaves the runner on `SYSTEM_VLEN`'s default.
    if let Some(vlen_bits) = vlen_bits {
        log::debug!("vlen: configured {vlen_bits} bits overrides the runner's default vlen");
        qemu_runner = qemu_runner.vlen(vlen_bits);
    }
    log::debug!("vlen: QEMU cpu vlen = {} bits", qemu_runner.vlen_bits());
    qemu_runner
}

/// Say which QEMU, tracer plugin and sysroot a run with no
/// `environment_config.json` falls back to, and whether each one is there.
///
/// Up front and in one place, because without it the first sign of a wrong
/// default is QEMU failing to start, or a plugin it cannot load, with nothing
/// connecting either to a config file that was never read.
fn log_builtin_env_defaults(runner: &QemuRunner) {
    let qemu = &runner.path_config().qemu_binary;
    let qemu_status = match crate::source::resolve_qemu_binary(qemu) {
        Ok(found) if found != *qemu => format!("found at {}", found.display()),
        Ok(_) => "found".to_string(),
        Err(_) => "NOT FOUND".to_string(),
    };
    let presence = |path: &std::path::Path| if path.exists() { "found" } else { "NOT FOUND" };
    let plugin = runner.plugin_config().tracer_plugin_path();
    let loader = &runner.path_config().loader_path;
    log::warn!(
        "No {} loaded (looked in {}); using built-in defaults:\n  \
         qemu_bin            = {} ({qemu_status})\n  \
         turbo_tracer_plugin = {} ({})\n  \
         loader_path         = {} ({})\n\
         Run `turbo config init` to write a config, and `turbo config check` to verify it.",
        EnvConfig::FILE_NAME,
        EnvConfig::searched_locations(),
        qemu.display(),
        plugin.display(),
        presence(&plugin),
        loader.display(),
        presence(loader),
    );
}

pub fn run_with_live_qemu(
    binary: PathBuf,
    output_dir: PathBuf,
    opts: RunWithQemuOptions,
) -> anyhow::Result<()> {
    let RunWithQemuOptions {
        bin_work_dir,
        no_server,
        env_config,
        bin_args,
        roi_flamegraph,
        annotate,
        cache_sim,
        perfetto,
        trace_flush_threshold,
        trace_counter_interval,
        summary,
        source_dirs,
        hart_filter,
        decode_threads,
        reference_traces,
        vlen_bits,
        guest_env,
        strict_trace,
    } = opts;

    // The spool `run` is about to write, named up front: the filter is moved
    // into the source below, and the report must cover exactly the harts that
    // were processed.
    let inspect_input = TraceInput::RecordedDir {
        dir: output_dir.clone(),
        harts: hart_filter.clone(),
    };

    log::info!(
        "Starting QEMU argument build with binary: {:?}, config = {env_config:?}",
        binary
    );

    let qemu_runner = build_qemu_runner(
        &output_dir,
        bin_work_dir,
        env_config.clone(),
        reference_traces,
        vlen_bits,
        guest_env,
    );

    let binary_str = binary.to_string_lossy().into_owned();

    // Spawns QEMU and returns immediately; the workflow polls it for spool data,
    // so decode+enrich overlaps guest execution.
    let source = source::live_qemu(&qemu_runner, &binary_str, &bin_args, hart_filter)
        .map_err(|err| anyhow::anyhow!("Failed to start live QEMU trace: {}", err))?;

    run_perf_workflow_opt(
        binary,
        source,
        Some(output_dir.clone()),
        no_server,
        annotate,
        roi_flamegraph,
        // VLEN comes from the runner rather than the recorded metadata: on the
        // live path the metadata file is still being written.
        cache_sim.map(|cfg| (u64::from(qemu_runner.vlen_bits()) / 8, cfg)),
        perfetto,
        trace_flush_threshold,
        trace_counter_interval,
        summary,
        // The plugin only ever emits BBT now.
        TraceFormat::Bbt,
        source_dirs,
        decode_threads,
        // `--decode-only` / `--dump-pcs` are `process`-side harnesses; the live
        // path always enriches.
        false,
        false,
        strict_trace,
        inspect_input,
    )
    .map_err(|e| anyhow::anyhow!("TURBO run failed: {}", e))
}

/// Options for [`record_with_qemu`].
#[derive(Default)]
pub struct RecordWithQemuOptions {
    pub bin_work_dir: Option<PathBuf>,
    pub env_config: Option<EnvConfig>,
    pub bin_args: Vec<String>,
    /// Also record the independent execlog/Paraver reference traces
    /// (`--reference-traces`).
    pub reference_traces: bool,
    /// Vector register length in bits: `--vlen`, else `cpu_config.json`'s
    /// `vlen`, else `None` for `SYSTEM_VLEN`'s default.
    pub vlen_bits: Option<u32>,
    /// Environment variables to set for the *guest* program (via QEMU's `-E`),
    /// e.g. `LD_LIBRARY_PATH` so a dynamically-linked guest finds its `.so`.
    pub guest_env: Vec<(String, String)>,
}

/// Run a binary under QEMU with the tracing plugin enabled and wait for it to
/// exit. The plugin writes the per-hart trace `.bin` files, `turbo_metadata.json`
/// and the VDSO dump straight to `output_dir` itself, independent of any
/// Rust-side consumer — so recording needs no decode/enrich step at all, just
/// "spawn QEMU, wait for it to finish".
pub fn record_with_qemu(
    binary: PathBuf,
    output_dir: PathBuf,
    opts: RecordWithQemuOptions,
) -> anyhow::Result<()> {
    let RecordWithQemuOptions {
        bin_work_dir,
        env_config,
        bin_args,
        reference_traces,
        vlen_bits,
        guest_env,
    } = opts;

    log::info!(
        "Recording {:?} under QEMU into {}, config = {env_config:?}",
        binary,
        output_dir.display()
    );

    let qemu_runner = build_qemu_runner(
        &output_dir,
        bin_work_dir,
        env_config.clone(),
        reference_traces,
        vlen_bits,
        guest_env,
    );
    let binary_str = binary.to_string_lossy().into_owned();
    let bin_args_refs: Vec<&str> = bin_args.iter().map(|s| s.as_str()).collect();

    let result = qemu_runner
        .run(&binary_str, &bin_args_refs, None)
        .map_err(|err| anyhow::anyhow!("QEMU run failed: {}", err))?;

    // A failing guest is still a recording (see `ld_library_path_missing_fails`),
    // so this is not an error; but QEMU refusing to start at all looks the same
    // from here, so say why rather than exiting as if nothing happened.
    if !result.success() {
        log::warn!(
            "{}",
            crate::source::qemu::qemu_failure_message(&result.output.status, &output_dir)
        );
    }

    Ok(())
}

pub fn run_from_input_rois(input_dir: PathBuf) -> anyhow::Result<()> {
    log::debug!("run from input rois: input dir is {}", input_dir.display());
    let input_path = input_dir.join("perf_data_final.json");

    // Check if file exists
    if !input_path.exists() {
        log::error!(
            "no performance data found, file does not exist {}",
            input_path.display()
        );
        return Err(anyhow!("{} not found", input_path.display()));
    }
    log::info!("Loading ROI data from: {:?}", input_path);
    let pd = PerformanceData::load(input_path)?;

    let perfdata_sender = tokio::sync::watch::Sender::new(PerformanceData::default());
    let perfdata_reciever = perfdata_sender.subscribe();
    let mut server = server::HttpServer::new(perfdata_reciever);

    // A prior `run`/`process --annotate` writes the source & assembly report as
    // gzipped `annotate_*.html.gz` files into the input dir's `html/`
    // subdirectory. There is no enrichment thread in load mode to publish them,
    // so read them off disk and populate the in-memory store here; otherwise
    // `/annotate/{file}` serves the "pending" placeholder even though the pages
    // exist.
    let annotate_pages = load_annotate_pages(&input_dir.join("html"));
    if annotate_pages.is_empty() {
        log::info!(
            "no annotate_*.html files found in {}; source/asm report unavailable in load mode",
            input_dir.join("html").display()
        );
    } else {
        log::info!("loaded {} annotate page(s) from disk", annotate_pages.len());
    }
    // If no report was captured for this dir, mark annotate disabled so the
    // server serves a clear "not available" page rather than the pending one.
    let had_annotate = !annotate_pages.is_empty();
    server = server.annotate_pages(Arc::new(std::sync::RwLock::new(annotate_pages)));
    server = server.annotate_enabled(had_annotate);
    // A prior `run --roi-flamegraph` writes roi_flamegraph.html straight to the
    // output dir; if it's there, serve it, otherwise show "not available".
    server = server.roi_flamegraph_enabled(input_dir.join("roi_flamegraph.html").exists());
    // Serve the input dir under /out/ so the UI can open generated artifacts.
    server = server.output_path(input_dir.clone());

    perfdata_sender
        .send(pd)
        .expect("sending PerformanceData to server must work");
    drop(perfdata_sender);

    server.run()?;
    Ok(())
}

/// Read the gzipped `annotate_*.html.gz` report pages (plus the shared
/// `annotate.css.gz` / `annotate.js.gz` assets) from a previously-generated
/// `html/` output subdirectory. The map is keyed by the *logical* filename
/// with the `.gz` suffix stripped (e.g. `annotate_index.html`), matching the
/// names the browser requests and how the enrichment thread keys the live
/// store. Values are the raw gzip bytes, served as-is with
/// `Content-Encoding: gzip`.
fn load_annotate_pages(html_dir: &std::path::Path) -> std::collections::HashMap<String, Vec<u8>> {
    let mut pages = std::collections::HashMap::new();
    let Ok(entries) = std::fs::read_dir(html_dir) else {
        return pages;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        let Some(name) = path.file_name().and_then(|n| n.to_str()) else {
            continue;
        };
        // Match our gzipped artifacts: annotate_*.html.gz, annotate.css.gz,
        // annotate.js.gz. The store key is the name without the `.gz` suffix.
        if name.starts_with("annotate") && name.ends_with(".gz") {
            match std::fs::read(&path) {
                Ok(gz) => {
                    let key = name.trim_end_matches(".gz").to_string();
                    pages.insert(key, gz);
                }
                Err(e) => log::warn!("failed to read annotate page {}: {}", path.display(), e),
            }
        }
    }
    pages
}

/// What `process` was pointed at.
///
/// The two shapes are genuinely different inputs, not two spellings of one: a
/// recorded spool is a *set* of framed per-hart logs, while a raw stream is one
/// unframed blob with no hart identity. Resolving this once, up front, is what
/// keeps `process` from silently analysing hart 0 of a multi-hart recording.
#[derive(Clone, Debug)]
pub enum TraceInput {
    /// A directory of `turbo_trace_vcpu<N>.bin` logs: every hart, or those in `harts`.
    RecordedDir {
        dir: PathBuf,
        harts: Option<Vec<u32>>,
    },
    /// One explicitly-named record log. Siblings are reported, not read.
    RecordedLog { path: PathBuf, hart_id: u32 },
}

/// Every `turbo_trace_vcpu<N>.bin` record log in `dir`, hart-ordered, keeping
/// only the harts in `harts` when a `--vcpu` set was requested.
///
/// One scan, shared by `process`'s input validation and `inspect`'s report, so
/// the logs a user is *told* about are exactly the logs that would be read. A
/// `dir` that cannot be listed at all (absent, not a directory, unreadable)
/// yields no logs rather than an error: every caller already has a better
/// message for that case than `read_dir` does.
pub fn record_logs_in_dir(dir: &Path, harts: Option<&[u32]>) -> Vec<(u32, PathBuf)> {
    let mut logs: Vec<(u32, PathBuf)> = std::fs::read_dir(dir)
        .into_iter()
        .flatten()
        .flatten()
        .filter_map(|entry| {
            let path = entry.path();
            Some((source::hart_id_from_path(&path)?, path))
        })
        .filter(|(hart, _)| harts.is_none_or(|only| only.contains(hart)))
        .collect();
    logs.sort();
    logs
}

impl TraceInput {
    /// Classify a user-supplied path, refusing up front anything that cannot be
    /// processed.
    ///
    /// A directory is a spool; a file is a single record log (so an oddly-named
    /// log still works, as it did before).
    ///
    /// The validation belongs here, at the edge where the user's own words are
    /// still in hand: a path that does not exist is not a directory either, so
    /// without these checks it is silently classified as a *named log*, and the
    /// first layer to notice is the metadata read -- which looks in that log's
    /// parent directory and so complains about a path the user never typed.
    pub fn resolve(path: PathBuf, harts: Option<Vec<u32>>) -> anyhow::Result<Self> {
        if !path.exists() {
            anyhow::bail!(
                "trace input {} does not exist; point this at the --output-dir of a \
                 `turbo record` run, or at one of its turbo_trace_vcpu<N>.bin logs",
                path.display()
            );
        }

        if !path.is_dir() {
            // A named log wins over --vcpu; the user already picked one file.
            let hart_id = source::hart_id_from_path(&path).unwrap_or(0);
            return Ok(Self::RecordedLog { path, hart_id });
        }

        // Checked against the whole directory first, so "no recording here at
        // all" and "your --vcpu set matched none of the harts that are here"
        // are two different messages rather than one ambiguous one.
        let present = record_logs_in_dir(&path, None);
        if present.is_empty() {
            anyhow::bail!(
                "{} contains no turbo_trace_vcpu<N>.bin record logs, so there is \
                 nothing to process; is this the --output-dir of a `turbo record` \
                 run, and did that run complete?",
                path.display()
            );
        }
        if let Some(requested) = harts.as_deref() {
            if record_logs_in_dir(&path, Some(requested)).is_empty() {
                let recorded: Vec<u32> = present.iter().map(|(hart, _)| *hart).collect();
                anyhow::bail!(
                    "--vcpu {:?} selects no hart recorded in {}; recorded hart(s): {:?}",
                    requested,
                    path.display(),
                    recorded
                );
            }
        }

        Ok(Self::RecordedDir { dir: path, harts })
    }

    /// The directory report outputs default into when `--output-dir` is absent:
    /// where the input came from, not a fresh unrelated timestamped directory.
    pub fn default_output_dir(&self) -> PathBuf {
        match self {
            Self::RecordedDir { dir, .. } => dir.clone(),
            Self::RecordedLog { path, .. } => path
                .parent()
                .map(|p| p.to_path_buf())
                .unwrap_or_else(|| PathBuf::from(".")),
        }
    }
}

/// Pick the wire format for an offline `process` run, and set `SYSTEM_VLEN`.
///
/// None of the offline paths construct a `QemuRunner`, which is what would
/// otherwise set the system VLEN that the enricher reads for any vector op. So
/// the format chosen must also establish that VLEN here, or vector traces panic
/// on a value nothing ever set.
fn resolve_trace_format(
    input: &TraceInput,
    configured_vlen_bits: Option<u32>,
) -> anyhow::Result<TraceFormat> {
    // The recording itself is authoritative: the plugin stamps the guest's
    // `vlenb` CSR into turbo_metadata.json, so a spool recorded at one VLEN is
    // never re-enriched at another (which silently mis-sizes every vector op).
    //
    // Unconditional, deliberately: `SYSTEM_VLEN` starts at a non-zero default,
    // so gating this on `is_set()` would never read the recording at all and
    // every `process` would silently enrich at that default.
    let work_dir = input.default_output_dir();
    // nosemgrep: input-validation-path-traversal-2 -- work_dir joined with a const file name
    let meta_file = work_dir.join(turbo_tracer_shared::TURBO_METADATA_FILE);

    // Three genuinely different faults, each with its own fix, so each gets its
    // own message rather than one that lists all three and makes the reader
    // work out which applies. All are returned, not panicked: a user pointing
    // `process` at the wrong directory is bad input, not a turbo bug.
    if !meta_file.exists() {
        anyhow::bail!(
            "{} has no {}, so it is not a `turbo record` output directory; the VLEN \
             (and the memory map) a trace was recorded with live in that file",
            work_dir.display(),
            turbo_tracer_shared::TURBO_METADATA_FILE
        );
    }
    let recorded = turbo_tracer_shared::TurboMetadata::read_from_workdir(&work_dir)
        .ok_or_else(|| {
            anyhow!(
                "{} could not be parsed; the recording it describes is unusable \
                 (a partial write, or a file from an incompatible turbo version) \
                 -- re-record this trace",
                meta_file.display()
            )
        })?
        .vlen_bits;
    if recorded == 0 {
        anyhow::bail!(
            "{} records vlen_bits = 0; the VLEN this trace was recorded with cannot \
             be recovered, and enriching at a guessed VLEN would mis-size every \
             vector op -- re-record this trace",
            meta_file.display()
        );
    }

    // The recording wins unconditionally; this only says so out loud when the
    // invocation's `cpu_config.json` claimed a different VLEN. Cache *geometry*
    // is a different matter and may freely differ from the recording -- the
    // memory-address replay exists precisely so one trace can be re-simulated
    // against several cache configurations -- but VLEN cannot, because it
    // changes what the recorded instructions mean.
    crate::common::warn_on_vlen_mismatch(configured_vlen_bits, recorded, "cpu_config.json");

    // `SYSTEM_VLEN` holds bytes in a `u16`, so a `vlen_bits` outside the legal
    // range would narrow into a plausible-looking wrong VLEN rather than fail.
    // 65535 is the specific value to watch for: it is what a tracer predating
    // the `u32` widening wrote for a 65536-bit run, having saturated a `u16`.
    if !recorded.is_power_of_two() || !(VLEN_BITS_MIN..=VLEN_BITS_MAX).contains(&recorded) {
        anyhow::bail!(
            "{} records vlen_bits = {recorded}, which is not a power of two in \
             [{VLEN_BITS_MIN}, {VLEN_BITS_MAX}] bits, so it is not a VLEN any machine \
             ran at; re-record this trace",
            meta_file.display()
        );
    }
    crate::common::SYSTEM_VLEN.set_bytes((recorded / 8) as u16);
    log::info!("vlen set to {recorded} bits from metadata.");

    // The turbo tracer plugin only ever emits BBT now.
    Ok(TraceFormat::Bbt)
}

/// Options for [`process_trace_input_file`].
pub struct ProcessTraceOptions {
    pub annotate: bool,
    /// Build the ROI call-tree flamegraph (`--no-roi-flamegraph` clears it).
    pub roi_flamegraph: bool,
    pub source_dirs: Vec<PathBuf>,
    /// Model an L1/L2 cache over the captured vector memory addresses
    /// (`--cache-sim`). `None` -- the default -- models nothing.
    pub cache_sim: Option<processing::cache_model::Config>,
    pub perfetto: bool,
    pub trace_flush_threshold: usize,
    pub trace_counter_interval: u64,
    pub summary: Option<String>,
    /// Cap on parallel decode+enrich threads (`--decode-threads`).
    pub decode_threads: Option<usize>,
    /// The VLEN this invocation was configured with (`cpu_config.json`, or the
    /// effective value when `run --overlap=off` composes record and process).
    /// Never used to *set* the VLEN -- the recording's own metadata is
    /// authoritative -- only to warn when the two disagree. `None` asserts
    /// nothing and so can never mismatch.
    pub configured_vlen_bits: Option<u32>,
    // Hidden measurement mode: decode only, no enrichment, no outputs.
    pub decode_only: bool,
    /// Hidden cross-check mode (`--dump-pcs`): write every decoded PC to
    /// `turbo_trace_pcs.txt`. Implies decode-only: no reports are produced.
    pub dump_pcs: bool,
    /// `--strict-trace`: fail instead of reporting a trace that does not cover
    /// the whole run. Default `false`, matching the CLI.
    pub strict_trace: bool,
}

/// Hand-written rather than derived, for the same reason as `ProcessArgs`'s:
/// `#[derive(Default)]` would give `false` for the report flags, i.e. silently
/// suppress the annotation and flamegraph reports that every CLI user gets. A
/// caller filling this struct with `..Default::default()` should inherit the
/// CLI's behaviour, not the opposite of it.
impl Default for ProcessTraceOptions {
    fn default() -> Self {
        Self {
            annotate: true,
            roi_flamegraph: true,
            source_dirs: Vec::new(),
            cache_sim: None,
            perfetto: false,
            trace_flush_threshold: 100_000,
            trace_counter_interval: 1000,
            summary: None,
            decode_threads: None,
            configured_vlen_bits: None,
            decode_only: false,
            dump_pcs: false,
            strict_trace: false,
        }
    }
}

/// Decode a recorded trace file from disk, enrich it, and write the report
/// files (`perf_data_final.json`, `annotate_*.html.gz`, ...) into
/// `output_dir`. Never starts the HTTP server — that's `load`'s job.
pub fn process_trace_input_file(
    binary: PathBuf,
    input: TraceInput,
    output_dir: PathBuf,
    opts: ProcessTraceOptions,
) -> anyhow::Result<()> {
    let ProcessTraceOptions {
        annotate,
        roi_flamegraph,
        source_dirs,
        cache_sim,
        perfetto,
        trace_flush_threshold,
        trace_counter_interval,
        summary,
        decode_threads,
        configured_vlen_bits,
        decode_only,
        dump_pcs,
        strict_trace,
    } = opts;

    log::info!("Starting trace-elf processing..");

    // Before the output directory is created, so a `process` that is going to
    // refuse its input leaves nothing behind to be mistaken for a result.
    let trace_format = resolve_trace_format(&input, configured_vlen_bits)?;

    // Nothing else creates it on this path: unlike `run`, `process` builds no
    // QemuRunner (which is what makes the directory for a recording). Without
    // this every report write fails with a warning and the run looks like it
    // succeeded while producing nothing.
    if let Err(e) = std::fs::create_dir_all(&output_dir) {
        return Err(anyhow::anyhow!(
            "could not create output directory {}: {}",
            output_dir.display(),
            e
        ));
    }

    // Cloned before `input` is destructured into a source: the report has to
    // name the very same logs that were just decoded.
    let inspect_input = input.clone();

    // Every arm runs the SAME consumer; only the source differs. `impl
    // TraceSource` is a distinct type per arm, so the call is repeated rather
    // than the source being stored in a variable (boxing it would add a dynamic
    // dispatch to the hot poll path for no benefit).
    macro_rules! run_with {
        ($src:expr) => {
            run_perf_workflow_opt(
                binary,
                $src,
                Some(output_dir),
                // `process` never serves live; that's `load`'s job.
                true,
                annotate,
                roi_flamegraph,
                // `resolve_trace_format` above set `SYSTEM_VLEN` from the
                // recording's own metadata, which is the authoritative VLEN for
                // a trace being re-processed.
                cache_sim.map(|cfg| (u64::from(crate::common::SYSTEM_VLEN.get_bytes()), cfg)),
                perfetto,
                trace_flush_threshold,
                trace_counter_interval,
                summary,
                trace_format,
                source_dirs,
                decode_threads,
                decode_only,
                dump_pcs,
                strict_trace,
                inspect_input,
            )
        };
    }

    let result = match input {
        TraceInput::RecordedDir { dir, harts } => {
            log::info!("Binary: {:?}, recorded spool: {:?}", binary, dir);
            let src = source::recorded_dir(dir, binary.clone(), harts);
            run_with!(src)
        }
        TraceInput::RecordedLog { path, hart_id } => {
            log::info!(
                "Binary: {:?}, record log: {:?} (hart {})",
                binary,
                path,
                hart_id
            );
            let src = source::recorded_single_log(&path, hart_id, binary.clone());
            run_with!(src)
        }
    };

    result
        .map(|_| ())
        .map_err(|e| anyhow::anyhow!("TURBO process failed: {}", e))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A fresh scratch directory per test: these tests run in parallel and
    /// clear the directory they are given, so a shared name would let one test
    /// delete another's files mid-check.
    fn tmp_dir(name: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("turbo_trace_input_{name}"));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    fn write_log(dir: &Path, hart: u32) -> PathBuf {
        let path = dir.join(format!("turbo_trace_vcpu{hart}.bin"));
        std::fs::write(&path, b"not decoded by these tests").unwrap();
        path
    }

    fn write_meta(dir: &Path, vlen_bits: u32) {
        turbo_tracer_shared::TurboMetadata {
            run_id: turbo_tracer_shared::RunId::parse(&format!("run_{:016x}", 0xe57)),
            vlen_bits,
            meminfo: None,
            libs: Vec::new(),
        }
        .write_to_workdir(dir)
        .unwrap();
    }

    /// The QTR-PRO-003 regression: a trace input that does not exist must be a
    /// controlled error naming the path the user typed. It used to classify as
    /// a *named log* (a missing path is not a directory), and so panicked much
    /// later about `turbo_metadata.json` in that path's PARENT directory.
    #[test]
    fn missing_trace_input_is_a_controlled_error() {
        let dir = tmp_dir("missing_input");
        let missing = dir.join("missing-trace");

        let err = TraceInput::resolve(missing.clone(), None)
            .expect_err("a non-existent trace input must be refused");
        let msg = err.to_string();
        assert!(
            msg.contains(&missing.display().to_string()),
            "the error must name the path the user typed, got: {msg}"
        );
        assert!(
            !msg.contains("turbo_metadata.json"),
            "the fault is the missing input, not the metadata, got: {msg}"
        );
    }

    /// An existing directory with no record logs is a different mistake from a
    /// path that is not there at all, and gets its own message.
    #[test]
    fn directory_without_record_logs_is_refused() {
        let dir = tmp_dir("no_logs");
        // Metadata alone is not a recording: `process` has nothing to decode.
        write_meta(&dir, 1024);

        let err = TraceInput::resolve(dir.clone(), None)
            .expect_err("a directory with no record logs must be refused");
        assert!(
            err.to_string().contains("no turbo_trace_vcpu<N>.bin"),
            "got: {err}"
        );
    }

    /// A `--vcpu` set that matches none of the recorded harts names the harts
    /// that ARE there, so the user can fix the flag without a second run.
    #[test]
    fn vcpu_filter_matching_nothing_names_the_recorded_harts() {
        let dir = tmp_dir("vcpu_filter");
        write_log(&dir, 0);
        write_log(&dir, 1);

        let err = TraceInput::resolve(dir.clone(), Some(vec![7]))
            .expect_err("a --vcpu set matching no recorded hart must be refused");
        let msg = err.to_string();
        assert!(msg.contains("[0, 1]"), "got: {msg}");

        // The same directory with a hart that IS present resolves.
        let ok = TraceInput::resolve(dir, Some(vec![1])).expect("hart 1 is recorded");
        assert!(matches!(ok, TraceInput::RecordedDir { .. }));
    }

    /// A directory of logs resolves as a spool, and a named log as one hart --
    /// the classification the validation must not have changed.
    #[test]
    fn valid_inputs_still_resolve() {
        let dir = tmp_dir("valid");
        let log = write_log(&dir, 3);

        assert!(matches!(
            TraceInput::resolve(dir, None).unwrap(),
            TraceInput::RecordedDir { .. }
        ));
        assert!(matches!(
            TraceInput::resolve(log, None).unwrap(),
            TraceInput::RecordedLog { hart_id: 3, .. }
        ));
    }

    /// The three metadata faults `resolve_trace_format` used to merge into one
    /// panic are three distinct errors, each naming its own fix.
    #[test]
    fn metadata_faults_are_distinct_errors() {
        let dir = tmp_dir("metadata_faults");
        write_log(&dir, 0);
        let input = TraceInput::resolve(dir.clone(), None).unwrap();

        let err = resolve_trace_format(&input, None).expect_err("absent metadata must be refused");
        assert!(
            err.to_string().contains("is not a `turbo record` output"),
            "got: {err}"
        );

        std::fs::write(dir.join("turbo_metadata.json"), b"{ not json").unwrap();
        let err =
            resolve_trace_format(&input, None).expect_err("unparseable metadata must be refused");
        assert!(
            err.to_string().contains("could not be parsed"),
            "got: {err}"
        );

        write_meta(&dir, 0);
        let err = resolve_trace_format(&input, None).expect_err("vlen_bits = 0 must be refused");
        assert!(err.to_string().contains("vlen_bits = 0"), "got: {err}");

        // 65535 is what a tracer predating the `u32` widening wrote for a
        // 65536-bit run, having saturated a `u16`: not a VLEN any machine ran
        // at, so it is refused rather than narrowed into a plausible one.
        write_meta(&dir, 65535);
        let err = resolve_trace_format(&input, None)
            .expect_err("a non-power-of-two VLEN must be refused");
        assert!(err.to_string().contains("vlen_bits = 65535"), "got: {err}");

        write_meta(&dir, 1024);
        resolve_trace_format(&input, None).expect("a well-formed recording resolves");
        assert_eq!(crate::common::SYSTEM_VLEN.get_bytes(), 128);
    }
}
