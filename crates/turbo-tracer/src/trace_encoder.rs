use std::sync::{Arc, Mutex};
/// BBT (basic-block-trace) encoder state: per-hart record log, dictionary
/// bookkeeping, and ROI side-channel buffering.
use turbo_tracer_shared::bbt::{
    bbt_vl, encode_chunk_into, push_mem_group, MemGroupKind, BBT_CHUNK_HEADER_BYTES,
};
use turbo_tracer_shared::{BatchExtrasRef, RecordLogWriter, RoiRtMarker};

use std::fs::{File, OpenOptions};
use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::sync::mpsc::{sync_channel, Receiver, Sender, SyncSender};
use std::sync::Once;
use std::thread::JoinHandle;

/// Emit one diagnostic line: to the encoder's debug log file if it has one,
/// otherwise through the `log` crate at the given level.
///
/// Taking `&File` (which implements `Write`) rather than a buffered writer keeps
/// this usable from `&self` methods and makes each line a single unbuffered
/// append, so lines from concurrent harts don't interleave mid-message.
macro_rules! dlog {
    ($self:expr, $level:ident, $($arg:tt)*) => {
        match $self.debug_log.as_ref() {
            Some(mut f) => {
                let _ = writeln!(
                    f,
                    "[{:<5}] {}",
                    stringify!($level).to_uppercase(),
                    format_args!($($arg)*)
                );
            }
            None => log::$level!($($arg)*),
        }
    };
}

/// The debug log is shared by every hart, so the first encoder to open it
/// truncates any file left over from a previous run and the rest append.
fn open_debug_log(path: &Path) -> Option<File> {
    static TRUNCATE_ONCE: Once = Once::new();
    TRUNCATE_ONCE.call_once(|| {
        let _ = File::create(path);
    });
    match OpenOptions::new().append(true).create(true).open(path) {
        Ok(f) => Some(f),
        Err(e) => {
            log::error!(
                "TraceEncoder: could not open debug log {}: {e}",
                path.display()
            );
            None
        }
    }
}

// ---------------------------------------------------------------------------
// Compress-and-write helper thread
// ---------------------------------------------------------------------------

/// Depth of the encoder -> writer queue, in flush windows.
///
/// Bounded on purpose: a writer that falls behind must apply backpressure, not
/// grow RAM. A hart's in-flight trace bytes are at most `DEPTH + 2` windows
/// (queued, plus the one the writer holds, plus the one the encoder is filling),
/// so this is ~240 KB per hart at the plugin's default `flush_interval=10_000`
/// and ~24 MB if that is raised to a million. Deep enough that a single slow
/// write or page fault on the writer side does not immediately stall the vCPU,
/// shallow enough that the memory is not worth counting at the default.
const WRITER_QUEUE_DEPTH: usize = 4;

/// Upper bound on the `flush_interval` used to *pre-size* a record buffer.
///
/// `flush_interval` comes straight off the plugin's argument string, so a typo
/// (or a value in the wrong unit) would otherwise turn into a reservation of
/// `4 * flush_interval` bytes per hart -- an immediate abort inside the guest's
/// own process, with WRITER_QUEUE_DEPTH+2 of them in flight. Clamping only the
/// *capacity* hint costs nothing: a window larger than this still works, the
/// buffer just grows into it once instead of being reserved up front.
const MAX_PRESIZED_FLUSH_INTERVAL: u64 = 8 << 20;

/// Initial capacity for one flush window's record buffer, in bytes.
fn record_buffer_capacity(flush_interval: u64) -> usize {
    4 * flush_interval.min(MAX_PRESIZED_FLUSH_INTERVAL) as usize
}

/// One flush window, handed to the hart's writer thread.
///
/// The *raw* pieces travel, not a finished payload: chunk assembly, postcard
/// serialization, deflate and `write(2)` all belong to the writer. That is the
/// whole point -- on measured 4-hart runs those four steps were 30-38% of every
/// vCPU thread's CPU.
struct WriteJob {
    /// Dictionary bytes interned since the previous window. Snapshotted on the
    /// vCPU thread under `bbt_dict_log`'s lock, because the "descriptor before
    /// any record naming it" guarantee depends on reading the tail at exactly
    /// this point; see [`TraceEncoder::dict_sent`].
    dict: Vec<u8>,
    /// This window's BBT record words, *moved* out of the encoder rather than
    /// copied. Returned to the encoder on the spare channel once written.
    records: Vec<u8>,
    /// ROI markers captured during this window.
    roi_markers: Vec<RoiRtMarker>,
    /// Encoder-side numbers for the per-batch debug line; `None` unless debug
    /// logging is on, in which case the writer emits the line (it owns the
    /// on-disk totals the line also needs).
    debug: Option<BatchDebug>,
}

/// The encoder-side half of one batch's debug line. See [`WriteJob::debug`].
struct BatchDebug {
    batch_index: u64,
    block_bytes: u64,
    vl_bytes: u64,
    mem_bytes: u64,
    dict_entries: u64,
}

/// A writer thread's final word: the file's byte totals, read after the `Done`
/// sentinel is on disk. Returned through the `JoinHandle` rather than shared
/// atomics so [`TraceEncoder::log_write_summary`] sees exact numbers with no
/// "the writer might still be behind" caveat.
struct WriterTotals {
    on_disk: u64,
    logical: u64,
    /// The file's uncompressed header, whose length depends on the run id it
    /// carries.
    header: u64,
}

/// Just enough of a `self` for the [`dlog!`] macro on the writer thread.
struct WriterCtx {
    debug_log: Option<File>,
}

/// Compress and write one hart's stream until its channel closes.
///
/// Owns the [`RecordLogWriter`] outright, so its reused `compressor` and
/// `scratch` (the reason a flush allocates nothing in steady state) keep working
/// unchanged -- one deflate state per file, which is what the format needs.
///
/// Closing the job channel is the shutdown signal: the loop drains what is
/// queued, appends the `Done` sentinel and `sync_all`s, then reports the totals.
///
/// I/O errors are logged and the loop continues, matching what the synchronous
/// flush path did. The thread never exits early, so a `send` on the encoder side
/// can only fail after the encoder itself closed the channel.
fn writer_loop(
    hart_id: u64,
    mut record_log: RecordLogWriter,
    jobs: Receiver<WriteJob>,
    spares: Sender<Vec<u8>>,
    debug_log: Option<File>,
) -> WriterTotals {
    let ctx = WriterCtx { debug_log };
    // Reused across windows, so assembling a chunk is two `memcpy`s and no
    // allocation.
    let mut chunk = Vec::new();

    for job in jobs {
        let WriteJob {
            dict,
            mut records,
            roi_markers,
            debug,
        } = job;

        encode_chunk_into(&mut chunk, &dict, &records);

        let before = record_log.bytes_written();
        let logical_before = record_log.logical_bytes();

        let extras = BatchExtrasRef {
            roi_markers: &roi_markers,
        };
        if let Err(e) = record_log.append_batch(&chunk, extras) {
            dlog!(
                ctx,
                error,
                "TraceEncoder[hart={}]: record-log append failed: {}",
                hart_id,
                e
            );
        } else if let Err(e) = record_log.flush() {
            dlog!(
                ctx,
                error,
                "TraceEncoder[hart={}]: record-log flush failed: {}",
                hart_id,
                e
            );
        }

        if let Some(d) = debug {
            // Whatever the file grew by beyond this chunk is envelope: the `[u32
            // len]` frame, postcard's enum tag and length varints, and the
            // serialized ROI markers.
            let on_disk = record_log.bytes_written() - before;
            let logical = record_log.logical_bytes() - logical_before;
            dlog!(
                ctx,
                debug,
                "TraceEncoder[hart={}]: BBT batch #{}: {} B on disk ({:.1}x) from {} B \
             uncompressed = {} B block records ({}) + {} B vl records ({}) + {} B mem \
             records + {} B dict ({} descriptors) + {} B chunk header + {} B envelope \
             ({} roi markers)",
                hart_id,
                d.batch_index,
                on_disk,
                logical as f64 / on_disk.max(1) as f64,
                logical,
                d.block_bytes,
                d.block_bytes / 4,
                d.vl_bytes,
                d.vl_bytes / 4,
                d.mem_bytes,
                dict.len(),
                d.dict_entries,
                BBT_CHUNK_HEADER_BYTES,
                logical.saturating_sub(chunk.len() as u64),
                roi_markers.len(),
            );
        }

        // Hand the (multi-megabyte) record buffer back for the next window. A
        // failed send just means the encoder is gone, and the channel this loop
        // is draining is about to close anyway.
        records.clear();
        let _ = spares.send(records);
    }

    if let Err(e) = record_log.finish() {
        dlog!(
            ctx,
            error,
            "TraceEncoder[hart={}]: record-log finish failed: {}",
            hart_id,
            e
        );
    }

    WriterTotals {
        on_disk: record_log.bytes_written(),
        logical: record_log.logical_bytes(),
        header: turbo_tracer_shared::RECORD_LOG_HEADER_BYTES as u64,
    }
}

pub struct TraceEncoder {
    /// Total number of steps processed
    pub total_steps: u64,
    /// Hart ID for this encoder (maps to vCPU index)
    hart_id: u64,
    /// Queue to this hart's compress-and-write thread, which owns the
    /// append-only record-log writer. Bounded, so a writer that falls behind
    /// blocks this thread instead of growing RAM. `None` once
    /// [`finish`](Self::finish) has closed it.
    writer_tx: Option<SyncSender<WriteJob>>,
    /// That thread's handle. Joining it yields the file's final byte totals.
    writer: Option<JoinHandle<WriterTotals>>,
    /// Return path for drained record buffers, so a flush *moves* its buffer to
    /// the writer instead of allocating a fresh multi-megabyte `Vec` per window.
    spare_rx: Receiver<Vec<u8>>,
    /// Set the first time a send to the writer thread fails, so the diagnostic
    /// is emitted once rather than once per window.
    writer_gone: bool,
    /// Counter for periodic record-log flushes
    steps_since_last_flush: u64,
    /// Flush interval (every N steps)
    flush_interval: u64,
    /// Debug log file, open for append when debug logging is enabled. When this
    /// is `Some`, every diagnostic this encoder emits goes here instead of the
    /// `log` crate; when it is `None`, the `log` crate is used as usual.
    debug_log: Option<File>,
    /// ROI markers accumulated since last flush
    roi_markers_since_last_flush: Vec<RoiRtMarker>,

    /// BBT record words (little-endian `u32`) accumulated since the last flush.
    records: Vec<u8>,
    /// Append-only dictionary byte log, shared by every hart. Block IDs are
    /// global (interning happens once, at translate time), so a block interned
    /// while one hart was running can be executed by another.
    dict_log: Arc<Mutex<Vec<u8>>>,
    /// How many bytes of `dict_log` this hart has already shipped.
    ///
    /// Snapshotting the log's tail at flush time is what makes "the dictionary
    /// entry arrives before any record naming it" true: a record for block X is
    /// only in this buffer because X executed, which can only happen after X was
    /// interned, so X's bytes are already in the log by the time this hart reads
    /// it.
    dict_sent: usize,

    /// Per-category byte/record accounting for the debug size report. Kept as
    /// plain counters bumped on paths that already exist, so the hot
    /// `record_block_word` append stays a store and two increments.
    stats: WriteStats,
}

/// Running attribution of everything this hart has put on disk.
///
/// The point of this is answering "what is the .bin actually made of" without
/// re-reading the file: every byte written falls into exactly one of these
/// buckets, and they sum to `RecordLogWriter::bytes_written`.
#[derive(Default, Clone, Copy)]
struct WriteStats {
    /// Flush windows shipped (one `StreamRecord::Batch` each).
    batches: u64,
    /// `BLOCK` record words appended (4 bytes each).
    block_records: u64,
    /// `VL` record words appended (4 bytes each).
    vl_records: u64,
    /// `VL` words appended since the last flush; reset per window.
    vl_since_last_flush: u64,
    /// `MEM` address groups appended (one per vector memory instruction).
    mem_groups: u64,
    /// `ADDR` half-words appended (two per captured address or stride).
    addr_records: u64,
    /// `MEM` + `ADDR` words appended since the last flush; reset per window.
    mem_words_since_last_flush: u64,
    /// Dictionary bytes shipped (block descriptors: id, start_pc, sizes).
    dict_bytes: u64,
    /// Dictionary entries shipped, counted by the descriptors in those bytes.
    dict_entries: u64,
    /// BBT chunk headers (magic + two section lengths) -- 12 bytes per batch.
    chunk_header_bytes: u64,
    /// ROI markers shipped as batch side-channel data.
    roi_markers: u64,
}

impl WriteStats {
    fn record_bytes(&self) -> u64 {
        4 * (self.block_records + self.vl_records + self.mem_groups + self.addr_records)
    }
}

impl TraceEncoder {
    /// Create a new encoder that streams to a per-hart append-only record log.
    ///
    /// The record-log file is created (truncated) at
    /// `workdir/turbo_trace_vcpu<hart_id>.bin`, giving a fresh file per run and
    /// per-hart isolation (no shared file handle across threads).
    ///
    /// # Arguments
    /// * `hart_id` - Hart ID for this encoder (maps to vCPU index)
    /// * `workdir` - Working directory in which to create the per-hart record log
    /// * `flush_interval` - Number of steps between record-log flushes
    /// * `debug_log_path` - Optional path to debug log file
    /// * `dict_log` - Shared BBT dictionary log this hart appends to and reads from
    /// * `compress_level` - Deflate level for record payloads (see `compress=`)
    /// * `run_id` - this run's identifier (the plugin's `run_id=` argument),
    ///   stamped into the record log's header so a consumer can tell that the
    ///   log, its siblings and the `turbo_metadata.json` beside them all came
    ///   from one execution. `None` when the plugin was invoked without one,
    ///   i.e. by hand rather than by `turbo run`
    ///
    /// # Errors
    /// Returns an error if the per-hart record-log file could not be created.
    pub fn new_with_record_log(
        hart_id: u64,
        workdir: &Path,
        flush_interval: u64,
        debug_log_path: Option<PathBuf>,
        dict_log: Arc<Mutex<Vec<u8>>>,
        compress_level: u32,
        run_id: Option<turbo_tracer_shared::RunId>,
    ) -> std::io::Result<Self> {
        // nosemgrep: input-validation-path-traversal-2 -- workdir joined with a name generated from a u32 hart id
        let record_log_path = workdir.join(turbo_tracer_shared::get_turbo_trace_bin_for_vcpu(
            hart_id as u32,
        ));
        // Created here, on the calling thread, and only then moved into the
        // writer: an `io::Error` opening the file still surfaces synchronously
        // from this constructor, and the file exists with a valid header the
        // moment the encoder does -- which the live-tailing consumer in
        // `turbo::source::record_log` relies on.
        let record_log =
            RecordLogWriter::create_with_level(&record_log_path, compress_level, run_id)?;

        let (job_tx, job_rx) = sync_channel(WRITER_QUEUE_DEPTH);
        let (spare_tx, spare_rx) = std::sync::mpsc::channel();
        // A second append handle on the shared debug log, which is already the
        // file's contract: one unbuffered append per line, so lines from
        // concurrent harts and their writers never interleave mid-message.
        let writer_debug_log = debug_log_path.as_deref().and_then(open_debug_log);
        let writer = std::thread::Builder::new()
            .name(format!("turbo-w{hart_id}"))
            .spawn(move || writer_loop(hart_id, record_log, job_rx, spare_tx, writer_debug_log))?;

        Ok(Self {
            total_steps: 0,
            hart_id,
            writer_tx: Some(job_tx),
            writer: Some(writer),
            spare_rx,
            writer_gone: false,
            steps_since_last_flush: 0,
            flush_interval,
            debug_log: debug_log_path.as_deref().and_then(open_debug_log),
            roi_markers_since_last_flush: Vec::new(),
            // One flush window is `flush_interval` block entries; sizing the
            // buffer for that up front keeps the hot append a store and a
            // length bump, with no growth reallocation in steady state.
            // nosemgrep: dos-unbounded-memory-allocation -- record_buffer_capacity clamps the flush_interval hint
            records: Vec::with_capacity(record_buffer_capacity(flush_interval)),
            dict_log,
            dict_sent: 0,
            stats: WriteStats::default(),
        })
    }

    /// Check if instruction is a vector configuration instruction (vsetvli, vsetivli, vsetvl)
    /// These instructions write to VL (0xC20) and VTYPE (0xC21) CSRs
    pub fn is_vector_set_instruction(disas: &str) -> bool {
        disas.starts_with("vsetvli")
            || disas.starts_with("vsetivli")
            || disas.starts_with("vsetvl ")
    }

    /// Record one block entry: append its precomputed record word.
    ///
    /// This is the whole per-block cost of BBT mode, and the reason the format
    /// exists. `word` is `bbt_block(id)` computed once at translate time and
    /// captured in the block's callback closure, so nothing here derives,
    /// branches on, or encodes anything.
    #[inline(always)]
    pub fn record_block_word(&mut self, word: u32) {
        // Flush *before* appending, not after. A window must never end between a
        // BLOCK record and the VL record that follows it: the VL is appended by
        // the `vsetvl*`'s instruction callback, which runs after this block
        // callback has already returned. Cutting here -- always immediately
        // before a BLOCK -- is the producer half of the batching invariant the
        // consumer relies on, and it is why a chunk boundary can never leave a
        // `vsetvl*` in one batch with its `vl` in the next.
        if self.steps_since_last_flush >= self.flush_interval {
            self.flush_records();
        }

        self.records.extend_from_slice(&word.to_le_bytes());

        self.stats.block_records += 1;
        self.total_steps += 1;
        self.steps_since_last_flush += 1;
    }

    /// Record the `vl` an instruction produced. Two call sites: the `vsetvl*`
    /// instruction callback, and the callback on the block a `vl*ff` falls
    /// through to.
    ///
    /// The record carries the value alone: which instruction produced it is its
    /// position in the stream, which the consumer already has from walking the
    /// block's PCs.
    ///
    /// The `vl` record must land immediately after the `BLOCK` record of the
    /// block whose last instruction wrote `vl`, and before the next block's --
    /// which it does for both producers. A `vsetvl*` is always its block's last
    /// instruction and its callback runs after the block callback; a `vl*ff` is
    /// likewise its block's last instruction, and the callback that reads its
    /// `vl` is registered on the successor block ahead of that block's own
    /// record-word callback. The consumer relies on that adjacency to pick safe
    /// batch boundaries.
    #[inline]
    pub fn record_vl(&mut self, vl: u64) {
        self.records.extend_from_slice(&bbt_vl(vl).to_le_bytes());
        self.stats.vl_records += 1;
        self.stats.vl_since_last_flush += 1;
    }

    /// Record one vector memory instruction's captured addresses.
    ///
    /// Called from that instruction's execute callback, so the group lands after
    /// the `BLOCK` record of the block containing it and in program order with
    /// the other memory ops of that block -- which is what lets the consumer
    /// match groups to PCs by walking the block.
    #[inline]
    pub fn record_mem_group(&mut self, kind: MemGroupKind, values: &[u64]) {
        push_mem_group(&mut self.records, kind, values);
        self.stats.mem_groups += 1;
        self.stats.addr_records += 2 * values.len() as u64;
        self.stats.mem_words_since_last_flush += 1 + 2 * values.len() as u64;
    }

    /// Hand this window's records, together with any dictionary entries interned
    /// since the last flush, to this hart's writer thread.
    ///
    /// Everything expensive -- chunk assembly, postcard, deflate, `write(2)` --
    /// happens on that thread. What is left here is a dictionary-tail snapshot,
    /// a few counter bumps, and moving two buffers into a channel.
    fn flush_records(&mut self) {
        if self.records.is_empty() {
            self.steps_since_last_flush = 0;
            return;
        }

        // Read the dictionary tail *now*, i.e. after this window's records were
        // appended, so every block they name is already described. See
        // `dict_sent`. This stays on the vCPU thread deliberately: the guarantee
        // is about reading the tail at exactly this point, so it cannot be
        // deferred to a thread that would observe a later dictionary state.
        let dict: Vec<u8> = {
            let log = self
                .dict_log
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            log[self.dict_sent..].to_vec()
        };
        self.dict_sent += dict.len();
        // Descriptor layout is `id: u32, start_pc: u64, n: u16, n size bytes`,
        // so entries are only countable by walking them -- done here, off the
        // per-block path, and only over bytes new in this window.
        let dict_entries = count_dict_entries(&dict);

        let record_bytes = self.records.len();
        let vl_bytes = 4 * self.stats.vl_since_last_flush;
        let mem_bytes = 4 * self.stats.mem_words_since_last_flush;
        let block_bytes = record_bytes as u64 - vl_bytes - mem_bytes;

        let roi_markers = std::mem::take(&mut self.roi_markers_since_last_flush);

        self.stats.batches += 1;
        self.stats.dict_bytes += dict.len() as u64;
        self.stats.dict_entries += dict_entries;
        self.stats.chunk_header_bytes += BBT_CHUNK_HEADER_BYTES;
        self.stats.roi_markers += roi_markers.len() as u64;
        self.stats.vl_since_last_flush = 0;
        self.stats.mem_words_since_last_flush = 0;

        let debug = self.debug_log.is_some().then_some(BatchDebug {
            batch_index: self.stats.batches,
            block_bytes,
            vl_bytes,
            mem_bytes,
            dict_entries,
        });

        // Move the window's records out rather than copying them: the writer
        // owns the buffer until it hands it back on `spare_rx`.
        let job = WriteJob {
            dict,
            records: std::mem::take(&mut self.records),
            roi_markers,
            debug,
        };

        // Blocks when the queue is full, which is the intended backpressure: it
        // costs this thread at most the time it would have spent doing the
        // compression itself, and it is what bounds the hart's queued bytes.
        let sent = match self.writer_tx.as_ref() {
            Some(tx) => tx.send(job).is_ok(),
            None => false,
        };
        if !sent && !self.writer_gone {
            self.writer_gone = true;
            dlog!(
                self,
                error,
                "TraceEncoder[hart={}]: writer thread is gone; trace data from \
                 this window on is dropped",
                self.hart_id
            );
        }

        let fresh = self.take_spare_buffer();
        self.records = fresh;
        self.steps_since_last_flush = 0;
    }

    /// A cleared record buffer for the next window: the one the writer just
    /// handed back if there is one, otherwise a fresh allocation.
    ///
    /// In steady state the pool cycles a fixed handful of buffers, so this is a
    /// `try_recv` and nothing else. It only allocates during the first few
    /// windows, or while the writer is far enough behind that every buffer is
    /// still in flight.
    fn take_spare_buffer(&self) -> Vec<u8> {
        self.spare_rx
            .try_recv()
            // nosemgrep: dos-unbounded-memory-allocation -- record_buffer_capacity clamps the flush_interval hint
            .unwrap_or_else(|_| Vec::with_capacity(record_buffer_capacity(self.flush_interval)))
    }

    /// One-line-per-category breakdown of everything this hart wrote, emitted
    /// once at `finish`. The same numbers `turbo --inspect` recomputes from the file
    /// on disk -- if the two disagree, one of the two is wrong.
    fn log_write_summary(&self, totals: &WriterTotals) {
        let s = &self.stats;
        // Two totals, because they answer different questions. `on_disk` is the
        // file's real size; the categories below are shares of the UNCOMPRESSED
        // payload, since deflate shares a window across a whole batch and there
        // is no honest way to attribute compressed bytes to one category.
        // Everything not in a BBT chunk -- the log header, the per-record framing,
        // postcard's tags and varints, the `Done` sentinel, and the serialized
        // ROI markers -- falls out as the residual.
        let on_disk = totals.on_disk;
        let logical = totals.logical + totals.header;
        let envelope =
            logical.saturating_sub(s.record_bytes() + s.dict_bytes + s.chunk_header_bytes);
        let pct = |n: u64| 100.0 * n as f64 / logical.max(1) as f64;
        dlog!(
            self,
            debug,
            "TraceEncoder[hart={}]: wrote {} B on disk in {} batches, {} B uncompressed \
             ({:.1}x compression):",
            self.hart_id,
            on_disk,
            s.batches,
            logical,
            logical as f64 / on_disk.max(1) as f64,
        );
        for (label, bytes, count) in [
            ("bbt block executions", 4 * s.block_records, s.block_records),
            ("bbt vl values", 4 * s.vl_records, s.vl_records),
            (
                "bbt memory addresses",
                4 * (s.mem_groups + s.addr_records),
                s.mem_groups,
            ),
            ("bbt block descriptors", s.dict_bytes, s.dict_entries),
            ("bbt chunk headers", s.chunk_header_bytes, s.batches),
            ("envelope + roi markers", envelope, s.roi_markers),
        ] {
            dlog!(
                self,
                debug,
                "TraceEncoder[hart={}]:   {:<24} {:>12} B ({:>5.1}%) over {:>10} item(s)",
                self.hart_id,
                label,
                bytes,
                pct(bytes),
                count,
            );
        }
    }

    pub fn record_roi_marker(&mut self, marker: RoiRtMarker) {
        dlog!(
            self,
            trace,
            "ROI marker recorded: {:?} at PC {:#x} rs1={:#x} rs2={:#x}",
            marker.kind,
            marker.pc,
            marker.rs1_val,
            marker.rs2_val
        );
        self.roi_markers_since_last_flush.push(marker);
    }

    /// End the trace, drain the writer thread and join it.
    ///
    /// Call this exactly once before dropping the encoder -- it is the only
    /// thing that waits for the hart's queued windows to reach the file. BBT
    /// carries no cross-record state, so the stream is complete as soon as the
    /// last record is written; this ships whatever remains, then the writer
    /// appends a `Done` record (flush+sync) so a live tailing reader knows the
    /// stream ended.
    pub fn finish(&mut self) {
        if !self.records.is_empty() {
            self.flush_records();
        }
        // Closing the queue is the writer's shutdown signal: it drains what is
        // still queued, appends `Done`, `sync_all`s and reports the file's byte
        // totals. Reading them only after the join is what keeps them exact.
        drop(self.writer_tx.take());
        match self.writer.take().map(JoinHandle::join) {
            Some(Ok(totals)) => self.log_write_summary(&totals),
            Some(Err(_)) => dlog!(
                self,
                error,
                "TraceEncoder[hart={}]: writer thread panicked; its record log \
                 may be truncated and no write summary is available",
                self.hart_id
            ),
            // Already finished; nothing left to close or join.
            None => {}
        }
    }
}

/// Count the block descriptors in a run of dictionary bytes, walking the
/// variable-length entries. Returns what it managed to count on a truncated
/// tail rather than erroring: this is accounting, not decoding.
fn count_dict_entries(mut bytes: &[u8]) -> u64 {
    let mut n = 0;
    while bytes.len() >= 14 {
        let insns = usize::from(u16::from_le_bytes([bytes[12], bytes[13]]));
        if bytes.len() < 14 + insns {
            break;
        }
        bytes = &bytes[14 + insns..];
        n += 1;
    }
    n
}
