//! Incremental reader for one hart's `turbo_trace_vcpu<N>.bin` record log.
//!
//! This is the *only* record-log decoder. Both supported inputs are the same
//! thing — a framed record log on disk — differing only in whether more bytes
//! can still be appended, so the live path and the offline path share this
//! reader rather than each carrying their own copy of the framing walk.
//!
//! The log is an 8-byte magic header followed by `[u32 LE len][postcard
//! StreamRecord]` records, ending with a `Done` record. Each `Batch` becomes one
//! [`TraceData::Trace`].
//!
//! Reads are bounded per [`pump`](HartLogReader::pump) call in two ways — a
//! batch cap and a buffer low-water mark — so a finished multi-gigabyte log is
//! streamed in bounded pieces instead of being faulted in whole. That matters
//! for the offline path in particular: reading the entire file and decoding
//! every batch up front (as the old `TraceFileSource` did) made peak RSS scale
//! with trace size, times the hart count.

use std::io::Read;
use std::path::Path;

use turbo_tracer_shared::{
    check_record_log_header, decode_next_record, RecordLogDecoder, RunId, StreamRecord,
    LEN_PREFIX_BYTES, RECORD_HEADER_BYTES, RECORD_LOG_HEADER_BYTES,
};

use crate::source::TraceData;

/// Read at most this many bytes from the file in one `pump` call.
const PUMP_READ_BUDGET: usize = 4 * 1024 * 1024;

/// Stop reading once this many undecoded bytes are buffered.
///
/// This is what actually bounds memory: decode drains the buffer as batches are
/// handed to the consumer, and reading tops it back up only once it runs low. So
/// the reader self-throttles to the consumer's rate instead of racing ahead of
/// it.
const BUFFER_LOW_WATER: usize = 1024 * 1024;

/// Outcome of one [`HartLogReader::pump`] call.
pub(crate) struct PumpStat {
    /// Batches appended to `out` by this call.
    #[cfg_attr(not(test), allow(dead_code))]
    pub batches: usize,
    /// Whether this reader may still have data to hand over *right now*
    /// (independent of whether the producer is still running): either the batch
    /// cap was hit with records still buffered, or the file has not been read to
    /// EOF yet. When false and the producer has finished, the reader is spent.
    pub more_available: bool,
}

/// One hart's record log, read incrementally.
///
/// Holds the open file, a persistent buffer with a decode cursor, and
/// header/termination state. Torn tails are left buffered for a later pass.
pub(crate) struct HartLogReader {
    hart_id: u32,
    file: std::fs::File,
    buf: Vec<u8>,
    cursor: usize,
    header_ok: bool,
    /// Set on `Done`, a decode error or a bad header: nothing further will ever
    /// be decoded from this reader.
    ///
    /// Deliberately NOT set by a partial record at EOF. While the producer is
    /// running, a torn tail is the normal state between two writes, so treating
    /// it as terminal here would silently drop the rest of the hart's trace.
    /// Whether a torn tail is "truncated log" or "producer paused" is knowable
    /// only from the producer, so [`SpoolSource`] decides it.
    ///
    /// [`SpoolSource`]: crate::source::spool_source::SpoolSource
    done: bool,
    /// Set when the log is structurally unusable: a bad magic header, a record
    /// that will not decode, or an I/O error mid-read. Distinct from `done`,
    /// which is also set by the normal `Done` sentinel -- this says the trace
    /// LOST DATA, which is a reason to fail the run rather than to report
    /// whatever was decoded before it.
    fault: Option<String>,
    /// The run this log says it belongs to. Checked against the run the memory
    /// map describes, so a log from another execution -- a stale file in a
    /// reused directory, or a spool assembled from two runs -- is refused
    /// rather than decoded against the wrong address layout.
    ///
    /// `None` both before the header has been parsed (a live reader can open
    /// the file before the producer's first write lands) and when the header
    /// carries no id. Deliberately one state: either way there is nothing to
    /// check this log against, which is what every caller acts on.
    run_id: Option<RunId>,
    /// Whether the most recent read attempt reached end-of-file.
    at_eof: bool,
    bytes_read: usize,
    /// Batches handed over so far, and whether the producer's terminal `Done`
    /// record was reached. Reported per hart in the trace-integrity block, so
    /// a caller can tell a whole log from one that was cut short.
    batches_decoded: u64,
    saw_done: bool,
    /// Inflate state and scratch, reused for every record this hart decodes.
    decoder: RecordLogDecoder,
}

impl HartLogReader {
    pub(crate) fn open(hart_id: u32, path: &Path) -> std::io::Result<Self> {
        let file = std::fs::File::open(path)?;
        Ok(Self {
            hart_id,
            file,
            buf: Vec::new(),
            cursor: 0,
            header_ok: false,
            done: false,
            fault: None,
            run_id: None,
            at_eof: false,
            bytes_read: 0,
            batches_decoded: 0,
            saw_done: false,
            decoder: RecordLogDecoder::new(),
        })
    }

    pub(crate) fn hart_id(&self) -> u32 {
        self.hart_id
    }

    pub(crate) fn done(&self) -> bool {
        self.done
    }

    /// Why this log is unusable, if it is. See [`fault`](Self::fault).
    pub(crate) fn fault(&self) -> Option<&str> {
        self.fault.as_deref()
    }

    /// The run id from this log's header; see [`run_id`](Self::run_id) for what
    /// `None` covers.
    pub(crate) fn run_id(&self) -> Option<RunId> {
        self.run_id
    }

    /// Record a fault found by a layer above this one -- the run-id
    /// cross-check, which needs the memory map this reader knows nothing
    /// about. Terminal, like the framing faults: nothing further is decoded.
    pub(crate) fn set_fault(&mut self, reason: String) {
        self.done = true;
        if self.fault.is_none() {
            self.fault = Some(reason);
        }
    }

    /// How completely this log was read, for the trace-integrity block.
    ///
    /// Only meaningful once the producer has finished: until then a log with no
    /// `Done` yet is simply a log still being written.
    pub(crate) fn status(&self) -> crate::common::HartTraceStatus {
        crate::common::HartTraceStatus {
            hart_id: self.hart_id,
            bytes_read: self.bytes_read as u64,
            batches: self.batches_decoded,
            complete: self.saw_done,
            partial_tail_bytes: self.partial_tail_bytes() as u64,
        }
    }

    /// Nothing more can be handed over right now: either permanently finished,
    /// or the file has been read to EOF. A caller that knows the producer is
    /// finished can treat this as terminal.
    pub(crate) fn spent(&self) -> bool {
        self.done || self.at_eof
    }

    /// Bytes of an incomplete trailing record. Non-zero once the producer has
    /// finished means the log was truncated mid-write.
    pub(crate) fn partial_tail_bytes(&self) -> usize {
        if self.done {
            0
        } else {
            self.buffered()
        }
    }

    pub(crate) fn bytes_read(&self) -> usize {
        self.bytes_read
    }

    /// Undecoded bytes currently buffered.
    fn buffered(&self) -> usize {
        self.buf.len() - self.cursor
    }

    /// How many buffered bytes the *head* record needs before
    /// [`decode_next_record`] can yield it: `RECORD_HEADER_BYTES +
    /// payload_len` once its length prefix has arrived, otherwise just the
    /// framing bytes.
    ///
    /// This is why the framing puts the COMPRESSED length up front — the reader
    /// can size a record before inflating any of it. Zero before the file's
    /// header has been consumed, when `cursor` does not point at a record.
    fn head_record_need(&self) -> usize {
        if !self.header_ok {
            return 0;
        }
        let rest = &self.buf[self.cursor.min(self.buf.len())..];
        match rest.first_chunk::<LEN_PREFIX_BYTES>() {
            Some(prefix) => RECORD_HEADER_BYTES + u32::from_le_bytes(*prefix) as usize,
            None => RECORD_HEADER_BYTES,
        }
    }

    /// Read newly-appended bytes (subject to the budget and low-water mark),
    /// then decode up to `max_batches` complete records, appending one
    /// [`TraceData::Trace`] per `Batch` to `out`.
    pub(crate) fn pump(
        &mut self,
        out: &mut Vec<TraceData>,
        message_id: &mut u32,
        max_batches: usize,
    ) -> PumpStat {
        if self.done {
            return PumpStat {
                batches: 0,
                more_available: false,
            };
        }

        // Always attempt a read, so `at_eof` reflects the file as it is NOW.
        // `fill` itself stops early once the buffer is topped up, which is what
        // paces reads against decode. (Deciding not to read at all based on the
        // buffer level would leave `at_eof` stale, and a stale "not at EOF" is
        // an endless poll once the producer has finished.)
        self.fill();

        if !self.header_ok {
            match check_record_log_header(&self.buf) {
                // Not enough bytes yet. If the producer has finished and the
                // file is this short, the caller's producer check ends the run.
                Ok(None) => {
                    return PumpStat {
                        batches: 0,
                        more_available: !self.at_eof,
                    }
                }
                Ok(Some(hdr)) => {
                    self.cursor = RECORD_LOG_HEADER_BYTES;
                    self.run_id = hdr.run_id;
                    self.header_ok = true;
                }
                Err(e) => {
                    self.fault = Some(e);
                    self.done = true;
                    return PumpStat {
                        batches: 0,
                        more_available: false,
                    };
                }
            }
        }

        let mut batches = 0usize;
        let mut hit_cap = false;
        while !self.done {
            if batches >= max_batches {
                hit_cap = true;
                break;
            }
            match decode_next_record(&mut self.decoder, &self.buf, &mut self.cursor) {
                Some(Ok(StreamRecord::Batch { trace_data, extras })) => {
                    out.push(TraceData::Trace {
                        trace_data,
                        message_id: *message_id,
                        roi_markers: extras.roi_markers,
                        hart_id: self.hart_id,
                    });
                    *message_id += 1;
                    batches += 1;
                    self.batches_decoded += 1;
                }
                Some(Ok(StreamRecord::Done)) => {
                    log::debug!("hart {} record log reached Done", self.hart_id);
                    self.saw_done = true;
                    self.done = true;
                }
                Some(Err(e)) => {
                    self.fault = Some(e);
                    self.done = true;
                }
                // Torn tail: wait for more bytes. Whether any are coming is the
                // producer's business, not ours -- see the `done` field.
                None => break,
            }
        }

        // Drop the consumed prefix, keeping any trailing incomplete record.
        if self.cursor > 0 {
            self.buf.drain(0..self.cursor);
            self.cursor = 0;
        }

        PumpStat {
            batches,
            more_available: !self.done && (hit_cap || !self.at_eof),
        }
    }

    /// Append newly-written bytes to the buffer, recording whether EOF was
    /// reached. Stops at [`PUMP_READ_BUDGET`] bytes, or as soon as the buffer is
    /// back above [`BUFFER_LOW_WATER`] — so the reader keeps pace with decode
    /// instead of racing ahead of it.
    ///
    /// The low-water mark is a pacing floor, never a reason to stall: a record
    /// bigger than it must still be read to completion, one `PUMP_READ_BUDGET`
    /// pass at a time. Stopping at the mark with an incomplete head record
    /// livelocks — `decode_next_record` yields nothing until the record is whole,
    /// so the next `fill` would see a buffer already past the mark and read
    /// nothing, forever. Hence [`head_record_need`](Self::head_record_need).
    fn fill(&mut self) {
        let mut chunk = [0u8; 64 * 1024];
        let mut read_this_pass = 0usize;
        self.at_eof = false;
        while read_this_pass < PUMP_READ_BUDGET
            && self.buffered() < BUFFER_LOW_WATER.max(self.head_record_need())
        {
            match self.file.read(&mut chunk) {
                Ok(0) => {
                    self.at_eof = true;
                    break;
                }
                Ok(n) => {
                    read_this_pass += n;
                    self.bytes_read += n;
                    self.buf.extend_from_slice(&chunk[..n]);
                }
                Err(e) => {
                    self.fault = Some(format!("read failed after {} bytes: {e}", self.bytes_read));
                    self.done = true;
                    break;
                }
            }
        }
    }
}

/// Parse the vCPU index `N` from an `turbo_trace_vcpu<N>.bin` path, or `None` for any
/// other filename shape.
pub(crate) fn hart_id_from_path(path: &Path) -> Option<u32> {
    let stem = path.file_stem()?.to_str()?;
    if path.extension()?.to_str()? != "bin" {
        return None;
    }
    stem.strip_prefix("turbo_trace_vcpu")?.parse::<u32>().ok()
}

#[cfg(test)]
mod tests {
    use super::*;
    use turbo_tracer_shared::{BatchExtrasRef, RecordLogWriter};

    /// The run id every log in these tests claims.
    fn test_run_id() -> Option<RunId> {
        Some(RunId::parse(&format!("run_{:016x}", 0x105)).expect("built in the one shape"))
    }

    /// Build a record log with the real writer, so these tests exercise the
    /// actual on-disk framing rather than a hand-rolled imitation of it.
    fn write_log(path: &Path, batches: usize, with_done: bool) {
        let mut w = RecordLogWriter::create(path, test_run_id()).unwrap();
        for i in 0..batches {
            w.append_batch(&[i as u8; 8], BatchExtrasRef::EMPTY)
                .unwrap();
        }
        if with_done {
            w.finish().unwrap();
        } else {
            w.flush().unwrap();
        }
    }

    /// The framed bytes of a `batches`-record log, for tests that need to write
    /// a log in pieces (torn/truncated tails).
    ///
    /// `name` must be unique per caller: tests run in parallel and `tmp_dir`
    /// clears the directory it returns, so a shared scratch name lets one test
    /// delete another's file mid-read.
    fn log_bytes(name: &str, batches: usize, with_done: bool) -> Vec<u8> {
        let dir = tmp_dir(&format!("scratch_{name}"));
        let path = dir.join("scratch.bin");
        write_log(&path, batches, with_done);
        std::fs::read(&path).unwrap()
    }

    fn tmp_dir(name: &str) -> std::path::PathBuf {
        let d = std::env::temp_dir().join(format!("turbo_record_log_{name}"));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    #[test]
    fn parses_hart_id_from_filename() {
        assert_eq!(
            hart_id_from_path(Path::new("/x/turbo_trace_vcpu0.bin")),
            Some(0),
            "vcpu0 log"
        );
        assert_eq!(
            hart_id_from_path(Path::new("turbo_trace_vcpu7.bin")),
            Some(7)
        );
        assert_eq!(hart_id_from_path(Path::new("some_other.bin")), None);
        assert_eq!(hart_id_from_path(Path::new("turbo_trace_vcpu0.txt")), None);
    }

    #[test]
    fn decodes_all_batches_and_stops_at_done() {
        let dir = tmp_dir("all_batches");
        let path = dir.join("turbo_trace_vcpu0.bin");
        write_log(&path, 5, true);

        let mut r = HartLogReader::open(0, &path).unwrap();
        let mut out = Vec::new();
        let mut mid = 0;
        let stat = r.pump(&mut out, &mut mid, 64);

        assert_eq!(stat.batches, 5);
        assert!(!stat.more_available, "Done seen, so nothing more");
        assert!(r.done());
        assert_eq!(out.len(), 5);
        assert_eq!(mid, 5);
    }

    /// The batch cap is what keeps `pending` bounded on a finished multi-GB log.
    #[test]
    fn batch_cap_bounds_one_pump_and_resumes() {
        let dir = tmp_dir("batch_cap");
        let path = dir.join("turbo_trace_vcpu0.bin");
        write_log(&path, 10, true);

        let mut r = HartLogReader::open(0, &path).unwrap();
        let mut out = Vec::new();
        let mut mid = 0;

        let stat = r.pump(&mut out, &mut mid, 3);
        assert_eq!(stat.batches, 3, "capped at 3");
        assert!(stat.more_available, "more records still buffered");
        assert!(!r.done());

        // Remaining batches arrive across later pumps, message_id continuing.
        let mut total = 3;
        while !r.done() {
            total += r.pump(&mut out, &mut mid, 3).batches;
        }
        assert_eq!(total, 10);
        assert_eq!(out.len(), 10);
        assert_eq!(mid, 10);
    }

    /// A live log grows between pumps; a record split across the boundary must
    /// be buffered, not dropped or double-counted.
    /// A single record whose COMPRESSED size exceeds `BUFFER_LOW_WATER` must
    /// still decode.
    ///
    /// `fill` reads only while `buffered() < BUFFER_LOW_WATER`, so once a
    /// partially-read record has topped the buffer past the low-water mark it
    /// stops reading entirely -- and `decode_next_record` needs the whole record
    /// before it will yield anything. That is a livelock, not a slow path:
    /// `pump` returns `more_available` forever while reading zero new bytes.
    ///
    /// The producer's flush window is what decides record size, so this is the
    /// hard ceiling on `flush_interval`.
    #[test]
    fn a_record_larger_than_the_low_water_mark_still_decodes() {
        let dir = tmp_dir("big_record");
        let path = dir.join("turbo_trace_vcpu0.bin");

        // Incompressible payload, so the compressed record is comfortably over
        // BUFFER_LOW_WATER rather than deflating back under it.
        let mut lcg: u64 = 0x2545_F491_4F6C_DD1D;
        let payload: Vec<u8> = (0..4 * BUFFER_LOW_WATER)
            .map(|_| {
                lcg = lcg
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                (lcg >> 33) as u8
            })
            .collect();

        let mut w = RecordLogWriter::create(&path, test_run_id()).unwrap();
        w.append_batch(&payload, BatchExtrasRef::EMPTY).unwrap();
        w.finish().unwrap();
        let on_disk = std::fs::metadata(&path).unwrap().len();
        assert!(
            on_disk as usize > BUFFER_LOW_WATER,
            "test needs a record bigger than the low-water mark, got {on_disk} B"
        );

        let mut r = HartLogReader::open(0, &path).unwrap();
        let mut out = Vec::new();
        let mut id = 0u32;
        // Generous but finite: a correct reader needs a handful of passes.
        for _ in 0..10_000 {
            r.pump(&mut out, &mut id, 64);
            if !out.is_empty() || r.done() {
                break;
            }
        }
        assert_eq!(out.len(), 1, "record never decoded ({on_disk} B on disk)");
    }

    #[test]
    fn torn_tail_is_buffered_until_the_rest_arrives() {
        let dir = tmp_dir("torn_tail");
        let path = dir.join("turbo_trace_vcpu0.bin");

        // Build a complete 2-batch log, then write it in two pieces, cutting
        // through the middle of the second record.
        let full = log_bytes("torn_tail", 2, false);
        let cut = full.len() - 4;
        std::fs::write(&path, &full[..cut]).unwrap();

        let mut r = HartLogReader::open(0, &path).unwrap();
        let mut out = Vec::new();
        let mut mid = 0;
        assert_eq!(r.pump(&mut out, &mut mid, 64).batches, 1, "only batch 0");
        assert!(!r.done(), "torn tail is not the end while more may arrive");

        // Append the rest; the buffered prefix must combine with it.
        use std::io::Write as _;
        let mut f = std::fs::OpenOptions::new()
            .append(true)
            .open(&path)
            .unwrap();
        f.write_all(&full[cut..]).unwrap();
        drop(f);

        assert_eq!(r.pump(&mut out, &mut mid, 64).batches, 1, "batch 1 now");
        assert_eq!(out.len(), 2);
        // No Done record was written, so the reader stays open.
        assert!(!r.done());
    }

    /// A truncated log (no `Done`, partial trailing record) must stop claiming
    /// data is available, so the source can terminate once the producer is done.
    /// It must NOT mark itself permanently done: while a producer is still
    /// running, that same state is just "paused between writes".
    #[test]
    fn truncated_tail_at_eof_reports_spent_not_done() {
        let dir = tmp_dir("truncated");
        let path = dir.join("turbo_trace_vcpu0.bin");
        let full = log_bytes("truncated", 2, false);
        std::fs::write(&path, &full[..full.len() - 3]).unwrap();

        let mut r = HartLogReader::open(0, &path).unwrap();
        let mut out = Vec::new();
        let mut mid = 0;
        let stat = r.pump(&mut out, &mut mid, 64);
        assert_eq!(stat.batches, 1, "the one complete record");
        assert!(!stat.more_available, "nothing more to hand over");
        assert!(r.spent(), "at EOF: terminal once the producer is finished");
        assert!(!r.done(), "not permanently done -- a live log could grow");
        assert!(r.partial_tail_bytes() > 0, "truncated tail is visible");
    }

    #[test]
    fn bad_header_marks_done() {
        let dir = tmp_dir("bad_header");
        let path = dir.join("turbo_trace_vcpu0.bin");
        std::fs::write(&path, b"not-a-record-log-at-all").unwrap();

        let mut r = HartLogReader::open(0, &path).unwrap();
        let mut out = Vec::new();
        let mut mid = 0;
        let stat = r.pump(&mut out, &mut mid, 64);
        assert_eq!(stat.batches, 0);
        assert!(!stat.more_available);
        assert!(r.done());
        assert!(out.is_empty());
    }
}
