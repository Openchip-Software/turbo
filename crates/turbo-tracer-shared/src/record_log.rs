//! Record-log wire format: the framed, compressed on-disk stream the QEMU
//! plugin appends per vCPU and every consumer reads back.
//!
//! Split out of `lib.rs` verbatim; this is the format only (framing, the
//! `StreamRecord` payloads, the writer, and the incremental decoder). The
//! *policy* around reading one - buffering, pacing, when a torn tail means
//! "truncated" rather than "producer paused" - belongs to the consumer, and
//! lives in turbo's `source::record_log`.

use serde::{Deserialize, Serialize};
use std::path::Path;

use crate::region_marker::RoiRtMarker;
#[cfg(test)]
use crate::region_marker::RoiRtMarkerKind;
use crate::run_id::{RunId, RUN_ID_BYTES};

// ---------------------------------------------------------------------------
// Batch side-channels
// ---------------------------------------------------------------------------

/// The side-channel payloads that accompany a batch's trace bytes on the wire.
///
/// A batch is overwhelmingly `trace_data`; everything else riding along with it
/// is small, optional, and independent of the trace encoding. Grouping those
/// payloads here rather than listing them positionally on
/// [`StreamRecord::Batch`], [`RecordLogWriter::append_batch`] and
/// [`BorrowedStreamRecord::Batch`] means a new side-channel is one field added
/// in one place, instead of a signature change rippling through every producer
/// and consumer call site.
///
/// Nesting costs nothing on the wire: postcard encodes a struct as the
/// concatenation of its fields with no tag or length, so a flattened
/// `Batch { a, b, c }` and a `Batch { extras: { a, b, c } }` serialize to
/// identical bytes.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct BatchExtras {
    /// ROI R-type markers captured on this hart since the previous batch.
    pub roi_markers: Vec<RoiRtMarker>,
}

/// Borrowed mirror of [`BatchExtras`] for the hot producer path, which already
/// holds each payload in a slice and must not clone them just to serialize.
///
/// Field ORDER and shape MUST match [`BatchExtras`] exactly, or the two encode
/// to different bytes. `append_batch_matches_owned_append` is the regression
/// test for that.
#[derive(Debug, Clone, Copy, Serialize)]
pub struct BatchExtrasRef<'a> {
    /// See [`BatchExtras::roi_markers`].
    pub roi_markers: &'a [RoiRtMarker],
}

impl<'a> BatchExtrasRef<'a> {
    /// An empty set of side-channels, for producers and tests that ship none.
    pub const EMPTY: Self = Self { roi_markers: &[] };
}

impl Default for BatchExtrasRef<'_> {
    fn default() -> Self {
        Self::EMPTY
    }
}

// ---------------------------------------------------------------------------
// Record-log wire format (producer appends to a growing on-disk file)
// ---------------------------------------------------------------------------

/// The first 8 bytes of every record-log file: "this is a turbo record log".
/// It answers only that question -- what *layout* the file uses is
/// [`RECORD_LOG_FORMAT_VERSION`], which sits right behind it -- so the magic
/// does not change when the layout does, and a file whose layout this build
/// cannot read is still recognisably one of ours.
pub const TRBOLOG_MAGIC: [u8; 8] = *b"TRBOLOG1";

/// Format version, written into the header behind the magic. Bump on ANY
/// incompatible change to [`StreamRecord`] or the framing (new/removed
/// variants, changed field types, reordered enum variants).
///
/// This is the check that makes drift loud, and it covers files older than
/// itself too: before this field existed, the four bytes it occupies were the
/// first record's compressed length. A pre-version file therefore arrives here
/// as an absurd "version" -- a few hundred, or whatever that record's length
/// was -- and is refused by the check below rather than mis-parsed. Which is
/// why the magic did not need a new generation to introduce it.
pub const RECORD_LOG_FORMAT_VERSION: u32 = 3;

/// Width of the little-endian `u32` length prefix that introduces every framed
/// record. Named because the writer reserves exactly this much room ahead of
/// the compressed payload and patches it afterwards.
pub const LEN_PREFIX_BYTES: usize = 4;

/// Width of the little-endian `u32` CRC-32 that follows the length prefix,
/// computed over the record's COMPRESSED payload bytes.
///
/// Over the compressed bytes, not the plaintext, for two reasons: the reader
/// can then reject a damaged record *before* handing it to the inflater (so a
/// bad record costs 4 bytes of arithmetic, not an inflate), and the writer
/// computes it on bytes it already has in hand -- the framed buffer it is about
/// to hand to `write_all` -- rather than making a second pass over the
/// pre-compression payload.
pub const CRC_BYTES: usize = 4;

/// Bytes of framing ahead of every record's compressed payload:
/// `[u32 LE len][u32 LE crc32]`.
pub const RECORD_HEADER_BYTES: usize = LEN_PREFIX_BYTES + CRC_BYTES;

/// CRC-32 of a record's compressed payload, as the framing stores it.
///
/// One function so the writer and the reader cannot disagree about which bytes
/// are covered, and so the choice of implementation is in one place: this is
/// zlib's CRC-32 via `flate2`, which the workspace already depends on for
/// deflate and which dispatches to a SIMD implementation at runtime.
pub fn payload_crc32(payload: &[u8]) -> u32 {
    let mut crc = flate2::Crc::new();
    crc.update(payload);
    crc.sum()
}

/// Every record-log file opens with exactly these bytes:
/// `[8-byte magic][u32 LE version][RUN_ID_BYTES run id]`.
///
/// A CONSTANT, which is the point: the run id is fixed-width (see [`RunId`]),
/// so the first record is always at this offset, a live-tailing reader needs
/// one length comparison to know whether the header has landed, and neither
/// side carries a length field that a damaged file could lie about.
pub const RECORD_LOG_HEADER_BYTES: usize = TRBOLOG_MAGIC.len() + 4 + RUN_ID_BYTES;

/// A parsed record-log file header; see [`RECORD_LOG_HEADER_BYTES`] for the
/// layout.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RecordLogHeader {
    /// [`RECORD_LOG_FORMAT_VERSION`] as the producer wrote it.
    pub version: u32,
    /// The run this log belongs to, so a consumer can refuse logs that are not
    /// from the run its metadata describes. `None` when the producer was given
    /// no run id -- a plugin invoked by hand rather than by `turbo record` --
    /// which is written as zeroed id bytes and reads back as "no run to check
    /// against", not as an id that matches nothing.
    pub run_id: Option<RunId>,
}

/// One appended record in a per-hart record-log file.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum StreamRecord {
    /// One flush window's worth of trace data for a single hart. The hart id is
    /// not carried in the record itself: it is encoded in the per-hart
    /// filename. One Batch == one downstream TraceData::Trace message.
    Batch {
        trace_data: Vec<u8>,
        /// Side-channel payloads for this window; see [`BatchExtras`].
        extras: BatchExtras,
    },
    /// Terminal sentinel appended exactly once when the producer finishes this
    /// hart's stream. Lets a *tailing* reader distinguish "producer paused
    /// between flushes" from "producer done" without inferring it from EOF.
    Done,
}

/// Default deflate level for record payloads.
pub const DEFAULT_COMPRESSION_LEVEL: u32 = 6;

/// Compress `input` and APPEND it to `out` as a raw deflate stream.
///
/// Appends rather than clears so a caller can put the frame's length prefix in
/// `out` first and have the compressed payload land directly behind it -- which
/// is what lets [`RecordLogWriter::write_framed`] hand one buffer to `write_all`
/// with no second copy. Callers wanting the old "just the stream" behaviour
/// clear `out` themselves.
///
/// `compressor` and `out` are both owned by the caller across calls so the hot
/// per-flush path neither allocates nor rebuilds deflate state. `compress_vec`
/// only ever writes into the vector's *spare capacity*, so this grows `out`
/// itself whenever the stream has more to emit than there is room for.
fn deflate_into(
    compressor: &mut flate2::Compress,
    input: &[u8],
    out: &mut Vec<u8>,
) -> std::io::Result<()> {
    use flate2::{FlushCompress, Status};

    compressor.reset();
    // A first guess big enough that the common case finishes in one call. Trace
    // payloads compress ~76x, so a quarter of the input is generous.
    out.reserve((input.len() / 4).max(1024));

    loop {
        let consumed = compressor.total_in() as usize;
        let produced_before = compressor.total_out();
        if out.len() == out.capacity() {
            out.reserve(out.capacity().max(1024));
        }
        let status = compressor
            .compress_vec(&input[consumed..], out, FlushCompress::Finish)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        match status {
            Status::StreamEnd => return Ok(()),
            // More to emit. If the call made no progress at all it can only be
            // for want of output room, so force growth rather than spin.
            Status::Ok | Status::BufError => {
                if compressor.total_out() == produced_before {
                    out.reserve(out.capacity().max(1024));
                }
            }
        }
    }
}

/// A `&[u8]` that serializes through `serialize_bytes` rather than as a
/// sequence.
///
/// serde's blanket `impl Serialize for [T]` is a *sequence*, so a derived
/// `trace_data: &[u8]` field is pushed into postcard's output one
/// `serialize_u8` -> `Vec::push` at a time -- one bounds-checked single-byte
/// push per trace byte, which measured as ~4% of every QEMU vCPU thread's CPU
/// on two 4-hart workloads. `serialize_bytes` is a length varint plus one
/// `extend_from_slice`.
///
/// The wire bytes are IDENTICAL either way -- postcard encodes both a
/// sequence-of-`u8` and a byte string as `varint(len)` followed by the raw
/// bytes -- so this is not a format change and needs no version bump.
/// `append_batch_matches_owned_append` is the regression test for that, and it
/// is why the owned `StreamRecord::Batch` can keep its plain `Vec<u8>`.
struct RawBytes<'a>(&'a [u8]);

impl Serialize for RawBytes<'_> {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_bytes(self.0)
    }
}

/// Append-only writer for a per-hart record-log file. Owns the underlying file
/// and writes the magic header on creation; every subsequent record is framed
/// as `[u32 LE length][deflate-compressed postcard payload]`.
///
/// # Compression
///
/// Payloads are compressed individually rather than the file being wrapped in
/// one gzip stream, because the log is *live-tailed* while the producer runs
/// (see `HartLogReader` in turbo). Per-record compression keeps the three
/// properties that path depends on: the plaintext `[u32 len]` prefix still
/// decides whether a record is complete (so a torn tail stays recoverable and
/// is never handed to the inflater), reads stay bounded, and any record can be
/// decoded without inflating the ones before it.
///
/// It costs nothing in ratio: a BBT stream's redundancy is local — hot loops
/// replay the same block IDs inside a single flush window — so per-batch
/// deflate measures *better* than whole-file `gzip -1` on the same trace.
pub struct RecordLogWriter {
    file: std::fs::File,
    /// Reused across records so the per-flush path does not rebuild deflate
    /// state or allocate.
    compressor: flate2::Compress,
    /// Reused output buffer holding one COMPLETE frame -- the 4-byte length
    /// prefix followed by the compressed payload, which is exactly the byte
    /// range handed to `write_all`. Reaches steady-state capacity within the
    /// first few records.
    framed: Vec<u8>,
    /// Total pre-compression payload bytes handed to [`write_framed`]. Paired
    /// with [`bytes_written`](Self::bytes_written) this gives the achieved
    /// ratio without re-reading the file.
    logical_bytes: u64,
    /// Running total of bytes handed to `write_all`, header included. Lets a
    /// producer attribute *on-disk* growth to a flush window (framing and
    /// postcard overhead included) by diffing this across an append, rather
    /// than guessing from the payload length it passed in.
    bytes_written: u64,
}

impl RecordLogWriter {
    /// Create (and TRUNCATE) the file, then write the header. Each run starts
    /// fresh: stale files from a previous run carry addresses relocated by a
    /// different ASLR layout, so appending to them would interleave
    /// incompatible data. Truncation guarantees the header is the file's prefix.
    ///
    /// `run_id` identifies the run this log belongs to and is stamped into
    /// the header, so a consumer can tell that a log, the
    /// `turbo_metadata.json` beside it and its sibling logs all came from the
    /// same execution. Pass `None` only from a producer that genuinely has no
    /// run of its own.
    pub fn create(path: &std::path::Path, run_id: Option<RunId>) -> std::io::Result<Self> {
        Self::create_with_level(path, DEFAULT_COMPRESSION_LEVEL, run_id)
    }

    /// As [`create`](Self::create), with an explicit deflate level (0-9).
    ///
    /// Level 0 is the "compression off" setting for A/B measurement: deflate
    /// emits *stored* blocks, so the file grows to roughly its uncompressed
    /// size but is still a valid stream that the ordinary reader decodes with
    /// no special case. That is why the format needs no per-record codec tag.
    pub fn create_with_level(
        path: &std::path::Path,
        level: u32,
        run_id: Option<RunId>,
    ) -> std::io::Result<Self> {
        use std::io::Write;
        let mut file = std::fs::OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .open(path)?;
        // One `write_all`: the header must never be observable in pieces by a
        // consumer that opens the file the instant it appears (the live path
        // does exactly that).
        let header = encode_record_log_header(run_id);
        file.write_all(&header)?;
        Ok(Self {
            file,
            // Raw deflate (no zlib/gzip wrapper): the framing already carries
            // the length, and a per-record header would be pure overhead.
            compressor: flate2::Compress::new(flate2::Compression::new(level.min(9)), false),
            framed: Vec::new(),
            logical_bytes: 0,
            bytes_written: header.len() as u64,
        })
    }

    /// Total bytes written to this file so far (header + every framed record),
    /// i.e. the size on disk, post-compression.
    pub fn bytes_written(&self) -> u64 {
        self.bytes_written
    }

    /// Total payload bytes before compression. `logical_bytes / bytes_written`
    /// is the achieved ratio.
    pub fn logical_bytes(&self) -> u64 {
        self.logical_bytes
    }

    /// Serialize `record` with postcard and append it as `[u32 LE len][payload]`.
    /// The framed bytes are concatenated and written with a single `write_all` so
    /// a concurrent tailing reader can never observe a length prefix torn from its
    /// payload. postcard errors are surfaced as `InvalidData` io errors.
    pub fn append(&mut self, record: &StreamRecord) -> std::io::Result<()> {
        let payload = postcard::to_allocvec(record)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        self.write_framed(&payload)
    }

    /// Borrowed fast path for the hot per-flush call: serialize a
    /// [`StreamRecord::Batch`] WITHOUT cloning `trace_data` (which can be large)
    /// or the marker/access vectors. A private `#[derive(Serialize)]` enum mirror
    /// with borrowed slices reproduces the exact variant order/shape of
    /// `StreamRecord`, so postcard emits byte-identical output to constructing an
    /// owned `Batch` and calling [`append`](Self::append).
    pub fn append_batch(
        &mut self,
        trace_data: &[u8],
        extras: BatchExtrasRef<'_>,
    ) -> std::io::Result<()> {
        // Mirror of StreamRecord with borrowed slices. Variant ORDER and field
        // shape MUST match StreamRecord exactly or the postcard encoding diverges.
        #[derive(Serialize)]
        enum BorrowedRecord<'a> {
            Batch {
                trace_data: RawBytes<'a>,
                extras: BatchExtrasRef<'a>,
            },
            #[allow(dead_code)]
            Done,
        }
        let borrowed = BorrowedRecord::Batch {
            trace_data: RawBytes(trace_data),
            extras,
        };
        let payload = postcard::to_allocvec(&borrowed)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        self.write_framed(&payload)
    }

    /// Compress a single already-serialized payload and write it with its
    /// length prefix in one `write_all`, so the `[len][payload]` framing is
    /// never torn on disk.
    ///
    /// The length prefix describes the COMPRESSED payload, which is what makes
    /// a live-tailing reader able to tell "record not fully written yet" from
    /// "corrupt" without ever feeding a partial stream to the inflater.
    fn write_framed(&mut self, payload: &[u8]) -> std::io::Result<()> {
        use std::io::Write;
        // One buffer, then one `write_all`: a reader tailing this file must
        // never observe a length prefix without its payload.
        //
        // The compressed bytes are deflated straight in behind a placeholder
        // prefix, which is then overwritten with the length now that it is
        // known. The obvious alternative -- compress into a scratch buffer,
        // then build a second `[len][payload]` vector -- costs a malloc/free
        // and a full copy of the payload on every flush window, on the vCPU's
        // own writer thread.
        let mut framed = std::mem::take(&mut self.framed);
        framed.clear();
        framed.extend_from_slice(&[0u8; RECORD_HEADER_BYTES]);
        let result = deflate_into(&mut self.compressor, payload, &mut framed);
        self.framed = framed;
        result?;

        let len = (self.framed.len() - RECORD_HEADER_BYTES) as u32;
        // Over the compressed bytes that are already sitting in this buffer, so
        // the checksum costs one pass over the SMALL side of the compression
        // ratio (payloads deflate ~76x) and no extra copy. Measured against
        // the deflate it rides behind, this is in the noise.
        let crc = payload_crc32(&self.framed[RECORD_HEADER_BYTES..]);
        self.framed[..LEN_PREFIX_BYTES].copy_from_slice(&len.to_le_bytes());
        self.framed[LEN_PREFIX_BYTES..RECORD_HEADER_BYTES].copy_from_slice(&crc.to_le_bytes());
        self.file.write_all(&self.framed)?;
        self.bytes_written += self.framed.len() as u64;
        self.logical_bytes += payload.len() as u64;
        Ok(())
    }

    /// Flush the OS-level buffer so a tailing reader sees appended records
    /// promptly. Tradeoff: calling this after every append minimizes reader
    /// latency but costs a syscall per flush window; batching flushes improves
    /// producer throughput at the cost of staleness on the reader side. Callers
    /// pick the cadence; this does NOT `fsync` (see [`finish`](Self::finish)).
    pub fn flush(&mut self) -> std::io::Result<()> {
        use std::io::Write;
        self.file.flush()
    }

    /// Append the terminal [`StreamRecord::Done`] sentinel, then flush and
    /// `sync_all` so the completed stream is durably on disk before returning.
    pub fn finish(&mut self) -> std::io::Result<()> {
        use std::io::Write;
        self.append(&StreamRecord::Done)?;
        self.file.flush()?;
        self.file.sync_all()
    }
}

/// The read-side mirror of [`StreamRecord`], with `trace_data` BORROWED out of
/// the caller's buffer instead of copied into a `Vec`.
///
/// This exists because deserializing the owned `Vec<u8>` is not a `memcpy`:
/// serde's `Vec` visitor pulls the payload one element at a time
/// (`next_element::<u8>()` plus a capacity check per byte), so a 15.7 MB trace
/// costs 15.7 M iterations of that loop. Asking postcard for a `&[u8]` instead
/// routes through `deserialize_bytes`, which just hands back a subslice.
///
/// The postcard encoding is identical either way -- a `Vec<u8>` and a `&[u8]`
/// both go out as a varint length followed by the raw bytes -- so this reads
/// records written by [`RecordLogWriter::append`] and
/// [`RecordLogWriter::append_batch`] unchanged. Variant ORDER and field shape
/// MUST match `StreamRecord` exactly, for the same reason the writer's
/// `BorrowedRecord` mirror must.
#[derive(Debug, Clone, Deserialize)]
pub enum BorrowedStreamRecord<'a> {
    /// See [`StreamRecord::Batch`].
    Batch {
        #[serde(borrow)]
        trace_data: &'a [u8],
        /// See [`StreamRecord::Batch::extras`]. Stays owned: it holds
        /// non-trivial structs (`String`s in the markers) that cannot be
        /// borrowed from the wire, and is orders of magnitude smaller than the
        /// trace payload.
        extras: BatchExtras,
    },
    /// See [`StreamRecord::Done`].
    Done,
}

impl BorrowedStreamRecord<'_> {
    /// Copy the borrowed payload into an owned [`StreamRecord`]. One `memcpy`
    /// for the trace bytes, versus the per-byte visitor loop that deserializing
    /// `StreamRecord` directly would run.
    pub fn into_owned(self) -> StreamRecord {
        match self {
            Self::Batch { trace_data, extras } => StreamRecord::Batch {
                trace_data: trace_data.to_vec(),
                extras,
            },
            Self::Done => StreamRecord::Done,
        }
    }
}

/// Decode the next complete `[u32 len][postcard payload]` record from
/// `buf[*cursor..]`, borrowing the trace payload out of `buf`.
///
/// Framing behaviour is identical to [`decode_next_record`]; see it for the
/// `None`-vs-`Some(Err(..))` contract. Prefer this when the record is consumed
/// before `buf` is reused, and [`decode_next_record`] when an owned record has
/// to outlive the buffer.
pub fn decode_next_record_borrowed<'a>(
    decoder: &'a mut RecordLogDecoder,
    buf: &[u8],
    cursor: &mut usize,
) -> Option<Result<BorrowedStreamRecord<'a>, String>> {
    let start = *cursor;
    // Need the whole `[len][crc]` framing before anything can be sized.
    if buf.len() < start + RECORD_HEADER_BYTES {
        return None;
    }
    let len = u32::from_le_bytes(
        buf[start..start + LEN_PREFIX_BYTES]
            .try_into()
            .expect("LEN_PREFIX_BYTES bytes are in hand"),
    ) as usize;
    let stored_crc = u32::from_le_bytes(
        buf[start + LEN_PREFIX_BYTES..start + RECORD_HEADER_BYTES]
            .try_into()
            .expect("CRC_BYTES bytes are in hand"),
    );
    let payload_start = start + RECORD_HEADER_BYTES;
    let payload_end = payload_start + len;
    // Full payload not yet present (torn tail of a growing file). Checked
    // BEFORE inflating, so a partial record is never mistaken for corruption.
    if buf.len() < payload_end {
        return None;
    }
    *cursor = payload_end;
    let payload = &buf[payload_start..payload_end];

    // Checked before inflating: a damaged record then costs one pass over the
    // compressed bytes instead of an inflate that either fails obscurely or --
    // the case this exists for -- succeeds and yields plausible garbage. Raw
    // deflate carries no checksum of its own, so without this the only
    // detection was "the bytes happened not to parse".
    let actual_crc = payload_crc32(payload);
    if actual_crc != stored_crc {
        return Some(Err(format!(
            "record log: the {len}-byte record at file offset {start} fails its \
             checksum (header says {stored_crc:#010x}, the bytes on disk are \
             {actual_crc:#010x}); this trace is damaged -- re-record it"
        )));
    }

    Some(decoder.inflate_and_parse(payload, start))
}

/// Decoder state for a record log: a reusable inflate context and the buffer
/// records are decompressed into.
///
/// This exists because [`BorrowedStreamRecord`] borrows its trace payload
/// rather than copying it (see that type's docs for why that matters), and
/// decompressed bytes need somewhere to live that outlives the call. Holding
/// the scratch here means one buffer per reader for the whole run, reaching
/// steady-state capacity within the first few records, instead of an
/// allocation per record.
pub struct RecordLogDecoder {
    inflate: flate2::Decompress,
    inflate_scratch: Vec<u8>,
}

impl Default for RecordLogDecoder {
    fn default() -> Self {
        Self::new()
    }
}

impl RecordLogDecoder {
    pub fn new() -> Self {
        Self {
            // Raw deflate, matching `RecordLogWriter`'s compressor.
            inflate: flate2::Decompress::new(false),
            inflate_scratch: Vec::new(),
        }
    }

    /// Uncompressed size of the most recently decoded record's payload. Paired
    /// with the record's framed size this gives the achieved ratio, which is
    /// what `turbo --inspect` reports.
    pub fn last_inflated_len(&self) -> usize {
        self.inflate_scratch.len()
    }

    /// Inflate one complete framed payload and deserialize it. `offset` is the
    /// record's byte position, used only to make errors locatable.
    fn inflate_and_parse(
        &mut self,
        payload: &[u8],
        offset: usize,
    ) -> Result<BorrowedStreamRecord<'_>, String> {
        use flate2::{FlushDecompress, Status};

        let inflate = &mut self.inflate;
        inflate.reset(false);
        self.inflate_scratch.clear();
        // Payloads compress ~76x, so guess generously rather than growing from
        // nothing; after the first few records the capacity is already right.
        self.inflate_scratch.reserve((payload.len() * 8).max(4096));

        loop {
            let consumed = inflate.total_in() as usize;
            let produced_before = inflate.total_out();
            if self.inflate_scratch.len() == self.inflate_scratch.capacity() {
                self.inflate_scratch
                    .reserve(self.inflate_scratch.capacity().max(4096));
            }
            let status = inflate
                .decompress_vec(
                    &payload[consumed..],
                    &mut self.inflate_scratch,
                    FlushDecompress::Finish,
                )
                .map_err(|e| decode_error(offset, payload.len(), &e.to_string()))?;
            match status {
                Status::StreamEnd => break,
                Status::Ok | Status::BufError => {
                    if inflate.total_out() == produced_before
                        && inflate.total_in() as usize == consumed
                    {
                        // No progress and no output written: the stream ended
                        // early rather than merely needing more room.
                        if self.inflate_scratch.len() < self.inflate_scratch.capacity() {
                            return Err(decode_error(
                                offset,
                                payload.len(),
                                "deflate stream ended before it was complete",
                            ));
                        }
                        self.inflate_scratch
                            .reserve(self.inflate_scratch.capacity().max(4096));
                    }
                }
            }
        }

        postcard::from_bytes(&self.inflate_scratch)
            .map_err(|e| decode_error(offset, payload.len(), &format!("postcard: {e}")))
    }
}

/// One message for every way a record can fail to decode. Points at the
/// overwhelmingly likely cause: this format carries no version tag (deliberately
/// — it is prototype-stage and old traces are worthless anyway, since their
/// addresses come from a different ASLR layout), so a producer/consumer mismatch
/// shows up here rather than at a header check.
fn decode_error(offset: usize, payload_len: usize, cause: &str) -> String {
    format!(
        "record log: could not decode the {payload_len}-byte record at file offset \
         {offset}: {cause}.\n\
         Record payloads are deflate-compressed postcard. This log carries no format \
         version, so the usual cause is a STALE TRACE -- one written by a producer \
         built before the current format. Re-run the workload to regenerate it \
         (`cargo b -r -p turbo-tracer` first, so QEMU loads the current plugin). \
         If the trace was just generated by this build, it is genuinely corrupt."
    )
}

/// Decode the next complete `[u32 len][postcard payload]` record from
/// `buf[*cursor..]`. On success, advance `*cursor` past it and return the
/// record. If fewer than 4 length bytes, or the full payload has not yet been
/// written (torn tail of a growing file), return `None` and leave `*cursor`
/// UNCHANGED so the caller retries after more bytes arrive. Returns
/// `Some(Err(..))` only when a length prefix is present and complete but the
/// payload fails to deserialize (real corruption / version mismatch).
///
/// Implemented on top of [`decode_next_record_borrowed`] so that even this
/// owned path copies the payload with one `memcpy` rather than serde's per-byte
/// `Vec` visitor.
pub fn decode_next_record(
    decoder: &mut RecordLogDecoder,
    buf: &[u8],
    cursor: &mut usize,
) -> Option<Result<StreamRecord, String>> {
    decode_next_record_borrowed(decoder, buf, cursor)
        .map(|r| r.map(BorrowedStreamRecord::into_owned))
}

/// Serialize the file header a [`RecordLogWriter`] writes; see
/// [`RECORD_LOG_HEADER_BYTES`] for the layout.
///
/// A producer with no run of its own passes `None`, which writes zeroed id
/// bytes: "no run id" has to occupy the field rather than shorten it, because
/// the whole point of a fixed-width header is that the first record's offset
/// does not depend on what is in it.
fn encode_record_log_header(run_id: Option<RunId>) -> [u8; RECORD_LOG_HEADER_BYTES] {
    let mut header = [0u8; RECORD_LOG_HEADER_BYTES];
    header[..TRBOLOG_MAGIC.len()].copy_from_slice(&TRBOLOG_MAGIC);
    header[TRBOLOG_MAGIC.len()..TRBOLOG_MAGIC.len() + 4]
        .copy_from_slice(&RECORD_LOG_FORMAT_VERSION.to_le_bytes());
    if let Some(id) = run_id {
        header[TRBOLOG_MAGIC.len() + 4..].copy_from_slice(id.as_bytes());
    }
    header
}

/// Parse and verify the file header. Returns `Ok(Some(header))` once the whole
/// header is present, `Ok(None)` while the buffer is still SHORTER than it
/// ("not enough bytes yet, retry" -- a tailing reader may open the file before
/// the producer's first write lands), and `Err` once there are enough bytes to
/// know the header is wrong.
///
/// The header is a constant width, so "not yet" is one length comparison and
/// there is no length field to sanity-check: bytes in the run-id field that are
/// not a run id read back as `None`, exactly like the zeroes a producer without
/// one writes. That is the right answer for both -- a consumer with nothing
/// trustworthy to compare against skips the staleness check rather than
/// declaring a file corrupt over a field that is only a hint.
pub fn check_record_log_header(buf: &[u8]) -> Result<Option<RecordLogHeader>, String> {
    if buf.len() < TRBOLOG_MAGIC.len() {
        return Ok(None);
    }
    let head = &buf[..TRBOLOG_MAGIC.len()];
    if head != TRBOLOG_MAGIC {
        let hex: String = head.iter().map(|b| format!("{b:02x} ")).collect();
        let ascii: String = head
            .iter()
            .map(|&b| if b.is_ascii_graphic() { b as char } else { '.' })
            .collect();
        return Err(format!(
            "bad record-log magic: expected {:?} (\"{}\"), got hex [{}] ascii \"{}\"",
            TRBOLOG_MAGIC,
            String::from_utf8_lossy(&TRBOLOG_MAGIC),
            hex.trim_end(),
            ascii
        ));
    }

    if buf.len() < RECORD_LOG_HEADER_BYTES {
        return Ok(None);
    }
    let version = u32::from_le_bytes(
        buf[TRBOLOG_MAGIC.len()..TRBOLOG_MAGIC.len() + 4]
            .try_into()
            .expect("4 bytes are in hand"),
    );
    if version != RECORD_LOG_FORMAT_VERSION {
        // Deliberately covers two cases in one message, because the user's fix
        // is the same for both: a genuinely different version, and a trace
        // written before the header carried a version at all -- where these
        // four bytes are the first record's length and so read as a large,
        // meaningless "version".
        return Err(format!(
            "record log format version {version}, but this build reads version \
             {RECORD_LOG_FORMAT_VERSION}; the trace was written by a different turbo \
             (a version far above {RECORD_LOG_FORMAT_VERSION} means it predates the \
             versioned header entirely) -- re-record it"
        ));
    }
    let id_bytes: [u8; RUN_ID_BYTES] = buf[TRBOLOG_MAGIC.len() + 4..RECORD_LOG_HEADER_BYTES]
        .try_into()
        .expect("RUN_ID_BYTES bytes are in hand");

    Ok(Some(RecordLogHeader {
        version,
        run_id: RunId::from_bytes(id_bytes),
    }))
}

/// Read a finished record-log file and return all `Batch` records in order.
///
/// Validates the magic header, then decodes records until a `Done` sentinel,
/// EOF, or a torn/corrupt tail (logged via the returned error only on a hard
/// I/O or header failure; a corrupt record mid-stream stops decoding and
/// returns what was decoded so far). `Done` records are not included in the
/// result. This is the shared entry point for consumers that want the whole
/// finished stream at once (offline `process`, direct PC-comparison tests),
/// as opposed to the incremental `decode_next_record` used by live tailing.
pub fn read_record_log(path: &Path) -> std::io::Result<Vec<StreamRecord>> {
    let buf = std::fs::read(path)?;
    let mut cursor = match check_record_log_header(&buf) {
        Ok(Some(_)) => RECORD_LOG_HEADER_BYTES,
        Ok(None) => {
            return Err(std::io::Error::new(
                std::io::ErrorKind::UnexpectedEof,
                format!("record-log file {path:?} too short for a header"),
            ))
        }
        Err(e) => return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, e)),
    };

    let mut records = Vec::new();
    let mut decoder = RecordLogDecoder::new();
    loop {
        match decode_next_record(&mut decoder, &buf, &mut cursor) {
            Some(Ok(StreamRecord::Done)) | None => break,
            Some(Ok(record)) => records.push(record),
            Some(Err(e)) => {
                eprintln!("record-log {path:?}: stopping at undecodable record: {e}");
                break;
            }
        }
    }
    Ok(records)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Frame a record exactly as [`RecordLogWriter`] does: deflate the postcard
    /// payload, then length-prefix the COMPRESSED bytes. Tests that hand-build
    /// framed bytes must go through this, or they would be asserting against a
    /// format the writer never produces.
    fn frame_record(record: &StreamRecord) -> Vec<u8> {
        let payload = postcard::to_allocvec(record).unwrap();
        let mut compressor =
            flate2::Compress::new(flate2::Compression::new(DEFAULT_COMPRESSION_LEVEL), false);
        let mut compressed = Vec::new();
        deflate_into(&mut compressor, &payload, &mut compressed).unwrap();
        let mut framed = (compressed.len() as u32).to_le_bytes().to_vec();
        framed.extend_from_slice(&payload_crc32(&compressed).to_le_bytes());
        framed.extend_from_slice(&compressed);
        framed
    }

    /// The run id every test in this module writes. Built from a number rather
    /// than `RunId::generate()` so the bytes a test compares against are fixed,
    /// and rather than a hand-typed literal so it cannot silently fail to be a
    /// run id (which would leave these tests exercising the "no id" path).
    fn test_run_id() -> Option<RunId> {
        Some(RunId::parse(&format!("run_{:016x}", 0x105)).expect("built in the one shape"))
    }

    #[test]
    fn borrowed_and_owned_record_decode_agree() {
        // The borrowed reader must accept bytes from BOTH writer paths and yield
        // exactly what the owned reader does -- it is a decode of the same wire
        // format, not a second format.
        let tmp = std::env::temp_dir().join("turbo_borrowed_record_decode.log");
        let mut w = RecordLogWriter::create(&tmp, test_run_id()).unwrap();
        let trace: Vec<u8> = (0..=255u8).cycle().take(4096).collect();
        let markers = vec![RoiRtMarker {
            pc: 0x1234,
            kind: RoiRtMarkerKind::Add,
            rs1_val: 1,
            rs2_val: 2,
            name: Some("matmul".to_string()),
        }];
        w.append_batch(
            &trace,
            BatchExtrasRef {
                roi_markers: &markers,
            },
        )
        .unwrap();
        w.append(&StreamRecord::Batch {
            trace_data: trace.clone(),
            extras: BatchExtras {
                roi_markers: markers.clone(),
            },
        })
        .unwrap();
        w.finish().unwrap();

        let buf = std::fs::read(&tmp).unwrap();
        check_record_log_header(&buf).unwrap().unwrap();
        let hdr = RECORD_LOG_HEADER_BYTES;
        let (mut c_borrowed, mut c_owned) = (hdr, hdr);
        let mut d_borrowed = RecordLogDecoder::new();
        let mut d_owned = RecordLogDecoder::new();
        for record in 0..2 {
            let o = decode_next_record(&mut d_owned, &buf, &mut c_owned)
                .unwrap()
                .unwrap();
            let b = decode_next_record_borrowed(&mut d_borrowed, &buf, &mut c_borrowed)
                .unwrap()
                .unwrap();
            match (&b, &o) {
                (
                    BorrowedStreamRecord::Batch {
                        trace_data,
                        extras: borrowed_extras,
                    },
                    StreamRecord::Batch {
                        trace_data: owned_trace,
                        extras: owned_extras,
                    },
                ) => {
                    let roi_markers = &borrowed_extras.roi_markers;
                    let owned_markers = &owned_extras.roi_markers;
                    assert_eq!(*trace_data, &trace[..], "record {record}");
                    assert_eq!(trace_data, &&owned_trace[..], "record {record}");
                    // RoiRtMarker has no PartialEq; compare the fields that
                    // would show a shape divergence between the two decodes.
                    let key = |m: &[RoiRtMarker]| -> Vec<(u64, Option<String>)> {
                        m.iter().map(|m| (m.pc, m.name.clone())).collect()
                    };
                    assert_eq!(key(roi_markers), key(owned_markers), "record {record}");
                    assert_eq!(key(roi_markers), key(&markers), "record {record}");
                }
                _ => panic!("record {record} is not a Batch: {b:?} / {o:?}"),
            }
            // Both readers must consume the same number of bytes, or a later
            // record would be read from a different offset by each.
            assert_eq!(
                c_borrowed, c_owned,
                "cursors diverged after record {record}"
            );
        }
        assert!(matches!(
            decode_next_record_borrowed(&mut d_borrowed, &buf, &mut c_borrowed)
                .unwrap()
                .unwrap(),
            BorrowedStreamRecord::Done
        ));
        std::fs::remove_file(&tmp).ok();
    }

    #[test]
    fn borrowed_decode_reports_a_torn_tail_as_incomplete() {
        // A growing file must not read as corruption -- same contract as the
        // owned path, checked here because the borrowed decode is what the live
        // tailing reader will use.
        // Payload is compressed, so a cut lands mid-DEFLATE-stream. The length
        // prefix is what must catch that, BEFORE the inflater ever sees the
        // bytes -- otherwise a growing file reads as corruption.
        let framed = frame_record(&StreamRecord::Batch {
            trace_data: vec![7; 512],
            extras: BatchExtras::default(),
        });
        let mut decoder = RecordLogDecoder::new();
        for cut in 0..framed.len() {
            let mut cursor = 0;
            assert!(
                decode_next_record_borrowed(&mut decoder, &framed[..cut], &mut cursor).is_none(),
                "a record truncated at {cut} must read as incomplete, not corrupt"
            );
            assert_eq!(cursor, 0, "an incomplete read must not advance the cursor");
        }
        let mut cursor = 0;
        assert!(
            decode_next_record_borrowed(&mut decoder, &framed, &mut cursor)
                .unwrap()
                .is_ok()
        );
        assert_eq!(cursor, framed.len());
    }

    fn sample_marker() -> RoiRtMarker {
        RoiRtMarker {
            pc: 0xdead_beef,
            kind: RoiRtMarkerKind::Add,
            rs1_val: 7,
            rs2_val: 9,
            name: Some("region".to_string()),
        }
    }

    #[test]
    fn record_log_round_trip() {
        let path = std::env::temp_dir().join("turbo_recordlog_test_roundtrip.bin");
        let _ = std::fs::remove_file(&path);

        let mut w = RecordLogWriter::create(&path, test_run_id()).unwrap();
        w.append(&StreamRecord::Batch {
            trace_data: vec![9, 8, 7, 6],
            extras: BatchExtras {
                roi_markers: vec![sample_marker()],
            },
        })
        .unwrap();
        w.append(&StreamRecord::Batch {
            trace_data: vec![],
            extras: BatchExtras::default(),
        })
        .unwrap();
        w.finish().unwrap();

        let bytes = std::fs::read(&path).unwrap();
        let header = check_record_log_header(&bytes).unwrap().unwrap();
        assert_eq!(header.version, RECORD_LOG_FORMAT_VERSION);
        assert_eq!(header.run_id, test_run_id());

        let mut cursor = RECORD_LOG_HEADER_BYTES;
        let mut records = Vec::new();
        let mut decoder = RecordLogDecoder::new();
        while let Some(r) = decode_next_record(&mut decoder, &bytes, &mut cursor) {
            records.push(r.unwrap());
        }
        assert_eq!(records.len(), 3);
        match &records[0] {
            StreamRecord::Batch { trace_data, extras } => {
                let roi_markers = &extras.roi_markers;
                assert_eq!(trace_data, &vec![9, 8, 7, 6]);
                assert_eq!(roi_markers.len(), 1);
                assert_eq!(roi_markers[0].name.as_deref(), Some("region"));
            }
            other => panic!("expected Batch, got {other:?}"),
        }
        match &records[1] {
            StreamRecord::Batch { trace_data, .. } => assert!(trace_data.is_empty()),
            other => panic!("expected Batch, got {other:?}"),
        }
        assert!(matches!(records[2], StreamRecord::Done));

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn every_compression_level_round_trips_to_the_same_records() {
        // The level is a size/CPU knob and nothing else: what comes back out
        // must not depend on it. Level 0 matters most -- it is the "compression
        // off" A/B setting, and it works only because deflate emits *stored*
        // blocks, keeping the file a valid stream with no reader special case.
        let trace: Vec<u8> = (0..=255u8).cycle().take(8192).collect();
        let mut sizes = Vec::new();
        for level in [0u32, 1, 6, 9] {
            let tmp = std::env::temp_dir().join(format!("turbo_level_{level}.log"));
            let mut w = RecordLogWriter::create_with_level(&tmp, level, test_run_id()).unwrap();
            w.append_batch(&trace, BatchExtrasRef::EMPTY).unwrap();
            w.finish().unwrap();
            sizes.push((level, w.bytes_written(), w.logical_bytes()));

            let records = read_record_log(&tmp).unwrap();
            assert_eq!(records.len(), 1, "level {level}");
            match &records[0] {
                StreamRecord::Batch { trace_data, .. } => {
                    assert_eq!(trace_data, &trace, "level {level} corrupted the payload")
                }
                other => panic!("level {level}: expected Batch, got {other:?}"),
            }
            std::fs::remove_file(&tmp).ok();
        }
        // Level 0 must be much bigger than level 1 on this highly compressible
        // input, which is what proves it really is the "off" setting rather
        // than silently compressing anyway.
        let stored = sizes[0].1;
        let deflated = sizes[1].1;
        assert!(
            stored > deflated * 4,
            "level 0 ({stored} B) should be far larger than level 1 ({deflated} B)"
        );
    }

    #[test]
    fn an_uncompressed_payload_fails_with_an_actionable_message() {
        // A record that is framed correctly and passes its checksum, but whose
        // payload is not a deflate stream: i.e. a producer that agreed with
        // this reader about the framing and disagreed about the payload. The
        // header's version tag cannot catch that, so the inflate failure's
        // message is the only thing telling the user what happened -- assert
        // that it says so.
        let record = StreamRecord::Batch {
            trace_data: vec![1, 2, 3, 4, 5],
            extras: BatchExtras::default(),
        };
        // Raw postcard, no deflate, but framed and checksummed as v2 requires.
        let payload = postcard::to_allocvec(&record).unwrap();
        let mut framed = (payload.len() as u32).to_le_bytes().to_vec();
        framed.extend_from_slice(&payload_crc32(&payload).to_le_bytes());
        framed.extend_from_slice(&payload);

        let mut decoder = RecordLogDecoder::new();
        let mut cursor = 0;
        let err = decode_next_record(&mut decoder, &framed, &mut cursor)
            .expect("a complete-but-undecodable record must not read as a torn tail")
            .unwrap_err();
        assert!(err.contains("STALE TRACE"), "unhelpful message: {err}");
        assert!(
            err.contains("Re-run the workload"),
            "no fix suggested: {err}"
        );
    }

    /// The writer deflates straight into the frame buffer, behind a placeholder
    /// length it patches afterwards, rather than compressing into scratch and
    /// concatenating. That is an allocation/copy optimisation ONLY: the bytes
    /// that reach the file must stay identical to the naive `[len][deflate]`
    /// concatenation every reader and every hand-built test fixture assumes.
    #[test]
    fn writer_frame_bytes_match_naive_concatenation() {
        let record = StreamRecord::Batch {
            trace_data: (0..=255u8).cycle().take(8192).collect(),
            extras: BatchExtras {
                roi_markers: vec![sample_marker()],
            },
        };
        let tmp = std::env::temp_dir().join("turbo_frame_bytes_match.log");
        let mut w = RecordLogWriter::create(&tmp, test_run_id()).unwrap();
        // Twice: the second frame reuses a buffer that already holds the first
        // one's bytes, which is exactly where a missing `clear()` would show up.
        w.append(&record).unwrap();
        w.append(&record).unwrap();
        w.finish().unwrap();

        let on_disk = std::fs::read(&tmp).unwrap();
        let expected: Vec<u8> = encode_record_log_header(test_run_id())
            .into_iter()
            .chain(frame_record(&record))
            .chain(frame_record(&record))
            .chain(frame_record(&StreamRecord::Done))
            .collect();
        assert_eq!(on_disk, expected);
        let _ = std::fs::remove_file(&tmp);
    }

    #[test]
    fn record_log_torn_tail() {
        let record = StreamRecord::Batch {
            trace_data: vec![1, 2, 3, 4, 5],
            extras: BatchExtras {
                roi_markers: vec![sample_marker()],
            },
        };
        let framed = frame_record(&record);
        let mut decoder = RecordLogDecoder::new();

        // Only the length prefix present -> None, cursor unchanged.
        let prefix_only = &framed[..4];
        let mut cursor = 0;
        assert!(decode_next_record(&mut decoder, prefix_only, &mut cursor).is_none());
        assert_eq!(cursor, 0);

        // Prefix + partial payload -> None, cursor unchanged.
        let partial = &framed[..4 + (framed.len() - 4) / 2];
        let mut cursor = 0;
        assert!(decode_next_record(&mut decoder, partial, &mut cursor).is_none());
        assert_eq!(cursor, 0);

        // Full bytes -> decodes and advances.
        let mut cursor = 0;
        let decoded = decode_next_record(&mut decoder, &framed, &mut cursor)
            .expect("full record must decode")
            .unwrap();
        assert_eq!(cursor, framed.len());
        assert!(matches!(decoded, StreamRecord::Batch { .. }));
    }

    #[test]
    fn grouping_fields_into_a_nested_struct_is_wire_compatible() {
        // Grouping fields into a nested struct must not shift a single byte
        // versus listing them flat: postcard writes a struct as the bare
        // concatenation of its fields, with no tag or length.
        //
        // This is the property that let BatchExtras be introduced WITHOUT
        // bumping RECORD_LOG_MAGIC, and it is what keeps record logs written by
        // earlier builds readable. It is asserted on a standalone pair of shapes
        // rather than on BatchExtras itself so that it stays a test of postcard's
        // encoding, independent of which side-channels BatchExtras happens to
        // carry. If postcard ever changes this, it fails loudly here rather than
        // as a mis-parse far downstream.
        #[derive(Serialize)]
        struct Inner<'a> {
            counter: u64,
            flag: bool,
            slice: &'a [u8],
        }
        #[derive(Serialize)]
        enum Nested<'a> {
            #[allow(dead_code)]
            First,
            Payload {
                head: &'a [u8],
                inner: Inner<'a>,
            },
        }
        #[derive(Serialize)]
        enum Flat<'a> {
            #[allow(dead_code)]
            First,
            Payload {
                head: &'a [u8],
                counter: u64,
                flag: bool,
                slice: &'a [u8],
            },
        }

        let head: Vec<u8> = (0..=255u8).cycle().take(600).collect();
        let slice = vec![7u8, 8, 9];

        let nested = postcard::to_allocvec(&Nested::Payload {
            head: &head,
            inner: Inner {
                counter: 1 << 40,
                flag: true,
                slice: &slice,
            },
        })
        .unwrap();
        let flat = postcard::to_allocvec(&Flat::Payload {
            head: &head,
            counter: 1 << 40,
            flag: true,
            slice: &slice,
        })
        .unwrap();

        assert_eq!(flat, nested, "struct nesting must be a no-op on the wire");
    }

    #[test]
    fn record_log_header_checks() {
        assert!(check_record_log_header(b"XXXXXXXX").is_err());
        // Shorter than the 8-byte magic -> pending.
        assert_eq!(check_record_log_header(b"TRBO").unwrap(), None);
        // Magic alone: the version and run id have not landed yet, so still
        // pending rather than an error. A live reader opens the file the moment
        // it appears and must tolerate seeing it mid-write.
        assert_eq!(check_record_log_header(&TRBOLOG_MAGIC).unwrap(), None);

        let header = encode_record_log_header(test_run_id());
        assert_eq!(header.len(), RECORD_LOG_HEADER_BYTES);
        let parsed = check_record_log_header(&header).unwrap().unwrap();
        assert_eq!(parsed.version, RECORD_LOG_FORMAT_VERSION);
        assert_eq!(parsed.run_id, test_run_id());
        // Byte-by-byte arrival: every prefix of a valid header is "pending",
        // never an error and never a short read of the run id.
        for n in TRBOLOG_MAGIC.len()..header.len() {
            assert_eq!(
                check_record_log_header(&header[..n]).unwrap(),
                None,
                "a {n}-byte prefix must be pending, not parsed"
            );
        }

        // A producer with no run of its own writes a full-width header whose id
        // field is zeroed, and it reads back as "no id" -- not as a short
        // header, and not as an error.
        let anonymous = encode_record_log_header(None);
        assert_eq!(anonymous.len(), RECORD_LOG_HEADER_BYTES);
        let parsed = check_record_log_header(&anonymous).unwrap().unwrap();
        assert_eq!(parsed.run_id, None);

        // A log written before the header carried a version: the magic still
        // matches (it identifies the file type, not the layout), and the four
        // bytes where the version now lives are that file's first record
        // length. It must be refused, not read as version 383.
        let mut pre_version = TRBOLOG_MAGIC.to_vec();
        pre_version.extend_from_slice(&383u32.to_le_bytes()); // a record length
        pre_version.resize(RECORD_LOG_HEADER_BYTES, 0xab); // payload bytes
        let e = check_record_log_header(&pre_version).unwrap_err();
        assert!(e.contains("version 383"), "got: {e}");
        assert!(e.contains("re-record"), "no fix suggested: {e}");

        // A version this build does not read is refused by number.
        let mut wrong_version = encode_record_log_header(test_run_id());
        wrong_version[TRBOLOG_MAGIC.len()] = 99;
        let e = check_record_log_header(&wrong_version).unwrap_err();
        assert!(e.contains("version 99"), "got: {e}");

        // Garbage where the run id belongs is NOT an error: the id is a
        // staleness hint, and a header that is otherwise well-formed still
        // locates the first record. It reads back as "no id", so the consumer
        // skips the check rather than declaring the file corrupt.
        let mut damaged_id = encode_record_log_header(test_run_id());
        damaged_id[TRBOLOG_MAGIC.len() + 4..].fill(0xff);
        let parsed = check_record_log_header(&damaged_id).unwrap().unwrap();
        assert_eq!(parsed.run_id, None);
        assert_eq!(parsed.version, RECORD_LOG_FORMAT_VERSION);
    }

    /// The CRC exists to catch the corruption that inflating cannot: a payload
    /// that still decompresses and still parses, into different bytes.
    #[test]
    fn a_damaged_payload_fails_its_checksum() {
        let path = std::env::temp_dir().join("turbo_recordlog_test_crc.bin");
        let _ = std::fs::remove_file(&path);
        let mut w = RecordLogWriter::create(&path, test_run_id()).unwrap();
        // Compressible, and long enough that a flipped bit lands inside the
        // deflate stream rather than in its final byte.
        let trace_data: Vec<u8> = (0..4000u32).map(|i| (i / 16) as u8).collect();
        w.append_batch(&trace_data, BatchExtrasRef::EMPTY).unwrap();
        w.finish().unwrap();

        let good = std::fs::read(&path).unwrap();
        check_record_log_header(&good).unwrap().unwrap();

        // Unmodified: decodes.
        let mut cursor = RECORD_LOG_HEADER_BYTES;
        let mut decoder = RecordLogDecoder::new();
        assert!(decode_next_record(&mut decoder, &good, &mut cursor)
            .unwrap()
            .is_ok());

        // Every byte of the first record's payload, flipped one at a time, is
        // caught -- which is the property a checksum buys and postcard alone
        // does not.
        let payload_start = RECORD_LOG_HEADER_BYTES + RECORD_HEADER_BYTES;
        let payload_len = u32::from_le_bytes(
            good[RECORD_LOG_HEADER_BYTES..RECORD_LOG_HEADER_BYTES + LEN_PREFIX_BYTES]
                .try_into()
                .unwrap(),
        ) as usize;
        for offset in payload_start..payload_start + payload_len {
            let mut damaged = good.clone();
            damaged[offset] ^= 0x01;
            let mut cursor = RECORD_LOG_HEADER_BYTES;
            let mut decoder = RecordLogDecoder::new();
            let err = decode_next_record(&mut decoder, &damaged, &mut cursor)
                .expect("the record is complete, so this is a decode attempt")
                .expect_err("a flipped payload bit must not decode");
            assert!(
                err.contains("fails its checksum"),
                "byte {offset}: expected a checksum failure, got: {err}"
            );
        }
    }

    #[test]
    fn append_batch_matches_owned_append() {
        // Long enough that the payload's postcard length needs a two-byte
        // varint, which is what makes this a real check that
        // `serialize_bytes` (borrowed path) and serde's seq-of-`u8` (owned
        // path) agree on the framing and not just on the bytes.
        let trace_data: Vec<u8> = (0..500u32).map(|i| (i * 7) as u8).collect();
        let roi_markers = vec![sample_marker()];

        // Owned via append.
        let owned_path = std::env::temp_dir().join("turbo_recordlog_test_owned.bin");
        let _ = std::fs::remove_file(&owned_path);
        let mut wo = RecordLogWriter::create(&owned_path, test_run_id()).unwrap();
        wo.append(&StreamRecord::Batch {
            trace_data: trace_data.clone(),
            extras: BatchExtras {
                roi_markers: roi_markers.clone(),
            },
        })
        .unwrap();
        wo.flush().unwrap();
        let owned_bytes = std::fs::read(&owned_path).unwrap();

        // Borrowed via append_batch.
        let borrowed_path = std::env::temp_dir().join("turbo_recordlog_test_borrowed.bin");
        let _ = std::fs::remove_file(&borrowed_path);
        let mut wb = RecordLogWriter::create(&borrowed_path, test_run_id()).unwrap();
        wb.append_batch(
            &trace_data,
            BatchExtrasRef {
                roi_markers: &roi_markers,
            },
        )
        .unwrap();
        wb.flush().unwrap();
        let borrowed_bytes = std::fs::read(&borrowed_path).unwrap();

        assert_eq!(
            owned_bytes, borrowed_bytes,
            "borrowed fast path must be byte-identical to owned append"
        );

        let _ = std::fs::remove_file(&owned_path);
        let _ = std::fs::remove_file(&borrowed_path);
    }
}
