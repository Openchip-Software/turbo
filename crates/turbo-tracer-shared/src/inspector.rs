//! Report what a per-vCPU `turbo_trace_vcpu<N>.bin` on disk is actually made of.
//!
//! The trace file's size is important, so it is compressed and the file sizes
//! are easily inspected with this tool. The BBT format is easily compressed, as
//! repeated executions are identified by Deflate algorithm, resulting in very
//! small file-sizes for repetitive workloads like HPC kernels and AI inferencing.

use std::collections::HashMap;
use std::path::Path;

use crate::bbt::{bbt_decode, parse_chunk, BbtRecord, BBT_CHUNK_HEADER_BYTES};
use crate::{
    check_record_log_header, decode_next_record, RecordLogDecoder, RunId, StreamRecord,
    RECORD_HEADER_BYTES, RECORD_LOG_HEADER_BYTES,
};

/// One row of the uncompressed size breakdown: what a category costs, and how
/// many items of it there were. Emitted as data rather than as a formatted
/// line so the caller owns every formatting decision.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SizeRow {
    /// Human-facing category name, e.g. `"bbt block executions"`.
    pub label: &'static str,
    /// Uncompressed bytes this category costs.
    pub bytes: u64,
    /// Items in the category (records, entries, batches, markers).
    pub count: u64,
    /// `bytes` as a percentage of the file's uncompressed size.
    pub percent: f64,
}

/// One block of the dictionary and how often the record stream executed it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BlockStat {
    pub id: u32,
    /// Instructions in the block, from its descriptor.
    pub insns: u32,
    pub executions: u64,
}

impl BlockStat {
    /// Guest instructions this block contributed to the whole stream.
    pub fn total_insns(&self) -> u64 {
        u64::from(self.insns) * self.executions
    }
}

/// Everything accumulated over one trace file.
#[derive(Debug, Default, Clone)]
pub struct TraceReport {
    /// The run this log was recorded by, from its header. `None` for a log
    /// whose producer had no run id, and for a set of logs that disagree about
    /// theirs (see [`TraceReport::combined`]).
    pub run_id: Option<RunId>,
    /// Size on disk, i.e. after per-record compression.
    pub file_bytes: u64,
    /// Sum of the record payloads *before* compression, plus the framing that
    /// is not compressed. This is the size the categories below add up to.
    pub logical_bytes: u64,
    pub batches: u64,
    /// Whether the writer's `Done` sentinel was reached: its absence means the
    /// file is truncated or still being written.
    pub saw_done: bool,
    pub truncated_tail: u64,

    pub block_records: u64,
    pub vl_records: u64,
    /// Address groups: one per vector memory instruction executed.
    pub mem_groups: u64,
    /// Address half-words, i.e. two per captured address or stride.
    pub addr_records: u64,
    pub reserved_records: u64,

    pub dict_bytes: u64,
    pub dict_entries: u64,
    /// Descriptors whose ID was already described earlier in THIS file. A
    /// non-zero count means the file carries the same block twice.
    pub dict_redundant_entries: u64,
    pub dict_redundant_bytes: u64,
    /// Instructions covered by the dictionary, i.e. the size bytes.
    pub dict_insns: u64,

    pub chunk_header_bytes: u64,
    pub roi_markers: u64,
    /// Postcard bytes the ROI side-channel costs, measured by re-serializing
    /// each batch's `extras` rather than estimated from the struct shape.
    pub roi_bytes: u64,
    /// Of `roi_bytes`, the part that is marker name strings.
    pub roi_name_bytes: u64,
    /// Distinct marker name strings seen, and how many times names repeat.
    pub roi_names: HashMap<String, u64>,

    /// Guest instructions the record stream expands to.
    pub retired_insns: u64,
    /// Records naming a block ID with no descriptor at or before that point.
    pub unresolved_block_records: u64,

    /// id -> (n_insns, executions)
    pub blocks: HashMap<u32, (u32, u64)>,
}

impl TraceReport {
    /// Uncompressed bytes spent on the fixed-width BBT record words.
    pub fn record_bytes(&self) -> u64 {
        4 * (self.block_records
            + self.vl_records
            + self.mem_groups
            + self.addr_records
            + self.reserved_records)
    }

    /// Bytes not inside a BBT chunk: the log magic, per-record `[u32 len]`
    /// framing, postcard tags/varints, and the serialized ROI markers.
    pub fn envelope_bytes(&self) -> u64 {
        self.logical_bytes
            .saturating_sub(self.record_bytes() + self.dict_bytes + self.chunk_header_bytes)
    }

    /// Envelope minus the ROI side-channel: pure per-record framing overhead.
    pub fn framing_bytes(&self) -> u64 {
        self.envelope_bytes().saturating_sub(self.roi_bytes)
    }

    /// Uncompressed size divided by on-disk size: what deflate achieved.
    pub fn compression_ratio(&self) -> f64 {
        self.logical_bytes as f64 / self.file_bytes.max(1) as f64
    }

    /// The uncompressed size breakdown, most-significant categories first.
    ///
    /// Percentages are of the UNCOMPRESSED size: deflate shares a window
    /// across a whole batch, so there is no honest way to attribute compressed
    /// bytes to one category. What each category costs before compression is
    /// the number that actually guides format work.
    ///
    /// The rows partition `logical_bytes` exactly (`framing_bytes` is the
    /// remainder), so a caller may sum them as a self-check. The reserved-tag
    /// row is present only when such records exist -- they are a format
    /// anomaly, not a normal cost line.
    pub fn size_rows(&self) -> Vec<SizeRow> {
        let mut rows = vec![
            (
                "bbt block executions",
                4 * self.block_records,
                self.block_records,
            ),
            ("bbt vl values", 4 * self.vl_records, self.vl_records),
            (
                "bbt memory addresses",
                4 * (self.mem_groups + self.addr_records),
                self.mem_groups,
            ),
            ("bbt block descriptors", self.dict_bytes, self.dict_entries),
            ("bbt chunk headers", self.chunk_header_bytes, self.batches),
            ("roi markers", self.roi_bytes, self.roi_markers),
            ("record framing", self.framing_bytes(), self.batches),
        ];
        if self.reserved_records != 0 {
            rows.push((
                "reserved-tag records",
                4 * self.reserved_records,
                self.reserved_records,
            ));
        }
        rows.into_iter()
            .map(|(label, bytes, count)| SizeRow {
                label,
                bytes,
                count,
                percent: pct(bytes, self.logical_bytes),
            })
            .collect()
    }

    /// Distinct block IDs described by the dictionary.
    pub fn distinct_blocks(&self) -> usize {
        self.blocks.len()
    }

    /// Mean instructions per described block.
    pub fn insns_per_block(&self) -> f64 {
        self.dict_insns as f64 / self.dict_entries.max(1) as f64
    }

    /// Mean guest instructions each block-execution record expands to.
    pub fn insns_per_record(&self) -> f64 {
        self.retired_insns as f64 / self.block_records.max(1) as f64
    }

    /// Uncompressed trace bytes per retired guest instruction.
    pub fn bytes_per_insn(&self) -> f64 {
        self.logical_bytes as f64 / self.retired_insns.max(1) as f64
    }

    /// On-disk (compressed) trace bytes per retired guest instruction.
    pub fn disk_bytes_per_insn(&self) -> f64 {
        self.file_bytes as f64 / self.retired_insns.max(1) as f64
    }

    /// The `n` most-executed blocks. These are the ones worth compressing
    /// further, so they are worth naming.
    pub fn hottest_blocks(&self, n: usize) -> Vec<BlockStat> {
        let mut top: Vec<BlockStat> = self
            .blocks
            .iter()
            .map(|(&id, &(insns, executions))| BlockStat {
                id,
                insns,
                executions,
            })
            .collect();
        top.sort_unstable_by_key(|b| (std::cmp::Reverse(b.executions), b.id));
        top.truncate(n);
        top
    }

    /// `(distinct marker names, total name strings shipped)`. The second being
    /// larger than the first is the ROI side-channel re-sending the same text.
    pub fn roi_name_repeats(&self) -> (usize, u64) {
        (self.roi_names.len(), self.roi_names.values().sum())
    }

    /// Fold another file's report into this one, to describe a whole run as if
    /// it were a single trace.
    ///
    /// Block IDs are safe to merge across harts: the plugin's dictionary is
    /// plugin-wide, keyed by `(start_pc, block hash)`, so one ID means the same
    /// block on every vCPU. `saw_done` becomes "every file was complete", since
    /// one truncated hart makes the combined picture incomplete.
    pub fn merge(&mut self, other: &TraceReport) {
        self.file_bytes += other.file_bytes;
        self.logical_bytes += other.logical_bytes;
        self.batches += other.batches;
        self.saw_done &= other.saw_done;
        self.truncated_tail += other.truncated_tail;

        self.block_records += other.block_records;
        self.vl_records += other.vl_records;
        self.mem_groups += other.mem_groups;
        self.addr_records += other.addr_records;
        self.reserved_records += other.reserved_records;

        self.dict_bytes += other.dict_bytes;
        self.dict_entries += other.dict_entries;
        self.dict_redundant_entries += other.dict_redundant_entries;
        self.dict_redundant_bytes += other.dict_redundant_bytes;
        self.dict_insns += other.dict_insns;

        self.chunk_header_bytes += other.chunk_header_bytes;
        self.roi_markers += other.roi_markers;
        self.roi_bytes += other.roi_bytes;
        self.roi_name_bytes += other.roi_name_bytes;
        for (name, n) in &other.roi_names {
            *self.roi_names.entry(name.clone()).or_default() += n;
        }

        self.retired_insns += other.retired_insns;
        self.unresolved_block_records += other.unresolved_block_records;

        for (&id, &(insns, execs)) in &other.blocks {
            let e = self.blocks.entry(id).or_insert((insns, 0));
            e.1 += execs;
        }
    }

    /// Combine per-file reports into one whole-run report. An empty input
    /// yields a zeroed report, which reads as an empty run rather than
    /// panicking.
    pub fn combined<'a>(reports: impl IntoIterator<Item = &'a TraceReport>) -> TraceReport {
        let mut out = TraceReport {
            saw_done: true,
            ..Default::default()
        };
        let mut run_id: Option<Option<RunId>> = None;
        for r in reports {
            // One run id for the whole set only when every file agrees; a
            // mixture reports none, rather than picking one file's and
            // implying the others share it.
            match run_id {
                None => run_id = Some(r.run_id),
                Some(seen) if seen == r.run_id => {}
                Some(_) => run_id = Some(None),
            }
            out.merge(r);
        }
        out.run_id = run_id.flatten();
        out
    }

    /// Share of `total` that `n` represents, as a percentage. Zero-total safe.
    pub fn percent_of_stream(&self, n: u64) -> f64 {
        pct(n, self.retired_insns)
    }
}

fn pct(n: u64, total: u64) -> f64 {
    if total == 0 {
        0.0
    } else {
        100.0 * n as f64 / total as f64
    }
}

/// Compressed bytes to consume between two `on_bytes` callbacks. The callback
/// is a shared counter in the caller, so it is coalesced here: 1 MiB of a
/// compressed BBT log is roughly 24 MiB of records, i.e. a small enough slice
/// of any real file to keep a progress report smooth, and large enough that
/// the reporting costs nothing next to the walk.
const PROGRESS_TICK_BYTES: usize = 1 << 20;

/// Walk one record log and attribute every byte in it.
///
/// Errors describe the file and the batch at which the walk gave up; a torn
/// tail (a length prefix or payload that never landed) is not an error, it is
/// reported as `truncated_tail` with `saw_done == false`.
pub fn inspect_trace_file(path: &Path) -> Result<TraceReport, String> {
    inspect_trace_file_with_progress(path, &|_| {})
}

/// [`inspect_trace_file`], reporting how far through the file it has read.
///
/// `on_bytes` is called with the number of *compressed* (on-disk) bytes
/// consumed since the previous call, coalesced to [`PROGRESS_TICK_BYTES`].
/// On-disk bytes are the unit because they are the only measure of the work a
/// file represents that is known before it is parsed, which is what lets a
/// caller inspecting several files build one accurate total.
///
/// The reported bytes do not add up to the file size when the walk stops
/// early -- at the `Done` sentinel, a torn tail, or an error -- so a caller
/// that needs the counts to close should credit the difference against the
/// file's length itself.
pub fn inspect_trace_file_with_progress(
    path: &Path,
    on_bytes: &dyn Fn(u64),
) -> Result<TraceReport, String> {
    let buf = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let mut r = TraceReport {
        file_bytes: buf.len() as u64,
        ..Default::default()
    };

    let header = match check_record_log_header(&buf) {
        Ok(Some(hdr)) => hdr,
        Ok(None) => {
            return Err(format!(
                "{}: only {} byte(s), too short for a record-log header ({} bytes: \
                 magic, version and run id)",
                path.display(),
                buf.len(),
                RECORD_LOG_HEADER_BYTES
            ))
        }
        Err(e) => return Err(format!("{}: {e}", path.display())),
    };
    let mut cursor = RECORD_LOG_HEADER_BYTES;
    r.run_id = header.run_id;

    r.logical_bytes = cursor as u64; // the header is not compressed

    // Descriptors are cumulative across chunks within a file: an ID described in
    // chunk 1 stays valid for records in chunk 9.
    // The owned decode is used deliberately: it copies the trace payload out of
    // the decoder's scratch, which frees the borrow so `last_inflated_len` can
    // be queried for the size accounting. One memcpy per batch is irrelevant to
    // an offline inspector.
    let mut decoder = RecordLogDecoder::new();
    // Offset already handed to `on_bytes`; the gap to `cursor` is the work done
    // since the last report.
    let mut reported = cursor;
    loop {
        let start = cursor;
        if start - reported >= PROGRESS_TICK_BYTES {
            on_bytes((start - reported) as u64);
            reported = start;
        }
        let decoded = decode_next_record(&mut decoder, &buf, &mut cursor);
        // Framing (`[len][crc]`, uncompressed) plus the payload as it was
        // before the writer compressed it.
        if decoded.is_some() {
            r.logical_bytes += RECORD_HEADER_BYTES as u64 + decoder.last_inflated_len() as u64;
        }
        match decoded {
            None => {
                // Torn tail: a length prefix or payload that never landed.
                r.truncated_tail = r.file_bytes - start as u64;
                break;
            }
            // The decoder's message already locates the record and explains
            // the likely cause; just name the file.
            Some(Err(e)) => return Err(format!("{}: {e}", path.display())),
            Some(Ok(StreamRecord::Done)) => {
                r.saw_done = true;
                r.truncated_tail = r.file_bytes - cursor as u64;
                break;
            }
            Some(Ok(StreamRecord::Batch { trace_data, extras })) => {
                r.batches += 1;
                r.roi_markers += extras.roi_markers.len() as u64;
                // Measured, not modelled: re-serialize the side-channel exactly
                // as the writer did, so the number is the real on-wire cost.
                if !extras.roi_markers.is_empty() {
                    r.roi_bytes += postcard::to_allocvec(&extras)
                        .map(|v| v.len() as u64)
                        .unwrap_or(0);
                    for m in &extras.roi_markers {
                        if let Some(name) = &m.name {
                            // postcard writes a String as varint-len + bytes.
                            r.roi_name_bytes += name.len() as u64 + 1;
                            *r.roi_names.entry(name.clone()).or_default() += 1;
                        }
                    }
                }

                let (dict, records) = parse_chunk(&trace_data)
                    .map_err(|e| format!("{}: batch {}: {e}", path.display(), r.batches))?;
                r.chunk_header_bytes += BBT_CHUNK_HEADER_BYTES;
                r.dict_bytes += dict.len() as u64;

                let mut rest = dict;
                while rest.len() >= 14 {
                    let id = u32::from_le_bytes(rest[0..4].try_into().unwrap());
                    let n = usize::from(u16::from_le_bytes(rest[12..14].try_into().unwrap()));
                    if rest.len() < 14 + n {
                        return Err(format!(
                            "{}: batch {}: truncated descriptor for id {id}",
                            path.display(),
                            r.batches
                        ));
                    }
                    r.dict_entries += 1;
                    r.dict_insns += n as u64;
                    // Re-inserting would reset an existing block's exec count,
                    // so only a genuinely new ID is inserted.
                    if let std::collections::hash_map::Entry::Vacant(e) = r.blocks.entry(id) {
                        e.insert((n as u32, 0));
                    } else {
                        r.dict_redundant_entries += 1;
                        r.dict_redundant_bytes += (14 + n) as u64;
                    }
                    rest = &rest[14 + n..];
                }

                for w in records.as_chunks::<4>().0 {
                    let word = u32::from_le_bytes(*w);
                    match bbt_decode(word) {
                        BbtRecord::Block(id) => {
                            r.block_records += 1;
                            match r.blocks.get_mut(&id) {
                                Some((n, execs)) => {
                                    r.retired_insns += u64::from(*n);
                                    *execs += 1;
                                }
                                None => r.unresolved_block_records += 1,
                            }
                        }
                        BbtRecord::Vl(_) => r.vl_records += 1,
                        BbtRecord::Mem { .. } => r.mem_groups += 1,
                        BbtRecord::Addr(_) => r.addr_records += 1,
                        BbtRecord::Reserved(_) => r.reserved_records += 1,
                    }
                }
            }
        }
    }

    Ok(r)
}
