//! A directory of per-hart `turbo_trace_vcpu<N>.bin` record logs, tailed as one unit.
//!
//! The recording side writes one log per hart (see
//! `turbo_tracer::trace_encoder`), so a spool is a *set* of streams, not one.
//! This type owns that set: it discovers logs, pumps each of them, and reports
//! when they are collectively spent. It is shared by the live QEMU path and the
//! offline `process` path, which is what makes multi-hart the default everywhere
//! rather than a property of one of them.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use turbo_tracer_shared::RunId;

use crate::source::record_log::{hart_id_from_path, HartLogReader};
use crate::source::TraceData;

/// Batches taken from each hart per pump pass.
///
/// The cap is per hart per pass, so `pending` stays bounded no matter how many
/// harts there are, and each hart makes progress on every pass instead of hart 0
/// being drained to completion before hart 1 is looked at. Chronological-ish
/// interleaving is what the Perfetto writer and any cross-hart view want.
const BATCHES_PER_HART_PER_PUMP: usize = 32;

/// Tails every `turbo_trace_vcpu<N>.bin` in a directory.
pub(crate) struct HartSpool {
    dir: PathBuf,
    /// `BTreeMap`, not `HashMap`: iteration order is the hart order, so the
    /// interleave of messages is reproducible run to run. With `HashMap` the
    /// multi-hart message order varied between runs of the same input.
    readers: BTreeMap<u32, HartLogReader>,
    /// When set, only these harts are read (`--vcpu`).
    hart_filter: Option<Vec<u32>>,
    /// Harts present in the directory but excluded by `hart_filter`, reported
    /// once so a filtered run never looks like a complete one.
    excluded: Vec<u32>,
    /// Set for an explicitly-named single log: the reader set is fixed, so
    /// [`rescan`](Self::rescan) must not adopt its siblings.
    fixed_set: bool,
    /// Why `excluded` harts are not being read, for the report.
    exclusion_reason: &'static str,
    message_id: u32,
    total_bytes: usize,
    /// A log that was discovered but could not be OPENED. Kept here rather
    /// than logged and dropped: a spool with an unreadable log is not a spool
    /// with fewer harts, it is a run that would report a fraction of the
    /// program as if it were all of it.
    open_fault: Option<(u32, String)>,
}

impl HartSpool {
    pub(crate) fn new(dir: PathBuf, hart_filter: Option<Vec<u32>>) -> Self {
        Self {
            dir,
            readers: BTreeMap::new(),
            hart_filter,
            excluded: Vec::new(),
            fixed_set: false,
            exclusion_reason: "the requested vCPU set",
            message_id: 0,
            total_bytes: 0,
            open_fault: None,
        }
    }

    /// A spool of exactly one explicitly-named record log.
    ///
    /// Used when `process` is pointed at a single file: the user named that log,
    /// so its siblings are not adopted — but they ARE reported, because silently
    /// analysing one hart of a multi-hart recording is the failure mode this
    /// whole type exists to prevent.
    pub(crate) fn single_log(path: &Path, hart_id: u32) -> Self {
        let dir = path
            .parent()
            .unwrap_or_else(|| Path::new("."))
            .to_path_buf();
        let mut spool = Self {
            dir,
            readers: BTreeMap::new(),
            hart_filter: None,
            excluded: Vec::new(),
            fixed_set: true,
            exclusion_reason: "a single record log being named explicitly",
            message_id: 0,
            total_bytes: 0,
            open_fault: None,
        };
        match HartLogReader::open(hart_id, path) {
            Ok(reader) => {
                log::info!("Reading record log for hart {} at {:?}", hart_id, path);
                spool.readers.insert(hart_id, reader);
            }
            Err(e) => {
                spool.open_fault = Some((hart_id, format!("cannot open {}: {e}", path.display())))
            }
        }
        // Report siblings the user is not getting.
        if let Ok(entries) = std::fs::read_dir(&spool.dir) {
            for entry in entries.flatten() {
                if let Some(id) = hart_id_from_path(&entry.path()) {
                    if id != hart_id {
                        spool.excluded.push(id);
                    }
                }
            }
        }
        spool.excluded.sort_unstable();
        spool
    }

    /// Open a reader for any `turbo_trace_vcpu<N>.bin` not already tracked.
    ///
    /// Called on every pump: on the live path harts appear as the guest spawns
    /// threads, so the set is not known up front. Once it stabilises this is a
    /// directory listing and nothing else.
    pub(crate) fn rescan(&mut self) {
        if self.fixed_set {
            return;
        }
        let entries = match std::fs::read_dir(&self.dir) {
            Ok(e) => e,
            Err(e) => {
                log::trace!("Could not read spool dir {:?}: {}", self.dir, e);
                return;
            }
        };
        for entry in entries.flatten() {
            let path = entry.path();
            let Some(hart_id) = hart_id_from_path(&path) else {
                continue;
            };
            if self.readers.contains_key(&hart_id) {
                continue;
            }
            if let Some(ref only) = self.hart_filter {
                if !only.contains(&hart_id) {
                    if !self.excluded.contains(&hart_id) {
                        self.excluded.push(hart_id);
                        // Sorted, not discovery-ordered: this list is reported
                        // to the user and recorded in the trace-integrity
                        // block, and `read_dir` order varies run to run.
                        self.excluded.sort_unstable();
                        log::info!(
                            "hart {} record log present but excluded by the requested vCPU set {:?}",
                            hart_id,
                            only
                        );
                    }
                    continue;
                }
            }
            match HartLogReader::open(hart_id, &path) {
                Ok(reader) => {
                    log::info!("Tailing record log for hart {} at {:?}", hart_id, path);
                    self.readers.insert(hart_id, reader);
                }
                Err(e) => {
                    self.open_fault =
                        Some((hart_id, format!("cannot open {}: {e}", path.display())));
                }
            }
        }
    }

    /// Discover new logs, then take a bounded slice from each hart in turn.
    ///
    /// Returns whether any reader may still have data available right now.
    pub(crate) fn pump(&mut self, out: &mut Vec<TraceData>) -> bool {
        self.rescan();
        let mut more = false;
        for reader in self.readers.values_mut() {
            let stat = reader.pump(out, &mut self.message_id, BATCHES_PER_HART_PER_PUMP);
            more |= stat.more_available;
        }
        self.total_bytes = self.readers.values().map(|r| r.bytes_read()).sum();
        more
    }

    /// True once no discovered log has anything left to hand over: each has
    /// either reached `Done` / failed, or been read to EOF. Combined with a
    /// finished producer this means the spool is fully consumed.
    pub(crate) fn all_spent(&self) -> bool {
        self.readers.values().all(|r| r.spent())
    }

    /// True if no logs have been discovered yet.
    pub(crate) fn is_empty(&self) -> bool {
        self.readers.is_empty()
    }

    /// Refuse any log that is not from the run `expected` names.
    ///
    /// The record logs, the `turbo_metadata.json` beside them and each other
    /// must all describe one execution. Nothing else can check this: a log from
    /// another run decodes perfectly -- its framing is valid and its records
    /// are intact -- and produces a profile resolved against the WRONG address
    /// layout, i.e. plausible numbers attributed to the wrong functions. The
    /// only evidence is the run id each header carries.
    ///
    /// `expected` of `None` means "no reference available" (metadata without a
    /// run id): the harts are then only checked against each other, which still
    /// catches a directory holding two runs' logs. Harts with no run id of
    /// their own are skipped, for the same reason -- there is nothing to
    /// compare.
    ///
    /// Faults are terminal, so calling this on every pump is idempotent.
    pub(crate) fn verify_run_ids(&mut self, expected: Option<RunId>) {
        // No metadata run id: the first hart that has one becomes the
        // reference, so a mixed directory is still caught.
        let reference = match expected.or_else(|| self.readers.values().find_map(|r| r.run_id())) {
            Some(id) => id,
            None => return,
        };

        for reader in self.readers.values_mut() {
            let Some(found) = reader.run_id() else {
                continue;
            };
            if found == reference {
                continue;
            }
            let reason = format!(
                "recorded by run {found:?}, but this spool is run {reference:?}; \
                 the log is left over from another run in the same directory, or \
                 the directory holds two runs' logs -- re-record, or process each \
                 run's own output directory"
            );
            reader.set_fault(reason);
        }
    }

    /// How completely each hart's log was read, plus the harts left out.
    ///
    /// Only meaningful once the producer has finished, for the same reason as
    /// [`warn_unfinished`](Self::warn_unfinished): until then, a log without a
    /// `Done` record is simply one still being written.
    pub(crate) fn integrity(&self) -> crate::common::TraceIntegrity {
        crate::common::TraceIntegrity {
            harts: self.readers.values().map(|r| r.status()).collect(),
            harts_excluded: self.excluded.clone(),
        }
    }

    /// The first hart whose log is unusable, with the reason.
    ///
    /// Hart-ordered rather than detection-ordered (`readers` is a `BTreeMap`),
    /// so the message a multi-hart run fails with is the same on every run.
    pub(crate) fn fault(&self) -> Option<(u32, &str)> {
        if let Some((hart, reason)) = &self.open_fault {
            return Some((*hart, reason.as_str()));
        }
        self.readers
            .values()
            .find_map(|r| r.fault().map(|f| (r.hart_id(), f)))
    }

    pub(crate) fn hart_count(&self) -> usize {
        self.readers.len()
    }

    pub(crate) fn message_count(&self) -> u32 {
        self.message_id
    }

    pub(crate) fn total_bytes(&self) -> usize {
        self.total_bytes
    }

    /// Warn about logs that stopped without a `Done` sentinel — the producer was
    /// killed, or the file was copied mid-write. Only meaningful once the
    /// producer has finished; until then this state is entirely normal.
    pub(crate) fn warn_unfinished(&self) {
        for reader in self.readers.values() {
            if reader.done() {
                continue;
            }
            let partial = reader.partial_tail_bytes();
            if partial > 0 {
                log::warn!(
                    "hart {} record log ends with a {}-byte partial record and no Done \
                     sentinel; the log was truncated mid-write and its tail is ignored",
                    reader.hart_id(),
                    partial
                );
            } else {
                log::warn!(
                    "hart {} record log ended without a Done record; drained anyway",
                    reader.hart_id()
                );
            }
        }
    }

    /// Harts present but not read, for reporting a deliberately partial run.
    pub(crate) fn excluded_harts(&self) -> &[u32] {
        &self.excluded
    }

    /// Why [`excluded_harts`](Self::excluded_harts) are not being read.
    pub(crate) fn exclusion_reason(&self) -> &'static str {
        self.exclusion_reason
    }

    pub(crate) fn dir(&self) -> &Path {
        &self.dir
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use turbo_tracer_shared::{BatchExtrasRef, RecordLogWriter};

    /// The run id every log in these tests claims. Shared so a test that means
    /// to exercise a MISMATCH has to say so explicitly.
    fn test_run_id() -> Option<RunId> {
        run_id_from(0xda7a)
    }

    /// A run id built from a number, so it cannot silently fail to be one --
    /// which would leave a test exercising the "no id" path instead.
    fn run_id_from(n: u64) -> Option<RunId> {
        Some(RunId::parse(&format!("run_{n:016x}")).expect("built in the one shape"))
    }

    fn tmp_dir(name: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("turbo_hart_spool_{name}"));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    fn write_hart_log(dir: &Path, hart: u32, batches: usize, done: bool) {
        write_hart_log_for_run(dir, hart, batches, done, test_run_id())
    }

    fn write_hart_log_for_run(
        dir: &Path,
        hart: u32,
        batches: usize,
        done: bool,
        run_id: Option<RunId>,
    ) {
        let path = dir.join(turbo_tracer_shared::get_turbo_trace_bin_for_vcpu(hart));
        let mut w = RecordLogWriter::create(&path, run_id).unwrap();
        for _ in 0..batches {
            w.append_batch(&[hart as u8; 8], BatchExtrasRef::EMPTY)
                .unwrap();
        }
        if done {
            w.finish().unwrap();
        } else {
            w.flush().unwrap();
        }
    }

    fn drain(spool: &mut HartSpool) -> Vec<TraceData> {
        let mut out = Vec::new();
        // Bounded pumping means one pass is not enough; loop to quiescence.
        for _ in 0..1000 {
            let more = spool.pump(&mut out);
            if !more && spool.all_spent() {
                break;
            }
        }
        out
    }

    /// A log from another run decodes perfectly and yields a profile resolved
    /// against the wrong addresses, so the run id in its header is the only
    /// thing that can catch it.
    #[test]
    fn a_log_from_another_run_is_refused() {
        let dir = tmp_dir("run_id_stale");
        write_hart_log(&dir, 0, 3, true);
        // Hart 1 left over from an earlier run in the same directory.
        write_hart_log_for_run(&dir, 1, 3, true, run_id_from(0xbead));

        let mut spool = HartSpool::new(dir.clone(), None);
        drain(&mut spool);
        spool.verify_run_ids(test_run_id());
        let (hart, reason) = spool.fault().expect("the stale log must be refused");
        assert_eq!(hart, 1, "the log that disagrees is the one named");
        assert!(reason.contains("000000000000bead"), "got: {reason}");

        // The same directory, checked with no reference run id at all (metadata
        // that records none): the harts are compared against each other, so the
        // mix is still caught.
        let mut spool = HartSpool::new(dir, None);
        drain(&mut spool);
        spool.verify_run_ids(None);
        assert!(
            spool.fault().is_some(),
            "two runs in one directory must be caught without a reference"
        );
    }

    /// The check must not fire on the ordinary case, including for a hart whose
    /// log appears mid-run.
    #[test]
    fn matching_run_ids_are_not_a_fault() {
        let dir = tmp_dir("run_id_match");
        write_hart_log(&dir, 0, 2, true);

        let mut spool = HartSpool::new(dir.clone(), None);
        drain(&mut spool);
        spool.verify_run_ids(test_run_id());
        assert!(spool.fault().is_none());

        // A hart that appears later is checked on the pump that adopts it.
        write_hart_log(&dir, 1, 2, true);
        drain(&mut spool);
        spool.verify_run_ids(test_run_id());
        assert!(spool.fault().is_none(), "a second hart of the same run");
        assert_eq!(spool.hart_count(), 2);
    }

    /// The trace-integrity block is the machine-readable answer to "do these
    /// numbers cover the whole run?", so each of its three answers is checked
    /// here rather than left to be read off a log line.
    #[test]
    fn integrity_reports_what_was_and_was_not_read() {
        // Whole: every hart read to its Done sentinel, none left out.
        let dir = tmp_dir("integrity_whole");
        write_hart_log(&dir, 0, 3, true);
        write_hart_log(&dir, 1, 2, true);
        let mut spool = HartSpool::new(dir, None);
        drain(&mut spool);
        let integrity = spool.integrity();
        assert!(integrity.is_complete(), "got: {integrity:?}");
        assert_eq!(integrity.harts.len(), 2);
        assert_eq!(integrity.harts[0].batches, 3);
        assert_eq!(integrity.harts[1].batches, 2);

        // Cut short: a log that stops mid-record is reported, with the size of
        // the tail that could not be decoded.
        let dir = tmp_dir("integrity_cut_short");
        write_hart_log(&dir, 0, 6, true);
        let path = dir.join(turbo_tracer_shared::get_turbo_trace_bin_for_vcpu(0));
        let bytes = std::fs::read(&path).unwrap();
        std::fs::write(&path, &bytes[..bytes.len() - 8]).unwrap();
        let mut spool = HartSpool::new(dir, None);
        drain(&mut spool);
        let integrity = spool.integrity();
        assert!(!integrity.is_complete(), "a cut-short log is not complete");
        assert_eq!(integrity.incomplete_harts(), vec![0]);

        // Partial by construction: `--vcpu` left harts unread.
        let dir = tmp_dir("integrity_filtered");
        write_hart_log(&dir, 0, 2, true);
        write_hart_log(&dir, 1, 2, true);
        write_hart_log(&dir, 2, 2, true);
        let mut spool = HartSpool::new(dir, Some(vec![1]));
        drain(&mut spool);
        let integrity = spool.integrity();
        assert!(
            !integrity.is_complete(),
            "a filtered run is not the whole run"
        );
        assert!(
            integrity.harts.iter().all(|h| h.complete),
            "the hart that WAS read is itself whole"
        );
        assert_eq!(integrity.harts_excluded, vec![0, 2], "sorted and reported");
    }

    /// A structurally broken log is reported as a fault, not quietly skipped:
    /// the run built from what survives would be a fraction of the program
    /// presented as all of it.
    #[test]
    fn a_broken_log_is_reported_as_a_fault() {
        let dir = tmp_dir("bad_magic");
        write_hart_log(&dir, 0, 3, true);
        let path = dir.join(turbo_tracer_shared::get_turbo_trace_bin_for_vcpu(0));

        // A healthy log has nothing to report.
        let mut spool = HartSpool::new(dir.clone(), None);
        drain(&mut spool);
        assert!(spool.fault().is_none(), "a complete log is not a fault");

        // Header clobbered: not a turbo record log at all.
        let mut bytes = std::fs::read(&path).unwrap();
        bytes[0] = 0xff;
        std::fs::write(&path, &bytes).unwrap();
        let mut spool = HartSpool::new(dir.clone(), None);
        drain(&mut spool);
        let (hart, reason) = spool.fault().expect("a bad magic header is a fault");
        assert_eq!(hart, 0);
        assert!(reason.contains("magic"), "got: {reason}");

        // A record body clobbered mid-stream: caught by that record's checksum,
        // and terminal either way -- the framing walk cannot get past a record
        // it cannot trust, so everything after it is lost.
        let mut bytes = std::fs::read(&path).unwrap();
        bytes[0] = b'T';
        let mid = bytes.len() / 2;
        bytes[mid..mid + 8].fill(0);
        std::fs::write(&path, &bytes).unwrap();
        let mut spool = HartSpool::new(dir, None);
        drain(&mut spool);
        let (_, reason) = spool.fault().expect("a corrupt record is a fault");
        assert!(reason.contains("fails its checksum"), "got: {reason}");
    }

    /// A log that stops mid-record is NOT a fault: a killed guest is the usual
    /// cause, and the prefix is real data worth reporting (flagged as partial
    /// by the trace-integrity block, and fatal only under `--strict-trace`).
    #[test]
    fn a_truncated_tail_is_not_a_fault() {
        let dir = tmp_dir("torn_tail");
        write_hart_log(&dir, 0, 6, true);
        let path = dir.join(turbo_tracer_shared::get_turbo_trace_bin_for_vcpu(0));
        let bytes = std::fs::read(&path).unwrap();
        std::fs::write(&path, &bytes[..bytes.len() / 2]).unwrap();

        let mut spool = HartSpool::new(dir, None);
        let out = drain(&mut spool);
        assert!(spool.fault().is_none(), "a torn tail is not a hard fault");
        assert!(!out.is_empty(), "the surviving prefix is still handed over");
    }

    /// The regression this whole type exists for: every hart in the directory is
    /// read, not just vCPU 0.
    #[test]
    fn reads_every_hart_in_the_directory() {
        let dir = tmp_dir("two_harts");
        write_hart_log(&dir, 0, 3, true);
        write_hart_log(&dir, 1, 4, true);

        let mut spool = HartSpool::new(dir, None);
        let out = drain(&mut spool);

        assert_eq!(spool.hart_count(), 2, "both harts discovered");
        let h0 = out.iter().filter(|d| d.hart_id() == 0).count();
        let h1 = out.iter().filter(|d| d.hart_id() == 1).count();
        assert_eq!((h0, h1), (3, 4), "every batch of both harts");
    }

    /// Live runs create hart logs as the guest spawns threads.
    #[test]
    fn picks_up_a_hart_that_appears_later() {
        let dir = tmp_dir("late_hart");
        write_hart_log(&dir, 0, 2, true);

        let mut spool = HartSpool::new(dir.clone(), None);
        let mut out = Vec::new();
        spool.pump(&mut out);
        assert_eq!(spool.hart_count(), 1);

        write_hart_log(&dir, 3, 2, true);
        for _ in 0..10 {
            spool.pump(&mut out);
        }

        assert_eq!(spool.hart_count(), 2, "late hart 3 discovered on a rescan");
        assert_eq!(out.iter().filter(|d| d.hart_id() == 3).count(), 2);
    }

    #[test]
    fn hart_filter_excludes_and_reports() {
        let dir = tmp_dir("filtered");
        write_hart_log(&dir, 0, 2, true);
        write_hart_log(&dir, 1, 2, true);
        write_hart_log(&dir, 2, 2, true);

        let mut spool = HartSpool::new(dir, Some(vec![1]));
        let out = drain(&mut spool);

        assert_eq!(spool.hart_count(), 1);
        assert!(out.iter().all(|d| d.hart_id() == 1));
        let mut excluded = spool.excluded_harts().to_vec();
        excluded.sort();
        assert_eq!(excluded, vec![0, 2], "dropped harts are reported");
    }

    /// One corrupt hart must not stop this type from pumping the others: the
    /// *run* fails (`SpoolSource` acts on `fault()`), but it fails having read
    /// every hart, so the report names the broken log rather than whichever
    /// hart happened to be pumped first.
    #[test]
    fn a_corrupt_hart_does_not_stop_the_others() {
        let dir = tmp_dir("corrupt_one");
        write_hart_log(&dir, 0, 3, true);
        std::fs::write(
            dir.join(turbo_tracer_shared::get_turbo_trace_bin_for_vcpu(1)),
            b"garbage-not-a-log",
        )
        .unwrap();

        let mut spool = HartSpool::new(dir, None);
        let out = drain(&mut spool);

        assert_eq!(spool.hart_count(), 2, "both logs opened");
        assert_eq!(
            out.iter().filter(|d| d.hart_id() == 0).count(),
            3,
            "hart 0 fully read despite hart 1 being unreadable"
        );
        assert_eq!(out.iter().filter(|d| d.hart_id() == 1).count(), 0);
        assert!(spool.all_spent());
        assert_eq!(
            spool.fault().map(|(hart, _)| hart),
            Some(1),
            "the broken hart is the one reported"
        );
    }

    #[test]
    fn interleaves_harts_rather_than_draining_one_at_a_time() {
        let dir = tmp_dir("interleave");
        // More batches than one pass takes from a single hart, so a
        // drain-hart-0-first implementation would show a long run of hart 0.
        write_hart_log(&dir, 0, BATCHES_PER_HART_PER_PUMP * 2, true);
        write_hart_log(&dir, 1, BATCHES_PER_HART_PER_PUMP * 2, true);

        let mut spool = HartSpool::new(dir, None);
        let out = drain(&mut spool);

        let first_h1 = out.iter().position(|d| d.hart_id() == 1).unwrap();
        assert!(
            first_h1 <= BATCHES_PER_HART_PER_PUMP,
            "hart 1 should appear within the first pass, not after all of hart 0 \
             (found at index {first_h1})"
        );
    }

    #[test]
    fn empty_directory_is_empty_and_done() {
        let dir = tmp_dir("empty");
        let mut spool = HartSpool::new(dir, None);
        let mut out = Vec::new();
        assert!(!spool.pump(&mut out));
        assert!(spool.is_empty());
        // Vacuously spent: there is nothing to wait for. Callers detect the
        // "no logs at all" case via is_empty().
        assert!(spool.all_spent());
        assert!(out.is_empty());
    }
}
