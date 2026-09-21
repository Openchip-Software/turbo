//! The one [`TraceSource`] over a hart spool, parameterised by what produces it.
//!
//! The live QEMU path and the offline `process` path used to be two separate
//! source implementations with two record-log decoders and two notions of
//! "finished". They are the same thing: tail a directory of per-hart record
//! logs until no more bytes can arrive. Only two things actually differ, and
//! both are now traits:
//!
//! * [`Producer`] — can more bytes still be appended? (a running QEMU child vs
//!   a directory that is already complete)
//! * [`MemoryMapProvider`] — where the address layout comes from (the prepared
//!   QEMU run vs the `turbo_metadata.json` side files on disk)
//!
//! Everything else — discovery, framing, bounded pumping, multi-hart
//! interleaving, the memory-map-before-trace-data handshake — is shared.

use std::path::PathBuf;
use std::time::Instant;

use turbo_tracer_shared::RunId;

use crate::source::hart_spool::HartSpool;
use crate::source::qemu::qemu_runner::{log_qemu_exit, PreparedQemuRun, QemuRunner};
use crate::source::qemu::{qemu_failure_message, MemoryMap};
use crate::source::{TraceData, TraceSource};
use crate::RerfError;

/// What decides whether more trace bytes can still arrive.
pub(crate) trait Producer {
    /// `Ok(true)` once nothing further will be written to the spool.
    fn finished(&mut self) -> Result<bool, RerfError>;
    /// Human-readable name for log messages.
    fn describe(&self) -> &'static str;
    /// Called when the consumer stops before the producer finished (decode
    /// error, enrichment failure, early return). Default: nothing to do.
    fn abandon(&mut self) {}
}

/// Where the run's address layout comes from.
pub(crate) trait MemoryMapProvider {
    /// `Ok(Some(map))` once available, `Ok(None)` while it may still show up.
    ///
    /// `producer_finished` distinguishes the two cases: while the producer runs,
    /// a failure just means "not written yet", but once it has finished the map
    /// must be there or the run cannot be processed at all.
    fn try_build(&mut self, producer_finished: bool) -> Result<Option<MemoryMap>, RerfError>;
}

/// A running QEMU child process.
pub(crate) struct QemuChild {
    child: std::process::Child,
    start: Instant,
    workdir: PathBuf,
    exit_logged: bool,
}

impl Producer for QemuChild {
    fn finished(&mut self) -> Result<bool, RerfError> {
        if self.exit_logged {
            return Ok(true);
        }
        match self.child.try_wait() {
            Ok(Some(status)) => {
                self.exit_logged = true;
                log_qemu_exit(self.start.elapsed().as_secs_f32(), &status);
                if !status.success() {
                    return Err(RerfError::trace_decoding_failed(qemu_failure_message(
                        &status,
                        &self.workdir,
                    )));
                }
                Ok(true)
            }
            Ok(None) => Ok(false),
            Err(e) => Err(RerfError::trace_decoding_failed(format!(
                "Failed to poll QEMU child process: {}",
                e
            ))),
        }
    }

    fn describe(&self) -> &'static str {
        "live QEMU"
    }

    /// Kill the child, so no orphaned QEMU keeps writing to a spool nobody reads.
    fn abandon(&mut self) {
        log::warn!("consumer stopped before QEMU exited; killing child process");
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

/// A spool that is already complete: nothing can be appended.
pub(crate) struct AlreadyFinished;

impl Producer for AlreadyFinished {
    fn finished(&mut self) -> Result<bool, RerfError> {
        Ok(true)
    }
    fn describe(&self) -> &'static str {
        "recorded spool"
    }
}

/// Memory map built from a prepared QEMU run's plugin output.
pub(crate) struct QemuPreparedMap {
    prepared: PreparedQemuRun,
}

impl MemoryMapProvider for QemuPreparedMap {
    fn try_build(&mut self, producer_finished: bool) -> Result<Option<MemoryMap>, RerfError> {
        // `build_memory_map(require_all_needed)`: while QEMU runs, insist every
        // needed side file is present (a partial map would resolve PCs against
        // the wrong bases); once it has exited, take what is there.
        match self.prepared.build_memory_map(!producer_finished) {
            Ok(map) => Ok(Some(map)),
            Err(err) if producer_finished => Err(RerfError::metadata_init_failed(format!(
                "No MemoryMap available after QEMU exit: {}. This likely means QEMU failed to run.",
                err
            ))),
            // Still running: the plugin has not written the map yet.
            Err(_) => Ok(None),
        }
    }
}

/// Memory map recovered from the `turbo_metadata.json` + `vdso_mem.dump` side
/// files a prior `record`/`run` wrote next to the trace logs.
pub(crate) struct RecordedDirMap {
    binary: PathBuf,
    dir: PathBuf,
    loader: PathBuf,
    /// Whether the "could not recover it" warning has been emitted.
    warned: bool,
}

impl RecordedDirMap {
    pub(crate) fn new(binary: PathBuf, dir: PathBuf) -> Self {
        Self {
            binary,
            dir,
            // `process` builds no QemuRunner, so there is no configured sysroot
            // to inherit; this matches QemuRunner's own default.
            loader: PathBuf::from("/usr/riscv64-linux-gnu/"),
            warned: false,
        }
    }
}

impl MemoryMapProvider for RecordedDirMap {
    fn try_build(&mut self, _producer_finished: bool) -> Result<Option<MemoryMap>, RerfError> {
        // No run of its own, so nothing to check the recording against: `None`
        // for run_id (skip the staleness check) and `None` for VLEN.
        // `resolve_trace_format` is what compares this recording's VLEN with the
        // invocation's configured one. `require_all_needed` is false: the guest
        // has already exited, so there is nothing left to wait for.
        match MemoryMap::from_qemu_output(&self.binary, &self.dir, &self.loader, None, None, false)
        {
            Ok(map) => {
                log::info!(
                    "Recovered memory-map info from {} in {:?}: {} object(s)",
                    turbo_tracer_shared::TURBO_METADATA_FILE,
                    self.dir,
                    map.objects.len(),
                );
                Ok(Some(map))
            }
            Err(e) => {
                if !self.warned {
                    self.warned = true;
                    log::warn!(
                        "dynamically-linked/shared library and VDSO info could not be recovered \
                         for {:?} ({e:#}). If 'AddressNotCovered' errors occur, it is likely due \
                         to this.",
                        self.dir
                    );
                }
                // No map, but the trace is still decodable against the main
                // binary alone. Hand back the legacy no-dynamic-info sentinel by
                // reporting "none available"; SpoolSource emits the sentinel.
                Ok(None)
            }
        }
    }
}

/// A hart spool as a pull-based [`TraceSource`].
pub(crate) struct SpoolSource<P: Producer, M: MemoryMapProvider> {
    spool: HartSpool,
    producer: P,
    map: M,
    memory_map_sent: bool,
    /// The run the memory map describes, once one has been built: the reference
    /// every record log's header run id is checked against (see
    /// [`HartSpool::verify_run_ids`]). `None` until a map has been built, for a
    /// run with no map at all, and for a map whose metadata carried no run id
    /// -- all three mean the same thing to the check, which then falls back to
    /// comparing the logs against each other.
    expected_run_id: Option<RunId>,
    /// Emitted instead of a real map when none can be recovered, so the
    /// consumer's startup handshake can still initialise `ElfMetadata`.
    fall_back_to_metadata_sentinel: bool,
    finished: bool,
}

impl<P: Producer, M: MemoryMapProvider> SpoolSource<P, M> {
    fn new(spool: HartSpool, producer: P, map: M, allow_sentinel: bool) -> Self {
        Self {
            spool,
            producer,
            map,
            memory_map_sent: false,
            expected_run_id: None,
            fall_back_to_metadata_sentinel: allow_sentinel,
            finished: false,
        }
    }

    /// Emit `MemoryMapReady` (or `NoMemoryMap`) as soon as the layout is known.
    /// Always precedes any Trace message.
    fn ensure_memory_map(
        &mut self,
        out: &mut Vec<TraceData>,
        producer_finished: bool,
    ) -> Result<(), RerfError> {
        if self.memory_map_sent {
            return Ok(());
        }
        match self.map.try_build(producer_finished)? {
            Some(map) => {
                log::info!(
                    "Sending {} MemoryMap: main base 0x{:x}, {} object(s)",
                    if producer_finished { "final" } else { "live" },
                    map.main_binary_base,
                    map.objects.len(),
                );
                // Kept before the map is handed over: it is the reference for
                // every log's run-id check below.
                self.expected_run_id = map.run_id;
                out.push(TraceData::MemoryMapReady(map));
                self.memory_map_sent = true;
            }
            None if producer_finished && self.fall_back_to_metadata_sentinel => {
                out.push(TraceData::NoMemoryMap);
                self.memory_map_sent = true;
            }
            // Producer still running: not written yet, try again next poll.
            None => {}
        }
        Ok(())
    }
}

impl<P: Producer, M: MemoryMapProvider> TraceSource for SpoolSource<P, M> {
    fn integrity(&self) -> crate::common::TraceIntegrity {
        self.spool.integrity()
    }

    fn poll(&mut self, out: &mut Vec<TraceData>) -> Result<bool, RerfError> {
        if self.finished {
            return Ok(false);
        }

        // Check the producer BEFORE pumping, so the pump below is guaranteed to
        // observe every record written before it finished. The other order could
        // see "still running", pump, and then have the producer append its last
        // records after the final pass had already happened.
        let producer_finished = match self.producer.finished() {
            Ok(f) => f,
            Err(e) => {
                // A failed producer is terminal; don't keep polling it.
                self.finished = true;
                return Err(e);
            }
        };

        self.ensure_memory_map(out, producer_finished)?;
        let more_available = if self.memory_map_sent {
            self.spool.pump(out)
        } else {
            false
        };

        // Every log that has revealed its header is checked against the run the
        // map describes. Done on each pump rather than once, because harts
        // appear as the guest spawns threads: a log adopted on pump 40 must be
        // checked too. Faults are terminal, so re-checking costs a comparison.
        self.spool.verify_run_ids(self.expected_run_id);

        // A structurally broken log ends the run, and does so as soon as the
        // break is reached rather than after enriching whatever preceded it.
        //
        // These faults all mean LOST TRACE DATA: a bad magic header (the file
        // is not a turbo record log), a record that will not decode (the rest
        // of that log is unreachable, since the framing walk cannot resynchronise
        // past it), or a log that cannot be read at all. Reporting a profile
        // built from the surviving prefix would be a partial answer presented
        // as a whole one -- silently, since the numbers look plausible. A
        // truncated *tail* is a different case and is deliberately NOT here:
        // see `HartSpool::warn_unfinished`.
        if let Some((hart, reason)) = self.spool.fault() {
            let err = RerfError::trace_decoding_failed(format!(
                "hart {hart} record log is unusable: {reason}"
            ));
            self.finished = true;
            return Err(err);
        }

        // Only finish when the producer is done AND every reader has handed over
        // everything it has. Bounded pumping means producer-exit alone is not
        // enough: a large recorded spool takes many passes to drain, and
        // stopping at exit would silently truncate the trace.
        if producer_finished && !more_available && self.spool.all_spent() {
            self.spool.warn_unfinished();
            let excluded = self.spool.excluded_harts();
            if !excluded.is_empty() {
                log::warn!(
                    "{} hart(s) present in the recording but NOT processed, due to {}: {:?}",
                    excluded.len(),
                    self.spool.exclusion_reason(),
                    excluded
                );
            }
            log::info!(
                "{} drained: {} hart(s), {} messages, {} bytes",
                self.producer.describe(),
                self.spool.hart_count(),
                self.spool.message_count(),
                self.spool.total_bytes()
            );
            self.finished = true;
            return Ok(false);
        }

        // A finished producer whose spool has no logs at all: nothing will ever
        // arrive, so report exhaustion rather than spinning. The consumer turns
        // this into a clear "QEMU produced no trace" error.
        if producer_finished && self.spool.is_empty() {
            log::error!(
                "no turbo_trace_vcpu<N>.bin record logs found in {}",
                self.spool.dir().display()
            );
            self.finished = true;
            return Ok(false);
        }

        Ok(true)
    }
}

impl<P: Producer, M: MemoryMapProvider> Drop for SpoolSource<P, M> {
    /// If the workflow bails out early (decode error, enrichment failure), tell
    /// the producer to stop rather than leaving it writing to an unread spool.
    fn drop(&mut self) {
        if !self.finished {
            self.producer.abandon();
        }
    }
}

/// Spawn QEMU and tail its spool live, so decode/enrich overlaps execution.
pub(crate) fn live_qemu(
    qemu_runner: &QemuRunner,
    binary_path: &str,
    binary_args: &[String],
    hart_filter: Option<Vec<u32>>,
) -> Result<SpoolSource<QemuChild, QemuPreparedMap>, String> {
    let prog_args_refs: Vec<&str> = binary_args.iter().map(|s| s.as_str()).collect();
    let prepared = qemu_runner
        .prepare_run(binary_path, &prog_args_refs, None)
        .map_err(|err| format!("Failed to prepare QEMU run: {}", err))?;

    let workdir = prepared.working_directory.clone();

    log::info!(
        "Starting QEMU with binary {}, tailing record logs in {}",
        binary_path,
        workdir.display()
    );

    let child = qemu_runner
        .spawn_prepared(&prepared)
        .map_err(|err| format!("Failed to spawn QEMU: {}", err))?;

    log::info!("Tailing QEMU spool live while QEMU runs...");

    Ok(SpoolSource::new(
        HartSpool::new(workdir.clone(), hart_filter),
        QemuChild {
            child,
            start: Instant::now(),
            workdir,
            exit_logged: false,
        },
        QemuPreparedMap { prepared },
        // The live path has the prepared run to build a real map from; falling
        // back to "no dynamic info" would silently mis-resolve shared-lib PCs.
        false,
    ))
}

/// Read an already-recorded spool directory: every hart, no waiting.
pub(crate) fn recorded_dir(
    dir: PathBuf,
    binary: PathBuf,
    hart_filter: Option<Vec<u32>>,
) -> SpoolSource<AlreadyFinished, RecordedDirMap> {
    SpoolSource::new(
        HartSpool::new(dir.clone(), hart_filter),
        AlreadyFinished,
        RecordedDirMap::new(binary, dir),
        // Offline: a trace moved away from its side files still decodes
        // against the main binary, with a warning.
        true,
    )
}

/// Read one explicitly-named record log. Its siblings are reported but not read.
pub(crate) fn recorded_single_log(
    path: &std::path::Path,
    hart_id: u32,
    binary: PathBuf,
) -> SpoolSource<AlreadyFinished, RecordedDirMap> {
    let spool = HartSpool::single_log(path, hart_id);
    let dir = spool.dir().to_path_buf();
    SpoolSource::new(
        spool,
        AlreadyFinished,
        RecordedDirMap::new(binary, dir),
        true,
    )
}
