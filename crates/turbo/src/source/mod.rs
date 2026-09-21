//! Input trace sources and execution adapters.
//!
//! This module provides adapters for different trace input methods:
//! - [`qemu`]: QEMU execution with the turbo tracer plugin, memory mapping and cache plugin support
//! - [`record_log`]: the one incremental reader for a per-hart `turbo_trace_vcpu<N>.bin` log
//! - [`hart_spool`]: a directory of per-hart logs, tailed as one multi-hart unit
//! - [`spool_source`]: the one `TraceSource` over a spool, parameterised by producer
//!
//! The main [`TraceData`] enum encapsulates all trace data flowing through the system,
//! either as decoded instruction chunks or as metadata (memory map, shared libraries).

use turbo_tracer_shared::RoiRtMarker;

pub(crate) mod hart_spool;
pub(crate) mod qemu;
pub(crate) mod record_log;
pub(crate) mod spool_source;

pub use qemu::{
    resolve_qemu_binary, MemoryMap, QemuError, QemuResult, QemuRunner, VLEN_BITS_MAX, VLEN_BITS_MIN,
};
pub(crate) use record_log::hart_id_from_path;
pub(crate) use spool_source::{live_qemu, recorded_dir, recorded_single_log};

/// A pull-based source of [`TraceData`].
///
/// Both supported inputs are files on disk: the live path tails the per-hart
/// `turbo_trace_vcpu<N>.bin` record logs the QEMU plugin writes, and the offline path
/// reads a finished log. Neither needs a thread of its own — that was a
/// requirement of the older design, where trace data arrived over a Unix domain
/// socket whose blocking `recvmsg` had to be drained somewhere to avoid
/// backpressuring QEMU. Polling instead lets the whole pipeline (source, decode,
/// enrich) run on one thread with no channel between the stages.
pub trait TraceSource {
    /// Append every message available *right now* to `out`.
    ///
    /// Returns `Ok(true)` while more data may still arrive, and `Ok(false)` once
    /// the source is permanently exhausted. Appending nothing while returning
    /// `Ok(true)` is normal and expected (a live source whose spool has no new
    /// bytes yet), so callers should back off briefly before polling again in
    /// that case rather than spinning.
    ///
    /// A source may append messages on the same call that returns `Ok(false)`;
    /// callers must drain `out` before acting on the end of the stream.
    fn poll(&mut self, out: &mut Vec<TraceData>) -> Result<bool, crate::RerfError>;

    /// How completely the trace behind this source was read, once it is
    /// exhausted. Reported in `perf_data_final.json` so a caller can tell a
    /// whole run from a partially recovered one without reading the log.
    ///
    /// Defaulted rather than required: a source with no notion of per-hart
    /// completeness has nothing to say here, and "no evidence" is correctly
    /// read as not-complete.
    fn integrity(&self) -> crate::common::TraceIntegrity {
        crate::common::TraceIntegrity::default()
    }
}

/// Represents trace data from execution tracing
#[derive(Clone, Debug)]
pub enum TraceData {
    /// One batch of trace bytes from one hart.
    Trace {
        trace_data: Vec<u8>,
        message_id: u32,
        roi_markers: Vec<RoiRtMarker>,
        hart_id: u32,
    },
    /// Memory map ready; sent once, before any trace data.
    MemoryMapReady(MemoryMap),
    /// No memory map could be recovered for this trace: the consumer
    /// initializes against the main binary alone. Sent in place of
    /// [`TraceData::MemoryMapReady`], and likewise before any trace data.
    NoMemoryMap,
}

impl TraceData {
    pub fn hart_id(&self) -> u32 {
        match self {
            TraceData::Trace { hart_id, .. } => *hart_id,
            TraceData::MemoryMapReady(_) | TraceData::NoMemoryMap => 0,
        }
    }
}
