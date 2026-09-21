//! Definitions shared between the QEMU tracer plugin and its consumers:
//! per-run identifiers, the side-file names and payloads the plugin emits, and
//! the ROI marker types. The trace formats themselves live in the submodules.

pub mod bbt;
pub mod inspector;
pub mod metadata;
pub mod record_log;
pub mod region_marker;
pub mod run_id;
pub mod vmem;

// Allow other crates to easily access these
pub use metadata::{LibMapping, MemMapInfo, TurboMetadata, TURBO_METADATA_FILE, VDSO_DUMP_FILE};
pub use region_marker::{RoiRtMarker, RoiRtMarkerKind};
pub use run_id::{RunId, RUN_ID_BYTES};

/// Re-exported at the crate root so every existing
/// `turbo_tracer_shared::<item>` path still resolves.
pub use record_log::*;

use std::path::PathBuf;

/// FNV-1a 64-bit hash of an arbitrary byte slice.
/// Constants: offset_basis = 0xcbf29ce484222325, prime = 0x100000001b3.
/// Avoids adding external dependancies e.g. sha256 cryptographic lib
pub fn fnv1a_hash_64(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf29ce484222325u64, |acc, &b| {
        (acc ^ b as u64).wrapping_mul(0x100000001b3)
    })
}

/// Returns the filename of the trace binary data for the given vCPU.
/// Note this is a file NAME only, NOT an absolute path!
/// You must .join() this with the directory you want to open it from
pub fn get_turbo_trace_bin_for_vcpu(vcpu: u32) -> PathBuf {
    PathBuf::from(format!("turbo_trace_vcpu{vcpu}.bin"))
}
