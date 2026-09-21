use std::path::Path;

/// Metadata about the trace, in JSON format
use serde::{Deserialize, Serialize};

/// File name (within the working directory) that the turbo tracer plugin writes its
/// metadata (main-binary meminfo + discovered shared-object mappings) to, as
/// human-readable JSON.
pub const TURBO_METADATA_FILE: &str = "turbo_metadata.json";

/// Filename (within the working directory) of the vDSO memory dump the plugin
/// captures from guest memory. The plugin reads the guest's vDSO region (located
/// via `qemu_plugin_get_vdso_info`) and writes the raw bytes here; the in-memory
/// vDSO is a complete ELF image, so consumers load it for symbol resolution.
pub const VDSO_DUMP_FILE: &str = "vdso_mem.dump";

/// Main-binary memory-map info obtained from the QEMU plugin API. All addresses
/// are guest runtime addresses (already relocated for PIE/ET_DYN binaries).
///
/// The module load bias needed to relocate ELF vaddrs is recoverable exactly as
/// `entry_code - e_entry` (with `e_entry` read from the ELF header): for a PIE
/// this yields the load base, and for a static ET_EXEC binary it yields 0.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MemMapInfo {
    /// Path to the main executable (from `qemu_plugin_path_to_binary`).
    pub binary_path: String,
    /// Runtime start of the main text segment (`qemu_plugin_start_code`).
    pub start_code: u64,
    /// Runtime end of the main text segment (`qemu_plugin_end_code`).
    pub end_code: u64,
    /// Runtime entry point of the main module (`qemu_plugin_entry_code`).
    pub entry_code: u64,
    /// Guest load address of the vDSO (`qemu_plugin_get_vdso_info`), or 0 if the
    /// target has no vDSO / the API is unavailable.
    pub vdso_addr: u64,
    /// Size of the vDSO region in bytes (`qemu_plugin_get_vdso_info`), or 0.
    pub vdso_size: u64,
}

/// One dynamically-loaded shared object: its guest path (as passed to `openat`)
/// and the runtime load base (the `mmap` return for the file-offset-0 segment,
/// == the module load bias). The consumer resolves the guest path to a host
/// path (sysroot prefix / `$ORIGIN` fallback) when loading the ELF.
///
/// Discovered by the turbo tracer plugin observing the guest's `openat`/`mmap`
/// syscalls — authoritative, and (unlike scraping LD_DEBUG stderr) it also
/// captures `dlopen`'d libraries. Does NOT include the dynamic linker itself or
/// the VDSO, which QEMU maps before the guest runs (no observable syscall).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LibMapping {
    pub path: String,
    pub base_address: u64,
}

/// Metadata the turbo tracer plugin emits for a run: the main-binary memory-map info
/// and the shared-object mappings discovered so far. Written as a single JSON
/// file so it is human-readable and there is one file to locate, rather than
/// several small binary sidecars.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TurboMetadata {
    /// Identifier of the run that wrote this file (see [`crate::RunId`]),
    /// passed to the QEMU plugin via its `run_id=` argument. This is used to
    /// ensure no stale metadata is being consumed when decoding traces.
    ///
    /// `None` -- absent, empty, or not an id -- means the plugin was invoked
    /// without one, i.e. by hand rather than by `turbo run`, and consumers skip
    /// the staleness check. Lenient by design: a run id is a hint, and losing
    /// it must not cost the caller the memory map recorded beside it. See
    /// [`crate::run_id::deserialize_optional`].
    #[serde(default, deserialize_with = "crate::run_id::deserialize_optional")]
    pub run_id: Option<crate::RunId>,
    /// VLEN in **bits**, as QEMU's `vlenb` CSR reported it. `u32`, not `u16`:
    /// the RVV maximum is 65536, which a `u16` cannot hold -- it used to be
    /// recorded as a saturated 65535, a VLEN no machine has.
    pub vlen_bits: u32,
    pub meminfo: Option<MemMapInfo>,
    pub libs: Vec<LibMapping>,
}

impl TurboMetadata {
    /// Serialize (pretty JSON) and write to `<workdir>/TURBO_METADATA_FILE`,
    /// atomically so a concurrent reader never observes a partial file.
    pub fn write_to_workdir(&self, workdir: &Path) -> std::io::Result<()> {
        let json = serde_json::to_string_pretty(self)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        let tmp = workdir.join(TURBO_METADATA_FILE).with_extension("tmp");
        std::fs::write(&tmp, json)?;
        std::fs::rename(&tmp, workdir.join(TURBO_METADATA_FILE))
    }

    /// Read and deserialize from `<workdir>/TURBO_METADATA_FILE`. Returns
    /// `None` if the file is absent (e.g. the plugin did not run) or
    /// unparseable.
    pub fn read_from_workdir(workdir: &Path) -> Option<Self> {
        let json = std::fs::read_to_string(workdir.join(TURBO_METADATA_FILE)).ok()?;
        // nosemgrep: deserialization-from-untrusted-source -- our own workdir file; a parse failure is just None
        serde_json::from_str(&json).ok()
    }
}
