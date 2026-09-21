use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::process::{Command, ExitStatus, Output, Stdio};
use std::time::Instant;

use turbo_tracer_shared::RunId;

use crate::common::SYSTEM_VLEN;
use crate::{get_build_artifact_dir, workspace_root};

// ============================================================================
// Error Types
// ============================================================================

/// Errors that can occur during QEMU execution
#[derive(Debug, thiserror::Error)]
pub enum QemuError {
    #[error("Binary not found: {0}")]
    BinaryNotFound(PathBuf),

    /// The QEMU binary itself could not be found. Its own variant, not a
    /// [`QemuError::BinaryNotFound`], because the fix is in the configuration
    /// rather than on the command line, and the message says where.
    #[error("{}", qemu_not_found_message(.0))]
    QemuNotFound(PathBuf),

    #[error("Failed to create directory {path}: {source}")]
    DirectoryCreation { path: PathBuf, source: io::Error },

    #[error("Failed to resolve path {path}: {source}")]
    PathResolution { path: PathBuf, source: io::Error },

    #[error("Failed to execute QEMU binary {qemu}: {source}")]
    ExecutionFailed { qemu: PathBuf, source: io::Error },

    #[error("{report}")]
    ProcessFailed {
        exit_code: i32,
        stderr: String,
        report: String,
    },

    #[error("Invalid configuration: {0}")]
    InvalidConfiguration(String),

    /// The guest needs a dynamic linker that the configured sysroot does not
    /// contain. Its own variant, not an `InvalidConfiguration`, because it is
    /// the one configuration mistake that cannot be diagnosed from the message
    /// QEMU would print: QEMU dies in its ELF loader before the plugin sees a
    /// vCPU, so everything downstream only ever sees zeroed metadata.
    #[error("{0}")]
    MissingSysroot(String),

    /// A field of `environment_config.json` that this run needs is unset.
    #[error("{0}")]
    MissingConfigValue(String),
}

pub type Result<T> = std::result::Result<T, QemuError>;

fn qemu_not_found_message(qemu: &Path) -> String {
    if is_bare_command_name(qemu) {
        format!(
            "QEMU binary `{}` was not found on PATH. Install QEMU 11.0.x built with \
             --enable-plugins, or set `qemu_bin` in environment_config.json to its \
             full path: `turbo config init` writes that file, and `turbo config check` \
             verifies it.",
            qemu.display()
        )
    } else {
        format!(
            "QEMU binary {} does not exist or is not executable. Fix `qemu_bin` in \
             environment_config.json, then verify it with `turbo config check`.",
            qemu.display()
        )
    }
}

/// The execlog reference trace (`--reference-traces`) was asked for, but there
/// is no directory to load QEMU's `libexeclog.so` from.
fn missing_qemu_plugin_dir() -> QemuError {
    QemuError::MissingConfigValue(
        "--reference-traces needs QEMU's contrib plugins: build them with \
         `ninja -C build contrib-plugins` in the QEMU tree, and set `qemu_plugin_dir` \
         in environment_config.json to <qemu>/build/contrib/plugins"
            .to_string(),
    )
}

/// A single path component with no directory separator, i.e. something the
/// shell (and [`Command::new`]) would look up on `PATH` rather than treat as a
/// path relative to the current directory.
fn is_bare_command_name(path: &Path) -> bool {
    let mut components = path.components();
    matches!(
        (components.next(), components.next()),
        (Some(std::path::Component::Normal(_)), None)
    )
}

fn is_executable_file(path: &Path) -> bool {
    use std::os::unix::fs::PermissionsExt;
    fs::metadata(path)
        .map(|meta| meta.is_file() && meta.permissions().mode() & 0o111 != 0)
        .unwrap_or(false)
}

/// The first executable `name` in the directories of `search_path`, in the
/// format of the `PATH` environment variable.
fn find_in_search_path(name: &Path, search_path: &std::ffi::OsStr) -> Option<PathBuf> {
    std::env::split_paths(search_path)
        // An empty `PATH` entry means the current directory, as it does to the
        // shell.
        .map(|dir| {
            if dir.as_os_str().is_empty() {
                PathBuf::from(".")
            } else {
                dir
            }
        })
        // nosemgrep: input-validation-path-traversal-2 -- PATH lookup, mirroring what Command::new executes
        .map(|dir| dir.join(name))
        .find(|candidate| is_executable_file(candidate))
}

/// Resolve the configured QEMU binary to the file that would be executed.
///
/// A bare name like the default `qemu-riscv64` is looked up on `PATH`, the way
/// [`Command::new`] will run it; anything with a directory in it is taken as a
/// path. Checking a bare name with `Path::exists()` instead looks in the
/// current directory, which rejects a QEMU that is installed and on `PATH` --
/// exactly the setup a source build with no config file relies on.
pub fn resolve_qemu_binary(qemu: &Path) -> Result<PathBuf> {
    let found = if is_bare_command_name(qemu) {
        std::env::var_os("PATH").and_then(|path| find_in_search_path(qemu, &path))
    } else {
        is_executable_file(qemu).then(|| qemu.to_path_buf())
    };
    found.ok_or_else(|| QemuError::QemuNotFound(qemu.to_path_buf()))
}

// ============================================================================
// Result Types
// ============================================================================

/// Result of a QEMU execution
#[derive(Debug)]
pub struct QemuResult {
    pub output: Output,
    pub exit_code: i32,
}

impl Default for QemuResult {
    fn default() -> Self {
        Self {
            output: Output {
                status: ExitStatus::default(),
                stdout: vec![],
                stderr: vec![],
            },
            exit_code: 0,
        }
    }
}

impl QemuResult {
    /// Check if the execution was successful (exit code 0)
    pub fn success(&self) -> bool {
        self.output.status.success()
    }
}

// ============================================================================
// Configuration Types
// ============================================================================

/// Smallest VLEN the RVV v1.0 specification permits for the `V` extension, in
/// bits. QEMU enforces the same floor (`riscv_cpu_validate_v`).
pub const VLEN_BITS_MIN: u32 = 128;

/// Largest VLEN the RVV v1.0 specification permits, in bits. Matches QEMU's own
/// `RV_VLEN_MAX`, and is the reason every VLEN-in-bits value is a `u32`: 65536
/// is one past the top of a `u16`.
pub const VLEN_BITS_MAX: u32 = 65536;

/// Configuration for QEMU CPU settings
#[derive(Debug, Clone, PartialEq)]
pub struct CpuConfig {
    /// CPU specification (e.g., "rv64")
    pub arch: String,
    /// Enable vector extensions
    pub vector_enabled: bool,
    /// Vector extension specification version
    pub vext_spec: String,
    /// Vector length in bits. `u32` because the RVV maximum, 65536, does not
    /// fit a `u16`; see [`VLEN_BITS_MAX`].
    pub vlen: u32,
}

impl Default for CpuConfig {
    fn default() -> Self {
        Self {
            arch: "max".to_string(),
            vector_enabled: true,
            vext_spec: "v1.0".to_string(),
            vlen: crate::common::DEFAULT_VLEN_BITS,
        }
    }
}

impl CpuConfig {
    /// Build the QEMU CPU specification string
    pub(crate) fn to_qemu_arg(&self) -> String {
        let arg = format!(
            "{},v={},vext_spec={},vlen={}",
            self.arch, self.vector_enabled, self.vext_spec, self.vlen
        );
        log::debug!("vlen: QEMU -cpu {arg}");
        arg
    }
}

/// Configuration for QEMU runtime paths
#[derive(Debug, Clone)]
pub struct PathConfig {
    /// Path to QEMU binary
    pub qemu_binary: PathBuf,
    /// Loader library path (e.g., "/usr/riscv64-linux-gnu/")
    pub loader_path: PathBuf,
    /// Working directory for QEMU execution
    pub work_dir: PathBuf,
}

impl Default for PathConfig {
    fn default() -> Self {
        Self {
            qemu_binary: PathBuf::from("qemu-riscv64"),
            loader_path: PathBuf::from("/usr/riscv64-linux-gnu/"),
            work_dir: workspace_root().join("target"),
        }
    }
}

/// Configuration for QEMU plugins
#[derive(Debug, Clone)]
pub struct PluginConfig {
    /// Enable turbo tracer plugin
    pub tracer_enabled: bool,
    /// Path to turbo tracer plugin (optional, uses default if None)
    pub tracer_plugin_path: Option<PathBuf>,
    /// Output file name for the turbo tracer
    pub tracer_output_file: String,

    /// Enable execution log plugin
    pub execlog_enabled: bool,
    /// Path to qemu plugin directory
    pub plugin_dir: Option<PathBuf>,
    /// Output file name for execlog
    pub execlog_output_file: String,
}

impl Default for PluginConfig {
    fn default() -> Self {
        Self {
            tracer_enabled: true,
            tracer_plugin_path: None,
            tracer_output_file: "turbo_trace_output.bin".to_string(),
            execlog_enabled: false,
            plugin_dir: None,
            execlog_output_file: "execlog_out.txt".to_string(),
        }
    }
}

impl PluginConfig {
    /// The tracer plugin QEMU will load: the configured one, else the one this
    /// checkout's cargo build produced.
    pub fn tracer_plugin_path(&self) -> PathBuf {
        self.tracer_plugin_path
            .clone()
            .unwrap_or_else(|| get_build_artifact_dir().join("libturbo_tracer.so"))
    }
}

#[derive(Debug, Clone)]
pub(crate) struct PreparedQemuRun {
    pub(crate) binary_full_path: PathBuf,
    pub(crate) program_args: Vec<String>,
    pub(crate) working_directory: PathBuf,
    pub(crate) bin_work_dir: PathBuf,
    pub(crate) qemu_args: Vec<String>,
    pub(crate) loader_path: PathBuf,
    /// Opaque identifier of this run (see [`turbo_tracer_shared::RunId`]),
    /// passed to the turbo tracer plugin via its `run_id=` argument and cross-checked
    /// against the metadata file it writes, so a stale leftover from a
    /// previous run in the same work_dir can never be mistaken for this run's.
    pub(crate) run_id: RunId,
    /// VLEN in bits this run asked QEMU for, cross-checked against the `vlenb`
    /// the plugin actually read: QEMU silently clamps a `vlen=` its build cannot
    /// provide, so the request is not evidence it was honoured.
    pub(crate) vlen_bits: u32,
}

impl PreparedQemuRun {
    pub(crate) fn build_memory_map(
        &self,
        require_all_needed: bool,
    ) -> anyhow::Result<crate::source::qemu::memory_map::MemoryMap> {
        crate::source::qemu::memory_map::MemoryMap::from_qemu_output(
            &self.binary_full_path,
            &self.working_directory,
            &self.loader_path,
            Some(self.run_id),
            Some(self.vlen_bits),
            require_all_needed,
        )
    }
}

// ============================================================================
// Helper Functions
// ============================================================================

/// Build QEMU command line arguments from configuration
fn build_qemu_args(
    cpu_config: &CpuConfig,
    path_config: &PathConfig,
    plugin_config: &PluginConfig,
    working_dir: &Path,
    run_id: RunId,
    guest_env: &[(String, String)],
) -> Result<Vec<String>> {
    let mut args = Vec::new();

    // CPU configuration
    args.push("-cpu".to_string());
    args.push(cpu_config.to_qemu_arg());

    // Guest environment variables requested by the caller (e.g.
    // LD_LIBRARY_PATH), passed to the guest via QEMU's `-E NAME=VALUE`.
    for (name, value) in guest_env {
        args.push("-E".into());
        args.push(format!("{name}={value}"));
    }

    // OpenMP config:
    if let Ok(omp_n_t) = std::env::var("OMP_NUM_THREADS") {
        if let Ok(t) = omp_n_t.parse::<usize>() {
            args.push("-E".into());
            args.push(format!("OMP_NUM_THREADS={t}"));
            log::info!(
                "detected OMP_NUM_THREADS variable in ENV, passing through to app, threads =  {}",
                t
            )
        } else {
            log::error!("your ENV variables have OMP_NUM_THREADS={} but this is not a valid value - ignoring this ENV var!", omp_n_t);
        }
    }

    // Loader path
    args.push("-L".to_string());
    args.push(path_config.loader_path.display().to_string());

    // turbo tracer plugin
    if plugin_config.tracer_enabled {
        let plugin_path = plugin_config.tracer_plugin_path();
        // No `vl_verify=` override: the plugin's default is `abort`, and it is
        // now correct to leave it there. It used to be forced to `warn` because
        // a fault-only-first vector load changed `vl` behind the oracle's back
        // and every guest using one tripped it; that `vl` is captured now
        // too, so a divergence is once again a real bug.
        let plugin_arg = format!(
            "{},workdir={},run_id={}",
            plugin_path.display(),
            working_dir.display(),
            run_id
        );

        args.push("-plugin".to_string());
        args.push(plugin_arg);
    }

    // Execlog plugin
    if plugin_config.execlog_enabled {
        let plugin_path = plugin_config
            .plugin_dir
            .clone()
            .ok_or_else(missing_qemu_plugin_dir)?
            .join("libexeclog.so");

        args.push("-D".to_string());
        args.push(plugin_config.execlog_output_file.clone());
        args.push("-plugin".to_string());
        args.push(plugin_path.display().to_string());

        // Plugin debug output and page layout for VDSO capture.
        args.push("-d".to_string());
        args.push("plugin".to_string());
    }

    Ok(args)
}

/// Inspect the target binary and environment to guess *why* QEMU failed, so the
/// user gets an actionable hint rather than just an exit code. QEMU-user is
/// notoriously terse: e.g. a target ELF that is missing its execute bit makes
/// QEMU exit 1 while printing nothing but `host mmap_min_addr=...`.
fn diagnose_binary_failure(binary: &Path) -> Vec<String> {
    use std::os::unix::fs::PermissionsExt;

    let mut hints = Vec::new();

    match fs::metadata(binary) {
        Ok(meta) => {
            let mode = meta.permissions().mode();
            if mode & 0o111 == 0 {
                hints.push(format!(
                    "The target binary is not executable (mode {:o}). QEMU-user requires \
                     the execute bit; run `chmod +x {}` and retry.",
                    mode & 0o777,
                    binary.display()
                ));
            }
            if meta.len() == 0 {
                hints.push("The target binary is empty (0 bytes).".to_string());
            }
        }
        Err(e) => {
            hints.push(format!(
                "Could not stat the target binary {}: {}",
                binary.display(),
                e
            ));
        }
    }

    // Sanity-check the ELF header / architecture — a host-arch or non-ELF file
    // handed to qemu-riscv64 fails in confusing ways.
    if let Ok(bytes) = fs::read(binary) {
        if bytes.len() < 20 || &bytes[0..4] != b"\x7fELF" {
            hints.push("The target file does not look like an ELF binary.".to_string());
        } else {
            // e_machine is a u16 at offset 18 (little-endian for our targets).
            let e_machine = u16::from_le_bytes([bytes[18], bytes[19]]);
            const EM_RISCV: u16 = 243;
            if e_machine != EM_RISCV {
                hints.push(format!(
                    "The target ELF is not RISC-V (e_machine={}); qemu-riscv64 cannot run it.",
                    e_machine
                ));
            }
        }
    }

    hints
}

// ============================================================================
// QEMU Runner
// ============================================================================

/// High-level interface for running RISC-V binaries with QEMU
#[derive(Debug, Clone)]
pub struct QemuRunner {
    cpu_config: CpuConfig,
    path_config: PathConfig,
    plugin_config: PluginConfig,
    bin_work_path: Option<PathBuf>,
    /// Environment variables to set for the *guest* program (via QEMU's `-E`).
    guest_env: Vec<(String, String)>,
}

impl Default for QemuRunner {
    fn default() -> Self {
        Self::new()
    }
}

impl QemuRunner {
    /// Create a new QEMU runner with default configuration
    pub fn new() -> Self {
        Self {
            cpu_config: CpuConfig::default(),
            path_config: PathConfig::default(),
            plugin_config: PluginConfig::default(),
            bin_work_path: None,
            guest_env: Vec::new(),
        }
    }

    // ========================================================================
    // Configuration Builders
    // ========================================================================

    /// Set CPU configuration
    pub fn with_cpu_config(mut self, config: CpuConfig) -> Self {
        self.cpu_config = config;
        self
    }

    /// Set path configuration
    pub fn with_path_config(mut self, config: PathConfig) -> Self {
        self.path_config = config;
        self
    }

    /// Set plugin configuration
    pub fn with_plugin_config(mut self, config: PluginConfig) -> Self {
        self.plugin_config = config;
        self
    }

    /// Set QEMU binary path
    pub fn with_qemu_binary(mut self, path: PathBuf) -> Self {
        self.path_config.qemu_binary = path;
        self
    }

    /// Set working directory
    pub fn with_work_dir(mut self, dir: PathBuf) -> Self {
        self.path_config.work_dir = dir;
        self
    }

    pub fn with_plugin_dir(mut self, path: PathBuf) -> Self {
        self.plugin_config.plugin_dir = Some(path);
        self
    }

    /// Set the loader library path (QEMU `-L` sysroot, e.g. "/usr/riscv64-linux-gnu/")
    pub fn with_loader_path(mut self, path: PathBuf) -> Self {
        self.path_config.loader_path = path;
        self
    }

    /// The directory that the command being run is run from, aka "pwd" for the profiled command
    pub fn with_bin_work_dir(mut self, bin_work_path: PathBuf) -> Self {
        self.bin_work_path = Some(bin_work_path);
        self
    }

    /// Set an environment variable for the *guest* program, passed to QEMU as
    /// `-E NAME=VALUE`. This is how a dynamically-linked guest is told where its
    /// shared libraries live (`LD_LIBRARY_PATH`), since QEMU-user does not
    /// forward the host's environment to the guest by default.
    pub fn with_guest_env(mut self, name: impl Into<String>, value: impl Into<String>) -> Self {
        self.guest_env.push((name.into(), value.into()));
        self
    }

    pub fn with_tracer_plugin_path(mut self, path: PathBuf) -> Self {
        self.plugin_config.tracer_plugin_path = Some(path);
        self
    }

    // ========================================================================
    // Convenience Builders (for backward compatibility)
    // ========================================================================

    /// Vector length (VLEN) in bits this runner will pass to QEMU.
    pub fn vlen_bits(&self) -> u32 {
        self.cpu_config.vlen
    }

    /// Set vector length (VLEN)
    pub fn vlen(mut self, vlen: u32) -> Self {
        self.cpu_config.vlen = vlen;
        self
    }

    /// Enable or disable turbo tracer plugin
    pub fn tracer(mut self, enabled: bool) -> Self {
        self.plugin_config.tracer_enabled = enabled;
        self
    }

    /// Enable or disable execlog plugin
    pub fn execlog(mut self, enabled: bool) -> Self {
        if enabled {
            log::warn!(
                "Execlog is enabled, this causes HUGE (10GB+!) trace files for long running binaries!!"
            );
        }
        self.plugin_config.execlog_enabled = enabled;
        self
    }

    // ========================================================================
    // Getters
    // ========================================================================

    /// Check if the turbo tracer plugin is enabled
    pub fn get_tracer_enabled(&self) -> bool {
        self.plugin_config.tracer_enabled
    }

    /// Check if execlog is enabled
    pub fn get_execlog(&self) -> bool {
        self.plugin_config.execlog_enabled
    }

    /// Get the working directory
    pub fn work_dir(&self) -> &PathBuf {
        &self.path_config.work_dir
    }

    /// Get the CPU configuration
    pub fn cpu_config(&self) -> &CpuConfig {
        &self.cpu_config
    }

    /// Get the path configuration
    pub fn path_config(&self) -> &PathConfig {
        &self.path_config
    }

    /// Get the plugin configuration
    pub fn plugin_config(&self) -> &PluginConfig {
        &self.plugin_config
    }

    // ========================================================================
    // Validation
    // ========================================================================

    /// Validate configuration before running
    fn validate(&self) -> Result<()> {
        let qemu = resolve_qemu_binary(&self.path_config.qemu_binary)?;
        log::debug!("QEMU binary resolved to {}", qemu.display());

        // Check execlog configuration
        if self.plugin_config.execlog_enabled && self.plugin_config.plugin_dir.is_none() {
            return Err(missing_qemu_plugin_dir());
        }

        // Validate VLEN against what the RVV specification actually allows.
        // The whole range is now checked here rather than half-implied by the
        // field's width: while VLEN was a `u16` the upper bound was enforced by
        // the type (badly -- it excluded the legal maximum), and an out-of-range
        // value reached QEMU to be rejected there, several hundred lines of
        // launch later and in QEMU's words rather than ours.
        let vlen = self.cpu_config.vlen;
        if !vlen.is_power_of_two() || !(VLEN_BITS_MIN..=VLEN_BITS_MAX).contains(&vlen) {
            return Err(QemuError::InvalidConfiguration(format!(
                "Invalid VLEN value: {vlen}. Must be a power of two in \
                 [{VLEN_BITS_MIN}, {VLEN_BITS_MAX}] bits"
            )));
        }

        Ok(())
    }

    pub(crate) fn prepare_run(
        &self,
        binary_path: &str,
        prog_args: &[&str],
        work_dir: Option<&Path>,
    ) -> Result<PreparedQemuRun> {
        self.validate()?;

        log::debug!("vlen: SYSTEM_VLEN set to {} bits", self.cpu_config.vlen);
        // `validate()` above bounds VLEN at 65536 bits, i.e. 8192 bytes, so the
        // narrowing cannot lose anything.
        SYSTEM_VLEN.set_bytes((self.cpu_config.vlen / 8) as u16);

        let binary_full_path = PathBuf::from(binary_path);

        if !binary_full_path.exists() {
            return Err(QemuError::BinaryNotFound(binary_full_path));
        }

        let binary_full_path =
            binary_full_path
                .canonicalize()
                .map_err(|e| QemuError::ExecutionFailed {
                    qemu: self.path_config.qemu_binary.clone(),
                    source: e,
                })?;

        // Warn early about likely-fatal issues with the target binary (e.g.
        // missing execute bit, wrong architecture) so the user gets an
        // actionable message before QEMU runs and fails opaquely.
        for hint in diagnose_binary_failure(&binary_full_path) {
            log::warn!("target binary issue: {}", hint);
        }

        // A dynamically-linked guest whose interpreter is not under `-L` cannot
        // run, full stop: QEMU exits inside its ELF loader having initialised no
        // vCPU, so the tracer plugin writes metadata full of zeroes and the run
        // fails later as a VLEN of 0 / absent meminfo -- symptoms several layers
        // away from the missing sysroot that caused them. Refuse here instead,
        // while the sysroot is still in hand and nothing has been spawned.
        if let Some(missing) = crate::source::qemu::MissingInterpreter::detect(
            &binary_full_path,
            &self.path_config.loader_path,
        ) {
            return Err(QemuError::MissingSysroot(missing.report()));
        }

        let working_directory = work_dir.unwrap_or(&self.path_config.work_dir);
        let working_directory = if working_directory.is_absolute() {
            working_directory.to_path_buf()
        } else {
            // nosemgrep: input-validation-path-traversal-2 -- only reached when the path is relative, and it is canonicalized below
            std::env::current_dir()
                .map_err(|e| QemuError::PathResolution {
                    path: working_directory.to_path_buf(),
                    source: e,
                })?
                .join(working_directory)
        };

        if !working_directory.exists() {
            fs::create_dir_all(&working_directory).map_err(|e| QemuError::DirectoryCreation {
                path: working_directory.to_path_buf(),
                source: e,
            })?;
        }
        let working_directory =
            working_directory
                .canonicalize()
                .map_err(|e| QemuError::PathResolution {
                    path: working_directory.clone(),
                    source: e,
                })?;

        let run_id = turbo_tracer_shared::RunId::generate();

        let qemu_args = build_qemu_args(
            &self.cpu_config,
            &self.path_config,
            &self.plugin_config,
            &working_directory,
            run_id,
            &self.guest_env,
        )?;

        let program_args: Vec<String> = prog_args.iter().map(|s| s.to_string()).collect();

        let bin_work_dir = self
            .bin_work_path
            .clone()
            .unwrap_or_else(|| working_directory.clone());

        Ok(PreparedQemuRun {
            binary_full_path,
            program_args,
            working_directory,
            bin_work_dir,
            qemu_args,
            loader_path: self.path_config.loader_path.clone(),
            run_id,
            // The value that went into `-cpu ...,vlen=`, so the check is against
            // what was asked of QEMU rather than against SYSTEM_VLEN, which this
            // same function has just set from it.
            vlen_bits: self.cpu_config.vlen,
        })
    }

    // ========================================================================
    // Main Execution Method
    // ========================================================================

    /// Run a RISC-V binary with QEMU
    ///
    /// # Arguments
    /// * `binary_path` - Path to the RISC-V binary relative to workspace root
    /// * `prog_args` - Additional arguments to pass to the binary
    /// * `work_dir` - Optional working directory override (uses configured if None)
    ///
    /// # Returns
    /// Result containing QemuResult with stdout, stderr, and exit code
    ///
    /// # Errors
    /// Returns QemuError if:
    /// - Binary not found
    /// - Configuration is invalid
    /// - QEMU execution fails
    /// - Required environment variables are missing
    pub fn run(
        &self,
        binary_path: &str,
        prog_args: &[&str],
        work_dir: Option<&Path>,
    ) -> Result<QemuResult> {
        let prepared = self.prepare_run(binary_path, prog_args, work_dir)?;
        self.run_prepared(prepared)
    }

    /// Run a prepared QEMU invocation to completion, blocking until QEMU exits.
    ///
    /// The live-trace path does NOT use this: it calls [`Self::spawn_prepared`]
    /// and polls the child with `try_wait()` so it can tail the spool from the
    /// same thread instead of parking one here.
    pub(crate) fn run_prepared(&self, prepared: PreparedQemuRun) -> Result<QemuResult> {
        let mut child = self.spawn_prepared(&prepared)?;
        let start = Instant::now();
        let status = child.wait().map_err(|e| QemuError::ExecutionFailed {
            qemu: self.path_config.qemu_binary.clone(),
            source: e,
        })?;
        log_qemu_exit(start.elapsed().as_secs_f32(), &status);

        Ok(QemuResult {
            output: Output {
                status,
                stdout: vec![],
                stderr: vec![],
            },
            exit_code: 0,
        })
    }

    /// Spawn a prepared QEMU invocation WITHOUT waiting for it.
    ///
    /// Performs the same pre-run cleanup as [`Self::run_prepared`] (stale VDSO
    /// dumps, stale metadata file, stale per-hart `.bin` record logs) and returns
    /// the live [`Child`] so the caller can poll it with `try_wait()`.
    ///
    /// stdout is inherited; stderr is redirected to `qemu_stderr.txt` in the
    /// working directory. It must not be a pipe we do not drain: a full pipe
    /// would block QEMU forever, since with `try_wait()` nobody is reading it.
    pub(crate) fn spawn_prepared(&self, prepared: &PreparedQemuRun) -> Result<std::process::Child> {
        log::debug!("Starting QEMU run with config: {:?}", self);

        log::info!(
            "Running {} {} {} {}",
            self.path_config.qemu_binary.display(),
            prepared.qemu_args.join(" "),
            prepared.binary_full_path.display(),
            prepared.program_args.join(" ")
        );

        let ld_lib_path = self
            .path_config
            .qemu_binary
            .parent()
            .map(|p| p.display().to_string())
            .unwrap_or_default();

        log::debug!("LD_LIBRARY_PATH = {}", ld_lib_path);

        // the "binary working dir" is where the program being profiled is running
        // the working_directory is where we want to place our output files
        // - TODO/bug: QEMU will likely write stuff to the "binary working dir.."?
        log::debug!("profiled binary working dir is {:?}", prepared.bin_work_dir);

        // Clear the plugin's side files from any previous run in the same
        // (uncleaned) work_dir. The live-trace path polls for the metadata file
        // to become readable as its memory-map readiness signal, so a leftover
        // would be picked up immediately with a completely wrong (and
        // self-consistent, so undetectable) address layout; the vDSO dump has no
        // run_id of its own, so removing it is its only staleness protection.
        for name in [
            turbo_tracer_shared::TURBO_METADATA_FILE,
            turbo_tracer_shared::VDSO_DUMP_FILE,
        ] {
            // nosemgrep: input-validation-path-traversal-2 -- canonicalized workdir joined with a const from the literal array above
            let path = prepared.working_directory.join(name);
            match fs::remove_file(&path) {
                Ok(()) => log::debug!("Cleared stale {name} before QEMU run"),
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
                Err(e) => log::warn!("Failed to remove stale {:?}: {e}", path),
            }
        }

        // The plugin writes each hart's trace directly to a per-vCPU
        // `turbo_trace_vcpu<N>.bin` record-log file in the working directory (via
        // RecordLogWriter, which truncates on create). We only clear STALE bin
        // files from a previous run up front, so a run with fewer harts than
        // last time cannot leave an orphaned higher-numbered file that a reader
        // would later pick up.
        if self.plugin_config.tracer_enabled {
            for entry in std::fs::read_dir(&prepared.working_directory)
                .into_iter()
                .flatten()
                .flatten()
            {
                let p = entry.path();
                if p.file_name()
                    .and_then(|n| n.to_str())
                    .map(|n| n.starts_with("turbo_trace_vcpu") && n.ends_with(".bin"))
                    .unwrap_or(false)
                {
                    let _ = std::fs::remove_file(&p);
                }
            }
        }

        // Persist QEMU's stderr (plugin log: `-d plugin,page` layout output)
        // to qemu_stderr.txt. A file also means we can never deadlock QEMU on
        // an undrained pipe.
        // nosemgrep: input-validation-path-traversal-2 -- canonicalized workdir joined with a const file name
        let stderr_path = prepared.working_directory.join(super::QEMU_STDERR_FILE);
        let stderr_file = fs::File::create(&stderr_path).map_err(|e| {
            log::error!("Failed to create QEMU stderr file {:?}: {}", stderr_path, e);
            QemuError::ExecutionFailed {
                qemu: self.path_config.qemu_binary.clone(),
                source: e,
            }
        })?;
        log::debug!("QEMU stderr -> {:?}", stderr_path);

        // Persist QEMU's stdout (the guest program's own console output) to
        // qemu_stdout.txt as well, for the same reasons as stderr above.
        // nosemgrep: input-validation-path-traversal-2 -- canonicalized workdir joined with a const file name
        let stdout_path = prepared.working_directory.join(super::QEMU_STDOUT_FILE);
        let stdout_file = fs::File::create(&stdout_path).map_err(|e| {
            log::error!("Failed to create QEMU stdout file {:?}: {}", stdout_path, e);
            QemuError::ExecutionFailed {
                qemu: self.path_config.qemu_binary.clone(),
                source: e,
            }
        })?;
        log::debug!("QEMU stdout -> {:?}", stdout_path);

        // nosemgrep: rust-security-command-injection-new -- choosing which QEMU to run is the feature; same trust level as the CLI
        let child = Command::new(&self.path_config.qemu_binary)
            .current_dir(&prepared.bin_work_dir)
            // .env("LD_LIBRARY_PATH", &ld_lib_path)
            .args(&prepared.qemu_args)
            .arg(&prepared.binary_full_path)
            .args(&prepared.program_args)
            .stdout(Stdio::from(stdout_file))
            .stderr(Stdio::from(stderr_file))
            .spawn()
            .map_err(|e| QemuError::ExecutionFailed {
                qemu: self.path_config.qemu_binary.clone(),
                source: e,
            })?;

        log::debug!("QEMU process spawned (pid {})", child.id());

        Ok(child)
    }
}

/// Log the QEMU wall time and exit status in one place, so the live-trace path
/// (which polls with `try_wait()`) and the blocking path report identically.
pub(crate) fn log_qemu_exit(run_time_secs: f32, status: &ExitStatus) {
    log::info!(
        "QEMU execution completed in {:.2}s, exit-code {}",
        run_time_secs,
        status
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn build_qemu_args_passes_workdir_and_run_id_to_tracer_plugin() {
        let cpu_config = CpuConfig::default();
        let path_config = PathConfig::default();
        let plugin_config = PluginConfig {
            tracer_plugin_path: Some(PathBuf::from("/tmp/libturbo_tracer.so")),
            ..Default::default()
        };

        let work_dir = Path::new("/tmp/turbo-output");
        let args = build_qemu_args(
            &cpu_config,
            &path_config,
            &plugin_config,
            work_dir,
            RunId::parse(&format!("run_{:016x}", 0x1d)).unwrap(),
            &[],
        )
        .unwrap();

        let plugin_idx = args.iter().position(|arg| arg == "-plugin").unwrap();
        let plugin_arg = &args[plugin_idx + 1];
        assert!(plugin_arg.contains("workdir=/tmp/turbo-output"));
        assert!(plugin_arg.contains("run_id=run_000000000000001d"));
        // The plugin has never parsed a socket_path key: trace data is appended
        // to an on-disk record log, so passing one would be silently ignored noise.
        assert!(!plugin_arg.contains("socket_path"));
    }

    /// With no `--vlen` and no `vlen` configured, QEMU is asked for 1024 bits:
    /// the default is a constant in bits, never read back from `SYSTEM_VLEN`
    /// (which holds bytes, and which an earlier run may have changed).
    #[test]
    fn default_vlen_is_1024_bits() {
        assert_eq!(crate::common::DEFAULT_VLEN_BITS, 1024);
        assert_eq!(CpuConfig::default().vlen, 1024);
        assert_eq!(QemuRunner::new().vlen_bits(), 1024);
        assert_eq!(
            CpuConfig::default().to_qemu_arg(),
            "max,v=true,vext_spec=v1.0,vlen=1024"
        );
    }

    /// A scratch directory holding one executable and one non-executable file.
    fn qemu_lookup_fixture(name: &str) -> PathBuf {
        use std::os::unix::fs::PermissionsExt;
        let dir = std::env::temp_dir().join(format!("turbo_qemu_lookup_{name}"));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        for (file, mode) in [("fake-qemu", 0o755), ("not-executable", 0o644)] {
            let path = dir.join(file);
            fs::write(&path, "#!/bin/sh\n").unwrap();
            fs::set_permissions(&path, fs::Permissions::from_mode(mode)).unwrap();
        }
        dir
    }

    #[test]
    fn bare_qemu_name_is_found_on_search_path() {
        let dir = qemu_lookup_fixture("search_path");
        let search_path = std::env::join_paths([Path::new("/nonexistent"), &dir]).unwrap();

        assert_eq!(
            find_in_search_path(Path::new("fake-qemu"), &search_path),
            Some(dir.join("fake-qemu"))
        );
        assert_eq!(
            find_in_search_path(Path::new("not-executable"), &search_path),
            None
        );
        assert_eq!(
            find_in_search_path(Path::new("missing"), &search_path),
            None
        );
    }

    #[test]
    fn qemu_path_with_a_directory_is_not_searched_for() {
        let dir = qemu_lookup_fixture("explicit_path");

        let qemu = dir.join("fake-qemu");
        assert_eq!(resolve_qemu_binary(&qemu).unwrap(), qemu);

        for rejected in [dir.join("not-executable"), dir.join("missing"), dir.clone()] {
            assert!(matches!(
                resolve_qemu_binary(&rejected),
                Err(QemuError::QemuNotFound(p)) if p == rejected
            ));
        }
    }

    #[test]
    fn qemu_not_found_message_says_where_it_looked() {
        assert!(qemu_not_found_message(Path::new("qemu-riscv64")).contains("not found on PATH"));
        assert!(qemu_not_found_message(Path::new("./qemu-riscv64")).contains("qemu_bin"));
        assert!(!qemu_not_found_message(Path::new("/opt/qemu")).contains("PATH"));
    }
}
