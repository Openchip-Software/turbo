use crate::{
    config::{ConfigFile, CpuConfig, EnvConfig},
    process_trace_input_file, record_with_qemu, run_from_input_rois,
};
use chrono::Local;
use clap::{Args, Parser, Subcommand, ValueEnum};
use std::path::{Path, PathBuf};

/// Fields shared by `record`, `process` and `run`: what binary to work with,
/// where to read/write files, and what CPU is being modelled.
#[derive(Args, Debug, Clone, PartialEq, Eq)]
pub struct CommonArgs {
    /// Binary to use for recording/decoding
    pub binary: PathBuf,

    /// Directory to save output files (default: output/YYYY_MM_DD_HH_MM_SS_<binary>)
    #[arg(short, long)]
    pub output_dir: Option<PathBuf>,

    /// Path to the CPU configuration file (`cpu_config.json`): the emulated
    /// VLEN and, for `--cache-sim`, the cache geometry. Default:
    /// `default_cpu_config` from `environment_config.json` if it names one,
    /// else the `cpu_config.json` in `~/.config/openchip/` if there is one,
    /// else the one in `/usr/share/turbo/`.
    ///
    /// Lives here rather than on the record- or process-specific argument
    /// groups because all three of `record`, `process` and `run` need it, and
    /// `run` flattens both of those groups -- one flag in each would be a
    /// duplicate-argument error.
    #[arg(long, value_name = "PATH")]
    pub cpu_config: Option<PathBuf>,
}

impl CommonArgs {
    /// Build the pair programmatically. Deliberately not `Default`: `binary` is
    /// a required positional on the CLI, so there is no such thing as a
    /// defaulted one.
    pub fn new(binary: impl Into<PathBuf>, output_dir: Option<PathBuf>) -> Self {
        Self {
            binary: binary.into(),
            output_dir,
            cpu_config: None,
        }
    }

    /// Point at a specific `cpu_config.json` instead of the default location.
    pub fn with_cpu_config(mut self, path: impl Into<PathBuf>) -> Self {
        self.cpu_config = Some(path.into());
        self
    }

    /// Resolve the output directory, defaulting to
    /// `output/YYYY_MM_DD_HH_MM_SS_<binary>` when none was given.
    pub fn resolve_output_dir(&self) -> PathBuf {
        resolve_output_dir(self.output_dir.clone(), &self.binary)
    }
}

/// CLI entry point for live and offline turbo workflows

#[derive(Parser, Debug)]
#[command(name = "turbo")]
#[command(about = "Tuning Utilities for Risc-v Benchmarking and Optimization")]
pub struct Cli {
    #[command(subcommand)]
    command: Command,
}

impl From<Command> for Cli {
    /// Build a `Cli` from a programmatically constructed `Command`, so callers
    /// (tests, other tools) can drive [`main_interface`] through exactly the
    /// same entry point as the binary does, without going via argv strings.
    fn from(command: Command) -> Self {
        Self { command }
    }
}

/// Options for spawning QEMU to record a binary, shared by `Record` and `Run`.
#[derive(Args, Debug, Clone, Default, PartialEq, Eq)]
pub struct RecordArgs {
    /// Path to the environment configuration file (`environment_config.json`):
    /// where QEMU, its plugin directory, the guest sysroot and the tracer
    /// plugin live. `--config` is kept as an alias for the pre-split flag.
    #[arg(long, alias = "config", value_name = "PATH")]
    pub env_config: Option<PathBuf>,

    /// Directory to use as the profiled binary's current working directory
    #[arg(long)]
    pub bin_work_dir: Option<PathBuf>,

    /// Also record the independent reference traces of the same execution
    /// (execlog `execlog_out.txt`). Only useful for
    /// cross-checking the trace decode against them -- see `--dump-pcs`.
    /// Expensive: execlog writes one text line per instruction.
    #[arg(long, hide = true)]
    pub reference_traces: bool,

    /// Vector register length in bits for the emulated CPU (QEMU `vlen=`).
    /// A power of two, 128 to 1024 on stock QEMU 11.0.0. RVV allows up to
    /// 65536, but QEMU refuses more than it supports; `turbo config check`
    /// reports the range yours accepts. Overrides `vlen` in `cpu_config.json`.
    #[arg(long, value_name = "BITS")]
    pub vlen: Option<u32>,

    /// Environment variable to set for the *guest* program (QEMU `-E
    /// NAME=VALUE`), repeatable. This is how a dynamically-linked guest is told
    /// where its shared libraries live (`--guest-env LD_LIBRARY_PATH=...`),
    /// since QEMU-user does not forward the host's environment by default.
    #[arg(long = "guest-env", value_name = "NAME=VALUE")]
    pub guest_env: Vec<String>,

    /// Arguments passed to the binary (must come after `--`)
    #[arg(allow_hyphen_values = true)]
    pub args: Vec<String>,
}

impl RecordArgs {
    /// The `--guest-env NAME=VALUE` strings as `(name, value)` pairs, rejecting
    /// any without an `=` (which QEMU's `-E` would not be able to set).
    pub fn guest_env_pairs(&self) -> anyhow::Result<Vec<(String, String)>> {
        self.guest_env
            .iter()
            .map(|s| {
                s.split_once('=')
                    .map(|(k, v)| (k.to_string(), v.to_string()))
                    .ok_or_else(|| anyhow::anyhow!("--guest-env must be NAME=VALUE, got '{s}'"))
            })
            .collect()
    }
}

/// Options for decoding+enriching a recorded trace file, shared by `Process`
/// and `Run`.
#[derive(Args, Debug, Clone, PartialEq, Eq)]
pub struct ProcessArgs {
    /// Emit the source & assembly annotation report (annotate_*.html) into
    /// the output directory.
    #[arg(long, default_value = "true")]
    pub annotate: bool,

    /// Skip the ROI call-tree flamegraph (`roi_flamegraph.folded`,
    /// `roi_flamegraph.json` and the interactive `roi_flamegraph.html`).
    ///
    /// The flamegraph is aggregated as the trace is enriched, keyed on interned
    /// region IDs, so it costs one hash probe per region entry and two adds per
    /// region exit and is bounded by the number of unique ROI call paths rather
    /// than by execution count. That is cheap enough to leave on; this flag is
    /// the escape hatch for a workload whose ROI markers fire often enough that
    /// even that is worth declining.
    #[arg(long)]
    pub no_roi_flamegraph: bool,

    /// Refuse to report a trace that does not cover the whole run: a hart whose
    /// record log never reached its `Done` sentinel (the guest was killed, the
    /// machine died, or the file was copied mid-write), or a hart present in
    /// the recording but excluded by `--vcpu`.
    ///
    /// Off by default, deliberately: the surviving prefix of a cut-short
    /// recording is real data, and refusing it would make a crashed workload
    /// unanalysable. Every run records what it covered in
    /// `trace_info.trace_integrity` either way -- this flag is for a caller
    /// (CI, validation) that must only ever accept whole runs, and would rather
    /// have a non-zero exit than a partial answer.
    #[arg(long)]
    pub strict_trace: bool,

    /// Source search root for the --annotate report (repeatable). DWARF
    /// records build-host source paths that usually don't exist locally;
    /// each --src-dir is searched by longest matching path suffix (gdb
    /// `dir`-style) to locate the real source, e.g.
    /// `--src-dir /path/to/checkout`.
    #[arg(long = "src-dir")]
    pub src_dir: Vec<PathBuf>,

    /// Model an L1/L2 cache over the trace's captured vector memory addresses,
    /// reporting hits and misses per ROI and per instruction.
    ///
    /// The geometry comes from the `cache` block of `cpu_config.json`
    /// (`--cpu-config`); there is no built-in default cache, so this flag fails
    /// rather than modelling one the user never described.
    ///
    /// Off by default: it costs a model lookup per vector memory instruction,
    /// and it needs a trace recorded *with* address capture (the default; see
    /// `TURBO_MEM_CAPTURE`). The counts cover vector loads and stores only --
    /// scalar accesses and instruction fetch are not captured at all -- so they
    /// describe the vector data side, not the machine.
    #[arg(long)]
    pub cache_sim: bool,

    /// Write perfetto trace of run
    #[arg(long)]
    pub perfetto: bool,

    /// Perfetto trace flush threshold (number of packets before flushing to disk)
    #[arg(long, default_value = "100000")]
    pub trace_flush_threshold: usize,

    /// Perfetto counter emit interval (emit counter tracks every N instructions)
    #[arg(long, default_value = "1000")]
    pub trace_counter_interval: u64,

    /// Print a summary after processing. Plain --summary shows all views.
    /// Specific values: "program" (global stats), "function" (per-function table),
    /// "regions" (per-ROI table)
    #[arg(long, default_missing_value = "all", num_args = 0..=1, require_equals = true)]
    pub summary: Option<String>,

    /// Restrict processing to these vCPUs/harts (repeatable). Default: every
    /// hart that has a recorded `turbo_trace_vcpu<N>.bin` log. Harts present but
    /// excluded are reported, so a filtered run never looks like a full one.
    #[arg(long = "vcpu")]
    pub vcpu: Vec<u32>,

    /// Maximum decode+enrich worker threads (1..=8, default 8). Harts are
    /// decoded in parallel, one worker per hart up to this cap; beyond it harts
    /// share workers. `--decode-threads=1` is the single-threaded path, useful
    /// for A/B timing and for a byte-reproducible Perfetto trace.
    #[arg(long, value_name = "N")]
    pub decode_threads: Option<usize>,
}

/// Hand-written rather than derived, because `#[derive(Default)]` cannot see
/// clap's `default_value` attributes and would silently give `annotate: false`
/// and zeroed Perfetto intervals -- i.e. a "default" that no CLI user ever
/// gets. `arg_defaults_match_clap` pins these to the clap defaults.
impl Default for ProcessArgs {
    fn default() -> Self {
        Self {
            annotate: true,
            no_roi_flamegraph: false,
            strict_trace: false,
            src_dir: Vec::new(),
            cache_sim: false,
            perfetto: false,
            trace_flush_threshold: 100_000,
            trace_counter_interval: 1000,
            summary: None,
            vcpu: Vec::new(),
            decode_threads: None,
        }
    }
}

impl ProcessArgs {
    /// `--vcpu` as an optional filter: no flags means "every hart".
    pub fn hart_filter(&self) -> Option<Vec<u32>> {
        (!self.vcpu.is_empty()).then(|| self.vcpu.clone())
    }
}

/// How `run` sequences recording and processing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum, Default)]
pub enum Overlap {
    /// Decode and enrich WHILE QEMU runs, tailing the spool live, and serve the
    /// web UI as results accumulate. One pass over the data.
    #[default]
    On,
    /// Record to completion first, then process, then serve. Slower (no
    /// overlap), but each stage is separately observable -- which is what makes
    /// it the right mode for A/B timing the enricher, and for re-processing a
    /// recording with different flags without re-recording it.
    Off,
}

impl Overlap {
    fn enabled(self) -> bool {
        self == Overlap::On
    }
}

#[derive(Subcommand, Debug)]
pub enum Command {
    /// Run a binary under QEMU and write trace files to disk. Does not
    /// decode or enrich anything.
    Record {
        #[command(flatten)]
        common: CommonArgs,

        #[command(flatten)]
        record: RecordArgs,
    },

    /// Decode a recorded trace file and enrich it, writing report files
    /// (perf_data_final.json, annotate_*.html.gz, ...) to disk. Never starts
    /// the web server.
    Process {
        #[command(flatten)]
        common: CommonArgs,

        #[command(flatten)]
        process: ProcessArgs,

        /// Measurement harness: decode the trace and throw the result away, so
        /// that `DataProcessor::decode_into` can be timed on its own instead of
        /// inferred by subtracting an enrich-inclusive run. Produces NO outputs.
        #[arg(long, hide = true)]
        decode_only: bool,

        /// Cross-check harness: write every decoded PC to `turbo_trace_pcs.txt` in
        /// the output directory, for comparison against an independently
        /// produced trace of the same run (see `record --reference-traces`).
        /// Implies `--decode-only`: produces NO reports.
        #[arg(long, hide = true)]
        dump_pcs: bool,

        /// Recorded spool directory, or a single `turbo_trace_vcpu<N>.bin` record
        /// log. A directory processes EVERY hart it contains; naming one log
        /// processes just that hart.
        trace_input: PathBuf,
    },

    /// Report what recorded trace files are made of, without decoding or
    /// enriching anything: on-disk vs uncompressed size and the achieved
    /// compression, the per-category byte breakdown, duplicate dictionary
    /// entries, and the hottest blocks. Harts are inspected in parallel; the
    /// combined whole-run report is logged at info, one line per hart at debug,
    /// and each hart's full report at trace.
    Inspect {
        /// Recorded spool directory, or a single `turbo_trace_vcpu<N>.bin`
        /// record log. A directory inspects EVERY hart it contains.
        trace_input: PathBuf,

        /// Restrict the report to these vCPUs/harts (repeatable).
        #[arg(long = "vcpu")]
        vcpu: Vec<u32>,
    },

    /// Run the web server over a previously processed output directory.
    /// Does not run or parse anything.
    Load {
        /// Input directory containing a previously written perf_data_final.json
        input_dir: PathBuf,
    },

    /// Convenience command: record, then process, then load, in one go.
    Run {
        #[command(flatten)]
        common: CommonArgs,

        #[command(flatten)]
        record: RecordArgs,

        #[command(flatten)]
        process: ProcessArgs,

        /// Skip the final `load` step (don't start the web server).
        #[arg(long)]
        no_server: bool,

        /// Whether to overlap decode/enrich with QEMU execution.
        #[arg(long, value_name = "MODE", default_value = "on")]
        overlap: Overlap,
    },

    /// Write or verify the configuration files that `record` and `run` read.
    Config {
        #[command(subcommand)]
        action: ConfigAction,
    },
}

/// `turbo config` verbs.
#[derive(Subcommand, Debug)]
pub enum ConfigAction {
    /// Write a starter `environment_config.json` and `cpu_config.json`, with
    /// paths found on this machine: `qemu-riscv64` from PATH and the tracer
    /// plugin from this build. Existing files are left alone unless `--force`
    /// is given.
    Init {
        /// Directory to write the files to (default: `~/.config/openchip/`).
        #[arg(long, value_name = "DIR")]
        dir: Option<PathBuf>,

        /// Overwrite files that already exist.
        #[arg(long)]
        force: bool,
    },

    /// Check the configuration `record` and `run` would use: that QEMU can be
    /// launched, its version and plugin support, that it loads the tracer
    /// plugin, and that the plugin's trace is readable by this turbo. Exits 1
    /// if any check fails.
    Check {
        /// Environment configuration file to check instead of the default one.
        #[arg(long, value_name = "PATH")]
        env_config: Option<PathBuf>,

        /// CPU configuration file to check instead of the default one.
        #[arg(long, value_name = "PATH")]
        cpu_config: Option<PathBuf>,
    },
}

pub fn resolve_output_dir(output_dir: Option<PathBuf>, binary: &Path) -> PathBuf {
    let bin_stem = binary
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("turbo_run");
    let timestamp = Local::now().format("%Y_%m_%d_%H_%M_%S");
    let subdir = output_dir.unwrap_or_else(|| PathBuf::from(format!("{}_{}", timestamp, bin_stem)));
    // nosemgrep: input-validation-path-traversal-2 -- subdir is the operator's own --output-dir
    PathBuf::from("output").join(subdir)
}

/// Load the `EnvConfig` for a `record`/`run` invocation: from `--env-config` if
/// given, otherwise from the default location, falling back to `None`.
fn load_env_config(config: Option<&PathBuf>) -> Option<EnvConfig> {
    if let Some(path) = config {
        match EnvConfig::from_file(path) {
            Ok(cfg) => {
                log::info!("Loaded config from {}", path.display());
                Some(cfg)
            }
            Err(e) => {
                log::error!("Failed to load config from {}: {}", path.display(), e);
                None
            }
        }
    } else {
        match EnvConfig::from_default_location() {
            Ok(Some((cfg, path))) => {
                log::info!("Loaded config from {}", path.display());
                Some(cfg)
            }
            // Only debug: `process` reads this file too and never needs it, and
            // `record`/`run` report the defaults they fall back to where the
            // QEMU runner is built (see `workflow::build_qemu_runner`).
            Ok(None) => {
                log::debug!(
                    "No {} in {}",
                    EnvConfig::FILE_NAME,
                    EnvConfig::searched_locations()
                );
                None
            }
            Err(e) => {
                log::warn!("Ignoring {}: {e}", EnvConfig::default_location().display());
                None
            }
        }
    }
}

/// The CPU config an invocation runs with, and the file it came from.
///
/// The pair travels together because the two must agree: `--cache-sim`'s error
/// tells the user which file to add a `cache` block to, and with three possible
/// sources for that file, re-deriving the name at the error site is how it would
/// come to name the wrong one.
#[derive(Debug)]
pub(crate) struct ResolvedCpuConfig {
    pub(crate) config: CpuConfig,
    /// The file `config` was read from, for error messages that must name the
    /// file this invocation actually chose.
    pub(crate) path: PathBuf,
}

/// Load the `CpuConfig` for any invocation, in precedence order: `--cpu-config`,
/// then `default_cpu_config` from `environment_config.json`, then the default
/// location.
///
/// The middle step exists so that a machine modelling something other than the
/// shipped `cpu_config.json` says so once, in the file that already describes
/// this machine, instead of on every command line -- and so that a checkout
/// keeping one file per core can switch between them by editing one line.
///
/// Returns an empty config rather than an `Option` when nothing is found: "no
/// file" and "a file that sets no fields" mean the same thing to every consumer,
/// and both are reported by whichever consumer actually needed a field --
/// `--cache-sim` with an error, `vlen` by falling back to `SYSTEM_VLEN`.
///
/// A *named* file that cannot be read IS fatal, whether the name came from the
/// flag or from the environment config: the user named a file, and silently
/// profiling a different CPU than the one they pointed at is worse than
/// stopping. Only the unnamed default location is allowed to be absent.
fn load_cpu_config(
    common: &CommonArgs,
    env_config: Option<&EnvConfig>,
) -> anyhow::Result<ResolvedCpuConfig> {
    load_cpu_config_from(common.cpu_config.as_deref(), env_config)
}

/// [`load_cpu_config`] for a caller with no [`CommonArgs`] (`turbo config
/// check`), so that it resolves the file by exactly the rules `run` does.
pub(crate) fn load_cpu_config_from(
    cpu_config_flag: Option<&Path>,
    env_config: Option<&EnvConfig>,
) -> anyhow::Result<ResolvedCpuConfig> {
    let named = cpu_config_flag
        .map(|path| (path, "--cpu-config"))
        .or_else(|| {
            env_config
                .and_then(|env| env.default_cpu_config.as_deref())
                .map(|path| (path, "default_cpu_config in environment_config.json"))
        });

    if let Some((path, source)) = named {
        let config = CpuConfig::from_file(path)
            .map_err(|e| anyhow::anyhow!("{source} {}: {e}", path.display()))?;
        log::info!(
            "Loaded cpu config from {} ({source}): {config:?}",
            path.display()
        );
        return Ok(ResolvedCpuConfig {
            config,
            path: path.to_path_buf(),
        });
    }

    let path = CpuConfig::default_location();
    let config = match CpuConfig::from_default_location() {
        Ok(Some((config, path))) => {
            log::info!("Loaded cpu config from {}: {config:?}", path.display());
            config
        }
        Ok(None) => {
            log::info!(
                "No {} in {}; modelling the default CPU (VLEN {} bits unless --vlen is given, \
                 no cache for --cache-sim)",
                CpuConfig::FILE_NAME,
                CpuConfig::searched_locations(),
                crate::common::DEFAULT_VLEN_BITS,
            );
            CpuConfig::default()
        }
        Err(e) => {
            log::warn!("Ignoring {}: {e}", path.display());
            CpuConfig::default()
        }
    };
    Ok(ResolvedCpuConfig { config, path })
}

pub fn main_interface(cli: Cli) -> anyhow::Result<()> {
    match cli.command {
        Command::Record { common, record } => cmd_record(common, record).map(|_| ()),
        Command::Process {
            common,
            process,
            decode_only,
            dump_pcs,
            trace_input,
        } => cmd_process(common, process, decode_only, dump_pcs, trace_input),
        Command::Inspect { trace_input, vcpu } => crate::inspect::report_trace_input(
            &crate::workflow::TraceInput::resolve(trace_input, (!vcpu.is_empty()).then_some(vcpu))?,
        ),
        Command::Load { input_dir } => run_from_input_rois(input_dir),
        Command::Run {
            common,
            record,
            process,
            no_server,
            overlap,
        } => cmd_run(common, record, process, no_server, overlap),
        Command::Config { action } => match action {
            ConfigAction::Init { dir, force } => crate::config_cmd::init(dir, force),
            ConfigAction::Check {
                env_config,
                cpu_config,
            } => crate::config_cmd::check(env_config.as_deref(), cpu_config.as_deref()),
        },
    }
}

/// Run and analyse a binary: launch QEMU and process the results.
fn cmd_record(common: CommonArgs, record: RecordArgs) -> anyhow::Result<PathBuf> {
    let env_config = load_env_config(record.env_config.as_ref());
    let cpu_config = load_cpu_config(&common, env_config.as_ref())?.config;

    let output_dir = common.resolve_output_dir();
    // Absolutise so every consumer (QEMU) refers to the exact same
    // directory regardless of the process working dir.
    let output_dir = std::path::absolute(&output_dir).unwrap_or(output_dir);

    // Computed before `record.args` is moved into the options below.
    let guest_env = record.guest_env_pairs()?;

    record_with_qemu(
        common.binary,
        output_dir.clone(),
        crate::workflow::RecordWithQemuOptions {
            bin_work_dir: record.bin_work_dir,
            env_config,
            bin_args: record.args,
            reference_traces: record.reference_traces,
            // Precedence, resolved once here rather than in three layers:
            // `--vlen` beats the config file, which beats `SYSTEM_VLEN`'s own
            // default (a `None` the runner leaves alone).
            vlen_bits: record.vlen.or(cpu_config.vlen),
            guest_env,
        },
    )?;

    Ok(output_dir)
}

/// Refuse a missing or unreadable ELF before the decode pipeline starts.
///
/// `record` and `run` get this from `QemuRunner::validate`, which never runs on
/// the `process` path: nothing there opens the binary until an enrichment
/// worker builds its `ElfMetadata`, several threads and one memory map later.
fn validate_process_binary(binary: &Path) -> anyhow::Result<()> {
    if !binary.exists() {
        anyhow::bail!(
            "binary {} does not exist; `process` needs the SAME ELF the trace was \
             recorded from, to resolve its symbols and instructions",
            binary.display()
        );
    }
    if binary.is_dir() {
        anyhow::bail!(
            "binary {} is a directory, not an ELF; the recorded trace directory is \
             the LAST argument, the binary comes first",
            binary.display()
        );
    }
    if let Err(e) = std::fs::File::open(binary) {
        anyhow::bail!("binary {} cannot be read: {e}", binary.display());
    }
    Ok(())
}

/// Decode+enrich a previously recorded trace spool or log.
fn cmd_process(
    common: CommonArgs,
    process: ProcessArgs,
    decode_only: bool,
    dump_pcs: bool,
    trace_input: PathBuf,
) -> anyhow::Result<()> {
    // `process` has no `--env-config` of its own -- it never launches QEMU, so
    // the rest of that file is nothing to it -- but it still reads the default
    // one, because `default_cpu_config` must mean the same thing to every
    // command or a `run` and a re-`process` of its recording would model
    // different caches.
    let env_config = load_env_config(None);
    let cpu_config = load_cpu_config(&common, env_config.as_ref())?;
    let cache_sim = cpu_config
        .config
        .cache_for_sim(process.cache_sim, &cpu_config.path)?;
    let cpu_vlen_bits = cpu_config.config.vlen;
    // Both inputs are checked before anything is loaded, decoded or created:
    // `process` reads the binary from a worker thread deep in the decode
    // pipeline, where a path typo surfaces as a thread panic rather than as the
    // one sentence that fixes it.
    validate_process_binary(&common.binary)?;
    let input = crate::workflow::TraceInput::resolve(trace_input, process.hart_filter())?;
    // Default the outputs next to the input rather than into a fresh
    // `output/<timestamp>_<binary>/`: reports about a recording belong beside
    // that recording, where `turbo load` will be pointed.
    let output_dir = match common.output_dir.clone() {
        Some(_) => common.resolve_output_dir(),
        None => input.default_output_dir(),
    };
    log::info!("process: outputs dir is {}", output_dir.display());
    process_trace_input_file(
        common.binary,
        input,
        output_dir,
        crate::workflow::ProcessTraceOptions {
            annotate: process.annotate,
            roi_flamegraph: !process.no_roi_flamegraph,
            source_dirs: process.src_dir,
            cache_sim,
            perfetto: process.perfetto,
            trace_flush_threshold: process.trace_flush_threshold,
            trace_counter_interval: process.trace_counter_interval,
            summary: process.summary,
            decode_threads: process.decode_threads,
            // Only to be checked against the recording's own metadata, which
            // stays authoritative. A `cache` block that differs IS legitimate --
            // re-simulating one trace against several cache geometries is the
            // point of the address replay -- but a differing VLEN is not.
            configured_vlen_bits: cpu_vlen_bits,
            decode_only,
            dump_pcs,
            strict_trace: process.strict_trace,
        },
    )
}

/// Record and analyse in one command.
///
/// With `--overlap=on` (the default) this is a single pass: QEMU runs while the
/// spool is tailed, decoded and enriched, and the web UI serves results as they
/// accumulate. With `--overlap=off` it is `record` -> `process` -> `load` in
/// sequence, sharing one resolved `output_dir`.
fn cmd_run(
    common: CommonArgs,
    record: RecordArgs,
    process: ProcessArgs,
    no_server: bool,
    overlap: Overlap,
) -> anyhow::Result<()> {
    let binary = common.binary.clone();
    let hart_filter = process.hart_filter();
    // Resolved before either branch: `--cache-sim` without a configured cache
    // must fail before QEMU is launched, not after the recording is made.
    // Loaded up here rather than in the overlapping branch alone: it is where
    // `default_cpu_config` comes from, and the cache resolution below must not
    // depend on which branch is taken.
    let env_config = load_env_config(record.env_config.as_ref());
    let cpu_config = load_cpu_config(&common, env_config.as_ref())?;
    let cache_sim = cpu_config
        .config
        .cache_for_sim(process.cache_sim, &cpu_config.path)?;
    let vlen_bits = record.vlen.or(cpu_config.config.vlen);

    if overlap.enabled() {
        let output_dir = std::path::absolute(common.resolve_output_dir())
            .unwrap_or_else(|_| common.resolve_output_dir());
        // Computed before `record.args` is moved into the options below.
        let guest_env = record.guest_env_pairs()?;
        // The live path starts (and joins) the server itself, so there is NO
        // trailing `load` here: a second server would race the first for
        // DEFAULT_PORT. That asymmetry is the whole reason the sequential mode
        // below can compose three commands and this one cannot.
        crate::run_with_live_qemu(
            binary,
            output_dir.clone(),
            crate::workflow::RunWithQemuOptions {
                bin_work_dir: record.bin_work_dir,
                no_server,
                env_config,
                bin_args: record.args,
                roi_flamegraph: !process.no_roi_flamegraph,
                annotate: process.annotate,
                cache_sim,
                perfetto: process.perfetto,
                trace_flush_threshold: process.trace_flush_threshold,
                trace_counter_interval: process.trace_counter_interval,
                summary: process.summary,
                source_dirs: process.src_dir,
                hart_filter,
                decode_threads: process.decode_threads,
                reference_traces: record.reference_traces,
                vlen_bits,
                guest_env,
                strict_trace: process.strict_trace,
            },
        )?;
        Ok(())
    } else {
        let output_dir = cmd_record(common, record)?;
        process_trace_input_file(
            binary,
            crate::workflow::TraceInput::RecordedDir {
                dir: output_dir.clone(),
                harts: hart_filter,
            },
            output_dir.clone(),
            crate::workflow::ProcessTraceOptions {
                annotate: process.annotate,
                roi_flamegraph: !process.no_roi_flamegraph,
                source_dirs: process.src_dir,
                cache_sim,
                perfetto: process.perfetto,
                trace_flush_threshold: process.trace_flush_threshold,
                trace_counter_interval: process.trace_counter_interval,
                summary: process.summary,
                decode_threads: process.decode_threads,
                // The effective VLEN, not `cpu_config.vlen`: `--vlen` is what
                // this recording was just made at, so passing the file's value
                // would report a mismatch against a deliberate override.
                configured_vlen_bits: vlen_bits,
                strict_trace: process.strict_trace,
                ..Default::default()
            },
        )?;

        if no_server {
            return Ok(());
        }
        run_from_input_rois(output_dir)
    }
}

#[cfg(test)]
mod tests {
    use std::fs;

    use super::*;
    use crate::common::PerformanceData;

    /// The programmatic `Default` impls must be byte-identical to what clap
    /// produces for a flagless invocation, so `ProcessArgs::default()` in a
    /// caller means exactly "what the user gets by typing no flags". Without
    /// this, `annotate`'s `default_value = "true"` and the two Perfetto
    /// intervals could drift out of sync with the hand-written `Default`.
    #[test]
    fn arg_defaults_match_clap() {
        #[derive(Parser)]
        struct ProcessOnly {
            #[command(flatten)]
            args: ProcessArgs,
        }
        #[derive(Parser)]
        struct RecordOnly {
            #[command(flatten)]
            args: RecordArgs,
        }

        assert_eq!(
            ProcessOnly::parse_from(["turbo"]).args,
            ProcessArgs::default(),
            "ProcessArgs::default() has drifted from the clap defaults"
        );
        assert_eq!(
            RecordOnly::parse_from(["turbo"]).args,
            RecordArgs::default(),
            "RecordArgs::default() has drifted from the clap defaults"
        );
    }

    /// `--cpu-config` beats `default_cpu_config` beats the default location,
    /// and a *named* file that is not there stops the run instead of quietly
    /// modelling some other CPU.
    ///
    /// The precedence is the whole point of `default_cpu_config`: a machine
    /// that sets it must still be able to profile one run against a different
    /// core without editing its environment config.
    #[test]
    fn cpu_config_precedence() {
        let dir = crate::workspace_root().join("target/test_output/cpu_config_precedence");
        fs::create_dir_all(&dir).expect("failed to create test directory");
        let from_flag = dir.join("from_flag.json");
        let from_env = dir.join("from_env.json");
        fs::write(&from_flag, r#"{"vlen": 256}"#).expect("failed to write from_flag.json");
        fs::write(&from_env, r#"{"vlen": 512}"#).expect("failed to write from_env.json");

        let env_config = EnvConfig {
            default_cpu_config: Some(from_env.clone()),
            ..Default::default()
        };
        let bare = CommonArgs::new("unused_binary", None);
        let with_flag = bare.clone().with_cpu_config(&from_flag);

        let resolved = load_cpu_config(&with_flag, Some(&env_config))
            .expect("--cpu-config names a readable file");
        assert_eq!(resolved.config.vlen, Some(256), "--cpu-config must win");
        assert_eq!(resolved.path, from_flag);

        let resolved = load_cpu_config(&bare, Some(&env_config))
            .expect("default_cpu_config names a readable file");
        assert_eq!(
            resolved.config.vlen,
            Some(512),
            "default_cpu_config must be used when --cpu-config is absent"
        );
        assert_eq!(
            resolved.path, from_env,
            "the path must be the file actually read, so --cache-sim's error names it"
        );

        // A flag still overrides it, even when the environment names one.
        let resolved = load_cpu_config(&with_flag, Some(&env_config)).expect("readable");
        assert_eq!(resolved.config.vlen, Some(256));

        // A default_cpu_config that is not there is fatal, not a silent
        // fallback to whatever CPU the default location describes.
        let missing = EnvConfig {
            default_cpu_config: Some(dir.join("no_such_cpu_config.json")),
            ..Default::default()
        };
        let err = load_cpu_config(&bare, Some(&missing))
            .expect_err("a named-but-missing cpu config must fail the run")
            .to_string();
        assert!(
            err.contains("default_cpu_config in environment_config.json"),
            "the error must name where the bad path came from, got: {err}"
        );

        // With neither, the default location is what gets reported.
        let resolved = load_cpu_config(&bare, None).expect("the default location is never fatal");
        assert_eq!(resolved.path, CpuConfig::default_location());
    }

    /// `record` then `process`, driven through [`main_interface`] itself rather
    /// than by calling `record_with_qemu`/`process_trace_input_file` directly,
    /// so a broken `cmd_record`/`cmd_process` wiring fails here.
    #[test]
    fn test_main_commands() {
        let _ = env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info"))
            .try_init();

        let workspace = crate::workspace_root();
        let binary = workspace.join("test_bins/basic_branch_asm/flops_test.riscv");
        let output_dir = workspace.join("target/test_output/turbo_main");
        fs::create_dir_all(&output_dir).expect("Failed to create output directory");
        assert!(binary.exists(), "Binary not found at {:?}", binary);

        let common = CommonArgs::new(&binary, Some(output_dir.clone()));

        // Step 1: `turbo record` -- QEMU writes a trace spool, nothing decoded.
        main_interface(
            Command::Record {
                common: common.clone(),
                record: RecordArgs::default(),
            }
            .into(),
        )
        .expect("record command failed");

        let turbo_trace_bin = output_dir.join(turbo_tracer_shared::get_turbo_trace_bin_for_vcpu(0));
        assert!(
            turbo_trace_bin.exists(),
            "Trace file not created at {:?}",
            turbo_trace_bin
        );

        // Step 2: `turbo process` on the DIRECTORY, covering the resolution path
        // `run --overlap=off` uses: every hart present, not just vCPU 0.
        main_interface(
            Command::Process {
                common,
                process: ProcessArgs::default(),
                decode_only: false,
                dump_pcs: false,
                trace_input: output_dir.clone(),
            }
            .into(),
        )
        .expect("process command failed");

        let perf_data_file = output_dir.join("perf_data_final.json");
        assert!(
            perf_data_file.exists(),
            "perf_data_final.json not created at {:?}",
            perf_data_file
        );

        // Step 3: the data `turbo load` would serve must be readable and non-empty.
        // `Load` itself is not invoked: it starts a blocking web server.
        let perf_data = PerformanceData::load(perf_data_file).expect("load perf_data_final.json");
        assert!(
            !perf_data.rois.is_empty() || !perf_data.functions.is_empty(),
            "Loaded performance data should contain ROIs or functions"
        );
    }

    /// `process` opens the binary only inside a decode worker, so a path typo
    /// used to surface as a thread panic. Each of the three ways the argument
    /// can be wrong gets its own message instead.
    #[test]
    fn process_refuses_an_unusable_binary() {
        let dir = std::env::temp_dir().join("turbo_validate_process_binary");
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();

        let err = validate_process_binary(&dir.join("nope.riscv")).expect_err("must be refused");
        assert!(err.to_string().contains("does not exist"), "got: {err}");

        // The commonest form of this mistake: the trace directory typed where
        // the binary belongs.
        let err = validate_process_binary(&dir).expect_err("a directory must be refused");
        assert!(err.to_string().contains("is a directory"), "got: {err}");

        let elf = dir.join("some.riscv");
        fs::write(&elf, b"\x7fELF").unwrap();
        validate_process_binary(&elf).expect("a readable file is accepted");
    }
}
