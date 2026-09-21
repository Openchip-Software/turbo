//! Shared test harness.
//!
//! Every integration test drives the tool through [`main_interface`] -- the
//! same entry point the `turbo` binary uses -- so tests exercise the real
//! top-level API rather than a parallel reimplementation of it. The only
//! test-specific concerns living here are output-directory lifecycle and
//! "read an artifact back and parse it"; everything about *what to run* is
//! expressed with the production `CommonArgs`/`RecordArgs`/`ProcessArgs`
//! structs, so a new CLI flag is reachable from tests with no changes here.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use turbo::config::{ConfigFile, CpuConfig, EnvConfig};
use turbo::main_interface::{
    main_interface, Command, CommonArgs, Overlap, ProcessArgs, RecordArgs,
};
use turbo::processing::read_page_gz;
use turbo::source::QemuRunner;
use turbo::{common::PerformanceData, processing::RoiInfo, workspace_root};

/// Record and process one guest binary, asserting only that the whole pipeline
/// completed. This is the body of most `turbo_tests!` entries: the guest is a
/// decoder regression case, and "the decode did not error" is the assertion.
pub fn smoke(name: &str, bin: &str, args: &[&str]) {
    if skip_if_missing(bin) {
        return;
    }
    TurboTest::new(name, bin).args(args).run();
}

/// Create a fresh output directory under `target/test_output/<name>`.
///
/// Always wiped first: a stale `perf_data_final.json` from a previous run of
/// the same test would otherwise satisfy assertions even if this run produced
/// nothing at all.
pub fn out_dir(name: &str) -> PathBuf {
    let dir = workspace_root()
        .join("target")
        .join("test_output")
        .join(name);
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("Failed to create test output directory");
    dir
}

/// Resolve a test binary path (workspace-relative or absolute) without
/// checking whether it exists. Kept separate from [`test_binary`] so the
/// skip-if-absent guard in the tests resolves the path exactly the way the
/// harness will later open it.
fn resolve_binary(bin: impl AsRef<Path>) -> PathBuf {
    let bin = bin.as_ref();
    if bin.is_absolute() {
        bin.to_path_buf()
    } else {
        workspace_root().join(bin)
    }
}

/// Skip the current test when `bin` was never built.
///
/// Not every guest is in the repository. The `rust_asm_bins` guests are built
/// from source for a musl target the Quickstart does not install, and guests
/// whose sources `#include` the RAVE headers are not built here at all: those
/// headers are external and deliberately not vendored, so `Makefile.tests`
/// skips `roi_validation` and `thread_roi` entirely. The skip message says
/// which of the two `bin` is, and how to build it. Same skip rules as
/// [`skip_unless_fixture_cpu`].
///
/// Returns `true` when the caller should return early.
#[must_use]
pub fn skip_if_missing(bin: impl AsRef<Path>) -> bool {
    let bin = bin.as_ref();
    let path = resolve_binary(bin);
    if path.exists() {
        return false;
    }
    let how = if bin.starts_with("test_bins/rust_asm_bins") {
        "clang, `rustup target add riscv64gc-unknown-linux-musl`, then \
         `cargo build --release` in test_bins/rust_asm_bins"
    } else if bin.starts_with("test_bins/roi_validation") || bin.starts_with("test_bins/thread_roi")
    {
        "the external RAVE headers, which are not in this repository"
    } else {
        "`make -f Makefile.tests`"
    };
    skip(format!("{} was not built; it needs {how}", path.display()))
}

/// Resolve a test binary path (workspace-relative or absolute) and assert it exists.
fn test_binary(bin: impl AsRef<Path>) -> PathBuf {
    let path = resolve_binary(bin);
    assert!(path.exists(), "Binary not found: {}", path.display());
    path
}

/// The `cpu_config.json` every harness invocation is pointed at: the one in the
/// tree, not whatever the developer happens to have in `$HOME`, unless
/// `TURBO_TEST_CPU_CONFIG` names another. Reading it from the default location
/// would make those assertions depend on an unversioned file, so the harness
/// passes `--cpu-config` explicitly and the tests read the same path.
///
/// The shipped file models the default CPU, not the one the fixtures were
/// computed for, so the tests that need that CPU skip (see [`skip_unless_fixture_cpu`]).
pub fn cpu_config_path() -> PathBuf {
    let path = std::env::var_os("TURBO_TEST_CPU_CONFIG")
        .map(PathBuf::from)
        .unwrap_or_else(|| workspace_root().join("crates/turbo/src/cpu_config.json"));
    assert!(path.exists(), "no cpu config at {}", path.display());
    path
}

/// The VLEN, in bits, that the hand-computed expectations of the
/// `test_bins/asm_validation` kernels assume: `expected_metrics.json`, and the
/// footprint, cache and element counts derived from the `.S` sources.
pub const FIXTURE_VLEN_BITS: u32 = 16384;

/// What a test's expectations need from the harness's `cpu_config.json`.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum FixtureCpu {
    /// VLEN [`FIXTURE_VLEN_BITS`].
    Vlen,
    /// VLEN [`FIXTURE_VLEN_BITS`], and a `cache` block for `--cache-sim`.
    VlenAndCache,
}

/// Whether `test` must skip, because the harness's `cpu_config.json` does not
/// model the CPU its expectations were computed for. Call it first, as
/// `if skip_unless_fixture_cpu(..) { return; }`.
///
/// Skipping rather than failing: at another VLEN these tests do not find a bug,
/// they just compare against numbers computed for a different machine. libtest
/// has no runtime skip, so a skipped test reports `ok`; the reason is printed
/// (shown with `--nocapture`). Set `TURBO_TEST_NO_SKIP=1` to make a skip a
/// failure instead, so that a CI job meant to run them cannot quietly run none.
#[must_use]
pub fn skip_unless_fixture_cpu(test: &str, needs: FixtureCpu) -> bool {
    let config = cpu_config();
    let mut missing = Vec::new();
    if config.vlen != Some(FIXTURE_VLEN_BITS) {
        missing.push(format!(
            "VLEN {FIXTURE_VLEN_BITS} (it has {})",
            config
                .vlen
                .map_or_else(|| "none".to_string(), |v| v.to_string())
        ));
    }
    if needs == FixtureCpu::VlenAndCache && config.cache.is_none() {
        missing.push("a cache block".to_string());
    }
    if missing.is_empty() {
        // The config is right; the QEMU still has to be able to run it.
        return skip_unless_qemu_vlen(test, FIXTURE_VLEN_BITS);
    }

    skip(format!(
        "{test}: needs {} in {}; set TURBO_TEST_CPU_CONFIG to a cpu_config.json that has it",
        missing.join(" and "),
        cpu_config_path().display()
    ))
}

/// Whether `test` must skip, because the QEMU the harness runs cannot start a
/// guest at `vlen_bits`. Call it first, as `if skip_unless_qemu_vlen(..) {
/// return; }`.
///
/// RVV permits up to 65536 bits, but upstream QEMU 11.0 stops at 1024, so on
/// the QEMU the README builds, a test at a wider VLEN finds nothing but that
/// QEMU refused to start. Same skip rules as [`skip_unless_fixture_cpu`].
#[must_use]
pub fn skip_unless_qemu_vlen(test: &str, vlen_bits: u32) -> bool {
    let max = qemu_max_vlen();
    if vlen_bits <= max {
        return false;
    }
    skip(format!(
        "{test}: needs a QEMU that runs VLEN {vlen_bits}; the configured one stops at {max}"
    ))
}

/// The largest VLEN the harness's QEMU ([`harness_runner`]) accepts, probed
/// once per test binary.
pub fn qemu_max_vlen() -> u32 {
    static MAX: OnceLock<u32> = OnceLock::new();
    *MAX.get_or_init(|| {
        turbo::config_cmd::qemu_max_vlen(&harness_runner())
            .expect("cannot probe the VLEN range of QEMU")
    })
}

/// Whether `test` must skip, because the harness's QEMU has no execlog plugin
/// for `--reference-traces` to load: `qemu_plugin_dir` is unset in the default
/// `environment_config.json`, or has no `libexeclog.so`. Same skip rules as
/// [`skip_unless_fixture_cpu`].
///
/// The Quickstart builds only `qemu-riscv64`, not QEMU's contrib plugins, so
/// on a README-built machine every PC-equivalence test lands here.
#[must_use]
pub fn skip_unless_execlog(test: &str) -> bool {
    let dir = harness_runner().plugin_config().plugin_dir.clone();
    let found = dir
        .as_ref()
        .is_some_and(|d| d.join("libexeclog.so").exists());
    if found {
        return false;
    }
    let dir = dir.map_or_else(|| "unset".to_string(), |d| d.display().to_string());
    skip(format!(
        "{test}: needs QEMU's execlog plugin, and qemu_plugin_dir is {dir}; build it \
         with `ninja -C build contrib-plugins` in the QEMU tree, and set \
         qemu_plugin_dir to <qemu>/build/contrib/plugins"
    ))
}

/// The `QemuRunner` the harness's runs get: the harness passes no
/// `--env-config`, so it is the one from the default `environment_config.json`
/// (or the built-in default).
fn harness_runner() -> QemuRunner {
    EnvConfig::from_default_location()
        .ok()
        .flatten()
        .map(|(config, _)| QemuRunner::from(config))
        .unwrap_or_default()
}

/// Report a skip as a no-op pass, or, with `TURBO_TEST_NO_SKIP` set, fail.
/// Always returns `true`, for the caller to return early on.
fn skip(reason: String) -> bool {
    assert!(
        std::env::var_os("TURBO_TEST_NO_SKIP").is_none(),
        "TURBO_TEST_NO_SKIP is set, so not skipping: {reason}"
    );
    eprintln!("SKIP: {reason}");
    true
}

/// The contents of [`cpu_config_path`], for tests that must agree with what the
/// pipeline was configured with.
pub fn cpu_config() -> CpuConfig {
    CpuConfig::from_file(cpu_config_path()).expect("the in-tree cpu_config.json must parse")
}

/// A `turbo` invocation under construction.
///
/// `name` is the output directory under `target/test_output/` and is always
/// given explicitly -- never derived from the binary's file name, which used to
/// couple the run and the assertions through a value neither side stated.
pub struct TurboTest {
    name: String,
    bin: PathBuf,
    record: RecordArgs,
    process: ProcessArgs,
    overlap: Overlap,
}

impl TurboTest {
    pub fn new(name: &str, bin: impl AsRef<Path>) -> Self {
        let _ =
            env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("debug"))
                .try_init();
        Self {
            name: name.to_string(),
            bin: test_binary(bin),
            record: RecordArgs::default(),
            process: ProcessArgs::default(),
            overlap: Overlap::default(),
        }
    }

    /// Arguments passed to the guest binary.
    pub fn args(mut self, args: &[&str]) -> Self {
        self.record.args = args.iter().map(|s| s.to_string()).collect();
        self
    }

    /// Set an environment variable for the *guest* program (QEMU `-E
    /// NAME=VALUE`), e.g. `.guest_env("LD_LIBRARY_PATH", "/path/to/libs")` so a
    /// dynamically-linked guest finds a `.so` that is not next to the binary.
    pub fn guest_env(mut self, name: &str, value: &str) -> Self {
        self.record.guest_env.push(format!("{name}={value}"));
        self
    }

    /// Set any `turbo process` / `turbo run` flag by field, e.g.
    /// `.process(|p| p.perfetto = true)`. A typed escape hatch instead of one
    /// wrapper method per flag, so `ProcessArgs` stays the single declaration
    /// of what is configurable.
    pub fn process(mut self, f: impl FnOnce(&mut ProcessArgs)) -> Self {
        f(&mut self.process);
        self
    }

    /// QEMU's vector register length in bits (`vlen=`), overriding the config
    /// file. A power of two in [128, 65536] -- the RVV range, which is why this
    /// is a `u32`: the maximum does not fit a `u16`.
    pub fn vlen(mut self, bits: u32) -> Self {
        self.record.vlen = Some(bits);
        self
    }

    /// Also record the execlog + Paraver reference traces of the same
    /// execution, for cross-checking the trace decode against them.
    pub fn reference_traces(mut self) -> Self {
        self.record.reference_traces = true;
        self
    }

    pub fn overlap(mut self, overlap: Overlap) -> Self {
        self.overlap = overlap;
        self
    }

    fn common(&self, dir: &Path) -> CommonArgs {
        CommonArgs::new(&self.bin, Some(dir.to_path_buf())).with_cpu_config(cpu_config_path())
    }

    /// `turbo run --no-server`: record and process in one go. The server is
    /// always skipped -- it blocks forever, which no test can survive.
    pub fn run(self) -> TestRun {
        let dir = out_dir(&self.name);
        main_interface(
            Command::Run {
                common: self.common(&dir),
                record: self.record,
                process: self.process,
                no_server: true,
                overlap: self.overlap,
            }
            .into(),
        )
        .expect("`turbo run` failed");
        TestRun { dir, bin: self.bin }
    }

    /// `turbo record`: QEMU writes a trace spool, nothing is decoded.
    pub fn record(self) -> TestRun {
        self.try_record().expect("`turbo record` failed")
    }

    /// [`TurboTest::record`] that returns the error instead of panicking, for
    /// tests that assert a run *fails* (e.g. a guest whose `.so` cannot be
    /// found without `LD_LIBRARY_PATH`).
    pub fn try_record(self) -> anyhow::Result<TestRun> {
        let dir = out_dir(&self.name);
        main_interface(
            Command::Record {
                common: self.common(&dir),
                record: self.record,
            }
            .into(),
        )?;
        Ok(TestRun { dir, bin: self.bin })
    }
}

/// A completed run and the artifacts in its output directory.
pub struct TestRun {
    pub dir: PathBuf,
    pub bin: PathBuf,
}

impl TestRun {
    /// The merged, whole-program `perf_data_final.json`.
    pub fn perf_data(&self) -> PerformanceData {
        let path = self.dir.join("perf_data_final.json");
        assert!(
            path.exists(),
            "perf_data_final.json not written: {}",
            path.display()
        );
        PerformanceData::load(path).expect("perf_data_final.json is readable and well-formed")
    }

    /// One hart's own `turbo_vcpu_<N>_perf.json`, if that hart was processed.
    pub fn vcpu_perf(&self, hart: u32) -> Option<PerformanceData> {
        let path = self.dir.join(format!("turbo_vcpu_{hart}_perf.json"));
        path.exists().then(|| {
            PerformanceData::load(path)
                .expect("turbo_vcpu_<N>_perf.json is readable and well-formed")
        })
    }

    pub fn has_vcpu_perf(&self, hart: u32) -> bool {
        self.dir
            .join(format!("turbo_vcpu_{hart}_perf.json"))
            .exists()
    }

    /// This run's `turbo_trace_vcpu<N>.bin` record log, asserting it was written.
    pub fn trace_log(&self, vcpu: u32) -> PathBuf {
        let path = self
            .dir
            .join(turbo_tracer_shared::get_turbo_trace_bin_for_vcpu(vcpu));
        assert!(
            path.exists(),
            "trace log for vcpu{vcpu} not written: {}",
            path.display()
        );
        path
    }

    /// The `turbo_metadata.json` the tracer plugin wrote for this run, if any.
    /// Its `libs` list is the shared objects the plugin observed being mapped --
    /// empty when a dynamically-linked guest died in its loader before mapping
    /// any (e.g. a missing `LD_LIBRARY_PATH`).
    pub fn metadata(&self) -> Option<turbo_tracer_shared::TurboMetadata> {
        turbo_tracer_shared::TurboMetadata::read_from_workdir(&self.dir)
    }

    /// Every recorded `(hart, log)` pair, sorted by hart. A multi-hart guest
    /// must yield more than one, or a "covers every hart" test passes vacuously.
    pub fn trace_logs(&self) -> Vec<(u32, PathBuf)> {
        let mut logs: Vec<(u32, PathBuf)> = std::fs::read_dir(&self.dir)
            .expect("output dir readable")
            .flatten()
            .filter_map(|e| {
                let path = e.path();
                let hart = path
                    .file_stem()?
                    .to_str()?
                    .strip_prefix("turbo_trace_vcpu")?
                    .parse()
                    .ok()?;
                (path.extension()? == "bin").then_some((hart, path))
            })
            .collect();
        logs.sort();
        logs
    }

    /// The global instruction count -- the single number that must not vary
    /// with any execution-strategy choice (overlap mode, decode threads).
    pub fn global_instructions(&self) -> u64 {
        global_instructions(&self.perf_data())
    }

    /// A rendered annotate page, e.g. `annotate_index.html`. The on-disk
    /// artifact is gzipped; `read_page_gz` is the production reader for it.
    pub fn annotate_page(&self, name: &str) -> String {
        let html_dir = self.dir.join("html");
        read_page_gz(&html_dir, name)
            .unwrap_or_else(|e| panic!("failed to read {name}.gz from {html_dir:?}: {e}"))
    }

    /// Per-function annotate pages present, excluding the index and hotspots.
    pub fn annotate_function_pages(&self) -> Vec<String> {
        let html_dir = self.dir.join("html");
        let mut pages: Vec<String> = std::fs::read_dir(&html_dir)
            .unwrap_or_else(|e| panic!("{html_dir:?} must be readable: {e}"))
            .flatten()
            .filter_map(|e| e.file_name().to_str().map(|n| n.to_string()))
            .filter(|n| {
                n.starts_with("annotate_")
                    && n.ends_with(".html.gz")
                    && n != "annotate_index.html.gz"
                    && n != "annotate_hotspots.html.gz"
            })
            .map(|n| n.trim_end_matches(".gz").to_string())
            .collect();
        pages.sort();
        pages
    }

    /// `turbo process` over this run's recording, into a fresh `name`
    /// directory. `input` is a spool directory or a single `turbo_trace_vcpu<N>.bin`.
    pub fn reprocess(
        &self,
        name: &str,
        input: impl AsRef<Path>,
        f: impl FnOnce(&mut ProcessArgs),
    ) -> TestRun {
        self.reprocess_with_cpu_config(name, input, cpu_config_path(), f)
    }

    /// [`TestRun::reprocess`] against a specific `cpu_config.json`, for the tests
    /// that re-simulate one recording under several CPU configurations.
    pub fn reprocess_with_cpu_config(
        &self,
        name: &str,
        input: impl AsRef<Path>,
        cpu_config: impl Into<PathBuf>,
        f: impl FnOnce(&mut ProcessArgs),
    ) -> TestRun {
        let dir = out_dir(name);
        let mut process = ProcessArgs::default();
        f(&mut process);
        main_interface(
            Command::Process {
                common: CommonArgs::new(&self.bin, Some(dir.clone())).with_cpu_config(cpu_config),
                process,
                decode_only: false,
                dump_pcs: false,
                trace_input: input.as_ref().to_path_buf(),
            }
            .into(),
        )
        .expect("`turbo process` failed");
        TestRun {
            dir,
            bin: self.bin.clone(),
        }
    }

    /// `turbo process --dump-pcs` over this run's spool, returning the path of
    /// the `turbo_trace_pcs.txt` it writes into this run's own directory (beside the
    /// `execlog_out.txt` / `qemutrace.prv` a `--reference-traces` recording left
    /// there, which is what makes the three-way comparison possible).
    pub fn dump_pcs(&self) -> PathBuf {
        main_interface(
            Command::Process {
                common: CommonArgs::new(&self.bin, Some(self.dir.clone()))
                    .with_cpu_config(cpu_config_path()),
                process: ProcessArgs::default(),
                decode_only: false,
                dump_pcs: true,
                trace_input: self.dir.clone(),
            }
            .into(),
        )
        .expect("`turbo process --dump-pcs` failed");

        let path = self.dir.join(turbo::workflow::TRACE_PCS_FILE);
        assert!(path.exists(), "--dump-pcs wrote no {}", path.display());
        path
    }

    /// `turbo process` over this run's whole spool directory.
    pub fn reprocess_dir(&self, name: &str) -> TestRun {
        let dir = self.dir.clone();
        self.reprocess(name, dir, |_| {})
    }
}

/// The `Global` ROI's instruction count.
pub fn global_instructions(pd: &PerformanceData) -> u64 {
    pd.rois
        .iter()
        .find(|r| &*r.name == "Global")
        .map(|r| r.instructions)
        .expect("Global ROI present")
}

// ---------------------------------------------------------------------------
// Expected-metrics fixtures (shared by the assembly-validation test binaries)
// ---------------------------------------------------------------------------

/// Load expected metrics from JSON file
pub fn load_expected_metrics() -> HashMap<String, Vec<RoiInfo>> {
    let metrics_file = workspace_root().join("test_bins/asm_validation/expected_metrics.json");
    let json = std::fs::read_to_string(&metrics_file)
        .unwrap_or_else(|e| panic!("Failed to read {}: {e}", metrics_file.display()));
    serde_json::from_str(&json).expect("Failed to deserialize expected_metrics.json")
}

/// Validate that actual metrics match expected metrics
pub fn validate_metrics(test_name: &str, run: &TestRun, expected_rois: &[RoiInfo]) {
    assert_eq!(
        cpu_config().vlen,
        Some(FIXTURE_VLEN_BITS),
        "assembly validation needs VLEN {FIXTURE_VLEN_BITS}; call \
         skip_unless_fixture_cpu first"
    );
    let actual_pd = run.perf_data();

    for expected in expected_rois {
        let actual_roi = actual_pd
            .functions
            .iter()
            .find(|roi| roi.name == expected.name)
            .unwrap_or_else(|| {
                panic!(
                    "Function not found in outputted performance data: {}",
                    expected.name
                )
            });

        eprintln!(
            "Actual: {}",
            serde_json::to_string_pretty(&actual_roi).expect("Actual ROI must serialize")
        );
        eprintln!(
            "Expected: {}",
            serde_json::to_string_pretty(&expected).expect("Expected ROI must serialize")
        );

        actual_roi.assert_equals(expected, &format!("{}/{}", test_name, actual_roi.name));
    }
}
