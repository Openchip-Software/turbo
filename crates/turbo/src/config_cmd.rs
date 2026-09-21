//! `turbo config init` and `turbo config check`.
//!
//! A source build has no install step to put config files in place, and a
//! wrong path in one is only discovered when a run fails -- often as QEMU
//! refusing the tracer plugin, several layers away from the file at fault.
//! `init` writes files that point at what this machine actually has; `check`
//! then exercises that configuration end to end, by the same rules `run` uses
//! to find it, without needing a guest binary of the user's own.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use anyhow::{anyhow, Context};
use object::{Object, ObjectSection, ObjectSymbol};

use crate::config::{user_config_dir, ConfigFile, CpuConfig, EnvConfig};
use crate::source::{resolve_qemu_binary, QemuRunner, VLEN_BITS_MAX, VLEN_BITS_MIN};

/// The QEMU release series turbo-tracer is built and tested against. It is the
/// last series with plugin API v6; 11.1 requires v7 and refuses a v6 plugin.
const TESTED_QEMU_SERIES: (u32, u32) = (11, 0);

/// The `cpu_config.json` a package installs, so `init` and a packaged install
/// model the same CPU.
const SHIPPED_CPU_CONFIG: &str = include_str!("cpu_config.json");

/// The dynamic linker a glibc riscv64 guest asks for, relative to the sysroot.
const RISCV_GLIBC_INTERPRETER: &str = "lib/ld-linux-riscv64-lp64d.so.1";

// ============================================================================
// init
// ============================================================================

/// `turbo config init`: write both config files into `dir` (default
/// `~/.config/openchip/`), keeping any that exist unless `force`.
pub fn init(dir: Option<PathBuf>, force: bool) -> anyhow::Result<()> {
    let dir = dir
        .or_else(user_config_dir)
        .ok_or_else(|| anyhow!("cannot find a home directory; pass --dir"))?;

    let (env_config, notes) = detect_env_config();
    // nosemgrep: input-validation-path-traversal-2 -- --dir is the user's chosen target; FILE_NAME is a const
    let env_path = dir.join(EnvConfig::FILE_NAME);
    // nosemgrep: input-validation-path-traversal-2 -- --dir is the user's chosen target; FILE_NAME is a const
    let cpu_path = dir.join(CpuConfig::FILE_NAME);

    if write_unless_exists(&env_path, force, |path| env_config.write_to(path))? {
        for note in &notes {
            println!("  note: {note}");
        }
    }
    write_unless_exists(&cpu_path, force, |path| {
        fs::write(path, SHIPPED_CPU_CONFIG)
            .with_context(|| format!("Failed to write {}", path.display()))
    })?;

    println!("Next: run `turbo config check` to verify QEMU and the tracer plugin.");
    Ok(())
}

/// Run `write` on `path` unless the file exists and `force` is off. Returns
/// whether it wrote.
fn write_unless_exists(
    path: &Path,
    force: bool,
    write: impl FnOnce(&Path) -> anyhow::Result<()>,
) -> anyhow::Result<bool> {
    if path.exists() && !force {
        println!(
            "kept    {} (already exists; --force overwrites it)",
            path.display()
        );
        return Ok(false);
    }
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("Failed to create {}", parent.display()))?;
    }
    write(path)?;
    println!("wrote   {}", path.display());
    Ok(true)
}

/// The environment config for this machine, and a note for each path that
/// could not be found and so will need editing.
///
/// Starts from the runner's built-in defaults, so a field this cannot improve
/// on is written as the value a run with no file would use anyway -- the file
/// then documents the defaults instead of changing them.
fn detect_env_config() -> (EnvConfig, Vec<String>) {
    let defaults = QemuRunner::default();
    let mut notes = Vec::new();

    let qemu = defaults.path_config().qemu_binary.clone();
    let qemu_bin = match resolve_qemu_binary(&qemu) {
        Ok(found) => found,
        Err(_) => {
            notes.push(format!(
                "{} is not on PATH; set qemu_bin to the QEMU 11.0.x you built",
                qemu.display()
            ));
            qemu
        }
    };

    // This build's plugin first, then where a package installs it.
    let built = defaults.plugin_config().tracer_plugin_path();
    let installed = PathBuf::from("/usr/lib/libturbo_tracer.so");
    let turbo_tracer_plugin = if built.exists() || !installed.exists() {
        if !built.exists() {
            notes.push(format!(
                "{} does not exist yet; build it with `cargo build --release`",
                built.display()
            ));
        }
        built
    } else {
        installed
    };

    let loader_path = defaults.path_config().loader_path.clone();
    if !loader_path.join(RISCV_GLIBC_INTERPRETER).exists() {
        notes.push(format!(
            "no riscv64 glibc sysroot at {}; dynamically linked guests need one \
             (Debian/Ubuntu: libc6-riscv64-cross)",
            loader_path.display()
        ));
    }

    let config = EnvConfig {
        qemu_bin: Some(qemu_bin),
        loader_path: Some(loader_path),
        turbo_tracer_plugin: Some(turbo_tracer_plugin),
        ..EnvConfig::default()
    };
    (config, notes)
}

// ============================================================================
// check
// ============================================================================

#[derive(Clone, Copy, PartialEq, Eq)]
enum Status {
    Ok,
    Warn,
    Fail,
    Skip,
}

/// Logs each check as it completes, and counts the failures.
#[derive(Default)]
struct Report {
    failures: usize,
    warnings: usize,
}

impl Report {
    fn line(&mut self, status: Status, what: &str, detail: impl std::fmt::Display) {
        match status {
            Status::Ok => log::info!("{what}: {detail}"),
            Status::Skip => log::info!("{what}: skipped, {detail}"),
            Status::Warn => {
                self.warnings += 1;
                log::warn!("{what}: {detail}");
            }
            Status::Fail => {
                self.failures += 1;
                log::error!("{what}: {detail}");
            }
        }
    }
}

/// `turbo config check`: verify the configuration `record`/`run` would use.
pub fn check(env_flag: Option<&Path>, cpu_flag: Option<&Path>) -> anyhow::Result<()> {
    let mut report = Report::default();

    let env_config = check_env_config(&mut report, env_flag);
    let vlen = check_cpu_config(&mut report, cpu_flag, env_config.as_ref());

    let mut runner = env_config.map(QemuRunner::from).unwrap_or_default();
    if let Some(vlen) = vlen {
        runner = runner.vlen(vlen);
    }

    let qemu = check_qemu(&mut report, &runner);
    let vlen_ok = qemu.is_some() && check_vlen(&mut report, &runner);
    let plugin_ok = check_tracer_plugin(&mut report, &runner);
    check_sysroot(&mut report, &runner);

    match qemu {
        Some(true) if plugin_ok && vlen_ok => check_test_run(&mut report, &runner),
        _ => report.line(
            Status::Skip,
            "test run",
            "needs a launchable QEMU with plugin support, a VLEN it accepts, and a tracer plugin",
        ),
    }

    if report.failures == 0 {
        log::info!(
            "All checks passed{}.",
            if report.warnings > 0 {
                ", with warnings"
            } else {
                ""
            }
        );
        Ok(())
    } else {
        Err(anyhow!(
            "{} check(s) failed; see the Configuration section of the README",
            report.failures
        ))
    }
}

/// Load the environment config the way `record`/`run` do, reporting where it
/// came from. A file that exists but does not parse is a failure here, even
/// though `run` only warns and carries on with the defaults: that is exactly
/// the silent fallback this command exists to catch.
fn check_env_config(report: &mut Report, flag: Option<&Path>) -> Option<EnvConfig> {
    let what = EnvConfig::FILE_NAME;
    let loaded = match flag {
        Some(path) => EnvConfig::from_file(path).map(|c| Some((c, path.to_path_buf()))),
        None => EnvConfig::from_default_location(),
    };
    match loaded {
        Ok(Some((config, path))) => {
            report.line(Status::Ok, what, path.display());
            Some(config)
        }
        Ok(None) => {
            report.line(
                Status::Ok,
                what,
                format!(
                    "none found in {}; using built-in defaults (`turbo config init` writes one)",
                    EnvConfig::searched_locations()
                ),
            );
            None
        }
        Err(e) => {
            let path = flag.map_or_else(EnvConfig::default_location, Path::to_path_buf);
            report.line(Status::Fail, what, format!("{}: {e}", path.display()));
            None
        }
    }
}

/// Load and validate the CPU config, returning the VLEN it asks for.
fn check_cpu_config(
    report: &mut Report,
    flag: Option<&Path>,
    env_config: Option<&EnvConfig>,
) -> Option<u32> {
    let what = CpuConfig::FILE_NAME;
    let resolved = match crate::main_interface::load_cpu_config_from(flag, env_config) {
        Ok(resolved) => resolved,
        Err(e) => {
            report.line(Status::Fail, what, e);
            return None;
        }
    };
    let source = if resolved.path.exists() {
        resolved.path.display().to_string()
    } else {
        "none found; using built-in defaults".to_string()
    };

    let vlen = resolved.config.vlen;
    let vlen_desc = match vlen {
        Some(v) if !v.is_power_of_two() || !(VLEN_BITS_MIN..=VLEN_BITS_MAX).contains(&v) => {
            report.line(
                Status::Fail,
                what,
                format!(
                    "{source}: vlen {v} is not a power of two in [{VLEN_BITS_MIN}, {VLEN_BITS_MAX}]"
                ),
            );
            return None;
        }
        Some(v) => format!("vlen {v}"),
        None => format!("vlen {} (default)", crate::common::DEFAULT_VLEN_BITS),
    };
    // `cache` is optional and only read by `--cache-sim`, so its absence is
    // not worth a mention; a block that is present must be valid, though.
    let cache_desc = match resolved.config.cache.map(|c| c.validate()) {
        None => String::new(),
        Some(Ok(())) => ", cache block ok".to_string(),
        Some(Err(e)) => {
            report.line(
                Status::Fail,
                what,
                format!("{source}: invalid cache block: {e}"),
            );
            return vlen;
        }
    };
    report.line(
        Status::Ok,
        what,
        format!("{source} ({vlen_desc}{cache_desc})"),
    );
    vlen
}

/// Find QEMU, launch it for its version, and look for plugin support.
///
/// `None` if QEMU cannot be run at all, else whether it supports plugins.
fn check_qemu(report: &mut Report, runner: &QemuRunner) -> Option<bool> {
    let configured = &runner.path_config().qemu_binary;
    let qemu = match resolve_qemu_binary(configured) {
        Ok(qemu) => qemu,
        Err(e) => {
            report.line(Status::Fail, "qemu_bin", e);
            return None;
        }
    };
    report.line(Status::Ok, "qemu_bin", qemu.display());

    let version = match run_capture(&qemu, &["--version"]) {
        Ok(out) => out,
        Err(e) => {
            report.line(Status::Fail, "QEMU launch", e);
            return None;
        }
    };
    let first_line = version.lines().next().unwrap_or_default();
    match parse_qemu_version(&version) {
        Some((major, minor, patch)) if (major, minor) == TESTED_QEMU_SERIES => report.line(
            Status::Ok,
            "QEMU version",
            format!("{major}.{minor}.{patch}"),
        ),
        Some((major, minor, patch)) => report.line(
            Status::Warn,
            "QEMU version",
            format!(
                "{major}.{minor}.{patch}; turbo-tracer is built for plugin API v6 and tested \
                 with QEMU {}.{}.x (11.1 and later refuse v6 plugins)",
                TESTED_QEMU_SERIES.0, TESTED_QEMU_SERIES.1
            ),
        ),
        None => report.line(
            Status::Warn,
            "QEMU version",
            format!("could not parse `{first_line}`"),
        ),
    }

    let plugins = run_capture(&qemu, &["--help"])
        .map(|help| help.lines().any(|l| l.trim_start().starts_with("-plugin")))
        .unwrap_or(false);
    if plugins {
        report.line(Status::Ok, "QEMU plugin support", "yes");
    } else {
        report.line(
            Status::Fail,
            "QEMU plugin support",
            "no -plugin option; rebuild QEMU with --enable-plugins",
        );
    }
    Some(plugins)
}

/// Probe the VLEN range QEMU accepts, and check the configured VLEN is in it.
fn check_vlen(report: &mut Report, runner: &QemuRunner) -> bool {
    let what = "QEMU VLEN support";
    let vlen = runner.vlen_bits();
    match qemu_max_vlen(runner) {
        Ok(max) if vlen <= max => {
            report.line(
                Status::Ok,
                what,
                format!("{VLEN_BITS_MIN} to {max} bits (configured: {vlen})"),
            );
            true
        }
        Ok(max) => {
            report.line(
                Status::Fail,
                what,
                format!(
                    "{VLEN_BITS_MIN} to {max} bits, but the CPU config asks for {vlen}; \
                     lower `vlen` in cpu_config.json (upstream QEMU 11.0 stops at 1024)"
                ),
            );
            false
        }
        Err(e) => {
            report.line(Status::Fail, what, format!("{e:#}"));
            false
        }
    }
}

/// The largest VLEN, in bits, at which `runner`'s QEMU will start a guest.
///
/// RVV permits up to [`VLEN_BITS_MAX`], but a QEMU build may stop well short
/// of that -- upstream 11.0 refuses anything above 1024 -- and it only says so
/// on stderr, once asked to run a guest. So ask it: launch the exit-only guest
/// at each power of two upward from [`VLEN_BITS_MIN`], with no plugin, until
/// one is refused. QEMU's accepted range is contiguous, so the last VLEN that
/// ran is the maximum.
pub fn qemu_max_vlen(runner: &QemuRunner) -> anyhow::Result<u32> {
    use std::os::unix::fs::{DirBuilderExt, PermissionsExt};
    use std::sync::atomic::{AtomicU32, Ordering};

    // Unique per call, so concurrent probes (parallel tests) never share a guest.
    static PROBE: AtomicU32 = AtomicU32::new(0);

    let qemu = resolve_qemu_binary(&runner.path_config().qemu_binary)?;
    // nosemgrep: input-validation-path-traversal-2 -- name is built from pid + counter only
    let dir = std::env::temp_dir().join(format!(
        "turbo_vlen_probe_{}_{}",
        std::process::id(),
        PROBE.fetch_add(1, Ordering::Relaxed)
    ));
    // Not create_dir_all: fail if the directory already exists (e.g. planted by
    // another user in a shared /tmp), and keep it private to us.
    fs::DirBuilder::new().mode(0o700).create(&dir)?;
    let guest = dir.join("exit0.riscv");
    fs::write(&guest, exit_only_riscv_elf())?;
    fs::set_permissions(&guest, fs::Permissions::from_mode(0o755))?;

    let mut max = None;
    let mut refusal = String::new();
    let mut vlen = VLEN_BITS_MIN;
    while vlen <= VLEN_BITS_MAX {
        let cpu = crate::source::qemu::qemu_runner::CpuConfig {
            vlen,
            ..runner.cpu_config().clone()
        };
        // nosemgrep: rust-security-command-injection-new -- the configured QEMU, the same one `run` executes
        let out = Command::new(&qemu)
            .arg("-cpu")
            .arg(cpu.to_qemu_arg())
            .arg(&guest)
            .output();
        match out {
            Ok(out) if out.status.success() => max = Some(vlen),
            Ok(out) => {
                refusal = String::from_utf8_lossy(&out.stderr).trim().to_string();
                break;
            }
            Err(e) => {
                refusal = format!("cannot execute {}: {e}", qemu.display());
                break;
            }
        }
        vlen *= 2;
    }
    let _ = fs::remove_dir_all(&dir);

    max.ok_or_else(|| {
        anyhow!(
            "{} will not run a vector guest even at VLEN {VLEN_BITS_MIN}: {refusal}",
            qemu.display()
        )
    })
}

/// Check the tracer plugin exists and read the QEMU plugin API it targets.
fn check_tracer_plugin(report: &mut Report, runner: &QemuRunner) -> bool {
    let what = "turbo_tracer_plugin";
    let plugin = runner.plugin_config().tracer_plugin_path();
    if !plugin.is_file() {
        report.line(
            Status::Fail,
            what,
            format!(
                "{} does not exist; run `cargo build --release` or fix turbo_tracer_plugin",
                plugin.display()
            ),
        );
        return false;
    }
    match plugin_api_version(&plugin) {
        Ok(api) => {
            report.line(
                Status::Ok,
                what,
                format!("{} (QEMU plugin API v{api})", plugin.display()),
            );
            true
        }
        Err(e) => {
            report.line(
                Status::Fail,
                what,
                format!("{}: not a QEMU plugin: {e}", plugin.display()),
            );
            false
        }
    }
}

/// The sysroot only matters for dynamically linked guests, so its absence is
/// a warning: static guests, and this command's own test run, work without it.
fn check_sysroot(report: &mut Report, runner: &QemuRunner) {
    let loader = &runner.path_config().loader_path;
    if loader.join(RISCV_GLIBC_INTERPRETER).exists() {
        report.line(Status::Ok, "loader_path", loader.display());
    } else {
        report.line(
            Status::Warn,
            "loader_path",
            format!(
                "{} has no {RISCV_GLIBC_INTERPRETER}; only statically linked guests will run \
                 (Debian/Ubuntu: libc6-riscv64-cross)",
                loader.display()
            ),
        );
    }
}

/// Trace a tiny guest exactly as `record` would, then read back what the
/// plugin wrote with this build's readers.
///
/// This is the check that settles compatibility: QEMU itself decides whether
/// it accepts the plugin's API version, and the metadata and record-log
/// readers decide whether this turbo understands that plugin's output.
fn check_test_run(report: &mut Report, runner: &QemuRunner) {
    let what = "test run";
    // nosemgrep: input-validation-path-traversal-2 -- name is built from the pid only
    let dir = std::env::temp_dir().join(format!("turbo_config_check_{}", std::process::id()));
    let result = test_run_in(runner, &dir);
    let _ = fs::remove_dir_all(&dir);
    match result {
        Ok(()) => report.line(
            Status::Ok,
            what,
            "QEMU loaded the tracer plugin, and turbo read the trace it wrote",
        ),
        Err(e) => report.line(Status::Fail, what, format!("{e:#}")),
    }
}

fn test_run_in(runner: &QemuRunner, dir: &Path) -> anyhow::Result<()> {
    use std::os::unix::fs::{DirBuilderExt, PermissionsExt};

    let _ = fs::remove_dir_all(dir);
    // Not create_dir_all: fail if the directory already exists (e.g. planted by
    // another user in a shared /tmp), and keep it private to us.
    fs::DirBuilder::new().mode(0o700).create(dir)?;
    let work_dir = dir.join("output");
    fs::create_dir(&work_dir)?;
    let guest = dir.join("exit0.riscv");
    fs::write(&guest, exit_only_riscv_elf())?;
    fs::set_permissions(&guest, fs::Permissions::from_mode(0o755))?;

    let guest = guest.to_str().context("temporary directory is not UTF-8")?;
    let prepared = runner.prepare_run(guest, &[], Some(&work_dir))?;
    let result = runner.run_prepared(prepared.clone())?;
    if !result.success() {
        return Err(anyhow!(
            "{}",
            crate::source::qemu::qemu_failure_message(&result.output.status, &work_dir)
        ));
    }

    prepared
        .build_memory_map(false)
        .context("plugin metadata does not match this turbo")?;

    let trace = work_dir.join("turbo_trace_vcpu0.bin");
    let bytes = fs::read(&trace)
        .with_context(|| format!("the plugin wrote no trace ({})", trace.display()))?;
    match turbo_tracer_shared::check_record_log_header(&bytes) {
        Ok(Some(header)) if header.run_id == Some(prepared.run_id) => Ok(()),
        Ok(Some(_)) => Err(anyhow!("the trace carries a different run id")),
        Ok(None) => Err(anyhow!("the trace is truncated")),
        Err(e) => Err(anyhow!("{e}")),
    }
}

// ============================================================================
// Helpers
// ============================================================================

/// Run `program args...` and return its stdout, or why it could not be run.
fn run_capture(program: &Path, args: &[&str]) -> anyhow::Result<String> {
    // nosemgrep: rust-security-command-injection-new -- the configured QEMU, the same one `run` executes
    let out = Command::new(program)
        .args(args)
        .output()
        .with_context(|| format!("cannot execute {}", program.display()))?;
    if !out.status.success() {
        let stderr = String::from_utf8_lossy(&out.stderr);
        return Err(anyhow!(
            "{} {} failed ({}): {}",
            program.display(),
            args.join(" "),
            out.status,
            stderr.trim()
        ));
    }
    Ok(String::from_utf8_lossy(&out.stdout).into_owned())
}

/// `(major, minor, patch)` from `qemu-riscv64 --version` output, whose first
/// line reads `qemu-riscv64 version 11.0.0` (optionally followed by a
/// distribution suffix such as ` (Debian 1:11.0.0+ds-1)`).
fn parse_qemu_version(output: &str) -> Option<(u32, u32, u32)> {
    let first = output.lines().next()?;
    let version = first.split("version ").nth(1)?.split_whitespace().next()?;
    let mut parts = version.split('.').map(|p| {
        let digits: String = p.chars().take_while(char::is_ascii_digit).collect();
        digits.parse::<u32>().ok()
    });
    Some((
        parts.next()??,
        parts.next()??,
        parts.next().flatten().unwrap_or(0),
    ))
}

/// The `qemu_plugin_version` a QEMU plugin exports: the plugin API version it
/// was built against, which QEMU compares with the range it supports.
fn plugin_api_version(path: &Path) -> anyhow::Result<i32> {
    let data = fs::read(path)?;
    let file = object::File::parse(&*data)?;
    let symbol = file
        .dynamic_symbols()
        .chain(file.symbols())
        .find(|s| s.name() == Ok("qemu_plugin_version"))
        .context("no qemu_plugin_version symbol")?;
    let section = file.section_by_index(
        symbol
            .section_index()
            .context("qemu_plugin_version is undefined")?,
    )?;
    let bytes = section
        .data_range(symbol.address(), 4)?
        .context("qemu_plugin_version lies outside its section")?;
    Ok(i32::from_le_bytes(bytes.try_into()?))
}

/// A minimal static riscv64 Linux ELF that calls `exit_group(0)`.
///
/// Built here rather than shipped as a file, so `check` works from a packaged
/// install with no test binaries, and without a RISC-V toolchain. One `PT_LOAD`
/// segment maps the whole file, headers included; the entry point is the code
/// right after the program header.
fn exit_only_riscv_elf() -> Vec<u8> {
    const BASE: u64 = 0x10000;
    const EHDR: u16 = 64;
    const PHDR: u16 = 56;
    const EM_RISCV: u16 = 243;
    let code: [u32; 3] = [
        0x0000_0513, // li a0, 0
        0x05e0_0893, // li a7, 94 (exit_group)
        0x0000_0073, // ecall
    ];
    let code_off = u64::from(EHDR + PHDR);
    let total = code_off + 4 * code.len() as u64;

    let mut elf = Vec::new();
    // e_ident: ELFCLASS64, ELFDATA2LSB, EV_CURRENT, ELFOSABI_SYSV
    elf.extend_from_slice(b"\x7fELF\x02\x01\x01\x00");
    elf.extend_from_slice(&[0; 8]);
    elf.extend_from_slice(&2u16.to_le_bytes()); // e_type: ET_EXEC
    elf.extend_from_slice(&EM_RISCV.to_le_bytes());
    elf.extend_from_slice(&1u32.to_le_bytes()); // e_version
    elf.extend_from_slice(&(BASE + code_off).to_le_bytes()); // e_entry
    elf.extend_from_slice(&u64::from(EHDR).to_le_bytes()); // e_phoff
    elf.extend_from_slice(&0u64.to_le_bytes()); // e_shoff
    elf.extend_from_slice(&0u32.to_le_bytes()); // e_flags
    elf.extend_from_slice(&EHDR.to_le_bytes()); // e_ehsize
    elf.extend_from_slice(&PHDR.to_le_bytes()); // e_phentsize
    elf.extend_from_slice(&1u16.to_le_bytes()); // e_phnum
    elf.extend_from_slice(&64u16.to_le_bytes()); // e_shentsize
    elf.extend_from_slice(&0u16.to_le_bytes()); // e_shnum
    elf.extend_from_slice(&0u16.to_le_bytes()); // e_shstrndx

    elf.extend_from_slice(&1u32.to_le_bytes()); // p_type: PT_LOAD
    elf.extend_from_slice(&5u32.to_le_bytes()); // p_flags: R | X
    elf.extend_from_slice(&0u64.to_le_bytes()); // p_offset
    elf.extend_from_slice(&BASE.to_le_bytes()); // p_vaddr
    elf.extend_from_slice(&BASE.to_le_bytes()); // p_paddr
    elf.extend_from_slice(&total.to_le_bytes()); // p_filesz
    elf.extend_from_slice(&total.to_le_bytes()); // p_memsz
    elf.extend_from_slice(&0x1000u64.to_le_bytes()); // p_align

    for insn in code {
        elf.extend_from_slice(&insn.to_le_bytes());
    }
    elf
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn qemu_version_lines_parse() {
        let upstream = "qemu-riscv64 version 11.0.0\nCopyright (c) 2003-2026 Fabrice Bellard\n";
        assert_eq!(parse_qemu_version(upstream), Some((11, 0, 0)));
        let distro = "qemu-riscv64 version 10.1.2 (Debian 1:10.1.2+ds-1)\n";
        assert_eq!(parse_qemu_version(distro), Some((10, 1, 2)));
        assert_eq!(
            parse_qemu_version("qemu-riscv64 version 11.1\n"),
            Some((11, 1, 0))
        );
        assert_eq!(parse_qemu_version("something else"), None);
    }

    #[test]
    fn exit_only_elf_is_a_static_riscv_executable() {
        let elf = exit_only_riscv_elf();
        let file = object::File::parse(&*elf).expect("parses as an object file");
        assert_eq!(file.architecture(), object::Architecture::Riscv64);
        assert_eq!(file.kind(), object::ObjectKind::Executable);
        assert_eq!(file.entry(), 0x10000 + 64 + 56);
        assert_eq!(elf.len(), 64 + 56 + 12);
        // Its one program header is PT_LOAD: no PT_INTERP, so no sysroot is
        // needed to run it.
        assert_eq!(elf[64..68], 1u32.to_le_bytes());
    }

    #[test]
    fn built_tracer_plugin_reports_its_api_version() {
        let plugin = crate::get_build_artifact_dir().join("libturbo_tracer.so");
        if !plugin.exists() {
            eprintln!("SKIP: {} not built", plugin.display());
            return;
        }
        assert_eq!(plugin_api_version(&plugin).unwrap(), 6);
    }

    #[test]
    fn shipped_cpu_config_parses() {
        let config: CpuConfig = serde_json::from_str(SHIPPED_CPU_CONFIG).unwrap();
        assert!(config.vlen.is_some());
    }
}
