//! `turbo config init` / `turbo config check`, driven through the real binary.
//!
//! These spawn `turbo` rather than calling `main_interface()` in-process,
//! because both verbs read `HOME` and `PATH`: changing those in the shared test
//! process would race every other test that loads the default config.

use std::fs;
use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

/// A fresh scratch directory to stand in for `HOME`.
fn scratch_home(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("turbo_config_cmd_{name}"));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    dir
}

fn turbo(home: &Path, path: &str, args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_turbo"))
        .args(args)
        .env("HOME", home)
        .env("PATH", path)
        // `config check` reports through the logger, on stderr.
        .env("RUST_LOG", "info")
        .output()
        .expect("turbo runs")
}

fn stdout(out: &Output) -> String {
    String::from_utf8_lossy(&out.stdout).into_owned()
}

fn stderr(out: &Output) -> String {
    String::from_utf8_lossy(&out.stderr).into_owned()
}

/// `init` writes both files, with `qemu_bin` resolved from PATH to an absolute
/// path; a second `init` keeps them, and `--force` replaces them.
#[test]
fn init_writes_config_and_keeps_existing_files() {
    let home = scratch_home("init");
    let bin = home.join("bin");
    fs::create_dir_all(&bin).unwrap();
    let fake_qemu = bin.join("qemu-riscv64");
    fs::write(&fake_qemu, "#!/bin/sh\n").unwrap();
    fs::set_permissions(&fake_qemu, fs::Permissions::from_mode(0o755)).unwrap();
    let path = bin.to_str().unwrap();

    let out = turbo(&home, path, &["config", "init"]);
    assert!(out.status.success(), "{out:?}");
    let dir = home.join(".config/openchip");
    let env: serde_json::Value =
        serde_json::from_str(&fs::read_to_string(dir.join("environment_config.json")).unwrap())
            .unwrap();
    assert_eq!(env["qemu_bin"], fake_qemu.to_str().unwrap());
    assert!(env["turbo_tracer_plugin"]
        .as_str()
        .unwrap()
        .ends_with("libturbo_tracer.so"));
    // Unset fields are left out rather than written as `null`.
    assert!(env.get("qemu_plugin_dir").is_none());
    let cpu: serde_json::Value =
        serde_json::from_str(&fs::read_to_string(dir.join("cpu_config.json")).unwrap()).unwrap();
    assert!(cpu["vlen"].is_u64());

    fs::write(dir.join("cpu_config.json"), "{\"vlen\": 256}").unwrap();
    let out = turbo(&home, path, &["config", "init"]);
    assert!(out.status.success(), "{out:?}");
    assert!(stdout(&out).contains("kept"));
    assert_eq!(
        fs::read_to_string(dir.join("cpu_config.json")).unwrap(),
        "{\"vlen\": 256}"
    );

    let out = turbo(&home, path, &["config", "init", "--force"]);
    assert!(out.status.success(), "{out:?}");
    assert_ne!(
        fs::read_to_string(dir.join("cpu_config.json")).unwrap(),
        "{\"vlen\": 256}"
    );
}

/// `check` exits 1 and names the field at fault when QEMU is missing, and
/// skips the test run it cannot do.
#[test]
fn check_fails_on_missing_qemu() {
    let home = scratch_home("check_missing_qemu");
    let env = home.join("env.json");
    fs::write(&env, r#"{"qemu_bin": "/nonexistent/qemu-riscv64"}"#).unwrap();

    let out = turbo(
        &home,
        "/nonexistent",
        &["config", "check", "--env-config", env.to_str().unwrap()],
    );
    assert_eq!(out.status.code(), Some(1), "{out:?}");
    let report = stderr(&out);
    assert!(
        report.contains("ERROR turbo::config_cmd] qemu_bin:")
            && report.contains("test run: skipped"),
        "{report}"
    );
}

/// With no `cpu_config.json` anywhere and no `--vlen`, a recording runs at the
/// 1024-bit default. Read back from the metadata the plugin wrote, i.e. what
/// QEMU actually ran with, not what turbo meant to ask for.
#[test]
fn record_without_a_vlen_runs_at_1024_bits() {
    use turbo::config::{ConfigFile, EnvConfig};

    if Path::new("/usr/share/turbo/cpu_config.json").exists() {
        eprintln!("SKIP: a packaged /usr/share/turbo/cpu_config.json would set the VLEN");
        return;
    }
    // This machine's QEMU and plugin, but a HOME with no cpu_config.json.
    let home = scratch_home("default_vlen");
    let env = EnvConfig::from_default_location()
        .expect("the machine's environment_config.json parses")
        .map(|(env, _)| env)
        .unwrap_or_default();
    env.write_to(&home.join(".config/openchip/environment_config.json"))
        .unwrap();

    let bin = turbo::workspace_root().join("test_bins/basic_branch_asm/add_1k.riscv");
    let out = Command::new(env!("CARGO_BIN_EXE_turbo"))
        .args(["record", "--output-dir", "default_vlen"])
        .arg(&bin)
        .current_dir(&home)
        .env("HOME", &home)
        .env("RUST_LOG", "off")
        .output()
        .expect("turbo runs");
    assert!(out.status.success(), "{}", stderr(&out));

    let metadata =
        turbo_tracer_shared::TurboMetadata::read_from_workdir(&home.join("output/default_vlen"))
            .expect("the plugin wrote its metadata");
    assert_eq!(metadata.vlen_bits, 1024);
}

/// `check` on this machine's own configuration -- the one every QEMU-driven
/// test in this suite runs with -- passes, test run included.
#[test]
fn check_passes_on_the_test_configuration() {
    let home = std::env::home_dir().expect("a home directory");
    let path = std::env::var("PATH").unwrap_or_default();
    let out = turbo(&home, &path, &["config", "check"]);
    let report = stderr(&out);
    assert!(out.status.success(), "{report}");
    assert!(
        report.contains("INFO  turbo::config_cmd] test run: QEMU loaded"),
        "{report}"
    );
}
