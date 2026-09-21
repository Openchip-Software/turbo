//! Build script: stamps the git commit of this checkout into the binary as
//! the `GIT_COMMIT` compile-time env var, so `turbo --version` can report it
//! alongside the cargo package version.

use std::process::Command;

fn main() {
    // Re-run on every build so the stamp tracks the current HEAD. The output
    // is hashed by cargo, so identical output (same commit) keeps everything
    // else fresh.
    println!("cargo:rerun-if-changed=");

    let commit = Command::new("git")
        .args(["rev-parse", "--short", "HEAD"])
        .output()
        .ok()
        .filter(|out| out.status.success())
        .map(|out| String::from_utf8_lossy(&out.stdout).trim().to_string())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "unknown".to_string());

    println!("cargo:rustc-env=GIT_COMMIT={commit}");
}
