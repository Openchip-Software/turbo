//! QEMU execution and trace capture integration.
//!
//! This module handles launching QEMU with the E-Trace plugin, receiving live trace data,
//! and managing the runtime memory map (shared libraries, VDSO regions).
//!
//! ## Components
//!
//! - [`qemu_runner`]: Builds and executes QEMU with turbo tracer plugin and optional ROI marker support
//! - [`memory_map`]: Manages virtual address space (main binary, shared libs, VDSO)

/// Filename (inside the run's working directory) that QEMU's stderr is
/// redirected to. Carries the `-d plugin,page` layout output.
pub const QEMU_STDERR_FILE: &str = "qemu_stderr.txt";

/// Filename (inside the run's working directory) that QEMU's stdout is
/// redirected to. Mostly the guest program's own console output.
pub const QEMU_STDOUT_FILE: &str = "qemu_stdout.txt";

/// How many trailing lines of [`QEMU_STDERR_FILE`] a QEMU failure quotes.
pub const QEMU_STDERR_TAIL_LINES: usize = 5;

/// Describe a QEMU that exited with `status`, quoting the last
/// [`QEMU_STDERR_TAIL_LINES`] non-empty lines of its stderr in `work_dir`.
///
/// QEMU reports why it refused to start (an unsupported `vlen=`, a plugin of
/// the wrong API version, a missing interpreter) on stderr and nowhere else, so
/// a failure that only names the file makes every user go and open it.
pub fn qemu_failure_message(
    status: &std::process::ExitStatus,
    work_dir: &std::path::Path,
) -> String {
    let path = work_dir.join(QEMU_STDERR_FILE);
    let stderr = std::fs::read_to_string(&path).unwrap_or_default();
    let lines: Vec<&str> = stderr.lines().filter(|l| !l.trim().is_empty()).collect();
    if lines.is_empty() {
        return format!(
            "QEMU exited with a failure status ({status}); {} is empty",
            path.display()
        );
    }
    let tail = &lines[lines.len().saturating_sub(QEMU_STDERR_TAIL_LINES)..];
    format!(
        "QEMU exited with a failure status ({status}); last lines of {}:\n    {}",
        path.display(),
        tail.join("\n    ")
    )
}

pub mod memory_map;
pub(crate) mod qemu_runner;

pub use memory_map::{MemoryMap, MissingInterpreter};
pub use qemu_runner::{
    resolve_qemu_binary, QemuError, QemuResult, QemuRunner, VLEN_BITS_MAX, VLEN_BITS_MIN,
};

#[cfg(test)]
mod tests {
    use super::*;
    use std::os::unix::process::ExitStatusExt;

    #[test]
    fn failure_message_quotes_the_last_stderr_lines() {
        let dir = std::env::temp_dir().join(format!("turbo_stderr_tail_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let status = std::process::ExitStatus::from_raw(1 << 8);

        let _ = std::fs::remove_file(dir.join(QEMU_STDERR_FILE));
        assert!(qemu_failure_message(&status, &dir).ends_with("qemu_stderr.txt is empty"));

        let lines: Vec<String> = (1..=8).map(|i| format!("line {i}")).collect();
        std::fs::write(dir.join(QEMU_STDERR_FILE), lines.join("\n\n") + "\n").unwrap();
        let msg = qemu_failure_message(&status, &dir);
        let _ = std::fs::remove_dir_all(&dir);

        assert!(msg.contains("exit status: 1"), "{msg}");
        let quoted: Vec<&str> = msg.lines().skip(1).map(str::trim).collect();
        assert_eq!(quoted, ["line 4", "line 5", "line 6", "line 7", "line 8"]);
    }
}
