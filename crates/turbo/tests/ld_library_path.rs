//! `LD_LIBRARY_PATH` discovery of a shared library that is NOT next to the
//! binary.
//!
//! `dyn_addso_reloc.riscv` is linked against `libadd.so` with no rpath, and the
//! library lives in a separate `libs/` directory (see
//! `test_bins/linker_flavours/Makefile`). The dynamic loader can therefore only
//! resolve it via `LD_LIBRARY_PATH`:
//!
//!   * without the env var the guest dies in its loader ("error while loading
//!     shared libraries: libadd.so") before `main` runs -- the tracer plugin
//!     records no shared-library mappings;
//!   * with `LD_LIBRARY_PATH=libs` the guest runs to completion and `libadd.so`
//!     shows up in the run's mapped libraries.
//!
//! The pass/fail signal is the tracer plugin's `turbo_metadata.json`: its
//! `libs` list is what the plugin observed being mapped, so it is empty exactly
//! when the loader could not find the `.so`.

use turbo::workspace_root;

use crate::harness::TurboTest;

const RELOC_BIN: &str = "test_bins/linker_flavours/dyn_addso_reloc.riscv";

/// Absolute path to the `libs/` directory holding the relocated `libadd.so`.
fn libs_dir() -> String {
    workspace_root()
        .join("test_bins/linker_flavours/libs")
        .display()
        .to_string()
}

/// True when the run's metadata shows `libadd.so` among the mapped libraries.
fn mapped_libadd(run: &crate::harness::TestRun) -> bool {
    run.metadata()
        .map(|m| m.libs.iter().any(|l| l.path.ends_with("libadd.so")))
        .unwrap_or(false)
}

/// Without `LD_LIBRARY_PATH` the loader cannot find `libadd.so`, so the guest
/// fails before mapping it: the run must NOT record `libadd.so` as mapped.
#[test]
fn ld_library_path_missing_fails() {
    let run = TurboTest::new("ld_library_path_missing_fails", RELOC_BIN)
        .try_record()
        .expect("`turbo record` itself must not error; the *guest* is what fails");

    assert!(
        !mapped_libadd(&run),
        "libadd.so was mapped without LD_LIBRARY_PATH; expected the loader to fail. \
         metadata.libs = {:?}",
        run.metadata().map(|m| m.libs)
    );
}

/// With `LD_LIBRARY_PATH` pointing at `libs/`, the loader finds `libadd.so` and
/// the guest runs to completion: the run must record `libadd.so` as mapped.
#[test]
fn ld_library_path_set_passes() {
    let run = TurboTest::new("ld_library_path_set_passes", RELOC_BIN)
        .guest_env("LD_LIBRARY_PATH", &libs_dir())
        .try_record()
        .expect("`turbo record` failed");

    assert!(
        mapped_libadd(&run),
        "libadd.so was not mapped even with LD_LIBRARY_PATH set. metadata.libs = {:?}",
        run.metadata().map(|m| m.libs)
    );
}
