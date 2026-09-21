use std::path::{Component, Path, PathBuf};

/// Join a guest-supplied relative path onto a host base directory, refusing any
/// component that would leave that directory.
///
/// The relative part comes out of the traced ELF -- `PT_INTERP`, the loader's
/// reported shared-library paths -- so it is only as trustworthy as the binary
/// being profiled. A plain `base.join(rel)` turns a `../../etc/shadow` in that
/// string into a read outside the sysroot, and an absolute `rel` makes `join`
/// discard `base` altogether. Both are rejected here rather than at each of the
/// three call sites, so the rule cannot drift between them.
///
/// `Some(base.join(rel))` when every component is a normal one (a leading `./`
/// is harmless and allowed), `None` otherwise. Callers turn `None` into
/// whatever "this library cannot be resolved" means for them.
pub fn join_under(base: &Path, rel: impl AsRef<Path>) -> Option<PathBuf> {
    let rel = rel.as_ref();
    let escapes = rel
        .components()
        .any(|c| !matches!(c, Component::Normal(_) | Component::CurDir));
    // nosemgrep: input-validation-path-traversal-2 -- this IS the traversal guard; every component was just checked
    (!escapes).then(|| base.join(rel))
}

pub fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf()
}

pub fn get_build_artifact_dir() -> PathBuf {
    let workspace = workspace_root();
    #[cfg(debug_assertions)]
    let profile = "debug";
    #[cfg(not(debug_assertions))]
    let profile = "release";

    let artifact_dir = format!("{}/target/{}", workspace.display(), profile);
    Path::new(&artifact_dir).to_path_buf()
}

pub fn extract_panic(panic: Box<dyn std::any::Any + Send>) -> String {
    if let Some(s) = panic.downcast_ref::<&str>() {
        s.to_string()
    } else if let Some(s) = panic.downcast_ref::<String>() {
        s.to_string()
    } else {
        "No error message".to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn join_under_accepts_ordinary_relative_paths() {
        let base = Path::new("/usr/riscv64-linux-gnu");
        assert_eq!(
            join_under(base, "lib/ld-linux-riscv64-lp64d.so.1"),
            Some(base.join("lib/ld-linux-riscv64-lp64d.so.1"))
        );
        assert_eq!(
            join_under(base, "./lib/libc.so.6"),
            Some(base.join("./lib/libc.so.6"))
        );
    }

    /// The whole point: a guest ELF naming `../` must not read outside the
    /// sysroot, and an absolute path must not silently replace the base.
    #[test]
    fn join_under_rejects_escapes() {
        let base = Path::new("/usr/riscv64-linux-gnu");
        assert_eq!(join_under(base, "../../etc/shadow"), None);
        assert_eq!(join_under(base, "lib/../../../etc/shadow"), None);
        assert_eq!(join_under(base, "/etc/shadow"), None);
    }
}
