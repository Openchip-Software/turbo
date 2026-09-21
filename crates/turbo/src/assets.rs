//! Runtime access to the browser-side assets shipped in `src/server/assets/`.
//!
//! The dashboard's own `index.html` / `app.js` / `oc.css` were always read from
//! disk (the server hands the directory to a `ServeDir`); the annotation
//! report's stylesheet and scripts used to be `const &str` blobs compiled into
//! the binary instead. They now live beside the dashboard assets as real
//! `.css` / `.js` files and are read through here, so an editor sees them as the
//! languages they are and a tweak needs no rebuild.
//!
//! Resolution order matches the server's (see `server::http`): an explicit
//! [`set_dir`] override, then the packaged `/usr/share/turbo/assets`, then the
//! in-tree path from `CARGO_MANIFEST_DIR` for development builds.

use std::path::{Path, PathBuf};
use std::sync::OnceLock;

/// The packaged location, populated by the `assets` list in `Cargo.toml`.
const PROD_DIR: &str = "/usr/share/turbo/assets";

static DIR: OnceLock<PathBuf> = OnceLock::new();

/// Point asset lookup at `path` instead of the default locations. Only takes
/// effect if called before the first [`dir`]/[`read`]; returns whether it did,
/// so a caller that must know can tell (the server ignores it -- by the time it
/// serves anything, whatever directory was resolved is the one in use).
pub fn set_dir<P: Into<PathBuf>>(path: P) -> bool {
    DIR.set(path.into()).is_ok()
}

/// The directory the assets are read from, resolved once per process.
pub fn dir() -> &'static Path {
    DIR.get_or_init(|| {
        let prod = PathBuf::from(PROD_DIR);
        if prod.is_dir() {
            log::debug!("assets: using packaged path {}", prod.display());
            return prod;
        }
        let dev = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("src")
            .join("server")
            .join("assets");
        log::debug!("assets: using development path {}", dev.display());
        dev
    })
}

/// Read one asset by file name, e.g. `read("annotate.css")`.
///
/// A missing or unreadable file yields an empty string and one logged error
/// rather than a panic: a report page with no stylesheet is still a readable
/// page of numbers, and killing a completed profiling run over a cosmetic
/// asset would throw away the work. The `assets_present` test below is what
/// keeps a rename from getting that far.
pub fn read(name: &str) -> String {
    // nosemgrep: input-validation-path-traversal-2 -- name is always a literal asset name; assets_present pins the set
    let path = dir().join(name);
    match std::fs::read_to_string(&path) {
        Ok(s) => s,
        Err(e) => {
            log::error!("assets: cannot read {}: {e}", path.display());
            String::new()
        }
    }
}

#[cfg(test)]
mod tests {
    /// Every asset the annotation report asks for by name must exist in the
    /// directory that ships. These are looked up as strings at runtime, so a
    /// rename or a missed `Cargo.toml` glob is invisible to the compiler; this
    /// is the check that isn't.
    #[test]
    fn assets_present() {
        for name in [
            "annotate.css",
            "annotate_hide_src.js",
            "annotate_arrow.js",
            "annotate_region.js",
            "annotate_hover.js",
            "flamegraph.css",
            "flamegraph.js",
            "flamegraph_theme.css",
        ] {
            let s = super::read(name);
            assert!(
                !s.is_empty(),
                "asset {name} is missing or empty in {}",
                super::dir().display()
            );
        }
    }

    /// The dashboard's favicon. Binary, so it goes through `fs`, not `read`;
    /// it is requested by `index.html` and served by the static fallback, so
    /// nothing in the compiler's view would notice it going missing from the
    /// shipped directory.
    #[test]
    fn favicon_present() {
        let path = super::dir().join("icon.png");
        let len = std::fs::metadata(&path).map(|m| m.len()).unwrap_or(0);
        assert!(len > 0, "favicon missing or empty at {}", path.display());
    }
}
