//! Configuration files, split by what they describe.
//!
//! Two files, each looked up in `~/.config/openchip/` if present and otherwise
//! in `/usr/share/turbo/`:
//!
//! * [`EnvConfig`] (`environment_config.json`) -- where this machine keeps
//!   QEMU, its plugin directory, the guest sysroot and the tracer plugin.
//! * [`CpuConfig`] (`cpu_config.json`) -- what the modelled CPU *is*: its VLEN
//!   and, for `--cache-sim`, its cache geometry.
//!
//! They are separate because they change for different reasons -- a new
//! checkout moves the paths, a new target core changes the CPU -- and because
//! `process` needs only the second: it decodes a recording and never launches
//! QEMU. Loading them independently means a machine with no QEMU can still
//! `process --cache-sim`.
//!
//! Which `cpu_config.json` to read is itself part of the environment, so
//! [`EnvConfig::default_cpu_config`] can name one for a machine that models a
//! core other than the default file's.
//!
//! The single `performance_config.json` that used to hold both is still read as
//! a fallback; see [`ConfigFile::from_default_location`].

use anyhow::anyhow;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::{Path, PathBuf};

use crate::processing::cache_model;
use crate::source::QemuRunner;

/// The pre-split config file, read only as a fallback so that an existing
/// developer setup keeps working. It carried the environment paths and `vlen`,
/// which is why both loaders can fall back to it -- and never a `cache` block,
/// which is why `--cache-sim` cannot be satisfied by it.
const LEGACY_CONFIG_FILE_NAME: &str = "performance_config.json";

/// A JSON config file with a fixed name, loadable from the default location.
///
/// A trait rather than the load logic written twice: the two configs differ only
/// in their file name, and a copy-pasted loader is where the two would drift
/// apart on which errors are fatal and which are warnings.
pub trait ConfigFile: Serialize + DeserializeOwned + Default + std::fmt::Debug {
    /// File name, in `~/.config/openchip/` or `/usr/share/turbo/`.
    const FILE_NAME: &'static str;

    fn from_file(path: impl AsRef<Path>) -> anyhow::Result<Self> {
        log::trace!(
            "attempting to load {} from {}",
            std::any::type_name::<Self>(),
            path.as_ref().display()
        );
        match fs::read_to_string(path.as_ref()) {
            // nosemgrep: deserialization-from-untrusted-source -- parse failure is matched into an Err with the path logged
            Ok(content) => match serde_json::from_str(&content) {
                Ok(config) => {
                    log::debug!(
                        "Successfully loaded config from {}",
                        path.as_ref().display()
                    );
                    Ok(config)
                }
                Err(e) => {
                    log::error!(
                        "Failed to deserialize config from {}: {}",
                        path.as_ref().display(),
                        e
                    );
                    Err(anyhow!("Failed to deserialise config: {}", e))
                }
            },
            Err(e) => {
                log::warn!(
                    "Failed to read config file from {}: {}",
                    path.as_ref().display(),
                    e
                );
                Err(e.into())
            }
        }
    }

    /// The user's copy if there is one, else the system-wide install.
    fn default_location() -> PathBuf {
        default_location(Self::FILE_NAME)
    }

    /// Load from [`ConfigFile::default_location`], falling back to a
    /// `performance_config.json` in the same directory. `Ok(None)` when neither
    /// exists, which is a normal state (a source build with no install step),
    /// not an error; a file that exists but cannot be read or parsed is one.
    ///
    /// The fallback exists because the failure mode of silently ignoring a
    /// pre-split config is bad: `qemu_bin` reverts to whatever `qemu-riscv64` is
    /// on `PATH`, so the run fails a long way from the cause. The warning names
    /// the file to create instead.
    fn from_default_location() -> anyhow::Result<Option<(Self, PathBuf)>> {
        let path = Self::default_location();
        log::trace!(
            "{} from location {}",
            std::any::type_name::<Self>(),
            path.display()
        );
        if path.exists() {
            return Self::from_file(&path).map(|config| Some((config, path)));
        }

        let legacy = default_location(LEGACY_CONFIG_FILE_NAME);
        if legacy.exists() {
            log::warn!(
                "{} not found; falling back to the deprecated {}. Split it into \
                 environment_config.json (qemu_bin, qemu_plugin_dir, loader_path, \
                 turbo_tracer_plugin) and cpu_config.json (vlen, cache).",
                path.display(),
                legacy.display()
            );
            return Self::from_file(&legacy).map(|config| Some((config, legacy)));
        }

        Ok(None)
    }

    /// Both places [`ConfigFile::default_location`] looks, for messages about
    /// a file that is in neither.
    fn searched_locations() -> String {
        searched_locations(Self::FILE_NAME)
    }

    /// Write this config to `path` as pretty-printed JSON, creating its
    /// directory. Overwrites: whether an existing file may be replaced is the
    /// caller's decision (`turbo config init --force`).
    fn write_to(&self, path: &Path) -> anyhow::Result<()> {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)
                .map_err(|e| anyhow!("Failed to create {}: {e}", parent.display()))?;
        }
        let json = serde_json::to_string_pretty(self)?;
        fs::write(path, json + "\n")
            .map_err(|e| anyhow!("Failed to write {}: {e}", path.display()))?;
        log::debug!("Wrote {}", path.display());
        Ok(())
    }
}

/// The per-user config directory, `~/.config/openchip/`, whether or not it
/// exists.
pub fn user_config_dir() -> Option<PathBuf> {
    std::env::home_dir().map(|home| home.join(".config").join("openchip"))
}

/// The system-wide config directory a package installs into.
pub const SYSTEM_CONFIG_DIR: &str = "/usr/share/turbo/";

fn searched_locations(file_name: &str) -> String {
    // nosemgrep: input-validation-path-traversal-2 -- file_name is the Self::FILE_NAME associated const
    let system = Path::new(SYSTEM_CONFIG_DIR).join(file_name);
    match user_config_dir() {
        // nosemgrep: input-validation-path-traversal-2 -- file_name is the Self::FILE_NAME associated const
        Some(user) => format!("{} or {}", user.join(file_name).display(), system.display()),
        None => system.display().to_string(),
    }
}

fn default_location(file_name: &str) -> PathBuf {
    user_config_dir()
        // nosemgrep: input-validation-path-traversal-2 -- file_name is the Self::FILE_NAME associated const
        .map(|dir| dir.join(file_name))
        .filter(|p| p.exists())
        // nosemgrep: input-validation-path-traversal-2 -- file_name is the Self::FILE_NAME associated const
        .unwrap_or_else(|| Path::new(SYSTEM_CONFIG_DIR).join(file_name))
}

/// Where the external tools this machine runs are installed.
///
/// Every field is optional because every one has a working fallback in
/// [`QemuRunner`]; unknown fields are ignored, which is what lets an older
/// user-written file with `mca_bin`/`gem5_bin` keys still load.
#[derive(Debug, Serialize, Deserialize, Clone, Default)]
pub struct EnvConfig {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub qemu_bin: Option<PathBuf>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub qemu_plugin_dir: Option<PathBuf>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub loader_path: Option<PathBuf>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub turbo_tracer_plugin: Option<PathBuf>,
    /// The [`CpuConfig`] file to read when `--cpu-config` is not given: the
    /// standing answer to "which core is this machine modelling?".
    ///
    /// It lives here, in the *environment*, rather than being a second CPU
    /// config: `cpu_config.json` says what a CPU is, and this says which of
    /// those descriptions to pick up by default. A checkout that keeps one file
    /// per core (`cores/rvv0.json`, `cores/rvv1.json`) can then name the one it
    /// is currently working on once, instead of repeating `--cpu-config` on
    /// every single invocation.
    ///
    /// Relative paths are resolved against the process working directory, not
    /// against this file, so absolute paths are what belong here -- the same as
    /// every other path in this struct.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub default_cpu_config: Option<PathBuf>,
}

impl ConfigFile for EnvConfig {
    const FILE_NAME: &'static str = "environment_config.json";
}

impl From<EnvConfig> for QemuRunner {
    fn from(config: EnvConfig) -> Self {
        let mut runner = QemuRunner::default();
        if let Some(path) = config.qemu_bin.clone() {
            runner = runner.with_qemu_binary(path);
        }

        if let Some(dir) = config.qemu_plugin_dir.clone() {
            runner = runner.with_plugin_dir(dir);
        }

        if let Some(path) = config.loader_path.clone() {
            runner = runner.with_loader_path(path);
        }

        if let Some(path) = config.turbo_tracer_plugin.clone() {
            runner = runner.with_tracer_plugin_path(path);
        }

        runner
    }
}

/// The CPU being modelled: its vector register length, and the cache geometry
/// `--cache-sim` models.
///
/// Not to be confused with [`crate::source::qemu::qemu_runner::CpuConfig`],
/// which builds QEMU's `-cpu` string. This one is the *file*; that one is the
/// argument. Nothing imports both.
#[derive(Debug, Serialize, Deserialize, Clone, Default)]
pub struct CpuConfig {
    /// Vector register length in bits. `None` leaves `SYSTEM_VLEN` at its
    /// default; `--vlen` overrides either way.
    ///
    /// `u32`: the RVV maximum of 65536 is one past the top of a `u16`, so a
    /// `cpu_config.json` asking for it used to fail to parse at all.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub vlen: Option<u32>,
    /// L1/L2 geometry for `--cache-sim`. `None` means the flag has nothing to
    /// model with -- see [`CpuConfig::cache_for_sim`]. There is deliberately no
    /// default: a cache the user never described is not a cache they can draw
    /// conclusions from.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cache: Option<cache_model::Config>,
}

impl ConfigFile for CpuConfig {
    const FILE_NAME: &'static str = "cpu_config.json";
}

impl CpuConfig {
    /// Resolve `--cache-sim` into the config the pipeline should model with.
    ///
    /// Errors rather than returning `None` when the flag is set and no `cache`
    /// block is configured: a run that reported all-zero cache counts because
    /// nothing was modelled looks exactly like a run whose working set fit, and
    /// the user asked for a number.
    /// `source` is only for the error message: the file this config was actually
    /// read from, which with `--cpu-config`, `default_cpu_config` and the
    /// default location is one of three.
    pub fn cache_for_sim(
        &self,
        enabled: bool,
        source: &Path,
    ) -> anyhow::Result<Option<cache_model::Config>> {
        if !enabled {
            return Ok(None);
        }
        let config = self.cache.ok_or_else(|| {
            anyhow!(
                "--cache-sim needs a \"cache\" block (l1_bytes, l1_ways, l2_bytes, l2_ways) in \
                 {}; there is no built-in default cache to fall back on",
                source.display()
            )
        })?;
        // Validated here rather than at first access, so a bad geometry costs a
        // startup error instead of a whole run.
        config.validate().map_err(|e| {
            anyhow!(
                "the \"cache\" block in {} is invalid: {e}",
                source.display()
            )
        })?;
        Ok(Some(config))
    }
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Haps {
    ssh_key_path: String,
    user_dir: String,
    ssh_target: String,
}
