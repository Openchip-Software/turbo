//! Memory map of the guest process, built from the turbo tracer plugin's outputs.
//!
//! Everything comes from the plugin (no stderr / `-d page` / LD_DEBUG text
//! parsing): the main binary from the plugin-emitted meminfo, the shared
//! libraries from the plugin's guest-syscall mapping table, the dynamic linker
//! from the main binary's `PT_INTERP` plus the plugin-emitted entry point, and
//! the vDSO from `qemu_plugin_get_vdso_info` plus a guest-memory dump.

use anyhow::{Context, Result};
use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};
use turbo_tracer_shared::RunId;

type Elf<'a> = elf::ElfBytes<'a, elf::endian::LittleEndian>;

/// One loaded object plus the load bias its ELF vaddrs must be relocated by.
///
/// Shared libraries, the dynamic linker and the vDSO all live in one list
/// because every consumer treats them identically: load the ELF at
/// `path`, relocate by `base_address`.
#[derive(Debug, Clone)]
pub struct ObjectMapping {
    /// A guest path for a shared library (resolved to a host path downstream by
    /// `ElfMetadata::add_shared_lib`), a host path for the dynamic linker, and
    /// the `vdso_mem.dump` path for the vDSO.
    pub path: PathBuf,
    pub base_address: u64,
}

/// Complete memory layout of the guest process.
#[derive(Debug, Clone)]
pub struct MemoryMap {
    /// The run this layout came from, as `turbo_metadata.json` records it (see
    /// [`turbo_tracer_shared::RunId`]). Carried so a consumer can check the
    /// record logs it is about to decode against the run whose addresses it is
    /// about to resolve them with. `None` when the metadata was written without
    /// one.
    pub run_id: Option<RunId>,
    /// Load bias of the main binary; 0 for a static `ET_EXEC`.
    pub main_binary_base: u64,
    /// Every other loaded object: shared libraries, the dynamic linker (absent
    /// for a statically-linked binary) and the vDSO (absent if the target or
    /// QEMU build has none).
    pub objects: Vec<ObjectMapping>,
}

impl MemoryMap {
    /// Build the memory map from the turbo tracer plugin's outputs: the
    /// `TURBO_METADATA_FILE` JSON it writes, plus the vDSO guest-memory dump.
    ///
    /// # Arguments
    /// * `binary_path` - host path to the main binary
    /// * `work_dir` - directory the plugin wrote its side files to
    /// * `loader_path` - sysroot prefix used to resolve the guest interpreter path
    /// * `run_id` - expected `run_id` of the metadata file (see
    ///   [`turbo_tracer_shared::RunId`]). A mismatch is an error rather than a
    ///   map silently built from another run's addresses. Pass `None` to skip
    ///   the check, i.e. from a consumer with no run of its own.
    /// * `expected_vlen_bits` - the VLEN this run asked QEMU for, warned about
    ///   loudly if the plugin recorded a different one (a QEMU build that caps
    ///   VLEN clamps it silently). `None` from a consumer that asked for
    ///   nothing, i.e. one reading a recording it did not make.
    /// * `require_all_needed` - error out unless at least as many shared objects
    ///   have been discovered as the binary's `DT_NEEDED` count. The live-trace
    ///   path sets this while QEMU runs, so it doesn't freeze a snapshot taken
    ///   before `ld.so` finished mapping: meminfo is written at first vCPU init,
    ///   while `libs` fills in as `ld.so`'s own openat/mmap syscalls execute.
    ///   Callers building a map after the guest exited pass `false` and accept
    ///   whatever was captured.
    pub fn from_qemu_output(
        binary_path: &Path,
        work_dir: &Path,
        loader_path: &Path,
        run_id: Option<RunId>,
        expected_vlen_bits: Option<u32>,
        require_all_needed: bool,
    ) -> Result<Self> {
        const FILE: &str = turbo_tracer_shared::TURBO_METADATA_FILE;

        // A memory map is only ever built to resolve trace PCs, which means the
        // plugin ran and wrote this file on its first vCPU init (before any
        // guest instruction executed). A missing file is a real error.
        let metadata = turbo_tracer_shared::TurboMetadata::read_from_workdir(work_dir)
            .with_context(|| {
                format!(
                    "QEMU plugin metadata ({FILE}) missing or unparseable in {}; \
                     the turbo tracer plugin must run and emit it",
                    work_dir.display()
                )
            })?;

        // Without this check a leftover file from a previous run in the same
        // (uncleaned) work_dir would be accepted as a self-consistent but wrong
        // address layout. A MISSING run id in the metadata is its own (more
        // likely) failure mode: a plugin .so older than this binary, which
        // ignored `run_id=`.
        if let Some(expected) = run_id.filter(|&expected| metadata.run_id != Some(expected)) {
            let (found, hint) = match metadata.run_id {
                None => (
                    "none".to_string(),
                    "the plugin .so is older than this binary and ignored the `run_id=` \
                     argument - rebuild the plugin",
                ),
                Some(found) => (
                    format!("{found:?}"),
                    "likely a stale file from a previous run in this work_dir",
                ),
            };
            anyhow::bail!(
                "QEMU plugin metadata ({FILE}) in {} is not from this run (expected \
                 run_id {expected:?}, found {found}): {hint}",
                work_dir.display(),
            );
        }

        // Taken before `metadata` is destructured below; this is the run the
        // returned map describes, which the record logs must agree with.
        let metadata_run_id = metadata.run_id;

        // Before the VLEN check below, deliberately: no meminfo means the plugin
        // never saw a vCPU init, i.e. the guest never started, and then every
        // other field is a zero that says nothing about the run. Complaining
        // that it "ran at 0 bits" would describe a run that never happened.
        let info = match metadata.meminfo {
            Some(info) => info,
            None => {
                let mut msg = format!(
                    "QEMU plugin metadata ({FILE}) in {} has no meminfo: the tracer plugin \
                     exited before QEMU reached the first vCPU init, so the guest never ran",
                    work_dir.display()
                );
                if let Some(missing) = MissingInterpreter::detect(binary_path, loader_path) {
                    msg.push_str("\n\n");
                    msg.push_str(&missing.report());
                }
                anyhow::bail!(msg);
            }
        };

        // Checked here because this is where the plugin's own account of the run
        // is first in hand, beside the run_id check above and for the same
        // reason: a file that does not describe the run we think it does. QEMU
        // clamps an unsupported `vlen=` rather than refusing it, so the request
        // succeeding is no evidence that it was honoured.
        crate::common::warn_on_vlen_mismatch(
            expected_vlen_bits,
            metadata.vlen_bits,
            "QEMU's -cpu vlen= argument",
        );

        let main_data = std::fs::read(binary_path)
            .with_context(|| format!("reading main binary {}", binary_path.display()))?;
        let main_elf = Elf::minimal_parse(&main_data)
            .with_context(|| format!("parsing main binary ELF {}", binary_path.display()))?;

        // QEMU defines start_code as `load_bias + min(p_vaddr of executable
        // PT_LOAD)`, so the bias is recoverable by subtraction. This is correct
        // for a PIE (yields the load base) and for a static ET_EXEC (yields 0),
        // and - unlike `entry_code - e_entry` - it does not assume there is no
        // interpreter: entry_code is the *interpreter's* entry for a
        // dynamically-linked binary.
        let main_binary_base = info
            .start_code
            .saturating_sub(min_exec_vaddr(&main_elf, binary_path)?);

        // Shared libraries, discovered by the plugin from the guest's
        // openat/mmap syscalls: authoritative runtime bases, and it catches
        // dlopen'd objects too.
        let mut objects: Vec<ObjectMapping> = metadata
            .libs
            .into_iter()
            .map(|m| ObjectMapping {
                path: PathBuf::from(m.path),
                base_address: m.base_address,
            })
            .collect();
        let lib_count = objects.len();

        match dynamic_linker(&main_elf, &info, binary_path, loader_path)? {
            Some(dl) => {
                if require_all_needed {
                    let needed = needed_count(&main_elf, binary_path)?;
                    if lib_count < needed {
                        anyhow::bail!(
                            "only {lib_count}/{needed} DT_NEEDED shared object(s) discovered \
                             so far in {}",
                            work_dir.display()
                        );
                    }
                }
                objects.push(dl);
            }
            None => log::debug!("No PT_INTERP in main binary; treating as statically linked"),
        }

        if let Some(vdso) = vdso(&info, work_dir)? {
            objects.push(vdso);
        }

        log::info!(
            "MemoryMap: main base 0x{main_binary_base:x}, {lib_count} shared lib(s), \
             {} object(s) total",
            objects.len()
        );

        Ok(MemoryMap {
            run_id: metadata_run_id,
            main_binary_base,
            objects,
        })
    }
}

/// Lowest vaddr of any executable `PT_LOAD` segment: the offset that QEMU's
/// `start_code` (and the vDSO's load address) is biased from.
fn min_exec_vaddr(elf: &Elf<'_>, path: &Path) -> Result<u64> {
    elf.segments()
        .and_then(|segs| {
            segs.iter()
                .filter(|p| p.p_type == elf::abi::PT_LOAD && (p.p_flags & elf::abi::PF_X) != 0)
                .map(|p| p.p_vaddr)
                .min()
        })
        .with_context(|| format!("{} has no executable PT_LOAD segment", path.display()))
}

/// Count `DT_NEEDED` entries: the number of shared objects `ld.so` is expected
/// to map at startup. Used to tell a live-trace snapshot taken mid-load from one
/// taken once loading has finished.
fn needed_count(elf: &Elf<'_>, path: &Path) -> Result<usize> {
    let Some(dynamic) = elf
        .dynamic()
        .with_context(|| format!("reading .dynamic section of {}", path.display()))?
    else {
        return Ok(0);
    };
    Ok(dynamic
        .iter()
        .filter(|d| d.d_tag == elf::abi::DT_NEEDED)
        .count())
}

/// Recover the dynamic linker's load base.
///
/// The interpreter is mapped by QEMU's own ELF loader before the guest runs, so
/// no guest syscall observes it (unlike the .so mappings). But QEMU sets the
/// guest entry point to the interpreter's entry (`entry_code`), and the
/// interpreter is an `ET_DYN` object loaded contiguously with no offset — for
/// both PIE and non-PIE executables — so its base is `entry_code - e_entry`,
/// with `e_entry` read from the interpreter's own ELF header on disk.
///
/// No `PT_INTERP` means a statically-linked binary: `Ok(None)`.
fn dynamic_linker(
    main_elf: &Elf<'_>,
    info: &turbo_tracer_shared::MemMapInfo,
    binary_path: &Path,
    loader_path: &Path,
) -> Result<Option<ObjectMapping>> {
    let Some(interp_phdr) = main_elf
        .segments()
        .and_then(|segs| segs.iter().find(|p| p.p_type == elf::abi::PT_INTERP))
    else {
        return Ok(None);
    };

    let seg = main_elf
        .segment_data(&interp_phdr)
        .with_context(|| format!("reading PT_INTERP data from {}", binary_path.display()))?;
    let interp_guest = std::str::from_utf8(seg)
        .context("PT_INTERP is not valid UTF-8")?
        .trim_end_matches('\0');

    // Resolve the guest interpreter path against the sysroot, mirroring how
    // shared libraries are resolved downstream.
    let interp_host = match interp_guest.strip_prefix('/') {
        Some(rel) => crate::util::join_under(loader_path, rel).with_context(|| {
            format!(
                "PT_INTERP {interp_guest} escapes the sysroot {}",
                loader_path.display()
            )
        })?,
        None => PathBuf::from(interp_guest),
    };

    let interp_data = std::fs::read(&interp_host).with_context(|| {
        format!(
            "reading dynamic linker {} (guest path {interp_guest})",
            interp_host.display(),
        )
    })?;
    let e_entry = Elf::minimal_parse(&interp_data)
        .with_context(|| format!("parsing dynamic linker ELF {}", interp_host.display()))?
        .ehdr
        .e_entry;
    let base_address = info.entry_code.saturating_sub(e_entry);

    // Called once per retry attempt; only log the first time we resolve a
    // given base address to avoid spamming the console.
    static SEEN: OnceLock<Mutex<HashSet<u64>>> = OnceLock::new();
    let seen = SEEN.get_or_init(|| Mutex::new(HashSet::new()));
    if seen.lock().unwrap().insert(base_address) {
        log::trace!(
            "Dynamic linker {interp_guest} (host {}): base 0x{base_address:x}",
            interp_host.display(),
        );
    }

    Ok(Some(ObjectMapping {
        path: interp_host,
        base_address,
    }))
}

/// Build the vDSO mapping from the plugin-emitted meminfo plus the guest-memory
/// dump the plugin wrote.
///
/// The vDSO is injected by QEMU's ELF loader before the guest runs, so no guest
/// syscall observes it. Its location comes from `qemu_plugin_get_vdso_info`
/// (`vdso_addr`/`vdso_size`) and its ELF image from the raw guest memory the
/// plugin dumped — the in-memory vDSO is a complete ELF, so it loads directly.
/// `Ok(None)` if the target (or QEMU build) reports no vDSO.
fn vdso(info: &turbo_tracer_shared::MemMapInfo, work_dir: &Path) -> Result<Option<ObjectMapping>> {
    if info.vdso_addr == 0 || info.vdso_size == 0 {
        log::debug!("No vDSO reported by plugin");
        return Ok(None);
    }

    // nosemgrep: input-validation-path-traversal-2 -- work_dir joined with a const file name
    let path = work_dir.join(turbo_tracer_shared::VDSO_DUMP_FILE);
    let dump = std::fs::read(&path).with_context(|| {
        format!(
            "plugin reported a vDSO at 0x{:x} (size 0x{:x}) but its guest-memory dump {} \
             could not be read",
            info.vdso_addr,
            info.vdso_size,
            path.display()
        )
    })?;

    // The dump's PT_LOAD vaddrs are already relocated to the runtime base
    // (unlike a normal .so on disk), so this bias is 0 — but computing it the
    // same way as the main binary keeps it correct for a relative image too.
    let elf = Elf::minimal_parse(&dump)
        .with_context(|| format!("parsing vDSO dump ELF {}", path.display()))?;
    let base_address = info.vdso_addr.saturating_sub(min_exec_vaddr(&elf, &path)?);

    log::debug!(
        "vDSO: load bias 0x{base_address:x}, dump {}",
        path.display()
    );

    Ok(Some(ObjectMapping { path, base_address }))
}

/// A dynamically-linked guest binary whose `PT_INTERP` dynamic linker exists
/// nowhere QEMU will look for it.
///
/// This is the whole reason `-L <sysroot>` exists: QEMU-user maps the
/// interpreter itself, before the guest executes a single instruction, and its
/// only clue where the guest's `/lib/ld-linux-riscv64-lp64d.so.1` lives on this
/// host is the `-L` prefix. Get that wrong and QEMU dies during ELF load with
/// one terse line on stderr -- no vCPU is ever initialised, so the tracer plugin
/// records nothing but zeroes, and every downstream check then reports the
/// symptom (metadata with no meminfo, a VLEN of 0) instead of the cause.
///
/// Detected up front from the ELF alone, which is why this is cheap enough to
/// run before QEMU is launched: no run has to fail to find out.
#[derive(Debug, Clone)]
pub struct MissingInterpreter {
    /// `PT_INTERP` verbatim: the interpreter path as the *guest* names it.
    pub guest_path: String,
    /// Where the configured sysroot says that path should be on this host.
    pub sysroot_candidate: PathBuf,
    /// The sysroot that was searched (QEMU's `-L`).
    pub loader_path: PathBuf,
}

impl MissingInterpreter {
    /// `Some(_)` only when the binary needs a dynamic linker AND nothing on this
    /// host provides it, so a caller can treat it as conclusive.
    ///
    /// The search mirrors QEMU's own (`util/path.c`): an absolute guest path is
    /// tried under the `-L` prefix first, then verbatim on the host, and a
    /// relative one is only ever tried verbatim. Anything that cannot be read or
    /// parsed yields `None` -- this exists to explain a failure, so it must never
    /// invent one.
    pub fn detect(binary_path: &Path, loader_path: &Path) -> Option<Self> {
        let data = std::fs::read(binary_path).ok()?;
        let elf = Elf::minimal_parse(&data).ok()?;
        let guest_path = interpreter_path(&elf)?;

        // A relative PT_INTERP is not something QEMU resolves through `-L` at
        // all, so this check has nothing to say about it.
        let rel = guest_path.strip_prefix('/')?;

        // A `..` in PT_INTERP has no sysroot candidate to report -- resolving it
        // would name a file outside the prefix that QEMU would never load.
        let sysroot_candidate = crate::util::join_under(loader_path, rel)?;
        if sysroot_candidate.exists() || Path::new(&guest_path).exists() {
            return None;
        }
        Some(Self {
            guest_path,
            sysroot_candidate,
            loader_path: loader_path.to_path_buf(),
        })
    }

    /// The message a user can act on: what is missing, where it was looked for,
    /// and the one setting that fixes it. Bannered like the VLEN mismatch it
    /// replaces, for the same reason -- this is the cause, and it should not be
    /// mistaken for one more line of run log.
    pub fn report(&self) -> String {
        format!(
            "\n\
             ============= NO SYSROOT FOR DYNAMICALLY LINKED BINARY =============\n\
             binary needs: {guest} (its PT_INTERP)\n\
             looked under: {candidate}\n\
             \x20             (the configured sysroot {sysroot}, QEMU's -L)\n\
             and also at : {guest} (the guest path, verbatim)\n\
             Fix: point `loader_path` in environment_config.json at a sysroot\n\
             containing that file, or link the binary statically, or\n\
             install this package (Debian/Ubuntu): libc6-riscv64-cross\n\
             ===================================================================",
            guest = self.guest_path,
            candidate = self.sysroot_candidate.display(),
            sysroot = self.loader_path.display(),
        )
    }
}

/// `PT_INTERP` as the guest spells it, or `None` for a static binary.
///
/// Best-effort, unlike the read in [`dynamic_linker`]: a malformed PT_INTERP is
/// `None` here rather than an error, because the only caller is a diagnostic
/// that must not turn one failure into a different one.
fn interpreter_path(elf: &Elf<'_>) -> Option<String> {
    let phdr = elf
        .segments()?
        .iter()
        .find(|p| p.p_type == elf::abi::PT_INTERP)?;
    let seg = elf.segment_data(&phdr).ok()?;
    Some(
        std::str::from_utf8(seg)
            .ok()?
            .trim_end_matches('\0')
            .to_string(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp_dir(name: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("turbo_memory_map_{name}"));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    /// The smallest little-endian ELF64 the `elf` crate will parse, with one
    /// `PT_INTERP` segment naming `interp` (`None` for a static binary).
    ///
    /// Synthesized rather than taken from `test_bins/linker_flavours/`: those
    /// are built by `make -f Makefile.tests linker-flavours` and are not in the
    /// tree, so a unit test that read them would pass or fail depending on
    /// whether someone had run the guest-fixture build.
    fn elf_with_interp(interp: Option<&str>) -> Vec<u8> {
        const EHDR_LEN: u64 = 64;
        const PHDR_LEN: u64 = 56;

        let phnum: u16 = u16::from(interp.is_some());
        let mut out = Vec::new();
        out.extend_from_slice(b"\x7fELF");
        out.push(2); // EI_CLASS: ELFCLASS64
        out.push(1); // EI_DATA: little-endian
        out.push(1); // EI_VERSION
        out.extend_from_slice(&[0u8; 9]); // EI_OSABI onwards
        out.extend_from_slice(&2u16.to_le_bytes()); // e_type: ET_EXEC
        out.extend_from_slice(&243u16.to_le_bytes()); // e_machine: EM_RISCV
        out.extend_from_slice(&1u32.to_le_bytes()); // e_version
        out.extend_from_slice(&0x1000u64.to_le_bytes()); // e_entry
        out.extend_from_slice(&EHDR_LEN.to_le_bytes()); // e_phoff
        out.extend_from_slice(&0u64.to_le_bytes()); // e_shoff
        out.extend_from_slice(&0u32.to_le_bytes()); // e_flags
        out.extend_from_slice(&(EHDR_LEN as u16).to_le_bytes()); // e_ehsize
        out.extend_from_slice(&(PHDR_LEN as u16).to_le_bytes()); // e_phentsize
        out.extend_from_slice(&phnum.to_le_bytes()); // e_phnum
        out.extend_from_slice(&0u16.to_le_bytes()); // e_shentsize
        out.extend_from_slice(&0u16.to_le_bytes()); // e_shnum
        out.extend_from_slice(&0u16.to_le_bytes()); // e_shstrndx
        assert_eq!(out.len() as u64, EHDR_LEN);

        let Some(interp) = interp else {
            return out;
        };

        // The interpreter string goes immediately after the one phdr.
        let str_off = EHDR_LEN + PHDR_LEN;
        let mut interp_bytes = interp.as_bytes().to_vec();
        interp_bytes.push(0);

        out.extend_from_slice(&elf::abi::PT_INTERP.to_le_bytes()); // p_type
        out.extend_from_slice(&elf::abi::PF_R.to_le_bytes()); // p_flags
        out.extend_from_slice(&str_off.to_le_bytes()); // p_offset
        out.extend_from_slice(&0u64.to_le_bytes()); // p_vaddr
        out.extend_from_slice(&0u64.to_le_bytes()); // p_paddr
        out.extend_from_slice(&(interp_bytes.len() as u64).to_le_bytes()); // p_filesz
        out.extend_from_slice(&(interp_bytes.len() as u64).to_le_bytes()); // p_memsz
        out.extend_from_slice(&1u64.to_le_bytes()); // p_align
        assert_eq!(out.len() as u64, str_off);

        out.extend_from_slice(&interp_bytes);
        out
    }

    fn write_elf(dir: &Path, name: &str, interp: Option<&str>) -> PathBuf {
        let path = dir.join(name);
        std::fs::write(&path, elf_with_interp(interp)).unwrap();
        path
    }

    /// The reported failure: a dynamically-linked guest and a sysroot that does
    /// not contain its interpreter.
    #[test]
    fn missing_interpreter_is_detected_and_names_both_places_searched() {
        let dir = tmp_dir("missing_interp");
        let bin = write_elf(&dir, "dyn.riscv", Some("/lib/ld-linux-riscv64-lp64d.so.1"));
        let sysroot = dir.join("empty-sysroot");

        let missing = MissingInterpreter::detect(&bin, &sysroot)
            .expect("no ld.so anywhere, so this must be detected");
        assert_eq!(missing.guest_path, "/lib/ld-linux-riscv64-lp64d.so.1");
        assert_eq!(
            missing.sysroot_candidate,
            sysroot.join("lib/ld-linux-riscv64-lp64d.so.1")
        );

        // The report has to name the cause, not the symptoms it used to be
        // reported as -- that is the whole point of it existing.
        let report = missing.report();
        assert!(report.contains("NO SYSROOT FOR DYNAMICALLY LINKED BINARY"));
        assert!(report.contains("/lib/ld-linux-riscv64-lp64d.so.1"));
        assert!(report.contains(&sysroot.display().to_string()));
        assert!(report.contains("loader_path"));
    }

    /// The interpreter under the sysroot is exactly the case that works, so it
    /// must not be reported.
    #[test]
    fn interpreter_present_under_sysroot_is_not_reported() {
        let dir = tmp_dir("interp_in_sysroot");
        let bin = write_elf(&dir, "dyn.riscv", Some("/lib/ld.so.1"));
        let sysroot = dir.join("sysroot");
        std::fs::create_dir_all(sysroot.join("lib")).unwrap();
        std::fs::write(sysroot.join("lib/ld.so.1"), b"not really an ld.so").unwrap();

        assert!(MissingInterpreter::detect(&bin, &sysroot).is_none());
    }

    /// QEMU falls back to the guest path verbatim when the sysroot has no such
    /// file (`util/path.c`), so a host that happens to have it at that absolute
    /// path runs fine and must not be flagged.
    #[test]
    fn interpreter_present_verbatim_on_the_host_is_not_reported() {
        let dir = tmp_dir("interp_verbatim");
        // This file's own path: absolute, and certainly present on this host.
        let self_path = dir.join("dyn.riscv");
        let bin = write_elf(&dir, "dyn.riscv", Some(&self_path.display().to_string()));

        assert!(MissingInterpreter::detect(&bin, Path::new("/no-such-sysroot")).is_none());
    }

    /// A static binary needs no dynamic linker, so no sysroot can be missing for
    /// it: breaking `-L` must leave the static flavours running.
    #[test]
    fn static_binary_needs_no_sysroot() {
        let dir = tmp_dir("static");
        let bin = write_elf(&dir, "static.riscv", None);

        assert!(MissingInterpreter::detect(&bin, Path::new("/no-such-sysroot")).is_none());
    }

    /// This is a diagnostic for a failure that has already happened, so anything
    /// it cannot read is not its business to report on.
    #[test]
    fn unreadable_or_non_elf_input_reports_nothing() {
        let dir = tmp_dir("not_elf");
        let junk = dir.join("junk.riscv");
        std::fs::write(&junk, b"#!/bin/sh\necho not an elf\n").unwrap();

        assert!(MissingInterpreter::detect(&junk, Path::new("/no-such-sysroot")).is_none());
        let absent = dir.join("absent.riscv");
        assert!(MissingInterpreter::detect(&absent, Path::new("/no-such-sysroot")).is_none());
    }
}
