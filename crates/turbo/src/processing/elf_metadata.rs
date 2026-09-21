//! ELF binary metadata and symbol information management.
//!
//! This module loads and caches ELF binary data, shared library symbols, and runtime
//! memory mappings. It provides the foundation for address-to-function mapping and
//! symbol resolution during trace analysis.
//!
//! The [`ElfMetadata`] struct holds:
//! - Main ELF binary data and path
//! - Shared library information (base addresses and loaded data), including the
//!   dynamic linker and the vDSO dump
//!
//! All data is wrapped in Arc<> to enable cheap cloning across threads while sharing
//! the underlying binary contents.

use crate::RerfError;
use std::{
    path::{Path, PathBuf},
    sync::Arc,
};

/// Cached shared library information with Arc-wrapped data for cheap cloning
#[derive(Clone)]
pub struct SharedLibInfo {
    pub base_address: u64,
    pub data: Arc<[u8]>,
    pub path: PathBuf,
}

/// Hand-written so the multi-megabyte `data` is not formatted.
impl std::fmt::Debug for SharedLibInfo {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SharedLibInfo")
            .field("base_address", &self.base_address)
            .field("path", &self.path)
            .finish()
    }
}

/// Metadata required for initializing data processing components
///
/// This struct contains all the information needed to configure
/// a DataProcessor and Pipeline for trace analysis.
#[derive(Debug, Clone)]
pub struct ElfMetadata {
    /// Path to the main ELF binary being analyzed
    pub elf_path: PathBuf,
    pub elf_data: Arc<[u8]>,
    /// Shared libraries with their base addresses: (base_address, path)
    pub shared_libs: Vec<SharedLibInfo>,
    /// Optional base address for the main binary (for PIE/dynamic executables)
    pub base_addr: Option<u64>,
}

impl ElfMetadata {
    /// Create a new ProcessorMetadata with the given ELF path
    pub fn new<P: Into<PathBuf>>(elf_path: P) -> Self {
        let mut s = ElfMetadata {
            elf_path: elf_path.into(),
            elf_data: Arc::new([]),
            shared_libs: vec![],
            base_addr: None, //Some(base_addr),
        };
        s.initialize()
            .expect("default settings ElfMetadata init should succeed.");
        s
    }

    pub fn new_uninit<P: Into<PathBuf>>(elf_path: P) -> Self {
        ElfMetadata {
            elf_path: elf_path.into(),
            elf_data: Arc::new([]),
            shared_libs: vec![],
            base_addr: None,
        }
    }

    /// Returns full ELF data with runtime base addresses (for enricher/InstructionDecoder)
    pub fn iterate_executable_segments(&self) -> Vec<(u64, Arc<[u8]>)> {
        let mut segs = vec![];

        // Determine if main binary is static (ET_EXEC) or dynamic (ET_DYN)
        // For static binaries, InstructionDecoder should use base_addr = 0
        // For dynamic binaries, InstructionDecoder should use the actual base_addr
        let main_base = if let Ok(elf) =
            elf::ElfBytes::<elf::endian::LittleEndian>::minimal_parse(&self.elf_data)
        {
            let is_dynamic = elf.ehdr.e_type == elf::abi::ET_DYN;
            if is_dynamic {
                self.base_addr.unwrap_or(0)
            } else {
                // Static executable - addresses are absolute, no relocation needed
                0
            }
        } else {
            // If we can't parse, assume dynamic and use base_addr
            self.base_addr.unwrap_or(0)
        };

        segs.push((main_base, self.elf_data.clone()));

        for s in self.shared_libs.iter() {
            segs.push((s.base_address, s.data.clone()));
        }

        segs
    }

    pub fn initialize(&mut self) -> anyhow::Result<()> {
        log::info!("loading ELF binary and initializing {:?}", self);

        // Load the ELF file
        self.elf_data = {
            let elf_tmp = std::fs::read(&self.elf_path)?;
            Arc::from(elf_tmp.into_boxed_slice())
        };

        let elf = elf::ElfBytes::<elf::endian::LittleEndian>::minimal_parse(&self.elf_data)?;

        // Determine the main binary's base address and size
        let (elf_base, main_size, is_dynamic) = calculate_elf_virt_range(&elf);

        if self.base_addr.is_none() {
            log::debug!(
                "base was {:?} (static binary?), initizalize() now setting it to 0x{:x?}",
                self.base_addr,
                elf_base
            );
            self.base_addr = Some(elf_base);
        }

        log::debug!(
            "Loaded ELF binary: {} bytes, elf_base 0x{:x} size 0x{:x} dynamic: {}",
            self.elf_data.len(),
            elf_base,
            main_size,
            is_dynamic
        );

        // The vDSO needs no special handling here: `MemoryMap` puts its dump in
        // `shared_libs` (with the right base) before initialize() is called.
        Ok(())
    }

    pub fn set_main_base_addr(&mut self, base: u64) {
        self.base_addr = Some(base);
        log::info!("main base addr is now 0x{:x}", base)
    }

    pub fn add_shared_lib(&mut self, base: u64, lib: &Path) -> Result<(), RerfError> {
        log::trace!(
            "shared lib being added, base 0x{:x}, path {:?}",
            base,
            lib.display()
        );
        let so_path = lib;

        // A vDSO dump (always named `QEMU_VDSO_DUMP_FILE`) and an
        // already-sysroot-prefixed path are used as-is; anything else is a guest
        // path that needs the RISC-V sysroot prepended.
        let is_vdso_dump =
            so_path.file_name() == Some(std::ffi::OsStr::new(turbo_tracer_shared::VDSO_DUMP_FILE));

        let already_riscv_prefixed = so_path.starts_with("/usr/riscv64-linux-gnu/");

        let so_path_buf = if (is_vdso_dump || already_riscv_prefixed) && so_path.exists() {
            log::debug!(
                "Using provided path as-is for library at base 0x{:x}: {:?}",
                base,
                so_path
            );
            so_path.to_path_buf()
        } else {
            // Standard shared library path transformation for /usr/riscv64-linux-gnu/
            let base_riscv_path = PathBuf::from("/usr/riscv64-linux-gnu/");

            // UGH: .join() with an absolute path throws away the "base" part.
            let Ok(stripped_path) = so_path.strip_prefix("/") else {
                log::error!("problem processing .so path {:?}", so_path);
                return Err(RerfError::path_processing_failed(format!(
                    "the so path must be absolute at this point: {:?}",
                    so_path
                )));
            };

            // strip the "RiscV prefix away" as its sometimes present for ld.so, not for other libs :/
            let stripped_path = stripped_path
                .strip_prefix("usr/riscv64-linux-gnu")
                .unwrap_or(stripped_path);

            // now join the "non absolute" path with our prefix for riscv shared
            // objects. The path came from the traced binary, so a `..` in it
            // would resolve to a host file outside the sysroot entirely.
            let Some(joined) = crate::util::join_under(&base_riscv_path, stripped_path) else {
                log::error!("library path escapes the sysroot: {:?}", so_path);
                return Err(RerfError::path_processing_failed(format!(
                    "the so path must stay under {}: {:?}",
                    base_riscv_path.display(),
                    so_path
                )));
            };
            joined
        };

        // The sysroot transform above assumes the library lives under the RISC-V
        // sysroot (true for libc, ld.so, etc.). A binary may also link against a
        // local shared object that lives outside the sysroot (e.g. resolved via
        // an $ORIGIN rpath), in which case the loader reports its real absolute
        // path. If the sysroot-translated path doesn't exist but the original
        // absolute path does, fall back to the original.
        let so_path_buf = if !so_path_buf.exists() && so_path.exists() {
            log::debug!(
                "Sysroot path {:?} not found; using original existing path {:?}",
                so_path_buf,
                so_path
            );
            so_path.to_path_buf()
        } else {
            so_path_buf
        };

        log::debug!(
            "Loading shared library {} at base 0x{:x}",
            so_path_buf.display(),
            base
        );

        if !so_path_buf.exists() {
            log::error!("Shared library {:?} does not exist, skipping", so_path_buf);
            return Ok(());
        }

        match std::fs::read(&so_path_buf) {
            Ok(so_data) => {
                // Parse to get size
                match elf::ElfBytes::<elf::endian::LittleEndian>::minimal_parse(so_data.as_ref()) {
                    Ok(so_elf) => {
                        let (_so_base, so_size, _is_dynamic) = calculate_elf_virt_range(&so_elf);

                        self.shared_libs.push(SharedLibInfo {
                            base_address: base,
                            data: Arc::from(so_data),
                            path: so_path_buf.clone(),
                        });

                        log::debug!(
                            "Successfully cached shared library {} at 0x{:x}, size 0x{:x}",
                            so_path.display(),
                            base,
                            so_size
                        );
                    }
                    Err(e) => {
                        log::warn!(
                            "Could not parse ELF from shared library {}: {:?}",
                            so_path.display(),
                            e
                        );
                    }
                }
            }
            Err(e) => {
                log::warn!(
                    "Could not read shared library {}: {:?}",
                    so_path.display(),
                    e
                );
            }
        }
        Ok(())
    }
}

/// Calculate the virtual memory size and base address of an ELF binary from its program headers
/// Returns (base_address, size)
fn calculate_elf_virt_range(elf: &elf::ElfBytes<elf::endian::LittleEndian>) -> (u64, u64, bool) {
    let is_dynamic = elf.ehdr.e_type == elf::abi::ET_DYN;

    let phdrs = match elf.segments() {
        Some(phdrs) => phdrs,
        None => return (0, 0x100000, is_dynamic), // Default: base 0, size 1MB if no program headers
    };

    let mut min_vaddr = u64::MAX;
    let mut max_vaddr = 0u64;

    for phdr in phdrs.iter() {
        // Only consider LOAD segments
        if phdr.p_type == elf::abi::PT_LOAD {
            let start = phdr.p_vaddr;
            let end = phdr.p_vaddr + phdr.p_memsz;

            if start < min_vaddr {
                min_vaddr = start;
            }
            if end > max_vaddr {
                max_vaddr = end;
            }
        }
    }

    if min_vaddr == u64::MAX || max_vaddr == 0 {
        // No LOAD segments found, use a default
        log::warn!("couldn't find ELF abi PT_LOAD, using 1MB as default, this will likely cause errors later..");
        return (0, 0x100000, is_dynamic); // base 0, size 1MB default
    }

    (min_vaddr, max_vaddr - min_vaddr, is_dynamic)
}
