//! ELF function name resolution module.
//!
//! This module provides efficient function name lookup from ELF binaries,
//! prioritizing DWARF debug information with fallbacks to symbol tables.

use std::borrow::Cow;
use std::collections::{BTreeMap, HashMap};
use std::ops::Range;

use object::{Object, ObjectSection, ObjectSymbol, SymbolKind};
use thiserror::Error;

/// Errors that can occur during ELF resolution.
#[derive(Debug, Error)]
pub enum ResolverError {
    #[error("failed to parse ELF data: {0}")]
    ParseError(#[from] object::Error),

    #[error("failed to read ELF file: {0}")]
    IoError(#[from] std::io::Error),

    #[error("failed to parse DWARF data: {0}")]
    DwarfError(#[from] gimli::Error),

    #[error("no executable sections found")]
    NoExecutableSections,

    #[error("address 0x{0:x} is outside all known ranges")]
    AddressOutOfRange(u64),

    #[error("function not found: {0}")]
    FunctionNotFound(String),

    #[error("index out of bounds: {0}")]
    IndexOutOfBounds(usize),
}

/// Result type for resolver operations.
pub type Result<T> = std::result::Result<T, ResolverError>;

/// Represents a resolved function with its metadata.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResolvedFunction<'a> {
    /// The function name (demangled if possible).
    pub name: Cow<'a, str>,
    /// The start address of the function.
    pub start_address: u64,
    /// The end address of the function (exclusive), if known.
    pub end_address: Option<u64>,
    /// The source of this resolution.
    pub source: ResolutionSource,
    /// Source file information, if available.
    pub source_file: Option<Cow<'a, str>>,
    /// Line number, if available.
    pub line_number: Option<u32>,
}

/// Indicates how the function name was resolved.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResolutionSource {
    /// Resolved from DWARF debug information.
    Dwarf,
    /// Resolved from the .symtab section.
    SymbolTable,
    /// Resolved from the .dynsym section.
    DynamicSymbols,
}

/// A function entry used for efficient lookup.
#[derive(Debug, Clone)]
struct FunctionEntry {
    name: String,
    demangled_name: Option<String>,
    start: u64,
    end: Option<u64>,
    source: ResolutionSource,
    file: Option<String>,
    line: Option<u32>,
}

/// Address → source line index built from the DWARF line-number program.
///
/// Best-effort: binaries without `.debug_line` yield an empty index and
/// [`ElfResolver::resolve_line`] simply returns `None`.
#[derive(Clone, Default)]
struct LineIndex {
    /// Interned source-file paths.
    files: Vec<String>,
    /// Row address → optional (file id, source line). The map is keyed by the
    /// address each line-program row *begins* at; a `range(..=pc).next_back()`
    /// lookup finds the row governing an arbitrary instruction address.
    ///
    /// A `None` value is a **sequence terminator**: the DWARF line program marks
    /// the end of each contiguous run of addresses with an `end_sequence` row
    /// whose address is one past the last covered instruction. Recording those
    /// as `None` bounds every sequence, so a PC that falls in a gap (no line
    /// coverage — e.g. a Fortran routine whose CU has no `.debug_line`) resolves
    /// to `None` instead of bleeding into the nearest lower row of an unrelated
    /// compilation unit.
    map: BTreeMap<u64, Option<(u32, u32)>>,
}

impl LineIndex {
    /// Intern a file path, returning its id (linear scan; file tables are small).
    fn intern(&mut self, path: String) -> u32 {
        if let Some(pos) = self.files.iter().position(|p| *p == path) {
            return pos as u32;
        }
        let id = self.files.len() as u32;
        self.files.push(path);
        id
    }

    /// Look up the (file, line) governing `address`, or `None` if the address
    /// is not covered by any line-program sequence.
    fn lookup(&self, address: u64) -> Option<(&str, u32)> {
        self.map
            .range(..=address)
            .next_back()
            .and_then(|(_, v)| v.as_ref())
            .map(|(fid, line)| (self.files[*fid as usize].as_str(), *line))
    }
}

/// Main resolver struct that holds parsed ELF data.
///
/// This struct pre-parses and indexes all function information for
/// efficient repeated lookups. Create once and reuse for multiple queries.
#[derive(Clone)]
pub struct ElfResolver {
    /// Functions indexed by start address for range lookups.
    /// Using BTreeMap for efficient range queries.
    functions: BTreeMap<u64, FunctionEntry>,
    /// Executable address ranges for validation.
    executable_ranges: Vec<Range<u64>>,
    /// Mapping from function name to unique index.
    name_to_index: HashMap<String, usize>,
    /// Mapping from index to function name for reverse lookup.
    index_to_name: Vec<String>,
    /// Address → source line index (from the DWARF line-number program).
    line_index: LineIndex,
}

/// Why a symbol was not turned into a [`FunctionEntry`].
///
/// Rejections are the common case (most of a symbol table is not functions),
/// so they are tallied in [`RejectTally`] and reported once per object instead
/// of logged per symbol.
#[derive(Clone, Copy)]
enum RejectReason {
    NotTextSection,
    WrongKind,
    EmptyName,
    MappingSymbol,
    ZeroAddress,
    NoSection,
}

/// Per-object counts of accepted symbols and rejections by reason.
#[derive(Default)]
struct RejectTally {
    accepted: u64,
    not_text_section: u64,
    wrong_kind: u64,
    empty_name: u64,
    mapping_symbol: u64,
    zero_address: u64,
    no_section: u64,
}

impl RejectTally {
    fn reject(&mut self, reason: RejectReason) {
        let slot = match reason {
            RejectReason::NotTextSection => &mut self.not_text_section,
            RejectReason::WrongKind => &mut self.wrong_kind,
            RejectReason::EmptyName => &mut self.empty_name,
            RejectReason::MappingSymbol => &mut self.mapping_symbol,
            RejectReason::ZeroAddress => &mut self.zero_address,
            RejectReason::NoSection => &mut self.no_section,
        };
        *slot += 1;
    }

    fn rejected(&self) -> u64 {
        self.not_text_section
            + self.wrong_kind
            + self.empty_name
            + self.mapping_symbol
            + self.zero_address
            + self.no_section
    }
}

impl std::fmt::Display for RejectTally {
    /// Totals, then a breakdown of only the reasons that actually fired.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "accepted {}, rejected {}",
            self.accepted,
            self.rejected()
        )?;
        let reasons = [
            ("not-text-section", self.not_text_section),
            ("wrong-kind", self.wrong_kind),
            ("empty-name", self.empty_name),
            ("mapping-symbol", self.mapping_symbol),
            ("zero-address", self.zero_address),
            ("no-section", self.no_section),
        ];
        let mut first = true;
        for (label, count) in reasons.iter().filter(|(_, c)| *c > 0) {
            f.write_str(if first { " (" } else { ", " })?;
            first = false;
            write!(f, "{label}: {count}")?;
        }
        if !first {
            f.write_str(")")?;
        }
        Ok(())
    }
}

impl ElfResolver {
    /// An empty resolver that resolves nothing. Used as a fallback for a
    /// module whose ELF data failed to parse, so callers can keep treating
    /// every module uniformly instead of special-casing a missing entry.
    pub fn empty() -> Self {
        Self {
            functions: BTreeMap::new(),
            executable_ranges: Vec::new(),
            name_to_index: HashMap::new(),
            index_to_name: Vec::new(),
            line_index: LineIndex::default(),
        }
    }

    /// Create a new resolver from raw ELF binary data, addresses left as the
    /// file's link-time virtual addresses (no runtime relocation applied).
    ///
    /// # Errors
    /// Returns an error if the ELF data cannot be parsed.
    pub fn new(data: &[u8]) -> Result<Self> {
        Self::new_at(data, 0)
    }

    /// Create a new resolver from raw ELF binary data, relocating every
    /// resolved address by `base_addr` — the same runtime base used to place
    /// this module's executable segments (see
    /// [`InstructionDecoder::new`](crate::processing::instruction_decoder::InstructionDecoder::new)).
    ///
    /// Position-independent binaries and shared libraries link with
    /// near-zero virtual addresses; without this shift, symbol/DWARF
    /// addresses parsed from the file never match runtime PCs. VDSO dumps
    /// are the exception: their addresses are already the runtime ones, so
    /// the same low-address heuristic used by `InstructionDecoder` is
    /// applied here to avoid double-relocating them.
    ///
    /// This parses DWARF info and symbol tables upfront for efficient
    /// subsequent lookups.
    ///
    /// # Arguments
    /// * `data` - The raw bytes of the ELF file.
    /// * `base_addr` - The runtime load base for this module (0 for static
    ///   binaries or already-relocated data such as VDSO dumps).
    ///
    /// # Errors
    /// Returns an error if the ELF data cannot be parsed.
    pub fn new_at(data: &[u8], base_addr: u64) -> Result<Self> {
        let mut resolver = Self::empty();
        let bias = Self::load_bias(data, base_addr);
        resolver.load_object(data, bias)?;
        resolver.rebuild_name_index();
        Ok(resolver)
    }

    /// Build a resolver covering the main executable **and** every shared
    /// library described by `metadata`, applying each object's runtime load
    /// bias so that PCs captured at runtime resolve to the correct symbol and
    /// source line.
    ///
    /// This is what makes hot code that lives in a shared library annotatable.
    /// A dynamically-linked binary (e.g. `test-quantize-perf`, whose RVV
    /// quantization kernels live in `libggml-cpu.so`) executes most of its hot
    /// instructions inside `.so` address ranges; [`new`](Self::new) alone only
    /// ever parses the main ELF, so those PCs resolve to no symbol/line and
    /// land in the annotate report's synthetic `(unknown)` bucket with no
    /// source interleaving. Loading each shared library at its loader-reported
    /// base fixes that.
    pub fn from_metadata(metadata: &crate::processing::ElfMetadata) -> Result<Self> {
        let mut resolver = Self::empty();

        // Main executable. For a non-PIE (ET_EXEC) binary the link-time vaddrs
        // already equal the runtime PCs (bias 0); for a PIE (ET_DYN) they are
        // file-relative and need the runtime load base added.
        let main_data = std::fs::read(&metadata.elf_path)?;
        let main_bias = Self::load_bias(&main_data, metadata.base_addr.unwrap_or(0));
        resolver.load_object(&main_data, main_bias)?;
        log::debug!(
            "resolver: loaded main ELF {:?} at bias 0x{:x} ({} functions so far)",
            metadata.elf_path,
            main_bias,
            resolver.functions.len()
        );

        // Shared libraries, each biased by the base the loader mapped it at, so
        // their symbols/DWARF line info are keyed at the same runtime addresses
        // the trace records.
        for lib in &metadata.shared_libs {
            let bias = Self::load_bias(&lib.data, lib.base_address);
            let before = resolver.functions.len();
            match resolver.load_object(&lib.data, bias) {
                Ok(()) => log::debug!(
                    "resolver: loaded shared lib {} at bias 0x{:x} (+{} functions)",
                    lib.path.display(),
                    bias,
                    resolver.functions.len() - before
                ),
                Err(e) => log::warn!(
                    "resolver: failed to parse shared lib {} for symbol/line resolution: {e}; \
                     its PCs will fall into the annotate (unknown) bucket",
                    lib.path.display()
                ),
            }
        }

        resolver.rebuild_name_index();
        log::info!(
            "resolver: {} functions across main ELF + {} shared libs",
            resolver.functions.len(),
            metadata.shared_libs.len()
        );
        Ok(resolver)
    }

    /// Determine the runtime load bias to add to an ELF's link-time addresses.
    ///
    /// A position-independent object (`ET_DYN` — a PIE executable or a shared
    /// library) has file-relative vaddrs starting near 0, so the bias is the
    /// runtime base the loader mapped it at. A non-PIE `ET_EXEC` binary already
    /// carries its runtime vaddrs, so the bias is 0. A pre-relocated object
    /// (e.g. a captured VDSO whose ELF vaddrs already sit at the runtime base)
    /// is detected by its first vaddr already matching `base`, and also gets 0
    /// — mirrors the same heuristic in `InstructionDecoder`.
    fn load_bias(data: &[u8], base: u64) -> u64 {
        let Ok(object) = object::File::parse(data) else {
            return 0;
        };
        if !matches!(object.kind(), object::ObjectKind::Dynamic) {
            return 0;
        }
        // Lowest non-zero section vaddr; if it already sits at/near the runtime
        // base the object is pre-relocated and needs no further offset.
        let first_vaddr = object
            .sections()
            .map(|s| s.address())
            .filter(|&a| a != 0)
            .min()
            .unwrap_or(0);
        if first_vaddr >= base && first_vaddr < base.saturating_add(0x10000) {
            0
        } else {
            base
        }
    }

    /// Parse one ELF's symbols and DWARF into this resolver, adding `bias` to
    /// every address so a shared library's (or PIE's) link-time addresses are
    /// keyed at the runtime PCs the trace records. Merges into the existing
    /// maps; call [`rebuild_name_index`](Self::rebuild_name_index) once after
    /// all objects are loaded.
    fn load_object(&mut self, data: &[u8], bias: u64) -> Result<()> {
        let object = object::File::parse(data)?;

        // Collect executable ranges (biased to runtime addresses).
        for section in object.sections() {
            if let Ok(name) = section.name() {
                if name == ".text" || section.kind() == object::SectionKind::Text {
                    let start = section.address() + bias;
                    let end = start + section.size();
                    self.executable_ranges.push(start..end);
                }
            }
        }

        // Ignore errors - in this case, we will just use the symbols table
        let _ = Self::parse_dwarf(&object, &mut self.functions, &mut self.line_index, bias);
        Self::parse_symbols(&object, &mut self.functions, bias)?;
        Ok(())
    }

    /// Rebuild the name↔index mappings from the current function set. Call once
    /// after all objects have been merged via [`load_object`](Self::load_object).
    fn rebuild_name_index(&mut self) {
        self.name_to_index.clear();
        self.index_to_name.clear();
        for entry in self.functions.values() {
            let display_name = entry.demangled_name.as_ref().unwrap_or(&entry.name);
            if !self.name_to_index.contains_key(display_name) {
                let index = self.index_to_name.len();
                self.name_to_index.insert(display_name.clone(), index);
                self.index_to_name.push(display_name.clone());
            }
        }
    }

    /// Parse DWARF debug information for function entries.
    fn parse_dwarf(
        object: &object::File<'_>,
        functions: &mut BTreeMap<u64, FunctionEntry>,
        line_index: &mut LineIndex,
        bias: u64,
    ) -> Result<()> {
        // Load DWARF sections
        let load_section =
            |id: gimli::SectionId| -> std::result::Result<Cow<'_, [u8]>, gimli::Error> {
                Ok(object
                    .section_by_name(id.name())
                    .and_then(|s| s.uncompressed_data().ok())
                    .unwrap_or(Cow::Borrowed(&[])))
            };

        let dwarf_sections = gimli::DwarfSections::load(load_section)?;

        // Borrow the sections for parsing
        let dwarf =
            dwarf_sections.borrow(|section| gimli::EndianSlice::new(section, gimli::LittleEndian));

        // Iterate through compilation units
        let mut units = dwarf.units();
        while let Some(header) = units.next()? {
            let unit = dwarf.unit(header)?;
            Self::parse_unit(&dwarf, &unit, functions, bias)?;
            // Line-program failures are non-fatal: fall back to asm-only rows.
            let _ = Self::parse_line_program(&dwarf, &unit, line_index, bias);
        }

        Ok(())
    }

    /// Walk a compilation unit's line-number program, indexing each row's
    /// address → (file, line) into `line_index`.
    fn parse_line_program<R: gimli::Reader>(
        dwarf: &gimli::Dwarf<R>,
        unit: &gimli::Unit<R>,
        line_index: &mut LineIndex,
        bias: u64,
    ) -> Result<()> {
        let program = match unit.line_program.clone() {
            Some(p) => p,
            None => return Ok(()),
        };

        let mut rows = program.rows();
        while let Some((header, row)) = rows.next_row()? {
            let address = row.address() + bias;

            // The end_sequence row marks one-past-the-end of a contiguous run;
            // record it as a terminator so a PC in the following gap resolves to
            // None rather than the nearest lower row of an unrelated CU. Use
            // or_insert so it never clobbers a real row that begins at the same
            // address (the terminator of one sequence can coincide with the
            // start of the next).
            if row.end_sequence() {
                line_index.map.entry(address).or_insert(None);
                continue;
            }

            let line = row.line().map(|l| l.get() as u32).unwrap_or(0);
            let file_id = match row.file(header) {
                Some(file_entry) => {
                    match Self::resolve_file_path(dwarf, unit, header, file_entry)? {
                        Some(path) => line_index.intern(path),
                        None => continue,
                    }
                }
                None => continue,
            };
            // A real row always wins over a coincident terminator.
            line_index.map.insert(address, Some((file_id, line)));
        }

        Ok(())
    }

    /// Parse a single DWARF compilation unit.
    fn parse_unit<R: gimli::Reader>(
        dwarf: &gimli::Dwarf<R>,
        unit: &gimli::Unit<R>,
        functions: &mut BTreeMap<u64, FunctionEntry>,
        bias: u64,
    ) -> Result<()> {
        let mut entries = unit.entries();

        while let Some(entry) = entries.next_dfs()? {
            if entry.tag() == gimli::DW_TAG_subprogram {
                if let Some(func) = Self::parse_function_entry(dwarf, unit, entry, bias)? {
                    // Only insert if we don't have a better entry (DWARF takes priority)
                    functions.entry(func.start).or_insert(func);
                }
            }
        }

        Ok(())
    }

    /// Parse a single DWARF function entry.
    fn parse_function_entry<R: gimli::Reader>(
        dwarf: &gimli::Dwarf<R>,
        unit: &gimli::Unit<R>,
        entry: &gimli::DebuggingInformationEntry<R>,
        bias: u64,
    ) -> Result<Option<FunctionEntry>> {
        let mut name: Option<R> = None;
        let mut low_pc: Option<u64> = None;
        let mut high_pc: Option<u64> = None;
        let mut high_pc_offset: Option<u64> = None;
        let mut file: Option<String> = None;
        let mut line: Option<u32> = None;

        let attrs = entry.attrs();
        for attr in attrs {
            match attr.name() {
                gimli::DW_AT_name => {
                    if let gimli::AttributeValue::DebugStrRef(offset) = attr.value() {
                        name = Some(dwarf.debug_str.get_str(offset)?);
                    } else if let gimli::AttributeValue::String(s) = attr.value() {
                        name = Some(s);
                    }
                }
                gimli::DW_AT_linkage_name | gimli::DW_AT_MIPS_linkage_name => {
                    // Prefer linkage name over regular name
                    if let gimli::AttributeValue::DebugStrRef(offset) = attr.value() {
                        name = Some(dwarf.debug_str.get_str(offset)?);
                    } else if let gimli::AttributeValue::String(s) = attr.value() {
                        name = Some(s);
                    }
                }
                gimli::DW_AT_low_pc => {
                    if let gimli::AttributeValue::Addr(addr) = attr.value() {
                        low_pc = Some(addr);
                    }
                }
                gimli::DW_AT_high_pc => match attr.value() {
                    gimli::AttributeValue::Addr(addr) => high_pc = Some(addr),
                    gimli::AttributeValue::Udata(offset) => high_pc_offset = Some(offset),
                    _ => {}
                },
                gimli::DW_AT_decl_file => {
                    if let gimli::AttributeValue::FileIndex(index) = attr.value() {
                        if let Some(line_program) = &unit.line_program {
                            let header = line_program.header();
                            if let Some(file_entry) = header.file(index) {
                                if let Some(path) =
                                    Self::resolve_file_path(dwarf, unit, header, file_entry)?
                                {
                                    file = Some(path);
                                }
                            }
                        }
                    }
                }
                gimli::DW_AT_decl_line => {
                    if let gimli::AttributeValue::Udata(l) = attr.value() {
                        line = Some(l as u32);
                    }
                }
                _ => {}
            }
        }

        // Need at least a name and address
        let (name_slice, start) = match (name, low_pc) {
            (Some(n), Some(s)) => (n, s),
            _ => return Ok(None),
        };

        let name_string = name_slice.to_string_lossy()?.into_owned();

        // Apply the runtime load bias so the entry is keyed at the same address
        // the trace records. `high_pc` as an absolute address is also a
        // link-time vaddr (bias it); as a `Udata` offset it is relative to the
        // (already-biased) start.
        let start = start + bias;

        // Calculate end address
        let end = match (high_pc, high_pc_offset) {
            (Some(hp), _) => Some(hp + bias),
            (_, Some(offset)) => Some(start + offset),
            _ => None,
        };

        let demangled_name = Self::demangle(&name_string);

        Ok(Some(FunctionEntry {
            name: name_string,
            demangled_name,
            start,
            end,
            source: ResolutionSource::Dwarf,
            file,
            line,
        }))
    }

    /// Resolve a file path from DWARF file entry.
    fn resolve_file_path<R: gimli::Reader>(
        dwarf: &gimli::Dwarf<R>,
        unit: &gimli::Unit<R>,
        header: &gimli::LineProgramHeader<R>,
        file_entry: &gimli::FileEntry<R>,
    ) -> Result<Option<String>> {
        let mut path = String::new();

        // Get directory
        if let Some(dir) = file_entry.directory(header) {
            let dir_str = dwarf.attr_string(unit, dir)?;
            path.push_str(&dir_str.to_string_lossy()?);
            if !path.ends_with('/') {
                path.push('/');
            }
        }

        // Get filename
        let file_str = dwarf.attr_string(unit, file_entry.path_name())?;
        path.push_str(&file_str.to_string_lossy()?);

        if path.is_empty() {
            Ok(None)
        } else {
            Ok(Some(path))
        }
    }

    /// Parse symbol tables for function entries.
    fn parse_symbols(
        object: &object::File<'_>,
        functions: &mut BTreeMap<u64, FunctionEntry>,
        bias: u64,
    ) -> Result<()> {
        // Parse regular symbol table
        let mut tally = RejectTally::default();

        // First, build a map of section indices to their kinds
        let mut section_kinds = HashMap::new();
        for section in object.sections() {
            section_kinds.insert(section.index(), section.kind());
        }

        for symbol in object.symbols() {
            // Check if symbol is in a text section for Unknown kind filtering
            let in_text_section = if symbol.kind() == SymbolKind::Unknown {
                if let object::SymbolSection::Section(idx) = symbol.section() {
                    matches!(section_kinds.get(&idx), Some(&object::SectionKind::Text))
                } else {
                    false
                }
            } else {
                true // Non-Unknown symbols pass through for other checks
            };

            if !in_text_section {
                tally.reject(RejectReason::NotTextSection);
                continue;
            }
            match Self::symbol_to_entry(&symbol, ResolutionSource::SymbolTable, bias) {
                Ok(entry) => {
                    // Don't overwrite DWARF entries
                    functions.entry(entry.start).or_insert(entry);
                    tally.accepted += 1;
                }
                Err(reason) => tally.reject(reason),
            }
        }

        // Parse dynamic symbol table
        let mut dyn_tally = RejectTally::default();
        for symbol in object.dynamic_symbols() {
            let in_text_section = if symbol.kind() == SymbolKind::Unknown {
                if let object::SymbolSection::Section(idx) = symbol.section() {
                    matches!(section_kinds.get(&idx), Some(&object::SectionKind::Text))
                } else {
                    false
                }
            } else {
                true
            };

            if !in_text_section {
                dyn_tally.reject(RejectReason::NotTextSection);
                continue;
            }
            match Self::symbol_to_entry(&symbol, ResolutionSource::DynamicSymbols, bias) {
                Ok(entry) => {
                    functions.entry(entry.start).or_insert(entry);
                    dyn_tally.accepted += 1;
                }
                Err(reason) => dyn_tally.reject(reason),
            }
        }

        // One line per object. `functions.len()` is cumulative across all
        // objects merged so far; `from_metadata` logs the final per-object
        // breakdown and grand total.
        log::debug!(
            "Symbols: {tally}; dynamic: {dyn_tally}; {} functions total so far",
            functions.len()
        );

        Ok(())
    }

    /// Convert an object symbol to a function entry.
    fn symbol_to_entry(
        symbol: &object::Symbol<'_, '_>,
        source: ResolutionSource,
        bias: u64,
    ) -> std::result::Result<FunctionEntry, RejectReason> {
        let name = symbol.name().map_err(|_| RejectReason::EmptyName)?;
        let kind = symbol.kind();
        let start = symbol.address();

        // log::trace!(
        //     "Checking symbol: name='{}', kind={:?}, address=0x{:x}, section={:?}",
        //     name,
        //     kind,
        //     start,
        //     symbol.section()
        // );

        // Accept function symbols and text symbols
        // Also accept Unknown symbols if they're in a text/code section
        let accept_kind = kind == SymbolKind::Text || kind == SymbolKind::Unknown;
        if !accept_kind {
            return Err(RejectReason::WrongKind);
        }

        if name.is_empty() {
            return Err(RejectReason::EmptyName);
        }

        // Filter out RISC-V mapping symbols (e.g., $x, $xrv64i2p1_m2p0_...)
        // These are ISA attribute markers, not actual functions
        if name.starts_with("$x") {
            return Err(RejectReason::MappingSymbol);
        }

        if start == 0 {
            return Err(RejectReason::ZeroAddress);
        }

        // Filter out symbols without a defined section (like absolute or undefined symbols)
        if matches!(
            symbol.section(),
            object::SymbolSection::Undefined
                | object::SymbolSection::None
                | object::SymbolSection::Absolute
        ) {
            return Err(RejectReason::NoSection);
        }

        // log::debug!("  Accepted symbol: '{}' at 0x{:x}", name, start);

        // Apply the runtime load bias now that the symbol has passed all
        // link-time-address filters (the `start == 0` reject above must see the
        // unbiased vaddr).
        let start = start + bias;
        let size = symbol.size();
        let end = if size > 0 { Some(start + size) } else { None };

        let name_string = name.to_string();
        let demangled_name = Self::demangle(&name_string);

        Ok(FunctionEntry {
            name: name_string,
            demangled_name,
            start,
            end,
            source,
            file: None,
            line: None,
        })
    }

    /// Resolve a function name for the given address.
    ///
    /// # Arguments
    /// * `address` - The instruction address to look up.
    ///
    /// # Returns
    /// The resolved function information, or an error if the address
    /// cannot be resolved.
    pub fn resolve(&self, address: u64) -> Result<ResolvedFunction<'_>> {
        // Find the function containing this address using range query
        // Get the largest key <= address
        if let Some((&start, entry)) = self.functions.range(..=address).next_back() {
            // Check if address is within function bounds
            let in_bounds = match entry.end {
                Some(end) => address < end,
                None => {
                    // No end address - check if there's a next function
                    // and if our address is before it
                    self.functions
                        .range((start + 1)..)
                        .next()
                        .map(|(&next_start, _)| address < next_start)
                        .unwrap_or(true)
                }
            };

            if in_bounds {
                return Ok(self.entry_to_resolved(entry));
            }
        }

        // Check if address is in executable range at all
        let in_executable = self.executable_ranges.iter().any(|r| r.contains(&address));

        if !in_executable {
            return Err(ResolverError::AddressOutOfRange(address));
        }

        // Address is executable but we don't know the function
        Err(ResolverError::AddressOutOfRange(address))
    }

    /// Try to resolve a function name, returning None instead of error.
    ///
    /// This is useful when you want to handle missing functions gracefully.
    #[inline]
    pub fn try_resolve(&self, address: u64) -> Option<ResolvedFunction<'_>> {
        self.resolve(address).ok()
    }

    /// Convert internal entry to public resolved function.
    fn entry_to_resolved<'a>(&self, entry: &'a FunctionEntry) -> ResolvedFunction<'a> {
        let name = entry.demangled_name.as_deref().unwrap_or(&entry.name);
        ResolvedFunction {
            name: Cow::Borrowed(name),
            start_address: entry.start,
            end_address: entry.end,
            source: entry.source,
            source_file: entry.file.as_deref().map(Cow::Borrowed),
            line_number: entry.line,
        }
    }

    fn demangle(name: &str) -> Option<String> {
        // arbitrary name-len limit for "shortening"
        const MAX_LEN: usize = 64;

        // Try C++ demangling
        if let Ok(demangled) = cpp_demangle::Symbol::new(name) {
            let mut result = demangled.demangle().ok()?;
            result.truncate(MAX_LEN);
            if result != name {
                return Some(result);
            }
        }

        // Try Rust demangling
        if let Ok(demangled) = rustc_demangle::try_demangle(name) {
            let mut result = demangled.to_string();
            result.truncate(MAX_LEN);
            if result != name {
                return Some(result);
            }
        }

        None
    }

    /// Resolve an instruction address to its `(source_file, line)` using the
    /// DWARF line-number program. Returns `None` when no `.debug_line` covers
    /// the address (asm-only degradation).
    pub fn resolve_line(&self, address: u64) -> Option<(&str, u32)> {
        self.line_index.lookup(address)
    }

    /// Whether `address` falls inside one of this module's executable
    /// sections. Used by [`MultiResolver`] to pick the resolver that can
    /// symbolize a given runtime PC.
    pub fn covers(&self, address: u64) -> bool {
        self.executable_ranges.iter().any(|r| r.contains(&address))
    }

    /// Get the number of functions indexed.
    #[inline]
    pub fn function_count(&self) -> usize {
        self.functions.len()
    }

    /// Check if the resolver has any functions indexed.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.functions.is_empty()
    }

    /// Iterate over all known functions.
    pub fn functions(&self) -> impl Iterator<Item = ResolvedFunction<'_>> + '_ {
        self.functions.values().map(|e| self.entry_to_resolved(e))
    }

    /// Convert a function name to its unique integer index.
    ///
    /// # Arguments
    /// * `name` - The function name to look up.
    ///
    /// # Returns
    /// The unique index for this function name, or an error if the
    /// function name is not found in the functions map.
    pub fn to_index(&self, name: &str) -> Result<usize> {
        self.name_to_index
            .get(name)
            .copied()
            .ok_or_else(|| ResolverError::FunctionNotFound(name.to_string()))
    }

    /// Convert an integer index back to its function name.
    ///
    /// # Arguments
    /// * `index` - The index to look up.
    ///
    /// # Returns
    /// The function name for this index, or an error if the index
    /// is out of bounds.
    pub fn from_index(&self, index: usize) -> Result<&str> {
        self.index_to_name
            .get(index)
            .map(|s| s.as_str())
            .ok_or(ResolverError::IndexOutOfBounds(index))
    }

    /// Get the total number of unique function names indexed.
    #[inline]
    pub fn name_count(&self) -> usize {
        self.index_to_name.len()
    }

    /// Resolve an address to both function name and index in a single call.
    /// This is more efficient than calling try_resolve() then to_index().
    pub fn resolve_to_name_and_index(&self, address: u64) -> Option<(&str, usize)> {
        // Find the function containing this address
        if let Some((&start, entry)) = self.functions.range(..=address).next_back() {
            let in_bounds = match entry.end {
                Some(end) => address < end,
                None => self
                    .functions
                    .range((start + 1)..)
                    .next()
                    .map(|(&next_start, _)| address < next_start)
                    .unwrap_or(true),
            };

            if in_bounds {
                let name = entry.demangled_name.as_deref().unwrap_or(&entry.name);
                let index = self.name_to_index.get(name).copied()?;
                return Some((name, index));
            }
        }
        None
    }

    /// Resolve an address to function name, index, and the function's address range.
    /// This allows caching the range to avoid repeated lookups for sequential PCs.
    #[hotpath::measure]
    pub fn resolve_to_name_index_and_range(
        &self,
        address: u64,
    ) -> Option<(&str, usize, Range<u64>)> {
        // Find the function containing this address
        if let Some((&start, entry)) = self.functions.range(..=address).next_back() {
            let end = match entry.end {
                Some(end) => end,
                None => self
                    .functions
                    .range((start + 1)..)
                    .next()
                    .map(|(&next_start, _)| next_start)
                    .unwrap_or(u64::MAX),
            };

            if address < end {
                let name = entry.demangled_name.as_deref().unwrap_or(&entry.name);
                let index = self.name_to_index.get(name).copied()?;
                return Some((name, index, start..end));
            }
        }
        None
    }
}

/// A collection of per-module [`ElfResolver`]s (main binary, each shared
/// library, VDSO) presented as a single symbolizer with one flat, collision-
/// free index space.
///
/// Each `ElfResolver` numbers its own functions starting at 0, so using a
/// bare per-module index as a cross-module key (e.g. to accumulate
/// per-function stats, or to symbolize the annotation report) would silently
/// merge unrelated functions that happen to land on the same local index in
/// different modules. `MultiResolver` assigns every module a fixed offset
/// (the running total of functions in the modules before it) and adds it to
/// each local index, so a "global index" uniquely names one function in one
/// module across the whole trace.
pub struct MultiResolver {
    resolvers: Vec<ElfResolver>,
    /// Global-index offset for each resolver, aligned 1:1 with `resolvers`.
    offsets: Vec<usize>,
}

impl MultiResolver {
    /// Build from one resolver per executable module, in any order — the
    /// offsets are derived from `resolvers`' own order, so callers that also
    /// keep a parallel per-module list (e.g. `Enricher::multi_decoder.dec`)
    /// must build both from the same iteration to keep module indices
    /// aligned.
    pub fn new(resolvers: Vec<ElfResolver>) -> Self {
        // nosemgrep: dos-unbounded-memory-allocation -- one entry per loaded module
        let mut offsets = Vec::with_capacity(resolvers.len());
        let mut acc = 0usize;
        for r in &resolvers {
            offsets.push(acc);
            acc += r.name_count();
        }
        Self { resolvers, offsets }
    }

    /// Number of modules (main binary + shared libs + VDSO).
    pub fn module_count(&self) -> usize {
        self.resolvers.len()
    }

    /// Find the module index whose executable range contains `address`.
    pub fn module_for(&self, address: u64) -> Option<usize> {
        self.resolvers.iter().position(|r| r.covers(address))
    }

    /// Resolve a PC using a module index already known to the caller (e.g.
    /// from [`MultiDecoder::find_decoder_index_for`](crate::processing::instruction_decoder::MultiDecoder::find_decoder_index_for),
    /// which the hot per-instruction path already computes to pick the
    /// decoder). Avoids re-scanning every module's address ranges when the
    /// caller already knows which one it wants.
    pub fn resolve_in_module(
        &self,
        module_index: usize,
        address: u64,
    ) -> Option<(&str, usize, Range<u64>)> {
        let (name, local_index, range) =
            self.resolvers[module_index].resolve_to_name_index_and_range(address)?;
        Some((name, self.offsets[module_index] + local_index, range))
    }

    /// Resolve a PC to `(name, global_index, address_range)`, scanning
    /// modules by address range to find the right one. Prefer
    /// [`resolve_in_module`](Self::resolve_in_module) when the module index
    /// is already known.
    pub fn resolve_to_name_index_and_range(
        &self,
        address: u64,
    ) -> Option<(&str, usize, Range<u64>)> {
        let module_index = self.module_for(address)?;
        self.resolve_in_module(module_index, address)
    }

    /// Resolve a PC to `(source_file, line)` via the DWARF line table of
    /// whichever module contains it.
    pub fn resolve_line(&self, address: u64) -> Option<(&str, u32)> {
        let module_index = self.module_for(address)?;
        self.resolvers[module_index].resolve_line(address)
    }

    /// Convert a global function-name index back to its name.
    ///
    /// Finds the last module whose offset is `<= global_index` (offsets are
    /// non-decreasing — a module with zero functions repeats the previous
    /// offset — so `partition_point`, not `binary_search`, is what correctly
    /// picks the rightmost match on duplicates). `offsets[0]` is always 0, so
    /// this never underflows for a `global_index` produced by this same
    /// `MultiResolver`.
    pub fn from_index(&self, global_index: usize) -> Result<&str> {
        let module_index = self.offsets.partition_point(|&o| o <= global_index) - 1;
        let local_index = global_index - self.offsets[module_index];
        self.resolvers[module_index].from_index(local_index)
    }
}

#[cfg(test)]
mod line_index_tests {
    use super::LineIndex;

    /// Build a two-sequence index by hand and check that end_sequence
    /// terminators bound each run — a PC in the gap between sequences must
    /// resolve to `None`, not bleed into the previous sequence's file.
    #[test]
    fn terminator_bounds_sequences() {
        let mut li = LineIndex::default();
        let a = li.intern("a.c".to_string());
        let b = li.intern("b.c".to_string());

        // Sequence 1: [0x100, 0x140) in a.c; end_sequence terminator at 0x140.
        li.map.insert(0x100, Some((a, 10)));
        li.map.insert(0x120, Some((a, 11)));
        li.map.entry(0x140).or_insert(None); // terminator
                                             // Sequence 2: [0x200, 0x210) in b.c; terminator at 0x210.
        li.map.insert(0x200, Some((b, 42)));
        li.map.entry(0x210).or_insert(None);

        // Inside sequence 1.
        assert_eq!(li.lookup(0x108), Some(("a.c", 10)));
        assert_eq!(li.lookup(0x120), Some(("a.c", 11)));
        // In the gap after sequence 1 — must NOT map to a.c.
        assert_eq!(li.lookup(0x140), None);
        assert_eq!(li.lookup(0x1a0), None);
        // Inside sequence 2.
        assert_eq!(li.lookup(0x200), Some(("b.c", 42)));
        // After sequence 2.
        assert_eq!(li.lookup(0x210), None);
        // Before anything.
        assert_eq!(li.lookup(0x10), None);
    }

    /// A real row must win over a coincident terminator regardless of insert
    /// order (terminator of one sequence == start of the next).
    #[test]
    fn real_row_wins_over_coincident_terminator() {
        let mut li = LineIndex::default();
        let a = li.intern("a.c".to_string());
        // Terminator inserted first, then a real row at the same address.
        li.map.entry(0x300).or_insert(None);
        li.map.insert(0x300, Some((a, 7)));
        assert_eq!(li.lookup(0x300), Some(("a.c", 7)));
    }
}
