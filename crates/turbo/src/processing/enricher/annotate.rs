//! Static source & assembly annotation report generator.
//!
//! Joins the per-PC profile ([`PcProfile`]) with the DWARF line table and
//! function names from a [`MultiResolver`] (one [`ElfResolver`] per module —
//! main binary, shared libraries, VDSO) to produce `perf annotate`-style
//! HTML: per function, the disassembly grouped under its source lines, each
//! asm row annotated with execution count, instruction class, and (for
//! vector ops) VL/SEW/LMUL and element counts.
//!
//! [`ElfResolver`]: crate::processing::function_lookup::ElfResolver
//!
//! The output is a set of self-contained static HTML files written next to the
//! existing `.asm` files, so it works with the `--no-server` offline path.

use std::collections::{HashMap, HashSet};
use std::fmt::Write as _;
use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use flate2::{write::GzEncoder, Compression};

use crate::processing::function_lookup::MultiResolver;
use crate::processing::instruction_decoder::MultiDecoder;

use super::pc_profile::{class, class_tags, PcProfile, PcRecord};

/// One executed instruction together with the source line DWARF attributes it
/// to. Rows are held in **ascending PC order** so the disassembly reads
/// linearly (like `objdump -S` / `perf annotate`); the source line is reprinted
/// at render time whenever it changes between consecutive rows.
struct AnnotatedRow {
    rec: PcRecord,
    line_no: u32,
    text: Option<String>,
}

/// One contiguous run of addresses attributed to a single source file. A file
/// that is inlined at several points in a function gets one of these per run,
/// so the page as a whole stays in address order.
struct AnnotatedFile {
    path: String,
    /// Rows sorted by PC (ascending). Instructions from different source lines
    /// interleave here exactly as the compiler laid them out in memory.
    rows: Vec<AnnotatedRow>,
}

/// A function's full annotation view.
struct AnnotatedFunction {
    name: String,
    safe_name: String,
    total_instructions: u64,
    files: Vec<AnnotatedFile>,
    /// Records with no source-line mapping (asm-only).
    orphan: Vec<PcRecord>,
    orphan_exec: u64,
}

/// Name of the side file listing every DWARF-requested source file that could
/// not be located locally, written into the run's output directory.
const MISSING_SRC_FILE: &str = "turbo_no_debug_symbols.json";

/// Collector for source files the DWARF line table references but which don't
/// exist locally (no local checkout, or the wrong/no `--src-dir`).
///
/// Every such file used to warn once at the point of the failed read, which on a
/// C++ binary means dozens of console lines for toolchain headers nobody is
/// going to annotate anyway. Instead the paths are accumulated here with the
/// functions that wanted them, dumped in full to [`MISSING_SRC_FILE`], and
/// summarized in a single `warn!`.
#[derive(Default)]
struct MissingSources {
    /// DWARF path -> names of the functions whose PCs resolved to it, in
    /// first-seen order (`by_path` keeps the paths ordered for a stable file).
    by_path: std::collections::BTreeMap<String, Vec<String>>,
}

impl MissingSources {
    fn record(&mut self, path: &str, func: &str) {
        let funcs = self.by_path.entry(path.to_string()).or_default();
        if !funcs.iter().any(|f| f == func) {
            funcs.push(func.to_string());
        }
    }

    fn is_empty(&self) -> bool {
        self.by_path.is_empty()
    }

    /// Write the full detail to `<out_dir>/turbo_no_debug_symbols.json`
    fn report(&self, out_dir: Option<&Path>, src_dirs: &[PathBuf]) {
        if self.is_empty() {
            return;
        }

        // Rank by how many functions each missing file would have annotated, so
        // the three shown are the ones most worth pointing --src-dir at.
        let mut ranked: Vec<(&String, &Vec<String>)> = self.by_path.iter().collect();
        ranked.sort_by(|a, b| b.1.len().cmp(&a.1.len()).then(a.0.cmp(b.0)));
        let head: Vec<String> = ranked
            .iter()
            .take(3)
            .map(|(path, funcs)| format!("{path} ({} function(s))", funcs.len()))
            .collect();

        let detail = serde_json::json!({
            "missing_source_file_count": self.by_path.len(),
            "src_dirs_searched": src_dirs.iter().map(|d| d.display().to_string())
                .collect::<Vec<_>>(),
            "note": "DWARF line-table paths (build-host absolute) that do not exist \
                     locally; the affected functions render as asm-only. Pass --src-dir \
                     <root> pointing at a checkout containing these files to interleave \
                     source.",
            "missing_sources": ranked.iter().map(|(path, funcs)| serde_json::json!({
                "path": path,
                "functions": funcs,
            })).collect::<Vec<_>>(),
        });

        let written =
            out_dir.map(|d| d.join(MISSING_SRC_FILE)).filter(
                |p| match serde_json::to_string_pretty(&detail)
                    .map_err(|e| e.to_string())
                    .and_then(|s| std::fs::write(p, s).map_err(|e| e.to_string()))
                {
                    Ok(()) => true,
                    Err(e) => {
                        log::warn!("annotate: could not write {p:?}: {e}");
                        false
                    }
                },
            );

        let where_ = match written {
            Some(p) => format!("full list in {}", p.display()),
            None => "run with RUST_LOG=debug for the full list".to_string(),
        };
        log::debug!(
            "DWARF symbols request {} source file(s), but not found. Searched literal paths \
             + {} --src-dir root(s)); functions are shown asm-only. Top 3: {} — {where_}",
            self.by_path.len(),
            src_dirs.len(),
            head.join(", "),
        );
    }
}

/// Render the annotation report as in-memory `(filename, html)` pairs, without
/// touching the filesystem. The filenames mirror the on-disk layout
/// (`annotate_index.html`, `annotate_<func>.html`) so the same names work for
/// both disk serving and the HTTP router. Best-effort: missing DWARF line info
/// or source files degrade to asm-only rows; nothing is ever dropped. Returns
/// an empty vec when there are no per-PC records.
pub fn render_all(
    resolver: &MultiResolver,
    profile: &PcProfile,
    src_dirs: &[PathBuf],
    decoder: &MultiDecoder,
    out_dir: Option<&Path>,
) -> Vec<(String, String)> {
    // Group records by resolving function index. Records without a function
    // fall into a synthetic "(unknown)" bucket keyed by usize::MAX.
    let mut by_func: HashMap<usize, Vec<PcRecord>> = HashMap::new();
    for rec in profile.records().values() {
        let key = rec.func_index.unwrap_or(usize::MAX);
        by_func.entry(key).or_default().push(rec.clone());
    }

    // A function's ELF symbol may cover PCs the trace never executed (dead
    // branches, error paths, etc.). Fill those in with exec_count == 0 rows so
    // the annotate view shows the whole function body, not just the subset
    // that happened to run — the symbol is exported with a known size, so its
    // full address range is knowable independent of any trace.
    for (&fkey, records) in by_func.iter_mut() {
        if fkey == usize::MAX {
            continue;
        }
        if let Some(first_pc) = records.first().map(|r| r.pc) {
            if let Some((_, _, range)) = resolver.resolve_to_name_index_and_range(first_pc) {
                fill_unexecuted_pcs(records, range, fkey, decoder);
            }
        }
    }

    if by_func.is_empty() {
        log::info!("annotate: no per-PC records to report");
        return Vec::new();
    }

    // Source-file text cache (path -> lines), read lazily once per file.
    let mut src_cache: HashMap<String, Option<Vec<String>>> = HashMap::new();
    // Track which --src-dir roots have resolved at least one file so we log a
    // detection message once per directory (not once per source file).
    let mut logged_roots: HashSet<usize> = HashSet::new();
    // Unresolvable DWARF source paths, summarized once after the loop below.
    let mut missing = MissingSources::default();

    let mut funcs: Vec<AnnotatedFunction> = Vec::new();
    for (fkey, records) in by_func {
        let name = if fkey == usize::MAX {
            "(unknown)".to_string()
        } else {
            resolver
                .from_index(fkey)
                .map(|s| s.to_string())
                .unwrap_or_else(|_| format!("func_{fkey}"))
        };
        funcs.push(build_function(
            name,
            records,
            resolver,
            &mut src_cache,
            src_dirs,
            &mut logged_roots,
            &mut missing,
        ));
    }
    missing.report(out_dir, src_dirs);

    // Aggregate summary of source-annotation coverage, so a --src-dir user can
    // see at a glance how many functions got interleaved source vs. degraded to
    // asm-only (per-function detail is at debug level to avoid a console flood).
    if !src_dirs.is_empty() {
        let named: Vec<&AnnotatedFunction> =
            funcs.iter().filter(|f| f.name != "(unknown)").collect();
        let no_line = named
            .iter()
            .filter(|f| f.files.is_empty() && !f.orphan.is_empty())
            .count();
        let with_src = named.iter().filter(|f| !f.files.is_empty()).count();
        if no_line > 0 {
            log::info!(
                "annotate: {with_src}/{} function(s) annotated with source; {no_line} resolved to \
                 a symbol but lack DWARF line info (built without -g / stripped, e.g. libc) — \
                 shown asm-only (run with RUST_LOG=debug for the per-function list)",
                named.len()
            );
        }
    }

    // Hottest functions first.
    funcs.sort_by_key(|f| std::cmp::Reverse(f.total_instructions));

    // Rank loop sections by "badness" (hot × poorly-vectorized) for the hotspots
    // page. Reads the profile directly (independent of the per-function grouping
    // above), so it needs no data threaded through `AnnotatedFunction`.
    let sections = super::hotspots::rank(resolver, profile);

    // nosemgrep: dos-unbounded-memory-allocation -- len of a resident Vec
    let mut pages: Vec<(String, String)> = Vec::with_capacity(funcs.len() + 2);
    pages.push((
        "annotate_index.html".to_string(),
        render_index(&funcs, !sections.is_empty()),
    ));
    if !sections.is_empty() {
        pages.push((
            "annotate_hotspots.html".to_string(),
            super::hotspots::render_page(&sections),
        ));
    }
    pages.extend(render_function_pages(&funcs));
    // Shared assets, downloaded once and cached by the browser instead of the
    // ~10 KB of CSS+JS previously inlined into every page.
    pages.push(("annotate.css".to_string(), emitted_stylesheet()));
    pages.push(("annotate.js".to_string(), annotate_js()));
    pages
}

/// The shared client-side script, served once as `annotate.js` and referenced
/// by every function page instead of inlining ~3 KB per file. Concatenates the
/// hide-source toggle, the branch-arrow overlay, and the loop-region highlighter
/// (each self-guards, so the single bundle is harmless on pages that lack the
/// elements it targets).
pub(super) fn annotate_js() -> String {
    format!(
        "{}\n{}\n{}\n{}",
        hide_src_js(),
        arrow_js(),
        region_js(),
        hover_js()
    )
}

/// Gzip-compress bytes. Used both for the on-disk `<name>.gz` artifacts and for
/// the in-memory pages the server sends with `Content-Encoding: gzip`.
pub(crate) fn gzip(bytes: &[u8]) -> Vec<u8> {
    let mut enc = GzEncoder::new(Vec::new(), Compression::default());
    let _ = enc.write_all(bytes);
    enc.finish().unwrap_or_default()
}

/// Render one page per function, spreading the functions across a small pool of
/// scoped threads.
///
/// `render_function` is a pure function of its `AnnotatedFunction` (all the
/// shared state -- source cache, resolver -- was consumed by `build_function`
/// above), so the only reason this was serial is that it was written that way.
/// It is part of the same end-of-run serial tail as the gzip stage (see
/// [`gzip_pages`]): the two together were the 0.13s `enrich-tail` on a
/// `flash_full 256 256` run. Order is preserved (hottest function first).
fn render_function_pages(funcs: &[AnnotatedFunction]) -> Vec<(String, String)> {
    let threads = std::thread::available_parallelism()
        .map(|n| n.get().min(8))
        .unwrap_or(1);
    let render =
        |f: &AnnotatedFunction| (format!("annotate_{}.html", f.safe_name), render_function(f));
    if threads <= 1 || funcs.len() < 8 {
        return funcs.iter().map(render).collect();
    }

    let chunk = funcs.len().div_ceil(threads);
    // nosemgrep: dos-unbounded-memory-allocation -- len of a resident Vec
    let mut out: Vec<(String, String)> = Vec::with_capacity(funcs.len());
    std::thread::scope(|scope| {
        let handles: Vec<_> = funcs
            .chunks(chunk)
            .map(|part| scope.spawn(move || part.iter().map(render).collect::<Vec<_>>()))
            .collect();
        for h in handles {
            match h.join() {
                Ok(part) => out.extend(part),
                // Best-effort, as everywhere in the annotate path: lose the
                // chunk's pages rather than the whole run.
                Err(_) => log::warn!("annotate: a page-render thread panicked"),
            }
        }
    });
    out
}

/// Gzip every rendered page, spreading the pages across a small pool of
/// scoped threads.
///
/// Compression is embarrassingly parallel (each page is independent) and it is
/// the last thing that happens before the process exits, so it sits directly on
/// the serial tail of the wall clock: 267 pages cost ~40ms single-threaded on a
/// `flash_full 256 256` run. Order is preserved (chunks are concatenated in
/// order) so the on-disk artifacts and the HTTP store see a stable sequence.
///
/// Scoped `std::thread` rather than a work-stealing pool: no new dependency,
/// and equal-sized chunks are good enough for a one-shot, uniform workload.
pub(crate) fn gzip_pages(pages: Vec<(String, String)>) -> Vec<(String, Vec<u8>)> {
    let threads = std::thread::available_parallelism()
        .map(|n| n.get().min(8))
        .unwrap_or(1);
    if threads <= 1 || pages.len() < 8 {
        return pages
            .into_iter()
            .map(|(name, html)| (name, gzip(html.as_bytes())))
            .collect();
    }

    let chunk = pages.len().div_ceil(threads);
    // nosemgrep: dos-unbounded-memory-allocation -- len of a resident Vec
    let mut out: Vec<(String, Vec<u8>)> = Vec::with_capacity(pages.len());
    std::thread::scope(|scope| {
        let handles: Vec<_> = pages
            .chunks(chunk)
            .map(|part| {
                scope.spawn(move || {
                    part.iter()
                        .map(|(name, html)| (name.clone(), gzip(html.as_bytes())))
                        .collect::<Vec<_>>()
                })
            })
            .collect();
        for h in handles {
            match h.join() {
                Ok(part) => out.extend(part),
                // A panicking compression thread must not take the run down:
                // the report is best-effort (see `write_pages_gz`).
                Err(_) => log::warn!("annotate: a page-compression thread panicked"),
            }
        }
    });
    out
}

/// Write each already-gzipped page to `out_dir` as `<name>.gz` (≈15× smaller
/// than the raw HTML). The logical name is preserved (e.g.
/// `annotate_<workload>.html.gz`), so the server and load path can key on it by
/// stripping the `.gz` suffix. Best-effort: a failed write is logged, never
/// fatal.
///
/// Takes the compressed bytes rather than the HTML because the caller needs
/// the very same bytes for the in-memory HTTP store: compressing here *and*
/// there deflated all 267 pages twice, and `miniz_oxide` was 3.5% of
/// whole-process samples on a `flash_full 256 256` run -- over half of the
/// serial 0.13s annotate tail.
pub fn write_pages_gz(pages: &[(String, Vec<u8>)], out_dir: &Path) {
    if pages.is_empty() {
        return;
    }
    if let Err(e) = std::fs::create_dir_all(out_dir) {
        log::warn!("annotate: could not create output dir {out_dir:?}: {e}");
        return;
    }

    let mut written = 0usize;
    for (name, gz) in pages {
        // nosemgrep: input-validation-path-traversal-2 -- name is annotate_<safe_name>.html; sanitize() drops every separator
        let path = out_dir.join(format!("{name}.gz"));
        match std::fs::write(&path, gz) {
            Ok(_) => written += 1,
            Err(e) => log::warn!("annotate: failed to write {path:?}: {e}"),
        }
    }
    log::debug!(
        "annotate: wrote {} gzipped page(s) (incl. index + assets) to {:?}",
        written,
        out_dir
    );
}

/// Read back one page written by [`write_pages_gz`] and decompress it.
///
/// `out_dir` is the `html/` directory and `name` the logical page name
/// (e.g. `annotate_index.html`) -- the `.gz` suffix is added here, so callers
/// never have to know that the on-disk artifact is compressed while the
/// in-memory HTTP store serves it plain.
pub fn read_page_gz(out_dir: &Path, name: &str) -> std::io::Result<String> {
    use std::io::Read as _;
    // nosemgrep: input-validation-path-traversal-2 -- name is annotate_<safe_name>.html; sanitize() drops every separator
    let path = out_dir.join(format!("{name}.gz"));
    let bytes = std::fs::read(&path)?;
    let mut html = String::new();
    // nosemgrep: dos-zip-bomb-vulnerability -- decompresses a .gz this same process wrote
    flate2::read::GzDecoder::new(&bytes[..]).read_to_string(&mut html)?;
    Ok(html)
}

/// Statically disassemble every instruction in `range` that isn't already
/// present in `records`, appending a synthetic `exec_count == 0` [`PcRecord`]
/// for each. Disassembly is purely a function of the ELF `.text` bytes (see
/// [`crate::processing::instruction_decoder::InstructionDecoder`]), so it
/// works for PCs the trace never visited. Best-effort: stops early if a PC in
/// the range can't be decoded (e.g. the range runs past the last known
/// symbol into padding/data).
fn fill_unexecuted_pcs(
    records: &mut Vec<PcRecord>,
    range: std::ops::Range<u64>,
    func_index: usize,
    decoder: &MultiDecoder,
) {
    let seen: HashSet<u64> = records.iter().map(|r| r.pc).collect();
    let Some(dec) = decoder.find_decoder_for(range.start) else {
        return;
    };

    let mut pc = range.start;
    while pc < range.end {
        let Ok((ins, ins_bytes)) = dec.get_instruction_at(pc) else {
            break;
        };
        let len = ins_bytes.len() as u64;
        if !seen.contains(&pc) {
            records.push(PcRecord::unexecuted(pc, &ins, ins_bytes, Some(func_index)));
        }
        pc += len.max(1);
    }
}

fn build_function(
    name: String,
    records: Vec<PcRecord>,
    resolver: &MultiResolver,
    src_cache: &mut HashMap<String, Option<Vec<String>>>,
    src_dirs: &[PathBuf],
    logged_roots: &mut HashSet<usize>,
    missing: &mut MissingSources,
) -> AnnotatedFunction {
    let safe_name = sanitize(&name);
    let total_instructions: u64 = records.iter().map(|r| r.exec_count).sum();

    // Walk the records in address order and cut a new section every time the
    // resolved source file changes. Bucketing by file instead would reorder the
    // asm: inlined header code is interleaved with the caller's own code in the
    // address space, so a per-file bucket tears those runs apart and the
    // disassembly no longer reads in sequence. One section per *contiguous run*
    // keeps addresses monotonic across the whole page; a file that is inlined
    // repeatedly simply gets a heading per run. Records with no line info at all
    // are collected apart and don't break the surrounding run.
    let mut records = records;
    records.sort_by_key(|r| r.pc);

    let mut afiles: Vec<AnnotatedFile> = Vec::new();
    let mut orphan: Vec<PcRecord> = Vec::new();

    for rec in records {
        let Some((path, line_no)) = resolver.resolve_line(rec.pc) else {
            orphan.push(rec);
            continue;
        };
        let path = path.to_string();

        let lines = src_cache
            .entry(path.clone())
            .or_insert_with(|| read_source(&path, src_dirs, logged_roots));
        if lines.is_none() {
            // Unresolved source: recorded (not logged) here, so the console gets
            // one aggregate warning instead of one line per DWARF file.
            missing.record(&path, &name);
        }
        let text = lines
            .as_ref()
            .and_then(|l| line_no.checked_sub(1).and_then(|i| l.get(i as usize)))
            .map(|s| s.to_string());

        if afiles.last().map(|f| f.path.as_str()) != Some(path.as_str()) {
            afiles.push(AnnotatedFile {
                path,
                rows: Vec::new(),
            });
        }
        afiles
            .last_mut()
            .expect("just pushed")
            .rows
            .push(AnnotatedRow { rec, line_no, text });
    }

    orphan.sort_by_key(|r| r.pc);
    let orphan_exec = orphan.iter().map(|r| r.exec_count).sum();

    // A --src-dir was given but none of this function's PCs resolved to a
    // source line (no DWARF line info for it), so its asm can never be
    // interleaved with source no matter how correct --src-dir is. Distinguish
    // the two root causes so the message is actionable:
    //   * "(unknown)" bucket  -> the PCs resolved to no symbol at all, i.e. they
    //     execute in an object whose symbols/DWARF weren't loaded (a stripped
    //     shared library, or one the loader mapped that we couldn't parse).
    //   * a named function     -> the symbol resolved but the object carries no
    //     .debug_line for it (compiled without -g / debug info stripped).
    if !src_dirs.is_empty() && afiles.is_empty() && !orphan.is_empty() {
        if name == "(unknown)" {
            // The important, actionable case: hot instructions attributed to no
            // symbol at all — kept at warn.
            log::warn!(
                "annotate: {} instruction(s) executed in code with no resolved symbol \
                 (not covered by the main ELF or any loaded shared library's symbols/DWARF); \
                 these cannot be interleaved with source. If the hot code lives in a shared \
                 library, ensure it is built with debug info (-g) and not stripped",
                orphan.iter().map(|r| r.exec_count).sum::<u64>()
            );
        } else {
            // A named symbol lacking .debug_line is routine, this is extremely spammy :/
            // log::debug!(
            //     "annotate: function {name:?} resolved to a symbol but has no DWARF line info \
            //      (built without -g or debug info stripped); asm shown without source"
            // );
        }
    }

    AnnotatedFunction {
        name,
        safe_name,
        total_instructions,
        files: afiles,
        orphan,
        orphan_exec,
    }
}

/// Read a source file's lines, resolving the DWARF-encoded `path` to a readable
/// local file first (see [`locate_source`]). Returns `None` when no candidate
/// exists (asm-only degradation).
fn read_source(
    path: &str,
    src_dirs: &[PathBuf],
    logged_roots: &mut HashSet<usize>,
) -> Option<Vec<String>> {
    let Some((local, root_idx)) = locate_source(path, src_dirs) else {
        // Name the DWARF-requested source file explicitly so it's obvious what
        // to point --src-dir at (the DWARF path is the build-host absolute path,
        // which usually doesn't exist locally). Debug level only: a build
        // typically references dozens of unavailable toolchain/libc sources, so
        // the console gets a single aggregate warning plus the full list in
        // `turbo_no_debug_symbols.json` (see [`MissingSources`]) instead.
        log::debug!(
            "annotate: source file requested by DWARF not found: {path:?} \
             (searched literal path + {} --src-dir root(s): {:?}); asm shown without source",
            src_dirs.len(),
            src_dirs
        );
        return None;
    };
    // Log once per --src-dir root that first resolves a file, so a correct
    // --src-dir is confirmed without a per-file flood.
    if let Some(idx) = root_idx {
        if logged_roots.insert(idx) {
            log::info!(
                "annotate: resolved source under --src-dir {}",
                src_dirs[idx].display()
            );
        }
    }
    match std::fs::read_to_string(&local) {
        Ok(s) => Some(s.lines().map(|l| l.to_string()).collect()),
        Err(_) => None,
    }
}

/// Resolve a DWARF-encoded source path to a readable local file.
///
/// DWARF records the absolute path the file had **on the build host** (e.g.
/// `/home/user/build/src/lib/foo.c`), which
/// usually does not exist on the machine running the analysis. Resolution order:
///
/// 1. The literal path — correct when analysing on the build host itself.
/// 2. For each user-supplied `--src-dir` root (gdb `dir`-style), the longest
///    path **suffix** of `path` that exists under that root. Suffixes are tried
///    longest-first (most specific wins), stripping one leading component at a
///    time, so a root of `/home/user/local-checkout` matches a DWARF path of
///    `.../local-checkout/src/lib/foo.c` via the
///    suffix `src/lib/foo.c`.
///
/// Returns the first existing candidate together with the index of the
/// `--src-dir` root that matched (`None` when the literal build-host path
/// existed and no root was needed), or `None` if nothing matches.
fn locate_source(path: &str, src_dirs: &[PathBuf]) -> Option<(PathBuf, Option<usize>)> {
    let literal = Path::new(path);
    if literal.is_file() {
        return Some((literal.to_path_buf(), None));
    }

    // Path components, skipping the root/prefix so suffixes are relative.
    let comps: Vec<&str> = path
        .split('/')
        .filter(|c| !c.is_empty() && *c != ".")
        .collect();

    for (idx, root) in src_dirs.iter().enumerate() {
        // Longest suffix first: start at the full component list, drop one
        // leading component per iteration down to just the basename.
        for start in 0..comps.len() {
            let mut candidate = root.clone();
            for c in &comps[start..] {
                candidate.push(c);
            }
            if candidate.is_file() {
                return Some((candidate, Some(idx)));
            }
        }
    }

    None
}

/// Sanitize a function name into a filesystem-safe stem used as
/// `annotate_<stem>.html`. Now that annotation covers every module (not just
/// the main binary — see [`MultiResolver`]), libc's `index` (the classic
/// `strchr` alias) and any hypothetical `hotspots` symbol are real
/// possibilities, and would otherwise collide with the report's own
/// `annotate_index.html`/`annotate_hotspots.html` — silently overwriting
/// them since [`write_pages_gz`] writes by filename with no dedup. Guard the
/// two reserved stems explicitly; this is a pure function of `name`, so both
/// call sites ([`build_function`] and `hotspots::sections_for_function`)
/// derive the same disambiguated stem independently and stay consistent
/// without needing to share state.
pub(super) fn sanitize(name: &str) -> String {
    let raw: String = name
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '_' || c == '.' {
                c
            } else {
                '_'
            }
        })
        .collect();
    match raw.as_str() {
        "index" | "hotspots" => format!("{raw}_fn"),
        _ => raw,
    }
}

pub(super) fn esc(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
}

/// SEW encoding → mnemonic (e8/e16/e32/e64).
fn sew_label(sew: Option<u8>) -> &'static str {
    match sew {
        Some(0) => "e8",
        Some(1) => "e16",
        Some(2) => "e32",
        Some(3) => "e64",
        _ => "e?",
    }
}

/// LMUL → the `.vbar-*` CSS class suffix (m1/m2/m4/m8/mf2/mf4/mf8), used to
/// colour the VL bars with the same palette as the dashboard LMUL histograms.
fn lmul_suffix(lmul: Option<crate::common::Lmul>) -> &'static str {
    use crate::common::Lmul::*;
    match lmul {
        Some(M1) => "m1",
        Some(M2) => "m2",
        Some(M4) => "m4",
        Some(M8) => "m8",
        Some(Mf2) => "mf2",
        Some(Mf4) => "mf4",
        Some(Mf8) => "mf8",
        None => "none",
    }
}

/// Render the VL histogram for the `vec` column of the `vsetvl*` that set the
/// length: the SEW/LMUL header plus one bar per distinct VL the instruction set
/// (VL value, a count-proportional bar coloured by LMUL, and how many times it
/// set that VL). Returns an empty string for records without a histogram (i.e.
/// everything except a length-setting `vsetvl*`).
///
/// The element total is derived from the histogram itself (Σ VL·count) rather
/// than `vector_elems`, which lives on the consuming vector ops.
fn vector_hist_html(rec: &PcRecord) -> String {
    if rec.vl_hist.is_empty() {
        return String::new();
    }
    let mut bins: Vec<(u32, u64)> = rec.vl_hist.iter().map(|(&vl, &c)| (vl, c)).collect();
    bins.sort_by_key(|&(vl, _)| vl);
    let max_c = bins.iter().map(|&(_, c)| c).max().unwrap_or(1);
    let elems: u64 = bins.iter().map(|&(vl, c)| vl as u64 * c).sum();
    let suffix = lmul_suffix(rec.lmul);
    let lmul_txt = rec.lmul.map(|l| format!("{l:?}")).unwrap_or_default();

    let mut html = String::from("<div class=\"vhist\">");
    // nosemgrep: error-handling-ignored-critical-result -- fmt::Write on a String is infallible
    let _ = write!(
        html,
        "<div class=\"vhist-head\">{} {} &middot; {} elems</div>",
        sew_label(rec.sew),
        esc(&lmul_txt),
        elems
    );
    for (vl, c) in bins {
        let w = ((c as f64 / max_c as f64) * 60.0).round() as u64;
        // nosemgrep: error-handling-ignored-critical-result -- fmt::Write on a String is infallible
        let _ = write!(
            html,
            "<div class=\"vbin\"><span class=\"vbin-label\">{}</span>\
             <span class=\"vbar vbar-{}\" style=\"width:{}px\"></span>\
             <span class=\"vbin-cnt\">{}</span></div>",
            vl, suffix, w, c
        );
    }
    html.push_str("</div>");
    html
}

/// Per-vector-op `data-*` attributes consumed by the top-right hover panel
/// (see the `annotate_hover.js` asset). Because the commit that moved the VL histogram onto the
/// `vsetvl*` left every consuming vector op blank in the `vec` column, these
/// attributes carry the SEW/LMUL/VL/exec data forward so the panel can show it
/// at the point of use — even when the controlling `vsetvl*` is off-screen or
/// ambiguous under branches. Returns an empty string for scalar instructions.
fn vec_data_attrs(r: &PcRecord) -> String {
    if r.class & class::VECTOR == 0 {
        return String::new();
    }
    let lmul_txt = r.lmul.map(|l| format!("{l:?}")).unwrap_or_default();
    format!(
        " data-vsew=\"{}\" data-vlmul=\"{}\" data-vlmin=\"{}\" \
         data-vlmax=\"{}\" data-velems=\"{}\" data-vexec=\"{}\"{}",
        sew_label(r.sew),
        esc(&lmul_txt),
        r.vl_min,
        r.vl_max,
        r.vector_elems,
        r.exec_count,
        cache_data_attrs(r),
    )
}

/// Per-instruction cache counts, for the hover panel.
///
/// Empty unless a cache was modelled *and* this instruction touched memory, so
/// a run without `--cache-sim` produces byte-identical pages to before, and a
/// scalar instruction in a modelled run does not get a row of zeroes that would
/// read as "never missed".
fn cache_data_attrs(r: &PcRecord) -> String {
    if r.cache.is_zero() {
        return String::new();
    }
    format!(
        " data-l1h=\"{}\" data-l1m=\"{}\" data-l2h=\"{}\" data-l2m=\"{}\"",
        r.cache.l1_hits, r.cache.l1_misses, r.cache.l2_hits, r.cache.l2_misses,
    )
}

/// Applies the theme the user picked in the main dashboard. The annotation
/// pages are same-origin, so they read the same `localStorage["theme"]` key the
/// dashboard's toggle writes (falling back to the OS preference). Runs in
/// `<head>` before paint to avoid a flash of the wrong theme.
pub(crate) const THEME_SCRIPT: &str = r#"<script>(function(){var t=localStorage.getItem("theme");if(t==="light"||t==="dark"){document.documentElement.setAttribute("data-theme",t);}})();</script>"#;

/// Pre-paint head script: restore the persisted "hide source" preference onto
/// `<html>` before the body renders, so source rows never flash into view.
/// Mirrors [`THEME_SCRIPT`].
const HIDE_SRC_HEAD: &str = r#"<script>(function(){if(localStorage.getItem("hidesrc")==="1"){document.documentElement.classList.add("hide-src");}})();</script>"#;

/// Wires the `#toggle-src` button to flip the `hide-src` class on `<html>` and
/// persist it. Toggling only reflows the on-screen `tbody.chunk`s (off-screen
/// ones stay `content-visibility`-skipped), so it stays responsive on huge pages.
/// Pressing the `s` key toggles source visibility too (ignored while typing in
/// an input/textarea or with a modifier held).
///
/// Lives in the `annotate_hide_src.js` asset rather than a string literal in this
/// file, so it is real JavaScript to an editor and needs no rebuild to change.
fn hide_src_js() -> &'static str {
    static S: OnceLock<String> = OnceLock::new();
    S.get_or_init(|| crate::assets::read("annotate_hide_src.js"))
}

/// Hover-arrow overlay. Purely additive to the static `[branch N ↑/↓]` links
/// (which already jump on click and highlight via `:target`). On hover over a
/// branch tag it draws **one** SVG curve from the left of the `[branch …]` tag
/// to the right edge of the target instruction, with an arrowhead, and
/// highlights the target row; everything clears on mouse-out. A single
/// event-delegated listener, never more than one curve in the DOM, and the
/// left-bow is clamped on-page — so it stays snappy at any page size.
///
/// Lives in the `annotate_arrow.js` asset rather than a string literal in this
/// file, so it is real JavaScript to an editor and needs no rebuild to change.
fn arrow_js() -> &'static str {
    static S: OnceLock<String> = OnceLock::new();
    S.get_or_init(|| crate::assets::read("annotate_arrow.js"))
}

/// Top-right vector-op hover panel. On hover over a vector instruction row
/// (which carries `data-v*` attributes from [`vec_data_attrs`]) it fills the
/// `#vhover` panel with that op's SEW / LMUL / VL range / exec / element count,
/// plus the full VL histogram of the controlling `vsetvl*`.
///
/// The histogram bins live only on the `vsetvl*` row (the earlier commit moved
/// them off the consuming ops), so the panel finds the nearest `.vhist` at or
/// above the hovered row within the same table and clones that graphic in —
/// labelling which `vsetvl*` PC it came from. This restores the histogram at
/// the point of use without scrolling, and makes the "nearest vsetvl* above"
/// attribution explicit so the reader can sanity-check it under branches.
///
/// The panel keeps showing the last vector op until another is hovered. A
/// single event-delegated listener; self-guards if `#vhover` is absent.
///
/// Lives in the `annotate_hover.js` asset rather than a string literal in this
/// file, so it is real JavaScript to an editor and needs no rebuild to change.
fn hover_js() -> &'static str {
    static S: OnceLock<String> = OnceLock::new();
    S.get_or_init(|| crate::assets::read("annotate_hover.js"))
}

/// Loop-region highlighter. When the page is opened from the hotspots list with
/// a `?region=<start_hex>-<end_hex>` query (start = loop head, end = back-edge
/// branch), it marks the two edge rows and the body between them, then scrolls
/// the loop head into view — so the start and end of the hot loop are obvious.
///
/// Lives in the `annotate_region.js` asset rather than a string literal in this
/// file, so it is real JavaScript to an editor and needs no rebuild to change.
fn region_js() -> &'static str {
    static S: OnceLock<String> = OnceLock::new();
    S.get_or_init(|| crate::assets::read("annotate_region.js"))
}

/// The base stylesheet of every annotation page. Opens with a theme-aware
/// palette mirroring the dashboard's `oc.css` variables, so the report matches
/// whichever theme (dark/light) is active.
///
/// Lives in the `annotate.css` asset rather than a string literal in this file,
/// so it is real CSS to an editor and needs no rebuild to change. It sits beside
/// the dashboard's `oc.css` and shares its custom-property names.
///
/// Cached for the process: the sheet is emitted once per page-build, which in a
/// live run happens per batch, and re-reading between batches could otherwise
/// give successive artifacts of the same run different bytes.
pub(super) fn css() -> &'static str {
    static CSS: OnceLock<String> = OnceLock::new();
    CSS.get_or_init(|| crate::assets::read("annotate.css"))
}

/// Render the `class` cell for one asm row: the instruction-class tag string
/// `[..]`, coloured by its dominant class. When `branch` is `Some((target,
/// delta))` — a branch/jump whose target is another row in the same table — the
/// distance and direction are folded into the tag (`[branch 20 ↑]`) and the
/// whole tag becomes the jump link (`#pc-<target>`; a click navigates and CSS
/// `:target` highlights the destination). `delta` is the signed row distance
/// (negative = backward/up, e.g. a loop back-edge).
fn class_cell(class_bits: u8, branch: Option<(u64, isize)>) -> String {
    let tags = class_tags(class_bits);
    let cls = if tags.contains("fmacc") {
        "tag-fmacc"
    } else if tags.contains("float") {
        "tag-float"
    } else if tags.contains("vector") {
        "tag-vector"
    } else if tags.contains("branch") {
        "tag-branch"
    } else {
        "tag"
    };
    match branch {
        Some((target, delta)) => {
            let (arrow, mag) = if delta < 0 {
                ('\u{2191}', -delta) // ↑ backward (loop back-edge)
            } else {
                ('\u{2193}', delta) // ↓ forward
            };
            format!(
                "<a class=\"brj\" href=\"#pc-{target:x}\" title=\"jump to 0x{target:x} ({mag} rows {})\">\
                 <span class=\"{cls}\">[{tags} {mag}{arrow}]</span></a>",
                if delta < 0 { "up" } else { "down" }
            )
        }
        None => format!("<span class=\"{cls}\">[{tags}]</span>"),
    }
}

/// Extra HTML fragments appended, in order, to each asm row's `vec` cell.
///
/// The `vec` cell is shared rather than given a column per metric because these
/// annotations are mutually exclusive with the VL histogram in practice (a
/// whole-register op carries no VL histogram), and an always-present empty
/// column would cost horizontal space on every row of every function page.
///
/// Registering here rather than editing the row `format!` keeps the argument
/// list and the `<td>` count in correspondence: those are positional, so an
/// added cell that misses one of them shifts every following column.
const VEC_CELL_EXTRAS: &[fn(&PcRecord) -> String] = &[];

/// Stylesheet fragments appended to [`css`] when `annotate.css` is emitted, one
/// per optional row annotation. Kept beside the code that generates the markup
/// it styles, so a fragment cannot outlive its renderer unnoticed.
pub(super) const EXTRA_CSS: &[&str] = &[];

/// The stylesheet as actually emitted: the base [`css`] plus every registered
/// [`EXTRA_CSS`] fragment. One definition, so tests cannot check a sheet the
/// report does not ship.
pub(super) fn emitted_stylesheet() -> String {
    css().to_string() + &EXTRA_CSS.concat()
}

/// Render every fragment in [`VEC_CELL_EXTRAS`] for one row, concatenated.
fn vec_cell_extras(r: &PcRecord) -> String {
    VEC_CELL_EXTRAS.iter().map(|f| f(r)).collect()
}

/// A rendered row in the function's single table: either a source-line header
/// or one asm row. A `Source` header is emitted whenever the source *file or
/// line* changes between consecutive (PC-ordered) asm rows, so the same line may
/// reappear several times when the compiler interleaves it with others.
///
enum Cell<'a> {
    Source {
        line_no: u32,
        text: Option<&'a str>,
        /// Σ exec of the contiguous run of asm rows this header introduces.
        exec_count: u64,
    },
    /// An asm row plus its `(file index, line)`, or `None` for orphan rows with
    /// no DWARF line info. A `line_no` of 0 means DWARF attributed the PC to no
    /// source line (compiler-generated code); the line cell is left blank.
    Asm(&'a PcRecord, Option<(usize, u32)>),
}

/// Signed row distance from an asm row to its in-table branch/jump target, or
/// `None` for fall-through instructions, indirect jumps, self-targets, and
/// cross-file targets (rare, from inlining). Negative = backward/up.
fn branch_link(
    rec: &PcRecord,
    from_seq: usize,
    pc_to_seq: &HashMap<u64, usize>,
) -> Option<(u64, isize)> {
    let target = rec.branch_target?;
    let &to_seq = pc_to_seq.get(&target)?;
    if to_seq == from_seq {
        return None;
    }
    Some((target, to_seq as isize - from_seq as isize))
}

/// Render one file's (or the orphan block's) table: a column-title header row
/// followed by interleaved source and asm rows. Each branch's `class` tag
/// carries a `[branch N ↑/↓]` jump link to its target row; target rows are
/// marked so landing points are visible at rest.
fn render_table(buf: &mut String, cells: &[Cell], max_exec: u64) {
    // Assign each asm row a sequence index and map its PC to it, so a branch can
    // report the signed row distance to (and link to) its target. O(n), no
    // per-row width — control flow lives entirely in the `class` cell now.
    let mut pc_to_seq: HashMap<u64, usize> = HashMap::new();
    // nosemgrep: dos-unbounded-memory-allocation -- len of a resident slice
    let mut asm_seq: Vec<Option<usize>> = Vec::with_capacity(cells.len());
    let mut seq = 0usize;
    for c in cells {
        match c {
            Cell::Asm(r, _) => {
                pc_to_seq.insert(r.pc, seq);
                asm_seq.push(Some(seq));
                seq += 1;
            }
            Cell::Source { .. } => asm_seq.push(None),
        }
    }
    // PCs landed on by an in-table branch, so we can mark target rows at rest.
    let mut is_target: std::collections::HashSet<u64> = std::collections::HashSet::new();
    for c in cells {
        if let Cell::Asm(r, _) = c {
            if let Some(t) = r.branch_target {
                if t != r.pc && pc_to_seq.contains_key(&t) {
                    is_target.insert(t);
                }
            }
        }
    }

    buf.push_str("<table>");
    // Column-title header explaining each column. `class` and `vec` are now
    // distinct columns. In its own <thead> so it always participates in layout.
    // nosemgrep: error-handling-ignored-critical-result -- fmt::Write on a String is infallible
    let _ = writeln!(
        buf,
        "<thead><tr class=\"col-header\"><td class=\"pc\">addr</td>\
         <td class=\"ln\">line</td><td class=\"count\">exec</td>\
         <td class=\"freq\">freq</td><td class=\"cls\">class</td>\
         <td class=\"vec\">Vector</td>\
         <td class=\"ins\">instruction</td></tr></thead>"
    );

    // Emit rows in fixed-size <tbody> chunks. Every chunk after the first is
    // `content-visibility:auto` (see STYLE), so the browser skips layout/paint
    // of off-screen chunks — this is what keeps 100k+-row pages scrollable and
    // makes the hide-source toggle cheap. The first chunk stays plain so it
    // anchors the auto-layout column widths (no jitter as later chunks render).
    const CHUNK: usize = 256;
    let mut chunk = 0usize;
    for (i, c) in cells.iter().enumerate() {
        if i % CHUNK == 0 {
            if i > 0 {
                buf.push_str("</tbody>");
            }
            buf.push_str(if chunk == 0 {
                "<tbody>"
            } else {
                "<tbody class=\"chunk\">"
            });
            chunk += 1;
        }
        match c {
            Cell::Source {
                line_no,
                text,
                exec_count,
                ..
            } => {
                // The fallback must be escaped like any other text: written raw
                // it parses as an unknown element and renders as a blank row of
                // full height, i.e. wasted space that looks like a bug.
                let text = text.map(esc).unwrap_or_else(|| esc("<source unavailable>"));
                // Source code spans the instruction/class/vec columns.
                // nosemgrep: error-handling-ignored-critical-result -- fmt::Write on a String is infallible
                let _ = writeln!(
                    buf,
                    "<tr class=\"src\"><td class=\"pc\"></td>\
                     <td class=\"ln\">{}</td>\
                     <td class=\"count\">{}</td><td class=\"freq\"></td>\
                     <td class=\"cls\"></td><td class=\"vec\"></td>\
                     <td class=\"code\">{}</td></tr>",
                    line_no, exec_count, text
                );
            }
            Cell::Asm(r, src) => {
                let pct = if max_exec > 0 {
                    (r.exec_count as f64 / max_exec as f64 * 38.0).round() as u64
                } else {
                    0
                };
                let from_seq = asm_seq[i].expect("asm cell has a seq index");
                let link = branch_link(r, from_seq, &pc_to_seq);
                let bt = if is_target.contains(&r.pc) { " bt" } else { "" };
                let zero = if r.exec_count == 0 { " zero" } else { "" };
                // Every asm row carries its own line number, so the `line`
                // column stays meaningful with source hidden.
                // Line 0 is DWARF's "no source line" (compiler-generated code):
                // print nothing rather than a meaningless "0".
                let ln_text = match src {
                    Some((_, n)) if *n > 0 => n.to_string(),
                    _ => String::new(),
                };
                let ln_cell = format!("<td class=\"ln\">{ln_text}</td>");
                // nosemgrep: error-handling-ignored-critical-result -- fmt::Write on a String is infallible
                let _ = writeln!(
                    buf,
                    "<tr id=\"pc-{:x}\" class=\"asm{}{}\"{}>\
                     <td class=\"pc\">0x{:x}</td>{}\
                     <td class=\"count\">{}</td>\
                     <td class=\"freq\"><span class=\"bar\" style=\"width:{}px\"></span></td>\
                     <td class=\"cls\">{}</td>\
                     <td class=\"vec\">{}{}</td>\
                     <td class=\"ins\">{}</td></tr>",
                    r.pc,
                    zero,
                    bt,
                    vec_data_attrs(r),
                    r.pc,
                    ln_cell,
                    r.exec_count,
                    pct,
                    class_cell(r.class, link),
                    vector_hist_html(r),
                    vec_cell_extras(r),
                    esc(&r.disasm),
                );
            }
        }
    }
    if !cells.is_empty() {
        buf.push_str("</tbody>");
    }
    buf.push_str("</table>");
}

fn render_function(f: &AnnotatedFunction) -> String {
    let max_exec = f
        .files
        .iter()
        .flat_map(|file| file.rows.iter().map(|row| &row.rec))
        .chain(f.orphan.iter())
        .map(|r| r.exec_count)
        .max()
        .unwrap_or(1);

    // nosemgrep: dos-unbounded-memory-allocation -- a literal, not a dynamic size
    let mut body = String::with_capacity(10000);
    // nosemgrep: error-handling-ignored-critical-result -- fmt::Write on a String is infallible
    let _ = write!(
        body,
        "<!doctype html><html><head><meta charset=\"utf-8\"><title>annotate: {}</title>{}{}\
         <link rel=\"stylesheet\" href=\"annotate.css\"></head><body>",
        esc(&f.name),
        THEME_SCRIPT,
        HIDE_SRC_HEAD,
    );
    // nosemgrep: error-handling-ignored-critical-result -- fmt::Write on a String is infallible
    let _ = write!(
        body,
        "<p><a href=\"annotate_index.html\">&larr; all functions</a></p><h1>{}</h1>\
         <p class=\"summary\">{} instructions executed</p>\
         <div class=\"toolbar\"><button id=\"toggle-src\" type=\"button\" title=\"Toggle source (s)\">Hide source</button></div>\
         <div class=\"hover-stack\">\
           <div id=\"vhover\" class=\"vhover-panel\">\
             <div class=\"vhover-title\">Vector op</div>\
             <div class=\"vhover-body\"><div class=\"vhover-empty\">Hover a vector instruction</div></div>\
           </div>\
         </div>",
        esc(&f.name),
        f.total_instructions
    );

    // One table for the whole function. Splitting per source file would print a
    // heading (and start a fresh table) at every inlining boundary, chopping the
    // disassembly into fragments; instead the file is carried on each row's
    // `line` cell and surfaced on hover. A single table also lets branch arrows
    // resolve across inlined code, since `render_table` only links targets it
    // can see.
    // Distinct source paths, in first-seen order; rows reference them by index.
    let mut paths: Vec<&str> = Vec::new();
    let mut cells: Vec<Cell> = Vec::new();
    let mut last: Option<(usize, u32)> = None;
    for file in &f.files {
        let fidx = match paths.iter().position(|p| *p == file.path) {
            Some(i) => i,
            None => {
                paths.push(&file.path);
                paths.len() - 1
            }
        };
        for (i, row) in file.rows.iter().enumerate() {
            let here = (fidx, row.line_no);
            if last != Some(here) {
                // Line 0 is compiler-generated code DWARF maps to no source
                // line. A header for it can only ever render as an empty row, so
                // skip it and let the asm run on uninterrupted; `last` still
                // advances so the run isn't retried per row.
                if row.line_no > 0 {
                    let run_exec = file.rows[i..]
                        .iter()
                        .take_while(|r| r.line_no == row.line_no)
                        .map(|r| r.rec.exec_count)
                        .sum();
                    cells.push(Cell::Source {
                        line_no: row.line_no,
                        text: row.text.as_deref(),
                        exec_count: run_exec,
                    });
                }
                last = Some(here);
            }
            cells.push(Cell::Asm(&row.rec, Some(here)));
        }
    }
    if !cells.is_empty() {
        render_table(&mut body, &cells, max_exec);
    }

    if !f.orphan.is_empty() {
        // nosemgrep: error-handling-ignored-critical-result -- fmt::Write on a String is infallible
        let _ = write!(
            body,
            "<h2>(no source line &mdash; {} instructions)</h2>",
            f.orphan_exec
        );
        let cells: Vec<Cell> = f.orphan.iter().map(|r| Cell::Asm(r, None)).collect();
        render_table(&mut body, &cells, max_exec);
    }

    body.push_str("<script src=\"annotate.js\"></script>");
    body.push_str("</body></html>");
    body
}

fn render_index(funcs: &[AnnotatedFunction], has_hotspots: bool) -> String {
    let mut body = String::new();
    // nosemgrep: error-handling-ignored-critical-result -- fmt::Write on a String is infallible
    let _ = write!(
        body,
        "<!doctype html><html><head><meta charset=\"utf-8\"><title>annotate index</title>{}\
         <link rel=\"stylesheet\" href=\"annotate.css\"></head><body>",
        THEME_SCRIPT,
    );
    body.push_str("<h1>Source &amp; Assembly Annotation</h1>");
    if has_hotspots {
        body.push_str(
            "<p class=\"summary\">\
             <a href=\"annotate_hotspots.html\">\u{1f525} Hotspots &mdash; worst loops (hot &amp; poorly vectorized)</a></p>",
        );
    }

    body.push_str("<table>");
    let total: u64 = funcs.iter().map(|f| f.total_instructions).sum();
    for f in funcs {
        let pct = if total > 0 {
            f.total_instructions as f64 / total as f64 * 100.0
        } else {
            0.0
        };
        let barpx = (pct * 2.0).round() as u64;
        // nosemgrep: error-handling-ignored-critical-result -- fmt::Write on a String is infallible
        let _ = writeln!(
            body,
            "<tr><td class=\"count\">{}</td><td>{:.1}%</td>\
             <td><span class=\"bar\" style=\"width:{}px\"></span></td>\
             <td><a href=\"annotate_{}.html\">{}</a></td></tr>",
            f.total_instructions,
            pct,
            barpx,
            f.safe_name,
            esc(&f.name)
        );
    }
    body.push_str("</table></body></html>");
    body
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every registered EXTRA_CSS fragment must reach the stylesheet that ships.
    ///
    /// The fragments are appended to CSS at emit time rather than living inside
    /// it, so a fragment that is registered but never concatenated would leave
    /// its markup unstyled while every other test still passed.
    ///
    /// This checks the mechanism only. That each *class name* the row extras emit
    /// has a matching rule is checked per-feature, next to the feature that
    /// registers it, because populating an extra requires that feature's own
    /// fields.
    #[test]
    fn registered_extra_css_reaches_the_stylesheet() {
        let stylesheet = emitted_stylesheet();
        for frag in EXTRA_CSS {
            assert!(
                stylesheet.contains(*frag),
                "a registered EXTRA_CSS fragment is missing from the emitted \
                 stylesheet: {frag:?}"
            );
        }
    }

    fn rec(pc: u64, target: Option<u64>) -> PcRecord {
        PcRecord {
            pc,
            disasm: format!("insn_{pc:x}"),
            exec_count: 1,
            class: if target.is_some() {
                super::super::pc_profile::class::BRANCH
            } else {
                0
            },
            branch_target: target,
            ..Default::default()
        }
    }

    /// A back-edge branch renders a `↑N` badge anchored to its target row, the
    /// target row is marked `bt`, every asm row is id-anchored, and the branch
    /// gutter stays a single cell wide (no lane-packed `│` blow-up).
    #[test]
    fn branch_badge_and_anchors() {
        let recs = [rec(0x10, None), rec(0x14, None), rec(0x18, Some(0x10))];
        let cells: Vec<Cell> = recs.iter().map(|r| Cell::Asm(r, None)).collect();
        let mut buf = String::new();
        render_table(&mut buf, &cells, 1);

        // Anchor ids for each asm row.
        assert!(buf.contains("id=\"pc-10\""));
        assert!(buf.contains("id=\"pc-18\""));
        // Back-edge tag `[branch 2 ↑]` linking to the target row.
        assert!(buf.contains("href=\"#pc-10\""));
        assert!(
            buf.contains("[branch 2\u{2191}]"),
            "expected [branch 2 ↑], got: {buf}"
        );
        // Target row (0x10) is marked as a branch target.
        assert!(buf.contains("id=\"pc-10\" class=\"asm bt\""));
        // No lane-packed vertical-bar gutter anywhere.
        assert!(!buf.contains('\u{2502}'));
    }

    /// A forward branch renders `[branch N ↓]`; indirect/out-of-table targets
    /// render a plain tag with no link.
    #[test]
    fn forward_and_absent_badges() {
        let recs = [
            rec(0x10, Some(0x18)),
            rec(0x14, Some(0x999)),
            rec(0x18, None),
        ];
        let cells: Vec<Cell> = recs.iter().map(|r| Cell::Asm(r, None)).collect();
        let mut buf = String::new();
        render_table(&mut buf, &cells, 1);
        assert!(buf.contains("[branch 2\u{2193}]"), "expected [branch 2 ↓]");
        // Target outside the table -> no anchor to it (plain tag).
        assert!(!buf.contains("#pc-999"));
    }

    /// Rows are emitted in balanced <thead>/<tbody> chunks: the header lives in
    /// a single <thead>, the first CHUNK rows in a plain <tbody> (anchors column
    /// widths), and each subsequent chunk in a `content-visibility` <tbody
    /// class="chunk"> so off-screen rows skip layout/paint on huge pages.
    #[test]
    fn rows_are_chunked_into_tbodies() {
        // 600 rows -> 3 chunks (256 + 256 + 88): 1 plain + 2 content-visibility.
        let recs: Vec<PcRecord> = (0..600).map(|i| rec(0x10 + i * 4, None)).collect();
        let cells: Vec<Cell> = recs.iter().map(|r| Cell::Asm(r, None)).collect();
        let mut buf = String::new();
        render_table(&mut buf, &cells, 1);

        assert_eq!(buf.matches("<thead>").count(), 1);
        // Balanced open/close, and exactly ceil(600/256)=3 groups.
        let opens = buf.matches("<tbody").count();
        assert_eq!(opens, 3, "expected 3 tbody chunks");
        assert_eq!(buf.matches("</tbody>").count(), opens);
        // First chunk is plain (width anchor); the rest are cv-skippable.
        assert_eq!(buf.matches("<tbody class=\"chunk\">").count(), 2);
    }
}
