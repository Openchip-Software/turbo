//! vDSO coverage: the guest vDSO must reach the enricher as a real module.
//!
//! The vDSO is the one loaded object no guest syscall and no on-disk file can
//! reveal: QEMU's ELF loader injects it before the guest runs, and the image
//! only ever exists in guest memory. The tracer plugin is therefore the sole
//! source for it -- it reads the base out of the guest's own auxiliary vector
//! (`AT_SYSINFO_EHDR`) at first vCPU init and dumps the image to
//! `VDSO_DUMP_FILE`, so the consumer, which has no guest-memory access, can
//! load it as a module.
//!
//! Losing that leaves a hole in module coverage that the enricher treats as
//! fatal (`RerfError::address_not_covered`), so a guest calling
//! `gettimeofday`/`clock_gettime` -- i.e. any real benchmark that times itself
//! -- fails its whole run over three instructions.
//!
//! Nothing else in the suite catches that. The vDSO-exercising fixtures
//! (`*_gtod`, `static_pie`, `dyn_nopie`, `dyn_sepcode`) are all used from
//! `pc_equivalence`, which goes through `--dump-pcs`: that path decodes but
//! does not enrich, and PC reconstruction alone does not need the vDSO module.
//! Every test that *does* enrich uses a static assembly fixture that never
//! enters the vDSO. So these two tests are the coverage for it: one on the
//! plugin's side files, one end to end.
//!
//! `static_gtod` is the guest for both: statically linked, so the vDSO is the
//! *only* object besides the main binary and nothing else can mask its
//! absence.

use turbo::processing::RoiInfo;

use crate::harness::TurboTest;

/// The fixture is built by `make -f Makefile.tests linker-flavours`, like every
/// other `linker_flavours` guest the suite uses.
const GTOD_BIN: &str = "test_bins/linker_flavours/static_gtod.riscv";

/// The plugin's half: it located the vDSO and dumped a loadable ELF image of
/// it. Checked separately from enrichment so that a failed auxv walk reports
/// itself here -- as "the plugin found no vDSO" -- rather than only as an
/// uncovered PC several stages downstream.
#[test]
fn plugin_reports_the_vdso_and_dumps_a_loadable_image() {
    let run = TurboTest::new("vdso_plugin_side_files", GTOD_BIN).record();

    let meminfo = run
        .metadata()
        .expect("the plugin must write turbo_metadata.json")
        .meminfo
        .expect("meminfo is captured on first vCPU init, before any guest instruction");

    assert!(
        meminfo.vdso_addr != 0 && meminfo.vdso_size != 0,
        "plugin reported no vDSO (addr 0x{:x}, size 0x{:x}): the guest's auxv either \
         could not be walked at first vCPU init or carried no AT_SYSINFO_EHDR",
        meminfo.vdso_addr,
        meminfo.vdso_size,
    );

    // The consumer's only route to the vDSO: it cannot read guest memory, so if
    // this file is absent or not an ELF there is no module to be had, however
    // well-known the address is.
    let dump = run.dir.join(turbo_tracer_shared::VDSO_DUMP_FILE);
    let bytes = std::fs::read(&dump).unwrap_or_else(|e| {
        panic!(
            "plugin reported a vDSO at 0x{:x} but wrote no dump at {}: {e}",
            meminfo.vdso_addr,
            dump.display()
        )
    });
    assert_eq!(
        bytes.len() as u64,
        meminfo.vdso_size,
        "vDSO dump is {} bytes but the plugin reported {}",
        bytes.len(),
        meminfo.vdso_size
    );
    assert_eq!(
        &bytes[..4],
        b"\x7fELF",
        "the in-memory vDSO is a complete ELF image, so the dump must be one too"
    );
}

/// The end-to-end half: a guest that actually executes vDSO instructions is
/// enriched, and those instructions are attributed to their `__vdso_*` symbol.
///
/// Both halves of the assertion matter. Today a missing vDSO module aborts the
/// run, so reaching the assertion at all is most of the test; the symbol check
/// is what keeps it honest if uncovered PCs are ever made non-fatal, since the
/// run would then succeed with the vDSO's instructions silently unattributed.
#[test]
fn vdso_instructions_are_enriched_and_symbolized() {
    let run = TurboTest::new("vdso_enriched_symbols", GTOD_BIN).run();
    let pd = run.perf_data();

    // Any `__vdso_*` symbol, not one by name: which stub glibc's
    // `gettimeofday` lands in is a libc detail (current glibc routes it through
    // `__vdso_clock_gettime`, older ones call `__vdso_gettimeofday`), and the
    // guest fixtures are rebuilt against CI's own toolchain. What is being
    // tested is that the vDSO module resolved at all.
    let vdso_fns: Vec<&RoiInfo> = pd
        .functions
        .iter()
        .filter(|f| f.name.starts_with("__vdso_"))
        .collect();

    // The guest calls `gettimeofday`, whose libc wrapper is resolved either way,
    // so name the syscall wrappers that *were* found: seeing `clock_gettime`
    // without `__vdso_clock_gettime` is the signature of a run that enriched
    // fine but had no vDSO module to attribute the stub itself to.
    assert!(
        !vdso_fns.is_empty(),
        "no `__vdso_*` function in perf_data_final.json, so the vDSO's instructions \
         went unattributed even though the run completed. {} function(s) were resolved, \
         among them: {:?}",
        pd.functions.len(),
        pd.functions
            .iter()
            .map(|f| f.name.as_ref())
            .filter(|n| n.contains("gettime") || n.contains("vdso"))
            .collect::<Vec<_>>(),
    );

    for f in &vdso_fns {
        assert!(
            f.instructions > 0,
            "`{}` was resolved as a symbol but credited no instructions",
            f.name
        );
    }
}
