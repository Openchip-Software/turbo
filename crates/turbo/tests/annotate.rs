//! End-to-end test for the source & assembly annotation report.
//!
//! Runs `turbo run` with `--annotate` and asserts the static `annotate_*.html`
//! pages are produced in the output directory.

use crate::harness::TurboTest;

#[test]
fn annotate_report_is_generated() {
    let run = TurboTest::new(
        "annotate_flops_test",
        "test_bins/basic_branch_asm/flops_test.riscv",
    )
    .process(|p| p.annotate = true)
    .run();

    // Index page must exist and list the hot function.
    let index = run.annotate_page("annotate_index.html");
    assert!(
        index.contains("Source &amp; Assembly Annotation"),
        "index missing header"
    );
    assert!(
        index.contains("annotate_"),
        "index should link to at least one per-function page"
    );

    // At least one per-function page must exist and contain disassembly rows.
    let per_func = run.annotate_function_pages();
    assert!(
        !per_func.is_empty(),
        "expected at least one annotate_<func>.html page, found none"
    );

    let page = run.annotate_page(&per_func[0]);
    assert!(
        page.contains("instructions executed"),
        "per-function page missing summary"
    );
    assert!(
        page.contains("0x"),
        "per-function page should contain disassembly PC rows"
    );
}
