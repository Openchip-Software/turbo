use std::fmt::Write as _;

use crate::common;

/// One optional byte-valued column appended to the per-ROI table, sitting
/// between the fixed `St_*` and `Intens` columns.
///
/// This exists so an optional metric is registered in ONE place instead of
/// being threaded through the five separate points the table is assembled from
/// (unit-suffixed header pair, header format string, per-row value computation,
/// row format string, row argument list). Getting one of those five wrong
/// silently misaligns the whole table.
struct ExtraColumn {
    /// Header text. Takes the table's unit choice so the label tracks the
    /// scaling, exactly as the fixed `Ld_KB`/`Ld_MB` pair does.
    header: fn(use_mb: bool) -> &'static str,
    /// This ROI's value for the column, in bytes. Scaled by the table divisor.
    value: fn(&common::RoiInfo) -> u64,
}

/// The extra byte-valued columns the ROI table renders, in display order.
///
/// Deliberately NOT folded into `bytes_moved`/`intensity`: those describe
/// traffic through the normal memory hierarchy, and mixing another address
/// space into the arithmetic-intensity denominator would silently change what
/// that number means.
const EXTRA_COLUMNS: &[ExtraColumn] = &[];

/// Extra `label -> value` lines appended to the program summary as their own
/// separator-delimited block. Same one-place-registration rationale as
/// [`EXTRA_COLUMNS`]; the block is omitted entirely when this is empty, so
/// removing every entry does not leave a doubled separator behind.
type SummaryExtra = (&'static str, fn(&common::RoiInfo) -> u64);

const SUMMARY_EXTRAS: &[SummaryExtra] = &[];

pub fn print_program_summary(g: &common::RoiInfo) {
    let line = "─".repeat(40);
    println!("\n{line}");
    println!("  Program Summary");
    println!("{line}");
    println!("  Instructions        {:>16}", g.instructions);
    println!("  Branches            {:>16}", g.branches);
    println!("{line}");
    println!("  Vector inst         {:>16}", g.vector_instructions);
    println!("  Vector elements     {:>16}", g.vector_elements);
    println!("  Vector ld inst      {:>16}", g.vector_load_instructions);
    println!("  Vector ld elements  {:>16}", g.vector_load_elements);
    println!("{line}");
    println!("  Float inst          {:>16}", g.float_instructions);
    println!("  Float elements      {:>16}", g.float_elements);
    println!("  FMA inst            {:>16}", g.fmacc_instructions);
    println!("  FMA elements        {:>16}", g.fmacc_elements);
    println!("{line}");
    println!("  Load bytes          {:>16}", g.load_bytes);
    println!("  Store bytes         {:>16}", g.store_bytes);
    // Only when a cache was actually modelled: an unmodelled run must not show
    // a row of zeroes that reads as "no misses".
    if !g.cache.is_zero() {
        println!("{line}");
        println!("  Cache lines (vec)   {:>16}", g.cache.lines());
        println!(
            "  L1 hits / misses    {:>16}",
            format!("{} / {}", g.cache.l1_hits, g.cache.l1_misses)
        );
        println!(
            "  L2 hits / misses    {:>16}",
            format!("{} / {}", g.cache.l2_hits, g.cache.l2_misses)
        );
        if let Some(rate) = g.cache.l1_hit_rate() {
            println!("  L1 hit rate         {:>15.2}%", rate * 100.0);
        }
        // Said in the output, not only in the docs: these counts cover the
        // vector loads and stores the tracer captures. Scalar accesses and
        // instruction fetch never reach the model, so this is not a hit rate a
        // machine would report.
        println!("  (vector data-side only: no scalar accesses, no instruction fetch)");
    }
    if !SUMMARY_EXTRAS.is_empty() {
        println!("{line}");
        for (label, value) in SUMMARY_EXTRAS {
            // Width 20 puts the value field at column 22, matching every fixed
            // line above (e.g. "  Instructions" + 8 spaces).
            println!("  {:<20}{:>16}", label, value(g));
        }
    }
    println!("{line}\n");
}

fn print_roi_table(rows: &[&common::RoiInfo], global: &common::RoiInfo, name_col_header: &str) {
    let total_instrs = global.instructions.max(1) as f64;

    let max_bytes = rows
        .iter()
        .map(|f| f.load_bytes.max(f.store_bytes))
        .max()
        .unwrap_or(0);
    let use_mb = max_bytes >= 10 * 1024 * 1024;
    let (ld_hdr, st_hdr) = if use_mb {
        ("Ld_MB", "St_MB")
    } else {
        ("Ld_KB", "St_KB")
    };
    let mut hdr = format!(
        "{:>12}  {:>5}  {:>5}  {:>6}  {:>5}  {:>5}  {:>8}  {:>8}",
        "Instrs", "%Tot", "Vec%", "AvgVL", "Flt%", "FMA%", ld_hdr, st_hdr,
    );
    for col in EXTRA_COLUMNS {
        // nosemgrep: error-handling-ignored-critical-result -- fmt::Write on a String is infallible
        let _ = write!(hdr, "  {:>8}", (col.header)(use_mb));
    }
    // nosemgrep: error-handling-ignored-critical-result -- fmt::Write on a String is infallible
    let _ = write!(hdr, "  {:>7}  {}", "Intens", name_col_header);
    let sep = "─".repeat(hdr.len());
    println!("\n{sep}");
    println!("{hdr}");
    println!("{sep}");

    for f in rows {
        let instrs = f.instructions;
        let pct_tot = instrs as f64 / total_instrs * 100.0;
        let vec_pct = if instrs > 0 {
            f.vector_instructions as f64 / instrs as f64 * 100.0
        } else {
            0.0
        };
        let avg_vl = if f.vector_instructions > 0 {
            f.vector_elements as f64 / f.vector_instructions as f64
        } else {
            0.0
        };
        let flt_pct = if instrs > 0 {
            f.float_instructions as f64 / instrs as f64 * 100.0
        } else {
            0.0
        };
        let fma_pct = if instrs > 0 {
            f.fmacc_instructions as f64 / instrs as f64 * 100.0
        } else {
            0.0
        };
        let divisor = if use_mb { 1024.0 * 1024.0 } else { 1024.0 };
        let ld = f.load_bytes as f64 / divisor;
        let st = f.store_bytes as f64 / divisor;
        let bytes_moved = f.load_bytes + f.store_bytes;
        let intensity = if bytes_moved > 0 {
            f.float_elements as f64 / bytes_moved as f64
        } else {
            0.0
        };
        let name = if f.name.len() > 60 {
            &f.name[..60]
        } else {
            &f.name
        };

        let mut row = format!(
            "{:>12}  {:>5.1}  {:>5.1}  {:>6.1}  {:>5.1}  {:>5.1}  {:>8.1}  {:>8.1}",
            instrs, pct_tot, vec_pct, avg_vl, flt_pct, fma_pct, ld, st,
        );
        for col in EXTRA_COLUMNS {
            // nosemgrep: error-handling-ignored-critical-result -- fmt::Write on a String is infallible
            let _ = write!(row, "  {:>8.1}", (col.value)(f) as f64 / divisor);
        }
        // nosemgrep: error-handling-ignored-critical-result -- fmt::Write on a String is infallible
        let _ = write!(row, "  {:>7.2}  {}", intensity, name);
        println!("{row}");
    }
    println!("{sep}\n");
}

pub fn print_function_summary(functions: &[common::RoiInfo], global: &common::RoiInfo) {
    let total_instrs = global.instructions.max(1) as f64;
    let mut sorted: Vec<&common::RoiInfo> = functions
        .iter()
        .filter(|f| &*f.name != "Global" && f.instructions as f64 / total_instrs >= 0.01)
        .collect();
    sorted.sort_by_key(|r| std::cmp::Reverse(r.instructions));
    print_roi_table(&sorted, global, "Function");
}

pub fn print_regions_summary(rois: &[common::RoiInfo], global: &common::RoiInfo) {
    let mut sorted: Vec<&common::RoiInfo> = rois.iter().filter(|r| &*r.name != "Global").collect();
    sorted.sort_by_key(|r| std::cmp::Reverse(r.instructions));
    print_roi_table(&sorted, global, "Region");
}

pub fn print_summary(pd: &common::PerformanceData, summary: &str) {
    let show_program = matches!(summary, "program" | "all");
    let show_function = matches!(summary, "function" | "all");
    let show_regions = matches!(summary, "regions" | "all");
    if !show_program && !show_function && !show_regions {
        log::warn!(
            "Unknown --summary value: '{}'. Supported: program, function, regions (or plain --summary for all)",
            summary
        );
        return;
    }
    let global = pd
        .rois
        .iter()
        .find(|r| &*r.name == "Global")
        .cloned()
        .unwrap_or_default();
    if show_program {
        print_program_summary(&global);
    }
    if show_regions {
        print_regions_summary(&pd.rois, &global);
    }
    if show_function {
        print_function_summary(&pd.functions, &global);
    }
}
