//! Hotspot ranking — find the "worst" loops in a run.
//!
//! Consumes the same per-PC profile ([`PcProfile`]) as the annotation view and
//! ranks **loop sections** by *badness*: a blend of how hot a loop is (dynamic
//! instruction count) and how improvable it looks (scalar, or vectorized but
//! running short vector lengths / drowning in address arithmetic). A section
//! that is hot **and** poorly vectorized ranks at the top; a hot but saturated
//! vector loop, or a cold scalar loop, ranks low.
//!
//! This implements loop bodies delimited by back-edges, the **Algo 1**
//! "hot × improvable" score as the default sort, the **Algo 4** tier as the
//! explanation badge, and **Algo 2** headroom as a secondary column.

use std::collections::HashMap;
use std::fmt::Write as _;

use crate::common::CacheCounts;
use crate::processing::function_lookup::MultiResolver;

use super::annotate::{esc, sanitize, THEME_SCRIPT};
use super::pc_profile::{class, PcProfile, PcRecord};

/// Improvability weight on VL under-utilization (`1 - vl_util`).
const W_VL: f64 = 0.6;
/// Improvability weight on the scalar fraction of a vectorized loop.
const W_SCALAR: f64 = 0.4;
/// Target vector fraction a healthy loop should reach, used by the Algo-2
/// "instructions recoverable" headroom estimate.
const IDEAL_VEC_FRAC: f64 = 0.9;
/// VL utilization at or above which a vectorized loop is considered to be
/// saturating its vector unit — the Algo-4 "healthy" threshold. Above this the
/// scalar instructions are unavoidable loop-control overhead amortized across
/// near-VLMAX vector work, *not* improvable address arithmetic, so the loop is
/// healthy regardless of its scalar fraction.
const HEALTHY_VL_UTIL: f64 = 0.8;
/// How many sections the hotspots page lists (worst first).
const TOP_K: usize = 50;

/// Health tier for a loop section — the Algo-4 classifier. Ordered worst-first
/// so it doubles as a coarse tiebreak.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Tier {
    /// Most of the loop's vector cache-line lookups missed the L1. Only ever
    /// reached when the run modelled a cache (`--cache-sim`); see [`Section::cache`].
    MemHeavy,
    /// Hot loop that never touched the vector unit.
    Scalar,
    /// Vectorized, but average VL is well below the widest VL it ever set.
    ShortVl,
    /// Vectorized at a good VL, but most dynamic instructions are scalar
    /// (address arithmetic / index math around the vector ops).
    AddrBound,
    /// Vectorized and running near its widest VL — leave it alone.
    Healthy,
}

impl Tier {
    fn label(self) -> &'static str {
        match self {
            Tier::MemHeavy => "mem-heavy",
            Tier::Scalar => "scalar",
            Tier::ShortVl => "short-VL",
            Tier::AddrBound => "addr-bound",
            Tier::Healthy => "healthy",
        }
    }
    /// CSS class suffix reusing the report palette (red→amber→green).
    fn css(self) -> &'static str {
        match self {
            Tier::MemHeavy => "t-mem",
            Tier::Scalar => "t-scalar",
            Tier::ShortVl => "t-shortvl",
            Tier::AddrBound => "t-addr",
            Tier::Healthy => "t-healthy",
        }
    }
}

/// One ranked loop section.
pub struct Section {
    func: String,
    safe_name: String,
    /// Loop head PC (the back-edge target) — the anchor linked on the function
    /// page (`annotate_<safe>.html#pc-<head>`).
    head_pc: u64,
    /// The back-edge branch PC that closes the loop (the "end" marker). Passed
    /// to the function page so it can highlight the start & end of the region.
    back_pc: u64,
    /// Source `(file, start_line, end_line)` spanned by the loop body, if DWARF
    /// resolved it. `start == end` for a single-line loop.
    src: Option<(String, u32, u32)>,
    /// Σ exec_count over the body — the hotness term.
    insns: u64,
    /// Times the body ran (back-edge exec count).
    trip: u64,
    vec_insns: u64,
    scalar_frac: f64,
    /// Mean VL across vector ops / the widest VL the loop ever set.
    vl_util: f64,
    vl_mean: f64,
    vlmax_obs: u32,
    /// Σ of the body's modelled cache counts. All-zero unless the run was
    /// processed with `--cache-sim`, and even then it covers **vector** loads
    /// and stores only — scalar accesses and instruction fetch never reach the
    /// model — so it describes this loop's vector memory traffic, not its full
    /// memory behaviour.
    cache: CacheCounts,
    tier: Tier,
    /// Algo-1 score: hotness × improvability. Primary sort key.
    score: f64,
    /// Algo-2 estimate of dynamic instructions recoverable by full
    /// vectorization at a healthy VL.
    headroom: f64,
}

/// Rank every loop section across all functions, worst (highest score) first,
/// truncated to [`TOP_K`]. Returns empty when the profile has no records or no
/// back-edges (no loops) were found.
pub fn rank(resolver: &MultiResolver, profile: &PcProfile) -> Vec<Section> {
    // Group records by resolving function so a back-edge only ever pairs with
    // instructions from its own function.
    let mut by_func: HashMap<usize, Vec<&PcRecord>> = HashMap::new();
    for rec in profile.records().values() {
        by_func
            .entry(rec.func_index.unwrap_or(usize::MAX))
            .or_default()
            .push(rec);
    }

    let mut out: Vec<Section> = Vec::new();
    for (fkey, recs) in by_func {
        let name = if fkey == usize::MAX {
            "(unknown)".to_string()
        } else {
            resolver
                .from_index(fkey)
                .map(|s| s.to_string())
                .unwrap_or_else(|_| format!("func_{fkey}"))
        };
        out.extend(sections_for_function(&name, &recs, resolver));
    }

    // Algo 1 primary sort (badness), Tier then insns as tiebreaks.
    out.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(b.insns.cmp(&a.insns))
    });
    out.truncate(TOP_K);
    out
}

/// Build the loop sections of one function. A **back-edge** is a control-flow
/// instruction whose `branch_target` resolves to a lower PC that is itself an
/// executed instruction in this function; the half-open PC range
/// `[target, branch]` is the loop body. Each PC is attributed to the
/// **innermost** enclosing back-edge (smallest range), so nested loops don't
/// double-count and the inner kernel is scored on its own.
fn sections_for_function(name: &str, recs: &[&PcRecord], resolver: &MultiResolver) -> Vec<Section> {
    let pcs: std::collections::HashSet<u64> = recs.iter().map(|r| r.pc).collect();

    // (head, back, trip) for each back-edge, deduped on (head, back).
    let mut backedges: Vec<(u64, u64, u64)> = Vec::new();
    for r in recs {
        if let Some(t) = r.branch_target {
            if t < r.pc && pcs.contains(&t) {
                backedges.push((t, r.pc, r.exec_count));
            }
        }
    }
    if backedges.is_empty() {
        return Vec::new();
    }

    // Attribute each record to the smallest-width back-edge that contains it.
    let mut buckets: HashMap<usize, Vec<&PcRecord>> = HashMap::new();
    for r in recs {
        let mut best: Option<(usize, u64)> = None;
        for (i, &(h, b, _)) in backedges.iter().enumerate() {
            if r.pc >= h && r.pc <= b {
                let width = b - h;
                if best.is_none_or(|(_, w)| width < w) {
                    best = Some((i, width));
                }
            }
        }
        if let Some((i, _)) = best {
            buckets.entry(i).or_default().push(r);
        }
    }

    let safe_name = sanitize(name);
    let mut sections = Vec::new();
    for (i, body) in buckets {
        let (head_pc, back_pc, trip) = backedges[i];
        sections.push(build_section(
            name, &safe_name, head_pc, back_pc, trip, &body, resolver,
        ));
    }
    sections
}

/// Fold one loop body's records into a scored [`Section`], resolving its source
/// span first. The scoring itself lives in [`fold_body`] so the unit tests can
/// exercise the real classifier without an `ElfResolver`.
fn build_section(
    name: &str,
    safe_name: &str,
    head_pc: u64,
    back_pc: u64,
    trip: u64,
    body: &[&PcRecord],
    resolver: &MultiResolver,
) -> Section {
    // Source line range: anchor on the loop head's file, then widen to the min
    // and max line any body instruction in that same file maps to. The compiler
    // interleaves lines within the loop, so the true span is min..max, not just
    // head..back.
    let src = resolver.resolve_line(head_pc).map(|(file, head_line)| {
        let file = file.to_string();
        // Ignore line 0 (DWARF "no line" / compiler-generated rows), which
        // would otherwise pin the range start to 0.
        let (mut lo, mut hi) = (u32::MAX, 0u32);
        for r in body {
            if let Some((p, l)) = resolver.resolve_line(r.pc) {
                if p == file && l > 0 {
                    lo = lo.min(l);
                    hi = hi.max(l);
                }
            }
        }
        if lo == u32::MAX {
            (file, head_line, head_line)
        } else {
            (file, lo, hi)
        }
    });

    fold_body(name, safe_name, head_pc, back_pc, trip, body, src)
}

/// The metric fold and Algo-1/2/4 scoring of one loop body.
fn fold_body(
    name: &str,
    safe_name: &str,
    head_pc: u64,
    back_pc: u64,
    trip: u64,
    body: &[&PcRecord],
    src: Option<(String, u32, u32)>,
) -> Section {
    let insns: u64 = body.iter().map(|r| r.exec_count).sum();
    // `vsetvl*` is classified scalar, so VECTOR-class records are the real
    // vector ops — their exec count is the number of vector-op executions.
    let vec_insns: u64 = body
        .iter()
        .filter(|r| r.class & class::VECTOR != 0)
        .map(|r| r.exec_count)
        .sum();
    let scalar_insns = insns.saturating_sub(vec_insns);
    // vector_elems lives on the consuming ops; VL histogram on the vsetvl*.
    let vec_elems: u64 = body.iter().map(|r| r.vector_elems).sum();
    // Widest VL the loop ever set (any vsetvl* histogram key) — the observed
    // ceiling. Avoids needing VLEN; a loop that reaches a full-width iteration
    // establishes its own VLMAX.
    let vlmax_obs: u32 = body
        .iter()
        .flat_map(|r| r.vl_hist.keys().copied())
        .max()
        .unwrap_or(0);
    // Already multiplied out by execution count on the record, so a plain sum.
    let mut cache = CacheCounts::default();
    for r in body {
        cache.add(&r.cache);
    }

    let insns_f = insns.max(1) as f64;
    let scalar_frac = scalar_insns as f64 / insns_f;
    let vl_mean = if vec_insns > 0 {
        vec_elems as f64 / vec_insns as f64
    } else {
        0.0
    };
    let vl_util = if vlmax_obs > 0 {
        (vl_mean / vlmax_obs as f64).clamp(0.0, 1.0)
    } else {
        0.0
    };

    // Algo 1: a fully-scalar hot loop is maximally improvable; a vectorized one
    // is penalized by short VL and by scalar overhead.
    let improvable = if vec_insns == 0 {
        1.0
    } else {
        (W_VL * (1.0 - vl_util) + W_SCALAR * scalar_frac).clamp(0.0, 1.0)
    };
    let score = insns_f * improvable;

    // Algo 2: instructions that would vanish reaching IDEAL_VEC_FRAC, plus the
    // slack from vector ops running below full VL.
    let vec_frac = vec_insns as f64 / insns_f;
    let headroom = insns_f * (IDEAL_VEC_FRAC - vec_frac).max(0.0)
        + vec_insns as f64 * (1.0 - vl_util).max(0.0);

    // Algo 4 tier. `mem-heavy` is tested first and wins outright: a loop that
    // misses the L1 on most of its lines is a memory problem whatever its VL
    // utilization says, and `healthy` in particular would be actively
    // misleading advice for it. Nothing below is disturbed by that ordering
    // because `score` and `headroom` are unchanged — the tier is only the
    // explanation badge, so a mem-heavy loop still ranks on its vector badness
    // and the badge tells the reader why that number may not be the whole story.
    // `lines() == 0` for every unmodelled run, so this branch is unreachable
    // there and classification is bit-identical to before cache modelling.
    let tier = if cache.lines() > 0 && cache.l1_misses * 2 > cache.lines() {
        Tier::MemHeavy
    } else if vec_insns == 0 {
        Tier::Scalar
    } else if vl_util < 0.5 {
        Tier::ShortVl
    } else if vl_util >= HEALTHY_VL_UTIL {
        // VL saturated: scalar insns are loop-control overhead, not
        // improvable addressing — healthy even if scalar_frac is high.
        Tier::Healthy
    } else if scalar_frac > 0.5 {
        Tier::AddrBound
    } else {
        Tier::Healthy
    };

    Section {
        func: name.to_string(),
        safe_name: safe_name.to_string(),
        head_pc,
        back_pc,
        src,
        insns,
        trip,
        vec_insns,
        scalar_frac,
        vl_util,
        vl_mean,
        vlmax_obs,
        cache,
        tier,
        score,
        headroom,
    }
}

/// Extra CSS for the hotspots table (tier badges + score bar), on top of the
/// shared report [`STYLE`].
const HOT_STYLE: &str = r#"<style>
.hot td { padding:3px 10px; }
.badge { display:inline-block; padding:0 6px; border-radius:3px; font-size:11px; font-weight:700; color:var(--bg); }
/* mem-heavy needs to read as "worse than scalar" while staying distinct from
   the red→green vectorization ramp, hence a red pushed towards the accent. */
.t-mem { background:color-mix(in srgb, var(--red) 55%, var(--accent)); }
.t-scalar { background:var(--red); }
.t-shortvl { background:var(--orange); }
.t-addr { background:var(--amber); }
.t-healthy { background:var(--green); }
.num { text-align:right; font-variant-numeric:tabular-nums; }
.sbar { display:inline-block; height:9px; border-radius:2px; background:var(--red); opacity:.55; }
.leg { color:var(--dim); margin:.4rem 0 1rem; font-size:12px; }
/* Per-row backgrounds (zebra striping) so a row reads cleanly across all
   columns; a brighter hover band tracks the row under the cursor. */
.hot tbody tr:nth-child(odd) td { background:var(--surface); }
.hot tbody tr:hover td { background:color-mix(in srgb, var(--accent) 16%, var(--surface)); }
</style>"#;

/// Thousands-separated decimal, so the seven-digit line counts a cache-modelled
/// run produces stay readable next to the plain instruction counts.
fn commas(n: u64) -> String {
    let d = n.to_string();
    // nosemgrep: dos-unbounded-memory-allocation -- d is a u64 rendered to at most 20 chars
    let mut out = String::with_capacity(d.len() + d.len() / 3);
    for (i, c) in d.chars().enumerate() {
        if i > 0 && (d.len() - i).is_multiple_of(3) {
            out.push(',');
        }
        out.push(c);
    }
    out
}

/// One cache-level cell: `(hits / misses) 99.9%`, or an em dash when the level
/// was never looked up. That covers both an unmodelled run and an L2 whose L1
/// never missed — neither is a 0 % hit rate, and printing one would read as a
/// pathology that did not happen.
fn cache_cell(hits: u64, misses: u64) -> String {
    let lookups = hits + misses;
    if lookups == 0 {
        return "&mdash;".to_string();
    }
    format!(
        "({} / {}) {:.1}%",
        commas(hits),
        commas(misses),
        hits as f64 / lookups as f64 * 100.0
    )
}

/// Render the `annotate_hotspots.html` page: a legend plus a table of the
/// worst loop sections, each linking to its loop head on the function page.
///
/// The two cache columns and the `mem-heavy` legend line are omitted entirely
/// when no listed section has any modelled cache data, so the table a run
/// without `--cache-sim` produces is exactly the one it produced before cache
/// modelling existed (the unused badge rule in [`HOT_STYLE`] aside).
pub fn render_page(sections: &[Section]) -> String {
    let max_score = sections
        .iter()
        .map(|s| s.score)
        .fold(0.0f64, f64::max)
        .max(1.0);
    let show_cache = sections.iter().any(|s| !s.cache.is_zero());

    let mut b = String::new();
    // nosemgrep: error-handling-ignored-critical-result -- fmt::Write on a String is infallible
    let _ = write!(
        b,
        "<!doctype html><html><head><meta charset=\"utf-8\"><title>annotate: hotspots</title>{}\
         <link rel=\"stylesheet\" href=\"annotate.css\">{}</head><body>",
        THEME_SCRIPT, HOT_STYLE
    );
    b.push_str("<p><a href=\"annotate_index.html\">&larr; all functions</a></p>");
    b.push_str("<h1>\u{1f525} Hotspots &mdash; worst loops</h1>");
    b.push_str(
        "<p class=\"leg\">Ranked by <b>badness</b> = hotness (instructions executed) \u{00d7} \
         improvability (scalar, short vector length, or address-bound). \
         <span class=\"badge t-scalar\">scalar</span> not vectorized &middot; \
         <span class=\"badge t-shortvl\">short-VL</span> VL below the widest set &middot; \
         <span class=\"badge t-addr\">addr-bound</span> mostly scalar address math &middot; \
         <span class=\"badge t-healthy\">healthy</span> leave it alone.</p>",
    );
    if show_cache {
        b.push_str(
            "<p class=\"leg\"><span class=\"badge t-mem\">mem-heavy</span> over half of \
             the loop's cache lines missed the L1 &mdash; outranks the tiers above. \
             The <b>L1</b>/<b>L2</b> columns are modelled \
             <b>(hits / misses)</b> of this loop's <b>vector</b> loads and stores only; \
             scalar accesses and instruction fetch are not captured.</p>",
        );
    }

    b.push_str(
        "<table class=\"hot\"><thead><tr class=\"col-header\">\
         <td>#</td><td>score</td><td></td><td>tier</td><td>function</td><td>source</td>\
         <td class=\"num\">insns</td><td class=\"num\">trips</td>\
         <td class=\"num\">scalar</td><td class=\"num\">VL</td>",
    );
    if show_cache {
        b.push_str("<td class=\"num\">L1</td><td class=\"num\">L2</td>");
    }
    b.push_str("<td class=\"num\">recoverable</td></tr></thead><tbody>");

    for (i, s) in sections.iter().enumerate() {
        let barpx = (s.score / max_score * 80.0).round() as u64;
        let src = match &s.src {
            Some((p, lo, hi)) => {
                let file = p.rsplit('/').next().unwrap_or(p);
                if lo == hi {
                    format!("{}:{}", esc(file), lo)
                } else {
                    format!("{}:{}-{}", esc(file), lo, hi)
                }
            }
            None => "&mdash;".to_string(),
        };
        let vl = if s.vec_insns > 0 && s.vlmax_obs > 0 {
            format!(
                "{:.0}/{} ({:.0}%)",
                s.vl_mean,
                s.vlmax_obs,
                s.vl_util * 100.0
            )
        } else {
            "&mdash;".to_string()
        };
        let cache_cells = if show_cache {
            format!(
                "<td class=\"num\">{}</td><td class=\"num\">{}</td>",
                cache_cell(s.cache.l1_hits, s.cache.l1_misses),
                cache_cell(s.cache.l2_hits, s.cache.l2_misses)
            )
        } else {
            String::new()
        };
        // nosemgrep: error-handling-ignored-critical-result -- fmt::Write on a String is infallible
        let _ = writeln!(
            b,
            "<tr><td class=\"num\">{}</td>\
             <td class=\"num\">{:.0}</td>\
             <td><span class=\"sbar\" style=\"width:{}px\"></span></td>\
             <td><span class=\"badge {}\">{}</span></td>\
             <td><a href=\"annotate_{}.html?region={:x}-{:x}#pc-{:x}\">{}</a></td>\
             <td class=\"pc\">{}</td>\
             <td class=\"num\">{}</td><td class=\"num\">{}</td>\
             <td class=\"num\">{:.0}%</td><td class=\"num\">{}</td>{}\
             <td class=\"num\">{:.0}</td></tr>",
            i + 1,
            s.score,
            barpx,
            s.tier.css(),
            s.tier.label(),
            s.safe_name,
            s.head_pc,
            s.back_pc,
            s.head_pc,
            esc(&s.func),
            src,
            s.insns,
            s.trip,
            s.scalar_frac * 100.0,
            vl,
            cache_cells,
            s.headroom,
        );
    }
    b.push_str("</tbody></table></body></html>");
    b
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rec(pc: u64, exec: u64, class_bits: u8, target: Option<u64>) -> PcRecord {
        PcRecord {
            pc,
            exec_count: exec,
            class: class_bits,
            branch_target: target,
            ..Default::default()
        }
    }

    /// A fully-scalar loop (a back-edge over scalar-only body) is tiered
    /// `scalar`, is maximally improvable (score == insns), and shows no VL.
    #[test]
    fn scalar_loop_is_worst() {
        let recs = [
            rec(0x10, 100, 0, None),
            rec(0x14, 100, 0, None),
            rec(0x18, 100, class::BRANCH, Some(0x10)), // back-edge
        ];
        let refs: Vec<&PcRecord> = recs.iter().collect();
        // resolver is only used for names/lines; build via a helper-free path.
        let secs = build_only(&refs);
        assert_eq!(secs.len(), 1);
        let s = &secs[0];
        assert!(matches!(s.tier, Tier::Scalar));
        assert_eq!(s.vec_insns, 0);
        assert_eq!(s.insns, 300);
        assert!((s.score - 300.0).abs() < 1e-6, "score {}", s.score);
    }

    /// A vector loop running near its widest VL is `healthy` and scores far
    /// below an equally-hot scalar loop.
    #[test]
    fn saturated_vector_loop_is_healthy() {
        // Vector-dominated body: one vsetvl + several wide-VL vector ops + a
        // back-edge, so the scalar fraction stays below 50 %.
        let mut vset = rec(0x0c, 100, 0, None);
        vset.vl_hist.insert(128, 100); // widest VL set = 128
        let mut recs = vec![vset];
        for k in 0..6 {
            let mut v = rec(0x10 + k * 4, 100, class::VECTOR, None);
            v.vector_elems = 100 * 128; // mean VL 128
            recs.push(v);
        }
        recs.push(rec(0x30, 100, class::BRANCH, Some(0x0c))); // back-edge
        let refs: Vec<&PcRecord> = recs.iter().collect();
        let secs = build_only(&refs);
        assert_eq!(secs.len(), 1);
        let s = &secs[0];
        assert!(s.vl_util > 0.9, "vl_util {}", s.vl_util);
        assert!(matches!(s.tier, Tier::Healthy));
        assert!(s.score < 100.0, "healthy loop scored too high: {}", s.score);
    }

    /// A vector loop stuck at a short VL is tiered `short-VL` and scores high
    /// despite being vectorized.
    #[test]
    fn short_vl_loop_flagged() {
        let mut v = rec(0x10, 100, class::VECTOR, None);
        v.vector_elems = 100 * 10; // mean VL 10
        let mut vset = rec(0x0c, 100, 0, None);
        vset.vl_hist.insert(128, 100); // could do 128
        let recs = [vset, v, rec(0x14, 100, class::BRANCH, Some(0x0c))];
        let refs: Vec<&PcRecord> = recs.iter().collect();
        let s = &build_only(&refs)[0];
        assert!(matches!(s.tier, Tier::ShortVl));
        assert!(s.vl_util < 0.1, "vl_util {}", s.vl_util);
    }

    /// The hotspots link carries the `?region=head-back` query (both edges of
    /// the loop) plus the `#pc-head` anchor, so the function page can highlight
    /// the region's start and end.
    #[test]
    fn link_carries_region_range() {
        let recs = [
            rec(0x10, 5, 0, None),
            rec(0x18, 5, class::BRANCH, Some(0x10)), // head 0x10, back 0x18
        ];
        let refs: Vec<&PcRecord> = recs.iter().collect();
        let secs = build_only(&refs);
        let html = render_page(&secs);
        assert!(
            html.contains("?region=10-18#pc-10"),
            "expected region link, got: {html}"
        );
    }

    /// A loop with no back-edge yields no section.
    #[test]
    fn no_backedge_no_section() {
        let recs = [rec(0x10, 5, 0, None), rec(0x14, 5, 0, None)];
        let refs: Vec<&PcRecord> = recs.iter().collect();
        assert!(build_only(&refs).is_empty());
    }

    /// A body whose vector accesses missed the L1 on most of their lines is
    /// `mem-heavy` even though every other signal says `healthy` — that
    /// precedence is the whole point of the tier.
    #[test]
    fn mem_heavy_outranks_healthy() {
        let mut vset = rec(0x0c, 100, 0, None);
        vset.vl_hist.insert(128, 100);
        let mut recs = vec![vset];
        for k in 0..6 {
            let mut v = rec(0x10 + k * 4, 100, class::VECTOR, None);
            v.vector_elems = 100 * 128; // saturated VL
            v.cache = CacheCounts {
                l1_hits: 100,
                l1_misses: 300, // 75 % miss
                l2_hits: 200,
                l2_misses: 100,
            };
            recs.push(v);
        }
        recs.push(rec(0x30, 100, class::BRANCH, Some(0x0c)));
        let refs: Vec<&PcRecord> = recs.iter().collect();
        let s = &build_only(&refs)[0];
        assert!(s.vl_util > 0.9, "vl_util {}", s.vl_util);
        assert!(matches!(s.tier, Tier::MemHeavy), "tier {}", s.tier.label());
    }

    /// The threshold is strictly greater than half: an even hit/miss split is
    /// not yet a memory problem, so the section keeps its vectorization tier.
    #[test]
    fn exactly_half_l1_misses_is_not_mem_heavy() {
        let mut v = rec(0x10, 100, class::VECTOR, None);
        v.vector_elems = 100 * 128;
        v.cache = CacheCounts {
            l1_hits: 500,
            l1_misses: 500,
            l2_hits: 500,
            l2_misses: 0,
        };
        let mut vset = rec(0x0c, 100, 0, None);
        vset.vl_hist.insert(128, 100);
        let recs = [vset, v, rec(0x14, 100, class::BRANCH, Some(0x0c))];
        let refs: Vec<&PcRecord> = recs.iter().collect();
        let s = &build_only(&refs)[0];
        assert_eq!(s.cache.l1_misses * 2, s.cache.lines(), "test setup");
        assert!(!matches!(s.tier, Tier::MemHeavy), "tier {}", s.tier.label());
    }

    /// A run without `--cache-sim` leaves every `PcRecord::cache` zero, so both
    /// the classification and the rendered table must be exactly as they were
    /// before cache modelling existed: no cache columns, no `mem-heavy` legend.
    #[test]
    fn no_cache_data_changes_nothing() {
        let mut vset = rec(0x0c, 100, 0, None);
        vset.vl_hist.insert(128, 100);
        let mut v = rec(0x10, 100, class::VECTOR, None);
        v.vector_elems = 100 * 128;
        let recs = [vset, v, rec(0x14, 100, class::BRANCH, Some(0x0c))];
        let refs: Vec<&PcRecord> = recs.iter().collect();
        let secs = build_only(&refs);
        let s = &secs[0];
        assert!(s.cache.is_zero());
        assert_eq!(s.cache.lines(), 0);
        assert!(matches!(s.tier, Tier::Healthy), "tier {}", s.tier.label());

        let html = render_page(&secs);
        assert!(!html.contains("badge t-mem"), "cache legend leaked: {html}");
        assert!(!html.contains(">L1<"), "cache column leaked: {html}");
    }

    /// The section aggregate is a plain sum over its body's records (each of
    /// which already covers all of that PC's executions).
    #[test]
    fn cache_aggregate_sums_over_body() {
        let mut a = rec(0x10, 10, class::VECTOR, None);
        a.cache = CacheCounts {
            l1_hits: 7,
            l1_misses: 3,
            l2_hits: 2,
            l2_misses: 1,
        };
        let mut c = rec(0x14, 10, class::VECTOR, None);
        c.cache = CacheCounts {
            l1_hits: 1,
            l1_misses: 5,
            l2_hits: 0,
            l2_misses: 5,
        };
        let recs = [a, c, rec(0x18, 10, class::BRANCH, Some(0x10))];
        let refs: Vec<&PcRecord> = recs.iter().collect();
        let s = &build_only(&refs)[0];
        assert_eq!(
            s.cache,
            CacheCounts {
                l1_hits: 8,
                l1_misses: 8,
                l2_hits: 2,
                l2_misses: 6,
            }
        );
        assert_eq!(s.cache.lines(), 16);
    }

    /// Test helper: run section detection + scoring without needing an
    /// `ElfResolver` by borrowing the detection logic and passing an empty
    /// source resolution. Mirrors `sections_for_function` minus the resolver
    /// name/line lookups.
    fn build_only(recs: &[&PcRecord]) -> Vec<Section> {
        let pcs: std::collections::HashSet<u64> = recs.iter().map(|r| r.pc).collect();
        let mut backedges: Vec<(u64, u64, u64)> = Vec::new();
        for r in recs {
            if let Some(t) = r.branch_target {
                if t < r.pc && pcs.contains(&t) {
                    backedges.push((t, r.pc, r.exec_count));
                }
            }
        }
        let mut buckets: HashMap<usize, Vec<&PcRecord>> = HashMap::new();
        for r in recs {
            let mut best: Option<(usize, u64)> = None;
            for (i, &(h, b, _)) in backedges.iter().enumerate() {
                if r.pc >= h && r.pc <= b {
                    let width = b - h;
                    if best.is_none_or(|(_, w)| width < w) {
                        best = Some((i, width));
                    }
                }
            }
            if let Some((i, _)) = best {
                buckets.entry(i).or_default().push(r);
            }
        }
        let mut out = Vec::new();
        for (i, body) in buckets {
            let (head, back, trip) = backedges[i];
            out.push(build_section_no_src(head, back, trip, &body));
        }
        out
    }

    /// [`build_section`] with `src = None` (no resolver in unit tests). Delegates
    /// to the real [`fold_body`], so the tiering these tests pin is the shipped
    /// one rather than a copy that can drift.
    fn build_section_no_src(head_pc: u64, back_pc: u64, trip: u64, body: &[&PcRecord]) -> Section {
        fold_body("", "", head_pc, back_pc, trip, body, None)
    }
}
