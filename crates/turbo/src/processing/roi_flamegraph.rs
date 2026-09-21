//! ROI flamegraph construction and rendering.
//!
//! [`StreamingRoiFlamegraph`] is the live path: it aggregates a call-tree of
//! [`RoiFlameNode`]s as regions enter and exit, so its memory is bounded by the
//! number of unique call paths rather than by the number of executions, and it
//! costs one integer-keyed hash probe per region entry plus two integer adds per
//! region exit.
//!
//! [`build_flamegraph`] is the older offline path, folding a flat
//! `Vec<RoiExecution>` (one entry per dynamic execution) into the same tree. It
//! is no longer on the production path -- the enricher streams instead -- and is
//! kept for tests and for callers holding a recorded execution list.

use std::collections::HashMap;

use serde::Serialize;

use crate::common::RoiExecution;
use crate::processing::enricher::annotate::THEME_SCRIPT;

/// Default hard ceiling for unique call paths in [`StreamingRoiFlamegraph`].
pub const DEFAULT_MAX_FLAME_NODES: usize = 10_000;

/// Index of a call-path node in [`StreamingRoiFlamegraph`]'s arena.
///
/// The arena holds only *real* regions; [`FLAME_ROOT`] stands in for the
/// synthetic "Global" root that [`StreamingRoiFlamegraph::finish`] materialises,
/// so a top-level region's parent is `FLAME_ROOT` rather than a stored node.
pub type FlameNodeIdx = u32;

/// Parent index meaning "child of the synthetic Global root", i.e. no enclosing
/// region. Also the index an [`Enricher`](crate::processing::enricher::Enricher)
/// carries for a region whose flamegraph tracking is switched off, which makes
/// [`StreamingRoiFlamegraph::accumulate`] a no-op for it.
pub const FLAME_ROOT: FlameNodeIdx = u32::MAX;

/// One node of the live call tree: a distinct `(call path, region)` pair.
#[derive(Debug, Clone)]
struct FlameNode {
    parent: FlameNodeIdx,
    region_id: usize,
    /// Written once, when the node is created -- never on the hot path.
    name: String,
    inclusive_instructions: u64,
    call_count: u64,
}

/// Online streaming builder for ROI flamegraphs.
///
/// Aggregates live as regions enter and exit, so memory is bounded by the number
/// of unique *call paths*, not by the number of executions.
///
/// The hot path is keyed on `(parent node index, region_id)` packed into a `u64`
/// rather than on region names, which is what keeps it allocation- and
/// string-hash-free: `region_id` is already a stable, dense, name-interned
/// integer (see `MarkerHandler`'s `region_name_ids`), so two entries of the same
/// named region always share an id and two different names never do. The node
/// index for a region is resolved **once, when the region is entered**
/// ([`enter`](Self::enter)) and carried on the enricher's open-region frame, so
/// the exit path ([`accumulate`](Self::accumulate)) is two integer adds.
#[derive(Debug, Clone)]
pub struct StreamingRoiFlamegraph {
    /// `(parent_idx, region_id)` packed into one `u64` -> node index. Integer
    /// key with `FxHash`: no string hashing while the trace is being enriched.
    children: rustc_hash::FxHashMap<u64, FlameNodeIdx>,
    nodes: Vec<FlameNode>,
    /// Name-keyed sibling index, populated *only* by [`merge`](Self::merge).
    /// See that method for why the merge cannot reuse `children`.
    merge_children: HashMap<(FlameNodeIdx, String), FlameNodeIdx>,
    /// Number of completed executions folded in. Distinct from `nodes.len()`:
    /// a region that is entered but never closed creates a node without ever
    /// producing an execution, and a report of nothing but the Global root is
    /// not worth writing.
    executions: u64,
    /// Hard ceiling on unique call paths, so a pathological workload (e.g.
    /// unbounded dynamic region names) cannot grow the arena without limit.
    max_nodes: usize,
}

impl Default for StreamingRoiFlamegraph {
    fn default() -> Self {
        Self::new(DEFAULT_MAX_FLAME_NODES)
    }
}

/// Pack a sibling lookup key. `region_id` is allocated densely from 0 by
/// `MarkerHandler`, so truncating it to 32 bits is safe for any real trace.
#[inline(always)]
fn child_key(parent: FlameNodeIdx, region_id: usize) -> u64 {
    ((parent as u64) << 32) | (region_id as u32 as u64)
}

impl StreamingRoiFlamegraph {
    pub fn new(max_nodes: usize) -> Self {
        Self {
            children: rustc_hash::FxHashMap::default(),
            nodes: Vec::new(),
            merge_children: HashMap::new(),
            executions: 0,
            max_nodes,
        }
    }

    /// Whether any call-path node has been created.
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Number of unique call-path nodes -- the quantity this builder bounds.
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Number of completed region executions folded in. This, not
    /// [`node_count`](Self::node_count), is the right gate for emitting a
    /// report: see the `executions` field.
    pub fn executions(&self) -> u64 {
        self.executions
    }

    /// Whether any region actually completed, i.e. whether there is anything to
    /// report.
    pub fn has_executions(&self) -> bool {
        self.executions > 0
    }

    /// Resolve (creating on first sight) the call-path node for `region_id`
    /// entered beneath `parent`. Called once per region *entry*: one
    /// integer-keyed hash probe, and no allocation once the node exists.
    ///
    /// On hitting `max_nodes` the region folds into its parent rather than
    /// minting a new node, so overflow costs accuracy but never memory, and
    /// every returned index stays a valid, already-registered node.
    #[inline]
    pub fn enter(&mut self, parent: FlameNodeIdx, region_id: usize, name: &str) -> FlameNodeIdx {
        let key = child_key(parent, region_id);
        if let Some(&idx) = self.children.get(&key) {
            return idx;
        }
        if self.nodes.len() >= self.max_nodes {
            return parent;
        }
        let idx = self.nodes.len() as FlameNodeIdx;
        self.nodes.push(FlameNode {
            parent,
            region_id,
            name: name.to_string(),
            inclusive_instructions: 0,
            call_count: 0,
        });
        self.children.insert(key, idx);
        idx
    }

    /// Fold one completed execution into `node`, whose index came from the
    /// matching [`enter`](Self::enter). Called once per region *exit*.
    ///
    /// A `node` of [`FLAME_ROOT`] (or any out-of-range index) is ignored, which
    /// is how a run with flamegraph tracking disabled costs nothing here.
    #[inline]
    pub fn accumulate(&mut self, node: FlameNodeIdx, duration_instr: u64) {
        if let Some(n) = self.nodes.get_mut(node as usize) {
            n.inclusive_instructions += duration_instr;
            n.call_count += 1;
            self.executions += 1;
        }
    }

    /// Merge another builder (one per hart) into this one.
    ///
    /// `region_id`s are allocated by each `Enricher` independently, i.e. per
    /// hart, so they are **not** comparable across builders: hart 0's id 0 and
    /// hart 1's id 0 are usually different regions, and an id-keyed merge would
    /// collapse them into one node. The merge therefore re-keys `other`'s nodes
    /// by *name*, which is the identity the per-hart interner itself uses. It
    /// runs once per hart at end of run, off any hot path, so a `String`-keyed
    /// probe here is free -- the live path stays integer-keyed.
    pub fn merge(&mut self, other: StreamingRoiFlamegraph) {
        // `enter`/`enter_by_name` only ever append, so a node's parent always
        // has a strictly lower index than the node itself. Walking `other` in
        // index order therefore always finds the parent's remapping already
        // recorded.
        // nosemgrep: dos-unbounded-memory-allocation -- one u32 per node of a Vec already resident and owned by value
        let mut remap: Vec<FlameNodeIdx> = Vec::with_capacity(other.nodes.len());
        for n in other.nodes.iter() {
            let parent = if n.parent == FLAME_ROOT {
                FLAME_ROOT
            } else {
                remap[n.parent as usize]
            };
            let idx = self.enter_by_name(parent, n.region_id, &n.name);
            remap.push(idx);
            if let Some(dst) = self.nodes.get_mut(idx as usize) {
                dst.inclusive_instructions += n.inclusive_instructions;
                dst.call_count += n.call_count;
            }
        }
        self.executions += other.executions;
    }

    /// Name-keyed sibling lookup, for the cold [`merge`](Self::merge) path only.
    fn enter_by_name(
        &mut self,
        parent: FlameNodeIdx,
        region_id: usize,
        name: &str,
    ) -> FlameNodeIdx {
        if let Some(&idx) = self.merge_children.get(&(parent, name.to_string())) {
            return idx;
        }
        if self.nodes.len() >= self.max_nodes {
            return parent;
        }
        let idx = self.nodes.len() as FlameNodeIdx;
        self.nodes.push(FlameNode {
            parent,
            region_id,
            name: name.to_string(),
            inclusive_instructions: 0,
            call_count: 0,
        });
        self.merge_children.insert((parent, name.to_string()), idx);
        idx
    }

    /// Materialise the finalised [`RoiFlameNode`] tree under a synthetic
    /// "Global" root whose inclusive count is `global_instructions`, so work
    /// outside any region shows up as root self-time.
    pub fn finish(&self, global_instructions: u64) -> RoiFlameNode {
        // Each node stores its parent, so the child lists come out of one
        // linear pass rather than a map keyed by parent.
        let mut kids: Vec<Vec<FlameNodeIdx>> = vec![Vec::new(); self.nodes.len()];
        let mut roots: Vec<FlameNodeIdx> = Vec::new();
        for (i, n) in self.nodes.iter().enumerate() {
            if n.parent == FLAME_ROOT {
                roots.push(i as FlameNodeIdx);
            } else {
                kids[n.parent as usize].push(i as FlameNodeIdx);
            }
        }

        fn build(
            idx: FlameNodeIdx,
            nodes: &[FlameNode],
            kids: &[Vec<FlameNodeIdx>],
        ) -> RoiFlameNode {
            let n = &nodes[idx as usize];
            let mut children: Vec<RoiFlameNode> = kids[idx as usize]
                .iter()
                .map(|&c| build(c, nodes, kids))
                .collect();
            children.sort_by(|a, b| a.name.cmp(&b.name));
            let children_inclusive: u64 = children.iter().map(|c| c.inclusive_instructions).sum();
            RoiFlameNode {
                name: n.name.clone(),
                region_id: n.region_id,
                inclusive_instructions: n.inclusive_instructions,
                self_instructions: n.inclusive_instructions.saturating_sub(children_inclusive),
                call_count: n.call_count,
                children,
            }
        }

        let mut root_children: Vec<RoiFlameNode> = roots
            .iter()
            .map(|&r| build(r, &self.nodes, &kids))
            .collect();
        root_children.sort_by(|a, b| a.name.cmp(&b.name));

        let children_inclusive: u64 = root_children.iter().map(|c| c.inclusive_instructions).sum();
        RoiFlameNode {
            name: "Global".to_string(),
            region_id: GLOBAL_REGION_ID,
            inclusive_instructions: global_instructions,
            self_instructions: global_instructions.saturating_sub(children_inclusive),
            call_count: 1,
            children: root_children,
        }
    }
}

/// A single node in the ROI call tree.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct RoiFlameNode {
    pub name: String,
    pub region_id: usize,
    /// Sum of `end_instr - begin_instr` over all executions merged into this node.
    pub inclusive_instructions: u64,
    /// `inclusive_instructions` minus the sum of children's inclusive instructions.
    pub self_instructions: u64,
    /// Number of dynamic executions merged into this node.
    pub call_count: u64,
    pub children: Vec<RoiFlameNode>,
}

impl RoiFlameNode {
    /// Number of nodes in this subtree, root included -- the count of distinct
    /// call paths the tree represents. Used to label a per-thread graph in
    /// `roi_flamegraph_harts.json`; not on any hot path.
    pub fn node_count(&self) -> usize {
        1 + self.children.iter().map(Self::node_count).sum::<usize>()
    }
}

/// Sentinel `region_id` used for the synthetic root "Global" node.
pub const GLOBAL_REGION_ID: usize = usize::MAX;

/// One entry of `roi_flamegraph_harts.json`: the index a viewer builds its
/// per-thread selector from, so the file-name pattern lives here rather than
/// being reconstructed by every consumer.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct HartFlamegraphEntry {
    pub hart_id: u32,
    /// Name of that hart's graph JSON, relative to the work directory.
    pub json: String,
    /// The hart's own retired-instruction count, i.e. its root's inclusive
    /// value. Every thread that retired anything is listed, so these sum to
    /// the merged graph's root -- `Global` is additive across harts.
    pub instructions: u64,
    /// Retired on this hart outside any region. For a worker thread this is
    /// the barrier/spin tail -- a quantity the merged graph can only expose by
    /// subtraction.
    pub self_instructions: u64,
    /// Distinct call paths in that hart's tree, root included.
    pub call_paths: usize,
}

/// Index of the per-thread graphs written beside the merged one, serialised to
/// `roi_flamegraph_harts.json`.
///
/// Covers every thread that retired an instruction, including one that never
/// entered a region: such a thread's graph is a bare root, but its self-time
/// is exactly the work the region-based view cannot see, and it is often most
/// of the run.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct HartFlamegraphIndex {
    /// Sorted by `hart_id`, so a selector's order is stable across runs --
    /// the enricher map this is built from is a `HashMap`.
    pub harts: Vec<HartFlamegraphEntry>,
}

/// The graph JSON file name for one hart. The single place the pattern is
/// spelled: `HartFlamegraphEntry::json` carries it to consumers.
pub fn hart_json_name(hart_id: u32) -> String {
    format!("roi_flamegraph_hart{hart_id}.json")
}

/// The folded-stacks file name for one hart, the `flamegraph.pl`-facing
/// companion to [`hart_json_name`].
pub fn hart_folded_name(hart_id: u32) -> String {
    format!("roi_flamegraph_hart{hart_id}.folded")
}

/// Build a merged multi-hart ROI flamegraph from a flat list of executions.
///
/// `global_instructions` is the total retired-instruction count (e.g.
/// `global_roi().instructions`), used as the inclusive count of the synthetic
/// root "Global" node so that work outside any region shows up as root
/// self-time.
pub fn build_flamegraph(executions: &[RoiExecution], global_instructions: u64) -> RoiFlameNode {
    // Work on a stable, begin_instr-ordered copy so that, if executions ever
    // arrive out of order (e.g. merged from multiple harts), sibling merge
    // order is still deterministic.
    let mut sorted: Vec<&RoiExecution> = executions.iter().collect();
    sorted.sort_by_key(|e| e.begin_instr);

    // Key identifying a merged node: (parent NodeKey, name). Using name (not
    // region_id) as part of the key is what collapses repeated dynamic
    // executions of the same region into one node with call_count > 1, per
    // the plan's `repeated.c` case.
    #[derive(PartialEq, Eq, Hash, Clone)]
    struct NodeKey {
        parent: Option<Box<NodeKey>>,
        name: String,
    }

    // Per merged-node accumulated state.
    struct Accum {
        region_id: usize,
        inclusive_instructions: u64,
        call_count: u64,
    }

    let mut accum: HashMap<NodeKey, Accum> = HashMap::new();
    // Map region_id -> NodeKey, so children can look up which merged-node key
    // their parent_region_id resolved to. Executions are processed in
    // begin_instr order, so a parent's execution is always folded in before
    // any of its children's (a child cannot begin before its parent begins).
    let mut region_id_to_key: HashMap<usize, NodeKey> = HashMap::new();
    // Track parent-of-key edges: child NodeKey -> parent NodeKey (None = root).
    let mut parent_of: HashMap<NodeKey, Option<NodeKey>> = HashMap::new();

    for exec in &sorted {
        // Resolve the parent's NodeKey: if this execution's parent_region_id
        // is Some(id), look up which merged node that id currently maps to;
        // if it hasn't been seen yet (shouldn't happen, but guard anyway)
        // treat as root.
        let parent_key: Option<NodeKey> = exec
            .parent_region_id
            .and_then(|pid| region_id_to_key.get(&pid).cloned());

        let key = NodeKey {
            parent: parent_key.clone().map(Box::new),
            name: exec.name.clone(),
        };

        let inclusive = exec.end_instr.saturating_sub(exec.begin_instr);
        let entry = accum.entry(key.clone()).or_insert(Accum {
            region_id: exec.region_id,
            inclusive_instructions: 0,
            call_count: 0,
        });
        entry.inclusive_instructions += inclusive;
        entry.call_count += 1;

        region_id_to_key.insert(exec.region_id, key.clone());
        parent_of.entry(key).or_insert(parent_key);
    }

    // Materialize the tree by grouping children under each parent key.
    let mut children_of: HashMap<Option<NodeKey>, Vec<NodeKey>> = HashMap::new();
    for (key, parent) in &parent_of {
        children_of
            .entry(parent.clone())
            .or_default()
            .push(key.clone());
    }

    fn build_node(
        key: &NodeKey,
        accum: &HashMap<NodeKey, Accum>,
        children_of: &HashMap<Option<NodeKey>, Vec<NodeKey>>,
    ) -> RoiFlameNode {
        let a = &accum[key];
        let mut children: Vec<RoiFlameNode> = children_of
            .get(&Some(key.clone()))
            .into_iter()
            .flatten()
            .map(|ck| build_node(ck, accum, children_of))
            .collect();
        // Deterministic ordering for tests/output: by name.
        children.sort_by(|a, b| a.name.cmp(&b.name));

        let children_inclusive: u64 = children.iter().map(|c| c.inclusive_instructions).sum();
        RoiFlameNode {
            name: key.name.clone(),
            region_id: a.region_id,
            inclusive_instructions: a.inclusive_instructions,
            self_instructions: a.inclusive_instructions.saturating_sub(children_inclusive),
            call_count: a.call_count,
            children,
        }
    }

    let mut root_children: Vec<RoiFlameNode> = children_of
        .get(&None)
        .into_iter()
        .flatten()
        .map(|k| build_node(k, &accum, &children_of))
        .collect();
    root_children.sort_by(|a, b| a.name.cmp(&b.name));

    let children_inclusive: u64 = root_children.iter().map(|c| c.inclusive_instructions).sum();
    RoiFlameNode {
        name: "Global".to_string(),
        region_id: GLOBAL_REGION_ID,
        inclusive_instructions: global_instructions,
        self_instructions: global_instructions.saturating_sub(children_inclusive),
        call_count: 1,
        children: root_children,
    }
}

/// Render the tree as Brendan Gregg `flamegraph.pl`-style folded stacks: one
/// line per node with `self_instructions > 0`, formatted as
/// `Global;region_1;region_2 <self_instructions>`.
pub fn to_folded_stacks(root: &RoiFlameNode) -> String {
    let mut out = String::new();
    let mut stack: Vec<&str> = Vec::new();
    fold_node(root, &mut stack, &mut out);
    out
}

fn fold_node<'a>(node: &'a RoiFlameNode, stack: &mut Vec<&'a str>, out: &mut String) {
    stack.push(&node.name);
    if node.self_instructions > 0 {
        out.push_str(&stack.join(";"));
        out.push(' ');
        out.push_str(&node.self_instructions.to_string());
        out.push('\n');
    }
    for child in &node.children {
        fold_node(child, stack, out);
    }
    stack.pop();
}

/// Thin JSON-serialization view of [`RoiFlameNode`], matching the
/// d3-flame-graph / Perfetto-compatible `{name, value, children}` schema.
#[derive(Debug, Serialize)]
struct JsonFlameNode {
    name: String,
    region_id: usize,
    /// Inclusive cost (self + descendants), used as the frame's `value`.
    value: u64,
    self_value: u64,
    call_count: u64,
    /// Position of this node in the tree, as a dotted child-index path
    /// (`""` for the root, `"0.2"` for the third child of the first child).
    ///
    /// `region_id` cannot serve as the identity a rendered frame is looked up
    /// by: repeated executions of the same region merge into several distinct
    /// nodes under different parents but keep one `region_id`.
    key: String,
    children: Vec<JsonFlameNode>,
}

impl From<&RoiFlameNode> for JsonFlameNode {
    fn from(n: &RoiFlameNode) -> Self {
        JsonFlameNode::from_node(n, String::new())
    }
}

impl JsonFlameNode {
    fn from_node(n: &RoiFlameNode, key: String) -> Self {
        JsonFlameNode {
            name: n.name.clone(),
            region_id: n.region_id,
            value: n.inclusive_instructions,
            self_value: n.self_instructions,
            call_count: n.call_count,
            children: n
                .children
                .iter()
                .enumerate()
                .map(|(i, c)| {
                    let child_key = if key.is_empty() {
                        i.to_string()
                    } else {
                        format!("{key}.{i}")
                    };
                    JsonFlameNode::from_node(c, child_key)
                })
                .collect(),
            key,
        }
    }
}

/// Serialize the tree to a pretty-printed JSON string, suitable for writing
/// to `roi_flamegraph.json`.
pub fn to_json(root: &RoiFlameNode) -> serde_json::Result<String> {
    serde_json::to_string_pretty(&JsonFlameNode::from(root))
}

/// Serialize the tree to a compact JSON string safe to inline inside a
/// `<script>` element.
///
/// `</` is the only sequence an HTML parser looks for while scanning a script
/// body, so escaping it (JSON `\/` is a legal escape for `/`) is enough to make
/// any region name — including one containing a literal `</script>` — inert.
fn to_embeddable_json(root: &RoiFlameNode) -> serde_json::Result<String> {
    let json = serde_json::to_string(&JsonFlameNode::from(root))?;
    Ok(json.replace("</", "<\\/"))
}

/// The widget's stylesheet and behaviour, read from the assets directory at
/// runtime (see [`crate::assets`]) and inlined into the rendered HTML.
///
/// Inlined rather than linked because the fragment has to keep working wherever
/// it is embedded and wherever the page is later copied to; reading it at
/// runtime rather than `include_str!`ing it keeps the CSS and JS editable as
/// their own languages, and tweakable without a rebuild.
fn flamegraph_assets() -> String {
    format!(
        "<style>{}</style><script>{}</script>",
        crate::assets::read("flamegraph.css"),
        crate::assets::read("flamegraph.js"),
    )
}

/// Render the tree as an embeddable HTML fragment: one `<div id="{id}">`
/// carrying the graph, its inline data, and the styles/behaviour it needs.
///
/// The fragment is self-contained and position-independent — it can be dropped
/// into any page or `<div>`, and several fragments with distinct `id`s can
/// coexist on one page. It brings no external files and no dependency on the
/// host page's CSS, but picks up `oc.css` custom properties when they are
/// present so it themes with the rest of the UI.
///
/// `id` must be a valid HTML id (it is also used to derive the data element's
/// id); callers pass something like `roi-flamegraph`.
pub fn to_html_fragment(root: &RoiFlameNode, id: &str) -> serde_json::Result<String> {
    let data = to_embeddable_json(root)?;
    Ok(format!(
        "{assets}\
         <div id=\"{id}\" class=\"ocfg\">\
           <div class=\"ocfg-bar\">\
             <span class=\"ocfg-title\">ROI Flamegraph</span>\
             <span class=\"ocfg-note\"></span>\
             <input class=\"ocfg-search\" type=\"search\" placeholder=\"Search regions\" \
                    aria-label=\"Search regions\">\
             <button class=\"ocfg-btn\" type=\"button\">Reset zoom</button>\
           </div>\
           <div class=\"ocfg-frames\"></div>\
           <div class=\"ocfg-tip\"></div>\
         </div>\
         <script type=\"application/json\" id=\"{id}-data\">{data}</script>\
         <script>window.ocfgInit({id_json});</script>",
        assets = flamegraph_assets(),
        id = id,
        data = data,
        id_json = serde_json::to_string(id)?,
    ))
}

/// Render the tree as a standalone interactive HTML page, suitable for writing
/// to `roi_flamegraph.html`.
///
/// Thin wrapper: the page is just [`to_html_fragment`] plus the shared theme
/// bootstrap and `flamegraph_theme.css` (the `oc.css` variables the widget
/// reads, which a lone file on disk has no stylesheet to inherit), so what is
/// written to disk and what gets embedded elsewhere are the same widget.
pub fn to_html(root: &RoiFlameNode) -> serde_json::Result<String> {
    Ok(format!(
        "<!doctype html><html><head><meta charset=\"utf-8\">\
         <meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">\
         <title>ROI Flamegraph</title>{theme}<style>{vars}</style>\
         </head><body>{fragment}</body></html>",
        theme = THEME_SCRIPT,
        vars = crate::assets::read("flamegraph_theme.css"),
        fragment = to_html_fragment(root, "roi-flamegraph")?,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::processing::RoiMarker;

    fn exec(
        name: &str,
        region_id: usize,
        parent_region_id: Option<usize>,
        depth: u32,
        begin_instr: u64,
        end_instr: u64,
    ) -> RoiExecution {
        RoiExecution {
            name: name.to_string(),
            region_id,
            begin_marker_ordinal: 0,
            begin_pc: 0,
            begin_marker_kind: RoiMarker::BeginRegion,
            end_marker_ordinal: 0,
            end_pc: 0,
            end_marker_kind: RoiMarker::EndRegion,
            hart_id: 0,
            begin_instr,
            end_instr,
            parent_region_id,
            depth,
        }
    }

    /// nested.S: region_1 [0,8) wraps region_2 [begin,end) with 2 instructions
    /// of nested work; region_1 inclusive=8, self=6; region_2 inclusive=2, self=2.
    #[test]
    fn nested_regions() {
        let execs = vec![
            exec("region_1", 1, None, 0, 0, 8),
            exec("region_2", 2, Some(1), 1, 3, 5),
        ];
        let root = build_flamegraph(&execs, 8);

        assert_eq!(root.name, "Global");
        assert_eq!(root.inclusive_instructions, 8);
        assert_eq!(root.children.len(), 1);

        let r1 = &root.children[0];
        assert_eq!(r1.name, "region_1");
        assert_eq!(r1.inclusive_instructions, 8);
        assert_eq!(r1.self_instructions, 6);
        assert_eq!(r1.call_count, 1);
        assert_eq!(r1.children.len(), 1);

        let r2 = &r1.children[0];
        assert_eq!(r2.name, "region_2");
        assert_eq!(r2.inclusive_instructions, 2);
        assert_eq!(r2.self_instructions, 2);
        assert_eq!(r2.call_count, 1);
        assert!(r2.children.is_empty());

        // Root self-time = instructions outside any region.
        assert_eq!(root.self_instructions, 0);
    }

    /// The streaming builder's whole point: a hot loop entered a million times
    /// must cost the same memory as one entered once. Two unique call paths
    /// (`region_1` and `region_1;region_2`) means two nodes, not two million
    /// buffered executions.
    #[test]
    fn streaming_bounded_by_unique_call_paths() {
        let mut builder = StreamingRoiFlamegraph::new(10_000);
        const ITERATIONS: u64 = 1_000_000;

        // region_1 wraps region_2, so region_2's parent is region_1's node.
        let r1 = builder.enter(FLAME_ROOT, 1, "region_1");
        let r2 = builder.enter(r1, 2, "region_2");
        for _ in 0..ITERATIONS {
            // Re-entering resolves to the same nodes without allocating.
            assert_eq!(builder.enter(FLAME_ROOT, 1, "region_1"), r1);
            assert_eq!(builder.enter(r1, 2, "region_2"), r2);
            builder.accumulate(r2, 50);
            builder.accumulate(r1, 100);
        }

        assert_eq!(builder.node_count(), 2);
        assert_eq!(builder.executions(), ITERATIONS * 2);

        let root = builder.finish(100_000_000);
        assert_eq!(root.name, "Global");
        assert_eq!(root.children.len(), 1);

        let n1 = &root.children[0];
        assert_eq!(n1.name, "region_1");
        assert_eq!(n1.region_id, 1);
        assert_eq!(n1.call_count, ITERATIONS);
        assert_eq!(n1.inclusive_instructions, 100_000_000);
        assert_eq!(n1.children.len(), 1);

        let n2 = &n1.children[0];
        assert_eq!(n2.name, "region_2");
        assert_eq!(n2.region_id, 2);
        assert_eq!(n2.call_count, ITERATIONS);
        assert_eq!(n2.inclusive_instructions, 50_000_000);
    }

    /// Every node carries the `region_id` of the region it represents, whether
    /// it was first seen as a leaf or as somebody's parent. The previous
    /// name-keyed builder created parent nodes with a hardcoded `region_id: 0`,
    /// so every interior node of a nested flamegraph reported region 0.
    #[test]
    fn streaming_interior_nodes_keep_their_region_id() {
        let mut builder = StreamingRoiFlamegraph::default();
        let outer = builder.enter(FLAME_ROOT, 7, "outer");
        let inner = builder.enter(outer, 9, "inner");
        builder.accumulate(inner, 5);
        builder.accumulate(outer, 20);

        let root = builder.finish(20);
        let o = &root.children[0];
        assert_eq!((o.name.as_str(), o.region_id), ("outer", 7));
        let i = &o.children[0];
        assert_eq!((i.name.as_str(), i.region_id), ("inner", 9));
    }

    /// Same region name under two different parents is two distinct call paths,
    /// which is the whole reason the trie is keyed on `(parent, region_id)`
    /// rather than on `region_id` alone.
    #[test]
    fn streaming_same_region_under_two_parents_is_two_nodes() {
        let mut builder = StreamingRoiFlamegraph::default();
        let a = builder.enter(FLAME_ROOT, 0, "a");
        let b = builder.enter(FLAME_ROOT, 1, "b");
        let shared_under_a = builder.enter(a, 2, "shared");
        let shared_under_b = builder.enter(b, 2, "shared");
        assert_ne!(shared_under_a, shared_under_b);
        assert_eq!(builder.node_count(), 4);

        builder.accumulate(shared_under_a, 3);
        builder.accumulate(shared_under_b, 4);
        builder.accumulate(a, 10);
        builder.accumulate(b, 10);

        let root = builder.finish(20);
        assert_eq!(root.children.len(), 2);
        assert_eq!(root.children[0].children[0].inclusive_instructions, 3);
        assert_eq!(root.children[1].children[0].inclusive_instructions, 4);
    }

    /// `max_nodes` is a hard ceiling, and overflow folds a region into its
    /// parent: accuracy degrades, memory does not grow, and every index handed
    /// out stays a registered node (so `merge`'s parents-come-first invariant
    /// survives overflow).
    #[test]
    fn streaming_node_cap_folds_into_parent() {
        const CAP: usize = 5;
        let mut builder = StreamingRoiFlamegraph::new(CAP);

        let parent = builder.enter(FLAME_ROOT, 0, "parent");
        let mut folded = 0;
        for region_id in 1..20 {
            let name = format!("dynamic_{region_id}");
            let node = builder.enter(parent, region_id, &name);
            if node == parent {
                folded += 1;
            }
            builder.accumulate(node, 10);
        }

        assert_eq!(
            builder.node_count(),
            CAP,
            "the cap is exact, not off by one"
        );
        assert!(
            folded > 0,
            "overflowing regions must fold into their parent"
        );
        // Nothing is lost, it is just attributed to the parent.
        assert_eq!(builder.executions(), 19);
    }

    /// Cross-hart merge must key on *name*, not `region_id`: ids are allocated
    /// per `Enricher`, so hart 0's id 0 and hart 1's id 0 are different
    /// regions. An id-keyed merge collapsed all of a multi-hart run's
    /// per-thread regions into a single node.
    #[test]
    fn merge_across_harts_keys_on_name_not_region_id() {
        let mut merged = StreamingRoiFlamegraph::default();
        for (hart, name) in ["thread_0_work", "thread_1_work", "thread_2_work"]
            .into_iter()
            .enumerate()
        {
            let mut per_hart = StreamingRoiFlamegraph::default();
            // Every hart's interner independently hands out region_id 0 first.
            let n = per_hart.enter(FLAME_ROOT, 0, name);
            per_hart.accumulate(n, 2 * (hart as u64 + 1));
            merged.merge(per_hart);
        }

        let root = merged.finish(100);
        let got: Vec<(&str, u64)> = root
            .children
            .iter()
            .map(|c| (c.name.as_str(), c.inclusive_instructions))
            .collect();
        assert_eq!(
            got,
            vec![
                ("thread_0_work", 2),
                ("thread_1_work", 4),
                ("thread_2_work", 6)
            ]
        );
        assert_eq!(merged.executions(), 3);
    }

    /// The per-thread graphs written beside the merged one must be rooted at
    /// *that hart's* retired count, not the
    /// run's, or the root's self-time is the difference between the whole run
    /// and one thread.
    #[test]
    fn per_hart_finish_uses_that_harts_global() {
        let mut per_hart = StreamingRoiFlamegraph::default();
        let n = per_hart.enter(FLAME_ROOT, 0, "work");
        per_hart.accumulate(n, 70);

        // 100 retired on this hart, of which 70 is inside a region.
        let hart_root = per_hart.finish(100);
        assert_eq!(hart_root.inclusive_instructions, 100);
        assert_eq!(hart_root.self_instructions, 30);

        // The run total, wrongly passed, would claim 330 of self-time on a
        // hart that only retired 100.
        assert_eq!(per_hart.finish(400).self_instructions, 330);
    }

    /// `Global` is additive across harts, so the per-thread roots sum to the
    /// merged root. This is what lets the two views be read against each
    /// other, and it holds only because every thread that retired anything is
    /// given a graph -- including one that entered no region at all.
    #[test]
    fn per_hart_root_inclusive_sums_to_merged_root() {
        let per_hart_retired = [100u64, 60, 60];
        let mut merged = StreamingRoiFlamegraph::default();
        let mut sum = 0u64;

        for (hart, &retired) in per_hart_retired.iter().enumerate() {
            let mut b = StreamingRoiFlamegraph::default();
            let n = b.enter(FLAME_ROOT, 0, "work");
            b.accumulate(n, retired / 2);
            sum += b.finish(retired).inclusive_instructions;
            // Same order as the real caller: finish first, then fold in.
            merged.merge(b);
            assert_eq!(merged.executions(), hart as u64 + 1);
        }

        let merged_root = merged.finish(per_hart_retired.iter().sum());
        assert_eq!(sum, merged_root.inclusive_instructions);
    }

    /// A thread that entered no region still gets a usable graph: a bare root
    /// whose self-time is everything it retired. This is the case the merged
    /// graph's `has_executions` gate throws away and the per-thread view must
    /// not -- `pthreads_roi`'s main thread is 98% of the run and enters
    /// nothing at all.
    #[test]
    fn region_free_thread_finishes_to_a_bare_root() {
        let builder = StreamingRoiFlamegraph::default();
        assert!(!builder.has_executions(), "nothing was ever entered");

        let root = builder.finish(170_103);
        assert_eq!(root.name, "Global");
        assert!(root.children.is_empty());
        // All of it is self-time -- the whole point of listing this thread.
        assert_eq!(root.inclusive_instructions, 170_103);
        assert_eq!(root.self_instructions, 170_103);
        assert_eq!(root.node_count(), 1);
    }

    /// The sum invariant across a realistic mix: one region-free thread that
    /// dominates the run, plus workers that each own one region. This is the
    /// `pthreads_roi` shape, and it only adds up because the region-free
    /// thread is included.
    #[test]
    fn region_free_thread_is_what_makes_the_index_sum_to_the_merged_root() {
        let mut merged = StreamingRoiFlamegraph::default();
        let mut per_hart_totals = Vec::new();

        // hart 0: 170,103 retired, no regions.
        let main = StreamingRoiFlamegraph::default();
        per_hart_totals.push(main.finish(170_103).inclusive_instructions);
        merged.merge(main);

        // harts 1..3: a region each.
        for (retired, region_instrs) in [(973u64, 2u64), (934, 4), (937, 6)] {
            let mut b = StreamingRoiFlamegraph::default();
            let n = b.enter(FLAME_ROOT, 0, "work");
            b.accumulate(n, region_instrs);
            per_hart_totals.push(b.finish(retired).inclusive_instructions);
            merged.merge(b);
        }

        let run_total: u64 = per_hart_totals.iter().sum();
        assert_eq!(run_total, 172_947);
        assert_eq!(
            merged.finish(run_total).inclusive_instructions,
            run_total,
            "the per-thread roots must account for the whole run"
        );
        // Dropping the region-free thread would account for 2% of it.
        let workers: u64 = per_hart_totals[1..].iter().sum();
        assert!(workers * 20 < run_total);
    }

    /// Self-time does *not* reconcile the same way, and that is not a bug:
    /// a per-hart root's children are only that hart's regions, while the
    /// merged root's are every hart's. Pinned because a reader diffing the two
    /// numbers will otherwise read the difference as one.
    #[test]
    fn per_hart_and_merged_self_time_differ_by_design() {
        let mut h0 = StreamingRoiFlamegraph::default();
        let a = h0.enter(FLAME_ROOT, 0, "a");
        h0.accumulate(a, 40);
        let h0_root = h0.finish(100);

        let mut h1 = StreamingRoiFlamegraph::default();
        let b = h1.enter(FLAME_ROOT, 0, "b");
        h1.accumulate(b, 10);
        let h1_root = h1.finish(100);

        let mut merged = StreamingRoiFlamegraph::default();
        merged.merge(h0);
        merged.merge(h1);
        let merged_root = merged.finish(200);

        // Inclusive adds up.
        assert_eq!(
            h0_root.inclusive_instructions + h1_root.inclusive_instructions,
            merged_root.inclusive_instructions
        );
        // Self-time does not: the merged root loses both harts' regions, each
        // per-hart root only its own.
        assert_eq!(h0_root.self_instructions, 60);
        assert_eq!(h1_root.self_instructions, 90);
        assert_eq!(merged_root.self_instructions, 150);
    }

    /// One region name on several harts is one node in the merged graph but
    /// stays separate per hart. That collapse is what makes per-thread
    /// imbalance invisible in the merged view -- the reason the per-hart
    /// graphs are written at all -- and it must not be "fixed" by suffixing
    /// names, which would break the ROI table's name join.
    #[test]
    fn shared_region_name_collapses_merged_but_stays_per_hart() {
        let mut merged = StreamingRoiFlamegraph::default();
        let mut per_hart_roots = Vec::new();
        // Hart 0 does 1.8x the work of each of the other three.
        for retired in [42u64, 23, 23, 23] {
            let mut b = StreamingRoiFlamegraph::default();
            let n = b.enter(FLAME_ROOT, 0, "Section 3: Physics");
            b.accumulate(n, retired);
            per_hart_roots.push(b.finish(retired));
            merged.merge(b);
        }

        let merged_root = merged.finish(111);
        assert_eq!(merged_root.children.len(), 1, "one node, four harts");
        assert_eq!(merged_root.children[0].inclusive_instructions, 111);
        assert_eq!(merged_root.children[0].call_count, 4);

        // The imbalance the merged 111 hides.
        let per_hart: Vec<u64> = per_hart_roots
            .iter()
            .map(|r| r.children[0].inclusive_instructions)
            .collect();
        assert_eq!(per_hart, vec![42, 23, 23, 23]);
    }

    /// `node_count` counts call paths including the synthetic root, which is
    /// what `roi_flamegraph_harts.json` reports per thread.
    #[test]
    fn flame_node_count_includes_the_root() {
        let mut b = StreamingRoiFlamegraph::default();
        let outer = b.enter(FLAME_ROOT, 0, "outer");
        let inner = b.enter(outer, 1, "inner");
        b.accumulate(inner, 1);
        b.accumulate(outer, 2);
        // Global + outer + inner.
        assert_eq!(b.finish(2).node_count(), 3);
    }

    /// The per-hart file names are spelled in exactly one place, since the
    /// index carries them to consumers rather than each rebuilding the pattern.
    #[test]
    fn hart_file_names_match_the_documented_pattern() {
        assert_eq!(hart_json_name(3), "roi_flamegraph_hart3.json");
        assert_eq!(hart_folded_name(0), "roi_flamegraph_hart0.folded");
    }

    /// Merging preserves nesting, and the same call path from two harts folds
    /// into one node with the counts summed.
    #[test]
    fn merge_sums_matching_call_paths_and_keeps_nesting() {
        let mut h0 = StreamingRoiFlamegraph::default();
        let o0 = h0.enter(FLAME_ROOT, 0, "outer");
        let i0 = h0.enter(o0, 1, "inner");
        h0.accumulate(i0, 3);
        h0.accumulate(o0, 10);

        let mut h1 = StreamingRoiFlamegraph::default();
        // Different local ids for the same names -- the merge must not care.
        let o1 = h1.enter(FLAME_ROOT, 4, "outer");
        let i1 = h1.enter(o1, 2, "inner");
        h1.accumulate(i1, 5);
        h1.accumulate(o1, 20);

        let mut merged = StreamingRoiFlamegraph::default();
        merged.merge(h0);
        merged.merge(h1);

        let root = merged.finish(30);
        assert_eq!(root.children.len(), 1);
        let outer = &root.children[0];
        assert_eq!(outer.name, "outer");
        assert_eq!(outer.inclusive_instructions, 30);
        assert_eq!(outer.call_count, 2);
        assert_eq!(outer.children.len(), 1);
        let inner = &outer.children[0];
        assert_eq!(inner.name, "inner");
        assert_eq!(inner.inclusive_instructions, 8);
        assert_eq!(inner.call_count, 2);
    }

    /// A region that is entered but never closed creates a node without ever
    /// completing an execution. `has_executions` is what gates the report, so
    /// such a run does not emit a flamegraph containing nothing but the root.
    #[test]
    fn opened_but_never_closed_region_reports_no_executions() {
        let mut builder = StreamingRoiFlamegraph::default();
        builder.enter(FLAME_ROOT, 0, "never_closed");

        assert!(!builder.is_empty(), "the node exists");
        assert!(!builder.has_executions(), "but nothing completed");
    }

    /// repeated.c: 10 iterations of the same named region at top level should
    /// merge into a single node with call_count = 10.
    #[test]
    fn repeated_siblings_merge() {
        let mut execs = Vec::new();
        let mut cursor = 0u64;
        for _ in 0..10 {
            let begin = cursor;
            let end = begin + 3;
            execs.push(exec("iterative loop", 1, None, 0, begin, end));
            cursor = end;
        }
        let root = build_flamegraph(&execs, cursor);

        assert_eq!(root.children.len(), 1);
        let node = &root.children[0];
        assert_eq!(node.name, "iterative loop");
        assert_eq!(node.call_count, 10);
        assert_eq!(node.inclusive_instructions, 30);
        assert_eq!(node.self_instructions, 30);
        assert_eq!(root.inclusive_instructions, 30);
        assert_eq!(root.self_instructions, 0);
    }

    /// non_overlapping.c: three sibling leaves under Global, no nesting.
    #[test]
    fn non_overlapping_siblings() {
        let execs = vec![
            exec("region_1", 1, None, 0, 0, 1),
            exec("region_2", 2, None, 0, 1, 3),
            exec("region_3", 3, None, 0, 3, 4),
        ];
        let root = build_flamegraph(&execs, 10);

        assert_eq!(root.children.len(), 3);
        let names: Vec<&str> = root.children.iter().map(|c| c.name.as_str()).collect();
        assert_eq!(names, vec!["region_1", "region_2", "region_3"]);
        for c in &root.children {
            assert!(c.children.is_empty());
            assert_eq!(c.call_count, 1);
        }
        // Global has 10 total instructions but only 1+2+1=4 attributed to regions.
        assert_eq!(root.self_instructions, 6);
    }

    #[test]
    fn folded_stacks_nested() {
        let execs = vec![
            exec("region_1", 1, None, 0, 0, 8),
            exec("region_2", 2, Some(1), 1, 3, 5),
        ];
        let root = build_flamegraph(&execs, 8);
        let folded = to_folded_stacks(&root);

        // region_1 self=6, region_2 self=2; Global self=0 so it's skipped.
        let lines: Vec<&str> = folded.lines().collect();
        assert_eq!(
            lines,
            vec!["Global;region_1 6", "Global;region_1;region_2 2"]
        );
    }

    #[test]
    fn folded_stacks_skips_zero_self() {
        let execs = vec![exec("region_1", 1, None, 0, 0, 4)];
        // Global inclusive == region_1 inclusive, so Global's self is 0 and
        // must not be emitted.
        let root = build_flamegraph(&execs, 4);
        let folded = to_folded_stacks(&root);
        assert_eq!(folded, "Global;region_1 4\n");
    }

    #[test]
    fn json_roundtrip_nested() {
        let execs = vec![
            exec("region_1", 1, None, 0, 0, 8),
            exec("region_2", 2, Some(1), 1, 3, 5),
        ];
        let root = build_flamegraph(&execs, 8);
        let json = to_json(&root).expect("serialize");

        let parsed: serde_json::Value = serde_json::from_str(&json).expect("valid json");
        assert_eq!(parsed["name"], "Global");
        assert_eq!(parsed["value"], 8);
        assert_eq!(parsed["self_value"], 0);
        let r1 = &parsed["children"][0];
        assert_eq!(r1["name"], "region_1");
        assert_eq!(r1["value"], 8);
        assert_eq!(r1["self_value"], 6);
        let r2 = &r1["children"][0];
        assert_eq!(r2["name"], "region_2");
        assert_eq!(r2["value"], 2);
        assert_eq!(r2["self_value"], 2);
        assert!(r2["children"].as_array().unwrap().is_empty());
    }

    fn nested_root() -> RoiFlameNode {
        let execs = vec![
            exec("region_1", 1, None, 0, 0, 8),
            exec("region_2", 2, Some(1), 1, 3, 5),
        ];
        build_flamegraph(&execs, 8)
    }

    #[test]
    fn html_page_is_self_contained_and_carries_the_data() {
        let html = to_html(&nested_root()).expect("html render must succeed");

        assert!(html.starts_with("<!doctype html>"));
        assert!(html.ends_with("</html>"));
        // Self-contained: no external stylesheet or script to fetch.
        assert!(!html.contains("<link"));
        assert!(!html.contains("src="));
        // The tree ships inline, and the widget is wired up.
        assert!(html.contains("\"name\":\"region_1\""));
        assert!(html.contains("\"name\":\"region_2\""));
        assert!(html.contains("\"name\":\"Global\""));
        assert!(html.contains("window.ocfgInit(\"roi-flamegraph\")"));
        assert!(html.contains("id=\"roi-flamegraph-data\""));
        // The runtime-loaded assets really are inlined, not just referenced.
        assert!(html.contains(".ocfg-frame {"));
        assert!(html.contains("window.ocfgInit = function"));
    }

    /// The fragment must be droppable into an arbitrary page/`<div>`: it may
    /// not carry document-level markup, and its container id has to follow the
    /// caller's choice so two graphs can share a page.
    #[test]
    fn fragment_is_embeddable_under_a_caller_chosen_id() {
        let frag = to_html_fragment(&nested_root(), "panel-fg").expect("fragment render");

        assert!(!frag.contains("<html"));
        assert!(!frag.contains("<body"));
        assert!(frag.contains("id=\"panel-fg\""));
        assert!(frag.contains("id=\"panel-fg-data\""));
        assert!(frag.contains("window.ocfgInit(\"panel-fg\")"));
    }

    /// A region name containing `</script>` must not be able to close the data
    /// element and inject markup into the host page.
    #[test]
    fn region_name_cannot_break_out_of_the_data_script() {
        let execs = vec![exec("</script><img onerror=x>", 1, None, 0, 0, 4)];
        let root = build_flamegraph(&execs, 8);
        let html = to_html(&root).expect("html render");

        assert!(!html.contains("</script><img"));
        assert!(html.contains("<\\/script>"));
    }
}
