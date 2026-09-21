"use strict";

/* ROI flamegraph widget (see processing/roi_flamegraph.rs).
 *
 * Registers one global `window.ocfgInit(id)` instead of running inline: each
 * container calls it with its own id, so several graphs can share a page and a
 * graph moved into a different <div> just needs another call. Re-defining is
 * skipped, since the fragment inlines this file once per embed.
 *
 * Two-way selection with a host page (the dashboard's ROI table) goes through
 * the region *name*, the only identity the two sides share: the graph publishes
 * `window.ocfgGraphs`, an id -> {select(name), selectedName()} map the host
 * calls to highlight, and emits a bubbling `ocfg:select` event on its container
 * when a frame is clicked. A host that ignores both still gets a working
 * standalone widget. */
(function () {
    /* Published before the early return so a second copy of this file (one per
       embedded fragment) still finds the map its first copy created. */
    window.ocfgGraphs = window.ocfgGraphs || new Map();
    if (window.ocfgInit) return;

    const ROW = 18; // Frame height + 1px gap, in px.
    const MIN_PCT = 0.08; // Frames thinner than this are not drawn; see draw().
    const intFmt = new Intl.NumberFormat();

    const esc = (s) =>
        String(s).replace(
            /[&<>"]/g,
            (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c],
        );

    /* Warm flamegraph palette, keyed on the region name so a region keeps its
       colour across zooms, runs and reports. Lightness is nudged by the name too,
       so adjacent frames with near hues still read as separate blocks. */
    function color(name) {
        let h = 0;
        for (const ch of name) h = (h * 31 + ch.charCodeAt(0)) | 0;
        h = Math.abs(h);
        return `hsl(${6 + (h % 44)},78%,${56 + (h % 5) * 3}%)`;
    }

    const depthOf = (node) =>
        1 + Math.max(0, ...node.children.map(depthOf));

    /* Whether `name` occurs anywhere in this subtree -- used to tell a
       highlight that would land outside the current zoom from one that is
       simply not in the graph at all. */
    const hasName = (node, name) =>
        node.name === name || node.children.some((c) => hasName(c, name));

    window.ocfgInit = function (id) {
        const el = document.getElementById(id);
        const dataEl = document.getElementById(id + "-data");
        if (!el || !dataEl) return;
        const root = JSON.parse(dataEl.textContent);

        const frames = el.querySelector(".ocfg-frames");
        const tip = el.querySelector(".ocfg-tip");
        const note = el.querySelector(".ocfg-note");
        const search = el.querySelector(".ocfg-search");
        const reset = el.querySelector(".ocfg-btn");

        // Index by key so a click maps back to its node without stashing objects
        // on DOM elements.
        const byKey = new Map();
        (function index(node) {
            byKey.set(node.key, node);
            node.children.forEach(index);
        })(root);

        const total = root.value || 1;
        let focus = root;
        /* Region name highlighted across the whole graph, not a single frame:
           one region reached by several call paths is several frames here but a
           single row in the host's ROI table, so all of them light up. */
        let selected = null;
        // Ancestors of the current zoom focus, so clicking the focus again (and
        // "reset") can walk back out.
        let zoomPath = [];

        function draw() {
            const span = focus.value || 1;
            const html = [];
            (function walk(node, x, depth) {
                const w = (node.value / span) * 100;
                // Sub-pixel frames cost more to render than they can ever show;
                // their cost is still visible in the parent, and zooming in
                // brings them back.
                if (w < MIN_PCT) return;
                html.push(
                    `<div class="ocfg-frame" data-k="${esc(node.key)}" style="left:${
                        (x / span) * 100
                    }%;width:${w}%;top:${depth * ROW}px;background:${color(node.name)}">${esc(
                        node.name,
                    )}</div>`,
                );
                let cx = x;
                for (const c of node.children) {
                    walk(c, cx, depth + 1);
                    cx += c.value;
                }
            })(focus, 0, 0);

            frames.style.height = depthOf(focus) * ROW + "px";
            // nosemgrep: insecure-document-method -- esc() on every node name and key
            frames.innerHTML = html.join("");
            // The frames the pointer was over are gone, so no mouseleave will
            // fire: drop the tooltip rather than leave it naming a moved frame.
            hideTip();
            note.textContent =
                `${intFmt.format(focus.value)} instructions` +
                (focus === root ? "" : ` — zoomed to ${focus.name}`);
            paint();
        }

        /* Search and selection are both per-frame classes over the frames the
           last draw() produced, so they share one pass: a selected frame is
           never faded out, or a search would hide the very frame the host page
           just asked to point at. */
        function paint() {
            const q = search.value.trim().toLowerCase();
            for (const f of frames.children) {
                const node = byKey.get(f.dataset.k);
                const sel = selected !== null && !!node && node.name === selected;
                const hit = q !== "" && f.textContent.toLowerCase().includes(q);
                f.classList.toggle("ocfg-hit", hit);
                f.classList.toggle("ocfg-sel", sel);
                f.classList.toggle("ocfg-fade", q !== "" && !hit && !sel);
            }
        }

        /* Highlight every frame of `name` (null clears). A region that is in
           the graph but not under the current zoom focus walks the zoom back
           out, so a click in the host's table is always visible; one that is
           nowhere in the graph leaves the zoom alone -- there is nothing to
           reveal, and resetting would throw away the user's zoom. */
        function select(name) {
            selected = name == null ? null : String(name);
            const reveal =
                selected !== null &&
                !hasName(focus, selected) &&
                hasName(root, selected);
            if (reveal) {
                focus = root;
                zoomPath = [];
                draw();
            } else {
                paint();
            }
        }

        const hideTip = () => tip.classList.remove("ocfg-on");
        const nodeAt = (ev) => {
            const f = ev.target.closest(".ocfg-frame");
            return f ? byKey.get(f.dataset.k) : undefined;
        };
        const pct = (v) => ((v / total) * 100).toFixed(2);

        frames.addEventListener("click", (ev) => {
            const n = nodeAt(ev);
            if (!n) return;
            selected = n.name;
            if (n === focus) focus = zoomPath.pop() || root;
            else {
                zoomPath.push(focus);
                focus = n;
            }
            draw();
            /* After draw(), so a host that reacts by calling select() back into
               this graph paints the frames that are actually on screen. */
            el.dispatchEvent(
                new CustomEvent("ocfg:select", {
                    bubbles: true,
                    detail: {
                        id,
                        name: n.name,
                        key: n.key,
                        region_id: n.region_id,
                        value: n.value,
                        self_value: n.self_value,
                        call_count: n.call_count,
                    },
                }),
            );
        });

        frames.addEventListener("mousemove", (ev) => {
            const n = nodeAt(ev);
            if (!n) return hideTip();
            // nosemgrep: insecure-document-method -- esc() on the name, Intl formatting on the numbers
            tip.innerHTML =
                `<div class="ocfg-tip-name">${esc(n.name)}</div>` +
                `<div>${intFmt.format(n.value)} instructions (${pct(n.value)}% of total)</div>` +
                `<div>${intFmt.format(n.self_value)} self (${pct(n.self_value)}%)</div>` +
                `<div>${intFmt.format(n.call_count)} executions</div>`;
            tip.classList.add("ocfg-on");
            // Flip to the left of the cursor near the right edge so the tooltip
            // never pushes the page wider than the viewport.
            tip.style.left =
                Math.min(ev.clientX + 14, window.innerWidth - tip.offsetWidth - 8) + "px";
            tip.style.top = ev.clientY + 16 + "px";
        });
        frames.addEventListener("mouseleave", hideTip);

        search.addEventListener("input", paint);
        reset.addEventListener("click", () => {
            focus = root;
            zoomPath = [];
            search.value = "";
            draw();
        });

        window.ocfgGraphs.set(id, {
            select,
            selectedName: () => selected,
        });

        draw();
    };
})();
