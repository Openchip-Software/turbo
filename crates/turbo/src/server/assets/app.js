"use strict";

/* -------------------------------------------------------------
   FORMATTERS  -- use pre-created Intl instances (much faster
   than calling .toLocaleString() / .toFixed() every call)
   ------------------------------------------------------------- */
const _intFmt = new Intl.NumberFormat();
const _floatFmt = new Intl.NumberFormat(undefined, {
    minimumFractionDigits: 4,
    maximumFractionDigits: 4,
});

function fmtInt(v) {
    if (v === null || v === undefined) return "\u2014";
    return _intFmt.format(v);
}

function fmtFloat(v) {
    if (v === null || v === undefined) return "\u2014";
    return _floatFmt.format(v);
}

const BYTE_UNITS = ["B", "KB", "MB", "GB", "TB"];
const _bytesFmt = new Intl.NumberFormat(undefined, {
    maximumFractionDigits: 2,
});
function fmtBytes(v) {
    if (v === null || v === undefined || v === 0) return "0\u00a0B";
    let n = Number(v),
        i = 0;
    while (n >= 1024 && i < BYTE_UNITS.length - 1) {
        n /= 1024;
        i++;
    }
    return (
        (i === 0 ? _intFmt.format(n) : _bytesFmt.format(n)) +
        "\u00a0" +
        BYTE_UNITS[i]
    );
}

/* Modelled cache behaviour as a two-bar infographic: one bar per level, green
   from 0 to hits and red from hits out to that level's accesses.

   Both bars share one x axis whose max is L1 accesses, because the L2 only ever
   sees what missed the L1 -- so its bar's *length* is its share of all traffic
   and the hit/miss split within it is its own rate. Normalising each bar to its
   own total would draw a 64-lookup L2 as wide as a 2.7M-access L1 and lose
   exactly the thing worth seeing. A non-empty level always gets at least a
   hairline so "tiny" stays distinguishable from "absent".

   The cell value is a `{l1, l2, axis, rate}` object rather than a number
   because it shows both levels at once; `getSortedRows` sorts it by `.rate`
   (the L1 hit rate) so the column still orders sensibly. `null` means no cache
   was modelled (the run had no `--cache-sim`), which must read as "no data" and
   not as a 0% hit rate. */
function _cacheBarHtml(name, lvl, axis, withCounts) {
    /* Widths are percentages of the axis; the green/red pair together span the
       level's own share of it, leaving the remainder as empty track. A non-zero
       segment keeps a 0.6% floor so "tiny" stays visible; a zero one gets
       nothing, or an all-hits level would still show a red sliver. */
    var pct = function (n) {
        return n > 0 ? Math.max(0.6, (100 * n) / axis) : 0;
    };
    return (
        '<span class="cbar-row">' +
        '<span class="cbar-lbl">' +
        name +
        "</span>" +
        '<span class="cbar-track">' +
        (lvl
            ? '<i class="cbar-hit" style="width:' +
              pct(lvl.h).toFixed(3) +
              '%"></i>' +
              '<i class="cbar-miss" style="width:' +
              pct(lvl.m).toFixed(3) +
              '%"></i>'
            : "") +
        "</span>" +
        (withCounts
            ? '<span class="cbar-counts">' +
              (lvl
                  ? _intFmt.format(lvl.h) + " / " + _intFmt.format(lvl.m)
                  : "\u2014") +
              "</span>"
            : "") +
        '<span class="cbar-rate">' +
        (lvl ? (lvl.rate * 100).toFixed(1) + "%" : "\u2014") +
        "</span>" +
        "</span>"
    );
}

/* Table-cell form: bars and hit rates only, no absolute counts. */
function fmtCacheBars(v) {
    if (!v) return "\u2014";
    return (
        '<span class="cbar">' +
        _cacheBarHtml("L1", v.l1, v.axis, false) +
        _cacheBarHtml("L2", v.l2, v.axis, false) +
        "</span>"
    );
}

/* Detail-pane form: the same bars plus the absolute hit/miss counts. */
function _cacheStatsBodyHtml(v) {
    if (!v) return '<div class="detail-empty-note">No cache modelled</div>';
    return (
        '<div class="cbar cbar-detail">' +
        _cacheBarHtml("L1", v.l1, v.axis, true) +
        _cacheBarHtml("L2", v.l2, v.axis, true) +
        '<div class="cbar-key">' +
        '<i class="cbar-hit"></i>hit<i class="cbar-miss"></i>miss' +
        "</div></div>"
    );
}

/* -------------------------------------------------------------
   COLUMN DEFINITIONS
   key         field name in the data JSON
   label       column header text
   cls         td class(es) -- "num" = right-aligned
   fmt(v)      value formatter; null = raw string
   ------------------------------------------------------------- */
const COLS = [
    { key: "name", label: "Name", cls: "name-cell", fmt: null },
    { key: "instructions", label: "Instructions", cls: "num", fmt: fmtInt },
    { key: "float_elements", label: "Float Ops", cls: "num", fmt: fmtInt },
    { key: "fmacc_elements", label: "FMA-Ops", cls: "num", fmt: fmtInt },
    { key: "vector_instructions", label: "Vec Instr", cls: "num", fmt: fmtInt },
    { key: "bytes_moved", label: "Bytes Moved", cls: "num", fmt: fmtBytes },
    { key: "flops_per_byte", label: "FLOPs/Byte", cls: "num", fmt: fmtFloat },
    {
        key: "cache_stats",
        label: "Cache Stats",
        cls: "cache-cell",
        fmt: fmtCacheBars,
        html: true,
    },
];

/* -------------------------------------------------------------
   HTML ESCAPING  -- fast path for the common case
   Most cell values are formatted numbers -- no special chars.
   ------------------------------------------------------------- */
const _escapeMap = {
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    '"': "&quot;",
    "'": "&#39;",
};
function escHtml(str) {
    if (str === null || str === undefined) return "\u2014";
    const s = String(str);
    if (
        s.indexOf("&") === -1 &&
        s.indexOf("<") === -1 &&
        s.indexOf(">") === -1 &&
        s.indexOf('"') === -1 &&
        s.indexOf("'") === -1
    )
        return s;
    return s.replace(/[&<>"']/g, (ch) => _escapeMap[ch]);
}

/* -------------------------------------------------------------
   STATE
   ------------------------------------------------------------- */
const state = {
    functions: [],
    rois: [],
    frozen: false,
    sort: {
        functions: { colIdx: 1, dir: "desc" },
        rois: { colIdx: 1, dir: "desc" },
    },
    filter: { functions: "", rois: "" },
    /* Sort result cache -- only re-sort when data or sort config changes */
    _sortedFunctionsKey: "",
    _sortedFunctions: [],
    _sortedRoisKey: "",
    _sortedRois: [],
    /* Selected row for detail pane */
    selectedRow: null,
    selectedDataKey: null,
    /* Fingerprints for data change detection */
    _lastFunctionsFingerprint: 0,
    _lastRoisFingerprint: 0,
    /* Track which row name is rendered in the detail pane */
    _detailPaneName: null,
    _detailPaneKey: null,
    /* Persisted open/closed state of the Instruction Mix expander,
       reused when switching between ROIs/functions */
    _instrMixOpen: true,
    /* Same, for the Cache Stats expander */
    _cacheStatsOpen: true,
};

/* -------------------------------------------------------------
   HELPERS
   ------------------------------------------------------------- */
const byId = (id) => document.getElementById(id);

/* Derive the client-side-only columns from the wire fields: the server sends
   load_bytes/store_bytes and float_elements, not the combined/ratio values. */
function deriveRowMetrics(rows) {
    if (!Array.isArray(rows)) return;
    for (const row of rows) {
        row.bytes_moved = (row.load_bytes || 0) + (row.store_bytes || 0);
        row.flops_per_byte =
            row.bytes_moved > 0
                ? (row.float_elements || 0) / row.bytes_moved
                : 0;
        /* `cache` is absent unless the run was processed with --cache-sim, and
           the L2 only ever sees what missed the L1 -- so its rate is over its
           own lookups (l1_misses), not over every access. */
        row.cache_stats = cacheStats(row.cache);
    }
}

/* Both levels plus the shared axis (L1 accesses) the bars are drawn against,
   or null when the L1 was never looked up -- there is nothing to draw and no
   axis to draw it on. `.rate` is the L1 hit rate, which is what the column
   sorts by. */
function cacheStats(cache) {
    const l1 = cacheCell(cache, "l1_hits", "l1_misses");
    if (!l1) return null;
    return {
        l1: l1,
        l2: cacheCell(cache, "l2_hits", "l2_misses"),
        axis: l1.h + l1.m,
        rate: l1.rate,
    };
}

/* One level's counts as the {h, m, rate} the cache columns render, or null when
   the level was never looked up -- which is both "no cache modelled" and "an L2
   whose L1 never missed", and neither is a 0% hit rate. */
function cacheCell(cache, hitKey, missKey) {
    if (!cache) return null;
    const h = cache[hitKey] || 0,
        m = cache[missKey] || 0;
    if (h + m === 0) return null;
    return { h: h, m: m, rate: h / (h + m) };
}

/*
   DATA CHANGE DETECTION
   FNV-1a 32-bit hash over all row fields.  Covers every numeric
   field (including large/fractional values) and the name string
   without allocating intermediate strings.  Returns an unsigned
   32-bit integer; identical payloads produce the same value.
*/
function computeFingerprint(rows) {
    const P = 0x01000193; // FNV-1a 32-bit prime
    let h = 0x811c9dc5; // FNV offset basis

    // Mix one 32-bit word into h
    function mixWord(v) {
        v = v | 0;
        h = Math.imul(h ^ (v & 0xff), P);
        h = Math.imul(h ^ ((v >>> 8) & 0xff), P);
        h = Math.imul(h ^ ((v >>> 16) & 0xff), P);
        h = Math.imul(h ^ ((v >>> 24) & 0xff), P);
    }

    // Mix a number that may exceed 2^31 or carry a fractional part
    function mixNum(v) {
        if (v == null) {
            mixWord(0x7fffffff);
            return;
        }
        mixWord(v | 0); // lower 32 bits
        const hi = Math.floor(v / 0x100000000) | 0;
        if (hi) mixWord(hi); // upper bits (bytes_moved etc.)
        const frac = ((v % 1) * 1e7) | 0;
        if (frac) mixWord(frac); // fractional precision
    }

    // Mix a string; final mixWord(h) acts as null-terminator so adjacent
    // fields with different lengths cannot alias each other.
    function mixStr(str) {
        if (!str) {
            mixWord(0);
            return;
        }
        for (let j = 0; j < str.length; j++) {
            const c = str.charCodeAt(j);
            h = Math.imul(h ^ (c & 0xff), P);
            if (c > 0xff) h = Math.imul(h ^ (c >>> 8), P);
        }
        h = Math.imul(h, P); // terminator
    }

    // Mix a sparse [[vl, count], ...] histogram.  Both halves of every pair
    // participate -- a length-plus-total shortcut would alias whenever counts
    // are redistributed across buckets, which is exactly the change the
    // detail pane needs to see.
    function mixSparse(arr) {
        if (!Array.isArray(arr) || arr.length === 0) {
            mixWord(0);
            return;
        }
        mixWord(arr.length);
        for (let j = 0; j < arr.length; j++) {
            mixNum(arr[j][0]);
            mixNum(arr[j][1]);
        }
    }

    mixWord(rows.length);
    for (let i = 0; i < rows.length; i++) {
        const r = rows[i];
        mixStr(r.name != null ? String(r.name) : "");
        mixNum(r.instructions);
        mixNum(r.float_elements);
        mixNum(r.fmacc_elements);
        mixNum(r.vector_instructions);
        mixNum(r.bytes_moved);
        mixNum(r.flops_per_byte);
        /* Cache counts move independently of every scalar above (the same
           instructions can hit this batch and miss the next), so they have to
           participate or a payload that only changed them hashes equal and the
           table never updates. */
        if (r.cache && typeof r.cache === "object") {
            mixNum(r.cache.l1_hits);
            mixNum(r.cache.l1_misses);
            mixNum(r.cache.l2_hits);
            mixNum(r.cache.l2_misses);
        } else {
            mixWord(0);
        }

        /* The detail pane renders instruction_mix and the VL histograms, so
           they have to participate: a payload whose scalars are unchanged but
           whose distributions moved must not hash equal, or the handlers
           short-circuit and the pane goes stale. */
        const mix = r.instruction_mix;
        if (mix && typeof mix === "object") {
            for (const k in mix) {
                mixStr(k);
                mixNum(mix[k]);
            }
        } else {
            mixWord(0);
        }

        // HIST_FIELDS is defined further down; safe because this only runs
        // once an SSE payload arrives, long after the script has evaluated.
        for (let f = 0; f < HIST_FIELDS.length; f++) {
            mixSparse(r[HIST_FIELDS[f].rawKey]);
        }
    }

    return h >>> 0; // unsigned 32-bit integer
}

/* -------------------------------------------------------------
   ROW PRE-PROCESSING
   Called once per row on first render after new data arrives.
   Escapes the name field so the render path can drop it straight
   into innerHTML without re-escaping on every scroll.
   ------------------------------------------------------------- */
function preprocessRow(row) {
    if (row._nameEsc === undefined) {
        row._nameEsc = escHtml(row.name != null ? String(row.name) : "");
    }
}

/* -------------------------------------------------------------
   SORTING -- cached, only re-runs when stale
   Cache key encodes colIdx + dir + row-count.  A row-count
   change always means new data was received, so the cache is
   implicitly invalidated on every SSE update.
   ------------------------------------------------------------- */
function getSortedRows(dataKey, rows, sortState) {
    const cacheKey = sortState.colIdx + ":" + sortState.dir + ":" + rows.length;
    const keyProp =
        "_sorted" + dataKey.charAt(0).toUpperCase() + dataKey.slice(1) + "Key";
    const arrProp =
        "_sorted" + dataKey.charAt(0).toUpperCase() + dataKey.slice(1);

    if (state[keyProp] === cacheKey) return state[arrProp];

    const colKey = COLS[sortState.colIdx].key;
    const asc = sortState.dir === "asc";

    const sorted = rows.slice().sort((a, b) => {
        let av = a[colKey],
            bv = b[colKey];
        /* The cache cell is an object of both levels; sort it by the L1 hit
           rate, and put "no data" below every real rate rather than wherever
           object comparison happens to land it. */
        if (av && typeof av === "object" && "rate" in av) av = av.rate;
        else if (av == null && COLS[sortState.colIdx].fmt === fmtCacheBars)
            av = -1;
        if (bv && typeof bv === "object" && "rate" in bv) bv = bv.rate;
        else if (bv == null && COLS[sortState.colIdx].fmt === fmtCacheBars)
            bv = -1;
        if (typeof av === "string") av = av.toLowerCase();
        if (typeof bv === "string") bv = bv.toLowerCase();
        if (av < bv) return asc ? -1 : 1;
        if (av > bv) return asc ? 1 : -1;
        return 0;
    });

    state[keyProp] = cacheKey;
    state[arrProp] = sorted;
    return sorted;
}

/* -------------------------------------------------------------
   VIRTUAL SCROLL CONSTANTS
   ------------------------------------------------------------- */
const ROW_H = 28; // estimated row height in px
const OVERSCAN = 10; // extra rows above/below viewport

/* Store per-container virtual-scroll state to avoid re-querying DOM */
const vsState = {};

function getFilteredRows(dataKey, sorted) {
    const filterText = state.filter[dataKey].toLowerCase();
    if (!filterText) return sorted;
    return sorted.filter((r) => {
        const name = r.name;
        return (
            name != null &&
            String(name).toLowerCase().indexOf(filterText) !== -1
        );
    });
}

/* Format a single cell value -- returns { text, html }.
   text is the plain comparable string; html is the innerHTML content. */
function _formatCellValue(col, row) {
    if (col.key === "name") {
        return { text: row.name || "", html: row._nameEsc };
    } else if (col.fmt) {
        const v = col.fmt(row[col.key]);
        return { text: v, html: v };
    } else {
        const v = row[col.key];
        const s = v != null ? String(v) : "\u2014";
        return { text: s, html: s };
    }
}

function renderVisibleRows(containerId) {
    const vs = vsState[containerId];
    if (!vs) return;

    const container = vs.container;
    const scrollTop = container.scrollTop;
    const viewH = container.clientHeight;
    const filtered = vs.filtered;
    const totalRows = filtered.length;

    const firstVisible = Math.floor(scrollTop / ROW_H);
    const visibleCount = Math.ceil(viewH / ROW_H);
    const startIdx = Math.max(0, firstVisible - OVERSCAN);
    const endIdx = Math.min(totalRows, firstVisible + visibleCount + OVERSCAN);

    /* A changed row count with an unchanged [startIdx, endIdx) -- a filter
       that shrank the list while scrolled past it -- must not take the
       in-place path: that path skips slots whose row is now undefined and
       would leave the dropped rows on screen. */
    const rangeChanged =
        startIdx !== vs.lastStart ||
        endIdx !== vs.lastEnd ||
        totalRows !== vs.lastTotal;
    const dataChanged = vs._dataChanged;

    // Nothing changed -- skip entirely
    if (!rangeChanged && !dataChanged) return;

    vs._dataChanged = false;
    vs.lastStart = startIdx;
    vs.lastEnd = endIdx;
    vs.lastTotal = totalRows;

    // Range changed (scroll or first render) -- full innerHTML rebuild
    if (rangeChanged) {
        _renderVisibleRowsFull(vs, startIdx, endIdx, totalRows);
        return;
    }

    // Data changed but range is stable -- update cells in-place
    _updateVisibleRowsInPlace(vs, startIdx, endIdx, totalRows);
}

/* Full innerHTML rebuild -- used when scroll range changes */
function _renderVisibleRowsFull(vs, startIdx, endIdx, totalRows) {
    const tbody = vs.tbody;
    const colCount = COLS.length;
    const filtered = vs.filtered;
    const selectedName = state.selectedRow ? state.selectedRow.name : null;
    const selectedKey = state.selectedDataKey;
    const dataKey = vs.dataKey;
    const slotsPerRow = 2 + colCount * 3;
    const count = endIdx - startIdx;
    const parts = new Array(count * slotsPerRow + 2);
    let pi = 0;

    // Top spacer
    const topH = startIdx * ROW_H;
    parts[pi++] =
        '<tr style="height:' +
        topH +
        'px"><td colspan="' +
        colCount +
        '"></td></tr>';

    for (let ri = startIdx; ri < endIdx; ri++) {
        const row = filtered[ri];
        const isSelected = selectedName === row.name && selectedKey === dataKey;
        parts[pi++] =
            '<tr data-row-index="' +
            ri +
            '" role="row" tabindex="0"' +
            (isSelected
                ? ' class="selected" aria-selected="true"'
                : ' aria-selected="false"') +
            ">";
        for (let ci = 0; ci < colCount; ci++) {
            const col = COLS[ci];
            parts[pi++] = '<td class="' + (col.cls || "") + '">';
            if (col.key === "name") {
                parts[pi++] = row._nameEsc;
            } else if (col.fmt) {
                parts[pi++] = col.fmt(row[col.key]);
            } else {
                const v = row[col.key];
                parts[pi++] = v != null ? String(v) : "\u2014";
            }
            parts[pi++] = "</td>";
        }
        parts[pi++] = "</tr>";
    }

    // Bottom spacer
    const bottomH = (totalRows - endIdx) * ROW_H;
    parts[pi++] =
        '<tr style="height:' +
        bottomH +
        'px"><td colspan="' +
        colCount +
        '"></td></tr>';

    // nosemgrep: insecure-document-method -- name cells use the pre-escaped row._nameEsc, the rest are numbers
    tbody.innerHTML = parts.slice(0, pi).join("");
}

/* In-place cell update -- used when data changed but scroll range is stable */
function _updateVisibleRowsInPlace(vs, startIdx, endIdx, totalRows) {
    const tbody = vs.tbody;
    const colCount = COLS.length;
    const filtered = vs.filtered;
    const selectedName = state.selectedRow ? state.selectedRow.name : null;
    const selectedKey = state.selectedDataKey;
    const dataKey = vs.dataKey;

    // Children: [top-spacer, row0, row1, ..., rowN, bottom-spacer]
    const trs = tbody.children;
    const rowCount = endIdx - startIdx;

    for (let i = 0; i < rowCount; i++) {
        const tr = trs[i + 1]; // +1 to skip top spacer
        if (!tr || !tr.dataset || !tr.dataset.rowIndex) continue;

        const ri = startIdx + i;
        const row = filtered[ri];
        if (!row) continue;

        // Update row index in case sort order changed
        tr.dataset.rowIndex = String(ri);

        // If a different row now occupies this slot, clear per-cell caches so
        // stale values from the previous row don't suppress cell updates
        if (tr._cachedRowName !== row.name) {
            tr._cachedRowName = row.name;
            const tdsToReset = tr.children;
            for (let k = 0; k < tdsToReset.length; k++) {
                tdsToReset[k]._cachedVal = undefined;
            }
        }

        // Update selection state
        const isSelected = selectedName === row.name && selectedKey === dataKey;
        if (isSelected && !tr.classList.contains("selected")) {
            tr.classList.add("selected");
            tr.setAttribute("aria-selected", "true");
        } else if (!isSelected && tr.classList.contains("selected")) {
            tr.classList.remove("selected");
            tr.setAttribute("aria-selected", "false");
        }

        // Update each cell
        const tds = tr.children;
        for (let ci = 0; ci < colCount && ci < tds.length; ci++) {
            const col = COLS[ci];
            const td = tds[ci];
            const val = _formatCellValue(col, row);

            // Skip if value unchanged
            if (td._cachedVal === val.text) continue;
            td._cachedVal = val.text;

            /* `col.html` marks a column whose formatter emits markup (the cache
               bars), so it takes the same innerHTML path as the name cell. */
            if (col.key === "name" || col.html) {
                // nosemgrep: insecure-document-method -- val.html is row._nameEsc or numeric markup from col.fmt
                td.innerHTML = val.html;
            } else {
                td.textContent = val.text;
            }
        }
    }

    // Update bottom spacer height
    const bottomSpacer = trs[trs.length - 1];
    if (bottomSpacer) {
        const bottomH = (totalRows - endIdx) * ROW_H;
        bottomSpacer.style.height = bottomH + "px";
    }
}

/* -------------------------------------------------------------
   TABLE RENDERING -- virtual-scroll strategy
   Only rows visible in the scroll viewport (plus an overscan
   buffer) are rendered into the DOM.  Two spacer <tr> elements
   at the top and bottom of the <tbody> simulate the full table
   height so the scrollbar behaves correctly.
   ------------------------------------------------------------- */
function buildTable(containerId, tableId, rows, sortState) {
    const container = byId(containerId);
    if (!container) return;

    /* Pre-process every row (idempotent -- skips rows already done) */
    for (let i = 0; i < rows.length; i++) preprocessRow(rows[i]);

    const dataKey = containerId === "fn-scroll" ? "functions" : "rois";
    const colCount = COLS.length;
    const sorted = getSortedRows(dataKey, rows, sortState);
    const filtered = getFilteredRows(dataKey, sorted);

    /* -- Build or reuse the <table> skeleton -- */
    let tbl = byId(tableId);
    if (!tbl) {
        tbl = document.createElement("table");
        tbl.id = tableId;
        tbl.className = "data-tbl";

        const thead = tbl.createTHead();
        const hrow = thead.insertRow();
        for (let ci = 0; ci < colCount; ci++) {
            const col = COLS[ci];
            const th = document.createElement("th");
            th.textContent = col.label;
            th.dataset.colIdx = ci;
            th.addEventListener("click", () =>
                handleSort(containerId, tableId, ci),
            );
            hrow.appendChild(th);
        }

        tbl.createTBody();
        container.innerHTML = "";
        container.appendChild(tbl);

        // Attach scroll listener for virtual scrolling
        container.addEventListener(
            "scroll",
            () => renderVisibleRows(containerId),
            { passive: true },
        );
    }

    /* -- Update sort indicators -- */
    const cells = tbl.tHead.rows[0].cells;
    for (let ci = 0; ci < cells.length; ci++) {
        const th = cells[ci];
        if (th.classList.contains("sort-col")) th.classList.remove("sort-col");
        if (th.classList.contains("sort-asc")) th.classList.remove("sort-asc");
        if (th.classList.contains("sort-desc"))
            th.classList.remove("sort-desc");
        if (ci === sortState.colIdx) {
            th.classList.add(
                "sort-col",
                sortState.dir === "asc" ? "sort-asc" : "sort-desc",
            );
        }
    }

    /* -- Store virtual scroll state (preserve range to enable in-place updates) -- */
    const tbody = tbl.tBodies[0];
    const existingVs = vsState[containerId];

    if (existingVs && existingVs.tbody === tbody) {
        // Update in place -- preserve lastStart/lastEnd/lastTotal so
        // renderVisibleRows can still take the in-place path
        existingVs.filtered = filtered;
        existingVs.dataKey = dataKey;
        existingVs._dataChanged = true;
    } else {
        vsState[containerId] = {
            container: container,
            tbody: tbody,
            filtered: filtered,
            dataKey: dataKey,
            lastStart: -1,
            lastEnd: -1,
            lastTotal: -1,
            _dataChanged: true,
        };
    }

    // Add click handler for row selection (only once to prevent memory leaks)
    if (!tbody.dataset.hasClickHandler) {
        tbody.dataset.hasClickHandler = "true";
        tbody.addEventListener("click", (e) => {
            const tr = e.target.closest("tr");
            if (!tr || !tr.dataset.rowIndex) return;

            const rowIndex = parseInt(tr.dataset.rowIndex, 10);
            const vs = vsState[containerId];
            if (!vs) return;

            const row = vs.filtered[rowIndex];

            if (row) {
                selectRow(row, dataKey);
                // Re-render to update selection styling
                renderVisibleRows(container.id);
            }
        });
    }

    // Add keyboard navigation handler (only once to prevent memory leaks)
    if (!tbody.dataset.hasKeyHandler) {
        tbody.dataset.hasKeyHandler = "true";
        tbody.addEventListener("keydown", (e) => {
            const tr = e.target.closest("tr");
            if (!tr || !tr.dataset.rowIndex) return;

            const rowIndex = parseInt(tr.dataset.rowIndex, 10);
            const vs = vsState[containerId];
            if (!vs) return;

            // Enter or Space: select row
            if (e.key === "Enter" || e.key === " ") {
                e.preventDefault();
                const row = vs.filtered[rowIndex];
                if (row) {
                    selectRow(row, dataKey);
                    renderVisibleRows(container.id);
                }
            }
            // Arrow Down: focus next row
            else if (e.key === "ArrowDown") {
                e.preventDefault();
                const nextRow = tr.nextElementSibling;
                if (nextRow && nextRow.dataset.rowIndex) {
                    nextRow.focus();
                }
            }
            // Arrow Up: focus previous row
            else if (e.key === "ArrowUp") {
                e.preventDefault();
                const prevRow = tr.previousElementSibling;
                if (prevRow && prevRow.dataset.rowIndex) {
                    prevRow.focus();
                }
            }
        });
    }

    /* -- Render the visible slice -- */
    renderVisibleRows(containerId);
}

/* -------------------------------------------------------------
   SORT HANDLER
   ------------------------------------------------------------- */
function handleSort(containerId, tableId, colIdx) {
    const key = containerId === "fn-scroll" ? "functions" : "rois";
    const s = state.sort[key];
    if (s.colIdx === colIdx) {
        s.dir = s.dir === "asc" ? "desc" : "asc";
    } else {
        s.colIdx = colIdx;
        s.dir = "desc";
    }
    /* Invalidate sort cache so getSortedRows re-runs */
    const keyProp =
        "_sorted" + key.charAt(0).toUpperCase() + key.slice(1) + "Key";
    state[keyProp] = "";
    buildTable(containerId, tableId, state[key], s);
}

/* -------------------------------------------------------------
   FILTER INPUT HANDLING
   ------------------------------------------------------------- */
function setupFilter(inputId, containerId, tableId, dataKey) {
    const input = byId(inputId);
    if (!input) return;
    let debounceTimer = null;
    input.addEventListener("input", () => {
        clearTimeout(debounceTimer);
        debounceTimer = setTimeout(() => {
            state.filter[dataKey] = input.value;
            if (state[dataKey].length > 0) {
                buildTable(
                    containerId,
                    tableId,
                    state[dataKey],
                    state.sort[dataKey],
                );
            }
        }, 50);
    });
}

setupFilter("fn-filter", "fn-scroll", "tbl-functions", "functions");
setupFilter("roi-filter", "roi-scroll", "tbl-rois", "rois");

/* -------------------------------------------------------------
   FREEZE
   ------------------------------------------------------------- */
byId("freeze-cb").addEventListener("change", (e) => {
    state.frozen = e.target.checked;
    byId("data-panels").classList.toggle("frozen", state.frozen);
    byId("detail-pane").classList.toggle("frozen", state.frozen);
});

/* Insert the generated widget into the dashboard so it reflows to the full
   panel width instead of being constrained by an embedded document.

   Two views over the same run: the merged graph every run writes, and -- for a
   run whose threads each got their own graph (see write_per_hart_flamegraphs in
   workflow.rs) -- one graph per thread, stacked. The merged graph re-keys by
   region name, so several threads' executions of one region are a single frame
   there; the per-thread view is the only place a work imbalance between them is
   visible at all. */
(function initInlineRoiFlamegraph() {
    const slot = byId("roi-flamegraph-slot");
    const panel = byId("roi-flamegraph-panel");
    const toggle = byId("roi-flamegraph-toggle");
    const threadsBtn = byId("roi-flamegraph-threads");
    if (!slot || !panel || !toggle) return;

    const label = toggle.querySelector(".roi-flamegraph-toggle-label");
    const setExpanded = (expanded) => {
        toggle.setAttribute("aria-expanded", expanded ? "true" : "false");
        panel.hidden = !expanded;
        if (label) {
            label.textContent = expanded ? "Hide flamegraph" : "Show flamegraph";
        }
    };
    toggle.addEventListener("click", () => {
        setExpanded(toggle.getAttribute("aria-expanded") !== "true");
    });
    // Open by default: the graph is the fastest read of where the
    // instructions went, so it should not need a click to appear.
    setExpanded(true);

    const GRAPH_ID = "roi-flamegraph";
    const HART_PREFIX = GRAPH_ID + "-hart";

    /* The widget's markup, taken from the generated page and kept as the
       template every graph in the slot is cloned from. Cloning rather than
       building it here keeps roi_flamegraph.rs the one definition of the
       widget's DOM, instead of a second copy that has to be kept in step. */
    let template = null;
    let mergedData = null;
    /* roi_flamegraph_harts.json's entries, and each hart's graph JSON. Empty
       for a single-threaded run, which is what keeps the switch hidden. */
    let harts = [];
    const hartData = new Map();
    let perThread = false;

    /* Drop the registrations of graphs that are about to leave the DOM.
       syncFlamegraphSelection() walks window.ocfgGraphs, and a stale entry
       there closes over a detached node, so its select() paints nothing while
       still being called on every row click. */
    const forgetGraphs = () => {
        if (!window.ocfgGraphs) return;
        for (const id of [...window.ocfgGraphs.keys()]) {
            if (id === GRAPH_ID || id.startsWith(HART_PREFIX)) {
                window.ocfgGraphs.delete(id);
            }
        }
    };

    /* One widget under `id`, plus the data element ocfgInit reads. That has to
       be a real (non-executing) script node in the document, since ocfgInit
       looks it up by id -- hence a fragment rather than a single element.

       `entry` labels a per-thread graph, and is absent for the merged one. */
    const graphNode = (id, dataText, entry) => {
        const frag = document.createDocumentFragment();
        const widget = template.cloneNode(true);
        widget.id = id;

        /* The widget's own title slot is what carries the thread label, rather
           than a caption row of our own: a stack of graphs is only useful if
           several fit on screen at once, and a separate row per thread costs
           more height than the label needs. The merged graph has the panel's
           section header and needs no title at all. */
        const title = widget.querySelector(".ocfg-title");
        if (title && !entry) {
            title.remove();
        } else if (title) {
            const name = document.createElement("span");
            name.textContent = "hart " + entry.hart_id;
            // What the bar's own note does not already say. For a worker
            // thread this is its barrier/spin time -- the quantity the merged
            // graph can only give by subtraction.
            const outside = document.createElement("span");
            outside.className = "roi-flamegraph-hart-stat";
            const pct = entry.instructions
                ? ((entry.self_instructions / entry.instructions) * 100).toFixed(1)
                : "0.0";
            outside.textContent =
                fmtInt(entry.self_instructions) + " outside regions (" + pct + "%)";
            title.replaceChildren(name, outside);
        }

        const dataScript = document.createElement("script");
        dataScript.type = "application/json";
        dataScript.id = id + "-data";
        dataScript.textContent = dataText;
        frag.append(widget, dataScript);
        return frag;
    };

    /* ocfgInit resolves its container with getElementById, so every graph has
       to be in the document before any of them is initialised -- build the
       whole slot first, then walk the ids. */
    const render = () => {
        if (!template || mergedData === null) return;
        forgetGraphs();

        const ids = [];
        let threadView = false;
        const frag = document.createDocumentFragment();
        if (perThread && harts.length) {
            for (const entry of harts) {
                const data = hartData.get(entry.hart_id);
                if (data === undefined) continue;
                const id = HART_PREFIX + entry.hart_id;
                const box = document.createElement("div");
                box.className = "roi-flamegraph-hart";
                box.append(graphNode(id, data, entry));
                frag.append(box);
                ids.push(id);
            }
            threadView = ids.length > 0;
        }
        // Also the fallback if every per-hart fetch failed, so the switch can
        // never leave the panel empty.
        if (!ids.length) {
            frag.append(graphNode(GRAPH_ID, mergedData));
            ids.push(GRAPH_ID);
        }

        slot.replaceChildren(frag);
        /* Drives the per-thread layout rules: the height cap that stops the
           stack starving the ROI table, and the thread labels the merged
           layout hides. Keyed on the mode, not on the graph count, so a run
           where only one hart's graph could be read still gets its label. */
        slot.classList.toggle("per-thread", threadView);
        ids.forEach((id) => window.ocfgInit(id));
        /* A ROI may have been selected while the graphs were being fetched, or
           before this mode switch. Pushing it into all of them at once is the
           point of the per-thread view: one click lights up that region on
           every thread that ran it. */
        syncFlamegraphSelection();
    };

    if (threadsBtn) {
        threadsBtn.addEventListener("click", () => {
            perThread = threadsBtn.getAttribute("aria-pressed") !== "true";
            threadsBtn.setAttribute("aria-pressed", perThread ? "true" : "false");
            const btnLabel = threadsBtn.querySelector(".roi-flamegraph-toggle-label");
            if (btnLabel) btnLabel.textContent = perThread ? "Merged" : "Per thread";
            // Switching view with the panel collapsed is a click with no
            // visible effect.
            setExpanded(true);
            render();
        });
    }

    /* Take only the graph and its inline data out of the generated page.
       Its <style>/<script> copies are deliberately left behind: the widget's
       real assets are linked by index.html (a run's generated HTML freezes
       the versions from when it was written, and the first copy of
       flamegraph.js to run is the one that defines window.ocfgInit), and its
       head also carries the standalone page's own `body { padding }` rules,
       which have no business restyling the dashboard. */
    const adopt = (html) => {
        const doc = new DOMParser().parseFromString(html, "text/html");
        const widget = doc.querySelector(".ocfg");
        const data = doc.querySelector('script[type="application/json"]');
        if (!widget || !widget.id || !data || typeof window.ocfgInit !== "function") {
            return false;
        }
        template = document.importNode(widget, true);
        mergedData = data.textContent;
        render();
        return true;
    };

    /* The per-thread files are written just after the merged page, so a poll
       can land between the two. Retry a few times rather than concluding from
       one 404 that this was a single-threaded run and hiding the switch for
       good; a truncated read of a file mid-write parses as failure too.
       Missing the switch on a run that has per-thread graphs is much worse
       than the cost of being wrong the other way, which is that a
       single-threaded run logs a handful of 404s for this file and stops. */
    const loadHarts = (triesLeft) => {
        fetch("/out/roi_flamegraph_harts.json")
            .then((r) => (r.ok ? r.json() : null))
            .then((index) => {
                const entries = index && Array.isArray(index.harts) ? index.harts : null;
                if (!entries || !entries.length) throw new Error("no per-thread index");
                /* The switch appears only once every graph it would show is
                   in hand, so pressing it cannot land on a half-populated
                   view. write_per_hart_flamegraphs writes the index *after*
                   every graph it lists, so reaching here means they exist --
                   the only reason one would be missing is a real error. */
                return Promise.all(
                    entries.map((e) =>
                        fetch("/out/" + e.json)
                            .then((r) => (r.ok ? r.text() : null))
                            .then((text) => {
                                if (text === null) return;
                                // Validated here rather than left to
                                // ocfgInit, which parses without a guard: one
                                // unreadable file would throw mid-loop and
                                // take every graph after it down with it.
                                try {
                                    JSON.parse(text);
                                } catch {
                                    return;
                                }
                                hartData.set(e.hart_id, text);
                            }),
                    ),
                ).then(() => {
                    harts = entries.filter((e) => hartData.has(e.hart_id));
                    if (harts.length && threadsBtn) threadsBtn.hidden = false;
                });
            })
            .catch(() => {
                if (triesLeft > 0) setTimeout(() => loadHarts(triesLeft - 1), 1000);
            });
    };

    const poll = () => {
        fetch("/out/roi_flamegraph.html")
            .then((response) => {
                if (!response.ok) return null;
                return response.text();
            })
            .then((html) => {
                // Nothing is inserted until the whole widget is in hand, so
                // a failed attempt leaves the placeholder alone.
                if (!html || !adopt(html)) {
                    setTimeout(poll, 1000);
                    return;
                }
                loadHarts(3);
            })
            .catch(() => setTimeout(poll, 1000));
    };
    poll();
})();

/* -------------------------------------------------------------
   SSE DATA HANDLERS
   ------------------------------------------------------------- */
function onFunctionsData(payload) {
    if (!Array.isArray(payload.functions)) return;

    deriveRowMetrics(payload.functions);

    // Fingerprint check -- skip all DOM work if data is unchanged
    const fp = computeFingerprint(payload.functions);
    const dataChanged = fp !== state._lastFunctionsFingerprint;
    state._lastFunctionsFingerprint = fp;
    state.functions = payload.functions;

    // Update detail pane if a function is selected (only when data changed and not frozen)
    if (
        dataChanged &&
        !state.frozen &&
        state.selectedDataKey === "functions" &&
        state.selectedRow
    ) {
        const updatedRow = payload.functions.find(
            (r) => r.name === state.selectedRow.name,
        );
        if (updatedRow) {
            state.selectedRow = updatedRow;
            renderDetailPane(updatedRow, "functions");
        }
    }

    if (state.frozen || !dataChanged) return;

    /* Invalidate sort cache */
    state._sortedFunctionsKey = "";

    const ph = byId("fn-placeholder");
    if (ph) ph.style.display = "none";
    byId("fn-info").textContent =
        _intFmt.format(payload.functions.length) + " rows";
    buildTable(
        "fn-scroll",
        "tbl-functions",
        state.functions,
        state.sort.functions,
    );
}

function onRoisData(payload) {
    if (!Array.isArray(payload.rois)) return;

    deriveRowMetrics(payload.rois);

    // Fingerprint check -- skip all DOM work if data is unchanged
    const fp = computeFingerprint(payload.rois);
    const dataChanged = fp !== state._lastRoisFingerprint;
    state._lastRoisFingerprint = fp;
    state.rois = payload.rois;

    // Update detail pane if a ROI is selected (only when data changed and not frozen)
    if (
        dataChanged &&
        !state.frozen &&
        state.selectedDataKey === "rois" &&
        state.selectedRow
    ) {
        const updatedRow = payload.rois.find(
            (r) => r.name === state.selectedRow.name,
        );
        if (updatedRow) {
            state.selectedRow = updatedRow;
            renderDetailPane(updatedRow, "rois");
        }
    }

    if (state.frozen || !dataChanged) return;

    /* Invalidate sort cache */
    state._sortedRoisKey = "";

    const ph = byId("roi-placeholder");
    if (ph) ph.style.display = "none";
    byId("roi-info").textContent =
        _intFmt.format(payload.rois.length) + " rows";
    buildTable("roi-scroll", "tbl-rois", state.rois, state.sort.rois);
}

const DATA_HANDLERS = {
    functions: onFunctionsData,
    rois: onRoisData,
};

function processSseData(sseData) {
    if (!sseData.data || !Array.isArray(sseData.data)) {
        console.warn("[SSE] Unexpected envelope:", sseData);
        return;
    }
    for (let i = 0; i < sseData.data.length; i++) {
        const item = sseData.data[i];
        const handler = DATA_HANDLERS[item.type];
        if (handler) handler(item.data);
        else console.warn("[SSE] No handler for type:", item.type);
    }
}

/* -------------------------------------------------------------
   DETAIL PANE
   ------------------------------------------------------------- */

/* ---------- LMUL colour palette ----------
   Each LMUL gets a distinct colour that works in both dark and light themes.
   Index 0 is the "combined" (all-LMUL) histogram for each SEW.             */
const LMUL_COLORS_DARK = {
    combined: "#5ab3f0", // accent blue (same as --accent dark)
    m1: "#5ab3f0", // blue
    m2: "#7ecb6a", // green
    m4: "#f0c75a", // amber/gold
    m8: "#ef7a6d", // coral/red
    mf2: "#c491e0", // lavender
    mf4: "#5ad8d2", // teal
    mf8: "#e08a5a", // orange
};
const LMUL_COLORS_LIGHT = {
    combined: "#1565c0", // accent blue (same as --accent light)
    m1: "#1565c0", // blue
    m2: "#2e7d32", // green
    m4: "#c17000", // amber
    m8: "#c62828", // red
    mf2: "#7b1fa2", // purple
    mf4: "#00838f", // teal
    mf8: "#d84315", // deep orange
};

function _isDarkTheme() {
    var el = document.documentElement;
    if (el.getAttribute("data-theme") === "light") return false;
    if (el.getAttribute("data-theme") === "dark") return true;
    return window.matchMedia("(prefers-color-scheme: dark)").matches;
}

function lmulColor(lmulKey) {
    var palette = _isDarkTheme() ? LMUL_COLORS_DARK : LMUL_COLORS_LIGHT;
    return palette[lmulKey] || palette.combined;
}

const LMUL_KEYS = ["m1", "m2", "m4", "m8", "mf2", "mf4", "mf8"];
const LMUL_LABELS = {
    m1: "LMUL 1",
    m2: "LMUL 2",
    m4: "LMUL 4",
    m8: "LMUL 8",
    mf2: "LMUL 1/2",
    mf4: "LMUL 1/4",
    mf8: "LMUL 1/8",
};

const SEW_BASES = ["e8", "e16", "e32", "e64"];

/* Build HIST_FIELDS: one chart per SEW x LMUL combination.
   Only entries whose _raw data is non-empty will render.    */
const HIST_FIELDS = [];
(function () {
    for (var si = 0; si < SEW_BASES.length; si++) {
        var sew = SEW_BASES[si];
        for (var li = 0; li < LMUL_KEYS.length; li++) {
            var lk = LMUL_KEYS[li];
            HIST_FIELDS.push({
                label: "VL " + sew + " : " + LMUL_LABELS[lk],
                rawKey: "vl_hist_" + sew + "_" + lk + "_raw",
                lmulKey: lk,
                sewGroup: sew,
            });
        }
    }
})();

let tooltipEl = null;
let _hoveredCanvas = null;

function getTooltip() {
    if (!tooltipEl) {
        tooltipEl = document.createElement("div");
        tooltipEl.className = "histogram-tooltip";
        document.body.appendChild(tooltipEl);
    }
    return tooltipEl;
}

// sparse = [[vl, count], ...] pairs
// returns { total, mean, p50, p95, min, max }
function computeSparseStats(sparse) {
    if (!sparse || sparse.length === 0) {
        return { total: 0, mean: 0, p50: 0, p95: 0, min: 0, max: 0 };
    }
    var sorted = sparse.slice().sort(function (a, b) {
        return a[0] - b[0];
    });
    var total = 0;
    var i;
    for (i = 0; i < sorted.length; i++) total += sorted[i][1];
    if (total === 0) {
        return { total: 0, mean: 0, p50: 0, p95: 0, min: 0, max: 0 };
    }
    var wsum = 0;
    for (i = 0; i < sorted.length; i++) wsum += sorted[i][0] * sorted[i][1];
    var mean = wsum / total;
    var minVl = sorted[0][0];
    var maxVl = sorted[sorted.length - 1][0];
    var p50 = 0,
        p95 = 0;
    var target50 = total * 0.5;
    var target95 = total * 0.95;
    var cum = 0;
    var found50 = false,
        found95 = false;
    for (i = 0; i < sorted.length; i++) {
        cum += sorted[i][1];
        if (!found50 && cum >= target50) {
            p50 = sorted[i][0];
            found50 = true;
        }
        if (!found95 && cum >= target95) {
            p95 = sorted[i][0];
            found95 = true;
        }
        if (found50 && found95) break;
    }
    return {
        total: total,
        mean: mean,
        p50: p50,
        p95: p95,
        min: minVl,
        max: maxVl,
    };
}

// Internal draw function. Returns bar column metadata array for hover.
function _drawHistogramOnCanvas(canvas, row, field) {
    var dpr = window.devicePixelRatio || 1;
    var rect = canvas.getBoundingClientRect();
    var cssW = rect.width > 0 ? rect.width : 300;
    var cssH = rect.height > 0 ? rect.height : 140;
    canvas.width = Math.round(cssW * dpr);
    canvas.height = Math.round(cssH * dpr);
    var ctx = canvas.getContext("2d");
    ctx.scale(dpr, dpr);

    var w = cssW;
    var h = cssH;
    var padL = 8,
        padR = 8,
        padT = 18,
        padB = 22;
    var chartW = w - padL - padR;
    var chartH = h - padT - padB;

    var cs = getComputedStyle(document.documentElement);
    var accentColor = cs.getPropertyValue("--accent").trim();
    var dimColor = cs.getPropertyValue("--dim").trim();
    var borderColor = cs.getPropertyValue("--border").trim();
    var surface2 = cs.getPropertyValue("--surface2").trim();

    ctx.clearRect(0, 0, w, h);
    ctx.fillStyle = surface2;
    ctx.fillRect(0, 0, w, h);

    var rawData = row[field.rawKey];
    var useRaw = Array.isArray(rawData) && rawData.length > 0;

    // Pick bar colour: per-LMUL colour when available, otherwise accent
    var barColor = field.lmulKey ? lmulColor(field.lmulKey) : accentColor;

    if (!useRaw) {
        ctx.fillStyle = dimColor;
        ctx.font = "11px sans-serif";
        ctx.textAlign = "center";
        ctx.textBaseline = "middle";
        ctx.fillText("No data", w / 2, h / 2);
        return [];
    }

    var bars = [];
    var maxCount = 0;
    var i;

    var sorted = rawData.slice().sort(function (a, b) {
        return a[0] - b[0];
    });
    for (i = 0; i < sorted.length; i++) {
        if (sorted[i][1] > maxCount) maxCount = sorted[i][1];
    }
    if (maxCount === 0) {
        ctx.fillStyle = dimColor;
        ctx.font = "11px sans-serif";
        ctx.textAlign = "center";
        ctx.textBaseline = "middle";
        ctx.fillText("All zeros", w / 2, h / 2);
        return [];
    }
    var n = sorted.length;
    var colW = chartW / n;
    var gap = Math.min(2, colW * 0.1);
    for (i = 0; i < n; i++) {
        bars.push({
            colStart: i * colW,
            colEnd: (i + 1) * colW,
            barX: padL + i * colW + gap / 2,
            barW: Math.max(1, colW - gap),
            fraction: sorted[i][1] / maxCount,
            count: sorted[i][1],
            label: String(sorted[i][0]),
        });
    }

    // Horizontal gridlines at 100% and 50%
    ctx.strokeStyle = borderColor;
    ctx.lineWidth = 0.5;
    ctx.setLineDash([3, 3]);
    ctx.beginPath();
    ctx.moveTo(padL, padT);
    ctx.lineTo(padL + chartW, padT);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(padL, padT + chartH * 0.5);
    ctx.lineTo(padL + chartW, padT + chartH * 0.5);
    ctx.stroke();
    ctx.setLineDash([]);

    // Draw bars
    var hoveredIdx = canvas._hoveredIdx !== undefined ? canvas._hoveredIdx : -1;
    for (i = 0; i < bars.length; i++) {
        if (bars[i].count === 0) continue;
        var barH = bars[i].fraction * chartH;
        var barY = padT + chartH - barH;
        if (i === hoveredIdx) {
            ctx.fillStyle = "#c0e8ff";
            ctx.globalAlpha = 1.0;
        } else {
            ctx.fillStyle = barColor;
            ctx.globalAlpha = 0.85;
        }
        ctx.fillRect(bars[i].barX, barY, bars[i].barW, barH);
    }
    ctx.globalAlpha = 1.0;

    // X-axis tick labels (only when column wide enough)
    ctx.fillStyle = dimColor;
    ctx.font = "9px sans-serif";
    ctx.textAlign = "center";
    ctx.textBaseline = "top";
    var minColForLabel = 28;
    for (i = 0; i < bars.length; i++) {
        if (bars[i].count === 0) continue;
        var cw = bars[i].colEnd - bars[i].colStart;
        if (cw < minColForLabel) continue;
        var labelX = padL + (bars[i].colStart + bars[i].colEnd) / 2;
        ctx.fillText(bars[i].label, labelX, padT + chartH + 3);
    }

    // Max count label top-left
    ctx.fillStyle = dimColor;
    ctx.font = "9px sans-serif";
    ctx.textAlign = "left";
    ctx.textBaseline = "top";
    ctx.fillText("max:" + fmtInt(maxCount), padL, 2);

    return bars;
}

/* Build the tooltip content for the currently hovered bar of `canvas`.
   Returns false when there is no valid hovered bar (caller hides the tooltip),
   true when `tooltip.innerHTML` was populated. Does NOT reposition the tooltip
   -- that stays where the last mousemove placed it. */
function _fillHistogramTooltip(canvas, tooltip) {
    var curBars = canvas._bars;
    var found = canvas._hoveredIdx;
    if (!curBars || found < 0 || found >= curBars.length) return false;
    var bar = curBars[found];
    if (!bar || bar.count <= 0) return false;

    var total = 0;
    var rd = canvas._row[canvas._field.rawKey];
    if (Array.isArray(rd)) {
        for (var j = 0; j < rd.length; j++) total += rd[j][1];
    }
    var pct = total > 0 ? ((bar.count / total) * 100).toFixed(1) : "0.0";
    // nosemgrep: insecure-document-method -- escHtml on the label, fmtInt/toFixed on the numbers
    tooltip.innerHTML =
        '<span class="histogram-tooltip-bucket">VL ' +
        escHtml(bar.label) +
        "</span>" +
        '<span class="histogram-tooltip-count">' +
        fmtInt(bar.count) +
        "</span>" +
        '<span style="color:var(--dim);margin-left:6px">' +
        pct +
        "%</span>";
    return true;
}

/* Refresh the tooltip in place for the currently hovered canvas after its
   underlying data changed (e.g. a live SSE update redrew the histogram).
   Keeps the tooltip at its current position. */
function refreshHoveredTooltip() {
    if (!_hoveredCanvas) return;
    var tooltip = getTooltip();
    if (!tooltip.classList.contains("visible")) return;
    if (!_fillHistogramTooltip(_hoveredCanvas, tooltip)) {
        tooltip.classList.remove("visible");
    }
}

function drawAndBindHistogram(canvas, row, field) {
    canvas._hoveredIdx = -1;
    canvas._row = row;
    canvas._field = field;
    var bars = _drawHistogramOnCanvas(canvas, row, field);
    canvas._bars = bars;

    var tooltip = getTooltip();

    canvas.addEventListener("mousemove", function (e) {
        _hoveredCanvas = canvas;
        var rect = canvas.getBoundingClientRect();
        var mouseX = e.clientX - rect.left;
        var padL = 8,
            padR = 8;
        var chartW = rect.width - padL - padR;
        var curBars = canvas._bars;

        if (
            !curBars ||
            curBars.length === 0 ||
            mouseX < padL ||
            mouseX > padL + chartW
        ) {
            tooltip.classList.remove("visible");
            if (canvas._hoveredIdx !== -1) {
                canvas._hoveredIdx = -1;
                canvas._bars = _drawHistogramOnCanvas(
                    canvas,
                    canvas._row,
                    canvas._field,
                );
            }
            return;
        }

        var chartMouseX = mouseX - padL;
        var found = -1;
        for (var i = 0; i < curBars.length; i++) {
            if (
                chartMouseX >= curBars[i].colStart &&
                chartMouseX <= curBars[i].colEnd
            ) {
                found = i;
                break;
            }
        }

        if (found !== canvas._hoveredIdx) {
            canvas._hoveredIdx = found;
            canvas._bars = _drawHistogramOnCanvas(
                canvas,
                canvas._row,
                canvas._field,
            );
            curBars = canvas._bars;
        }

        if (found >= 0 && curBars[found].count > 0) {
            _fillHistogramTooltip(canvas, tooltip);
            tooltip.classList.add("visible");

            var tooltipRect = tooltip.getBoundingClientRect();
            var tooltipWidth = tooltipRect.width;
            var tooltipHeight = tooltipRect.height;

            var mouseX = e.clientX;
            var mouseY = e.clientY;
            var offsetY = 18;

            var left = mouseX - tooltipWidth / 2;
            var top = mouseY + offsetY;

            var viewportWidth = window.innerWidth;
            var viewportHeight = window.innerHeight;

            if (left < 5) {
                left = 5;
            } else if (left + tooltipWidth > viewportWidth - 5) {
                left = viewportWidth - tooltipWidth - 5;
            }

            if (top + tooltipHeight > viewportHeight - 5) {
                top = mouseY - tooltipHeight - offsetY;
            }

            tooltip.style.left = left + "px";
            tooltip.style.top = top + "px";
        } else {
            tooltip.classList.remove("visible");
        }
    });

    canvas.addEventListener("mouseleave", function () {
        tooltip.classList.remove("visible");
        canvas._hoveredIdx = -1;
        canvas._bars = _drawHistogramOnCanvas(
            canvas,
            canvas._row,
            canvas._field,
        );
        if (_hoveredCanvas === canvas) _hoveredCanvas = null;
    });
}

function renderDetailPane(row, dataKey) {
    var content = byId("detail-content");
    if (!row) {
        content.innerHTML =
            '<div class="detail-placeholder">Select a ROI or Function to view detailed histogram information</div>';
        state._detailPaneName = null;
        state._detailPaneKey = null;
        return;
    }

    // Same row and same data type -- update values in-place (preserves text selection)
    if (
        state._detailPaneName === row.name &&
        state._detailPaneKey === dataKey
    ) {
        _updateDetailPaneInPlace(content, row);
        return;
    }

    // Different row or different data type -- full rebuild
    state._detailPaneName = row.name;
    state._detailPaneKey = dataKey;
    _buildDetailPaneFull(content, row, dataKey);
}

/* Full detail pane build -- creates skeleton with stable IDs for in-place updates */
/* Build the collapsed-by-default "Instruction Mix" expander: per-opcode
   execution counts (mnemonic -> count), sorted by count descending. Returns
   an empty string when the row carries no instruction_mix. */
function _instructionMixEntries(row) {
    var mix = row.instruction_mix;
    if (!mix || typeof mix !== "object") return null;
    var entries = Object.keys(mix).map(function (k) {
        return [k, mix[k]];
    });
    if (entries.length === 0) return null;
    entries.sort(function (a, b) {
        return b[1] - a[1] || (a[0] < b[0] ? -1 : a[0] > b[0] ? 1 : 0);
    });
    return entries;
}

function _instructionMixBodyHtml(entries) {
    var total = 0;
    for (var i = 0; i < entries.length; i++) total += entries[i][1];

    var html = "";
    for (var j = 0; j < entries.length; j++) {
        var mnemonic = entries[j][0];
        var count = entries[j][1];
        var pct = total > 0 ? (count / total) * 100 : 0;
        html += '<div class="instr-mix-row">';
        html += '<span class="instr-mix-mnemonic">' + escHtml(mnemonic) + "</span>";
        html += '<span class="instr-mix-value">' + fmtInt(count) + "</span>";
        html += '<span class="instr-mix-pct">' + pct.toFixed(1) + "%</span>";
        html += "</div>";
    }
    return html;
}

function _buildInstructionMixHtml(row) {
    var entries = _instructionMixEntries(row);
    if (!entries) return "";

    var html = '<div class="detail-section detail-instr-mix-section">';
    html +=
        '<details class="instr-mix" id="dp-instr-mix"' +
        (state._instrMixOpen ? " open" : "") +
        ">";
    html +=
        '<summary class="detail-section-title instr-mix-summary">' +
        "Instruction Mix" +
        '<span class="instr-mix-count">' +
        fmtInt(entries.length) +
        " opcodes</span></summary>";
    html += '<div class="instr-mix-body" id="dp-instr-mix-body">';
    html += _instructionMixBodyHtml(entries);
    html += "</div></details></div>";
    return html;
}

/* The "Cache Stats" expander: the same two bars the table cell draws, plus the
   absolute hit/miss counts the cell has no room for. Absent entirely on a row
   with no modelled cache -- an empty expander is worse than no expander. */
function _buildCacheStatsHtml(row) {
    if (!row.cache_stats) return "";

    var html = '<div class="detail-section detail-cache-stats-section">';
    html +=
        '<details class="cache-stats" id="dp-cache-stats"' +
        (state._cacheStatsOpen ? " open" : "") +
        ">";
    html +=
        '<summary class="detail-section-title cache-stats-summary">' +
        "Cache Stats" +
        '<span class="cache-stats-count">' +
        fmtInt(row.cache_stats.axis) +
        " accesses</span></summary>";
    html += '<div class="cache-stats-body" id="dp-cache-stats-body">';
    html += _cacheStatsBodyHtml(row.cache_stats);
    html += "</div></details></div>";
    return html;
}

/* Filesystem-safe stem for a name, mirroring Rust annotate::sanitize():
   keep [A-Za-z0-9_.], replace every other char with '_'. */
function annotateFileFor(name) {
    return "annotate_" + String(name || "").replace(/[^A-Za-z0-9_.]/g, "_") + ".html";
}

function _buildDetailPaneFull(content, row, dataKey) {
    var html = '<div class="detail-name">' + escHtml(row.name || "") + "</div>";

    // Link into the source & assembly annotation report (served in-memory
    // under /annotate/ by the HTTP router). The per-item page is keyed by
    // function, so only show the direct button for functions; ROIs (region
    // names) have no page and would 404. The annotation index and hotspots
    // pages live in the Overview panel now, so they're not repeated here.
    if (dataKey === "functions") {
        html += '<div class="detail-annotate">';
        html +=
            '<a class="annotate-btn" target="_blank" rel="noopener" href="/annotate/' +
            encodeURIComponent(annotateFileFor(row.name)) +
            '">Source &amp; ASM annotation</a>';
        html += "</div>";
    }

    html += '<div class="detail-top-row">';
    html += '<div class="detail-section detail-stats-section">';
    html += '<div class="detail-section-title">Statistics</div>';
    html += '<div class="detail-stats">';
    html +=
        '<div class="detail-stat"><div class="detail-stat-label">Instructions</div>' +
        '<div class="detail-stat-value" id="dp-instructions">' +
        fmtInt(row.instructions) +
        "</div></div>";
    html +=
        '<div class="detail-stat"><div class="detail-stat-label">Float Ops</div>' +
        '<div class="detail-stat-value" id="dp-float_elements">' +
        fmtInt(row.float_elements) +
        "</div></div>";
    html +=
        '<div class="detail-stat"><div class="detail-stat-label">Vec Instr</div>' +
        '<div class="detail-stat-value" id="dp-vector_instructions">' +
        fmtInt(row.vector_instructions) +
        "</div></div>";
    html +=
        '<div class="detail-stat"><div class="detail-stat-label">Bytes Moved</div>' +
        '<div class="detail-stat-value" id="dp-bytes_moved">' +
        fmtBytes(row.bytes_moved) +
        "</div></div>";
    html +=
        '<div class="detail-stat"><div class="detail-stat-label">FLOPs/Byte</div>' +
        '<div class="detail-stat-value" id="dp-flops_per_byte">' +
        fmtFloat(row.flops_per_byte) +
        "</div></div>";
    html += "</div></div>";

    html += _buildInstructionMixHtml(row);

    html += "</div>"; // close .detail-top-row

    /* Full width below the top row: the bars want a wide track, and the top
       row's two sections already fill it. */
    html += _buildCacheStatsHtml(row);

    html += '<div class="detail-section">';
    html += '<div class="detail-section-title">Vector Length Histograms</div>';

    var histCount = 0;
    for (var i = 0; i < HIST_FIELDS.length; i++) {
        var field = HIST_FIELDS[i];
        var rawData = row[field.rawKey];
        if (!Array.isArray(rawData) || rawData.length === 0) continue;
        histCount++;

        var stats = computeSparseStats(rawData);

        html += '<div class="histogram-container">';
        html +=
            '<div class="histogram-title">' + escHtml(field.label) + "</div>";
        html +=
            '<div class="histogram-summary" id="dp-hist-summary-' + i + '">';
        html +=
            '<span id="dp-hist-avg-' +
            i +
            '">Avg: ' +
            fmtFloat(stats.mean) +
            "</span>";
        html +=
            '<span id="dp-hist-max-' +
            i +
            '">Max: ' +
            fmtInt(stats.max) +
            "</span>";
        html +=
            '<span id="dp-hist-total-' +
            i +
            '">Total: ' +
            fmtInt(stats.total) +
            "</span>";
        html +=
            '<span id="dp-hist-p50-' +
            i +
            '">p50: ' +
            fmtInt(stats.p50) +
            "</span>";
        html +=
            '<span id="dp-hist-p95-' +
            i +
            '">p95: ' +
            fmtInt(stats.p95) +
            "</span>";
        html += "</div>";
        html +=
            '<canvas class="histogram-canvas"' +
            ' data-field="' +
            escHtml(field.rawKey) +
            '"' +
            ' data-idx="' +
            i +
            '"' +
            ' role="img"' +
            ' aria-label="' +
            escHtml(field.label + " distribution histogram") +
            '"' +
            "></canvas>";
        html += "</div>";
    }

    if (histCount === 0) {
        html +=
            '<div class="detail-empty-note">No vector instructions executed</div>';
    }

    html += "</div>";
    // nosemgrep: insecure-document-method -- html is assembled from escHtml'd labels and keys
    content.innerHTML = html;

    // Remember the Instruction Mix expander state so it carries across ROIs
    var mixDetails = byId("dp-instr-mix");
    if (mixDetails) {
        mixDetails.addEventListener("toggle", function () {
            state._instrMixOpen = mixDetails.open;
        });
    }

    // Likewise for the Cache Stats expander
    var cacheDetails = byId("dp-cache-stats");
    if (cacheDetails) {
        cacheDetails.addEventListener("toggle", function () {
            state._cacheStatsOpen = cacheDetails.open;
        });
    }

    var canvases = content.querySelectorAll("canvas.histogram-canvas");
    for (var ci = 0; ci < canvases.length; ci++) {
        var cv = canvases[ci];
        var idx = parseInt(cv.getAttribute("data-idx"), 10);
        if (idx >= 0 && idx < HIST_FIELDS.length) {
            drawAndBindHistogram(cv, row, HIST_FIELDS[idx]);
        }
    }
}

/* In-place detail pane update -- only touches changed values */
function _updateDetailPaneInPlace(content, row) {
    var el;
    el = byId("dp-instructions");
    if (el) el.textContent = fmtInt(row.instructions);
    el = byId("dp-float_elements");
    if (el) el.textContent = fmtInt(row.float_elements);
    el = byId("dp-vector_instructions");
    if (el) el.textContent = fmtInt(row.vector_instructions);
    el = byId("dp-bytes_moved");
    if (el) el.textContent = fmtBytes(row.bytes_moved);
    el = byId("dp-flops_per_byte");
    if (el) el.textContent = fmtFloat(row.flops_per_byte);
    /* Present only when the pane was built for a row that had cache data. A row
       that gains it mid-run (the first modelled access lands in a later batch)
       gets the section on the next full rebuild, which a selection change
       forces. Rewriting the body preserves the <details> open/closed state. */
    el = byId("dp-cache-stats-body");
    if (el && row.cache_stats) {
        // nosemgrep: insecure-document-method -- _cacheStatsBodyHtml renders numbers only
        el.innerHTML = _cacheStatsBodyHtml(row.cache_stats);
        var axisEl = el.parentNode.querySelector(".cache-stats-count");
        if (axisEl)
            axisEl.textContent = fmtInt(row.cache_stats.axis) + " accesses";
    }

    // Refresh instruction-mix body (preserves the <details> open/closed state)
    el = byId("dp-instr-mix-body");
    if (el) {
        var mixEntries = _instructionMixEntries(row);
        if (mixEntries) {
            // nosemgrep: insecure-document-method -- _instructionMixBodyHtml escapes the mnemonic
            el.innerHTML = _instructionMixBodyHtml(mixEntries);
            var cntEl = el.parentNode.querySelector(".instr-mix-count");
            if (cntEl) cntEl.textContent = fmtInt(mixEntries.length) + " opcodes";
        }
    }

    // Update histogram summaries and redraw canvases
    for (var i = 0; i < HIST_FIELDS.length; i++) {
        var field = HIST_FIELDS[i];
        var rawData = row[field.rawKey];
        var hasRaw = Array.isArray(rawData) && rawData.length > 0;

        if (byId("dp-hist-summary-" + i)) {
            var stats = computeSparseStats(hasRaw ? rawData : null);
            el = byId("dp-hist-avg-" + i);
            if (el) el.textContent = "Avg: " + fmtFloat(stats.mean);
            el = byId("dp-hist-max-" + i);
            if (el) el.textContent = "Max: " + fmtInt(stats.max);
            el = byId("dp-hist-total-" + i);
            if (el) el.textContent = "Total: " + fmtInt(stats.total);
            el = byId("dp-hist-p50-" + i);
            if (el) el.textContent = "p50: " + fmtInt(stats.p50);
            el = byId("dp-hist-p95-" + i);
            if (el) el.textContent = "p95: " + fmtInt(stats.p95);
        }
    }

    // Redraw histogram canvases with updated row data
    var canvases = content.querySelectorAll("canvas.histogram-canvas");
    for (var ci = 0; ci < canvases.length; ci++) {
        var cv = canvases[ci];
        cv._row = row;
        cv._bars = _drawHistogramOnCanvas(cv, row, cv._field);
    }

    // If a tooltip is currently shown over one of these histograms, refresh
    // its text to reflect the newly redrawn data.
    refreshHoveredTooltip();
}

function selectRow(row, dataKey) {
    state.selectedRow = row;
    state.selectedDataKey = dataKey;
    renderDetailPane(row, dataKey);

    // Mark both tables dirty so renderVisibleRows re-evaluates selection on all visible rows
    for (const cid of Object.keys(vsState)) {
        vsState[cid]._dataChanged = true;
    }

    syncFlamegraphSelection();
}

/* -------------------------------------------------------------
   ROI TABLE <-> FLAMEGRAPH SELECTION BRIDGE
   The two views share only the region name, so that is the link:
   a ROI row highlights every frame of that region, and a frame
   click selects the row of the same name.
   ------------------------------------------------------------- */

/* Push the current selection into every flamegraph on the page. Selecting a
   function (or nothing) clears the highlight, since no frame corresponds to it.

   Safe to call before the widget exists -- the fragment is fetched
   asynchronously, so a row can be clicked first; initInlineRoiFlamegraph calls
   this again once the graph has registered itself. */
function syncFlamegraphSelection() {
    const graphs = window.ocfgGraphs;
    if (!graphs) return;
    const name =
        state.selectedDataKey === "rois" && state.selectedRow
            ? state.selectedRow.name
            : null;
    graphs.forEach((g) => g.select(name));
}

/* Scroll a table row into view by name, for a selection that came from outside
   the table. Virtual scrolling means the row may not be in the DOM at all, so
   the offset comes from its index in the filtered list rather than from an
   element. A row the filter has excluded is simply left alone: the detail pane
   still updates, and clearing the filter brings the highlight back. */
function revealTableRow(containerId, name) {
    const vs = vsState[containerId];
    if (!vs) return;
    const idx = vs.filtered.findIndex((r) => r.name === name);
    if (idx < 0) return;

    const container = vs.container;
    const top = idx * ROW_H;
    const viewH = container.clientHeight;
    if (top < container.scrollTop || top + ROW_H > container.scrollTop + viewH) {
        // Centre it: the <thead> is sticky, so a row parked at the very top of
        // the scroll box would sit underneath it.
        container.scrollTop = Math.max(0, top - (viewH - ROW_H) / 2);
    }
    renderVisibleRows(containerId);
}

/* A frame click anywhere on the page (the event bubbles out of the widget's
   container) selects the matching ROI row. `name` is a region name, so it
   matches at most one row; "Global" is a real ROI row, and a name with no row
   at all -- a region the ROI table has not seen yet -- is ignored. */
document.addEventListener("ocfg:select", (ev) => {
    const name = ev.detail && ev.detail.name;
    if (!name) return;
    const row = state.rois.find((r) => r.name === name);
    if (!row) return;
    selectRow(row, "rois");
    revealTableRow("roi-scroll", name);
});

/* -------------------------------------------------------------
   SSE CONNECTION
   ------------------------------------------------------------- */
const dot = byId("sse-dot");
const label = byId("sse-label");
const updInfo = byId("upd-info");
const updTime = byId("upd-time");

let es = null;

function setStatus(cls, text) {
    dot.className = "status-dot " + cls;
    label.textContent = text;
}

const _timeFmt = new Intl.DateTimeFormat(undefined, {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
});

function connectSSE() {
    if (es) es.close();

    setStatus("reconnecting", "Connecting...");
    /* Relative: the dashboard is served by the same process, so this keeps
       working when the page is opened from another host or a forwarded port. */
    es = new EventSource("/sse");

    es.onopen = () => {
        setStatus("connected", "Live");
    };

    es.onmessage = (e) => {
        try {
            const data = JSON.parse(e.data);
            processSseData(data);
            updInfo.style.display = "";
            updTime.textContent = _timeFmt.format(new Date());
        } catch (err) {
            console.error("[SSE] Parse error:", err);
        }
    };

    es.onerror = () => {
        setStatus("error", "Disconnected -- retrying in 5 s...");
        es.close();
        es = null;
        setTimeout(connectSSE, 5000);
    };
}

connectSSE();

function redrawHistogramsForTheme() {
    var content = byId("detail-content");
    if (!content) return;
    var canvases = content.querySelectorAll("canvas.histogram-canvas");
    for (var ci = 0; ci < canvases.length; ci++) {
        var cv = canvases[ci];
        if (cv._row && cv._field) {
            cv._bars = _drawHistogramOnCanvas(cv, cv._row, cv._field);
        }
    }
}

window
    .matchMedia("(prefers-color-scheme: dark)")
    .addEventListener("change", redrawHistogramsForTheme);

new MutationObserver(redrawHistogramsForTheme).observe(
    document.documentElement,
    {
        attributes: true,
        attributeFilter: ["class", "data-theme", "color-scheme"],
    },
);
