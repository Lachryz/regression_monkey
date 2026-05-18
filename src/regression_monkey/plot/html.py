"""
regression_monkey · html  ·  redesigned edition
=================================================
独立交互式网页脚本：只读取标准结果文件和绘图元数据，然后生成自包含 HTML。
视觉风格：学术感精致极简 · 键盘导航 · 暗色 tooltip · 内置图例
"""

from __future__ import annotations

import argparse
import base64
import html
import json
import math
import pathlib
from time import perf_counter
from typing import Any

import pandas as pd

from . import png as rm_plot
from ..engine import py as rm_py


# ── Colour palette ────────────────────────────────────────────────────────────
#   p < 0.01  →  crimson
#   p < 0.05  →  forest green
#   p < 0.10  →  steel blue
#   n.s.      →  black

_SIG_COLOR = ["#111827", "#0433FF", "#00F900", "#FF2600"]  # nsig, 10%, 5%, 1%
_SIG_BG = [
    "rgba(17,24,39,.12)",
    "rgba(29,78,216,.16)",
    "rgba(21,128,61,.16)",
    "rgba(185,28,28,.16)",
]
_SIG_LABEL = ["n.s.", "p<0.10", "p<0.05", "p<0.01"]

_OBS_FILL = "#9CA3AF"
_MAX_HTML_PANEL_PLOT_WIDTH = 1120
_ALT_GROUP_COLORS = [
    "#0B3A75",  # deep blue
    "#14532D",  # deep green
    "#7F1D1D",  # deep red
    "#581C87",  # deep purple
    "#7C2D12",  # deep amber
    "#164E63",  # deep cyan
]
_COURIER_NEW_FONT_PATH = pathlib.Path("/System/Library/Fonts/Supplemental/Courier New.ttf")
_EMBEDDED_COURIER_NEW_CSS: str | None = None


# ── Helpers ───────────────────────────────────────────────────────────────────


def _json_for_html(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, allow_nan=False).replace("</", "<\\/")


def _embedded_courier_new_css() -> str:
    global _EMBEDDED_COURIER_NEW_CSS
    if _EMBEDDED_COURIER_NEW_CSS is not None:
        return _EMBEDDED_COURIER_NEW_CSS
    if not _COURIER_NEW_FONT_PATH.exists():
        _EMBEDDED_COURIER_NEW_CSS = ""
        return _EMBEDDED_COURIER_NEW_CSS
    data = base64.b64encode(_COURIER_NEW_FONT_PATH.read_bytes()).decode("ascii")
    _EMBEDDED_COURIER_NEW_CSS = (
        "@font-face {\n"
        "      font-family: 'RM Courier New';\n"
        "      src: url(data:font/ttf;base64,"
        + data
        + ") format('truetype');\n"
        "      font-weight: 400;\n"
        "      font-style: normal;\n"
        "      font-display: swap;\n"
        "    }\n"
    )
    return _EMBEDDED_COURIER_NEW_CSS


def _star_level(p: float) -> int:
    return 3 if p < 0.01 else 2 if p < 0.05 else 1 if p < 0.10 else 0


def _point_color(p: float) -> str:
    return _SIG_COLOR[_star_level(p)]


def _record_payload(
    record: rm_py.SpecRecord, index: int, matrix_controls: list[str]
) -> dict[str, Any]:
    included = [n for n in matrix_controls if n in record["controls_all"]]
    adj_r2_raw = float(record.get("adj_r2", float("nan")))
    adj_r2 = adj_r2_raw if math.isfinite(adj_r2_raw) else None
    within_r2_raw = float(record.get("within_r2", float("nan")))
    within_r2 = within_r2_raw if math.isfinite(within_r2_raw) else None
    f_stat_raw = float(record.get("f_stat", float("nan")))
    f_stat = f_stat_raw if math.isfinite(f_stat_raw) else None
    return {
        "index": index,
        "coef": float(record["coef"]),
        "se": float(record["se"]),
        "t_value": float(record["t_value"]),
        "p_value": float(record["p_value"]),
        "adj_r2": adj_r2,
        "within_r2": within_r2,
        "f_stat": f_stat,
        "ci99_lo": float(record["ci99_lo"]),
        "ci99_hi": float(record["ci99_hi"]),
        "ci95_lo": float(record["ci95_lo"]),
        "ci95_hi": float(record["ci95_hi"]),
        "ci90_lo": float(record["ci90_lo"]),
        "ci90_hi": float(record["ci90_hi"]),
        "obs": int(record["obs"]),
        "is_full": bool(record["is_full"]),
        "is_no_controls_test": not bool(record["controls_test"]),
        "is_best_test": False,
        "star": _star_level(float(record["p_value"])),
        "color": _point_color(float(record["p_value"])),
        "controls_all": sorted(record["controls_all"]),
        "included_matrix_controls": included,
        "control_stats": list(record.get("control_stats", [])),
    }


def _mark_best_test_record(payload: dict[str, Any]) -> None:
    records = list(payload.get("records", []))
    controls_test_names = set(_payload_controls_test_names(payload))
    for record in records:
        record["is_best_test"] = False
    three_star_records = [record for record in records if int(record.get("star", 0)) == 3]
    if not three_star_records or any(bool(record.get("is_full")) for record in three_star_records):
        return

    def candidate_key(record: dict[str, Any]) -> tuple[int, float, int]:
        included = set(record.get("included_matrix_controls", []))
        test_count = len(included & controls_test_names)
        p_value = float(record.get("p_value", float("inf")))
        index = int(record.get("index", 0))
        return (-test_count, p_value, index)

    best = min(three_star_records, key=candidate_key)
    best["is_best_test"] = True


def _display_subtitle(value: Any) -> str:
    text = str(value or "")
    return text.split(" - ", 1)[0]


def _controls_must_line(values: Any, max_width: int = 150) -> str:
    controls = list(values or [])
    if not controls:
        return "controls_must = (none)"
    prefix = "controls_must = "
    indent = " " * len(prefix)
    lines: list[str] = []
    current = prefix
    for item in controls:
        sep = "" if current == prefix else ", "
        candidate = current + sep + str(item)
        if current != prefix and len(candidate) > max_width:
            lines.append(current)
            current = indent + str(item)
        else:
            current = candidate
    lines.append(current)
    return "\n".join(lines)


def _controls_test_line(values: Any, alt_groups: Any = None) -> str:
    controls = [str(v) for v in list(values or [])]
    if not controls:
        return "controls_test = (none)"

    groups = [
        {
            "start": int(g.get("start", -1)),
            "end": int(g.get("end", -1)),
            "kind": str(g.get("kind", "")),
        }
        for g in list(alt_groups or [])
        if str(g.get("kind", "")) == "controls_test"
    ]
    group_by_start = {
        g["start"]: g for g in groups if 0 <= g["start"] <= g["end"] < len(controls)
    }
    grouped_idx = {
        idx
        for g in group_by_start.values()
        for idx in range(g["start"], g["end"] + 1)
    }

    pieces: list[str] = []
    idx = 0
    while idx < len(controls):
        grp = group_by_start.get(idx)
        if grp and grp["end"] > grp["start"]:
            pieces.append("[" + ", ".join(controls[idx : grp["end"] + 1]) + "]")
            idx = grp["end"] + 1
            continue
        if idx not in grouped_idx:
            pieces.append(controls[idx])
        else:
            pieces.append(controls[idx])
        idx += 1
    return "controls_test = " + (", ".join(pieces) if pieces else "(none)")


def _controls_test_line_html(values: Any, alt_groups: Any = None, max_width: int = 150) -> str:
    controls = [str(v) for v in list(values or [])]
    if not controls:
        return "controls_test = (none)"

    groups = [
        {
            "start": int(g.get("start", -1)),
            "end": int(g.get("end", -1)),
            "kind": str(g.get("kind", "")),
        }
        for g in list(alt_groups or [])
        if str(g.get("kind", "")) == "controls_test"
    ]
    group_by_start = {
        g["start"]: g for g in groups if 0 <= g["start"] <= g["end"] < len(controls)
    }

    # Build pieces as (html_markup, text_char_count) so we can wrap on text length
    pieces: list[tuple[str, int]] = []
    color_idx = 0
    idx = 0
    while idx < len(controls):
        grp = group_by_start.get(idx)
        if grp and grp["end"] > grp["start"]:
            fill = _ALT_GROUP_COLORS[color_idx % len(_ALT_GROUP_COLORS)]
            color_idx += 1
            text = "[" + ", ".join(controls[idx : grp["end"] + 1]) + "]"
            pieces.append((
                f'<span class="ctrl-group-title" style="color:{html.escape(fill, quote=True)}">{html.escape(text)}</span>',
                len(text),
            ))
            idx = grp["end"] + 1
            continue
        pieces.append((html.escape(controls[idx]), len(controls[idx])))
        idx += 1

    prefix = "controls_test = "
    indent_html = "&nbsp;" * len(prefix)
    lines: list[str] = []
    current_html = prefix
    current_len = len(prefix)
    first = True
    for piece_html, piece_len in pieces:
        sep = ", " if not first else ""
        candidate_len = current_len + (2 if not first else 0) + piece_len
        if not first and candidate_len > max_width:
            lines.append(current_html)
            current_html = indent_html + piece_html
            current_len = len(prefix) + piece_len
        else:
            current_html += sep + piece_html
            current_len = candidate_len
        first = False
    lines.append(current_html)
    return "<br>".join(lines)


def _payload_controls_must_line(payload: dict[str, Any]) -> str:
    existing = str(payload.get("controlsMustLine") or "")
    if existing:
        return existing
    if "controls_must_flat" in payload:
        return _controls_must_line(payload.get("controls_must_flat"))
    records = list(payload.get("records", []))
    if not records:
        return _controls_must_line([])
    common = set(records[0].get("controls_all", []))
    for rec in records[1:]:
        common &= set(rec.get("controls_all", []))
    varying = set(payload.get("matrixControls", []))
    return _controls_must_line(sorted(common - varying))


def _payload_controls_test_line(payload: dict[str, Any]) -> str:
    existing = str(payload.get("controlsTestLine") or "")
    if existing:
        return existing
    if "controls_test_flat" in payload:
        return _controls_test_line(
            payload.get("controls_test_flat"), payload.get("matrixAltGroups")
        )
    return _controls_test_line(
        payload.get("controlsTestNames", payload.get("matrixControls", [])),
        payload.get("matrixAltGroups"),
    )


def _payload_controls_test_line_html(payload: dict[str, Any]) -> str:
    if "controls_test_flat" in payload:
        return _controls_test_line_html(
            payload.get("controls_test_flat"), payload.get("matrixAltGroups")
        )
    return _controls_test_line_html(
        payload.get("controlsTestNames", payload.get("matrixControls", [])),
        payload.get("matrixAltGroups"),
    )


def _payload_controls_test_names(payload: dict[str, Any]) -> list[str]:
    existing = payload.get("controlsTestNames")
    if existing is not None:
        return [str(v) for v in list(existing)]
    if "controls_test_flat" in payload:
        return [str(v) for v in list(payload.get("controls_test_flat") or [])]
    controls = [str(v) for v in list(payload.get("matrixControls", []))]
    must_idx: set[int] = set()
    for grp in list(payload.get("matrixAltGroups", [])):
        if str(grp.get("kind", "")) != "controls_must":
            continue
        start = int(grp.get("start", -1))
        end = int(grp.get("end", -1))
        if 0 <= start <= end < len(controls):
            must_idx.update(range(start, end + 1))
    return [name for idx, name in enumerate(controls) if idx not in must_idx]


# ── Public entry point ────────────────────────────────────────────────────────


def html_from_files(
    *,
    results_path: str | pathlib.Path,
    meta_path: str | pathlib.Path,
    output_path: str | pathlib.Path | None = None,
) -> pathlib.Path:
    """Render one interactive specification-curve HTML from standard handoff files."""
    payload = payload_from_files(
        results_path=results_path,
        meta_path=meta_path,
    )
    default_out = pathlib.Path(
        str(payload.get("outputPath", pathlib.Path(results_path).with_suffix(".html")))
    ).with_suffix(".html")
    out = pathlib.Path(output_path) if output_path is not None else default_out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(_build_canvas_html(payload), encoding="utf-8")
    return out


def payload_from_files(
    *,
    results_path: str | pathlib.Path,
    meta_path: str | pathlib.Path,
) -> dict[str, Any]:
    """Build the single-chart HTML payload from standard handoff files."""
    render_t0 = perf_counter()
    results_file = pathlib.Path(results_path)
    meta_file = pathlib.Path(meta_path)
    meta = rm_plot.load_plot_meta(meta_file)
    records = rm_py.records_from_dataframe(pd.read_csv(results_file))

    matrix_controls = list(
        meta.get("matrix_controls", meta.get("controls_test_flat", []))
    )
    payload = {
        "title": f"{meta.get('y', '')} × {meta.get('x', '')}",
        "subtitle": _display_subtitle(meta.get("title_suffix")),
        "controlsMustLine": _controls_must_line(meta.get("controls_must_flat")),
        "controlsTestLine": _controls_test_line(
            meta.get("controls_test_flat"), meta.get("matrix_alt_groups")
        ),
        "controlsMustNames": list(meta.get("controls_must_flat", [])),
        "controlsTestNames": list(meta.get("controls_test_flat", [])),
        "y": meta.get("y", ""),
        "x": meta.get("x", ""),
        "engine": meta.get("engine", ""),
        "specName": meta.get("spec_name", ""),
        "outputPath": str(meta.get("output_path", results_file.with_suffix(".html"))),
        "matrixControls": matrix_controls,
        "matrixAltGroups": list(meta.get("matrix_alt_groups", [])),
        "showSpecialMarkers": bool(meta.get("show_special_markers", True)),
        "elapsedSeconds": meta.get("elapsed_seconds_preplot"),
        "records": [
            _record_payload(r, i, matrix_controls) for i, r in enumerate(records)
        ],
    }
    if payload["elapsedSeconds"] is not None:
        payload["elapsedSeconds"] = float(payload["elapsedSeconds"]) + (
            perf_counter() - render_t0
        )
    return payload


def html_bundle_from_payloads(
    payloads: list[dict[str, Any]],
    *,
    output_path: str | pathlib.Path,
) -> pathlib.Path:
    """Render one HTML wrapper that switches among multiple chart payloads."""
    out = pathlib.Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(_build_bundle_html(payloads), encoding="utf-8")
    return out


# ── HTML builder ─────────────────────────────────────────────────────────────


def _build_bundle_html(payloads: list[dict[str, Any]]) -> str:
    views: list[dict[str, str]] = []
    for idx, payload in enumerate(payloads):
        spec_label = str(payload.get("subtitle") or payload.get("specName") or "Spec")
        views.append({
            "id": f"view-{idx}",
            "y": str(payload.get("y", "")),
            "x": str(payload.get("x", "")),
            "spec": spec_label,
            "title": str(payload.get("title", "")),
            "srcdoc": _build_canvas_html(payload),
        })
    data_json = _json_for_html(views)
    title = "Regression Monkey Interactive"
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(title)}</title>
  <style>
    :root {{
      --ink: #111827;
      --muted: #6B7280;
      --line: #E5E7EB;
      --line-2: #F3F4F6;
      --bg: #FFFFFF;
      --active: #7C3AED;
      --mono: 'Courier New', Courier, monospace;
      --sans: Arial, Helvetica, sans-serif;
    }}
    *, *::before, *::after {{ box-sizing: border-box; }}
    html, body {{ height: 100%; margin: 0; }}
    body {{
      display: flex;
      flex-direction: column;
      overflow: hidden;
      background: var(--bg);
      color: var(--ink);
      font-family: var(--sans);
      font-size: 12px;
    }}
    .bundle-toolbar {{
      flex: 0 0 auto;
      display: flex;
      align-items: center;
      gap: 10px;
      padding: 9px 14px;
      border-bottom: 1px solid var(--line);
      background: var(--bg);
      font-family: var(--mono);
      min-height: 44px;
    }}
    .bundle-brand {{
      font-weight: 700;
      font-size: 12px;
      margin-right: 8px;
      white-space: nowrap;
    }}
    .bundle-field {{
      display: inline-flex;
      align-items: center;
      gap: 6px;
      min-width: 0;
    }}
    .bundle-field select {{
      height: 25px;
      max-width: 310px;
      border: 1px solid var(--line);
      border-radius: 5px;
      background: var(--line-2);
      color: var(--ink);
      font-family: var(--mono);
      font-size: 11px;
      padding: 2px 24px 2px 8px;
    }}
    .bundle-spacer {{ flex: 1; min-width: 8px; }}
    .bundle-count {{
      color: var(--muted);
      white-space: nowrap;
      font-family: var(--mono);
      font-size: 10.5px;
    }}
    #rmFrame {{
      flex: 1 1 auto;
      width: 100%;
      min-height: 0;
      border: 0;
      display: block;
      background: #FFFFFF;
    }}
    @media (max-width: 860px) {{
      .bundle-toolbar {{ align-items: flex-start; flex-wrap: wrap; }}
      .bundle-spacer {{ display: none; }}
      .bundle-field {{ flex: 1 1 100%; }}
      .bundle-field select {{ flex: 1; max-width: none; }}
    }}
  </style>
</head>
<body class="mode-compact">
  <div class="bundle-toolbar">
    <div class="bundle-brand">Regression Monkey</div>
    <div class="bundle-field">
      <select id="bundleY" aria-label="Y"></select>
    </div>
    <div class="bundle-field">
      <select id="bundleX" aria-label="X"></select>
    </div>
    <div class="bundle-field">
      <select id="bundleSpec" aria-label="SPEC"></select>
    </div>
    <div class="bundle-spacer"></div>
    <div id="bundleCount" class="bundle-count"></div>
  </div>
  <iframe id="rmFrame" title="Regression Monkey chart" allow="clipboard-write"></iframe>
  <script>
    const VIEWS = {data_json};
    const ySel = document.getElementById("bundleY");
    const xSel = document.getElementById("bundleX");
    const specSel = document.getElementById("bundleSpec");
    const frame = document.getElementById("rmFrame");
    const count = document.getElementById("bundleCount");
    window.__rmBundleState = window.__rmBundleState || {{ sort: "signed_p", mode: "", controlColor: "run" }};
    window.addEventListener("message", event => {{
      const data = event.data || {{}};
      if (data.type !== "rmBundleState") return;
      window.__rmBundleState = {{
        sort: data.sort || window.__rmBundleState.sort || "signed_p",
        mode: data.mode || window.__rmBundleState.mode || "",
        controlColor: data.controlColor || window.__rmBundleState.controlColor || "run",
      }};
    }});

    function uniq(values) {{
      return [...new Set(values.map(v => String(v)))];
    }}

    function option(value) {{
      const opt = document.createElement("option");
      opt.value = value;
      opt.textContent = value || "(blank)";
      return opt;
    }}

    function setOptions(select, values, preferred) {{
      const old = preferred !== undefined ? preferred : select.value;
      select.replaceChildren(...values.map(option));
      if (values.includes(old)) select.value = old;
      else if (values.length) select.value = values[0];
    }}

    function setSpecOptions(matches, preferredSpec) {{
      const oldId = specSel.value;
      const oldView = VIEWS.find(v => v.id === oldId);
      const oldSpec = preferredSpec !== undefined ? preferredSpec : (oldView ? oldView.spec : "");
      specSel.replaceChildren(...matches.map(v => {{
        const opt = option(v.id);
        opt.textContent = v.spec || "Spec";
        return opt;
      }}));
      const sameId = matches.find(v => v.id === oldId);
      const sameSpec = matches.find(v => String(v.spec || "") === String(oldSpec || ""));
      if (sameId) specSel.value = sameId.id;
      else if (sameSpec) specSel.value = sameSpec.id;
      else if (matches.length) specSel.value = matches[0].id;
    }}

    function filtered() {{
      return VIEWS.filter(v => String(v.y) === ySel.value && String(v.x) === xSel.value);
    }}

    function renderSelectors(changed) {{
      if (!VIEWS.length) return;
      const previousY = ySel.value;
      const previousX = xSel.value;
      const previousSpec = (VIEWS.find(v => v.id === specSel.value) || {{}}).spec || "";
      if (changed !== "y") setOptions(ySel, uniq(VIEWS.map(v => v.y)), previousY);
      const xs = uniq(VIEWS.filter(v => String(v.y) === ySel.value).map(v => v.x));
      if (changed !== "x") setOptions(xSel, xs, previousX);
      setSpecOptions(filtered(), previousSpec);
      renderFrame();
    }}

    function renderFrame() {{
      const matches = filtered();
      const view = matches.find(v => v.id === specSel.value) || matches[0] || VIEWS[0];
      if (!view) return;
      ySel.value = String(view.y);
      xSel.value = String(view.x);
      specSel.value = view.id;
      frame.srcdoc = view.srcdoc;
      count.textContent = `${{VIEWS.indexOf(view) + 1}} / ${{VIEWS.length}}`;
    }}

    ySel.addEventListener("change", () => renderSelectors("y"));
    xSel.addEventListener("change", () => renderSelectors("x"));
    specSel.addEventListener("change", renderFrame);
    renderSelectors();
  </script>
</body>
</html>
"""

# ── Canvas HTML builder ───────────────────────────────────────────────────────


def _build_canvas_html(payload: dict[str, Any]) -> str:  # noqa: C901
    """Generate a self-contained Canvas-based interactive HTML.

    - Data is zlib-compressed + base64-encoded (reduces payload size ~70%).
    - Rendering uses two stacked <canvas> elements: cv-main (static) and
      cv-ov (overlay for hover/pin highlights).
    - Virtual rendering skips columns outside the visible viewport + buffer,
      so 8000+ spec charts stay fast.
    - No SVG elements are emitted; the entire chart is drawn imperatively.
    """
    import zlib as _zlib

    records = list(payload["records"])
    controls = list(payload["matrixControls"])
    def _flat_control_names(values: Any) -> list[str]:
        names: list[str] = []
        for value in list(values or []):
            if isinstance(value, (list, tuple)):
                names.extend(_flat_control_names(value))
            else:
                names.append(str(value))
        return names

    must_controls = _flat_control_names(payload.get("controlsMustNames", []))
    sig_controls = list(dict.fromkeys([*must_controls, *[str(c) for c in controls]]))
    alt_groups = list(payload.get("matrixAltGroups", []))

    n = len(records)
    n_controls = len(sig_controls)
    compact_threshold = 1024
    compact_base_n = 8192
    compact_enabled = n > compact_threshold
    initial_mode = "compact" if compact_enabled else "detail"
    body_classes = ["rm-preinit"]
    if compact_enabled:
        body_classes.append("mode-compact")
    body_attrs = f' class="{" ".join(body_classes)}"'
    compact_button_attrs = (
        ' class="active" aria-pressed="true"'
        if compact_enabled
        else ' aria-pressed="false" disabled'
    )
    detail_button_attrs = (
        ' aria-pressed="false"'
        if compact_enabled
        else ' class="active" aria-pressed="true"'
    )
    compact_enabled_js = "true" if compact_enabled else "false"

    # ── Geometry (mirrors _build_svg) ─────────────────────────────────────────
    label_chars = max([len(str(c)) for c in sig_controls] + [8])
    grp_pad = 46 if alt_groups else 0
    left = max(160, min(440, 76 + label_chars * 7 + grp_pad))
    right = 82
    coef_h = 296
    row_h = 18
    obs_h = 88
    gap = 12
    x_step = 7 if n <= 1600 else 5
    star_cell_gap = 3
    star_cell_size = 2 * max(1.6, min(2.6, x_step * 0.36))
    star_h = int(math.ceil(3 * star_cell_size + 4 * star_cell_gap))
    matrix_h = max(row_h, n_controls * row_h)

    star_y = 20
    coef_y = star_y + star_h + gap
    matrix_y = coef_y + coef_h + gap
    obs_y = matrix_y + matrix_h + gap + 12
    total_h = int(obs_y + obs_h + 22)

    # ── Value ranges ──────────────────────────────────────────────────────────
    if records:
        clo = min(float(r["ci99_lo"]) for r in records)
        chi = max(float(r["ci99_hi"]) for r in records)
        if clo == chi:
            clo -= 1.0
            chi += 1.0
        else:
            pad = (chi - clo) * 0.09
            clo -= pad
            chi += pad
        obs_values = [int(r["obs"]) for r in records]
        obs_min_v = min(obs_values)
        obs_max_v = max(obs_values)
        obs_mean_v = sum(obs_values) / len(obs_values)
    else:
        clo, chi, obs_min_v, obs_max_v, obs_mean_v = -1.0, 1.0, 0, 1, 0.0

    # Compute grid tick values (5 evenly spaced, top→bottom)
    grid_coefs = [chi - (chi - clo) * j / 4 for j in range(5)]

    # Alt-group color assignment
    alt_group_color_by_control: dict[str, str] = {}
    color_idx_py = 0
    for grp in alt_groups:
        if str(grp.get("kind", "")) != "controls_test":
            continue
        s = int(grp.get("start", -1))
        e = int(grp.get("end", -1))
        if s < 0 or e <= s or e >= len(controls):
            continue
        fill = _ALT_GROUP_COLORS[color_idx_py % len(_ALT_GROUP_COLORS)]
        color_idx_py += 1
        for row in range(s, e + 1):
            alt_group_color_by_control[str(controls[row])] = fill

    # Three-star starred controls (yellow highlight)
    from ..engine import py as _rm_py  # already imported at module level, reuse below
    controls_test_names_set = set(_payload_controls_test_names(payload))
    three_star_records = [r for r in records if int(r.get("star", 0)) == 3]
    starred_controls: set[str] = {
        name
        for name in controls
        if name in controls_test_names_set
        and three_star_records
        and all(name in set(r.get("included_matrix_controls", [])) for r in three_star_records)
    }

    # Header info
    title_esc = html.escape(str(payload.get("title", "")))
    y_title = html.escape(str(payload.get("y", "")))
    x_title = html.escape(str(payload.get("x", "")))
    subtitle = html.escape(str(payload.get("subtitle") or ""))
    controls_must_line_s = html.escape(_payload_controls_must_line(payload))
    controls_test_line_html_s = _payload_controls_test_line_html(payload)
    elapsed = payload.get("elapsedSeconds")
    elapsed_text = f"Elapsed = {float(elapsed):.2f}s" if elapsed is not None else "Elapsed = n/a"
    engine = str(payload.get("engine") or "").strip()
    engine_text = f"engine = {html.escape(engine)}" if engine else ""
    n1 = sum(1 for r in records if r.get("star") == 3)
    n5 = sum(1 for r in records if r.get("star") == 2)
    n10 = sum(1 for r in records if r.get("star") == 1)

    # Left sidebar labels (Python-generated static HTML)
    tick_items: list[str] = []
    for j in range(5):
        frac = j / 4
        gv = chi - (chi - clo) * frac
        # y position within the canvas (in px from top of canvas)
        ypos = coef_y + coef_h * frac
        tick_items.append(
            f'<div class="tick-lbl coef-tick" style="top:{ypos - 6:.1f}px">{gv:.3f}</div>'
        )

    obs_tick_data = [
        ("max", float(obs_max_v)),
        ("mean", float(obs_mean_v)),
        ("min", float(obs_min_v)),
    ]

    def _obs_scale(v: float) -> float:
        lo_o = min(float(obs_min_v), float(obs_mean_v), float(obs_max_v))
        hi_o = max(float(obs_min_v), float(obs_mean_v), float(obs_max_v))
        if lo_o == hi_o:
            lo_o -= 1
            hi_o += 1
        else:
            p_ = (hi_o - lo_o) * 0.06
            lo_o -= p_
            hi_o += p_
        if hi_o == lo_o:
            return obs_y + obs_h / 2
        return obs_y + obs_h - ((v - lo_o) / (hi_o - lo_o)) * obs_h

    for label_o, val_o in obs_tick_data:
        ypos_o = _obs_scale(val_o)
        tick_items.append(
            f'<div class="tick-lbl obs-tick" style="top:{ypos_o - 6:.1f}px">{val_o:,.0f}</div>'
        )

    control_label_items: list[str] = []
    for row, name in enumerate(controls):
        ry = matrix_y + row * row_h
        # y center for the label
        label_y_center = ry + row_h * 0.65
        group_color = alt_group_color_by_control.get(str(name), "")
        color_style = f"color:{group_color};" if group_color else ""
        highlight_class = " starred" if name in starred_controls else ""
        control_label_items.append(
            f'<div class="tick-lbl ctrl-lbl ctrl-lbl-gray{highlight_class}" data-control="{html.escape(str(name), quote=True)}"'
            f' style="top:{label_y_center - 7:.1f}px;{color_style}">{html.escape(str(name))}</div>'
        )
    for row, name in enumerate(sig_controls):
        ry = matrix_y + row * row_h
        label_y_center = ry + row_h * 0.65
        group_color = alt_group_color_by_control.get(str(name), "")
        color_style = f"color:{group_color};" if group_color else ""
        highlight_class = " starred" if name in starred_controls else ""
        control_label_items.append(
            f'<div class="tick-lbl ctrl-lbl ctrl-lbl-sig{highlight_class}" data-control="{html.escape(str(name), quote=True)}"'
            f' style="top:{label_y_center - 7:.1f}px;{color_style}">{html.escape(str(name))}</div>'
        )

    # Panel side-labels (STARS / COEF / CONTROLS / OBS) — rotated 90°
    panels_py = [
        (star_y, star_h, "STARS"),
        (coef_y, coef_h, "COEF"),
        (matrix_y, matrix_h, "CONTROLS"),
        (obs_y, obs_h, "OBS"),
    ]
    left_sidebar_html = "\n".join(tick_items + control_label_items)

    # Alt-group bracket markers HTML (inside the left sidebar, absolute pos)
    alt_marker_items: list[str] = []
    for grp in alt_groups:
        s = int(grp.get("start", -1))
        e = int(grp.get("end", -1))
        if s < 0 or e < s or e >= len(controls):
            continue
        y0 = matrix_y + s * row_h
        y1 = matrix_y + (e + 1) * row_h
        is_test = str(grp.get("kind", "")) == "controls_test"
        if is_test and y1 - y0 > 6:
            y0 += 2
            y1 -= 2
        marker_color = (
            alt_group_color_by_control.get(str(controls[s]), "#6B7280")
            if is_test
            else "#6B7280"
        )
        marker_class = "alt-marker is-dashed" if is_test else "alt-marker"
        alt_marker_items.append(
            f'<div class="{marker_class}" style="top:{y0:.1f}px;height:{y1-y0:.1f}px;--alt-marker-color:{marker_color};"></div>'
        )

    alt_markers_html = "\n".join(alt_marker_items)

    _mark_best_test_record(payload)

    # ── Compress payload JSON ──────────────────────────────────────────────────
    json_bytes = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    compressed = base64.b64encode(_zlib.compress(json_bytes, level=9)).decode("ascii")

    # ── Embed font ────────────────────────────────────────────────────────────
    embedded_font_css = _embedded_courier_new_css()

    # Pre-compute JSON fragments for JS constants
    alt_groups_js = _json_for_html(alt_groups)
    controls_js = _json_for_html(controls)
    must_controls_js = _json_for_html(must_controls)
    sig_controls_js = _json_for_html(sig_controls)
    alt_group_colors_js = _json_for_html(_ALT_GROUP_COLORS)
    alt_group_color_map_js = _json_for_html(alt_group_color_by_control)
    starred_controls_js = _json_for_html(list(starred_controls))
    sig_colors_js = json.dumps(_SIG_COLOR)
    sig_bg_js = json.dumps(_SIG_BG)
    sig_labels_js = json.dumps(_SIG_LABEL)

    # ── Assemble HTML ─────────────────────────────────────────────────────────
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{title_esc}</title>
  <style>
    {embedded_font_css}
    :root {{
      --ink:        #111827;
      --ink-2:      #374151;
      --muted:      #6B7280;
      --muted-2:    #9CA3AF;
      --line:       #E5E7EB;
      --line-2:     #F3F4F6;
      --bg:         #FFFFFF;
      --bg-2:       #F9FAFB;
      --active:     #7C3AED;
      --sig1:       #FF2600;
      --sig5:       #00F900;
      --sig10:      #0433FF;
      --nsig:       #111827;
      --mono: "RM Courier New", "Courier New", monospace;
      --sans: "RM Courier New", "Courier New", monospace;
      --r-sm:  4px;
      --r-md:  8px;
      --shadow-sm: 0 1px 3px rgba(0,0,0,.08), 0 1px 2px rgba(0,0,0,.06);
      --header-pad-x: 22px;
    }}
    *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
    html, body {{ height: 100%; }}
    body {{
      font-family: var(--sans);
      font-size: 13px;
      color: var(--ink);
      background: var(--bg);
      -webkit-font-smoothing: antialiased;
      text-rendering: optimizeLegibility;
      display: flex;
      flex-direction: column;
      overflow: hidden;
    }}
    body.rm-preinit {{ visibility: hidden; }}
    header {{
      position: sticky;
      top: 0;
      z-index: 100;
      padding: 11px var(--header-pad-x) 9px;
      background: rgba(255,255,255,.95);
      backdrop-filter: saturate(180%) blur(14px);
      -webkit-backdrop-filter: saturate(180%) blur(14px);
    }}
    header::after {{
      content: "";
      position: absolute;
      left: 0; right: 0; bottom: 0;
      height: 1px;
      background: var(--line);
      pointer-events: none;
    }}
    .h-row {{ display: flex; align-items: center; gap: 10px; flex-wrap: wrap; }}
    .title-stack {{ display: flex; flex-direction: column; gap: 3px; align-items: flex-start; }}
    .title-meta {{
      display: inline-flex; align-items: center; gap: 8px;
      padding: 3px 10px;
      border: 1px solid var(--line); border-radius: 99px;
      background: var(--line-2);
      font-family: var(--mono); font-size: 10.5px; font-weight: 500;
      color: var(--muted); white-space: nowrap;
    }}
    .title-meta .sep {{ width: 1px; height: 11px; background: var(--line); display: inline-block; }}
    h1 {{ font-family: var(--mono); font-size: 13.5px; font-weight: 600; letter-spacing: -0.01em; color: var(--ink); line-height: 1.35; }}
    .kbd-row {{ margin-left: auto; display: flex; align-items: center; gap: 5px; font-size: 10px; color: var(--muted-2); }}
    kbd {{
      display: inline-flex; align-items: center; justify-content: center;
      min-width: 18px; height: 18px; padding: 0 4px;
      border: 1px solid var(--line); border-radius: 3px;
      background: var(--line-2); font-family: var(--sans); font-size: 9.5px;
      color: var(--ink-2); box-shadow: 0 1px 0 var(--line);
    }}
    .subtitle {{ margin-top: 2px; font-family: var(--mono); font-size: 10.5px; color: var(--muted); white-space: pre-wrap; word-break: break-word; }}
    .ctrl-group-title {{ font-weight: 700; }}
    .meta-row {{ margin-top: 4px; display: flex; align-items: center; gap: 8px; flex-wrap: wrap; }}
    .rm-tools {{
      position: relative; margin-top: 10px; padding-top: 10px;
      display: flex; align-items: center; gap: 10px; flex-wrap: wrap;
      font-family: var(--mono); font-size: 10.5px; color: var(--muted);
    }}
    .rm-tools::before {{
      content: "";
      position: absolute;
      left: calc(-1 * var(--header-pad-x)); right: calc(-1 * var(--header-pad-x));
      top: 0; height: 1px; background: var(--line); pointer-events: none;
    }}
    .rm-lbl {{ color: var(--muted-2); text-transform: uppercase; letter-spacing: .08em; font-weight: 600; font-size: 9.5px; }}
    .rm-seg {{ display: inline-flex; background: var(--line-2); border-radius: 5px; padding: 2px; gap: 2px; }}
    .rm-seg button {{
      background: transparent; border: 0; outline: none;
      font-family: var(--mono); font-size: 10.5px;
      padding: 3px 8px; border-radius: 4px;
      color: var(--muted); cursor: pointer; font-weight: 500;
    }}
    .rm-seg button.active, .rm-seg button[aria-pressed="true"] {{
      background: #FFFFFF; color: var(--ink);
      box-shadow: 0 1px 2px rgba(0,0,0,.06);
    }}
    .rm-seg button:focus, .rm-seg button:focus-visible {{ outline: none; }}
    .rm-seg button:disabled {{
      opacity: .42;
      cursor: default;
      background: transparent;
      box-shadow: none;
    }}
    body.mode-compact .info-panel {{ display: none; }}
    body.mode-compact #cv-wrap {{ cursor: default; }}
    body.mode-compact #cv-ov {{ pointer-events: none; }}
    .rm-chip {{
      display: inline-flex; align-items: center; gap: 5px;
      padding: 2px 8px; border: 1px solid var(--line); border-radius: 99px;
      color: var(--muted-2); background: #FFFFFF;
      cursor: pointer; user-select: none; font-size: 10px;
    }}
    .rm-chip i {{ width: 7px; height: 7px; border-radius: 50%; display: inline-block; }}
    .rm-chip.on {{ border-color: var(--ink); color: var(--ink); background: var(--bg); }}
    .rm-divider {{ width: 1px; height: 14px; background: var(--line); }}
    .main-body {{ flex: 1; display: flex; min-height: 0; overflow: hidden; }}
    /* Chart column */
    #chart-col {{ flex: 1; min-width: 0; display: flex; flex-direction: column; overflow: hidden; position: relative; }}
    /* Vertical scroll wrapper — contains canvas + left sidebar, scrolls vertically */
    #chart-vscroll {{ flex: 1; min-height: 0; overflow-x: hidden; overflow-y: auto; position: relative; }}
    /* Left sidebar */
    #left-sb {{
      position: absolute; left: 0; top: 0;
      width: {left}px;
      height: {total_h}px;
      pointer-events: none;
      z-index: 5;
      background: #fff;
    }}
    .tick-lbl {{
      position: absolute;
      right: 8px;
      font-family: var(--mono); font-size: 9.5px; color: #9CA3AF;
      white-space: nowrap;
      line-height: 1.2;
    }}
    .ctrl-lbl {{
      position: absolute;
      right: 8px;
      max-width: calc(100% - 52px);
      overflow: hidden;
      text-overflow: ellipsis;
      font-family: var(--mono); font-size: 10px; color: #4B5563;
      white-space: nowrap;
      line-height: 1.2;
      z-index: 2;
    }}
    .ctrl-lbl.starred {{ background: #FEF3C7; border-radius: 2px; padding: 0 2px; }}
    .ctrl-lbl-sig {{ display: none; }}
    body.control-stats .ctrl-lbl-gray {{ display: none; }}
    body.control-stats .ctrl-lbl-sig {{ display: block; }}
    body.control-stats .alt-marker {{ display: none; }}
    .alt-marker {{
      position: absolute;
      left: 16px;
      width: 12px;
      z-index: 8;
      --alt-marker-color: #6B7280;
      background-image: linear-gradient(var(--alt-marker-color), var(--alt-marker-color));
      background-position: center top;
      background-repeat: no-repeat;
      background-size: 1.4px 100%;
    }}
    .alt-marker.is-dashed {{
      background-image: repeating-linear-gradient(to bottom, var(--alt-marker-color) 0 3px, transparent 3px 6px);
    }}
    .alt-marker::before, .alt-marker::after {{
      content: "";
      position: absolute;
      left: 0;
      width: 12px;
      border-top: 1.4px solid var(--alt-marker-color);
    }}
    .alt-marker::before {{ top: 0; }}
    .alt-marker::after {{ bottom: 0; }}
    /* Canvas wrap */
    #cv-wrap {{
      position: relative;
      height: {total_h}px;
      overflow: hidden;
      cursor: crosshair;
    }}
    #cv-main, #cv-ov {{
      position: absolute; top: 0; left: 0;
      display: block;
    }}
    #cv-ov {{ z-index: 2; }}
    /* Scrollbar */
    #sb-track {{
      height: 10px;
      margin: 2px {right}px 4px {left}px;
      background: var(--line-2);
      border-radius: 5px;
      position: relative;
      cursor: pointer;
      flex-shrink: 0;
    }}
    #sb-thumb {{
      position: absolute; top: 0; height: 100%;
      background: #9CA3AF; border-radius: 5px;
      min-width: 20px;
      cursor: grab;
    }}
    #sb-thumb:active {{ cursor: grabbing; }}
    /* Info panel */
    .info-panel {{
      width: 300px; flex-shrink: 0;
      border-left: 1px solid var(--line);
      background: var(--bg); overflow-y: auto;
    }}
    .panel-placeholder {{ padding: 32px 16px; color: var(--muted-2); font-size: 11px; text-align: center; line-height: 1.7; }}
    .panel-head {{
      display: flex; align-items: center; justify-content: space-between;
      gap: 8px;
      padding: 10px 14px 8px; border-bottom: 1px solid var(--line);
    }}
    .panel-head-left {{ display: inline-flex; align-items: center; gap: 8px; min-width: 0; }}
    .panel-title {{ font-family: var(--mono); font-size: 11px; font-weight: 600; color: var(--ink); }}
    .panel-copy {{
      border: 1px solid var(--line); background: var(--bg-2); color: var(--ink-2);
      border-radius: 3px; padding: 2px 6px;
      font-family: var(--mono); font-size: 9px; font-weight: 700; letter-spacing: .04em;
      cursor: pointer; line-height: 1.2;
    }}
    .panel-copy:hover {{ background: #E5E7EB; color: var(--ink); }}
    .panel-copy:disabled {{ opacity: .45; cursor: default; background: var(--bg-2); color: var(--muted-2); }}
    .panel-copy:disabled:hover {{ background: var(--bg-2); color: var(--muted-2); }}
    .panel-copy:active {{ transform: translateY(1px); }}
    .panel-copy.copied {{ color: #7C3AED; border-color: rgba(124,58,237,.35); background: rgba(124,58,237,.08); }}
    .panel-sig {{
      display: inline-flex; align-items: center;
      padding: 1px 7px; border-radius: 99px;
      font-size: 10px; font-weight: 600; letter-spacing: .01em;
      white-space: nowrap;
    }}
    .panel-table {{
      display: grid; grid-template-columns: auto 1fr;
      gap: 2px 10px; padding: 10px 14px;
    }}
    .panel-key {{ color: var(--muted); font-size: 10.5px; align-self: center; }}
    .panel-val {{ font-family: var(--mono); font-size: 10.5px; color: var(--ink-2); font-weight: 500; text-align: right; }}
    .panel-divider {{ grid-column: 1 / -1; height: 1px; border-top: 1px dotted var(--line); margin: 4px 0; }}
    .panel-controls {{ grid-column: 1 / -1; margin-top: 2px; color: var(--muted); font-size: 10px; line-height: 1.45; }}
    .panel-controls em {{ font-style: normal; color: var(--ink-2); }}
    .panel-coefs {{ grid-column: 1 / -1; display: flex; flex-direction: column; margin-top: 2px; }}
    .panel-coefs-head {{
      display: flex; justify-content: space-between; align-items: baseline;
      font-family: var(--mono); font-size: 9px; letter-spacing: .1em;
      text-transform: uppercase; color: var(--muted-2); font-weight: 600;
      padding-bottom: 6px; margin-bottom: 4px;
    }}
    .panel-coefs-meta {{ font-weight: 500; opacity: .85; letter-spacing: 0; text-transform: none; }}
    .coef-group-label {{
      font-family: var(--mono); font-size: 8.5px; color: var(--muted-2);
      text-transform: uppercase; letter-spacing: .12em;
      padding: 8px 14px 4px; font-weight: 600;
      border-top: 1px solid var(--line); margin: 6px -14px 0;
    }}
    .coef-group-label .grp-count {{ color: var(--muted); font-weight: 500; letter-spacing: 0; }}
    .coef-row {{
      display: grid; grid-template-columns: minmax(0, 1fr) 68px 74px;
      gap: 12px; align-items: baseline;
      padding: 4px; border-radius: 3px;
      font-family: var(--mono); font-size: 10.5px; line-height: 1.25;
      transition: background .12s;
      cursor: pointer;
    }}
    .coef-row + .coef-row {{ border-top: 1px dotted var(--line); }}
    .coef-row:hover {{ background: var(--bg-2); }}
    .coef-row.filter-selected {{ background: rgba(124,58,237,.11); box-shadow: inset 0 0 0 1px rgba(124,58,237,.28); }}
    .coef-row.filter-selected .coef-name {{ color: #7C3AED; font-weight: 700; }}
    .coef-row.is-test .coef-name {{ font-weight: 600; }}
    .coef-filter-bar {{
      grid-column: 1 / -1;
      display: none; align-items: center; gap: 6px; flex-wrap: wrap;
      padding: 2px 0 6px; margin-bottom: 2px;
      font-family: var(--mono); font-size: 9.5px; color: var(--muted);
    }}
    .coef-filter-bar.has-filters {{ display: flex; }}
    .coef-filter-chip {{
      display: inline-flex; align-items: center; gap: 4px;
      border: 1px solid rgba(124,58,237,.22); background: rgba(124,58,237,.08);
      color: #5B21B6; border-radius: 999px; padding: 1px 6px;
      max-width: 118px;
    }}
    .coef-filter-chip span {{ overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }}
    .coef-filter-clear {{
      border: 0; background: transparent; color: var(--muted-2);
      font-family: var(--mono); font-size: 9.5px; padding: 1px 4px; cursor: pointer;
    }}
    .coef-filter-clear:hover {{ color: var(--ink); }}
    .coef-name-wrap {{ display: inline-flex; align-items: center; gap: 6px; min-width: 0; }}
    .coef-badge {{
      width: 22px; height: 18px; border-radius: 3px;
      display: inline-flex; align-items: center; justify-content: center;
      flex: 0 0 22px; font-family: var(--mono); font-size: 9px; font-weight: 700;
      line-height: 1; font-variant-numeric: tabular-nums;
      border: 1px solid rgba(17,24,39,.08);
    }}
    .coef-badge.pos-dir {{ color: #FFFFFF; }}
    .coef-badge.neg-dir {{ color: #FFFFFF; }}
    .coef-badge.pos-zero {{ color: #DC2626; }}
    .coef-badge.neg-zero {{ color: #0433FF; }}
    .coef-badge.pos-sig-1 {{ background: #FECACA; }}
    .coef-badge.pos-sig-2 {{ background: #B91C1C; }}
    .coef-badge.pos-sig-3 {{ background: #FF2600; }}
    .coef-badge.neg-sig-1 {{ background: #BFDBFE; }}
    .coef-badge.neg-sig-2 {{ background: #1E3A8A; }}
    .coef-badge.neg-sig-3 {{ background: #0433FF; }}
    .coef-badge.pos-sig-1, .coef-badge.neg-sig-1 {{ color: #111827; }}
    .coef-badge.zero {{ background: #E5E7EB; }}
    .coef-badge.missing {{ background: #F3F4F6; color: var(--muted-2); }}
    .coef-badge.blank {{ background: #FFFFFF; color: transparent; border-color: #D1D5DB; }}
    .coef-name {{ color: var(--ink); font-weight: 500; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; letter-spacing: 0; }}
    .coef-row.not-included .coef-name {{ color: var(--muted); font-weight: 500; }}
    .coef-val {{ color: var(--ink); font-variant-numeric: tabular-nums; text-align: right; font-size: 10.5px; letter-spacing: 0; min-width: 0; }}
    .coef-val.placeholder {{ color: var(--muted-2); }}
    .coef-val.pos {{ color: #B91C1C; }}
    .coef-val.neg {{ color: #0433FF; }}
    .coef-p {{ display: inline-flex; align-items: center; gap: 4px; font-size: 9.5px; color: var(--muted); font-variant-numeric: tabular-nums; justify-content: flex-end; min-width: 0; white-space: nowrap; }}
    .coef-p.placeholder {{ color: var(--muted-2); }}
    .coef-p .coef-stars {{ font-weight: 700; letter-spacing: 0; font-size: 11px; color: var(--ink-2); }}
    .coef-empty {{ font-family: var(--mono); font-size: 10px; color: var(--muted-2); padding: 6px 4px; font-style: italic; }}
    @media print {{
      header {{ position: static; border-bottom: 1px solid #ccc; }}
      .info-panel {{ display: none; }}
    }}
  </style>
</head>
<body{body_attrs}>

<header>
  <div class="h-row">
    <div class="title-stack">
      <div class="title-meta">
        <span>{n} specs</span>
        {f'<span class="sep"></span><span>{engine_text}</span>' if engine_text else ""}
        <span class="sep"></span>
        <span>{elapsed_text}</span>
      </div>
      <h1>Y: {y_title}<br>X: {x_title}</h1>
    </div>
    <div class="kbd-row">
      Navigate <kbd>&#8592;</kbd><kbd>&#8594;</kbd>&nbsp; Dismiss <kbd>Esc</kbd>
    </div>
  </div>
  {f'<div class="subtitle">{subtitle}</div>' if subtitle else ""}
  {f'<div class="subtitle">{controls_must_line_s}</div>' if controls_must_line_s else ""}
  {f'<div class="subtitle">{controls_test_line_html_s}</div>' if controls_test_line_html_s else ""}
  <div class="meta-row">
    <span style="font-family:var(--mono);font-size:10.5px;font-weight:600;color:var(--ink-2)">@Lachryz</span>
  </div>
  <div class="rm-tools">
    <span class="rm-lbl">Sort</span>
    <div class="rm-seg" id="rmSort">
      <button type="button" data-v="coef" aria-pressed="false">coef</button>
      <button type="button" data-v="obs" aria-pressed="false">obs</button>
      <button type="button" data-v="signed_p" class="active" aria-pressed="true">sig(coef)/p</button>
    </div>
    <span class="rm-lbl">Mode</span>
    <div class="rm-seg" id="rmMode">
      <button type="button" data-v="compact"{compact_button_attrs}>compact</button>
      <button type="button" data-v="detail"{detail_button_attrs}>detail</button>
    </div>
    <span class="rm-divider"></span>
    <span class="rm-lbl">Controls</span>
    <div class="rm-seg" id="rmControlColor">
      <button type="button" data-v="run" class="active" aria-pressed="true">gray</button>
      <button type="button" data-v="stats" aria-pressed="false">sig</button>
    </div>
    <span class="rm-divider"></span>
    <span class="rm-lbl">Significance</span>
    <span class="rm-chip on" data-sig="3"><i style="background:#FF2600"></i>p&lt;.01</span>
    <span class="rm-chip on" data-sig="2"><i style="background:#00F900"></i>p&lt;.05</span>
    <span class="rm-chip on" data-sig="1"><i style="background:#0433FF"></i>p&lt;.10</span>
    <span class="rm-chip on" data-sig="0"><i style="background:#111827"></i>n.s.</span>
    <span class="rm-divider"></span>
    <span class="rm-lbl">CI bands</span>
    <span class="rm-chip on" data-ci="90"><i style="background:#9CA3AF;opacity:.55"></i>90%</span>
    <span class="rm-chip on" data-ci="95"><i style="background:#6B7280;opacity:.7"></i>95%</span>
    <span class="rm-chip on" data-ci="99"><i style="background:#374151;opacity:.85"></i>99%</span>
    <span class="rm-divider"></span>
    <span class="rm-lbl">TEST GUIDES</span>
    <span class="rm-chip on" data-special="full"><i style="background:#FF2F92"></i>all</span>
    <span class="rm-chip on" data-special="nocontrol"><i style="background:#ff8c00"></i>no</span>
    <span class="rm-chip on" data-special="besttest"><i style="background:#FACC15"></i>best</span>
  </div>
</header>

<div class="main-body">
  <div id="chart-col">
    <!-- Vertical scroll wrapper: left sidebar + canvas scroll together -->
    <div id="chart-vscroll">
      <!-- Left sidebar: tick labels, control labels -->
      <div id="left-sb">
        {left_sidebar_html}
        {alt_markers_html}
      </div>
      <!-- Canvas area -->
      <div id="cv-wrap">
        <canvas id="cv-main"></canvas>
        <canvas id="cv-ov"></canvas>
      </div>
    </div>
    <!-- Custom horizontal scrollbar — stays fixed at bottom -->
    <div id="sb-track"><div id="sb-thumb"></div></div>
  </div>
  <div class="info-panel" id="info-panel">
    <div class="panel-placeholder" id="panel-placeholder">Hover or click<br>a specification</div>
    <div id="panel-content" style="display:none"></div>
  </div>
</div>

<!-- Compressed payload -->
<script>const _RAW = '{compressed}';</script>

<script>
(async function() {{
  /* ── Decompress payload ─────────────────────────────────── */
  async function loadData() {{
    const b64 = _RAW;
    const bin = Uint8Array.from(atob(b64), c => c.charCodeAt(0));
    const ds = new DecompressionStream('deflate');
    const writer = ds.writable.getWriter();
    writer.write(bin);
    writer.close();
    const buf = await new Response(ds.readable).arrayBuffer();
    return JSON.parse(new TextDecoder().decode(buf));
  }}

  const DATA = await loadData();
  const records = DATA.records;
  const N = records.length;
  const COMPACT_THRESHOLD = {compact_threshold};
  const COMPACT_BASE_N = {compact_base_n};
  const COMPACT_ENABLED = {compact_enabled_js};
  const MAX_COMPACT_BITMAP_DIM = 32760;
  const MAX_COMPACT_BITMAP_AREA = 80000000;

  /* ── Constants (must match Python geometry) ─────────────── */
  const LEFT        = {left};
  const RIGHT       = {right};
  const STAR_Y      = {star_y};
  const STAR_H      = {star_h};
  const COEF_Y      = {coef_y};
  const COEF_H      = {coef_h};
  const MATRIX_Y    = {matrix_y};
  const MATRIX_H    = {matrix_h};
  const OBS_Y       = {obs_y};
  const OBS_H       = {obs_h};
  const ROW_H       = {row_h};
  const TOTAL_H     = {total_h};
  const X_STEP      = {x_step};
  const STAR_CELL_GAP = {star_cell_gap};
  const STAR_CELL_SIZE = {star_cell_size!r};
  const COEF_LO     = {clo!r};
  const COEF_HI     = {chi!r};
  const OBS_MIN     = {obs_min_v!r};
  const OBS_MAX     = {obs_max_v!r};
  const OBS_MEAN    = {obs_mean_v!r};

  const STAR_POS_LEVEL = ["", "#FECACA", "#B91C1C", "#FF2600"];
  const STAR_NEG_LEVEL = ["", "#BFDBFE", "#1E3A8A", "#0433FF"];
  const STAR_ZERO_POS  = "#DC2626";
  const STAR_ZERO_NEG  = "#0433FF";
  const STAR_ZERO_BG   = "#E5E7EB";
  const STAR_ZERO_STROKE = "#9CA3AF";
  const OBS_FILL    = "#9CA3AF";
  const SPECIAL_FULL    = "#FF2F92";
  const SPECIAL_NOTEST  = "#ff8c00";
  const SPECIAL_BESTTEST = "#FACC15";
  const ACTIVE_COL  = "#7C3AED";

  const SIG_COLOR   = {sig_colors_js};
  const SIG_BG      = {sig_bg_js};
  const SIG_LABEL   = {sig_labels_js};

  const ALT_GROUP_COLOR_MAP = {alt_group_color_map_js};
  const STARRED_CONTROLS    = new Set({starred_controls_js});
  const BASE_MATRIX_CONTROLS = {controls_js};
  const MUST_CONTROLS        = {must_controls_js};
  const SIG_MATRIX_CONTROLS  = {sig_controls_js};

  function matrixControls() {{
    return state.controlColor === 'stats' ? SIG_MATRIX_CONTROLS : BASE_MATRIX_CONTROLS;
  }}

  function matrixH() {{
    return Math.max(ROW_H, matrixControls().length * ROW_H);
  }}

  function obsPanelY() {{
    return MATRIX_Y + matrixH() + 24;
  }}

  function totalH() {{
    return obsPanelY() + OBS_H + 22;
  }}

  function controlIncluded(rec, name) {{
    const names = state.controlColor === 'stats'
      ? (rec.controls_all || [])
      : (rec.included_matrix_controls || []);
    return names.includes(name);
  }}

  /* ── Scale helpers ──────────────────────────────────────── */
  function cy(v) {{
    if (Math.abs(COEF_HI - COEF_LO) < 1e-12) return COEF_Y + COEF_H / 2;
    const t = (v - COEF_LO) / (COEF_HI - COEF_LO);
    return COEF_Y + COEF_H - t * COEF_H;
  }}

  function obsY(v) {{
    let lo = Math.min(OBS_MIN, OBS_MEAN, OBS_MAX);
    let hi = Math.max(OBS_MIN, OBS_MEAN, OBS_MAX);
    if (lo === hi) {{ lo -= 1; hi += 1; }} else {{ const p = (hi - lo) * 0.06; lo -= p; hi += p; }}
    const oy = obsPanelY();
    if (Math.abs(hi - lo) < 1e-12) return oy + OBS_H / 2;
    return oy + OBS_H - ((v - lo) / (hi - lo)) * OBS_H;
  }}

  /* ── State ──────────────────────────────────────────────── */
  const BUNDLE_STATE = (() => {{
    try {{
      return (window.parent && window.parent !== window && window.parent.__rmBundleState)
        ? window.parent.__rmBundleState
        : {{}};
    }} catch (e) {{
      return {{}};
    }}
  }})();
  const INITIAL_SORT = ['coef', 'obs', 'signed_p'].includes(BUNDLE_STATE.sort)
    ? BUNDLE_STATE.sort
    : 'signed_p';
  const INITIAL_MODE = (BUNDLE_STATE.mode === 'compact' && COMPACT_ENABLED)
    ? 'compact'
    : (BUNDLE_STATE.mode === 'detail' ? 'detail' : '{initial_mode}');
  const INITIAL_CONTROL_COLOR = ['run', 'stats'].includes(BUNDLE_STATE.controlColor)
    ? BUNDLE_STATE.controlColor
    : 'run';
  const state = {{
    sort: INITIAL_SORT,
    mode: INITIAL_MODE,
    sigFilter: new Set([0, 1, 2, 3]),
    controlSigFilters: new Set(),
    showCI: {{ 99: true, 95: true, 90: true }},
    showFull: true,
    showNotest: true,
    showBestTest: true,
    controlColor: INITIAL_CONTROL_COLOR,
    scrollX: 0,
  }};
  const guideJumpCursor = {{ full: -1, nocontrol: -1, besttest: -1 }};
  function publishBundleState() {{
    try {{
      if (window.parent && window.parent !== window) {{
        window.parent.__rmBundleState = {{ sort: state.sort, mode: state.mode, controlColor: state.controlColor }};
        window.parent.postMessage({{
          type: 'rmBundleState',
          sort: state.sort,
          mode: state.mode,
          controlColor: state.controlColor,
        }}, '*');
      }}
    }} catch (e) {{
      // Standalone HTML has no parent bundle state.
    }}
  }}
  let sortedOrder = [];   // all record indices after current sort
  let visibleOrder = [];  // filtered record indices after current sort
  let activeIdx = -1;
  let pinnedIdx = -1;

  /* ── DOM refs ───────────────────────────────────────────── */
  const chartVScroll = document.getElementById('chart-vscroll');
  const cvWrap     = document.getElementById('cv-wrap');
  const cvMain     = document.getElementById('cv-main');
  const cvOv       = document.getElementById('cv-ov');
  const sbTrack    = document.getElementById('sb-track');
  const sbThumb    = document.getElementById('sb-thumb');
  const leftSb     = document.getElementById('left-sb');
  const infoPanel  = document.getElementById('info-panel');
  const panelPH    = document.getElementById('panel-placeholder');
  const panelCt    = document.getElementById('panel-content');

  const mainCtx  = cvMain.getContext('2d');
  const ovCtx    = cvOv.getContext('2d');

  /* ── Resize canvas to match container ──────────────────── */
  function resizeCanvas() {{
    const w = chartVScroll.clientWidth;
    const dpr = window.devicePixelRatio || 1;
    const th = totalH();
    cvMain.width  = w * dpr;
    cvMain.height = th * dpr;
    cvMain.style.width  = w + 'px';
    cvMain.style.height = th + 'px';
    cvOv.width  = w * dpr;
    cvOv.height = th * dpr;
    cvOv.style.width  = w + 'px';
    cvOv.style.height = th + 'px';
    mainCtx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ovCtx.setTransform(dpr, 0, 0, dpr, 0, 0);
    leftSb.style.height = th + 'px';
    leftSb.style.width  = LEFT + 'px';
    syncObsTickPositions();
  }}

  function syncObsTickPositions() {{
    const vals = [OBS_MAX, OBS_MEAN, OBS_MIN];
    document.querySelectorAll('.obs-tick').forEach((el, idx) => {{
      el.style.top = (obsY(vals[idx] ?? OBS_MEAN) - 6) + 'px';
    }});
  }}

  const ro = new ResizeObserver(() => {{
    invalidateCompactBitmap();
    resizeCanvas();
    requestRender();
    updateScrollbar();
  }});
  ro.observe(chartVScroll);
  resizeCanvas();

  /* ── Sort ───────────────────────────────────────────────── */
  function updateVisibleOrder() {{
    visibleOrder = sortedOrder.filter(idx => passesControlSigFilters(records[idx]));
  }}

  function displayN() {{
    return visibleOrder.length;
  }}

  function reSort() {{
    sortedOrder = records.map((_, i) => i);
    if (state.sort === 'coef') {{
      sortedOrder.sort((a, b) => records[a].coef - records[b].coef);
    }} else if (state.sort === 'signed_p') {{
      sortedOrder.sort((a, b) => {{
        const ra = records[a], rb = records[b];
        const sa = Math.sign(ra.coef) / Math.max(ra.p_value, 1e-300);
        const sb = Math.sign(rb.coef) / Math.max(rb.p_value, 1e-300);
        return sa - sb;
      }});
    }} else {{
      sortedOrder.sort((a, b) => records[a].obs - records[b].obs);
    }}
    updateVisibleOrder();
  }}
  reSort();

  /* ── Run-length matrix colors (recomputed after every sort) ─ */
  let runTValues = null; // Float32Array[displayN() * nRows], index = col * nRows + row

  function computeRunColors() {{
    const controls = matrixControls();
    const nRows = controls.length;
    const nCols = displayN();
    if (nRows === 0 || nCols === 0) {{ runTValues = null; return; }}
    runTValues = new Float32Array(nCols * nRows);
    let maxRunLen = 1;
    const runs = [];
    for (let row = 0; row < nRows; row++) {{
      const name = controls[row];
      let start = null;
      for (let col = 0; col <= nCols; col++) {{
        const included = col < nCols && controlIncluded(records[visibleOrder[col]], name);
        if (included && start === null) {{ start = col; }}
        else if (!included && start !== null) {{
          const rlen = col - start;
          runs.push([row, start, col, rlen]);
          if (rlen > maxRunLen) maxRunLen = rlen;
          start = null;
        }}
      }}
    }}
    for (const [row, s, e, rlen] of runs) {{
      const t = Math.pow(rlen / maxRunLen, 0.6);
      for (let col = s; col < e; col++) runTValues[col * nRows + row] = t;
    }}
  }}
  computeRunColors();

  /* ── Visible range ──────────────────────────────────────── */
  const VIRT_BUF = 1500;

  function visibleRange() {{
    const nCols = displayN();
    const vpW = chartVScroll.clientWidth - LEFT;
    const sc  = state.scrollX;
    const step = xStep();
    const fc  = Math.max(0, Math.floor((sc - VIRT_BUF) / step));
    const lc  = Math.min(nCols - 1, Math.ceil((sc + vpW + VIRT_BUF) / step));
    return [fc, lc];
  }}

  /* ── Draw helpers ───────────────────────────────────────── */
  function plotViewportW() {{
    return Math.max(0, chartVScroll.clientWidth - LEFT - RIGHT);
  }}

  function xStep() {{
    const nCols = displayN();
    if (state.mode !== 'compact' || nCols <= 0) return X_STEP;
    const compactBase = Math.min(nCols, COMPACT_BASE_N);
    return Math.max(0.02, Math.min(X_STEP, plotViewportW() / Math.max(compactBase, 1)));
  }}

  function maxScrollX() {{
    const vpW = chartVScroll.clientWidth - LEFT;
    return Math.max(0, displayN() * xStep() - vpW);
  }}

  function clampScrollX() {{
    state.scrollX = Math.max(0, Math.min(maxScrollX(), state.scrollX));
  }}

  function colX(col) {{
    const step = xStep();
    return LEFT + col * step + step / 2;
  }}

  function colLeft(col) {{ return LEFT + col * xStep(); }}

  function isDimmed(rec) {{
    return !state.sigFilter.has(rec.star);
  }}

  function specialVisible(kind) {{
    if (state.mode === 'detail') return true;
    if (kind === 'full') return state.showFull;
    if (kind === 'nocontrol') return state.showNotest;
    return state.showBestTest;
  }}

  function specialMatches(rec, kind) {{
    if (kind === 'full') return rec.is_full;
    if (kind === 'nocontrol') return rec.is_no_controls_test;
    return rec.is_best_test;
  }}

  function controlStat(rec, name) {{
    return (rec.control_stats || []).find(item => item.name === name);
  }}

  function controlIsSignificant(rec, name) {{
    const stat = controlStat(rec, name);
    return !!stat && starLevel(Number(stat.p_value)) > 0;
  }}

  function passesControlSigFilters(rec) {{
    if (state.controlSigFilters.size === 0) return true;
    for (const name of state.controlSigFilters) {{
      if (!controlIsSignificant(rec, name)) return false;
    }}
    return true;
  }}

  /* ── Panel frames & grid (static background, full width) ── */
  function drawBackground(ctx, viewportW = chartVScroll.clientWidth) {{
    const vpW = viewportW;
    const plotW = Math.max(0, vpW - LEFT - RIGHT);
    const rxR = vpW - RIGHT; // right edge of clip region (screen coords)
    const controls = matrixControls();
    // COEF grid lines (5 levels) — use screen coords to avoid 40k-px paths
    ctx.lineWidth = 0.75;
    for (let j = 0; j < 5; j++) {{
      const gy = COEF_Y + COEF_H * j / 4;
      ctx.strokeStyle = (j === 0 || j === 2 || j === 4) ? '#E5E7EB' : '#F3F4F6';
      ctx.beginPath();
      ctx.moveTo(LEFT, gy);
      ctx.lineTo(rxR, gy);
      ctx.stroke();
    }}
    // MATRIX row separators
    for (let r = 0; r < controls.length; r++) {{
      const ry = MATRIX_Y + r * ROW_H;
      ctx.strokeStyle = '#F3F4F6';
      ctx.lineWidth = 0.6;
      ctx.beginPath();
      ctx.moveTo(LEFT, ry);
      ctx.lineTo(rxR, ry);
      ctx.stroke();
    }}
    // OBS guide lines (max/mean/min)
    const obsRefVals = [OBS_MAX, OBS_MEAN, OBS_MIN];
    for (let i = 0; i < 3; i++) {{
      const yt = obsY(obsRefVals[i]);
      const isMean = i === 1;
      ctx.strokeStyle = isMean ? '#EF4444' : '#F3F4F6';
      ctx.lineWidth = isMean ? 1.5 : 0.7;
      if (isMean) {{
        ctx.setLineDash([5, 4]);
        ctx.globalAlpha = 0.55;
      }}
      ctx.beginPath();
      ctx.moveTo(LEFT, yt);
      ctx.lineTo(rxR, yt);
      ctx.stroke();
      if (isMean) {{
        ctx.setLineDash([]);
        ctx.globalAlpha = 1.0;
      }}
    }}
    // Swimlane backgrounds for alt groups
    const altGroups = DATA.matrixAltGroups || [];
    const altGroupColors = {alt_group_colors_js};
    let cIdx = 0;
    for (const grp of altGroups) {{
      if (grp.kind !== 'controls_test') {{ cIdx++; continue; }}
      const groupNames = BASE_MATRIX_CONTROLS.slice(grp.start, grp.end + 1);
      const rows = groupNames.map(name => controls.indexOf(name)).filter(row => row >= 0);
      if (!rows.length) {{ cIdx++; continue; }}
      const s = Math.min(...rows), e = Math.max(...rows);
      if (s < 0 || e <= s || e >= controls.length) {{ cIdx++; continue; }}
      const fill = altGroupColors[cIdx % altGroupColors.length];
      cIdx++;
      ctx.globalAlpha = 0.12;
      ctx.fillStyle = fill;
      ctx.fillRect(LEFT, MATRIX_Y + s * ROW_H, plotW, (e - s + 1) * ROW_H);
      ctx.globalAlpha = 1.0;
    }}
  }}

  /* ── Guides (special vertical lines) ───────────────────── */
  function drawGuides(ctx, fc, lc) {{
    ctx.lineWidth = 1;
    for (let col = fc; col <= lc; col++) {{
      const rec = records[visibleOrder[col]];
      if (!passesControlSigFilters(rec)) continue;
      const isFull    = specialVisible('full') && rec.is_full;
      const isNotest  = specialVisible('nocontrol') && rec.is_no_controls_test;
      const isBestTest = specialVisible('besttest') && rec.is_best_test;
      if (!isFull && !isNotest && !isBestTest) continue;
      const x = colX(col) - state.scrollX;
      ctx.strokeStyle = isFull ? SPECIAL_FULL : isBestTest ? SPECIAL_BESTTEST : SPECIAL_NOTEST;
      ctx.setLineDash([3, 2]);
      ctx.beginPath();
      ctx.moveTo(x, STAR_Y); ctx.lineTo(x, STAR_Y + STAR_H);
      if (state.mode === 'compact') {{
        ctx.moveTo(x, COEF_Y); ctx.lineTo(x, COEF_Y + COEF_H);
      }} else {{
        const coefAt = cy(rec.coef);
        const gapR = 4.8;
        ctx.moveTo(x, COEF_Y); ctx.lineTo(x, Math.max(COEF_Y, coefAt - gapR));
        ctx.moveTo(x, Math.min(COEF_Y + COEF_H, coefAt + gapR)); ctx.lineTo(x, COEF_Y + COEF_H);
      }}
      ctx.moveTo(x, MATRIX_Y); ctx.lineTo(x, MATRIX_Y + matrixH());
      const oy = obsPanelY();
      ctx.moveTo(x, oy); ctx.lineTo(x, oy + OBS_H);
      ctx.stroke();
      ctx.setLineDash([]);
    }}
  }}

  /* ── STARS panel ────────────────────────────────────────── */
  function drawStars(ctx, fc, lc) {{
    for (let col = fc; col <= lc; col++) {{
      const rec = records[visibleOrder[col]];
      if (!passesControlSigFilters(rec)) continue;
      const dimmed = isDimmed(rec);
      const x = colX(col) - state.scrollX;
      const step = xStep();
      const compact = state.mode === 'compact';
      ctx.globalAlpha = dimmed ? 0.13 : 1.0;
      const sign = rec.coef < 0 ? -1 : 1;
      const starColors = sign < 0 ? STAR_NEG_LEVEL : STAR_POS_LEVEL;
      if (compact) {{
        const barW = Math.max(0.5, Math.min(step, step * 0.82));
        const bx = colLeft(col) - state.scrollX + (step - barW) / 2;
        const segGap = Math.max(0.35, Math.min(1.2, STAR_H * 0.045));
        const segmentH = (STAR_H - 2 * segGap) / 3;
        if (rec.star === 0) {{
          const by = STAR_Y + STAR_H - segmentH;
          ctx.fillStyle = STAR_ZERO_STROKE;
          ctx.fillRect(bx, by, barW, segmentH / 2);
          ctx.fillStyle = sign < 0 ? STAR_NEG_LEVEL[3] : STAR_POS_LEVEL[3];
          ctx.fillRect(bx, by + segmentH / 2, barW, segmentH / 2);
          continue;
        }}
        for (let blk = 0; blk < rec.star; blk++) {{
          const by = STAR_Y + STAR_H - (blk + 1) * segmentH - blk * segGap;
          ctx.fillStyle = starColors[blk + 1];
          ctx.fillRect(bx, by, barW, segmentH);
        }}
        continue;
      }}
      const gap = STAR_CELL_GAP;
      const side = STAR_CELL_SIZE;
      const baseY = STAR_Y + STAR_H - gap;
      if (rec.star === 0) {{
        const cy0 = baseY - side / 2;
        ctx.beginPath();
        ctx.arc(x, cy0, side / 2, 0, Math.PI * 2);
        ctx.fillStyle = '#FFFFFF';
        ctx.globalAlpha = 1.0;
        ctx.fill();
        ctx.globalAlpha = dimmed ? 0.13 : 1.0;
        ctx.strokeStyle = sign < 0 ? STAR_ZERO_NEG : STAR_ZERO_POS;
        ctx.lineWidth = 1.65;
        ctx.stroke();
        continue;
      }}
      for (let blk = 0; blk < rec.star; blk++) {{
        const cyDot = baseY - (blk + 0.5) * side - blk * gap;
        ctx.beginPath();
        ctx.arc(x, cyDot, side / 2, 0, Math.PI * 2);
        ctx.fillStyle = starColors[blk + 1];
        ctx.fill();
        ctx.strokeStyle = 'rgba(17,24,39,.08)';
        ctx.lineWidth = 0.55;
        ctx.stroke();
      }}
    }}
    ctx.globalAlpha = 1.0;
  }}

  /* ── CI bands ───────────────────────────────────────────── */
  function drawCIBands(ctx, fc, lc) {{
    if (lc < fc) return;
    const levels = [
      {{ lo: 'ci99_lo', hi: 'ci99_hi', key: 99, alpha: 0.16 }},
      {{ lo: 'ci95_lo', hi: 'ci95_hi', key: 95, alpha: 0.22 }},
      {{ lo: 'ci90_lo', hi: 'ci90_hi', key: 90, alpha: 0.28 }},
    ];
    const step = xStep();
    const bandW = Math.max(0.25, step);
    for (const lvl of levels) {{
      if (!state.showCI[lvl.key]) continue;
      ctx.fillStyle = `rgba(17,24,39,${{lvl.alpha}})`;
      for (let col = fc; col <= lc; col++) {{
        const rec = records[visibleOrder[col]];
        if (!passesControlSigFilters(rec)) continue;
        const x = colX(col) - state.scrollX;
        const yHi = cy(rec[lvl.hi]);
        const yLo = cy(rec[lvl.lo]);
        ctx.fillRect(x - bandW / 2, yHi, bandW, yLo - yHi);
      }}
    }}
  }}

  /* ── Coef points & zero line ────────────────────────────── */
  function drawCoef(ctx, fc, lc, viewportW = chartVScroll.clientWidth) {{
    // zero line — use screen coords to avoid anti-aliasing bleed at clip boundary
    const zy = cy(0);
    if (zy >= COEF_Y && zy <= COEF_Y + COEF_H) {{
      const rxR = viewportW - RIGHT;
      ctx.strokeStyle = '#EF4444';
      ctx.lineWidth = 1.5;
      ctx.setLineDash([5, 4]);
      ctx.globalAlpha = 0.55;
      ctx.beginPath();
      ctx.moveTo(LEFT, zy);
      ctx.lineTo(rxR, zy);
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.globalAlpha = 1.0;
    }}
    // points
    const step = xStep();
    const pointR = state.mode === 'compact'
      ? Math.max(0.55, Math.min(1.45, step * 0.58))
      : Math.max(1.6, Math.min(2.6, step * 0.36));
    for (let col = fc; col <= lc; col++) {{
      const rec = records[visibleOrder[col]];
      if (!passesControlSigFilters(rec)) continue;
      const dimmed = isDimmed(rec);
      ctx.globalAlpha = dimmed ? 0.13 : 1.0;
      ctx.fillStyle = rec.color;
      ctx.beginPath();
      ctx.arc(colX(col) - state.scrollX, cy(rec.coef), pointR, 0, Math.PI * 2);
      ctx.fill();
    }}
    ctx.globalAlpha = 1.0;
  }}

  /* ── Control matrix ─────────────────────────────────────── */
  function statColorForControl(rec, name) {{
    const stat = (rec.control_stats || []).find(item => item.name === name);
    if (!stat) return '#9CA3AF';
    const coef = Number(stat.coef);
    const level = starLevel(Number(stat.p_value));
    if (level === 0) return '#BFC5CF';
    return (coef < 0 ? STAR_NEG_LEVEL : STAR_POS_LEVEL)[level];
  }}

  function drawMatrix(ctx, fc, lc) {{
    const controls = matrixControls();
    const nRows = controls.length;
    for (let ci = 0; ci < nRows; ci++) {{
      const name = controls[ci];
      const ry = MATRIX_Y + ci * ROW_H;
      for (let col = fc; col <= lc; col++) {{
        const rec = records[visibleOrder[col]];
        if (!passesControlSigFilters(rec)) continue;
        if (!controlIncluded(rec, name)) continue;
        const dimmed = isDimmed(rec);
        ctx.globalAlpha = dimmed ? 0.13 : 1.0;
        const isFull   = specialVisible('full') && rec.is_full;
        const isNotest = specialVisible('nocontrol') && rec.is_no_controls_test;
        const isBestTest = specialVisible('besttest') && rec.is_best_test;
        const groupFill = ALT_GROUP_COLOR_MAP[name];
        const step = xStep();
        const compact = state.mode === 'compact';
        let normalFill = '#1F2937';
        if (!compact && runTValues) {{
          const t = runTValues[col * nRows + ci];
          const v = Math.round(200 - 200 * t);
          normalFill = `rgb(${{v}},${{v}},${{v}})`;
        }}
        const statsFill = statColorForControl(rec, name);
        if (state.controlColor === 'stats') {{
          ctx.fillStyle = statsFill;
        }} else {{
          ctx.fillStyle = compact ? (groupFill || normalFill) : (isFull ? SPECIAL_FULL : isBestTest ? SPECIAL_BESTTEST : isNotest ? SPECIAL_NOTEST : (groupFill || normalFill));
        }}
        const rw = compact ? Math.min(2.0, step * 0.55) : Math.max(1, step * 0.80);
        const rx = compact
          ? colX(col) - state.scrollX - rw / 2
          : colLeft(col) - state.scrollX + step * 0.10;
        const rh = compact ? ROW_H : ROW_H - 4;
        ctx.beginPath();
        if (!compact && ctx.roundRect) ctx.roundRect(rx, ry + 2, rw, rh, 1.5);
        else ctx.rect(rx, compact ? ry : ry + 2, rw, rh);
        ctx.fill();
      }}
    }}
    ctx.globalAlpha = 1.0;
  }}

  /* ── OBS bars ───────────────────────────────────────────── */
  function drawObs(ctx, fc, lc) {{
    const baseline = obsY(OBS_MEAN);
    const obsGap = 2.0;
    const obsMinH = 0.7;
    for (let col = fc; col <= lc; col++) {{
      const rec = records[visibleOrder[col]];
      if (!passesControlSigFilters(rec)) continue;
      const dimmed = isDimmed(rec);
      const step = xStep();
      const compact = state.mode === 'compact';
      const isFull   = specialVisible('full') && rec.is_full;
      const isNotest = specialVisible('nocontrol') && rec.is_no_controls_test;
      const isBestTest = specialVisible('besttest') && rec.is_best_test;
      ctx.globalAlpha = dimmed ? 0.13 : 0.78;
      ctx.fillStyle = compact ? OBS_FILL : (isFull ? SPECIAL_FULL : isBestTest ? SPECIAL_BESTTEST : isNotest ? SPECIAL_NOTEST : OBS_FILL);
      if (!compact && (isFull || isNotest || isBestTest)) ctx.globalAlpha = dimmed ? 0.13 : 1.0;
      const vy = obsY(rec.obs);
      let barY, barH;
      if (vy < baseline) {{
        const bot = baseline - obsGap;
        barY = Math.min(vy, bot - obsMinH);
        barH = bot - barY;
      }} else if (vy > baseline) {{
        barY = baseline + obsGap;
        const bot = Math.max(vy, barY + obsMinH);
        barH = bot - barY;
      }} else {{
        barY = baseline + obsGap;
        barH = obsMinH;
      }}
      const bx = colLeft(col) - state.scrollX + (compact ? 0 : step * 0.12);
      const bw = compact ? Math.max(0.5, step) : Math.max(1, step * 0.76);
      ctx.beginPath();
      if (!compact && ctx.roundRect) ctx.roundRect(bx, barY, bw, barH, 1.5);
      else ctx.rect(bx, barY, bw, barH);
      ctx.fill();
    }}
    ctx.globalAlpha = 1.0;
  }}

  /* ── White sidebar mask (sticky left labels) ────────────── */
  function drawSidebarMask(ctx) {{
    ctx.fillStyle = '#FFFFFF';
    ctx.fillRect(0, 0, LEFT, totalH());
  }}

  /* ── Main render ────────────────────────────────────────── */
  let _raf = 0;
  let compactBitmapCanvas = null;
  let compactBitmapKey = "";

  function compactBitmapStateKey() {{
    const dpr = window.devicePixelRatio || 1;
    const sig = [...state.sigFilter].sort((a, b) => a - b).join('');
    const controlSig = [...state.controlSigFilters].sort().join(',');
    const ci = [90, 95, 99].filter(k => state.showCI[k]).join('');
    return [
      chartVScroll.clientWidth,
      totalH(),
      dpr,
      N,
      displayN(),
      xStep().toFixed(8),
      state.sort,
      state.controlColor,
      sig,
      controlSig,
      ci,
      state.showFull ? 'F1' : 'F0',
      state.showNotest ? 'N1' : 'N0',
      state.showBestTest ? 'B1' : 'B0',
    ].join('|');
  }}

  function invalidateCompactBitmap() {{
    compactBitmapCanvas = null;
    compactBitmapKey = "";
  }}

  function compactBitmapCssWidth() {{
    return Math.max(1, Math.ceil(LEFT + displayN() * xStep() + RIGHT));
  }}

  function canUseCompactBitmap() {{
    if (state.mode !== 'compact' || displayN() <= 0) return false;
    const dpr = window.devicePixelRatio || 1;
    const w = compactBitmapCssWidth();
    const h = Math.max(1, Math.ceil(totalH()));
    return (
      w * dpr <= MAX_COMPACT_BITMAP_DIM
      && h * dpr <= MAX_COMPACT_BITMAP_DIM
      && w * h * dpr * dpr <= MAX_COMPACT_BITMAP_AREA
    );
  }}

  function ensureCompactBitmap() {{
    if (!canUseCompactBitmap()) return null;
    const key = compactBitmapStateKey();
    if (compactBitmapCanvas && compactBitmapKey === key) return compactBitmapCanvas;

    const dpr = window.devicePixelRatio || 1;
    const cssW = compactBitmapCssWidth();
    const cssPlotW = Math.max(0, cssW - LEFT - RIGHT);
    const th = totalH();
    const bmp = document.createElement('canvas');
    bmp.width = Math.ceil(cssW * dpr);
    bmp.height = Math.ceil(th * dpr);
    const bmpCtx = bmp.getContext('2d');
    bmpCtx.setTransform(dpr, 0, 0, dpr, 0, 0);
    bmpCtx.clearRect(0, 0, cssW, th);

    const oldScrollX = state.scrollX;
    state.scrollX = 0;
    bmpCtx.save();
    try {{
      bmpCtx.beginPath();
      bmpCtx.rect(LEFT, 0, cssPlotW, th);
      bmpCtx.clip();
      drawBackground(bmpCtx, cssW);
      drawCIBands(bmpCtx, 0, displayN() - 1);
      drawStars(bmpCtx, 0, displayN() - 1);
      drawCoef(bmpCtx, 0, displayN() - 1, cssW);
      drawMatrix(bmpCtx, 0, displayN() - 1);
      drawObs(bmpCtx, 0, displayN() - 1);
      drawGuides(bmpCtx, 0, displayN() - 1);
    }} finally {{
      bmpCtx.restore();
      state.scrollX = oldScrollX;
    }}
    compactBitmapCanvas = bmp;
    compactBitmapKey = key;
    return compactBitmapCanvas;
  }}

  function renderCompactBitmap() {{
    const bmp = ensureCompactBitmap();
    if (!bmp) return false;
    mainCtx.clearRect(0, 0, cvMain.width, cvMain.height);
    const dpr = window.devicePixelRatio || 1;
    const th = totalH();
    const visibleW = Math.max(0, chartVScroll.clientWidth - LEFT - RIGHT);
    const sourceX = LEFT + state.scrollX;
    const availableW = Math.max(0, compactBitmapCssWidth() - RIGHT - sourceX);
    const sw = Math.min(visibleW, availableW);
    if (sw > 0) {{
      mainCtx.drawImage(
        bmp,
        sourceX * dpr,
        0,
        sw * dpr,
        th * dpr,
        LEFT,
        0,
        sw,
        th
      );
    }}
    drawSidebarMask(mainCtx);
    drawPanelFrames(mainCtx);
    drawPanelLabels(mainCtx);
    return true;
  }}

  function requestRender() {{
    if (!_raf) _raf = requestAnimationFrame(() => {{ _raf = 0; render(); }});
  }}

  function render() {{
    if (state.mode === 'compact' && renderCompactBitmap()) return;
    mainCtx.clearRect(0, 0, cvMain.width, cvMain.height);
    const [fc, lc] = visibleRange();
    // Clip to chart area to avoid bleeding into left/right margins during scroll
    mainCtx.save();
    mainCtx.beginPath();
    mainCtx.rect(LEFT, 0, chartVScroll.clientWidth - LEFT - RIGHT, totalH());
    mainCtx.clip();
    drawBackground(mainCtx);
    drawCIBands(mainCtx, fc, lc);
    drawStars(mainCtx, fc, lc);
    drawCoef(mainCtx, fc, lc);
    drawMatrix(mainCtx, fc, lc);
    drawObs(mainCtx, fc, lc);
    drawGuides(mainCtx, fc, lc);
    mainCtx.restore();
    // These three are unclipped — drawn over the full canvas
    drawSidebarMask(mainCtx);
    drawPanelFrames(mainCtx);
    drawPanelLabels(mainCtx);
  }}

  /* ── Panel frames (drawn outside clip so borders are fully visible) ── */
  function drawPanelFrames(ctx) {{
    const vpW = chartVScroll.clientWidth;
    const frameW = Math.max(0, vpW - LEFT - RIGHT);
    const panelDefs = [
      [STAR_Y, STAR_H],
      [COEF_Y, COEF_H],
      [MATRIX_Y, matrixH()],
      [obsPanelY(), OBS_H],
    ];
    ctx.strokeStyle = '#D1D5DB';
    ctx.lineWidth = 1.25;
    for (const [py, ph] of panelDefs) {{
      ctx.beginPath();
      if (ctx.roundRect) ctx.roundRect(LEFT, py, frameW, ph, 2);
      else ctx.rect(LEFT, py, frameW, ph);
      ctx.stroke();
    }}
  }}

  /* ── Panel labels (right side, rotated 90°) ─────────────── */
  function drawPanelLabels(ctx) {{
    const vpW = chartVScroll.clientWidth;
    const labelX = vpW - RIGHT + 18;
    ctx.save();
    ctx.fillStyle = '#9CA3AF';
    ctx.font = '600 9px "RM Courier New","Courier New",monospace';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    const panelDefs = [
      [STAR_Y, STAR_H, 'STARS'],
      [COEF_Y, COEF_H, 'COEF'],
      [MATRIX_Y, matrixH(), 'CONTROLS'],
      [obsPanelY(), OBS_H, 'OBS'],
    ];
    for (const [py, ph, lbl] of panelDefs) {{
      const mid = py + ph / 2;
      ctx.save();
      ctx.translate(labelX, mid);
      ctx.rotate(Math.PI / 2);
      ctx.fillText(lbl, 0, 0);
      ctx.restore();
    }}
    ctx.restore();
  }}

  /* ── Overlay (hover/pin highlight) ─────────────────────── */
  function drawOverlay(col) {{
    ovCtx.clearRect(0, 0, cvOv.width, cvOv.height);
    if (state.mode === 'compact') return;
    if (col < 0) return;
    const x = colX(col) - state.scrollX;
    const vpW = chartVScroll.clientWidth;
    ovCtx.save();
    ovCtx.beginPath();
    ovCtx.rect(LEFT, 0, vpW - LEFT - RIGHT, totalH());
    ovCtx.clip();
    // Purple column tint — only within panel areas, skip inter-panel gaps
    ovCtx.fillStyle = 'rgba(124,58,237,0.15)';
    const step = xStep();
    const tw = step;
    const tx = x - step / 2;
    for (const [py, ph] of [[STAR_Y,STAR_H],[COEF_Y,COEF_H],[MATRIX_Y,matrixH()],[obsPanelY(),OBS_H]]) {{
      ovCtx.fillRect(tx, py, tw, ph);
    }}
    // Active ring on coef point
    const rec = records[visibleOrder[col]];
    ovCtx.strokeStyle = ACTIVE_COL;
    ovCtx.lineWidth = 2;
    ovCtx.beginPath();
    ovCtx.arc(x, cy(rec.coef), 6, 0, Math.PI * 2);
    ovCtx.stroke();
    ovCtx.restore();
  }}

  function clearOverlay() {{
    ovCtx.clearRect(0, 0, cvOv.width, cvOv.height);
  }}

  /* ── Hit testing ────────────────────────────────────────── */
  function colFromEvent(e) {{
    const rect = cvOv.getBoundingClientRect();
    const worldX = (e.clientX - rect.left) + state.scrollX;
    const col = Math.floor((worldX - LEFT) / xStep());
    if (col < 0 || col >= displayN()) return -1;
    return col;
  }}

  function nextVisibleCol(fromCol, dir) {{
    for (let col = fromCol + dir; col >= 0 && col < displayN(); col += dir) {{
      return col;
    }}
    return -1;
  }}

  function guideCols(kind) {{
    const cols = [];
    for (let col = 0; col < displayN(); col++) {{
      const rec = records[visibleOrder[col]];
      if (specialMatches(rec, kind)) cols.push(col);
    }}
    return cols;
  }}

  function revealCol(col) {{
    const target = colX(col) - LEFT - plotViewportW() / 2;
    state.scrollX = Math.max(0, Math.min(maxScrollX(), target));
    requestRender();
    updateScrollbar();
  }}

  function jumpToGuide(kind) {{
    if (state.mode !== 'detail') return false;
    const cols = guideCols(kind);
    if (cols.length === 0) return true;
    const currentPos = cols.indexOf(activeCol);
    const basePos = currentPos >= 0 ? currentPos : guideJumpCursor[kind];
    const nextPos = (basePos + 1 + cols.length) % cols.length;
    const col = cols[nextPos];
    guideJumpCursor[kind] = nextPos;
    revealCol(col);
    pinnedIdx = -1;
    activate(col, true);
    return true;
  }}

  /* ── Info panel update ──────────────────────────────────── */
  function fmt(v) {{ return Number(v).toFixed(4); }}

  function starLevel(p) {{
    return p < 0.01 ? 3 : p < 0.05 ? 2 : p < 0.10 ? 1 : 0;
  }}

  function starsForP(p) {{
    const level = starLevel(p);
    return level === 0 ? '.' : '*'.repeat(level);
  }}

  function escHtml(value) {{
    return String(value)
      .replaceAll('&', '&amp;')
      .replaceAll('<', '&lt;')
      .replaceAll('>', '&gt;')
      .replaceAll('"', '&quot;')
      .replaceAll("'", '&#39;');
  }}

  function controlBadge(stat, blank = false) {{
    if (blank) return `<span class="coef-badge blank">&nbsp;</span>`;
    if (!stat) return `<span class="coef-badge missing">--</span>`;
    const coef = Number(stat.coef);
    const level = starLevel(Number(stat.p_value));
    const sign = coef < 0 ? -1 : 1;
    const signedLevel = sign * level;
    const label = level === 0 ? `0${{sign < 0 ? '-' : '+'}}` : `${{signedLevel > 0 ? '+' : ''}}${{signedLevel}}`;
    const badgeClass = level === 0
      ? `${{sign < 0 ? 'neg' : 'pos'}}-zero zero`
      : `${{sign < 0 ? 'neg' : 'pos'}}-dir ${{sign < 0 ? 'neg' : 'pos'}}-sig-${{level}}`;
    return `<span class="coef-badge ${{badgeClass}}">${{label}}</span>`;
  }}

  function controlFilterBarHtml() {{
    if (state.controlSigFilters.size === 0) return `<div class="coef-filter-bar"></div>`;
    return `<div class="coef-filter-bar has-filters"><span>keep significant:</span>${{[...state.controlSigFilters].map(name => `<span class="coef-filter-chip" title="${{escHtml(name)}}"><span>${{escHtml(name)}}</span></span>`).join('')}}<button type="button" class="coef-filter-clear" data-filter-clear="1">clear</button></div>`;
  }}

  function copyTextFallback(text) {{
    const ta = document.createElement('textarea');
    ta.value = text;
    ta.setAttribute('readonly', '');
    ta.style.position = 'fixed';
    ta.style.left = '-9999px';
    document.body.appendChild(ta);
    ta.focus();
    ta.select();
    ta.setSelectionRange(0, ta.value.length);
    try {{
      const ok = document.execCommand('copy');
      if (!ok) throw new Error('execCommand copy returned false');
      return Promise.resolve();
    }} catch (err) {{
      return Promise.reject(err);
    }} finally {{
      document.body.removeChild(ta);
    }}
  }}

  function copyTextToClipboard(text) {{
    if (navigator.clipboard && window.isSecureContext) {{
      return navigator.clipboard.writeText(text).catch(() => copyTextFallback(text));
    }}
    return copyTextFallback(text);
  }}

  function showInfo(recIdx = -1) {{
    const hasSelection = Number.isInteger(recIdx) && recIdx >= 0 && recIdx < records.length;
    const r = hasSelection ? records[recIdx] : null;
    const star = hasSelection ? r.star : 0;
    const ci99 = hasSelection ? `[${{fmt(r.ci99_lo)}}, ${{fmt(r.ci99_hi)}}]` : '';
    const ci95 = hasSelection ? `[${{fmt(r.ci95_lo)}}, ${{fmt(r.ci95_hi)}}]` : '';
    const ci90 = hasSelection ? `[${{fmt(r.ci90_lo)}}, ${{fmt(r.ci90_hi)}}]` : '';
    const adjR2 = hasSelection
      ? ((r.adj_r2 === null || r.adj_r2 === undefined) ? '-' : Number(r.adj_r2).toFixed(4))
      : '';
    const withinR2 = hasSelection
      ? ((r.within_r2 === null || r.within_r2 === undefined) ? '-' : Number(r.within_r2).toFixed(4))
      : '';
    const fStat = hasSelection
      ? ((r.f_stat === null || r.f_stat === undefined) ? '-' : Number(r.f_stat).toFixed(3))
      : '';

    const includedControls = new Set(hasSelection ? (r.controls_all || []) : []);
    const testOrder = DATA.controlsTestNames || DATA.matrixControls || [];
    const mustOrder = DATA.controlsMustNames || [];
    const testRows = testOrder;
    const testIncl = hasSelection ? testOrder.filter(c => includedControls.has(c)) : [];
    const mustIncl = hasSelection ? mustOrder.filter(c => includedControls.has(c)) : [];
    const orderedKnown = new Set([...testIncl, ...mustIncl]);
    const extraIncl = hasSelection ? (r.controls_all || []).filter(c => !orderedKnown.has(c)) : [];
    const orderedControls = [...mustIncl, ...extraIncl, ...testIncl];
    const copyControls = orderedControls.join(' ');
    const controlStats = new Map(hasSelection ? (r.control_stats || []).map(item => [item.name, item]) : []);

    const coefRow = (name, group, included = true) => {{
      const stat = controlStats.get(name);
      const selected = state.controlSigFilters.has(name);
      const showValues = hasSelection && included;
      return `
        <div class="coef-row ${{group === 'test' ? 'is-test' : ''}} ${{selected ? 'filter-selected' : ''}} ${{included ? '' : 'not-included'}}" data-filter-control="${{escHtml(name)}}" title="Click to filter by significant ${{escHtml(name)}}">
          <span class="coef-name-wrap">${{controlBadge(stat, !hasSelection || !included)}}<span class="coef-name" title="${{escHtml(name)}}">${{escHtml(name)}}</span></span>
          ${{showValues && stat
            ? `<span class="coef-val ${{Number(stat.coef) < 0 ? 'neg' : 'pos'}}">${{fmt(Number(stat.coef))}}</span>
               <span class="coef-p s${{starLevel(Number(stat.p_value))}}"><span class="coef-stars">${{starsForP(Number(stat.p_value))}}</span><span>${{Number(stat.p_value).toFixed(4)}}</span></span>`
            : hasSelection && included
              ? `<span class="coef-val placeholder">-</span><span class="coef-p placeholder"><span class="coef-stars">.</span><span>-</span></span>`
              : `<span class="coef-val placeholder"></span><span class="coef-p placeholder"><span class="coef-stars"></span><span></span></span>`
          }}
        </div>`;
    }};

    const mustRows = hasSelection ? [...mustIncl, ...extraIncl] : mustOrder;
    const coefBlock = mustRows.length === 0 && testRows.length === 0
      ? `<div class="coef-empty">${{hasSelection ? 'No controls included in this specification.' : 'No controls configured.'}}</div>`
      : `<div class="panel-coefs-head"><span>Control coefficients</span><span class="panel-coefs-meta">${{testIncl.length}}/${{testRows.length}} test · ${{mustIncl.length + extraIncl.length}} must</span></div>
         ${{mustRows.length ? `<div class="coef-group-label">MUST <span class="grp-count">(${{hasSelection ? mustIncl.length + extraIncl.length : mustRows.length}})</span></div>${{mustRows.map(c => coefRow(c, 'base', hasSelection)).join('')}}` : ''}}
         ${{testRows.length ? `<div class="coef-group-label">TEST <span class="grp-count">(${{testIncl.length}}/${{testRows.length}})</span></div>${{testRows.map(c => coefRow(c, 'test', includedControls.has(c))).join('')}}` : ''}}`;
    const filterBar = controlFilterBarHtml();
    const copyDisabled = hasSelection ? '' : ' disabled';

    panelCt.innerHTML = `
      <div class="panel-head">
        <span class="panel-head-left">
          <span class="panel-title">${{hasSelection ? `Spec #${{recIdx + 1}}&thinsp;/&thinsp;${{records.length}}` : `Spec -&thinsp;/&thinsp;${{records.length}}`}}</span>
          <button type="button" class="panel-copy" data-copy-controls="${{escHtml(copyControls)}}" title="Copy included control names"${{copyDisabled}}>COPY</button>
        </span>
        <span class="panel-sig" style="background:${{hasSelection ? SIG_BG[star] : '#F3F4F6'}};color:${{hasSelection ? SIG_COLOR[star] : 'var(--muted-2)'}}">${{hasSelection ? SIG_LABEL[star] : '--'}}</span>
      </div>
      <div class="panel-table">
        <span class="panel-key">coef</span>         <span class="panel-val">${{hasSelection ? r.coef.toFixed(5) : ''}}</span>
        <span class="panel-key">std&nbsp;err</span>  <span class="panel-val">${{hasSelection ? r.se.toFixed(5) : ''}}</span>
        <span class="panel-key">t&#8209;stat</span>  <span class="panel-val">${{hasSelection ? r.t_value.toFixed(3) : ''}}</span>
        <span class="panel-key">p&#8209;value</span> <span class="panel-val">${{hasSelection ? r.p_value.toFixed(4) : ''}}</span>
        <div class="panel-divider"></div>
        <span class="panel-key">90% CI</span>       <span class="panel-val">${{ci90}}</span>
        <span class="panel-key">95% CI</span>       <span class="panel-val">${{ci95}}</span>
        <span class="panel-key">99% CI</span>       <span class="panel-val">${{ci99}}</span>
        <div class="panel-divider"></div>
        <span class="panel-key">obs</span>           <span class="panel-val">${{hasSelection ? r.obs.toLocaleString() : ''}}</span>
        <span class="panel-key">adj&nbsp;R²</span>   <span class="panel-val">${{adjR2}}</span>
        <span class="panel-key">within&nbsp;R²</span><span class="panel-val">${{withinR2}}</span>
        <span class="panel-key">F</span>             <span class="panel-val">${{fStat}}</span>
        <div class="panel-divider"></div>
        ${{filterBar}}
        <div class="panel-coefs">${{coefBlock}}</div>
      </div>`;

    panelPH.style.display = 'none';
    panelCt.style.display = 'block';

    // highlight control labels
    document.querySelectorAll('.ctrl-lbl').forEach(el => el.classList.remove('active-ctrl'));
    const highlightedControls = hasSelection
      ? (state.controlColor === 'stats' ? (r.controls_all || []) : (r.included_matrix_controls || []))
      : [];
    highlightedControls.forEach(name => {{
      document.querySelectorAll(`.ctrl-lbl[data-control="${{escHtml(name)}}"]`)
        .forEach(el => el.classList.add('active-ctrl'));
    }});
  }}

  function clearInfo() {{
    document.querySelectorAll('.ctrl-lbl').forEach(el => el.classList.remove('active-ctrl'));
    showInfo();
  }}

  function refreshAfterControlSigFilterChange() {{
    updateVisibleOrder();
    clampScrollX();
    invalidateCompactBitmap();
    requestRender();
    updateScrollbar();
    const currentIdx = activeIdx;
    if (currentIdx >= 0 && passesControlSigFilters(records[currentIdx])) {{
      const col = visibleOrder.indexOf(currentIdx);
      activeCol = col;
      if (col >= 0 && state.mode !== 'compact') drawOverlay(col);
      showInfo(currentIdx);
    }} else {{
      clearOverlay();
      activeCol = -1;
      activeIdx = -1;
      pinnedIdx = -1;
      if (currentIdx >= 0) showInfo(currentIdx);
      else clearInfo();
    }}
  }}

  /* ── Activation ─────────────────────────────────────────── */
  let activeCol = -1;  // column index (in sorted order)

  function activate(col, pin) {{
    if (state.mode === 'compact') return;
    if (col < 0 || col >= displayN()) return;
    const recIdx = visibleOrder[col];
    if (pinnedIdx >= 0 && !pin) return;
    activeCol = col;
    activeIdx = recIdx;
    if (pin) {{
      if (pinnedIdx === recIdx) {{
        // clicking same pinned spec → deselect
        pinnedIdx = -1;
        activeCol = -1;
        activeIdx = -1;
        clearOverlay();
        clearInfo();
        return;
      }}
      pinnedIdx = recIdx;
    }}
    drawOverlay(col);
    showInfo(recIdx);
  }}

  function deactivate() {{
    if (pinnedIdx >= 0) return;
    activeCol = -1;
    activeIdx = -1;
    clearOverlay();
    clearInfo();
  }}

  /* ── Mouse events ───────────────────────────────────────── */
  cvOv.addEventListener('mousemove', e => {{
    if (state.mode === 'compact') return;
    const col = colFromEvent(e);
    if (col >= 0) {{
      if (pinnedIdx >= 0) return;  // don't move while pinned
      if (col !== activeCol) activate(col, false);
    }} else {{
      if (pinnedIdx < 0) deactivate();
    }}
  }});

  cvOv.addEventListener('mouseleave', () => {{
    if (state.mode === 'compact') return;
    if (pinnedIdx < 0) deactivate();
  }});

  cvOv.addEventListener('click', e => {{
    if (state.mode === 'compact') return;
    const col = colFromEvent(e);
    if (col < 0) {{
      pinnedIdx = -1;
      deactivate();
      return;
    }}
    const recIdx = visibleOrder[col];
    if (pinnedIdx === recIdx) {{
      // toggle off
      pinnedIdx = -1;
      deactivate();
    }} else {{
      activate(col, true);
    }}
  }});

  panelCt.addEventListener('click', e => {{
    const clearBtn = e.target.closest('[data-filter-clear]');
    if (clearBtn) {{
      state.controlSigFilters.clear();
      refreshAfterControlSigFilterChange();
      return;
    }}
    const copyBtn = e.target.closest('[data-copy-controls]');
    if (copyBtn) {{
      const text = copyBtn.dataset.copyControls || '';
      copyTextToClipboard(text).then(() => {{
        copyBtn.classList.add('copied');
        copyBtn.textContent = 'COPIED';
        window.setTimeout(() => {{
          copyBtn.classList.remove('copied');
          copyBtn.textContent = 'COPY';
        }}, 900);
      }}).catch(() => {{
        copyBtn.textContent = 'FAILED';
        window.setTimeout(() => {{ copyBtn.textContent = 'COPY'; }}, 1200);
      }});
      return;
    }}
    const row = e.target.closest('.coef-row[data-filter-control]');
    if (!row) return;
    const name = row.dataset.filterControl;
    if (!name) return;
    if (state.controlSigFilters.has(name)) state.controlSigFilters.delete(name);
    else state.controlSigFilters.add(name);
    refreshAfterControlSigFilterChange();
  }});

  /* ── Keyboard navigation ────────────────────────────────── */
  document.addEventListener('keydown', e => {{
    if (state.mode === 'compact') return;
    if (e.key === 'Escape') {{
      pinnedIdx = -1;
      activeCol = -1;
      activeIdx = -1;
      clearOverlay();
      clearInfo();
      return;
    }}
    if (e.key === 'ArrowRight' || e.key === 'ArrowLeft') {{
      e.preventDefault();
      const dir = e.key === 'ArrowRight' ? 1 : -1;
      const cur = activeCol >= 0 ? activeCol : (dir > 0 ? -1 : displayN());
      const next = nextVisibleCol(cur, dir);
      if (next < 0) return;
      activate(next, true);
      // scroll if needed
      const x = colX(next);
      const vpW = chartVScroll.clientWidth - LEFT;
      if (x - state.scrollX < LEFT + 20) {{
        state.scrollX = Math.max(0, x - LEFT - 20);
        requestRender();
        updateScrollbar();
      }} else if (x - state.scrollX > vpW - 20) {{
        const maxSc = maxScrollX();
        state.scrollX = Math.min(maxSc, x - vpW + 20);
        requestRender();
        updateScrollbar();
      }}
    }}
  }});

  /* ── Wheel scroll ───────────────────────────────────────── */
  cvOv.addEventListener('wheel', e => {{
    const isHorizontal = Math.abs(e.deltaX) >= Math.abs(e.deltaY);
    if (isHorizontal) {{
      e.preventDefault();
      state.scrollX = Math.max(0, Math.min(maxScrollX(), state.scrollX + e.deltaX));
      requestRender();
      updateScrollbar();
      if (activeCol >= 0 && pinnedIdx >= 0) drawOverlay(activeCol);
    }}
    // vertical: don't preventDefault — let it bubble to #chart-vscroll for native scroll
  }}, {{ passive: false }});
  cvWrap.addEventListener('wheel', e => {{
    if (state.mode !== 'compact') return;
    const isHorizontal = Math.abs(e.deltaX) >= Math.abs(e.deltaY);
    if (!isHorizontal) return;
    e.preventDefault();
    state.scrollX = Math.max(0, Math.min(maxScrollX(), state.scrollX + e.deltaX));
    requestRender();
    updateScrollbar();
  }}, {{ passive: false }});

  /* ── Scrollbar ──────────────────────────────────────────── */
  function updateScrollbar() {{
    clampScrollX();
    const trackW = sbTrack.clientWidth;
    const vpW = chartVScroll.clientWidth - LEFT;
    const totalW = displayN() * xStep();
    if (totalW <= vpW) {{
      sbThumb.style.display = 'none';
      return;
    }}
    sbThumb.style.display = '';
    const ratio = vpW / totalW;
    const thumbW = Math.max(20, trackW * ratio);
    const maxScroll = Math.max(0, totalW - vpW);
    const thumbLeft = maxScroll > 0 ? (state.scrollX / maxScroll) * (trackW - thumbW) : 0;
    sbThumb.style.width = thumbW + 'px';
    sbThumb.style.left  = thumbLeft + 'px';
  }}

  // Drag scrollbar
  let sbDrag = null;
  sbThumb.addEventListener('mousedown', e => {{
    sbDrag = {{ startX: e.clientX, startSc: state.scrollX }};
    e.preventDefault();
  }});
  document.addEventListener('mousemove', e => {{
    if (!sbDrag) return;
    const trackW = sbTrack.clientWidth;
    const vpW = chartVScroll.clientWidth - LEFT;
    const totalW = displayN() * xStep();
    const maxSc = maxScrollX();
    const thumbW = Math.max(20, trackW * (vpW / totalW));
    const dx = e.clientX - sbDrag.startX;
    const sc = sbDrag.startSc + dx * (maxSc / (trackW - thumbW));
    state.scrollX = Math.max(0, Math.min(maxSc, sc));
    requestRender();
    updateScrollbar();
  }});
  document.addEventListener('mouseup', () => {{ sbDrag = null; }});
  sbTrack.addEventListener('click', e => {{
    if (e.target === sbThumb) return;
    const rect = sbTrack.getBoundingClientRect();
    const trackW = sbTrack.clientWidth;
    const totalW = displayN() * xStep();
    const maxSc = maxScrollX();
    const frac = (e.clientX - rect.left) / trackW;
    state.scrollX = Math.max(0, Math.min(maxSc, frac * totalW));
    requestRender();
    updateScrollbar();
  }});

  /* ── Sort buttons ───────────────────────────────────────── */
  function syncSortButtons() {{
    document.querySelectorAll('#rmSort button').forEach(btn => {{
      const selected = btn.dataset.v === state.sort;
      btn.classList.toggle('active', selected);
      btn.setAttribute('aria-pressed', selected ? 'true' : 'false');
    }});
  }}
  syncSortButtons();

  document.getElementById('rmSort').addEventListener('click', e => {{
    const btn = e.target.closest('button');
    if (!btn) return;
    state.sort = btn.dataset.v || 'signed_p';
    publishBundleState();
    reSort();
    computeRunColors();
    invalidateCompactBitmap();
    syncSortButtons();
    pinnedIdx = -1;
    activeCol = -1;
    activeIdx = -1;
    clearOverlay();
    clearInfo();
    requestRender();
    updateScrollbar();
  }});

  /* ── Mode buttons ───────────────────────────────────────── */
  function syncModeButtons() {{
    document.body.classList.toggle('mode-compact', state.mode === 'compact');
    document.querySelectorAll('#rmMode button').forEach(btn => {{
      if (btn.dataset.v === 'compact') btn.disabled = !COMPACT_ENABLED;
      const selected = btn.dataset.v === state.mode;
      btn.classList.toggle('active', selected);
      btn.setAttribute('aria-pressed', selected ? 'true' : 'false');
    }});
  }}
  syncModeButtons();

  document.getElementById('rmMode').addEventListener('click', e => {{
    const btn = e.target.closest('button');
    if (!btn || btn.disabled) return;
    if (btn.dataset.v === 'compact' && !COMPACT_ENABLED) return;
    state.mode = btn.dataset.v || 'detail';
    syncGuideChips();
    publishBundleState();
    state.scrollX = 0;
    invalidateCompactBitmap();
    pinnedIdx = -1;
    activeCol = -1;
    activeIdx = -1;
    clearOverlay();
    clearInfo();
    syncModeButtons();
    resizeCanvas();
    requestRender();
    updateScrollbar();
  }});

  /* ── Control color mode ─────────────────────────────────── */
  function syncControlColorButtons() {{
    document.body.classList.toggle('control-stats', state.controlColor === 'stats');
    document.querySelectorAll('#rmControlColor button').forEach(btn => {{
      const selected = btn.dataset.v === state.controlColor;
      btn.classList.toggle('active', selected);
      btn.setAttribute('aria-pressed', selected ? 'true' : 'false');
    }});
  }}
  syncControlColorButtons();

  document.getElementById('rmControlColor').addEventListener('click', e => {{
    const btn = e.target.closest('button');
    if (!btn) return;
    state.controlColor = btn.dataset.v || 'run';
    publishBundleState();
    computeRunColors();
    invalidateCompactBitmap();
    syncControlColorButtons();
    resizeCanvas();
    updateScrollbar();
    clearOverlay();
    requestRender();
  }});

  /* ── Significance filter chips ──────────────────────────── */
  document.querySelectorAll('.rm-chip[data-sig]').forEach(chip => {{
    chip.addEventListener('click', () => {{
      const star = Number(chip.dataset.sig);
      if (state.sigFilter.has(star)) {{
        if (state.sigFilter.size <= 1) return;
        state.sigFilter.delete(star);
      }} else {{
        state.sigFilter.add(star);
      }}
      chip.classList.toggle('on', state.sigFilter.has(star));
      invalidateCompactBitmap();
      requestRender();
    }});
  }});

  /* ── CI band chips ──────────────────────────────────────── */
  document.querySelectorAll('.rm-chip[data-ci]').forEach(chip => {{
    chip.addEventListener('click', () => {{
      const ci = Number(chip.dataset.ci);
      const enabled = !chip.classList.contains('on');
      chip.classList.toggle('on', enabled);
      state.showCI[ci] = enabled;
      invalidateCompactBitmap();
      requestRender();
    }});
  }});

  /* ── Guide chips ────────────────────────────────────────── */
  function syncGuideChips() {{
    document.querySelectorAll('.rm-chip[data-special]').forEach(chip => {{
      const kind = chip.dataset.special;
      const enabled = state.mode === 'detail'
        || (kind === 'full' ? state.showFull : kind === 'nocontrol' ? state.showNotest : state.showBestTest);
      chip.classList.toggle('on', enabled);
    }});
  }}
  syncGuideChips();

  document.querySelectorAll('.rm-chip[data-special]').forEach(chip => {{
    chip.addEventListener('click', () => {{
      const kind = chip.dataset.special;
      if (jumpToGuide(kind)) {{
        syncGuideChips();
        return;
      }}
      const enabled = !chip.classList.contains('on');
      chip.classList.toggle('on', enabled);
      if (kind === 'full') state.showFull = enabled;
      else if (kind === 'nocontrol') state.showNotest = enabled;
      else state.showBestTest = enabled;
      invalidateCompactBitmap();
      requestRender();
    }});
  }});

  /* ── Initial render ─────────────────────────────────────── */
  render();
  updateScrollbar();
  clearInfo();
  document.body.classList.remove('rm-preinit');

  /* add ctrl-lbl active style */
  const style = document.createElement('style');
  style.textContent = '.ctrl-lbl.active-ctrl {{ color: #7C3AED !important; font-weight: 600; }}';
  document.head.appendChild(style);

}})();
</script>
</body>
</html>
"""


# ── CLI entry ─────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="regression_monkey_html",
        description=(
            "Render interactive Regression Monkey HTML "
            "from *_results.csv and *_plot_meta.json."
        ),
    )
    parser.add_argument("--results", required=True, metavar="CSV")
    parser.add_argument("--meta", required=True, metavar="JSON")
    parser.add_argument("--output", metavar="HTML")
    args = parser.parse_args()

    out = html_from_files(
        results_path=args.results,
        meta_path=args.meta,
        output_path=args.output,
    )
    print(f"✓  {out}")


if __name__ == "__main__":
    main()
