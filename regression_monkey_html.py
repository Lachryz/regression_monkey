# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "pandas",
#   "numpy",
# ]
# ///
"""
regression_monkey_html.py  ·  redesigned edition
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

import regression_monkey_plot as rm_plot
import regression_monkey_py as rm_py


# ── Colour palette ────────────────────────────────────────────────────────────
#   p < 0.01  →  crimson
#   p < 0.05  →  forest green
#   p < 0.10  →  steel blue
#   n.s.      →  warm gray

_SIG_COLOR = ["#111827", "#1D4ED8", "#15803D", "#B91C1C"]  # nsig, 10%, 5%, 1%
_SIG_BG = [
    "rgba(17,24,39,.12)",
    "rgba(29,78,216,.16)",
    "rgba(21,128,61,.16)",
    "rgba(185,28,28,.16)",
]
_SIG_LABEL = ["n.s.", "p<0.10", "p<0.05", "p<0.01"]

_OBS_FILL = "#9CA3AF"
_STAR_POS = "#FF2F92"  # positive coef star cells
_STAR_NEG = "#0433FF"  # negative coef star cells
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
    return {
        "index": index,
        "coef": float(record["coef"]),
        "se": float(record["se"]),
        "t_value": float(record["t_value"]),
        "p_value": float(record["p_value"]),
        "adj_r2": adj_r2,
        "ci99_lo": float(record["ci99_lo"]),
        "ci99_hi": float(record["ci99_hi"]),
        "ci95_lo": float(record["ci95_lo"]),
        "ci95_hi": float(record["ci95_hi"]),
        "ci90_lo": float(record["ci90_lo"]),
        "ci90_hi": float(record["ci90_hi"]),
        "obs": int(record["obs"]),
        "is_full": bool(record["is_full"]),
        "is_no_controls_test": not bool(record["controls_test"]),
        "star": _star_level(float(record["p_value"])),
        "color": _point_color(float(record["p_value"])),
        "controls_all": sorted(record["controls_all"]),
        "included_matrix_controls": included,
        "control_stats": list(record.get("control_stats", [])),
    }


def _display_subtitle(value: Any) -> str:
    text = str(value or "")
    return text.split(" - ", 1)[0]


def _controls_must_line(values: Any, max_width: int = 100) -> str:
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


def _controls_test_line_html(values: Any, alt_groups: Any = None, max_width: int = 100) -> str:
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
    order: str | None = None,
    sort_by_signed_p: bool | None = None,
) -> pathlib.Path:
    """Render one interactive specification-curve HTML from standard handoff files."""
    payload = payload_from_files(
        results_path=results_path,
        meta_path=meta_path,
        order=order,
        sort_by_signed_p=sort_by_signed_p,
    )
    default_out = pathlib.Path(
        str(payload.get("outputPath", pathlib.Path(results_path).with_suffix(".html")))
    ).with_suffix(".html")
    out = pathlib.Path(output_path) if output_path is not None else default_out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(_build_html(payload), encoding="utf-8")
    return out


def payload_from_files(
    *,
    results_path: str | pathlib.Path,
    meta_path: str | pathlib.Path,
    order: str | None = None,
    sort_by_signed_p: bool | None = None,
) -> dict[str, Any]:
    """Build the single-chart HTML payload from standard handoff files."""
    render_t0 = perf_counter()
    results_file = pathlib.Path(results_path)
    meta_file = pathlib.Path(meta_path)
    meta = rm_plot.load_plot_meta(meta_file)
    records = rm_py.records_from_dataframe(pd.read_csv(results_file))

    p_alias = (
        bool(sort_by_signed_p)
        if sort_by_signed_p is not None
        else (
            bool(meta.get("sort_by_p_mode", meta.get("sort_by_signed_p", False)))
            if order is None
            else False
        )
    )
    plot_order = rm_py._normalize_plot_order(
        str(meta.get("order", "coef") if order is None else order),
        p_alias=p_alias,
    )
    records = rm_py._sort_records_for_plot(
        records,
        sort_by_signed_p=rm_py._order_uses_p_mode(plot_order),
    )

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
        "specName": meta.get("spec_name", ""),
        "outputPath": str(meta.get("output_path", results_file.with_suffix(".html"))),
        "matrixControls": matrix_controls,
        "matrixAltGroups": list(meta.get("matrix_alt_groups", [])),
        "showSpecialMarkers": bool(meta.get("show_special_markers", True)),
        "elapsedSeconds": meta.get("elapsed_seconds_preplot"),
        "order": plot_order,
        "sort_by_p_mode": rm_py._order_uses_p_mode(plot_order),
        "sort_by_signed_p": rm_py._order_uses_p_mode(plot_order),
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
            "srcdoc": _build_html(payload),
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
    .bundle-field label {{
      color: var(--muted);
      font-size: 9.5px;
      font-weight: 700;
      letter-spacing: .08em;
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
<body>
  <div class="bundle-toolbar">
    <div class="bundle-brand">Regression Monkey</div>
    <div class="bundle-field">
      <label for="bundleY">Y</label>
      <select id="bundleY"></select>
    </div>
    <div class="bundle-field">
      <label for="bundleX">X</label>
      <select id="bundleX"></select>
    </div>
    <div class="bundle-field">
      <label for="bundleSpec">SPEC</label>
      <select id="bundleSpec"></select>
    </div>
    <div class="bundle-spacer"></div>
    <div id="bundleCount" class="bundle-count"></div>
  </div>
  <iframe id="rmFrame" title="Regression Monkey chart"></iframe>
  <script>
    const VIEWS = {data_json};
    const ySel = document.getElementById("bundleY");
    const xSel = document.getElementById("bundleX");
    const specSel = document.getElementById("bundleSpec");
    const frame = document.getElementById("rmFrame");
    const count = document.getElementById("bundleCount");

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

    function setSpecOptions(matches) {{
      const old = specSel.value;
      specSel.replaceChildren(...matches.map(v => {{
        const opt = option(v.id);
        opt.textContent = v.spec || "Spec";
        return opt;
      }}));
      if (matches.some(v => v.id === old)) specSel.value = old;
      else if (matches.length) specSel.value = matches[0].id;
    }}

    function filtered() {{
      return VIEWS.filter(v => String(v.y) === ySel.value && String(v.x) === xSel.value);
    }}

    function renderSelectors(changed) {{
      if (!VIEWS.length) return;
      if (changed !== "y") setOptions(ySel, uniq(VIEWS.map(v => v.y)));
      const xs = uniq(VIEWS.filter(v => String(v.y) === ySel.value).map(v => v.x));
      if (changed !== "x") setOptions(xSel, xs);
      setSpecOptions(filtered());
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


def _build_html(payload: dict[str, Any]) -> str:
    if (
        bool(payload.get("sort_by_p_mode", payload.get("sort_by_signed_p", False)))
        or str(payload.get("order", "")) == "p"
    ):
        payload = {
            **payload,
            "records": rm_py._sort_records_by_p_mode(list(payload["records"])),
        }
    title = html.escape(str(payload["title"]))
    y_title = html.escape(str(payload.get("y", "")))
    x_title = html.escape(str(payload.get("x", "")))
    subtitle = html.escape(str(payload.get("subtitle") or ""))
    controls_must_line = html.escape(_payload_controls_must_line(payload))
    controls_test_line = _payload_controls_test_line_html(payload)
    elapsed = payload.get("elapsedSeconds")
    elapsed_item = (
        f'<span class="leg-meta">Elapsed = {float(elapsed):.2f}s</span><span class="leg-sep"></span><span class="leg-meta">@Lachryz</span>'
        if elapsed is not None
        else ""
    )
    n = len(payload["records"])
    n1 = sum(1 for r in payload["records"] if r["star"] == 3)
    n5 = sum(1 for r in payload["records"] if r["star"] == 2)
    n10 = sum(1 for r in payload["records"] if r["star"] == 1)

    data_json = _json_for_html(payload)
    svg_markup, width, height = _build_svg(payload)

    sig_colors_js = json.dumps(_SIG_COLOR)
    sig_bg_js = json.dumps(_SIG_BG)
    sig_labels_js = json.dumps(_SIG_LABEL)
    embedded_font_css = _embedded_courier_new_css()

    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{title}</title>
  <style>
    {embedded_font_css}
    /* ── Design tokens ──────────────────────────────────── */
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

      --sig1:       #B91C1C;
      --sig5:       #15803D;
      --sig10:      #1D4ED8;
      --nsig:       #111827;

      --mono: "RM Courier New", "Courier New", monospace;
      --sans: "RM Courier New", "Courier New", monospace;

      --r-sm:  4px;
      --r-md:  8px;
      --shadow-sm: 0 1px 3px rgba(0,0,0,.08), 0 1px 2px rgba(0,0,0,.06);
      --shadow-tt: 0 20px 40px rgba(0,0,0,.22), 0 6px 16px rgba(0,0,0,.14);
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

    /* ── Header ─────────────────────────────────────────── */
    header {{
      position: sticky;
      top: 0;
      z-index: 100;
      padding: 11px 22px 9px;
      border-bottom: 1px solid var(--line);
      background: rgba(255,255,255,.95);
      backdrop-filter: saturate(180%) blur(14px);
      -webkit-backdrop-filter: saturate(180%) blur(14px);
    }}

    .h-row {{
      display: flex;
      align-items: center;
      gap: 10px;
      flex-wrap: wrap;
    }}

    h1 {{
      font-family: var(--mono);
      font-size: 13.5px;
      font-weight: 600;
      letter-spacing: -0.01em;
      color: var(--ink);
      line-height: 1.35;
    }}

    .badge {{
      display: inline-flex;
      align-items: center;
      gap: 3px;
      padding: 2px 7px;
      border-radius: 99px;
      font-size: 10.5px;
      font-weight: 500;
      letter-spacing: .01em;
      border: 1px solid var(--line);
      color: var(--muted);
      background: var(--line-2);
    }}

    .leg-meta {{
      font-family: var(--mono);
      font-size: 10.5px;
      font-weight: 600;
      color: var(--ink-2);
    }}

    .kbd-row {{
      margin-left: auto;
      display: flex;
      align-items: center;
      gap: 5px;
      font-size: 10px;
      color: var(--muted-2);
    }}

    kbd {{
      display: inline-flex;
      align-items: center;
      justify-content: center;
      min-width: 18px;
      height: 18px;
      padding: 0 4px;
      border: 1px solid var(--line);
      border-radius: 3px;
      background: var(--line-2);
      font-family: var(--sans);
      font-size: 9.5px;
      color: var(--ink-2);
      box-shadow: 0 1px 0 var(--line);
    }}

    .subtitle {{
      margin-top: 2px;
      font-family: var(--mono);
      font-size: 10.5px;
      color: var(--muted);
      white-space: pre-wrap;
      word-break: break-word;
    }}

    .ctrl-group-title {{
      font-weight: 700;
    }}

    /* ── Legend ─────────────────────────────────────────── */
    .legend {{
      display: flex;
      align-items: center;
      gap: 10px;
      margin-top: 8px;
      flex-wrap: wrap;
    }}

    .leg-grp {{
      display: flex;
      align-items: center;
      gap: 10px;
    }}

    .leg-item {{
      display: flex;
      align-items: center;
      gap: 5px;
      font-size: 11px;
      color: var(--ink-2);
    }}

    .leg-dot {{
      width: 7px;
      height: 7px;
      border-radius: 50%;
      flex-shrink: 0;
    }}

    .leg-count {{
      font-family: var(--mono);
      font-size: 10px;
      color: var(--muted);
    }}

    .leg-sep {{
      width: 1px;
      height: 13px;
      background: var(--line);
    }}

    .leg-ci {{
      display: flex;
      align-items: center;
      gap: 6px;
      font-size: 11px;
      color: var(--muted);
    }}

    .leg-ci-band {{
      position: relative;
      width: 30px;
      height: 14px;
    }}

    .leg-ci-band span {{
      position: absolute;
      left: 0;
      right: 0;
      border-radius: 2px;
    }}

    .meta-row {{
      margin-top: 4px;
      display: flex;
      align-items: center;
      gap: 8px;
      flex-wrap: wrap;
    }}

    .rm-tools {{
      margin-top: 10px;
      padding-top: 10px;
      border-top: 1px solid var(--line);
      display: flex;
      align-items: center;
      gap: 10px;
      flex-wrap: wrap;
      font-family: var(--mono);
      font-size: 10.5px;
      color: var(--muted);
    }}

    .rm-lbl {{
      color: var(--muted-2);
      text-transform: uppercase;
      letter-spacing: .08em;
      font-weight: 600;
      font-size: 9.5px;
    }}

    .rm-seg {{
      display: inline-flex;
      background: var(--line-2);
      border-radius: 5px;
      padding: 2px;
      gap: 2px;
    }}

    .rm-seg button {{
      background: transparent;
      border: 0;
      font-family: var(--mono);
      font-size: 10.5px;
      padding: 3px 8px;
      border-radius: 4px;
      color: var(--muted);
      cursor: pointer;
      font-weight: 500;
    }}

    .rm-seg button.active {{
      background: #FFFFFF;
      color: var(--ink);
      box-shadow: 0 1px 2px rgba(0,0,0,.06);
    }}

    .rm-chip {{
      display: inline-flex;
      align-items: center;
      gap: 5px;
      padding: 2px 8px;
      border: 1px solid var(--line);
      border-radius: 99px;
      color: var(--muted-2);
      background: #FFFFFF;
      cursor: pointer;
      user-select: none;
      font-size: 10px;
    }}

    .rm-chip i {{
      width: 7px;
      height: 7px;
      border-radius: 50%;
      display: inline-block;
    }}

    .rm-chip.on {{
      border-color: var(--ink);
      color: var(--ink);
      background: var(--bg);
    }}

    .rm-divider {{
      width: 1px;
      height: 14px;
      background: var(--line);
    }}

    #chart g[data-spec] {{
      transition: transform .35s cubic-bezier(.22,.7,.25,1);
    }}

    #chart g[data-spec].rm-dim > * {{
      opacity: .13;
      transition: opacity .2s;
    }}

    /* ── Layout ─────────────────────────────────────────── */
    .main-body {{
      flex: 1;
      display: flex;
      min-height: 0;
      overflow: hidden;
    }}

    .chart-col {{
      flex: 1;
      min-width: 0;
      overflow-x: auto;
      overflow-y: auto;
    }}

    /* ── Canvas ─────────────────────────────────────────── */
    .wrap {{
      padding: 18px 22px 36px;
    }}

    svg {{
      width: auto;
      height: auto;
      display: block;
    }}

    /* ── SVG element styles ─────────────────────────────── */
    .axis-label {{
      font-size: 9px;
      font-family: var(--sans);
      fill: #9CA3AF;
      font-weight: 600;
      letter-spacing: .08em;
    }}

    .sticky-x {{
      pointer-events: none;
    }}

    .sticky-label {{
      paint-order: stroke;
      stroke: #FFFFFF;
      stroke-width: 4px;
      stroke-linejoin: round;
    }}

    .tick-label {{
      font-size: 9.5px;
      font-family: var(--mono);
      fill: #9CA3AF;
    }}

    .control-label {{
      font-size: 10px;
      font-family: var(--mono);
      fill: var(--control-label-fill, #4B5563);
      transition: fill .15s ease;
    }}

    .control-label.active {{
      fill: var(--active);
      font-weight: 600;
    }}

    .control-label-highlight {{
      fill: none;
      stroke: #F59E0B;
      stroke-width: 1;
    }}

    .grid-major {{
      stroke: #E5E7EB;
      stroke-width: .75;
    }}

    .grid-minor {{
      stroke: #F3F4F6;
      stroke-width: .6;
    }}

    .zero-line {{
      stroke: #EF4444;
      stroke-width: .8;
      stroke-dasharray: 5 4;
      opacity: .55;
    }}

    .star-zero-line {{
      stroke: #111827;
      stroke-width: .8;
      opacity: .78;
    }}

    .star-cell.active {{
      fill: var(--active);
    }}

    .star-zero-segment.active {{
      stroke: var(--active);
      opacity: 1;
    }}

    /* CI bands — layered opacities create natural gradient */
    .ci99 {{ fill: #9CA3AF; opacity: .16; }}
    .ci95 {{ fill: #6B7280; opacity: .22; }}
    .ci90 {{ fill: #374151; opacity: .28; }}

    .point {{
      cursor: crosshair;
      stroke: none;
    }}

    .active-ring {{
      pointer-events: none;
      fill: none;
      stroke: var(--active);
      stroke-width: 2;
      opacity: 0;
      transition: opacity .12s ease;
    }}

    .active-ring.visible {{
      opacity: 1;
    }}

    .special-line {{
      pointer-events: none;
      opacity: .86;
      stroke-width: 1.15;
      transition: opacity .12s ease;
    }}

    .special-line.hidden {{
      opacity: 0;
    }}

    .special-full {{
      stroke: #FF2F92;
    }}

    .special-nocontrol {{
      stroke: #ff8c00;
    }}

    .guide {{
      pointer-events: none;
      opacity: 0;
      stroke: var(--active);
      stroke-width: .9;
      stroke-dasharray: 3 3;
      transition: opacity .12s ease;
    }}

    .guide.active {{ opacity: .4; }}

    .matrix-cell {{
      opacity: 1;
    }}

    .matrix-cell.group-control-cell {{
      fill: var(--group-fill);
    }}

    .matrix-cell.full-control-cell {{
      fill: #FF2F92;
    }}

    .matrix-cell.active {{
      fill: var(--active);
      opacity: 1;
    }}

    .obs-bar {{
      fill: var(--obs-fill, #9CA3AF);
      opacity: .78;
    }}

    .obs-bar.active {{
      fill: var(--active);
      opacity: 1;
    }}

    .hoverable {{ cursor: crosshair; }}

    /* ── Info panel ─────────────────────────────────────── */
    .info-panel {{
      width: 300px;
      flex-shrink: 0;
      border-left: 1px solid var(--line);
      background: var(--bg);
      overflow-y: auto;
    }}

    .panel-placeholder {{
      padding: 32px 16px;
      color: var(--muted-2);
      font-size: 11px;
      text-align: center;
      line-height: 1.7;
    }}

    .panel-head {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 10px 14px 8px;
      border-bottom: 1px solid var(--line);
    }}

    .panel-title {{
      font-family: var(--mono);
      font-size: 11px;
      font-weight: 600;
      color: var(--ink);
    }}

    .panel-sig {{
      display: inline-flex;
      align-items: center;
      padding: 1px 7px;
      border-radius: 99px;
      font-size: 10px;
      font-weight: 600;
      letter-spacing: .01em;
    }}

    .panel-table {{
      display: grid;
      grid-template-columns: auto 1fr;
      gap: 2px 10px;
      padding: 10px 14px;
    }}

    .panel-key {{
      color: var(--muted);
      font-size: 10.5px;
      align-self: center;
    }}

    .panel-val {{
      font-family: var(--mono);
      font-size: 10.5px;
      color: var(--ink-2);
      font-weight: 500;
      text-align: right;
    }}

    .panel-divider {{
      grid-column: 1 / -1;
      height: 1px;
      border-top: 1px dotted var(--line);
      margin: 4px 0;
    }}

    .panel-controls {{
      grid-column: 1 / -1;
      margin-top: 2px;
      color: var(--muted);
      font-size: 10px;
      line-height: 1.45;
    }}

    .panel-controls em {{
      font-style: normal;
      color: var(--ink-2);
    }}

    /* ── Per-control coefficient list ─────────────────── */
    .panel-coefs {{
      grid-column: 1 / -1;
      display: flex;
      flex-direction: column;
      margin-top: 2px;
    }}

    .panel-coefs-head {{
      display: flex;
      justify-content: space-between;
      align-items: baseline;
      font-family: var(--mono);
      font-size: 9px;
      letter-spacing: .1em;
      text-transform: uppercase;
      color: var(--muted-2);
      font-weight: 600;
      padding-bottom: 6px;
      margin-bottom: 4px;
    }}

    .panel-coefs-meta {{
      font-weight: 500;
      opacity: .85;
      letter-spacing: 0;
      text-transform: none;
    }}

    .coef-group-label {{
      font-family: var(--mono);
      font-size: 8.5px;
      color: var(--muted-2);
      text-transform: uppercase;
      letter-spacing: .12em;
      padding: 8px 14px 4px;
      font-weight: 600;
      border-top: 1px solid var(--line);
      margin: 6px -14px 0;
    }}

    .coef-group-label .grp-count {{
      color: var(--muted);
      font-weight: 500;
      letter-spacing: 0;
    }}

    .coef-row {{
      display: grid;
      grid-template-columns: minmax(0, 1fr) 68px 74px;
      gap: 12px;
      align-items: baseline;
      padding: 4px;
      border-radius: 3px;
      font-family: var(--mono);
      font-size: 10.5px;
      line-height: 1.25;
      transition: background .12s;
    }}

    .coef-row + .coef-row {{
      border-top: 1px dotted var(--line);
    }}

    .coef-row:hover {{
      background: var(--bg-2);
    }}

    .coef-row.is-test .coef-name {{
      font-weight: 600;
    }}

    .coef-name {{
      color: var(--ink);
      font-weight: 500;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
      letter-spacing: 0;
    }}

    .coef-val {{
      color: var(--ink);
      font-variant-numeric: tabular-nums;
      text-align: right;
      font-size: 10.5px;
      letter-spacing: 0;
      min-width: 0;
    }}

    .coef-val.placeholder {{
      color: var(--muted-2);
    }}

    .coef-val.pos {{
      color: #B91C1C;
    }}

    .coef-val.neg {{
      color: #1D4ED8;
    }}

    .coef-p {{
      display: inline-flex;
      align-items: center;
      gap: 4px;
      font-size: 9.5px;
      color: var(--muted);
      font-variant-numeric: tabular-nums;
      justify-content: flex-end;
      min-width: 0;
      white-space: nowrap;
    }}

    .coef-p.placeholder {{
      color: var(--muted-2);
    }}

    .coef-p .coef-stars {{
      font-weight: 700;
      letter-spacing: 0;
      font-size: 11px;
      color: var(--ink-2);
    }}

    .coef-empty {{
      font-family: var(--mono);
      font-size: 10px;
      color: var(--muted-2);
      padding: 6px 4px;
      font-style: italic;
    }}

    /* ── Print ──────────────────────────────────────────── */
    @media print {{
      header {{ position: static; border-bottom: 1px solid #ccc; }}
      .info-panel {{ display: none; }}
      .chart-col {{ flex: 1; }}
    }}
  </style>
</head>
<body>

<header>
  <div class="h-row">
    <h1>Y: {y_title}<br>X: {x_title}</h1>
    <div class="kbd-row">
      Navigate <kbd>←</kbd><kbd>→</kbd>&nbsp; Dismiss <kbd>Esc</kbd>
    </div>
  </div>
  {f'<div class="subtitle">{subtitle}</div>' if subtitle else ""}
  {f'<div class="subtitle">{controls_must_line}</div>' if controls_must_line else ""}
  {f'<div class="subtitle">{controls_test_line}</div>' if controls_test_line else ""}
  <div class="legend">
    <span class="badge" style="gap:8px;padding:3px 10px">
      {n} specs
      <span style="display:inline-block;width:1px;height:11px;background:var(--line);"></span>
      <span class="leg-meta" style="font-weight:500;color:var(--muted)">{f"Elapsed = {float(elapsed):.2f}s" if elapsed is not None else "Elapsed = n/a"}</span>
    </span>
  </div>
  <div class="meta-row">
    <span class="leg-meta">@Lachryz</span>
  </div>
  <div class="rm-tools">
    <span class="rm-lbl">Sort</span>
    <div class="rm-seg" id="rmSort">
      <button type="button" data-v="index" class="active">order</button>
      <button type="button" data-v="p">|p|</button>
      <button type="button" data-v="t">|t|</button>
      <button type="button" data-v="coef">coef</button>
      <button type="button" data-v="obs">obs</button>
    </div>
    <span class="rm-divider"></span>
    <span class="rm-lbl">Significance</span>
    <span class="rm-chip on" data-sig="3"><i style="background:#B91C1C"></i>p&lt;.01</span>
    <span class="rm-chip on" data-sig="2"><i style="background:#15803D"></i>p&lt;.05</span>
    <span class="rm-chip on" data-sig="1"><i style="background:#1D4ED8"></i>p&lt;.10</span>
    <span class="rm-chip on" data-sig="0"><i style="background:#9CA3AF"></i>n.s.</span>
    <span class="rm-divider"></span>
    <span class="rm-lbl">CI bands</span>
    <span class="rm-chip on" data-ci="90"><i style="background:#9CA3AF;opacity:.55"></i>90%</span>
    <span class="rm-chip on" data-ci="95"><i style="background:#6B7280;opacity:.7"></i>95%</span>
    <span class="rm-chip on" data-ci="99"><i style="background:#374151;opacity:.85"></i>99%</span>
  </div>
</header>

<div class="main-body">
  <div class="chart-col" id="chart-col">
    <div class="wrap" id="chart-wrap">
      {svg_markup}
    </div>
  </div>
  <div class="info-panel" id="info-panel">
    <div class="panel-placeholder" id="panel-placeholder">Hover or click<br>a specification</div>
    <div id="panel-content" style="display:none"></div>
  </div>
</div>

<script>
  const DATA       = {data_json};
  const SIG_COLOR  = {sig_colors_js};
  const SIG_BG     = {sig_bg_js};
  const SIG_LABEL  = {sig_labels_js};
  const records    = DATA.records;

  const panelPlaceholder = document.getElementById("panel-placeholder");
  const panelContent     = document.getElementById("panel-content");
  const chartCol = document.getElementById("chart-col");
  let activeIdx = null;
  let pinnedIdx = null;

  /* ── Activation ─────────────────────────────────────── */
  function activate(idx, pin = false) {{
    if (pinnedIdx !== null && !pin) return;
    if (idx === activeIdx) {{
      if (pin) pinnedIdx = idx;
      return;
    }}
    clearActive(false, true);
    activeIdx = idx;
    if (pin) pinnedIdx = idx;

    const r = records[idx];

    /* highlight SVG elements */
    document.querySelectorAll(`.special-line[data-special-index="${{idx}}"]`)
      .forEach(el => el.classList.add("hidden"));
    document.querySelectorAll(`[data-index="${{idx}}"]`)
      .forEach(el => el.classList.add("active"));
    const pt = document.querySelector(`.point[data-index="${{idx}}"]`);
    const ring = document.getElementById("active-ring");
    if (pt && ring) {{
      ring.setAttribute("cx", pt.getAttribute("cx"));
      ring.setAttribute("cy", pt.getAttribute("cy"));
      const specGroup = document.querySelector(`g[data-spec="${{idx}}"]`);
      ring.setAttribute("transform", specGroup ? (specGroup.getAttribute("transform") || "") : "");
      ring.classList.add("visible");
    }}

    r.included_matrix_controls.forEach(name =>
      document.querySelectorAll(`.control-label[data-control="${{name}}"]`)
        .forEach(el => el.classList.add("active"))
    );

    /* panel content */
    const star   = r.star;
    const ci99   = `[${{fmt(r.ci99_lo)}}, ${{fmt(r.ci99_hi)}}]`;
    const ci90   = `[${{fmt(r.ci90_lo)}}, ${{fmt(r.ci90_hi)}}]`;
    const ci95   = `[${{fmt(r.ci95_lo)}}, ${{fmt(r.ci95_hi)}}]`;
    const adjR2  = r.adj_r2 === null || r.adj_r2 === undefined ? "-" : Number(r.adj_r2).toFixed(4);
    const includedControls = new Set(r.controls_all || []);
    const testOrder = DATA.controlsTestNames || DATA.matrixControls || [];
    const mustOrder = DATA.controlsMustNames || [];
    const testIncl = testOrder.filter(c => includedControls.has(c));
    const mustIncl = mustOrder.filter(c => includedControls.has(c));
    const orderedKnown = new Set([...testIncl, ...mustIncl]);
    const extraIncl = (r.controls_all || []).filter(c => !orderedKnown.has(c));
    const controlStats = new Map((r.control_stats || []).map(item => [item.name, item]));
    const coefRow = (name, group) => `
      <div class="coef-row ${{group === "test" ? "is-test" : ""}}">
        <span class="coef-name" title="${{escapeHtml(name)}}">${{escapeHtml(name)}}</span>
        ${{controlStats.has(name)
          ? `<span class="coef-val ${{Number(controlStats.get(name).coef) < 0 ? "neg" : "pos"}}">${{fmt(Number(controlStats.get(name).coef))}}</span>
             <span class="coef-p s${{starLevel(Number(controlStats.get(name).p_value))}}"><span class="coef-stars">${{starsForP(Number(controlStats.get(name).p_value))}}</span><span>${{Number(controlStats.get(name).p_value).toFixed(4)}}</span></span>`
          : `<span class="coef-val placeholder">-</span><span class="coef-p placeholder"><span class="coef-stars">.</span><span>-</span></span>`}}
      </div>`;
    const coefBlock = r.controls_all.length === 0
      ? `<div class="coef-empty">No controls included in this specification.</div>`
      : `
        <div class="panel-coefs-head">
          <span>Control coefficients</span>
          <span class="panel-coefs-meta">${{testIncl.length}} test · ${{mustIncl.length + extraIncl.length}} must</span>
        </div>
        ${{testIncl.length ? `<div class="coef-group-label">TEST <span class="grp-count">(${{testIncl.length}})</span></div>${{testIncl.map(c => coefRow(c, "test")).join("")}}` : ""}}
        ${{mustIncl.length + extraIncl.length ? `<div class="coef-group-label">MUST <span class="grp-count">(${{mustIncl.length + extraIncl.length}})</span></div>${{[...mustIncl, ...extraIncl].map(c => coefRow(c, "base")).join("")}}` : ""}}
      `;

    panelContent.innerHTML = `
      <div class="panel-head">
        <span class="panel-title">Spec #${{idx + 1}}&thinsp;/&thinsp;${{records.length}}</span>
        <span class="panel-sig" style="background:${{SIG_BG[star]}};color:${{SIG_COLOR[star]}}">${{SIG_LABEL[star]}}</span>
      </div>
      <div class="panel-table">
        <span class="panel-key">coef</span>        <span class="panel-val">${{r.coef.toFixed(5)}}</span>
        <span class="panel-key">std&nbsp;err</span> <span class="panel-val">${{r.se.toFixed(5)}}</span>
        <span class="panel-key">t&#8209;stat</span> <span class="panel-val">${{r.t_value.toFixed(3)}}</span>
        <span class="panel-key">p&#8209;value</span><span class="panel-val">${{r.p_value.toFixed(4)}}</span>
        <div class="panel-divider"></div>
        <span class="panel-key">90% CI</span>      <span class="panel-val">${{ci90}}</span>
        <span class="panel-key">95% CI</span>      <span class="panel-val">${{ci95}}</span>
        <span class="panel-key">99% CI</span>      <span class="panel-val">${{ci99}}</span>
        <div class="panel-divider"></div>
        <span class="panel-key">obs</span>          <span class="panel-val">${{r.obs.toLocaleString()}}</span>
        <span class="panel-key">adj&nbsp;R²</span>  <span class="panel-val">${{adjR2}}</span>
        <div class="panel-divider"></div>
        <div class="panel-coefs">${{coefBlock}}</div>
      </div>`;

    panelPlaceholder.style.display = "none";
    panelContent.style.display     = "block";
  }}

  function clearActive(hide = true, force = false) {{
    if (pinnedIdx !== null && !force) return;
    document.querySelectorAll(".active").forEach(el => el.classList.remove("active"));
    document.querySelectorAll(".special-line.hidden")
      .forEach(el => el.classList.remove("hidden"));
    const ring = document.getElementById("active-ring");
    if (ring) {{
      ring.classList.remove("visible");
      ring.removeAttribute("transform");
    }}
    activeIdx = null;
    pinnedIdx = null;
    panelContent.style.display     = "none";
    panelPlaceholder.style.display = "";
  }}

  function togglePin(idx, event) {{
    event.preventDefault();
    event.stopPropagation();
    if (pinnedIdx === idx) {{
      clearActive(true, true);
      return;
    }}
    activate(idx, true);
  }}

  function fmt(v) {{ return v.toFixed(4); }}

  function starLevel(p) {{
    return p < 0.01 ? 3 : p < 0.05 ? 2 : p < 0.10 ? 1 : 0;
  }}

  function starsForP(p) {{
    const level = starLevel(p);
    return level === 0 ? "." : "*".repeat(level);
  }}

  function escapeHtml(value) {{
    return String(value)
      .replaceAll("&", "&amp;")
      .replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;")
      .replaceAll('"', "&quot;")
      .replaceAll("'", "&#39;");
  }}

  function syncStickyLabels() {{
    const x = chartCol ? chartCol.scrollLeft : 0;
    document.querySelectorAll(".sticky-x")
      .forEach(el => el.setAttribute("transform", `translate(${{x}},0)`));
  }}

  if (chartCol) {{
    chartCol.addEventListener("scroll", syncStickyLabels, {{ passive: true }});
    syncStickyLabels();
  }}

  /* ── Keyboard navigation ─────────────────────────────── */
  document.addEventListener("keydown", e => {{
    const n = records.length;
    if (!n) return;

    if (e.key === "ArrowRight" || e.key === "ArrowLeft") {{
      e.preventDefault();
      const cur  = activeIdx ?? (e.key === "ArrowRight" ? -1 : 0);
      const next = e.key === "ArrowRight"
        ? Math.min(cur + 1, n - 1)
        : Math.max(cur - 1, 0);
      pinnedIdx = null;
      activate(next);
    }}

    if (e.key === "Escape") clearActive(true, true);
  }});

  /* ── Header options ─────────────────────────────────── */
  (function initHeaderOptions() {{
    const svg = document.getElementById("chart");
    if (!svg || !records.length) return;

    const left = Number(svg.dataset.left || 0);
    const right = Number(svg.dataset.right || 0);
    const xStep = Number(svg.dataset.xStep || 7);
    const firstX = left + xStep * 0.5;
    const plotRight = Number(svg.dataset.plotRight || (Number(svg.getAttribute("width")) - right));
    const starY = Number(svg.dataset.starY || 20);
    const starBottom = Number(svg.dataset.starBottom || 86);
    const coefY = Number(svg.dataset.coefY || 98);
    const coefBottom = Number(svg.dataset.coefBottom || 394);
    const matrixY = Number(svg.dataset.matrixY || 406);
    const matrixBottom = Number(svg.dataset.matrixBottom || 622);
    const obsY = Number(svg.dataset.obsY || 646);
    const obsBottom = Number(svg.dataset.obsBottom || 734);

    const byIdx = {{}};
    svg.querySelectorAll("[data-index]").forEach(el => {{
      const idx = el.getAttribute("data-index");
      (byIdx[idx] = byIdx[idx] || []).push(el);
    }});

    Object.keys(byIdx).forEach(idx => {{
      const els = byIdx[idx];
      if (!els.length || els[0].closest("g[data-spec]")) return;
      const group = document.createElementNS("http://www.w3.org/2000/svg", "g");
      group.setAttribute("data-spec", idx);
      els[0].parentNode.insertBefore(group, els[0]);
      els.forEach(el => group.appendChild(el));
    }});

    const origCol = {{}};
    records.forEach((r, pos) => {{ origCol[r.index] = pos; }});

    const anchors = [];
    records.forEach(r => {{
      const pt = svg.querySelector(`.point[data-index="${{r.index}}"]`);
      if (pt) anchors.push({{ coef: Number(r.coef), y: Number(pt.getAttribute("cy")) }});
    }});
    let yCoef = v => coefY + (coefBottom - coefY) / 2;
    for (let i = 0; i < anchors.length; i++) {{
      for (let j = i + 1; j < anchors.length; j++) {{
        if (Math.abs(anchors[j].coef - anchors[i].coef) < 1e-12) continue;
        const slope = (anchors[j].y - anchors[i].y) / (anchors[j].coef - anchors[i].coef);
        const intercept = anchors[i].y - slope * anchors[i].coef;
        yCoef = v => slope * v + intercept;
        i = anchors.length;
        break;
      }}
    }}

    const state = {{
      sort: "index",
      sigFilter: new Set([0, 1, 2, 3]),
    }};

    function sortedRecords() {{
      const arr = records.slice();
      if (state.sort === "coef") arr.sort((a, b) => a.coef - b.coef);
      else if (state.sort === "t") arr.sort((a, b) => Math.abs(b.t_value) - Math.abs(a.t_value));
      else if (state.sort === "p") {{
        arr.sort((a, b) => {{
          const signA = a.coef < 0 ? -1 : 1;
          const signB = b.coef < 0 ? -1 : 1;
          const scoreA = signA * Math.log10(Math.max(a.p_value, Number.MIN_VALUE));
          const scoreB = signB * Math.log10(Math.max(b.p_value, Number.MIN_VALUE));
          return scoreA - scoreB;
        }});
      }} else if (state.sort === "obs") arr.sort((a, b) => b.obs - a.obs);
      else arr.sort((a, b) => a.index - b.index);
      return arr;
    }}

    function isDimmed(record) {{
      if (!state.sigFilter.has(record.star)) return true;
      return false;
    }}

    function ciPath(arr, loKey, hiKey, colW) {{
      if (!arr.length) return "";
      const upper = arr.map((r, pos) => {{
        const x = firstX + pos * colW;
        return `${{pos ? "L" : "M"}} ${{x.toFixed(3)}} ${{yCoef(r[hiKey]).toFixed(3)}}`;
      }});
      const lower = [];
      for (let pos = arr.length - 1; pos >= 0; pos--) {{
        const x = firstX + pos * colW;
        lower.push(`L ${{x.toFixed(3)}} ${{yCoef(arr[pos][loKey]).toFixed(3)}}`);
      }}
      return [...upper, ...lower, "Z"].join(" ");
    }}

    function updateSpecialLines(arr, colW) {{
      svg.querySelectorAll(".special-line").forEach(line => {{
        const idx = Number(line.getAttribute("data-special-index"));
        const pos = arr.findIndex(r => Number(r.index) === idx);
        if (pos < 0) return;
        const x = firstX + pos * colW;
        const r = records[idx];
        if (!r) return;
        const cy = yCoef(Number(r.coef));
        const gap = 4.8;
        const d = [
          `M ${{x.toFixed(3)}} ${{starY}} L ${{x.toFixed(3)}} ${{starBottom}}`,
          `M ${{x.toFixed(3)}} ${{coefY}} L ${{x.toFixed(3)}} ${{Math.max(coefY, cy - gap).toFixed(3)}}`,
          `M ${{x.toFixed(3)}} ${{Math.min(coefBottom, cy + gap).toFixed(3)}} L ${{x.toFixed(3)}} ${{coefBottom}}`,
          `M ${{x.toFixed(3)}} ${{matrixY}} L ${{x.toFixed(3)}} ${{matrixBottom}}`,
          `M ${{x.toFixed(3)}} ${{obsY}} L ${{x.toFixed(3)}} ${{obsBottom}}`,
        ].join(" ");
        line.setAttribute("d", d);
      }});
    }}

    function renderHeaderOptions() {{
      const arr = sortedRecords();
      const colW = xStep;

      arr.forEach((record, newPos) => {{
        const group = svg.querySelector(`g[data-spec="${{record.index}}"]`);
        if (!group) return;
        const oldX = firstX + (origCol[record.index] || 0) * xStep;
        const newX = firstX + newPos * colW;
        const scaleX = colW / xStep;
        const dx = newX - oldX * scaleX;
        group.setAttribute("transform", `translate(${{dx.toFixed(3)}},0) scale(${{scaleX.toFixed(5)}},1)`);
        group.classList.toggle("rm-dim", isDimmed(record));
      }});

      [["ci99", "ci99_lo", "ci99_hi"], ["ci95", "ci95_lo", "ci95_hi"], ["ci90", "ci90_lo", "ci90_hi"]]
        .forEach(([cls, loKey, hiKey]) => {{
          const path = svg.querySelector(`path.${{cls}}`);
          if (path) path.setAttribute("d", ciPath(arr, loKey, hiKey, colW));
        }});

      updateSpecialLines(arr, colW);
      if (activeIdx !== null) {{
        const group = svg.querySelector(`g[data-spec="${{activeIdx}}"]`);
        const ring = document.getElementById("active-ring");
        if (ring) ring.setAttribute("transform", group ? (group.getAttribute("transform") || "") : "");
      }}
      syncStickyLabels();
    }}

    const sortEl = document.getElementById("rmSort");
    if (sortEl) {{
      sortEl.addEventListener("click", event => {{
        const button = event.target.closest("button");
        if (!button) return;
        sortEl.querySelectorAll("button").forEach(item => item.classList.toggle("active", item === button));
        state.sort = button.dataset.v || "index";
        renderHeaderOptions();
      }});
    }}

    document.querySelectorAll(".rm-chip[data-sig]").forEach(chip => {{
      chip.addEventListener("click", () => {{
        const star = Number(chip.dataset.sig);
        if (state.sigFilter.has(star)) {{
          if (state.sigFilter.size <= 1) return;
          state.sigFilter.delete(star);
        }} else {{
          state.sigFilter.add(star);
        }}
        chip.classList.toggle("on", state.sigFilter.has(star));
        renderHeaderOptions();
      }});
    }});

    document.querySelectorAll(".rm-chip[data-ci]").forEach(chip => {{
      chip.addEventListener("click", () => {{
        const ci = chip.dataset.ci;
        const enabled = !chip.classList.contains("on");
        chip.classList.toggle("on", enabled);
        const path = svg.querySelector(`path.ci${{ci}}`);
        if (path) path.style.display = enabled ? "" : "none";
      }});
    }});

    renderHeaderOptions();
  }})();
</script>
</body>
</html>
"""


# ── SVG builder ───────────────────────────────────────────────────────────────


def _fmt(v: float) -> str:
    s = f"{v:.3f}".rstrip("0").rstrip(".")
    return s or "0"


def _svg_attrs(attrs: dict[str, Any]) -> str:
    return " ".join(
        f'{k}="{html.escape(str(v), quote=True)}"'
        for k, v in attrs.items()
        if v is not None
    )


def _tag(name: str, attrs: dict[str, Any], text: str | None = None) -> str:
    a = _svg_attrs(attrs)
    if text is None:
        return f"<{name} {a}/>"
    return f"<{name} {a}>{html.escape(text)}</{name}>"


def _scale(value: float, lo: float, hi: float, top: float, height: float) -> float:
    if hi == lo:
        return top + height / 2
    return top + height - ((value - lo) / (hi - lo)) * height


def _ci_path(
    records: list[dict],
    x_fn: Any,
    y_fn: Any,
    lo: str,
    hi: str,
    *,
    x_left: float | None = None,
    x_right: float | None = None,
) -> str:
    if not records:
        return ""
    upper: list[str] = []
    if x_left is not None:
        upper.append(f"M {_fmt(x_left)} {_fmt(y_fn(float(records[0][hi])))}")
    for i, r in enumerate(records):
        command = "M" if i == 0 and not upper else "L"
        upper.append(f"{command} {_fmt(x_fn(i))} {_fmt(y_fn(float(r[hi])))}")
    if x_right is not None:
        upper.append(f"L {_fmt(x_right)} {_fmt(y_fn(float(records[-1][hi])))}")

    lower: list[str] = []
    if x_right is not None:
        lower.append(f"L {_fmt(x_right)} {_fmt(y_fn(float(records[-1][lo])))}")
    lower.extend(
        f"L {_fmt(x_fn(i))} {_fmt(y_fn(float(r[lo])))}"
        for i, r in reversed(list(enumerate(records)))
    )
    if x_left is not None:
        lower.append(f"L {_fmt(x_left)} {_fmt(y_fn(float(records[0][lo])))}")
    return " ".join(upper + lower + ["Z"])


def _build_svg(payload: dict[str, Any]) -> tuple[str, int, int]:
    records = list(payload["records"])
    controls = list(payload["matrixControls"])
    alt_groups = list(payload.get("matrixAltGroups", []))
    controls_test_names = set(_payload_controls_test_names(payload))
    n = len(records)
    three_star_records = [r for r in records if int(r.get("star", 0)) == 3]
    starred_controls = {
        name
        for name in controls
        if name in controls_test_names
        and three_star_records
        and all(name in set(r.get("included_matrix_controls", [])) for r in three_star_records)
    }
    alt_group_color_by_control: dict[str, str] = {}
    color_idx = 0
    for grp in alt_groups:
        if str(grp.get("kind", "")) != "controls_test":
            continue
        s = int(grp.get("start", -1))
        e = int(grp.get("end", -1))
        if s < 0 or e <= s or e >= len(controls):
            continue
        fill = _ALT_GROUP_COLORS[color_idx % len(_ALT_GROUP_COLORS)]
        color_idx += 1
        for row in range(s, e + 1):
            alt_group_color_by_control[str(controls[row])] = fill

    # ── Geometry ──────────────────────────────────────────
    label_chars = max([len(str(c)) for c in controls] + [8])
    grp_pad = 46 if alt_groups else 0
    left = max(160, min(440, 76 + label_chars * 7 + grp_pad))
    right = 82
    top = 20

    star_h = 66
    coef_h = 296
    row_h = 18
    matrix_pad = 0
    matrix_h = max(row_h, len(controls) * row_h)
    obs_h = 88
    gap = 12

    x_step = 7 if n <= 1600 else 5
    plot_w_min = max(420, n * x_step)
    width = int(left + right + plot_w_min)
    plot_w = width - left - right

    star_y = top
    coef_y = star_y + star_h + gap
    matrix_y = coef_y + coef_h + gap
    obs_y = matrix_y + matrix_h + gap + 12

    height = int(obs_y + obs_h + 22)

    def xc(i: int) -> float:
        return left + x_step * (i + 0.5)

    # ── Value ranges ──────────────────────────────────────
    if records:
        clo = min(float(r["ci99_lo"]) for r in records)
        chi = max(float(r["ci99_hi"]) for r in records)
        if clo == chi:
            clo -= 1
            chi += 1
        else:
            pad = (chi - clo) * 0.09
            clo -= pad
            chi += pad
        obs_values = [int(r["obs"]) for r in records]
        obs_min = min(obs_values)
        obs_max = max(obs_values)
        obs_mean = sum(obs_values) / len(obs_values)
    else:
        clo, chi, obs_min, obs_max, obs_mean = -1.0, 1.0, 0, 1, 0.0

    def cy(v: float) -> float:
        return _scale(v, clo, chi, coef_y, coef_h)

    def oy(v: float) -> float:
        lo = min(float(obs_min), float(obs_mean), float(obs_max))
        hi = max(float(obs_min), float(obs_mean), float(obs_max))
        if lo == hi:
            lo -= 1
            hi += 1
        else:
            pad = (hi - lo) * 0.06
            lo -= pad
            hi += pad
        return _scale(v, lo, hi, obs_y, obs_h)

    def sy(v: float) -> float:
        return star_y + star_h / 2 - (v / 3) * (star_h / 2 - 9)

    # ── Build ─────────────────────────────────────────────
    p: list[str] = []

    p.append(
        f'<svg id="chart" role="img"'
        f' aria-label="{html.escape(str(payload["title"]), quote=True)}"'
        f' width="{width}" height="{height}"'
        f' viewBox="0 0 {width} {height}"'
        f' data-left="{_fmt(left)}" data-right="{_fmt(right)}"'
        f' data-x-step="{_fmt(x_step)}" data-plot-right="{_fmt(width - right)}"'
        f' data-star-y="{_fmt(star_y)}" data-star-bottom="{_fmt(star_y + star_h)}"'
        f' data-coef-y="{_fmt(coef_y)}" data-coef-bottom="{_fmt(coef_y + coef_h)}"'
        f' data-matrix-y="{_fmt(matrix_y)}" data-matrix-bottom="{_fmt(matrix_y + matrix_h)}"'
        f' data-obs-y="{_fmt(obs_y)}" data-obs-bottom="{_fmt(obs_y + obs_h)}"'
        f' xmlns="http://www.w3.org/2000/svg">'
    )

    # ── Panel rects ───────────────────────────────────────
    panels = [
        (star_y, star_h, "STARS"),
        (coef_y, coef_h, "COEF"),
        (matrix_y, matrix_h, "CONTROLS"),
        (obs_y, obs_h, "OBS"),
    ]
    for py_, ph, label in panels:
        # subtle background
        p.append(
            _tag(
                "rect",
                {
                    "x": left,
                    "y": _fmt(py_),
                    "width": plot_w,
                    "height": _fmt(ph),
                    "fill": "#FFFFFF",
                    "stroke": "#D1D5DB",
                    "stroke-width": "0.75",
                    "rx": "2",
                },
            )
        )
        # panel label on the right side of the frame
        label_x = width - right + 18
        mid = py_ + ph / 2
        p.append(
            f'<text x="{_fmt(label_x)}" y="{_fmt(mid)}"'
            f' class="axis-label" text-anchor="middle"'
            f' dominant-baseline="middle" transform="rotate(90 {_fmt(label_x)} {_fmt(mid)})"'
            f">{html.escape(label)}</text>"
        )

    # ── Coef grid lines (5 levels) ────────────────────────
    for j in range(5):
        frac = j / 4
        gy = coef_y + coef_h * frac
        gv = chi - (chi - clo) * frac
        cls = "grid-major" if j in (0, 2, 4) else "grid-minor"
        p.append(
            _tag(
                "line",
                {
                    "x1": left,
                    "x2": width - right,
                    "y1": _fmt(gy),
                    "y2": _fmt(gy),
                    "class": cls,
                },
            )
        )
        p.append(
            _tag(
                "text",
                {
                    "x": left - 7,
                    "y": _fmt(gy + 3.5),
                    "text-anchor": "end",
                    "class": "tick-label sticky-x sticky-label",
                },
                f"{gv:.3f}",
            )
        )

    # ── Zero lines ────────────────────────────────────────
    p.append(
        _tag(
            "line",
            {
                "x1": left,
                "x2": width - right,
                "y1": _fmt(cy(0)),
                "y2": _fmt(cy(0)),
                "class": "zero-line",
            },
        )
    )

    # ── Obs guide line and labels ─────────────────────────
    if records:
        obs_ticks = [
            ("max", float(obs_max)),
            ("mean", float(obs_mean)),
            ("min", float(obs_min)),
        ]
        for label, value in obs_ticks:
            y_tick = oy(value)
            p.append(
                _tag(
                    "line",
                    {
                        "x1": left,
                        "x2": width - right,
                        "y1": _fmt(y_tick),
                        "y2": _fmt(y_tick),
                        "stroke": "#F3F4F6" if label != "mean" else "#9CA3AF",
                        "stroke-width": "0.7",
                    },
                )
            )
            p.append(
                _tag(
                    "text",
                    {
                        "x": left - 7,
                        "y": _fmt(y_tick + 3.5),
                        "text-anchor": "end",
                        "class": "tick-label sticky-x sticky-label",
                    },
                    f"{value:,.0f}",
                )
            )
    p.append(
        _tag(
            "line",
            {
                "x1": left,
                "x2": width - right,
                "y1": _fmt(sy(0)),
                "y2": _fmt(sy(0)),
                "class": "star-zero-line",
            },
        )
    )

    # ── CI bands ─────────────────────────────────────────
    if records:
        p.append(
            _tag(
                "path",
                {
                    "d": _ci_path(
                        records,
                        xc,
                        cy,
                        "ci99_lo",
                        "ci99_hi",
                        x_left=left,
                        x_right=width - right,
                    ),
                    "class": "ci99",
                },
            )
        )
        p.append(
            _tag(
                "path",
                {
                    "d": _ci_path(
                        records,
                        xc,
                        cy,
                        "ci95_lo",
                        "ci95_hi",
                        x_left=left,
                        x_right=width - right,
                    ),
                    "class": "ci95",
                },
            )
        )
        p.append(
            _tag(
                "path",
                {
                    "d": _ci_path(
                        records,
                        xc,
                        cy,
                        "ci90_lo",
                        "ci90_hi",
                        x_left=left,
                        x_right=width - right,
                    ),
                    "class": "ci90",
                },
            )
        )

    # ── Matrix control labels & row separators ────────────
    label_x = left - 36 if alt_groups else left - 8

    for row, name in enumerate(controls):
        ry = matrix_y + matrix_pad + row * row_h
        group_label_fill = alt_group_color_by_control.get(str(name))
        label_text = str(name)
        if name in starred_controls:
            highlight_w = max(26, len(label_text) * 6.2 + 6)
            p.append(
                _tag(
                    "rect",
                    {
                        "x": _fmt(label_x - highlight_w + 3),
                        "y": _fmt(ry + 3),
                        "width": _fmt(highlight_w),
                        "height": _fmt(row_h - 5),
                        "rx": "2",
                        "class": "control-label-highlight sticky-x",
                    },
                )
            )
        p.append(
            _tag(
                "text",
                {
                    "x": label_x,
                    "y": _fmt(ry + row_h * 0.65),
                    "text-anchor": "end",
                    "class": "control-label sticky-x sticky-label",
                    "style": f"--control-label-fill: {group_label_fill}" if group_label_fill else None,
                    "data-control": name,
                },
                label_text,
            )
        )
        p.append(
            _tag(
                "line",
                {
                    "x1": left,
                    "x2": width - right,
                    "y1": _fmt(ry),
                    "y2": _fmt(ry),
                    "stroke": "#F3F4F6",
                    "stroke-width": "0.6",
                },
            )
        )
    if controls:
        p.append(
            _tag(
                "line",
                {
                    "x1": left,
                    "x2": width - right,
                    "y1": _fmt(matrix_y + matrix_h),
                    "y2": _fmt(matrix_y + matrix_h),
                    "stroke": "#F3F4F6",
                    "stroke-width": "0.6",
                },
            )
        )

    # ── Alt-group bracket markers ─────────────────────────
    for grp in alt_groups:
        s = int(grp.get("start", -1))
        e = int(grp.get("end", -1))
        if s < 0 or e < s or e >= len(controls):
            continue
        y0 = matrix_y + matrix_pad + s * row_h
        y1 = matrix_y + matrix_pad + (e + 1) * row_h
        xm = left - 18
        dash = "4 3" if str(grp.get("kind", "")) == "controls_test" else None
        for attrs, line_dash in [
            ({"x1": xm, "x2": xm, "y1": _fmt(y0), "y2": _fmt(y1)}, dash),
            ({"x1": _fmt(xm - 6), "x2": _fmt(xm + 6), "y1": _fmt(y0), "y2": _fmt(y0)}, None),
            ({"x1": _fmt(xm - 6), "x2": _fmt(xm + 6), "y1": _fmt(y1), "y2": _fmt(y1)}, None),
        ]:
            p.append(
                _tag(
                    "line",
                    {
                        **attrs,
                        "class": "alt-marker sticky-x",
                        "stroke": "#6B7280",
                        "stroke-width": "1.4",
                        "stroke-linecap": "square",
                        "stroke-dasharray": line_dash,
                    },
                )
            )

    # ── Matrix run-length colors, same idea as PNG output ─
    matrix_cell_fill: dict[tuple[int, int], str] = {}
    row_runs: list[tuple[int, int, int]] = []
    max_run_len = 1
    for row, name in enumerate(controls):
        start: int | None = None
        for idx, rec in enumerate(records + [{"included_matrix_controls": []}]):
            included = name in set(rec["included_matrix_controls"])
            if included and start is None:
                start = idx
            elif not included and start is not None:
                run_len = idx - start
                row_runs.append((row, start, idx))
                max_run_len = max(max_run_len, run_len)
                start = None
    for row, start, end in row_runs:
        run_len = end - start
        t = (run_len / max_run_len) ** 0.6
        value = round(200 - 200 * t)
        fill = f"rgb({value},{value},{value})"
        for idx in range(start, end):
            matrix_cell_fill[(row, idx)] = fill

    # ── Swimlane backgrounds for alt groups ───────────────
    for grp in alt_groups:
        s = int(grp.get("start", -1))
        e = int(grp.get("end", -1))
        if s < 0 or e <= s or e >= len(controls):
            continue
        grp_color = alt_group_color_by_control.get(str(controls[s]), _ALT_GROUP_COLORS[0])
        ry0 = matrix_y + s * row_h
        p.append(
            _tag(
                "rect",
                {
                    "x": _fmt(left),
                    "y": _fmt(ry0),
                    "width": _fmt(width - left - right),
                    "height": _fmt((e - s + 1) * row_h),
                    "fill": grp_color,
                    "opacity": "0.12",
                    "pointer-events": "none",
                },
            )
        )

    # ── Per-record elements ────────────────────────────────
    for idx, rec in enumerate(records):
        x = xc(idx)
        star = int(rec["star"])
        coef = float(rec["coef"])
        color = rec["color"]
        _show_sp_star = payload.get("showSpecialMarkers", True)
        _is_full_star = _show_sp_star and bool(rec.get("is_full", False))
        _is_noc_star = _show_sp_star and bool(rec.get("is_no_controls_test", False))
        star_fill = (
            "#FF2F92" if _is_full_star
            else "#ff8c00" if _is_noc_star
            else (_STAR_NEG if coef < 0 else _STAR_POS)
        )
        star_zero_stroke = "#FF2F92" if _is_full_star else "#ff8c00" if _is_noc_star else "#D1D5DB"
        dirn = -1 if coef < 0 else 1

        # Star block
        if star == 0:
            p.append(
                _tag(
                    "line",
                    {
                        "x1": _fmt(x - x_step * 0.36),
                        "x2": _fmt(x + x_step * 0.36),
                        "y1": _fmt(sy(0)),
                        "y2": _fmt(sy(0)),
                        "stroke": star_zero_stroke,
                        "stroke-width": "0.9",
                        "class": "star-zero-segment",
                        "data-index": idx,
                    },
                )
            )
        else:
            for blk in range(star):
                by = sy(dirn * (blk + 0.74))
                p.append(
                    _tag(
                        "rect",
                        {
                            "x": _fmt(x - x_step * 0.36),
                            "y": _fmt(by - 3.6),
                            "width": max(1, x_step * 0.72),
                            "height": 7,
                            "rx": "1",
                            "fill": star_fill,
                            "class": "star-cell",
                            "data-index": idx,
                        },
                    )
                )

        # Coefficient point
        p.append(
            _tag(
                "circle",
                {
                    "cx": _fmt(x),
                    "cy": _fmt(cy(coef)),
                    "r": "3.2",
                    "fill": color,
                    "class": "point hoverable",
                    "data-index": idx,
                },
            )
        )

        # Control matrix cells
        included = set(rec["included_matrix_controls"])
        for row, name in enumerate(controls):
            if name not in included:
                continue
            ry = matrix_y + matrix_pad + row * row_h
            normal_fill = matrix_cell_fill.get((row, idx), "#1F2937")
            group_fill = alt_group_color_by_control.get(str(name))
            cell_fill = (
                "#FF2F92" if _is_full_star
                else "#ff8c00" if _is_noc_star
                else normal_fill
            )
            p.append(
                _tag(
                    "rect",
                    {
                        "x": _fmt(x - x_step * 0.40),
                        "y": _fmt(ry + 2),
                        "width": max(1, x_step * 0.80),
                        "height": row_h - 4,
                        "rx": "1.5",
                        "fill": cell_fill,
                        "style": f"--normal-fill: {normal_fill}; --group-fill: {group_fill or normal_fill}",
                        "class": "matrix-cell",
                        "data-index": idx,
                        "data-control": name,
                    },
                )
            )

        # Obs bar
        obs_value = int(rec["obs"])
        obs_base_y = oy(float(obs_mean))
        obs_value_y = oy(float(obs_value))
        obs_gap = 2.0
        obs_min_h = 0.7
        _show_sp = payload.get("showSpecialMarkers", True)
        _is_full_bar = _show_sp and bool(rec.get("is_full", False))
        _is_noc_bar = _show_sp and bool(rec.get("is_no_controls_test", False))
        obs_bar_fill = "#FF2F92" if _is_full_bar else "#ff8c00" if _is_noc_bar else _OBS_FILL
        obs_bar_opacity = "1" if (_is_full_bar or _is_noc_bar) else ""
        if obs_value_y < obs_base_y:
            obs_bar_bottom = obs_base_y - obs_gap
            obs_bar_y = min(obs_value_y, obs_bar_bottom - obs_min_h)
            obs_bar_h = obs_bar_bottom - obs_bar_y
        elif obs_value_y > obs_base_y:
            obs_bar_y = obs_base_y + obs_gap
            obs_bar_bottom = max(obs_value_y, obs_bar_y + obs_min_h)
            obs_bar_h = obs_bar_bottom - obs_bar_y
        else:
            obs_bar_y = obs_base_y + obs_gap
            obs_bar_h = obs_min_h
        p.append(
            _tag(
                "rect",
                {
                    "x": _fmt(x - x_step * 0.38),
                    "y": _fmt(obs_bar_y),
                    "width": max(1, x_step * 0.76),
                    "height": _fmt(obs_bar_h),
                    "style": f"--obs-fill: {obs_bar_fill}" + (f"; opacity: {obs_bar_opacity}" if obs_bar_opacity else ""),
                    "rx": "1.5",
                    "class": "obs-bar",
                    "data-obs-gap": _fmt(obs_gap),
                    "data-index": idx,
                },
            )
        )

        # Vertical guide line (broken around point)
        gap_r = 7.0
        pts = [
            f"M {_fmt(x)} {_fmt(star_y)} L {_fmt(x)} {_fmt(star_y + star_h)}",
            f"M {_fmt(x)} {_fmt(coef_y)} L {_fmt(x)} {_fmt(max(coef_y, cy(coef) - gap_r))}",
            f"M {_fmt(x)} {_fmt(min(coef_y + coef_h, cy(coef) + gap_r))} L {_fmt(x)} {_fmt(coef_y + coef_h)}",
            f"M {_fmt(x)} {_fmt(matrix_y)} L {_fmt(x)} {_fmt(matrix_y + matrix_h)}",
            f"M {_fmt(x)} {_fmt(obs_y)} L {_fmt(x)} {_fmt(obs_y + obs_h)}",
        ]
        p.append(
            _tag("path", {"d": " ".join(pts), "class": "guide", "data-index": idx})
        )

        # Invisible wide hit region for hover
        p.append(
            _tag(
                "rect",
                {
                    "x": _fmt(x - x_step / 2),
                    "y": _fmt(star_y),
                    "width": max(3, x_step),
                    "height": _fmt(obs_y + obs_h - star_y),
                    "fill": "transparent",
                    "class": "hoverable",
                    "data-index": idx,
                    "onmouseenter": f"activate({idx})",
                    "onmouseleave": "clearActive()",
                    "onclick": f"togglePin({idx},event)",
                },
            )
        )

    # ── PNG-style special vertical markers ────────────────
    if payload.get("showSpecialMarkers", True):
        for kind, css_name in [
            ("is_no_controls_test", "special-nocontrol"),
            ("is_full", "special-full"),
        ]:
            for idx, rec in enumerate(records):
                fallback_no_controls = not rec.get("included_matrix_controls", [])
                default_marker = fallback_no_controls if kind == "is_no_controls_test" else False
                if not bool(rec.get(kind, default_marker)):
                    continue
                x = xc(idx)
                coef_y_at_x = cy(float(rec["coef"]))
                gap_r = 4.8
                pts = [
                    f"M {_fmt(x)} {_fmt(star_y)} L {_fmt(x)} {_fmt(star_y + star_h)}",
                    f"M {_fmt(x)} {_fmt(coef_y)} L {_fmt(x)} {_fmt(max(coef_y, coef_y_at_x - gap_r))}",
                    f"M {_fmt(x)} {_fmt(min(coef_y + coef_h, coef_y_at_x + gap_r))} L {_fmt(x)} {_fmt(coef_y + coef_h)}",
                    f"M {_fmt(x)} {_fmt(matrix_y)} L {_fmt(x)} {_fmt(matrix_y + matrix_h)}",
                    f"M {_fmt(x)} {_fmt(obs_y)} L {_fmt(x)} {_fmt(obs_y + obs_h)}",
                ]
                p.append(
                    _tag(
                        "path",
                        {
                            "d": " ".join(pts),
                            "class": f"special-line {css_name}",
                            "data-special-index": idx,
                        },
                    )
                )

    p.append(
        _tag(
            "circle",
            {
                "id": "active-ring",
                "class": "active-ring",
                "cx": "0",
                "cy": "0",
                "r": "6",
            },
        )
    )

    p.append("</svg>")
    return "\n  ".join(p), width, height


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
    parser.add_argument(
        "--order", choices=["coef", "p"], help="网页排序方式：coef 或 p"
    )
    parser.add_argument(
        "--p", action="store_true", default=None, help="兼容别名；等价于 --order p"
    )
    args = parser.parse_args()

    out = html_from_files(
        results_path=args.results,
        meta_path=args.meta,
        output_path=args.output,
        order=args.order,
        sort_by_signed_p=args.p,
    )
    print(f"✓  {out}")


if __name__ == "__main__":
    main()
