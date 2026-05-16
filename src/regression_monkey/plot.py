"""
regression_monkey · plot
=========================
独立绘图脚本：只读取分析脚本导出的标准结果文件和绘图元数据，然后生成 PNG。
"""

from __future__ import annotations

import argparse
import json
import pathlib
from typing import Any, cast

import pandas as pd

from . import py as rm_py


def load_plot_meta(meta_path: pathlib.Path) -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(meta_path.read_text(encoding="utf-8")))


def plot_from_files(
    *,
    results_path: str | pathlib.Path,
    meta_path: str | pathlib.Path,
    output_path: str | pathlib.Path | None = None,
    dpi: int | None = None,
    fig_width: float | None = None,
    order: str | None = None,
    sort_by_signed_p: bool | None = None,
    verbose: bool = True,
) -> None:
    """Render one specification-curve PNG from standard handoff files."""
    results_file = pathlib.Path(results_path)
    meta_file = pathlib.Path(meta_path)
    meta = load_plot_meta(meta_file)
    records = rm_py.records_from_dataframe(pd.read_csv(results_file))
    p_alias = bool(sort_by_signed_p) if sort_by_signed_p is not None else (
        bool(meta.get("sort_by_p_mode", meta.get("sort_by_signed_p", False)))
        if order is None
        else False
    )
    plot_order = rm_py._normalize_plot_order(
        str(meta.get("order", "coef") if order is None else order),
        p_alias=p_alias,
    )

    out = pathlib.Path(output_path) if output_path is not None else pathlib.Path(meta["output_path"])
    out.parent.mkdir(parents=True, exist_ok=True)
    _alt_groups = list(meta.get("matrix_alt_groups", []))
    _swimlane_ranges = [
        (int(g["start"]), int(g["end"]))
        for g in _alt_groups
        if int(g.get("end", -1)) > int(g.get("start", -1))
    ] or None
    rm_py._plot(
        records=records,
        y_name=str(meta["y"]),
        x_name=str(meta["x"]),
        controls_test=list(meta.get("controls_test_flat", [])),
        controls_must=list(meta.get("controls_must_flat", [])),
        matrix_controls=list(meta.get("matrix_controls", meta.get("controls_test_flat", []))),
        show_special_markers=bool(meta.get("show_special_markers", True)),
        fig_width=float(fig_width if fig_width is not None else meta.get("fig_width", 14.0)),
        dpi=int(dpi if dpi is not None else meta.get("dpi", 150)),
        output_path=str(out),
        title_suffix=meta.get("title_suffix"),
        elapsed_seconds_preplot=meta.get("elapsed_seconds_preplot"),
        engine=meta.get("engine"),
        grouping_variable=meta.get("grouping_variable"),
        grouped_plot_records=list(meta.get("grouped_plot_records", [])),
        interaction_plot_records=list(meta.get("interaction_plot_records", [])),
        sort_by_signed_p=rm_py._order_uses_p_mode(plot_order),
        verbose=verbose,
        matrix_swimlane_ranges=_swimlane_ranges,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="regression_monkey_plot",
        description="Plot Regression Monkey results from *_results.csv and *_plot_meta.json.",
    )
    parser.add_argument("--results", required=True, metavar="CSV")
    parser.add_argument("--meta", required=True, metavar="JSON")
    parser.add_argument("--output", metavar="PNG")
    parser.add_argument("--dpi", type=int)
    parser.add_argument("--fig-width", type=float)
    parser.add_argument("--order", choices=["coef", "p"], help="绘图排序方式：coef 或 p")
    parser.add_argument("--p", action="store_true", default=None, help="兼容别名；等价于 --order p")
    args = parser.parse_args()

    plot_from_files(
        results_path=args.results,
        meta_path=args.meta,
        output_path=args.output,
        dpi=args.dpi,
        fig_width=args.fig_width,
        order=args.order,
        sort_by_signed_p=args.p,
    )


if __name__ == "__main__":
    main()
