# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "numpy",
#   "pandas",
#   "matplotlib",
#   "scipy",
# ]
# ///
"""
regression_monkey_plot.py
=========================
独立绘图脚本：只读取分析脚本导出的标准结果文件和绘图元数据，然后生成 PNG。
"""

from __future__ import annotations

import argparse
import json
import pathlib
from typing import Any, cast

import pandas as pd

import regression_monkey_py as rm_py


def load_plot_meta(meta_path: pathlib.Path) -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(meta_path.read_text(encoding="utf-8")))


def plot_from_files(
    *,
    results_path: str | pathlib.Path,
    meta_path: str | pathlib.Path,
    output_path: str | pathlib.Path | None = None,
    dpi: int | None = None,
    fig_width: float | None = None,
) -> None:
    """Render one specification-curve PNG from standard handoff files."""
    results_file = pathlib.Path(results_path)
    meta_file = pathlib.Path(meta_path)
    meta = load_plot_meta(meta_file)
    records = rm_py.records_from_dataframe(pd.read_csv(results_file))

    out = pathlib.Path(output_path) if output_path is not None else pathlib.Path(meta["output_path"])
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
    args = parser.parse_args()

    plot_from_files(
        results_path=args.results,
        meta_path=args.meta,
        output_path=args.output,
        dpi=args.dpi,
        fig_width=args.fig_width,
    )


if __name__ == "__main__":
    main()
