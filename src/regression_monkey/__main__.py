"""
regression_monkey
=================
主入口：读取配置/CLI，调度 Python 或 Stata 分析引擎，再调用独立绘图脚本。
"""

from __future__ import annotations

from datetime import datetime, timedelta
import argparse
from collections import Counter, defaultdict
import json
import itertools
import pathlib
import sys
from time import perf_counter
from typing import Any, Callable, cast

import pandas as pd

from . import common as rm_common
from . import html as rm_html
from . import plot as rm_plot
from . import py as rm_py


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--engine", choices=["python", "stata"], default="python")
    parser.add_argument("--data", metavar="FILE")
    parser.add_argument("--y", metavar="VAR", nargs="+")
    parser.add_argument("--x", metavar="VAR", nargs="+")
    parser.add_argument("--controls", metavar="VAR", nargs="+", help="兼容旧配置；等价于 --controls-test")
    parser.add_argument("--controls-test", dest="controls_test", metavar="VAR", nargs="+")
    parser.add_argument("--controls-must", dest="controls_must", metavar="VAR", nargs="+")
    parser.add_argument("--grouping-variable", dest="grouping_variable", metavar="VAR", nargs="+", help="兼容别名；等价于 --grouping-variable-by-ind-time")
    parser.add_argument("--grouping-variable-by-ind-time", dest="grouping_variable_by_ind_time", metavar="VAR", nargs="+")
    parser.add_argument("--grouping-variable-by-time", dest="grouping_variable_by_time", metavar="VAR", nargs="+")
    parser.add_argument("--grouping-variable-by-none", dest="grouping_variable_by_none", metavar="VAR", nargs="+")
    parser.add_argument("--Firm-FE", dest="firm_fe", default="code", metavar="COL")
    parser.add_argument("--Ind-FE", dest="ind_fe", default="ind", metavar="COL")
    parser.add_argument("--Time-FE", dest="time_fe", default="year", metavar="COL")
    parser.add_argument("--Region-FE", dest="region_fe", default=None, metavar="COL")
    parser.add_argument("--fe", metavar="COL", nargs="+")
    parser.add_argument("--clust", metavar="COL", nargs="+")
    parser.add_argument("--gen-clust2", dest="gen_clust2", action="store_true")
    parser.add_argument("--output", default="outputs", metavar="DIR")
    parser.add_argument("--export-format", choices=["png", "html", "both"], default="png", help="导出格式：png、html 或 both")
    parser.add_argument("--dpi", default=150, type=int)
    parser.add_argument("--fig-width", default=14.0, type=float, metavar="INCHES")
    parser.add_argument("--n-jobs", default=0, type=int, metavar="N")
    parser.add_argument("--order", choices=["coef", "p"], default="coef", help="绘图排序方式：coef 或 p")
    parser.add_argument("--p", action="store_true", help="兼容别名；等价于 --order p")
    parser.add_argument("--stata-path", default="stata-mp", metavar="EXE")
    parser.add_argument("--keep-temp", action="store_true")
    for spec_name in rm_py._ALL_SPEC_NAMES:
        parser.add_argument(f"--{spec_name.replace('_', '-')}", dest=spec_name, action="store_true")


def _enabled_specs(args: argparse.Namespace) -> dict[str, bool]:
    return {name: bool(getattr(args, name, False)) for name in rm_py._ALL_SPEC_NAMES}


def _spec_columns(spec_def: dict[str, Any], args: argparse.Namespace) -> tuple[list[str], list[str], str | None]:
    var_map = {"firm": args.firm_fe, "ind": args.ind_fe, "time": args.time_fe}
    if args.region_fe:
        var_map["region"] = args.region_fe
    fe_cols: list[str] = []
    for key in spec_def["fe_keys"]:
        if key == "_ind_time":
            fe_cols.append(f"_spec_{args.ind_fe}_{args.time_fe}")
        elif key == "_region_time":
            fe_cols.append(f"_spec_{args.region_fe}_{args.time_fe}")
        else:
            fe_cols.append(var_map[key])
    clust_cols = [var_map[k] for k in spec_def["cl_keys"]]
    vce_label = "robust" if spec_def["vce"] == "robust" else None
    return fe_cols, clust_cols, vce_label


def _drawable_auto_specs(spec_flags: dict[str, bool], region_fe: str | None) -> list[dict[str, Any]]:
    return [
        spec_def for spec_def in rm_py._SPEC_CATALOG
        if spec_flags.get(spec_def["name"], False)
        and not (spec_def["needs_region"] and region_fe is None)
    ]


def _safe_path_part(value: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in value)
    cleaned = "_".join(part for part in cleaned.split("_") if part)
    return cleaned or "spec"


def _plot_output_path(run_output_dir: pathlib.Path, group_name: str, filename: str) -> pathlib.Path:
    output_dir = run_output_dir / _safe_path_part(group_name)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / filename


def _render_output_path(run_output_dir: pathlib.Path, group_name: str, filename: str, export_format: str) -> pathlib.Path:
    if export_format == "html":
        return run_output_dir / filename
    return _plot_output_path(run_output_dir, group_name, filename)


def _manual_plot_group(fe_cols: list[str], clust_cols: list[str]) -> str:
    fe_part = "_".join(_safe_path_part(col) for col in fe_cols)
    clust_part = "_".join(_safe_path_part(col) for col in clust_cols) if clust_cols else "robust"
    return f"manual_ab_{fe_part}_cl_{clust_part}"


def _matrix_alternative_groups(
    *,
    controls_must_slots: list[rm_py.ControlSlot],
    controls_test_slots: list[rm_py.ControlSlot],
) -> list[dict[str, Any]]:
    groups: list[dict[str, Any]] = []
    cursor = 0
    for slot in controls_must_slots:
        if len(slot) <= 1:
            continue
        start = cursor
        end = cursor + len(slot) - 1
        groups.append({"kind": "controls_must", "start": start, "end": end, "names": list(slot), "label": f"1 of {len(slot)}"})
        cursor = end + 1

    for slot in controls_test_slots:
        if len(slot) <= 1:
            cursor += 1
            continue
        start = cursor
        end = cursor + len(slot) - 1
        groups.append({"kind": "controls_test", "start": start, "end": end, "names": list(slot), "label": f"0/1 of {len(slot)}"})
        cursor = end + 1
    return groups


def _write_and_plot(
    *,
    records: list[rm_py.SpecRecord],
    results_path: pathlib.Path,
    meta_path: pathlib.Path,
    output_path: pathlib.Path,
    meta: dict[str, Any],
    verbose: bool = True,
    html_bundle_payloads: list[dict[str, Any]] | None = None,
) -> None:
    rm_py.write_analysis_artifacts(
        records=records,
        results_path=results_path,
        meta_path=meta_path,
        meta={**meta, "output_path": str(output_path)},
        verbose=verbose,
    )
    _render_from_files(
        results_path=results_path,
        meta_path=meta_path,
        output_path=output_path,
        export_format=str(meta.get("export_format", "png")),
        verbose=verbose,
        html_bundle_payloads=html_bundle_payloads,
    )


def _render_from_files(
    *,
    results_path: pathlib.Path,
    meta_path: pathlib.Path,
    output_path: pathlib.Path,
    export_format: str,
    verbose: bool = True,
    html_bundle_payloads: list[dict[str, Any]] | None = None,
) -> None:
    if export_format in {"png", "both"}:
        rm_plot.plot_from_files(
            results_path=results_path,
            meta_path=meta_path,
            output_path=output_path,
            verbose=verbose,
        )
    if export_format in {"html", "both"}:
        if html_bundle_payloads is not None:
            html_bundle_payloads.append(
                rm_html.payload_from_files(
                    results_path=results_path,
                    meta_path=meta_path,
                )
            )
        else:
            html_output_path = output_path.with_suffix(".html")
            rm_html.html_from_files(
                results_path=results_path,
                meta_path=meta_path,
                output_path=html_output_path,
            )
            if verbose:
                print(f"[Saved] {html_output_path}")


def _cleanup_plot_handoff(*, results_path: pathlib.Path, meta_path: pathlib.Path, keep_temp: bool) -> None:
    if keep_temp:
        return
    rm_common.safe_unlink(results_path)
    rm_common.safe_unlink(meta_path)


def _format_plot_progress(done: int, total: int, *, width: int = 30) -> str:
    if total <= 0:
        return "[导出进度] 无待导出结果"
    done = min(max(done, 0), total)
    filled = round(width * done / total)
    bar = "#" * filled + "-" * (width - filled)
    pct = done * 100 / total
    return f"[导出进度] |{bar}| {done}/{total} ({pct:5.1f}%)"


def _format_duration(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{secs:02d}s"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


def _fe_type_label(fe_keys: tuple[str, ...]) -> str:
    return "+".join(fe_keys) if fe_keys else "none"


class _PlotProgressEstimator:
    def __init__(self, planned_fe_types: list[tuple[str, ...]]) -> None:
        self.total = len(planned_fe_types)
        self.done = 0
        self.remaining = Counter(planned_fe_types)
        self.samples: dict[tuple[str, ...], list[float]] = defaultdict(list)

    def update(self, fe_type: tuple[str, ...], elapsed_seconds: float) -> str:
        self.done += 1
        if self.remaining[fe_type] > 0:
            self.remaining[fe_type] -= 1
        self.samples[fe_type].append(elapsed_seconds)

        parts = [_format_plot_progress(self.done, self.total)]
        if self.total:
            parts.append(f"FE={_fe_type_label(fe_type)}")
            parts.append(f"本张={_format_duration(elapsed_seconds)}")
            eta = self.estimate_remaining_seconds()
            if eta is None:
                missing = [
                    _fe_type_label(key)
                    for key, count in self.remaining.items()
                    if count > 0 and key not in self.samples
                ]
                parts.append("ETA=等待各FE类型首张样本")
                if missing:
                    parts.append(f"待样本={','.join(missing)}")
            else:
                finish_at = datetime.now() + timedelta(seconds=eta)
                parts.append(f"剩余≈{_format_duration(eta)}")
                parts.append(f"预计完成≈{finish_at:%H:%M:%S}")
        return "  ".join(parts)

    def estimate_remaining_seconds(self) -> float | None:
        if self.total <= 0 or not self.remaining:
            return 0.0
        for fe_type, count in self.remaining.items():
            if count > 0 and fe_type not in self.samples:
                return None
        remaining_seconds = 0.0
        for fe_type, count in self.remaining.items():
            if count <= 0:
                continue
            durations = self.samples[fe_type]
            remaining_seconds += count * (sum(durations) / len(durations))
        return remaining_seconds


def _run_python_pair(
    *,
    df: pd.DataFrame,
    args: argparse.Namespace,
    y_var: str,
    x_var: str,
    controls_test: rm_py.ControlSpecInput,
    controls_must: rm_py.ControlSpecInput,
    controls_test_flat: list[str],
    controls_must_flat: list[str],
    matrix_controls: list[str],
    matrix_alt_groups: list[dict[str, Any]],
    spec_flags: dict[str, bool],
    is_auto: bool,
    run_output_dir: pathlib.Path,
    resolved_n_jobs: int,
    on_plot_done: Callable[[pathlib.Path, tuple[str, ...], float], None] | None = None,
    html_bundle_payloads: list[dict[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], int]:
    pair_sig_rows: list[dict[str, Any]] = []
    pair_total_specs = 0
    pair_stem = f"{y_var}_{x_var}"

    if is_auto:
        fmt = {
            "firm": args.firm_fe,
            "ind": args.ind_fe,
            "time": args.time_fe,
            "region": args.region_fe or "region",
        }
        enabled_spec_defs = [
            spec_def for spec_def in rm_py._SPEC_CATALOG
            if spec_flags.get(spec_def["name"], False)
        ]
        for spec_def in enabled_spec_defs:
            spec_name = spec_def["name"]
            spec_t0 = perf_counter()
            auto_results = rm_py.regression_monkey_auto(
                df=df,
                y=y_var,
                x=x_var,
                controls_test=controls_test,
                controls_must=controls_must,
                firm_fe=args.firm_fe,
                ind_fe=args.ind_fe,
                time_fe=args.time_fe,
                region_fe=args.region_fe,
                specs={name: name == spec_name for name in rm_py._ALL_SPEC_NAMES},
                output_path=None,
                dpi=args.dpi,
                fig_width=args.fig_width,
                n_jobs=resolved_n_jobs,
                export_sig_table=False,
                render_plot=False,
            )
            if not auto_results:
                continue
            _, records, _fig = auto_results[0]
            spec_def = next(s for s in rm_py._SPEC_CATALOG if s["name"] == spec_name)
            out_png = _render_output_path(
                run_output_dir,
                str(spec_def["tag"]),
                f"{pair_stem}_{spec_def['tag']}.png",
                args.export_format,
            )
            results_path = run_output_dir / f"{pair_stem}_{spec_def['tag']}_results.csv"
            meta_path = run_output_dir / f"{pair_stem}_{spec_def['tag']}_plot_meta.json"
            _write_and_plot(
                records=records,
                results_path=results_path,
                meta_path=meta_path,
                output_path=out_png,
                meta={
                    "engine": "python",
                    "spec_name": spec_name,
                    "y": y_var,
                    "x": x_var,
                    "controls_test_flat": controls_test_flat,
                    "controls_must_flat": controls_must_flat,
                    "matrix_controls": matrix_controls,
                    "matrix_alt_groups": matrix_alt_groups,
                    "show_special_markers": True,
                    "fig_width": args.fig_width,
                    "dpi": args.dpi,
                    "order": args.order,
                    "sort_by_p_mode": rm_py._order_uses_p_mode(args.order),
                    "sort_by_signed_p": rm_py._order_uses_p_mode(args.order),
                    "title_suffix": spec_def["help"].format(**fmt),
                    "elapsed_seconds_preplot": perf_counter() - spec_t0,
                    "export_format": args.export_format,
                },
                html_bundle_payloads=html_bundle_payloads,
            )
            _cleanup_plot_handoff(
                results_path=results_path,
                meta_path=meta_path,
                keep_temp=bool(args.keep_temp),
            )
            if on_plot_done is not None:
                on_plot_done(out_png, tuple(spec_def["fe_keys"]), perf_counter() - spec_t0)
            fe_cols, clust_cols, vce_label = _spec_columns(spec_def, args)
            rows = rm_py._build_sig_rows(records, y_var, x_var, controls_must_flat, controls_test_flat, fe_cols, clust_cols, vce_label)
            pair_sig_rows.extend(rows)
            pair_total_specs += len(records)
        return pair_sig_rows, pair_total_specs

    clust_cols = list(args.clust) if args.clust else []
    if args.gen_clust2:
        clust2_col = f"{args.fe[0]}_{args.fe[1]}"
        df[clust2_col] = df[args.fe[0]].astype(str) + "_" + df[args.fe[1]].astype(str)
        print(f"已生成聚类变量：{clust2_col}")
        clust_cols.append(clust2_col)

    spec_t0 = perf_counter()
    records, _fig = rm_py.regression_monkey(
        df=df,
        y=y_var,
        x=x_var,
        controls_test=controls_test,
        controls_must=controls_must,
        fe_cols=list(args.fe),
        clust_cols=clust_cols,
        output_path=None,
        dpi=args.dpi,
        fig_width=args.fig_width,
        n_jobs=resolved_n_jobs,
        export_sig_table=False,
        render_plot=False,
    )
    out_png = _render_output_path(
        run_output_dir,
        _manual_plot_group(list(args.fe), clust_cols),
        f"{pair_stem}.png",
        args.export_format,
    )
    results_path = run_output_dir / f"{pair_stem}_results.csv"
    meta_path = run_output_dir / f"{pair_stem}_plot_meta.json"
    _write_and_plot(
        records=records,
        results_path=results_path,
        meta_path=meta_path,
        output_path=out_png,
        meta={
            "engine": "python",
            "spec_name": "manual",
            "y": y_var,
            "x": x_var,
            "controls_test_flat": controls_test_flat,
            "controls_must_flat": controls_must_flat,
            "matrix_controls": matrix_controls,
            "matrix_alt_groups": matrix_alt_groups,
            "show_special_markers": True,
            "fig_width": args.fig_width,
            "dpi": args.dpi,
            "order": args.order,
            "sort_by_p_mode": rm_py._order_uses_p_mode(args.order),
            "sort_by_signed_p": rm_py._order_uses_p_mode(args.order),
            "title_suffix": f"manual FE = {', '.join(args.fe)}",
            "elapsed_seconds_preplot": perf_counter() - spec_t0,
            "export_format": args.export_format,
        },
        html_bundle_payloads=html_bundle_payloads,
    )
    _cleanup_plot_handoff(
        results_path=results_path,
        meta_path=meta_path,
        keep_temp=bool(args.keep_temp),
    )
    if on_plot_done is not None:
        on_plot_done(out_png, tuple(args.fe), perf_counter() - spec_t0)
    pair_sig_rows.extend(rm_py._build_sig_rows(records, y_var, x_var, controls_must_flat, controls_test_flat, list(args.fe), clust_cols))
    pair_total_specs += len(records)
    return pair_sig_rows, pair_total_specs


def main() -> None:
    try:
        cfg, cli_args = rm_common.load_toml_config(sys.argv[1:])
    except FileNotFoundError as exc:
        print(f"错误：{exc}", file=sys.stderr)
        sys.exit(1)

    parser = argparse.ArgumentParser(
        prog="regression_monkey",
        description="规格曲线分析主入口：调度 Python/Stata 分析并独立绘图。",
    )
    _add_common_args(parser)
    if cfg:
        allowed = {
            "engine", "data", "y", "x", "controls", "controls_test", "controls_must",
            "grouping_variable", "grouping_variable_by_ind_time",
            "grouping_variable_by_time", "grouping_variable_by_none",
            "output", "dpi", "fig_width", "n_jobs", "order", "p", "firm_fe", "ind_fe", "time_fe",
            "region_fe", "fe", "clust", "gen_clust2", "stata_path", "keep_temp", "export_format",
        } | set(rm_py._ALL_SPEC_NAMES)
        normalized = {k.lower(): v for k, v in cfg.items()}
        parser.set_defaults(**{k: v for k, v in normalized.items() if k in allowed})

    args = parser.parse_args(cli_args)
    try:
        args.order = rm_py._normalize_plot_order(args.order, p_alias=bool(args.p))
    except ValueError as exc:
        parser.error(str(exc))
    try:
        args.y = rm_py._expand_space_separated_names(args.y)
        args.x = rm_py._expand_space_separated_names(args.x)
        args.fe = rm_py._expand_space_separated_names(args.fe)
        args.clust = rm_py._expand_space_separated_names(args.clust)
        args.grouping_variable = rm_py._expand_space_separated_names(args.grouping_variable)
        args.grouping_variable_by_ind_time = rm_py._expand_space_separated_names(args.grouping_variable_by_ind_time)
        args.grouping_variable_by_time = rm_py._expand_space_separated_names(args.grouping_variable_by_time)
        args.grouping_variable_by_none = rm_py._expand_space_separated_names(args.grouping_variable_by_none)
    except ValueError as exc:
        parser.error(str(exc))
    controls_test = list(args.controls_test) if args.controls_test else (list(args.controls) if args.controls else [])
    controls_must = list(args.controls_must) if args.controls_must else []
    try:
        controls_test_flat, controls_test_slots = rm_py._normalize_controls_test(controls_test)
        controls_must_flat, controls_must_slots = rm_py._normalize_controls_must(controls_must)
        rm_py._validate_control_lists_do_not_overlap(controls_test_flat, controls_must_flat)
    except ValueError as exc:
        parser.error(str(exc))
    matrix_controls = rm_py._varying_must_controls(controls_must_slots) + controls_test_flat
    matrix_alt_groups = _matrix_alternative_groups(
        controls_must_slots=controls_must_slots,
        controls_test_slots=controls_test_slots,
    )

    if not args.data or not args.y or not args.x:
        parser.error("必须提供 data / y / x（可通过 TOML 或 CLI 指定）")
    if not controls_test and not controls_must:
        parser.error("至少提供一类控制变量：controls_test/controls 或 controls_must")

    spec_flags = _enabled_specs(args)
    is_auto = any(spec_flags.values())
    if not is_auto and args.engine == "stata":
        parser.error("Stata 引擎仅支持自动规格模式，请至少启用一个 absorb_* flag。")
    grouping_specs = rm_py._collect_grouping_variable_specs(
        grouping_variable=list(args.grouping_variable or []),
        grouping_variable_by_ind_time=list(args.grouping_variable_by_ind_time or []),
        grouping_variable_by_time=list(args.grouping_variable_by_time or []),
        grouping_variable_by_none=list(args.grouping_variable_by_none or []),
    )
    if grouping_specs and args.engine != "stata":
        parser.error("grouping_variable_* 仅支持 --engine stata。")
    if not is_auto:
        if not args.fe:
            parser.error("手动模式需要 --fe。")
        if not args.clust and not args.gen_clust2:
            parser.error("手动模式需要 --clust 或 --gen-clust2。")
        if args.gen_clust2 and len(args.fe) < 2:
            parser.error("--gen-clust2 需要至少提供 2 个 --fe 列。")
        if grouping_specs:
            parser.error("grouping_variable_* 仅支持自动规格模式。")

    data_path = pathlib.Path(args.data).expanduser()
    print(f"读取数据：{data_path}")
    df = rm_common.load_dataframe(data_path)
    print(f"数据读取完成：{len(df):,} 行 × {len(df.columns)} 列")
    try:
        grouping_specs = rm_py._validate_grouping_variable_specs(
            df,
            grouping_specs,
        )
    except ValueError as exc:
        parser.error(str(exc))

    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = pathlib.Path(args.output).expanduser()
    if output_root.suffix:
        output_root = output_root.parent
    if output_root == pathlib.Path("."):
        output_root = pathlib.Path.cwd() / "outputs"
    run_output_dir = output_root / run_timestamp
    run_output_dir.mkdir(parents=True, exist_ok=True)
    print(f"输出目录：{run_output_dir}")

    resolved_n_jobs = rm_py._resolve_n_jobs(args.n_jobs)
    snapshot_config = {
        "generated_at": run_timestamp,
        "engine": args.engine,
        "data": str(args.data),
        "y": list(args.y),
        "x": list(args.x),
        "controls_test": controls_test,
        "controls_test_flat": controls_test_flat,
        "controls_must": controls_must,
        "controls_must_flat": controls_must_flat,
        "grouping_variable_by_ind_time": [var for scope, var in grouping_specs if scope == "by_ind_time"],
        "grouping_variable_by_time": [var for scope, var in grouping_specs if scope == "by_time"],
        "grouping_variable_by_none": [var for scope, var in grouping_specs if scope == "by_none"],
        "output": str(output_root),
        "run_output_dir": str(run_output_dir),
        "dpi": args.dpi,
        "fig_width": args.fig_width,
        "export_format": args.export_format,
        "order": args.order,
        "n_jobs": args.n_jobs,
        "resolved_n_jobs": resolved_n_jobs,
        "firm_fe": args.firm_fe,
        "ind_fe": args.ind_fe,
        "time_fe": args.time_fe,
    }
    if args.region_fe:
        snapshot_config["region_fe"] = args.region_fe
    if args.engine == "stata":
        snapshot_config["stata_path"] = args.stata_path
    snapshot_config.update({name: enabled for name, enabled in spec_flags.items() if enabled})
    rm_py._write_config_snapshot(snapshot_config, run_output_dir / "config_snapshot.toml")

    combos = list(itertools.product(args.y, args.x))
    all_sig_rows: list[dict[str, Any]] = []
    total_sig_specs = 0
    combo_summaries: list[dict[str, Any]] = []
    planned_plot_fe_types: list[tuple[str, ...]] = []

    if args.engine == "python":
        if is_auto:
            drawable_specs = _drawable_auto_specs(spec_flags, args.region_fe)
            planned_plot_fe_types = [
                tuple(spec_def["fe_keys"])
                for _combo in combos
                for spec_def in drawable_specs
            ]
        else:
            planned_plot_fe_types = [tuple(args.fe) for _combo in combos]
    else:
        drawable_specs = _drawable_auto_specs(spec_flags, args.region_fe)
        grouping_multiplier = max(1, len(grouping_specs))
        planned_plot_fe_types = [
            tuple(spec_def["fe_keys"])
            for _combo in combos
            for spec_def in drawable_specs
            for _group in range(grouping_multiplier)
        ]

    plot_progress = _PlotProgressEstimator(planned_plot_fe_types)
    html_bundle_payloads: list[dict[str, Any]] | None = (
        [] if args.export_format in {"html", "both"} else None
    )

    def on_plot_done(output_path: pathlib.Path, fe_type: tuple[str, ...], elapsed_seconds: float) -> None:
        print(plot_progress.update(fe_type, elapsed_seconds))

    def plot_stata_item(item: dict[str, Any]) -> None:
        meta_path = item["meta_path"]
        results_path = item["results_path"]
        output_path = item["output_path"]
        meta = rm_plot.load_plot_meta(meta_path)
        meta["elapsed_seconds_preplot"] = float(item["elapsed_seconds"])
        meta["export_format"] = args.export_format
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        _render_from_files(
            results_path=results_path,
            meta_path=meta_path,
            output_path=output_path,
            export_format=args.export_format,
            verbose=False,
            html_bundle_payloads=html_bundle_payloads,
        )
        _cleanup_plot_handoff(
            results_path=results_path,
            meta_path=meta_path,
            keep_temp=bool(args.keep_temp),
        )
        on_plot_done(output_path, tuple(item.get("fe_type", ())), float(item["elapsed_seconds"]))

    print(f"{_format_plot_progress(0, plot_progress.total)}  总导出数：{plot_progress.total}")

    if args.engine == "stata":
        import regression_monkey_stata as rm_stata

        stata_results = rm_stata.run_stata_engine(
            df=df,
            data_path=data_path.resolve(),
            args=args,
            controls_test=controls_test,
            controls_must=controls_must,
            controls_test_flat=controls_test_flat,
            controls_test_slots=controls_test_slots,
            controls_must_flat=controls_must_flat,
            controls_must_slots=controls_must_slots,
            grouping_variables=grouping_specs,
            matrix_controls=matrix_controls,
            matrix_alt_groups=matrix_alt_groups,
            spec_flags=spec_flags,
            run_output_dir=run_output_dir,
            on_item_ready=plot_stata_item,
        )
    else:
        stata_results = {}

    for idx, (y_var, x_var) in enumerate(combos, 1):
        print(f"\n{'#'*60}")
        print(f"[{idx}/{len(combos)}]  Y = {y_var}  ×  X = {x_var}")
        print("#" * 60)
        pair_t0 = perf_counter()

        if args.engine == "python":
            pair_rows, pair_total_specs = _run_python_pair(
                df=df,
                args=args,
                y_var=y_var,
                x_var=x_var,
                controls_test=controls_test,
                controls_must=controls_must,
                controls_test_flat=controls_test_flat,
                controls_must_flat=controls_must_flat,
                matrix_controls=matrix_controls,
                matrix_alt_groups=matrix_alt_groups,
                spec_flags=spec_flags,
                is_auto=is_auto,
                run_output_dir=run_output_dir,
                resolved_n_jobs=resolved_n_jobs,
                on_plot_done=on_plot_done,
                html_bundle_payloads=html_bundle_payloads,
            )
        else:
            pair_rows = []
            pair_total_specs = 0
            pair_summary_rows = []
            for item in stata_results.get((y_var, x_var), []):
                records = item["records"]
                pair_rows.extend(item["sig_rows"])
                if item.get("counts_as_base_spec", True):
                    pair_total_specs += len(records)
                    pair_summary_rows.extend(item.get("summary_sig_rows", item["sig_rows"]))
            summary_rows = pair_summary_rows

        all_sig_rows.extend(pair_rows)
        total_sig_specs += pair_total_specs
        if args.engine == "python":
            summary_rows = pair_rows
        combo_summaries.append({
            "y": y_var,
            "x": x_var,
            "n_specs": pair_total_specs,
            "n_sig": len(summary_rows),
            "star_counts": rm_py._sig_star_counts(summary_rows),
        })

    rm_py._export_sig_table(
        rows=all_sig_rows,
        output_path=str(run_output_dir / "sig.csv"),
        n_specs=total_sig_specs,
        print_summary=False,
    )
    for line in rm_py._format_combo_summary_lines(combo_summaries):
        print(line)
    if html_bundle_payloads is not None and html_bundle_payloads:
        bundle_path = run_output_dir / "interactive.html"
        rm_html.html_bundle_from_payloads(
            html_bundle_payloads,
            output_path=bundle_path,
        )
        print(f"[Saved] {bundle_path}")
    print(f"\n全部完成：{len(combos)} 个 y×x 组合")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt as exc:
        print("\n已中断。", file=sys.stderr)
        if exc.args:
            print(str(exc.args[0]), file=sys.stderr)
        sys.exit(130)
