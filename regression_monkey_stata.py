# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "numpy",
#   "pandas",
#   "matplotlib",
#   "pyreadstat",
#   "scipy",
#   "tomli >= 2.0 ; python_version < '3.11'",
# ]
# ///
"""
regression_monkey_stata.py
==========================
使用 Stata batch 模式 + reghdfe 执行规格曲线回归，并复用当前 Python 绘图逻辑。
"""

from __future__ import annotations

from datetime import datetime
import argparse
import itertools
import pathlib
import subprocess
import sys
from time import perf_counter
from typing import Any, cast

import pandas as pd

import regression_monkey as rm

try:
    import tomllib
except ImportError:
    import tomli as tomllib


def _stata_quote(text: str) -> str:
    return '"' + text.replace('"', '""') + '"'


def _load_toml_config(cli_args: list[str]) -> tuple[dict[str, Any], list[str]]:
    if cli_args and not cli_args[0].startswith("-"):
        config_path = pathlib.Path(cli_args[0]).expanduser().resolve()
        if not config_path.exists():
            raise FileNotFoundError(f"配置文件不存在：{config_path}")
        with config_path.open("rb") as f:
            return cast(dict[str, Any], tomllib.load(f)), cli_args[1:]

    default_cfg = pathlib.Path(__file__).with_name("regression_monkey_config.toml")
    if default_cfg.exists():
        with default_cfg.open("rb") as f:
            print(f"[配置] 加载默认配置：{default_cfg}")
            return cast(dict[str, Any], tomllib.load(f)), cli_args
    return {}, cli_args


def _load_dataframe(data_path: pathlib.Path) -> pd.DataFrame:
    suffix = data_path.suffix.lower()
    if suffix == ".dta":
        return cast(pd.DataFrame, pd.read_stata(data_path))
    if suffix == ".csv":
        return cast(pd.DataFrame, pd.read_csv(data_path))
    if suffix in (".parquet", ".pq"):
        return cast(pd.DataFrame, pd.read_parquet(data_path))
    raise ValueError(f"不支持的文件格式：{suffix}（支持 .dta / .csv / .parquet）")


def _ensure_stata_dta(
    df: pd.DataFrame,
    src_path: pathlib.Path,
    output_dir: pathlib.Path,
    stem_suffix: str = "stata_input",
) -> pathlib.Path:
    """
    Always materialize the dataframe used by Python into a Stata-readable .dta.

    Even when the original input is already a .dta, auto mode may have created
    derived columns such as `_spec_ind_year`; Stata batch must read the
    post-processed dataframe rather than the original source file.

    Important: do not pre-drop rows based on optional `controls_test` columns.
    Each specification must be estimated on the full available base sample and
    let Stata/reghdfe determine its own effective estimation sample and `e(N)`.
    """
    dta_path = output_dir / f"{src_path.stem}_{stem_suffix}.dta"
    df.to_stata(dta_path, write_index=False, version=118)
    print(f"[Stata] 已生成临时 dta：{dta_path}")
    return dta_path


def _prepare_auto_dataframe(
    df: pd.DataFrame,
    specs: dict[str, bool],
    firm_fe: str,
    ind_fe: str,
    time_fe: str,
    region_fe: str | None,
) -> tuple[pd.DataFrame, dict[str, str], dict[str, str]]:
    base_var_map: dict[str, str] = {"firm": firm_fe, "ind": ind_fe, "time": time_fe}
    if region_fe is not None:
        base_var_map["region"] = region_fe

    fmt = {
        "firm": firm_fe,
        "ind": ind_fe,
        "time": time_fe,
        "region": region_fe or "region",
    }
    return df, base_var_map, fmt


def _enumerate_control_specs(
    controls_must_slots: list[rm.ControlSlot],
    controls_test_slots: list[rm.ControlSlot],
) -> list[tuple[int, list[str], list[str], bool]]:
    """
    枚举所有合法规格：
    - controls_must: 每个槽位必须选一个
    - controls_test: 每个槽位可不选，若为替代组则至多选一个
    """
    subsets: list[tuple[int, list[str], list[str], bool]] = []
    total_specs = rm._spec_count_from_slots(controls_must_slots, controls_test_slots)
    for bits in range(total_specs):
        rem, chosen_must_cols, chosen_must = rm._decode_required_choice(bits, controls_must_slots)
        chosen_test_cols, chosen_test, is_full = rm._decode_optional_choice(rem, controls_test_slots)
        _ = chosen_must_cols, chosen_test_cols
        subsets.append((bits, chosen_must, chosen_test, is_full))
    return subsets


def _spec_absorb_and_vce(spec_def: dict[str, Any], var_map: dict[str, str]) -> tuple[str, str]:
    def _stata_fe_term(key: str) -> str:
        if key == "firm":
            return f"i.{var_map['firm']}"
        if key == "ind":
            return f"i.{var_map['ind']}"
        if key == "time":
            return f"i.{var_map['time']}"
        if key == "region":
            return f"i.{var_map['region']}"
        if key == "_ind_time":
            return f"i.{var_map['time']}#i.{var_map['ind']}"
        if key == "_region_time":
            return f"i.{var_map['time']}#i.{var_map['region']}"
        raise ValueError(f"unknown FE key: {key}")

    absorb_expr = " ".join(_stata_fe_term(k) for k in spec_def["fe_keys"])
    clust_cols = [var_map[k] for k in spec_def["cl_keys"]]
    if spec_def["vce"] == "robust":
        vce = "vce(robust)"
    elif len(clust_cols) == 1:
        vce = f"vce(cluster {clust_cols[0]})"
    elif len(clust_cols) == 2:
        vce = f"vce(cluster {' '.join(clust_cols)})"
    else:
        raise ValueError(f"unsupported vce setting: {spec_def}")
    return absorb_expr, vce


def _spec_fe_labels(spec_def: dict[str, Any], var_map: dict[str, str]) -> list[str]:
    """Return human-readable FE labels aligned with the Stata absorb() spec."""
    labels: list[str] = []
    for key in spec_def["fe_keys"]:
        if key == "firm":
            labels.append(var_map["firm"])
        elif key == "ind":
            labels.append(var_map["ind"])
        elif key == "time":
            labels.append(var_map["time"])
        elif key == "region":
            labels.append(var_map["region"])
        elif key == "_ind_time":
            labels.append(f"{var_map['time']}#{var_map['ind']}")
        elif key == "_region_time":
            labels.append(f"{var_map['time']}#{var_map['region']}")
        else:
            raise ValueError(f"unknown FE key: {key}")
    return labels


def _write_reghdfe_do(
    do_path: pathlib.Path,
    log_path: pathlib.Path,
    data_path: pathlib.Path,
    results_dta: pathlib.Path,
    y: str,
    x: str,
    controls_must: list[str],
    controls_must_slots: list[rm.ControlSlot],
    controls_test: list[str],
    controls_test_slots: list[rm.ControlSlot],
    spec_def: dict[str, Any],
    var_map: dict[str, str],
) -> None:
    absorb_expr, vce = _spec_absorb_and_vce(spec_def, var_map)
    subsets = _enumerate_control_specs(controls_must_slots, controls_test_slots)
    lines = [
        "version 18.0",
        "clear all",
        "set more off",
        f'log using {_stata_quote(str(log_path))}, replace text',
        "capture which reghdfe",
        "if _rc {",
        '    di as error "reghdfe not installed in this Stata environment."',
        "    exit 199",
        "}",
        f"use {_stata_quote(str(data_path))}, clear",
        f"tempname posth",
        f'postfile `posth\' str128 spec_name long bits str2045 chosen_must_controls str2045 chosen_test_controls double coef double se double obs double df_resid using {_stata_quote(str(results_dta))} , replace',
    ]

    for bits, chosen_must, chosen_test, _is_full in subsets:
        rhs_terms = [x]
        if chosen_must:
            rhs_terms.extend(chosen_must)
        if chosen_test:
            rhs_terms.extend(chosen_test)
        rhs = " ".join(rhs_terms)
        chosen_must_txt = "|".join(chosen_must)
        chosen_test_txt = "|".join(chosen_test)
        spec_name = spec_def["name"]
        lines.extend([
            f"* Run each spec on the full dataset currently in memory; do not pre-filter",
            f"* by optional controls_test missingness outside reghdfe. Let e(sample)/e(N)",
            f"* be determined by this exact RHS + absorb() combination.",
            f"capture reghdfe {y} {rhs}, absorb({absorb_expr}) {vce}",
            "if _rc == 0 {",
            f"    scalar __b = _b[{x}]",
            f"    scalar __se = _se[{x}]",
            "    scalar __N = e(N)",
            "    scalar __df = e(df_r)",
            f'    post `posth\' ({_stata_quote(spec_name)}) ({bits}) ({_stata_quote(chosen_must_txt)}) ({_stata_quote(chosen_test_txt)}) (__b) (__se) (__N) (__df)',
            "}",
        ])

    lines.extend([
        "postclose `posth'",
        "log close",
        "exit 0",
    ])
    do_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _run_stata_do(stata_path: str, do_path: pathlib.Path, cwd: pathlib.Path) -> None:
    cmd = [stata_path, "-b", "do", str(do_path)]
    subprocess.run(cmd, cwd=cwd, check=True)


def _safe_unlink(path: pathlib.Path) -> None:
    """Best-effort delete for temporary files."""
    try:
        path.unlink()
    except FileNotFoundError:
        return


def _records_from_stata_dta(
    dta_path: pathlib.Path,
    controls_must_slots: list[rm.ControlSlot],
    controls_test_slots: list[rm.ControlSlot],
) -> list[rm.SpecRecord]:
    df_res = cast(pd.DataFrame, pd.read_stata(dta_path))
    records: list[rm.SpecRecord] = []
    for _, row in df_res.iterrows():
        coef = float(row["coef"])
        se = float(row["se"])
        obs = int(round(float(row["obs"])))
        df_resid = max(1, int(round(float(row["df_resid"]))))
        chosen_must = [c for c in str(row["chosen_must_controls"]).split("|") if c]
        chosen_test = [c for c in str(row["chosen_test_controls"]).split("|") if c]
        chosen_test_set = set(chosen_test)
        chosen_all_set = set(chosen_must) | chosen_test_set
        rem_bits, _, _ = rm._decode_required_choice(int(row["bits"]), controls_must_slots)
        _, _, is_full = rm._decode_optional_choice(rem_bits, controls_test_slots)
        t_value = coef / se
        p_value = rm._p_value_from_t(abs(t_value), df_resid)
        crit99, crit95, crit90 = rm._crit_values(df_resid)
        records.append({
            "coef": coef,
            "se": se,
            "t_value": t_value,
            "p_value": p_value,
            "df_resid": df_resid,
            "ci99_lo": coef - crit99 * se,
            "ci99_hi": coef + crit99 * se,
            "ci95_lo": coef - crit95 * se,
            "ci95_hi": coef + crit95 * se,
            "ci90_lo": coef - crit90 * se,
            "ci90_hi": coef + crit90 * se,
            "controls_test": chosen_test_set,
            "controls_all": chosen_all_set,
            "is_full": is_full,
            "obs": obs,
        })
    records.sort(key=lambda r: r["coef"])
    return records


def main() -> None:
    cfg, cli_args = _load_toml_config(sys.argv[1:])
    parser = argparse.ArgumentParser(
        prog="regression_monkey_stata",
        description="Use Stata batch mode + reghdfe to run specification-curve regressions.",
    )
    parser.add_argument("--data", metavar="FILE")
    parser.add_argument("--y", metavar="VAR", nargs="+")
    parser.add_argument("--x", metavar="VAR", nargs="+")
    parser.add_argument("--controls", metavar="VAR", nargs="+", help="compat alias for --controls-test")
    parser.add_argument("--controls-test", dest="controls_test", metavar="VAR", nargs="+")
    parser.add_argument("--controls-must", dest="controls_must", metavar="VAR", nargs="+")
    parser.add_argument("--Firm-FE", dest="firm_fe", default="code", metavar="COL")
    parser.add_argument("--Ind-FE", dest="ind_fe", default="ind", metavar="COL")
    parser.add_argument("--Time-FE", dest="time_fe", default="year", metavar="COL")
    parser.add_argument("--Region-FE", dest="region_fe", default=None, metavar="COL")
    parser.add_argument("--output", default="outputs", metavar="DIR")
    parser.add_argument("--dpi", default=150, type=int)
    parser.add_argument("--fig-width", default=14.0, type=float, metavar="INCHES")
    parser.add_argument("--stata-path", default="stata-mp", metavar="EXE")
    parser.add_argument("--keep-temp", action="store_true", help="保留 .do / .log / 中间 Stata 结果文件")
    for spec_name in rm._ALL_SPEC_NAMES:
        parser.add_argument(f"--{spec_name.replace('_', '-')}", dest=spec_name, action="store_true")

    if cfg:
        allowed = {
            "data", "y", "x", "controls", "controls_test", "controls_must",
            "output", "dpi", "fig_width", "firm_fe", "ind_fe", "time_fe",
            "region_fe", "stata_path",
        } | set(rm._ALL_SPEC_NAMES)
        normalized = {k.lower(): v for k, v in cfg.items()}
        parser.set_defaults(**{k: v for k, v in normalized.items() if k in allowed})

    args = parser.parse_args(cli_args)
    controls_test = list(args.controls_test) if args.controls_test else (list(args.controls) if args.controls else [])
    controls_must = list(args.controls_must) if args.controls_must else []
    try:
        controls_test_flat, controls_test_slots = rm._normalize_controls_test(controls_test)
        controls_must_flat, controls_must_slots = rm._normalize_controls_must(controls_must)
    except ValueError as exc:
        parser.error(str(exc))
    varying_must_controls = rm._varying_must_controls(controls_must_slots)
    matrix_controls = varying_must_controls + controls_test_flat
    show_special_markers = not varying_must_controls
    if not args.data or not args.y or not args.x:
        parser.error("必须提供 data / y / x（可通过 TOML 或 CLI 指定）")
    if not controls_test and not controls_must:
        parser.error("至少提供一类控制变量：controls_test/controls 或 controls_must")

    spec_flags = {name: getattr(args, name, False) for name in rm._ALL_SPEC_NAMES}
    if not any(spec_flags.values()):
        parser.error("当前 Stata 脚本仅支持自动规格模式，请至少启用一个 absorb_* flag。")

    data_path = pathlib.Path(args.data).expanduser().resolve()
    print(f"读取数据：{data_path}")
    df = _load_dataframe(data_path)
    print(f"数据读取完成：{len(df):,} 行 × {len(df.columns)} 列")

    df, var_map, fmt = _prepare_auto_dataframe(
        df=df,
        specs=spec_flags,
        firm_fe=args.firm_fe,
        ind_fe=args.ind_fe,
        time_fe=args.time_fe,
        region_fe=args.region_fe,
    )

    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = pathlib.Path(args.output).expanduser().resolve()
    if output_root.suffix:
        output_root = output_root.parent
    run_output_dir = output_root / run_timestamp
    run_output_dir.mkdir(parents=True, exist_ok=True)
    print(f"输出目录：{run_output_dir}")

    dta_path = _ensure_stata_dta(df, data_path, run_output_dir)
    snapshot_config = {
        "generated_at": run_timestamp,
        "engine": "stata-reghdfe",
        "stata_path": args.stata_path,
        "data": str(args.data),
        "y": list(args.y),
        "x": list(args.x),
        "controls_test": controls_test,
        "controls_test_flat": controls_test_flat,
        "controls_must": controls_must,
        "controls_must_flat": controls_must_flat,
        "output": str(output_root),
        "run_output_dir": str(run_output_dir),
        "dpi": args.dpi,
        "fig_width": args.fig_width,
        "firm_fe": args.firm_fe,
        "ind_fe": args.ind_fe,
        "time_fe": args.time_fe,
    }
    if args.region_fe:
        snapshot_config["region_fe"] = args.region_fe
    snapshot_config.update({name: enabled for name, enabled in spec_flags.items() if enabled})
    rm._write_config_snapshot(snapshot_config, run_output_dir / "config_snapshot.toml")

    combos = list(itertools.product(args.y, args.x))
    all_sig_rows: list[dict[str, Any]] = []
    total_sig_specs = 0

    for idx, (y_var, x_var) in enumerate(combos, 1):
        print(f"\n{'#'*60}")
        print(f"[{idx}/{len(combos)}]  Y = {y_var}  ×  X = {x_var}")
        print("#" * 60)
        for spec_def in rm._SPEC_CATALOG:
            spec_t0 = perf_counter()
            spec_name = spec_def["name"]
            if not spec_flags.get(spec_name, False):
                continue
            if spec_def["needs_region"] and args.region_fe is None:
                print(f"[跳过] {spec_name}：需要 Region_FE 但未指定")
                continue

            do_path = run_output_dir / f"{y_var}_{x_var}_{spec_def['tag']}.do"
            log_path = run_output_dir / f"{y_var}_{x_var}_{spec_def['tag']}.log"
            dta_result_path = run_output_dir / f"{y_var}_{x_var}_{spec_def['tag']}_stata_results.dta"
            out_png = run_output_dir / f"{y_var}_{x_var}_{spec_def['tag']}.png"
            title_suffix = spec_def["help"].format(**fmt)

            print(f"[Stata] 运行规格：{spec_name}")
            _write_reghdfe_do(
                do_path=do_path,
                log_path=log_path,
                data_path=dta_path,
                results_dta=dta_result_path,
                y=y_var,
                x=x_var,
                controls_must=controls_must_flat,
                controls_must_slots=controls_must_slots,
                controls_test=controls_test_flat,
                controls_test_slots=controls_test_slots,
                spec_def=spec_def,
                var_map=var_map,
            )
            _run_stata_do(args.stata_path, do_path, run_output_dir)
            records = _records_from_stata_dta(
                dta_result_path,
                controls_must_slots=controls_must_slots,
                controls_test_slots=controls_test_slots,
            )
            if not records:
                print(f"[Stata] {spec_name} 未返回有效回归结果")
                print(f"[Stata] 已保留调试文件：{do_path.name}, {log_path.name}, {dta_result_path.name}")
                continue

            rm._plot(
                records=records,
                y_name=y_var,
                x_name=x_var,
                controls_test=controls_test_flat,
                controls_must=controls_must_flat,
                matrix_controls=matrix_controls,
                show_special_markers=show_special_markers,
                fig_width=args.fig_width,
                dpi=args.dpi,
                output_path=str(out_png),
                title_suffix=title_suffix,
                elapsed_seconds_preplot=perf_counter() - spec_t0,
            )
            fe_cols = _spec_fe_labels(spec_def, var_map)
            clust_cols = [var_map[k] for k in spec_def["cl_keys"]]
            all_sig_rows.extend(
                rm._build_sig_rows(
                    records=records,
                    y=y_var,
                    x=x_var,
                    controls_must=controls_must_flat,
                    controls_test=controls_test_flat,
                    fe_cols=fe_cols,
                    clust_cols=clust_cols,
                    vce_label="robust" if spec_def["vce"] == "robust" else None,
                )
            )
            total_sig_specs += len(records)

            if not args.keep_temp:
                _safe_unlink(do_path)
                _safe_unlink(log_path)
                _safe_unlink(dta_result_path)

    rm._export_sig_table(
        rows=all_sig_rows,
        output_path=str(run_output_dir / "sig.csv"),
        n_specs=total_sig_specs,
    )
    if not args.keep_temp and dta_path != data_path:
        _safe_unlink(dta_path)


if __name__ == "__main__":
    main()
