"""
regression_monkey · stata
==========================
使用 Stata batch 模式 + reghdfe 执行规格曲线回归，并复用当前 Python 绘图逻辑。
"""

from __future__ import annotations

from datetime import datetime
import argparse
import itertools
import os
import pathlib
import signal
import subprocess
import sys
from time import perf_counter
from typing import Any, Callable, cast

import pandas as pd

from . import common as rm_common
from . import py as rm


GroupingSpec = tuple[str, str]


def _stata_quote(text: str) -> str:
    return '"' + text.replace('"', '""') + '"'


def _ensure_stata_dta(
    df: pd.DataFrame,
    src_path: pathlib.Path,
    output_dir: pathlib.Path,
    stem_suffix: str = "stata_input",
    verbose: bool = True,
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
    if verbose:
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


def _safe_path_part(value: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in value)
    cleaned = "_".join(part for part in cleaned.split("_") if part)
    return cleaned or "spec"


def _dynamic_group_var_name(grouping_variable: str) -> str:
    return f"b_{grouping_variable}"


def _grouping_scope_suffix(scope: str) -> str:
    if scope == "by_ind_time":
        return "by_ind_time"
    if scope == "by_time":
        return "by_time"
    if scope == "by_none":
        return "by_none"
    raise ValueError(f"unknown grouping scope: {scope}")


def _grouping_display_name(grouping_variable: str, grouping_scope: str) -> str:
    return f"{grouping_variable}[{_grouping_scope_suffix(grouping_scope)}]"


def _grouping_path_part(grouping_variable: str, grouping_scope: str) -> str:
    return f"{_safe_path_part(grouping_variable)}_{_grouping_scope_suffix(grouping_scope)}"


def _grouping_quantiles_line(grouping_variable: str, grouping_scope: str, var_map: dict[str, str]) -> list[str]:
    if grouping_scope == "by_ind_time":
        return [
            f"    bysort {var_map['ind']} {var_map['time']} ({var_map['firm']}): quantiles {grouping_variable}, gen(_temp) n(2) stable",
        ]
    if grouping_scope == "by_time":
        return [
            f"    bysort {var_map['time']} ({var_map['firm']}): quantiles {grouping_variable}, gen(_temp) n(2) stable",
        ]
    if grouping_scope == "by_none":
        return [
            f"    sort {var_map['firm']}",
            f"    quantiles {grouping_variable}, gen(_temp) n(2) stable",
        ]
    raise ValueError(f"unknown grouping scope: {grouping_scope}")


def _control_spec_count(
    controls_must_slots: list[rm.ControlSlot],
    controls_test_slots: list[rm.ControlSlot],
) -> int:
    return rm._spec_count_from_slots(controls_must_slots, controls_test_slots)


def _append_control_stats_lines(lines: list[str], chosen_controls: list[str], *, indent: str = "    ") -> None:
    if not chosen_controls:
        lines.append(f'{indent}local __ctrl_stats ""')
        return
    controls_macro = " ".join(chosen_controls)
    lines.extend([
        f'{indent}local __rm_controls "{controls_macro}"',
        f'{indent}local __ctrl_stats ""',
        f'{indent}local __ctrl_sep ""',
        f'{indent}foreach __rm_c of local __rm_controls {{',
        f'{indent}    capture scalar __ctrl_b = _b[`__rm_c\']',
        f'{indent}    if _rc == 0 {{',
        f'{indent}        capture scalar __ctrl_se = _se[`__rm_c\']',
        f'{indent}        if _rc == 0 & __ctrl_se < . & __ctrl_se != 0 {{',
        f'{indent}            scalar __ctrl_t = __ctrl_b / __ctrl_se',
        f'{indent}            scalar __ctrl_p = 2 * ttail(e(df_r), abs(__ctrl_t))',
        f'{indent}            local __ctrl_piece = "`__rm_c\'=" + string(__ctrl_b, "%21.15g") + "," + string(__ctrl_se, "%21.15g") + "," + string(__ctrl_t, "%21.15g") + "," + string(__ctrl_p, "%21.15g")',
        f'{indent}            local __ctrl_stats "`__ctrl_stats\'`__ctrl_sep\'`__ctrl_piece\'"',
        f'{indent}            local __ctrl_sep ";"',
        f'{indent}        }}',
        f'{indent}    }}',
        f'{indent}}}',
    ])


def _plot_output_path(run_output_dir: pathlib.Path, group_name: str, filename: str) -> pathlib.Path:
    output_dir = run_output_dir / _safe_path_part(group_name)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / filename


def _render_output_path(run_output_dir: pathlib.Path, group_name: str, filename: str, export_format: str) -> pathlib.Path:
    if export_format == "html":
        return run_output_dir / filename
    return _plot_output_path(run_output_dir, group_name, filename)


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
    grouping_variable: str | None = None,
    grouping_scope: str = "by_ind_time",
) -> None:
    absorb_expr, vce = _spec_absorb_and_vce(spec_def, var_map)
    subsets = _enumerate_control_specs(controls_must_slots, controls_test_slots)
    _ = log_path
    lines = [
        "version 18.0",
        "clear all",
        "set more off",
        "capture which reghdfe",
        "if _rc {",
        '    di as error "reghdfe not installed in this Stata environment."',
        "    exit 199",
        "}",
    ]
    if grouping_variable is not None:
        lines.extend([
            "capture which quantiles",
            "if _rc {",
            '    di as error "quantiles not installed in this Stata environment."',
            "    exit 199",
            "}",
        ])
    lines.extend([
        f"use {_stata_quote(str(data_path))}, clear",
        f"tempname posth",
        f'postfile `posth\' str128 spec_name long bits str2045 chosen_must_controls str2045 chosen_test_controls str128 grouping_variable double group_value double coef double se double obs double df_resid double adj_r2 double within_r2 double f_stat str2045 control_stats using {_stata_quote(str(results_dta))} , replace',
    ])

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
        if grouping_variable is None:
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
                "    scalar __adj_r2 = e(r2_a)",
                "    scalar __within_r2 = e(r2_within)",
                "    scalar __f_stat = e(F)",
            ])
            _append_control_stats_lines(lines, chosen_must + chosen_test)
            lines.extend([
                f'    post `posth\' ({_stata_quote(spec_name)}) ({bits}) ({_stata_quote(chosen_must_txt)}) ({_stata_quote(chosen_test_txt)}) ("") (.) (__b) (__se) (__N) (__df) (__adj_r2) (__within_r2) (__f_stat) ("`__ctrl_stats\'")',
                "}",
            ])
        else:
            b_group = _dynamic_group_var_name(grouping_variable)
            group_name_literal = _stata_quote(_grouping_display_name(grouping_variable, grouping_scope))
            quantiles_lines = _grouping_quantiles_line(grouping_variable, grouping_scope, var_map)
            lines.extend([
                f"* First run the all-sample spec to capture its exact e(sample).",
                f"capture reghdfe {y} {rhs}, absorb({absorb_expr}) {vce}",
                "if _rc == 0 {",
                "    tempvar __rm_esample",
                "    gen byte `__rm_esample' = e(sample)",
                "    preserve",
                "    keep if `__rm_esample'",
                f"    capture drop {b_group}",
                "    capture drop _temp",
                *quantiles_lines,
                f"    gen byte {b_group} = _temp - 1",
                "    drop _temp",
                "    forvalues __rm_g = 0/1 {",
                f"        capture reghdfe {y} {rhs} if {b_group} == `__rm_g', absorb({absorb_expr}) {vce}",
                "        if _rc == 0 {",
                f"            scalar __b = _b[{x}]",
                f"            scalar __se = _se[{x}]",
                "            scalar __N = e(N)",
                "            scalar __df = e(df_r)",
                "            scalar __adj_r2 = e(r2_a)",
                "            scalar __within_r2 = e(r2_within)",
                "            scalar __f_stat = e(F)",
            ])
            _append_control_stats_lines(lines, chosen_must + chosen_test, indent="            ")
            lines.extend([
                f'            post `posth\' ({_stata_quote(spec_name)}) ({bits}) ({_stata_quote(chosen_must_txt)}) ({_stata_quote(chosen_test_txt)}) ({group_name_literal}) (`__rm_g\') (__b) (__se) (__N) (__df) (__adj_r2) (__within_r2) (__f_stat) ("`__ctrl_stats\'")',
                "        }",
                "    }",
                f"    drop {b_group}",
                "    restore",
                "}",
            ])

    lines.extend([
        "postclose `posth'",
        "exit 0",
    ])
    do_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_interaction_reghdfe_do(
    do_path: pathlib.Path,
    log_path: pathlib.Path,
    data_path: pathlib.Path,
    results_dta: pathlib.Path,
    y: str,
    x: str,
    z: str,
    controls_must: list[str],
    controls_must_slots: list[rm.ControlSlot],
    controls_test: list[str],
    controls_test_slots: list[rm.ControlSlot],
    spec_def: dict[str, Any],
    var_map: dict[str, str],
) -> None:
    absorb_expr, vce = _spec_absorb_and_vce(spec_def, var_map)
    subsets = _enumerate_control_specs(controls_must_slots, controls_test_slots)
    _ = log_path, controls_must, controls_test
    interaction_term = f"c.{x}#c.{z}"
    lines = [
        "version 18.0",
        "clear all",
        "set more off",
        "capture which reghdfe",
        "if _rc {",
        '    di as error "reghdfe not installed in this Stata environment."',
        "    exit 199",
        "}",
        f"use {_stata_quote(str(data_path))}, clear",
        "tempname posth",
        f'postfile `posth\' str128 spec_name long bits str2045 chosen_must_controls str2045 chosen_test_controls str128 grouping_variable double group_value double coef double se double obs double df_resid double adj_r2 double within_r2 double f_stat str2045 control_stats using {_stata_quote(str(results_dta))} , replace',
    ]
    for bits, chosen_must, chosen_test, _is_full in subsets:
        rhs_terms = [x]
        rhs_terms.extend(chosen_must)
        rhs_terms.extend(chosen_test)
        rhs_terms.append(interaction_term)
        rhs = " ".join(rhs_terms)
        chosen_must_txt = "|".join(chosen_must)
        chosen_test_txt = "|".join(chosen_test)
        spec_name = spec_def["name"]
        lines.extend([
            f"capture reghdfe {y} {rhs}, absorb({absorb_expr}) {vce}",
            "if _rc == 0 {",
            f"    scalar __b = _b[{interaction_term}]",
            f"    scalar __se = _se[{interaction_term}]",
            "    scalar __N = e(N)",
            "    scalar __df = e(df_r)",
            "    scalar __adj_r2 = e(r2_a)",
            "    scalar __within_r2 = e(r2_within)",
            "    scalar __f_stat = e(F)",
        ])
        _append_control_stats_lines(lines, chosen_must + chosen_test)
        lines.extend([
            f'    post `posth\' ({_stata_quote(spec_name)}) ({bits}) ({_stata_quote(chosen_must_txt)}) ({_stata_quote(chosen_test_txt)}) ("") (.) (__b) (__se) (__N) (__df) (__adj_r2) (__within_r2) (__f_stat) ("`__ctrl_stats\'")',
            "}",
        ])
    lines.extend([
        "postclose `posth'",
        "exit 0",
    ])
    do_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _run_stata_do(
    stata_path: str,
    do_path: pathlib.Path,
    log_path: pathlib.Path,
    cwd: pathlib.Path,
    verbose: bool = True,
) -> None:
    cmd = [stata_path, "-b", "do", do_path.name]
    try:
        proc = subprocess.Popen(cmd, cwd=cwd, start_new_session=True)
    except FileNotFoundError as exc:
        raise RuntimeError(
            f"Stata executable not found: {stata_path}\n"
            "Use --stata-path or set stata_path in the TOML config."
        ) from exc
    if verbose:
        print(f"[Stata] PID={proc.pid}  do={do_path.name}  log={log_path.name}")
    try:
        returncode = proc.wait()
    except KeyboardInterrupt as exc:
        if proc.poll() is None:
            try:
                os.killpg(proc.pid, signal.SIGTERM)
                proc.wait(timeout=10)
            except ProcessLookupError:
                pass
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
        log_tail = _tail_text(log_path)
        raise KeyboardInterrupt(
            "Stata batch run was interrupted.\n"
            f"Do file: {do_path.resolve()}\n"
            f"Log file: {log_path.resolve()}\n"
            "Stata log tail:\n"
            f"{log_tail}"
        ) from exc
    if returncode != 0:
        log_tail = _tail_text(log_path)
        raise RuntimeError(
            f"Stata exited with status {returncode}.\n"
            f"Do file: {do_path.resolve()}\n"
            f"Log file: {log_path.resolve()}\n"
            "Stata log tail:\n"
            f"{log_tail}"
        )


def _tail_text(path: pathlib.Path, max_lines: int = 80) -> str:
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except FileNotFoundError:
        return "(log file not found)"
    return "\n".join(lines[-max_lines:])


def _ensure_stata_result_exists(
    *,
    results_dta: pathlib.Path,
    do_path: pathlib.Path,
    log_path: pathlib.Path,
) -> None:
    if results_dta.exists():
        return
    log_tail = _tail_text(log_path)
    raise RuntimeError(
        "Stata did not create the expected result file.\n"
        f"Expected result: {results_dta.resolve()}\n"
        f"Do file: {do_path.resolve()}\n"
        f"Log file: {log_path.resolve()}\n"
        "Stata log tail:\n"
        f"{log_tail}"
    )


def _raise_empty_stata_records(
    *,
    spec_display: str,
    results_dta: pathlib.Path,
    do_path: pathlib.Path,
    log_path: pathlib.Path,
) -> None:
    log_tail = _tail_text(log_path)
    raise RuntimeError(
        "Stata returned no valid regression results.\n"
        f"Spec: {spec_display}\n"
        f"Result file: {results_dta.resolve()}\n"
        f"Do file: {do_path.resolve()}\n"
        f"Log file: {log_path.resolve()}\n"
        "Stata log tail:\n"
        f"{log_tail}"
    )


def _parse_stata_control_stats(value: Any) -> list[dict[str, Any]]:
    if value is None or pd.isna(value):
        return []
    text = str(value).strip()
    if not text:
        return []
    stats: list[dict[str, Any]] = []
    for piece in text.split(";"):
        if not piece or "=" not in piece:
            continue
        name, raw_values = piece.split("=", 1)
        parts = raw_values.split(",")
        if len(parts) != 4:
            continue
        try:
            coef, se, t_value, p_value = (float(part) for part in parts)
        except ValueError:
            continue
        stats.append({
            "name": name,
            "coef": coef,
            "se": se,
            "t_value": t_value,
            "p_value": p_value,
        })
    return stats


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
        adj_r2 = float(row["adj_r2"]) if "adj_r2" in row.index and not pd.isna(row["adj_r2"]) else float("nan")
        within_r2 = float(row["within_r2"]) if "within_r2" in row.index and not pd.isna(row["within_r2"]) else float("nan")
        f_stat = float(row["f_stat"]) if "f_stat" in row.index and not pd.isna(row["f_stat"]) else float("nan")
        crit99, crit95, crit90 = rm._crit_values(df_resid)
        records.append({
            "coef": coef,
            "se": se,
            "t_value": t_value,
            "p_value": p_value,
            "adj_r2": adj_r2,
            "within_r2": within_r2,
            "f_stat": f_stat,
            "df_resid": df_resid,
            "ci99_lo": coef - crit99 * se,
            "ci99_hi": coef + crit99 * se,
            "ci95_lo": coef - crit95 * se,
            "ci95_hi": coef + crit95 * se,
            "ci90_lo": coef - crit90 * se,
            "ci90_hi": coef + crit90 * se,
            "controls_test": chosen_test_set,
            "controls_all": chosen_all_set,
            "control_stats": _parse_stata_control_stats(row.get("control_stats")),
            "is_full": is_full,
            "obs": obs,
        })
    records.sort(key=lambda r: r["coef"])
    return records


def _grouped_records_from_stata_dta(
    dta_path: pathlib.Path,
    controls_must_slots: list[rm.ControlSlot],
    controls_test_slots: list[rm.ControlSlot],
) -> list[rm.GroupedPlotRecord]:
    df_res = cast(pd.DataFrame, pd.read_stata(dta_path))
    grouped_records: list[rm.GroupedPlotRecord] = []
    for _, row in df_res.iterrows():
        group_name = str(row.get("grouping_variable", "") or "")
        group_value_raw = row.get("group_value")
        if not group_name or pd.isna(group_value_raw):
            continue
        coef = float(row["coef"])
        se = float(row["se"])
        obs = int(round(float(row["obs"])))
        df_resid = max(1, int(round(float(row["df_resid"]))))
        chosen_must = [c for c in str(row["chosen_must_controls"]).split("|") if c]
        chosen_test = [c for c in str(row["chosen_test_controls"]).split("|") if c]
        chosen_all = sorted(set(chosen_must) | set(chosen_test))
        rem_bits, _, _ = rm._decode_required_choice(int(row["bits"]), controls_must_slots)
        _, _, _is_full = rm._decode_optional_choice(rem_bits, controls_test_slots)
        t_value = coef / se
        p_value = rm._p_value_from_t(abs(t_value), df_resid)
        _crit99, crit95, _crit90 = rm._crit_values(df_resid)
        grouped_records.append({
            "grouping_variable": group_name,
            "group_value": int(round(float(group_value_raw))),
            "coef": coef,
            "p_value": p_value,
            "ci99_lo": coef - _crit99 * se,
            "ci99_hi": coef + _crit99 * se,
            "ci95_lo": coef - crit95 * se,
            "ci95_hi": coef + crit95 * se,
            "ci90_lo": coef - _crit90 * se,
            "ci90_hi": coef + _crit90 * se,
            "obs": obs,
            "controls_all": chosen_all,
        })
    return grouped_records


def run_stata_engine(
    *,
    df: pd.DataFrame,
    data_path: pathlib.Path,
    args: argparse.Namespace,
    controls_test: rm.ControlSpecInput,
    controls_must: rm.ControlSpecInput,
    controls_test_flat: list[str],
    controls_test_slots: list[rm.ControlSlot],
    controls_must_flat: list[str],
    controls_must_slots: list[rm.ControlSlot],
    grouping_variables: list[GroupingSpec],
    matrix_controls: list[str],
    matrix_alt_groups: list[dict[str, Any]] | None = None,
    spec_flags: dict[str, bool],
    run_output_dir: pathlib.Path,
    on_item_ready: Callable[[dict[str, Any]], None] | None = None,
) -> dict[tuple[str, str], list[dict[str, Any]]]:
    _ = controls_test, controls_must
    df, var_map, fmt = _prepare_auto_dataframe(
        df=df,
        specs=spec_flags,
        firm_fe=args.firm_fe,
        ind_fe=args.ind_fe,
        time_fe=args.time_fe,
        region_fe=args.region_fe,
    )
    dta_path = _ensure_stata_dta(df, data_path, run_output_dir, verbose=False)
    outputs: dict[tuple[str, str], list[dict[str, Any]]] = {}

    for y_var, x_var in itertools.product(args.y, args.x):
        pair_items: list[dict[str, Any]] = []
        for spec_def in rm._SPEC_CATALOG:
            spec_name = spec_def["name"]
            spec_display = rm._format_spec_display(spec_def, fmt)
            if not spec_flags.get(spec_name, False):
                continue
            if spec_def["needs_region"] and args.region_fe is None:
                print(f"[跳过] {spec_display}：需要 Region_FE 但未指定")
                continue

            do_path = run_output_dir / f"{y_var}_{x_var}_{spec_def['tag']}.do"
            log_path = run_output_dir / f"{y_var}_{x_var}_{spec_def['tag']}.log"
            dta_result_path = run_output_dir / f"{y_var}_{x_var}_{spec_def['tag']}_stata_results.dta"
            results_path = run_output_dir / f"{y_var}_{x_var}_{spec_def['tag']}_results.csv"
            title_suffix = spec_def["help"].format(**fmt)
            base_regression_count = _control_spec_count(controls_must_slots, controls_test_slots)

            print(f"[Stata] 运行规格：{spec_display}")
            if grouping_variables:
                print(f"[Stata] 预先运行 all 样本规格：{base_regression_count:,} 个回归")
            else:
                print(rm._format_plot_regression_count(base_regression_count))
            spec_t0 = perf_counter()
            _write_reghdfe_do(
                do_path=do_path.resolve(),
                log_path=log_path.resolve(),
                data_path=dta_path.resolve(),
                results_dta=dta_result_path.resolve(),
                y=y_var,
                x=x_var,
                controls_must=controls_must_flat,
                controls_must_slots=controls_must_slots,
                controls_test=controls_test_flat,
                controls_test_slots=controls_test_slots,
                spec_def=spec_def,
                var_map=var_map,
                grouping_variable=None,
            )
            _run_stata_do(args.stata_path, do_path, log_path, run_output_dir, verbose=False)
            _ensure_stata_result_exists(
                results_dta=dta_result_path,
                do_path=do_path,
                log_path=log_path,
            )
            records = _records_from_stata_dta(
                dta_result_path,
                controls_must_slots=controls_must_slots,
                controls_test_slots=controls_test_slots,
            )
            if not records:
                _raise_empty_stata_records(
                    spec_display=spec_display,
                    results_dta=dta_result_path,
                    do_path=do_path,
                    log_path=log_path,
                )

            fe_cols = _spec_fe_labels(spec_def, var_map)
            clust_cols = [var_map[k] for k in spec_def["cl_keys"]]
            base_sig_rows = rm._build_sig_rows(
                records=records,
                y=y_var,
                x=x_var,
                controls_must=controls_must_flat,
                controls_test=controls_test_flat,
                fe_cols=fe_cols,
                clust_cols=clust_cols,
                vce_label="robust" if spec_def["vce"] == "robust" else None,
            )
            if grouping_variables:
                for grouping_idx, (grouping_scope, grouping_variable) in enumerate(grouping_variables):
                    grouping_path_part = _grouping_path_part(grouping_variable, grouping_scope)
                    grouping_display_name = _grouping_display_name(grouping_variable, grouping_scope)
                    grouped_do_path = run_output_dir / f"{y_var}_{x_var}_{spec_def['tag']}_{grouping_path_part}_grouped.do"
                    grouped_log_path = run_output_dir / f"{y_var}_{x_var}_{spec_def['tag']}_{grouping_path_part}_grouped.log"
                    grouped_dta_path = run_output_dir / f"{y_var}_{x_var}_{spec_def['tag']}_{grouping_path_part}_grouped_stata_results.dta"
                    interaction_do_path = run_output_dir / f"{y_var}_{x_var}_{spec_def['tag']}_{grouping_path_part}_interaction.do"
                    interaction_log_path = run_output_dir / f"{y_var}_{x_var}_{spec_def['tag']}_{grouping_path_part}_interaction.log"
                    interaction_dta_path = run_output_dir / f"{y_var}_{x_var}_{spec_def['tag']}_{grouping_path_part}_interaction_stata_results.dta"
                    grouped_results_path = run_output_dir / f"{y_var}_{x_var}_{spec_def['tag']}_{grouping_path_part}_results.csv"
                    grouped_meta_path = run_output_dir / f"{y_var}_{x_var}_{spec_def['tag']}_{grouping_path_part}_plot_meta.json"
                    grouped_output_path = _render_output_path(
                        run_output_dir,
                        str(spec_def["tag"]),
                        f"{y_var}_{x_var}_{spec_def['tag']}_{grouping_path_part}.png",
                        args.export_format,
                    )

                    print(rm._format_plot_regression_count(base_regression_count * 4))
                    _write_reghdfe_do(
                        do_path=grouped_do_path.resolve(),
                        log_path=grouped_log_path.resolve(),
                        data_path=dta_path.resolve(),
                        results_dta=grouped_dta_path.resolve(),
                        y=y_var,
                        x=x_var,
                        controls_must=controls_must_flat,
                        controls_must_slots=controls_must_slots,
                        controls_test=controls_test_flat,
                        controls_test_slots=controls_test_slots,
                        spec_def=spec_def,
                        var_map=var_map,
                        grouping_variable=grouping_variable,
                        grouping_scope=grouping_scope,
                    )
                    _run_stata_do(args.stata_path, grouped_do_path, grouped_log_path, run_output_dir, verbose=False)
                    _ensure_stata_result_exists(
                        results_dta=grouped_dta_path,
                        do_path=grouped_do_path,
                        log_path=grouped_log_path,
                    )
                    grouped_plot_records = _grouped_records_from_stata_dta(
                        grouped_dta_path,
                        controls_must_slots=controls_must_slots,
                        controls_test_slots=controls_test_slots,
                    )

                    _write_interaction_reghdfe_do(
                        do_path=interaction_do_path.resolve(),
                        log_path=interaction_log_path.resolve(),
                        data_path=dta_path.resolve(),
                        results_dta=interaction_dta_path.resolve(),
                        y=y_var,
                        x=x_var,
                        z=grouping_variable,
                        controls_must=controls_must_flat,
                        controls_must_slots=controls_must_slots,
                        controls_test=controls_test_flat,
                        controls_test_slots=controls_test_slots,
                        spec_def=spec_def,
                        var_map=var_map,
                    )
                    _run_stata_do(args.stata_path, interaction_do_path, interaction_log_path, run_output_dir, verbose=False)
                    _ensure_stata_result_exists(
                        results_dta=interaction_dta_path,
                        do_path=interaction_do_path,
                        log_path=interaction_log_path,
                    )
                    interaction_records = _records_from_stata_dta(
                        interaction_dta_path,
                        controls_must_slots=controls_must_slots,
                        controls_test_slots=controls_test_slots,
                    )
                    if not interaction_records:
                        _raise_empty_stata_records(
                            spec_display=f"{spec_display} interaction {x_var}#{grouping_variable}",
                            results_dta=interaction_dta_path,
                            do_path=interaction_do_path,
                            log_path=interaction_log_path,
                        )
                    rm.write_analysis_artifacts(
                        records=records,
                        results_path=grouped_results_path,
                        meta_path=grouped_meta_path,
                        meta={
                            "engine": "stata",
                            "spec_name": spec_name,
                            "y": y_var,
                            "x": x_var,
                            "controls_test_flat": controls_test_flat,
                            "controls_must_flat": controls_must_flat,
                            "matrix_controls": matrix_controls,
                            "matrix_alt_groups": matrix_alt_groups or [],
                            "show_special_markers": True,
                            "fig_width": args.fig_width,
                            "dpi": args.dpi,
                            "order": args.order,
                            "sort_by_p_mode": rm._order_uses_p_mode(args.order),
                            "sort_by_signed_p": rm._order_uses_p_mode(args.order),
                            "title_suffix": f"{title_suffix} - grouped + interaction c.{x_var}#c.{grouping_variable}",
                            "output_path": str(grouped_output_path),
                            "grouping_variable": grouping_display_name,
                            "grouped_plot_records": grouped_plot_records,
                            "interaction_plot_records": interaction_records,
                        },
                        verbose=False,
                    )
                    interaction_sig_rows = rm._build_sig_rows(
                        records=interaction_records,
                        y=y_var,
                        x=f"c.{x_var}#c.{grouping_variable}",
                        controls_must=controls_must_flat,
                        controls_test=controls_test_flat,
                        fe_cols=fe_cols,
                        clust_cols=clust_cols,
                        vce_label="robust" if spec_def["vce"] == "robust" else None,
                    )
                    grouped_sig_rows = rm._build_sig_rows(
                        records=records,
                        y=y_var,
                        x=x_var,
                        controls_must=controls_must_flat,
                        controls_test=controls_test_flat,
                        fe_cols=fe_cols,
                        clust_cols=clust_cols,
                        vce_label="robust" if spec_def["vce"] == "robust" else None,
                        grouping_variable=grouping_display_name,
                        grouped_records=grouped_plot_records,
                    )
                    item = {
                        "records": records,
                        "sig_rows": grouped_sig_rows + interaction_sig_rows,
                        "summary_sig_rows": base_sig_rows,
                        "counts_as_base_spec": grouping_idx == 0,
                        "results_path": grouped_results_path,
                        "meta_path": grouped_meta_path,
                        "output_path": grouped_output_path,
                        "fe_type": tuple(spec_def["fe_keys"]),
                        "elapsed_seconds": perf_counter() - spec_t0,
                    }
                    if on_item_ready is not None:
                        on_item_ready(item)
                        item["plotted"] = True
                    pair_items.append(item)
                    if not args.keep_temp:
                        rm_common.safe_unlink(grouped_do_path)
                        rm_common.safe_unlink(grouped_log_path)
                        rm_common.safe_unlink(grouped_dta_path)
                        rm_common.safe_unlink(interaction_do_path)
                        rm_common.safe_unlink(interaction_log_path)
                        rm_common.safe_unlink(interaction_dta_path)
            else:
                meta_path = run_output_dir / f"{y_var}_{x_var}_{spec_def['tag']}_plot_meta.json"
                output_path = _render_output_path(
                    run_output_dir,
                    str(spec_def["tag"]),
                    f"{y_var}_{x_var}_{spec_def['tag']}.png",
                    args.export_format,
                )
                rm.write_analysis_artifacts(
                    records=records,
                    results_path=results_path,
                    meta_path=meta_path,
                    meta={
                        "engine": "stata",
                        "spec_name": spec_name,
                        "y": y_var,
                        "x": x_var,
                        "controls_test_flat": controls_test_flat,
                        "controls_must_flat": controls_must_flat,
                        "matrix_controls": matrix_controls,
                        "matrix_alt_groups": matrix_alt_groups or [],
                        "show_special_markers": True,
                        "fig_width": args.fig_width,
                        "dpi": args.dpi,
                        "order": args.order,
                        "sort_by_p_mode": rm._order_uses_p_mode(args.order),
                        "sort_by_signed_p": rm._order_uses_p_mode(args.order),
                        "title_suffix": title_suffix,
                        "output_path": str(output_path),
                    },
                    verbose=False,
                )
                item = {
                    "records": records,
                    "sig_rows": base_sig_rows,
                    "summary_sig_rows": base_sig_rows,
                    "counts_as_base_spec": True,
                    "results_path": results_path,
                    "meta_path": meta_path,
                    "output_path": output_path,
                    "fe_type": tuple(spec_def["fe_keys"]),
                    "elapsed_seconds": perf_counter() - spec_t0,
                }
                if on_item_ready is not None:
                    on_item_ready(item)
                    item["plotted"] = True
                pair_items.append(item)

            if not args.keep_temp:
                rm_common.safe_unlink(do_path)
                rm_common.safe_unlink(log_path)
                rm_common.safe_unlink(dta_result_path)
        outputs[(y_var, x_var)] = pair_items

    if not args.keep_temp and dta_path != data_path:
        rm_common.safe_unlink(dta_path)
    return outputs


def main() -> None:
    cfg, cli_args = rm_common.load_toml_config(sys.argv[1:])
    parser = argparse.ArgumentParser(
        prog="regression_monkey_stata",
        description="Run Stata/reghdfe analysis and write standard Regression Monkey result files.",
    )
    parser.add_argument("--data", metavar="FILE")
    parser.add_argument("--y", metavar="VAR", nargs="+")
    parser.add_argument("--x", metavar="VAR", nargs="+")
    parser.add_argument("--controls", metavar="VAR", nargs="+", help="compat alias for --controls-test")
    parser.add_argument("--controls-test", dest="controls_test", metavar="VAR", nargs="+")
    parser.add_argument("--controls-must", dest="controls_must", metavar="VAR", nargs="+")
    parser.add_argument("--grouping-variable", dest="grouping_variable", metavar="VAR", nargs="+", help="compat alias for --grouping-variable-by-ind-time")
    parser.add_argument("--grouping-variable-by-ind-time", dest="grouping_variable_by_ind_time", metavar="VAR", nargs="+")
    parser.add_argument("--grouping-variable-by-time", dest="grouping_variable_by_time", metavar="VAR", nargs="+")
    parser.add_argument("--grouping-variable-by-none", dest="grouping_variable_by_none", metavar="VAR", nargs="+")
    parser.add_argument("--Firm-FE", dest="firm_fe", default="code", metavar="COL")
    parser.add_argument("--Ind-FE", dest="ind_fe", default="ind", metavar="COL")
    parser.add_argument("--Time-FE", dest="time_fe", default="year", metavar="COL")
    parser.add_argument("--Region-FE", dest="region_fe", default=None, metavar="COL")
    parser.add_argument("--output", default="outputs", metavar="DIR")
    parser.add_argument("--dpi", default=150, type=int)
    parser.add_argument("--fig-width", default=14.0, type=float, metavar="INCHES")
    parser.add_argument("--order", choices=["coef", "p"], default="coef", help="绘图排序方式：coef 或 p")
    parser.add_argument("--p", action="store_true", help="兼容别名；等价于 --order p")
    parser.add_argument("--stata-path", default="stata-mp", metavar="EXE")
    parser.add_argument("--keep-temp", action="store_true", help="保留 .do / .log / 中间 Stata 结果文件")
    for spec_name in rm._ALL_SPEC_NAMES:
        parser.add_argument(f"--{spec_name.replace('_', '-')}", dest=spec_name, action="store_true")

    if cfg:
        allowed = {
            "data", "y", "x", "controls", "controls_test", "controls_must",
            "grouping_variable", "grouping_variable_by_ind_time",
            "grouping_variable_by_time", "grouping_variable_by_none",
            "output", "dpi", "fig_width", "order", "p", "firm_fe", "ind_fe", "time_fe",
            "region_fe", "stata_path", "keep_temp",
        } | set(rm._ALL_SPEC_NAMES)
        normalized = {k.lower(): v for k, v in cfg.items()}
        parser.set_defaults(**{k: v for k, v in normalized.items() if k in allowed})

    args = parser.parse_args(cli_args)
    try:
        args.order = rm._normalize_plot_order(args.order, p_alias=bool(args.p))
    except ValueError as exc:
        parser.error(str(exc))
    try:
        args.y = rm._expand_space_separated_names(args.y)
        args.x = rm._expand_space_separated_names(args.x)
        args.grouping_variable = rm._expand_space_separated_names(args.grouping_variable)
        args.grouping_variable_by_ind_time = rm._expand_space_separated_names(args.grouping_variable_by_ind_time)
        args.grouping_variable_by_time = rm._expand_space_separated_names(args.grouping_variable_by_time)
        args.grouping_variable_by_none = rm._expand_space_separated_names(args.grouping_variable_by_none)
    except ValueError as exc:
        parser.error(str(exc))
    controls_test = list(args.controls_test) if args.controls_test else (list(args.controls) if args.controls else [])
    controls_must = list(args.controls_must) if args.controls_must else []
    controls_test_flat, controls_test_slots = rm._normalize_controls_test(controls_test)
    controls_must_flat, controls_must_slots = rm._normalize_controls_must(controls_must)
    rm._validate_control_lists_do_not_overlap(controls_test_flat, controls_must_flat)
    matrix_controls = rm._varying_must_controls(controls_must_slots) + controls_test_flat
    if not args.data or not args.y or not args.x:
        parser.error("必须提供 data / y / x（可通过 TOML 或 CLI 指定）")
    spec_flags = {name: getattr(args, name, False) for name in rm._ALL_SPEC_NAMES}
    if not any(spec_flags.values()):
        parser.error("当前 Stata 脚本仅支持自动规格模式，请至少启用一个 absorb_* flag。")

    data_path = pathlib.Path(args.data).expanduser().resolve()
    print(f"读取数据：{data_path}")
    df = rm_common.load_dataframe(data_path)
    print(f"数据读取完成：{len(df):,} 行 × {len(df.columns)} 列")
    try:
        grouping_specs = rm._collect_grouping_variable_specs(
            grouping_variable=list(args.grouping_variable or []),
            grouping_variable_by_ind_time=list(args.grouping_variable_by_ind_time or []),
            grouping_variable_by_time=list(args.grouping_variable_by_time or []),
            grouping_variable_by_none=list(args.grouping_variable_by_none or []),
        )
        grouping_specs = rm._validate_grouping_variable_specs(df, grouping_specs)
    except ValueError as exc:
        parser.error(str(exc))
    output_root = pathlib.Path(args.output).expanduser().resolve()
    if output_root.suffix:
        output_root = output_root.parent
    run_output_dir = output_root / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_output_dir.mkdir(parents=True, exist_ok=True)
    print(f"输出目录：{run_output_dir}")

    run_stata_engine(
        df=df,
        data_path=data_path,
        args=args,
        controls_test=controls_test,
        controls_must=controls_must,
        controls_test_flat=controls_test_flat,
        controls_test_slots=controls_test_slots,
        controls_must_flat=controls_must_flat,
        controls_must_slots=controls_must_slots,
        grouping_variables=grouping_specs,
        matrix_controls=matrix_controls,
        spec_flags=spec_flags,
        run_output_dir=run_output_dir,
    )


if __name__ == "__main__":
    main()
