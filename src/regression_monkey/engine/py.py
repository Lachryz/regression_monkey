"""
regression_monkey · py
====================
规格曲线分析（Specification Curve Analysis）

主入口：
  regression_monkey(df, y, x, controls, fe_cols, clust_cols)
      手动模式：自定 FE 和聚类列
  regression_monkey_auto(df, y, x, controls, firm_fe, ind_fe,
                         time_fe, region_fe, specs)
      自动模式：按 _SPEC_CATALOG 枚举

算法：
  1. FWL 定理：将所有变量对 N 向固定效应做迭代残差化（Gauss-Seidel）
  2. 枚举所有合法控制变量组合（支持组内互斥替代），逐一 OLS 估计
  3. 双向聚类 SE（Cameron-Gelbach-Miller）或单向聚类 SE：
       SSC = G_min/(G_min-1) × (N-1)/(N-k)
       k = k_reg + k_fe_absorbed
       k_fe_absorbed = sum(n_levels_i) - n_connected_components
  4. 绘制规格曲线图（黑白配色，全变量规格红点高亮，黑色块按游程长度深浅）

依赖：numpy, pandas, matplotlib（无需 pyfixest / statsmodels）
"""

from __future__ import annotations

from datetime import datetime
import json
import multiprocessing as mp
import os
import pathlib
import platform
import tempfile
from time import perf_counter
from typing import TYPE_CHECKING, Any, NotRequired, TypeAlias, TypedDict, cast
import numpy as np
import pandas as pd
from scipy import linalg as sp_linalg
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components as _csc_connected_components
from scipy.stats import t as student_t

try:
    import polars as pl
except ImportError:
    pl = None

for _env_name in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    _ = os.environ.setdefault(_env_name, "1")

# 强制使用非交互式后端，避免 CLI 场景下因 MacOSX/Qt 事件循环卡住
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(pathlib.Path(tempfile.gettempdir()) / "regression_monkey_mpl"))
os.environ.setdefault("XDG_CACHE_HOME", str(pathlib.Path(tempfile.gettempdir()) / "regression_monkey_cache"))

if TYPE_CHECKING:
    from matplotlib.figure import Figure


SeArgs = tuple[Any, ...]
_MAX_AUTO_JOBS = 9
ControlSlot: TypeAlias = tuple[str, ...]
ControlSpecInput: TypeAlias = list[str | list[str] | tuple[str, ...]]


class SpecRecord(TypedDict):
    coef: float
    se: float
    t_value: float
    p_value: float
    adj_r2: NotRequired[float]
    within_r2: NotRequired[float]
    f_stat: NotRequired[float]
    df_resid: int
    ci99_lo: float
    ci99_hi: float
    ci95_lo: float
    ci95_hi: float
    ci90_lo: float
    ci90_hi: float
    controls_test: set[str]
    controls_all: set[str]
    control_stats: NotRequired[list[dict[str, Any]]]
    is_full: bool
    obs: int


class GroupedPlotRecord(TypedDict):
    grouping_variable: str
    group_value: int
    coef: float
    p_value: float
    ci99_lo: float
    ci99_hi: float
    ci95_lo: float
    ci95_hi: float
    ci90_lo: float
    ci90_hi: float
    obs: int
    controls_all: list[str]


# ────────────────────────────────────────────────────────────
# 内部工具函数
# ────────────────────────────────────────────────────────────

def _normalize_control_spec(
    controls: ControlSpecInput,
    *,
    field_name: str,
) -> tuple[list[str], list[ControlSlot]]:
    """
    将 controls 规范化为：
    - flat_controls: 扁平变量顺序，用于画图、导出、缺失列检查
    - control_slots: 枚举槽位；单变量槽位长度为 1，替代组槽位长度 >= 2
    """
    flat_controls: list[str] = []
    control_slots: list[ControlSlot] = []
    seen: set[str] = set()

    for item in controls:
        if isinstance(item, str):
            names = item.split()
            if not names:
                raise ValueError(f"{field_name} 不能包含空字符串变量名")
            slots = [(name,) for name in names]
        elif isinstance(item, (list, tuple)):
            if not item:
                raise ValueError(f"{field_name} 中的替代组不能为空列表")
            slot_names: list[str] = []
            for value in item:
                if not isinstance(value, str):
                    raise ValueError(f"{field_name} 的替代组必须全部由非空字符串列名组成")
                names = value.split()
                if not names:
                    raise ValueError(f"{field_name} 的替代组必须全部由非空字符串列名组成")
                slot_names.extend(names)
            slots = [tuple(slot_names)]
        else:
            raise ValueError(f"{field_name} 仅支持 str 或由 str 组成的 list/tuple")

        for slot in slots:
            dup_in_slot = [name for name in slot if slot.count(name) > 1]
            if dup_in_slot:
                raise ValueError(f"{field_name} 存在重复列名：{sorted(set(dup_in_slot))}")
            dup = [name for name in slot if name in seen]
            if dup:
                raise ValueError(f"{field_name} 存在重复列名：{dup}")

            control_slots.append(slot)
            flat_controls.extend(slot)
            seen.update(slot)

    return flat_controls, control_slots


def _expand_space_separated_names(values: list[str] | tuple[str, ...] | None) -> list[str]:
    """Expand flat variable-name lists where an item may contain whitespace-separated names."""
    if values is None:
        return []
    names: list[str] = []
    for value in values:
        parts = value.split()
        if not parts:
            raise ValueError("变量列表不能包含空字符串变量名")
        names.extend(parts)
    return names


def _normalize_controls_test(
    controls_test: ControlSpecInput,
) -> tuple[list[str], list[ControlSlot]]:
    """
    将 controls_test 规范化为扁平变量顺序和枚举槽位。

    槽位语义为“至多选择一个”：
    - "size" -> (none, size)
    - ["SOE1", "SOE2"] -> (none, SOE1, SOE2)
    """
    return _normalize_control_spec(controls_test, field_name="controls_test")


def _normalize_controls_must(
    controls_must: ControlSpecInput,
) -> tuple[list[str], list[ControlSlot]]:
    """
    将 controls_must 规范化为扁平变量顺序和枚举槽位。

    槽位语义为“必须选择一个”：
    - "size" -> (size)
    - ["SOE1", "SOE2"] -> (SOE1, SOE2)
    """
    return _normalize_control_spec(controls_must, field_name="controls_must")


def _varying_must_controls(must_slots: list[ControlSlot]) -> list[str]:
    """返回 controls_must 中会随规格变化的变量（即替代组成员）。"""
    varying: list[str] = []
    for slot in must_slots:
        if len(slot) > 1:
            varying.extend(slot)
    return varying


def _wrap_title_line(prefix: str, items: list[str], max_width: int = 90) -> str:
    """将 'prefix = item1, item2, ...' 超出 max_width 时按逗号换行并缩进对齐。"""
    if not items:
        return f"{prefix} = (none)"
    full_prefix = f"{prefix} = "
    indent = " " * len(full_prefix)
    lines: list[str] = []
    current = full_prefix
    for item in items:
        sep = "" if current == full_prefix else ", "
        candidate = current + sep + item
        if current != full_prefix and len(candidate) > max_width:
            lines.append(current)
            current = indent + item
        else:
            current = candidate
    lines.append(current)
    return "\n".join(lines)


def _compute_swimlane_ranges(
    matrix_controls: list[str],
    must_slots: list[ControlSlot],
    test_slots: list[ControlSlot],
) -> list[tuple[int, int]]:
    """返回控制变量矩阵中各替代组的行范围 (start_row, end_row)，用于泳道背景。"""
    row_map = {c: i for i, c in enumerate(matrix_controls)}
    ranges: list[tuple[int, int]] = []
    for slot in must_slots + test_slots:
        if len(slot) < 2:
            continue
        rows = [row_map[c] for c in slot if c in row_map]
        if len(rows) >= 2:
            ranges.append((min(rows), max(rows)))
    return ranges


def _validate_control_lists_do_not_overlap(
    controls_test_flat: list[str],
    controls_must_flat: list[str],
) -> None:
    overlap = sorted(set(controls_test_flat) & set(controls_must_flat))
    if overlap:
        raise ValueError(
            "变量不可同时出现在 controls_test 和 controls_must 中："
            + ", ".join(overlap)
        )


def _validate_grouping_variables(
    df: pd.DataFrame,
    grouping_variables: list[str],
) -> list[str]:
    validated: list[str] = []
    for col in grouping_variables:
        if col in validated:
            raise ValueError(f"grouping_variable 存在重复列名：{col}")
        if col not in df.columns:
            raise ValueError(f"grouping_variable 指定的列 '{col}' 不存在于数据中")
        if not pd.api.types.is_numeric_dtype(df[col]):
            raise ValueError(f"grouping_variable '{col}' 需要是数值变量")
        validated.append(col)
    return validated


def _collect_grouping_variable_specs(
    *,
    grouping_variable: list[str] | None = None,
    grouping_variable_by_ind_time: list[str] | None = None,
    grouping_variable_by_time: list[str] | None = None,
    grouping_variable_by_none: list[str] | None = None,
) -> list[tuple[str, str]]:
    specs: list[tuple[str, str]] = []
    scope_inputs = [
        ("by_ind_time", list(grouping_variable_by_ind_time or []) + list(grouping_variable or [])),
        ("by_time", list(grouping_variable_by_time or [])),
        ("by_none", list(grouping_variable_by_none or [])),
    ]
    for scope, variables in scope_inputs:
        seen: set[str] = set()
        for variable in variables:
            if variable in seen:
                raise ValueError(f"grouping_variable_{scope} 存在重复列名：{variable}")
            seen.add(variable)
            specs.append((scope, variable))
    return specs


def _validate_grouping_variable_specs(
    df: pd.DataFrame,
    specs: list[tuple[str, str]],
) -> list[tuple[str, str]]:
    for scope, variables in (
        ("by_ind_time", [variable for spec_scope, variable in specs if spec_scope == "by_ind_time"]),
        ("by_time", [variable for spec_scope, variable in specs if spec_scope == "by_time"]),
        ("by_none", [variable for spec_scope, variable in specs if spec_scope == "by_none"]),
    ):
        _validate_grouping_variables(df, variables)
    for scope, _variable in specs:
        if scope not in {"by_ind_time", "by_time", "by_none"}:
            raise ValueError(f"grouping_variable scope 不支持：{scope}")
    return specs


def _spec_count_from_slots(
    must_slots: list[ControlSlot],
    test_slots: list[ControlSlot],
) -> int:
    """返回规格总数；must 槽位必须选一，test 槽位允许不选。"""
    total_specs = 1
    for slot in must_slots:
        total_specs *= len(slot)
    for slot in test_slots:
        total_specs *= len(slot) + 1
    return total_specs


def _format_plot_regression_count(count: int) -> str:
    return f"[本图回归数] {count:,} 个回归"


def _sort_records_by_p_mode(records: list[SpecRecord]) -> list[SpecRecord]:
    """p 模式：按 sign(coef) / p_value 从小到大排序。"""
    return sorted(
        records,
        key=lambda record: (
            (-1.0 if float(record["coef"]) < 0 else 1.0)
            / max(float(record["p_value"]), np.finfo(float).tiny)
        ),
    )


def _sort_records_for_plot(
    records: list[SpecRecord],
    *,
    sort_by_signed_p: bool = False,
) -> list[SpecRecord]:
    if sort_by_signed_p:
        return _sort_records_by_p_mode(list(records))
    return list(records)


def _normalize_plot_order(order: str | None, *, p_alias: bool = False) -> str:
    if p_alias:
        return "p"
    normalized = (order or "coef").lower()
    if normalized not in {"coef", "p"}:
        raise ValueError("order 只能是 coef 或 p")
    return normalized


def _order_uses_p_mode(order: str | None, *, p_alias: bool = False) -> bool:
    return _normalize_plot_order(order, p_alias=p_alias) == "p"


def _decode_required_choice(
    bits: int,
    control_slots: list[ControlSlot],
) -> tuple[int, list[int], list[str]]:
    """解码必须选择一个的槽位（用于 controls_must）。"""
    chosen_cols: list[int] = []
    chosen_names: list[str] = []
    flat_idx = 0

    for slot in control_slots:
        radix = len(slot)
        state = bits % radix
        bits //= radix
        chosen_cols.append(flat_idx + state)
        chosen_names.append(slot[state])
        flat_idx += len(slot)

    return bits, chosen_cols, chosen_names


def _decode_optional_choice(bits: int, control_slots: list[ControlSlot]) -> tuple[list[int], list[str], bool]:
    """
    解码可不选的槽位（用于 controls_test）。

    返回：
    - chosen_cols: ct_arr 中被选中的列索引
    - chosen_names: 本规格实际纳入的 test controls
    - is_full: 每个槽位都选择了一个变量（替代组则为组内某一个）
    """
    chosen_cols: list[int] = []
    chosen_names: list[str] = []
    is_full = True
    flat_idx = 0

    for slot in control_slots:
        radix = len(slot) + 1
        state = bits % radix
        bits //= radix
        if state == 0:
            is_full = False
        else:
            chosen_cols.append(flat_idx + state - 1)
            chosen_names.append(slot[state - 1])
        flat_idx += len(slot)

    return chosen_cols, chosen_names, is_full

def _absorb_n(vec: np.ndarray, groups: list[np.ndarray],
              tol: float = 1e-10, max_iter: int = 2000) -> np.ndarray:
    """迭代去除 N 向固定效应（Gauss-Seidel within 变换）。"""
    r = vec.copy().astype(float)
    cnts = [np.bincount(g).astype(float) for g in groups]
    for _ in range(max_iter):
        max_delta = 0.0
        for g, cnt in zip(groups, cnts):
            upd = (np.bincount(g, weights=r) / cnt)[g]
            r -= upd
            d = float(np.max(np.abs(upd)))
            if d > max_delta:
                max_delta = d
        if max_delta < tol:
            break
    return r


def _connected_components_n(*groups: np.ndarray) -> int:
    """
    计算 N 向固定效应的连通分量数（多部图连通分量）。
    k_fe_absorbed = sum(n_levels_i) - n_components

    使用 scipy.sparse.csgraph.connected_components 替代纯 Python
    union-find 循环，大幅减少 Python 层迭代次数。
    """
    if len(groups) == 0:
        return 0
    if len(groups) == 1:
        return 1

    n_levels = [int(g.max()) + 1 for g in groups]
    offsets = np.zeros(len(n_levels), dtype=np.intp)
    for i in range(1, len(n_levels)):
        offsets[i] = offsets[i - 1] + n_levels[i - 1]
    total_nodes = int(offsets[-1]) + n_levels[-1]

    g0 = groups[0].astype(np.intp) + offsets[0]
    row_parts, col_parts = [], []
    for k in range(1, len(groups)):
        gk = groups[k].astype(np.intp) + offsets[k]
        row_parts.append(g0)
        col_parts.append(gk)

    rows = np.concatenate(row_parts)
    cols = np.concatenate(col_parts)
    data = np.ones(len(rows), dtype=np.float32)
    # 对称邻接矩阵
    mat = csr_matrix(
        (np.concatenate([data, data]),
         (np.concatenate([rows, cols]), np.concatenate([cols, rows]))),
        shape=(total_nodes, total_nodes),
    )
    n_comp, _ = _csc_connected_components(mat, directed=False)
    return int(n_comp)


def _precomp(c: np.ndarray):
    """预计算聚类排序索引和组边界（只需算一次）。"""
    order = np.argsort(c, kind="stable")
    bdry = np.concatenate([[0], np.where(np.diff(c[order]))[0] + 1])
    return order, bdry


def _meat(Xe: np.ndarray, order: np.ndarray, bdry: np.ndarray) -> np.ndarray:
    gs = np.add.reduceat(Xe[order], bdry)
    return gs.T @ gs


def _cgm_se(X2: np.ndarray, e: np.ndarray,
            ord1, bd1, ord2, bd2, ord12, bd12,
            nc1: int, nc2: int, nc12: int,
            N: int, k_total: int) -> np.ndarray:
    """
    双向 CGM 聚类标准误，与 pyfixest CRV1 完全一致：
      SSC = G_min/(G_min-1) × (N-1)/(N-k_total)
      V   = XtX⁻¹ · SSC·(B1 + B2 - B12) · XtX⁻¹
    """
    XtX_inv = np.linalg.inv(X2.T @ X2)
    Xe = X2 * e[:, None]
    B1  = _meat(Xe, ord1,  bd1)
    B2  = _meat(Xe, ord2,  bd2)
    B12 = _meat(Xe, ord12, bd12)
    G_min = min(nc1, nc2, nc12)
    ssc = G_min / (G_min - 1) * (N - 1) / (N - k_total)
    V = XtX_inv @ (ssc * (B1 + B2 - B12)) @ XtX_inv
    return np.sqrt(np.diag(V))


def _se_single(X2: np.ndarray, e: np.ndarray,
               ord1, bd1,
               nc1: int, N: int, k_total: int) -> np.ndarray:
    """
    单向聚类 SE，与 pyfixest CRV1 一致：
      SSC = G/(G-1) × (N-1)/(N-k_total)
      V   = XtX⁻¹ · SSC · B1 · XtX⁻¹
    """
    XtX_inv = np.linalg.inv(X2.T @ X2)
    Xe = X2 * e[:, None]
    B1 = _meat(Xe, ord1, bd1)
    ssc = nc1 / (nc1 - 1) * (N - 1) / (N - k_total)
    V = XtX_inv @ (ssc * B1) @ XtX_inv
    return np.sqrt(np.diag(V))


def _se_robust(X2: np.ndarray, e: np.ndarray, N: int, k_total: int) -> np.ndarray:
    """
    异方差稳健 SE（HC1）：
      V = XtX⁻¹ · [N/(N-k_total)] · (X'e²X) · XtX⁻¹
    """
    XtX_inv = np.linalg.inv(X2.T @ X2)
    Xe = X2 * e[:, None]
    ssc = N / (N - k_total)
    V = XtX_inv @ (ssc * (Xe.T @ Xe)) @ XtX_inv
    return np.sqrt(np.diag(V))


def _resid_df(N: int, k_total: int) -> int:
    """返回用于 t 分布检验的残差自由度。"""
    return max(1, N - k_total)


def _crit_values(df_resid: int) -> tuple[float, float, float]:
    """返回 99% / 95% / 90% 双侧置信区间对应的 t 临界值。"""
    return (
        float(student_t.ppf(0.995, df_resid)),
        float(student_t.ppf(0.975, df_resid)),
        float(student_t.ppf(0.95, df_resid)),
    )


def _p_value_from_t(t_abs: float, df_resid: int) -> float:
    """根据 t 统计量绝对值和自由度计算双侧精确 p 值。"""
    return float(2.0 * student_t.sf(t_abs, df_resid))


_enum_worker_state: dict[str, Any] = {}


def _resolve_n_jobs(n_jobs: int | None) -> int:
    """
    解析用户请求的并行核数。

    - `n_jobs <= 0` 或 `None`：自动模式，尽量使用更多物理核/逻辑核；
    - 自动模式与手动模式都统一封顶到 9 核，避免规格枚举时过度争抢资源。
    """
    cpu_count = os.cpu_count() or mp.cpu_count() or 1
    auto_jobs = max(1, min(_MAX_AUTO_JOBS, int(cpu_count)))
    if n_jobs is None:
        return auto_jobs
    requested = int(n_jobs)
    if requested <= 0:
        return auto_jobs
    return max(1, min(requested, _MAX_AUTO_JOBS))


def _plan_task_parallelism(n_jobs: int, n_tasks: int) -> tuple[int, list[int]]:
    """
    为自动模式中的多个规格任务分配并行度。

    设计原则：
    - 总并行度不超过 `n_jobs`；
    - 自动模式始终避免“Pool 内再起 Pool”，因为 multiprocessing.Pool
      的 worker 是 daemon 进程，不能再创建子进程；
    - 因此有多个规格任务时，只使用外层并行；仅当任务数为 1 时，
      才把全部核数交给该任务的内层规格枚举。
    """
    if n_tasks <= 0:
        return 1, []
    if n_jobs <= 1:
        return 1, [1] * n_tasks
    if n_tasks == 1:
        return 1, [n_jobs]
    return min(n_jobs, n_tasks), [1] * n_tasks


def _best_mp_context() -> Any:
    """
    为规格枚举选择更合适的 multiprocessing 上下文。

    macOS 上 fork 会继承 Objective-C 运行时状态，Python 3.12+ 已将
    macOS 默认改为 spawn；此处主动对齐，避免 uv/fork 留下僵尸进程。
    Linux 仍使用 fork 以保留启动速度优势。
    """
    methods = set(mp.get_all_start_methods())
    if platform.system() != "Darwin" and "fork" in methods:
        return mp.get_context("fork")
    return mp.get_context("spawn")


def _valid_mask(arr: np.ndarray) -> np.ndarray:
    """返回数组的非缺失布尔掩码。"""
    return ~pd.isna(arr).to_numpy() if isinstance(arr, pd.Series) else ~pd.isna(arr)


def _drop_fe_singletons(mask: np.ndarray, fe_arrs: list[np.ndarray]) -> np.ndarray:
    """
    迭代剔除 FE singleton observations，更接近 reghdfe 默认行为。

    只要某个样本在任一固定效应维度上属于单例组，就剔除；
    剔除后重新计数，直到没有新的 singleton。
    """
    if not fe_arrs:
        return mask

    work_mask = mask.copy()
    while True:
        singleton_obs = np.zeros(work_mask.shape[0], dtype=bool)
        active_idx = np.flatnonzero(work_mask)
        if active_idx.size <= 1:
            return work_mask

        for fe in fe_arrs:
            codes, _ = pd.factorize(fe[work_mask], sort=False)
            counts = np.bincount(codes)
            singleton_obs_active = counts[codes] == 1
            if singleton_obs_active.any():
                singleton_obs[active_idx[singleton_obs_active]] = True

        if not singleton_obs.any():
            return work_mask
        work_mask &= ~singleton_obs


def _drop_collinear_controls(
    xr_: np.ndarray,
    cm_resid: np.ndarray,
    ct_resid: np.ndarray,
    chosen_must: list[str],
    chosen_test: list[str],
    tol: float = 1e-10,
) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    """
    更接近 reghdfe 的吸收后共线性处理：
    - 主解释变量 `x` 必须保留；若其在吸收后近乎为零，则该规格不可估；
    - controls_must / controls_test 中与已有列共线的列会被丢弃。
    """
    if float(np.linalg.norm(xr_)) <= tol:
        raise ValueError("absorbed regressor is collinear with fixed effects")

    x_blocks = [xr_[:, None]]
    col_meta: list[tuple[str, str | None]] = [("x", None)]

    if cm_resid.size:
        for j in range(cm_resid.shape[1]):
            col = cm_resid[:, j]
            if float(np.linalg.norm(col)) <= tol:
                continue
            trial = np.column_stack([*x_blocks, col[:, None]])
            if np.linalg.matrix_rank(trial, tol=tol) > len(x_blocks):
                x_blocks.append(col[:, None])
                col_meta.append(("must", chosen_must[j]))

    if ct_resid.size:
        for j in range(ct_resid.shape[1]):
            col = ct_resid[:, j]
            if float(np.linalg.norm(col)) <= tol:
                continue
            trial = np.column_stack([*x_blocks, col[:, None]])
            if np.linalg.matrix_rank(trial, tol=tol) > len(x_blocks):
                x_blocks.append(col[:, None])
                col_meta.append(("test", chosen_test[j]))

    X2 = np.column_stack(x_blocks)

    # 最后再用带 pivoting 的 QR 做一次稳健筛选，避免近共线残留。
    _q, r, piv = sp_linalg.qr(X2, mode="economic", pivoting=True)
    rank = int(np.sum(np.abs(np.diag(r)) > tol))
    keep_piv = set(int(v) for v in piv[:rank])
    if 0 not in keep_piv:
        raise ValueError("main regressor dropped by collinearity check")
    if rank < X2.shape[1]:
        keep_idx = [j for j in range(X2.shape[1]) if j in keep_piv]
        X2 = X2[:, keep_idx]
        col_meta = [col_meta[j] for j in keep_idx]

    kept_must = [name for kind, name in col_meta if kind == "must" and name is not None]
    kept_chosen = [name for kind, name in col_meta if kind == "test" and name is not None]
    ct_plot_resid = (
        ct_resid[:, [j for j, name in enumerate(chosen_test) if name in set(kept_chosen)]]
        if ct_resid.size and kept_chosen else np.empty((len(xr_), 0))
    )
    return X2, ct_plot_resid, kept_must, kept_chosen


def _init_enum_worker(
    y_arr: np.ndarray,
    x_arr: np.ndarray,
    cm_arr: np.ndarray,
    ct_arr: np.ndarray,
    fe_arrs: list[np.ndarray],
    cl_arrs: list[np.ndarray],
    base_mask: np.ndarray,
    controls_must: list[str],
    must_slots: list[ControlSlot],
    controls_test: list[str],
    test_slots: list[ControlSlot],
    se_kind: str,
) -> None:
    """在子进程中初始化只读枚举状态。"""
    import signal
    signal.signal(signal.SIGINT, signal.SIG_IGN)  # 子进程忽略 Ctrl+C，由主进程统一处理
    global _enum_worker_state
    _enum_worker_state = {
        "y_arr": y_arr,
        "x_arr": x_arr,
        "cm_arr": cm_arr,
        "ct_arr": ct_arr,
        "fe_arrs": fe_arrs,
        "cl_arrs": cl_arrs,
        "base_mask": base_mask,
        "controls_must": controls_must,
        "must_slots": must_slots,
        "controls_test": controls_test,
        "test_slots": test_slots,
        "se_kind": se_kind,
    }


def _enumerate_specs_chunk(bit_range: tuple[int, int]) -> tuple[list[SpecRecord], list[str]]:
    """在单个进程中处理一段规格位掩码。"""
    start, end = bit_range
    y_arr = cast(np.ndarray, _enum_worker_state["y_arr"])
    x_arr = cast(np.ndarray, _enum_worker_state["x_arr"])
    cm_arr = cast(np.ndarray, _enum_worker_state["cm_arr"])
    ct_arr = cast(np.ndarray, _enum_worker_state["ct_arr"])
    fe_arrs = cast(list[np.ndarray], _enum_worker_state["fe_arrs"])
    cl_arrs = cast(list[np.ndarray], _enum_worker_state["cl_arrs"])
    base_mask = cast(np.ndarray, _enum_worker_state["base_mask"])
    controls_must = cast(list[str], _enum_worker_state["controls_must"])
    must_slots = cast(list[ControlSlot], _enum_worker_state["must_slots"])
    controls_test = cast(list[str], _enum_worker_state["controls_test"])
    test_slots = cast(list[ControlSlot], _enum_worker_state["test_slots"])
    se_kind = cast(str, _enum_worker_state["se_kind"])

    records: list[SpecRecord] = []
    skipped: list[str] = []

    for bits in range(start, end):
        rem_bits, must_cols, chosen_must = _decode_required_choice(bits, must_slots)
        test_cols, chosen_test, is_full = _decode_optional_choice(rem_bits, test_slots)

        mask = base_mask.copy()
        for j in must_cols:
            mask &= _valid_mask(cm_arr[:, j])
        for j in test_cols:
            mask &= _valid_mask(ct_arr[:, j])
        mask = _drop_fe_singletons(mask, fe_arrs)

        N = int(mask.sum())
        if N <= 1:
            chosen_all = chosen_must + chosen_test
            skipped.append(f"  [skip] controls={chosen_all or '(none)'}: 样本量不足")
            continue

        try:
            y_sub = y_arr[mask].astype(float)
            x_sub = x_arr[mask].astype(float)
            cm_sub = cm_arr[np.ix_(mask, must_cols)].astype(float) if must_cols else np.empty((N, 0))
            ct_sub = ct_arr[np.ix_(mask, test_cols)].astype(float) if test_cols else np.empty((N, 0))

            groups: list[np.ndarray] = []
            n_levels_list: list[int] = []
            for fe in fe_arrs:
                g = pd.factorize(fe[mask])[0]
                groups.append(g)
                n_levels_list.append(int(g.max()) + 1)

            vars_to_absorb = [y_sub, x_sub]
            vars_to_absorb.extend(cm_sub[:, j] for j in range(cm_sub.shape[1]))
            vars_to_absorb.extend(ct_sub[:, j] for j in range(ct_sub.shape[1]))
            absorbed = [_absorb_n(v, groups) for v in vars_to_absorb]

            yr_ = absorbed[0]
            xr_ = absorbed[1]
            offset = 2
            cm_resid = np.column_stack(absorbed[offset:offset + cm_sub.shape[1]]) if cm_sub.shape[1] else np.empty((N, 0))
            offset += cm_sub.shape[1]
            ct_resid = np.column_stack(absorbed[offset:offset + ct_sub.shape[1]]) if ct_sub.shape[1] else np.empty((N, 0))

            X2, _ct_plot_resid, kept_must, kept_chosen = _drop_collinear_controls(
                xr_=xr_,
                cm_resid=cm_resid,
                ct_resid=ct_resid,
                chosen_must=chosen_must,
                chosen_test=chosen_test,
            )
            kept_must_set = set(kept_must)
            chosen_set = set(kept_chosen)

            beta, *_ = np.linalg.lstsq(X2, yr_, rcond=None)
            e = yr_ - X2 @ beta
            coef = float(beta[0])

            n_comp = _connected_components_n(*groups)
            k_fe_absorbed = sum(n_levels_list) - n_comp
            k_total = X2.shape[1] + k_fe_absorbed

            if se_kind == "robust":
                se = float(_se_robust(X2, e, N, k_total)[0])
            elif se_kind == "one_way":
                c1arr = pd.factorize(cl_arrs[0][mask])[0]
                nc1 = int(c1arr.max()) + 1
                ord1, bd1 = _precomp(c1arr)
                se = float(_se_single(X2, e, ord1, bd1, nc1, N, k_total)[0])
            else:
                c1arr = pd.factorize(cl_arrs[0][mask])[0]
                c2arr = pd.factorize(cl_arrs[1][mask])[0]
                nc1 = int(c1arr.max()) + 1
                nc2 = int(c2arr.max()) + 1
                c12arr = c1arr * nc2 + c2arr
                nc12 = int(np.unique(c12arr).size)
                ord1, bd1 = _precomp(c1arr)
                ord2, bd2 = _precomp(c2arr)
                ord12, bd12 = _precomp(c12arr)
                se = float(_cgm_se(X2, e, ord1, bd1, ord2, bd2, ord12, bd12, nc1, nc2, nc12, N, k_total)[0])

            if not (np.isfinite(coef) and np.isfinite(se) and se > 0):
                raise ValueError(f"non-finite result: coef={coef}, se={se}")

            df_resid = _resid_df(N, k_total)
            t_value = coef / se
            p_value = _p_value_from_t(abs(t_value), df_resid)
            sse = float(np.dot(e, e))
            tss = float(np.dot(yr_, yr_))
            r2 = 1.0 - sse / tss if tss > 0 else float("nan")
            within_r2 = r2
            adj_r2 = 1.0 - (1.0 - r2) * ((N - 1) / df_resid) if np.isfinite(r2) and df_resid > 0 else float("nan")
            k_model = X2.shape[1]
            ms_model = (tss - sse) / k_model if k_model > 0 else float("nan")
            ms_resid = sse / df_resid if df_resid > 0 else float("nan")
            f_stat = ms_model / ms_resid if np.isfinite(ms_model) and np.isfinite(ms_resid) and ms_resid > 0 else float("nan")
            crit99, crit95, crit90 = _crit_values(df_resid)
        except Exception as _exc:
            chosen_all = chosen_must + chosen_test
            skipped.append(f"  [skip] controls={chosen_all or '(none)'}: {_exc}")
            continue

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
            "controls_test": chosen_set,
            "controls_all": kept_must_set | chosen_set,
            "is_full": is_full,
            "obs": N,
        })

    return records, skipped


def _enumerate_specs(
    y_arr: np.ndarray,
    x_arr: np.ndarray,
    cm_arr: np.ndarray,
    ct_arr: np.ndarray,
    fe_arrs: list[np.ndarray],
    cl_arrs: list[np.ndarray],
    base_mask: np.ndarray,
    controls_must: list[str],
    must_slots: list[ControlSlot],
    controls_test: list[str],
    test_slots: list[ControlSlot],
    se_kind: str,
    n_jobs: int = 1,
) -> list[SpecRecord]:
    """枚举所有合法规格，返回排序后的 records 列表（内部共用）。"""
    total_specs = _spec_count_from_slots(must_slots, test_slots)
    n_jobs = max(1, int(n_jobs))
    parallel_threshold = max(32, n_jobs * 4)

    # 规格数较少时，进程池启动与数组传递的成本常常高于并行收益。
    if n_jobs == 1 or total_specs <= 1 or total_specs < parallel_threshold:
        _init_enum_worker(
            y_arr, x_arr, cm_arr, ct_arr, fe_arrs, cl_arrs, base_mask,
            controls_must, must_slots, controls_test, test_slots, se_kind,
        )
        records, skipped = _enumerate_specs_chunk((0, total_specs))
        for msg in skipped:
            print(msg)
        records.sort(key=lambda r: r["coef"])
        return records

    chunk_size = max(1, (total_specs + n_jobs * 4 - 1) // (n_jobs * 4))
    bit_ranges = [
        (start, min(start + chunk_size, total_specs))
        for start in range(0, total_specs, chunk_size)
    ]
    total_chunks = len(bit_ranges)
    print(
        f"开始枚举 {total_specs} 个规格"
        f"（并行 {n_jobs} 进程，{total_chunks} 个任务块，每块约 {chunk_size} 个规格）"
    )

    records = []
    import signal as _signal
    ctx = _best_mp_context()
    # 创建 Pool 期间暂时忽略 SIGINT，确保 worker spawn 阶段不被中断
    _orig_handler = _signal.signal(_signal.SIGINT, _signal.SIG_IGN)
    try:
        pool = ctx.Pool(
            processes=n_jobs,
            initializer=_init_enum_worker,
            initargs=(
                y_arr, x_arr, cm_arr, ct_arr, fe_arrs, cl_arrs, base_mask,
                controls_must, must_slots, controls_test, test_slots, se_kind,
            ),
        )
    finally:
        _signal.signal(_signal.SIGINT, _orig_handler)  # 恢复主进程的 Ctrl+C
    try:
        completed_chunks = 0
        next_report = 1
        for chunk_records, skipped in pool.imap_unordered(_enumerate_specs_chunk, bit_ranges):
            records.extend(chunk_records)
            for msg in skipped:
                print(msg)
            completed_chunks += 1
            progress = completed_chunks * 100 // total_chunks
            if progress >= next_report or completed_chunks == total_chunks:
                print(
                    f"  枚举进度：{completed_chunks}/{total_chunks} 块完成"
                    f"（约 {progress}%）"
                )
                next_report = min(100, progress + 10)
        pool.close()   # 正常完成：通知 worker 不再有新任务
    except BaseException:
        pool.terminate()  # 异常时强制终止 worker
        raise
    finally:
        pool.join()    # 等待所有 worker 退出

    records.sort(key=lambda r: r["coef"])
    return records


def _run_spec_task(args: tuple) -> tuple[str, list[SpecRecord]]:
    """
    执行单个 spec 的完整规格枚举。
    必须定义在模块顶层以支持 spawn 序列化。
    """
    import signal as _sig
    _sig.signal(_sig.SIGINT, _sig.SIG_IGN)

    (spec_name, y_arr, x_arr, cm_arr, ct_arr,
     fe_arrs, cl_arrs, base_mask,
     controls_must, must_slots, controls_test, test_slots, se_kind, n_inner) = args

    records = _enumerate_specs(
        y_arr, x_arr, cm_arr, ct_arr, fe_arrs, cl_arrs, base_mask,
        controls_must, must_slots, controls_test, test_slots, se_kind,
        n_jobs=n_inner,
    )
    return spec_name, records


def _spec_bit_ranges(total_specs: int, n_jobs: int) -> list[tuple[int, int]]:
    """按 worker 数量为单个 spec 切分 bitmask 区间。"""
    chunk_size = max(1, (total_specs + n_jobs * 4 - 1) // (n_jobs * 4))
    return [
        (start, min(start + chunk_size, total_specs))
        for start in range(0, total_specs, chunk_size)
    ]


def _run_flat_spec_chunk(args: tuple) -> tuple[str, list[SpecRecord], list[str], tuple[int, int]]:
    """
    扁平化 worker：处理某个 spec 的一个 bitmask 区间。

    该函数不创建新的进程池，只在当前 worker 内执行一段枚举，
    用于把多个 spec 的工作块统一丢进同一个总进程池。
    """
    (
        spec_name,
        bit_range,
        y_arr,
        x_arr,
        cm_arr,
        ct_arr,
        fe_arrs,
        cl_arrs,
        base_mask,
        controls_must,
        must_slots,
        controls_test,
        test_slots,
        se_kind,
    ) = args
    _init_enum_worker(
        y_arr, x_arr, cm_arr, ct_arr, fe_arrs, cl_arrs, base_mask,
        controls_must, must_slots, controls_test, test_slots, se_kind,
    )
    records, skipped = _enumerate_specs_chunk(bit_range)
    return spec_name, records, skipped, bit_range


# ────────────────────────────────────────────────────────────
# 规格目录（自动模式下的有限枚举组合）
# ────────────────────────────────────────────────────────────

_SPEC_CATALOG: list[dict[str, Any]] = [
    {
        "name":         "absorb_firm_time_vce_cluster_firm",
        "fe_keys":      ["firm", "time"],
        "cl_keys":      ["firm"],
        "vce":          "cluster",
        "derived":      [],
        "needs_region": False,
        "tag":          "ab_firm_time_cl_firm",
        "help":         "absorb({firm} {time}) vce(cluster {firm}) - firm and time FE, clustered by firm, baseline specification",
    },
    {
        "name":         "absorb_firm_time_vce_robust",
        "fe_keys":      ["firm", "time"],
        "cl_keys":      [],
        "vce":          "robust",
        "derived":      [],
        "needs_region": False,
        "tag":          "ab_firm_time_robust",
        "help":         "absorb({firm} {time}) vce(robust) - firm and time FE with heteroskedasticity-robust SE",
    },
    {
        "name":         "absorb_firm_indtime_vce_cluster_firm",
        "fe_keys":      ["firm", "_ind_time"],
        "cl_keys":      ["firm"],
        "vce":          "cluster",
        "derived":      [("_ind_time", "ind", "time")],
        "needs_region": False,
        "tag":          "ab_firm_indtime_cl_firm",
        "help":         "absorb({firm} i.{ind}#i.{time}) vce(cluster {firm}) - firm FE plus industry-by-time FE, clustered by firm",
    },
    {
        "name":         "absorb_firm_indtime_vce_robust",
        "fe_keys":      ["firm", "_ind_time"],
        "cl_keys":      [],
        "vce":          "robust",
        "derived":      [("_ind_time", "ind", "time")],
        "needs_region": False,
        "tag":          "ab_firm_indtime_robust",
        "help":         "absorb({firm} i.{ind}#i.{time}) vce(robust) - firm FE plus industry-by-time FE with heteroskedasticity-robust SE",
    },
    {
        "name":         "absorb_firm_regiontime_vce_cluster_firm",
        "fe_keys":      ["firm", "_region_time"],
        "cl_keys":      ["firm"],
        "vce":          "cluster",
        "derived":      [("_region_time", "region", "time")],
        "needs_region": True,
        "tag":          "ab_firm_regiontime_cl_firm",
        "help":         "absorb({firm} i.{region}#i.{time}) vce(cluster {firm}) - firm FE plus region-by-time FE, clustered by firm",
    },
    {
        "name":         "absorb_firm_regiontime_vce_robust",
        "fe_keys":      ["firm", "_region_time"],
        "cl_keys":      [],
        "vce":          "robust",
        "derived":      [("_region_time", "region", "time")],
        "needs_region": True,
        "tag":          "ab_firm_regiontime_robust",
        "help":         "absorb({firm} i.{region}#i.{time}) vce(robust) - firm FE plus region-by-time FE with heteroskedasticity-robust SE",
    },
    {
        "name":         "absorb_firm_indtime_regiontime_vce_cluster_firm",
        "fe_keys":      ["firm", "_ind_time", "_region_time"],
        "cl_keys":      ["firm"],
        "vce":          "cluster",
        "derived":      [("_ind_time", "ind", "time"), ("_region_time", "region", "time")],
        "needs_region": True,
        "tag":          "ab_firm_indtime_regiontime_cl_firm",
        "help":         "absorb({firm} i.{ind}#i.{time} i.{region}#i.{time}) vce(cluster {firm}) - firm FE with industry-by-time and region-by-time FE",
    },
    {
        "name":         "absorb_firm_indtime_regiontime_vce_robust",
        "fe_keys":      ["firm", "_ind_time", "_region_time"],
        "cl_keys":      [],
        "vce":          "robust",
        "derived":      [("_ind_time", "ind", "time"), ("_region_time", "region", "time")],
        "needs_region": True,
        "tag":          "ab_firm_indtime_regiontime_robust",
        "help":         "absorb({firm} i.{ind}#i.{time} i.{region}#i.{time}) vce(robust) - firm FE with industry-by-time and region-by-time FE and heteroskedasticity-robust SE",
    },
    {
        "name":         "absorb_firm_time_vce_cluster_region",
        "fe_keys":      ["firm", "time"],
        "cl_keys":      ["region"],
        "vce":          "cluster",
        "derived":      [],
        "needs_region": True,
        "tag":          "ab_firm_time_cl_region",
        "help":         "absorb({firm} {time}) vce(cluster {region}) - firm and time FE, clustered by region",
    },
    {
        "name":         "absorb_firm_time_vce_cluster_ind",
        "fe_keys":      ["firm", "time"],
        "cl_keys":      ["ind"],
        "vce":          "cluster",
        "derived":      [],
        "needs_region": False,
        "tag":          "ab_firm_time_cl_ind",
        "help":         "absorb({firm} {time}) vce(cluster {ind}) - firm and time FE, clustered by industry",
    },
    {
        "name":         "absorb_ind_region_time_vce_cluster_ind",
        "fe_keys":      ["ind", "region", "time"],
        "cl_keys":      ["ind"],
        "vce":          "cluster",
        "derived":      [],
        "needs_region": True,
        "tag":          "ab_ind_region_time_cl_ind",
        "help":         "absorb({ind} {region} {time}) vce(cluster {ind}) - industry, region, and time FE, clustered by industry",
    },
    {
        "name":         "absorb_ind_region_time_vce_robust",
        "fe_keys":      ["ind", "region", "time"],
        "cl_keys":      [],
        "vce":          "robust",
        "derived":      [],
        "needs_region": True,
        "tag":          "ab_ind_region_time_robust",
        "help":         "absorb({ind} {region} {time}) vce(robust) - industry, region, and time FE with heteroskedasticity-robust SE",
    },
    {
        "name":         "absorb_firm_time_vce_cluster_firm_time",
        "fe_keys":      ["firm", "time"],
        "cl_keys":      ["firm", "time"],
        "vce":          "cluster",
        "derived":      [],
        "needs_region": False,
        "tag":          "ab_firm_time_cl_firm_time",
        "help":         "absorb({firm} {time}) vce(cluster {firm} {time}) - firm and time FE with CGM two-way clustering",
    },
    {
        "name":         "absorb_ind_time_vce_cluster_firm",
        "fe_keys":      ["ind", "time"],
        "cl_keys":      ["firm"],
        "vce":          "cluster",
        "derived":      [],
        "needs_region": False,
        "tag":          "ab_ind_time_cl_firm",
        "help":         "absorb({ind} {time}) vce(cluster {firm}) - industry and time FE, clustered by firm",
    },
    {
        "name":         "absorb_ind_time_vce_robust",
        "fe_keys":      ["ind", "time"],
        "cl_keys":      [],
        "vce":          "robust",
        "derived":      [],
        "needs_region": False,
        "tag":          "ab_ind_time_robust",
        "help":         "absorb({ind} {time}) vce(robust) - industry and time FE with heteroskedasticity-robust SE",
    },
]

# 所有规格的名称列表（用于 CLI/TOML 键校验）
_ALL_SPEC_NAMES: list[str] = [s["name"] for s in _SPEC_CATALOG]


def _format_spec_display(spec_def: dict[str, Any], fmt: dict[str, str]) -> str:
    """Return terminal-facing Stata-style spec text without changing internal names."""
    return str(spec_def["help"]).format(**fmt).split(" - ", 1)[0]


# ────────────────────────────────────────────────────────────
# 主函数：手动模式
# ────────────────────────────────────────────────────────────

def regression_monkey(
    df: pd.DataFrame,
    y: str,
    x: str,
    controls_test: ControlSpecInput,
    controls_must: ControlSpecInput,
    fe_cols: list[str],
    clust_cols: list[str],
    output_path: str | None = None,
    dpi: int = 150,
    fig_width: float = 14.0,
    n_jobs: int = 0,
    export_sig_table: bool = True,
    title_suffix: str | None = None,
    render_plot: bool = True,
    sort_by_signed_p: bool = False,
) -> tuple[list[SpecRecord], Figure | None]:
    """
    规格曲线分析：枚举所有控制变量组合，绘制主变量系数的规格曲线。

    Parameters
    ----------
    df           : 数据框（列需包含 y, x, controls_test, controls_must, fe_cols, clust_cols）
    y            : 被解释变量列名
    x            : 主解释变量列名（绘图中的 β）
    controls_test: 参与组合枚举的控制变量列表；元素可为列名，或表示“组内互斥替代”的列名列表
    controls_must: 强制纳入的控制变量列表；元素可为列名，或表示“组内互斥替代”的列名列表
    fe_cols      : 固定效应列名列表（1 个或多个，支持 N 向 FE）
    clust_cols   : 聚类变量列名列表（1 个 = 单向聚类；2 个 = CGM 双向聚类）
    output_path  : 图片保存路径；None 则不保存
    dpi          : 图像分辨率
    fig_width    : 图像宽度（英寸）
    n_jobs       : 并行进程数；`<=0` 为自动模式，最多使用 9 核
    export_sig_table: 是否导出显著性汇总表
    title_suffix : 附加到图标题的说明文字（可选）

    Returns
    -------
    records : list[dict]，每条包含 coef / se / ci99_lo / ci99_hi 等字段
    fig     : matplotlib Figure 对象
    """
    total_t0 = perf_counter()
    controls_must_flat, must_slots = _normalize_controls_must(controls_must)
    controls_test_flat, test_slots = _normalize_controls_test(controls_test)
    _validate_control_lists_do_not_overlap(controls_test_flat, controls_must_flat)
    varying_must_controls = _varying_must_controls(must_slots)
    matrix_controls = varying_must_controls + controls_test_flat
    N = len(df)
    y_arr = df[y].to_numpy()
    x_arr = df[x].to_numpy()
    cm_arr = (
        np.column_stack([df[c].to_numpy() for c in controls_must_flat])
        if controls_must_flat else np.empty((N, 0), dtype=float)
    )
    ct_arr = (
        np.column_stack([df[c].to_numpy() for c in controls_test_flat])
        if controls_test_flat else np.empty((N, 0), dtype=float)
    )
    fe_arrs = [df[c].to_numpy() for c in fe_cols]
    cl_arrs = [df[c].to_numpy() for c in clust_cols]

    base_mask = _valid_mask(y_arr) & _valid_mask(x_arr)
    for j in range(cm_arr.shape[1]):
        base_mask &= _valid_mask(cm_arr[:, j])
    for fe in fe_arrs:
        base_mask &= _valid_mask(fe)
    for cl in cl_arrs:
        base_mask &= _valid_mask(cl)

    n_jobs = _resolve_n_jobs(n_jobs)
    total_regressions = _spec_count_from_slots(must_slots, test_slots)
    print(_format_plot_regression_count(total_regressions))

    print(
        f"开始逐规格回归（FE = {', '.join(fe_cols)}；"
        f"controls_must = {len(controls_must_flat)}；controls_test = {len(controls_test_flat)}；"
        f"n_jobs = {n_jobs}）"
    )

    if len(clust_cols) == 0:
        se_kind = "robust"
    elif len(clust_cols) == 2:
        se_kind = "two_way"
    else:
        se_kind = "one_way"

    # ── Step 3: 枚举所有规格 ──────────────────────────────
    records = _enumerate_specs(
        y_arr, x_arr, cm_arr, ct_arr, fe_arrs, cl_arrs, base_mask,
        controls_must_flat, must_slots, controls_test_flat, test_slots, se_kind, n_jobs=n_jobs,
    )
    n_full  = sum(r["is_full"] for r in records)
    print(f"完成 {len(records)} 个规格（{n_full} 个全变量规格）")
    print(f"系数范围：[{records[0]['coef']:.4f}, {records[-1]['coef']:.4f}]")

    # ── Step 4: 绘图 ─────────────────────────────────────
    fig = None
    if render_plot:
        fig = _plot(records, y_name=y, x_name=x, controls_test=controls_test_flat,
                    controls_must=controls_must_flat,
                    matrix_controls=matrix_controls,
                    show_special_markers=True,
                    fig_width=fig_width, dpi=dpi,
                    output_path=output_path, title_suffix=title_suffix,
                    elapsed_seconds_preplot=perf_counter() - total_t0,
                    sort_by_signed_p=sort_by_signed_p,
                    matrix_swimlane_ranges=_compute_swimlane_ranges(matrix_controls, must_slots, test_slots) or None)

    # ── Step 5: 导出显著性汇总表 ──────────────────────────
    if output_path is not None and export_sig_table:
        _stem = pathlib.Path(output_path).stem
        _parent = pathlib.Path(output_path).parent
        tbl_path = str(_parent / f"{_stem}_sig.csv")
        _export_sig_table(
            rows       = _build_sig_rows(
                records    = records,
                y          = y,
                x          = x,
                controls_must = controls_must_flat,
                controls_test = controls_test_flat,
                fe_cols    = fe_cols,
                clust_cols = clust_cols,
            ),
            output_path = tbl_path,
            n_specs    = len(records),
        )

    return records, fig


# ────────────────────────────────────────────────────────────
# 主函数：自动模式
# ────────────────────────────────────────────────────────────

def regression_monkey_auto(
    df: pd.DataFrame,
    y: str,
    x: str,
    controls_test: ControlSpecInput,
    controls_must: ControlSpecInput,
    firm_fe: str,
    ind_fe: str,
    time_fe: str,
    region_fe: str | None,
    specs: dict[str, bool],
    output_path: str | None = None,
    dpi: int = 150,
    fig_width: float = 14.0,
    n_jobs: int = 0,
    export_sig_table: bool = True,
    render_plot: bool = True,
    sort_by_signed_p: bool = False,
) -> list[tuple[str, list[SpecRecord], Figure | None]]:
    """
    自动规格曲线：按 _SPEC_CATALOG 枚举启用的规格，每个规格输出一张图。

    Parameters
    ----------
    df         : 数据框
    y          : 被解释变量
    x          : 主解释变量
    controls_test: 参与组合枚举的控制变量列表；元素可为列名，或表示“组内互斥替代”的列名列表
    controls_must: 强制纳入的控制变量列表；元素可为列名，或表示“组内互斥替代”的列名列表
    firm_fe    : 个体（企业）固定效应列名
    ind_fe     : 行业固定效应列名
    time_fe    : 时间固定效应列名
    region_fe  : 地区固定效应列名（县级 city / 地级 pref / 省级 prov），
                 needs_region=True 的规格需提供
    specs      : dict，键为规格名，值为 True/False；
                 例如 {"spec_firm_year": True, "spec_firm_indyear": True}
    output_path: 基础输出路径；None 则不保存；
                 实际文件名会自动拼接规格标签
    dpi / fig_width / n_jobs : 图像和并行参数；`n_jobs <= 0` 时自动并行且封顶 9 核

    Returns
    -------
    list of (spec_name, records, fig) tuples
    """
    n_jobs = _resolve_n_jobs(n_jobs)
    controls_must_flat, must_slots = _normalize_controls_must(controls_must)
    controls_test_flat, test_slots = _normalize_controls_test(controls_test)
    _validate_control_lists_do_not_overlap(controls_test_flat, controls_must_flat)
    varying_must_controls = _varying_must_controls(must_slots)
    matrix_controls = varying_must_controls + controls_test_flat

    # 基础变量名映射（"逻辑键" → 实际列名）
    base_var_map: dict[str, str] = {
        "firm":   firm_fe,
        "ind":    ind_fe,
        "time":   time_fe,
    }
    if region_fe is not None:
        base_var_map["region"] = region_fe

    # 预先生成所有启用规格所需的交叉项列；若可用则优先使用 Polars 加速预处理
    df = df.copy()
    use_polars = pl is not None
    pl_df = None
    if use_polars:
        try:
            pl_df = pl.from_pandas(df)
            print("[预处理] 使用 Polars 加速交叉项生成与缺失值筛选")
        except Exception as exc:
            use_polars = False
            print(f"[预处理] Polars 回退到 pandas：{exc}")
    derived_map: dict[str, str] = {}   # 逻辑键 → 实际列名（交叉项）
    derived_exprs = []

    for spec_def in _SPEC_CATALOG:
        if not specs.get(spec_def["name"], False):
            continue
        for new_key, src_key1, src_key2 in spec_def["derived"]:
            if new_key in derived_map:
                continue
            src1 = base_var_map[src_key1]
            src2 = base_var_map[src_key2]
            col_name = f"_spec_{src1}_{src2}"
            if use_polars and pl_df is not None:
                if col_name not in pl_df.columns:
                    derived_exprs.append(
                        pl.concat_str(
                            [
                                pl.col(src1).cast(pl.String),
                                pl.lit("_"),
                                pl.col(src2).cast(pl.String),
                            ]
                        ).alias(col_name)
                    )
                    print(f"[生成] 交叉项列：{col_name} = {src1}#{src2}")
            elif col_name not in df.columns:
                df[col_name] = (
                    df[src1].astype(str) + "_" + df[src2].astype(str)
                )
                print(f"[生成] 交叉项列：{col_name} = {src1}#{src2}")
            derived_map[new_key] = col_name

    if use_polars and pl_df is not None and derived_exprs:
        pl_df = pl_df.with_columns(derived_exprs)

    base_vars = [y, x] + controls_must_flat
    results: list[tuple[str, list[SpecRecord], Figure | None]] = []
    all_sig_rows: list[dict] = []
    total_specs = 0
    var_map = {**base_var_map, **derived_map}
    fmt = {
        "firm":   firm_fe,
        "ind":    ind_fe,
        "time":   time_fe,
        "region": region_fe or "region",
    }

    # ── 阶段一：准备所有任务的数据（主进程完成，避免传输 DataFrame）──
    task_args:  list[tuple]  = []   # 单 spec 枚举参数
    task_metas: list[dict]   = []   # 主进程绘图所需的元信息

    for spec_def in _SPEC_CATALOG:
        spec_name = spec_def["name"]
        spec_display = _format_spec_display(spec_def, fmt)
        if not specs.get(spec_name, False):
            continue
        if spec_def["needs_region"] and region_fe is None:
            print(f"[跳过] {spec_display}：需要 Region_FE 但未指定")
            continue
        spec_t0 = perf_counter()

        fe_cols    = [var_map[k] for k in spec_def["fe_keys"]]
        clust_cols = [var_map[k] for k in spec_def["cl_keys"]]

        needed = list(dict.fromkeys(base_vars + controls_test_flat + fe_cols + clust_cols))
        if use_polars and pl_df is not None:
            df_spec = cast(
                pd.DataFrame,
                pl_df.select(needed).drop_nulls(subset=base_vars + fe_cols + clust_cols).to_pandas(),
            )
        else:
            df_spec = df[needed].dropna(subset=base_vars + fe_cols + clust_cols).copy()
        n_drop = len(df) - len(df_spec)
        if n_drop > 0:
            print(f"[{spec_display}] 剔除缺失值：{n_drop} 行（剩余 {len(df_spec):,} 行）")

        N = len(df_spec)
        y_arr  = df_spec[y].to_numpy()
        x_arr  = df_spec[x].to_numpy()
        cm_arr = (
            np.column_stack([df_spec[c].to_numpy() for c in controls_must_flat])
            if controls_must_flat else np.empty((N, 0), dtype=float)
        )
        ct_arr = (
            np.column_stack([df_spec[c].to_numpy() for c in controls_test_flat])
            if controls_test_flat else np.empty((N, 0), dtype=float)
        )
        fe_arrs  = [df_spec[c].to_numpy() for c in fe_cols]
        cl_arrs  = [df_spec[c].to_numpy() for c in clust_cols]
        base_mask = _valid_mask(y_arr) & _valid_mask(x_arr)
        for j in range(cm_arr.shape[1]):
            base_mask &= _valid_mask(cm_arr[:, j])
        for fe in fe_arrs:
            base_mask &= _valid_mask(fe)
        for cl in cl_arrs:
            base_mask &= _valid_mask(cl)

        se_kind = (
            "robust"  if len(clust_cols) == 0 else
            "two_way" if len(clust_cols) == 2 else
            "one_way"
        )
        plot_regression_count = _spec_count_from_slots(must_slots, test_slots)
        print(_format_plot_regression_count(plot_regression_count))

        out_i = None
        if output_path is not None:
            p     = pathlib.Path(output_path)
            out_i = str(p.with_stem(f"{p.stem}_{spec_def['tag']}"))

        task_args.append((
            spec_name, y_arr, x_arr, cm_arr, ct_arr,
            fe_arrs, cl_arrs, base_mask,
            controls_must_flat, must_slots, controls_test_flat, test_slots, se_kind,
            1,   # 手动/单规格路径可直接复用 _run_spec_task
        ))
        task_metas.append({
            "spec_name":    spec_name,
            "spec_display": spec_display,
            "spec_def":     spec_def,
            "fe_cols":      fe_cols,
            "clust_cols":   clust_cols,
            "title_suffix": spec_def["help"].format(**fmt),
            "out_i":        out_i,
            "vce_label":    "robust" if spec_def["vce"] == "robust" else None,
            "t0":           spec_t0,
        })

    if not task_args:
        return []

    # ── 阶段二：枚举规格（串行 / 单规格内层并行 / 多规格扁平化并行）──
    n_tasks = len(task_args)
    if n_tasks == 1:
        print(f"\n[枚举] 单个规格组合，内层并行枚举最多使用 {n_jobs} 核")
        single_args = (*task_args[0][:-1], n_jobs)
        enum_results = [_run_spec_task(single_args)]
    elif n_jobs <= 1:
        print(f"\n[枚举] 串行执行 {n_tasks} 个规格组合")
        enum_results = [_run_spec_task(a) for a in task_args]
    else:
        flat_chunk_tasks: list[tuple] = []
        spec_chunk_counts: dict[str, int] = {}
        spec_display_by_name = {meta["spec_name"]: meta["spec_display"] for meta in task_metas}
        for spec_args in task_args:
            (
                spec_name, y_arr, x_arr, cm_arr, ct_arr,
                fe_arrs, cl_arrs, base_mask,
                controls_must_i, must_slots_i, controls_test_i, test_slots_i, se_kind_i, _n_inner,
            ) = spec_args
            total_spec_count = _spec_count_from_slots(must_slots_i, test_slots_i)
            bit_ranges = _spec_bit_ranges(total_spec_count, n_jobs)
            spec_chunk_counts[spec_name] = len(bit_ranges)
            for bit_range in bit_ranges:
                flat_chunk_tasks.append((
                    spec_name, bit_range, y_arr, x_arr, cm_arr, ct_arr,
                    fe_arrs, cl_arrs, base_mask,
                    controls_must_i, must_slots_i, controls_test_i, test_slots_i, se_kind_i,
                ))

        total_chunks = len(flat_chunk_tasks)
        worker_count = min(n_jobs, total_chunks)
        print(
            f"\n[枚举] 扁平化并行执行 {n_tasks} 个规格组合"
            f"（{worker_count} 个 worker，{total_chunks} 个总任务块，合计 {n_jobs} 核上限）"
        )
        for spec_name, chunk_count in spec_chunk_counts.items():
            print(f"  - {spec_display_by_name[spec_name]}: {chunk_count} 个块")

        enum_records: dict[str, list[SpecRecord]] = {
            meta["spec_name"]: [] for meta in task_metas
        }
        completed_chunks = 0
        next_report = 1

        import signal as _sig
        ctx = _best_mp_context()
        _orig = _sig.signal(_sig.SIGINT, _sig.SIG_IGN)
        try:
            pool = ctx.Pool(processes=worker_count)
        finally:
            _sig.signal(_sig.SIGINT, _orig)
        try:
            for spec_name, chunk_records, skipped, _bit_range in pool.imap_unordered(
                _run_flat_spec_chunk, flat_chunk_tasks
            ):
                enum_records[spec_name].extend(chunk_records)
                for msg in skipped:
                    print(msg)
                completed_chunks += 1
                progress = completed_chunks * 100 // total_chunks
                if progress >= next_report or completed_chunks == total_chunks:
                    print(
                        f"  总进度：{completed_chunks}/{total_chunks} 块完成"
                        f"（约 {progress}%）"
                    )
                    next_report = min(100, progress + 10)
            pool.close()
        except BaseException:
            pool.terminate()
            raise
        finally:
            pool.join()

        enum_results = []
        for meta in task_metas:
            spec_name = meta["spec_name"]
            records = enum_records[spec_name]
            records.sort(key=lambda r: r["coef"])
            enum_results.append((spec_name, records))

    # ── 阶段三：主进程绘图 & 汇总（Figure 无需跨进程传输）────────
    for (_, records), meta in zip(enum_results, task_metas):
        spec_name = meta["spec_name"]
        spec_display = meta["spec_display"]
        print(f"\n{'='*60}")
        print(f"规格：{spec_display}")
        print("=" * 60)
        print(f"完成 {len(records)} 个规格，系数范围：[{records[0]['coef']:.4f}, {records[-1]['coef']:.4f}]")

        fig = None
        if render_plot:
            fig = _plot(
                records,
                y_name        = y,
                x_name        = x,
                controls_test = controls_test_flat,
                controls_must = controls_must_flat,
                matrix_controls = matrix_controls,
                show_special_markers = True,
                fig_width     = fig_width,
                dpi           = dpi,
                output_path   = meta["out_i"],
                title_suffix  = meta["title_suffix"],
                elapsed_seconds_preplot = perf_counter() - meta["t0"],
                sort_by_signed_p = sort_by_signed_p,
                matrix_swimlane_ranges = _compute_swimlane_ranges(matrix_controls, must_slots, test_slots) or None,
            )

        total_specs += len(records)
        all_sig_rows.extend(
            _build_sig_rows(
                records       = records,
                y             = y,
                x             = x,
                controls_must = controls_must_flat,
                controls_test = controls_test_flat,
                fe_cols       = meta["fe_cols"],
                clust_cols    = meta["clust_cols"],
                vce_label     = meta["vce_label"],
            )
        )
        results.append((spec_name, records, fig))

    if output_path is not None and export_sig_table:
        p = pathlib.Path(output_path)
        tbl_path = str(p.with_name(f"{p.stem}_sig.csv"))
        _export_sig_table(
            rows        = all_sig_rows,
            output_path = tbl_path,
            n_specs     = total_specs,
        )

    return results


# ────────────────────────────────────────────────────────────
# 显著性汇总表（内部）
# ────────────────────────────────────────────────────────────

def _build_sig_rows(
    records:     list[SpecRecord],
    y:           str,
    x:           str,
    controls_must: list[str],
    controls_test: list[str],
    fe_cols:     list[str],
    clust_cols:  list[str],
    vce_label:   str | None = None,
    grouping_variable: str | None = None,
    grouped_records: list[GroupedPlotRecord] | None = None,
) -> list[dict[str, Any]]:
    """
    提取所有 90% 及以上显著的规格行为行记录。

    列说明
    ------
    Star     : 按显著性和系数符号编码；例如 -3 = 99% 显著且系数为负
    coef     : 主变量系数估计值
    obs      : 该规格回归样本量
    Y        : 被解释变量
    X        : 主解释变量
    Controls : 该规格实际纳入的控制变量（含强制控制变量，逗号分隔）
    FE       : 固定效应列（逗号分隔）
    cluster  : 聚类变量列（逗号分隔）；若为 robust，则显示 "robust"
    """
    rows = []
    cluster_label = vce_label or (", ".join(clust_cols) if clust_cols else "robust")
    grouped_lookup: dict[tuple[str, ...], dict[int, GroupedPlotRecord]] = {}
    if grouping_variable and grouped_records:
        for grouped_record in grouped_records:
            key = _controls_key_from_values(grouped_record["controls_all"])
            grouped_lookup.setdefault(key, {})[int(grouped_record["group_value"])] = grouped_record
    for r in records:
        p_value = r["p_value"]
        if p_value < 0.01:
            star_abs = 3
        elif p_value < 0.05:
            star_abs = 2
        elif p_value < 0.10:
            star_abs = 1
        else:
            continue
        sign = -1 if r["coef"] < 0 else 1
        ordered_controls = [c for c in controls_must if c in r["controls_all"]]
        ordered_controls.extend(
            c for c in controls_test
            if c in r["controls_all"] and c not in controls_must
        )
        rows.append({
            "Star":     sign * star_abs,
            "coef":     round(r["coef"], 6),
            "p_value":  round(r["p_value"], 6),
            "t_value":  round(r["t_value"], 6),
            "obs":      r["obs"],
            "Y":        y,
            "X":        x,
            "Controls": ", ".join(ordered_controls) if ordered_controls else "(none)",
            "FE":       ", ".join(fe_cols),
            "cluster":  cluster_label,
        })
        if grouping_variable:
            grouped_pair = grouped_lookup.get(_controls_key_from_values(r["controls_all"]), {})
            rows[-1]["GroupingVariable"] = grouping_variable
            rows[-1]["Group0"] = _signed_star_label(_signed_star_value(grouped_pair.get(0)))
            rows[-1]["Group1"] = _signed_star_label(_signed_star_value(grouped_pair.get(1)))

    return rows


def _export_sig_table(
    rows:        list[dict],
    output_path: str,
    n_specs:     int,
    print_summary: bool = True,
) -> "pd.DataFrame | None":
    """将显著规格行写出为单张 CSV 汇总表。"""
    rows = [{
        **row,
        "GroupingVariable": row.get("GroupingVariable", ""),
        "Group0": row.get("Group0", ""),
        "Group1": row.get("Group1", ""),
        "Specs": n_specs,
    } for row in rows]

    if not rows:
        if print_summary:
            print("  [汇总表] 无 90% 及以上显著的规格，跳过导出")
        return None

    tbl = pd.DataFrame(rows)
    tbl = tbl.sort_values(["p_value"], ascending=[True], kind="stable")
    preferred_columns = [
        "Star", "coef", "p_value", "t_value", "obs", "Y", "X", "Controls", "FE", "cluster",
        "GroupingVariable", "Group0", "Group1", "Specs",
    ]
    tbl = tbl[[col for col in preferred_columns if col in tbl.columns]]
    tbl.to_csv(output_path, index=False, encoding="utf-8-sig")
    star_counts = {k: 0 for k in (3, 2, 1, -1, -2, -3)}
    for r in rows:
        star = int(r["Star"])
        if star in star_counts:
            star_counts[star] += 1
    if print_summary:
        print(f"  [汇总表] {len(rows)}/{n_specs} 个规格显著"
              f"（+3:{star_counts[3]}  +2:{star_counts[2]}  +1:{star_counts[1]}  "
              f"-1:{star_counts[-1]}  -2:{star_counts[-2]}  -3:{star_counts[-3]}）→ {output_path}")
    return tbl


def _sig_star_counts(rows: list[dict[str, Any]]) -> dict[int, int]:
    """按签名显著性星级汇总行记录。"""
    star_counts = {k: 0 for k in (3, 2, 1, -1, -2, -3)}
    for row in rows:
        star = int(row["Star"])
        if star in star_counts:
            star_counts[star] += 1
    return star_counts


def _format_sig_summary(n_sig: int, n_specs: int, star_counts: dict[int, int]) -> str:
    """格式化终端中的显著性汇总文本。"""
    if n_sig == 0:
        return "无 90% 及以上显著的规格"
    return (
        f"{n_sig}/{n_specs} 个规格显著"
        f"（+3:{star_counts[3]}  +2:{star_counts[2]}  +1:{star_counts[1]}  "
        f"-1:{star_counts[-1]}  -2:{star_counts[-2]}  -3:{star_counts[-3]}）"
    )


def _controls_key_from_values(controls_all: set[str] | list[str] | tuple[str, ...]) -> tuple[str, ...]:
    return tuple(sorted(controls_all))


def _signed_star_value(record: SpecRecord | GroupedPlotRecord | None) -> int:
    if record is None:
        return 0
    p_value = float(record["p_value"])
    if p_value < 0.01:
        star_abs = 3
    elif p_value < 0.05:
        star_abs = 2
    elif p_value < 0.10:
        star_abs = 1
    else:
        return 0
    sign = -1 if float(record["coef"]) < 0 else 1
    return sign * star_abs


def _signed_star_label(value: int) -> str:
    if value > 0:
        return f"+{value}"
    if value < 0:
        return str(value)
    return "0"


def _format_combo_summary_lines(combo_summaries: list[dict[str, Any]]) -> list[str]:
    """格式化所有 Y × X 汇总行，并对齐分隔符与汇总文本起点。"""
    if not combo_summaries:
        return []
    y_width = max(len(str(summary["y"])) for summary in combo_summaries)
    x_width = max(len(str(summary["x"])) for summary in combo_summaries)
    lines = []
    for summary in combo_summaries:
        sig_summary = _format_sig_summary(
            int(summary["n_sig"]),
            int(summary["n_specs"]),
            summary["star_counts"],
        )
        lines.append(
            f"Y = {str(summary['y']):<{y_width}}  ×  "
            f"X = {str(summary['x']):<{x_width}} {sig_summary}"
        )
    return lines


def _toml_literal(value) -> str:
    """将常见 Python 值转为简单 TOML 字面量。"""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int | float):
        return str(value)
    if isinstance(value, str):
        escaped = value.replace("\\", "\\\\").replace('"', '\\"')
        return f'"{escaped}"'
    if isinstance(value, list):
        return "[" + ", ".join(_toml_literal(v) for v in value) + "]"
    if value is None:
        return '""'
    escaped = str(value).replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def _write_config_snapshot(config: dict, output_path: pathlib.Path) -> None:
    """将本次运行的有效配置写为 TOML 快照。"""
    lines = ["# config_snapshot.toml", ""]
    for key, value in config.items():
        lines.append(f"{key} = {_toml_literal(value)}")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[Saved] {output_path}")


RESULT_COLUMNS = [
    "coef", "se", "t_value", "p_value", "adj_r2", "within_r2", "f_stat", "df_resid",
    "ci99_lo", "ci99_hi", "ci95_lo", "ci95_hi", "ci90_lo", "ci90_hi",
    "controls_test", "controls_all", "control_stats", "is_full", "obs",
]


def records_to_dataframe(records: list[SpecRecord]) -> pd.DataFrame:
    """Convert in-memory records to the standard CSV file contract."""
    rows: list[dict[str, Any]] = []
    for r in records:
        rows.append({
            "coef": r["coef"],
            "se": r["se"],
            "t_value": r["t_value"],
            "p_value": r["p_value"],
            "adj_r2": r.get("adj_r2", np.nan),
            "within_r2": r.get("within_r2", np.nan),
            "f_stat": r.get("f_stat", np.nan),
            "df_resid": r["df_resid"],
            "ci99_lo": r["ci99_lo"],
            "ci99_hi": r["ci99_hi"],
            "ci95_lo": r["ci95_lo"],
            "ci95_hi": r["ci95_hi"],
            "ci90_lo": r["ci90_lo"],
            "ci90_hi": r["ci90_hi"],
            "controls_test": json.dumps(sorted(r["controls_test"]), ensure_ascii=False),
            "controls_all": json.dumps(sorted(r["controls_all"]), ensure_ascii=False),
            "control_stats": json.dumps(r.get("control_stats", []), ensure_ascii=False),
            "is_full": bool(r["is_full"]),
            "obs": r["obs"],
        })
    return pd.DataFrame(rows, columns=RESULT_COLUMNS)


def records_from_dataframe(df: pd.DataFrame) -> list[SpecRecord]:
    """Load records from the standard CSV file contract."""
    records: list[SpecRecord] = []
    for _, row in df.iterrows():
        controls_test = set(json.loads(str(row["controls_test"])))
        controls_all = set(json.loads(str(row["controls_all"])))
        control_stats_raw = row.get("control_stats", "[]")
        try:
            control_stats = json.loads(str(control_stats_raw)) if not pd.isna(control_stats_raw) else []
        except json.JSONDecodeError:
            control_stats = []
        records.append({
            "coef": float(row["coef"]),
            "se": float(row["se"]),
            "t_value": float(row["t_value"]),
            "p_value": float(row["p_value"]),
            "adj_r2": float(row["adj_r2"]) if "adj_r2" in row.index and not pd.isna(row["adj_r2"]) else float("nan"),
            "within_r2": float(row["within_r2"]) if "within_r2" in row.index and not pd.isna(row["within_r2"]) else float("nan"),
            "f_stat": float(row["f_stat"]) if "f_stat" in row.index and not pd.isna(row["f_stat"]) else float("nan"),
            "df_resid": int(row["df_resid"]),
            "ci99_lo": float(row["ci99_lo"]),
            "ci99_hi": float(row["ci99_hi"]),
            "ci95_lo": float(row["ci95_lo"]),
            "ci95_hi": float(row["ci95_hi"]),
            "ci90_lo": float(row["ci90_lo"]),
            "ci90_hi": float(row["ci90_hi"]),
            "controls_test": controls_test,
            "controls_all": controls_all,
            "control_stats": control_stats if isinstance(control_stats, list) else [],
            "is_full": bool(row["is_full"]),
            "obs": int(row["obs"]),
        })
    return records


def write_analysis_artifacts(
    *,
    records: list[SpecRecord],
    results_path: pathlib.Path,
    meta_path: pathlib.Path,
    meta: dict[str, Any],
    verbose: bool = True,
) -> None:
    """Write the standard analysis-to-plot handoff files."""
    records_to_dataframe(records).to_csv(results_path, index=False, encoding="utf-8-sig")
    meta_path.write_text(
        json.dumps(meta, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if verbose:
        print(f"[Saved] {results_path}")
        print(f"[Saved] {meta_path}")


# ────────────────────────────────────────────────────────────
# 绘图（内部）
# ────────────────────────────────────────────────────────────

def _plot(records: list[SpecRecord], y_name: str, x_name: str,
          controls_test: list[str], controls_must: list[str],
          fig_width: float, dpi: int,
          output_path: str | None,
          matrix_controls: list[str] | None = None,
          show_special_markers: bool = True,
          title_suffix: str | None = None,
          elapsed_seconds_preplot: float | None = None,
          engine: str | None = None,
          grouping_variable: str | None = None,
          grouped_plot_records: list[GroupedPlotRecord] | None = None,
          interaction_plot_records: list[SpecRecord] | None = None,
          sort_by_signed_p: bool = False,
          verbose: bool = True,
          matrix_swimlane_ranges: list[tuple[int, int]] | None = None) -> Figure:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.lines import Line2D
    plot_t0 = perf_counter()
    records = _sort_records_for_plot(records, sort_by_signed_p=sort_by_signed_p)

    matrix_controls = matrix_controls if matrix_controls is not None else controls_test
    K_test = len(matrix_controls)
    K_total = K_test + len(controls_must)
    matrix_rows = K_test
    n = len(records)
    xs = np.arange(n)
    obs_arr = np.array([r["obs"] for r in records])

    coefs  = np.array([r["coef"]    for r in records])
    lo99   = np.array([r["ci99_lo"] for r in records])
    hi99   = np.array([r["ci99_hi"] for r in records])
    lo95   = np.array([r["ci95_lo"] for r in records])
    hi95   = np.array([r["ci95_hi"] for r in records])
    lo90   = np.array([r["ci90_lo"] for r in records])
    hi90   = np.array([r["ci90_hi"] for r in records])
    is_full = np.array([r["is_full"] for r in records])
    is_nocontrol = np.array([len(r["controls_test"]) == 0 for r in records])

    # ── 游程编码（黑色块深浅） ──────────────────────────
    runs: list[list[dict]] = []
    global_max_black = 1
    for k in range(K_test):
        row_runs: list[dict] = []
        start = 0
        inc = matrix_controls[k] in records[0]["controls_all"]
        for i in range(1, n + 1):
            cur = (matrix_controls[k] in records[i]["controls_all"]) if i < n else (not inc)
            if cur != inc:
                length = i - start
                row_runs.append({"start": start, "len": length, "inc": inc})
                if inc and length > global_max_black:
                    global_max_black = length
                start = i
                inc = cur
        runs.append(row_runs)

    def black_value(length: int) -> float:
        t = (length / global_max_black) ** 0.6
        v = round(200 - 200 * t)
        return v / 255.0

    # ── 布局 ────────────────────────────────────────────
    has_grouped_panel = bool(grouping_variable and grouped_plot_records is not None)
    has_interaction_panel = bool(has_grouped_panel and interaction_plot_records is not None)
    title_h = 1.55
    pval_h = 0.95
    upper_h = 3.8
    lower_h = max(1.2, matrix_rows * 0.30 + 0.55)
    stats_h = 0.45
    obs_h = 1.0
    group_ci_h = 1.7 if has_grouped_panel else 0.0
    interaction_h = 1.45 if has_interaction_panel else 0.0
    group_star_h = 0.95 if has_grouped_panel else 0.0
    group_obs_h = 1.0 if has_grouped_panel else 0.0

    height_ratios = [title_h, pval_h, upper_h, lower_h, stats_h, obs_h]
    if has_grouped_panel:
        height_ratios.append(group_ci_h)
        if has_interaction_panel:
            height_ratios.append(interaction_h)
        height_ratios.extend([group_star_h, group_obs_h])

    fig, fig_axes = plt.subplots(
        len(height_ratios), 1,
        figsize=(fig_width, sum(height_ratios)),
        facecolor="white",
        gridspec_kw={"height_ratios": height_ratios},
        layout="constrained",
        dpi=dpi,
    )
    axes = list(fig_axes if isinstance(fig_axes, np.ndarray) else [fig_axes])
    ax_title, ax_p, ax1, ax2, ax_stats, ax3 = axes[:6]
    ax_group = axes[6] if has_grouped_panel else None
    ax_interaction = axes[7] if has_interaction_panel else None
    star_idx = 8 if has_interaction_panel else 7
    obs_idx = 9 if has_interaction_panel else 8
    ax_group_star = axes[star_idx] if has_grouped_panel else None
    ax_group_obs = axes[obs_idx] if has_grouped_panel else None
    fig.set_constrained_layout_pads(h_pad=0.06, w_pad=0.04, hspace=0.03, wspace=0.02)

    # ── 上图：三层颜色区间 ───────────────────────────────
    ax1.axhline(0, color="#cc2222", lw=0.9, ls="--", zorder=10)
    ax1.set_facecolor("white")
    ax1.grid(axis="y", color="#eeeeee", lw=0.5, zorder=0)

    _CI99 = "#cccccc"
    _CI95 = "#888888"
    _CI90 = "#333333"

    ax1.fill_between(xs, lo99, hi99, color=_CI99, alpha=0.9, zorder=1, linewidth=0)
    ax1.fill_between(xs, lo95, hi95, color=_CI95, alpha=0.8, zorder=2, linewidth=0)
    ax1.fill_between(xs, lo90, hi90, color=_CI90, alpha=0.75, zorder=3, linewidth=0)

    ax1.autoscale()
    ax1.set_xlim(-0.5, n - 0.5)

    # ── 显著性颜色（按 t 统计量） ───────────────────────
    p_values = np.array([r["p_value"] for r in records])

    sig99  = p_values < 0.01                      # 3 stars → 红
    sig95  = (p_values < 0.05) & ~sig99           # 2 stars → Spring
    sig90  = (p_values < 0.10) & ~sig99 & ~sig95  # 1 star  → BlueBerry
    insig  = ~sig99 & ~sig95 & ~sig90             # 不显著   → 黑

    _C90  = "#0433FF"   # BlueBerry（1 star）
    _C95  = "#00F900"   # Spring（2 stars）
    _C99  = "#FF2600"   # 红（3 stars）
    _CINS = "#000000"   # 黑（不显著）
    _CFUL = "#FF2F92"   # Full controls 特殊规格竖线/外圈
    _CSTAR_NEG = _C90   # 负系数 Star 色块：BlueBerry
    _CSTAR_POS = _CFUL  # 正系数 Star 色块：红/玫红
    _CSTAR0 = "#000000"  # 黑（Star 面板中的不显著 0 线）
    _CNOC = "#ff8c00"   # 橙（无 test 控制变量规格外圈）
    _CSWITCH = "#2255cc"  # 蓝（最接近 0 的系数点）

    def _set_horizontal_ylabel(ax, label: str, *, fontsize: int = 8, labelpad: float = 18.0) -> None:
        ax.set_ylabel(label, fontsize=fontsize, rotation=0, ha="right", va="center", labelpad=labelpad)

    # 仅当系数序列跨过 0 时，才标记最接近 0 的单个规格点
    is_sign_switch = np.zeros(n, dtype=bool)
    if np.any(coefs < 0) and np.any(coefs > 0):
        is_sign_switch[int(np.argmin(np.abs(coefs)))] = True

    # ── Star 条形图：方向由系数符号决定，格数由显著性星级决定 ─────
    ax_p.set_facecolor("white")
    star_abs = np.where(
        p_values < 0.01,
        3,
        np.where(
            p_values < 0.05,
            2,
            np.where(p_values < 0.10, 1, 0),
        ),
    )
    p_gap = 0.16
    p_outer_pad_x = 0.12
    p_outer_pad_y = 0.18
    p_half_w = 0.5 - p_gap
    p_half_h = 0.33
    for i in range(n):
        _sp_full = show_special_markers and bool(is_full[i])
        _sp_noc = show_special_markers and bool(is_nocontrol[i])
        blocks = int(star_abs[i])
        if blocks == 0:
            tick_color = _CFUL if _sp_full else _CNOC if _sp_noc else _CSTAR0
            ax_p.plot(
                [i - p_half_w, i + p_half_w],
                [0.0, 0.0],
                color=tick_color,
                lw=1.0,
                solid_capstyle="butt",
                zorder=4,
            )
            continue
        direction = -1 if coefs[i] < 0 else 1
        color_code = _CFUL if _sp_full else _CNOC if _sp_noc else (_CSTAR_NEG if coefs[i] < 0 else _CSTAR_POS)
        for block_idx in range(blocks):
            y_center = direction * (block_idx + 0.68)
            ax_p.add_patch(
                mpatches.Rectangle(
                    (i - p_half_w, y_center - p_half_h),
                    2 * p_half_w,
                    2 * p_half_h,
                    facecolor=color_code,
                    edgecolor="none",
                    linewidth=0.0,
                    zorder=2,
                )
            )
    ax_p.axhline(0, color="#111111", lw=0.8, ls="-", zorder=1)
    if show_special_markers and is_full.any():
        for fi in np.where(is_full)[0]:
            ax_p.axvline(fi, color=_CFUL, lw=1.1, ls="-", zorder=5)
    if show_special_markers and is_nocontrol.any():
        for ni in np.where(is_nocontrol)[0]:
            ax_p.axvline(ni, color=_CNOC, lw=1.1, ls="-", zorder=5)
    for si in np.where(is_sign_switch)[0]:
        ax_p.axvline(si, color=_CSWITCH, lw=1.1, ls="-", zorder=5)
    ax_p.set_xlim(-0.5 - p_outer_pad_x, n - 0.5 + p_outer_pad_x)
    ax_p.set_ylim(-3.0 - p_outer_pad_y, 3.0 + p_outer_pad_y)
    _set_horizontal_ylabel(ax_p, "Star", fontsize=8)
    ax_p.set_yticks([-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0])
    ax_p.set_yticklabels(["-3", "-2", "-1", "0", "+1", "+2", "+3"], fontfamily="monospace")
    ax_p.tick_params(axis="y", labelsize=8)
    ax_p.tick_params(axis="x", bottom=False, labelbottom=False)
    ax_p.spines[["top", "right", "left", "bottom"]].set_visible(True)
    for spine in ax_p.spines.values():
        spine.set_color("#000000")
        spine.set_linewidth(0.8)

    # 特殊规格保留竖线标记：仅对 full / no-controls_test 绘制
    if show_special_markers and is_full.any():
        ax1.vlines(xs[is_full], lo99[is_full], hi99[is_full],
                   colors=_CFUL, linewidth=1.1, zorder=5)
    if show_special_markers and is_nocontrol.any():
        ax1.vlines(xs[is_nocontrol], lo99[is_nocontrol], hi99[is_nocontrol],
                   colors=_CNOC, linewidth=1.1, zorder=5.5)
    if is_sign_switch.any():
        ax1.vlines(xs[is_sign_switch], lo99[is_sign_switch], hi99[is_sign_switch],
                   colors=_CSWITCH, linewidth=1.1, zorder=5.25)

    draw_order = [(insig, _CINS), (sig90, _C90), (sig95, _C95), (sig99, _C99)]

    # 按显著性分组绘制普通点（非全变量规格）：
    # 先画低显著性，后画高显著性，让更显著的点覆盖在上面。
    for mask, color in draw_order:
        m = mask & ~is_full
        if m.any():
            ax1.scatter(xs[m], coefs[m], s=6, color=color, zorder=6, lw=0)

    _SPECIAL_S = 54
    _SPECIAL_LW = 1.2

    # 无控制变量规格：保留显著性填色，仅用白色描边高亮
    for mask, color in draw_order:
        m = mask & is_nocontrol & show_special_markers
        if m.any():
            ax1.scatter(
                xs[m],
                coefs[m],
                s=_SPECIAL_S,
                color=color,
                edgecolors=_CNOC,
                linewidths=_SPECIAL_LW,
                zorder=8,
            )

    # 全变量规格：最后绘制，确保覆盖在普通点与其他特殊点之上
    for mask, color in draw_order:
        m = mask & is_full & show_special_markers
        if m.any():
            ax1.scatter(xs[m], coefs[m], s=_SPECIAL_S, color=color,
                        edgecolors=_CFUL, linewidths=_SPECIAL_LW, zorder=10)

    switch_color = _CINS
    if is_sign_switch.any():
        switch_idx = int(np.flatnonzero(is_sign_switch)[0])
        if sig99[switch_idx]:
            switch_color = _C99
        elif sig95[switch_idx]:
            switch_color = _C95
        elif sig90[switch_idx]:
            switch_color = _C90

    # 最接近 0 的系数点：样式与 No controls_test 一致，仅边框改为蓝色
    if is_sign_switch.any():
        ax1.scatter(
            xs[is_sign_switch],
            coefs[is_sign_switch],
            s=_SPECIAL_S,
            color=switch_color,
            edgecolors=_CSWITCH,
            linewidths=_SPECIAL_LW,
            zorder=9,
        )

    _set_horizontal_ylabel(ax1, "Coef.", fontsize=9)
    ax1.tick_params(axis="y", labelsize=8)
    ax1.tick_params(axis="x", bottom=False, labelbottom=False)
    ax1.spines[["top", "right", "left", "bottom"]].set_visible(True)
    for spine in ax1.spines.values():
        spine.set_color("#000000")
        spine.set_linewidth(0.8)

    legend_elems = [
        mpatches.Patch(facecolor=_CI99, edgecolor="none", label="99% CI"),
        mpatches.Patch(facecolor=_CI95, edgecolor="none", label="95% CI"),
        mpatches.Patch(facecolor=_CI90, edgecolor="none", label="90% CI"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=_C99,
               markersize=7, label="p<0.01"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=_C95,
               markersize=7, label="p<0.05"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=_C90,
               markersize=7, label="p<0.10"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=_CINS,
               markersize=7, label="n.s."),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#777777",
               markeredgecolor=_CSWITCH, markeredgewidth=_SPECIAL_LW,
               markersize=9, label="Closest to zero"),
    ]
    if show_special_markers:
        legend_elems.insert(
            7,
            Line2D([0], [0], marker="o", color="w", markerfacecolor="#777777",
                   markeredgecolor=_CFUL, markeredgewidth=_SPECIAL_LW,
                   markersize=9, label="Full controls"),
        )
        legend_elems.insert(
            8,
            Line2D([0], [0], marker="o", color="w", markerfacecolor="#777777",
                   markeredgecolor=_CNOC, markeredgewidth=_SPECIAL_LW,
                   markersize=9, label="No controls_test"),
        )
    ax1.legend(handles=legend_elems, fontsize=7.5, frameon=True,
               loc="lower center", bbox_to_anchor=(0.5, -0.08), ncol=len(legend_elems),
               columnspacing=0.9, handletextpad=0.4, borderaxespad=0.0,
               facecolor="#eeeeee", edgecolor="none")
    spec_line = title_suffix.split(" - ", 1)[0] if title_suffix else None
    elapsed_total = (elapsed_seconds_preplot or 0.0) + (perf_counter() - plot_t0)
    engine_line = f"  |  engine = {engine}" if engine else ""
    title_lines = [
        "Regression Monkey",
        f"Y = {y_name}  |  X = {x_name}",
        f"specs = {n}  |  controls = {K_total}{engine_line}",
        spec_line or "",
        _wrap_title_line("controls_must", list(controls_must)),
        f"grouping_variable = {grouping_variable}" if has_grouped_panel and grouping_variable else "",
        f"Elapsed = {elapsed_total:.2f}s  |  @Lachryz",
    ]
    _title = "\n".join(line for line in title_lines if line)
    ax_title.set_axis_off()
    title_text = ax_title.text(
        0.5, 0.52,
        _title,
        ha="center", va="center",
        fontsize=10.5, fontweight="bold",
        linespacing=1.25,
        transform=ax_title.transAxes,
    )

    # ── 下图：黑白方格矩阵（游程合并，黑块深浅） ────────
    ax2.set_facecolor("white")
    ax2.set_xlim(-0.5, n - 0.5)
    ax2.set_ylim(-0.5, matrix_rows - 0.5)
    ax2.set_yticks(range(matrix_rows))
    ax2.set_yticklabels(matrix_controls[::-1], fontsize=8)
    ax2.tick_params(axis="y", length=0, pad=4)
    ax2.tick_params(axis="x", bottom=False, labelbottom=False)
    ax2.spines[["top", "right", "left", "bottom"]].set_visible(True)
    for spine in ax2.spines.values():
        spine.set_color("#000000")
        spine.set_linewidth(0.8)

    # 先栅格化控制变量矩阵，再一次性绘制，避免大量 barh patch 拖慢首轮渲染
    if K_test > 0:
        _SWIM_COLORS = ["#0B3A75", "#14532D", "#7F1D1D", "#581C87", "#7C2D12", "#164E63"]
        if matrix_swimlane_ranges:
            for gi, (r0, r1) in enumerate(matrix_swimlane_ranges):
                y_lo = matrix_rows - 1 - r1 - 0.5
                y_hi = matrix_rows - 1 - r0 + 0.5
                ax2.axhspan(y_lo, y_hi, alpha=0.12, color=_SWIM_COLORS[gi % len(_SWIM_COLORS)], zorder=0)

        # RGBA grid：非选中格透明（泳道背景可透出），选中格不透明并按特殊规格着色
        _h2rgb = lambda h: tuple(int(h.lstrip("#")[i:i+2], 16) / 255 for i in (0, 2, 4))
        _CFUL_rgb = _h2rgb(_CFUL)
        _CNOC_rgb = _h2rgb(_CNOC)
        grid_rgba = np.zeros((matrix_rows, n, 4), dtype=float)  # 全透明起点
        for k_idx, row_runs in enumerate(runs):
            row_ax = matrix_rows - 1 - k_idx
            for run in row_runs:
                if not run["inc"]:
                    continue
                x0 = run["start"]
                x1 = x0 + run["len"]
                g = black_value(run["len"])
                grid_rgba[row_ax, x0:x1, :3] = g
                grid_rgba[row_ax, x0:x1, 3] = 1.0
        if show_special_markers:
            for sp_mask, sp_rgb in [(is_full, _CFUL_rgb), (is_nocontrol, _CNOC_rgb)]:
                for col in np.where(sp_mask)[0]:
                    inc_rows = grid_rgba[:, col, 3] == 1.0
                    grid_rgba[inc_rows, col, :3] = sp_rgb

        ax2.imshow(
            grid_rgba,
            aspect="auto",
            interpolation="nearest",
            origin="lower",
            extent=(-0.5, n - 0.5, -0.5, matrix_rows - 0.5),
            zorder=1,
        )

        # 各控制变量行之间添加浅灰分隔线
        for y_sep in range(matrix_rows - 1):
            ax2.axhline(
                y_sep + 0.5,
                color="#000000",
                lw=0.6,
                zorder=2,
            )

    # Full controls 特殊规格竖线
    if show_special_markers:
        for fi in np.where(is_full)[0]:
            ax2.axvline(fi, color=_CFUL, lw=1.1, ls="-", zorder=3)
        for ni in np.where(is_nocontrol)[0]:
            ax2.axvline(ni, color=_CNOC, lw=1.1, ls="-", zorder=3)
    for si in np.where(is_sign_switch)[0]:
        ax2.axvline(si, color=_CSWITCH, lw=1.1, ls="-", zorder=3)

    ax2.set_xlabel("Specifications", fontsize=8)
    ax2.tick_params(axis="x", labelsize=8)

    # ── 描述性统计横栏 ──────────────────────────────────
    _fmt = lambda v: f"{v:,.0f}" if v >= 100 else f"{v:.2f}"
    _pcts = np.percentile(obs_arr, [1, 25, 50, 75, 99])
    _stats = [
        ("Mean",   np.mean(obs_arr)),
        ("Std",    np.std(obs_arr)),
        ("Min",    np.min(obs_arr)),
        ("1%",     _pcts[0]),
        ("25%",    _pcts[1]),
        ("Median", _pcts[2]),
        ("75%",    _pcts[3]),
        ("99%",    _pcts[4]),
        ("Max",    np.max(obs_arr)),
    ]
    stat_text = "   ".join(f"{lbl}: {_fmt(val)}" for lbl, val in _stats)
    ax_stats.text(
        0.5, 0.5, stat_text,
        ha="center", va="center",
        fontsize=10, color="#444444",
        transform=ax_stats.transAxes,
    )
    ax_stats.set_axis_off()

    # ── 下下图：obs 条形图（原始取值） ─────────────────────
    ax3.set_facecolor("white")
    obs_mean = float(np.mean(obs_arr))
    _COBS = "#9CA3AF"
    ax3.bar(
        xs, obs_arr - obs_mean, bottom=obs_mean, width=1.0,
        color=_COBS, edgecolor=_COBS, alpha=0.35, linewidth=0.25,
        antialiased=True, snap=False, zorder=2, align="center",
    )
    if show_special_markers:
        for sp_mask, sp_color in [(is_full, _CFUL), (is_nocontrol, _CNOC)]:
            sp_idx = np.where(sp_mask)[0]
            if sp_idx.size:
                ax3.bar(
                    xs[sp_idx], (obs_arr - obs_mean)[sp_idx], bottom=obs_mean, width=1.0,
                    color=sp_color, edgecolor=sp_color, alpha=1.0, linewidth=0.25,
                    antialiased=True, snap=False, zorder=3, align="center",
                )
    ax3.axhline(obs_mean, color="#444444", lw=0.8, ls="-", zorder=3)
    if show_special_markers:
        for fi in np.where(is_full)[0]:
            ax3.axvline(fi, color=_CFUL, lw=1.1, ls="-", zorder=4)
        for ni in np.where(is_nocontrol)[0]:
            ax3.axvline(ni, color=_CNOC, lw=1.1, ls="-", zorder=4)
    for si in np.where(is_sign_switch)[0]:
        ax3.axvline(si, color=_CSWITCH, lw=1.1, ls="-", zorder=4)
    ax3.set_xlim(-0.5, n - 0.5)
    _set_horizontal_ylabel(ax3, "Obs.", fontsize=8)
    obs_min = float(np.min(obs_arr))
    obs_max = float(np.max(obs_arr))
    ax3.set_yticks([obs_min, obs_mean, obs_max])
    ax3.set_yticklabels([f"{obs_min:.2f}", f"{obs_mean:.2f}", f"{obs_max:.2f}"])
    ax3.tick_params(axis="y", labelsize=8)
    ax3.tick_params(axis="x", bottom=False, labelbottom=False)
    ax3.grid(axis="y", color="#eeeeee", lw=0.5, zorder=0)
    ax3.spines[["top", "right", "left", "bottom"]].set_visible(True)
    for spine in ax3.spines.values():
        spine.set_color("#000000")
        spine.set_linewidth(0.8)

    if has_grouped_panel and ax_group is not None and ax_group_star is not None and ax_group_obs is not None:
        _CG0 = "#FF5FA2"
        _CG1 = "#8ED8FF"
        _CGZERO = "#39FF14"
        grouped_by_controls: dict[tuple[str, ...], dict[int, GroupedPlotRecord]] = {}
        for grouped_record in grouped_plot_records or []:
            key = _controls_key_from_values(grouped_record["controls_all"])
            grouped_by_controls.setdefault(key, {})[int(grouped_record["group_value"])] = grouped_record

        group0_coef = np.full(n, np.nan)
        group1_coef = np.full(n, np.nan)
        group0_lo99 = np.full(n, np.nan)
        group0_hi99 = np.full(n, np.nan)
        group0_lo = np.full(n, np.nan)
        group0_hi = np.full(n, np.nan)
        group0_lo90 = np.full(n, np.nan)
        group0_hi90 = np.full(n, np.nan)
        group1_lo99 = np.full(n, np.nan)
        group1_hi99 = np.full(n, np.nan)
        group1_lo = np.full(n, np.nan)
        group1_hi = np.full(n, np.nan)
        group1_lo90 = np.full(n, np.nan)
        group1_hi90 = np.full(n, np.nan)
        group0_obs = np.zeros(n)
        group1_obs = np.zeros(n)
        star_all = np.array([float(_signed_star_value(record)) for record in records], dtype=float)
        star_group0 = np.full(n, np.nan)
        star_group1 = np.full(n, np.nan)

        for idx, record in enumerate(records):
            pair = grouped_by_controls.get(_controls_key_from_values(record["controls_all"]), {})
            if 0 in pair:
                group0 = pair[0]
                group0_coef[idx] = float(group0["coef"])
                group0_lo99[idx] = float(group0["ci99_lo"])
                group0_hi99[idx] = float(group0["ci99_hi"])
                group0_lo[idx] = float(group0["ci95_lo"])
                group0_hi[idx] = float(group0["ci95_hi"])
                group0_lo90[idx] = float(group0["ci90_lo"])
                group0_hi90[idx] = float(group0["ci90_hi"])
                group0_obs[idx] = float(group0["obs"])
                star_group0[idx] = float(_signed_star_value(group0))
            if 1 in pair:
                group1 = pair[1]
                group1_coef[idx] = float(group1["coef"])
                group1_lo99[idx] = float(group1["ci99_lo"])
                group1_hi99[idx] = float(group1["ci99_hi"])
                group1_lo[idx] = float(group1["ci95_lo"])
                group1_hi[idx] = float(group1["ci95_hi"])
                group1_lo90[idx] = float(group1["ci90_lo"])
                group1_hi90[idx] = float(group1["ci90_hi"])
                group1_obs[idx] = float(group1["obs"])
                star_group1[idx] = float(_signed_star_value(group1))

        ax_group.set_facecolor("white")
        ax_group.axhline(0, color=_CGZERO, lw=0.9, ls="--", zorder=8)
        valid0 = ~np.isnan(group0_coef)
        valid1 = ~np.isnan(group1_coef)
        if valid0.any():
            x0 = xs[valid0]
            ax_group.fill_between(x0, group0_lo99[valid0], group0_hi99[valid0], color=_CG0, alpha=0.14, zorder=1, linewidth=0)
            ax_group.fill_between(x0, group0_lo[valid0], group0_hi[valid0], color=_CG0, alpha=0.22, zorder=2, linewidth=0)
            ax_group.fill_between(x0, group0_lo90[valid0], group0_hi90[valid0], color=_CG0, alpha=0.32, zorder=3, linewidth=0)
            ax_group.plot(x0, group0_lo99[valid0], color=_CG0, lw=1.0, alpha=0.95, zorder=4)
            ax_group.plot(x0, group0_hi99[valid0], color=_CG0, lw=1.0, alpha=0.95, zorder=4)
            ax_group.scatter(x0, group0_coef[valid0], s=10, color=_CG0, zorder=5, lw=0)
        if valid1.any():
            x1 = xs[valid1]
            ax_group.fill_between(x1, group1_lo99[valid1], group1_hi99[valid1], color=_CG1, alpha=0.14, zorder=1, linewidth=0)
            ax_group.fill_between(x1, group1_lo[valid1], group1_hi[valid1], color=_CG1, alpha=0.22, zorder=2, linewidth=0)
            ax_group.fill_between(x1, group1_lo90[valid1], group1_hi90[valid1], color=_CG1, alpha=0.32, zorder=3, linewidth=0)
            ax_group.plot(x1, group1_lo99[valid1], color=_CG1, lw=1.0, alpha=0.95, zorder=4)
            ax_group.plot(x1, group1_hi99[valid1], color=_CG1, lw=1.0, alpha=0.95, zorder=4)
            ax_group.scatter(x1, group1_coef[valid1], s=10, color=_CG1, zorder=5, lw=0)
        if show_special_markers:
            for fi in np.where(is_full)[0]:
                ax_group.axvline(fi, color=_CFUL, lw=1.1, ls="-", zorder=7)
            for ni in np.where(is_nocontrol)[0]:
                ax_group.axvline(ni, color=_CNOC, lw=1.1, ls="-", zorder=7)
        for si in np.where(is_sign_switch)[0]:
            ax_group.axvline(si, color=_CSWITCH, lw=1.1, ls="-", zorder=7)
        ax_group.set_xlim(-0.5, n - 0.5)
        _set_horizontal_ylabel(ax_group, "Grouped\nCoef.", fontsize=8, labelpad=26)
        ax_group.tick_params(axis="y", labelsize=8)
        ax_group.tick_params(axis="x", bottom=False, labelbottom=False)
        ax_group.legend(
            handles=[
                Line2D([0], [0], color=_CG0, lw=6, alpha=0.32, label="Group 0"),
                Line2D([0], [0], color=_CG1, lw=6, alpha=0.32, label="Group 1"),
            ],
            fontsize=7.5,
            frameon=True,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.18),
            ncol=2,
            facecolor="#eeeeee",
            edgecolor="none",
        )
        ax_group.spines[["top", "right", "left", "bottom"]].set_visible(True)
        for spine in ax_group.spines.values():
            spine.set_color("#000000")
            spine.set_linewidth(0.8)

        if has_interaction_panel and ax_interaction is not None:
            interaction_by_controls: dict[tuple[str, ...], SpecRecord] = {
                _controls_key_from_values(record["controls_all"]): record
                for record in interaction_plot_records or []
            }
            interaction_coef = np.full(n, np.nan)
            interaction_lo99 = np.full(n, np.nan)
            interaction_hi99 = np.full(n, np.nan)
            interaction_lo = np.full(n, np.nan)
            interaction_hi = np.full(n, np.nan)
            interaction_lo90 = np.full(n, np.nan)
            interaction_hi90 = np.full(n, np.nan)
            for idx, record in enumerate(records):
                interaction_record = interaction_by_controls.get(_controls_key_from_values(record["controls_all"]))
                if interaction_record is None:
                    continue
                interaction_coef[idx] = float(interaction_record["coef"])
                interaction_lo99[idx] = float(interaction_record["ci99_lo"])
                interaction_hi99[idx] = float(interaction_record["ci99_hi"])
                interaction_lo[idx] = float(interaction_record["ci95_lo"])
                interaction_hi[idx] = float(interaction_record["ci95_hi"])
                interaction_lo90[idx] = float(interaction_record["ci90_lo"])
                interaction_hi90[idx] = float(interaction_record["ci90_hi"])

            valid_interaction = ~np.isnan(interaction_coef)
            ax_interaction.set_facecolor("white")
            ax_interaction.axhline(0, color=_CGZERO, lw=0.9, ls="--", zorder=8)
            if valid_interaction.any():
                xi = xs[valid_interaction]
                ax_interaction.fill_between(xi, interaction_lo99[valid_interaction], interaction_hi99[valid_interaction], color="#7B3FF2", alpha=0.14, zorder=1, linewidth=0)
                ax_interaction.fill_between(xi, interaction_lo[valid_interaction], interaction_hi[valid_interaction], color="#7B3FF2", alpha=0.22, zorder=2, linewidth=0)
                ax_interaction.fill_between(xi, interaction_lo90[valid_interaction], interaction_hi90[valid_interaction], color="#7B3FF2", alpha=0.32, zorder=3, linewidth=0)
                ax_interaction.plot(xi, interaction_lo99[valid_interaction], color="#7B3FF2", lw=1.0, alpha=0.95, zorder=4)
                ax_interaction.plot(xi, interaction_hi99[valid_interaction], color="#7B3FF2", lw=1.0, alpha=0.95, zorder=4)
                ax_interaction.scatter(xi, interaction_coef[valid_interaction], s=10, color="#7B3FF2", zorder=5, lw=0)
            if show_special_markers:
                for fi in np.where(is_full)[0]:
                    ax_interaction.axvline(fi, color=_CFUL, lw=1.1, ls="-", zorder=7)
                for ni in np.where(is_nocontrol)[0]:
                    ax_interaction.axvline(ni, color=_CNOC, lw=1.1, ls="-", zorder=7)
            for si in np.where(is_sign_switch)[0]:
                ax_interaction.axvline(si, color=_CSWITCH, lw=1.1, ls="-", zorder=7)
            ax_interaction.set_xlim(-0.5, n - 0.5)
            _set_horizontal_ylabel(ax_interaction, "Interaction\nCoef.", fontsize=8, labelpad=34)
            ax_interaction.tick_params(axis="y", labelsize=8)
            ax_interaction.tick_params(axis="x", bottom=False, labelbottom=False)
            ax_interaction.legend(
                handles=[Line2D([0], [0], color="#7B3FF2", lw=6, alpha=0.32, label="c.X#c.Z")],
                fontsize=7.5,
                frameon=True,
                loc="upper center",
                bbox_to_anchor=(0.5, -0.18),
                ncol=1,
                facecolor="#eeeeee",
                edgecolor="none",
            )
            ax_interaction.spines[["top", "right", "left", "bottom"]].set_visible(True)
            for spine in ax_interaction.spines.values():
                spine.set_color("#000000")
                spine.set_linewidth(0.8)

        ax_group_star.set_facecolor("white")
        ax_group_star.axhline(0, color=_CGZERO, lw=0.9, ls="--", zorder=0)
        ax_group_star.plot(
            xs,
            star_all,
            color="#111111",
            lw=2.8,
            alpha=0.95,
            drawstyle="steps-mid",
            zorder=0.6,
        )
        if np.any(~np.isnan(star_group1)):
            ax_group_star.plot(
                xs,
                star_group1,
                color="#2E6BFF",
                lw=2.2,
                alpha=0.95,
                drawstyle="steps-mid",
                zorder=0.8,
            )
        if np.any(~np.isnan(star_group0)):
            ax_group_star.plot(
                xs,
                star_group0,
                color="#FF2F5E",
                lw=1.2,
                alpha=0.98,
                drawstyle="steps-mid",
                zorder=1.0,
            )
        if show_special_markers:
            for fi in np.where(is_full)[0]:
                ax_group_star.axvline(fi, color=_CFUL, lw=1.1, ls="-", zorder=5)
            for ni in np.where(is_nocontrol)[0]:
                ax_group_star.axvline(ni, color=_CNOC, lw=1.1, ls="-", zorder=5)
        for si in np.where(is_sign_switch)[0]:
            ax_group_star.axvline(si, color=_CSWITCH, lw=1.1, ls="-", zorder=5)
        ax_group_star.set_xlim(-0.5, n - 0.5)
        ax_group_star.set_ylim(-3.3, 3.3)
        _set_horizontal_ylabel(ax_group_star, "Star", fontsize=8)
        ax_group_star.set_yticks([-3, -2, -1, 0, 1, 2, 3])
        ax_group_star.set_yticklabels(["-3", "-2", "-1", "0", "+1", "+2", "+3"], fontfamily="monospace")
        ax_group_star.tick_params(axis="y", labelsize=8)
        ax_group_star.tick_params(axis="x", bottom=False, labelbottom=False)
        ax_group_star.legend(
            handles=[
                Line2D([0], [0], color="#111111", lw=2.8, label="Star all"),
                Line2D([0], [0], color="#2E6BFF", lw=2.2, label="Star 1"),
                Line2D([0], [0], color="#FF2F5E", lw=1.2, label="Star 0"),
            ],
            fontsize=7.5,
            frameon=True,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.18),
            ncol=3,
            facecolor="#eeeeee",
            edgecolor="none",
        )
        ax_group_star.spines[["top", "right", "left", "bottom"]].set_visible(True)
        for spine in ax_group_star.spines.values():
            spine.set_color("#000000")
            spine.set_linewidth(0.8)

        ax_group_obs.set_facecolor("white")
        ax_group_obs.axhline(0, color=_CGZERO, lw=0.8, ls="--", zorder=0)
        if np.any(group0_obs > 0):
            ax_group_obs.bar(xs, group0_obs, width=1.0, color=_CG0, edgecolor=_CG0,
                             alpha=0.35, linewidth=0.25, zorder=2, align="center")
        if np.any(group1_obs > 0):
            ax_group_obs.bar(xs, -group1_obs, width=1.0, color=_CG1, edgecolor=_CG1,
                             alpha=0.35, linewidth=0.25, zorder=2, align="center")
        if show_special_markers:
            for sp_mask, sp_color in [(is_full, _CFUL), (is_nocontrol, _CNOC)]:
                sp_idx = np.where(sp_mask)[0]
                if sp_idx.size:
                    if np.any(group0_obs[sp_idx] > 0):
                        ax_group_obs.bar(xs[sp_idx], group0_obs[sp_idx], width=1.0,
                                         color=sp_color, edgecolor=sp_color,
                                         alpha=1.0, linewidth=0.25, zorder=3, align="center")
                    if np.any(group1_obs[sp_idx] > 0):
                        ax_group_obs.bar(xs[sp_idx], -group1_obs[sp_idx], width=1.0,
                                         color=sp_color, edgecolor=sp_color,
                                         alpha=1.0, linewidth=0.25, zorder=3, align="center")
            for fi in np.where(is_full)[0]:
                ax_group_obs.axvline(fi, color=_CFUL, lw=1.1, ls="-", zorder=4)
            for ni in np.where(is_nocontrol)[0]:
                ax_group_obs.axvline(ni, color=_CNOC, lw=1.1, ls="-", zorder=4)
        for si in np.where(is_sign_switch)[0]:
            ax_group_obs.axvline(si, color=_CSWITCH, lw=1.1, ls="-", zorder=4)
        max_group_obs = max(
            float(np.max(group0_obs)) if group0_obs.size else 0.0,
            float(np.max(group1_obs)) if group1_obs.size else 0.0,
            1.0,
        )
        ax_group_obs.set_ylim(-max_group_obs * 1.1, max_group_obs * 1.1)
        ax_group_obs.set_xlim(-0.5, n - 0.5)
        _set_horizontal_ylabel(ax_group_obs, "Grouped\nObs.", fontsize=8, labelpad=26)
        ax_group_obs.set_yticks([-max_group_obs, 0.0, max_group_obs])
        ax_group_obs.set_yticklabels([f"{max_group_obs:.0f}", "0", f"{max_group_obs:.0f}"])
        ax_group_obs.tick_params(axis="y", labelsize=8)
        ax_group_obs.tick_params(axis="x", bottom=False, labelbottom=False)
        ax_group_obs.spines[["top", "right", "left", "bottom"]].set_visible(True)
        for spine in ax_group_obs.spines.values():
            spine.set_color("#000000")
            spine.set_linewidth(0.8)

    if output_path:
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="white")
        elapsed_total = (elapsed_seconds_preplot or 0.0) + (perf_counter() - plot_t0)
        title_lines[-1] = f"Elapsed = {elapsed_total:.2f}s  |  @Lachryz"
        title_text.set_text("\n".join(title_lines))
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="white")
        if verbose:
            print(f"[Saved] {output_path}")

    plt.close(fig)
    return fig


# ────────────────────────────────────────────────────────────
# CLI 入口
# ────────────────────────────────────────────────────────────

def main() -> None:
    import argparse, itertools, sys

    try:
        import tomllib
    except ImportError:
        import tomli as tomllib

    # ── 预读 TOML 配置文件 ────────────────────────────────────
    # 优先级：命令行显式传入的路径 > 脚本同目录下的 regression_monkey_config.toml
    _toml_cfg: dict = {}
    _cli_args = sys.argv[1:]
    _script_dir = pathlib.Path(__file__).parent
    _default_config = _script_dir / "regression_monkey_config.toml"

    if _cli_args and pathlib.Path(_cli_args[0]).suffix == ".toml":
        _config_path = pathlib.Path(_cli_args.pop(0))
        if not _config_path.exists():
            print(f"错误：配置文件不存在：{_config_path}", file=sys.stderr)
            sys.exit(1)
        with open(_config_path, "rb") as _f:
            _toml_cfg = tomllib.load(_f)
        print(f"[配置] 加载：{_config_path}")
    elif _default_config.exists():
        with open(_default_config, "rb") as _f:
            _toml_cfg = tomllib.load(_f)
        print(f"[配置] 加载默认配置：{_default_config}")

    # ── 构建 ArgumentParser ───────────────────────────────────
    parser = argparse.ArgumentParser(
        prog="regression_monkey",
        description="规格曲线分析（Specification Curve Analysis）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例（TOML 配置文件，推荐）：
  uv run regression_monkey.py                              # 自动加载同目录 regression_monkey_config.toml
  uv run regression_monkey.py my.toml                      # 指定配置文件
  uv run regression_monkey.py my.toml --dpi 600 --n-jobs 0 # 命令行参数覆盖配置（自动并行，最多 9 核）

示例（自动模式，按规格标志选择 FE+聚类组合）：
  uv run regression_monkey.py --data panel.dta \\
      --y ln_spread --x treat \\
      --controls-test size lev roa age tobinq \\
      --controls-must turnover cashflow \\
      --Firm-FE code --Ind-FE ind --Time-FE year --Region-FE pref \\
      --absorb-firm-time-vce-cluster-firm --absorb-firm-indtime-vce-cluster-firm

示例（手动模式，自定义 FE 列和聚类列）：
  uv run regression_monkey.py --data panel.dta \\
      --y ln_spread --x treat \\
      --controls-test size lev roa age tobinq \\
      --fe ind year --clust firm
""",
    )

    # ── 数据与变量 ──────────────────────────────────────
    parser.add_argument("--data",     metavar="FILE",
                        help=".dta / .csv / .parquet 数据文件路径")
    parser.add_argument("--y",        metavar="VAR", nargs="+",
                        help="被解释变量列名（可传多个，与 --x 两两组合）")
    parser.add_argument("--x",        metavar="VAR", nargs="+",
                        help="主解释变量列名（可传多个，与 --y 两两组合）")
    parser.add_argument("--controls", metavar="VAR", nargs="+",
                        help="兼容旧配置；等价于 --controls-test")
    parser.add_argument("--controls-test", dest="controls_test", metavar="VAR", nargs="+",
                        help="参与组合枚举的控制变量列名（CLI 传平铺列名；TOML/API 可用嵌套列表表示组内互斥替代）")
    parser.add_argument("--controls-must", dest="controls_must", metavar="VAR", nargs="+",
                        help="强制纳入回归、但不参与组合枚举的控制变量列名")

    # ── FE 变量标识符（自动模式和手动模式共用） ──────────
    parser.add_argument("--Firm-FE",   dest="firm_fe",   default="code",
                        metavar="COL",
                        help="个体（企业）固定效应列名（默认 code）")
    parser.add_argument("--Ind-FE",    dest="ind_fe",    default="ind",
                        metavar="COL",
                        help="行业固定效应列名（默认 ind）")
    parser.add_argument("--Time-FE",   dest="time_fe",   default="year",
                        metavar="COL",
                        help="时间固定效应列名（默认 year）")
    parser.add_argument("--Region-FE", dest="region_fe", default=None,
                        metavar="COL",
                        help="地区固定效应列名（县级 city / 地级 pref / 省级 prov）；"
                             "needs_region=True 的规格需提供")

    # ── 自动模式：规格标志（每个对应一种 FE+聚类组合） ───
    spec_grp = parser.add_argument_group(
        "自动模式（规格标志）",
        "选择一个或多个规格；未选择任何规格时进入手动模式",
    )
    spec_grp.add_argument(
        "--absorb-firm-time-vce-cluster-firm",
        dest="absorb_firm_time_vce_cluster_firm",
        action="store_true",
        help="absorb(Firm_FE Time_FE) vce(cluster Firm_FE) — 个体+年度FE，公司聚类，会计实证默认基准",
    )
    spec_grp.add_argument(
        "--absorb-firm-time-vce-robust",
        dest="absorb_firm_time_vce_robust",
        action="store_true",
        help="absorb(Firm_FE Time_FE) vce(robust) — 个体+年度FE，异方差稳健标准误",
    )
    spec_grp.add_argument(
        "--absorb-firm-indtime-vce-cluster-firm",
        dest="absorb_firm_indtime_vce_cluster_firm",
        action="store_true",
        help="absorb(Firm_FE i.Ind_FE#i.Time_FE) vce(cluster Firm_FE) — 行业×年度FE",
    )
    spec_grp.add_argument(
        "--absorb-firm-indtime-vce-robust",
        dest="absorb_firm_indtime_vce_robust",
        action="store_true",
        help="absorb(Firm_FE i.Ind_FE#i.Time_FE) vce(robust) — 行业×年度FE，异方差稳健标准误",
    )
    spec_grp.add_argument(
        "--absorb-firm-regiontime-vce-cluster-firm",
        dest="absorb_firm_regiontime_vce_cluster_firm",
        action="store_true",
        help="absorb(Firm_FE i.Region_FE#i.Time_FE) vce(cluster Firm_FE) — 地区×年度FE",
    )
    spec_grp.add_argument(
        "--absorb-firm-regiontime-vce-robust",
        dest="absorb_firm_regiontime_vce_robust",
        action="store_true",
        help="absorb(Firm_FE i.Region_FE#i.Time_FE) vce(robust) — 地区×年度FE，异方差稳健标准误",
    )
    spec_grp.add_argument(
        "--absorb-firm-indtime-regiontime-vce-cluster-firm",
        dest="absorb_firm_indtime_regiontime_vce_cluster_firm",
        action="store_true",
        help="absorb(Firm_FE i.Ind_FE#i.Time_FE i.Region_FE#i.Time_FE) vce(cluster Firm_FE) — 双重剥离",
    )
    spec_grp.add_argument(
        "--absorb-firm-indtime-regiontime-vce-robust",
        dest="absorb_firm_indtime_regiontime_vce_robust",
        action="store_true",
        help="absorb(Firm_FE i.Ind_FE#i.Time_FE i.Region_FE#i.Time_FE) vce(robust) — 双重剥离，异方差稳健标准误",
    )
    spec_grp.add_argument(
        "--absorb-firm-time-vce-cluster-region",
        dest="absorb_firm_time_vce_cluster_region",
        action="store_true",
        help="absorb(Firm_FE Time_FE) vce(cluster Region_FE) — 地区层面聚类",
    )
    spec_grp.add_argument(
        "--absorb-firm-time-vce-cluster-ind",
        dest="absorb_firm_time_vce_cluster_ind",
        action="store_true",
        help="absorb(Firm_FE Time_FE) vce(cluster Ind_FE) — 行业层面聚类",
    )
    spec_grp.add_argument(
        "--absorb-ind-region-time-vce-cluster-ind",
        dest="absorb_ind_region_time_vce_cluster_ind",
        action="store_true",
        help="absorb(Ind_FE Region_FE Time_FE) vce(cluster Ind_FE) — 三向FE，无个体层面重复观测",
    )
    spec_grp.add_argument(
        "--absorb-ind-region-time-vce-robust",
        dest="absorb_ind_region_time_vce_robust",
        action="store_true",
        help="absorb(Ind_FE Region_FE Time_FE) vce(robust) — 三向FE，异方差稳健标准误",
    )
    spec_grp.add_argument(
        "--absorb-firm-time-vce-cluster-firm-time",
        dest="absorb_firm_time_vce_cluster_firm_time",
        action="store_true",
        help="absorb(Firm_FE Time_FE) vce(cluster Firm_FE Time_FE) — CGM双向聚类",
    )
    spec_grp.add_argument(
        "--absorb-ind-time-vce-cluster-firm",
        dest="absorb_ind_time_vce_cluster_firm",
        action="store_true",
        help="absorb(Ind_FE Time_FE) vce(cluster Firm_FE) — 行业+年度FE，公司聚类",
    )
    spec_grp.add_argument(
        "--absorb-ind-time-vce-robust",
        dest="absorb_ind_time_vce_robust",
        action="store_true",
        help="absorb(Ind_FE Time_FE) vce(robust) — 行业+年度FE，异方差稳健标准误",
    )

    # ── 手动模式 ──────────────────────────────────────────
    manual_grp = parser.add_argument_group(
        "手动模式",
        "未选择任何规格标志时使用；直接指定 FE 列和聚类列",
    )
    manual_grp.add_argument("--fe",    metavar="COL", nargs="+",
                             help="固定效应列名（1 个或多个，已存在于数据中）")
    manual_grp.add_argument("--clust", metavar="COL", nargs="+",
                             help="聚类变量列名（1 个 = 单向；2 个 = CGM 双向）")
    manual_grp.add_argument("--gen-clust2", dest="gen_clust2",
                             action="store_true",
                             help="自动生成 clust2 = fe[0]_fe[1] 交叉列并追加到 --clust")

    # ── 通用可选 ──────────────────────────────────────────
    parser.add_argument("--output",    default="outputs", metavar="DIR",
                        help="输出目录路径（默认 outputs；程序会在其中创建时间戳子目录，图片固定导出为 PNG）")
    parser.add_argument("--dpi",       default=150, type=int,
                        help="图像分辨率（默认 150）")
    parser.add_argument("--fig-width", default=14.0, type=float, metavar="INCHES",
                        help="图像宽度，单位英寸（默认 14）")
    parser.add_argument("--n-jobs",    default=0, type=int, metavar="N",
                        help="并行进程数（默认 0 = 自动；最多 9 核；1 表示串行）")
    parser.add_argument("--order", choices=["coef", "p"], default="coef",
                        help="绘图排序方式：coef 或 p")
    parser.add_argument("--p", action="store_true",
                        help="兼容别名；等价于 --order p")

    # ── 将 TOML 配置值注入为默认值（命令行参数仍可覆盖） ──
    if _toml_cfg:
        _allowed_keys = {
            "data", "y", "x", "controls", "controls_test", "controls_must",
            "output", "dpi", "fig_width", "n_jobs", "order", "p",
            "firm_fe", "ind_fe", "time_fe", "region_fe",
            "fe", "clust", "gen_clust2",
        } | set(_ALL_SPEC_NAMES)
        # TOML 键全部小写后与 argparse dest 对应
        _normalized = {k.lower(): v for k, v in _toml_cfg.items()}
        parser.set_defaults(**{k: v for k, v in _normalized.items()
                                if k in _allowed_keys})

    args = parser.parse_args(_cli_args)
    try:
        args.order = _normalize_plot_order(args.order, p_alias=bool(args.p))
    except ValueError as exc:
        parser.error(str(exc))
    resolved_n_jobs = _resolve_n_jobs(args.n_jobs)
    controls_test = list(args.controls_test) if args.controls_test else (
        list(args.controls) if args.controls else []
    )
    controls_must = list(args.controls_must) if args.controls_must else []
    try:
        controls_must_flat, _must_slots = _normalize_controls_must(controls_must)
        controls_test_flat, _control_slots = _normalize_controls_test(controls_test)
        _validate_control_lists_do_not_overlap(controls_test_flat, controls_must_flat)
    except ValueError as exc:
        parser.error(str(exc))
    matrix_controls = _varying_must_controls(_must_slots) + controls_test_flat

    # ── 必填参数校验 ──────────────────────────────────────
    missing_required = [f for f, v in [
        ("--data / data",         args.data),
        ("--y / y",               args.y),
        ("--x / x",               args.x),
    ] if v is None]
    if missing_required:
        parser.error(
            f"以下参数未提供（可在配置文件或命令行中指定）："
            f"{', '.join(missing_required)}"
        )
    if not controls_test and not controls_must:
        parser.error("至少提供一类控制变量：--controls-test / controls_test / controls，或 --controls-must / controls_must。")

    # ── 确定运行模式 ──────────────────────────────────────
    _spec_flags = {
        name: getattr(args, name, False)
        for name in _ALL_SPEC_NAMES
    }
    is_auto = any(_spec_flags.values())

    # ── 手动模式参数校验 ──────────────────────────────────
    if not is_auto:
        if not args.fe:
            parser.error(
                "请选择至少一个规格标志（--absorb-*-vce-cluster-*）进入自动模式，"
                "或使用 --fe 指定固定效应列名进入手动模式。"
            )
        if not args.clust and not args.gen_clust2:
            parser.error("手动模式下必须指定 --clust 或 --gen-clust2。")

    # ── 读取数据 ──────────────────────────────────────────
    data_path = pathlib.Path(args.data)
    suffix = data_path.suffix.lower()
    print(f"读取数据：{data_path}")

    if suffix == ".dta":
        df = cast(pd.DataFrame, pd.read_stata(data_path))
    elif suffix == ".csv":
        df = cast(pd.DataFrame, pd.read_csv(data_path))
    elif suffix in (".parquet", ".pq"):
        df = cast(pd.DataFrame, pd.read_parquet(data_path))
    else:
        parser.error(f"不支持的文件格式：{suffix}（支持 .dta / .csv / .parquet）")

    print(f"数据读取完成：{len(df):,} 行 × {len(df.columns)} 列")

    # ── 输出目录与配置快照 ────────────────────────────────
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    raw_output = pathlib.Path(args.output)
    output_root = raw_output.parent if raw_output.suffix else raw_output
    if output_root == pathlib.Path("."):
        output_root = pathlib.Path.cwd() / "outputs"
    run_output_dir = output_root / run_timestamp
    run_output_dir.mkdir(parents=True, exist_ok=True)
    print(f"输出目录：{run_output_dir}")

    snapshot_config = {
        "generated_at": run_timestamp,
        "data": str(args.data),
        "y": list(args.y),
        "x": list(args.x),
        "controls_test": controls_test_flat,
        "controls_must": controls_must_flat,
        "output": str(output_root),
        "run_output_dir": str(run_output_dir),
        "dpi": args.dpi,
        "fig_width": args.fig_width,
        "n_jobs": args.n_jobs,
        "resolved_n_jobs": resolved_n_jobs,
        "order": args.order,
        "firm_fe": args.firm_fe,
        "ind_fe": args.ind_fe,
        "time_fe": args.time_fe,
    }
    if args.region_fe:
        snapshot_config["region_fe"] = args.region_fe
    snapshot_config.update({name: enabled for name, enabled in _spec_flags.items() if enabled})
    _write_config_snapshot(snapshot_config, run_output_dir / "config_snapshot.toml")
    print(f"并行策略：请求 n_jobs={args.n_jobs}，实际使用 {resolved_n_jobs} 核（上限 {_MAX_AUTO_JOBS}）")

    # ── 遍历所有 y × x 组合 ──────────────────────────────
    combos   = list(itertools.product(args.y, args.x))
    n_combos = len(combos)
    all_sig_rows: list[dict] = []
    total_sig_specs = 0
    combo_summaries: list[dict[str, Any]] = []
    clust_cols: list[str] = []

    if not is_auto:
        # 手动模式：处理 --gen-clust2
        clust_cols = list(args.clust) if args.clust else []
        if args.gen_clust2:
            if len(args.fe) < 2:
                parser.error("--gen-clust2 需要至少提供 2 个 --fe 列。")
            clust2_col = f"{args.fe[0]}_{args.fe[1]}"
            df[clust2_col] = (
                df[args.fe[0]].astype(str) + "_" + df[args.fe[1]].astype(str)
            )
            print(f"已生成聚类变量：{clust2_col}")
            clust_cols.append(clust2_col)
        if len(clust_cols) > 2:
            parser.error("--clust 最多接受 2 个聚类列（单向 / CGM 双向）。")

    for idx, (y_var, x_var) in enumerate(combos, 1):
        print(f"\n{'#'*60}")
        print(f"[{idx}/{n_combos}]  Y = {y_var}  ×  X = {x_var}")
        print("#" * 60)
        pair_sig_rows: list[dict[str, Any]] = []
        pair_total_specs = 0

        # 默认输出文件名不再附带 regression_monkey 前缀
        pair_stem = f"{y_var}_{x_var}"
        out_pair  = str((run_output_dir / f"{pair_stem}.png"))

        if is_auto:
            # 自动模式：检查 FE 列存在性
            fe_check = [
                ("--Firm-FE",   args.firm_fe),
                ("--Ind-FE",    args.ind_fe),
                ("--Time-FE",   args.time_fe),
            ]
            if args.region_fe:
                fe_check.append(("--Region-FE", args.region_fe))
            for flag, col in fe_check:
                if col not in df.columns:
                    parser.error(f"{flag} 指定的列 '{col}' 不存在于数据中")

            auto_results = regression_monkey_auto(
                df          = df,
                y           = y_var,
                x           = x_var,
                controls_test = controls_test,
                controls_must = controls_must,
                firm_fe     = args.firm_fe,
                ind_fe      = args.ind_fe,
                time_fe     = args.time_fe,
                region_fe   = args.region_fe,
                specs       = _spec_flags,
                output_path = out_pair,
                dpi         = args.dpi,
                fig_width   = args.fig_width,
                n_jobs      = resolved_n_jobs,
                export_sig_table = False,
                render_plot = False,
                sort_by_signed_p = _order_uses_p_mode(args.order),
            )
            for spec_name, records, _fig in auto_results:
                if not records:
                    continue
                spec_def = next(s for s in _SPEC_CATALOG if s["name"] == spec_name)
                out_png = run_output_dir / f"{pair_stem}_{spec_def['tag']}.png"
                write_analysis_artifacts(
                    records=records,
                    results_path=run_output_dir / f"{pair_stem}_{spec_def['tag']}_results.csv",
                    meta_path=run_output_dir / f"{pair_stem}_{spec_def['tag']}_plot_meta.json",
                    meta={
                        "engine": "python",
                        "spec_name": spec_name,
                        "y": y_var,
                        "x": x_var,
                        "controls_test_flat": controls_test_flat,
                        "controls_must_flat": controls_must_flat,
                        "matrix_controls": matrix_controls,
                        "show_special_markers": True,
                        "fig_width": args.fig_width,
                        "dpi": args.dpi,
                        "order": args.order,
                        "sort_by_p_mode": _order_uses_p_mode(args.order),
                        "sort_by_signed_p": _order_uses_p_mode(args.order),
                        "title_suffix": spec_def["help"].format(
                            firm=args.firm_fe,
                            ind=args.ind_fe,
                            time=args.time_fe,
                            region=args.region_fe or "region",
                        ),
                        "output_path": str(out_png),
                    },
                )
                var_map = {
                    "firm": args.firm_fe,
                    "ind": args.ind_fe,
                    "time": args.time_fe,
                }
                if args.region_fe:
                    var_map["region"] = args.region_fe
                fe_cols = []
                clust_cols = []
                for key in spec_def["fe_keys"]:
                    if key.startswith("_"):
                        if key == "_ind_time":
                            fe_cols.append(f"_spec_{args.ind_fe}_{args.time_fe}")
                        elif key == "_region_time":
                            fe_cols.append(f"_spec_{args.region_fe}_{args.time_fe}")
                    else:
                        fe_cols.append(var_map[key])
                for key in spec_def["cl_keys"]:
                    clust_cols.append(var_map[key])
                pair_rows = _build_sig_rows(
                    records=records,
                    y=y_var,
                    x=x_var,
                    controls_must=controls_must_flat,
                    controls_test=controls_test_flat,
                    fe_cols=fe_cols,
                    clust_cols=clust_cols,
                )
                all_sig_rows.extend(pair_rows)
                pair_sig_rows.extend(pair_rows)
                total_sig_specs += len(records)
                pair_total_specs += len(records)
        else:
            # 手动模式：列存在性检查
            needed = [y_var, x_var] + controls_must_flat + controls_test_flat + list(args.fe) + clust_cols
            missing = [c for c in needed if c not in df.columns]
            if missing:
                print(f"  [跳过] 以下列不存在：{missing}")
                continue

            records, _fig = regression_monkey(
                df           = df,
                y            = y_var,
                x            = x_var,
                controls_test = controls_test,
                controls_must = controls_must,
                fe_cols      = list(args.fe),
                clust_cols   = clust_cols,
                output_path  = out_pair,
                dpi          = args.dpi,
                fig_width    = args.fig_width,
                n_jobs       = resolved_n_jobs,
                export_sig_table = False,
                render_plot = False,
                sort_by_signed_p = _order_uses_p_mode(args.order),
            )
            write_analysis_artifacts(
                records=records,
                results_path=run_output_dir / f"{pair_stem}_results.csv",
                meta_path=run_output_dir / f"{pair_stem}_plot_meta.json",
                meta={
                    "engine": "python",
                    "spec_name": "manual",
                    "y": y_var,
                    "x": x_var,
                    "controls_test_flat": controls_test_flat,
                    "controls_must_flat": controls_must_flat,
                    "matrix_controls": matrix_controls,
                    "show_special_markers": True,
                    "fig_width": args.fig_width,
                    "dpi": args.dpi,
                    "order": args.order,
                    "sort_by_p_mode": _order_uses_p_mode(args.order),
                    "sort_by_signed_p": _order_uses_p_mode(args.order),
                    "title_suffix": f"manual FE = {', '.join(args.fe)}",
                    "output_path": str(run_output_dir / f"{pair_stem}.png"),
                },
            )
            pair_rows = _build_sig_rows(
                records=records,
                y=y_var,
                x=x_var,
                controls_must=controls_must_flat,
                controls_test=controls_test_flat,
                fe_cols=list(args.fe),
                clust_cols=clust_cols,
            )
            all_sig_rows.extend(pair_rows)
            pair_sig_rows.extend(pair_rows)
            total_sig_specs += len(records)
            pair_total_specs += len(records)

        combo_summaries.append({
            "y": y_var,
            "x": x_var,
            "n_specs": pair_total_specs,
            "n_sig": len(pair_sig_rows),
            "star_counts": _sig_star_counts(pair_sig_rows),
        })

    _export_sig_table(
        rows=all_sig_rows,
        output_path=str(run_output_dir / "sig.csv"),
        n_specs=total_sig_specs,
        print_summary=False,
    )

    for line in _format_combo_summary_lines(combo_summaries):
        print(line)
    print(f"\n全部完成：{n_combos} 个 y×x 组合")


if __name__ == "__main__":
    import sys
    main()
    sys.exit(0)
