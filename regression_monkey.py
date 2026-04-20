# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "numpy",
#   "pandas",
#   "polars",
#   "pyarrow",
#   "matplotlib",
#   "pyreadstat",
#   "scipy",
#   "tomli >= 2.0 ; python_version < '3.11'",
# ]
# ///
"""
regression_monkey.py
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
import multiprocessing as mp
import os
import pathlib
import platform
from time import perf_counter
from typing import TYPE_CHECKING, Any, TypeAlias, TypedDict, cast
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
    df_resid: int
    ci99_lo: float
    ci99_hi: float
    ci95_lo: float
    ci95_hi: float
    ci90_lo: float
    ci90_hi: float
    controls_test: set[str]
    controls_all: set[str]
    is_full: bool
    obs: int

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
            slot = (item,)
        elif isinstance(item, (list, tuple)):
            if not item:
                raise ValueError(f"{field_name} 中的替代组不能为空列表")
            if not all(isinstance(v, str) and v for v in item):
                raise ValueError(f"{field_name} 的替代组必须全部由非空字符串列名组成")
            slot = tuple(item)
        else:
            raise ValueError(f"{field_name} 仅支持 str 或由 str 组成的 list/tuple")

        dup = [name for name in slot if name in seen]
        if dup:
            raise ValueError(f"{field_name} 存在重复列名：{dup}")

        control_slots.append(slot)
        flat_controls.extend(slot)
        seen.update(slot)

    return flat_controls, control_slots


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


def _calc_se(
    X2: np.ndarray,
    e: np.ndarray,
    k_total: int,
    se_kind: str,
    se_args: SeArgs,
) -> np.ndarray:
    """根据预计算的聚类信息计算标准误。"""
    if se_kind == "two_way":
        return _cgm_se(X2, e, *se_args, k_total)
    if se_kind == "one_way":
        return _se_single(X2, e, *se_args, k_total)
    if se_kind == "robust":
        return _se_robust(X2, e, *se_args, k_total)
    raise ValueError(f"unknown se_kind: {se_kind}")


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
) -> tuple[list[SpecRecord], Figure]:
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
    varying_must_controls = _varying_must_controls(must_slots)
    matrix_controls = varying_must_controls + controls_test_flat
    show_special_markers = not varying_must_controls
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
    fig = _plot(records, y_name=y, x_name=x, controls_test=controls_test_flat,
                controls_must=controls_must_flat,
                matrix_controls=matrix_controls,
                show_special_markers=show_special_markers,
                fig_width=fig_width, dpi=dpi,
                output_path=output_path, title_suffix=title_suffix,
                elapsed_seconds_preplot=perf_counter() - total_t0)

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
) -> list[tuple[str, list[SpecRecord], Figure]]:
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
    varying_must_controls = _varying_must_controls(must_slots)
    matrix_controls = varying_must_controls + controls_test_flat
    show_special_markers = not varying_must_controls

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
    results: list[tuple[str, list[SpecRecord], Figure]] = []
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
        if not specs.get(spec_name, False):
            continue
        if spec_def["needs_region"] and region_fe is None:
            print(f"[跳过] {spec_name}：需要 Region_FE 但未指定")
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
            print(f"[{spec_name}] 剔除缺失值：{n_drop} 行（剩余 {len(df_spec):,} 行）")

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
            print(f"  - {spec_name}: {chunk_count} 个块")

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
        print(f"\n{'='*60}")
        print(f"规格：{spec_name}")
        print("=" * 60)
        print(f"完成 {len(records)} 个规格，系数范围：[{records[0]['coef']:.4f}, {records[-1]['coef']:.4f}]")

        fig = _plot(
            records,
            y_name        = y,
            x_name        = x,
            controls_test = controls_test_flat,
            controls_must = controls_must_flat,
            matrix_controls = matrix_controls,
            show_special_markers = show_special_markers,
            fig_width     = fig_width,
            dpi           = dpi,
            output_path   = meta["out_i"],
            title_suffix  = meta["title_suffix"],
            elapsed_seconds_preplot = perf_counter() - meta["t0"],
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
            "obs":      r["obs"],
            "Y":        y,
            "X":        x,
            "Controls": ", ".join(ordered_controls) if ordered_controls else "(none)",
            "FE":       ", ".join(fe_cols),
            "cluster":  cluster_label,
        })

    return rows


def _export_sig_table(
    rows:        list[dict],
    output_path: str,
    n_specs:     int,
    print_summary: bool = True,
) -> "pd.DataFrame | None":
    """将显著规格行写出为单张 CSV 汇总表。"""
    rows = [{**row, "Specs": n_specs} for row in rows]

    if not rows:
        if print_summary:
            print("  [汇总表] 无 90% 及以上显著的规格，跳过导出")
        return None

    tbl = pd.DataFrame(rows)
    tbl["_star_abs"] = tbl["Star"].abs()
    tbl = tbl.sort_values(
        ["_star_abs", "Star", "coef"], ascending=[False, False, False]
    ).drop(columns=["_star_abs"])
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
    return (
        f"[汇总表] {n_sig}/{n_specs} 个规格显著"
        f"（+3:{star_counts[3]}  +2:{star_counts[2]}  +1:{star_counts[1]}  "
        f"-1:{star_counts[-1]}  -2:{star_counts[-2]}  -3:{star_counts[-3]}）"
    )


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
          elapsed_seconds_preplot: float | None = None) -> Figure:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.lines import Line2D
    plot_t0 = perf_counter()

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
    title_h = 1.55
    upper_h = 3.8
    lower_h = max(1.2, matrix_rows * 0.30 + 0.55)
    stats_h = 0.45
    obs_h = 1.0

    fig, (ax_title, ax1, ax2, ax_stats, ax3) = plt.subplots(
        5, 1,
        figsize=(fig_width, title_h + upper_h + lower_h + stats_h + obs_h),
        facecolor="white",
        gridspec_kw={"height_ratios": [title_h, upper_h, lower_h, stats_h, obs_h]},
        layout="constrained",
        dpi=dpi,
    )
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

    sig99  = p_values < 0.01                      # 99% 显著 → 黄色
    sig95  = (p_values < 0.05) & ~sig99           # 95% 显著 → 草绿
    sig90  = (p_values < 0.10) & ~sig99 & ~sig95  # 90% 显著 → 蓝色
    insig  = ~sig99 & ~sig95 & ~sig90             # 不显著   → 深灰

    _C99  = "#FFC107"   # 黄
    _C95  = "#7CCD7C"   # 草绿
    _C90  = "#1F77B4"   # 蓝
    _CINS = "#000000"   # 黑（不显著）
    _CFUL = "#cc2222"   # 红（全变量规格外圈）
    _CNOC = "#ff8c00"   # 橙（无 test 控制变量规格外圈）
    _CSWITCH = "#2255cc"  # 蓝（最接近 0 的系数点）

    # 仅当系数序列跨过 0 时，才标记最接近 0 的单个规格点
    is_sign_switch = np.zeros(n, dtype=bool)
    if np.any(coefs < 0) and np.any(coefs > 0):
        is_sign_switch[int(np.argmin(np.abs(coefs)))] = True

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

    ax1.set_ylabel("Coef.", fontsize=9)
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
    title_lines = [
        "Regression Monkey",
        f"Y = {y_name}  |  X = {x_name}",
        f"specs = {n}  |  controls = {K_total}",
        spec_line or "",
        f"controls_must = {', '.join(controls_must)}" if controls_must else "controls_must = (none)",
        f"Elapsed = {elapsed_total:.2f}s  |  @Dfhaklhd",
    ]
    _title = "\n".join(title_lines)
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
        grid = np.ones((matrix_rows, n), dtype=float)
        for k_idx, row_runs in enumerate(runs):
            row_ax = matrix_rows - 1 - k_idx
            for run in row_runs:
                if not run["inc"]:
                    continue
                x0 = run["start"]
                x1 = x0 + run["len"]
                grid[row_ax, x0:x1] = black_value(run["len"])

        ax2.imshow(
            grid,
            cmap="gray",
            vmin=0.0,
            vmax=1.0,
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

    # 全变量列红色虚线
    if show_special_markers:
        for fi in np.where(is_full)[0]:
            ax2.axvline(fi, color="#cc2222", lw=1.1, ls="-", zorder=3)
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
    pos_mask = coefs >= 0
    neg_mask = ~pos_mask

    if pos_mask.any():
        ax3.bar(
            xs[pos_mask],
            obs_arr[pos_mask] - obs_mean,
            bottom=obs_mean,
            width=1.0,
            color="#cc2222",
            alpha=0.35,
            linewidth=0,
            zorder=2,
            align="center",
        )
    if neg_mask.any():
        ax3.bar(
            xs[neg_mask],
            obs_arr[neg_mask] - obs_mean,
            bottom=obs_mean,
            width=1.0,
            color="#2255cc",
            alpha=0.35,
            linewidth=0,
            zorder=2,
            align="center",
        )
    ax3.axhline(obs_mean, color="#444444", lw=0.8, ls="--", zorder=3)
    if show_special_markers:
        for fi in np.where(is_full)[0]:
            ax3.axvline(fi, color=_CFUL, lw=1.1, ls="-", zorder=4)
        for ni in np.where(is_nocontrol)[0]:
            ax3.axvline(ni, color=_CNOC, lw=1.1, ls="-", zorder=4)
    for si in np.where(is_sign_switch)[0]:
        ax3.axvline(si, color=_CSWITCH, lw=1.1, ls="-", zorder=4)
    ax3.set_xlim(-0.5, n - 0.5)
    ax3.set_ylabel("Obs.", fontsize=8)
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

    if output_path:
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="white")
        elapsed_total = (elapsed_seconds_preplot or 0.0) + (perf_counter() - plot_t0)
        title_lines[-1] = f"Elapsed = {elapsed_total:.2f}s  |  @Dfhaklhd"
        title_text.set_text("\n".join(title_lines))
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="white")
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

    # ── 将 TOML 配置值注入为默认值（命令行参数仍可覆盖） ──
    if _toml_cfg:
        _allowed_keys = {
            "data", "y", "x", "controls", "controls_test", "controls_must",
            "output", "dpi", "fig_width", "n_jobs",
            "firm_fe", "ind_fe", "time_fe", "region_fe",
            "fe", "clust", "gen_clust2",
        } | set(_ALL_SPEC_NAMES)
        # TOML 键全部小写后与 argparse dest 对应
        _normalized = {k.lower(): v for k, v in _toml_cfg.items()}
        parser.set_defaults(**{k: v for k, v in _normalized.items()
                                if k in _allowed_keys})

    args = parser.parse_args(_cli_args)
    resolved_n_jobs = _resolve_n_jobs(args.n_jobs)
    controls_test = list(args.controls_test) if args.controls_test else (
        list(args.controls) if args.controls else []
    )
    controls_must = list(args.controls_must) if args.controls_must else []
    try:
        controls_must_flat, _must_slots = _normalize_controls_must(controls_must)
        controls_test_flat, _control_slots = _normalize_controls_test(controls_test)
    except ValueError as exc:
        parser.error(str(exc))

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
        "controls_test": controls_test,
        "controls_test_flat": controls_test_flat,
        "controls_must": controls_must,
        "controls_must_flat": controls_must_flat,
        "output": str(output_root),
        "run_output_dir": str(run_output_dir),
        "dpi": args.dpi,
        "fig_width": args.fig_width,
        "n_jobs": args.n_jobs,
        "resolved_n_jobs": resolved_n_jobs,
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
            )
            for spec_name, records, _fig in auto_results:
                if not records:
                    continue
                spec_def = next(s for s in _SPEC_CATALOG if s["name"] == spec_name)
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

    print("\n各 Y-X 组合显著性汇总：")
    for summary in combo_summaries:
        print(
            f"  Y={summary['y']}  X={summary['x']}  "
            f"{_format_sig_summary(summary['n_sig'], summary['n_specs'], summary['star_counts'])}"
        )
    print(f"\n全部完成：{n_combos} 个 y×x 组合")


if __name__ == "__main__":
    import sys
    main()
    sys.exit(0)
