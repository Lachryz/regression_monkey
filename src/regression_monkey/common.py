"""
regression_monkey_common.py
===========================
共享工具函数，被 regression_monkey.py 和 regression_monkey_stata.py 导入。
不提供 CLI 入口，不作为 uv 脚本独立运行。
"""

from __future__ import annotations

import pathlib
from typing import Any, cast

import pandas as pd

try:
    import tomllib
except ImportError:
    import tomli as tomllib  # type: ignore[no-redef]


def load_toml_config(cli_args: list[str]) -> tuple[dict[str, Any], list[str]]:
    """
    从 cli_args 中读取可选 TOML 配置文件路径并加载。

    规则：
    - 若 cli_args[0] 以 .toml 结尾，将其作为配置路径并从返回的 args 列表中移除
    - 否则尝试加载脚本目录下的 regression_monkey_config.toml
    - 均不存在时返回空 dict

    Returns: (config_dict, remaining_cli_args)
    """
    if cli_args and pathlib.Path(cli_args[0]).suffix == ".toml":
        config_path = pathlib.Path(cli_args[0]).expanduser()
        if not config_path.exists():
            raise FileNotFoundError(f"配置文件不存在：{config_path}")
        with config_path.open("rb") as f:
            print(f"[配置] 加载：{config_path}")
            return cast(dict[str, Any], tomllib.load(f)), cli_args[1:]

    default_cfg = pathlib.Path.cwd() / "regression_monkey_config.toml"
    if not default_cfg.exists():
        default_cfg = pathlib.Path.cwd() / "config" / "regression_monkey_config.toml"
    if not default_cfg.exists():
        default_cfg = pathlib.Path(__file__).with_name("regression_monkey_config.toml")
    if default_cfg.exists():
        with default_cfg.open("rb") as f:
            print(f"[配置] 加载默认配置：{default_cfg}")
            return cast(dict[str, Any], tomllib.load(f)), cli_args
    return {}, cli_args


def load_dataframe(data_path: pathlib.Path) -> pd.DataFrame:
    """加载 .dta / .csv / .parquet 格式的数据文件，返回 DataFrame。"""
    suffix = data_path.suffix.lower()
    if suffix == ".dta":
        return cast(pd.DataFrame, pd.read_stata(data_path))
    if suffix == ".csv":
        return cast(pd.DataFrame, pd.read_csv(data_path))
    if suffix in (".parquet", ".pq"):
        return cast(pd.DataFrame, pd.read_parquet(data_path))
    raise ValueError(f"不支持的文件格式：{suffix}（支持 .dta / .csv / .parquet）")


def safe_unlink(path: pathlib.Path) -> None:
    """安全删除文件，文件不存在时静默忽略。"""
    try:
        path.unlink()
    except FileNotFoundError:
        return


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


def _tail_text(path: pathlib.Path, max_lines: int = 80) -> str:
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except FileNotFoundError:
        return "(log file not found)"
    return "\n".join(lines[-max_lines:])
