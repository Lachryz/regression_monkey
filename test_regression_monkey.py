"""
test_regression_monkey.py
==========================
pytest 测试套件。不依赖 Stata 环境，不依赖真实数据文件。
全部使用合成 CSV 数据（固定随机种子，可重现）。
"""

from __future__ import annotations

import json
import pathlib
import re
import numpy as np
import pandas as pd
import pytest

import regression_monkey_common as rm_common
import regression_monkey_html as rm_html
import regression_monkey as rm_main
import regression_monkey_py as rm_py
import regression_monkey_stata as rm_stata


# ─────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def synthetic_df() -> pd.DataFrame:
    """200 行合成面板数据。treatment 对 outcome 有真实正向效应（系数约 2.0）。"""
    rng = np.random.default_rng(42)
    n = 200
    firm_id = rng.integers(0, 10, size=n)
    industry = firm_id % 5
    year = rng.integers(2018, 2022, size=n)
    treatment = rng.standard_normal(n)
    ctrl_a = rng.standard_normal(n)
    ctrl_b = rng.standard_normal(n)
    ctrl_c = rng.standard_normal(n)
    firm_fe = rng.standard_normal(10)[firm_id]
    outcome = 2.0 * treatment + 0.5 * ctrl_c + firm_fe + rng.standard_normal(n) * 0.5
    return pd.DataFrame({
        "outcome": outcome,
        "treatment": treatment,
        "ctrl_a": ctrl_a,
        "ctrl_b": ctrl_b,
        "ctrl_c": ctrl_c,
        "firm_id": firm_id,
        "industry": industry,
        "year": year,
    })


@pytest.fixture(scope="session")
def test_toml_path() -> pathlib.Path:
    """regression_monkey_test.toml 的绝对路径。"""
    p = pathlib.Path(__file__).with_name("regression_monkey_test.toml")
    assert p.exists(), f"测试 TOML 不存在：{p}"
    return p


@pytest.fixture
def tmp_csv(tmp_path: pathlib.Path, synthetic_df: pd.DataFrame) -> pathlib.Path:
    """把合成数据写到临时 CSV 并返回路径。"""
    p = tmp_path / "test_data.csv"
    synthetic_df.to_csv(p, index=False)
    return p


# ─────────────────────────────────────────────────────────────
# TestLoadTomlConfig
# ─────────────────────────────────────────────────────────────

class TestLoadTomlConfig:

    def test_loads_explicit_toml(self, test_toml_path: pathlib.Path) -> None:
        """显式 .toml 路径被加载，且从 remaining 中移除。"""
        cfg, remaining = rm_common.load_toml_config([str(test_toml_path), "--dpi", "600"])
        assert "y" in cfg
        assert remaining == ["--dpi", "600"]

    def test_explicit_toml_consumed_alone(self, test_toml_path: pathlib.Path) -> None:
        """单独传入 .toml 路径时，remaining 为空列表。"""
        cfg, remaining = rm_common.load_toml_config([str(test_toml_path)])
        assert isinstance(cfg, dict)
        assert remaining == []

    def test_raises_for_missing_toml(self, tmp_path: pathlib.Path) -> None:
        """不存在的 .toml 路径抛 FileNotFoundError。"""
        with pytest.raises(FileNotFoundError):
            rm_common.load_toml_config([str(tmp_path / "nonexistent.toml")])

    def test_value_types(self, test_toml_path: pathlib.Path) -> None:
        """TOML 值类型正确：y/x 为 list，dpi 为 int，fig_width 为 float。"""
        cfg, _ = rm_common.load_toml_config([str(test_toml_path)])
        assert isinstance(cfg["y"], list)
        assert isinstance(cfg["x"], list)
        assert isinstance(cfg["dpi"], int)
        assert isinstance(cfg["fig_width"], float)

    def test_non_toml_arg_not_consumed(self, test_toml_path: pathlib.Path) -> None:
        """非 .toml 首参数不会被消费（仍留在 remaining 中）。"""
        # 首个参数是 flag，不匹配 .toml 条件，remaining 不变
        cfg, remaining = rm_common.load_toml_config(["--dpi", "300"])
        assert "--dpi" in remaining
        assert "300" in remaining


# ─────────────────────────────────────────────────────────────
# TestLoadDataframe
# ─────────────────────────────────────────────────────────────

class TestLoadDataframe:

    def test_loads_csv(self, tmp_csv: pathlib.Path) -> None:
        """CSV 加载后返回正确行数的 DataFrame。"""
        df = rm_common.load_dataframe(tmp_csv)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 200

    def test_csv_columns_preserved(self, tmp_csv: pathlib.Path) -> None:
        """加载的 CSV 包含原始列名。"""
        df = rm_common.load_dataframe(tmp_csv)
        assert set(["outcome", "treatment", "firm_id"]).issubset(df.columns)

    def test_unsupported_extension_raises(self, tmp_path: pathlib.Path) -> None:
        """.xlsx 后缀抛 ValueError，错误信息含'不支持的文件格式'。"""
        p = tmp_path / "data.xlsx"
        p.write_text("dummy")
        with pytest.raises(ValueError, match="不支持的文件格式"):
            rm_common.load_dataframe(p)


# ─────────────────────────────────────────────────────────────
# TestSafeUnlink
# ─────────────────────────────────────────────────────────────

class TestSafeUnlink:

    def test_deletes_existing_file(self, tmp_path: pathlib.Path) -> None:
        """存在的文件被删除。"""
        p = tmp_path / "dummy.txt"
        p.write_text("x")
        rm_common.safe_unlink(p)
        assert not p.exists()

    def test_no_error_on_missing_file(self, tmp_path: pathlib.Path) -> None:
        """文件不存在时不抛异常。"""
        p = tmp_path / "nonexistent.txt"
        rm_common.safe_unlink(p)  # 不应抛出


# ─────────────────────────────────────────────────────────────
# TestPlotProgress
# ─────────────────────────────────────────────────────────────

class TestPlotProgress:

    def test_format_plot_progress(self) -> None:
        line = rm_main._format_plot_progress(1, 4, width=8)
        assert line == "[导出进度] |##------| 1/4 ( 25.0%)"

    def test_eta_waits_until_each_fe_type_has_sample(self) -> None:
        estimator = rm_main._PlotProgressEstimator([
            ("firm", "time"),
            ("firm", "_ind_time"),
            ("firm", "time"),
        ])
        first = estimator.update(("firm", "time"), 10.0)
        assert "ETA=等待各FE类型首张样本" in first

        second = estimator.update(("firm", "_ind_time"), 20.0)
        assert "剩余≈10s" in second


# ─────────────────────────────────────────────────────────────
# TestNormalizeControls
# ─────────────────────────────────────────────────────────────

class TestNormalizeControls:

    def test_flat_controls_test(self) -> None:
        """平铺字符串列表正常规范化，每个槽位长度为 1。"""
        flat, slots = rm_py._normalize_controls_test(["a", "b", "c"])
        assert flat == ["a", "b", "c"]
        assert all(len(s) == 1 for s in slots)

    def test_alternative_group_controls_test(self) -> None:
        """嵌套列表视为替代组，槽位长度 >= 2。"""
        flat, slots = rm_py._normalize_controls_test(["a", ["b1", "b2"]])
        assert flat == ["a", "b1", "b2"]
        assert slots[1] == ("b1", "b2")

    def test_space_separated_controls_expand_to_flat_slots(self) -> None:
        """顶层空格分隔字符串会展开为多个普通变量槽位。"""
        flat, slots = rm_py._normalize_controls_test(["a b c"])
        assert flat == ["a", "b", "c"]
        assert slots == [("a",), ("b",), ("c",)]

    def test_space_separated_alternative_group_expands_inside_group(self) -> None:
        """替代组内的空格分隔字符串会展开为同一个互斥组的成员。"""
        flat, slots = rm_py._normalize_controls_test(["a", ["b1 b2", "b3"]])
        assert flat == ["a", "b1", "b2", "b3"]
        assert slots[1] == ("b1", "b2", "b3")

    def test_flat_controls_must(self) -> None:
        """must 平铺规范化正确。"""
        flat, slots = rm_py._normalize_controls_must(["c1", "c2"])
        assert flat == ["c1", "c2"]
        assert len(slots) == 2

    def test_alternative_group_controls_must(self) -> None:
        """must 替代组（必选其一）解析正确。"""
        flat, slots = rm_py._normalize_controls_must([["m1", "m2"], "c"])
        assert set(flat) == {"m1", "m2", "c"}
        assert slots[0] == ("m1", "m2")
        assert slots[1] == ("c",)

    def test_duplicate_variable_across_control_lists_raises(self) -> None:
        """同名变量不能同时出现在 must/test。"""
        with pytest.raises(ValueError, match="变量不可同时出现在 controls_test 和 controls_must 中"):
            rm_py._validate_control_lists_do_not_overlap(["a", "dup"], ["dup", "c"])

    def test_grouping_variable_accepts_continuous_numeric(self, synthetic_df: pd.DataFrame) -> None:
        """grouping_variable 支持连续数值变量。"""
        df = synthetic_df.copy()
        df["grouping"] = np.linspace(-1.0, 1.0, len(df))
        assert rm_py._validate_grouping_variables(df, ["grouping"]) == ["grouping"]

    def test_grouping_variable_requires_numeric_values(self, synthetic_df: pd.DataFrame) -> None:
        """grouping_variable 必须是数值列。"""
        df = synthetic_df.copy()
        df["grouping"] = np.where(df.index % 2 == 0, "low", "high")
        with pytest.raises(ValueError, match="需要是数值变量"):
            rm_py._validate_grouping_variables(df, ["grouping"])

    def test_grouping_specs_allow_same_variable_across_scopes(self, synthetic_df: pd.DataFrame) -> None:
        """同一个连续变量可以按不同中位数口径重复绘图。"""
        df = synthetic_df.copy()
        df["grouping"] = np.linspace(-1.0, 1.0, len(df))
        specs = rm_py._collect_grouping_variable_specs(
            grouping_variable_by_ind_time=["grouping"],
            grouping_variable_by_time=["grouping"],
            grouping_variable_by_none=["grouping"],
        )
        assert rm_py._validate_grouping_variable_specs(df, specs) == [
            ("by_ind_time", "grouping"),
            ("by_time", "grouping"),
            ("by_none", "grouping"),
        ]

    def test_duplicate_variable_raises(self) -> None:
        """重复变量名抛 ValueError。"""
        with pytest.raises(ValueError, match="重复"):
            rm_py._normalize_controls_test(["a", "a"])

    def test_empty_group_raises(self) -> None:
        """空替代组抛 ValueError。"""
        with pytest.raises(ValueError, match="不能为空"):
            rm_py._normalize_controls_test([[]])

    def test_space_separated_flat_name_helper(self) -> None:
        """普通变量列表也支持单项内空格分隔。"""
        names = rm_py._expand_space_separated_names(["y1 y2", "y3"])
        assert names == ["y1", "y2", "y3"]

    def test_stata_grouping_do_uses_dynamic_esample_quantiles(self, tmp_path: pathlib.Path) -> None:
        """分组 Stata do 文件应基于每个 all 回归的 e(sample) 动态二分连续变量。"""
        do_path = tmp_path / "grouped.do"
        log_path = tmp_path / "grouped.log"
        data_path = tmp_path / "input.dta"
        results_path = tmp_path / "results.dta"
        rm_stata._write_reghdfe_do(
            do_path=do_path,
            log_path=log_path,
            data_path=data_path,
            results_dta=results_path,
            y="outcome",
            x="treatment",
            controls_must=["ctrl_c"],
            controls_must_slots=[("ctrl_c",)],
            controls_test=["ctrl_a"],
            controls_test_slots=[("ctrl_a",)],
            spec_def=rm_py._SPEC_CATALOG[0],
            var_map={"firm": "firm_id", "ind": "industry", "time": "year"},
            grouping_variable="sue",
        )
        text = do_path.read_text(encoding="utf-8")
        assert "gen byte `__rm_esample' = e(sample)" in text
        assert "keep if `__rm_esample'" in text
        assert "bysort industry year (firm_id): quantiles sue, gen(_temp) n(2) stable" in text
        assert "gen byte b_sue = _temp - 1" in text
        assert "capture reghdfe outcome treatment ctrl_c if b_sue == `__rm_g'" in text
        assert "drop b_sue" in text

    def test_stata_grouping_do_supports_time_and_none_scopes(self, tmp_path: pathlib.Path) -> None:
        """分组 Stata do 文件支持 time 和 none 两种中位数构造口径。"""
        kwargs = dict(
            log_path=tmp_path / "grouped.log",
            data_path=tmp_path / "input.dta",
            results_dta=tmp_path / "results.dta",
            y="outcome",
            x="treatment",
            controls_must=["ctrl_c"],
            controls_must_slots=[("ctrl_c",)],
            controls_test=["ctrl_a"],
            controls_test_slots=[("ctrl_a",)],
            spec_def=rm_py._SPEC_CATALOG[0],
            var_map={"firm": "firm_id", "ind": "industry", "time": "year"},
            grouping_variable="sue",
        )
        by_time_path = tmp_path / "by_time.do"
        rm_stata._write_reghdfe_do(do_path=by_time_path, grouping_scope="by_time", **kwargs)
        by_time_text = by_time_path.read_text(encoding="utf-8")
        assert "bysort year (firm_id): quantiles sue, gen(_temp) n(2) stable" in by_time_text
        assert '(\"sue[by_time]\")' in by_time_text

        by_none_path = tmp_path / "by_none.do"
        rm_stata._write_reghdfe_do(do_path=by_none_path, grouping_scope="by_none", **kwargs)
        by_none_text = by_none_path.read_text(encoding="utf-8")
        assert "sort firm_id" in by_none_text
        assert "quantiles sue, gen(_temp) n(2) stable" in by_none_text
        assert '(\"sue[by_none]\")' in by_none_text

    def test_stata_interaction_grouping_do_uses_continuous_interaction(self, tmp_path: pathlib.Path) -> None:
        """分组模式的交乘图应估计并抽取 c.x#c.z 的系数。"""
        do_path = tmp_path / "interaction.do"
        rm_stata._write_interaction_reghdfe_do(
            do_path=do_path,
            log_path=tmp_path / "interaction.log",
            data_path=tmp_path / "input.dta",
            results_dta=tmp_path / "results.dta",
            y="outcome",
            x="treatment",
            z="sue",
            controls_must=["ctrl_c"],
            controls_must_slots=[("ctrl_c",)],
            controls_test=["ctrl_a"],
            controls_test_slots=[("ctrl_a",)],
            spec_def=rm_py._SPEC_CATALOG[0],
            var_map={"firm": "firm_id", "ind": "industry", "time": "year"},
        )
        text = do_path.read_text(encoding="utf-8")
        assert "capture reghdfe outcome treatment ctrl_c c.treatment#c.sue, absorb(i.firm_id i.year) vce(cluster firm_id)" in text
        assert "scalar __b = _b[c.treatment#c.sue]" in text
        assert "scalar __se = _se[c.treatment#c.sue]" in text


# ─────────────────────────────────────────────────────────────
# TestSpecCount
# ─────────────────────────────────────────────────────────────

class TestSpecCount:

    def test_two_test_vars_gives_four_specs(self) -> None:
        """2 个独立 test 变量 → 2^2 = 4 个规格。"""
        _, test_slots = rm_py._normalize_controls_test(["a", "b"])
        _, must_slots = rm_py._normalize_controls_must(["c"])
        assert rm_py._spec_count_from_slots(must_slots, test_slots) == 4

    def test_no_test_vars_gives_one_spec(self) -> None:
        """无 test 变量 → 1 个规格。"""
        _, test_slots = rm_py._normalize_controls_test([])
        _, must_slots = rm_py._normalize_controls_must(["c"])
        assert rm_py._spec_count_from_slots(must_slots, test_slots) == 1

    def test_must_group_multiplies_specs(self) -> None:
        """must 替代组（2选1）× test（1个变量 0/1） = 2×2 = 4。"""
        _, test_slots = rm_py._normalize_controls_test(["a"])
        _, must_slots = rm_py._normalize_controls_must([["m1", "m2"]])
        assert rm_py._spec_count_from_slots(must_slots, test_slots) == 4

    def test_alternative_test_group_count(self) -> None:
        """test = ["a", ["b1","b2"]] → (1+1)*(2+1) = 6 个规格。"""
        _, test_slots = rm_py._normalize_controls_test(["a", ["b1", "b2"]])
        _, must_slots = rm_py._normalize_controls_must([])
        assert rm_py._spec_count_from_slots(must_slots, test_slots) == 6

    def test_plot_regression_count_message(self) -> None:
        """每张图回归数提示格式稳定。"""
        msg = rm_py._format_plot_regression_count(3000)
        assert msg == "[本图回归数] 3,000 个回归"

    def test_p_mode_sort_splits_negative_and_positive_coefficients(self) -> None:
        """order=p：负系数在左侧按 p 从大到小，正系数在右侧按 p 从小到大。"""
        def rec(coef: float, p_value: float) -> rm_py.SpecRecord:
            return {
                "coef": coef,
                "se": 1.0,
                "t_value": coef,
                "p_value": p_value,
                "df_resid": 10,
                "ci99_lo": coef - 1,
                "ci99_hi": coef + 1,
                "ci95_lo": coef - 1,
                "ci95_hi": coef + 1,
                "ci90_lo": coef - 1,
                "ci90_hi": coef + 1,
                "controls_test": set(),
                "controls_all": set(),
                "is_full": False,
                "obs": 10,
            }

        records = [
            rec(9.0, 0.001),
            rec(1.0, 0.05),
            rec(-1.0, 0.10),
            rec(-2.0, 0.01),
            rec(2.0, 0.02),
        ]
        sorted_records = rm_py._sort_records_for_plot(records, sort_by_signed_p=True)
        assert [(r["coef"], r["p_value"]) for r in sorted_records] == [
            (-1.0, 0.10),
            (-2.0, 0.01),
            (9.0, 0.001),
            (2.0, 0.02),
            (1.0, 0.05),
        ]
        signs = [float(r["coef"]) < 0 for r in sorted_records]
        assert signs == [True, True, False, False, False]
        neg_p = [float(r["p_value"]) for r in sorted_records if float(r["coef"]) < 0]
        pos_p = [float(r["p_value"]) for r in sorted_records if float(r["coef"]) >= 0]
        assert neg_p == sorted(neg_p, reverse=True)
        assert pos_p == sorted(pos_p)

    def test_plot_order_normalization(self) -> None:
        """order 参数和 --p 兼容别名解析正确。"""
        assert rm_py._normalize_plot_order("coef") == "coef"
        assert rm_py._normalize_plot_order("p") == "p"
        assert rm_py._normalize_plot_order("coef", p_alias=True) == "p"
        with pytest.raises(ValueError, match="order 只能是 coef 或 p"):
            rm_py._normalize_plot_order("bad")


# ─────────────────────────────────────────────────────────────
# TestEndToEndPython
# ─────────────────────────────────────────────────────────────

class TestEndToEndPython:

    def _run(self, synthetic_df: pd.DataFrame, controls_test: list, controls_must: list) -> list:
        records, fig = rm_py.regression_monkey(
            df=synthetic_df,
            y="outcome",
            x="treatment",
            controls_test=controls_test,
            controls_must=controls_must,
            fe_cols=["firm_id", "year"],
            clust_cols=["firm_id"],
            output_path=None,
            n_jobs=1,
            render_plot=False,
            export_sig_table=False,
        )
        assert fig is None  # render_plot=False 不生成 figure
        return records

    def test_returns_correct_spec_count(self, synthetic_df: pd.DataFrame) -> None:
        """controls_test 2 个变量 → 4 条记录（2^2 规格）。"""
        records = self._run(synthetic_df, ["ctrl_a", "ctrl_b"], ["ctrl_c"])
        assert len(records) == 4

    def test_records_sorted_by_coef(self, synthetic_df: pd.DataFrame) -> None:
        """记录按系数从小到大排序。"""
        records = self._run(synthetic_df, ["ctrl_a"], ["ctrl_c"])
        coefs = [r["coef"] for r in records]
        assert coefs == sorted(coefs)

    def test_coef_positive(self, synthetic_df: pd.DataFrame) -> None:
        """治疗效应系数为正（数据生成时设置 β=2.0）。"""
        records = self._run(synthetic_df, [], ["ctrl_c"])
        assert all(r["coef"] > 0 for r in records)

    def test_se_positive(self, synthetic_df: pd.DataFrame) -> None:
        """所有标准误均为正数。"""
        records = self._run(synthetic_df, ["ctrl_a", "ctrl_b"], ["ctrl_c"])
        assert all(r["se"] > 0 for r in records)

    def test_write_and_read_artifacts(
        self, synthetic_df: pd.DataFrame, tmp_path: pathlib.Path
    ) -> None:
        """write_analysis_artifacts 写出 CSV，records_from_dataframe 还原结果无精度损失。"""
        records = self._run(synthetic_df, ["ctrl_a"], ["ctrl_c"])

        results_csv = tmp_path / "results.csv"
        meta_json = tmp_path / "meta.json"
        rm_py.write_analysis_artifacts(
            records=records,
            results_path=results_csv,
            meta_path=meta_json,
            meta={
                "engine": "python",
                "spec_name": "manual",
                "y": "outcome",
                "x": "treatment",
                "controls_test_flat": ["ctrl_a"],
                "controls_must_flat": ["ctrl_c"],
                "matrix_controls": ["ctrl_a"],
                "matrix_alt_groups": [{"kind": "controls_test", "start": 0, "end": 0, "names": ["ctrl_a"], "label": "0/1 of 1"}],
                "show_special_markers": True,
                "fig_width": 8.0,
                "dpi": 72,
                "output_path": str(tmp_path / "out.png"),
            },
        )

        assert results_csv.exists()
        assert meta_json.exists()

        loaded = rm_py.records_from_dataframe(pd.read_csv(results_csv))
        assert len(loaded) == len(records)
        for orig, back in zip(records, loaded):
            assert abs(orig["coef"] - back["coef"]) < 1e-9
            assert abs(orig["se"] - back["se"]) < 1e-9

    def test_html_from_files_writes_interactive_document(
        self, synthetic_df: pd.DataFrame, tmp_path: pathlib.Path
    ) -> None:
        """HTML 渲染脚本读取标准 handoff 文件，并写出悬停交互所需结构。"""
        records = self._run(synthetic_df, ["ctrl_a"], ["ctrl_c"])
        results_csv = tmp_path / "results.csv"
        meta_json = tmp_path / "meta.json"
        output_html = tmp_path / "out.html"
        rm_py.write_analysis_artifacts(
            records=records,
            results_path=results_csv,
            meta_path=meta_json,
            meta={
                "engine": "python",
                "spec_name": "manual",
                "y": "outcome",
                "x": "treatment",
                "controls_test_flat": ["ctrl_a"],
                "controls_must_flat": ["ctrl_c"],
                "matrix_controls": ["ctrl_a"],
                "matrix_alt_groups": [{"kind": "controls_test", "start": 0, "end": 0, "names": ["ctrl_a"], "label": "0/1 of 1"}],
                "show_special_markers": True,
                "fig_width": 8.0,
                "dpi": 72,
                "elapsed_seconds_preplot": 1.25,
                "title_suffix": "absorb(firm_id year) vce(cluster firm_id) - firm and time FE, clustered by firm, baseline specification",
                "output_path": str(tmp_path / "out.png"),
            },
            verbose=False,
        )

        rm_html.html_from_files(
            results_path=results_csv,
            meta_path=meta_json,
            output_path=output_html,
        )

        html_text = output_html.read_text(encoding="utf-8")
        assert 'class="guide"' in html_text
        assert "control-label" in html_text
        assert "sticky-x" in html_text
        assert "sticky-label" in html_text
        assert "@font-face" in html_text
        assert "RM Courier New" in html_text
        assert "data:font/ttf;base64," in html_text
        assert "syncStickyLabels" in html_text
        assert "mouseenter" in html_text
        assert "togglePin" in html_text
        assert "pinnedIdx" in html_text
        assert "onclick" in html_text
        assert '<circle cx="' in html_text
        assert "overflow-x: auto" in html_text
        assert "0/1 of 1" in html_text
        assert 'class="alt-marker sticky-x" stroke="#6B7280"' in html_text
        assert "is_no_controls_test" in html_text
        assert "special-full" in html_text
        assert "special-nocontrol" in html_text
        assert "full-control-cell" in html_text
        assert 'data-special-index="' in html_text
        assert "Elapsed = " in html_text
        assert "@Lachryz" in html_text
        assert "absorb(firm_id year) vce(cluster firm_id)" in html_text
        assert "controls_must = ctrl_c" in html_text
        assert "controls_test = ctrl_a" in html_text
        assert "star-zero-line" in html_text
        assert 'transform="rotate(90' in html_text
        assert 'class="obs-bar"' in html_text
        assert 'rx="1.5"' in html_text
        assert 'data-obs-gap="2"' in html_text
        assert "baseline specification" not in html_text

    def test_html_highlights_control_present_in_all_three_star_specs(self) -> None:
        """所有 3 星显著规格都包含某个 controls_test 时，控制变量名加黄色底纹。"""
        base = {
            "se": 0.1,
            "t_value": 3.0,
            "ci99_lo": 0.1,
            "ci99_hi": 0.4,
            "ci95_lo": 0.12,
            "ci95_hi": 0.38,
            "ci90_lo": 0.14,
            "ci90_hi": 0.36,
            "obs": 100,
            "is_full": False,
            "is_no_controls_test": False,
            "color": "#B91C1C",
            "controls_all": ["ctrl_a"],
            "included_matrix_controls": ["ctrl_a"],
        }
        payload = {
            "title": "outcome × treatment",
            "subtitle": "absorb(firm_id year) vce(cluster firm_id)",
            "controlsMustLine": "controls_must = (none)",
            "controlsTestLine": "controls_test = ctrl_a",
            "controlsTestNames": ["ctrl_a"],
            "y": "outcome",
            "x": "treatment",
            "matrixControls": ["ctrl_a"],
            "matrixAltGroups": [],
            "showSpecialMarkers": True,
            "elapsedSeconds": 1.0,
            "records": [
                {**base, "index": 0, "coef": 0.2, "p_value": 0.005, "star": 3},
                {**base, "index": 1, "coef": 0.1, "p_value": 0.200, "star": 0},
            ],
        }

        html_text = rm_html._build_html(payload)
        assert "☆ ctrl_a" not in html_text
        assert "ctrl_a" in html_text
        assert "control-label-highlight sticky-x" in html_text
        assert "#FDE68A" in html_text

    def test_html_colors_controls_test_alt_groups(self) -> None:
        """controls_test 的多变量子 list 使用同一组深色。"""
        def rec(index: int, controls: list[str], *, is_full: bool = False) -> dict:
            return {
                "index": index,
                "coef": -0.2,
                "se": 0.1,
                "t_value": -2.0,
                "p_value": 0.05,
                "ci99_lo": -0.3,
                "ci99_hi": -0.1,
                "ci95_lo": -0.28,
                "ci95_hi": -0.12,
                "ci90_lo": -0.25,
                "ci90_hi": -0.15,
                "obs": 100,
                "is_full": is_full,
                "is_no_controls_test": False,
                "star": 2,
                "color": "#15803D",
                "controls_all": controls,
                "included_matrix_controls": controls,
            }

        payload = {
            "title": "outcome × treatment",
            "subtitle": "",
            "controlsMustLine": "controls_must = (none)",
            "controlsTestLine": "controls_test = ctrl_a, [ctrl_b, ctrl_c]",
            "controlsTestNames": ["ctrl_a", "ctrl_b", "ctrl_c"],
            "y": "outcome",
            "x": "treatment",
            "matrixControls": ["ctrl_a", "ctrl_b", "ctrl_c"],
            "matrixAltGroups": [
                {"kind": "controls_test", "start": 1, "end": 2, "names": ["ctrl_b", "ctrl_c"], "label": "0/1 of 2"}
            ],
            "showSpecialMarkers": False,
            "elapsedSeconds": None,
            "records": [rec(0, ["ctrl_a", "ctrl_b"]), rec(1, ["ctrl_c"]), rec(2, ["ctrl_b", "ctrl_c"], is_full=True)],
        }

        html_text = rm_html._build_html(payload)
        assert "group-control-cell" in html_text
        assert "group-control-cell full-control-cell" in html_text
        assert "--group-fill: #0B3A75" in html_text
        assert html_text.count("--group-fill: #0B3A75") == 4
        assert '<span class="ctrl-group-title" style="color:#0B3A75">[ctrl_b, ctrl_c]</span>' in html_text
        assert 'data-control="ctrl_b"' in html_text
        assert '--control-label-fill: #0B3A75' in html_text
        assert html_text.index(".matrix-cell.group-control-cell") < html_text.index(".matrix-cell.full-control-cell")

    def test_html_payload_p_order_sorts_negative_p_descending(self) -> None:
        """HTML payload 标记 p 模式时，也按新 p 排序重排嵌入 records。"""
        def rec(index: int, coef: float, p_value: float) -> dict:
            return {
                "index": index,
                "coef": coef,
                "se": 0.1,
                "t_value": coef,
                "p_value": p_value,
                "ci99_lo": coef - 0.1,
                "ci99_hi": coef + 0.1,
                "ci95_lo": coef - 0.08,
                "ci95_hi": coef + 0.08,
                "ci90_lo": coef - 0.05,
                "ci90_hi": coef + 0.05,
                "obs": 100,
                "is_full": False,
                "is_no_controls_test": False,
                "star": 0,
                "color": "#111827",
                "controls_all": [],
                "included_matrix_controls": [],
            }

        payload = {
            "title": "outcome × treatment",
            "subtitle": "",
            "controlsMustLine": "controls_must = (none)",
            "controlsTestLine": "controls_test = (none)",
            "controlsTestNames": [],
            "y": "outcome",
            "x": "treatment",
            "matrixControls": [],
            "matrixAltGroups": [],
            "showSpecialMarkers": False,
            "elapsedSeconds": None,
            "order": "p",
            "sort_by_signed_p": True,
            "records": [
                rec(4, 9.0, 0.001),
                rec(0, -1.0, 0.01),
                rec(1, -2.0, 0.10),
                rec(2, 2.0, 0.02),
                rec(3, 1.0, 0.05),
            ],
        }

        html_text = rm_html._build_html(payload)
        data = json.loads(
            re.search(r"const DATA\s*=\s*(\{.*?\});\n\s*const SIG_COLOR", html_text, re.S).group(1)
        )
        assert [(r["coef"], r["p_value"]) for r in data["records"]] == [
            (-2.0, 0.10),
            (-1.0, 0.01),
            (9.0, 0.001),
            (2.0, 0.02),
            (1.0, 0.05),
        ]

    def test_sig_table_includes_p_t_and_sorts_by_p(
        self, synthetic_df: pd.DataFrame, tmp_path: pathlib.Path
    ) -> None:
        """sig.csv 导出 p/t 列，并按 p_value 从小到大排序。"""
        records = self._run(synthetic_df, ["ctrl_a", "ctrl_b"], ["ctrl_c"])
        rows = rm_py._build_sig_rows(
            records=records,
            y="outcome",
            x="treatment",
            controls_must=["ctrl_c"],
            controls_test=["ctrl_a", "ctrl_b"],
            fe_cols=["firm_id", "year"],
            clust_cols=["firm_id"],
        )
        out_csv = tmp_path / "sig.csv"
        tbl = rm_py._export_sig_table(rows=rows, output_path=str(out_csv), n_specs=len(records), print_summary=False)

        assert out_csv.exists()
        assert tbl is not None
        assert "p_value" in tbl.columns
        assert "t_value" in tbl.columns
        assert tbl["p_value"].tolist() == sorted(tbl["p_value"].tolist())
        assert list(tbl.columns[:4]) == ["Star", "coef", "p_value", "t_value"]

    def test_toml_config_has_required_fields(self, test_toml_path: pathlib.Path) -> None:
        """regression_monkey_test.toml 包含所有必需配置字段。"""
        cfg, _ = rm_common.load_toml_config([str(test_toml_path)])
        cfg_lower = {k.lower(): v for k, v in cfg.items()}
        required = ["y", "x", "controls_test", "controls_must", "firm_fe", "ind_fe", "time_fe"]
        for field in required:
            assert field in cfg_lower, f"TOML 缺少字段：{field}"
