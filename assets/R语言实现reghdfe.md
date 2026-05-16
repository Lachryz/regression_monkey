# R 复现 Stata `reghdfe` 的完整指南

## 一、核心结论：用 `fixest::feols`

**`lfe::felm`**（旧方案）已基本被弃用，
当前标准答案是 `fixest` 包的 `feols` 函数。
从 fixest 0.7.0 版本起，其标准误和 p 值的计算方式
已与 reghdfe 保持一致，涵盖单向和多向聚类标准误。

---

## 二、语法对照表

| 场景 | Stata `reghdfe` | R `fixest::feols` |
| --- | --- | --- |
| 基本固定效应 | `absorb(code year)` | `\| code + year` |
| 交互固定效应 | `absorb(code i.ind#i.year)` | `\| code + ind^year` |
| 单向聚类 | `vce(cluster code)` | `cluster = "code"` |
| 双向聚类 | `vce(cluster code year)` | `cluster = c("code","year")` |
| 异方差稳健 | `vce(robust)` | `vcov = "hetero"` |
| IV（工具变量） | `ivreghdfe` | `y ~ x \| fe \| endo ~ iv` |
| DID 事件研究 | 手动生成交互项 | `i(period, treat, ref)` |

## 关键语法示例

```r
library(fixest)

# 等价于: reghdfe y x1 x2, absorb(code year) vce(cluster code)
feols(y ~ x1 + x2 | code + year, cluster = "code", data = df)

# 等价于: absorb(code i.ind#i.year) vce(cluster code)
feols(y ~ x1 + x2 | code + ind^year, cluster = "code", data = df)

# 双向聚类
feols(y ~ x1 + x2 | code + year, cluster = c("code", "year"), data = df)

# IV: 等价于 ivreghdfe y x1 (x_endo = iv1 iv2), absorb(code year)
feols(y ~ x1 | code + year | x_endo ~ iv1 + iv2, cluster = "code", data = df)
```

---

## 三、标准误一致性的核心机制：DOF 校正

这是 R 与 Stata 产生差异的最主要来源，需要重点理解。

## 3.1 `fixest` 的 `ssc()` 函数

标准误差由两个乘数共同决定：

### ① VCOV 乘数（方差-协方差矩阵缩放）

$$\hat{V} = \frac{N-1}{N-K} \cdot \frac{G}{G-1} \cdot B$$

- $N$：观测数；$K$：估计参数数（含 FE）；$G$：聚类数
- `K.adj = TRUE`（默认）对应 $\frac{N-1}{N-K}$
- `G.adj = TRUE`（默认）对应 $\frac{G}{G-1}$

### ② t 统计量自由度（影响 p 值）

- `t.df = "min"`（默认）：使用最小聚类数 $-1$，与 reghdfe 一致
- `t.df = "conventional"`：使用 $N - K$（不推荐，会偏离 reghdfe）

## 3.2 FE 消耗自由度的处理

reghdfe 的一项关键步骤是计算因固定效应损失的自由度——在超过两个固定效应层级时，这仍是一个开放问题，reghdfe 提供保守近似。

`fixest` 的 `K.fixef` 参数对应这一处理：

| 参数值 | 含义 | 对应 reghdfe 行为 |
| --- | --- | --- |
| `"nonnested"`（默认） | 仅计数非嵌套的 FE | ✅ 与 reghdfe 一致 |
| `"full"` | 计数所有 FE 虚拟变量 | ❌ 过于保守 |
| `"none"` | 不扣除 FE 自由度 | ❌ 过于宽松 |

## 3.3 Singleton 删除

fixest 中 `fixef.rm = "singletons"`
（或 `"perfect_fit"`，为默认值）会递归删除单例观测——
在一个 FE 中删除单例可能在另一个 FE 中制造新的单例，
算法持续迭代直到无单例残留。
这与 reghdfe 迭代删除 singleton 观测
以避免标准误偏误的机制完全对应。

---

## 四、`lfe::felm` 的遗留问题（为何不推荐）

lfe 与 fixest 的差异在于：lfe 的聚类标准误不能被 feols 直接复现——即使设置相同的参数，二者数值仍然不同。若一定要用 `lfe`，则需要：

```r
felm(y ~ x | code + year | 0 | code,
     data = df, cmethod = 'cgm2', exactDOF = TRUE)
```

需要注意的是，必须使用最新版本的 lfe 才能使用 `cgm2` 方法，否则标准误将与 Stata 不同。

---

## 五、实践中的常见差异来源（排查清单）

| 差异来源 | reghdfe 默认行为 | R 侧注意事项 |
| --- | --- | --- |
| Singleton | 递归删除 | `fixef.rm = "perfect_fit"` 已默认处理 |
| FE 嵌套 | nonnested DOF | `ssc(K.fixef = "nonnested")` 已默认 |
| 聚类 G 校正 | `G/(G-1)` | `ssc(G.adj = TRUE)` 已默认 |
| t 分布自由度 | min cluster - 1 | `ssc(t.df = "min")` 已默认 |
| 交互 FE（`i.ind#i.year`） | 列出所有组合 | `ind^year` 在 `\|` 后，等价 |
| 含权重 | `[aw=weight]` | `weights = ~weight` |
| 多向聚类方向 | 无方向依赖 | `cluster = c("a","b")` 顺序不影响 |

> **重要提示**：如果复现结果仍有差异，
> 首先用一个**简单单向聚类、单 FE** 的情形对齐，
> 再逐步增加复杂度。
> 差异最可能来自 singleton 数量不同
> （检查 `df$fixef_removed`）或 FE 中是否存在完美共线。

---

## 六、输出表格：`etable` vs Stata 的 `esttab`

`fixest` 配套的 `etable()` 可直接生成学术格式表格，语法更接近 Stata 的 `esttab`：

```r
etable(model1, model2, model3,
       cluster = "code",          # 统一指定SE
       digits = 3,
       stars = c("*"=0.1,"**"=0.05,"***"=0.01))
```

配合 `modelsummary` 包可输出 Word/LaTeX 格式，与 Stata `outreg2`/`esttab` 对应。

---

## 小结

**推荐路径**：`fixest::feols` + 默认 `ssc()` 参数，
在绝大多数学术场景
（`absorb(code year)` 或 `absorb(code i.ind#i.year)`，
`vce(cluster code)`）
下可直接得到与 `reghdfe` 数值上一致的结果，
无需手动调整自由度校正。
