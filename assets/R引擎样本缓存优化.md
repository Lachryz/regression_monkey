# R 引擎样本缓存优化说明

本文记录 `src/regression_monkey/engine/r.py` 中 R/fixest 路径的性能修复逻辑。背景场景是单张图约 8,192 个规格、约 86,736 行样本，原始目标耗时约 22 秒。

## 问题现象

R 引擎需要对每个控制变量组合计算有效样本，然后复用 `fixest::demean()` 后的吸收矩阵做 `lm.fit()`。有效样本由以下条件共同决定：

- `y`、`x`、固定效应列、聚类列非缺失
- 每个规格实际包含的 `controls_must` / `controls_test` 非缺失
- 开启 `drop_singletons` 时，按固定效应迭代删除 singleton 观测

修复 R 变量名长度报错时曾出现两类性能退化：

- 把完整样本 mask 打包成 20KB 以上长字符串并作为 cache key，会触发 R 的变量名长度限制。
- 改成 list + `match()` 或对长 key 再做哈希，虽然避开了变量名限制，但在 8,192 个规格中反复线性查找或扫描长字符串，导致单图耗时从约 22 秒升到约 33 到 48 秒。

## 最终修复逻辑

最终实现要同时满足两个条件：cache key 足够短，且正式回归阶段不重复计算样本 mask。

### 1. 短样本 key

`mask_key()` 不再返回完整 packed mask 字符串，而是用 `packBits()` 后的 raw bytes 计算几个向量化校验和：

```r
mask_key <- function(m) {
  if (anyNA(m)) m[is.na(m)] <- FALSE
  pad <- (-length(m)) %% 8L
  bits <- if (pad > 0L) c(m, rep(FALSE, pad)) else m
  vals <- as.integer(packBits(bits))
  idx <- seq_along(vals)
  h1 <- sum((vals + 1) * ((idx %% 1009) + 1)) %% 2147483647
  h2 <- sum((vals + 3) * ((idx %% 9176) + 7)) %% 2147483629
  h3 <- sum((vals + 5) * ((idx %% 65521) + 11)) %% 2147483587
  paste0("k_", length(m), "_", sum(m), "_", as.integer(h1), "_", as.integer(h2), "_", as.integer(h3))
}
```

这样 86,736 行样本会生成几十字节的 key，例如：

```text
k_86736_52042_828419494_352413113_716171182
```

短 key 可以安全地作为 R environment 名称使用：

```r
cache <- new.env(hash = TRUE, parent = emptyenv())
cache[[key]] <- value
cache[[key]]
```

这恢复了接近 O(1) 的 cache lookup，同时避免 R 的 10,000 字节变量名限制。

### 2. 公共 base mask

每个规格都共享一批基础非缺失条件，包括 `y`、`x`、固定效应列、聚类列，以及所有规格共同包含的 `controls_must`。这些条件只计算一次：

```r
common_must <- if (length(spec_must)) Reduce(intersect, spec_must) else character(0)
base_spec_vars <- unique(c(y_var, x_var, common_must, cluster_vars, fe_base_vars_all))
base_mask <- fast_complete(base_spec_vars)
```

每个规格只在 `base_mask` 上追加本规格相对 `common_must` 多出来的控制变量条件：

```r
complete_from_base <- function(chosen_must, chosen_test) {
  mask <- base_mask
  extra_vars <- unique(setdiff(c(chosen_must, chosen_test), common_must))
  for (v in extra_vars) {
    if (v %in% names(col_notna)) mask <- mask & col_notna[[v]]
  }
  mask
}
```

这避免了对每个规格重复检查所有固定效应列、聚类列和固定控制变量。

### 3. 预扫描 sample key

预扫描阶段负责一次性完成：

- 计算每个规格的 mask
- 执行 singleton 删除
- 生成该规格的 sample key
- 收集唯一 mask 并预热 `fixest::demean()` cache

正式回归阶段不再重复 `fast_complete()`、`drop_singletons_c()` 或 `mask_key()`，而是直接用预扫描保存的 key 取样本：

```r
spec_sample_keys <- rep(NA_character_, length(spec_ids))
spec_valid <- rep(FALSE, length(spec_ids))

# pre-scan
spec_sample_keys[[i]] <- k
spec_valid[[i]] <- TRUE

# regression
if (!spec_valid[[i]]) return(NULL)
sample <- get_sample_by_key(spec_sample_keys[[i]])
```

唯一 mask 预热时也复用已算好的 key：

```r
for (j in seq_along(unique_masks)) {
  get_sample(unique_masks[[j]], unique_sample_keys[[j]])
}
```

## 性能结果

使用真实配置验证时，可用任意保存下来的 `config_snapshot.toml`，并限制到单个 `Y × X` 以便比较单图耗时：

```bash
uv run regression-monkey outputs/<timestamp>/config_snapshot.toml --y analyst_error_quarter_pa --x quant
```

近期在 86,736 行、8,192 个回归的真实配置上，单图进度输出稳定在约 18 到 20 秒：

```text
[本图回归数] 8,192 个回归
[导出进度] ... 本张=18s  # 或同量级的 20s
```

这个结果低于原先约 22 秒的目标，并明显优于退化版本的 33 到 48 秒。

## 后续维护规则

- 不要把完整 packed mask 字符串作为 R environment key；大样本下会超过 R 变量名限制。
- 不要用 list + `match()` 管理样本 cache；规格数增长后会线性退化。
- 不要在 `process_one_spec()` 中重新计算样本 mask、singleton 删除或 sample key；这些应保留在预扫描阶段。
- 如果后续改动 effective-sample 逻辑，必须同时检查 `base_mask`、`complete_from_base()`、`spec_sample_keys` 和 `get_sample_by_key()` 是否仍然一致。
- 如果需要调试 R 端耗时，建议先用单个 `Y × X` 加真实配置验证，例如：

```bash
uv run regression-monkey outputs/<timestamp>/config_snapshot.toml --y analyst_error_quarter_pa --x quant
```
