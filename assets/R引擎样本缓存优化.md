# R 引擎优化说明

本文记录 `src/regression_monkey/engine/r.py` 中 R/fixest 路径的全部性能优化逻辑。背景场景是单张图约 8,192 个规格、约 86,736 行样本，原始目标耗时约 22 秒。

## 一、数据通道优化（Python 侧）

### 1.1 列裁剪

写出输入数据集之前，先计算本次运行实际用到的列子集，避免把无关列序列化到磁盘：

```python
def _required_cols_for_run(df, args, controls_must_flat, controls_test_flat, var_map, spec_flags):
    cols_numeric = {y, x} | set(controls_must_flat) | set(controls_test_flat)
    cols_group = set()  # FE 底层列 + cluster 列，跨所有启用规格取并集
    ...
    all_cols = sorted((cols_numeric | cols_group) & set(df.columns))
    factor_cols = (cols_group - cols_numeric) & set(all_cols)
    return all_cols, factor_cols
```

`factor_cols` 是只用作 FE / cluster 的纯分组列，可以安全地用 `int32` 编码，进一步压缩传输体积。

### 1.2 R 包探测

每次 `run_r_engine` 调用一次 Rscript 子进程，检测 R 端是否安装了 `arrow` 和 `data.table`：

```python
def _probe_r_packages(rscript_path: str) -> dict[str, bool]:
    script = "cat(as.integer(requireNamespace('arrow', quietly=TRUE)), ..."
    ...
    return {"arrow": parts[0] == "1", "data.table": parts[1] == "1"}
```

探测结果决定后续数据通道选择。

### 1.3 feather 通道（需 R arrow 包）

当 `r_packages["arrow"]` 为真时，Python 写出 `.feather` 文件；否则回退 CSV。

- feather 保留 IEEE-754 double 精度，无转换损失。
- `factor_cols` 列用 `pa.int32()` 类型加 null bitmap 编码（原始 `NaN` → null），体积缩小约 75%。

```python
def _write_input_dataset(df, path, factor_cols, *, use_feather: bool) -> None:
    if use_feather:
        # pa.int32 for factor_cols, pa.array(s) for others
        ...
        pa_feather.write_feather(table, str(path))
    else:
        df.to_csv(path, index=False, encoding="utf-8")
```

## 二、R 端读写自适配

R 脚本根据 Python 注入的 `data_format` 变量以及运行时包检测结果选择读取方式：

```r
read_dataset <- function(path) {
  if (data_format == 'feather' && has_arrow)
    return(as.data.frame(arrow::read_feather(path)))
  if (has_dt)
    return(as.data.frame(data.table::fread(path, check.names = FALSE)))
  read.csv(path, check.names = FALSE, stringsAsFactors = FALSE)
}
```

三级回退：feather（最快） → `data.table::fread`（较快） → base `read.csv`（兜底）。
结果写出也有类似三级回退，当 `data.table` 可用时用 `fwrite`，否则用 `write.csv`。

## 三、规格 JSON 预解析

早期版本在每个 `process_one_spec(i)` 内部调用 `read_json_vec`。现在在主进程加载 specs 后一次性解析：

```r
spec_must   <- lapply(specs$chosen_must_controls, read_json_vec)
spec_test   <- lapply(specs$chosen_test_controls, read_json_vec)
spec_is_full <- as.logical(specs$is_full)
```

`process_one_spec(i)` 直接读取 `spec_must[[i]]`、`spec_test[[i]]`、`spec_is_full[[i]]`，
避免在 8,192 个规格中重复解析 JSON 字符串。

## 四、样本缓存

### 4.1 问题背景

早期版本维护两个平行向量 `sample_masks` / `sample_values`，每次命中检查都用
`identical(mask, sample_masks[[j]])` 线性扫描，K 个规格累计 O(K²·N) 代价。

修复时曾出现两类退化：

- 把完整 packed mask 打包成 20 KB 以上长字符串作为 key → 超过 R 变量名 10,000 字节限制。
- 改成 list + `match()` 或对长 key 再做哈希 → 在 8,192 个规格中反复线性查找或扫描长字符串，单图耗时从约 22 秒升至约 33 到 48 秒。

### 4.2 短样本 key

`mask_key()` 用 `packBits()` 后的 raw bytes 计算三个多项式校验和，生成几十字节的 key：

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

86,736 行样本生成的 key 仅约 40 字节，例如：

```text
k_86736_52042_828419494_352413113_716171182
```

Key 中包含 `length(m)` 和 `sum(m)` 作为快速鉴别字段，三个哈希值覆盖不同模数，碰撞概率极低。

### 4.3 environment 哈希 cache

短 key 可以安全地用作 R `environment` 的变量名，实现 O(1) 命中：

```r
sample_cache <- new.env(hash = TRUE, parent = emptyenv())

get_sample <- function(mask, key = NULL) {
  if (is.null(key)) key <- mask_key(mask)
  cached <- cache_get(sample_cache, key)
  if (!is.null(cached)) return(cached)
  # ... 计算并存储
  cache_set(sample_cache, key, value)
  value
}
```

`drop_singletons` 选项同样维护独立的 `ds_cache` environment，避免重复迭代删除 singleton。

### 4.4 公共 base mask

每个规格都共享一批基础非缺失条件（`y`、`x`、所有规格共有的 `controls_must`、cluster 列、FE 底层列）。这些条件只计算一次：

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

### 4.5 预扫描 + spec_sample_keys

预扫描阶段一次性完成：计算每个规格的 mask、执行 singleton 删除（可选）、生成该规格的 sample key、收集唯一 mask 并预热 `fixest::demean()` cache。

正式回归阶段不再重复任何 mask 计算，直接读预存的 key：

```r
spec_sample_keys <- rep(NA_character_, length(spec_ids))
spec_valid <- rep(FALSE, length(spec_ids))

# 预扫描
spec_sample_keys[[i]] <- k
spec_valid[[i]] <- TRUE

# 正式回归
if (!spec_valid[[i]]) return(NULL)
sample <- get_sample_by_key(spec_sample_keys[[i]])
```

唯一 mask 预热时复用已算好的 key，避免重复 `mask_key()`：

```r
for (j in seq_along(unique_masks)) get_sample(unique_masks[[j]], unique_sample_keys[[j]])
```

## 五、因子代码缓存（fe_codes）

每次 `get_sample` 建立样本时，同步构建该样本的因子代码缓存：

```r
fe_codes <- lapply(fe_list, function(v) {
  f <- factor(v)
  list(codes = as.integer(f), nlev = nlevels(f))
})
cluster_factors <- lapply(cluster_vars, function(nm) factor(work[[nm]]))
```

`k_fe_full` 和 `k_fe_se` 随样本一起存入 cache value，`process_one_spec` 直接读取，
避免对每个规格重复调用 `factor()` 和 `nlevels()`。

`connected_components_from_codes()`、`k_fe_count_from_codes()`、`k_fe_nonnested_from_codes()` 等函数接收 `fe_codes` 结构，代替反复从 data frame 重建因子。

## 六、`.lm.fit` 替换 `lm.fit`

```r
fit <- tryCatch(.lm.fit(X, y_dm), error = function(e) NULL)
```

`.lm.fit` 跳过参数检查，直接调用底层 LAPACK QR 分解，与 `lm.fit` 数值完全一致。
因返回值不带 `names`，需手动恢复：

```r
names(coef_full) <- kept_vars          # 恢复 coef 名称
x_idx <- match(x_var, kept_vars)       # 按位置索引，不再按名称
```

## 七、结果合并优化

```r
if (has_dt) {
  res <- as.data.frame(data.table::rbindlist(out, fill = TRUE))
} else {
  res <- do.call(rbind, out)
}
```

`data.table::rbindlist` 比 `do.call(rbind, ...)` 快约一个数量级，对 8,192 个结果行效果明显。

## 八、多路聚类的处理路径

### 8.1 问题

在 2-way cluster 路径中，手写 `calc_vcov` 与 `feols` 在标准误上存在约 0.3% 系统偏差，
根因是 `fixest::ssc()` 默认 `cluster.df='conventional'`，对每个 cluster 维度分别施加
`G_i/(G_i-1)` 校正：

```
V = (G1/(G1-1))·meat_cl1 + (G2/(G2-1))·meat_cl2 - (G12/(G12-1))·meat_cl12
```

而 CGM(2011) 标准 SSC 用全局 `Gmin/(Gmin-1)` 统一校正，两者不等价，
在 G1 ≪ G2 时差异尤为明显。重现 fixest 内部的 per-cluster 校正逻辑需要访问其私有接口，
维护成本过高。

### 8.2 决策

1-way / robust：走共享 demean cache + `.lm.fit` + `calc_vcov` 路径，数值与 fixest 完全一致（已验证）。

2-way：保留 `run_feols_spec` fallback，直接调用 `fixest::feols`，完全沿用 fixest 原生 SSC。

```r
process_one_spec <- function(i) {
  if (length(cluster_vars) > 1L) {
    # 2-way cluster：直接走 feols，避免 SSC 分歧
    exact_row <- run_feols_spec(chosen_must, chosen_test)
    if (!is.null(exact_row)) exact_row$is_full <- spec_is_full[[i]]
    return(exact_row)
  }
  # 1-way / robust：走共享 demean cache 路径
  ...
}
```

## 九、mclapply COW 预热门控

主进程预热 `sample_cache` 后 fork，子进程通过 copy-on-write 零开销共享缓存。
由于 2-way cluster 路径绕过 `get_sample`，预热循环仅对单路或 robust 规格执行：

```r
if (length(cluster_vars) <= 1L) for (i in spec_ids) {
  mask_i <- complete_from_base(spec_must[[i]], spec_test[[i]])
  if (drop_singletons_option) mask_i <- drop_singletons_c(mask_i)
  if (sum(mask_i) <= 1L) next
  k <- mask_key(mask_i)
  spec_sample_keys[[i]] <- k
  spec_valid[[i]] <- TRUE
  if (is.null(cache_get(seen_sample_keys, k))) {
    cache_set(seen_sample_keys, k, TRUE)
    unique_masks[[length(unique_masks) + 1L]] <- mask_i
    unique_sample_keys <- c(unique_sample_keys, k)
  }
}
for (j in seq_along(unique_masks)) get_sample(unique_masks[[j]], unique_sample_keys[[j]])
```

Windows 平台 `mclapply` 自动退回串行，预热同样有效（顺序执行时缓存复用更明显）。

## 性能结果

在 86,736 行、8,192 个回归的真实配置上，单图进度输出稳定约 18 到 20 秒：

```text
[本图回归数] 8,192 个回归
[导出进度] ... 本张=18s
```

低于原始约 22 秒的目标，明显优于退化版本的 33 到 48 秒。

可用保存下来的 `config_snapshot.toml` 加 `--y`/`--x` 限定单图验证：

```bash
uv run regression-monkey outputs/<timestamp>/config_snapshot.toml \
    --y analyst_error_quarter_pa --x quant
```

## 后续维护规则

- 不要把完整 packed mask 字符串作为 R environment key；大样本下会超过 R 变量名 10,000 字节限制。
- 不要用 list + `match()` 管理样本 cache；规格数增长后会线性退化。
- 不要在 `process_one_spec()` 中重新计算样本 mask、singleton 删除或 sample key；这些应保留在预扫描阶段。
- 2-way cluster 路径必须走 `run_feols_spec`；手写 `calc_vcov` 的 SSC 无法精确复现 fixest `cluster.df='conventional'` 的 per-cluster 校正。
- 如果后续改动 effective-sample 逻辑，必须同时检查 `base_mask`、`complete_from_base()`、`spec_sample_keys` 和 `get_sample_by_key()` 是否仍然一致。
- `_probe_r_packages` 在每次 `run_r_engine` 调用时执行一次（约 0.5 秒）；如在同一进程内多次调用，可考虑缓存结果，但目前调用频率低，不需要优化。
- 列裁剪后，`factor_cols` 限定为纯分组列（只做 FE / cluster，不作为数值回归变量）；如某列兼作 FE 和控制变量，它会保留在 `cols_numeric` 中，不会被 int32 因子化，避免精度损失。
