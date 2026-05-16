"""
regression_monkey · r
=====================
Use Rscript + fixest::demean to run fast reghdfe-style specification curves.
"""

from __future__ import annotations

from datetime import datetime
import argparse
import itertools
import json
import pathlib
import subprocess
import sys
from time import perf_counter
from typing import Any, Callable, cast

import numpy as np
import pandas as pd

from .. import common as rm_common
from . import py as rm


def _r_string(text: str) -> str:
    return json.dumps(text, ensure_ascii=False)


def _r_vector(values: list[str]) -> str:
    if not values:
        return "character(0)"
    return "c(" + ", ".join(_r_string(v) for v in values) + ")"


def _prepare_auto_dataframe(
    df: pd.DataFrame,
    specs: dict[str, bool],
    firm_fe: str,
    ind_fe: str,
    time_fe: str,
    region_fe: str | None,
) -> tuple[pd.DataFrame, dict[str, str], dict[str, str]]:
    _ = specs
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
    subsets: list[tuple[int, list[str], list[str], bool]] = []
    total_specs = rm._spec_count_from_slots(controls_must_slots, controls_test_slots)
    for bits in range(total_specs):
        rem, _chosen_must_cols, chosen_must = rm._decode_required_choice(bits, controls_must_slots)
        _chosen_test_cols, chosen_test, is_full = rm._decode_optional_choice(rem, controls_test_slots)
        subsets.append((bits, chosen_must, chosen_test, is_full))
    return subsets


def _spec_fe_labels(spec_def: dict[str, Any], var_map: dict[str, str]) -> list[str]:
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
            labels.append(f"{var_map['ind']}^{var_map['time']}")
        elif key == "_region_time":
            labels.append(f"{var_map['region']}^{var_map['time']}")
        else:
            raise ValueError(f"unknown FE key: {key}")
    return labels


def _spec_r_fe_terms(spec_def: dict[str, Any], var_map: dict[str, str]) -> list[str]:
    terms: list[str] = []
    for key in spec_def["fe_keys"]:
        if key == "firm":
            terms.append(var_map["firm"])
        elif key == "ind":
            terms.append(var_map["ind"])
        elif key == "time":
            terms.append(var_map["time"])
        elif key == "region":
            terms.append(var_map["region"])
        elif key == "_ind_time":
            terms.append(f"{var_map['ind']}^{var_map['time']}")
        elif key == "_region_time":
            terms.append(f"{var_map['region']}^{var_map['time']}")
        else:
            raise ValueError(f"unknown FE key: {key}")
    return terms


def _write_r_specs_csv(
    path: pathlib.Path,
    *,
    controls_must_slots: list[rm.ControlSlot],
    controls_test_slots: list[rm.ControlSlot],
) -> None:
    rows: list[dict[str, Any]] = []
    for bits, chosen_must, chosen_test, is_full in _enumerate_control_specs(controls_must_slots, controls_test_slots):
        rows.append({
            "bits": bits,
            "chosen_must_controls": json.dumps(chosen_must, ensure_ascii=False),
            "chosen_test_controls": json.dumps(chosen_test, ensure_ascii=False),
            "is_full": bool(is_full),
        })
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8")


def _required_cols_for_run(
    df: pd.DataFrame,
    args: argparse.Namespace,
    controls_must_flat: list[str],
    controls_test_flat: list[str],
    var_map: dict[str, str],
    spec_flags: dict[str, bool],
) -> tuple[list[str], set[str]]:
    cols_numeric: set[str] = set()
    for y in args.y:
        cols_numeric.add(y)
    for x in args.x:
        cols_numeric.add(x)
    cols_numeric.update(controls_must_flat)
    cols_numeric.update(controls_test_flat)

    cols_group: set[str] = set()
    for spec_def in rm._SPEC_CATALOG:
        if not spec_flags.get(spec_def["name"], False):
            continue
        if spec_def["needs_region"] and args.region_fe is None:
            continue
        for k in spec_def["cl_keys"]:
            cols_group.add(var_map[k])
        for term in _spec_r_fe_terms(spec_def, var_map):
            for part in term.split("^"):
                if part:
                    cols_group.add(part)

    all_cols = sorted((cols_numeric | cols_group) & set(df.columns))
    factor_cols = (cols_group - cols_numeric) & set(all_cols)
    return all_cols, factor_cols


def _probe_r_packages(rscript_path: str) -> dict[str, bool]:
    script = (
        "cat(as.integer(requireNamespace('arrow', quietly=TRUE)),"
        "as.integer(requireNamespace('data.table', quietly=TRUE)),sep=',')"
    )
    try:
        proc = subprocess.run(
            [rscript_path, "-e", script],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return {"arrow": False, "data.table": False}
    parts = (proc.stdout or "").strip().split(",")
    if len(parts) != 2:
        return {"arrow": False, "data.table": False}
    return {"arrow": parts[0] == "1", "data.table": parts[1] == "1"}


def _write_input_dataset(
    df: pd.DataFrame,
    path: pathlib.Path,
    factor_cols: set[str],
    *,
    use_feather: bool,
) -> None:
    if use_feather:
        import pyarrow as pa
        import pyarrow.feather as pa_feather

        arrays: dict[str, Any] = {}
        for col in df.columns:
            s = df[col]
            if col in factor_cols:
                codes_arr, _uniques = pd.factorize(s, use_na_sentinel=True)
                mask = codes_arr == -1
                codes_safe = np.where(mask, 0, codes_arr).astype(np.int32)
                arrays[col] = pa.array(codes_safe, mask=mask, type=pa.int32())
            else:
                arrays[col] = pa.array(s)
        table = pa.Table.from_pydict(arrays)
        pa_feather.write_feather(table, str(path))
    else:
        df.to_csv(path, index=False, encoding="utf-8")


def _write_fixest_script(
    *,
    script_path: pathlib.Path,
    data_path: pathlib.Path,
    data_format: str,
    specs_csv: pathlib.Path,
    results_path: pathlib.Path,
    y: str,
    x: str,
    fe_terms: list[str],
    clust_cols: list[str],
    robust: bool,
    nthreads: int,
    drop_singletons_option: bool = False,
) -> None:
    se_kind = "robust" if robust else "cluster"
    lines: list[str] = [
        "options(warn = 1)",
        "if (!requireNamespace('fixest', quietly = TRUE)) {",
        "  stop('R package fixest is not installed. Install it with install.packages(\"fixest\").')",
        "}",
        "has_arrow <- requireNamespace('arrow', quietly = TRUE)",
        "has_dt <- requireNamespace('data.table', quietly = TRUE)",
        f"data_format <- {_r_string(data_format)}",
        "qvar <- function(x) paste0('`', gsub('`', '``', x, fixed = TRUE), '`')",
        "json_escape <- function(x) {",
        "  x <- gsub('\\\\\\\\', '\\\\\\\\\\\\\\\\', x)",
        "  x <- gsub('\"', '\\\\\"', x)",
        "  paste0('\"', x, '\"')",
        "}",
        "json_vec <- function(xs) {",
        "  if (length(xs) == 0) return('[]')",
        "  paste0('[', paste(vapply(xs, json_escape, character(1)), collapse = ','), ']')",
        "}",
        "json_stats <- function(stats) {",
        "  if (length(stats) == 0) return('[]')",
        "  pieces <- vapply(stats, function(s) {",
        "    paste0('{\"name\":', json_escape(s[['name']]),",
        "           ',\"coef\":', as.character(s[['coef']]),",
        "           ',\"se\":', as.character(s[['se']]),",
        "           ',\"t_value\":', as.character(s[['t_value']]),",
        "           ',\"p_value\":', as.character(s[['p_value']]), '}')",
        "  }, character(1))",
        "  paste0('[', paste(pieces, collapse = ','), ']')",
        "}",
        "read_json_vec <- function(x) {",
        "  x <- trimws(x)",
        "  if (x == '[]') return(character(0))",
        "  x <- sub('^\\\\[', '', sub('\\\\]$', '', x))",
        "  if (!nzchar(x)) return(character(0))",
        "  parts <- strsplit(x, ',', fixed = TRUE)[[1]]",
        "  parts <- trimws(parts)",
        "  parts <- sub('^\"', '', sub('\"$', '', parts))",
        "  parts <- gsub('\\\\\"', '\"', parts)",
        "  gsub('\\\\\\\\', '\\\\', parts)",
        "}",
        "scalar1 <- function(x) {",
        "  y <- suppressWarnings(as.numeric(x))",
        "  if (length(y) == 0) return(NA_real_)",
        "  y[[1]]",
        "}",
        "fitstat_value <- function(model, type, field = NULL) {",
        "  value <- fixest::fitstat(model, type)[[type]]",
        "  if (!is.null(field) && is.list(value) && !is.null(value[[field]])) {",
        "    return(scalar1(value[[field]]))",
        "  }",
        "  scalar1(value)",
        "}",
        "wald_stat_value <- function(model) {",
        "  value <- tryCatch(fixest::wald(model, print = FALSE), error = function(e) NULL)",
        "  if (is.null(value) || is.null(value$stat)) return(NA_real_)",
        "  scalar1(value$stat)",
        "}",
        "fe_formula_part <- function(term) {",
        "  parts <- strsplit(term, '^', fixed = TRUE)[[1]]",
        "  paste(vapply(parts, qvar, character(1)), collapse = '^')",
        "}",
        "run_feols_spec <- function(chosen_must, chosen_test) {",
        "  rhs_vars <- unique(c(x_var, chosen_must, chosen_test))",
        "  rhs <- paste(vapply(rhs_vars, qvar, character(1)), collapse = ' + ')",
        "  fe_rhs <- paste(vapply(fe_terms, fe_formula_part, character(1)), collapse = ' + ')",
        "  fml <- stats::as.formula(paste0(qvar(y_var), ' ~ ', rhs, ' | ', fe_rhs))",
        "  model <- tryCatch({",
        "    if (se_kind == 'robust') {",
        "      fixest::feols(fml, data = df, vcov = 'hetero', ssc = fixest::ssc(K.fixef = 'nonnested', t.df = 'min'), nthreads = demean_nthreads, notes = FALSE)",
        "    } else {",
        "      cluster_rhs <- paste(vapply(cluster_vars, qvar, character(1)), collapse = ' + ')",
        "      fixest::feols(fml, data = df, cluster = stats::as.formula(paste('~', cluster_rhs)), ssc = fixest::ssc(K.fixef = 'nonnested', t.df = 'min'), nthreads = demean_nthreads, notes = FALSE)",
        "    }",
        "  }, error = function(e) NULL)",
        "  if (is.null(model)) return(NULL)",
        "  ct <- tryCatch(fixest::coeftable(model), error = function(e) NULL)",
        "  if (is.null(ct) || !(x_var %in% rownames(ct))) return(NULL)",
        "  coef <- unname(ct[x_var, 'Estimate'])",
        "  se <- unname(ct[x_var, 'Std. Error'])",
        "  t_value <- unname(ct[x_var, 't value'])",
        "  p_value <- unname(ct[x_var, 'Pr(>|t|)'])",
        "  if (!is.finite(coef) || !is.finite(se) || se <= 0) return(NULL)",
        "  df_resid <- max(1L, as.integer(stats::df.residual(model)))",
        "  crit99 <- stats::qt(0.995, df = df_resid)",
        "  crit95 <- stats::qt(0.975, df = df_resid)",
        "  crit90 <- stats::qt(0.950, df = df_resid)",
        "  kept_controls <- c(chosen_must, chosen_test)[c(chosen_must, chosen_test) %in% rownames(ct)]",
        "  kept_test <- chosen_test[chosen_test %in% kept_controls]",
        "  ctrl_stats <- list()",
        "  for (ctrl in kept_controls) {",
        "    ctrl_stats[[length(ctrl_stats) + 1L]] <- list(",
        "      name = ctrl,",
        "      coef = unname(ct[ctrl, 'Estimate']),",
        "      se = unname(ct[ctrl, 'Std. Error']),",
        "      t_value = unname(ct[ctrl, 't value']),",
        "      p_value = unname(ct[ctrl, 'Pr(>|t|)'])",
        "    )",
        "  }",
        "  data.frame(",
        "    coef = coef, se = se, t_value = t_value, p_value = p_value,",
        "    adj_r2 = fitstat_value(model, 'ar2'),",
        "    within_r2 = fitstat_value(model, 'wr2'),",
        "    f_stat = wald_stat_value(model),",
        "    df_resid = df_resid,",
        "    ci99_lo = coef - crit99 * se, ci99_hi = coef + crit99 * se,",
        "    ci95_lo = coef - crit95 * se, ci95_hi = coef + crit95 * se,",
        "    ci90_lo = coef - crit90 * se, ci90_hi = coef + crit90 * se,",
        "    controls_test = json_vec(kept_test),",
        "    controls_all = json_vec(kept_controls),",
        "    control_stats = json_stats(ctrl_stats),",
        "    is_full = FALSE,",
        "    obs = stats::nobs(model),",
        "    check.names = FALSE",
        "  )",
        "}",
        "read_dataset <- function(path) {",
        "  if (data_format == 'feather' && has_arrow) {",
        "    return(as.data.frame(arrow::read_feather(path)))",
        "  }",
        "  if (has_dt) return(as.data.frame(data.table::fread(path, check.names = FALSE)))",
        "  read.csv(path, check.names = FALSE, stringsAsFactors = FALSE)",
        "}",
        "make_interaction <- function(parts, data) {",
        "  vals <- lapply(parts, function(nm) data[[nm]])",
        "  do.call(interaction, c(vals, list(drop = TRUE, sep = '^')))",
        "}",
        "make_fe_list <- function(fe_terms, data) {",
        "  lapply(fe_terms, function(term) {",
        "    parts <- strsplit(term, '^', fixed = TRUE)[[1]]",
        "    if (length(parts) == 1) data[[parts[[1]]]] else make_interaction(parts, data)",
        "  })",
        "}",
        "fe_base_vars <- function(fe_terms) unique(unlist(strsplit(fe_terms, '^', fixed = TRUE), use.names = FALSE))",
        "mask_key <- function(m) {",
        "  if (anyNA(m)) m[is.na(m)] <- FALSE",
        "  pad <- (-length(m)) %% 8L",
        "  bits <- if (pad > 0L) c(m, rep(FALSE, pad)) else m",
        "  vals <- as.integer(packBits(bits))",
        "  idx <- seq_along(vals)",
        "  h1 <- sum((vals + 1) * ((idx %% 1009) + 1)) %% 2147483647",
        "  h2 <- sum((vals + 3) * ((idx %% 9176) + 7)) %% 2147483629",
        "  h3 <- sum((vals + 5) * ((idx %% 65521) + 11)) %% 2147483587",
        "  paste0('k_', length(m), '_', sum(m), '_', as.integer(h1), '_', as.integer(h2), '_', as.integer(h3))",
        "}",
        "new_cache <- function() {",
        "  new.env(hash = TRUE, parent = emptyenv())",
        "}",
        "cache_get <- function(cache, key) {",
        "  cache[[key]]",
        "}",
        "cache_set <- function(cache, key, value) {",
        "  cache[[key]] <- value",
        "  invisible(value)",
        "}",
        *([
            "drop_singletons <- function(mask, fe_list) {",
            "  if (!length(fe_list)) return(mask)",
            "  work <- mask",
            "  repeat {",
            "    active <- which(work)",
            "    if (length(active) <= 1) return(work)",
            "    drop <- rep(FALSE, length(work))",
            "    for (fe in fe_list) {",
            "      codes <- as.integer(factor(fe[work]))",
            "      tab <- tabulate(codes)",
            "      single <- tab[codes] == 1L",
            "      if (any(single)) drop[active[single]] <- TRUE",
            "    }",
            "    if (!any(drop)) return(work)",
            "    work <- work & !drop",
            "  }",
            "}",
        ] if drop_singletons_option else []),
        "connected_components_from_codes <- function(fe_codes) {",
        "  if (!length(fe_codes)) return(0L)",
        "  if (length(fe_codes) == 1L) return(1L)",
        "  nlev <- vapply(fe_codes, function(c) c$nlev, integer(1))",
        "  offsets <- c(0L, cumsum(nlev)[-length(nlev)])",
        "  total <- sum(nlev)",
        "  parent <- seq_len(total)",
        "  find <- function(x) {",
        "    while (parent[[x]] != x) {",
        "      parent[[x]] <<- parent[[parent[[x]]]]",
        "      x <- parent[[x]]",
        "    }",
        "    x",
        "  }",
        "  union_op <- function(a, b) {",
        "    ra <- find(a); rb <- find(b)",
        "    if (ra != rb) parent[[rb]] <<- ra",
        "  }",
        "  n <- length(fe_codes[[1]]$codes)",
        "  for (i in seq_len(n)) {",
        "    root <- offsets[[1]] + fe_codes[[1]]$codes[[i]]",
        "    for (j in 2:length(fe_codes)) union_op(root, offsets[[j]] + fe_codes[[j]]$codes[[i]])",
        "  }",
        "  length(unique(vapply(seq_len(total), find, integer(1))))",
        "}",
        "k_fe_count_from_codes <- function(fe_codes) {",
        "  if (!length(fe_codes)) return(0L)",
        "  sum(vapply(fe_codes, function(c) c$nlev, integer(1))) - connected_components_from_codes(fe_codes)",
        "}",
        "is_nested_in_cluster_codes <- function(fe_one, cluster_factors) {",
        "  if (!length(cluster_factors)) return(FALSE)",
        "  fe_factor <- factor(fe_one$codes)",
        "  for (cl in cluster_factors) {",
        "    combo <- interaction(fe_factor, cl, drop = TRUE)",
        "    if (nlevels(combo) == fe_one$nlev) return(TRUE)",
        "  }",
        "  FALSE",
        "}",
        "k_fe_nonnested_from_codes <- function(fe_codes, cluster_factors, se_kind) {",
        "  if (se_kind == 'robust' || !length(cluster_factors)) return(k_fe_count_from_codes(fe_codes))",
        "  keep <- vapply(fe_codes, function(c) !is_nested_in_cluster_codes(c, cluster_factors), logical(1))",
        "  kept <- fe_codes[keep]",
        "  if (!length(kept)) return(0L)",
        "  sum(vapply(kept, function(c) c$nlev, integer(1)))",
        "}",
        "meat_cluster <- function(Xe, cl) {",
        "  sums <- rowsum(Xe, group = cl, reorder = FALSE)",
        "  crossprod(as.matrix(sums))",
        "}",
        "calc_vcov <- function(X, e, clusters, se_kind, k_total) {",
        "  N <- nrow(X)",
        "  XtX_inv <- tryCatch(solve(crossprod(X)), error = function(err) NULL)",
        "  if (is.null(XtX_inv) || N <= k_total) return(NULL)",
        "  Xe <- X * as.numeric(e)",
        "  if (se_kind == 'robust') {",
        "    V <- XtX_inv %*% ((N / (N - k_total)) * crossprod(Xe)) %*% XtX_inv",
        "  } else if (length(clusters) == 1L) {",
        "    cl1 <- if (is.factor(clusters[[1]])) clusters[[1]] else factor(clusters[[1]])",
        "    G <- nlevels(cl1)",
        "    if (G <= 1L) return(NULL)",
        "    ssc <- G / (G - 1) * (N - 1) / (N - k_total)",
        "    V <- XtX_inv %*% (ssc * meat_cluster(Xe, cl1)) %*% XtX_inv",
        "  } else if (length(clusters) == 2L) {",
        "    cl1 <- if (is.factor(clusters[[1]])) clusters[[1]] else factor(clusters[[1]])",
        "    cl2 <- if (is.factor(clusters[[2]])) clusters[[2]] else factor(clusters[[2]])",
        "    cl12 <- interaction(cl1, cl2, drop = TRUE)",
        "    G1 <- nlevels(cl1); G2 <- nlevels(cl2); G12 <- nlevels(cl12)",
        "    if (G1 <= 1L || G2 <= 1L || G12 <= 1L) return(NULL)",
        "    # fixest ssc() default cluster.df='conventional': per-cluster G/(G-1) factor",
        "    # combined with the (N-1)/(N-K) finite-sample correction (K.fixef='nonnested').",
        "    nk <- (N - 1) / (N - k_total)",
        "    meat <- (G1 / (G1 - 1)) * meat_cluster(Xe, cl1) +",
        "            (G2 / (G2 - 1)) * meat_cluster(Xe, cl2) -",
        "            (G12 / (G12 - 1)) * meat_cluster(Xe, cl12)",
        "    V <- XtX_inv %*% (nk * meat) %*% XtX_inv",
        "  } else {",
        "    stop('unsupported cluster count')",
        "  }",
        "  colnames(V) <- colnames(X)",
        "  rownames(V) <- colnames(X)",
        "  V",
        "}",
        "keep_independent <- function(X, tol = 1e-9) {",
        "  if (!ncol(X)) return(integer(0))",
        "  q <- qr(X, tol = tol)",
        "  sort(q$pivot[seq_len(q$rank)])",
        "}",
        f"df <- read_dataset({_r_string(str(data_path))})",
        f"specs <- read.csv({_r_string(str(specs_csv))}, check.names = FALSE, stringsAsFactors = FALSE)",
        f"y_var <- {_r_string(y)}",
        f"x_var <- {_r_string(x)}",
        f"fe_terms <- {_r_vector(fe_terms)}",
        f"cluster_vars <- {_r_vector(clust_cols)}",
        f"se_kind <- {_r_string(se_kind)}",
        f"nthreads <- {int(nthreads)}",
        "n_workers <- if (nthreads <= 0L) 8L else max(1L, as.integer(nthreads))",
        "demean_nthreads <- n_workers",
        f"results_path <- {_r_string(str(results_path))}",
        "spec_must <- lapply(specs$chosen_must_controls, read_json_vec)",
        "spec_test <- lapply(specs$chosen_test_controls, read_json_vec)",
        "spec_is_full <- as.logical(specs$is_full)",
        "all_must <- unique(unlist(spec_must, use.names = FALSE))",
        "all_test <- unique(unlist(spec_test, use.names = FALSE))",
        "all_controls <- unique(c(all_must, all_test))",
        "fe_base_vars_all <- fe_base_vars(fe_terms)",
        "base_vars <- unique(c(y_var, x_var, all_controls, cluster_vars, fe_base_vars_all))",
        "base_vars <- base_vars[nzchar(base_vars)]",
        "missing_vars <- setdiff(base_vars, names(df))",
        "if (length(missing_vars)) stop(paste('missing variables:', paste(missing_vars, collapse = ', ')))",
        *([
            "fe_list_full <- make_fe_list(fe_terms, df)",
        ] if drop_singletons_option else []),
        "col_notna <- lapply(df, function(v) !is.na(v))",
        "fast_complete <- function(spec_vars) {",
        "  mask <- rep(TRUE, nrow(df))",
        "  for (v in spec_vars) { if (v %in% names(col_notna)) mask <- mask & col_notna[[v]] }",
        "  mask",
        "}",
        "common_must <- if (length(spec_must)) Reduce(intersect, spec_must) else character(0)",
        "base_spec_vars <- unique(c(y_var, x_var, common_must, cluster_vars, fe_base_vars_all))",
        "base_mask <- fast_complete(base_spec_vars)",
        "complete_from_base <- function(chosen_must, chosen_test) {",
        "  mask <- base_mask",
        "  extra_vars <- unique(setdiff(c(chosen_must, chosen_test), common_must))",
        "  for (v in extra_vars) { if (v %in% names(col_notna)) mask <- mask & col_notna[[v]] }",
        "  mask",
        "}",
        "sample_cache <- new_cache()",
        *([
            "ds_cache <- new_cache()",
            "drop_singletons_c <- function(init_mask) {",
            "  key <- mask_key(init_mask)",
            "  cached <- cache_get(ds_cache, key)",
            "  if (!is.null(cached)) return(cached)",
            "  r <- drop_singletons(init_mask, fe_list_full)",
            "  cache_set(ds_cache, key, r)",
            "  r",
            "}",
        ] if drop_singletons_option else []),
        "get_sample <- function(mask, key = NULL) {",
        "  if (is.null(key)) key <- mask_key(mask)",
        "  cached <- cache_get(sample_cache, key)",
        "  if (!is.null(cached)) return(cached)",
        "  work <- df[mask, , drop = FALSE]",
        "  fe_list <- make_fe_list(fe_terms, work)",
        "  fe_codes <- lapply(fe_list, function(v) { f <- factor(v); list(codes = as.integer(f), nlev = nlevels(f)) })",
        "  cluster_factors <- lapply(cluster_vars, function(nm) factor(work[[nm]]))",
        "  available_controls <- all_controls",
        "  if (length(available_controls)) {",
        "    complete_control <- vapply(work[, available_controls, drop = FALSE], function(v) all(!is.na(v)), logical(1))",
        "    available_controls <- available_controls[complete_control]",
        "  }",
        "  demean_vars <- unique(c(y_var, x_var, available_controls))",
        "  dm <- fixest::demean(",
        "    as.matrix(work[, demean_vars, drop = FALSE]),",
        "    f = fe_list,",
        "    nthreads = demean_nthreads,",
        "    notes = FALSE",
        "  )",
        "  dm <- as.matrix(dm)",
        "  colnames(dm) <- demean_vars",
        "  value <- list(",
        "    dm = dm,",
        "    clusters = cluster_factors,",
        "    k_fe_full = k_fe_count_from_codes(fe_codes),",
        "    k_fe_se = k_fe_nonnested_from_codes(fe_codes, cluster_factors, se_kind),",
        "    N = nrow(work)",
        "  )",
        "  cache_set(sample_cache, key, value)",
        "  value",
        "}",
        "get_sample_by_key <- function(key) {",
        "  cached <- cache_get(sample_cache, key)",
        "  if (is.null(cached)) stop('internal error: sample cache miss')",
        "  cached",
        "}",
        "process_one_spec <- function(i) {",
        "  chosen_must <- spec_must[[i]]",
        "  chosen_test <- spec_test[[i]]",
        "  if (length(cluster_vars) > 1L) {",
        "    # Multi-way clusters: defer to feols to match its native SSC scheme",
        "    # exactly (the shared-demean path uses Gmin SSC which diverges).",
        "    exact_row <- run_feols_spec(chosen_must, chosen_test)",
        "    if (!is.null(exact_row)) exact_row$is_full <- spec_is_full[[i]]",
        "    return(exact_row)",
        "  }",
        "  rhs_vars <- unique(c(x_var, chosen_must, chosen_test))",
        "  if (!spec_valid[[i]]) return(NULL)",
        "  sample <- get_sample_by_key(spec_sample_keys[[i]])",
        "  dm <- sample$dm",
        "  if (!all(rhs_vars %in% colnames(dm))) return(NULL)",
        "  X_raw <- dm[, rhs_vars, drop = FALSE]",
        "  keep <- keep_independent(X_raw)",
        "  if (!length(keep) || !(1L %in% keep)) return(NULL)",
        "  X <- X_raw[, keep, drop = FALSE]",
        "  kept_vars <- colnames(X)",
        "  y_dm <- as.numeric(dm[, y_var])",
        "  fit <- tryCatch(.lm.fit(X, y_dm), error = function(e) NULL)",
        "  if (is.null(fit) || anyNA(fit$coefficients)) return(NULL)",
        "  x_idx <- match(x_var, kept_vars)",
        "  if (is.na(x_idx)) return(NULL)",
        "  e <- as.numeric(fit$residuals)",
        "  N <- nrow(X)",
        "  k_total_df <- ncol(X) + sample$k_fe_full",
        "  k_total_se <- ncol(X) + sample$k_fe_se",
        "  df_resid <- max(1L, as.integer(N - k_total_df))",
        "  vcov_mat <- calc_vcov(X, e, sample$clusters, se_kind, k_total_se)",
        "  if (is.null(vcov_mat)) return(NULL)",
        "  se_vec <- sqrt(pmax(diag(vcov_mat), 0))",
        "  names(se_vec) <- kept_vars",
        "  coef_full <- fit$coefficients",
        "  names(coef_full) <- kept_vars",
        "  coef <- unname(coef_full[[x_idx]])",
        "  se <- unname(se_vec[[x_idx]])",
        "  if (!is.finite(coef) || !is.finite(se) || se <= 0) return(NULL)",
        "  t_value <- coef / se",
        "  p_df <- df_resid",
        "  if (se_kind == 'cluster' && length(sample$clusters)) {",
        "    p_df <- max(1L, min(vapply(sample$clusters, function(cl) nlevels(cl), integer(1))) - 1L)",
        "  }",
        "  p_value <- 2 * stats::pt(abs(t_value), df = p_df, lower.tail = FALSE)",
        "  crit99 <- stats::qt(0.995, df = p_df)",
        "  crit95 <- stats::qt(0.975, df = p_df)",
        "  crit90 <- stats::qt(0.950, df = p_df)",
        "  sse <- sum(e^2)",
        "  tss <- sum(y_dm^2)",
        "  within_r2 <- if (tss > 0) 1 - sse / tss else NA_real_",
        "  adj_r2 <- if (is.finite(within_r2) && df_resid > 0) 1 - (1 - within_r2) * ((N - 1) / df_resid) else NA_real_",
        "  f_stat <- NA_real_",
        "  vcov_inv <- tryCatch(solve(vcov_mat), error = function(err) NULL)",
        "  if (!is.null(vcov_inv)) {",
        "    beta <- as.numeric(coef_full)",
        "    f_stat <- as.numeric(t(beta) %*% vcov_inv %*% beta / length(beta))",
        "  }",
        "  ctrl_stats <- list()",
        "  for (ctrl in c(chosen_must, chosen_test)) {",
        "    if (ctrl %in% kept_vars) {",
        "      idx <- match(ctrl, kept_vars)",
        "      ctrl_coef <- unname(coef_full[[idx]])",
        "      ctrl_se <- unname(se_vec[[idx]])",
        "      ctrl_t <- ctrl_coef / ctrl_se",
        "      ctrl_stats[[length(ctrl_stats) + 1L]] <- list(",
        "        name = ctrl,",
        "        coef = ctrl_coef,",
        "        se = ctrl_se,",
        "        t_value = ctrl_t,",
        "        p_value = 2 * stats::pt(abs(ctrl_t), df = p_df, lower.tail = FALSE)",
        "      )",
        "    }",
        "  }",
        "  controls_all <- kept_vars[kept_vars != x_var]",
        "  kept_test <- intersect(chosen_test, controls_all)",
        "  data.frame(",
        "    coef = coef, se = se, t_value = t_value, p_value = p_value,",
        "    adj_r2 = adj_r2, within_r2 = within_r2, f_stat = f_stat, df_resid = df_resid,",
        "    ci99_lo = coef - crit99 * se, ci99_hi = coef + crit99 * se,",
        "    ci95_lo = coef - crit95 * se, ci95_hi = coef + crit95 * se,",
        "    ci90_lo = coef - crit90 * se, ci90_hi = coef + crit90 * se,",
        "    controls_test = json_vec(kept_test),",
        "    controls_all = json_vec(controls_all),",
        "    control_stats = json_stats(ctrl_stats),",
        "    is_full = spec_is_full[[i]],",
        "    obs = N,",
        "    check.names = FALSE,",
        "    stringsAsFactors = FALSE",
        "  )",
        "}",
        "spec_ids <- seq_len(nrow(specs))",
        "seen_sample_keys <- new_cache()",
        "unique_masks <- list()",
        "unique_sample_keys <- character(0)",
        "spec_sample_keys <- rep(NA_character_, length(spec_ids))",
        "spec_valid <- rep(FALSE, length(spec_ids))",
        "if (length(cluster_vars) <= 1L) for (i in spec_ids) {",
        "  mask_i <- complete_from_base(spec_must[[i]], spec_test[[i]])",
        *([
            "  mask_i <- drop_singletons_c(mask_i)",
        ] if drop_singletons_option else []),
        "  if (sum(mask_i) <= 1L) next",
        "  k <- mask_key(mask_i)",
        "  spec_sample_keys[[i]] <- k",
        "  spec_valid[[i]] <- TRUE",
        "  if (is.null(cache_get(seen_sample_keys, k))) {",
        "    cache_set(seen_sample_keys, k, TRUE)",
        "    unique_masks[[length(unique_masks) + 1L]] <- mask_i",
        "    unique_sample_keys <- c(unique_sample_keys, k)",
        "  }",
        "}",
        "for (j in seq_along(unique_masks)) get_sample(unique_masks[[j]], unique_sample_keys[[j]])",
        "if (n_workers > 1L && .Platform$OS.type != 'windows') {",
        "  out <- parallel::mclapply(spec_ids, process_one_spec, mc.cores = n_workers, mc.preschedule = TRUE)",
        "} else {",
        "  out <- lapply(spec_ids, process_one_spec)",
        "}",
        "out <- Filter(Negate(is.null), out)",
        "if (length(out) == 0) {",
        "  write.csv(data.frame(), results_path, row.names = FALSE)",
        "} else {",
        "  if (has_dt) {",
        "    res <- as.data.frame(data.table::rbindlist(out, fill = TRUE))",
        "  } else {",
        "    res <- do.call(rbind, out)",
        "  }",
        "  res <- res[order(res$coef), , drop = FALSE]",
        "  write.csv(res, results_path, row.names = FALSE, fileEncoding = 'UTF-8')",
        "}",
    ]
    script_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _run_rscript(rscript_path: str, script_path: pathlib.Path, log_path: pathlib.Path, cwd: pathlib.Path) -> None:
    with log_path.open("w", encoding="utf-8") as log:
        try:
            proc = subprocess.run(
                [rscript_path, str(script_path.name)],
                cwd=cwd,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            )
        except FileNotFoundError as exc:
            raise RuntimeError(
                f"Rscript executable not found: {rscript_path}\n"
                "Use --rscript-path or set rscript_path in the TOML config."
            ) from exc
    if proc.returncode != 0:
        raise RuntimeError(
            f"Rscript exited with status {proc.returncode}.\n"
            f"R script: {script_path.resolve()}\n"
            f"Log file: {log_path.resolve()}\n"
            "R log tail:\n"
            f"{rm_common._tail_text(log_path)}"
        )



def _records_from_r_csv(results_path: pathlib.Path, log_path: pathlib.Path, script_path: pathlib.Path) -> list[rm.SpecRecord]:
    try:
        df_res = cast(pd.DataFrame, pd.read_csv(results_path))
    except FileNotFoundError as exc:
        raise RuntimeError(
            "R did not create the expected result file.\n"
            f"Expected result: {results_path.resolve()}\n"
            f"R script: {script_path.resolve()}\n"
            f"Log file: {log_path.resolve()}\n"
            "R log tail:\n"
            f"{rm_common._tail_text(log_path)}"
        ) from exc
    if df_res.empty:
        raise RuntimeError(
            "R/fixest returned no valid regression results.\n"
            f"Result file: {results_path.resolve()}\n"
            f"R script: {script_path.resolve()}\n"
            f"Log file: {log_path.resolve()}\n"
            "R log tail:\n"
            f"{rm_common._tail_text(log_path)}"
        )
    return rm.records_from_dataframe(df_res)


def run_r_engine(
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

    r_packages = _probe_r_packages(args.rscript_path)
    use_feather = r_packages["arrow"]

    required_cols, factor_cols = _required_cols_for_run(
        df=df,
        args=args,
        controls_must_flat=controls_must_flat,
        controls_test_flat=controls_test_flat,
        var_map=var_map,
        spec_flags=spec_flags,
    )
    df_subset = df[required_cols]

    if use_feather:
        input_path = run_output_dir / f"{data_path.stem}_r_input.feather"
        data_format = "feather"
    else:
        input_path = run_output_dir / f"{data_path.stem}_r_input.csv"
        data_format = "csv"
    _write_input_dataset(df_subset, input_path, factor_cols, use_feather=use_feather)

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

            script_path = run_output_dir / f"{y_var}_{x_var}_{spec_def['tag']}.R"
            log_path = run_output_dir / f"{y_var}_{x_var}_{spec_def['tag']}.R.log"
            specs_csv = run_output_dir / f"{y_var}_{x_var}_{spec_def['tag']}_r_specs.csv"
            results_path = run_output_dir / f"{y_var}_{x_var}_{spec_def['tag']}_results.csv"
            meta_path = run_output_dir / f"{y_var}_{x_var}_{spec_def['tag']}_plot_meta.json"
            output_path = rm_common._render_output_path(
                run_output_dir,
                str(spec_def["tag"]),
                f"{y_var}_{x_var}_{spec_def['tag']}.png",
                args.export_format,
            )
            title_suffix = spec_def["help"].format(**fmt)
            base_regression_count = rm._spec_count_from_slots(controls_must_slots, controls_test_slots)

            print(f"[R] 运行规格：{spec_display}")
            print(rm._format_plot_regression_count(base_regression_count))
            spec_t0 = perf_counter()
            _write_r_specs_csv(
                specs_csv,
                controls_must_slots=controls_must_slots,
                controls_test_slots=controls_test_slots,
            )
            _write_fixest_script(
                script_path=script_path,
                data_path=input_path.resolve(),
                data_format=data_format,
                specs_csv=specs_csv.resolve(),
                results_path=results_path.resolve(),
                y=y_var,
                x=x_var,
                fe_terms=_spec_r_fe_terms(spec_def, var_map),
                clust_cols=[var_map[k] for k in spec_def["cl_keys"]],
                robust=spec_def["vce"] == "robust",
                nthreads=max(0, int(getattr(args, "n_jobs", 0))),
                drop_singletons_option=bool(getattr(args, "drop_singletons", True)),
            )
            _run_rscript(args.rscript_path, script_path, log_path, run_output_dir)
            records = _records_from_r_csv(results_path, log_path, script_path)

            fe_cols = _spec_fe_labels(spec_def, var_map)
            clust_cols = [var_map[k] for k in spec_def["cl_keys"]]
            sig_rows = rm._build_sig_rows(
                records=records,
                y=y_var,
                x=x_var,
                controls_must=controls_must_flat,
                controls_test=controls_test_flat,
                fe_cols=fe_cols,
                clust_cols=clust_cols,
                vce_label="robust" if spec_def["vce"] == "robust" else None,
            )
            rm.write_analysis_artifacts(
                records=records,
                results_path=results_path,
                meta_path=meta_path,
                meta={
                    "engine": "r",
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
                "sig_rows": sig_rows,
                "summary_sig_rows": sig_rows,
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
                rm_common.safe_unlink(script_path)
                rm_common.safe_unlink(log_path)
                rm_common.safe_unlink(specs_csv)
        outputs[(y_var, x_var)] = pair_items

    if not args.keep_temp:
        rm_common.safe_unlink(input_path)
    return outputs


def main() -> None:
    cfg, cli_args = rm_common.load_toml_config(sys.argv[1:])
    parser = argparse.ArgumentParser(
        prog="regression_monkey_r",
        description="Run R/fixest analysis and write standard Regression Monkey result files.",
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
    parser.add_argument("--n-jobs", default=0, type=int, metavar="N")
    parser.add_argument("--order", choices=["coef", "p"], default="coef", help="绘图排序方式：coef 或 p")
    parser.add_argument("--p", action="store_true", help="兼容别名；等价于 --order p")
    parser.add_argument("--rscript-path", default="Rscript", metavar="EXE")
    parser.add_argument("--keep-temp", action="store_true", help="保留 .R / .log / 中间规格文件")
    parser.add_argument("--drop-singletons", dest="drop_singletons", action="store_true", default=True, help="估计前删除 singleton 观测（默认开启，与 reghdfe 行为一致）")
    parser.add_argument("--no-drop-singletons", dest="drop_singletons", action="store_false", help="保留 singleton 观测")
    for spec_name in rm._ALL_SPEC_NAMES:
        parser.add_argument(f"--{spec_name.replace('_', '-')}", dest=spec_name, action="store_true")

    if cfg:
        allowed = {
            "data", "y", "x", "controls", "controls_test", "controls_must",
            "output", "dpi", "fig_width", "order", "p", "firm_fe", "ind_fe", "time_fe",
            "region_fe", "rscript_path", "keep_temp", "n_jobs", "drop_singletons",
        } | set(rm._ALL_SPEC_NAMES)
        normalized = {k.lower(): v for k, v in cfg.items()}
        parser.set_defaults(**{k: v for k, v in normalized.items() if k in allowed})

    args = parser.parse_args(cli_args)
    try:
        args.order = rm._normalize_plot_order(args.order, p_alias=bool(args.p))
        args.y = rm._expand_space_separated_names(args.y)
        args.x = rm._expand_space_separated_names(args.x)
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
        parser.error("当前 R 脚本仅支持自动规格模式，请至少启用一个 absorb_* flag。")

    data_path = pathlib.Path(args.data).expanduser().resolve()
    print(f"读取数据：{data_path}")
    df = rm_common.load_dataframe(data_path)
    print(f"数据读取完成：{len(df):,} 行 × {len(df.columns)} 列")
    output_root = pathlib.Path(args.output).expanduser().resolve()
    if output_root.suffix:
        output_root = output_root.parent
    run_output_dir = output_root / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_output_dir.mkdir(parents=True, exist_ok=True)
    print(f"输出目录：{run_output_dir}")

    run_r_engine(
        df=df,
        data_path=data_path,
        args=args,
        controls_test=controls_test,
        controls_must=controls_must,
        controls_test_flat=controls_test_flat,
        controls_test_slots=controls_test_slots,
        controls_must_flat=controls_must_flat,
        controls_must_slots=controls_must_slots,
        matrix_controls=matrix_controls,
        spec_flags=spec_flags,
        run_output_dir=run_output_dir,
    )


if __name__ == "__main__":
    main()
