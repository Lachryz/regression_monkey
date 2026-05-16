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

import pandas as pd

from . import common as rm_common
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


def _safe_path_part(value: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in value)
    cleaned = "_".join(part for part in cleaned.split("_") if part)
    return cleaned or "spec"


def _render_output_path(run_output_dir: pathlib.Path, group_name: str, filename: str, export_format: str) -> pathlib.Path:
    if export_format == "html":
        return run_output_dir / filename
    output_dir = run_output_dir / _safe_path_part(group_name)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / filename


def _control_spec_count(
    controls_must_slots: list[rm.ControlSlot],
    controls_test_slots: list[rm.ControlSlot],
) -> int:
    return rm._spec_count_from_slots(controls_must_slots, controls_test_slots)


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


def _write_fixest_script(
    *,
    script_path: pathlib.Path,
    data_csv: pathlib.Path,
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
    lines = [
        "options(warn = 1)",
        "if (!requireNamespace('fixest', quietly = TRUE)) {",
        "  stop('R package fixest is not installed. Install it with install.packages(\"fixest\").')",
        "}",
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
        "connected_components_n <- function(fe_list) {",
        "  if (!length(fe_list)) return(0L)",
        "  codes <- lapply(fe_list, function(v) as.integer(factor(v)))",
        "  nlev <- vapply(codes, max, integer(1))",
        "  if (length(codes) == 1L) return(1L)",
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
        "  union <- function(a, b) {",
        "    ra <- find(a); rb <- find(b)",
        "    if (ra != rb) parent[[rb]] <<- ra",
        "  }",
        "  n <- length(codes[[1]])",
        "  for (i in seq_len(n)) {",
        "    root <- offsets[[1]] + codes[[1]][[i]]",
        "    for (j in 2:length(codes)) union(root, offsets[[j]] + codes[[j]][[i]])",
        "  }",
        "  length(unique(vapply(seq_len(total), find, integer(1))))",
        "}",
        "k_fe_count <- function(fe_list) {",
        "  if (!length(fe_list)) return(0L)",
        "  groups <- lapply(fe_list, function(v) as.integer(factor(v)))",
        "  n_levels <- vapply(groups, function(g) max(g), integer(1))",
        "  sum(n_levels) - connected_components_n(fe_list)",
        "}",
        "is_nested_in_cluster <- function(fe, clusters) {",
        "  if (!length(clusters)) return(FALSE)",
        "  fe_codes <- factor(fe)",
        "  for (cl in clusters) {",
        "    combo <- interaction(fe_codes, factor(cl), drop = TRUE)",
        "    if (nlevels(combo) == nlevels(fe_codes)) return(TRUE)",
        "  }",
        "  FALSE",
        "}",
        "k_fe_nonnested <- function(fe_list, clusters, se_kind) {",
        "  if (se_kind == 'robust' || !length(clusters)) return(k_fe_count(fe_list))",
        "  keep <- vapply(fe_list, function(fe) !is_nested_in_cluster(fe, clusters), logical(1))",
        "  kept <- fe_list[keep]",
        "  if (!length(kept)) return(0L)",
        "  sum(vapply(kept, function(fe) nlevels(factor(fe)), integer(1)))",
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
        "    cl1 <- factor(clusters[[1]])",
        "    G <- nlevels(cl1)",
        "    if (G <= 1L) return(NULL)",
        "    ssc <- G / (G - 1) * (N - 1) / (N - k_total)",
        "    V <- XtX_inv %*% (ssc * meat_cluster(Xe, cl1)) %*% XtX_inv",
        "  } else if (length(clusters) == 2L) {",
        "    cl1 <- factor(clusters[[1]])",
        "    cl2 <- factor(clusters[[2]])",
        "    cl12 <- interaction(cl1, cl2, drop = TRUE)",
        "    Gmin <- min(nlevels(cl1), nlevels(cl2), nlevels(cl12))",
        "    if (Gmin <= 1L) return(NULL)",
        "    ssc <- Gmin / (Gmin - 1) * (N - 1) / (N - k_total)",
        "    meat <- meat_cluster(Xe, cl1) + meat_cluster(Xe, cl2) - meat_cluster(Xe, cl12)",
        "    V <- XtX_inv %*% (ssc * meat) %*% XtX_inv",
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
        f"df <- read.csv({_r_string(str(data_csv))}, check.names = FALSE, stringsAsFactors = FALSE)",
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
        "all_must <- unique(unlist(lapply(specs$chosen_must_controls, read_json_vec), use.names = FALSE))",
        "all_test <- unique(unlist(lapply(specs$chosen_test_controls, read_json_vec), use.names = FALSE))",
        "all_controls <- unique(c(all_must, all_test))",
        "base_vars <- unique(c(y_var, x_var, all_controls, cluster_vars, fe_base_vars(fe_terms)))",
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
        "sample_masks <- list()",
        "sample_values <- list()",
        *([
            "ds_cache_keys <- list()",
            "ds_cache_vals <- list()",
            "drop_singletons_c <- function(init_mask) {",
            "  for (j in seq_along(ds_cache_keys)) {",
            "    if (identical(init_mask, ds_cache_keys[[j]])) return(ds_cache_vals[[j]])",
            "  }",
            "  r <- drop_singletons(init_mask, fe_list_full)",
            "  ds_cache_keys[[length(ds_cache_keys) + 1L]] <<- init_mask",
            "  ds_cache_vals[[length(ds_cache_vals) + 1L]] <<- r",
            "  r",
            "}",
        ] if drop_singletons_option else []),
        "get_sample <- function(mask) {",
        "  if (length(sample_masks)) {",
        "    for (j in seq_along(sample_masks)) {",
        "      if (identical(mask, sample_masks[[j]])) return(sample_values[[j]])",
        "    }",
        "  }",
        "  work <- df[mask, , drop = FALSE]",
        "  fe_list <- make_fe_list(fe_terms, work)",
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
        "  clusters <- lapply(cluster_vars, function(nm) work[[nm]])",
        "  value <- list(",
        "    dm = dm,",
        "    clusters = clusters,",
        "    k_fe_full = k_fe_count(fe_list),",
        "    k_fe_se = k_fe_nonnested(fe_list, clusters, se_kind),",
        "    N = nrow(work)",
        "  )",
        "  sample_masks[[length(sample_masks) + 1L]] <<- mask",
        "  sample_values[[length(sample_values) + 1L]] <<- value",
        "  value",
        "}",
        "process_one_spec <- function(i) {",
        "  chosen_must <- read_json_vec(specs$chosen_must_controls[[i]])",
        "  chosen_test <- read_json_vec(specs$chosen_test_controls[[i]])",
        "  if (length(cluster_vars) > 1L) {",
        "    exact_row <- run_feols_spec(chosen_must, chosen_test)",
        "    if (!is.null(exact_row)) {",
        "      exact_row$is_full <- as.logical(specs$is_full[[i]])",
        "    }",
        "    return(exact_row)",
        "  }",
        "  rhs_vars <- unique(c(x_var, chosen_must, chosen_test))",
        "  spec_vars <- unique(c(y_var, rhs_vars, cluster_vars, fe_base_vars(fe_terms)))",
        "  mask <- fast_complete(spec_vars)",
        *([
            "  mask <- drop_singletons_c(mask)",
        ] if drop_singletons_option else []),
        "  if (sum(mask) <= 1L) return(NULL)",
        "  sample <- get_sample(mask)",
        "  dm <- sample$dm",
        "  if (!all(rhs_vars %in% colnames(dm))) return(NULL)",
        "  X_raw <- dm[, rhs_vars, drop = FALSE]",
        "  keep <- keep_independent(X_raw)",
        "  if (!length(keep) || !(1L %in% keep)) return(NULL)",
        "  X <- X_raw[, keep, drop = FALSE]",
        "  kept_vars <- colnames(X)",
        "  y_dm <- as.numeric(dm[, y_var])",
        "  fit <- tryCatch(lm.fit(x = X, y = y_dm), error = function(e) NULL)",
        "  if (is.null(fit) || anyNA(fit$coefficients) || !(x_var %in% names(fit$coefficients))) return(NULL)",
        "  e <- as.numeric(fit$residuals)",
        "  N <- nrow(X)",
        "  k_total_df <- ncol(X) + sample$k_fe_full",
        "  k_total_se <- ncol(X) + sample$k_fe_se",
        "  df_resid <- max(1L, as.integer(N - k_total_df))",
        "  vcov_mat <- calc_vcov(X, e, sample$clusters, se_kind, k_total_se)",
        "  if (is.null(vcov_mat)) return(NULL)",
        "  se_vec <- sqrt(pmax(diag(vcov_mat), 0))",
        "  names(se_vec) <- kept_vars",
        "  coef <- unname(fit$coefficients[[x_var]])",
        "  se <- unname(se_vec[[x_var]])",
        "  if (!is.finite(coef) || !is.finite(se) || se <= 0) return(NULL)",
        "  t_value <- coef / se",
        "  p_df <- df_resid",
        "  if (se_kind == 'cluster' && length(sample$clusters)) {",
        "    p_df <- max(1L, min(vapply(sample$clusters, function(cl) nlevels(factor(cl)), integer(1))) - 1L)",
        "  }",
        "  p_value <- 2 * stats::pt(abs(t_value), df = p_df, lower.tail = FALSE)",
        "  crit99 <- stats::qt(0.995, df = p_df)",
        "  crit95 <- stats::qt(0.975, df = p_df)",
        "  crit90 <- stats::qt(0.950, df = p_df)",
        "  sse <- sum(e^2)",
        "  tss <- sum((y_dm)^2)",
        "  within_r2 <- if (tss > 0) 1 - sse / tss else NA_real_",
        "  adj_r2 <- if (is.finite(within_r2) && df_resid > 0) 1 - (1 - within_r2) * ((N - 1) / df_resid) else NA_real_",
        "  f_stat <- NA_real_",
        "  vcov_inv <- tryCatch(solve(vcov_mat), error = function(err) NULL)",
        "  if (!is.null(vcov_inv)) {",
        "    beta <- as.numeric(fit$coefficients)",
        "    f_stat <- as.numeric(t(beta) %*% vcov_inv %*% beta / length(beta))",
        "  }",
        "  ctrl_stats <- list()",
        "  for (ctrl in c(chosen_must, chosen_test)) {",
        "    if (ctrl %in% kept_vars) {",
        "      ctrl_coef <- unname(fit$coefficients[[ctrl]])",
        "      ctrl_se <- unname(se_vec[[ctrl]])",
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
        "    is_full = as.logical(specs$is_full[[i]]),",
        "    obs = N,",
        "    check.names = FALSE",
        "  )",
        "}",
        "if (length(cluster_vars) <= 1L) {",
        "  union_vars <- unique(c(y_var, x_var, all_controls, cluster_vars, fe_base_vars(fe_terms)))",
        "  union_vars <- union_vars[nzchar(union_vars) & union_vars %in% names(df)]",
        "  union_mask <- fast_complete(union_vars)",
        *([
            "  union_mask <- drop_singletons_c(union_mask)",
        ] if drop_singletons_option else []),
        "  if (sum(union_mask) > 1L) get_sample(union_mask)",
        "}",
        "spec_ids <- seq_len(nrow(specs))",
        "if (n_workers > 1L && .Platform$OS.type != 'windows') {",
        "  out <- parallel::mclapply(spec_ids, process_one_spec, mc.cores = n_workers, mc.preschedule = TRUE)",
        "} else {",
        "  out <- lapply(spec_ids, process_one_spec)",
        "}",
        "out <- Filter(Negate(is.null), out)",
        "if (length(out) == 0) {",
        "  write.csv(data.frame(), results_path, row.names = FALSE)",
        "} else {",
        "  res <- do.call(rbind, out)",
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
            f"{_tail_text(log_path)}"
        )


def _tail_text(path: pathlib.Path, max_lines: int = 80) -> str:
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except FileNotFoundError:
        return "(log file not found)"
    return "\n".join(lines[-max_lines:])


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
            f"{_tail_text(log_path)}"
        ) from exc
    if df_res.empty:
        raise RuntimeError(
            "R/fixest returned no valid regression results.\n"
            f"Result file: {results_path.resolve()}\n"
            f"R script: {script_path.resolve()}\n"
            f"Log file: {log_path.resolve()}\n"
            "R log tail:\n"
            f"{_tail_text(log_path)}"
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
    input_csv = run_output_dir / f"{data_path.stem}_r_input.csv"
    df.to_csv(input_csv, index=False, encoding="utf-8")
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
            output_path = _render_output_path(
                run_output_dir,
                str(spec_def["tag"]),
                f"{y_var}_{x_var}_{spec_def['tag']}.png",
                args.export_format,
            )
            title_suffix = spec_def["help"].format(**fmt)
            base_regression_count = _control_spec_count(controls_must_slots, controls_test_slots)

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
                data_csv=input_csv.resolve(),
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
        rm_common.safe_unlink(input_csv)
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
