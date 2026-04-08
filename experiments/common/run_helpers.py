import os

import numpy as np

from experiments.common.anchor_selection import select_anchors
from experiments.common.math1d import COMBOS, apply_combo_with_supervised_train_idx
from experiments.common.math1d import (
    fit_mapping_model,
    l1,
    mean_ci95,
    pearson,
    predict_mapping_model,
    project_1d,
    project_1d_regression_direction,
    r2,
    random_l1_baseline,
    spearman,
)
from experiments.common.plots import save_table_png


STOCHASTIC_ANCHOR_METHODS = {
    "random_sample_first_last",
    "random_median",
    "random_median_regression",
}


def parse_feature_cols_env():
    raw = os.environ.get("FEATURE_SELECTION_COLS", "").strip()
    if not raw:
        return None
    cols = [int(token.strip()) for token in raw.split(",") if token.strip()]
    return cols if cols else None


def feature_selection_space_env():
    value = os.environ.get("FEATURE_SELECTION_SPACE", "transformed").strip().lower()
    if value not in {"transformed", "raw"}:
        raise ValueError(f"Invalid FEATURE_SELECTION_SPACE: {value}")
    return value


def selected_combos(forced_combo_name):
    if not forced_combo_name:
        return COMBOS
    combos = [combo for combo in COMBOS if combo["name"] == forced_combo_name]
    if not combos:
        raise ValueError(
            f"Unknown combo in FEATURE_SELECTION_COMBO: {forced_combo_name}"
        )
    return combos


def transform_with_feature_selection(
    matrix,
    labels,
    combo,
    target_scores,
    feature_cols,
    feature_space,
    supervised_train_idx=None,
):
    combo_input = matrix
    if feature_cols is not None and feature_space == "raw":
        combo_input = select_feature_columns(combo_input, feature_cols, "raw")
    transformed = apply_combo_with_supervised_train_idx(
        combo_input,
        labels,
        combo,
        target_scores=target_scores,
        supervised_train_idx=supervised_train_idx,
    )
    if feature_cols is not None and feature_space == "transformed":
        transformed = select_feature_columns(transformed, feature_cols, "transformed")
    return transformed


def retry_count(anchor_selection_method):
    if anchor_selection_method in STOCHASTIC_ANCHOR_METHODS:
        return int(os.environ.get("ANCHOR_RETRIES", "5").strip() or "5")
    return 1


def evaluate_mapping_payload_with_anchor_retries(
    matrix,
    relevant_rows,
    labels,
    target_scores,
    query_to_index,
    anchor_selection_method,
    anchor_n,
    combo,
    feature_cols,
    feature_space,
    target_key,
    group_key,
    best_pairs_separation,
    mappings,
):
    query_to_row = {row["query_id"]: row for row in relevant_rows}
    retries = retry_count(anchor_selection_method)
    metric_lists = {
        mapping_name: {"pearson": [], "spearman": [], "r2": [], "l1": []}
        for mapping_name in mappings
    }
    random_values = []
    first_retry_payload = {}
    successful_retries = 0
    for retry_idx in range(retries):
        anchor_a_ids, anchor_b_ids, eval_rows = select_anchors(
            relevant_rows,
            anchor_selection_method,
            target_key=target_key,
            n=anchor_n,
            best_pairs_separation=best_pairs_separation,
            seed=42 + retry_idx,
        )
        if len(eval_rows) < 2:
            continue
        anchor_a_idx = [query_to_index[query_id] for query_id in anchor_a_ids]
        anchor_b_idx = [query_to_index[query_id] for query_id in anchor_b_ids]
        anchor_train_idx = anchor_a_idx + anchor_b_idx
        if len(anchor_train_idx) < 2:
            continue
        transformed = transform_with_feature_selection(
            matrix,
            labels,
            combo,
            target_scores,
            feature_cols,
            feature_space,
            supervised_train_idx=anchor_train_idx,
        )
        eval_idx = [query_to_index[row["query_id"]] for row in eval_rows]
        real = np.asarray(
            [float(row[target_key]) for row in eval_rows], dtype=np.float64
        )
        if anchor_selection_method in {
            "random_sample_regression",
            "random_median_regression",
        }:
            sample_ids = anchor_a_ids + anchor_b_ids
            sample_idx = [query_to_index[query_id] for query_id in sample_ids]
            sample_targets = [
                float(query_to_row[query_id][target_key]) for query_id in sample_ids
            ]
            raw_projection = project_1d_regression_direction(
                transformed,
                sample_idx,
                sample_targets,
                eval_idx,
            )
            anchor_projection = project_1d_regression_direction(
                transformed,
                sample_idx,
                sample_targets,
                sample_idx,
            )
            anchor_targets = np.asarray(sample_targets, dtype=np.float64)
        else:
            raw_projection = project_1d(
                transformed, anchor_a_idx, anchor_b_idx, eval_idx
            )
            anchor_projection = project_1d(
                transformed,
                anchor_a_idx,
                anchor_b_idx,
                anchor_train_idx,
            )
            anchor_targets = np.asarray(
                [
                    float(query_to_row[query_id][target_key])
                    for query_id in (anchor_a_ids + anchor_b_ids)
                ],
                dtype=np.float64,
            )
        random_values.append(random_l1_baseline(real))
        anchors_a = np.asarray(
            [
                float(row[target_key])
                for row in relevant_rows
                if row["query_id"] in anchor_a_ids
            ],
            dtype=np.float64,
        )
        anchors_b = np.asarray(
            [
                float(row[target_key])
                for row in relevant_rows
                if row["query_id"] in anchor_b_ids
            ],
            dtype=np.float64,
        )
        groups = [row[group_key] for row in eval_rows]
        for mapping_name in mappings:
            if mapping_name == "raw":
                predicted = np.asarray(raw_projection, dtype=np.float64)
            else:
                model = fit_mapping_model(
                    mapping_name, anchor_projection, anchor_targets
                )
                predicted = np.asarray(
                    predict_mapping_model(model, raw_projection), dtype=np.float64
                )
            metric_lists[mapping_name]["pearson"].append(pearson(real, predicted))
            metric_lists[mapping_name]["spearman"].append(spearman(real, predicted))
            metric_lists[mapping_name]["r2"].append(r2(real, predicted))
            metric_lists[mapping_name]["l1"].append(
                float("nan") if mapping_name == "raw" else l1(real, predicted)
            )
            if mapping_name not in first_retry_payload:
                first_retry_payload[mapping_name] = {
                    "real": np.asarray(real, dtype=np.float64),
                    "predicted": np.asarray(predicted, dtype=np.float64),
                    "groups": groups,
                    "query_ids": [row["query_id"] for row in eval_rows],
                    "anchors_a": anchors_a,
                    "anchors_b": anchors_b,
                }
        successful_retries += 1
    if successful_retries == 0:
        return None
    random_mean, _ = mean_ci95(random_values)
    payload = {}
    for mapping_name in mappings:
        pearson_mean, pearson_ci = mean_ci95(metric_lists[mapping_name]["pearson"])
        spearman_mean, spearman_ci = mean_ci95(metric_lists[mapping_name]["spearman"])
        r2_mean, r2_ci = mean_ci95(metric_lists[mapping_name]["r2"])
        l1_mean, l1_ci = mean_ci95(metric_lists[mapping_name]["l1"])
        first = first_retry_payload[mapping_name]
        payload[mapping_name] = {
            "real": first["real"],
            "predicted": first["predicted"],
            "averaged_model_predicted": first["predicted"],
            "groups": first["groups"],
            "pearson": pearson_mean,
            "pearson_ci95": pearson_ci,
            "spearman": spearman_mean,
            "spearman_ci95": spearman_ci,
            "r2": r2_mean,
            "r2_ci95": r2_ci,
            "l1": l1_mean,
            "l1_ci95": l1_ci,
            "random": random_mean,
            "sample_count": len(first["real"]),
            "fold_count": successful_retries,
            "query_ids": first["query_ids"],
            "anchors_a": first["anchors_a"],
            "anchors_b": first["anchors_b"],
        }
    return payload


def select_feature_columns(matrix, feature_cols, label):
    if matrix.shape[1] == 0:
        raise ValueError(f"{label.capitalize()} matrix has zero columns")
    max_col = max(feature_cols)
    if max_col >= matrix.shape[1]:
        raise ValueError(
            f"FEATURE_SELECTION_COLS contains out-of-range index {max_col} "
            f"for {label} dim {matrix.shape[1]}"
        )
    return matrix[:, feature_cols]


def save_metric_table(
    output_path, combos, rows, key_fields, column_specs, metric_name, title
):
    value_map = {tuple(row[field] for field in key_fields): row for row in rows}
    body = []
    for combo in combos:
        line = [combo["name"]]
        for _, key in column_specs:
            row = value_map.get((combo["name"],) + key)
            if row is None:
                line.append("nan")
                continue
            value = row[metric_name]
            ci_value = row.get(f"{metric_name}_ci95", float("nan"))
            if np.isnan(value):
                line.append("nan")
            elif np.isnan(ci_value):
                line.append(f"{value:.3f}")
            else:
                line.append(f"{value:.3f} +/- {ci_value:.3f}")
        body.append(line)
    headers = ["combo"] + [label for label, _ in column_specs]
    save_table_png(output_path, headers, body, title)
