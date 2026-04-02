import os

import numpy as np

from experiments.common.math1d import COMBOS, apply_combo
from experiments.common.plots import save_table_png


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
):
    combo_input = matrix
    if feature_cols is not None and feature_space == "raw":
        combo_input = select_feature_columns(combo_input, feature_cols, "raw")
    transformed = apply_combo(combo_input, labels, combo, target_scores=target_scores)
    if feature_cols is not None and feature_space == "transformed":
        transformed = select_feature_columns(transformed, feature_cols, "transformed")
    return transformed


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
    value_map = {
        tuple(row[field] for field in key_fields): row[metric_name] for row in rows
    }
    body = []
    for combo in combos:
        line = [combo["name"]]
        for _, key in column_specs:
            value = value_map.get((combo["name"],) + key, float("nan"))
            line.append("nan" if np.isnan(value) else f"{value:.3f}")
        body.append(line)
    headers = ["combo"] + [label for label, _ in column_specs]
    save_table_png(output_path, headers, body, title)
