from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr


def spearman_metric(y_true, y_pred):
    y_true_array = np.asarray(y_true, dtype=np.float64)
    y_pred_array = np.asarray(y_pred, dtype=np.float64)
    if y_pred_array.ndim == 1:
        y_pred_array = y_pred_array.reshape(1, -1)
    if y_pred_array.ndim != 2:
        return -1.0
    scores = []
    for row in y_pred_array:
        corr, _ = spearmanr(y_true_array, row)
        if not np.isnan(corr):
            scores.append(float(corr))
    if not scores:
        return -1.0
    return float(np.mean(np.asarray(scores, dtype=np.float64)))


def mean_ci95(values):
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return float("nan"), float("nan")
    mean = float(np.mean(array))
    if array.size < 2:
        return mean, 0.0
    std = float(np.std(array, ddof=1))
    ci = 1.96 * std / float(np.sqrt(array.size))
    return mean, float(ci)


def remove_target_outliers(rows, y, group_labels=None, iqr_factor=1.5):
    if iqr_factor is None:
        return rows, y, group_labels, 0, float("nan"), float("nan")
    iqr_factor = float(iqr_factor)
    if iqr_factor <= 0.0:
        return rows, y, group_labels, 0, float("nan"), float("nan")
    if int(y.size) < 4:
        return rows, y, group_labels, 0, float("nan"), float("nan")

    q1, q3 = np.percentile(y, [25.0, 75.0])
    iqr = float(q3 - q1)
    if not np.isfinite(iqr) or iqr == 0.0:
        return rows, y, group_labels, 0, float("nan"), float("nan")

    low = float(q1 - iqr_factor * iqr)
    high = float(q3 + iqr_factor * iqr)
    keep_mask = (y >= low) & (y <= high)
    removed = int(np.sum(~keep_mask))
    if removed <= 0:
        return rows, y, group_labels, 0, low, high

    filtered_rows = [row for row, keep in zip(rows, keep_mask) if keep]
    filtered_y = np.asarray(y, dtype=np.float64)[keep_mask]
    filtered_groups = None
    if group_labels is not None:
        filtered_groups = np.asarray(group_labels, dtype=object)[keep_mask]
    return filtered_rows, filtered_y, filtered_groups, removed, low, high
