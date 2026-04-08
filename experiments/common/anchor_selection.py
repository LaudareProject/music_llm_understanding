import random

import numpy as np


def equally_spaced(relevant_rows, target_key, n=10):
    n_anchors = min(int(n), len(relevant_rows) // 2)
    targets = _target_values(relevant_rows, target_key)
    min_target = np.min(targets)
    max_target = np.max(targets)
    step = (max_target - min_target) / (2*n_anchors)
    anchor_a_ids = []
    for idx in range(2*n_anchors):
        value = min_target + step * idx
        targets_diff = np.abs(targets - value)
        new_anchor_pos = np.argmin(targets_diff)
        anchor_a_ids.append(relevant_rows[new_anchor_pos]["query_id"])
    anchor_b_ids = []
    sampled_ids = set(anchor_a_ids + anchor_b_ids)
    eval_rows = [row for row in relevant_rows if row["query_id"] not in sampled_ids]
    return anchor_a_ids, anchor_b_ids, eval_rows


def first_vs_last(relevant_rows, target_key, n=10):
    n_anchors = min(int(n), len(relevant_rows) // 2)
    ranked_rows = sorted(relevant_rows, key=lambda row: float(row[target_key]))
    anchor_a_rows = ranked_rows[:n_anchors]
    anchor_b_rows = ranked_rows[-n_anchors:]
    anchor_a_ids = [row["query_id"] for row in anchor_a_rows]
    anchor_b_ids = [row["query_id"] for row in anchor_b_rows]
    eval_rows = ranked_rows[n_anchors:-n_anchors]
    return anchor_a_ids, anchor_b_ids, eval_rows


def random_sample_first_vs_last(relevant_rows, target_key, n=10, seed=42):
    n_anchors = min(int(n), len(relevant_rows) // 2)
    sample_size = 2 * n_anchors
    rng = random.Random(seed)
    sampled_rows = rng.sample(relevant_rows, sample_size)
    ranked_sample = sorted(sampled_rows, key=lambda row: float(row[target_key]))
    anchor_a_ids = [row["query_id"] for row in ranked_sample[:n_anchors]]
    anchor_b_ids = [row["query_id"] for row in ranked_sample[-n_anchors:]]
    sampled_ids = set(anchor_a_ids + anchor_b_ids)
    eval_rows = [row for row in relevant_rows if row["query_id"] not in sampled_ids]
    return anchor_a_ids, anchor_b_ids, eval_rows


def _target_values(relevant_rows, target_key):
    return np.asarray(
        [float(row[target_key]) for row in relevant_rows], dtype=np.float64
    )


def random_after_and_before_median(relevant_rows, target_key, n=10, seed=42):
    n_anchors = min(int(n), len(relevant_rows) // 2)
    midpoint_median = np.median(_target_values(relevant_rows, target_key))
    anchor_a_ids = [
        row["query_id"]
        for row in relevant_rows
        if float(row[target_key]) < midpoint_median
    ]
    anchor_b_ids = [
        row["query_id"]
        for row in relevant_rows
        if float(row[target_key]) >= midpoint_median
    ]
    rng = random.Random(seed)
    rng.shuffle(anchor_a_ids)
    rng.shuffle(anchor_b_ids)
    anchor_a_ids = anchor_a_ids[:n_anchors]
    anchor_b_ids = anchor_b_ids[:n_anchors]
    eval_rows = [
        row
        for row in relevant_rows
        if row["query_id"] not in anchor_a_ids + anchor_b_ids
    ]
    return anchor_a_ids, anchor_b_ids, eval_rows


def select_anchors(
    relevant_rows,
    method,
    target_key,
    n=10,
    seed=42,
):
    if method in {"first_vs_last", "anchor_column"}:
        return first_vs_last(relevant_rows, target_key, n=n)
    if method == "random_sample_first_last":
        return random_sample_first_vs_last(relevant_rows, target_key, n=n, seed=seed)
    if method == "equally_spaced_regression":
        return equally_spaced(relevant_rows, target_key, n=n)
    if method == "random_sample_regression":
        return random_sample_first_vs_last(relevant_rows, target_key, n=n)
    if method == "random_median_regression":
        return random_after_and_before_median(relevant_rows, target_key, n=n, seed=seed)
    if method == "random_median":
        return random_after_and_before_median(relevant_rows, target_key, n=n, seed=seed)
    raise ValueError(f"Unknown anchor selection method: {method}")
