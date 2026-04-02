import random

import numpy as np


def use_anchor_column(relevant_rows, anchor_column):
    anchor_a_ids = [
        row["query_id"] for row in relevant_rows if row[anchor_column] == "a"
    ]
    anchor_b_ids = [
        row["query_id"] for row in relevant_rows if row[anchor_column] == "b"
    ]
    eval_rows = [row for row in relevant_rows if row[anchor_column] not in {"a", "b"}]
    return anchor_a_ids, anchor_b_ids, eval_rows


def _target_values(relevant_rows, target_key):
    return np.asarray(
        [float(row[target_key]) for row in relevant_rows], dtype=np.float64
    )


def random_after_and_before_median(relevant_rows, target_key, n=5, seed=42):
    n_anchors = n if isinstance(n, int) else int(len(relevant_rows) * n)
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


def n_best_pairs_target_separation(
    relevant_rows, target_key, separation=100, n=5, include_anchors_in_eval=False
):
    pairs = []
    for i in range(len(relevant_rows)):
        for j in range(i + 1, len(relevant_rows)):
            first_item = (
                relevant_rows[i]
                if float(relevant_rows[i][target_key])
                < float(relevant_rows[j][target_key])
                else relevant_rows[j]
            )
            last_item = (
                relevant_rows[j] if first_item == relevant_rows[i] else relevant_rows[i]
            )
            pairs.append(
                (
                    first_item,
                    last_item,
                    abs(float(last_item[target_key]) - float(first_item[target_key])),
                )
            )
    pairs.sort(key=lambda pair: abs(pair[2] - separation))
    anchor_a_ids = [pair[0]["query_id"] for pair in pairs[:n]]
    anchor_b_ids = [pair[1]["query_id"] for pair in pairs[:n]]
    if include_anchors_in_eval:
        eval_rows = list(relevant_rows)
    else:
        eval_rows = [
            row
            for row in relevant_rows
            if row["query_id"] not in anchor_a_ids + anchor_b_ids
        ]
    return anchor_a_ids, anchor_b_ids, eval_rows


def select_anchors(
    relevant_rows,
    anchor_column,
    method,
    target_key,
    random_n=10,
    best_pairs_n=10,
    best_pairs_separation=100,
):
    if method == "anchor_column":
        return use_anchor_column(relevant_rows, anchor_column)
    if method == "random_median":
        return random_after_and_before_median(relevant_rows, target_key, n=random_n)
    if method == "best_pairs":
        return n_best_pairs_target_separation(
            relevant_rows,
            target_key,
            separation=best_pairs_separation,
            n=best_pairs_n,
        )
    raise ValueError(f"Unknown anchor selection method: {method}")
