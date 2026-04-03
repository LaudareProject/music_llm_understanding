import random

import numpy as np


def first_vs_last(relevant_rows, target_key, n=10):
    n_anchors = min(int(n), len(relevant_rows) // 2)
    ranked_rows = sorted(relevant_rows, key=lambda row: float(row[target_key]))
    anchor_a_rows = ranked_rows[:n_anchors]
    anchor_b_rows = ranked_rows[-n_anchors:]
    anchor_a_ids = [row["query_id"] for row in anchor_a_rows]
    anchor_b_ids = [row["query_id"] for row in anchor_b_rows]
    eval_rows = ranked_rows[n_anchors:-n_anchors]
    return anchor_a_ids, anchor_b_ids, eval_rows


def _target_values(relevant_rows, target_key):
    return np.asarray(
        [float(row[target_key]) for row in relevant_rows], dtype=np.float64
    )


def random_after_and_before_median(relevant_rows, target_key, n=10, seed=42):
    n_anchors = int(n)
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
    relevant_rows, target_key, separation=100, n=10, include_anchors_in_eval=False
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
    anchor_a_ids = []
    anchor_b_ids = []
    used_a_ids = set()
    used_b_ids = set()
    for first_item, last_item, _ in pairs:
        first_id = first_item["query_id"]
        last_id = last_item["query_id"]
        if first_id in used_a_ids or last_id in used_b_ids:
            continue
        anchor_a_ids.append(first_id)
        anchor_b_ids.append(last_id)
        used_a_ids.add(first_id)
        used_b_ids.add(last_id)
        if len(anchor_a_ids) >= int(n):
            break
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
    method,
    target_key,
    n=10,
    best_pairs_separation=100,
):
    if method in {"first_vs_last", "anchor_column"}:
        return first_vs_last(relevant_rows, target_key, n=n)
    if method == "random_median":
        return random_after_and_before_median(relevant_rows, target_key, n=n)
    if method == "best_pairs":
        return n_best_pairs_target_separation(
            relevant_rows,
            target_key,
            separation=best_pairs_separation,
            n=n,
        )
    raise ValueError(f"Unknown anchor selection method: {method}")
