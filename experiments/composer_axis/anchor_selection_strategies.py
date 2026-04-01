import random

import numpy as np


def random_after_and_before_median(relevant_rows, n: int | float = 5):
    _n = n if isinstance(n, int) else int(len(relevant_rows) * n)
    mid_year_median = np.median(
        np.array(
            [float(row["midpoint_year"]) for row in relevant_rows], dtype=np.float64
        )
    )
    # all above median are "a", all below median are "b"
    anchor_a_ids = [
        row["query_id"]
        for row in relevant_rows
        if float(row["midpoint_year"]) < mid_year_median
    ]
    anchor_b_ids = [
        row["query_id"]
        for row in relevant_rows
        if float(row["midpoint_year"]) >= mid_year_median
    ]
    # take 5 random samples from each side as anchors, the rest are eval;
    rng = random.Random(42)
    rng.shuffle(anchor_a_ids)
    anchor_a_ids = anchor_a_ids[:_n]
    rng.shuffle(anchor_b_ids)
    anchor_b_ids = anchor_b_ids[:_n]

    eval_rows = [
        row
        for row in relevant_rows
        if row["query_id"] not in anchor_a_ids + anchor_b_ids
    ]
    return anchor_a_ids, anchor_b_ids, eval_rows


def n_best_pairs_year_separation(
    relevant_rows, year_sep: int = 100, n=5, include_anchors_in_eval: bool = False
):
    """
    Selects the top N pairs for which the time difference is as close as
    possible to the specified year separation. This can help ensure that the
    anchors are well-distributed across the timeline, which may lead to a more
    meaningful projection.
    """

    pairs = []
    for i in range(len(relevant_rows)):
        for j in range(i + 1, len(relevant_rows)):
            first_item = (
                relevant_rows[i]
                if float(relevant_rows[i]["midpoint_year"])
                < float(relevant_rows[j]["midpoint_year"])
                else relevant_rows[j]
            )
            last_item = (
                relevant_rows[j] if first_item == relevant_rows[i] else relevant_rows[i]
            )
            pairs.append(
                (
                    first_item,
                    last_item,
                    abs(
                        float(last_item["midpoint_year"])
                        - float(first_item["midpoint_year"])
                    ),
                )
            )

    pairs.sort(key=lambda x: abs(x[2] - year_sep))
    anchor_a_ids = [pair[0]["query_id"] for pair in pairs[:n]]
    anchor_b_ids = [pair[1]["query_id"] for pair in pairs[:n]]
    eval_rows = [
        row
        for row in relevant_rows
        if not include_anchors_in_eval
        and row["query_id"] not in anchor_a_ids + anchor_b_ids
    ]
    return anchor_a_ids, anchor_b_ids, eval_rows


def use_anchor_column(relevant_rows, anchor_column):
    anchor_a_ids = [
        row["query_id"] for row in relevant_rows if row[anchor_column] == "a"
    ]
    anchor_b_ids = [
        row["query_id"] for row in relevant_rows if row[anchor_column] == "b"
    ]
    eval_rows = [row for row in relevant_rows if row[anchor_column] not in {"a", "b"}]
    return anchor_a_ids, anchor_b_ids, eval_rows
