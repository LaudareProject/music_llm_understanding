import numpy as np


def backward_eliminate_columns(
    score_fn, n_features, use_tqdm=False, on_improvement=None, on_stop=None
):
    """
    Greedy backward elimination over column indices.

    score_fn(cols) must return a scalar score where higher is better.
    Returns (history, final_cols, final_score).
    """
    active_cols = list(range(int(n_features)))
    score_before = float(score_fn(active_cols))
    history = []
    step = 0

    while len(active_cols) > 1:
        best_candidate_score = float("-inf")
        best_candidate_col = None
        iterator = active_cols
        if use_tqdm:
            from tqdm import tqdm

            iterator = tqdm(
                active_cols, desc=f"step {step + 1}", unit="col", leave=False
            )

        for col in iterator:
            candidate = [value for value in active_cols if value != col]
            score = float(score_fn(candidate))
            if score > best_candidate_score:
                best_candidate_score = score
                best_candidate_col = col

        if best_candidate_score <= score_before:
            if on_stop is not None:
                on_stop(step, score_before, best_candidate_score)
            break

        step += 1
        active_cols = [value for value in active_cols if value != best_candidate_col]
        history.append(
            {
                "step": step,
                "removed_col": int(best_candidate_col),
                "score_before": float(score_before),
                "score_after": float(best_candidate_score),
                "n_remaining": len(active_cols),
                "remaining_cols": active_cols[:],
            }
        )
        if on_improvement is not None:
            on_improvement(
                step,
                int(best_candidate_col),
                score_before,
                best_candidate_score,
                len(active_cols),
            )
        score_before = best_candidate_score

    return history, active_cols, score_before


def best_k_from_ranked(ranked_cols, score_fn, on_improvement=None):
    """Evaluate prefixes of ranked_cols and return (best_cols, best_score, curve)."""
    best_score = float("-inf")
    best_cols = None
    curve = []
    for k in range(1, len(ranked_cols) + 1):
        cols = ranked_cols[:k]
        score = float(score_fn(cols))
        curve.append({"k": k, "score": score, "cols": cols[:]})
        if score > best_score:
            best_score = score
            best_cols = cols[:]
            if on_improvement is not None:
                on_improvement(k, best_score)
    return best_cols, best_score, curve


def overlap_stats(cols_a, cols_b):
    set_a = set(cols_a)
    set_b = set(cols_b)
    inter = len(set_a & set_b)
    union = len(set_a | set_b)
    min_size = min(len(set_a), len(set_b))
    jaccard = inter / union if union > 0 else float("nan")
    overlap_min = inter / min_size if min_size > 0 else float("nan")
    return inter, union, jaccard, overlap_min


def pairwise_overlap_rows(method_sets):
    rows = []
    for i in range(len(method_sets)):
        for j in range(i + 1, len(method_sets)):
            left = method_sets[i]
            right = method_sets[j]
            inter, union, jaccard, overlap_min = overlap_stats(
                left["best_cols"], right["best_cols"]
            )
            rows.append(
                {
                    "method_a": left["method"],
                    "method_b": right["method"],
                    "n_a": len(left["best_cols"]),
                    "n_b": len(right["best_cols"]),
                    "intersection": inter,
                    "union": union,
                    "jaccard": f"{jaccard:.10f}" if not np.isnan(jaccard) else "nan",
                    "overlap_on_min": f"{overlap_min:.10f}"
                    if not np.isnan(overlap_min)
                    else "nan",
                }
            )
    return rows
