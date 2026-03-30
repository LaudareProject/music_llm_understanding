"""
Backward elimination on transformed (post-combo) features for the pitch axis.

Procedure:
  1. Load (or recompute) Aria embeddings from the shared cache.
  2. Re-evaluate all combos to find the best one using the same scoring logic as run.py.
  3. Apply the best combo transform to obtain the transformed matrix.
  4. Run backward elimination on the columns of that transformed matrix:
       - At every step, try removing each remaining column.
       - Keep the removal that improves the score the most; stop when no removal helps.
  5. Write feature_selection TSV outputs to a timestamped dir.

Scoring:
  - mapping == "raw"  ->  abs(spearman)
  - mapping != "raw"  ->  -l1_ratio   (lower ratio = better, so we negate it)
"""

from datetime import datetime
import os
from pathlib import Path
import subprocess
import sys

import mido
import numpy as np
from sklearn.feature_selection import (
    RFE,
    f_regression,
    mutual_info_regression,
    r_regression,
)
from sklearn.linear_model import LinearRegression
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.common.tsv import read_tsv, write_tsv
from experiments.common.math1d import (
    apply_combo,
    fit_exp_linear,
    fit_linear,
    fit_log_linear,
    l1,
    pearson,
    project_1d,
    random_l1_baseline,
    spearman,
)
from experiments.common.embed_aria import encode_midi


REPO_ROOT = Path(__file__).resolve().parents[2]
TSV_PATH = REPO_ROOT / "experiments" / "pitch_axis" / "notes.tsv"
OUTPUT_ROOT = REPO_ROOT / "outputs" / "experiments" / "pitch_axis"
CACHE_DIR = OUTPUT_ROOT / "cache"

COMBOS = [
    {"name": "baseline", "whiten": False, "lda": False, "pca": False},
    {"name": "whiten", "whiten": True, "lda": False, "pca": False},
    {"name": "lda", "whiten": False, "lda": True, "pca": False},
    {"name": "pca", "whiten": False, "lda": False, "pca": True},
    {"name": "whiten+lda", "whiten": True, "lda": True, "pca": False},
    {"name": "whiten+pca", "whiten": True, "lda": False, "pca": True},
]


def timestamp_dir():
    out_dir = OUTPUT_ROOT / (
        "feature_selection_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def write_note_midi(midi_path, midi_note):
    midi = mido.MidiFile()
    track = mido.MidiTrack()
    midi.tracks.append(track)
    track.append(mido.Message("program_change", program=0, channel=0, time=0))
    track.append(
        mido.Message("note_on", note=int(midi_note), velocity=64, channel=0, time=0)
    )
    track.append(
        mido.Message("note_off", note=int(midi_note), velocity=64, channel=0, time=480)
    )
    midi.save(str(midi_path))


def load_or_compute_embedding(query_id, midi_path):
    cache_path = CACHE_DIR / f"{query_id}.npy"
    if cache_path.exists():
        return np.load(cache_path)
    embedding = encode_midi(midi_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, embedding)
    return embedding


def score_projection(raw_projection, real, mapping_name):
    """Return a scalar score (higher = better) for the given mapping."""
    if mapping_name == "raw":
        return abs(spearman(real, raw_projection))
    predicted_map = {
        "linear": fit_linear,
        "log_linear": fit_log_linear,
        "exp_linear": fit_exp_linear,
    }
    predicted = predicted_map[mapping_name](raw_projection, real)
    baseline = random_l1_baseline(real)
    ratio = l1(real, predicted) / baseline if baseline > 0 else float("nan")
    if np.isnan(ratio):
        return float("-inf")
    return -ratio  # negate: lower l1_ratio is better


def evaluate_combo(
    transformed, anchor_a_idx, anchor_b_idx, eval_idx, real, mapping_name
):
    raw_proj = project_1d(transformed, anchor_a_idx, anchor_b_idx, eval_idx)
    return score_projection(raw_proj, real, mapping_name)


def find_best_combo(matrix, labels, anchor_a_idx, anchor_b_idx, eval_idx, real):
    """Return (best_combo_name, best_mapping_name, best_score, best_transformed)."""
    best_score = float("-inf")
    best_combo_name = None
    best_mapping_name = None
    best_transformed = None
    for combo in COMBOS:
        transformed = apply_combo(
            matrix, labels, combo["whiten"], combo["lda"], combo["pca"]
        )
        raw_proj = project_1d(transformed, anchor_a_idx, anchor_b_idx, eval_idx)
        for mapping_name in ["raw", "linear", "log_linear", "exp_linear"]:
            sc = score_projection(raw_proj, real, mapping_name)
            if sc > best_score:
                best_score = sc
                best_combo_name = combo["name"]
                best_mapping_name = mapping_name
                best_transformed = transformed
    return best_combo_name, best_mapping_name, best_score, best_transformed


def backward_elimination(
    transformed, anchor_a_idx, anchor_b_idx, eval_idx, real, mapping_name
):
    """
    Iterative backward elimination on columns of `transformed`.

    Returns a list of dicts (one per elimination step) with keys:
      step, removed_col, score_before, score_after, remaining_cols
    """
    n_features = transformed.shape[1]
    active_cols = list(range(n_features))

    def current_score(cols):
        sub = transformed[:, cols]
        return evaluate_combo(
            sub, anchor_a_idx, anchor_b_idx, eval_idx, real, mapping_name
        )

    score_before = current_score(active_cols)
    history = []
    step = 0

    print(
        f"  starting score ({mapping_name}): {score_before:.6f}  [{n_features} features]"
    )

    while len(active_cols) > 1:
        best_candidate_score = float("-inf")
        best_candidate_col = None
        for col in active_cols:
            candidate = [c for c in active_cols if c != col]
            sc = current_score(candidate)
            if sc > best_candidate_score:
                best_candidate_score = sc
                best_candidate_col = col

        if best_candidate_score <= score_before:
            print(
                f"  stopping: no removal improves score (best candidate: {best_candidate_score:.6f})"
            )
            break

        step += 1
        removed = best_candidate_col
        active_cols = [c for c in active_cols if c != removed]
        history.append(
            {
                "step": step,
                "removed_col": removed,
                "score_before": score_before,
                "score_after": best_candidate_score,
                "n_remaining": len(active_cols),
                "remaining_cols": active_cols[:],
            }
        )
        print(
            f"  step {step}: removed col {removed}  score {score_before:.6f} -> {best_candidate_score:.6f}  [{len(active_cols)} remaining]"
        )
        score_before = best_candidate_score

    return history, active_cols, score_before


def evaluate_subset_score(
    transformed, cols, anchor_a_idx, anchor_b_idx, eval_idx, real, mapping_name
):
    subset = transformed[:, cols]
    return evaluate_combo(
        subset, anchor_a_idx, anchor_b_idx, eval_idx, real, mapping_name
    )


def best_k_from_ranked(
    ranked_cols,
    transformed,
    anchor_a_idx,
    anchor_b_idx,
    eval_idx,
    real,
    mapping_name,
    method_name,
):
    best_score = float("-inf")
    best_cols = None
    curve = []
    print(f"    - {method_name}: evaluating k=1..{len(ranked_cols)}")
    for k in range(1, len(ranked_cols) + 1):
        cols = ranked_cols[:k]
        score = evaluate_subset_score(
            transformed,
            cols,
            anchor_a_idx,
            anchor_b_idx,
            eval_idx,
            real,
            mapping_name,
        )
        curve.append({"k": k, "score": score, "cols": cols[:]})
        if score > best_score:
            best_score = score
            best_cols = cols[:]
            print(f"      {method_name}: new best at k={k} score={best_score:.6f}")
    print(
        f"    - {method_name}: done | best_score={best_score:.6f} best_k={len(best_cols)}"
    )
    return best_cols, best_score, curve


def sklearn_selector_comparison(
    transformed,
    anchor_a_idx,
    anchor_b_idx,
    eval_idx,
    real,
    mapping_name,
):
    x_eval = transformed[np.asarray(eval_idx, dtype=int)]
    y_eval = np.asarray(real, dtype=np.float64)
    n_features = transformed.shape[1]
    out = []

    f_scores, _ = f_regression(x_eval, y_eval)
    f_scores = np.nan_to_num(f_scores, nan=-np.inf)
    f_rank = list(np.argsort(-f_scores).astype(int))
    f_cols, f_best, f_curve = best_k_from_ranked(
        f_rank,
        transformed,
        anchor_a_idx,
        anchor_b_idx,
        eval_idx,
        real,
        mapping_name,
        "f_regression",
    )
    out.append(
        {
            "method": "f_regression",
            "best_score": f_best,
            "best_n_features": len(f_cols),
            "best_cols": f_cols,
            "k_curve": f_curve,
        }
    )

    r_scores = r_regression(x_eval, y_eval)
    r_scores = np.nan_to_num(np.abs(r_scores), nan=-np.inf)
    r_rank = list(np.argsort(-r_scores).astype(int))
    r_cols, r_best, r_curve = best_k_from_ranked(
        r_rank,
        transformed,
        anchor_a_idx,
        anchor_b_idx,
        eval_idx,
        real,
        mapping_name,
        "r_regression_abs",
    )
    out.append(
        {
            "method": "r_regression_abs",
            "best_score": r_best,
            "best_n_features": len(r_cols),
            "best_cols": r_cols,
            "k_curve": r_curve,
        }
    )

    mi_scores = mutual_info_regression(x_eval, y_eval, random_state=0)
    mi_scores = np.nan_to_num(mi_scores, nan=-np.inf)
    mi_rank = list(np.argsort(-mi_scores).astype(int))
    mi_cols, mi_best, mi_curve = best_k_from_ranked(
        mi_rank,
        transformed,
        anchor_a_idx,
        anchor_b_idx,
        eval_idx,
        real,
        mapping_name,
        "mutual_info_regression",
    )
    out.append(
        {
            "method": "mutual_info_regression",
            "best_score": mi_best,
            "best_n_features": len(mi_cols),
            "best_cols": mi_cols,
            "k_curve": mi_curve,
        }
    )

    rfe_best_score = float("-inf")
    rfe_best_cols = None
    rfe_curve = []
    print(f"    - rfe_linear_regression: evaluating k=1..{n_features}")
    for k in tqdm(range(1, n_features + 1), desc="RFE", unit="k", leave=False):
        selector = RFE(estimator=LinearRegression(), n_features_to_select=k)
        selector.fit(x_eval, y_eval)
        cols = list(np.where(selector.support_)[0].astype(int))
        score = evaluate_subset_score(
            transformed,
            cols,
            anchor_a_idx,
            anchor_b_idx,
            eval_idx,
            real,
            mapping_name,
        )
        rfe_curve.append({"k": k, "score": score, "cols": cols[:]})
        if score > rfe_best_score:
            rfe_best_score = score
            rfe_best_cols = cols[:]
            print(
                f"      rfe_linear_regression: new best at k={k} score={rfe_best_score:.6f}"
            )
    print(
        "    - rfe_linear_regression: done | "
        f"best_score={rfe_best_score:.6f} best_k={len(rfe_best_cols)}"
    )
    out.append(
        {
            "method": "rfe_linear_regression",
            "best_score": rfe_best_score,
            "best_n_features": len(rfe_best_cols),
            "best_cols": rfe_best_cols,
            "k_curve": rfe_curve,
        }
    )

    return out


def overlap_stats(cols_a, cols_b):
    set_a = set(cols_a)
    set_b = set(cols_b)
    inter = len(set_a & set_b)
    union = len(set_a | set_b)
    min_size = min(len(set_a), len(set_b))
    jaccard = inter / union if union > 0 else float("nan")
    overlap_min = inter / min_size if min_size > 0 else float("nan")
    return inter, union, jaccard, overlap_min


def main():
    rows = read_tsv(TSV_PATH)
    output_dir = timestamp_dir()
    midi_dir = output_dir / "midi"
    midi_dir.mkdir(parents=True, exist_ok=True)

    print("[1/6] writing midi")
    midi_paths = {}
    for row in rows:
        midi_path = midi_dir / f"{row['query_id']}.mid"
        write_note_midi(midi_path, int(row["midi_note"]))
        midi_paths[row["query_id"]] = midi_path

    print("[2/6] loading embeddings")
    query_ids = [row["query_id"] for row in rows]
    matrix = np.stack(
        [
            load_or_compute_embedding(query_id, midi_paths[query_id])
            for query_id in tqdm(query_ids, desc="Embeddings", unit="query")
        ],
        axis=0,
    )
    labels = [f"octave_{int(row['midi_note']) // 12 - 1}" for row in rows]
    query_to_index = {query_id: i for i, query_id in enumerate(query_ids)}

    anchor_a_idx = [
        query_to_index[row["query_id"]] for row in rows if row["axis_side"] == "a"
    ]
    anchor_b_idx = [
        query_to_index[row["query_id"]] for row in rows if row["axis_side"] == "b"
    ]
    eval_rows = [row for row in rows if row["axis_side"] not in {"a", "b"}]
    eval_idx = [query_to_index[row["query_id"]] for row in eval_rows]
    real = np.asarray([float(row["midi_note"]) for row in eval_rows], dtype=np.float64)

    print("[3/6] finding best combo+mapping and running backward elimination")
    best_combo_name, best_mapping_name, best_score, best_transformed = find_best_combo(
        matrix, labels, anchor_a_idx, anchor_b_idx, eval_idx, real
    )
    print(
        f"  best combo: {best_combo_name}  mapping: {best_mapping_name}  score: {best_score:.6f}"
    )
    print(f"  transformed shape: {best_transformed.shape}")

    history, final_cols, final_score = backward_elimination(
        best_transformed, anchor_a_idx, anchor_b_idx, eval_idx, real, best_mapping_name
    )

    print("[4/6] comparing sklearn selectors")
    sklearn_comparisons = sklearn_selector_comparison(
        best_transformed,
        anchor_a_idx,
        anchor_b_idx,
        eval_idx,
        real,
        best_mapping_name,
    )

    print("[5/6] writing outputs")
    tsv_rows = []
    for entry in history:
        tsv_rows.append(
            {
                "step": entry["step"],
                "removed_col": entry["removed_col"],
                "score_before": f"{entry['score_before']:.10f}",
                "score_after": f"{entry['score_after']:.10f}",
                "n_remaining": entry["n_remaining"],
                "remaining_cols": ",".join(str(c) for c in entry["remaining_cols"]),
            }
        )
    write_tsv(
        output_dir / "feature_selection.tsv",
        tsv_rows,
        [
            "step",
            "removed_col",
            "score_before",
            "score_after",
            "n_remaining",
            "remaining_cols",
        ],
    )

    comparison_rows = [
        {
            "method": "backward_elimination",
            "best_score": f"{final_score:.10f}",
            "best_n_features": len(final_cols),
            "best_cols": ",".join(str(c) for c in final_cols),
        }
    ]
    for item in sklearn_comparisons:
        comparison_rows.append(
            {
                "method": item["method"],
                "best_score": f"{item['best_score']:.10f}",
                "best_n_features": item["best_n_features"],
                "best_cols": ",".join(str(c) for c in item["best_cols"]),
            }
        )
    write_tsv(
        output_dir / "feature_selection_comparison.tsv",
        comparison_rows,
        ["method", "best_score", "best_n_features", "best_cols"],
    )

    method_sets = [
        {
            "method": "backward_elimination",
            "best_score": float(final_score),
            "best_cols": final_cols[:],
        }
    ] + [
        {
            "method": item["method"],
            "best_score": float(item["best_score"]),
            "best_cols": item["best_cols"][:],
        }
        for item in sklearn_comparisons
    ]

    overlap_rows = []
    for i in range(len(method_sets)):
        for j in range(i + 1, len(method_sets)):
            left = method_sets[i]
            right = method_sets[j]
            inter, union, jaccard, overlap_min = overlap_stats(
                left["best_cols"], right["best_cols"]
            )
            overlap_rows.append(
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
    write_tsv(
        output_dir / "feature_set_overlap.tsv",
        overlap_rows,
        [
            "method_a",
            "method_b",
            "n_a",
            "n_b",
            "intersection",
            "union",
            "jaccard",
            "overlap_on_min",
        ],
    )

    best_method = max(method_sets, key=lambda row: row["best_score"])

    print("[6/6] running run.py with selected feature set")
    print(
        f"  selected method: {best_method['method']}  score={best_method['best_score']:.6f}"
    )
    print(f"  selected cols: {len(best_method['best_cols'])}")
    env = os.environ.copy()
    env["FEATURE_SELECTION_COMBO"] = best_combo_name
    env["FEATURE_SELECTION_COLS"] = ",".join(
        str(col) for col in best_method["best_cols"]
    )
    subprocess.run(
        [str(sys.executable), str(REPO_ROOT / "experiments" / "pitch_axis" / "run.py")],
        check=True,
        cwd=str(REPO_ROOT),
        env=env,
    )

    print(f"  initial features: {best_transformed.shape[1]}")
    print(f"  final features:   {len(final_cols)}  {final_cols}")
    print(f"  final score:      {final_score:.6f}")
    print(output_dir)


if __name__ == "__main__":
    main()
