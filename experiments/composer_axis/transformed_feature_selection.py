import os
from pathlib import Path
import subprocess
import sys

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
from experiments.common.anchor_selection import select_anchors
from experiments.common.folder_handling import create_folder
from experiments.common.math1d import (
    COMBOS,
    apply_combo,
    combo_config_by_name,
    fit_exp_linear,
    fit_linear,
    fit_log_linear,
    hann_density,
    l1,
    project_1d,
    spearman,
)
from experiments.common.feature_select import (
    backward_eliminate_columns,
    best_k_from_ranked,
    pairwise_overlap_rows,
)
from experiments.common.humdrum import kern_to_midi, musicxml_to_midi
from experiments.common.embed_aria import encode_midi


REPO_ROOT = Path(__file__).resolve().parents[2]
TSV_PATH = REPO_ROOT / "experiments" / "composer_axis" / "songs.tsv"
OUTPUT_ROOT = REPO_ROOT / "outputs" / "experiments" / "composer_axis"
CACHE_DIR = REPO_ROOT / "cache" / "experiments" / "composer_axis"
BIN_DIR = REPO_ROOT / "bin"

DATASETS = ["asap", "music21", "ewld", "overall"]
ANCHOR_N = 10


def prepare_midi(row, midi_dir):
    source_path = (REPO_ROOT / row["path"]).resolve()
    midi_path = midi_dir / f"{row['query_id']}.mid"
    if row["input_kind"] == "midi":
        return source_path
    if row["input_kind"] == "musicxml":
        musicxml_to_midi(source_path, midi_path, BIN_DIR)
        return midi_path
    if row["input_kind"] == "kern":
        kern_to_midi(source_path, midi_path, BIN_DIR)
        return midi_path
    raise ValueError(row["input_kind"])


def load_or_compute_embedding(query_id, midi_path):
    cache_path = CACHE_DIR / f"{query_id}.npy"
    if cache_path.exists():
        return np.load(cache_path)
    embedding = encode_midi(midi_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, embedding)
    return embedding


def dataset_index_sets(rows, query_to_index, dataset_name):
    """Return anchor/eval indices plus chronology targets for one dataset."""
    if dataset_name == "overall":
        relevant = rows
    else:
        relevant = [row for row in rows if row["dataset"] == dataset_name]
    anchor_a_ids, anchor_b_ids, eval_rows = select_anchors(
        relevant,
        "first_vs_last",
        target_key="midpoint_year",
        n=ANCHOR_N,
        best_pairs_separation=100,
    )
    anchor_a_idx = [query_to_index[query_id] for query_id in anchor_a_ids]
    anchor_b_idx = [query_to_index[query_id] for query_id in anchor_b_ids]
    if len(eval_rows) < 2:
        return None
    eval_idx = [query_to_index[row["query_id"]] for row in eval_rows]
    real = np.asarray(
        [float(row["midpoint_year"]) for row in eval_rows], dtype=np.float64
    )
    birth_years = np.asarray(
        [float(row["birth_year"]) for row in eval_rows], dtype=np.float64
    )
    death_years = np.asarray(
        [float(row["death_year"]) for row in eval_rows], dtype=np.float64
    )
    return anchor_a_idx, anchor_b_idx, eval_idx, real, birth_years, death_years


def score_one_dataset(
    transformed,
    anchor_a_idx,
    anchor_b_idx,
    eval_idx,
    real,
    birth_years,
    death_years,
    mapping_name,
):
    raw_proj = project_1d(transformed, anchor_a_idx, anchor_b_idx, eval_idx)
    if mapping_name == "raw":
        sc = abs(spearman(real, raw_proj))
        return sc if not np.isnan(sc) else None
    fit_fn = {
        "linear": fit_linear,
        "log_linear": fit_log_linear,
        "exp_linear": fit_exp_linear,
    }[mapping_name]
    predicted = fit_fn(raw_proj, real)
    value = hann_density(predicted, birth_years, death_years)
    if np.isnan(value):
        return None
    return value


def combo_score(transformed, rows, query_to_index, mapping_name):
    """Mean score over all datasets for a given mapping (higher = better)."""
    index_sets = {}
    for ds in DATASETS:
        result = dataset_index_sets(rows, query_to_index, ds)
        if result is not None:
            index_sets[ds] = result

    scores = []
    for ds, (aa, ab, ei, real, birth_years, death_years) in index_sets.items():
        sc = score_one_dataset(
            transformed,
            aa,
            ab,
            ei,
            real,
            birth_years,
            death_years,
            mapping_name,
        )
        if sc is not None:
            scores.append(sc)
    return float(np.mean(scores)) if scores else float("-inf")


def find_best_combo(matrix, labels, target_scores, rows, query_to_index):
    """Return (best_combo_name, best_mapping_name, best_score, best_transformed)."""
    best_score = float("-inf")
    best_combo_name = None
    best_mapping_name = None
    best_transformed = None
    for combo in COMBOS:
        transformed = apply_combo(matrix, labels, combo, target_scores=target_scores)
        for mapping_name in ["linear", "log_linear", "exp_linear"]:
            sc = combo_score(transformed, rows, query_to_index, mapping_name)
            if sc > best_score:
                best_score = sc
                best_combo_name = combo["name"]
                best_mapping_name = mapping_name
                best_transformed = transformed
    return best_combo_name, best_mapping_name, best_score, best_transformed


def evaluate_subset_score(transformed, cols, rows, query_to_index, mapping_name):
    subset = transformed[:, cols]
    return combo_score(subset, rows, query_to_index, mapping_name)


def ranked_method(
    ranked_cols,
    score_fn,
    method_name,
):
    print(f"    - {method_name}: evaluating k=1..{len(ranked_cols)}")
    best_cols, best_score, curve = best_k_from_ranked(
        ranked_cols,
        score_fn,
        on_improvement=lambda k, score: print(
            f"      {method_name}: new best at k={k} score={score:.6f}"
        ),
    )
    print(
        f"    - {method_name}: done | best_score={best_score:.6f} best_k={len(best_cols)}"
    )
    return best_cols, best_score, curve


def sklearn_selector_comparison(feature_matrix, score_fn, rows, query_to_index):
    overall = dataset_index_sets(rows, query_to_index, "overall")
    if overall is None:
        return []
    _, _, eval_idx, real, _, _ = overall

    x_eval = feature_matrix[np.asarray(eval_idx, dtype=int)]
    y_eval = np.asarray(real, dtype=np.float64)
    n_features = feature_matrix.shape[1]
    out = []

    f_scores, _ = f_regression(x_eval, y_eval)
    f_scores = np.nan_to_num(f_scores, nan=-np.inf)
    f_rank = list(np.argsort(-f_scores).astype(int))
    f_cols, f_best, f_curve = ranked_method(f_rank, score_fn, "f_regression")
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
    r_cols, r_best, r_curve = ranked_method(r_rank, score_fn, "r_regression_abs")
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
    mi_cols, mi_best, mi_curve = ranked_method(
        mi_rank, score_fn, "mutual_info_regression"
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

    # rfe_best_score = float("-inf")
    # rfe_best_cols = None
    # rfe_curve = []
    # print(f"    - rfe_linear_regression: evaluating k=1..{n_features}")
    # for k in tqdm(range(1, (n_features + 1) // 2, 4), desc="RFE", unit="k", leave=False):
    #     selector = RFE(estimator=LinearRegression(), n_features_to_select=k)
    #     selector.fit(x_eval, y_eval)
    #     cols = list(np.where(selector.support_)[0].astype(int))
    #     score = float(score_fn(cols))
    #     rfe_curve.append({"k": k, "score": score, "cols": cols[:]})
    #     if score > rfe_best_score:
    #         rfe_best_score = score
    #         rfe_best_cols = cols[:]
    #         print(
    #             f"      rfe_linear_regression: new best at k={k} score={rfe_best_score:.6f}"
    #         )
    # print(
    #     "    - rfe_linear_regression: done | "
    #     f"best_score={rfe_best_score:.6f} best_k={len(rfe_best_cols)}"
    # )
    # out.append(
    #     {
    #         "method": "rfe_linear_regression",
    #         "best_score": rfe_best_score,
    #         "best_n_features": len(rfe_best_cols),
    #         "best_cols": rfe_best_cols,
    #         "k_curve": rfe_curve,
    #     }
    # )

    return out


def run_feature_selection(selection_space="transformed"):
    if selection_space not in {"transformed", "raw"}:
        raise ValueError(f"Unknown selection space: {selection_space}")
    rows = read_tsv(TSV_PATH)
    output_dir = create_folder(OUTPUT_ROOT, {"selection_space": selection_space})
    midi_dir = output_dir / "midi"
    midi_dir.mkdir(parents=True, exist_ok=True)

    print("[1/6] preparing midi")
    midi_paths = {}
    skipped = set()
    for row in tqdm(rows, desc="MIDI", unit="song"):
        query_id = row["query_id"]
        if (CACHE_DIR / f"{query_id}.npy").exists():
            continue
        try:
            midi_paths[query_id] = prepare_midi(row, midi_dir)
        except Exception as exc:
            print(f"  skipping {query_id}: {exc}")
            skipped.add(query_id)
    rows = [row for row in rows if row["query_id"] not in skipped]

    print("[2/6] loading embeddings")
    query_ids = [row["query_id"] for row in rows]
    embeddings = []
    for query_id in tqdm(query_ids, desc="Embeddings", unit="query"):
        embeddings.append(load_or_compute_embedding(query_id, midi_paths.get(query_id)))
    matrix = np.stack(embeddings, axis=0)
    labels = [row["composer"] for row in rows]
    target_scores = np.asarray(
        [float(row["midpoint_year"]) for row in rows], dtype=np.float64
    )
    query_to_index = {query_id: i for i, query_id in enumerate(query_ids)}

    print("[3/6] finding best combo+mapping")
    best_combo_name, best_mapping_name, best_score, best_transformed = find_best_combo(
        matrix, labels, target_scores, rows, query_to_index
    )
    print(
        f"  best combo: {best_combo_name}  mapping: {best_mapping_name}  score: {best_score:.6f}"
    )
    print(f"  transformed shape: {best_transformed.shape}")

    combo_cfg = combo_config_by_name(best_combo_name)
    if selection_space == "transformed":
        selection_matrix = best_transformed

        def score_fn(cols):
            return evaluate_subset_score(
                best_transformed, cols, rows, query_to_index, best_mapping_name
            )

    else:
        selection_matrix = matrix

        def score_fn(cols):
            raw_subset = matrix[:, cols]
            transformed_subset = apply_combo(
                raw_subset, labels, combo_cfg, target_scores=target_scores
            )
            return combo_score(
                transformed_subset, rows, query_to_index, best_mapping_name
            )

    initial_score = float(score_fn(list(range(selection_matrix.shape[1]))))
    print(
        f"  starting score ({best_mapping_name}): {initial_score:.6f}  [{selection_matrix.shape[1]} features]"
    )
    # history, final_cols, final_score = backward_eliminate_columns(
    #     score_fn,
    #     selection_matrix.shape[1],
    #     use_tqdm=True,
    #     on_improvement=lambda step, removed, before, after, remaining: print(
    #         f"  step {step}: removed col {removed}  score {before:.6f} -> {after:.6f}  [{remaining} remaining]"
    #     ),
    #     on_stop=lambda step, before, best_candidate: print(
    #         "  stopping: no removal improves score "
    #         f"(best candidate: {best_candidate:.6f})"
    #     ),
    # )

    print("[4/6] comparing sklearn selectors")
    sklearn_comparisons = sklearn_selector_comparison(
        selection_matrix, score_fn, rows, query_to_index
    )

    print("[5/6] writing outputs")
    tsv_rows = []
    # for entry in history:
    #     tsv_rows.append(
    #         {
    #             "step": entry["step"],
    #             "removed_col": entry["removed_col"],
    #             "score_before": f"{entry['score_before']:.10f}",
    #             "score_after": f"{entry['score_after']:.10f}",
    #             "n_remaining": entry["n_remaining"],
    #             "remaining_cols": ",".join(str(c) for c in entry["remaining_cols"]),
    #         }
    #     )
    # write_tsv(
    #     output_dir / "feature_selection.tsv",
    #     tsv_rows,
    #     [
    #         "step",
    #         "removed_col",
    #         "score_before",
    #         "score_after",
    #         "n_remaining",
    #         "remaining_cols",
    #     ],
    # )

    comparison_rows = [
        # {
        #     "method": "backward_elimination",
        #     "best_score": f"{final_score:.10f}",
        #     "best_n_features": len(final_cols),
        #     "best_cols": ",".join(str(c) for c in final_cols),
        # }
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

    method_sets = (
        [
            # {
            #     "method": "backward_elimination",
            #     "best_score": float(final_score),
            #     "best_cols": final_cols[:],
            # }
        ]
        + [
            {
                "method": item["method"],
                "best_score": float(item["best_score"]),
                "best_cols": item["best_cols"][:],
            }
            for item in sklearn_comparisons
        ]
    )

    overlap_rows = pairwise_overlap_rows(method_sets)
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
    env["FEATURE_SELECTION_SPACE"] = selection_space
    env["FEATURE_SELECTION_COLS"] = ",".join(
        str(col) for col in best_method["best_cols"]
    )
    subprocess.run(
        [
            str(sys.executable),
            str(REPO_ROOT / "experiments" / "composer_axis" / "run.py"),
        ],
        check=True,
        cwd=str(REPO_ROOT),
        env=env,
    )

    final_cols = best_method["best_cols"]
    final_score = best_method["best_score"]
    print(f"  initial features: {selection_matrix.shape[1]}")
    print(f"  final features:   {len(final_cols)}  {final_cols}")
    print(f"  final score:      {final_score:.6f}")
    print(output_dir)


def main():
    run_feature_selection("transformed")


if __name__ == "__main__":
    main()
