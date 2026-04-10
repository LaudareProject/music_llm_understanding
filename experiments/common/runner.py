from __future__ import annotations

import csv
import json
import os
from pathlib import Path

import mido
import numpy as np
from scipy.stats import spearmanr
from sklearn.base import clone
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV, KFold

from experiments.common.feature_backend import (
    FEATURE_BACKEND_ARIA,
    FEATURE_BACKEND_ENV,
    FEATURE_BACKEND_MUSIF,
    aria_embedding_path_for_row,
    cache_dir_for_experiment,
    feature_backend_name,
    matrix_for_rows,
    output_root_for_experiment,
)
from experiments.common.folder_handling import create_folder
from experiments.common.humdrum import kern_to_midi, musicxml_to_midi
from experiments.common.plotting import save_n_vs_spearman, save_predicted_vs_real_plot
from experiments.common.tsv import read_tsv, write_tsv


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


def _crossval_oof_predictions(estimator, matrix, y, cv):
    n_samples = int(matrix.shape[0])
    oof_pred = np.full(n_samples, np.nan, dtype=np.float64)
    anchor_counts = np.zeros(n_samples, dtype=np.int64)
    trial_scores = []

    for train_idx, test_idx in cv.split(matrix, y):
        model = clone(estimator)
        model.fit(matrix[train_idx], y[train_idx])
        fold_pred = np.asarray(model.predict(matrix[test_idx]), dtype=np.float64)
        if fold_pred.ndim == 1:
            fold_pred = fold_pred.reshape(1, -1)
        oof_pred[test_idx] = np.mean(fold_pred, axis=0)

        for trial_pred in fold_pred:
            trial_scores.append(float(spearman_metric(y[test_idx], trial_pred)))

        for metadata in model.trial_anchor_metadata_:
            local_anchor_idx = np.asarray(metadata["anchor_indices"], dtype=np.int64)
            if local_anchor_idx.size == 0:
                continue
            global_anchor_idx = np.asarray(train_idx, dtype=np.int64)[local_anchor_idx]
            anchor_counts[global_anchor_idx] += 1

    return oof_pred, anchor_counts, trial_scores


def _normalize_midi_to_piano(midi_path):
    midi = mido.MidiFile(str(midi_path))
    changed = False
    for track in midi.tracks:
        for msg in track:
            if msg.is_meta:
                continue
            if hasattr(msg, "channel") and int(msg.channel) != 0:
                msg.channel = 0
                changed = True
            if msg.type == "program_change" and int(msg.program) != 0:
                msg.program = 0
                changed = True
    if changed:
        midi.save(str(midi_path))


def _prepare_midi_for_row(row, repo_root, midi_dir, bin_dir):
    source_path = (repo_root / row["path"]).resolve()
    midi_path = midi_dir / f"{row['query_id']}.mid"
    if row["input_kind"] == "midi":
        _normalize_midi_to_piano(source_path)
        return source_path
    if row["input_kind"] == "musicxml":
        musicxml_to_midi(source_path, midi_path, bin_dir)
        _normalize_midi_to_piano(midi_path)
        return midi_path
    if row["input_kind"] == "kern":
        kern_to_midi(source_path, midi_path, bin_dir)
        _normalize_midi_to_piano(midi_path)
        return midi_path
    raise ValueError(row["input_kind"])


def _prepare_rows_and_midi(experiment_name, tsv_path, repo_root, output_dir):
    rows = read_tsv(tsv_path)
    midi_dir = output_dir / "midi"
    midi_dir.mkdir(parents=True, exist_ok=True)
    bin_dir = repo_root / "bin"
    midi_paths = {}
    skipped = set()
    for row in rows:
        query_id = row["query_id"]
        cache_embedding = aria_embedding_path_for_row(row, repo_root)
        if cache_embedding.exists():
            continue
        try:
            midi_paths[query_id] = _prepare_midi_for_row(
                row, repo_root, midi_dir, bin_dir
            )
        except Exception as exc:
            print(f"  skipping {experiment_name}:{query_id}: {exc}")
            skipped.add(query_id)
    filtered = [row for row in rows if row["query_id"] not in skipped]
    return filtered, midi_paths


def _write_gridsearch_csv(path, grid_search):
    rows = []
    for i in range(len(grid_search.cv_results_["params"])):
        row = {
            "rank_test_score": int(grid_search.cv_results_["rank_test_score"][i]),
            "mean_test_score": float(grid_search.cv_results_["mean_test_score"][i]),
            "std_test_score": float(grid_search.cv_results_["std_test_score"][i]),
            "mean_train_score": float(grid_search.cv_results_["mean_train_score"][i]),
            "std_train_score": float(grid_search.cv_results_["std_train_score"][i]),
            "params": json.dumps(grid_search.cv_results_["params"][i], sort_keys=True),
        }
        rows.append(row)
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "rank_test_score",
                "mean_test_score",
                "std_test_score",
                "mean_train_score",
                "std_train_score",
                "params",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def _same_params_except_n(candidate_params, best_params):
    for key, best_value in best_params.items():
        if key == "projection__n":
            continue
        if key not in candidate_params:
            return False
        if candidate_params[key] != best_value:
            return False
    return True


def _n_sweep_from_grid(grid, n_values):
    cv_results = grid.cv_results_
    split_keys = sorted(
        [
            key
            for key in cv_results.keys()
            if key.startswith("split") and key.endswith("_test_score")
        ]
    )

    by_n = {}
    for idx, params in enumerate(cv_results["params"]):
        if not _same_params_except_n(params, grid.best_params_):
            continue
        if "projection__n" not in params:
            continue
        n_value = int(params["projection__n"])
        split_scores = []
        for key in split_keys:
            value = float(cv_results[key][idx])
            if np.isfinite(value):
                split_scores.append(value)
        mean_score, ci95_score = mean_ci95(split_scores)
        by_n[n_value] = (mean_score, ci95_score)

    n_scores = []
    n_ci95 = []
    for n in n_values:
        if int(n) in by_n:
            mean_score, ci95_score = by_n[int(n)]
        else:
            mean_score, ci95_score = float("nan"), float("nan")
        n_scores.append(mean_score)
        n_ci95.append(ci95_score)
    return n_scores, n_ci95


def run_experiment(settings, experiment_name):
    if experiment_name not in settings.EXPERIMENTS:
        raise ValueError(f"Unknown experiment: {experiment_name}")

    spec = settings.EXPERIMENTS[experiment_name]
    backend = (
        str(
            spec.get(
                "feature_backend", getattr(settings, "DEFAULT_FEATURE_BACKEND", "aria")
            )
        )
        .strip()
        .lower()
    )
    if backend not in {FEATURE_BACKEND_ARIA, FEATURE_BACKEND_MUSIF}:
        raise ValueError(
            f"Invalid feature_backend '{backend}' for {experiment_name}. "
            f"Use '{FEATURE_BACKEND_ARIA}' or '{FEATURE_BACKEND_MUSIF}'."
        )
    os.environ[FEATURE_BACKEND_ENV] = backend

    repo_root = Path(__file__).resolve().parents[2]
    output_root = output_root_for_experiment(repo_root, experiment_name)
    cache_dir = cache_dir_for_experiment(repo_root, experiment_name)
    output_dir = create_folder(
        output_root,
        {
            "experiment": experiment_name,
            "feature_backend": feature_backend_name(),
            "grid_size": len(spec["param_grid"]),
        },
    )

    print("[1/5] preparing rows, midi, embeddings")
    rows, midi_paths = _prepare_rows_and_midi(
        experiment_name,
        spec["tsv_path"],
        repo_root,
        output_dir,
    )
    query_ids = [row["query_id"] for row in rows]
    y = np.asarray([float(row[spec["target_col"]]) for row in rows], dtype=np.float64)
    matrix = matrix_for_rows(rows, midi_paths, cache_dir, repo_root, "Features")
    np.save(output_dir / "embeddings.npy", matrix)
    write_tsv(
        output_dir / "rows.tsv",
        rows,
        list(rows[0].keys()) if rows else ["query_id"],
    )

    print("[2/5] running grid search")
    base_estimator = clone(spec["pipeline"])
    scorer = make_scorer(spearman_metric, greater_is_better=True)
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    grid = GridSearchCV(
        estimator=base_estimator,
        param_grid=spec["param_grid"],
        scoring=scorer,
        cv=cv,
        n_jobs=1,
        refit=True,
        return_train_score=True,
        error_score=np.nan,
    )
    grid.fit(matrix, y)
    _write_gridsearch_csv(output_dir / "gridsearch_results.csv", grid)
    (output_dir / "best_params.json").write_text(
        json.dumps(grid.best_params_, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print("[3/5] cross-validated predictions for best pipeline")
    best_estimator = clone(spec["pipeline"]).set_params(**grid.best_params_)
    y_hat_all, anchor_counts, trial_scores = _crossval_oof_predictions(
        best_estimator,
        matrix,
        y,
        cv,
    )
    valid_mask = np.isfinite(y_hat_all)
    y_test = y[valid_mask]
    y_test_hat = y_hat_all[valid_mask]
    anchor_mask = anchor_counts > 0
    y_anchor = y[anchor_mask]
    y_anchor_hat = y_hat_all[anchor_mask]
    score_mean, score_ci95 = mean_ci95(trial_scores)
    scatter_title = (
        f"{experiment_name} best pipeline | "
        f"spearman={score_mean:.4f} +/- {score_ci95:.4f}"
    )
    save_predicted_vs_real_plot(
        output_dir / "predicted_vs_real.png",
        y_test,
        y_test_hat,
        y_anchor,
        y_anchor_hat,
        scatter_title,
    )

    projection_rows = []
    for i, query_id in enumerate(query_ids):
        projection_rows.append(
            {
                "query_id": query_id,
                "target": f"{y[i]:.10f}",
                "predicted": f"{y_hat_all[i]:.10f}",
                "is_anchor": "1" if anchor_mask[i] else "0",
            }
        )
    write_tsv(
        output_dir / "projections.tsv",
        projection_rows,
        ["query_id", "target", "predicted", "is_anchor"],
    )
    write_tsv(
        output_dir / "metrics.tsv",
        [
            {
                "metric": "spearman",
                "mean": f"{score_mean:.10f}",
                "ci95": f"{score_ci95:.10f}",
                "folds": "5",
            }
        ],
        ["metric", "mean", "ci95", "folds"],
    )

    print("[4/5] sweeping n for best pipeline")
    n_values = list(spec["n_sweep"])
    n_scores, n_ci95 = _n_sweep_from_grid(grid, n_values)

    save_n_vs_spearman(
        output_dir / "n_vs_spearman.png",
        n_values,
        n_scores,
        f"{experiment_name} | n vs Spearman",
        ci95_values=n_ci95,
    )
    write_tsv(
        output_dir / "n_vs_spearman.tsv",
        [
            {"n": str(n), "spearman": f"{s:.10f}", "ci95": f"{c:.10f}"}
            for n, s, c in zip(n_values, n_scores, n_ci95)
        ],
        ["n", "spearman", "ci95"],
    )

    print("[5/5] done")
    print(output_dir)
    return output_dir
