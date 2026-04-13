from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from sklearn.base import clone
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV, KFold

from experiments.settings import CV_FOLDS, GRID_JOBS, SCORE_TOLERANCE
from experiments.common.feature_backend import (
    FEATURE_BACKEND_ARIA,
    FEATURE_BACKEND_ENV,
    FEATURE_BACKEND_MUSIF,
    cache_dir_for_experiment,
    feature_backend_name,
    matrix_for_rows,
    output_root_for_experiment,
)
from experiments.common.folder_handling import create_folder
from experiments.common.plotting import (
    save_n_vs_spearman_combined,
    save_predicted_vs_real_plot,
)
from experiments.common.runner_data import prepare_rows_and_midi
from experiments.common.runner_io import (
    write_gridsearch_csv,
    write_metrics_output,
    write_n_sweep_output,
    write_prediction_outputs,
)
from experiments.common.runner_search import (
    crossval_oof_predictions,
    default_anchor_config,
    n_sweep_extra_metrics,
    n_sweep_from_grid,
    select_grid_params_with_min_n,
    split_score_stats_for_params,
)
from experiments.common.runner_stats import (
    mean_ci95,
    remove_target_outliers,
    spearman_metric,
)
from experiments.common.tsv import write_tsv


def _check_backend(settings, experiment_name, spec):
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


def run_experiment(settings, experiment_name):
    if experiment_name not in settings.EXPERIMENTS:
        raise ValueError(f"Unknown experiment: {experiment_name}")

    spec = settings.EXPERIMENTS[experiment_name]
    _check_backend(settings, experiment_name, spec)

    repo_root = Path(__file__).resolve().parents[2]
    output_dir = create_folder(
        output_root_for_experiment(repo_root, experiment_name),
        {
            "experiment": experiment_name,
            "feature_backend": feature_backend_name(),
            "grid_size": len(spec["param_grid"]),
        },
    )
    cache_dir = cache_dir_for_experiment(repo_root, experiment_name)

    print("[1/5] preparing rows, midi, embeddings")
    rows, midi_paths = prepare_rows_and_midi(
        experiment_name,
        spec["tsv_path"],
        repo_root,
        output_dir,
        row_filter_col=spec.get("row_filter_col"),
        row_filter_value=spec.get("row_filter_value"),
    )
    total_rows_before_outliers = len(rows)
    y = np.asarray([float(row[spec["target_col"]]) for row in rows], dtype=np.float64)
    group_labels = None
    if spec.get("group_col") is not None:
        group_labels = np.asarray(
            [row[spec["group_col"]] for row in rows], dtype=object
        )

    outlier_iqr_factor = spec.get(
        "outlier_iqr_factor", getattr(settings, "OUTLIER_IQR_FACTOR", 1.5)
    )
    rows, y, group_labels, removed_outliers, outlier_low, outlier_high = (
        remove_target_outliers(
            rows,
            y,
            group_labels,
            iqr_factor=outlier_iqr_factor,
        )
    )
    print(
        "  target outlier filter before CV: "
        f"removed={removed_outliers}, kept={len(rows)}, total={total_rows_before_outliers}, "
        f"iqr_factor={outlier_iqr_factor}, range=[{outlier_low:.4f}, {outlier_high:.4f}]"
    )

    matrix = matrix_for_rows(rows, midi_paths, cache_dir, repo_root, "Features")
    np.save(output_dir / "embeddings.npy", matrix)
    write_tsv(
        output_dir / "rows.tsv", rows, list(rows[0].keys()) if rows else ["query_id"]
    )

    print("[2/5] running grid search")
    cv = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=42)
    grid = GridSearchCV(
        estimator=clone(spec["pipeline"]),
        param_grid=spec["param_grid"],
        scoring=make_scorer(spearman_metric, greater_is_better=True),
        cv=cv,
        n_jobs=GRID_JOBS,
        refit=False,
        return_train_score=True,
        verbose=1,
        error_score=np.nan,
    )
    if group_labels is None:
        grid.fit(matrix, y)
    else:
        grid.fit(matrix, y, group_labels=group_labels)
    write_gridsearch_csv(output_dir / "gridsearch_results.csv", grid)

    print("[3/5] cross-validated predictions for best pipeline")
    selected_params, selected_score, best_score = select_grid_params_with_min_n(
        grid, SCORE_TOLERANCE
    )
    best_estimator = clone(spec["pipeline"]).set_params(**selected_params)
    print(
        "  selected params with minimum n under tolerance: "
        f"best={best_score:.4f}, selected={selected_score:.4f}, "
        f"tolerance={SCORE_TOLERANCE:.4f}, n={selected_params.get('anchor__n', 'unknown')}"
    )
    y_hat_all, anchor_counts, trial_scores, _, _ = crossval_oof_predictions(
        best_estimator,
        matrix,
        y,
        cv,
        group_labels=group_labels,
    )

    valid_mask = np.isfinite(y_hat_all)
    score_mean, score_ci95 = split_score_stats_for_params(grid, selected_params)
    if not np.isfinite(score_mean):
        score_mean, score_ci95 = mean_ci95(trial_scores)
    anchor_mask = anchor_counts > 0
    anchor_method, anchor_n = default_anchor_config(best_estimator)
    group_plot = (
        np.asarray(["all"] * int(np.sum(valid_mask)), dtype=object)
        if group_labels is None
        else np.asarray(group_labels, dtype=object)[valid_mask]
    )
    save_predicted_vs_real_plot(
        output_dir / "predicted_vs_real.png",
        y[valid_mask],
        y_hat_all[valid_mask],
        group_plot,
        anchor_mask[valid_mask],
        f"{experiment_name} best pipeline | anchor={anchor_method}, n={anchor_n}\n"
        f"spearman={score_mean:.4f} +/- {score_ci95:.4f}",
    )
    write_prediction_outputs(
        output_dir, rows, y, y_hat_all, anchor_mask, removed_outliers
    )
    write_metrics_output(
        output_dir,
        score_mean,
        score_ci95,
        removed_outliers,
        total_rows_before_outliers,
        len(rows),
    )

    print("[4/5] sweeping n for best pipeline")
    n_sweep = n_sweep_from_grid(grid, reference_params=selected_params)
    n_sweep_extra = n_sweep_extra_metrics(
        grid,
        spec["pipeline"],
        matrix,
        y,
        cv,
        group_labels,
        selected_params,
    )
    save_n_vs_spearman_combined(
        output_dir / "n_vs_spearman_combined.png",
        n_sweep,
        f"{experiment_name} | n vs Spearman | direction-based + regression hyper-plane",
    )
    write_n_sweep_output(output_dir, n_sweep, n_sweep_extra, removed_outliers)

    print("[5/5] done")
    print(output_dir)
    return output_dir
