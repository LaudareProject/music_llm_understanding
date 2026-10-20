from __future__ import annotations

import numpy as np
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error, r2_score

from experiments.common.runner_stats import mean_ci95, spearman_metric


def crossval_oof_predictions(estimator, matrix, y, cv, group_labels=None):
    n_samples = int(matrix.shape[0])
    oof_pred = np.full(n_samples, np.nan, dtype=np.float64)
    anchor_counts = np.zeros(n_samples, dtype=np.int64)
    trial_scores = []
    trial_r2 = []
    trial_l1 = []

    for train_idx, test_idx in cv.split(matrix, y):
        model = clone(estimator)
        if group_labels is None:
            model.fit(matrix[train_idx], y[train_idx])
        else:
            model.fit(
                matrix[train_idx],
                y[train_idx],
                group_labels=np.asarray(group_labels)[train_idx],
            )
        fold_pred = np.asarray(model.predict(matrix[test_idx]), dtype=np.float64)
        if fold_pred.ndim == 1:
            fold_pred = fold_pred.reshape(1, -1)
        oof_pred[test_idx] = np.mean(fold_pred, axis=0)

        for trial_pred in fold_pred:
            trial_scores.append(float(spearman_metric(y[test_idx], trial_pred)))
            try:
                trial_r2.append(float(r2_score(y[test_idx], trial_pred)))
            except Exception:
                trial_r2.append(float("nan"))
            try:
                trial_l1.append(float(mean_absolute_error(y[test_idx], trial_pred)))
            except Exception:
                trial_l1.append(float("nan"))

        for metadata in model.trial_anchor_metadata_:
            local_anchor_idx = np.asarray(metadata["anchor_indices"], dtype=np.int64)
            if local_anchor_idx.size == 0:
                continue
            global_anchor_idx = np.asarray(train_idx, dtype=np.int64)[local_anchor_idx]
            anchor_counts[global_anchor_idx] += 1

    return oof_pred, anchor_counts, trial_scores, trial_r2, trial_l1


def _same_params_except_anchor_keys(candidate_params, best_params, ignored_keys):
    for key, best_value in best_params.items():
        if key in ignored_keys:
            continue
        if key not in candidate_params or candidate_params[key] != best_value:
            return False
    return True


def _params_equal(candidate_params, target_params):
    all_keys = set(candidate_params.keys()) | set(target_params.keys())
    for key in all_keys:
        if key not in candidate_params or key not in target_params:
            return False
        if candidate_params[key] != target_params[key]:
            return False
    return True


def anchor_method_label(method):
    labels = {
        "first_vs_last": "Deterministic extremes",
        "random_sample_first_last": "Random sample",
        "random_median": "Median random samples",
        "percentile_medians": "Percentile medians",
    }
    return labels.get(str(method), str(method))


def select_grid_params_with_min_n(grid, score_tolerance):
    cv_results = grid.cv_results_
    scores = np.asarray(cv_results["mean_test_score"], dtype=np.float64)
    params_list = cv_results["params"]

    best_idx = int(grid.best_index_)
    best_score = float(scores[best_idx])
    threshold = float(best_score - float(score_tolerance))

    candidate_indices = []
    for idx, score in enumerate(scores):
        if np.isfinite(score) and float(score) >= threshold:
            candidate_indices.append(idx)
    if not candidate_indices:
        return dict(grid.best_params_), best_score, best_score

    def _anchor_n_value(index):
        value = params_list[index].get("anchor__n")
        if value is None:
            return np.inf
        try:
            return int(value)
        except Exception:
            return np.inf

    selected_idx = min(
        candidate_indices,
        key=lambda idx: (
            _anchor_n_value(idx),
            -float(scores[idx]),
            int(cv_results["rank_test_score"][idx]),
        ),
    )
    return dict(params_list[selected_idx]), float(scores[selected_idx]), best_score


def n_sweep_from_grid(grid, reference_params=None):
    cv_results = grid.cv_results_
    split_keys = sorted(
        key
        for key in cv_results.keys()
        if key.startswith("split") and key.endswith("_test_score")
    )
    if reference_params is None:
        reference_params = grid.best_params_

    by_mode = {False: {}, True: {}}
    for idx, params in enumerate(cv_results["params"]):
        if not _same_params_except_anchor_keys(
            params,
            reference_params,
            ignored_keys={"anchor__n", "anchor__method", "projection__regression"},
        ):
            continue
        if "anchor__n" not in params or "anchor__method" not in params:
            continue
        split_scores = []
        for key in split_keys:
            value = float(cv_results[key][idx])
            if np.isfinite(value):
                split_scores.append(value)
        mean_score, ci95_score = mean_ci95(split_scores)
        projection_regression = bool(params.get("projection__regression", False))
        line_label = anchor_method_label(str(params["anchor__method"]))
        by_mode.setdefault(projection_regression, {}).setdefault(line_label, []).append(
            (int(params["anchor__n"]), mean_score, ci95_score)
        )

    sweep = {False: {}, True: {}}
    for projection_regression in [False, True]:
        for anchor_method in sorted(by_mode[projection_regression].keys()):
            points = sorted(
                by_mode[projection_regression][anchor_method], key=lambda row: row[0]
            )
            sweep[projection_regression][anchor_method] = {
                "n_values": [n for n, _, _ in points],
                "scores": [s for _, s, _ in points],
                "ci95": [c for _, _, c in points],
            }
    return sweep


def split_score_stats_for_params(grid, params):
    cv_results = grid.cv_results_
    params_list = cv_results["params"]
    split_keys = sorted(
        key
        for key in cv_results.keys()
        if key.startswith("split") and key.endswith("_test_score")
    )
    for idx, candidate in enumerate(params_list):
        if not _params_equal(candidate, params):
            continue
        split_scores = []
        for key in split_keys:
            value = float(cv_results[key][idx])
            if np.isfinite(value):
                split_scores.append(value)
        return mean_ci95(split_scores)
    return float("nan"), float("nan")


def n_sweep_extra_metrics(
    grid, base_pipeline, matrix, y, cv, group_labels, reference_params
):
    cv_results = grid.cv_results_
    extra = {}
    for params in cv_results["params"]:
        if not _same_params_except_anchor_keys(
            params,
            reference_params,
            ignored_keys={"anchor__n", "anchor__method", "projection__regression"},
        ):
            continue
        if "anchor__n" not in params or "anchor__method" not in params:
            continue

        key = (
            bool(params.get("projection__regression", False)),
            anchor_method_label(str(params["anchor__method"])),
            int(params["anchor__n"]),
        )
        estimator = clone(base_pipeline).set_params(**params)
        try:
            _, _, _, trial_r2, trial_l1 = crossval_oof_predictions(
                estimator,
                matrix,
                y,
                cv,
                group_labels=group_labels,
            )
            r2_mean, r2_ci95 = mean_ci95(trial_r2)
            l1_mean, l1_ci95 = mean_ci95(trial_l1)
            extra[key] = {
                "r2": float(r2_mean),
                "r2_ci95": float(r2_ci95),
                "l1": float(l1_mean),
                "l1_ci95": float(l1_ci95),
            }
        except Exception as exc:
            print(
                "  skipping extra metrics for sweep point: "
                f"anchor_method={params['anchor__method']}, "
                f"n={params['anchor__n']}, "
                f"projection_regression={params.get('projection__regression', False)} ({exc})"
            )
            extra[key] = {
                "r2": float("nan"),
                "r2_ci95": float("nan"),
                "l1": float("nan"),
                "l1_ci95": float("nan"),
            }
    return extra


def projection_results_from_grid(grid):
    cv_results = grid.cv_results_

    def _projection_label(params):
        projection = params.get("projection")
        projection_name = type(projection).__name__
        if projection_name == "FunctionTransformer":
            return "identity"
        elif projection_name == "Projection1D":
            return "project1d"
        else:
            return projection_name

    def _params_key_without_projection(params):
        key = []

        def _normalize(value):
            if isinstance(value, (str, int, float, bool, type(None))):
                return value
            if isinstance(value, np.generic):
                return value.item()
            if hasattr(value, "get_params"):
                estimator_params = value.get_params(deep=False)
                estimator_key = []
                for param_name in sorted(estimator_params.keys()):
                    estimator_key.append(
                        (param_name, repr(estimator_params[param_name]))
                    )
                return (type(value).__name__, tuple(estimator_key))
            return repr(value)

        for param_name in sorted(params.keys()):
            if param_name == "projection" or param_name.startswith("projection__"):
                continue
            key.append((param_name, _normalize(params[param_name])))
        return tuple(key)

    grouped_scores = {}
    for idx, params in enumerate(cv_results["params"]):
        value = float(cv_results["mean_test_score"][idx])
        if not np.isfinite(value):
            continue
        key = _params_key_without_projection(params)
        grouped_scores.setdefault(key, {})
        grouped_scores[key][_projection_label(params)] = float(value)

    identity_scores = []
    project1d_scores = []
    for projection_scores in grouped_scores.values():
        if "identity" not in projection_scores or "project1d" not in projection_scores:
            continue
        identity_scores.append(float(projection_scores["identity"]))
        project1d_scores.append(float(projection_scores["project1d"]))

    return {
        "identity_scores": identity_scores,
        "project1d_scores": project1d_scores,
    }


def default_anchor_config(estimator):
    params = estimator.get_params(deep=True)
    return str(params.get("anchor__method", "unknown")), params.get(
        "anchor__n", "unknown"
    )
