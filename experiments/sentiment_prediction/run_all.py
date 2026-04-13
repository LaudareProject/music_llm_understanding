from __future__ import annotations
from pathlib import Path
import sys
import os

import numpy as np
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import KFold


sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments import settings


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
from experiments.common.runner_data import prepare_rows_and_midi
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


def _mean_prediction(y_pred):
    y_array = np.asarray(y_pred, dtype=np.float64)
    if y_array.ndim == 1:
        return y_array
    return np.mean(y_array, axis=0)


def _class_label(value):
    return "+1" if int(value) == 1 else "-1"


def run_sentiment_prediction(settings, experiment_name="sentiment_prediction"):
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
            "cv_folds": int(spec.get("cv_folds", 10)),
            "source_model": "valence_axis",
        },
    )
    cache_dir = cache_dir_for_experiment(repo_root, experiment_name)

    print("[1/4] preparing rows, midi, embeddings")
    rows, midi_paths = prepare_rows_and_midi(
        experiment_name,
        spec["tsv_path"],
        repo_root,
        output_dir,
        row_filter_col=spec.get("row_filter_col"),
        row_filter_value=spec.get("row_filter_value"),
    )
    target_col = spec["target_col"]

    filtered_rows = []
    for row in rows:
        try:
            float(row[target_col])
        except Exception:
            continue
        filtered_rows.append(row)
    rows = filtered_rows

    if len(rows) < int(spec.get("cv_folds", 10)):
        raise ValueError(
            f"Need at least {int(spec.get('cv_folds', 10))} rows for 10-fold CV; got {len(rows)}"
        )

    matrix = matrix_for_rows(rows, midi_paths, cache_dir, repo_root, "Embeddings")
    y = np.asarray([float(row[target_col]) for row in rows], dtype=np.float64)
    y_class = (y >= 0.0).astype(np.int64)

    group_labels = None
    if spec.get("group_col") is not None:
        group_labels = np.asarray(
            [row[spec["group_col"]] for row in rows], dtype=object
        )

    print("[2/4] running 10-fold valence cross-validation")
    cv_folds = int(spec.get("cv_folds", 10))
    cv = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
    y_pred_oof = np.full(len(rows), np.nan, dtype=np.float64)
    y_class_pred_oof = np.full(len(rows), np.nan, dtype=np.float64)
    y_class_proba_oof = np.full(len(rows), np.nan, dtype=np.float64)
    fold_ids = np.full(len(rows), -1, dtype=np.int64)

    for fold_idx, (train_idx, test_idx) in enumerate(
        cv.split(matrix, y_class), start=1
    ):
        model = clone(spec["pipeline"])

        if group_labels is None:
            model.fit(matrix[train_idx], y[train_idx])
        else:
            model.fit(
                matrix[train_idx],
                y[train_idx],
                group_labels=group_labels[train_idx],
            )

        y_train_pred = _mean_prediction(model.predict(matrix[train_idx])).reshape(-1, 1)
        y_test_pred = _mean_prediction(model.predict(matrix[test_idx])).reshape(-1, 1)
        classifier = LogisticRegression(
            max_iter=2000,
            random_state=42,
            class_weight="balanced",
            solver="lbfgs",
        )
        classifier.fit(y_train_pred, y_class[train_idx])

        y_pred_oof[test_idx] = y_test_pred.ravel()
        y_class_pred_oof[test_idx] = classifier.predict(y_test_pred).astype(np.float64)
        y_class_proba_oof[test_idx] = classifier.predict_proba(y_test_pred)[:, 1]
        fold_ids[test_idx] = fold_idx

    print("[3/4] computing classes and accuracy")
    class_order = ["+1", "-1"]
    prediction_rows = []
    confusion = {(truth, pred): 0 for truth in class_order for pred in class_order}

    correct = 0
    for i, row in enumerate(rows):
        true_class = _class_label(y_class[i])
        pred_class = _class_label(int(y_class_pred_oof[i]))
        is_correct = int(true_class == pred_class)
        correct += is_correct
        confusion[(true_class, pred_class)] = (
            confusion.get((true_class, pred_class), 0) + 1
        )
        prediction_rows.append(
            {
                "query_id": row["query_id"],
                "fold": str(int(fold_ids[i])),
                "target_true": f"{y[i]:.10f}",
                "target_pred": f"{y_pred_oof[i]:.10f}",
                "class_true_numeric": str(int(y_class[i])),
                "class_pred_numeric": str(int(y_class_pred_oof[i])),
                "class_pred_proba": f"{y_class_proba_oof[i]:.10f}",
                "class_true": true_class,
                "class_pred": pred_class,
                "correct": str(is_correct),
            }
        )

    y_pred_labels = y_class_pred_oof.astype(np.int64)
    accuracy = float(accuracy_score(y_class, y_pred_labels))
    balanced_accuracy = float(balanced_accuracy_score(y_class, y_pred_labels))
    print(f"  accuracy={accuracy:.4f}")
    print(f"  balanced_accuracy={balanced_accuracy:.4f}")

    per_class_rows = []
    for class_label in class_order:
        support = sum(1 for row in prediction_rows if row["class_true"] == class_label)
        if support == 0:
            class_acc = float("nan")
        else:
            class_correct = sum(
                int(row["correct"])
                for row in prediction_rows
                if row["class_true"] == class_label
            )
            class_acc = float(class_correct) / float(support)
        per_class_rows.append(
            {
                "metric": f"accuracy_true_class_{class_label}",
                "value": f"{class_acc:.10f}",
                "support": str(int(support)),
            }
        )

    confusion_rows = []
    for true_class in class_order:
        for pred_class in class_order:
            confusion_rows.append(
                {
                    "class_true": true_class,
                    "class_pred": pred_class,
                    "count": str(int(confusion.get((true_class, pred_class), 0))),
                }
            )

    print("[4/4] writing outputs")
    write_tsv(
        output_dir / "metrics.tsv",
        [
            {
                "metric": "class_accuracy",
                "value": f"{accuracy:.10f}",
                "support": str(len(rows)),
            },
            {
                "metric": "balanced_class_accuracy",
                "value": f"{balanced_accuracy:.10f}",
                "support": str(len(rows)),
            },
        ]
        + per_class_rows,
        ["metric", "value", "support"],
    )
    write_tsv(
        output_dir / "sentiment_predictions.tsv",
        prediction_rows,
        [
            "query_id",
            "fold",
            "target_true",
            "target_pred",
            "class_true_numeric",
            "class_pred_numeric",
            "class_pred_proba",
            "class_true",
            "class_pred",
            "correct",
        ],
    )
    write_tsv(
        output_dir / "class_confusion.tsv",
        confusion_rows,
        ["class_true", "class_pred", "count"],
    )

    print(output_dir)
    return output_dir


if __name__ == "__main__":
    run_sentiment_prediction(settings, "sentiment_prediction")
