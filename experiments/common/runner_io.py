from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from experiments.common.tsv import write_tsv


def write_prediction_outputs(
    output_dir, rows, y, y_hat_all, anchor_mask, removed_outliers
):
    projection_rows = []
    for i, row in enumerate(rows):
        projection_rows.append(
            {
                "query_id": row["query_id"],
                "target": f"{y[i]:.10f}",
                "predicted": f"{y_hat_all[i]:.10f}",
                "is_anchor": "1" if anchor_mask[i] else "0",
                "outliers_removed": str(removed_outliers),
            }
        )
    write_tsv(
        output_dir / "projections.tsv",
        projection_rows,
        ["query_id", "target", "predicted", "is_anchor", "outliers_removed"],
    )


def write_metrics_output(
    output_dir, score_mean, score_ci95, removed_outliers, total_rows, kept_rows
):
    write_tsv(
        output_dir / "metrics.tsv",
        [
            {
                "metric": "spearman",
                "mean": f"{score_mean:.10f}",
                "ci95": f"{score_ci95:.10f}",
                "folds": "5",
                "outliers_removed": str(removed_outliers),
                "rows_total": str(total_rows),
                "rows_kept": str(kept_rows),
            }
        ],
        [
            "metric",
            "mean",
            "ci95",
            "folds",
            "outliers_removed",
            "rows_total",
            "rows_kept",
        ],
    )


def write_n_sweep_output(output_dir, n_sweep, n_sweep_extra, removed_outliers):
    n_rows = []
    for projection_regression, by_method in n_sweep.items():
        for anchor_method, sweep_values in by_method.items():
            for n, s, c in zip(
                sweep_values["n_values"],
                sweep_values["scores"],
                sweep_values["ci95"],
            ):
                extra = n_sweep_extra.get(
                    (projection_regression, anchor_method, int(n)), {}
                )
                n_rows.append(
                    {
                        "projection_regression": "1" if projection_regression else "0",
                        "anchor_method": anchor_method,
                        "n": str(n),
                        "spearman": f"{s:.10f}",
                        "ci95": f"{c:.10f}",
                        "outliers_removed": str(removed_outliers),
                        "r2": f"{extra.get('r2', float('nan')):.10f}",
                        "r2_ci95": f"{extra.get('r2_ci95', float('nan')):.10f}",
                        "l1": f"{extra.get('l1', float('nan')):.10f}",
                        "l1_ci95": f"{extra.get('l1_ci95', float('nan')):.10f}",
                    }
                )
    write_tsv(
        output_dir / "n_vs_spearman.tsv",
        n_rows,
        [
            "projection_regression",
            "anchor_method",
            "n",
            "spearman",
            "ci95",
            "outliers_removed",
            "r2",
            "r2_ci95",
            "l1",
            "l1_ci95",
        ],
    )


def write_gridsearch_csv(path, grid_search):
    def _export_param_value(value):
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, dict):
            return {str(k): _export_param_value(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [_export_param_value(v) for v in value]
        if hasattr(value, "get_params"):
            return type(value).__name__
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        return str(value)

    def _format_param_value(value):
        if isinstance(value, str):
            return value
        if isinstance(value, list):
            return "[" + ", ".join(_format_param_value(v) for v in value) + "]"
        if isinstance(value, dict):
            parts = []
            for k in sorted(value.keys()):
                parts.append(f"{k}:{_format_param_value(value[k])}")
            return "{" + ", ".join(parts) + "}"
        return str(value)

    rows = []
    for i in range(len(grid_search.cv_results_["params"])):
        params = grid_search.cv_results_["params"][i]
        params_text = "; ".join(
            f"{key}={_format_param_value(_export_param_value(params[key]))}"
            for key in sorted(params.keys())
        )
        rows.append(
            {
                "rank_test_score": int(grid_search.cv_results_["rank_test_score"][i]),
                "mean_test_score": float(grid_search.cv_results_["mean_test_score"][i]),
                "std_test_score": float(grid_search.cv_results_["std_test_score"][i]),
                "mean_train_score": float(
                    grid_search.cv_results_["mean_train_score"][i]
                ),
                "std_train_score": float(grid_search.cv_results_["std_train_score"][i]),
                "params": params_text,
            }
        )

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
