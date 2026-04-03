from pathlib import Path
import os
import sys


sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.common.aggregate import (
    read_tsv,
    run_script_and_capture_output_dir,
    to_float,
)
from experiments.common.folder_handling import create_folder
from experiments.common.plots import (
    save_heatmap,
    save_image_matrix,
    save_multi_line_grid_chart,
)
from experiments.common.math1d import pearson, spearman
from experiments.common.tsv import write_tsv

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = REPO_ROOT / "outputs" / "experiments" / "pitch_axis"
NOTES_TSV = REPO_ROOT / "experiments" / "pitch_axis" / "notes.tsv"
DATASETS = ["pitch"]
MAPPINGS = ["raw", "linear", "log_linear", "exp_linear"]
BASE_ANCHOR_N = 10
ANCHOR_N_SWEEP = [1, 2, 4, 8, 16, 32]
BEST_METRIC = os.environ.get("BEST_METRIC", "spearman").strip().lower()
VALID_BEST_METRICS = {"spearman", "pearson", "r2", "l1"}
if BEST_METRIC not in VALID_BEST_METRICS:
    raise ValueError(f"BEST_METRIC must be one of {sorted(VALID_BEST_METRICS)}")
PITCH_COVARIATE_COLUMNS = ["duration_ticks", "velocity"]

RUN_SPECS = [
    {"script": "run_base.py", "anchor": "first_vs_last", "label": "first_last"},
    {"script": "run_random_median.py", "anchor": "random_median", "label": "random"},
    {"script": "run_best_pairs.py", "anchor": "best_pairs", "label": "pairs"},
    {
        "script": "run_random_sample_first_last.py",
        "anchor": "random_sample_first_last",
        "label": "random_sample_first_last",
    },
    {
        "script": "run_random_sample_regression.py",
        "anchor": "random_sample_regression",
        "label": "random_sample_regression",
    },
]


def load_mode_result(spec, output_dir):
    rows = read_tsv(output_dir / "metrics.tsv")
    return {**spec, "output_dir": output_dir, "rows": rows}


def row_metric(row, metric_name=BEST_METRIC):
    return to_float(row[metric_name])


def metric_maximize(metric_name):
    return metric_name != "l1"


def mode_score(rows, combo):
    values = [
        row_metric(row)
        for row in rows
        if row["combo"] == combo and not np.isnan(row_metric(row))
    ]
    if not values:
        return float("-inf") if metric_maximize(BEST_METRIC) else float("inf")
    return max(values) if metric_maximize(BEST_METRIC) else min(values)


def best_rows(mode_result):
    combos = sorted({row["combo"] for row in mode_result["rows"]})
    best_combo = (
        max(combos, key=lambda combo: mode_score(mode_result["rows"], combo))
        if metric_maximize(BEST_METRIC)
        else min(combos, key=lambda combo: mode_score(mode_result["rows"], combo))
    )
    return {
        mapping: next(
            row
            for row in mode_result["rows"]
            if row["combo"] == best_combo and row["mapping"] == mapping
        )
        for mapping in MAPPINGS
    }


def row_spearman(row):
    return to_float(row["spearman"])


def row_l1(row):
    return to_float(row["l1"])


def row_l1_ratio(row):
    l1_value = to_float(row["l1"])
    random_value = to_float(row["random"])
    if np.isnan(l1_value) or np.isnan(random_value) or random_value <= 1e-12:
        return float("nan")
    return l1_value / random_value


def cell_label(row):
    best_value = row_metric(row)
    best_text = "nan" if np.isnan(best_value) else f"{best_value:.3f}"
    return (
        f"best({BEST_METRIC})={best_text}\n"
        f"rho={to_float(row['spearman']):.3f}+/-{to_float(row.get('spearman_ci95', 'nan')):.3f}"
        f"\nl1={to_float(row['l1']):.3f}+/-{to_float(row.get('l1_ci95', 'nan')):.3f}"
    )


def best_value(rows, metric_name, maximize):
    if metric_name == "l1_ratio":
        values = [row_l1_ratio(row) for row in rows if not np.isnan(row_l1_ratio(row))]
    else:
        values = [
            to_float(row[metric_name])
            for row in rows
            if not np.isnan(to_float(row[metric_name]))
        ]
    if not values:
        return float("nan")
    return max(values) if maximize else min(values)


def dataset_mapping_image_matrix(mode_results):
    images = [[]]
    texts = [[]]
    for mapping in MAPPINGS:
        candidates = [
            (row_metric(mode["best_rows"][mapping]), mode)
            for mode in mode_results
            if not np.isnan(row_metric(mode["best_rows"][mapping]))
        ]
        if candidates:
            best_mode = (
                max(candidates, key=lambda item: item[0])[1]
                if metric_maximize(BEST_METRIC)
                else min(candidates, key=lambda item: item[0])[1]
            )
        else:
            best_mode = None
        images[0].append(
            None
            if best_mode is None
            else best_mode["output_dir"] / f"scatter_pitch_{mapping}.png"
        )
        texts[0].append(
            "" if best_mode is None else cell_label(best_mode["best_rows"][mapping])
        )
    return list(MAPPINGS), images, texts


def dataset_mapping_value_matrix(mode_results, metric_name, maximize):
    values = [
        [
            best_value(
                [mode["best_rows"][mapping] for mode in mode_results],
                metric_name,
                maximize,
            )
            for mapping in MAPPINGS
        ]
    ]
    return list(MAPPINGS), np.asarray(values, dtype=np.float64)


def anchor_dataset_image_matrix(mode_results):
    images = [[]]
    texts = [[]]
    for mode in mode_results:
        candidate_mappings = [
            mapping
            for mapping in MAPPINGS
            if not np.isnan(row_metric(mode["best_rows"][mapping]))
        ]
        if not candidate_mappings:
            images[0].append(None)
            texts[0].append("")
            continue
        best_mapping = (
            max(
                candidate_mappings,
                key=lambda mapping: row_metric(mode["best_rows"][mapping]),
            )
            if metric_maximize(BEST_METRIC)
            else min(
                candidate_mappings,
                key=lambda mapping: row_metric(mode["best_rows"][mapping]),
            )
        )
        images[0].append(mode["output_dir"] / f"scatter_pitch_{best_mapping}.png")
        texts[0].append(cell_label(mode["best_rows"][best_mapping]))
    return [mode["label"] for mode in mode_results], images, texts


def anchor_dataset_value_matrix(mode_results, metric_name, maximize):
    values = [
        [
            best_value(
                [mode["best_rows"][mapping] for mapping in MAPPINGS],
                metric_name,
                maximize,
            )
            for mode in mode_results
        ]
    ]
    return [mode["label"] for mode in mode_results], np.asarray(
        values, dtype=np.float64
    )


def dataset_transformation_image_matrix(mode_results):
    columns = sorted({row["combo"] for mode in mode_results for row in mode["rows"]})
    images = [[]]
    texts = [[]]
    for combo in columns:
        candidates = []
        for mode in mode_results:
            for row in mode["rows"]:
                if row["combo"] == combo:
                    value = row_metric(row)
                    if not np.isnan(value):
                        candidates.append((value, mode, row))
        if candidates:
            best = (
                max(candidates, key=lambda item: item[0])
                if metric_maximize(BEST_METRIC)
                else min(candidates, key=lambda item: item[0])
            )
        else:
            best = None
        if best is None:
            images[0].append(None)
            texts[0].append("")
            continue
        _, mode, row = best
        images[0].append(
            mode["output_dir"]
            / "scatters_by_combo"
            / f"scatter_pitch__{combo}__{row['mapping']}.png"
        )
        texts[0].append(cell_label(row))
    return columns, images, texts


def dataset_transformation_value_matrix(mode_results, metric_name, maximize):
    columns = sorted({row["combo"] for mode in mode_results for row in mode["rows"]})
    values = [
        [
            best_value(
                [
                    row
                    for mode in mode_results
                    for row in mode["rows"]
                    if row["combo"] == combo
                ],
                metric_name,
                maximize,
            )
            for combo in columns
        ]
    ]
    return columns, np.asarray(values, dtype=np.float64)


def run_modes(anchor_n):
    mode_results = []
    for spec in RUN_SPECS:
        script_path = REPO_ROOT / "experiments" / "pitch_axis" / spec["script"]
        run_output_dir = run_script_and_capture_output_dir(
            script_path,
            OUTPUT_ROOT,
            extra_env={"ANCHOR_N": str(anchor_n), "BEST_METRIC": BEST_METRIC},
        )
        mode_result = load_mode_result({**spec, "anchor_n": anchor_n}, run_output_dir)
        mode_result["best_rows"] = best_rows(mode_result)
        mode_results.append(mode_result)
    return mode_results


def best_combo_mapping(mode_results_by_n):
    candidates = []
    for anchor_n in ANCHOR_N_SWEEP:
        for mode in mode_results_by_n[anchor_n]:
            for row in mode["rows"]:
                value = row_metric(row)
                if np.isnan(value):
                    continue
                candidates.append((value, row["combo"], row["mapping"]))
    if not candidates:
        return None
    chooser = max if metric_maximize(BEST_METRIC) else min
    _, combo, mapping = chooser(candidates, key=lambda item: item[0])
    return {"combo": combo, "mapping": mapping}


def anchor_n_curves(target_combo, target_mapping, mode_results_by_n, metric_name):
    series = []
    for spec in RUN_SPECS:
        y_values = []
        for anchor_n in ANCHOR_N_SWEEP:
            mode = next(
                mode
                for mode in mode_results_by_n[anchor_n]
                if mode["anchor"] == spec["anchor"]
            )
            candidates = [
                row
                for row in mode["rows"]
                if row["combo"] == target_combo and row["mapping"] == target_mapping
            ]
            if not candidates:
                y_values.append(float("nan"))
            else:
                y_values.append(to_float(candidates[0][metric_name]))
        series.append((spec["label"], y_values))
    return series


def best_combo_mapping_anchor(mode_results_by_n):
    candidates = []
    for anchor_n in ANCHOR_N_SWEEP:
        for mode in mode_results_by_n[anchor_n]:
            for row in mode["rows"]:
                value = row_metric(row)
                if np.isnan(value):
                    continue
                candidates.append((value, anchor_n, mode, row))
    if not candidates:
        return None
    chooser = max if metric_maximize(BEST_METRIC) else min
    score, anchor_n, mode, row = chooser(candidates, key=lambda item: item[0])
    return {
        "anchor_n": anchor_n,
        "anchor": mode["anchor"],
        "combo": row["combo"],
        "mapping": row["mapping"],
        "best_metric": score,
        "mode_output_dir": mode["output_dir"],
    }


def covariate_correlation_matrices(output_dir, mode_results_by_n):
    best = best_combo_mapping_anchor(mode_results_by_n)
    if best is None:
        return
    notes = read_tsv(NOTES_TSV)
    note_by_query_id = {row["query_id"]: row for row in notes}
    projection_rows = read_tsv(best["mode_output_dir"] / "projections.tsv")
    selected = [
        row
        for row in projection_rows
        if row["dataset"] == "pitch"
        and row["mapping"] == "raw"
        and row["combo"] == best["combo"]
    ]
    projected = np.asarray(
        [to_float(row["predicted"]) for row in selected], dtype=np.float64
    )
    target = np.asarray([to_float(row["real"]) for row in selected], dtype=np.float64)

    projection_pearson_row = []
    projection_spearman_row = []
    target_pearson_row = []
    target_spearman_row = []
    table_rows = []
    for covariate in PITCH_COVARIATE_COLUMNS:
        covariate_values = np.asarray(
            [
                to_float(note_by_query_id[row["query_id"]].get(covariate, "nan"))
                for row in selected
            ],
            dtype=np.float64,
        )
        projection_mask = (~np.isnan(projected)) & (~np.isnan(covariate_values))
        target_mask = (~np.isnan(target)) & (~np.isnan(covariate_values))
        projection_p_value = pearson(
            projected[projection_mask], covariate_values[projection_mask]
        )
        projection_s_value = spearman(
            projected[projection_mask], covariate_values[projection_mask]
        )
        target_p_value = pearson(target[target_mask], covariate_values[target_mask])
        target_s_value = spearman(target[target_mask], covariate_values[target_mask])
        projection_pearson_row.append(projection_p_value)
        projection_spearman_row.append(projection_s_value)
        target_pearson_row.append(target_p_value)
        target_spearman_row.append(target_s_value)
        table_rows.append(
            {
                "dataset": "pitch",
                "anchor": best["anchor"],
                "anchor_n": best["anchor_n"],
                "combo": best["combo"],
                "mapping": best["mapping"],
                "projection_mapping": "raw",
                "best_metric_name": BEST_METRIC,
                "best_metric": f"{best['best_metric']:.10f}",
                "covariate": covariate,
                "projection_pearson": "nan"
                if np.isnan(projection_p_value)
                else f"{projection_p_value:.10f}",
                "projection_spearman": "nan"
                if np.isnan(projection_s_value)
                else f"{projection_s_value:.10f}",
                "target_pearson": "nan"
                if np.isnan(target_p_value)
                else f"{target_p_value:.10f}",
                "target_spearman": "nan"
                if np.isnan(target_s_value)
                else f"{target_s_value:.10f}",
                "projection_count": int(np.sum(projection_mask)),
                "target_count": int(np.sum(target_mask)),
            }
        )

    write_tsv(
        output_dir / "best_strategy_covariate_correlations.tsv",
        table_rows,
        [
            "dataset",
            "anchor",
            "anchor_n",
            "combo",
            "mapping",
            "projection_mapping",
            "best_metric_name",
            "best_metric",
            "covariate",
            "projection_pearson",
            "projection_spearman",
            "target_pearson",
            "target_spearman",
            "projection_count",
            "target_count",
        ],
    )
    projection_pearson_matrix = np.asarray([projection_pearson_row], dtype=np.float64)
    projection_spearman_matrix = np.asarray([projection_spearman_row], dtype=np.float64)
    target_pearson_matrix = np.asarray([target_pearson_row], dtype=np.float64)
    target_spearman_matrix = np.asarray([target_spearman_row], dtype=np.float64)

    save_heatmap(
        output_dir / "best_strategy_covariate_pearson_heatmap.png",
        DATASETS,
        PITCH_COVARIATE_COLUMNS,
        projection_pearson_matrix,
        "Pitch best combo/anchor x covariates vs projection | Pearson",
        cmap="coolwarm",
    )
    save_heatmap(
        output_dir / "best_strategy_covariate_spearman_heatmap.png",
        DATASETS,
        PITCH_COVARIATE_COLUMNS,
        projection_spearman_matrix,
        "Pitch best combo/anchor x covariates vs projection | Spearman",
        cmap="coolwarm",
    )
    save_heatmap(
        output_dir / "best_strategy_main_target_covariate_pearson_heatmap.png",
        DATASETS,
        PITCH_COVARIATE_COLUMNS,
        target_pearson_matrix,
        "Pitch best combo/anchor x covariates vs pitch | Pearson",
        cmap="coolwarm",
    )
    save_heatmap(
        output_dir / "best_strategy_main_target_covariate_spearman_heatmap.png",
        DATASETS,
        PITCH_COVARIATE_COLUMNS,
        target_spearman_matrix,
        "Pitch best combo/anchor x covariates vs pitch | Spearman",
        cmap="coolwarm",
    )
    save_heatmap(
        output_dir / "best_strategy_covariate_pearson_fuzzy_masked_heatmap.png",
        DATASETS,
        PITCH_COVARIATE_COLUMNS,
        (1.0 - np.abs(target_pearson_matrix)) * projection_pearson_matrix,
        "Pitch covariates vs projection fuzzy-masked by pitch | Pearson",
        cmap="coolwarm",
    )
    save_heatmap(
        output_dir / "best_strategy_covariate_spearman_fuzzy_masked_heatmap.png",
        DATASETS,
        PITCH_COVARIATE_COLUMNS,
        (1.0 - np.abs(target_spearman_matrix)) * projection_spearman_matrix,
        "Pitch covariates vs projection fuzzy-masked by pitch | Spearman",
        cmap="coolwarm",
    )


def main():
    output_dir = create_folder(
        OUTPUT_ROOT,
        {
            "aggregate": True,
            "scripts": [spec["script"] for spec in RUN_SPECS],
            "base_anchor_n": BASE_ANCHOR_N,
            "anchor_n_sweep": ANCHOR_N_SWEEP,
            "best_metric": BEST_METRIC,
        },
    )
    mode_results = run_modes(BASE_ANCHOR_N)
    mode_results_by_n = {anchor_n: run_modes(anchor_n) for anchor_n in ANCHOR_N_SWEEP}

    mapping_columns, dataset_mapping_images, dataset_mapping_texts = (
        dataset_mapping_image_matrix(mode_results)
    )
    _, dataset_mapping_l1 = dataset_mapping_value_matrix(
        mode_results, "l1_ratio", False
    )
    _, dataset_mapping_spearman = dataset_mapping_value_matrix(
        mode_results, "spearman", True
    )
    anchor_columns, anchor_dataset_images, anchor_dataset_texts = (
        anchor_dataset_image_matrix(mode_results)
    )
    _, anchor_dataset_l1 = anchor_dataset_value_matrix(mode_results, "l1_ratio", False)
    _, anchor_dataset_spearman = anchor_dataset_value_matrix(
        mode_results, "spearman", True
    )
    transformation_columns, transformation_images, transformation_texts = (
        dataset_transformation_image_matrix(mode_results)
    )
    _, transformation_l1 = dataset_transformation_value_matrix(
        mode_results, "l1_ratio", False
    )
    _, transformation_spearman = dataset_transformation_value_matrix(
        mode_results, "spearman", True
    )

    save_heatmap(
        output_dir / "dataset_mapping_l1_heatmap.png",
        DATASETS,
        mapping_columns,
        dataset_mapping_l1,
        "Pitch dataset x mapping | best anchor by cell | l1 ratio (inference over random guessing)",
        cmap="magma_r",
    )
    save_heatmap(
        output_dir / "dataset_mapping_spearman_heatmap.png",
        DATASETS,
        mapping_columns,
        dataset_mapping_spearman,
        "Pitch dataset x mapping | best anchor by cell | Spearman",
    )
    save_heatmap(
        output_dir / "dataset_anchor_l1_heatmap.png",
        DATASETS,
        anchor_columns,
        anchor_dataset_l1,
        "Pitch dataset x anchor mode | best mapping by cell | l1 ratio (inference over random guessing)",
        cmap="magma_r",
    )
    save_heatmap(
        output_dir / "dataset_anchor_spearman_heatmap.png",
        DATASETS,
        anchor_columns,
        anchor_dataset_spearman,
        "Pitch dataset x anchor mode | best mapping by cell | Spearman",
    )
    save_heatmap(
        output_dir / "dataset_transformation_l1_heatmap.png",
        DATASETS,
        transformation_columns,
        transformation_l1,
        "Pitch dataset x transformation | best mode by cell | l1 ratio (inference over random guessing)",
        cmap="magma_r",
    )
    save_heatmap(
        output_dir / "dataset_transformation_spearman_heatmap.png",
        DATASETS,
        transformation_columns,
        transformation_spearman,
        "Pitch dataset x transformation | best mode by cell | Spearman",
    )

    save_image_matrix(
        output_dir / "dataset_mapping_spearman_matrix.png",
        dataset_mapping_images,
        DATASETS,
        mapping_columns,
        f"Pitch dataset x mapping | best anchor by {BEST_METRIC}",
        cell_texts=dataset_mapping_texts,
    )
    save_image_matrix(
        output_dir / "dataset_anchor_spearman_matrix.png",
        anchor_dataset_images,
        DATASETS,
        anchor_columns,
        f"Pitch dataset x anchor mode | best mapping by {BEST_METRIC}",
        cell_texts=anchor_dataset_texts,
    )
    save_image_matrix(
        output_dir / "dataset_transformation_spearman_matrix.png",
        transformation_images,
        DATASETS,
        transformation_columns,
        f"Pitch dataset x transformation | best mode by {BEST_METRIC}",
        cell_texts=transformation_texts,
    )

    target = best_combo_mapping(mode_results_by_n)
    if target is not None:
        panel_title = f"pitch | combo={target['combo']} | mapping={target['mapping']}"
        save_multi_line_grid_chart(
            output_dir / "anchor_n_l1_grid.png",
            [
                (
                    panel_title,
                    ANCHOR_N_SWEEP,
                    anchor_n_curves(
                        target["combo"],
                        target["mapping"],
                        mode_results_by_n,
                        "l1",
                    ),
                )
            ],
            ncols=1,
            title="L1 vs anchor n by dataset and method",
            xlabel="Anchor n",
            ylabel="L1",
        )
        save_multi_line_grid_chart(
            output_dir / "anchor_n_spearman_grid.png",
            [
                (
                    panel_title,
                    ANCHOR_N_SWEEP,
                    anchor_n_curves(
                        target["combo"],
                        target["mapping"],
                        mode_results_by_n,
                        "spearman",
                    ),
                )
            ],
            ncols=1,
            title="Spearman vs anchor n by dataset and method",
            xlabel="Anchor n",
            ylabel="Spearman",
        )
    covariate_correlation_matrices(output_dir, mode_results_by_n)
    print(output_dir)


if __name__ == "__main__":
    main()
