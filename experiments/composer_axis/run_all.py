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
    save_stacked_bar_chart,
)
from experiments.common.math1d import pearson, spearman
from experiments.common.tsv import write_tsv

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = REPO_ROOT / "outputs" / "experiments" / "composer_axis"
SONGS_TSV = REPO_ROOT / "experiments" / "composer_axis" / "songs.tsv"
DATASETS = ["asap", "music21", "ewld", "overall"]
MAPPINGS = ["raw", "linear", "log_linear", "exp_linear"]
BASE_ANCHOR_N = 10
ANCHOR_N_SWEEP = [1, 2, 4, 8, 16, 32]
BEST_METRIC = os.environ.get("BEST_METRIC", "spearman").strip().lower()
VALID_BEST_METRICS = {"spearman", "pearson", "r2", "l1"}
if BEST_METRIC not in VALID_BEST_METRICS:
    raise ValueError(f"BEST_METRIC must be one of {sorted(VALID_BEST_METRICS)}")
COMPOSER_COVARIATE_COLUMNS = [
    "midi_note_count",
    "midi_duration_seconds",
    "midi_track_count",
    "midi_pitch_q0",
    "midi_pitch_q25",
    "midi_pitch_q50",
    "midi_pitch_q75",
    "midi_pitch_q100",
    "midi_note_duration_q0",
    "midi_note_duration_q25",
    "midi_note_duration_q50",
    "midi_note_duration_q75",
    "midi_note_duration_q100",
]

RUN_SPECS = [
    {
        "script": "run_base.py",
        "anchor": "first_vs_last",
        "label": "first_last",
    },
    {
        "script": "run_random_median.py",
        "anchor": "random_median",
        "label": "random",
    },
    {
        "script": "run_best_pairs.py",
        "anchor": "best_pairs",
        "label": "pairs",
    },
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


def metric_better(metric_name, candidate, incumbent):
    if np.isnan(candidate):
        return False
    if incumbent is None:
        return True
    if metric_maximize(metric_name):
        return candidate > incumbent
    return candidate < incumbent


def best_mapping_for_rows(rows, metric_name=BEST_METRIC):
    best_mapping = None
    best_score = None
    for mapping in MAPPINGS:
        value = row_metric(rows[mapping], metric_name)
        if metric_better(metric_name, value, best_score):
            best_score = value
            best_mapping = mapping
    return best_mapping


def mode_score(rows, dataset, combo):
    values = [
        row_metric(row)
        for row in rows
        if row["dataset"] == dataset
        and row["combo"] == combo
        and not np.isnan(row_metric(row))
    ]
    if not values:
        return float("-inf") if metric_maximize(BEST_METRIC) else float("inf")
    return max(values) if metric_maximize(BEST_METRIC) else min(values)


def best_rows_by_dataset(mode_result):
    result = {}
    combos = sorted({row["combo"] for row in mode_result["rows"]})
    for dataset in DATASETS:
        available = [
            combo
            for combo in combos
            if any(
                row["dataset"] == dataset and row["combo"] == combo
                for row in mode_result["rows"]
            )
        ]
        if not available:
            continue
        best_combo = (
            max(
                available,
                key=lambda combo: mode_score(mode_result["rows"], dataset, combo),
            )
            if metric_maximize(BEST_METRIC)
            else min(
                available,
                key=lambda combo: mode_score(mode_result["rows"], dataset, combo),
            )
        )
        result[dataset] = {
            mapping: next(
                row
                for row in mode_result["rows"]
                if row["dataset"] == dataset
                and row["combo"] == best_combo
                and row["mapping"] == mapping
            )
            for mapping in MAPPINGS
        }
    return result


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
    columns = [f"{mapping}" for mapping in MAPPINGS]
    images = []
    texts = []
    for dataset in DATASETS:
        row_images = []
        row_texts = []
        for mapping in MAPPINGS:
            candidates = []
            for mode in mode_results:
                dataset_rows = mode["best_rows_by_dataset"].get(dataset)
                if dataset_rows is None:
                    continue
                row = dataset_rows[mapping]
                value = row_metric(row)
                if not np.isnan(value):
                    candidates.append((value, mode))
            if candidates:
                best_mode = (
                    max(candidates, key=lambda item: item[0])[1]
                    if metric_maximize(BEST_METRIC)
                    else min(candidates, key=lambda item: item[0])[1]
                )
            else:
                best_mode = None
            row_images.append(
                None
                if best_mode is None
                else best_mode["output_dir"] / f"scatter_{dataset}_{mapping}.png"
            )
            row_texts.append(
                ""
                if best_mode is None
                else cell_label(best_mode["best_rows_by_dataset"][dataset][mapping])
            )
        images.append(row_images)
        texts.append(row_texts)
    return columns, images, texts


def dataset_mapping_value_matrix(mode_results, metric_name, maximize):
    columns = [f"{mapping}" for mapping in MAPPINGS]
    values = []
    for dataset in DATASETS:
        row_values = []
        for mapping in MAPPINGS:
            candidates = []
            for mode in mode_results:
                dataset_rows = mode["best_rows_by_dataset"].get(dataset)
                if dataset_rows is not None:
                    candidates.append(dataset_rows[mapping])
            row_values.append(best_value(candidates, metric_name, maximize))
        values.append(row_values)
    return columns, np.asarray(values, dtype=np.float64)


def anchor_dataset_image_matrix(mode_results):
    columns = [mode["label"] for mode in mode_results]
    images = []
    texts = []
    for dataset in DATASETS:
        row_images = []
        row_texts = []
        for mode in mode_results:
            dataset_rows = mode["best_rows_by_dataset"].get(dataset)
            if dataset_rows is None:
                row_images.append(None)
                row_texts.append("")
                continue
            candidate_mappings = [
                mapping
                for mapping in MAPPINGS
                if not np.isnan(row_metric(dataset_rows[mapping]))
            ]
            if not candidate_mappings:
                row_images.append(None)
                row_texts.append("")
                continue
            best_mapping = (
                max(
                    candidate_mappings,
                    key=lambda mapping: row_metric(dataset_rows[mapping]),
                )
                if metric_maximize(BEST_METRIC)
                else min(
                    candidate_mappings,
                    key=lambda mapping: row_metric(dataset_rows[mapping]),
                )
            )
            row_images.append(
                mode["output_dir"] / f"scatter_{dataset}_{best_mapping}.png"
            )
            row_texts.append(cell_label(dataset_rows[best_mapping]))
        images.append(row_images)
        texts.append(row_texts)
    return columns, images, texts


def anchor_dataset_value_matrix(mode_results, metric_name, maximize):
    columns = [mode["label"] for mode in mode_results]
    values = []
    for dataset in DATASETS:
        row_values = []
        for mode in mode_results:
            dataset_rows = mode["best_rows_by_dataset"].get(dataset)
            if dataset_rows is None:
                row_values.append(float("nan"))
                continue
            mapping_rows = [dataset_rows[mapping] for mapping in MAPPINGS]
            row_values.append(best_value(mapping_rows, metric_name, maximize))
        values.append(row_values)
    return columns, np.asarray(values, dtype=np.float64)


def dataset_transformation_image_matrix(mode_results):
    columns = sorted({row["combo"] for mode in mode_results for row in mode["rows"]})
    images = []
    texts = []
    for dataset in DATASETS:
        row_images = []
        row_texts = []
        for combo in columns:
            candidates = []
            for mode in mode_results:
                for row in mode["rows"]:
                    if row["dataset"] == dataset and row["combo"] == combo:
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
                row_images.append(None)
                row_texts.append("")
                continue
            _, mode, row = best
            row_images.append(
                mode["output_dir"]
                / "scatters_by_combo"
                / f"scatter_{dataset}__{combo}__{row['mapping']}.png"
            )
            row_texts.append(cell_label(row))
        images.append(row_images)
        texts.append(row_texts)
    return columns, images, texts


def dataset_transformation_value_matrix(mode_results, metric_name, maximize):
    columns = sorted({row["combo"] for mode in mode_results for row in mode["rows"]})
    values = []
    for dataset in DATASETS:
        row_values = []
        for combo in columns:
            candidates = [
                row
                for mode in mode_results
                for row in mode["rows"]
                if row["dataset"] == dataset and row["combo"] == combo
            ]
            row_values.append(best_value(candidates, metric_name, maximize))
        values.append(row_values)
    return columns, np.asarray(values, dtype=np.float64)


def run_modes(anchor_n):
    mode_results = []
    for spec in RUN_SPECS:
        script_path = REPO_ROOT / "experiments" / "composer_axis" / spec["script"]
        run_output_dir = run_script_and_capture_output_dir(
            script_path,
            OUTPUT_ROOT,
            extra_env={"ANCHOR_N": str(anchor_n), "BEST_METRIC": BEST_METRIC},
        )
        mode_result = load_mode_result({**spec, "anchor_n": anchor_n}, run_output_dir)
        mode_result["best_rows_by_dataset"] = best_rows_by_dataset(mode_result)
        mode_results.append(mode_result)
    return mode_results


def best_combo_mapping_by_dataset(mode_results_by_n):
    best_by_dataset = {}
    for dataset in DATASETS:
        candidates = []
        for anchor_n in ANCHOR_N_SWEEP:
            for mode in mode_results_by_n[anchor_n]:
                for row in mode["rows"]:
                    if row["dataset"] != dataset:
                        continue
                    value = row_metric(row)
                    if np.isnan(value):
                        continue
                    candidates.append((value, row["combo"], row["mapping"]))
        if candidates:
            chooser = max if metric_maximize(BEST_METRIC) else min
            _, combo, mapping = chooser(candidates, key=lambda item: item[0])
            best_by_dataset[dataset] = {"combo": combo, "mapping": mapping}
    return best_by_dataset


def anchor_n_curves_for_dataset(
    dataset, target_combo, target_mapping, mode_results_by_n, metric_name
):
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
                if row["dataset"] == dataset
                and row["combo"] == target_combo
                and row["mapping"] == target_mapping
            ]
            if not candidates:
                y_values.append(float("nan"))
            else:
                y_values.append(to_float(candidates[0][metric_name]))
        series.append((spec["label"], y_values))
    return series


def best_combo_mapping_anchor_by_dataset(mode_results_by_n):
    best_by_dataset = {}
    for dataset in DATASETS:
        candidates = []
        for anchor_n in ANCHOR_N_SWEEP:
            for mode in mode_results_by_n[anchor_n]:
                for row in mode["rows"]:
                    if row["dataset"] != dataset:
                        continue
                    value = row_metric(row)
                    if np.isnan(value):
                        continue
                    candidates.append((value, anchor_n, mode, row))
        if not candidates:
            continue
        chooser = max if metric_maximize(BEST_METRIC) else min
        score, anchor_n, mode, row = chooser(candidates, key=lambda item: item[0])
        best_by_dataset[dataset] = {
            "anchor_n": anchor_n,
            "anchor": mode["anchor"],
            "combo": row["combo"],
            "mapping": row["mapping"],
            "best_metric": score,
            "mode_output_dir": mode["output_dir"],
        }
    return best_by_dataset


def covariate_correlation_matrices(output_dir, mode_results_by_n):
    songs = read_tsv(SONGS_TSV)
    song_by_query_id = {row["query_id"]: row for row in songs}
    best_by_dataset = best_combo_mapping_anchor_by_dataset(mode_results_by_n)
    projection_pearson_values = []
    projection_spearman_values = []
    target_pearson_values = []
    target_spearman_values = []
    table_rows = []

    for dataset in DATASETS:
        best = best_by_dataset.get(dataset)
        if best is None:
            projection_pearson_values.append(
                [float("nan")] * len(COMPOSER_COVARIATE_COLUMNS)
            )
            projection_spearman_values.append(
                [float("nan")] * len(COMPOSER_COVARIATE_COLUMNS)
            )
            target_pearson_values.append(
                [float("nan")] * len(COMPOSER_COVARIATE_COLUMNS)
            )
            target_spearman_values.append(
                [float("nan")] * len(COMPOSER_COVARIATE_COLUMNS)
            )
            continue
        projection_rows = read_tsv(best["mode_output_dir"] / "projections.tsv")
        selected = [
            row
            for row in projection_rows
            if row["dataset"] == dataset
            and row["mapping"] == "raw"
            and row["combo"] == best["combo"]
        ]
        projected = np.asarray(
            [to_float(row["predicted"]) for row in selected], dtype=np.float64
        )
        target = np.asarray(
            [to_float(row["real"]) for row in selected], dtype=np.float64
        )
        dataset_projection_pearson = []
        dataset_projection_spearman = []
        dataset_target_pearson = []
        dataset_target_spearman = []
        for covariate in COMPOSER_COVARIATE_COLUMNS:
            covariate_values = np.asarray(
                [
                    to_float(song_by_query_id[row["query_id"]].get(covariate, "nan"))
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
            target_s_value = spearman(
                target[target_mask], covariate_values[target_mask]
            )
            dataset_projection_pearson.append(projection_p_value)
            dataset_projection_spearman.append(projection_s_value)
            dataset_target_pearson.append(target_p_value)
            dataset_target_spearman.append(target_s_value)
            table_rows.append(
                {
                    "dataset": dataset,
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
        projection_pearson_values.append(dataset_projection_pearson)
        projection_spearman_values.append(dataset_projection_spearman)
        target_pearson_values.append(dataset_target_pearson)
        target_spearman_values.append(dataset_target_spearman)

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
    projection_pearson_matrix = np.asarray(projection_pearson_values, dtype=np.float64)
    projection_spearman_matrix = np.asarray(
        projection_spearman_values, dtype=np.float64
    )
    target_pearson_matrix = np.asarray(target_pearson_values, dtype=np.float64)
    target_spearman_matrix = np.asarray(target_spearman_values, dtype=np.float64)

    save_heatmap(
        output_dir / "best_strategy_covariate_pearson_heatmap.png",
        DATASETS,
        COMPOSER_COVARIATE_COLUMNS,
        projection_pearson_matrix,
        "Composer best combo/anchor x covariates vs projection | Pearson",
        cmap="coolwarm",
    )
    save_heatmap(
        output_dir / "best_strategy_covariate_spearman_heatmap.png",
        DATASETS,
        COMPOSER_COVARIATE_COLUMNS,
        projection_spearman_matrix,
        "Composer best combo/anchor x covariates vs projection | Spearman",
        cmap="coolwarm",
    )
    save_heatmap(
        output_dir / "best_strategy_main_target_covariate_pearson_heatmap.png",
        DATASETS,
        COMPOSER_COVARIATE_COLUMNS,
        target_pearson_matrix,
        "Composer best combo/anchor x covariates vs year | Pearson",
        cmap="coolwarm",
    )
    save_heatmap(
        output_dir / "best_strategy_main_target_covariate_spearman_heatmap.png",
        DATASETS,
        COMPOSER_COVARIATE_COLUMNS,
        target_spearman_matrix,
        "Composer best combo/anchor x covariates vs year | Spearman",
        cmap="coolwarm",
    )
    save_heatmap(
        output_dir / "best_strategy_covariate_pearson_fuzzy_masked_heatmap.png",
        DATASETS,
        COMPOSER_COVARIATE_COLUMNS,
        (1.0 - target_pearson_matrix) * projection_pearson_matrix,
        "Composer covariates vs projection fuzzy-masked by year | Pearson",
        cmap="coolwarm",
    )
    save_heatmap(
        output_dir / "best_strategy_covariate_spearman_fuzzy_masked_heatmap.png",
        DATASETS,
        COMPOSER_COVARIATE_COLUMNS,
        (1.0 - target_spearman_matrix) * projection_spearman_matrix,
        "Composer covariates vs projection fuzzy-masked by year | Spearman",
        cmap="coolwarm",
    )


def year_source_group(source):
    if source.startswith("imslp_"):
        return "imslp"
    if source.startswith("wikipedia_"):
        return "wikipedia"
    if source.startswith("rism_"):
        return "rism"
    if source == "ewld_first_performance_date":
        return "ewld"
    return "lifespan"


def save_year_source_stacked_histograms(output_dir):
    rows = read_tsv(SONGS_TSV)
    for dataset in ["asap", "music21", "ewld"]:
        dataset_rows = [row for row in rows if row["dataset"] == dataset]
        composers = sorted({row["composer"] for row in dataset_rows})
        counts = {
            group: [] for group in ["imslp", "wikipedia", "rism", "ewld", "lifespan"]
        }
        for composer in composers:
            composer_rows = [row for row in dataset_rows if row["composer"] == composer]
            group_counts = {group: 0 for group in counts}
            for row in composer_rows:
                group = year_source_group(row["year_source"])
                group_counts[group] += 1
            for group in counts:
                if dataset != "ewld" and group == "ewld":
                    counts[group].append(0)
                else:
                    counts[group].append(group_counts[group])
        groups = ["imslp", "wikipedia", "rism", "ewld", "lifespan"]
        if dataset != "ewld":
            groups = ["imslp", "wikipedia", "rism", "lifespan"]
        series = [(group, counts[group]) for group in groups]
        save_stacked_bar_chart(
            output_dir / f"year_source_histogram_{dataset}.png",
            composers,
            series,
            f"{dataset} year sources by composer",
            "Count",
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
    save_year_source_stacked_histograms(output_dir)
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
        "Composer dataset x mapping | best anchor by cell | l1 ratio (inference over random guessing)",
        cmap="magma_r",
    )
    save_heatmap(
        output_dir / "dataset_mapping_spearman_heatmap.png",
        DATASETS,
        mapping_columns,
        dataset_mapping_spearman,
        "Composer dataset x mapping | best anchor by cell | Spearman",
    )
    save_heatmap(
        output_dir / "dataset_anchor_l1_heatmap.png",
        DATASETS,
        anchor_columns,
        anchor_dataset_l1,
        "Composer dataset x anchor mode | best mapping by cell | l1 ratio (inference over random guessing)",
        cmap="magma_r",
    )
    save_heatmap(
        output_dir / "dataset_anchor_spearman_heatmap.png",
        DATASETS,
        anchor_columns,
        anchor_dataset_spearman,
        "Composer dataset x anchor mode | best mapping by cell | Spearman",
    )
    save_heatmap(
        output_dir / "dataset_transformation_l1_heatmap.png",
        DATASETS,
        transformation_columns,
        transformation_l1,
        "Composer dataset x transformation | best mode by cell | l1 ratio (inference over random guessing)",
        cmap="magma_r",
    )
    save_heatmap(
        output_dir / "dataset_transformation_spearman_heatmap.png",
        DATASETS,
        transformation_columns,
        transformation_spearman,
        "Composer dataset x transformation | best mode by cell | Spearman",
    )

    save_image_matrix(
        output_dir / "dataset_mapping_spearman_matrix.png",
        dataset_mapping_images,
        DATASETS,
        mapping_columns,
        f"Composer dataset x mapping | best anchor by {BEST_METRIC}",
        cell_texts=dataset_mapping_texts,
    )
    save_image_matrix(
        output_dir / "dataset_anchor_spearman_matrix.png",
        anchor_dataset_images,
        DATASETS,
        anchor_columns,
        f"Composer dataset x anchor mode | best mapping by {BEST_METRIC}",
        cell_texts=anchor_dataset_texts,
    )
    save_image_matrix(
        output_dir / "dataset_transformation_spearman_matrix.png",
        transformation_images,
        DATASETS,
        transformation_columns,
        f"Composer dataset x transformation | best mode by {BEST_METRIC}",
        cell_texts=transformation_texts,
    )

    best_combo_mapping = best_combo_mapping_by_dataset(mode_results_by_n)
    l1_panels = []
    spearman_panels = []
    for dataset in DATASETS:
        if dataset not in best_combo_mapping:
            continue
        target_combo = best_combo_mapping[dataset]["combo"]
        target_mapping = best_combo_mapping[dataset]["mapping"]
        panel_title = f"{dataset} | combo={target_combo} | mapping={target_mapping}"
        l1_panels.append(
            (
                panel_title,
                ANCHOR_N_SWEEP,
                anchor_n_curves_for_dataset(
                    dataset,
                    target_combo,
                    target_mapping,
                    mode_results_by_n,
                    "l1",
                ),
            )
        )
        spearman_panels.append(
            (
                panel_title,
                ANCHOR_N_SWEEP,
                anchor_n_curves_for_dataset(
                    dataset,
                    target_combo,
                    target_mapping,
                    mode_results_by_n,
                    "spearman",
                ),
            )
        )

    if l1_panels:
        save_multi_line_grid_chart(
            output_dir / "anchor_n_l1_grid.png",
            l1_panels,
            ncols=2,
            title="L1 vs anchor n by dataset and method",
            xlabel="Anchor n",
            ylabel="L1",
        )
    if spearman_panels:
        save_multi_line_grid_chart(
            output_dir / "anchor_n_spearman_grid.png",
            spearman_panels,
            ncols=2,
            title="Spearman vs anchor n by dataset and method",
            xlabel="Anchor n",
            ylabel="Spearman",
        )
    covariate_correlation_matrices(output_dir, mode_results_by_n)
    print(output_dir)


if __name__ == "__main__":
    main()
