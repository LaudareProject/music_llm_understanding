from pathlib import Path
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

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = REPO_ROOT / "outputs" / "experiments" / "composer_axis"
SONGS_TSV = REPO_ROOT / "experiments" / "composer_axis" / "songs.tsv"
DATASETS = ["asap", "music21", "ewld", "overall"]
MAPPINGS = ["raw", "linear", "log_linear", "exp_linear"]
BASE_ANCHOR_N = 10
ANCHOR_N_SWEEP = [1, 2, 4, 8, 16, 32]

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


def mode_score(rows, dataset, combo):
    values = [
        -to_float(row["l1"])
        for row in rows
        if row["dataset"] == dataset
        and row["combo"] == combo
        and row["mapping"] in MAPPINGS[1:]
        and not np.isnan(to_float(row["l1"]))
    ]
    return max(values) if values else float("-inf")


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
        best_combo = max(
            available,
            key=lambda combo: mode_score(mode_result["rows"], dataset, combo),
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


def cell_label(row):
    return (
        f"rho={to_float(row['spearman']):.3f}+/-{to_float(row.get('spearman_ci95', 'nan')):.3f}"
        f"\nl1={to_float(row['l1']):.3f}+/-{to_float(row.get('l1_ci95', 'nan')):.3f}"
    )


def best_value(rows, metric_name, maximize):
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
                if not np.isnan(row_l1(row)):
                    candidates.append((-row_l1(row), mode))
            best_mode = (
                max(candidates, key=lambda item: item[0])[1] if candidates else None
            )
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
                if not np.isnan(row_l1(dataset_rows[mapping]))
            ]
            if not candidate_mappings:
                row_images.append(None)
                row_texts.append("")
                continue
            best_mapping = min(
                candidate_mappings,
                key=lambda mapping: row_l1(dataset_rows[mapping]),
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
            mapping_rows = [
                dataset_rows[mapping]
                for mapping in (MAPPINGS if metric_name == "spearman" else MAPPINGS[1:])
            ]
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
                        l1_value = row_l1(row)
                        if not np.isnan(l1_value):
                            candidates.append((-l1_value, mode, row))
            best = max(candidates, key=lambda item: item[0]) if candidates else None
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
            extra_env={"ANCHOR_N": str(anchor_n)},
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
                    l1_value = row_l1(row)
                    if np.isnan(l1_value):
                        continue
                    candidates.append((l1_value, row["combo"], row["mapping"]))
        if candidates:
            _, combo, mapping = min(candidates, key=lambda item: item[0])
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
            elif metric_name == "spearman":
                y_values.append(row_spearman(candidates[0]))
            else:
                y_values.append(row_l1(candidates[0]))
        series.append((spec["label"], y_values))
    return series


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
        },
    )
    save_year_source_stacked_histograms(output_dir)
    mode_results = run_modes(BASE_ANCHOR_N)
    mode_results_by_n = {anchor_n: run_modes(anchor_n) for anchor_n in ANCHOR_N_SWEEP}

    mapping_columns, dataset_mapping_images, dataset_mapping_texts = (
        dataset_mapping_image_matrix(mode_results)
    )
    _, dataset_mapping_l1 = dataset_mapping_value_matrix(mode_results, "l1", False)
    _, dataset_mapping_spearman = dataset_mapping_value_matrix(
        mode_results, "spearman", True
    )
    anchor_columns, anchor_dataset_images, anchor_dataset_texts = (
        anchor_dataset_image_matrix(mode_results)
    )
    _, anchor_dataset_l1 = anchor_dataset_value_matrix(mode_results, "l1", False)
    _, anchor_dataset_spearman = anchor_dataset_value_matrix(
        mode_results, "spearman", True
    )
    transformation_columns, transformation_images, transformation_texts = (
        dataset_transformation_image_matrix(mode_results)
    )
    _, transformation_l1 = dataset_transformation_value_matrix(
        mode_results, "l1", False
    )
    _, transformation_spearman = dataset_transformation_value_matrix(
        mode_results, "spearman", True
    )

    save_heatmap(
        output_dir / "dataset_mapping_l1_heatmap.png",
        DATASETS,
        mapping_columns,
        dataset_mapping_l1,
        "Composer dataset x mapping | best anchor by cell | L1",
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
        "Composer dataset x anchor mode | best mapping by cell | L1",
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
        "Composer dataset x transformation | best mode by cell | L1",
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
        "Composer dataset x mapping | best anchor by L1",
        cell_texts=dataset_mapping_texts,
    )
    save_image_matrix(
        output_dir / "dataset_anchor_spearman_matrix.png",
        anchor_dataset_images,
        DATASETS,
        anchor_columns,
        "Composer dataset x anchor mode | best mapping by L1",
        cell_texts=anchor_dataset_texts,
    )
    save_image_matrix(
        output_dir / "dataset_transformation_spearman_matrix.png",
        transformation_images,
        DATASETS,
        transformation_columns,
        "Composer dataset x transformation | best mode by L1",
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
    print(output_dir)


if __name__ == "__main__":
    main()
