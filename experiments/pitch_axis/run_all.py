from pathlib import Path
import sys


sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.common.aggregate import (
    read_tsv,
    run_script_and_capture_output_dir,
    to_float,
)
from experiments.common.folder_handling import create_folder
from experiments.common.plots import save_heatmap, save_image_matrix

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = REPO_ROOT / "outputs" / "experiments" / "pitch_axis"
DATASETS = ["pitch"]
MAPPINGS = ["raw", "linear", "log_linear", "exp_linear"]

RUN_SPECS = [
    {"script": "run_base.py", "anchor": "anchor_column", "label": "column"},
    {"script": "run_random_median.py", "anchor": "random_median", "label": "random"},
    {"script": "run_best_pairs.py", "anchor": "best_pairs", "label": "pairs"},
]


def load_mode_result(spec, output_dir):
    rows = read_tsv(output_dir / "metrics.tsv")
    return {**spec, "output_dir": output_dir, "rows": rows}


def mode_score(rows, combo):
    values = [
        -to_float(row["l1"])
        for row in rows
        if row["combo"] == combo
        and row["mapping"] in MAPPINGS[1:]
        and not np.isnan(to_float(row["l1"]))
    ]
    return max(values) if values else float("-inf")


def best_rows(mode_result):
    combos = sorted({row["combo"] for row in mode_result["rows"]})
    best_combo = max(combos, key=lambda combo: mode_score(mode_result["rows"], combo))
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
    images = [[]]
    texts = [[]]
    for mapping in MAPPINGS:
        candidates = [
            (-row_l1(mode["best_rows"][mapping]), mode)
            for mode in mode_results
            if not np.isnan(row_l1(mode["best_rows"][mapping]))
        ]
        best_mode = max(candidates, key=lambda item: item[0])[1] if candidates else None
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
            if not np.isnan(row_l1(mode["best_rows"][mapping]))
        ]
        if not candidate_mappings:
            images[0].append(None)
            texts[0].append("")
            continue
        best_mapping = min(
            candidate_mappings,
            key=lambda mapping: row_l1(mode["best_rows"][mapping]),
        )
        images[0].append(mode["output_dir"] / f"scatter_pitch_{best_mapping}.png")
        texts[0].append(cell_label(mode["best_rows"][best_mapping]))
    return [mode["label"] for mode in mode_results], images, texts


def anchor_dataset_value_matrix(mode_results, metric_name, maximize):
    values = [
        [
            best_value(
                [
                    mode["best_rows"][mapping]
                    for mapping in (
                        MAPPINGS if metric_name == "spearman" else MAPPINGS[1:]
                    )
                ],
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
                    l1_value = row_l1(row)
                    if not np.isnan(l1_value):
                        candidates.append((-l1_value, mode, row))
        best = max(candidates, key=lambda item: item[0]) if candidates else None
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


def main():
    output_dir = create_folder(
        OUTPUT_ROOT,
        {"aggregate": True, "scripts": [spec["script"] for spec in RUN_SPECS]},
    )
    mode_results = []
    for spec in RUN_SPECS:
        script_path = REPO_ROOT / "experiments" / "pitch_axis" / spec["script"]
        run_output_dir = run_script_and_capture_output_dir(script_path, OUTPUT_ROOT)
        mode_result = load_mode_result(spec, run_output_dir)
        mode_result["best_rows"] = best_rows(mode_result)
        mode_results.append(mode_result)

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
        "Pitch dataset x mapping | best anchor by cell | L1",
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
        "Pitch dataset x anchor mode | best mapping by cell | L1",
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
        "Pitch dataset x transformation | best mode by cell | L1",
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
        "Pitch dataset x mapping | best anchor by L1",
        cell_texts=dataset_mapping_texts,
    )
    save_image_matrix(
        output_dir / "dataset_anchor_spearman_matrix.png",
        anchor_dataset_images,
        DATASETS,
        anchor_columns,
        "Pitch dataset x anchor mode | best mapping by L1",
        cell_texts=anchor_dataset_texts,
    )
    save_image_matrix(
        output_dir / "dataset_transformation_spearman_matrix.png",
        transformation_images,
        DATASETS,
        transformation_columns,
        "Pitch dataset x transformation | best mode by L1",
        cell_texts=transformation_texts,
    )
    print(output_dir)


if __name__ == "__main__":
    main()
