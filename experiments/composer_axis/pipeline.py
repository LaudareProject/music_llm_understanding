import os
from pathlib import Path

import numpy as np
from tqdm import tqdm

from experiments.common.anchor_selection import select_anchors
from experiments.common.embed_aria import encode_midi
from experiments.common.folder_handling import create_folder
from experiments.common.humdrum import kern_to_midi, musicxml_to_midi
from experiments.common.math1d import (
    cross_validated_mapping_summary,
    fit_exp_linear,
    fit_linear,
    fit_log_linear,
    l1,
    pearson,
    project_1d,
    project_1d_regression_direction,
    random_l1_baseline,
    r2,
    spearman,
)
from experiments.common.plots import save_scatter
from experiments.common.run_helpers import (
    feature_selection_space_env,
    parse_feature_cols_env,
    save_metric_table,
    selected_combos,
    transform_with_feature_selection,
)
from experiments.common.tsv import read_tsv, write_tsv


REPO_ROOT = Path(__file__).resolve().parents[2]
TSV_PATH = REPO_ROOT / "experiments" / "composer_axis" / "songs.tsv"
OUTPUT_ROOT = REPO_ROOT / "outputs" / "experiments" / "composer_axis"
CACHE_DIR = REPO_ROOT / "cache" / "experiments" / "composer_axis"
BIN_DIR = REPO_ROOT / "bin"

DATASETS = ["asap", "music21", "ewld", "overall"]
MAPPINGS = ["raw", "linear", "log_linear", "exp_linear"]
BEST_METRIC = os.environ.get("BEST_METRIC", "spearman").strip().lower()
VALID_BEST_METRICS = {"spearman", "pearson", "r2", "l1"}
if BEST_METRIC not in VALID_BEST_METRICS:
    raise ValueError(f"BEST_METRIC must be one of {sorted(VALID_BEST_METRICS)}")


def metric_value(metrics, metric_name):
    if metric_name == "l1":
        return float(metrics["l1"])
    if metric_name == "spearman":
        return float(metrics["spearman"])
    if metric_name == "pearson":
        return float(metrics["pearson"])
    if metric_name == "r2":
        return float(metrics["r2"])
    raise ValueError(metric_name)


def better_score(value, best_value, metric_name):
    if np.isnan(value):
        return False
    if best_value is None:
        return True
    if metric_name == "l1":
        return value < best_value
    return value > best_value


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


def combo_selection_score(payload):
    scores = []
    for mapping_name in MAPPINGS:
        values = [
            metric_value(payload[dataset][mapping_name], BEST_METRIC)
            for dataset in DATASETS
            if payload[dataset] is not None
            and not np.isnan(metric_value(payload[dataset][mapping_name], BEST_METRIC))
        ]
        if values:
            scores.append(float(np.mean(values)))
    if not scores:
        return float("inf") if BEST_METRIC == "l1" else float("-inf")
    return min(scores) if BEST_METRIC == "l1" else max(scores)


def dataset_combo_selection_score(dataset_payload):
    scores = [
        metric_value(dataset_payload[mapping_name], BEST_METRIC)
        for mapping_name in MAPPINGS
        if not np.isnan(metric_value(dataset_payload[mapping_name], BEST_METRIC))
    ]
    if not scores:
        return float("inf") if BEST_METRIC == "l1" else float("-inf")
    return min(scores) if BEST_METRIC == "l1" else max(scores)


def prepare_rows(output_dir):
    rows = read_tsv(TSV_PATH)
    midi_dir = output_dir / "midi"
    midi_dir.mkdir(parents=True, exist_ok=True)
    print("[1/5] preparing midi")
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
    filtered_rows = [row for row in rows if row["query_id"] not in skipped]
    return filtered_rows, midi_paths


def relevant_rows_for_dataset(rows, dataset):
    return (
        rows
        if dataset == "overall"
        else [row for row in rows if row["dataset"] == dataset]
    )


def evaluate_dataset(
    matrix, rows, query_to_index, dataset, anchor_selection_method, anchor_n
):
    relevant_rows = relevant_rows_for_dataset(rows, dataset)
    anchor_a_ids, anchor_b_ids, eval_rows = select_anchors(
        relevant_rows,
        anchor_selection_method,
        target_key="midpoint_year",
        n=anchor_n,
        best_pairs_separation=100,
    )
    if len(eval_rows) < 2:
        return None
    anchor_a_idx = [query_to_index[query_id] for query_id in anchor_a_ids]
    anchor_b_idx = [query_to_index[query_id] for query_id in anchor_b_ids]
    eval_idx = [query_to_index[row["query_id"]] for row in eval_rows]
    real = np.asarray(
        [float(row["midpoint_year"]) for row in eval_rows], dtype=np.float64
    )
    if anchor_selection_method == "random_sample_regression":
        sample_ids = anchor_a_ids + anchor_b_ids
        sample_idx = [query_to_index[query_id] for query_id in sample_ids]
        query_to_row = {row["query_id"]: row for row in relevant_rows}
        sample_targets = [
            float(query_to_row[query_id]["midpoint_year"]) for query_id in sample_ids
        ]
        raw_projection = project_1d_regression_direction(
            matrix,
            sample_idx,
            sample_targets,
            eval_idx,
        )
    else:
        raw_projection = project_1d(matrix, anchor_a_idx, anchor_b_idx, eval_idx)
    anchors_a = np.asarray(
        [
            float(row["midpoint_year"])
            for row in relevant_rows
            if row["query_id"] in anchor_a_ids
        ],
        dtype=np.float64,
    )
    anchors_b = np.asarray(
        [
            float(row["midpoint_year"])
            for row in relevant_rows
            if row["query_id"] in anchor_b_ids
        ],
        dtype=np.float64,
    )
    baseline_l1 = random_l1_baseline(real)
    groups = [row["composer"] for row in eval_rows]
    cv_groups = np.asarray(groups)
    payload = {}
    for mapping_name in MAPPINGS:
        summary = cross_validated_mapping_summary(
            raw_projection,
            real,
            cv_groups,
            mapping_name,
        )
        predicted = np.asarray(summary["predicted"], dtype=np.float64)
        payload[mapping_name] = {
            "real": real,
            "predicted": np.asarray(summary["groupkfold_predicted"], dtype=np.float64),
            "averaged_model_predicted": predicted,
            "groups": groups,
            "pearson": summary["pearson"],
            "pearson_ci95": summary["pearson_ci95"],
            "spearman": summary["spearman"],
            "spearman_ci95": summary["spearman_ci95"],
            "r2": summary["r2"],
            "r2_ci95": summary["r2_ci95"],
            "l1": summary["l1"],
            "l1_ci95": summary["l1_ci95"],
            "random": baseline_l1,
            "sample_count": len(eval_rows),
            "fold_count": summary["fold_count"],
            "query_ids": [row["query_id"] for row in eval_rows],
            "anchors_a": anchors_a,
            "anchors_b": anchors_b,
        }
    return payload


def metric_rows_for_combo(combo_name, payload):
    rows = []
    for dataset in DATASETS:
        if payload[dataset] is None:
            continue
        for mapping_name in MAPPINGS:
            metrics = payload[dataset][mapping_name]
            rows.append(
                {
                    "combo": combo_name,
                    "dataset": dataset,
                    "mapping": mapping_name,
                    "pearson": metrics["pearson"],
                    "pearson_ci95": metrics["pearson_ci95"],
                    "spearman": metrics["spearman"],
                    "spearman_ci95": metrics["spearman_ci95"],
                    "r2": metrics["r2"],
                    "r2_ci95": metrics["r2_ci95"],
                    "l1": metrics["l1"],
                    "l1_ci95": metrics["l1_ci95"],
                    "random": metrics["random"],
                    "sample_count": metrics["sample_count"],
                    "fold_count": metrics["fold_count"],
                }
            )
    return rows


def combo_score_row(combo_name, payload):
    raw_scores = []
    mapped_l1_values = []
    random_values = []
    for dataset in DATASETS:
        if payload[dataset] is None:
            continue
        raw_scores.append(abs(payload[dataset]["raw"]["spearman"]))
        random_values.append(payload[dataset]["raw"]["random"])
        for mapping_name in MAPPINGS[1:]:
            metrics = payload[dataset][mapping_name]
            if not np.isnan(metrics["l1"]):
                mapped_l1_values.append(metrics["l1"])
    return {
        "combo": combo_name,
        "raw_spearman_abs_mean": float(np.mean(raw_scores))
        if raw_scores
        else float("-inf"),
        "mapped_l1_mean": float(np.mean(mapped_l1_values))
        if mapped_l1_values
        else float("nan"),
        "random": float(np.mean(random_values)) if random_values else float("nan"),
    }


def scatter_title(dataset, combo_name, mapping_name, metrics):
    title = (
        f"{dataset} | {mapping_name} | {combo_name}\n"
        f"pearson={metrics['pearson']:.3f}+/-{metrics['pearson_ci95']:.3f}, "
        f"spearman={metrics['spearman']:.3f}+/-{metrics['spearman_ci95']:.3f}"
    )
    if mapping_name != "raw":
        title += (
            f"\nl1={metrics['l1']:.2f}+/-{metrics['l1_ci95']:.2f}, "
            f"random={metrics['random']:.2f}"
        )
    return title


def write_outputs(
    output_dir,
    combos,
    metric_rows,
    combo_scores,
    payloads_by_combo,
):
    write_tsv(
        output_dir / "metrics.tsv",
        [
            {
                "combo": row["combo"],
                "dataset": row["dataset"],
                "mapping": row["mapping"],
                "pearson": f"{row['pearson']:.10f}"
                if not np.isnan(row["pearson"])
                else "nan",
                "pearson_ci95": f"{row['pearson_ci95']:.10f}"
                if not np.isnan(row["pearson_ci95"])
                else "nan",
                "spearman": f"{row['spearman']:.10f}"
                if not np.isnan(row["spearman"])
                else "nan",
                "spearman_ci95": f"{row['spearman_ci95']:.10f}"
                if not np.isnan(row["spearman_ci95"])
                else "nan",
                "r2": f"{row['r2']:.10f}" if not np.isnan(row["r2"]) else "nan",
                "r2_ci95": f"{row['r2_ci95']:.10f}"
                if not np.isnan(row["r2_ci95"])
                else "nan",
                "l1": f"{row['l1']:.10f}" if not np.isnan(row["l1"]) else "nan",
                "l1_ci95": f"{row['l1_ci95']:.10f}"
                if not np.isnan(row["l1_ci95"])
                else "nan",
                "random": f"{row['random']:.10f}"
                if not np.isnan(row["random"])
                else "nan",
                "sample_count": row["sample_count"],
                "fold_count": row["fold_count"],
            }
            for row in metric_rows
        ],
        [
            "combo",
            "dataset",
            "mapping",
            "pearson",
            "pearson_ci95",
            "spearman",
            "spearman_ci95",
            "r2",
            "r2_ci95",
            "l1",
            "l1_ci95",
            "random",
            "sample_count",
            "fold_count",
        ],
    )
    write_tsv(
        output_dir / "combo_scores.tsv",
        [
            {
                "combo": row["combo"],
                "raw_spearman_abs_mean": f"{row['raw_spearman_abs_mean']:.10f}"
                if not np.isnan(row["raw_spearman_abs_mean"])
                else "nan",
                "mapped_l1_mean": f"{row['mapped_l1_mean']:.10f}"
                if not np.isnan(row["mapped_l1_mean"])
                else "nan",
                "random": f"{row['random']:.10f}"
                if not np.isnan(row["random"])
                else "nan",
            }
            for row in combo_scores
        ],
        [
            "combo",
            "raw_spearman_abs_mean",
            "mapped_l1_mean",
            "random",
        ],
    )
    column_specs = [
        (f"{dataset}:{mapping}", (dataset, mapping))
        for dataset in DATASETS
        for mapping in MAPPINGS
    ]
    save_metric_table(
        output_dir / "pearson_table.png",
        combos,
        metric_rows,
        ["combo", "dataset", "mapping"],
        column_specs,
        "pearson",
        "Composer chronology Pearson",
    )
    save_metric_table(
        output_dir / "spearman_table.png",
        combos,
        metric_rows,
        ["combo", "dataset", "mapping"],
        column_specs,
        "spearman",
        "Composer chronology Spearman",
    )
    save_metric_table(
        output_dir / "l1_table.png",
        combos,
        metric_rows,
        ["combo", "dataset", "mapping"],
        column_specs,
        "l1",
        "Composer chronology L1",
    )

    scatter_dir = output_dir / "scatters_by_combo"
    for combo_name, payload in payloads_by_combo.items():
        for dataset in DATASETS:
            if payload[dataset] is None:
                continue
            candidate_mappings = [
                mapping_name
                for mapping_name in MAPPINGS
                if not np.isnan(payload[dataset][mapping_name]["l1"])
            ]
            if not candidate_mappings:
                continue
            mapping_name = min(
                candidate_mappings,
                key=lambda mapping: payload[dataset][mapping]["l1"],
            )
            metrics = payload[dataset][mapping_name]
            save_scatter(
                scatter_dir / f"scatter_{dataset}__{combo_name}__{mapping_name}.png",
                metrics["real"],
                metrics["predicted"],
                metrics["groups"],
                scatter_title(dataset, combo_name, mapping_name, metrics),
                "Real value (composer midpoint year)",
                "Projected value"
                if mapping_name == "raw"
                else "Mapped projected value",
                ref_line=mapping_name != "raw",
                anchors_a=metrics["anchors_a"] if mapping_name != "raw" else None,
                anchors_b=metrics["anchors_b"] if mapping_name != "raw" else None,
            )

    best_by_dataset = {}
    for dataset in DATASETS:
        candidates = [
            (combo_name, payload[dataset])
            for combo_name, payload in payloads_by_combo.items()
            if payload[dataset] is not None
        ]
        if not candidates:
            continue
        scored = sorted(
            candidates,
            key=lambda item: dataset_combo_selection_score(item[1]),
            reverse=BEST_METRIC != "l1",
        )
        combo_name, dataset_payload = scored[0]
        best_by_dataset[dataset] = {"combo": combo_name, "payload": dataset_payload}

    projection_rows = []
    for dataset, best in best_by_dataset.items():
        for mapping_name in MAPPINGS:
            metrics = best["payload"][mapping_name]
            save_scatter(
                output_dir / f"scatter_{dataset}_{mapping_name}.png",
                metrics["real"],
                metrics["predicted"],
                metrics["groups"],
                scatter_title(dataset, best["combo"], mapping_name, metrics),
                "Real value (composer midpoint year)",
                "Projected value"
                if mapping_name == "raw"
                else "Mapped projected value",
                ref_line=mapping_name != "raw",
                anchors_a=metrics["anchors_a"] if mapping_name != "raw" else None,
                anchors_b=metrics["anchors_b"] if mapping_name != "raw" else None,
            )
            for query_id, real_value, predicted_value, group in zip(
                metrics["query_ids"],
                metrics["real"],
                metrics["predicted"],
                metrics["groups"],
            ):
                projection_rows.append(
                    {
                        "combo": best["combo"],
                        "dataset": dataset,
                        "mapping": mapping_name,
                        "query_id": query_id,
                        "composer": group,
                        "real": f"{float(real_value):.10f}",
                        "predicted": f"{float(predicted_value):.10f}",
                    }
                )
    write_tsv(
        output_dir / "projections.tsv",
        projection_rows,
        ["combo", "dataset", "mapping", "query_id", "composer", "real", "predicted"],
    )


def main(anchor_selection_method="first_vs_last"):
    forced_combo_name = os.environ.get("FEATURE_SELECTION_COMBO", "").strip()
    anchor_n = int(os.environ.get("ANCHOR_N", "10").strip() or "10")
    feature_cols = parse_feature_cols_env()
    feature_space = feature_selection_space_env()
    combos = selected_combos(forced_combo_name)
    output_dir = create_folder(
        OUTPUT_ROOT,
        {
            "feature_selection_space": feature_space,
            "feature_selection_combo": forced_combo_name or None,
            "feature_selection_cols": feature_cols,
            "anchor_selection": anchor_selection_method,
            "anchor_n": anchor_n,
            "best_metric": BEST_METRIC,
        },
    )
    rows, midi_paths = prepare_rows(output_dir)

    print("[2/5] embedding with aria")
    query_ids = [row["query_id"] for row in rows]
    matrix = np.stack(
        [
            load_or_compute_embedding(query_id, midi_paths.get(query_id))
            for query_id in tqdm(query_ids, desc="Embeddings", unit="query")
        ],
        axis=0,
    )
    labels = [row["composer"] for row in rows]
    target_scores = np.asarray(
        [float(row["midpoint_year"]) for row in rows], dtype=np.float64
    )
    query_to_index = {query_id: i for i, query_id in enumerate(query_ids)}
    np.save(output_dir / "embeddings.npy", matrix)

    print("[3/5] evaluating combos")
    if forced_combo_name:
        print(f"  forced combo: {forced_combo_name}")
    if feature_cols is not None:
        print(f"  using selected {feature_space} cols: {len(feature_cols)}")
    metric_rows = []
    combo_scores = []
    payloads_by_combo = {}
    for combo in combos:
        print(f"  - {combo['name']}")
        transformed = transform_with_feature_selection(
            matrix, labels, combo, target_scores, feature_cols, feature_space
        )
        payload = {
            dataset: evaluate_dataset(
                transformed,
                rows,
                query_to_index,
                dataset,
                anchor_selection_method,
                anchor_n,
            )
            for dataset in DATASETS
        }
        metric_rows.extend(metric_rows_for_combo(combo["name"], payload))
        combo_scores.append(combo_score_row(combo["name"], payload))
        payloads_by_combo[combo["name"]] = payload

    print("[4/5] writing outputs")
    write_outputs(
        output_dir,
        combos,
        metric_rows,
        combo_scores,
        payloads_by_combo,
    )
    print("[5/5] done")
    print(output_dir)
    return output_dir
