import os
from pathlib import Path

import numpy as np
from tqdm import tqdm

from experiments.common.anchor_selection import select_anchors
from experiments.common.embed_aria import encode_midi
from experiments.common.folder_handling import create_folder
from experiments.common.humdrum import kern_to_midi, musicxml_to_midi
from experiments.common.math1d import (
    _hann_weights,
    fit_exp_linear,
    fit_linear,
    fit_log_linear,
    hann_density,
    l1,
    pearson,
    project_1d,
    random_l1_baseline,
    spearman,
    weighted_pearson,
    weighted_spearman,
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


def combo_selection_score(payload, use_hann_selection):
    key = "hann_score" if use_hann_selection else "l1"
    sign = 1.0 if use_hann_selection else -1.0
    scores = []
    for mapping_name in MAPPINGS[1:]:
        values = [
            payload[dataset][mapping_name][key]
            for dataset in DATASETS
            if payload[dataset] is not None
            and not np.isnan(payload[dataset][mapping_name][key])
        ]
        if values:
            scores.append(sign * float(np.mean(values)))
    return max(scores) if scores else float("-inf")


def dataset_combo_selection_score(dataset_payload, use_hann_selection):
    key = "hann_score" if use_hann_selection else "l1"
    sign = 1.0 if use_hann_selection else -1.0
    scores = [
        sign * dataset_payload[mapping_name][key]
        for mapping_name in MAPPINGS[1:]
        if not np.isnan(dataset_payload[mapping_name][key])
    ]
    return max(scores) if scores else float("-inf")


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


def evaluate_dataset(matrix, rows, query_to_index, dataset, anchor_selection_method):
    relevant_rows = relevant_rows_for_dataset(rows, dataset)
    anchor_column = "overall_side" if dataset == "overall" else "axis_side"
    anchor_a_ids, anchor_b_ids, eval_rows = select_anchors(
        relevant_rows,
        anchor_column,
        anchor_selection_method,
        target_key="midpoint_year",
        random_n=10,
        best_pairs_n=10,
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
    raw_projection = project_1d(matrix, anchor_a_idx, anchor_b_idx, eval_idx)
    birth_years = np.asarray(
        [float(row["birth_year"]) for row in eval_rows], dtype=np.float64
    )
    death_years = np.asarray(
        [float(row["death_year"]) for row in eval_rows], dtype=np.float64
    )
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
    predictions = {
        "raw": raw_projection,
        "linear": fit_linear(raw_projection, real),
        "log_linear": fit_log_linear(raw_projection, real),
        "exp_linear": fit_exp_linear(raw_projection, real),
    }
    baseline_l1 = random_l1_baseline(real)
    groups = [row["composer"] for row in eval_rows]
    payload = {}
    for mapping_name, predicted in predictions.items():
        predicted = np.asarray(predicted, dtype=np.float64)
        weights = _hann_weights(predicted, birth_years, death_years)
        payload[mapping_name] = {
            "real": real,
            "predicted": predicted,
            "groups": groups,
            "pearson": pearson(real, predicted),
            "spearman": spearman(real, predicted),
            "hann_pearson": weighted_pearson(real, predicted, weights),
            "hann_spearman": weighted_spearman(real, predicted, weights),
            "hann_score": hann_density(predicted, birth_years, death_years),
            "l1": float("nan") if mapping_name == "raw" else l1(real, predicted),
            "random": baseline_l1,
            "sample_count": len(eval_rows),
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
                    "spearman": metrics["spearman"],
                    "hann_pearson": metrics["hann_pearson"],
                    "hann_spearman": metrics["hann_spearman"],
                    "hann_score": metrics["hann_score"],
                    "l1": metrics["l1"],
                    "random": metrics["random"],
                    "sample_count": metrics["sample_count"],
                }
            )
    return rows


def combo_score_row(combo_name, payload):
    raw_scores = []
    mapped_l1_values = []
    hann_score_values = []
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
            if not np.isnan(metrics["hann_score"]):
                hann_score_values.append(metrics["hann_score"])
    return {
        "combo": combo_name,
        "raw_spearman_abs_mean": float(np.mean(raw_scores))
        if raw_scores
        else float("-inf"),
        "mapped_l1_mean": float(np.mean(mapped_l1_values))
        if mapped_l1_values
        else float("nan"),
        "mapped_hann_score_mean": float(np.mean(hann_score_values))
        if hann_score_values
        else float("nan"),
        "random": float(np.mean(random_values)) if random_values else float("nan"),
    }


def write_outputs(
    output_dir, combos, metric_rows, combo_scores, payloads_by_combo, use_hann_selection
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
                "spearman": f"{row['spearman']:.10f}"
                if not np.isnan(row["spearman"])
                else "nan",
                "hann_pearson": f"{row['hann_pearson']:.10f}"
                if not np.isnan(row["hann_pearson"])
                else "nan",
                "hann_spearman": f"{row['hann_spearman']:.10f}"
                if not np.isnan(row["hann_spearman"])
                else "nan",
                "hann_score": f"{row['hann_score']:.10f}"
                if not np.isnan(row["hann_score"])
                else "nan",
                "l1": f"{row['l1']:.10f}" if not np.isnan(row["l1"]) else "nan",
                "random": f"{row['random']:.10f}"
                if not np.isnan(row["random"])
                else "nan",
                "sample_count": row["sample_count"],
            }
            for row in metric_rows
        ],
        [
            "combo",
            "dataset",
            "mapping",
            "pearson",
            "spearman",
            "hann_pearson",
            "hann_spearman",
            "hann_score",
            "l1",
            "random",
            "sample_count",
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
                "mapped_hann_score_mean": f"{row['mapped_hann_score_mean']:.10f}"
                if not np.isnan(row["mapped_hann_score_mean"])
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
            "mapped_hann_score_mean",
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
    save_metric_table(
        output_dir / "hann_pearson_table.png",
        combos,
        metric_rows,
        ["combo", "dataset", "mapping"],
        column_specs,
        "hann_pearson",
        "Composer chronology Hann-weighted Pearson",
    )
    save_metric_table(
        output_dir / "hann_spearman_table.png",
        combos,
        metric_rows,
        ["combo", "dataset", "mapping"],
        column_specs,
        "hann_spearman",
        "Composer chronology Hann-weighted Spearman",
    )
    save_metric_table(
        output_dir / "hann_score_table.png",
        combos,
        metric_rows,
        ["combo", "dataset", "mapping"],
        column_specs,
        "hann_score",
        "Composer chronology Hann overlap score",
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
        combo_name, dataset_payload = max(
            candidates,
            key=lambda item: dataset_combo_selection_score(item[1], use_hann_selection),
        )
        best_by_dataset[dataset] = {"combo": combo_name, "payload": dataset_payload}

    projection_rows = []
    for dataset, best in best_by_dataset.items():
        for mapping_name in MAPPINGS:
            metrics = best["payload"][mapping_name]
            title = f"{dataset} | {mapping_name} | {best['combo']}\npearson={metrics['pearson']:.3f}, spearman={metrics['spearman']:.3f}"
            if mapping_name != "raw":
                title += f"\nl1={metrics['l1']:.2f}, random={metrics['random']:.2f}, hann={metrics['hann_score']:.3f}"
            save_scatter(
                output_dir / f"scatter_{dataset}_{mapping_name}.png",
                metrics["real"],
                metrics["predicted"],
                metrics["groups"],
                title,
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


def main(use_hann_selection=False, anchor_selection_method="anchor_column"):
    forced_combo_name = os.environ.get("FEATURE_SELECTION_COMBO", "").strip()
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
            "use_hann_selection": use_hann_selection,
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
                transformed, rows, query_to_index, dataset, anchor_selection_method
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
        use_hann_selection,
    )
    print("[5/5] done")
    print(output_dir)
    return output_dir
