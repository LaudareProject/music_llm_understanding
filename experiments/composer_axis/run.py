import os
from pathlib import Path
import sys

import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.common.tsv import read_tsv, write_tsv
from experiments.common.folder_handling import create_folder
from experiments.common.plots import save_scatter, save_table_png, save_projection_strip
from experiments.common.math1d import (
    COMBOS,
    apply_combo,
    fit_exp_linear,
    fit_linear,
    fit_log_linear,
    hann_density,
    _hann_weights,
    l1,
    pearson,
    project_1d,
    random_l1_baseline,
    spearman,
    weighted_pearson,
    weighted_spearman,
)
from experiments.common.humdrum import kern_to_midi, musicxml_to_midi
from experiments.common.embed_aria import encode_midi


REPO_ROOT = Path(__file__).resolve().parents[2]
TSV_PATH = REPO_ROOT / "experiments" / "composer_axis" / "songs.tsv"
OUTPUT_ROOT = REPO_ROOT / "outputs" / "experiments" / "composer_axis"
CACHE_DIR = REPO_ROOT / "cache" / "experiments" / "composer_axis"
BIN_DIR = REPO_ROOT / "bin"


def parse_feature_cols_env():
    raw = os.environ.get("FEATURE_SELECTION_COLS", "").strip()
    if not raw:
        return None
    cols = [int(token.strip()) for token in raw.split(",") if token.strip()]
    return cols if cols else None


def feature_selection_space_env():
    value = os.environ.get("FEATURE_SELECTION_SPACE", "transformed").strip().lower()
    if value not in {"transformed", "raw"}:
        raise ValueError(f"Invalid FEATURE_SELECTION_SPACE: {value}")
    return value


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
    if use_hann_selection:
        return combo_selection_score_hann(payload)
    return combo_selection_score_standard(payload)


def combo_selection_score_standard(payload):
    scores = []
    for mapping_name in ["linear", "log_linear", "exp_linear"]:
        l1_values = [
            payload[dataset][mapping_name]["l1"]
            for dataset in ["asap", "music21", "ewld", "overall"]
            if payload[dataset] is not None
            and not np.isnan(payload[dataset][mapping_name]["l1"])
        ]
        if l1_values:
            scores.append(-float(np.mean(l1_values)))
    return max(scores) if scores else float("-inf")


def combo_selection_score_hann(payload):
    scores = []
    for mapping_name in ["linear", "log_linear", "exp_linear"]:
        hann_values = [
            payload[dataset][mapping_name]["hann_score"]
            for dataset in ["asap", "music21", "ewld", "overall"]
            if payload[dataset] is not None
            and not np.isnan(payload[dataset][mapping_name]["hann_score"])
        ]
        if hann_values:
            scores.append(float(np.mean(hann_values)))
    return max(scores) if scores else float("-inf")


def dataset_combo_selection_score(dataset_payload, use_hann_selection):
    if use_hann_selection:
        return dataset_combo_selection_score_hann(dataset_payload)
    return dataset_combo_selection_score_standard(dataset_payload)


def dataset_combo_selection_score_standard(dataset_payload):
    scores = []
    for mapping_name in ["linear", "log_linear", "exp_linear"]:
        value = dataset_payload[mapping_name]["l1"]
        if not np.isnan(value):
            scores.append(-value)
    return max(scores) if scores else float("-inf")


def dataset_combo_selection_score_hann(dataset_payload):
    scores = []
    for mapping_name in ["linear", "log_linear", "exp_linear"]:
        value = dataset_payload[mapping_name]["hann_score"]
        if not np.isnan(value):
            scores.append(value)
    return max(scores) if scores else float("-inf")


def save_metric_table(output_path, combos, metric_rows, metric_name, title):
    value_map = {
        (row["combo"], row["dataset"], row["mapping"]): row[metric_name]
        for row in metric_rows
    }
    headers = ["combo"] + [
        f"{dataset}:{mapping}"
        for dataset in ["asap", "music21", "ewld", "overall"]
        for mapping in ["raw", "linear", "log_linear", "exp_linear"]
    ]
    body = []
    for combo in combos:
        combo_name = combo["name"]
        line = [combo_name]
        for dataset in ["asap", "music21", "ewld", "overall"]:
            for mapping in ["raw", "linear", "log_linear", "exp_linear"]:
                value = value_map.get((combo_name, dataset, mapping), float("nan"))
                line.append("nan" if np.isnan(value) else f"{value:.3f}")
        body.append(line)
    save_table_png(output_path, headers, body, title)


def evaluate_dataset(name, matrix, rows, query_to_index, anchor_column):
    if name == "overall":
        relevant_rows = rows
    else:
        relevant_rows = [row for row in rows if row["dataset"] == name]
    anchor_a_ids = [
        row["query_id"] for row in relevant_rows if row[anchor_column] == "a"
    ]
    anchor_b_ids = [
        row["query_id"] for row in relevant_rows if row[anchor_column] == "b"
    ]
    eval_rows = [row for row in relevant_rows if row[anchor_column] not in {"a", "b"}]
    if len(eval_rows) < 2:
        return None
    anchor_a_idx = [query_to_index[query_id] for query_id in anchor_a_ids]
    anchor_b_idx = [query_to_index[query_id] for query_id in anchor_b_ids]
    eval_idx = [query_to_index[row["query_id"]] for row in eval_rows]
    raw_projection = project_1d(matrix, anchor_a_idx, anchor_b_idx, eval_idx)
    real = np.asarray(
        [float(row["midpoint_year"]) for row in eval_rows], dtype=np.float64
    )
    birth_years = np.asarray(
        [float(row["birth_year"]) for row in eval_rows], dtype=np.float64
    )
    death_years = np.asarray(
        [float(row["death_year"]) for row in eval_rows], dtype=np.float64
    )
    mappings = {
        "raw": raw_projection,
        "linear": fit_linear(raw_projection, real),
        "log_linear": fit_log_linear(raw_projection, real),
        "exp_linear": fit_exp_linear(raw_projection, real),
    }
    groups = [row["composer"] for row in eval_rows]
    out = {}
    baseline_l1 = random_l1_baseline(real)
    for mapping_name, predicted in mappings.items():
        predicted = np.asarray(predicted, dtype=np.float64)
        w = _hann_weights(predicted, birth_years, death_years)
        if mapping_name == "raw":
            mapped_l1 = float("nan")
        else:
            mapped_l1 = l1(real, predicted)
        out[mapping_name] = {
            "real": real,
            "predicted": predicted,
            "groups": groups,
            "pearson": pearson(real, predicted),
            "spearman": spearman(real, predicted),
            "hann_pearson": weighted_pearson(real, predicted, w),
            "hann_spearman": weighted_spearman(real, predicted, w),
            "hann_score": hann_density(predicted, birth_years, death_years),
            "l1": mapped_l1,
            "random": baseline_l1,
            "sample_count": len(eval_rows),
            "query_ids": [row["query_id"] for row in eval_rows],
        }
    return out


def main(use_hann_selection=False):
    forced_combo_name = os.environ.get("FEATURE_SELECTION_COMBO", "").strip()
    feature_cols = parse_feature_cols_env()
    feature_space = feature_selection_space_env()
    combos = COMBOS
    if forced_combo_name:
        combos = [combo for combo in COMBOS if combo["name"] == forced_combo_name]
        if not combos:
            raise ValueError(
                f"Unknown combo in FEATURE_SELECTION_COMBO: {forced_combo_name}"
            )

    rows = read_tsv(TSV_PATH)
    output_dir = create_folder(
        OUTPUT_ROOT,
        {
            "feature_selection_space": feature_space,
            "feature_selection_combo": forced_combo_name or None,
            "feature_selection_cols": feature_cols,
        },
    )
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
    rows = [row for row in rows if row["query_id"] not in skipped]

    print("[2/5] embedding with aria")
    query_ids = [row["query_id"] for row in rows]
    embeddings = []
    for query_id in tqdm(query_ids, desc="Embeddings", unit="query"):
        embeddings.append(load_or_compute_embedding(query_id, midi_paths.get(query_id)))
    matrix = np.stack(embeddings, axis=0)
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
    best_score = None
    for combo in combos:
        combo_name = combo["name"]
        print(f"  - {combo_name}")
        combo_input = matrix
        if feature_cols is not None and feature_space == "raw":
            if combo_input.shape[1] == 0:
                raise ValueError("Raw matrix has zero columns")
            max_col = max(feature_cols)
            if max_col >= combo_input.shape[1]:
                raise ValueError(
                    "FEATURE_SELECTION_COLS contains out-of-range index "
                    f"{max_col} for raw dim {combo_input.shape[1]}"
                )
            combo_input = combo_input[:, feature_cols]
        transformed = apply_combo(
            combo_input, labels, combo, target_scores=target_scores
        )
        if feature_cols is not None and feature_space == "transformed":
            if transformed.shape[1] == 0:
                raise ValueError("Transformed matrix has zero columns")
            max_col = max(feature_cols)
            if max_col >= transformed.shape[1]:
                raise ValueError(
                    "FEATURE_SELECTION_COLS contains out-of-range index "
                    f"{max_col} for transformed dim {transformed.shape[1]}"
                )
            transformed = transformed[:, feature_cols]
        payload = {
            "asap": evaluate_dataset(
                "asap", transformed, rows, query_to_index, "axis_side"
            ),
            "music21": evaluate_dataset(
                "music21", transformed, rows, query_to_index, "axis_side"
            ),
            "ewld": evaluate_dataset(
                "ewld", transformed, rows, query_to_index, "axis_side"
            ),
            "overall": evaluate_dataset(
                "overall", transformed, rows, query_to_index, "overall_side"
            ),
        }
        raw_scores = []
        mapped_l1_ratios = []
        for dataset in ["asap", "music21", "ewld", "overall"]:
            if payload[dataset] is None:
                continue
            for mapping_name in ["raw", "linear", "log_linear", "exp_linear"]:
                metrics = payload[dataset][mapping_name]
                metric_rows.append(
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
                if mapping_name == "raw" and not np.isnan(metrics["spearman"]):
                    raw_scores.append(abs(metrics["spearman"]))
                if mapping_name != "raw" and not np.isnan(metrics["l1"]):
                    mapped_l1_ratios.append(metrics["l1"])
        score = combo_selection_score(payload, use_hann_selection)
        random_values = [
            payload[dataset]["raw"]["random"]
            for dataset in ["asap", "music21", "ewld", "overall"]
            if payload[dataset] is not None
        ]
        hann_score_values = [
            payload[dataset][mapping_name]["hann_score"]
            for dataset in ["asap", "music21", "ewld", "overall"]
            if payload[dataset] is not None
            for mapping_name in ["linear", "log_linear", "exp_linear"]
            if not np.isnan(payload[dataset][mapping_name]["hann_score"])
        ]
        combo_scores.append(
            {
                "combo": combo_name,
                "raw_spearman_abs_mean": float(np.mean(raw_scores))
                if raw_scores
                else float("-inf"),
                "mapped_l1_mean": float(np.mean(mapped_l1_ratios))
                if mapped_l1_ratios
                else float("nan"),
                "mapped_hann_score_mean": float(np.mean(hann_score_values))
                if hann_score_values
                else float("nan"),
                "random": float(np.mean(random_values))
                if random_values
                else float("nan"),
            }
        )
        payloads_by_combo[combo_name] = payload
        if best_score is None or score > best_score:
            best_score = score

    print("[4/5] writing outputs")
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
    save_metric_table(
        output_dir / "pearson_table.png",
        combos,
        metric_rows,
        "pearson",
        "Composer chronology Pearson",
    )
    save_metric_table(
        output_dir / "spearman_table.png",
        combos,
        metric_rows,
        "spearman",
        "Composer chronology Spearman",
    )
    save_metric_table(
        output_dir / "l1_table.png",
        combos,
        metric_rows,
        "l1",
        "Composer chronology L1",
    )
    save_metric_table(
        output_dir / "hann_pearson_table.png",
        combos,
        metric_rows,
        "hann_pearson",
        "Composer chronology Hann-weighted Pearson",
    )
    save_metric_table(
        output_dir / "hann_spearman_table.png",
        combos,
        metric_rows,
        "hann_spearman",
        "Composer chronology Hann-weighted Spearman",
    )
    save_metric_table(
        output_dir / "hann_score_table.png",
        combos,
        metric_rows,
        "hann_score",
        "Composer chronology Hann overlap score",
    )

    best_by_dataset = {}
    for dataset in ["asap", "music21", "ewld", "overall"]:
        candidates = [
            (combo_name, payload[dataset])
            for combo_name, payload in payloads_by_combo.items()
            if payload[dataset] is not None
        ]
        if not candidates:
            continue
        best_dataset_combo_name, best_dataset_payload = max(
            candidates,
            key=lambda item: dataset_combo_selection_score(item[1], use_hann_selection),
        )
        best_by_dataset[dataset] = {
            "combo": best_dataset_combo_name,
            "payload": best_dataset_payload,
        }

    projection_rows = []
    for dataset in ["asap", "music21", "ewld", "overall"]:
        if dataset not in best_by_dataset:
            continue
        for mapping in ["raw", "linear", "log_linear", "exp_linear"]:
            metrics = best_by_dataset[dataset]["payload"][mapping]
            title = (
                f"{dataset} | {mapping} | {best_by_dataset[dataset]['combo']}\n"
                f"pearson={metrics['pearson']:.3f}, spearman={metrics['spearman']:.3f}"
            )
            if mapping != "raw":
                title += (
                    f"\nl1={metrics['l1']:.2f}, random={metrics['random']:.2f}, "
                    f"hann={metrics['hann_score']:.3f}"
                )
            save_scatter(
                output_dir / f"scatter_{dataset}_{mapping}.png",
                metrics["real"],
                metrics["predicted"],
                metrics["groups"],
                title,
                "Real value (composer midpoint year)",
                "Projected value" if mapping == "raw" else "Mapped projected value",
            )
            for query_id, real_value, predicted_value, group in zip(
                metrics["query_ids"],
                metrics["real"],
                metrics["predicted"],
                metrics["groups"],
            ):
                projection_rows.append(
                    {
                        "combo": best_by_dataset[dataset]["combo"],
                        "dataset": dataset,
                        "mapping": mapping,
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
    for dataset in ["asap", "music21", "ewld", "overall"]:
        if dataset not in best_by_dataset:
            continue
        raw_metrics = best_by_dataset[dataset]["payload"]["raw"]
        save_projection_strip(
            output_dir / f"projection_strip_{dataset}.png",
            raw_metrics["predicted"],
            raw_metrics["groups"],
            raw_metrics["groups"],
            f"Projection per song | {dataset} | {best_by_dataset[dataset]['combo']}",
            "Song",
        )

    print("[5/5] done")
    print(output_dir)


if __name__ == "__main__":
    main()
