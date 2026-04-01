import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from tqdm import tqdm

from experiments.composer_axis.anchor_selection_strategies import (
    random_after_and_before_median,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.common.embed_aria import encode_midi
from experiments.common.humdrum import kern_to_midi, musicxml_to_midi
from experiments.common.math1d import (
    _hann_weights,
    apply_combo,
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
from experiments.common.plots import save_projection_strip, save_scatter, save_table_png
from experiments.common.tsv import read_tsv, write_tsv

REPO_ROOT = Path(__file__).resolve().parents[2]
TSV_PATH = REPO_ROOT / "experiments" / "composer_axis" / "songs.tsv"
OUTPUT_ROOT = REPO_ROOT / "outputs" / "experiments" / "composer_axis"
CACHE_DIR = REPO_ROOT / "cache" / "experiments" / "composer_axis"
BIN_DIR = REPO_ROOT / "bin"

COMBOS = [
    {"name": "baseline", "whiten": False, "lda": False, "pca": False},
    {"name": "whiten", "whiten": True, "lda": False, "pca": False},
    {"name": "lda", "whiten": False, "lda": True, "pca": False},
    {"name": "pca", "whiten": False, "lda": False, "pca": True},
    {"name": "whiten+lda", "whiten": True, "lda": True, "pca": False},
    {"name": "whiten+pca", "whiten": True, "lda": False, "pca": True},
    # {"name": "lda+pca", "whiten": False, "lda": True, "pca": True},
    # {"name": "whiten+lda+pca", "whiten": True, "lda": True, "pca": True},
]


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


def timestamp_dir():
    out_dir = OUTPUT_ROOT / datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


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


def evaluate_dataset(name, matrix, rows, query_to_index, anchor_column):
    if name == "overall":
        relevant_rows = rows
    else:
        relevant_rows = [row for row in rows if row["dataset"] == name]

    # filter out rows of ewld where year is below 1900
    if name == "ewld":
        relevant_rows = [
            row for row in relevant_rows if float(row["midpoint_year"]) >= 1900
        ]

    anchor_a_ids, anchor_b_ids, eval_rows = random_after_and_before_median(
        relevant_rows, n=10
    )
    # anchor_a_ids, anchor_b_ids, eval_rows = use_anchor_column(
    #     relevant_rows, anchor_column
    # )
    # anchor_a_ids, anchor_b_ids, eval_rows = n_best_pairs_year_separation(
    #     relevant_rows, year_sep=100, n=10
    # )

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
    anchors_a_years = np.asarray(
        [
            float(row["midpoint_year"])
            for row in relevant_rows
            if row["query_id"] in anchor_a_ids
        ],
    )
    anchors_b_years = np.asarray(
        [
            float(row["midpoint_year"])
            for row in relevant_rows
            if row["query_id"] in anchor_b_ids
        ],
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
            mapped_ratio = float("nan")
        else:
            mapped_l1 = l1(real, predicted)
            mapped_ratio = mapped_l1 / baseline_l1 if baseline_l1 > 0 else float("nan")
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
            "random_l1": baseline_l1,
            "l1_ratio": mapped_ratio,
            "sample_count": len(eval_rows),
            "query_ids": [row["query_id"] for row in eval_rows],
            "anchors_a": anchors_a_years,
            "anchors_b": anchors_b_years,
        }
    return out


def main():
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
    output_dir = timestamp_dir()
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
    query_to_index = {query_id: i for i, query_id in enumerate(query_ids)}

    np.save(output_dir / "embeddings.npy", matrix)

    print("[3/5] evaluating combos")
    if forced_combo_name:
        print(f"  forced combo: {forced_combo_name}")
    if feature_cols is not None:
        print(f"  using selected {feature_space} cols: {len(feature_cols)}")
    metric_rows = []
    combo_scores = []
    best_payload = None
    best_score = None
    best_combo_name = None
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
            combo_input, labels, combo["whiten"], combo["lda"], combo["pca"]
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
                        "random_l1": metrics["random_l1"],
                        "l1_ratio": metrics["l1_ratio"],
                        "sample_count": metrics["sample_count"],
                    }
                )
                if mapping_name == "raw" and not np.isnan(metrics["spearman"]):
                    raw_scores.append(abs(metrics["spearman"]))
                if mapping_name != "raw" and not np.isnan(metrics["l1_ratio"]):
                    mapped_l1_ratios.append(metrics["l1_ratio"])
        score = float(np.mean(raw_scores)) if raw_scores else float("-inf")
        combo_scores.append(
            {
                "combo": combo_name,
                "raw_spearman_abs_mean": score,
                "mapped_l1_ratio_mean": float(np.mean(mapped_l1_ratios))
                if mapped_l1_ratios
                else float("nan"),
            }
        )
        if best_score is None or score > best_score:
            best_score = score
            best_payload = payload
            best_combo_name = combo_name

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
                "random_l1": f"{row['random_l1']:.10f}"
                if not np.isnan(row["random_l1"])
                else "nan",
                "l1_ratio": f"{row['l1_ratio']:.10f}"
                if not np.isnan(row["l1_ratio"])
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
            "random_l1",
            "l1_ratio",
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
                "mapped_l1_ratio_mean": f"{row['mapped_l1_ratio_mean']:.10f}"
                if not np.isnan(row["mapped_l1_ratio_mean"])
                else "nan",
            }
            for row in combo_scores
        ],
        ["combo", "raw_spearman_abs_mean", "mapped_l1_ratio_mean"],
    )

    spearman_map = {
        (row["combo"], row["dataset"], row["mapping"]): row["spearman"]
        for row in metric_rows
    }
    hann_pearson_map = {
        (row["combo"], row["dataset"], row["mapping"]): row["hann_pearson"]
        for row in metric_rows
    }
    hann_spearman_map = {
        (row["combo"], row["dataset"], row["mapping"]): row["hann_spearman"]
        for row in metric_rows
    }
    hann_score_map = {
        (row["combo"], row["dataset"], row["mapping"]): row["hann_score"]
        for row in metric_rows
    }
    score_map = {row["combo"]: row["raw_spearman_abs_mean"] for row in combo_scores}

    def _table_body(value_map):
        body = []
        for combo in combos:
            combo_name = combo["name"]
            line = [
                combo_name,
                f"{score_map[combo_name]:.3f}"
                if not np.isnan(score_map[combo_name])
                else "nan",
            ]
            for dataset in ["asap", "music21", "ewld", "overall"]:
                for mapping in ["raw", "linear", "log_linear", "exp_linear"]:
                    value = value_map.get((combo_name, dataset, mapping), float("nan"))
                    line.append("nan" if np.isnan(value) else f"{value:.3f}")
            body.append(line)
        return body

    col_headers = ["combo", "raw_spear_mean"] + [
        f"{dataset}:{mapping}"
        for dataset in ["asap", "music21", "ewld", "overall"]
        for mapping in ["raw", "linear", "log_linear", "exp_linear"]
    ]
    save_table_png(
        output_dir / "correlations_table.png",
        col_headers,
        _table_body(spearman_map),
        "Composer chronology correlations (Spearman)",
    )
    save_table_png(
        output_dir / "hann_pearson_table.png",
        col_headers,
        _table_body(hann_pearson_map),
        "Composer chronology Hann-weighted Pearson",
    )
    save_table_png(
        output_dir / "hann_spearman_table.png",
        col_headers,
        _table_body(hann_spearman_map),
        "Composer chronology Hann-weighted Spearman",
    )
    save_table_png(
        output_dir / "hann_score_table.png",
        col_headers,
        _table_body(hann_score_map),
        "Composer chronology Hann scores",
    )

    projection_rows = []
    for dataset in ["asap", "music21", "ewld", "overall"]:
        for mapping in ["raw", "linear", "log_linear", "exp_linear"]:
            if best_payload[dataset] is None:
                continue
            metrics = best_payload[dataset][mapping]
            title = (
                f"{dataset} | {mapping} | {best_combo_name}\n"
                f"pearson={metrics['pearson']:.3f}, spearman={metrics['spearman']:.3f}"
            )
            if mapping != "raw":
                title += (
                    f"\nl1={metrics['l1']:.2f}, random_l1={metrics['random_l1']:.2f}"
                )
            save_scatter(
                output_dir / f"scatter_{dataset}_{mapping}.png",
                metrics["real"],
                metrics["predicted"],
                metrics["groups"],
                title,
                "Real value (composer midpoint year)",
                "Projected value" if mapping == "raw" else "Mapped projected value",
                ref_line=mapping != "raw",
                anchors_a=metrics["anchors_a"] if mapping != "raw" else None,
                anchors_b=metrics["anchors_b"] if mapping != "raw" else None,
            )
            for query_id, real_value, predicted_value, group in zip(
                metrics["query_ids"],
                metrics["real"],
                metrics["predicted"],
                metrics["groups"],
            ):
                projection_rows.append(
                    {
                        "combo": best_combo_name,
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
        if best_payload[dataset] is None:
            continue
        raw_metrics = best_payload[dataset]["raw"]
        save_projection_strip(
            output_dir / f"projection_strip_{dataset}.png",
            raw_metrics["predicted"],
            raw_metrics["groups"],
            raw_metrics["groups"],
            f"Projection per song | {dataset} | {best_combo_name}",
            "Song",
        )

    print("[5/5] done")
    print(output_dir)


if __name__ == "__main__":
    main()
