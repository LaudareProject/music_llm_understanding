from datetime import datetime
import json
from pathlib import Path
import sys

import mido
import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.common.tsv import read_tsv, write_tsv
from experiments.common.plots import save_scatter, save_table_png, save_projection_strip
from experiments.common.math1d import (
    apply_combo,
    fit_exp_linear,
    fit_linear,
    fit_log_linear,
    l1,
    pearson,
    project_1d,
    random_l1_baseline,
    spearman,
)
from experiments.common.embed_aria import encode_midi


REPO_ROOT = Path(__file__).resolve().parents[2]
TSV_PATH = REPO_ROOT / "experiments" / "pitch_axis" / "notes.tsv"
OUTPUT_ROOT = REPO_ROOT / "outputs" / "experiments" / "pitch_axis"
CACHE_DIR = OUTPUT_ROOT / "cache"

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


def timestamp_dir():
    out_dir = OUTPUT_ROOT / datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def write_note_midi(midi_path, midi_note):
    midi = mido.MidiFile()
    track = mido.MidiTrack()
    midi.tracks.append(track)
    track.append(mido.Message("program_change", program=0, channel=0, time=0))
    track.append(
        mido.Message("note_on", note=int(midi_note), velocity=64, channel=0, time=0)
    )
    track.append(
        mido.Message("note_off", note=int(midi_note), velocity=64, channel=0, time=480)
    )
    midi.save(str(midi_path))


def load_or_compute_embedding(query_id, midi_path):
    cache_path = CACHE_DIR / f"{query_id}.npy"
    if cache_path.exists():
        return np.load(cache_path)
    embedding = encode_midi(midi_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, embedding)
    return embedding


def main():
    rows = read_tsv(TSV_PATH)
    output_dir = timestamp_dir()
    midi_dir = output_dir / "midi"
    midi_dir.mkdir(parents=True, exist_ok=True)

    print("[1/5] writing midi")
    midi_paths = {}
    for row in rows:
        midi_path = midi_dir / f"{row['query_id']}.mid"
        write_note_midi(midi_path, int(row["midi_note"]))
        midi_paths[row["query_id"]] = midi_path

    print("[2/5] embedding with aria")
    query_ids = [row["query_id"] for row in rows]
    matrix = np.stack(
        [
            load_or_compute_embedding(query_id, midi_paths[query_id])
            for query_id in tqdm(query_ids, desc="Embeddings", unit="query")
        ],
        axis=0,
    )
    labels = [f"octave_{int(row['midi_note']) // 12 - 1}" for row in rows]
    query_to_index = {query_id: i for i, query_id in enumerate(query_ids)}
    np.save(output_dir / "embeddings.npy", matrix)

    anchor_a_idx = [
        query_to_index[row["query_id"]] for row in rows if row["axis_side"] == "a"
    ]
    anchor_b_idx = [
        query_to_index[row["query_id"]] for row in rows if row["axis_side"] == "b"
    ]
    eval_rows = [row for row in rows if row["axis_side"] not in {"a", "b"}]
    eval_idx = [query_to_index[row["query_id"]] for row in eval_rows]
    real = np.asarray([float(row["midi_note"]) for row in eval_rows], dtype=np.float64)
    groups = [row["name"] for row in eval_rows]

    print("[3/5] evaluating combos")
    metric_rows = []
    combo_scores = []
    best_payload = None
    best_score = None
    best_combo_name = None
    for combo in COMBOS:
        combo_name = combo["name"]
        print(f"  - {combo_name}")
        transformed = apply_combo(
            matrix, labels, combo["whiten"], combo["lda"], combo["pca"]
        )
        raw_projection = project_1d(transformed, anchor_a_idx, anchor_b_idx, eval_idx)
        mappings = {
            "raw": raw_projection,
            "linear": fit_linear(raw_projection, real),
            "log_linear": fit_log_linear(raw_projection, real),
            "exp_linear": fit_exp_linear(raw_projection, real),
        }
        baseline_l1 = random_l1_baseline(real)
        payload = {}
        for mapping_name, predicted in mappings.items():
            if mapping_name == "raw":
                mapped_l1 = float("nan")
                mapped_ratio = float("nan")
            else:
                mapped_l1 = l1(real, predicted)
                mapped_ratio = (
                    mapped_l1 / baseline_l1 if baseline_l1 > 0 else float("nan")
                )
            payload[mapping_name] = {
                "real": real,
                "predicted": np.asarray(predicted, dtype=np.float64),
                "groups": groups,
                "pearson": pearson(real, predicted),
                "spearman": spearman(real, predicted),
                "l1": mapped_l1,
                "random_l1": baseline_l1,
                "l1_ratio": mapped_ratio,
                "sample_count": len(eval_rows),
                "query_ids": [row["query_id"] for row in eval_rows],
            }
            metric_rows.append(
                {
                    "combo": combo_name,
                    "mapping": mapping_name,
                    "pearson": payload[mapping_name]["pearson"],
                    "spearman": payload[mapping_name]["spearman"],
                    "l1": payload[mapping_name]["l1"],
                    "random_l1": payload[mapping_name]["random_l1"],
                    "l1_ratio": payload[mapping_name]["l1_ratio"],
                    "sample_count": payload[mapping_name]["sample_count"],
                }
            )
        score = abs(payload["raw"]["spearman"])
        combo_scores.append(
            {
                "combo": combo_name,
                "raw_spearman_abs_mean": score,
                "mapped_l1_ratio_mean": float(
                    np.mean(
                        [
                            payload[name]["l1_ratio"]
                            for name in ["linear", "log_linear", "exp_linear"]
                            if not np.isnan(payload[name]["l1_ratio"])
                        ]
                    )
                ),
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
                "mapping": row["mapping"],
                "pearson": f"{row['pearson']:.10f}"
                if not np.isnan(row["pearson"])
                else "nan",
                "spearman": f"{row['spearman']:.10f}"
                if not np.isnan(row["spearman"])
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
            "mapping",
            "pearson",
            "spearman",
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
                "raw_spearman_abs_mean": f"{row['raw_spearman_abs_mean']:.10f}",
                "mapped_l1_ratio_mean": f"{row['mapped_l1_ratio_mean']:.10f}",
            }
            for row in combo_scores
        ],
        ["combo", "raw_spearman_abs_mean", "mapped_l1_ratio_mean"],
    )
    headers = [
        "combo",
        "raw_spearman",
        "raw_pearson",
        "linear_l1_ratio",
        "log_linear_l1_ratio",
        "exp_linear_l1_ratio",
    ]
    metric_map = {(row["combo"], row["mapping"]): row for row in metric_rows}
    body = []
    for combo in COMBOS:
        combo_name = combo["name"]
        raw_metrics = metric_map[(combo_name, "raw")]
        linear_metrics = metric_map[(combo_name, "linear")]
        log_metrics = metric_map[(combo_name, "log_linear")]
        exp_metrics = metric_map[(combo_name, "exp_linear")]
        line = [
            combo_name,
            "nan"
            if np.isnan(raw_metrics["spearman"])
            else f"{raw_metrics['spearman']:.3f}",
            "nan"
            if np.isnan(raw_metrics["pearson"])
            else f"{raw_metrics['pearson']:.3f}",
            "nan"
            if np.isnan(linear_metrics["l1_ratio"])
            else f"{linear_metrics['l1_ratio']:.3f}",
            "nan"
            if np.isnan(log_metrics["l1_ratio"])
            else f"{log_metrics['l1_ratio']:.3f}",
            "nan"
            if np.isnan(exp_metrics["l1_ratio"])
            else f"{exp_metrics['l1_ratio']:.3f}",
        ]
        body.append(line)
    save_table_png(
        output_dir / "correlations_table.png", headers, body, "Pitch axis summary"
    )

    projection_rows = []
    for mapping_name in ["raw", "linear", "log_linear", "exp_linear"]:
        metrics = best_payload[mapping_name]
        title = (
            f"pitch | {mapping_name} | {best_combo_name}\n"
            f"pearson={metrics['pearson']:.3f}, spearman={metrics['spearman']:.3f}"
        )
        if mapping_name != "raw":
            title += f"\nl1={metrics['l1']:.2f}, random_l1={metrics['random_l1']:.2f}"
        save_scatter(
            output_dir / f"scatter_pitch_{mapping_name}.png",
            metrics["real"],
            metrics["predicted"],
            metrics["groups"],
            title,
            "Real value (MIDI note)",
            "Projected value" if mapping_name == "raw" else "Mapped projected value",
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
                    "mapping": mapping_name,
                    "query_id": query_id,
                    "name": group,
                    "real": f"{float(real_value):.10f}",
                    "predicted": f"{float(predicted_value):.10f}",
                }
            )
    write_tsv(
        output_dir / "projections.tsv",
        projection_rows,
        ["combo", "mapping", "query_id", "name", "real", "predicted"],
    )
    raw_metrics = best_payload["raw"]
    save_projection_strip(
        output_dir / "projection_strip.png",
        raw_metrics["predicted"],
        raw_metrics["groups"],
        raw_metrics["groups"],
        f"Projection per note | {best_combo_name}",
        "Note",
    )
    (output_dir / "best_combo.json").write_text(
        json.dumps(
            {"best_combo": best_combo_name, "combo_scores": combo_scores}, indent=2
        ),
        encoding="utf-8",
    )

    print("[5/5] done")
    print(output_dir)


if __name__ == "__main__":
    main()
