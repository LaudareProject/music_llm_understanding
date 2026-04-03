import os
from pathlib import Path

import mido
import numpy as np
from tqdm import tqdm

from experiments.common.anchor_selection import select_anchors
from experiments.common.embed_aria import encode_midi
from experiments.common.folder_handling import create_folder
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
TSV_PATH = REPO_ROOT / "experiments" / "pitch_axis" / "notes.tsv"
OUTPUT_ROOT = REPO_ROOT / "outputs" / "experiments" / "pitch_axis"
CACHE_DIR = REPO_ROOT / "cache" / "experiments" / "pitch_axis"

MAPPINGS = ["raw", "linear", "log_linear", "exp_linear"]


def write_note_midi(midi_path, midi_note, duration_ticks, velocity):
    midi = mido.MidiFile()
    track = mido.MidiTrack()
    midi.tracks.append(track)
    track.append(mido.Message("program_change", program=0, channel=0, time=0))
    track.append(
        mido.Message(
            "note_on", note=int(midi_note), velocity=int(velocity), channel=0, time=0
        )
    )
    track.append(
        mido.Message(
            "note_off",
            note=int(midi_note),
            velocity=int(velocity),
            channel=0,
            time=int(duration_ticks),
        )
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


def combo_selection_score(payload):
    scores = [
        -payload[mapping_name]["l1"]
        for mapping_name in MAPPINGS[1:]
        if not np.isnan(payload[mapping_name]["l1"])
    ]
    return max(scores) if scores else float("-inf")


def prepare_rows(output_dir):
    rows = read_tsv(TSV_PATH)
    midi_dir = output_dir / "midi"
    midi_dir.mkdir(parents=True, exist_ok=True)
    print("[1/5] writing midi")
    midi_paths = {}
    for row in rows:
        midi_path = midi_dir / f"{row['query_id']}.mid"
        write_note_midi(
            midi_path,
            int(row["midi_note"]),
            int(row["duration_ticks"]),
            int(row["velocity"]),
        )
        midi_paths[row["query_id"]] = midi_path
    return rows, midi_paths


def build_payload(transformed, rows, query_to_index, anchor_selection_method, anchor_n):
    anchor_a_ids, anchor_b_ids, eval_rows = select_anchors(
        rows,
        anchor_selection_method,
        target_key="midi_note",
        n=anchor_n,
        best_pairs_separation=12,
    )
    anchor_a_idx = [query_to_index[query_id] for query_id in anchor_a_ids]
    anchor_b_idx = [query_to_index[query_id] for query_id in anchor_b_ids]
    eval_idx = [query_to_index[row["query_id"]] for row in eval_rows]
    real = np.asarray([float(row["midi_note"]) for row in eval_rows], dtype=np.float64)
    if anchor_selection_method == "random_sample_regression":
        sample_ids = anchor_a_ids + anchor_b_ids
        sample_idx = [query_to_index[query_id] for query_id in sample_ids]
        query_to_row = {row["query_id"]: row for row in rows}
        sample_targets = [
            float(query_to_row[query_id]["midi_note"]) for query_id in sample_ids
        ]
        raw_projection = project_1d_regression_direction(
            transformed,
            sample_idx,
            sample_targets,
            eval_idx,
        )
    else:
        raw_projection = project_1d(transformed, anchor_a_idx, anchor_b_idx, eval_idx)
    groups = [row["name"] for row in eval_rows]
    cv_groups = np.asarray([row["midi_note"] for row in eval_rows])
    anchors_a = np.asarray(
        [float(row["midi_note"]) for row in rows if row["query_id"] in anchor_a_ids],
        dtype=np.float64,
    )
    anchors_b = np.asarray(
        [float(row["midi_note"]) for row in rows if row["query_id"] in anchor_b_ids],
        dtype=np.float64,
    )
    baseline_l1 = random_l1_baseline(real)
    payload = {}
    for mapping_name in MAPPINGS:
        summary = cross_validated_mapping_summary(
            raw_projection, real, cv_groups, mapping_name
        )
        payload[mapping_name] = {
            "real": real,
            "predicted": summary["lolo_predicted"],
            "averaged_model_predicted": summary["predicted"],
            "groups": groups,
            "pearson": summary["pearson"],
            "pearson_ci95": summary["pearson_ci95"],
            "spearman": summary["spearman"],
            "spearman_ci95": summary["spearman_ci95"],
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
    return [
        {
            "combo": combo_name,
            "mapping": mapping_name,
            "pearson": payload[mapping_name]["pearson"],
            "pearson_ci95": payload[mapping_name]["pearson_ci95"],
            "spearman": payload[mapping_name]["spearman"],
            "spearman_ci95": payload[mapping_name]["spearman_ci95"],
            "l1": payload[mapping_name]["l1"],
            "l1_ci95": payload[mapping_name]["l1_ci95"],
            "random": payload[mapping_name]["random"],
            "sample_count": payload[mapping_name]["sample_count"],
            "fold_count": payload[mapping_name]["fold_count"],
        }
        for mapping_name in MAPPINGS
    ]


def combo_score_row(combo_name, payload):
    mapped_l1_values = [
        payload[mapping_name]["l1"]
        for mapping_name in MAPPINGS[1:]
        if not np.isnan(payload[mapping_name]["l1"])
    ]
    return {
        "combo": combo_name,
        "raw_spearman_abs_mean": abs(payload["raw"]["spearman"]),
        "mapped_l1_mean": float(np.mean(mapped_l1_values)),
        "random": float(
            np.mean([payload[mapping_name]["random"] for mapping_name in MAPPINGS[1:]])
        ),
    }


def scatter_title(combo_name, mapping_name, metrics):
    title = (
        f"pitch | {mapping_name} | {combo_name}\n"
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
    best_combo_name,
    best_payload,
):
    write_tsv(
        output_dir / "metrics.tsv",
        [
            {
                "combo": row["combo"],
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
            "mapping",
            "pearson",
            "pearson_ci95",
            "spearman",
            "spearman_ci95",
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
                "raw_spearman_abs_mean": f"{row['raw_spearman_abs_mean']:.10f}",
                "mapped_l1_mean": f"{row['mapped_l1_mean']:.10f}",
                "random": f"{row['random']:.10f}",
            }
            for row in combo_scores
        ],
        ["combo", "raw_spearman_abs_mean", "mapped_l1_mean", "random"],
    )
    column_specs = [(mapping_name, (mapping_name,)) for mapping_name in MAPPINGS]
    save_metric_table(
        output_dir / "pearson_table.png",
        combos,
        metric_rows,
        ["combo", "mapping"],
        column_specs,
        "pearson",
        "Pitch axis Pearson",
    )
    save_metric_table(
        output_dir / "spearman_table.png",
        combos,
        metric_rows,
        ["combo", "mapping"],
        column_specs,
        "spearman",
        "Pitch axis Spearman",
    )
    save_metric_table(
        output_dir / "l1_table.png",
        combos,
        metric_rows,
        ["combo", "mapping"],
        column_specs,
        "l1",
        "Pitch axis L1",
    )

    scatter_dir = output_dir / "scatters_by_combo"
    for combo_name, payload in payloads_by_combo.items():
        for mapping_name in MAPPINGS:
            metrics = payload[mapping_name]
            save_scatter(
                scatter_dir / f"scatter_pitch__{combo_name}__{mapping_name}.png",
                metrics["real"],
                metrics["predicted"],
                metrics["groups"],
                scatter_title(combo_name, mapping_name, metrics),
                "Real value (MIDI note)",
                "Projected value"
                if mapping_name == "raw"
                else "Mapped projected value",
                ref_line=mapping_name != "raw",
                anchors_a=metrics["anchors_a"] if mapping_name != "raw" else None,
                anchors_b=metrics["anchors_b"] if mapping_name != "raw" else None,
            )

    projection_rows = []
    for mapping_name in MAPPINGS:
        metrics = best_payload[mapping_name]
        save_scatter(
            output_dir / f"scatter_pitch_{mapping_name}.png",
            metrics["real"],
            metrics["predicted"],
            metrics["groups"],
            scatter_title(best_combo_name, mapping_name, metrics),
            "Real value (MIDI note)",
            "Projected value" if mapping_name == "raw" else "Mapped projected value",
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
                    "dataset": "pitch",
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
        ["dataset", "combo", "mapping", "query_id", "name", "real", "predicted"],
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
        },
    )
    rows, midi_paths = prepare_rows(output_dir)

    print("[2/5] embedding with aria")
    query_ids = [row["query_id"] for row in rows]
    matrix = np.stack(
        [
            load_or_compute_embedding(query_id, midi_paths[query_id])
            for query_id in tqdm(query_ids, desc="Embeddings", unit="query")
        ],
        axis=0,
    )
    labels = [f"pitch_{int(row['midi_note'])}" for row in rows]
    target_scores = np.asarray(
        [float(row["midi_note"]) for row in rows], dtype=np.float64
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
    best_payload = None
    best_score = None
    best_combo_name = None
    for combo in combos:
        print(f"  - {combo['name']}")
        transformed = transform_with_feature_selection(
            matrix, labels, combo, target_scores, feature_cols, feature_space
        )
        payload = build_payload(
            transformed, rows, query_to_index, anchor_selection_method, anchor_n
        )
        payloads_by_combo[combo["name"]] = payload
        metric_rows.extend(metric_rows_for_combo(combo["name"], payload))
        combo_scores.append(combo_score_row(combo["name"], payload))
        score = combo_selection_score(payload)
        if best_score is None or score > best_score:
            best_score = score
            best_payload = payload
            best_combo_name = combo["name"]

    print("[4/5] writing outputs")
    write_outputs(
        output_dir,
        combos,
        metric_rows,
        combo_scores,
        payloads_by_combo,
        best_combo_name,
        best_payload,
    )
    print("[5/5] done")
    print(output_dir)
    return output_dir
