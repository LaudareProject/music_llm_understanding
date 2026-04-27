from collections import defaultdict
import json
import os
from pathlib import Path
import sys

import mido
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.common.tsv import write_tsv
from experiments.common.musif_filter import filter_rows_with_musif_nan


REPO_ROOT = Path(__file__).resolve().parents[2]
VGMIDI_ROOT = Path(os.environ.get("VGMIDI_ROOT", str(REPO_ROOT / "vgmidi"))).resolve()
LABELLED_ROOT = VGMIDI_ROOT / "labelled"
ANNOTATIONS_DIR = LABELLED_ROOT / "annotations"
MIDI_DIR = LABELLED_ROOT / "midi"
OUTPUT_TSV = REPO_ROOT / "experiments" / "valence_axis" / "songs.tsv"
FEATURES_MIDI_DIR = (
    REPO_ROOT / "cache" / "experiments" / "valence_axis" / "prepare_tsv_midi"
)
DATASET_NAME = "vgmidi"


def quantiles5(values):
    if not values:
        return [float("nan")] * 5
    return [
        float(v)
        for v in np.quantile(
            np.asarray(values, dtype=np.float64), [0, 0.25, 0.5, 0.75, 1.0]
        )
    ]


def midi_covariates(midi_path):
    midi = mido.MidiFile(str(midi_path))
    note_pitches = []
    note_durations = []
    active_notes = defaultdict(list)
    tempo = 500000
    elapsed_seconds = 0.0
    for msg in mido.merge_tracks(midi.tracks):
        if msg.time:
            elapsed_seconds += mido.tick2second(msg.time, midi.ticks_per_beat, tempo)
        if msg.type == "set_tempo":
            tempo = msg.tempo
            continue
        if msg.type not in {"note_on", "note_off"}:
            continue
        channel = getattr(msg, "channel", 0)
        note_key = (channel, msg.note)
        if msg.type == "note_on" and msg.velocity > 0:
            note_pitches.append(float(msg.note))
            active_notes[note_key].append(elapsed_seconds)
            continue
        if active_notes[note_key]:
            start_time = active_notes[note_key].pop()
            note_durations.append(max(0.0, elapsed_seconds - start_time))
    pitch_q = quantiles5(note_pitches)
    duration_q = quantiles5(note_durations)
    return {
        "midi_note_count": len(note_pitches),
        "midi_duration_seconds": float(midi.length),
        "midi_track_count": len(midi.tracks),
        "pitch_q0": pitch_q[0],
        "pitch_q25": pitch_q[1],
        "pitch_q50": pitch_q[2],
        "pitch_q75": pitch_q[3],
        "pitch_q100": pitch_q[4],
        "note_duration_q0": duration_q[0],
        "note_duration_q25": duration_q[1],
        "note_duration_q50": duration_q[2],
        "note_duration_q75": duration_q[3],
        "note_duration_q100": duration_q[4],
    }


def format_float(value):
    number = float(value)
    if np.isnan(number):
        return "nan"
    return f"{number:.6f}"


def format_int(value):
    number = float(value)
    if np.isnan(number):
        return "nan"
    return str(int(number))


def as_float_list(values):
    if not isinstance(values, list):
        return []
    out = []
    for value in values:
        try:
            out.append(float(value))
        except Exception:
            continue
    return out


def midi_basename(path_text):
    return Path(str(path_text or "")).name.lower()


def relpath(path):
    return os.path.relpath(path.resolve(), REPO_ROOT)


def build_midi_index():
    index = defaultdict(list)
    for midi_path in MIDI_DIR.rglob("*.mid"):
        index[midi_path.name.lower()].append(midi_path.resolve())
    for midi_path in MIDI_DIR.rglob("*.midi"):
        index[midi_path.name.lower()].append(midi_path.resolve())
    return index


def resolve_midi_path(midi_hint, midi_index):
    hint = str(midi_hint or "").strip()
    if not hint:
        return None
    candidates = [
        MIDI_DIR / hint,
        LABELLED_ROOT / hint,
        VGMIDI_ROOT / hint,
    ]
    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return candidate.resolve()
    basename = midi_basename(hint)
    if basename in midi_index and midi_index[basename]:
        return sorted(midi_index[basename])[0]
    return None


def parse_annotations():
    if not ANNOTATIONS_DIR.exists():
        raise FileNotFoundError(f"Missing annotations directory: {ANNOTATIONS_DIR}")
    if not MIDI_DIR.exists():
        raise FileNotFoundError(f"Missing MIDI directory: {MIDI_DIR}")

    midi_index = build_midi_index()
    aggregated = {}

    for json_path in sorted(ANNOTATIONS_DIR.glob("*.json")):
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        pieces = payload.get("pieces", {})
        annotations = payload.get("annotations", {})
        for annotation_id, annotation in annotations.items():
            piece_id = str(annotation_id).split("_")[0]
            if piece_id not in pieces:
                continue
            piece = pieces[piece_id]
            midi_hint = piece.get("midi", "")
            midi_path = resolve_midi_path(midi_hint, midi_index)
            if midi_path is None:
                continue
            key = relpath(midi_path)
            if key not in aggregated:
                stem = midi_path.stem
                game = stem.split("_")[0] if "_" in stem else stem
                aggregated[key] = {
                    "dataset": DATASET_NAME,
                    "game": game,
                    "title": str(piece.get("name") or stem),
                    "input_kind": "midi",
                    "path": key,
                    "valence_annotator_means": [],
                    "arousal_annotator_means": [],
                }
            valence_curve = as_float_list(annotation.get("valence", []))
            arousal_curve = as_float_list(annotation.get("arousal", []))
            if valence_curve:
                aggregated[key]["valence_annotator_means"].append(
                    float(np.mean(np.asarray(valence_curve, dtype=np.float64)))
                )
            if arousal_curve:
                aggregated[key]["arousal_annotator_means"].append(
                    float(np.mean(np.asarray(arousal_curve, dtype=np.float64)))
                )

    rows = []
    for path_key, row in sorted(aggregated.items(), key=lambda item: item[0].lower()):
        valence_values = np.asarray(row["valence_annotator_means"], dtype=np.float64)
        arousal_values = np.asarray(row["arousal_annotator_means"], dtype=np.float64)
        if valence_values.size == 0 or arousal_values.size == 0:
            continue
        row["valence_mean"] = float(np.mean(valence_values))
        row["valence_std"] = float(np.std(valence_values, ddof=0))
        row["valence_annotator_count"] = int(valence_values.size)
        row["arousal_mean"] = float(np.mean(arousal_values))
        row["arousal_std"] = float(np.std(arousal_values, ddof=0))
        row["arousal_annotator_count"] = int(arousal_values.size)
        rows.append(row)
    return rows


def assign_query_ids(rows):
    counters = defaultdict(int)
    for row in sorted(
        rows, key=lambda item: (item["dataset"], item["game"], item["path"])
    ):
        key = (row["dataset"], row["game"])
        counters[key] += 1
        row["query_id"] = f"{row['dataset']}_{row['game'][:20]}_{counters[key]:04d}"


def assign_midi_covariates(rows):
    FEATURES_MIDI_DIR.mkdir(parents=True, exist_ok=True)
    for row in rows:
        try:
            stats = midi_covariates((REPO_ROOT / row["path"]).resolve())
        except Exception as exc:
            print(f"  midi covariates failed for {row['query_id']}: {exc}")
            stats = {
                "midi_note_count": float("nan"),
                "midi_duration_seconds": float("nan"),
                "midi_track_count": float("nan"),
                "pitch_q0": float("nan"),
                "pitch_q25": float("nan"),
                "pitch_q50": float("nan"),
                "pitch_q75": float("nan"),
                "pitch_q100": float("nan"),
                "note_duration_q0": float("nan"),
                "note_duration_q25": float("nan"),
                "note_duration_q50": float("nan"),
                "note_duration_q75": float("nan"),
                "note_duration_q100": float("nan"),
            }
        row.update(stats)


def main():
    print("[1/3] loading annotations")
    rows = parse_annotations()
    assign_query_ids(rows)

    print("[2/3] filtering rows with musif NaNs")
    rows, dropped = filter_rows_with_musif_nan(rows, REPO_ROOT)
    print(f"  dropped rows with musif NaN values: {len(dropped)}")

    print("[3/3] writing tsv")
    output_rows = []
    for row in sorted(rows, key=lambda item: item["query_id"]):
        output_rows.append(
            {
                "query_id": row["query_id"],
                "dataset": row["dataset"],
                "game": row["game"],
                "title": row["title"],
                "valence_mean": format_float(row["valence_mean"]),
                "valence_std": format_float(row["valence_std"]),
                "valence_annotator_count": row["valence_annotator_count"],
                "arousal_mean": format_float(row["arousal_mean"]),
                "arousal_std": format_float(row["arousal_std"]),
                "arousal_annotator_count": row["arousal_annotator_count"],
                "input_kind": row["input_kind"],
                "path": row["path"],
            }
        )

    write_tsv(
        OUTPUT_TSV,
        output_rows,
        [
            "query_id",
            "dataset",
            "game",
            "title",
            "valence_mean",
            "valence_std",
            "valence_annotator_count",
            "arousal_mean",
            "arousal_std",
            "arousal_annotator_count",
            "input_kind",
            "path",
        ],
    )
    print(f"wrote {OUTPUT_TSV}")
    print(f"rows: {len(output_rows)}")


if __name__ == "__main__":
    main()
