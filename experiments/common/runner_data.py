from __future__ import annotations

from pathlib import Path

import mido

from experiments.common.feature_backend import aria_embedding_path_for_row
from experiments.common.humdrum import kern_to_midi, musicxml_to_midi
from experiments.common.tsv import read_tsv


def _normalize_midi_to_piano(midi_path):
    midi = mido.MidiFile(str(midi_path))
    changed = False
    for track in midi.tracks:
        for msg in track:
            if msg.is_meta:
                continue
            if hasattr(msg, "channel") and int(msg.channel) != 0:
                msg.channel = 0
                changed = True
            if msg.type == "program_change" and int(msg.program) != 0:
                msg.program = 0
                changed = True
    if changed:
        midi.save(str(midi_path))


def _write_pitch_note_midi(row, midi_path):
    midi_note = int(float(row["midi_note"]))
    duration_ticks = max(1, int(float(row["duration_ticks"])))
    velocity = int(float(row["velocity"]))
    midi = mido.MidiFile(ticks_per_beat=480)
    track = mido.MidiTrack()
    track.append(mido.MetaMessage("set_tempo", tempo=500_000, time=0))
    track.append(
        mido.Message("note_on", note=midi_note, velocity=velocity, time=0, channel=0)
    )
    track.append(
        mido.Message(
            "note_off", note=midi_note, velocity=0, time=duration_ticks, channel=0
        )
    )
    midi.tracks.append(track)
    midi_path.parent.mkdir(parents=True, exist_ok=True)
    midi.save(str(midi_path))


def _prepare_midi_for_row(row, repo_root, midi_dir, bin_dir):
    query_id = row["query_id"]
    midi_path = midi_dir / f"{query_id}.mid"
    source_path_value = row.get("path")
    if source_path_value:
        source_path = (repo_root / source_path_value).resolve()
        if row["input_kind"] == "midi":
            _normalize_midi_to_piano(source_path)
            return source_path
        if row["input_kind"] == "musicxml":
            musicxml_to_midi(source_path, midi_path, bin_dir)
            _normalize_midi_to_piano(midi_path)
            return midi_path
        if row["input_kind"] == "kern":
            kern_to_midi(source_path, midi_path, bin_dir)
            _normalize_midi_to_piano(midi_path)
            return midi_path
        raise ValueError(row["input_kind"])

    if row.get("dataset") == "pitch":
        _write_pitch_note_midi(row, midi_path)
        _normalize_midi_to_piano(midi_path)
        return midi_path

    raise ValueError(f"No source path or pitch generator for {query_id}")


def prepare_rows_and_midi(
    experiment_name,
    tsv_path,
    repo_root,
    output_dir,
    row_filter_col=None,
    row_filter_value=None,
):
    rows = read_tsv(tsv_path)
    if row_filter_col is not None:
        rows = [
            row
            for row in rows
            if str(row.get(row_filter_col, "")).strip() == str(row_filter_value)
        ]
    midi_dir = output_dir / "midi"
    midi_dir.mkdir(parents=True, exist_ok=True)
    bin_dir = repo_root / "bin"
    midi_paths = {}
    skipped = set()
    for row in rows:
        query_id = row["query_id"]
        cache_embedding = aria_embedding_path_for_row(row, repo_root)
        if cache_embedding.exists():
            continue
        try:
            midi_paths[query_id] = _prepare_midi_for_row(
                row, repo_root, midi_dir, bin_dir
            )
        except Exception as exc:
            print(f"  skipping {experiment_name}:{query_id}: {exc}")
            skipped.add(query_id)
    filtered = [row for row in rows if row["query_id"] not in skipped]
    return filtered, midi_paths
