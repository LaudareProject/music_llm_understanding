import csv
import os
from pathlib import Path
import shutil
import sys

import numpy as np
from tqdm import tqdm

from experiments.common.embed_aria import encode_midi


FEATURE_BACKEND_ENV = "FEATURE_BACKEND"
FEATURE_BACKEND_ARIA = "aria"
FEATURE_BACKEND_MUSIF = "musif"
MUSIF_FEATURES = [
    "core",
    "ambitus",
    "melody",
    "tempo",
    "density",
    "texture",
    "scale",
    "key",
    "dynamics",
    "rhythm",
    "music21",
    "jsymbolic",
]
MUSIF_IGNORE_COLUMNS = {"FileName", "Id", "WindowId", "WindowRange"}


def feature_backend_name():
    value = os.environ.get(FEATURE_BACKEND_ENV, FEATURE_BACKEND_ARIA).strip().lower()
    if value not in {FEATURE_BACKEND_ARIA, FEATURE_BACKEND_MUSIF}:
        raise ValueError(
            f"{FEATURE_BACKEND_ENV} must be one of ['{FEATURE_BACKEND_ARIA}', '{FEATURE_BACKEND_MUSIF}']"
        )
    return value


def output_root_for_experiment(repo_root, experiment_name):
    backend = feature_backend_name()
    top = "baseline_features" if backend == FEATURE_BACKEND_MUSIF else "aria_embeddings"
    return repo_root / "outputs" / top / experiment_name


def cache_dir_for_experiment(repo_root, experiment_name):
    backend = feature_backend_name()
    if backend == FEATURE_BACKEND_MUSIF:
        return repo_root / "cache" / "baseline_features" / experiment_name
    return repo_root / "cache" / "experiments" / experiment_name


def matrix_for_rows(rows, midi_paths, cache_dir, repo_root, progress_label="Features"):
    backend = feature_backend_name()
    if backend == FEATURE_BACKEND_ARIA:
        return _matrix_for_rows_aria(
            rows, midi_paths, cache_dir, repo_root, progress_label
        )
    return _matrix_for_rows_musif(rows, midi_paths, cache_dir, repo_root)


def _resolve_midi_path_for_row(row, midi_paths, repo_root):
    query_id = row["query_id"]
    path = midi_paths.get(query_id)
    if path is not None:
        return Path(path).resolve()
    return (repo_root / row["path"]).resolve()


def _matrix_for_rows_aria(rows, midi_paths, cache_dir, repo_root, progress_label):
    vectors = []
    for row in tqdm(rows, desc=progress_label, unit="query"):
        query_id = row["query_id"]
        cache_path = cache_dir / f"{query_id}.npy"
        if cache_path.exists():
            vectors.append(np.load(cache_path))
            continue
        midi_path = _resolve_midi_path_for_row(row, midi_paths, repo_root)
        embedding = encode_midi(midi_path)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache_path, embedding)
        vectors.append(embedding)
    return np.stack(vectors, axis=0)


def _dataset_name(row):
    return str(row.get("dataset", "default") or "default")


def _to_float(value):
    text = str(value or "").strip()
    if not text or text in {"-", "nan", "NaN", "None"}:
        return float("nan")
    lowered = text.lower()
    if lowered == "true":
        return 1.0
    if lowered == "false":
        return 0.0
    try:
        return float(text)
    except Exception:
        return float("nan")


def _extract_musif_csv(source_dir, output_csv, cache_dir, repo_root):
    musif_repo = repo_root / "musif"
    if not musif_repo.exists():
        raise FileNotFoundError(
            f"Missing musif repository at {musif_repo}. Add submodule from https://codeberg.org/00sapo/musif"
        )
    sys.path.insert(0, str(musif_repo))
    from musif.__main__ import main as musif_main

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    musif_main(
        source_dir=str(source_dir),
        extension=".mid",
        output_path=str(output_csv),
        cache_dir=str(cache_dir),
        ignore_errors=True,
        njobs=1,
        tweaks={
            "features": MUSIF_FEATURES,
            "basic_modules": ["scoring"],
        },
    )


def _cached_qids_from_musif_csv(csv_path):
    if not csv_path.exists():
        return set()
    qids = set()
    with csv_path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            file_name = str(row.get("FileName", "") or "")
            if not file_name:
                continue
            qids.add(Path(file_name).stem)
    return qids


def _ensure_dataset_musif_cache(dataset_rows, midi_paths, cache_dir, repo_root):
    dataset_name = _dataset_name(dataset_rows[0])
    dataset_dir = cache_dir / "musif" / dataset_name
    source_dir = dataset_dir / "midi"
    csv_path = dataset_dir / "features.csv"
    musif_cache_dir = dataset_dir / "musif_cache"
    source_dir.mkdir(parents=True, exist_ok=True)

    expected_qids = set()
    for row in dataset_rows:
        query_id = row["query_id"]
        expected_qids.add(query_id)
        src = _resolve_midi_path_for_row(row, midi_paths, repo_root)
        dst = source_dir / f"{query_id}.mid"
        if not dst.exists():
            shutil.copy2(src, dst)

    cached_qids = _cached_qids_from_musif_csv(csv_path)
    if expected_qids != cached_qids:
        _extract_musif_csv(source_dir, csv_path, musif_cache_dir, repo_root)
    return csv_path


def _read_dataset_musif_features(csv_path):
    out = {}
    with csv_path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            file_name = str(row.get("FileName", "") or "")
            if not file_name:
                continue
            query_id = Path(file_name).stem
            numeric = {}
            for key, value in row.items():
                if key in MUSIF_IGNORE_COLUMNS:
                    continue
                number = _to_float(value)
                if not np.isnan(number):
                    numeric[key] = number
            out[query_id] = numeric
    return out


def _matrix_for_rows_musif(rows, midi_paths, cache_dir, repo_root):
    by_dataset = {}
    for row in rows:
        by_dataset.setdefault(_dataset_name(row), []).append(row)

    feature_by_query = {}
    for dataset_rows in by_dataset.values():
        csv_path = _ensure_dataset_musif_cache(
            dataset_rows, midi_paths, cache_dir, repo_root
        )
        feature_by_query.update(_read_dataset_musif_features(csv_path))

    columns = sorted(
        {
            key
            for row in rows
            for key in feature_by_query.get(row["query_id"], {}).keys()
        }
    )
    if not columns:
        raise ValueError("No numeric musif features found")

    matrix = np.full((len(rows), len(columns)), np.nan, dtype=np.float64)
    for i, row in enumerate(rows):
        feats = feature_by_query.get(row["query_id"], {})
        for j, column in enumerate(columns):
            if column in feats:
                matrix[i, j] = feats[column]

    for j in range(matrix.shape[1]):
        col = matrix[:, j]
        finite = col[np.isfinite(col)]
        fill_value = float(np.mean(finite)) if finite.size else 0.0
        col[~np.isfinite(col)] = fill_value
        matrix[:, j] = col

    return matrix
