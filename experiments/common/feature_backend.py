import os
from pathlib import Path
import sys

import numpy as np
from tqdm import tqdm

from experiments.common.embed_aria import encode_midi
from experiments.common.musif_filter import read_musif_features_by_query


FEATURE_BACKEND_ENV = "FEATURE_BACKEND"
FEATURE_BACKEND_ARIA = "aria"
FEATURE_BACKEND_MUSIF = "musif"
FEATURE_BACKEND_BEAT = "beat"
FEATURE_BACKEND_CLAMP3_C2 = "clamp3_c2"
FEATURE_BACKEND_ADVERSARIAL_MIDIBERT = "adversarial_midibert"
FEATURE_BACKEND_MIDIGPT = "midigpt"
EMBEDDING_BACKENDS = (
    FEATURE_BACKEND_ARIA,
    FEATURE_BACKEND_BEAT,
    FEATURE_BACKEND_CLAMP3_C2,
    FEATURE_BACKEND_ADVERSARIAL_MIDIBERT,
    FEATURE_BACKEND_MIDIGPT,
)
FEATURE_BACKENDS = (*EMBEDDING_BACKENDS, FEATURE_BACKEND_MUSIF)
BACKEND_OUTPUT_DIRS = {
    FEATURE_BACKEND_ARIA: "aria_embeddings",
    FEATURE_BACKEND_MUSIF: "baseline_features",
    FEATURE_BACKEND_BEAT: "beat_embeddings",
    FEATURE_BACKEND_CLAMP3_C2: "clamp3_c2_embeddings",
    FEATURE_BACKEND_ADVERSARIAL_MIDIBERT: "adversarial_midibert_embeddings",
    FEATURE_BACKEND_MIDIGPT: "midigpt_embeddings",
}
BACKEND_ENCODERS = {
    FEATURE_BACKEND_BEAT: ("experiments.common.embed_beat", "encode_midi"),
    FEATURE_BACKEND_CLAMP3_C2: ("experiments.common.embed_clamp3", "encode_midi"),
    FEATURE_BACKEND_ADVERSARIAL_MIDIBERT: (
        "experiments.common.embed_adversarial_midibert",
        "encode_midi",
    ),
    FEATURE_BACKEND_MIDIGPT: ("experiments.common.embed_midigpt", "encode_midi"),
}
_LOADED_ENCODERS = {}
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


def feature_backend_name():
    value = os.environ.get(FEATURE_BACKEND_ENV, FEATURE_BACKEND_ARIA).strip().lower()
    if value not in FEATURE_BACKENDS:
        choices = ", ".join(f"'{name}'" for name in FEATURE_BACKENDS)
        raise ValueError(f"{FEATURE_BACKEND_ENV} must be one of [{choices}]")
    return value


def output_root_for_experiment(repo_root, experiment_name):
    backend = feature_backend_name()
    return repo_root / "outputs" / BACKEND_OUTPUT_DIRS[backend] / experiment_name


def cache_dir_for_experiment(repo_root, experiment_name):
    backend = feature_backend_name()
    if backend == FEATURE_BACKEND_MUSIF:
        return repo_root / "cache" / "baseline_features" / experiment_name
    return repo_root / "cache" / "embeddings" / backend


def embedding_cache_path_for_row(row, repo_root, backend=None):
    backend = backend or feature_backend_name()
    dataset_name = _dataset_name(row)
    return (
        repo_root
        / "cache"
        / "embeddings"
        / backend
        / dataset_name
        / f"{row['query_id']}.npy"
    )

def musif_data_dir(repo_root):
    return repo_root / "data" / "musif"


def musif_features_csv_path(repo_root, dataset_name):
    return musif_data_dir(repo_root) / dataset_name / "features.csv"


def musif_cache_dir(repo_root, dataset_name):
    return repo_root / "cache" / "musif_features" / dataset_name


def musif_sources_dir(repo_root, dataset_name):
    return repo_root / "cache" / "musif_sources" / dataset_name / "midi"


def run_musif_extraction(source_dir, output_csv, cache_dir, repo_root, njobs=2):
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
        njobs=njobs,
        tweaks={
            "max_nan_columns": 1000,
            "features": MUSIF_FEATURES,
            "basic_modules": ["scoring"],
            "jsymbolic_max_ram": os.environ.get("JSYMBOLIC_MAX_RAM", "50g"),
        },
    )

def matrix_for_rows(rows, midi_paths, cache_dir, repo_root, progress_label="Features"):
    backend = feature_backend_name()
    if backend in EMBEDDING_BACKENDS:
        return _matrix_for_rows_embeddings(
            rows, midi_paths, repo_root, progress_label, backend
        )
    return _matrix_for_rows_musif(rows, midi_paths, cache_dir, repo_root)


def _encoder_for_backend(backend):
    if backend == FEATURE_BACKEND_ARIA:
        return encode_midi
    if backend not in _LOADED_ENCODERS:
        import importlib

        try:
            module_name, function_name = BACKEND_ENCODERS[backend]
        except KeyError as exc:
            raise NotImplementedError(
                f"No official adapter is available for backend '{backend}'; its model/checkpoint identity is unresolved"
            ) from exc
        _LOADED_ENCODERS[backend] = getattr(
            importlib.import_module(module_name), function_name
        )
    return _LOADED_ENCODERS[backend]


def _matrix_for_rows_embeddings(rows, midi_paths, repo_root, progress_label, backend):
    vectors = []
    encoder = _encoder_for_backend(backend)
    for row in tqdm(rows, desc=progress_label, unit="query"):
        cache_path = (
            aria_embedding_path_for_row(row, repo_root)
            if backend == FEATURE_BACKEND_ARIA
            else embedding_cache_path_for_row(row, repo_root, backend)
        )
        if cache_path.exists():
            embedding = np.load(cache_path, allow_pickle=False)
        else:
            midi_path = _resolve_midi_path_for_row(row, midi_paths, repo_root)
            embedding = np.asarray(encoder(midi_path), dtype=np.float64).reshape(-1)
            if not embedding.size or not np.isfinite(embedding).all():
                raise ValueError(
                    f"Backend '{backend}' returned an empty or non-finite embedding "
                    f"for {row['query_id']}"
                )
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(cache_path, embedding, allow_pickle=False)
        vector = np.asarray(embedding, dtype=np.float64).reshape(-1)
        if not vector.size or not np.isfinite(vector).all():
            raise ValueError(f"Invalid cached embedding for {row['query_id']}")
        vectors.append(vector)
    if not vectors:
        raise ValueError("Cannot build an embedding matrix from zero rows")
    dimensions = {vector.shape for vector in vectors}
    if len(dimensions) != 1:
        raise ValueError(f"Embedding dimension mismatch for backend '{backend}'")
    return np.stack(vectors, axis=0)

def _resolve_midi_path_for_row(row, midi_paths, repo_root):
    path = midi_paths.get(row["query_id"])
    if path is not None:
        return Path(path).resolve()
    return (repo_root / row["path"]).resolve()



def _dataset_name(row):
    return str(row.get("dataset", "default") or "default")


def aria_dataset_embeddings_dir(repo_root, dataset_name):
    return repo_root / "data" / dataset_name / "aria_embeddings"


def aria_embedding_path_for_row(row, repo_root):
    dataset_name = _dataset_name(row)
    dataset_dir = aria_dataset_embeddings_dir(repo_root, dataset_name)
    candidate = dataset_dir / f"{row['query_id']}.npy"
    if candidate.exists():
        return candidate
    if dataset_name != "default":
        fallback_dir = aria_dataset_embeddings_dir(repo_root, "default")
        fallback = fallback_dir / f"{row['query_id']}.npy"
        if fallback.exists():
            return fallback
    return candidate


def _matrix_for_rows_musif(rows, midi_paths, cache_dir, repo_root):
    by_dataset = {}
    for row in rows:
        by_dataset.setdefault(_dataset_name(row), []).append(row)

    feature_by_query = {}
    for dataset_rows in by_dataset.values():
        dataset_name = _dataset_name(dataset_rows[0])
        csv_path = musif_features_csv_path(repo_root, dataset_name)
        if not csv_path.exists():
            raise FileNotFoundError(
                (
                    f"Musif features missing for dataset '{dataset_name}'. "
                    f"Run scripts/extract_musif_features.py --dataset {dataset_name} --tsv path/to/tsv"
                )
            )
        feature_by_query.update(read_musif_features_by_query(csv_path))

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
