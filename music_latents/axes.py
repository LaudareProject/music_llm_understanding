from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List

import numpy as np

from music_latents.config import AxisSpec, AxesOptions


@dataclass(frozen=True)
class AxisVector:
    name: str
    query_a: str
    query_b: str
    method: str
    params: Dict
    origin: np.ndarray
    direction: np.ndarray


def build_axes(
    axes: Iterable[AxisSpec],
    embeddings: Dict[str, np.ndarray],
    options: AxesOptions,
) -> List[AxisVector]:
    axis_vectors: List[AxisVector] = []

    directions: List[np.ndarray] = []
    origins: List[np.ndarray] = []

    for axis in axes:
        emb_a = embeddings[axis.query_a]
        emb_b = embeddings[axis.query_b]

        direction = _compute_axis_direction(axis, emb_a, emb_b)
        origin = emb_a

        if options.normalize_vectors:
            direction = _normalize_vector(direction, axis.name)

        directions.append(direction)
        origins.append(origin)

        axis_vectors.append(
            AxisVector(
                name=axis.name,
                query_a=axis.query_a,
                query_b=axis.query_b,
                method=axis.method,
                params=axis.params,
                origin=origin,
                direction=direction,
            )
        )

    if not axis_vectors:
        return axis_vectors

    if options.orthonormalize:
        direction_matrix = np.column_stack(directions)
        if direction_matrix.shape[1] > direction_matrix.shape[0]:
            raise ValueError(
                "Cannot orthonormalize more axes than embedding dimensions."
            )

        orthonormal, _ = np.linalg.qr(direction_matrix)

        if options.translate:
            if options.origin_strategy == "mean":
                shared_origin = np.mean(origins, axis=0)
            else:
                shared_origin = origins[0]
        else:
            shared_origin = np.zeros_like(origins[0])

        for idx, axis in enumerate(axis_vectors):
            axis_vectors[idx] = AxisVector(
                name=axis.name,
                query_a=axis.query_a,
                query_b=axis.query_b,
                method=axis.method,
                params=axis.params,
                origin=shared_origin,
                direction=orthonormal[:, idx],
            )

    return axis_vectors


def compute_projection_scores(
    axes: Iterable[AxisVector],
    embeddings: Dict[str, np.ndarray],
    options: AxesOptions,
) -> Dict[str, Dict[str, float]]:
    scores: Dict[str, Dict[str, float]] = {}
    axis_list = list(axes)

    for query_id, vector in embeddings.items():
        query_scores: Dict[str, float] = {}
        for axis in axis_list:
            base = vector - axis.origin if options.translate else vector
            if options.normalize_vectors:
                norm = np.linalg.norm(base)
                if norm == 0:
                    query_scores[axis.name] = 0.0
                    continue
                base = base / norm
            score = float(np.dot(base, axis.direction))
            query_scores[axis.name] = score
        scores[query_id] = query_scores

    return scores


def _compute_axis_direction(
    axis: AxisSpec, emb_a: np.ndarray, emb_b: np.ndarray
) -> np.ndarray:
    if axis.method in {"explicit", "method-name-1"}:
        return emb_b - emb_a

    raise ValueError(f"Unsupported axis method: {axis.method}")


def _normalize_vector(vector: np.ndarray, label: str) -> np.ndarray:
    norm = np.linalg.norm(vector)
    if norm == 0:
        raise ValueError(f"Cannot normalize zero vector for '{label}'.")
    return vector / norm
