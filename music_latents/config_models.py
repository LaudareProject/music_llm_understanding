from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class ExperimentSettings:
    name: str
    output_root: Path
    models: Optional[List[str]]


@dataclass(frozen=True)
class QuerySpec:
    query_id: str
    name: str
    color: Optional[str]
    target: Optional[Any]
    input_kind: str
    value: Any


@dataclass(frozen=True)
class AxisSpec:
    name: str
    query_a: str
    query_b: str
    method: str
    params: Dict[str, Any]


@dataclass(frozen=True)
class AxesOptions:
    center_and_whiten: bool
    translate: bool
    orthonormalize: bool
    normalize_vectors: bool
    origin_strategy: str
    pca: bool
    lda_groups: bool
    averaged_only: bool
    group_filter: str


@dataclass(frozen=True)
class PlotSpec:
    mode: str
    axes: List[str]
    cosine_references: Optional[List[str]]
    options: AxesOptions


@dataclass(frozen=True)
class AverageGroupSpec:
    group_id: str
    query_ids: List[str]
    name: str
    color: Optional[str]
    method: str


@dataclass(frozen=True)
class ExperimentConfig:
    settings: ExperimentSettings
    models: Dict[str, Dict]
    queries: List[QuerySpec]
    averages: List[AverageGroupSpec]
    medians: List[AverageGroupSpec]
    axes: List[AxisSpec]
    plots: Dict[str, PlotSpec]
    supervised: Optional[Dict[str, Any]]
    base_dir: Path


class ConfigError(ValueError):
    pass


PCA_COMPONENT_AXIS_RE = re.compile(r"^pca(\d{2,})$")
