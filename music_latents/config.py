from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml


@dataclass(frozen=True)
class ExperimentSettings:
    name: str
    output_root: Path
    models: Optional[List[str]]


@dataclass(frozen=True)
class QuerySpec:
    query_id: str
    input_kind: str
    value: str


@dataclass(frozen=True)
class ExperimentConfig:
    settings: ExperimentSettings
    models: Dict[str, Dict]
    queries: List[QuerySpec]
    base_dir: Path


class ConfigError(ValueError):
    pass


def load_experiment_config(path: Path) -> ExperimentConfig:
    if not path.exists():
        raise ConfigError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)

    if not isinstance(raw, dict):
        raise ConfigError("Experiment config must be a mapping.")

    settings_raw = raw.get("experiment")
    if not isinstance(settings_raw, dict):
        raise ConfigError("Missing or invalid 'experiment' section.")

    name = settings_raw.get("name")
    if not name or not isinstance(name, str):
        raise ConfigError("Experiment name must be a non-empty string.")

    output_root = settings_raw.get("output_root", "outputs")
    if not isinstance(output_root, str):
        raise ConfigError("experiment.output_root must be a string path.")

    models_setting = settings_raw.get("models")
    if models_setting is not None and not isinstance(models_setting, list):
        raise ConfigError(
            "experiment.models must be a list of model names if provided."
        )

    models = raw.get("models")
    if not isinstance(models, dict) or not models:
        raise ConfigError("Missing or invalid 'models' section.")

    queries_raw = raw.get("queries")
    if not isinstance(queries_raw, list) or not queries_raw:
        raise ConfigError("Missing or invalid 'queries' section.")

    queries = [normalize_query(item) for item in queries_raw]

    settings = ExperimentSettings(
        name=name,
        output_root=Path(output_root),
        models=models_setting,
    )

    return ExperimentConfig(
        settings=settings,
        models=models,
        queries=queries,
        base_dir=path.parent,
    )


def normalize_query(raw: Dict) -> QuerySpec:
    if not isinstance(raw, dict):
        raise ConfigError("Each query must be a mapping.")

    query_id = raw.get("id")
    if not query_id or not isinstance(query_id, str):
        raise ConfigError("Each query must include a non-empty 'id'.")

    input_fields: List[Tuple[str, str]] = []
    for key in ("abc", "midi_path", "musicxml_path", "mei_path", "custom"):
        if key in raw and raw[key] is not None:
            value = raw[key]
            if not isinstance(value, str):
                raise ConfigError(f"Query '{query_id}' field '{key}' must be a string.")
            input_fields.append((key, value))

    if len(input_fields) != 1:
        raise ConfigError(
            f"Query '{query_id}' must define exactly one input field: abc, midi_path, musicxml_path, mei_path, or custom."
        )

    input_kind, value = input_fields[0]
    return QuerySpec(query_id=query_id, input_kind=input_kind, value=value)
