from __future__ import annotations

from pathlib import Path

import yaml

from music_latents.config_axes import (
    normalize_axis,
    normalize_axes_options,
    normalize_plots,
)
from music_latents.config_models import (
    ConfigError,
    ExperimentConfig,
    ExperimentSettings,
)
from music_latents.config_queries import normalize_averages, normalize_queries
from music_latents.config_supervised import normalize_supervised


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

    queries = normalize_queries(queries_raw)
    query_ids = {query.query_id for query in queries}
    if len(query_ids) != len(queries):
        raise ConfigError("Query ids must be unique.")

    averages = normalize_averages(raw.get("average"), queries)
    medians = normalize_averages(raw.get("median"), queries, method="median")
    aggregate_ids = {group.group_id for group in averages + medians}

    axes_raw = raw.get("axes", [])
    if axes_raw is None:
        axes_raw = []
    if not isinstance(axes_raw, list):
        raise ConfigError("'axes' must be a list if provided.")

    axes = [normalize_axis(item, query_ids | aggregate_ids) for item in axes_raw]
    axis_names = [axis.name for axis in axes]
    if len(axis_names) != len(set(axis_names)):
        raise ConfigError("Axis names must be unique.")

    default_plot_options = normalize_axes_options(raw.get("axes_options"))
    plots = normalize_plots(raw.get("plots"), axis_names, default_plot_options)

    settings = ExperimentSettings(
        name=name,
        output_root=Path(output_root),
        models=models_setting,
    )

    return ExperimentConfig(
        settings=settings,
        models=models,
        queries=queries,
        averages=averages,
        medians=medians,
        axes=axes,
        plots=plots,
        supervised=normalize_supervised(raw.get("supervised")),
        base_dir=path.parent,
    )
