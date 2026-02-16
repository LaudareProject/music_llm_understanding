from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml


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
    input_kind: str
    value: str


@dataclass(frozen=True)
class AxisSpec:
    name: str
    query_a: str
    query_b: str
    method: str
    params: Dict[str, Any]


@dataclass(frozen=True)
class AxesOptions:
    translate: bool
    orthonormalize: bool
    normalize_vectors: bool
    origin_strategy: str
    pca: bool
    averaged_only: bool


@dataclass(frozen=True)
class PlotSpec:
    axes: List[str]
    options: AxesOptions


@dataclass(frozen=True)
class AverageGroupSpec:
    group_id: str
    query_ids: List[str]
    name: str
    color: Optional[str]


@dataclass(frozen=True)
class ExperimentConfig:
    settings: ExperimentSettings
    models: Dict[str, Dict]
    queries: List[QuerySpec]
    averages: List[AverageGroupSpec]
    axes: List[AxisSpec]
    plots: Dict[str, PlotSpec]
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

    queries = normalize_queries(queries_raw)
    query_ids = {query.query_id for query in queries}
    if len(query_ids) != len(queries):
        raise ConfigError("Query ids must be unique.")

    averages = normalize_averages(raw.get("average"), queries)
    average_ids = {group.group_id for group in averages}

    axes_raw = raw.get("axes", [])
    if axes_raw is None:
        axes_raw = []
    if not isinstance(axes_raw, list):
        raise ConfigError("'axes' must be a list if provided.")

    axes = [normalize_axis(item, query_ids | average_ids) for item in axes_raw]
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
        axes=axes,
        plots=plots,
        base_dir=path.parent,
    )


def normalize_queries(raw_items: List[Any]) -> List[QuerySpec]:
    queries: List[QuerySpec] = []
    for item in raw_items:
        if not isinstance(item, dict):
            raise ConfigError("Each query must be a mapping.")

        if "id" in item:
            queries.append(normalize_query(item, inherited_color=None))
            continue

        nested_queries = item.get("queries")
        if nested_queries is None:
            raise ConfigError(
                "Each queries item must be a query mapping or a query block with 'queries'."
            )
        if not isinstance(nested_queries, list) or not nested_queries:
            raise ConfigError("Query block 'queries' must be a non-empty list.")

        block_color = item.get("color")
        if block_color is not None and (
            not isinstance(block_color, str) or not block_color
        ):
            raise ConfigError("Query block 'color' must be a non-empty string.")

        for nested in nested_queries:
            queries.append(normalize_query(nested, inherited_color=block_color))

    return queries


def normalize_query(raw: Dict, inherited_color: Optional[str]) -> QuerySpec:
    if not isinstance(raw, dict):
        raise ConfigError("Each query must be a mapping.")

    query_id = raw.get("id")
    if not query_id or not isinstance(query_id, str):
        raise ConfigError("Each query must include a non-empty 'id'.")

    name = raw.get("name", query_id)
    if not isinstance(name, str) or not name:
        raise ConfigError(f"Query '{query_id}' field 'name' must be a string.")

    color = raw.get("color", inherited_color)
    if color is not None and (not isinstance(color, str) or not color):
        raise ConfigError(f"Query '{query_id}' field 'color' must be a string.")

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
    return QuerySpec(
        query_id=query_id,
        name=name,
        color=color,
        input_kind=input_kind,
        value=value,
    )


def normalize_axis(raw: Dict, query_ids: set[str]) -> AxisSpec:
    if not isinstance(raw, dict):
        raise ConfigError("Each axis must be a mapping.")

    query_a = raw.get("query_a")
    query_b = raw.get("query_b")
    if not query_a or not isinstance(query_a, str):
        raise ConfigError("Each axis must include a non-empty 'query_a'.")
    if not query_b or not isinstance(query_b, str):
        raise ConfigError("Each axis must include a non-empty 'query_b'.")
    if query_a not in query_ids:
        raise ConfigError(f"Axis query_a '{query_a}' is not defined in queries.")
    if query_b not in query_ids:
        raise ConfigError(f"Axis query_b '{query_b}' is not defined in queries.")

    method = raw.get("method", "explicit")
    if not isinstance(method, str) or not method:
        raise ConfigError("Axis 'method' must be a non-empty string.")

    name = raw.get("name")
    if name is None:
        name = f"{query_a}_vs_{query_b}"
    if not isinstance(name, str) or not name:
        raise ConfigError("Axis 'name' must be a non-empty string if provided.")

    params = raw.get("params", {})
    if params is None:
        params = {}
    if not isinstance(params, dict):
        raise ConfigError("Axis 'params' must be a mapping if provided.")

    return AxisSpec(
        name=name,
        query_a=query_a,
        query_b=query_b,
        method=method,
        params=params,
    )


def normalize_averages(raw: Any, queries: List[QuerySpec]) -> List[AverageGroupSpec]:
    if raw is None:
        return []
    if not isinstance(raw, dict):
        raise ConfigError("'average' must be a mapping if provided.")

    query_map = {query.query_id: query for query in queries}
    groups: List[AverageGroupSpec] = []

    for group_id, group_raw in raw.items():
        if not isinstance(group_id, str) or not group_id:
            raise ConfigError("Average group ids must be non-empty strings.")
        if group_id in query_map:
            raise ConfigError(
                f"Average group id '{group_id}' conflicts with an existing query id."
            )

        if isinstance(group_raw, list):
            query_ids = group_raw
            name = group_id
            color = None
        elif isinstance(group_raw, dict):
            query_ids = group_raw.get("queries")
            name = group_raw.get("name", group_id)
            color = group_raw.get("color")
        else:
            raise ConfigError(
                f"Average group '{group_id}' must be a list or a mapping."
            )

        if not isinstance(query_ids, list) or not query_ids:
            raise ConfigError(
                f"Average group '{group_id}' must define a non-empty 'queries' list."
            )

        normalized_query_ids: List[str] = []
        for query_id in query_ids:
            if not isinstance(query_id, str) or not query_id:
                raise ConfigError(
                    f"Average group '{group_id}' contains an invalid query id."
                )
            if query_id not in query_map:
                raise ConfigError(
                    f"Average group '{group_id}' references unknown query '{query_id}'."
                )
            normalized_query_ids.append(query_id)

        if len(normalized_query_ids) != len(set(normalized_query_ids)):
            raise ConfigError(
                f"Average group '{group_id}' has duplicate query references."
            )

        if not isinstance(name, str) or not name:
            raise ConfigError(f"Average group '{group_id}' name must be a string.")
        if color is not None and (not isinstance(color, str) or not color):
            raise ConfigError(f"Average group '{group_id}' color must be a string.")

        if color is None:
            colors = {query_map[qid].color for qid in normalized_query_ids}
            if len(colors) == 1:
                color = next(iter(colors))

        groups.append(
            AverageGroupSpec(
                group_id=group_id,
                query_ids=normalized_query_ids,
                name=name,
                color=color,
            )
        )

    return groups


def normalize_axes_options(raw: Any) -> AxesOptions:
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ConfigError("'axes_options' must be a mapping if provided.")

    translate = raw.get("translate", True)
    orthonormalize = raw.get("orthonormalize", False)
    normalize_vectors = raw.get("normalize_vectors", True)
    origin_strategy = raw.get("origin_strategy", "mean")
    pca = raw.get("pca", False)
    averaged_only = raw.get("averaged_only", False)

    if not isinstance(translate, bool):
        raise ConfigError("axes_options.translate must be a boolean.")
    if not isinstance(orthonormalize, bool):
        raise ConfigError("axes_options.orthonormalize must be a boolean.")
    if not isinstance(normalize_vectors, bool):
        raise ConfigError("axes_options.normalize_vectors must be a boolean.")
    if origin_strategy not in {"mean", "first"}:
        raise ConfigError("axes_options.origin_strategy must be 'mean' or 'first'.")
    if not isinstance(pca, bool):
        raise ConfigError("axes_options.pca must be a boolean.")
    if not isinstance(averaged_only, bool):
        raise ConfigError("axes_options.averaged_only must be a boolean.")

    return AxesOptions(
        translate=translate,
        orthonormalize=orthonormalize,
        normalize_vectors=normalize_vectors,
        origin_strategy=origin_strategy,
        pca=pca,
        averaged_only=averaged_only,
    )


def normalize_plots(
    raw: Any,
    axis_names: List[str],
    default_options: AxesOptions,
) -> Dict[str, PlotSpec]:
    if not axis_names:
        return {}

    if raw is None:
        return {"default": PlotSpec(axes=list(axis_names), options=default_options)}

    if isinstance(raw, list):
        if len(raw) != 1 or not isinstance(raw[0], dict):
            raise ConfigError("'plots' list format must contain one mapping item.")
        raw = raw[0]

    if not isinstance(raw, dict):
        raise ConfigError("'plots' must be a mapping of plot name to axis names.")

    known_axes = set(axis_names)
    plots: Dict[str, PlotSpec] = {}
    for plot_name, plot_raw in raw.items():
        if not isinstance(plot_name, str) or not plot_name:
            raise ConfigError("Each plot name must be a non-empty string.")

        plot_axes: Any
        plot_options = default_options
        if isinstance(plot_raw, list):
            plot_axes = plot_raw
        elif isinstance(plot_raw, dict):
            plot_axes = plot_raw.get("axes")
            if plot_axes is None:
                raise ConfigError(f"Plot '{plot_name}' is missing 'axes'.")
            plot_options = normalize_axes_options(plot_raw.get("options", {}))
        else:
            raise ConfigError(
                f"Plot '{plot_name}' must be a list of axis names or a mapping."
            )

        if not isinstance(plot_axes, list) or not plot_axes:
            raise ConfigError(f"Plot '{plot_name}' must list at least one axis.")

        normalized_axes: List[str] = []
        for axis_name in plot_axes:
            if not isinstance(axis_name, str) or not axis_name:
                raise ConfigError(f"Plot '{plot_name}' axis names must be strings.")
            if axis_name not in known_axes:
                raise ConfigError(
                    f"Plot '{plot_name}' references unknown axis '{axis_name}'."
                )
            normalized_axes.append(axis_name)

        if len(normalized_axes) != len(set(normalized_axes)):
            raise ConfigError(f"Plot '{plot_name}' has duplicate axis names.")
        plots[plot_name] = PlotSpec(axes=normalized_axes, options=plot_options)

    return plots
