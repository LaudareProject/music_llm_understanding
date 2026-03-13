from __future__ import annotations

from typing import Any, Dict, List

from music_latents.config_models import (
    AxisSpec,
    AxesOptions,
    ConfigError,
    PCA_COMPONENT_AXIS_RE,
    PlotSpec,
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


def normalize_axes_options(raw: Any) -> AxesOptions:
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ConfigError("'axes_options' must be a mapping if provided.")

    center_and_whiten = raw.get("center_and_whiten", False)
    translate = raw.get("translate", True)
    orthonormalize = raw.get("orthonormalize", False)
    normalize_vectors = raw.get("normalize_vectors", False)
    origin_strategy = raw.get("origin_strategy", "mean")
    pca = raw.get("pca", False)
    lda_groups = raw.get("lda_groups", False)
    averaged_only = raw.get("averaged_only", False)
    group_filter = raw.get("group_filter", "all")

    if not isinstance(center_and_whiten, bool):
        raise ConfigError("axes_options.center_and_whiten must be a boolean.")
    if center_and_whiten:
        raise ConfigError(
            "axes_options.center_and_whiten cannot be true in config; use CLI --center-and-whiten."
        )
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
    if not isinstance(lda_groups, bool):
        raise ConfigError("axes_options.lda_groups must be a boolean.")
    if not isinstance(averaged_only, bool):
        raise ConfigError("axes_options.averaged_only must be a boolean.")
    if group_filter not in {"all", "average", "median"}:
        raise ConfigError(
            "axes_options.group_filter must be one of: all, average, median."
        )

    return AxesOptions(
        center_and_whiten=center_and_whiten,
        translate=translate,
        orthonormalize=orthonormalize,
        normalize_vectors=normalize_vectors,
        origin_strategy=origin_strategy,
        pca=pca,
        lda_groups=lda_groups,
        averaged_only=averaged_only,
        group_filter=group_filter,
    )


def normalize_plots(
    raw: Any,
    axis_names: List[str],
    default_options: AxesOptions,
) -> Dict[str, PlotSpec]:
    if raw is None:
        if not axis_names:
            return {}
        return {
            "default": PlotSpec(
                mode="axes",
                axes=list(axis_names),
                cosine_references=None,
                options=default_options,
            )
        }

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
        plot_mode = "axes"
        cosine_references = None
        plot_options = default_options
        if isinstance(plot_raw, list):
            plot_axes = plot_raw
        elif isinstance(plot_raw, dict):
            plot_options = normalize_axes_options(plot_raw.get("options", {}))
            plot_axes = plot_raw.get("axes")
            cosine_raw = plot_raw.get("cosine")
            mds_raw = plot_raw.get("mds")

            mode_count = 0
            if plot_axes is not None:
                mode_count += 1
            if cosine_raw is not None:
                mode_count += 1
            if bool(mds_raw):
                mode_count += 1
            if mode_count != 1:
                raise ConfigError(
                    f"Plot '{plot_name}' must define exactly one mode: axes, cosine, or mds."
                )

            if bool(mds_raw):
                if mds_raw is not True:
                    raise ConfigError(f"Plot '{plot_name}' mds must be true when set.")
                plot_mode = "mds"
                if (
                    plot_options.pca
                    or plot_options.orthonormalize
                    or plot_options.lda_groups
                ):
                    raise ConfigError(
                        f"Plot '{plot_name}' cannot combine mds with pca, orthonormalize, or lda_groups."
                    )
                plot_axes = []
            elif cosine_raw is not None:
                plot_mode = "cosine"
                if (
                    not isinstance(cosine_raw, list)
                    or len(cosine_raw) != 2
                    or any(
                        not isinstance(reference, str) or not reference
                        for reference in cosine_raw
                    )
                ):
                    raise ConfigError(
                        f"Plot '{plot_name}' cosine must be a list with exactly two non-empty strings."
                    )
                cosine_references = list(cosine_raw)
                plot_axes = []
            else:
                plot_mode = "axes"
                if plot_axes is None:
                    raise ConfigError(f"Plot '{plot_name}' is missing 'axes'.")
        else:
            raise ConfigError(
                f"Plot '{plot_name}' must be a list of axis names or a mapping."
            )

        if plot_mode == "axes":
            if not isinstance(plot_axes, list) or not plot_axes:
                raise ConfigError(f"Plot '{plot_name}' must list at least one axis.")
        else:
            if not isinstance(plot_axes, list):
                raise ConfigError(f"Plot '{plot_name}' has invalid mode payload.")

        normalized_axes: List[str] = []
        if plot_mode == "axes":
            for axis_name in plot_axes:
                if not isinstance(axis_name, str) or not axis_name:
                    raise ConfigError(f"Plot '{plot_name}' axis names must be strings.")
                if axis_name not in known_axes and not is_pca_component_axis_name(
                    axis_name
                ):
                    raise ConfigError(
                        f"Plot '{plot_name}' references unknown axis '{axis_name}'."
                    )
                normalized_axes.append(axis_name)

        if len(normalized_axes) != len(set(normalized_axes)):
            raise ConfigError(f"Plot '{plot_name}' has duplicate axis names.")
        plots[plot_name] = PlotSpec(
            mode=plot_mode,
            axes=normalized_axes,
            cosine_references=cosine_references,
            options=plot_options,
        )

    return plots


def is_pca_component_axis_name(axis_name: str) -> bool:
    return PCA_COMPONENT_AXIS_RE.match(axis_name) is not None
