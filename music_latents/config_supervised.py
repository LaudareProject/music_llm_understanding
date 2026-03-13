from __future__ import annotations

from typing import Any, Dict, Optional

from music_latents.config_models import ConfigError


def normalize_supervised(raw: Any) -> Optional[Dict[str, Any]]:
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ConfigError("'supervised' must be a mapping if provided.")

    task = raw.get("task")
    if task not in {"classification", "regression"}:
        raise ConfigError("supervised.task must be 'classification' or 'regression'.")

    features = raw.get("features", ["embedding"])
    if not isinstance(features, list) or not features:
        raise ConfigError("supervised.features must be a non-empty list.")
    valid_features = {"embedding", "norm", "polar", "descriptive"}
    for feature_name in features:
        if not isinstance(feature_name, str) or feature_name not in valid_features:
            raise ConfigError(
                "supervised.features entries must be one of: embedding, norm, polar, descriptive."
            )

    l1_ratio = raw.get("l1_ratio", 0.5)
    if not isinstance(l1_ratio, (int, float)):
        raise ConfigError("supervised.l1_ratio must be numeric.")
    l1_ratio = float(l1_ratio)
    if not (0.0 <= l1_ratio <= 1.0):
        raise ConfigError("supervised.l1_ratio must be in [0, 1].")

    alpha = raw.get("alpha", 1.0)
    if not isinstance(alpha, (int, float)) or float(alpha) <= 0.0:
        raise ConfigError("supervised.alpha must be a positive number.")

    max_iter = raw.get("max_iter", 5000)
    if not isinstance(max_iter, int) or max_iter <= 0:
        raise ConfigError("supervised.max_iter must be a positive integer.")

    top_k = raw.get("top_k", 50)
    if not isinstance(top_k, int) or top_k <= 0:
        raise ConfigError("supervised.top_k must be a positive integer.")

    normalize = raw.get("normalize", False)
    if not isinstance(normalize, bool):
        raise ConfigError("supervised.normalize must be a boolean.")

    return {
        "task": task,
        "features": list(features),
        "l1_ratio": l1_ratio,
        "alpha": float(alpha),
        "max_iter": max_iter,
        "top_k": top_k,
        "normalize": normalize,
    }
