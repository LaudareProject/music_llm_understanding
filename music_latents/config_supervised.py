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

    default_estimator = "elasticnet" if task == "regression" else "logistic_regression"
    estimator = raw.get("estimator", default_estimator)
    if not isinstance(estimator, str):
        raise ConfigError("supervised.estimator must be a string.")
    valid_estimators = {
        "regression": {"elasticnet", "linear_svr"},
        "classification": {"logistic_regression", "linear_svm"},
    }
    if estimator not in valid_estimators[task]:
        supported = ", ".join(sorted(valid_estimators[task]))
        raise ConfigError(
            f"supervised.estimator for task '{task}' must be one of: {supported}."
        )

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

    C = raw.get("C", 1.0)
    if not isinstance(C, (int, float)) or float(C) <= 0.0:
        raise ConfigError("supervised.C must be a positive number.")

    epsilon = raw.get("epsilon", 0.0)
    if not isinstance(epsilon, (int, float)) or float(epsilon) < 0.0:
        raise ConfigError("supervised.epsilon must be a non-negative number.")

    loss = raw.get("loss", "epsilon_insensitive")
    if not isinstance(loss, str):
        raise ConfigError("supervised.loss must be a string.")
    valid_losses = {"epsilon_insensitive", "squared_epsilon_insensitive"}
    if loss not in valid_losses:
        raise ConfigError(
            "supervised.loss must be one of: epsilon_insensitive, squared_epsilon_insensitive."
        )

    class_weight = raw.get("class_weight", None)
    if class_weight is not None and class_weight != "balanced":
        raise ConfigError("supervised.class_weight must be null or 'balanced'.")

    max_iter = raw.get("max_iter", 5000)
    if not isinstance(max_iter, int) or max_iter <= 0:
        raise ConfigError("supervised.max_iter must be a positive integer.")

    top_k = raw.get("top_k", 50)
    if not isinstance(top_k, int) or top_k <= 0:
        raise ConfigError("supervised.top_k must be a positive integer.")

    normalize = raw.get("normalize", False)
    if not isinstance(normalize, bool):
        raise ConfigError("supervised.normalize must be a boolean.")

    use_pca = raw.get("use_pca", False)
    if not isinstance(use_pca, bool):
        raise ConfigError("supervised.use_pca must be a boolean.")

    rerun_on_residualized_inputs = raw.get("rerun_on_residualized_inputs", False)
    if not isinstance(rerun_on_residualized_inputs, bool):
        raise ConfigError("supervised.rerun_on_residualized_inputs must be a boolean.")

    return {
        "task": task,
        "estimator": estimator,
        "features": list(features),
        "l1_ratio": l1_ratio,
        "alpha": float(alpha),
        "C": float(C),
        "epsilon": float(epsilon),
        "loss": loss,
        "class_weight": class_weight,
        "use_pca": use_pca,
        "rerun_on_residualized_inputs": rerun_on_residualized_inputs,
        "max_iter": max_iter,
        "top_k": top_k,
        "normalize": normalize,
    }
