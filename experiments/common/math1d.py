import numpy as np


COMBOS = [
    {"name": "baseline", "whiten": False, "method": None},
    {"name": "whiten", "whiten": True, "method": None},
    {"name": "lda", "whiten": False, "method": "lda"},
    {"name": "pca", "whiten": False, "method": "pca"},
    {"name": "pls", "whiten": False, "method": "pls"},
    {"name": "olda", "whiten": False, "method": "olda"},
    {"name": "whiten+lda", "whiten": True, "method": "lda"},
    {"name": "whiten+pca", "whiten": True, "method": "pca"},
    {"name": "whiten+pls", "whiten": True, "method": "pls"},
    {"name": "whiten+olda", "whiten": True, "method": "olda"},
]


def whiten(matrix):
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.shape[0] < 2:
        return matrix - np.mean(matrix, axis=0, keepdims=True)
    from sklearn.decomposition import PCA

    n_components = min(matrix.shape[0], matrix.shape[1])
    return PCA(n_components=n_components, whiten=True, svd_solver="full").fit_transform(
        matrix
    )


def pca_reduce(matrix, variance=0.9):
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.shape[0] < 2:
        return matrix
    from sklearn.decomposition import PCA

    n_components = min(matrix.shape[0], matrix.shape[1])
    model = PCA(n_components=n_components, svd_solver="full")
    scores = model.fit_transform(matrix)
    out = scores @ model.components_
    return out


def lda_reduce(matrix, labels):
    matrix = np.asarray(matrix, dtype=np.float64)
    labels = list(labels)
    if len(set(labels)) < 2:
        return matrix
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

    n_components = min(len(set(labels)) - 1, matrix.shape[1])
    if n_components < 1:
        return matrix
    model = LinearDiscriminantAnalysis(n_components=n_components, solver="svd")
    scores = model.fit_transform(matrix, labels)
    scores = np.asarray(scores, dtype=np.float64)
    if scores.ndim == 1:
        scores = scores.reshape(-1, 1)
    out = scores @ model.scalings_[:, : scores.shape[1]].T
    return out


def pls_reduce(matrix, target_scores):
    matrix = np.asarray(matrix, dtype=np.float64)
    target_scores = np.asarray(target_scores, dtype=np.float64)
    if matrix.shape[0] < 2 or np.std(target_scores) <= 1e-12:
        return matrix
    from sklearn.cross_decomposition import PLSRegression

    n_components = min(matrix.shape[0] - 1, matrix.shape[1])
    if n_components < 1:
        return matrix
    try:
        model = PLSRegression(n_components=n_components)
        scores = model.fit_transform(matrix, target_scores.reshape(-1, 1))[0]
    except Exception:
        model = PLSRegression(n_components=1)
        scores = model.fit_transform(matrix, target_scores.reshape(-1, 1))[0]
    scores = np.asarray(scores, dtype=np.float64)
    if scores.ndim == 1:
        scores = scores.reshape(-1, 1)
    return scores @ model.x_loadings_[:, : scores.shape[1]].T


def olda_reduce(matrix, labels, target_scores):
    matrix = np.asarray(matrix, dtype=np.float64)
    labels = np.asarray(labels)
    target_scores = np.asarray(target_scores, dtype=np.float64)
    unique_labels = np.unique(labels)
    if unique_labels.shape[0] < 2:
        return matrix
    n_components = min(unique_labels.shape[0] - 1, matrix.shape[1])
    if n_components < 1:
        return matrix
    from experiments.common.olda import OrdinalLinearDiscriminantAnalysis

    class_scores = np.asarray(
        [np.mean(target_scores[labels == label]) for label in unique_labels],
        dtype=np.float64,
    )
    try:
        model = OrdinalLinearDiscriminantAnalysis(
            n_components=n_components,
            solver="eigen",
            class_scores=class_scores,
        )
        scores = model.fit_transform(matrix, labels)
    except Exception:
        return lda_reduce(matrix, labels)
    scores = np.asarray(scores, dtype=np.float64)
    if scores.ndim == 1:
        scores = scores.reshape(-1, 1)
    out = scores @ model.scalings_[:, : scores.shape[1]].T
    return out


def apply_combo(matrix, labels, combo, target_scores=None):
    out = np.asarray(matrix, dtype=np.float64)
    if combo["whiten"]:
        out = whiten(out)
    method = combo["method"]
    if method == "lda":
        out = lda_reduce(out, labels)
    elif method == "pca":
        out = pca_reduce(out, 0.9)
    elif method == "pls":
        if target_scores is None:
            raise ValueError("PLS requires target_scores")
        out = pls_reduce(out, target_scores)
    elif method == "olda":
        if target_scores is None:
            raise ValueError("OLDA requires target_scores")
        out = olda_reduce(out, labels, target_scores)
    return out


def combo_config_by_name(combo_name):
    for combo in COMBOS:
        if combo["name"] == combo_name:
            return combo
    raise ValueError(f"Unknown combo: {combo_name}")


def project_1d(matrix, anchor_a_idx, anchor_b_idx, eval_idx):
    """
    Compute cosine similarity dot(u, v) / (||u|| ||v||), where u is each evaluated row centered
    by mean(rows `anchor_a_idx`) and v is the axis from mean(rows `anchor_a_idx`) to mean(rows
    `anchor_b_idx`); the returned value is exactly the standard cosine similarity in that
    anchor-centered space.
    """
    matrix = np.asarray(matrix, dtype=np.float64)
    mean_a = np.mean(matrix[np.asarray(anchor_a_idx, dtype=int)], axis=0)
    mean_b = np.mean(matrix[np.asarray(anchor_b_idx, dtype=int)], axis=0)
    direction = mean_b - mean_a
    centered = matrix[np.asarray(eval_idx, dtype=int)] - mean_a
    numerator = centered @ direction
    direction_norm = max(float(np.linalg.norm(direction)), 1e-12)
    centered_norms = np.maximum(np.linalg.norm(centered, axis=1), 1e-12)
    return numerator / (centered_norms * direction_norm)


def _fit_affine(feature_values, y):
    feature_values = np.asarray(feature_values, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if feature_values.shape[0] < 2 or np.std(feature_values) <= 1e-12:
        return 0.0, float(np.mean(y) if y.size else 0.0)
    from sklearn.linear_model import LinearRegression

    model = LinearRegression()
    model.fit(feature_values.reshape(-1, 1), y)
    return float(model.coef_[0]), float(model.intercept_)


def fit_mapping_model(mapping_name, x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if mapping_name == "raw":
        return {"name": "raw"}
    if mapping_name == "linear":
        coef, intercept = _fit_affine(x, y)
        return {"name": "linear", "coef": coef, "intercept": intercept}
    if mapping_name == "log_linear":
        min_shift = max(0.0, -float(np.min(x)) + 1e-9) if x.size else 0.0
        shift = (
            min_shift + max(1.0, 0.05 * float(np.max(x) - np.min(x))) if x.size else 1.0
        )
        feature_values = np.log(np.maximum(x + shift, 1e-9))
        coef, intercept = _fit_affine(feature_values, y)
        return {
            "name": "log_linear",
            "coef": coef,
            "intercept": intercept,
            "shift": float(shift),
        }
    if mapping_name == "exp_linear":
        centered = x - np.mean(x) if x.size else x
        corr = pearson(x, y)
        rate = 0.1 / max(float(np.std(centered)), 1e-6) if x.size else 0.0
        if not np.isnan(corr) and corr < 0:
            rate = -rate
        feature_values = np.exp(rate * centered)
        coef, intercept = _fit_affine(feature_values, y)
        return {
            "name": "exp_linear",
            "coef": coef,
            "intercept": intercept,
            "rate": float(rate),
            "center": float(np.mean(x) if x.size else 0.0),
        }
    raise ValueError(f"Unknown mapping model: {mapping_name}")


def predict_mapping_model(model, x):
    x = np.asarray(x, dtype=np.float64)
    name = model["name"]
    if name == "raw":
        return x
    if name == "linear":
        return model["coef"] * x + model["intercept"]
    if name == "log_linear":
        feature_values = np.log(np.maximum(x + model["shift"], 1e-9))
        return model["coef"] * feature_values + model["intercept"]
    if name == "exp_linear":
        feature_values = np.exp(model["rate"] * (x - model["center"]))
        return model["coef"] * feature_values + model["intercept"]
    raise ValueError(f"Unknown mapping model: {name}")


def average_mapping_models(models):
    if not models:
        return None
    name = models[0]["name"]
    if name == "raw":
        return {"name": "raw"}
    averaged = {"name": name}
    for key in models[0]:
        if key == "name":
            continue
        averaged[key] = float(np.mean([model[key] for model in models]))
    return averaged


def mean_ci95(values):
    values = np.asarray(values, dtype=np.float64)
    values = values[~np.isnan(values)]
    if values.size == 0:
        return float("nan"), float("nan")
    if values.size == 1:
        return float(values[0]), 0.0
    return float(np.mean(values)), float(
        1.96 * np.std(values, ddof=1) / np.sqrt(values.size)
    )


def grouped_kfold_splits(groups, n_splits=10):
    groups = np.asarray(groups)
    unique_groups = np.unique(groups)
    if unique_groups.shape[0] < 2:
        return [(np.arange(groups.shape[0]), np.arange(groups.shape[0]))]
    from sklearn.model_selection import GroupKFold

    splitter = GroupKFold(n_splits=min(n_splits, unique_groups.shape[0]))
    indices = np.arange(groups.shape[0])
    return list(splitter.split(indices, groups=groups))


def lolo_predictions(raw_projection, real, groups, mapping_name):
    raw_projection = np.asarray(raw_projection, dtype=np.float64)
    real = np.asarray(real, dtype=np.float64)
    groups = np.asarray(groups)
    unique_groups = np.unique(groups)
    if unique_groups.shape[0] < 2:
        model = fit_mapping_model(mapping_name, raw_projection, real)
        return np.asarray(
            predict_mapping_model(model, raw_projection), dtype=np.float64
        )
    from sklearn.model_selection import LeaveOneGroupOut

    predicted = np.full(raw_projection.shape[0], np.nan, dtype=np.float64)
    splitter = LeaveOneGroupOut()
    indices = np.arange(raw_projection.shape[0])
    for train_idx, test_idx in splitter.split(indices, groups=groups):
        model = fit_mapping_model(
            mapping_name, raw_projection[train_idx], real[train_idx]
        )
        predicted[test_idx] = predict_mapping_model(model, raw_projection[test_idx])
    return predicted


def cross_validated_mapping_summary(
    raw_projection,
    real,
    groups,
    mapping_name,
    extra_metric_fns=None,
):
    raw_projection = np.asarray(raw_projection, dtype=np.float64)
    real = np.asarray(real, dtype=np.float64)
    groups = np.asarray(groups)
    extra_metric_fns = {} if extra_metric_fns is None else dict(extra_metric_fns)
    fold_metrics = {"pearson": [], "spearman": [], "l1": []}
    fold_metrics.update({name: [] for name in extra_metric_fns})
    models = []
    for train_idx, test_idx in grouped_kfold_splits(groups, n_splits=10):
        model = fit_mapping_model(
            mapping_name, raw_projection[train_idx], real[train_idx]
        )
        models.append(model)
        predicted = predict_mapping_model(model, raw_projection[test_idx])
        fold_metrics["pearson"].append(pearson(real[test_idx], predicted))
        fold_metrics["spearman"].append(spearman(real[test_idx], predicted))
        fold_metrics["l1"].append(
            float("nan") if mapping_name == "raw" else l1(real[test_idx], predicted)
        )
        for name, metric_fn in extra_metric_fns.items():
            fold_metrics[name].append(metric_fn(test_idx, predicted))
    averaged_model = average_mapping_models(models)
    full_predicted = predict_mapping_model(averaged_model, raw_projection)
    summary = {
        "model": averaged_model,
        "predicted": np.asarray(full_predicted, dtype=np.float64),
        "lolo_predicted": lolo_predictions(raw_projection, real, groups, mapping_name),
        "fold_count": len(fold_metrics["spearman"]),
    }
    for name, values in fold_metrics.items():
        mean_value, ci_value = mean_ci95(values)
        summary[name] = mean_value
        summary[f"{name}_ci95"] = ci_value
    return summary


def fit_linear(x, y):
    """Fit ordinary least squares with sklearn LinearRegression and return in-sample predictions."""
    return predict_mapping_model(fit_mapping_model("linear", x, y), x)


def fit_log_linear(x, y):
    """Fit y ~= a * log(x + c) + b by nonlinear least squares with scipy curve_fit and return predictions."""
    return predict_mapping_model(fit_mapping_model("log_linear", x, y), x)


def fit_exp_linear(x, y):
    """Fit y ~= a * exp(r * (x - mean(x))) + b by nonlinear least squares with scipy curve_fit and return predictions."""
    return predict_mapping_model(fit_mapping_model("exp_linear", x, y), x)


def pearson(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.shape[0] < 2 or y.shape[0] < 2:
        return float("nan")
    if np.std(x) <= 1e-12 or np.std(y) <= 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def spearman(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.shape[0] < 2 or y.shape[0] < 2:
        return float("nan")
    if np.std(x) <= 1e-12 or np.std(y) <= 1e-12:
        return float("nan")
    from scipy.stats import spearmanr

    value = spearmanr(x, y).statistic
    return float(value) if not np.isnan(value) else float("nan")


def l1(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    return float(np.mean(np.abs(y_true - y_pred)))


def random_l1_baseline(y_true):
    """Expected MAE of random guesses drawn from other empirical target values."""
    y_true = np.asarray(y_true, dtype=np.float64)
    pairwise = np.abs(y_true[:, None] - y_true[None, :])
    return float(np.mean(pairwise))


def hann_density(predicted, birth_years, death_years):
    """
    Mean normalized overlap area between the predicted Hann window and the real lifespan.

    Each predicted window is centered at the predicted midpoint, but it reuses the real
    lifespan length. The score for a sample is the fraction of the predicted Hann-window
    area that falls inside the real lifespan, so exact alignment gives 1 and no overlap
    gives 0.
    """
    density = _hann_weights(predicted, birth_years, death_years)
    return float(np.mean(density))


def _hann_weights(predicted, birth_years, death_years):
    """Per-sample normalized Hann overlap areas, used as correlation weights."""
    predicted = np.asarray(predicted, dtype=np.float64)
    birth = np.asarray(birth_years, dtype=np.float64)
    death = np.asarray(death_years, dtype=np.float64)
    half_span = np.maximum((death - birth) / 2.0, 1e-9)
    left = np.maximum(birth, predicted - half_span)
    right = np.minimum(death, predicted + half_span)
    overlap = right > left
    u_left = (left - predicted) / half_span
    u_right = (right - predicted) / half_span
    area_fraction = 0.5 * (u_right - u_left) + (
        np.sin(np.pi * u_right) - np.sin(np.pi * u_left)
    ) / (2.0 * np.pi)
    return np.where(overlap, area_fraction, 0.0)


def weighted_pearson(x, y, w):
    """Weighted Pearson correlation using numpy.cov with analytical weights."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)
    if x.shape[0] < 2 or np.sum(w) <= 1e-12:
        return float("nan")
    cov = np.cov(x, y, aweights=w)
    denom = (cov[0, 0] * cov[1, 1]) ** 0.5
    if denom <= 1e-12:
        return float("nan")
    return float(cov[0, 1] / denom)


def weighted_spearman(x, y, w):
    """Weighted Spearman: weighted fractional ranks fed into weighted_pearson."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)
    if x.shape[0] < 2:
        return float("nan")

    def weighted_ranks(v, w):
        order = np.argsort(v)
        cumw = np.cumsum(w[order])
        total = cumw[-1]
        if total <= 1e-12:
            return None
        prev = np.concatenate(([0.0], cumw[:-1]))
        ranks = np.empty(len(v), dtype=np.float64)
        ranks[order] = (prev + cumw) / total
        return ranks

    rx = weighted_ranks(x, w)
    ry = weighted_ranks(y, w)
    if rx is None or ry is None:
        return float("nan")
    return weighted_pearson(rx, ry, w)
