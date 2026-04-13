from __future__ import annotations

import random

import numpy as np
from scipy.stats import spearmanr
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin, clone
from sklearn.cross_decomposition import PLSRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline


STOCHASTIC_ANCHOR_METHODS = {
    "random_sample_first_last",
    "random_median",
}


def _as_float_array(y):
    return np.asarray(y, dtype=np.float64)


class AnchorSelection(BaseEstimator, TransformerMixin):
    def __init__(self, method="first_vs_last", n=4, seed=1992, max_anchor_size=None):
        self.method = method
        self.n = n
        self.seed = seed
        self.max_anchor_size = float(max_anchor_size)

    def _first_last_counts(self, n_value, n_samples):
        if n_samples < 3:
            return 1, 1
        n_effective = max(2, min(int(n_value), n_samples - 1))
        if self.max_anchor_size is not None:
            max_total = int(np.round(self.max_anchor_size * n_samples))
            if n_effective > max_total:
                raise RuntimeError("max_anchor_size reached!")
        first_count = n_effective // 2
        last_count = first_count + (n_effective % 2)
        return first_count, last_count

    def _first_vs_last(self, idx, y_array, n_value):
        ranked = idx[np.argsort(y_array)]
        first_count, last_count = self._first_last_counts(n_value, len(idx))
        return ranked[:first_count], ranked[-last_count:]

    def _random_sample_first_last(self, idx, y_array, n_value):
        rng = random.Random(int(self.seed))
        first_count, last_count = self._first_last_counts(n_value, len(idx))
        sample_size = min(len(idx), max(first_count + last_count, 2))
        sampled = np.asarray(rng.sample(idx.tolist(), sample_size), dtype=np.int64)
        ranked = sampled[np.argsort(y_array[sampled])]
        first_count = min(first_count, len(ranked) - 1)
        last_count = min(last_count, len(ranked) - first_count)
        return ranked[:first_count], ranked[-last_count:]

    def _random_median(self, idx, y_array, n_value):
        rng = random.Random(int(self.seed))
        first_count, last_count = self._first_last_counts(n_value, len(idx))
        median = float(np.median(y_array))
        lower = [int(i) for i in idx if y_array[i] < median]
        upper = [int(i) for i in idx if y_array[i] >= median]
        rng.shuffle(lower)
        rng.shuffle(upper)
        lower_split = max(1, min(first_count, len(lower)))
        upper_split = max(1, min(last_count, len(upper)))
        return (
            np.asarray(lower[:lower_split], dtype=np.int64),
            np.asarray(upper[:upper_split], dtype=np.int64),
        )

    def _percentile_medians(self, idx, y_array, n_value):
        n_percentiles = int(n_value)
        first_count, last_count = self._first_last_counts(n_percentiles, len(idx))
        n_effective = first_count + last_count
        percentile_count = max(2, min(int(n_percentiles), len(idx), n_effective))
        ranked = idx[np.argsort(y_array)]
        boundaries = np.linspace(0, len(ranked), percentile_count + 1, dtype=np.int64)
        picked = []
        for left, right in zip(boundaries[:-1], boundaries[1:]):
            if right <= left:
                continue
            segment = ranked[left:right]
            median_pos = int((len(segment) - 1) // 2)
            picked.append(int(segment[median_pos]))
        if len(picked) < 2:
            return self._first_vs_last(idx, y_array, n_value)
        unique = np.asarray(
            sorted(set(picked), key=lambda j: y_array[j]), dtype=np.int64
        )
        first_count = max(1, min(first_count, len(unique) - 1))
        last_count = max(1, min(last_count, len(unique) - first_count))
        return unique[:first_count], unique[-last_count:]

    def fit(self, X, y):
        y_array = _as_float_array(y)
        n_samples = int(y_array.shape[0])
        if n_samples < 2:
            raise ValueError("AnchorSelection requires at least 2 samples")
        idx = np.arange(n_samples, dtype=np.int64)
        n_value = max(1, int(self.n))

        method = str(self.method)
        if method in {"first_vs_last"}:
            a_idx, b_idx = self._first_vs_last(idx, y_array, n_value)
        elif method in {"random_sample_first_last"}:
            a_idx, b_idx = self._random_sample_first_last(idx, y_array, n_value)
        elif method in {"random_median"}:
            a_idx, b_idx = self._random_median(idx, y_array, n_value)
        elif method in {"percentile_medians"}:
            a_idx, b_idx = self._percentile_medians(idx, y_array, n_value)
        else:
            raise ValueError(f"Unknown anchor selection method: {self.method}")

        if len(a_idx) == 0 or len(b_idx) == 0:
            raise ValueError("AnchorSelection produced an empty anchor side")

        self.anchor_indices_a_ = np.asarray(a_idx, dtype=np.int64)
        self.anchor_indices_b_ = np.asarray(b_idx, dtype=np.int64)
        self.anchor_targets_a_ = y_array[self.anchor_indices_a_]
        self.anchor_targets_b_ = y_array[self.anchor_indices_b_]
        self.anchor_indices_ = np.asarray(
            sorted(
                set(self.anchor_indices_a_.tolist() + self.anchor_indices_b_.tolist())
            ),
            dtype=np.int64,
        )
        return self

    def transform(self, X):
        return np.asarray(X, dtype=np.float64)

    def anchor_metadata(self):
        return {
            "anchor_indices_a": np.asarray(self.anchor_indices_a_, dtype=np.int64),
            "anchor_indices_b": np.asarray(self.anchor_indices_b_, dtype=np.int64),
            "anchor_indices": np.asarray(self.anchor_indices_, dtype=np.int64),
            "anchor_targets_a": np.asarray(self.anchor_targets_a_, dtype=np.float64),
            "anchor_targets_b": np.asarray(self.anchor_targets_b_, dtype=np.float64),
            "method": str(self.method),
        }


class AnchorPLS(BaseEstimator, TransformerMixin):
    def __init__(self, scale=False):
        self.scale = scale

    def set_anchor_metadata(self, metadata):
        self.anchor_metadata_ = metadata
        return self

    def fit(self, X, y):
        if not hasattr(self, "anchor_metadata_"):
            raise ValueError("AnchorPLS requires anchor metadata from AnchorSelection")
        X_array = np.asarray(X, dtype=np.float64)
        y_array = _as_float_array(y)
        anchor_idx = np.asarray(self.anchor_metadata_["anchor_indices"], dtype=np.int64)
        n_components = max(1, min(int(anchor_idx.shape[0]), int(X_array.shape[1])))
        self.n_components_ = n_components
        self.model_ = PLSRegression(n_components=n_components, scale=bool(self.scale))
        self.model_.fit(X_array[anchor_idx], y_array[anchor_idx])
        return self

    def transform(self, X):
        X_array = np.asarray(X, dtype=np.float64)
        return np.asarray(self.model_.transform(X_array), dtype=np.float64)


class Projection1D(BaseEstimator, TransformerMixin):
    def __init__(self, regression=False):
        self.regression = regression
        self.anchor_metadata_ = None

    def set_anchor_metadata(self, metadata):
        self.anchor_metadata_ = metadata
        return self

    def fit(self, X, y=None):
        if not hasattr(self, "anchor_metadata_"):
            raise ValueError(
                "Projection1D requires anchor metadata from AnchorSelection"
            )
        X_array = np.asarray(X, dtype=np.float64)
        a_idx = np.asarray(self.anchor_metadata_["anchor_indices_a"], dtype=np.int64)
        b_idx = np.asarray(self.anchor_metadata_["anchor_indices_b"], dtype=np.int64)
        a_targets = np.asarray(
            self.anchor_metadata_["anchor_targets_a"], dtype=np.float64
        )
        b_targets = np.asarray(
            self.anchor_metadata_["anchor_targets_b"], dtype=np.float64
        )

        n_use = max(1, min(len(a_idx), len(b_idx)))
        a_order = np.argsort(a_targets)
        b_order = np.argsort(b_targets)
        a_use = a_idx[a_order[:n_use]]
        b_use = b_idx[b_order[-n_use:]]

        center_a = np.mean(X_array[a_use], axis=0).astype(np.float64, copy=False)
        center_b = np.mean(X_array[b_use], axis=0).astype(np.float64, copy=False)
        self.use_regression_projection_ = False

        if bool(self.regression):
            anchor_idx = np.asarray(
                self.anchor_metadata_["anchor_indices"], dtype=np.int64
            )
            if y is not None and anchor_idx.size > 1:
                y_array = _as_float_array(y)
                y_anchor = y_array[anchor_idx]
            else:
                y_anchor = np.concatenate([a_targets, b_targets], axis=0)
                anchor_idx = np.concatenate([a_idx, b_idx], axis=0)

            if anchor_idx.size > 1:
                model = LinearRegression()
                model.fit(X_array[anchor_idx], y_anchor)
                direction = np.asarray(model.coef_, dtype=np.float64).ravel()
                if float(np.linalg.norm(direction)) > 1e-12:
                    self.direction_ = direction
                    self.origin_ = np.mean(X_array[anchor_idx], axis=0).astype(
                        np.float64, copy=False
                    )
                    self.use_regression_projection_ = True
                    self.n_used_ = int(anchor_idx.size)
                    return self

        direction = np.asarray(center_b - center_a, dtype=np.float64)
        norm = float(np.linalg.norm(direction))
        if norm <= 1e-12:
            direction = np.zeros_like(direction)
        else:
            direction = direction / norm

        self.direction_ = direction
        self.origin_ = center_a
        self.n_used_ = n_use
        return self

    def transform(self, X):
        X_array = np.asarray(X, dtype=np.float64)
        projected = (X_array - self.origin_) @ self.direction_
        return projected.reshape(-1, 1)


class TransparentAnchorPipeline(BaseEstimator, RegressorMixin):
    def __init__(self, pipeline, n_trials=1, random_state=42):
        self.pipeline = pipeline
        self.n_trials = n_trials
        self.random_state = random_state

    def get_params(self, deep=True):
        params = {
            "pipeline": self.pipeline,
            "n_trials": self.n_trials,
            "random_state": self.random_state,
        }
        if deep:
            params.update(self.pipeline.get_params(deep=True))
        return params

    def set_params(self, **params):
        if "pipeline" in params:
            self.pipeline = params.pop("pipeline")
        if "n_trials" in params:
            self.n_trials = params.pop("n_trials")
        if "random_state" in params:
            self.random_state = params.pop("random_state")
        if params:
            self.pipeline.set_params(**params)
        return self

    def _fit_once(self, pipe, X_array, y_array, group_labels_array=None):
        X_current = np.asarray(X_array, dtype=np.float64)
        anchor_metadata = None

        for _, step in pipe.steps[:-1]:
            if hasattr(step, "set_anchor_metadata") and anchor_metadata is not None:
                step.set_anchor_metadata(anchor_metadata)
            if isinstance(step, LinearDiscriminantAnalysis):
                if group_labels_array is None:
                    raise ValueError(
                        "LinearDiscriminantAnalysis requires group_labels from group_col"
                    )
                if anchor_metadata is None:
                    raise ValueError(
                        "LinearDiscriminantAnalysis requires AnchorSelection before it"
                    )
                anchor_idx = np.asarray(
                    anchor_metadata["anchor_indices"], dtype=np.int64
                )
                labels_anchor = np.asarray(group_labels_array)[anchor_idx]
                unique_classes = np.unique(labels_anchor)
                if unique_classes.shape[0] < 2:
                    raise ValueError(
                        "LinearDiscriminantAnalysis requires at least 2 classes among anchors"
                    )
                step.fit(X_current[anchor_idx], labels_anchor)
            else:
                step.fit(X_current, y_array)
            if hasattr(step, "anchor_metadata"):
                anchor_metadata = step.anchor_metadata()
            X_current = step.transform(X_current)

        estimator = pipe.steps[-1][1]
        if anchor_metadata is not None:
            anchor_idx = np.asarray(anchor_metadata["anchor_indices"], dtype=np.int64)
            estimator.fit(X_current[anchor_idx], y_array[anchor_idx])
        else:
            estimator.fit(X_current, y_array)
        y_hat = np.asarray(estimator.predict(X_current), dtype=np.float64)
        if anchor_metadata is not None:
            eval_mask = np.ones(len(y_array), dtype=bool)
            eval_mask[np.asarray(anchor_metadata["anchor_indices"], dtype=np.int64)] = (
                False
            )
            if np.any(eval_mask):
                score, _ = spearmanr(y_array[eval_mask], y_hat[eval_mask])
            else:
                score, _ = spearmanr(y_array, y_hat)
        else:
            score, _ = spearmanr(y_array, y_hat)
        score = -1.0 if np.isnan(score) else float(score)
        return pipe, anchor_metadata, score

    def _resolve_trial_count(self):
        probe_pipe = clone(self.pipeline)
        probe_anchor = None
        for _, step in probe_pipe.steps[:-1]:
            if isinstance(step, AnchorSelection):
                probe_anchor = step
                break
        anchor_method = (
            probe_anchor.method if probe_anchor is not None else "first_vs_last"
        )
        deterministic = anchor_method not in STOCHASTIC_ANCHOR_METHODS
        return 1 if deterministic else max(1, int(self.n_trials))

    def fit(self, X, y, group_labels=None):
        X_array = np.asarray(X, dtype=np.float64)
        y_array = _as_float_array(y)
        group_labels_array = None
        if group_labels is not None:
            group_labels_array = np.asarray(group_labels)
            if int(group_labels_array.shape[0]) != int(y_array.shape[0]):
                raise ValueError("group_labels must have same length as y")
        base_seed = int(self.random_state)
        trials = self._resolve_trial_count()

        trial_pipes = []
        trial_metadata = []
        trial_scores = []
        for trial_idx in range(trials):
            trial_seed = int(base_seed + trial_idx)
            pipe = clone(self.pipeline)
            for _, step in pipe.steps[:-1]:
                if (
                    isinstance(step, AnchorSelection)
                    and step.method in STOCHASTIC_ANCHOR_METHODS
                ):
                    step.seed = trial_seed
            fitted_pipe, metadata, score = self._fit_once(
                pipe,
                X_array,
                y_array,
                group_labels_array=group_labels_array,
            )
            trial_pipes.append(fitted_pipe)
            trial_metadata.append(metadata)
            trial_scores.append(float(score))

        self.trial_pipelines_ = trial_pipes
        self.trial_anchor_metadata_ = trial_metadata
        self.trial_scores_ = np.asarray(trial_scores, dtype=np.float64)
        self.pipeline_ = trial_pipes[0]
        self.anchor_metadata_ = trial_metadata[0]
        self.mean_trial_score_ = float(np.mean(self.trial_scores_))
        self.effective_n_trials_ = int(trials)
        return self

    def _transform_for_predict(self, X, pipeline, anchor_metadata):
        X_current = np.asarray(X, dtype=np.float64)
        for _, step in pipeline.steps[:-1]:
            if hasattr(step, "set_anchor_metadata") and anchor_metadata is not None:
                step.set_anchor_metadata(anchor_metadata)
            X_current = step.transform(X_current)
        return X_current

    def predict(self, X):
        if not hasattr(self, "trial_pipelines_"):
            raise ValueError("Estimator is not fitted")
        per_trial = []
        for pipeline, metadata in zip(
            self.trial_pipelines_, self.trial_anchor_metadata_
        ):
            X_final = self._transform_for_predict(X, pipeline, metadata)
            estimator = pipeline.steps[-1][1]
            y_hat = np.asarray(estimator.predict(X_final), dtype=np.float64)
            per_trial.append(y_hat)
        return np.asarray(per_trial, dtype=np.float64)


def build_transparent_pipeline(steps, n_trials=1, random_state=42):
    return TransparentAnchorPipeline(
        Pipeline(steps),
        n_trials=n_trials,
        random_state=random_state,
    )
