from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import numpy as np


class RunnerSupervisedMixin:
    def _save_supervised_outputs(
        self,
        output_dir: Path,
        embeddings_by_model: Dict[str, Dict[str, np.ndarray]],
    ) -> None:
        assert self.config.supervised is not None

        supervised_dir = output_dir / "supervised"
        supervised_dir.mkdir(parents=True, exist_ok=True)

        task = self.config.supervised["task"]
        query_ids, y = self._build_supervised_targets(task)

        _, cv_label, cv_splits = self._determine_cv_strategy(
            sample_count=len(query_ids)
        )

        for model_name, embeddings in embeddings_by_model.items():
            model_dir = supervised_dir / model_name
            model_dir.mkdir(parents=True, exist_ok=True)

            matrix, feature_names = self._build_supervised_feature_matrix(
                embeddings=embeddings,
                query_ids=query_ids,
                features=self.config.supervised["features"],
            )

            result = self._fit_supervised_model(
                X=matrix,
                y=y,
                task=task,
                l1_ratio=float(self.config.supervised["l1_ratio"]),
                alpha=float(self.config.supervised["alpha"]),
                max_iter=int(self.config.supervised["max_iter"]),
            )

            importances = self._compute_importances_from_pca_model(
                scaler=result["scaler"],
                pca=result["pca"],
                model=result["model"],
                feature_names=feature_names,
            )

            selected_importances, selected_share = (
                self._select_importances_for_cumulative_share(
                    importances=importances,
                    target_share=0.5,
                )
            )

            payload = {
                "task": task,
                "cv_strategy": cv_label,
                "cv_folds": cv_splits,
                "n_samples": int(matrix.shape[0]),
                "n_features": int(matrix.shape[1]),
                "n_components": int(result["pca"].n_components_),
                "features": self.config.supervised["features"],
                "model": result["model_info"],
                "score_metric": result["score_metric"],
                "loo_score": float(result["training_score"]),
                "training_score": float(result["training_score"]),
                "feature_importance_target_share": 0.5,
                "feature_importance_selected_share": float(selected_share),
                "top_feature_importances": selected_importances,
            }
            self._save_json(model_dir / "supervised_report.json", payload)
            self._save_supervised_importance_plot(
                model_dir=model_dir,
                importances=selected_importances,
                training_score=float(result["training_score"]),
                score_metric=str(result["score_metric"]),
            )

    def _determine_cv_strategy(self, sample_count: int) -> tuple[Any, str, int]:
        if sample_count > 1:
            from sklearn.model_selection import LeaveOneOut

            return LeaveOneOut(), "leave_one_out", sample_count
        return 1, "single_sample", 1

    def _build_supervised_targets(self, task: str) -> tuple[list[str], np.ndarray]:
        labeled_queries = [
            query for query in self.config.queries if query.target is not None
        ]
        if labeled_queries:
            query_ids = [query.query_id for query in labeled_queries]
            raw_targets = [query.target for query in labeled_queries]
            if task == "regression":
                try:
                    y = np.asarray(
                        [float(value) for value in raw_targets], dtype=np.float64
                    )
                except Exception as exc:
                    raise ValueError(
                        "Regression targets must be numeric (query target/label)."
                    ) from exc
            else:
                y = np.asarray([str(value) for value in raw_targets], dtype=object)
            return query_ids, y

        if task != "regression":
            raise ValueError(
                "Supervised classification requires query targets (target/label)."
            )

        value_queries = [
            query for query in self.config.queries if query.input_kind == "value"
        ]
        if not value_queries:
            raise ValueError(
                "Supervised regression requires query targets or numeric query 'value' inputs."
            )

        try:
            raw_values = np.asarray(
                [float(query.value) for query in value_queries], dtype=np.float64
            )
        except Exception as exc:
            raise ValueError(
                "Numeric query 'value' inputs must be parseable as numbers."
            ) from exc

        if bool(self.config.supervised.get("normalize", False)):
            min_value = float(np.min(raw_values))
            max_value = float(np.max(raw_values))
            span = max_value - min_value
            if span <= 1e-12:
                raise ValueError(
                    "Cannot normalize regression targets from 'value' because all values are identical."
                )
            targets = (raw_values - min_value) / span
        else:
            targets = raw_values

        return [query.query_id for query in value_queries], targets

    def _build_supervised_feature_matrix(
        self,
        embeddings: Dict[str, np.ndarray],
        query_ids: list[str],
        features: list[str],
    ) -> tuple[np.ndarray, list[str]]:
        rows: list[np.ndarray] = []
        feature_names: list[str] | None = None

        for query_id in query_ids:
            vector = embeddings.get(query_id)
            if vector is None:
                raise ValueError(
                    f"Supervised query '{query_id}' missing embedding vector."
                )
            row, names = self._extract_vector_features(vector, features)
            rows.append(row)
            if feature_names is None:
                feature_names = names

        assert feature_names is not None
        return np.stack(rows, axis=0), feature_names

    def _extract_vector_features(
        self,
        vector: np.ndarray,
        features: list[str],
    ) -> tuple[np.ndarray, list[str]]:
        feature_chunks: list[np.ndarray] = []
        feature_names: list[str] = []
        vec = np.asarray(vector, dtype=np.float64)

        for feature_name in features:
            if feature_name == "embedding":
                feature_chunks.append(vec)
                feature_names.extend(
                    [f"embedding_{idx}" for idx in range(vec.shape[0])]
                )
            elif feature_name == "norm":
                feature_chunks.append(np.array([np.linalg.norm(vec)], dtype=np.float64))
                feature_names.append("norm")
            elif feature_name == "polar":
                polar_values, polar_names = self._spherical_coordinates(vec)
                feature_chunks.append(polar_values)
                feature_names.extend(polar_names)
            elif feature_name == "descriptive":
                descriptor_values, descriptor_names = self._descriptive_features(vec)
                feature_chunks.append(descriptor_values)
                feature_names.extend(descriptor_names)
            else:
                raise ValueError(f"Unsupported feature extractor '{feature_name}'.")

        return np.concatenate(feature_chunks), feature_names

    def _spherical_coordinates(
        self, vector: np.ndarray
    ) -> tuple[np.ndarray, list[str]]:
        norm = float(np.linalg.norm(vector))
        n_dims = int(vector.shape[0])
        if n_dims == 0:
            return np.array([0.0], dtype=np.float64), ["polar_radius"]

        values = [norm]
        names = ["polar_radius"]
        if n_dims == 1:
            return np.asarray(values, dtype=np.float64), names

        for idx in range(n_dims - 1):
            if idx == n_dims - 2:
                angle = float(np.arctan2(vector[-1], vector[-2]))
                names.append(f"polar_phi_{idx}")
            else:
                denom = float(np.linalg.norm(vector[idx:]))
                if denom <= 1e-12:
                    angle = 0.0
                else:
                    cosine = float(np.clip(vector[idx] / denom, -1.0, 1.0))
                    angle = float(np.arccos(cosine))
                names.append(f"polar_theta_{idx}")
            values.append(angle)

        return np.asarray(values, dtype=np.float64), names

    def _descriptive_features(self, vector: np.ndarray) -> tuple[np.ndarray, list[str]]:
        vec = np.asarray(vector, dtype=np.float64)
        if vec.size == 0:
            zeros = np.zeros(28, dtype=np.float64)
            names = [
                "mean",
                "median",
                "std",
                "variance",
                "iqr",
                "mad",
                "min",
                "max",
                "range",
                "skewness",
                "kurtosis",
                "trimmed_mean_10",
                "q10",
                "q25",
                "q50",
                "q75",
                "q90",
                "l1",
                "l2",
                "linf",
                "rms",
                "energy",
                "near_zero_fraction",
                "gini_concentration",
                "binned_entropy",
                "positive_fraction",
                "negative_fraction",
            ]
            return zeros[: len(names)], names

        mean = float(np.mean(vec))
        median = float(np.median(vec))
        std = float(np.std(vec))
        variance = float(np.var(vec))
        q10, q25, q50, q75, q90 = np.quantile(vec, [0.1, 0.25, 0.5, 0.75, 0.9])
        iqr = float(q75 - q25)
        mad = float(np.median(np.abs(vec - median)))
        vmin = float(np.min(vec))
        vmax = float(np.max(vec))
        vrange = float(vmax - vmin)

        centered = vec - mean
        centered_std = float(np.std(centered))
        if centered_std <= 1e-12:
            skewness = 0.0
            kurtosis = 0.0
        else:
            z = centered / centered_std
            skewness = float(np.mean(z**3))
            kurtosis = float(np.mean(z**4) - 3.0)

        trim = int(np.floor(vec.size * 0.1))
        if trim > 0 and vec.size > 2 * trim:
            sorted_vec = np.sort(vec)
            trimmed_mean = float(np.mean(sorted_vec[trim:-trim]))
        else:
            trimmed_mean = mean

        abs_vec = np.abs(vec)
        l1 = float(np.sum(abs_vec))
        l2 = float(np.linalg.norm(vec))
        linf = float(np.max(abs_vec))
        rms = float(np.sqrt(np.mean(vec**2)))
        energy = float(np.sum(vec**2))

        near_zero_threshold = max(1e-8, 1e-6 * max(1.0, linf))
        near_zero_fraction = float(np.mean(abs_vec <= near_zero_threshold))

        gini_concentration = self._gini_coefficient(abs_vec)
        binned_entropy = self._binned_entropy(vec)

        positive_fraction = float(np.mean(vec > 0.0))
        negative_fraction = float(np.mean(vec < 0.0))

        values = np.array(
            [
                mean,
                median,
                std,
                variance,
                iqr,
                mad,
                vmin,
                vmax,
                vrange,
                skewness,
                kurtosis,
                trimmed_mean,
                float(q10),
                float(q25),
                float(q50),
                float(q75),
                float(q90),
                l1,
                l2,
                linf,
                rms,
                energy,
                near_zero_fraction,
                gini_concentration,
                binned_entropy,
                positive_fraction,
                negative_fraction,
            ],
            dtype=np.float64,
        )
        names = [
            "mean",
            "median",
            "std",
            "variance",
            "iqr",
            "mad",
            "min",
            "max",
            "range",
            "skewness",
            "kurtosis",
            "trimmed_mean_10",
            "q10",
            "q25",
            "q50",
            "q75",
            "q90",
            "l1",
            "l2",
            "linf",
            "rms",
            "energy",
            "near_zero_fraction",
            "gini_concentration",
            "binned_entropy",
            "positive_fraction",
            "negative_fraction",
        ]
        return values, names

    def _gini_coefficient(self, values: np.ndarray) -> float:
        arr = np.asarray(values, dtype=np.float64)
        if arr.size == 0:
            return 0.0
        total = float(np.sum(arr))
        if total <= 1e-12:
            return 0.0
        sorted_arr = np.sort(arr)
        n = sorted_arr.size
        index = np.arange(1, n + 1, dtype=np.float64)
        gini = float((2.0 * np.sum(index * sorted_arr) / (n * total)) - (n + 1) / n)
        return max(0.0, min(1.0, gini))

    def _binned_entropy(self, values: np.ndarray, bins: int = 16) -> float:
        arr = np.asarray(values, dtype=np.float64)
        if arr.size == 0:
            return 0.0
        if float(np.max(arr)) == float(np.min(arr)):
            return 0.0
        counts, _ = np.histogram(arr, bins=bins)
        total = np.sum(counts)
        if total <= 0:
            return 0.0
        probs = counts.astype(np.float64) / float(total)
        probs = probs[probs > 0]
        return float(-np.sum(probs * np.log(probs)))

    def _fit_supervised_model(
        self,
        X: np.ndarray,
        y: np.ndarray,
        task: str,
        l1_ratio: float,
        alpha: float,
        max_iter: int,
    ) -> Dict[str, Any]:
        from sklearn.decomposition import PCA
        from sklearn.linear_model import ElasticNet, LogisticRegression
        from sklearn.metrics import balanced_accuracy_score, mean_absolute_error
        from sklearn.model_selection import LeaveOneOut
        from sklearn.preprocessing import StandardScaler

        if X.shape[0] <= 1:
            raise ValueError("Leave-one-out evaluation requires at least two samples.")

        if task == "classification":
            classes, counts = np.unique(y, return_counts=True)
            if classes.shape[0] < 2:
                raise ValueError("Classification requires at least two classes.")
            if np.min(counts) < 2:
                raise ValueError(
                    "Leave-one-out classification requires at least two samples per class."
                )

        y_true: list[Any] = []
        y_pred: list[Any] = []
        for train_idx, test_idx in LeaveOneOut().split(X):
            X_train = X[train_idx]
            X_test = X[test_idx]
            y_train = y[train_idx]
            y_test = y[test_idx]

            fold_scaler = StandardScaler()
            X_train_scaled = fold_scaler.fit_transform(X_train)
            X_test_scaled = fold_scaler.transform(X_test)

            fold_pca = PCA(n_components=0.95, svd_solver="full")
            X_train_reduced = fold_pca.fit_transform(X_train_scaled)
            X_test_reduced = fold_pca.transform(X_test_scaled)

            if task == "regression":
                fold_model = ElasticNet(
                    alpha=alpha, l1_ratio=l1_ratio, max_iter=max_iter
                )
            else:
                fold_model = LogisticRegression(
                    penalty="elasticnet",
                    solver="saga",
                    l1_ratio=l1_ratio,
                    max_iter=max_iter,
                )

            fold_model.fit(X_train_reduced, y_train)
            fold_pred = fold_model.predict(X_test_reduced)
            y_true.extend(np.ravel(y_test).tolist())
            y_pred.extend(np.ravel(fold_pred).tolist())

        if task == "regression":
            training_score = float(
                mean_absolute_error(
                    np.asarray(y_true, dtype=np.float64),
                    np.asarray(y_pred, dtype=np.float64),
                )
            )
            score_metric = "l1_error"
        else:
            training_score = float(
                balanced_accuracy_score(
                    np.asarray(y_true, dtype=object),
                    np.asarray(y_pred, dtype=object),
                )
            )
            score_metric = "balanced_accuracy"

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        pca = PCA(n_components=0.95, svd_solver="full")
        X_reduced = pca.fit_transform(X_scaled)

        if task == "regression":
            model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=max_iter)
        else:
            model = LogisticRegression(
                penalty="elasticnet",
                solver="saga",
                l1_ratio=l1_ratio,
                max_iter=max_iter,
            )

        model.fit(X_reduced, y)

        model_info: Dict[str, Any] = {
            "estimator": type(model).__name__,
            "l1_ratio": l1_ratio,
        }
        if hasattr(model, "alpha_"):
            model_info["alpha"] = float(model.alpha_)
        elif hasattr(model, "alpha"):
            model_info["alpha"] = float(model.alpha)
        if hasattr(model, "C_"):
            model_info["C"] = float(np.ravel(model.C_)[0])

        return {
            "scaler": scaler,
            "pca": pca,
            "model": model,
            "training_score": training_score,
            "score_metric": score_metric,
            "model_info": model_info,
        }

    def _compute_importances_from_pca_model(
        self,
        scaler,
        pca,
        model,
        feature_names: list[str],
    ) -> list[Dict[str, Any]]:
        model_weights = np.asarray(model.coef_, dtype=np.float64)
        if model_weights.ndim == 1:
            model_weights = model_weights.reshape(1, -1)

        pca_components = np.asarray(pca.components_, dtype=np.float64)
        weights_scaled = model_weights @ pca_components
        weights_original = weights_scaled / scaler.scale_.reshape(1, -1)
        if weights_original.shape[0] == 1:
            aggregate = weights_original[0]
        else:
            aggregate = np.mean(np.abs(weights_original), axis=0)

        return [
            {
                "feature": feature_names[idx],
                "importance": float(aggregate[idx]),
            }
            for idx in range(len(feature_names))
        ]

    def _select_importances_for_cumulative_share(
        self,
        importances: list[Dict[str, Any]],
        target_share: float,
    ) -> tuple[list[Dict[str, Any]], float]:
        if not importances:
            return [], 0.0

        sorted_importances = sorted(
            importances,
            key=lambda item: abs(float(item["importance"])),
            reverse=True,
        )

        total_abs = float(
            np.sum([abs(float(item["importance"])) for item in sorted_importances])
        )
        if total_abs <= 1e-12:
            return sorted_importances[:1], 0.0

        cumulative_abs = 0.0
        selected: list[Dict[str, Any]] = []
        for item in sorted_importances:
            selected.append(item)
            cumulative_abs += abs(float(item["importance"]))
            if cumulative_abs / total_abs >= target_share:
                break

        return selected, cumulative_abs / total_abs

    def _save_supervised_importance_plot(
        self,
        model_dir: Path,
        importances: list[Dict[str, Any]],
        training_score: float,
        score_metric: str,
    ) -> None:
        if not importances:
            return

        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "matplotlib is required to render supervised plots."
            ) from exc

        labels = [item["feature"] for item in importances][::-1]
        values = [float(item["importance"]) for item in importances][::-1]

        fig_height = max(4.0, 0.25 * len(labels))
        fig, ax = plt.subplots(figsize=(8, fig_height))
        ax.barh(labels, values, color="#4c72b0")
        ax.set_xlabel("Importance")
        ax.set_ylabel("Feature")
        ax.set_title(
            f"Supervised feature importance (LOO {score_metric}={training_score:.4f})"
        )
        ax.grid(axis="x", linestyle="--", alpha=0.35)

        fig.tight_layout()
        fig.savefig(model_dir / "feature_importance.png", dpi=self._plot_dpi)
        plt.close(fig)
