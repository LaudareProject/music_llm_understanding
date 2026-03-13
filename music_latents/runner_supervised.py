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

        labeled_queries = [
            query for query in self.config.queries if query.target is not None
        ]
        if not labeled_queries:
            raise ValueError(
                "Supervised experiment requires query targets (target/label)."
            )

        query_ids = [query.query_id for query in labeled_queries]
        raw_targets = [query.target for query in labeled_queries]

        task = self.config.supervised["task"]
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

        cv_strategy, cv_label, cv_splits = self._determine_cv_strategy(
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
                cv_strategy=cv_strategy,
                cv_splits=cv_splits,
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

            top_k = int(self.config.supervised["top_k"])
            sorted_importances = sorted(
                importances,
                key=lambda item: abs(float(item["importance"])),
                reverse=True,
            )
            top_importances = sorted_importances[:top_k]

            payload = {
                "task": task,
                "cv_strategy": cv_label,
                "cv_folds": cv_splits,
                "n_samples": int(matrix.shape[0]),
                "n_features": int(matrix.shape[1]),
                "n_components": int(result["pca"].n_components_),
                "features": self.config.supervised["features"],
                "model": result["model_info"],
                "training_score": float(result["training_score"]),
                "top_feature_importances": top_importances,
            }
            self._save_json(model_dir / "supervised_report.json", payload)
            self._save_supervised_importance_plot(
                model_dir=model_dir,
                importances=top_importances,
            )




    def _determine_cv_strategy(self, sample_count: int) -> tuple[Any, str, int]:
        if sample_count > 50:
            return 5, "kfold_5", 5
        if sample_count > 1:
            from sklearn.model_selection import LeaveOneOut

            return LeaveOneOut(), "leave_one_out", sample_count
        return 1, "single_sample", 1




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
        cv_strategy: Any,
        cv_splits: int,
        l1_ratio: float,
        alpha: float,
        max_iter: int,
    ) -> Dict[str, Any]:
        from sklearn.decomposition import PCA
        from sklearn.linear_model import (
            ElasticNet,
            ElasticNetCV,
            LogisticRegression,
            LogisticRegressionCV,
        )
        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        pca = PCA(n_components=0.95, svd_solver="full")
        X_reduced = pca.fit_transform(X_scaled)

        if task == "regression":
            if cv_splits > 1:
                model = ElasticNetCV(
                    l1_ratio=[l1_ratio], cv=cv_strategy, max_iter=max_iter
                )
            else:
                model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=max_iter)
        else:
            classes = np.unique(y)
            if classes.shape[0] < 2:
                raise ValueError("Classification requires at least two classes.")
            if cv_splits > 1:
                model = LogisticRegressionCV(
                    cv=cv_strategy,
                    penalty="elasticnet",
                    solver="saga",
                    l1_ratios=[l1_ratio],
                    max_iter=max_iter,
                )
            else:
                model = LogisticRegression(
                    penalty="elasticnet",
                    solver="saga",
                    l1_ratio=l1_ratio,
                    max_iter=max_iter,
                )

        model.fit(X_reduced, y)
        training_score = float(model.score(X_reduced, y))

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




    def _save_supervised_importance_plot(
        self,
        model_dir: Path,
        importances: list[Dict[str, Any]],
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
        ax.set_title("Supervised feature importance")
        ax.grid(axis="x", linestyle="--", alpha=0.35)

        fig.tight_layout()
        fig.savefig(model_dir / "feature_importance.png", dpi=self._plot_dpi)
        plt.close(fig)

