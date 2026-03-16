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

            supervised_cfg = self.config.supervised
            base_pass = self._run_supervised_pass(
                X=matrix,
                y=y,
                task=task,
                feature_names=feature_names,
            )

            self._save_json(
                model_dir / "supervised_report.json",
                self._build_supervised_report_payload(
                    task=task,
                    cv_label=cv_label,
                    cv_splits=cv_splits,
                    matrix=matrix,
                    features=self.config.supervised["features"],
                    pass_result=base_pass,
                    extra_payload={
                        "residualized_rerun": bool(
                            supervised_cfg.get("rerun_on_residualized_inputs", False)
                        )
                    },
                ),
            )
            self._save_supervised_model_artifacts(
                model_dir=model_dir,
                scaler=base_pass["result"]["scaler"],
                pca=base_pass["result"]["pca"],
                model=base_pass["result"]["model"],
                feature_names=feature_names,
                task=task,
                model_info=base_pass["result"]["model_info"],
            )
            self._save_supervised_importance_plot(
                model_dir=model_dir,
                importances=base_pass["selected_importances"],
                training_score=float(base_pass["result"]["training_score"]),
                score_metric=str(base_pass["result"]["score_metric"]),
            )

            if bool(supervised_cfg.get("rerun_on_residualized_inputs", False)):
                weights_original = self._compute_original_space_weights(
                    scaler=base_pass["result"]["scaler"],
                    pca=base_pass["result"]["pca"],
                    model=base_pass["result"]["model"],
                )
                basis = self._orthonormal_basis_from_weights(weights_original)
                residual_matrix = self._project_matrix_to_orthogonal_complement(
                    matrix,
                    basis,
                )
                residual_pass = self._run_supervised_pass(
                    X=residual_matrix,
                    y=y,
                    task=task,
                    feature_names=feature_names,
                )

                residual_dir = model_dir / "residualized"
                residual_dir.mkdir(parents=True, exist_ok=True)
                self._save_json(
                    residual_dir / "supervised_report.json",
                    self._build_supervised_report_payload(
                        task=task,
                        cv_label=cv_label,
                        cv_splits=cv_splits,
                        matrix=residual_matrix,
                        features=self.config.supervised["features"],
                        pass_result=residual_pass,
                        extra_payload={
                            "input_space_transform": "x_perp_from_base_supervised_weights",
                            "projection_basis_rank": int(basis.shape[1]),
                        },
                    ),
                )
                self._save_supervised_model_artifacts(
                    model_dir=residual_dir,
                    scaler=residual_pass["result"]["scaler"],
                    pca=residual_pass["result"]["pca"],
                    model=residual_pass["result"]["model"],
                    feature_names=feature_names,
                    task=task,
                    model_info=residual_pass["result"]["model_info"],
                )
                self._save_supervised_importance_plot(
                    model_dir=residual_dir,
                    importances=residual_pass["selected_importances"],
                    training_score=float(residual_pass["result"]["training_score"]),
                    score_metric=str(residual_pass["result"]["score_metric"]),
                )

    def _run_supervised_pass(
        self,
        X: np.ndarray,
        y: np.ndarray,
        task: str,
        feature_names: list[str],
    ) -> Dict[str, Any]:
        assert self.config.supervised is not None
        supervised_cfg = self.config.supervised
        result = self._fit_supervised_model(
            X=X,
            y=y,
            task=task,
            estimator=str(supervised_cfg.get("estimator", "")),
            l1_ratio=float(supervised_cfg["l1_ratio"]),
            alpha=float(supervised_cfg["alpha"]),
            C=float(supervised_cfg.get("C", 1.0)),
            epsilon=float(supervised_cfg.get("epsilon", 0.0)),
            loss=str(supervised_cfg.get("loss", "epsilon_insensitive")),
            class_weight=supervised_cfg.get("class_weight"),
            use_pca=bool(supervised_cfg.get("use_pca", False)),
            max_iter=int(supervised_cfg["max_iter"]),
        )

        importances = self._compute_cv_importances_from_pca_models(
            X=X,
            y=y,
            task=task,
            estimator=str(supervised_cfg.get("estimator", "")),
            l1_ratio=float(supervised_cfg["l1_ratio"]),
            alpha=float(supervised_cfg["alpha"]),
            C=float(supervised_cfg.get("C", 1.0)),
            epsilon=float(supervised_cfg.get("epsilon", 0.0)),
            loss=str(supervised_cfg.get("loss", "epsilon_insensitive")),
            class_weight=supervised_cfg.get("class_weight"),
            use_pca=bool(supervised_cfg.get("use_pca", False)),
            max_iter=int(supervised_cfg["max_iter"]),
            feature_names=feature_names,
        )
        selected_importances, selected_share = (
            self._select_importances_for_cumulative_share(
                importances=importances,
                target_share=0.5,
            )
        )
        return {
            "result": result,
            "selected_importances": selected_importances,
            "selected_share": float(selected_share),
        }

    def _build_supervised_report_payload(
        self,
        task: str,
        cv_label: str,
        cv_splits: int,
        matrix: np.ndarray,
        features: list[str],
        pass_result: Dict[str, Any],
        extra_payload: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        result = pass_result["result"]
        payload: Dict[str, Any] = {
            "task": task,
            "cv_strategy": cv_label,
            "cv_folds": cv_splits,
            "n_samples": int(matrix.shape[0]),
            "n_features": int(matrix.shape[1]),
            "n_components": int(result["n_components"]),
            "features": list(features),
            "model": result["model_info"],
            "score_metric": result["score_metric"],
            "loo_score": float(result["training_score"]),
            "training_score": float(result["training_score"]),
            "feature_importance_target_share": 0.5,
            "feature_importance_selected_share": float(pass_result["selected_share"]),
            "feature_importance_method": "loo_mean_abs_coefficient",
            "top_feature_importances": pass_result["selected_importances"],
        }
        if extra_payload:
            payload.update(extra_payload)
        return payload

    def _orthonormal_basis_from_weights(self, weights: np.ndarray) -> np.ndarray:
        basis = np.asarray(weights, dtype=np.float64).T
        q, _ = np.linalg.qr(basis)
        return q

    def _project_matrix_to_orthogonal_complement(
        self,
        matrix: np.ndarray,
        basis: np.ndarray,
    ) -> np.ndarray:
        array = np.asarray(matrix, dtype=np.float64).copy()
        if array.ndim != 2:
            raise ValueError("Expected a 2D supervised feature matrix for projection.")
        if array.shape[1] != basis.shape[0]:
            raise ValueError(
                "Cannot project supervised feature matrix to residual space due to "
                f"dimension mismatch: features={array.shape[1]}, basis={basis.shape[0]}."
            )
        projected = array @ basis @ basis.T
        return array - projected

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
        estimator: str,
        l1_ratio: float,
        alpha: float,
        C: float,
        epsilon: float,
        loss: str,
        class_weight: Any,
        use_pca: bool,
        max_iter: int,
    ) -> Dict[str, Any]:
        from sklearn.decomposition import PCA
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

            if use_pca:
                fold_pca = PCA(n_components=0.95, svd_solver="full")
                X_train_model = fold_pca.fit_transform(X_train_scaled)
                X_test_model = fold_pca.transform(X_test_scaled)
            else:
                X_train_model = X_train_scaled
                X_test_model = X_test_scaled

            fold_model = self._build_supervised_estimator(
                task=task,
                estimator=estimator,
                l1_ratio=l1_ratio,
                alpha=alpha,
                C=C,
                epsilon=epsilon,
                loss=loss,
                class_weight=class_weight,
                max_iter=max_iter,
            )

            fold_model.fit(X_train_model, y_train)
            fold_pred = fold_model.predict(X_test_model)
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

        pca = None
        X_model = X_scaled
        if use_pca:
            pca = PCA(n_components=0.95, svd_solver="full")
            X_model = pca.fit_transform(X_scaled)

        model = self._build_supervised_estimator(
            task=task,
            estimator=estimator,
            l1_ratio=l1_ratio,
            alpha=alpha,
            C=C,
            epsilon=epsilon,
            loss=loss,
            class_weight=class_weight,
            max_iter=max_iter,
        )

        model.fit(X_model, y)

        model_info: Dict[str, Any] = {
            "estimator": type(model).__name__,
        }
        if task == "classification" and estimator == "linear_svm":
            model_info["C"] = float(C)
            model_info["class_weight"] = class_weight
        elif task == "regression" and estimator == "linear_svr":
            model_info["C"] = float(C)
            model_info["epsilon"] = float(epsilon)
            model_info["loss"] = loss
        else:
            model_info["l1_ratio"] = l1_ratio
        if hasattr(model, "alpha_"):
            model_info["alpha"] = float(model.alpha_)
        elif hasattr(model, "alpha"):
            model_info["alpha"] = float(model.alpha)
        if hasattr(model, "C_"):
            model_info["C"] = float(np.ravel(model.C_)[0])

        return {
            "scaler": scaler,
            "model": model,
            "n_components": int(X_model.shape[1]),
            "pca": pca,
            "training_score": training_score,
            "score_metric": score_metric,
            "model_info": model_info,
        }

    def _build_supervised_estimator(
        self,
        task: str,
        l1_ratio: float,
        alpha: float,
        max_iter: int,
        estimator: str = "",
        C: float = 1.0,
        epsilon: float = 0.0,
        loss: str = "epsilon_insensitive",
        class_weight: Any = None,
    ):
        from sklearn.linear_model import ElasticNet, LogisticRegression
        from sklearn.svm import LinearSVC, LinearSVR

        if not estimator:
            estimator = "elasticnet" if task == "regression" else "logistic_regression"

        if task == "regression" and estimator == "elasticnet":
            return ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=max_iter)

        if task == "regression" and estimator == "linear_svr":
            return LinearSVR(
                C=C,
                epsilon=epsilon,
                loss=loss,
                max_iter=max_iter,
            )

        if task == "classification" and estimator == "linear_svm":
            return LinearSVC(
                C=C,
                class_weight=class_weight,
                max_iter=max_iter,
                dual="auto",
            )

        return LogisticRegression(
            penalty="elasticnet",
            solver="saga",
            l1_ratio=l1_ratio,
            C=1.0 / float(alpha),
            max_iter=max_iter,
        )

    def _compute_cv_importances_from_pca_models(
        self,
        X: np.ndarray,
        y: np.ndarray,
        task: str,
        estimator: str,
        l1_ratio: float,
        alpha: float,
        C: float,
        epsilon: float,
        loss: str,
        class_weight: Any,
        use_pca: bool,
        max_iter: int,
        feature_names: list[str],
    ) -> list[Dict[str, Any]]:
        from sklearn.model_selection import LeaveOneOut
        from sklearn.preprocessing import StandardScaler
        from sklearn.decomposition import PCA

        fold_importances: list[np.ndarray] = []
        for train_idx, _ in LeaveOneOut().split(X):
            X_train = X[train_idx]
            y_train = y[train_idx]

            fold_scaler = StandardScaler()
            X_train_scaled = fold_scaler.fit_transform(X_train)

            fold_pca = None
            X_train_model = X_train_scaled
            if use_pca:
                fold_pca = PCA(n_components=0.95, svd_solver="full")
                X_train_model = fold_pca.fit_transform(X_train_scaled)

            fold_model = self._build_supervised_estimator(
                task=task,
                estimator=estimator,
                l1_ratio=l1_ratio,
                alpha=alpha,
                C=C,
                epsilon=epsilon,
                loss=loss,
                class_weight=class_weight,
                max_iter=max_iter,
            )
            fold_model.fit(X_train_model, y_train)

            fold_items = self._compute_importances_from_pca_model(
                scaler=fold_scaler,
                pca=fold_pca,
                model=fold_model,
                feature_names=feature_names,
            )
            fold_importances.append(
                np.asarray(
                    [abs(float(item["importance"])) for item in fold_items],
                    dtype=np.float64,
                )
            )

        if not fold_importances:
            return []

        stacked = np.stack(fold_importances, axis=0)
        mean_importance = np.mean(stacked, axis=0)
        std_importance = np.std(stacked, axis=0)
        nonzero_fraction = np.mean(stacked > 1e-12, axis=0)

        return [
            {
                "feature": feature_names[idx],
                "importance": float(mean_importance[idx]),
                "importance_std": float(std_importance[idx]),
                "importance_nonzero_fraction": float(nonzero_fraction[idx]),
            }
            for idx in range(len(feature_names))
        ]

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

        if pca is None:
            weights_original = model_weights / scaler.scale_.reshape(1, -1)
        else:
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

    def _save_supervised_model_artifacts(
        self,
        model_dir: Path,
        scaler,
        pca,
        model,
        feature_names: list[str],
        task: str,
        model_info: Dict[str, Any],
    ) -> None:
        import joblib

        model_payload = {
            "task": task,
            "model_info": dict(model_info),
            "feature_names": list(feature_names),
            "scaler": scaler,
            "pca": pca,
            "model": model,
        }
        joblib.dump(model_payload, model_dir / "supervised_model.joblib")

        weights_original = self._compute_original_space_weights(
            scaler=scaler,
            pca=pca,
            model=model,
        )
        np.savez(
            model_dir / "supervised_model_projection.npz",
            weights_original=weights_original,
        )

    def _compute_original_space_weights(self, scaler, pca, model) -> np.ndarray:
        model_weights = np.asarray(model.coef_, dtype=np.float64)
        if model_weights.ndim == 1:
            model_weights = model_weights.reshape(1, -1)

        if pca is None:
            return model_weights / scaler.scale_.reshape(1, -1)

        pca_components = np.asarray(pca.components_, dtype=np.float64)
        weights_scaled = model_weights @ pca_components
        return weights_scaled / scaler.scale_.reshape(1, -1)

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
