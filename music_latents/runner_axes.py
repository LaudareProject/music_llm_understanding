from __future__ import annotations

import os
from pathlib import Path
import re
from typing import Any, Dict

import numpy as np

from music_latents.axes import build_axes, compute_projection_scores
from music_latents.config import AxisSpec, PlotSpec, is_pca_component_axis_name


class RunnerAxesMixin:
    def _save_axis_outputs(
        self,
        output_dir: Path,
        embeddings_by_model: Dict[str, Dict[str, np.ndarray]],
    ) -> None:
        axes_specs_by_name = {axis.name: axis for axis in self.config.axes}

        query_names = {query.query_id: query.name for query in self.config.queries}
        query_colors = {
            query.query_id: self._resolve_query_color(
                query_id=query.query_id,
                query_name=query.name,
                configured_color=query.color,
            )
            for query in self.config.queries
        }
        for average_group in self.config.averages:
            query_names[average_group.group_id] = average_group.name
            query_colors[average_group.group_id] = self._resolve_query_color(
                query_id=average_group.group_id,
                query_name=average_group.name,
                configured_color=average_group.color,
            )
        for median_group in self.config.medians:
            query_names[median_group.group_id] = median_group.name
            query_colors[median_group.group_id] = self._resolve_query_color(
                query_id=median_group.group_id,
                query_name=median_group.name,
                configured_color=median_group.color,
            )

        for model_name, model_embeddings in embeddings_by_model.items():
            raw_embeddings = {
                query_id: np.asarray(vector, dtype=np.float64)
                for query_id, vector in model_embeddings.items()
            }
            needs_center_and_whiten = any(
                plot_spec.options.center_and_whiten
                for plot_spec in self.config.plots.values()
            )
            centered_whitened_embeddings = (
                self._center_and_whiten_embeddings(raw_embeddings)
                if needs_center_and_whiten
                else raw_embeddings
            )
            model_dir = output_dir / model_name
            plots_root = model_dir / "plots"
            json_root = model_dir / "json"
            plots_root.mkdir(parents=True, exist_ok=True)
            json_root.mkdir(parents=True, exist_ok=True)

            for plot_name, plot_spec in self.config.plots.items():
                plot_embeddings = (
                    centered_whitened_embeddings
                    if plot_spec.options.center_and_whiten
                    else raw_embeddings
                )
                lda_info: Dict[str, Any] = {
                    "enabled": plot_spec.options.lda_groups,
                    "applied": False,
                }
                if plot_spec.options.lda_groups:
                    plot_embeddings, lda_info = self._apply_group_lda_embeddings(
                        embeddings=plot_embeddings
                    )

                pca_info: Dict[str, Any] = {
                    "enabled": plot_spec.options.pca,
                    "applied": False,
                    "variance_threshold": 0.9,
                }
                if plot_spec.options.pca:
                    plot_embeddings, pca_info = self._apply_pca_embeddings(
                        plot_embeddings
                    )

                mds_info: Dict[str, Any] = {
                    "enabled": plot_spec.mode == "mds",
                    "applied": False,
                }
                mds_embeddings: Dict[str, np.ndarray] | None = None
                mds_dissimilarities: np.ndarray | None = None
                if plot_spec.mode == "mds":
                    (
                        mds_embeddings,
                        mds_info,
                        mds_dissimilarities,
                    ) = self._apply_mds_embeddings(plot_embeddings)

                query_ids = self._select_plot_query_ids(
                    embeddings=plot_embeddings,
                    averaged_only=plot_spec.options.averaged_only,
                    group_filter=plot_spec.options.group_filter,
                )

                plot_slug = "".join(
                    ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in plot_name
                )
                processing_labels = self._active_processing_labels(plot_spec)
                processing_folder = self._processing_folder_name(processing_labels)
                canonical_plots_dir = plots_root / processing_folder
                canonical_json_dir = json_root / processing_folder
                canonical_plots_dir.mkdir(parents=True, exist_ok=True)
                canonical_json_dir.mkdir(parents=True, exist_ok=True)
                alias_plot_dirs = [plots_root / label for label in processing_labels]
                alias_json_dirs = [json_root / label for label in processing_labels]

                if plot_spec.mode == "cosine":
                    plot_file_suffix = "_lda" if plot_spec.options.lda_groups else ""
                    self._save_cosine_polar_plot(
                        plots_dir=canonical_plots_dir,
                        json_dir=canonical_json_dir,
                        plot_slug=plot_slug,
                        plot_file_suffix=plot_file_suffix,
                        query_ids=query_ids,
                        embeddings=plot_embeddings,
                        query_names=query_names,
                        query_colors=query_colors,
                        cosine_references=plot_spec.cosine_references,
                        variant="lda" if lda_info.get("applied", False) else "standard",
                    )
                    self._link_plot_outputs(
                        file_name=f"cosine_{plot_slug}{plot_file_suffix}.png",
                        canonical_dir=canonical_plots_dir,
                        alias_dirs=alias_plot_dirs,
                    )
                    self._link_plot_outputs(
                        file_name=f"cosine_{plot_slug}.json",
                        canonical_dir=canonical_json_dir,
                        alias_dirs=alias_json_dirs,
                    )
                    continue

                axes = []
                scores: Dict[str, Dict[str, float]] = {}
                if plot_spec.mode == "axes":
                    plot_axes_specs = self._resolve_plot_axes_specs(
                        plot_axes=plot_spec.axes,
                        axes_specs_by_name=axes_specs_by_name,
                        pca_info=pca_info,
                    )
                    axes = build_axes(
                        plot_axes_specs, plot_embeddings, plot_spec.options
                    )
                    scores = compute_projection_scores(
                        axes, plot_embeddings, plot_spec.options
                    )

                axes_payload = {
                    "plot": plot_name,
                    "options": {
                        "center_and_whiten": plot_spec.options.center_and_whiten,
                        "translate": plot_spec.options.translate,
                        "orthonormalize": plot_spec.options.orthonormalize,
                        "normalize_vectors": plot_spec.options.normalize_vectors,
                        "origin_strategy": plot_spec.options.origin_strategy,
                        "pca": plot_spec.options.pca,
                        "lda_groups": plot_spec.options.lda_groups,
                        "averaged_only": plot_spec.options.averaged_only,
                        "group_filter": plot_spec.options.group_filter,
                        "mode": plot_spec.mode,
                        "cosine": plot_spec.cosine_references,
                    },
                    "lda": lda_info,
                    "pca": pca_info,
                    "mds": mds_info,
                    "axes": [
                        {
                            "name": axis.name,
                            "query_a": axis.query_a,
                            "query_b": axis.query_b,
                            "method": axis.method,
                            "params": axis.params,
                            "origin": axis.origin.tolist(),
                            "direction": axis.direction.tolist(),
                        }
                        for axis in axes
                    ],
                }
                self._save_json(
                    canonical_json_dir / f"axes_{plot_slug}.json", axes_payload
                )

                scores_payload = {
                    "plot": plot_name,
                    "axes": [axis.name for axis in axes],
                    "queries": [
                        {
                            "id": query_id,
                            "name": query_names.get(query_id, query_id),
                            "color": query_colors.get(query_id),
                            "scores": scores.get(query_id, {}),
                        }
                        for query_id in query_ids
                    ],
                }
                self._save_json(
                    canonical_json_dir / f"projections_{plot_slug}.json", scores_payload
                )

                if plot_spec.mode == "axes":
                    plot_file_suffix = "_lda" if plot_spec.options.lda_groups else ""
                    self._save_axis_plot(
                        canonical_plots_dir,
                        plot_slug,
                        plot_file_suffix,
                        axes,
                        scores,
                        query_ids,
                        query_names,
                        query_colors,
                    )
                    if len(axes) == 1:
                        self._link_plot_outputs(
                            file_name=f"projections_{plot_slug}_1d{plot_file_suffix}.png",
                            canonical_dir=canonical_plots_dir,
                            alias_dirs=alias_plot_dirs,
                        )
                    else:
                        self._link_plot_outputs(
                            file_name=f"projections_{plot_slug}_2d{plot_file_suffix}.png",
                            canonical_dir=canonical_plots_dir,
                            alias_dirs=alias_plot_dirs,
                        )
                    self._link_plot_outputs(
                        file_name=f"axes_{plot_slug}.json",
                        canonical_dir=canonical_json_dir,
                        alias_dirs=alias_json_dirs,
                    )
                    self._link_plot_outputs(
                        file_name=f"projections_{plot_slug}.json",
                        canonical_dir=canonical_json_dir,
                        alias_dirs=alias_json_dirs,
                    )
                if plot_spec.mode == "mds" and mds_embeddings is not None:
                    self._save_mds_plots(
                        plots_dir=canonical_plots_dir,
                        json_dir=canonical_json_dir,
                        plot_slug=plot_slug,
                        plot_name=plot_name,
                        query_ids=list(mds_embeddings.keys()),
                        query_names=query_names,
                        query_colors=query_colors,
                        mds_embeddings=mds_embeddings,
                        dissimilarities=mds_dissimilarities,
                        mds_info=mds_info,
                    )
                    self._link_plot_outputs(
                        file_name=f"mds_{plot_slug}_2d.png",
                        canonical_dir=canonical_plots_dir,
                        alias_dirs=alias_plot_dirs,
                    )
                    self._link_plot_outputs(
                        file_name=f"mds_{plot_slug}_shepard.png",
                        canonical_dir=canonical_plots_dir,
                        alias_dirs=alias_plot_dirs,
                    )
                    self._link_plot_outputs(
                        file_name=f"mds_{plot_slug}.json",
                        canonical_dir=canonical_json_dir,
                        alias_dirs=alias_json_dirs,
                    )

    def _active_processing_labels(self, plot_spec: PlotSpec) -> list[str]:
        labels: list[str] = []
        options = plot_spec.options
        if options.lda_groups:
            labels.append("lda")
        if plot_spec.mode == "mds":
            labels.append("mds")
        if options.orthonormalize:
            labels.append("orthonormal")
        if options.pca:
            labels.append("pca")
        return labels

    def _processing_folder_name(self, labels: list[str]) -> str:
        if not labels:
            return "standard"
        return "+".join(labels)

    def _link_plot_outputs(
        self,
        file_name: str,
        canonical_dir: Path,
        alias_dirs: list[Path],
    ) -> None:
        if len(alias_dirs) <= 1:
            return

        source = canonical_dir / file_name
        if not source.exists():
            return

        for alias_dir in alias_dirs:
            if alias_dir == canonical_dir:
                continue
            alias_dir.mkdir(parents=True, exist_ok=True)
            alias_path = alias_dir / file_name
            if alias_path.exists() or alias_path.is_symlink():
                if alias_path.is_dir() and not alias_path.is_symlink():
                    continue
                alias_path.unlink()
            target = Path(os.path.relpath(source, alias_dir))
            alias_path.symlink_to(target)

    def _save_cosine_polar_plot(
        self,
        plots_dir: Path,
        json_dir: Path,
        plot_slug: str,
        plot_file_suffix: str,
        query_ids: list[str],
        embeddings: Dict[str, np.ndarray],
        query_names: Dict[str, str],
        query_colors: Dict[str, Any],
        cosine_references: list[str],
        variant: str,
    ) -> None:
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "matplotlib is required to render cosine polar plots."
            ) from exc

        reference_a, reference_b = cosine_references
        if reference_a not in embeddings:
            raise ValueError(f"Cosine reference '{reference_a}' is not available.")
        if reference_b not in embeddings:
            raise ValueError(f"Cosine reference '{reference_b}' is not available.")

        ref_a = embeddings[reference_a]
        ref_b = embeddings[reference_b]
        ref_a_norm = np.linalg.norm(ref_a)
        ref_b_norm = np.linalg.norm(ref_b)

        if ref_a_norm == 0:
            raise ValueError(f"Cosine reference '{reference_a}' has zero norm.")
        if ref_b_norm == 0:
            raise ValueError(f"Cosine reference '{reference_b}' has zero norm.")

        ref_a_unit = ref_a / ref_a_norm
        ref_b_unit = ref_b / ref_b_norm
        cos_ab = float(np.dot(ref_a_unit, ref_b_unit))

        ref_b_orth = ref_b_unit - cos_ab * ref_a_unit
        ref_b_orth_norm = float(np.linalg.norm(ref_b_orth))
        references_collinear = ref_b_orth_norm <= 1e-12

        points_payload: list[Dict[str, Any]] = []
        fig, ax = plt.subplots(figsize=(6, 5))
        for query_id in query_ids:
            vector = embeddings.get(query_id)
            if vector is None:
                continue

            vector_norm = np.linalg.norm(vector)
            if vector_norm == 0:
                cos_a = 0.0
                cos_b = 0.0
            else:
                vector_unit = vector / vector_norm
                cos_a = float(np.dot(vector_unit, ref_a_unit))
                cos_b = float(np.dot(vector_unit, ref_b_unit))

            x = float(cos_a)
            if references_collinear:
                y = 0.0
            else:
                y = float((cos_b - (cos_a * cos_ab)) / ref_b_orth_norm)
            plane_norm = float(np.sqrt((x * x) + (y * y)))

            ax.scatter(x, y, s=40, c=query_colors.get(query_id))
            ax.text(x, y, f" {query_names.get(query_id, query_id)}", fontsize=7)

            points_payload.append(
                {
                    "id": query_id,
                    "name": query_names.get(query_id, query_id),
                    "color": query_colors.get(query_id),
                    "cosine_reference_a": cos_a,
                    "cosine_reference_b": cos_b,
                    "cosine_reference_pair": cos_ab,
                    "references_collinear": references_collinear,
                    "plane_norm": plane_norm,
                    "x": x,
                    "y": y,
                }
            )

        unit_circle = plt.Circle(
            (0.0, 0.0), 1.0, fill=False, color="#777777", linewidth=1.0, linestyle="--"
        )
        ax.add_artist(unit_circle)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel(f"coord along {reference_a}")
        ax.set_ylabel(f"coord orthogonal to {reference_a}")
        ax.set_title(
            f"Cosine-derived 2D map ({variant}) | ref: {reference_a}, {reference_b}"
        )
        ax.grid(True, linestyle="--", alpha=0.35)

        fig.tight_layout()
        fig.savefig(
            plots_dir / f"cosine_{plot_slug}{plot_file_suffix}.png", dpi=self._plot_dpi
        )
        plt.close(fig)

        payload = {
            "plot": plot_slug,
            "variant": variant,
            "references": {
                "a": reference_a,
                "b": reference_b,
                "cosine_ab": cos_ab,
                "orthogonal_norm": ref_b_orth_norm,
                "collinear": references_collinear,
            },
            "points": points_payload,
        }
        self._save_json(json_dir / f"cosine_{plot_slug}.json", payload)

    def _save_mds_plots(
        self,
        plots_dir: Path,
        json_dir: Path,
        plot_slug: str,
        plot_name: str,
        query_ids: list[str],
        query_names: Dict[str, str],
        query_colors: Dict[str, Any],
        mds_embeddings: Dict[str, np.ndarray],
        dissimilarities: np.ndarray | None,
        mds_info: Dict[str, Any],
    ) -> None:
        if dissimilarities is None:
            return

        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "matplotlib is required to render MDS plots."
            ) from exc

        coords_matrix = np.stack(
            [mds_embeddings[query_id] for query_id in query_ids], axis=0
        )

        fig, ax = plt.subplots(figsize=(6, 5))
        points_payload: list[Dict[str, Any]] = []
        for query_id, coords in zip(query_ids, coords_matrix):
            x, y = float(coords[0]), float(coords[1])
            ax.scatter(x, y, s=40, c=query_colors.get(query_id))
            ax.text(x, y, f" {query_names.get(query_id, query_id)}", fontsize=7)
            points_payload.append(
                {
                    "id": query_id,
                    "name": query_names.get(query_id, query_id),
                    "color": query_colors.get(query_id),
                    "x": x,
                    "y": y,
                }
            )
        ax.set_xlabel("MDS dim 1")
        ax.set_ylabel("MDS dim 2")
        ax.set_title(f"MDS embedding ({plot_name})")
        ax.grid(True, linestyle="--", alpha=0.35)

        fig.tight_layout()
        fig.savefig(plots_dir / f"mds_{plot_slug}_2d.png", dpi=self._plot_dpi)
        plt.close(fig)

        distances = np.linalg.norm(
            coords_matrix[:, None, :] - coords_matrix[None, :, :], axis=-1
        )
        upper = np.triu_indices(len(query_ids), k=1)
        shepard_payload: list[Dict[str, Any]] = []
        shepard_x = dissimilarities[upper]
        shepard_y = distances[upper]
        for idx in range(len(shepard_x)):
            i = upper[0][idx]
            j = upper[1][idx]
            shepard_payload.append(
                {
                    "query_a": query_ids[i],
                    "query_b": query_ids[j],
                    "dissimilarity": float(shepard_x[idx]),
                    "distance": float(shepard_y[idx]),
                }
            )

        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(shepard_x, shepard_y, s=24, alpha=0.6, color="#4c72b0")
        ax.set_xlabel("Cosine dissimilarity")
        ax.set_ylabel("MDS Euclidean distance")
        ax.set_title(f"Shepard plot ({plot_name})")
        ax.grid(True, linestyle="--", alpha=0.35)

        fig.tight_layout()
        fig.savefig(plots_dir / f"mds_{plot_slug}_shepard.png", dpi=self._plot_dpi)
        plt.close(fig)

        payload = {
            "plot": plot_slug,
            "name": plot_name,
            "mds": {
                "stress": mds_info.get("stress"),
                "n_components": mds_info.get("n_components"),
            },
            "queries": points_payload,
            "shepard": shepard_payload,
        }
        self._save_json(json_dir / f"mds_{plot_slug}.json", payload)

    def _resolve_plot_axes_specs(
        self,
        plot_axes: list[str],
        axes_specs_by_name: Dict[str, AxisSpec],
        pca_info: Dict[str, Any],
    ) -> list[AxisSpec]:
        resolved: list[AxisSpec] = []
        for axis_name in plot_axes:
            explicit = axes_specs_by_name.get(axis_name)
            if explicit is not None:
                resolved.append(explicit)
                continue

            if not is_pca_component_axis_name(axis_name):
                raise ValueError(f"Unknown axis '{axis_name}'.")

            component_index = int(axis_name[3:]) - 1
            if component_index < 0:
                raise ValueError(f"Axis '{axis_name}' has invalid component index.")

            if pca_info.get("applied", False):
                n_components = int(pca_info.get("n_components", 0))
                if component_index >= n_components:
                    raise ValueError(
                        f"Axis '{axis_name}' out of range for PCA components: "
                        f"available 1..{n_components}."
                    )

            resolved.append(
                AxisSpec(
                    name=axis_name,
                    query_a="__pca_origin__",
                    query_b="__pca_component__",
                    method="pca_component",
                    params={"component_index": component_index},
                )
            )
        return resolved

    def _apply_pca_embeddings(
        self,
        embeddings: Dict[str, np.ndarray],
    ) -> tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        query_ids = list(embeddings.keys())
        matrix = np.stack([embeddings[qid] for qid in query_ids], axis=0)

        if matrix.ndim != 2:
            raise ValueError("Expected 2D embedding matrix for PCA.")

        n_samples, n_features = matrix.shape
        if n_samples < 2:
            return embeddings, {
                "enabled": True,
                "applied": False,
                "variance_threshold": 0.9,
                "reason": "insufficient_samples",
                "n_samples": n_samples,
                "n_features": n_features,
            }

        mean = np.mean(matrix, axis=0, keepdims=True)
        centered = matrix - mean
        _, singular_values, vt = np.linalg.svd(centered, full_matrices=False)

        explained_variance = (singular_values**2) / max(1, n_samples - 1)
        total_variance = float(np.sum(explained_variance))

        if total_variance <= 0:
            return embeddings, {
                "enabled": True,
                "applied": False,
                "variance_threshold": 0.9,
                "reason": "zero_variance",
                "n_samples": n_samples,
                "n_features": n_features,
            }

        explained_ratio = explained_variance / total_variance
        cumulative_ratio = np.cumsum(explained_ratio)
        n_components = int(np.searchsorted(cumulative_ratio, 0.9, side="left") + 1)
        n_components = max(1, min(n_components, vt.shape[0]))

        components = vt[:n_components, :]
        transformed = centered @ components.T

        transformed_map = {
            query_id: transformed[idx].astype(np.float64)
            for idx, query_id in enumerate(query_ids)
        }

        return transformed_map, {
            "enabled": True,
            "applied": True,
            "variance_threshold": 0.9,
            "n_samples": n_samples,
            "n_features": n_features,
            "n_components": n_components,
            "explained_variance_ratio": float(cumulative_ratio[n_components - 1]),
        }

    def _apply_mds_embeddings(
        self,
        embeddings: Dict[str, np.ndarray],
    ) -> tuple[
        Dict[str, np.ndarray] | None,
        Dict[str, Any],
        np.ndarray | None,
    ]:
        query_ids = list(embeddings.keys())
        info: Dict[str, Any] = {"enabled": True, "applied": False}
        n_samples = len(query_ids)
        if n_samples < 2:
            info.update({"reason": "insufficient_samples", "n_samples": n_samples})
            return None, info, None

        matrix = np.stack([embeddings[qid] for qid in query_ids], axis=0).astype(
            np.float64
        )

        try:
            from sklearn.metrics.pairwise import cosine_distances
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "scikit-learn is required to compute MDS embeddings."
            ) from exc

        dissimilarities = np.clip(cosine_distances(matrix), 0.0, 2.0)

        try:
            from sklearn.manifold import MDS
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "scikit-learn is required to compute MDS embeddings."
            ) from exc

        try:
            mds = MDS(
                n_components=2,
                metric_mds=False,
                metric="precomputed",
                n_init=1,
                init="classical_mds",
                random_state=0,
            )
            transformed = mds.fit_transform(dissimilarities)
        except Exception as exc:
            info.update({"reason": "mds_failed", "error": str(exc)})
            return None, info, dissimilarities

        coords = {
            query_id: transformed[idx].astype(np.float64)
            for idx, query_id in enumerate(query_ids)
        }
        info.update(
            {
                "applied": True,
                "n_samples": n_samples,
                "n_features": matrix.shape[1],
                "n_components": transformed.shape[1],
                "stress": float(getattr(mds, "stress_", 0.0)),
                "dissimilarity_metric": "cosine_distance",
            }
        )
        return coords, info, dissimilarities

    def _apply_group_lda_embeddings(
        self,
        embeddings: Dict[str, np.ndarray],
    ) -> tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        groups = self.config.averages + self.config.medians
        group_ids = [group.group_id for group in groups if group.group_id in embeddings]

        if len(group_ids) < 2:
            return embeddings, {
                "enabled": True,
                "applied": False,
                "reason": "insufficient_groups",
                "n_groups": len(group_ids),
            }

        query_ids = list(embeddings.keys())
        matrix = np.stack([embeddings[qid] for qid in query_ids], axis=0).astype(
            np.float64
        )
        n_samples, n_features = matrix.shape

        class_index = {gid: idx for idx, gid in enumerate(group_ids)}
        y = np.array(
            [class_index[qid] if qid in class_index else -1 for qid in query_ids],
            dtype=int,
        )

        global_mean = np.mean(matrix, axis=0)
        sw = np.zeros((n_features, n_features), dtype=np.float64)
        sb = np.zeros((n_features, n_features), dtype=np.float64)

        for gid, cls_idx in class_index.items():
            cls_mask = y == cls_idx
            cls_vectors = matrix[cls_mask]
            if cls_vectors.size == 0:
                continue
            cls_mean = np.mean(cls_vectors, axis=0)
            centered = cls_vectors - cls_mean
            sw += centered.T @ centered

            mean_diff = (cls_mean - global_mean).reshape(-1, 1)
            sb += cls_vectors.shape[0] * (mean_diff @ mean_diff.T)

        sw += 1e-6 * np.eye(n_features, dtype=np.float64)

        try:
            lda_matrix = np.linalg.solve(sw, sb)
        except np.linalg.LinAlgError:
            return embeddings, {
                "enabled": True,
                "applied": False,
                "reason": "singular_within_scatter",
                "n_groups": len(group_ids),
                "n_samples": n_samples,
                "n_features": n_features,
            }

        eigenvalues, eigenvectors = np.linalg.eig(lda_matrix)
        order = np.argsort(np.real(eigenvalues))[::-1]
        eigenvectors = np.real(eigenvectors[:, order])
        n_components = max(1, min(len(group_ids) - 1, n_features))
        components = eigenvectors[:, :n_components]

        transformed = (matrix - global_mean) @ components
        transformed_map = {
            query_id: transformed[idx].astype(np.float64)
            for idx, query_id in enumerate(query_ids)
        }
        return transformed_map, {
            "enabled": True,
            "applied": True,
            "n_groups": len(group_ids),
            "n_samples": n_samples,
            "n_features": n_features,
            "n_components": n_components,
        }

    def _save_axis_plot(
        self,
        plots_dir: Path,
        plot_slug: str,
        plot_file_suffix: str,
        axes,
        scores: Dict[str, Dict[str, float]],
        query_ids: list[str],
        query_names: Dict[str, str],
        query_colors: Dict[str, Any],
    ) -> None:
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "matplotlib is required to render axis plots."
            ) from exc

        axis_names = [axis.name for axis in axes]
        points = [
            (query_id, [scores[query_id][name] for name in axis_names])
            for query_id in query_ids
        ]

        fig = None
        if len(axes) == 1:
            x = np.arange(len(points))
            y = np.array([coords[0] for _, coords in points], dtype=np.float64)
            labels = [query_names.get(query_id, query_id) for query_id, _ in points]
            colors = [query_colors.get(query_id) for query_id, _ in points]

            fig_width = max(6.0, len(points) * 0.35)
            fig, ax = plt.subplots(figsize=(fig_width, 4))
            ax.plot(x, y, color="#777777", linewidth=1.2, zorder=1)
            ax.scatter(x, y, c=colors, s=38, zorder=2)
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=90)
            ax.set_xlabel("Query")
            ax.set_ylabel(axis_names[0])
            ax.grid(axis="y", linestyle="--", alpha=0.35)
            plot_path = plots_dir / f"projections_{plot_slug}_1d{plot_file_suffix}.png"
        else:
            fig, ax = plt.subplots(figsize=(6, 5))
            for query_id, coords in points:
                ax.scatter(coords[0], coords[1], s=40, c=query_colors.get(query_id))
                ax.text(
                    coords[0],
                    coords[1],
                    f" {query_names.get(query_id, query_id)}",
                    fontsize=7,
                )
            ax.set_xlabel(axis_names[0])
            ax.set_ylabel(axis_names[1])
            plot_path = plots_dir / f"projections_{plot_slug}_2d{plot_file_suffix}.png"

        if fig is not None:
            fig.tight_layout()
            fig.savefig(plot_path, dpi=self._plot_dpi)
            plt.close(fig)

    def _resolve_query_color(
        self,
        query_id: str,
        query_name: str,
        configured_color: str | None,
    ) -> str | None:
        if configured_color is not None:
            return configured_color

        pitch_class = self._infer_pitch_class(query_id=query_id, query_name=query_name)
        if pitch_class is None:
            return None
        return self._pitch_class_palette[pitch_class]

    def _infer_pitch_class(self, query_id: str, query_name: str) -> int | None:
        token_re = re.compile(r"[a-zA-Z#]+[0-9]*")
        for source in (query_id, query_name):
            for token in token_re.findall(source.lower()):
                note = re.sub(r"\d+$", "", token)
                if note in self._pitch_class_aliases:
                    return self._pitch_class_aliases[note]
        return None

    def _select_plot_query_ids(
        self,
        embeddings: Dict[str, np.ndarray],
        averaged_only: bool,
        group_filter: str,
    ) -> list[str]:
        if not averaged_only:
            return list(embeddings.keys())
        average_ids = [
            group.group_id
            for group in self.config.averages
            if group.group_id in embeddings
        ]
        median_ids = [
            group.group_id
            for group in self.config.medians
            if group.group_id in embeddings
        ]
        if group_filter == "average" and average_ids:
            return average_ids
        if group_filter == "median" and median_ids:
            return median_ids

        combined = average_ids + median_ids
        if combined:
            return combined
        return list(embeddings.keys())

    def _center_and_whiten_embeddings(
        self,
        embeddings: Dict[str, np.ndarray],
    ) -> Dict[str, np.ndarray]:
        query_ids = list(embeddings.keys())
        if not query_ids:
            return embeddings

        matrix = np.stack([embeddings[qid] for qid in query_ids], axis=0).astype(
            np.float64
        )

        n_samples = matrix.shape[0]
        if n_samples < 2:
            centered = matrix - np.mean(matrix, axis=0, keepdims=True)
            return {
                query_id: centered[idx].astype(np.float64)
                for idx, query_id in enumerate(query_ids)
            }

        try:
            from sklearn.preprocessing import StandardScaler

            centered = StandardScaler(with_mean=True, with_std=False).fit_transform(
                matrix
            )
        except Exception:
            centered = matrix - np.mean(matrix, axis=0, keepdims=True)

        eps = 1e-9
        n_samples_centered = centered.shape[0]
        _, singular_values, vt = np.linalg.svd(centered, full_matrices=False)
        scales = np.sqrt(max(1, n_samples_centered - 1)) / (singular_values + eps)
        projected = centered @ vt.T
        transformed = (projected * scales) @ vt

        return {
            query_id: transformed[idx].astype(np.float64)
            for idx, query_id in enumerate(query_ids)
        }
