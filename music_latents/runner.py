from __future__ import annotations

from datetime import datetime
import hashlib
import json
from pathlib import Path
from typing import Dict, Any

import numpy as np

from music_latents.config import (
    AxisSpec,
    is_pca_component_axis_name,
    load_experiment_config,
)
from music_latents.axes import build_axes, compute_projection_scores
from music_latents.inputs import prepare_midi_input
from music_latents.models import create_model_client


class ExperimentRunner:
    _bootstrap_samples: int = 1000

    def __init__(
        self,
        config_path: Path,
        reuse_results_dir: Path | None = None,
        output_root_override: Path | None = None,
        experiment_name_override: str | None = None,
    ) -> None:
        self.config_path = config_path
        self.config = load_experiment_config(config_path)
        self.reuse_results_dir = reuse_results_dir
        self.output_root_override = output_root_override
        self.experiment_name_override = experiment_name_override

    def run(self) -> Path:
        model_trims = self._resolve_model_trims()

        if self.reuse_results_dir is not None:
            output_dir = self._resolve_reuse_dir(self.reuse_results_dir)
            embeddings_by_model = self._load_existing_embeddings(
                output_dir, model_trims
            )
            embeddings_by_model = self._apply_aggregate_groups(embeddings_by_model)
            self._save_config_copy(output_dir)
            if self.config.plots:
                self._save_axis_outputs(output_dir, embeddings_by_model)
            return output_dir

        output_dir = self._create_output_dir()
        (output_dir / "queries").mkdir(parents=True, exist_ok=True)

        self._save_config_copy(output_dir)

        manifest = {
            "experiment": {
                "name": self.config.settings.name,
                "timestamp": output_dir.name,
            },
            "models": list(self._iter_model_names()),
            "queries": [],
        }

        model_clients = {}
        embeddings_by_model = {
            model_name: {} for model_name in self._iter_model_names()
        }
        for model_name in self._iter_model_names():
            model_cfg = self.config.models[model_name]
            model_type = model_cfg.get("type")
            if not model_type:
                raise ValueError(
                    f"Model '{model_name}' is missing required field 'type'."
                )

            model_clients[model_name] = create_model_client(
                model_type=model_type,
                name=model_name,
                config={
                    k: v for k, v in model_cfg.items() if k not in {"type", "trim"}
                },
            )

        for query in self.config.queries:
            query_dir = output_dir / "queries" / query.query_id
            query_dir.mkdir(parents=True, exist_ok=True)

            input_info = prepare_midi_input(
                query=query,
                base_dir=self.config.base_dir,
                query_dir=query_dir,
            )

            query_entry = {
                "id": query.query_id,
                "name": query.name,
                "color": query.color,
                "input_kind": input_info.input_kind,
                "source": input_info.source,
                "stored_path": str(input_info.stored_path)
                if input_info.stored_path
                else None,
                "midi_path": str(input_info.midi_path),
                "models": [],
            }

            for model_name in self._iter_model_names():
                model_client = model_clients[model_name]
                result = model_client.encode_midi(input_info.midi_path)
                latent, metadata = self._prepare_result_for_storage(
                    result.latent,
                    result.metadata,
                    trim=model_trims.get(model_name),
                )
                model_dir = query_dir / model_name
                model_dir.mkdir(parents=True, exist_ok=True)
                self._save_model_result(model_dir, latent, metadata)

                embeddings_by_model[model_name][query.query_id] = self._flatten_latent(
                    latent
                )

                query_entry["models"].append(
                    {
                        "name": model_name,
                        "type": model_client.model_type,
                        "latent_path": str(model_dir / "latent.npy"),
                        "metadata_path": str(model_dir / "metadata.json"),
                    }
                )

            manifest["queries"].append(query_entry)

        self._save_json(output_dir / "manifest.json", manifest)

        if self.config.plots:
            embeddings_by_model = self._apply_aggregate_groups(embeddings_by_model)
            self._save_axis_outputs(output_dir, embeddings_by_model)
        return output_dir

    def _resolve_reuse_dir(self, reuse_results_dir: Path) -> Path:
        if reuse_results_dir.is_absolute():
            output_dir = reuse_results_dir.resolve()
            if output_dir.exists() and output_dir.is_dir():
                return output_dir
            raise FileNotFoundError(f"Reuse results directory not found: {output_dir}")

        cwd_candidate = (Path.cwd() / reuse_results_dir).resolve()
        if cwd_candidate.exists() and cwd_candidate.is_dir():
            return cwd_candidate

        config_candidate = (self.config.base_dir / reuse_results_dir).resolve()
        if config_candidate.exists() and config_candidate.is_dir():
            return config_candidate

        raise FileNotFoundError(
            "Reuse results directory not found. Checked: "
            f"{cwd_candidate} and {config_candidate}"
        )

    def _load_existing_embeddings(
        self,
        output_dir: Path,
        model_trims: Dict[str, int | None],
    ) -> Dict[str, Dict[str, np.ndarray]]:
        embeddings_by_model: Dict[str, Dict[str, np.ndarray]] = {
            model_name: {} for model_name in self._iter_model_names()
        }

        for query in self.config.queries:
            for model_name in self._iter_model_names():
                latent_path = (
                    output_dir / "queries" / query.query_id / model_name / "latent.npy"
                )
                if not latent_path.exists():
                    raise FileNotFoundError(
                        f"Expected latent file not found for reuse mode: {latent_path}"
                    )

                latent = np.load(latent_path)
                trim = model_trims.get(model_name)
                if trim is not None:
                    latent = self._trim_latent(latent, trim, model_name)
                embeddings_by_model[model_name][query.query_id] = self._flatten_latent(
                    latent
                )

        return embeddings_by_model

    def _iter_model_names(self):
        if self.config.settings.models:
            return self.config.settings.models
        return list(self.config.models.keys())

    def _create_output_dir(self) -> Path:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_root = self.output_root_override or self.config.settings.output_root
        if not output_root.is_absolute():
            output_root = (self.config.base_dir / output_root).resolve()

        experiment_name = self.experiment_name_override or self.config.settings.name
        output_dir = output_root / experiment_name / timestamp
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir

    def _save_config_copy(self, output_dir: Path) -> None:
        config_copy = output_dir / "config.yaml"
        config_copy.write_text(
            self.config_path.read_text(encoding="utf-8"), encoding="utf-8"
        )

    def _save_model_result(
        self,
        model_dir: Path,
        latent: np.ndarray,
        metadata: Dict[str, Any],
    ) -> None:
        latent_path = model_dir / "latent.npy"
        np.save(latent_path, latent)

        self._save_latent_plot(model_dir, latent)

        metadata_path = model_dir / "metadata.json"
        self._save_json(metadata_path, metadata)

    def _resolve_model_trims(self) -> Dict[str, int | None]:
        trims: Dict[str, int | None] = {}
        for model_name in self._iter_model_names():
            model_cfg = self.config.models[model_name]
            trim = model_cfg.get("trim")
            if trim is None:
                trims[model_name] = None
                continue
            if not isinstance(trim, int) or trim <= 0:
                raise ValueError(
                    f"Model '{model_name}' field 'trim' must be a positive integer."
                )
            trims[model_name] = trim
        return trims

    def _prepare_result_for_storage(
        self,
        latent: np.ndarray,
        metadata: Dict[str, Any],
        trim: int | None,
    ) -> tuple[np.ndarray, Dict[str, Any]]:
        prepared_latent = np.asarray(latent)
        if trim is not None:
            model_name = str(metadata.get("model_name", "model"))
            prepared_latent = self._trim_latent(prepared_latent, trim, model_name)

        prepared_metadata = dict(metadata)
        prepared_metadata["latent_shape"] = list(prepared_latent.shape)
        if trim is not None:
            prepared_metadata["trim"] = trim
        return prepared_latent, prepared_metadata

    def _trim_latent(
        self, latent: np.ndarray, trim: int, model_name: str
    ) -> np.ndarray:
        if latent.ndim == 0:
            raise ValueError(f"Cannot trim scalar latent for model '{model_name}'.")
        if latent.shape[-1] < trim:
            raise ValueError(
                f"Model '{model_name}' trim={trim} exceeds latent size {latent.shape[-1]}."
            )
        return latent[..., :trim]

    def _save_json(self, path: Path, payload: Dict[str, Any]) -> None:
        path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8"
        )

    def _flatten_latent(self, latent: np.ndarray) -> np.ndarray:
        array = np.asarray(latent)
        array = np.squeeze(array)
        if array.ndim == 0:
            array = array.reshape(1)
        if array.ndim > 1:
            array = array.reshape(-1)
        return array.astype(np.float64)

    def _save_latent_plot(self, model_dir: Path, latent: np.ndarray) -> None:
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "matplotlib is required to render latent plots."
            ) from exc

        array = np.asarray(latent)
        array = np.squeeze(array)
        if array.ndim == 0:
            array = array.reshape(1, 1)

        plot_path = model_dir / "latent.png"

        if array.ndim == 1:
            fig, ax = plt.subplots(figsize=(8, 3))
            ax.bar(range(array.shape[0]), array)
            ax.set_xlabel("Index")
            ax.set_ylabel("Value")
        else:
            if array.ndim > 2:
                array = array.reshape(array.shape[0], -1)
            fig, ax = plt.subplots(figsize=(6, 4))
            im = ax.imshow(array, aspect="auto", cmap="viridis")
            fig.colorbar(im, ax=ax)
            ax.set_xlabel("Feature")
            ax.set_ylabel("Row")

        fig.tight_layout()
        fig.savefig(plot_path, dpi=150)
        plt.close(fig)

    def _save_axis_outputs(
        self,
        output_dir: Path,
        embeddings_by_model: Dict[str, Dict[str, np.ndarray]],
    ) -> None:
        axes_specs_by_name = {axis.name: axis for axis in self.config.axes}

        query_names = {query.query_id: query.name for query in self.config.queries}
        query_colors = {query.query_id: query.color for query in self.config.queries}
        for average_group in self.config.averages:
            query_names[average_group.group_id] = average_group.name
            query_colors[average_group.group_id] = average_group.color
        for median_group in self.config.medians:
            query_names[median_group.group_id] = median_group.name
            query_colors[median_group.group_id] = median_group.color

        for model_name, embeddings in embeddings_by_model.items():
            axes_dir = output_dir / "axes" / model_name
            axes_dir.mkdir(parents=True, exist_ok=True)

            for plot_name, plot_spec in self.config.plots.items():
                plot_embeddings = embeddings
                pre_lda_embeddings = embeddings
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

                query_ids = self._select_plot_query_ids(
                    embeddings=embeddings,
                    averaged_only=plot_spec.options.averaged_only,
                    group_filter=plot_spec.options.group_filter,
                )

                plot_axes_specs = self._resolve_plot_axes_specs(
                    plot_axes=plot_spec.axes,
                    axes_specs_by_name=axes_specs_by_name,
                    pca_info=pca_info,
                )
                axes = build_axes(plot_axes_specs, plot_embeddings, plot_spec.options)
                scores = compute_projection_scores(
                    axes, plot_embeddings, plot_spec.options
                )

                plot_slug = "".join(
                    ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in plot_name
                )

                axes_payload = {
                    "plot": plot_name,
                    "options": {
                        "translate": plot_spec.options.translate,
                        "orthonormalize": plot_spec.options.orthonormalize,
                        "normalize_vectors": plot_spec.options.normalize_vectors,
                        "origin_strategy": plot_spec.options.origin_strategy,
                        "pca": plot_spec.options.pca,
                        "lda_groups": plot_spec.options.lda_groups,
                        "averaged_only": plot_spec.options.averaged_only,
                        "group_filter": plot_spec.options.group_filter,
                        "cosine_references": plot_spec.options.cosine_references,
                    },
                    "lda": lda_info,
                    "pca": pca_info,
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
                self._save_json(axes_dir / f"axes_{plot_slug}.json", axes_payload)

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
                        for query_id in self._select_plot_query_ids(
                            embeddings=embeddings,
                            averaged_only=plot_spec.options.averaged_only,
                            group_filter=plot_spec.options.group_filter,
                        )
                    ],
                }
                self._save_json(axes_dir / f"scores_{plot_slug}.json", scores_payload)

                if plot_spec.options.cosine_references is not None:
                    if len(plot_spec.options.cosine_references) != 2:
                        raise ValueError(
                            f"Plot '{plot_name}' must define exactly two cosine references."
                        )

                    if lda_info.get("applied", False):
                        no_lda_embeddings = pre_lda_embeddings
                        if plot_spec.options.pca:
                            no_lda_embeddings, _ = self._apply_pca_embeddings(
                                no_lda_embeddings
                            )
                        self._save_cosine_polar_plot(
                            axes_dir=axes_dir,
                            plot_slug=plot_slug,
                            query_ids=query_ids,
                            embeddings=no_lda_embeddings,
                            query_names=query_names,
                            query_colors=query_colors,
                            cosine_references=plot_spec.options.cosine_references,
                            variant="no_lda",
                        )

                    self._save_cosine_polar_plot(
                        axes_dir=axes_dir,
                        plot_slug=plot_slug,
                        query_ids=query_ids,
                        embeddings=plot_embeddings,
                        query_names=query_names,
                        query_colors=query_colors,
                        cosine_references=plot_spec.options.cosine_references,
                        variant="lda" if lda_info.get("applied", False) else "standard",
                    )

                if len(plot_spec.axes) <= 3:
                    self._save_axis_plot(
                        axes_dir,
                        plot_slug,
                        axes,
                        scores,
                        query_ids,
                        query_names,
                        query_colors,
                    )

    def _save_cosine_polar_plot(
        self,
        axes_dir: Path,
        plot_slug: str,
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

        points_payload: list[Dict[str, Any]] = []
        fig, ax = plt.subplots(figsize=(6, 5))
        for query_id in query_ids:
            vector = embeddings.get(query_id)
            if vector is None:
                continue

            vector_norm = np.linalg.norm(vector)
            if vector_norm == 0:
                cos_distance_a = 1.0
                cos_distance_b = 1.0
            else:
                cos_distance_a = 1.0 - float(
                    np.dot(vector, ref_a) / (vector_norm * ref_a_norm)
                )
                cos_distance_b = 1.0 - float(
                    np.dot(vector, ref_b) / (vector_norm * ref_b_norm)
                )

            theta = float(np.arctan2(cos_distance_b, cos_distance_a))
            radius = float(np.sqrt(cos_distance_a**2 + cos_distance_b**2))
            x = float(radius * np.cos(theta))
            y = float(radius * np.sin(theta))

            ax.scatter(x, y, s=40, c=query_colors.get(query_id))
            ax.text(x, y, f" {query_names.get(query_id, query_id)}")

            points_payload.append(
                {
                    "id": query_id,
                    "name": query_names.get(query_id, query_id),
                    "color": query_colors.get(query_id),
                    "cosine_distance_reference_a": cos_distance_a,
                    "cosine_distance_reference_b": cos_distance_b,
                    "theta": theta,
                    "radius": radius,
                    "x": x,
                    "y": y,
                }
            )

        ax.set_xlabel(f"cos-dist to {reference_a}")
        ax.set_ylabel(f"cos-dist to {reference_b}")
        ax.set_title(
            f"Cosine polar map ({variant}) | ref: {reference_a}, {reference_b}"
        )
        ax.grid(True, linestyle="--", alpha=0.35)

        fig.tight_layout()
        fig.savefig(axes_dir / f"cosine_polar_{plot_slug}_{variant}.png", dpi=150)
        plt.close(fig)

        payload = {
            "plot": plot_slug,
            "variant": variant,
            "references": {
                "a": reference_a,
                "b": reference_b,
            },
            "points": points_payload,
        }
        self._save_json(axes_dir / f"cosine_polar_{plot_slug}_{variant}.json", payload)

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

    def _apply_aggregate_groups(
        self,
        embeddings_by_model: Dict[str, Dict[str, np.ndarray]],
    ) -> Dict[str, Dict[str, np.ndarray]]:
        aggregate_groups = self.config.averages + self.config.medians
        if not aggregate_groups:
            return embeddings_by_model

        result: Dict[str, Dict[str, np.ndarray]] = {}
        for model_name, embeddings in embeddings_by_model.items():
            merged = dict(embeddings)
            for average_group in aggregate_groups:
                vectors: list[np.ndarray] = []
                for query_id in average_group.query_ids:
                    vector = embeddings.get(query_id)
                    if vector is None:
                        raise ValueError(
                            f"Cannot average missing query '{query_id}' for group "
                            f"'{average_group.group_id}' in model '{model_name}'."
                        )
                    vectors.append(vector)
                if average_group.method == "median":
                    merged[average_group.group_id] = self._median(vectors=vectors)
                else:
                    merged[average_group.group_id] = self._bootstrap_average(
                        vectors=vectors,
                        group_id=average_group.group_id,
                        model_name=model_name,
                    )
            result[model_name] = merged

        return result

    def _bootstrap_average(
        self,
        vectors: list[np.ndarray],
        group_id: str,
        model_name: str,
    ) -> np.ndarray:
        if len(vectors) == 1:
            return vectors[0]

        stacked = np.stack(vectors, axis=0)
        n_vectors = stacked.shape[0]

        seed_material = f"{model_name}:{group_id}".encode("utf-8")
        seed = int.from_bytes(hashlib.sha256(seed_material).digest()[:8], "little")
        rng = np.random.default_rng(seed)

        estimate = np.zeros(stacked.shape[1], dtype=np.float64)
        for _ in range(self._bootstrap_samples):
            sampled_indices = rng.integers(0, n_vectors, size=n_vectors)
            estimate += np.mean(stacked[sampled_indices], axis=0)

        return estimate / float(self._bootstrap_samples)

    def _median(self, vectors: list[np.ndarray]) -> np.ndarray:
        if len(vectors) == 1:
            return vectors[0]
        stacked = np.stack(vectors, axis=0)
        return np.median(stacked, axis=0).astype(np.float64)

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
        axes_dir: Path,
        plot_slug: str,
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
            plot_path = axes_dir / f"projections_{plot_slug}_1d.png"
        elif len(axes) == 2:
            fig, ax = plt.subplots(figsize=(6, 5))
            for query_id, coords in points:
                ax.scatter(coords[0], coords[1], s=40, c=query_colors.get(query_id))
                ax.text(
                    coords[0],
                    coords[1],
                    f" {query_names.get(query_id, query_id)}",
                )
            ax.set_xlabel(axis_names[0])
            ax.set_ylabel(axis_names[1])
            plot_path = axes_dir / f"projections_{plot_slug}_2d.png"
        else:
            from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

            fig = plt.figure(figsize=(7, 6))
            ax = fig.add_subplot(111, projection="3d")
            for query_id, coords in points:
                ax.scatter(
                    coords[0],
                    coords[1],
                    coords[2],
                    s=40,
                    c=query_colors.get(query_id),
                )
                ax.text(
                    coords[0],
                    coords[1],
                    coords[2],
                    f" {query_names.get(query_id, query_id)}",
                )
            ax.set_xlabel(axis_names[0])
            ax.set_ylabel(axis_names[1])
            ax.set_zlabel(axis_names[2])
            plot_path = axes_dir / f"projections_{plot_slug}_3d.png"

        if fig is not None:
            fig.tight_layout()
            fig.savefig(plot_path, dpi=150)
            plt.close(fig)

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
