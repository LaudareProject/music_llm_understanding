from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
from typing import Dict, Any

import numpy as np

from music_latents.config import load_experiment_config
from music_latents.axes import build_axes, compute_projection_scores
from music_latents.inputs import prepare_midi_input
from music_latents.models import create_model_client


class ExperimentRunner:
    def __init__(
        self,
        config_path: Path,
        reuse_results_dir: Path | None = None,
    ) -> None:
        self.config_path = config_path
        self.config = load_experiment_config(config_path)
        self.reuse_results_dir = reuse_results_dir

    def run(self) -> Path:
        if self.reuse_results_dir is not None:
            output_dir = self._resolve_reuse_dir(self.reuse_results_dir)
            embeddings_by_model = self._load_existing_embeddings(output_dir)
            self._save_config_copy(output_dir)
            if self.config.axes:
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
                config={k: v for k, v in model_cfg.items() if k != "type"},
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
                model_dir = query_dir / model_name
                model_dir.mkdir(parents=True, exist_ok=True)
                self._save_model_result(model_dir, result)

                embeddings_by_model[model_name][query.query_id] = self._flatten_latent(
                    result.latent
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

        if self.config.axes:
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
        output_root = self.config.settings.output_root
        if not output_root.is_absolute():
            output_root = (self.config.base_dir / output_root).resolve()
        output_dir = output_root / self.config.settings.name / timestamp
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir

    def _save_config_copy(self, output_dir: Path) -> None:
        config_copy = output_dir / "config.yaml"
        config_copy.write_text(
            self.config_path.read_text(encoding="utf-8"), encoding="utf-8"
        )

    def _save_model_result(self, model_dir: Path, result) -> None:
        latent_path = model_dir / "latent.npy"
        np.save(latent_path, result.latent)

        self._save_latent_plot(model_dir, result.latent)

        metadata_path = model_dir / "metadata.json"
        self._save_json(metadata_path, result.metadata)

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

        for model_name, embeddings in embeddings_by_model.items():
            axes_dir = output_dir / "axes" / model_name
            axes_dir.mkdir(parents=True, exist_ok=True)

            for plot_name, plot_spec in self.config.plots.items():
                plot_axes_specs = [axes_specs_by_name[name] for name in plot_spec.axes]
                axes = build_axes(plot_axes_specs, embeddings, plot_spec.options)
                scores = compute_projection_scores(axes, embeddings, plot_spec.options)

                plot_slug = "".join(
                    ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in plot_name
                )
                plot_dir = axes_dir / plot_slug
                plot_dir.mkdir(parents=True, exist_ok=True)

                axes_payload = {
                    "plot": plot_name,
                    "options": {
                        "translate": plot_spec.options.translate,
                        "orthonormalize": plot_spec.options.orthonormalize,
                        "normalize_vectors": plot_spec.options.normalize_vectors,
                        "origin_strategy": plot_spec.options.origin_strategy,
                    },
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
                self._save_json(plot_dir / "axes.json", axes_payload)

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
                        for query_id in embeddings.keys()
                    ],
                }
                self._save_json(plot_dir / "scores.json", scores_payload)

                if len(plot_spec.axes) <= 3:
                    self._save_axis_plot(
                        plot_dir,
                        axes,
                        scores,
                        query_names,
                        query_colors,
                    )

    def _save_axis_plot(
        self,
        plot_dir: Path,
        axes,
        scores: Dict[str, Dict[str, float]],
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
            for query_id in scores.keys()
        ]

        fig = None
        if len(axes) == 1:
            fig, ax = plt.subplots(figsize=(6, 3))
            for query_id, coords in points:
                ax.scatter(coords[0], 0.0, s=40, c=query_colors.get(query_id))
                ax.text(
                    coords[0],
                    0.0,
                    f" {query_names.get(query_id, query_id)}",
                    va="center",
                )
            ax.set_xlabel(axis_names[0])
            ax.set_yticks([])
            plot_path = plot_dir / "projections_1d.png"
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
            plot_path = plot_dir / "projections_2d.png"
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
            plot_path = plot_dir / "projections_3d.png"

        if fig is not None:
            fig.tight_layout()
            fig.savefig(plot_path, dpi=150)
            plt.close(fig)
