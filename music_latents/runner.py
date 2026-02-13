from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
from typing import Dict, Any

import numpy as np

from music_latents.config import load_experiment_config
from music_latents.inputs import prepare_midi_input
from music_latents.models import create_model_client


class ExperimentRunner:
    def __init__(self, config_path: Path) -> None:
        self.config_path = config_path
        self.config = load_experiment_config(config_path)

    def run(self) -> Path:
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
        return output_dir

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
