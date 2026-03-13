from __future__ import annotations

from datetime import datetime
import hashlib
import json
from pathlib import Path
from typing import Any, Dict

import numpy as np

from music_latents.config import load_experiment_config
from music_latents.inputs import prepare_midi_input
from music_latents.models import create_model_client


class RunnerCoreMixin:
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
        self.embedding_cache_dir = (Path.cwd() / "embedding_cache").resolve()
        self.embedding_cache_dir.mkdir(parents=True, exist_ok=True)




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
            if self.config.supervised is not None:
                self._save_supervised_outputs(output_dir, embeddings_by_model)
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
                cached = self._load_embedding_cache(
                    model_name=model_name,
                    input_string=query.value,
                )
                if cached is None:
                    result = model_client.encode_midi(input_info.midi_path)
                    raw_latent = np.asarray(result.latent)
                    raw_metadata = dict(result.metadata)
                    self._save_embedding_cache(
                        model_name=model_name,
                        input_string=query.value,
                        latent=raw_latent,
                        metadata=raw_metadata,
                    )
                else:
                    raw_latent, raw_metadata = cached

                latent, metadata = self._prepare_result_for_storage(
                    raw_latent,
                    raw_metadata,
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
        else:
            embeddings_by_model = self._apply_aggregate_groups(embeddings_by_model)

        if self.config.supervised is not None:
            self._save_supervised_outputs(output_dir, embeddings_by_model)
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




    def _embedding_cache_paths(
        self,
        model_name: str,
        input_string: str,
    ) -> tuple[Path, Path, Path]:
        cache_key = hashlib.sha256(input_string.encode("utf-8")).hexdigest()
        model_cache_dir = self.embedding_cache_dir / model_name
        model_cache_dir.mkdir(parents=True, exist_ok=True)
        entry_dir = model_cache_dir / cache_key
        latent_path = entry_dir / "latent.npy"
        metadata_path = entry_dir / "metadata.json"
        key_path = entry_dir / "input_key.txt"
        return latent_path, metadata_path, key_path




    def _load_embedding_cache(
        self,
        model_name: str,
        input_string: str,
    ) -> tuple[np.ndarray, Dict[str, Any]] | None:
        latent_path, metadata_path, key_path = self._embedding_cache_paths(
            model_name=model_name,
            input_string=input_string,
        )
        if (
            not latent_path.exists()
            or not metadata_path.exists()
            or not key_path.exists()
        ):
            return None

        try:
            cached_key = key_path.read_text(encoding="utf-8")
            if cached_key != input_string:
                return None
            latent = np.load(latent_path)
            metadata_raw = json.loads(metadata_path.read_text(encoding="utf-8"))
            if not isinstance(metadata_raw, dict):
                return None
        except Exception:
            return None

        return np.asarray(latent), metadata_raw




    def _save_embedding_cache(
        self,
        model_name: str,
        input_string: str,
        latent: np.ndarray,
        metadata: Dict[str, Any],
    ) -> None:
        latent_path, metadata_path, key_path = self._embedding_cache_paths(
            model_name=model_name,
            input_string=input_string,
        )
        entry_dir = latent_path.parent
        entry_dir.mkdir(parents=True, exist_ok=True)
        np.save(latent_path, np.asarray(latent))
        metadata_path.write_text(
            json.dumps(metadata, indent=2, ensure_ascii=True), encoding="utf-8"
        )
        key_path.write_text(input_string, encoding="utf-8")




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
        fig.savefig(plot_path, dpi=self._plot_dpi)
        plt.close(fig)



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

