from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from music_latents.runner import ExperimentRunner


def test_experiment_runner_creates_outputs(tmp_path: Path) -> None:
    midi_path = tmp_path / "input.mid"
    midi_path.write_bytes(b"dummy-midi")

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
experiment:
  name: "unit_test"
  output_root: "outputs"

models:
  dummy_model:
    type: "dummy"
    seed: "test"

queries:
  - id: "q1"
    midi_path: "input.mid"
""".lstrip(),
        encoding="utf-8",
    )

    runner = ExperimentRunner(config_path)
    output_dir = runner.run()

    assert output_dir.exists()
    assert (output_dir / "config.yaml").exists()
    assert (output_dir / "manifest.json").exists()

    latent_path = output_dir / "queries" / "q1" / "dummy_model" / "latent.npy"
    metadata_path = output_dir / "queries" / "q1" / "dummy_model" / "metadata.json"
    assert latent_path.exists()
    assert metadata_path.exists()

    latent = np.load(latent_path)
    assert latent.shape == (1, 16)

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["model_type"] == "dummy"
    assert metadata["model_name"] == "dummy_model"
