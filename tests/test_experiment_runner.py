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
    midi_path_two = tmp_path / "input_two.mid"
    midi_path_two.write_bytes(b"dummy-midi-2")

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
  - color: "red"
    queries:
      - id: "q1"
        name: "Query One"
        midi_path: "input.mid"
  - color: "blue"
    queries:
      - id: "q2"
        name: "Query Two"
        midi_path: "input_two.mid"

axes:
  - name: "q1_vs_q2"
    query_a: "q1"
    query_b: "q2"
    method: "explicit"

plots:
  plot1:
    axes:
      - "q1_vs_q2"
    options:
      translate: true
      orthonormalize: false
      normalize_vectors: true
      origin_strategy: "mean"
      pca: true
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

    axes_dir = output_dir / "axes" / "dummy_model"
    axes_path = axes_dir / "axes_plot1.json"
    scores_path = axes_dir / "scores_plot1.json"
    plot_path = axes_dir / "projections_plot1_1d.png"

    assert axes_path.exists()
    assert scores_path.exists()
    assert plot_path.exists()

    axes_payload = json.loads(axes_path.read_text(encoding="utf-8"))
    assert axes_payload["plot"] == "plot1"
    assert axes_payload["options"]["translate"] is True
    assert axes_payload["options"]["normalize_vectors"] is True
    assert axes_payload["options"]["pca"] is True
    assert axes_payload["pca"]["applied"] is True
    assert len(axes_payload["axes"]) == 1

    scores_payload = json.loads(scores_path.read_text(encoding="utf-8"))
    assert scores_payload["axes"] == ["q1_vs_q2"]
    assert len(scores_payload["queries"]) == 2
    colors = {q["id"]: q["color"] for q in scores_payload["queries"]}
    assert colors == {"q1": "red", "q2": "blue"}


def test_experiment_runner_reuses_existing_embeddings(tmp_path: Path) -> None:
    midi_path = tmp_path / "input.mid"
    midi_path.write_bytes(b"dummy-midi")
    midi_path_two = tmp_path / "input_two.mid"
    midi_path_two.write_bytes(b"dummy-midi-2")

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
experiment:
  name: "unit_test_reuse"
  output_root: "outputs"

models:
  dummy_model:
    type: "dummy"
    seed: "test"

queries:
  - id: "q1"
    name: "Query One"
    midi_path: "input.mid"
  - id: "q2"
    name: "Query Two"
    midi_path: "input_two.mid"

axes:
  - name: "q1_vs_q2"
    query_a: "q1"
    query_b: "q2"
    method: "explicit"

plots:
  plot1:
    axes:
      - "q1_vs_q2"
    options:
      translate: true
      orthonormalize: false
      normalize_vectors: true
      origin_strategy: "mean"
""".lstrip(),
        encoding="utf-8",
    )

    output_dir = ExperimentRunner(config_path).run()

    midi_path.unlink()
    midi_path_two.unlink()

    reused_output_dir = ExperimentRunner(
        config_path,
        reuse_results_dir=output_dir,
    ).run()

    assert reused_output_dir == output_dir
    assert (output_dir / "axes" / "dummy_model" / "projections_plot1_1d.png").exists()


def test_experiment_runner_supports_average_groups(tmp_path: Path) -> None:
    midi_a = tmp_path / "a.mid"
    midi_a.write_bytes(b"a")
    midi_b = tmp_path / "b.mid"
    midi_b.write_bytes(b"b")
    midi_c = tmp_path / "c.mid"
    midi_c.write_bytes(b"c")

    config_path = tmp_path / "config_avg.yaml"
    config_path.write_text(
        """
experiment:
  name: "unit_test_average"
  output_root: "outputs"

models:
  dummy_model:
    type: "dummy"
    seed: "test"

queries:
  - id: "q1"
    name: "Q1"
    color: "red"
    midi_path: "a.mid"
  - id: "q2"
    name: "Q2"
    color: "red"
    midi_path: "b.mid"
  - id: "q3"
    name: "Q3"
    color: "blue"
    midi_path: "c.mid"

average:
  q12_avg:
    name: "Q12 avg"
    queries:
      - "q1"
      - "q2"

axes:
  - name: "avg_vs_q3"
    query_a: "q12_avg"
    query_b: "q3"
    method: "explicit"

plots:
  avg_plot:
    axes:
      - "avg_vs_q3"
    options:
      translate: true
      orthonormalize: false
      normalize_vectors: true
      origin_strategy: "mean"
""".lstrip(),
        encoding="utf-8",
    )

    output_dir = ExperimentRunner(config_path).run()
    scores_path = output_dir / "axes" / "dummy_model" / "scores_avg_plot.json"
    assert scores_path.exists()

    scores_payload = json.loads(scores_path.read_text(encoding="utf-8"))
    by_id = {entry["id"]: entry for entry in scores_payload["queries"]}
    assert "q12_avg" in by_id
    assert by_id["q12_avg"]["name"] == "Q12 avg"
    assert by_id["q12_avg"]["color"] == "red"


def test_experiment_runner_model_trim_option(tmp_path: Path) -> None:
    midi_path = tmp_path / "trim.mid"
    midi_path.write_bytes(b"trim-midi")

    config_path = tmp_path / "config_trim.yaml"
    config_path.write_text(
        """
experiment:
  name: "unit_test_trim"
  output_root: "outputs"

models:
  dummy_model:
    type: "dummy"
    seed: "test"
    trim: 8

queries:
  - id: "q1"
    name: "Q1"
    midi_path: "trim.mid"

""".lstrip(),
        encoding="utf-8",
    )

    output_dir = ExperimentRunner(config_path).run()
    latent_path = output_dir / "queries" / "q1" / "dummy_model" / "latent.npy"
    metadata_path = output_dir / "queries" / "q1" / "dummy_model" / "metadata.json"

    latent = np.load(latent_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    assert latent.shape == (1, 8)
    assert metadata["latent_shape"] == [1, 8]
    assert metadata["trim"] == 8
