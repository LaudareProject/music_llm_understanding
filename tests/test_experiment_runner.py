from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from music_latents.config import ConfigError
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

    axes_path = output_dir / "dummy_model" / "json" / "pca" / "axes_plot1.json"
    scores_path = output_dir / "dummy_model" / "json" / "pca" / "projections_plot1.json"
    plot_path = (
        output_dir / "dummy_model" / "plots" / "pca" / "projections_plot1_1d.png"
    )

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


def test_experiment_runner_creates_mds_outputs(tmp_path: Path) -> None:
    pytest.importorskip("sklearn")
    midi_path = tmp_path / "input.mid"
    midi_path.write_bytes(b"dummy-midi")
    midi_path_two = tmp_path / "input_two.mid"
    midi_path_two.write_bytes(b"dummy-midi-2")

    config_path = tmp_path / "config_mds.yaml"
    config_path.write_text(
        """
experiment:
  name: "unit_test_mds"
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
      mds: true
""".lstrip(),
        encoding="utf-8",
    )

    runner = ExperimentRunner(config_path)
    output_dir = runner.run()

    plots_dir = output_dir / "dummy_model" / "plots" / "mds"
    json_dir = output_dir / "dummy_model" / "json" / "mds"
    assert (plots_dir / "mds_plot1_2d.png").exists()
    assert (plots_dir / "mds_plot1_shepard.png").exists()
    assert (json_dir / "mds_plot1.json").exists()

    payload = json.loads((json_dir / "mds_plot1.json").read_text(encoding="utf-8"))
    assert payload["plot"] == "plot1"
    assert payload["mds"]["n_components"] == 2
    assert payload["shepard"]


def test_experiment_runner_links_intersections_for_processing_folders(
    tmp_path: Path,
) -> None:
    midi_path = tmp_path / "input.mid"
    midi_path.write_bytes(b"dummy-midi")
    midi_path_two = tmp_path / "input_two.mid"
    midi_path_two.write_bytes(b"dummy-midi-2")

    config_path = tmp_path / "config_links.yaml"
    config_path.write_text(
        """
experiment:
  name: "unit_test_links"
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
      orthonormalize: true
      pca: true
""".lstrip(),
        encoding="utf-8",
    )

    output_dir = ExperimentRunner(config_path).run()

    canonical_plot = (
        output_dir
        / "dummy_model"
        / "plots"
        / "orthonormal+pca"
        / "projections_plot1_1d.png"
    )
    canonical_json = (
        output_dir
        / "dummy_model"
        / "json"
        / "orthonormal+pca"
        / "projections_plot1.json"
    )
    assert canonical_plot.exists()
    assert canonical_json.exists()

    for alias_folder in ("orthonormal", "pca"):
        alias_plot = (
            output_dir
            / "dummy_model"
            / "plots"
            / alias_folder
            / "projections_plot1_1d.png"
        )
        alias_json = (
            output_dir
            / "dummy_model"
            / "json"
            / alias_folder
            / "projections_plot1.json"
        )
        assert alias_plot.is_symlink()
        assert alias_json.is_symlink()
        assert alias_plot.resolve() == canonical_plot.resolve()
        assert alias_json.resolve() == canonical_json.resolve()


def test_experiment_runner_cosine_projection_is_exclusive(tmp_path: Path) -> None:
    midi_path = tmp_path / "input.mid"
    midi_path.write_bytes(b"dummy-midi")
    midi_path_two = tmp_path / "input_two.mid"
    midi_path_two.write_bytes(b"dummy-midi-2")

    config_path = tmp_path / "config_cosine_only.yaml"
    config_path.write_text(
        """
experiment:
  name: "unit_test_cosine_only"
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
      cosine_references: ["q1", "q2"]
""".lstrip(),
        encoding="utf-8",
    )

    output_dir = ExperimentRunner(config_path).run()
    plots_dir = output_dir / "dummy_model" / "plots" / "standard"
    json_dir = output_dir / "dummy_model" / "json" / "standard"

    assert (plots_dir / "cosine_plot1.png").exists()
    assert (json_dir / "cosine_plot1.json").exists()
    assert not (json_dir / "axes_plot1.json").exists()
    assert not (json_dir / "projections_plot1.json").exists()


def test_experiment_runner_rejects_pca_and_mds(tmp_path: Path) -> None:
    midi_path = tmp_path / "input.mid"
    midi_path.write_bytes(b"dummy-midi")
    midi_path_two = tmp_path / "input_two.mid"
    midi_path_two.write_bytes(b"dummy-midi-2")

    config_path = tmp_path / "config_conflict.yaml"
    config_path.write_text(
        """
experiment:
  name: "unit_test_conflict"
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
      pca: true
      mds: true
""".lstrip(),
        encoding="utf-8",
    )

    with pytest.raises(ConfigError):
        ExperimentRunner(config_path)


def test_experiment_runner_rejects_mds_and_cosine_references(tmp_path: Path) -> None:
    midi_path = tmp_path / "input.mid"
    midi_path.write_bytes(b"dummy-midi")
    midi_path_two = tmp_path / "input_two.mid"
    midi_path_two.write_bytes(b"dummy-midi-2")

    config_path = tmp_path / "config_mds_cosine_conflict.yaml"
    config_path.write_text(
        """
experiment:
  name: "unit_test_mds_cosine_conflict"
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
      mds: true
      cosine_references: ["q1", "q2"]
""".lstrip(),
        encoding="utf-8",
    )

    with pytest.raises(ConfigError):
        ExperimentRunner(config_path)


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
    assert (
        output_dir / "dummy_model" / "plots" / "standard" / "projections_plot1_1d.png"
    ).exists()


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
    scores_path = (
        output_dir / "dummy_model" / "json" / "standard" / "projections_avg_plot.json"
    )
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


def test_experiment_runner_supports_median_groups(tmp_path: Path) -> None:
    midi_a = tmp_path / "a.mid"
    midi_a.write_bytes(b"a")
    midi_b = tmp_path / "b.mid"
    midi_b.write_bytes(b"b")
    midi_c = tmp_path / "c.mid"
    midi_c.write_bytes(b"c")

    config_path = tmp_path / "config_median.yaml"
    config_path.write_text(
        """
experiment:
  name: "unit_test_median"
  output_root: "outputs"

models:
  dummy_model:
    type: "dummy"
    seed: "test"

queries:
  - id: "q1"
    name: "Q1"
    midi_path: "a.mid"
  - id: "q2"
    name: "Q2"
    midi_path: "b.mid"
  - id: "q3"
    name: "Q3"
    midi_path: "c.mid"

average:
  avg_all:
    name: "Average"
    queries: ["q1", "q2", "q3"]

median:
  med_all:
    name: "Median"
    queries: ["q1", "q2", "q3"]

axes:
  - name: "med_vs_avg"
    query_a: "med_all"
    query_b: "avg_all"
    method: "explicit"

plots:
  average_plot:
    axes:
      - "med_vs_avg"
    options:
      averaged_only: true
      group_filter: "average"
  median_plot:
    axes:
      - "med_vs_avg"
    options:
      averaged_only: true
      group_filter: "median"
""".lstrip(),
        encoding="utf-8",
    )

    runner = ExperimentRunner(config_path)
    output_dir = runner.run()

    avg_scores = json.loads(
        (
            output_dir
            / "dummy_model"
            / "json"
            / "standard"
            / "projections_average_plot.json"
        ).read_text(encoding="utf-8")
    )
    med_scores = json.loads(
        (
            output_dir
            / "dummy_model"
            / "json"
            / "standard"
            / "projections_median_plot.json"
        ).read_text(encoding="utf-8")
    )

    assert [entry["id"] for entry in avg_scores["queries"]] == ["avg_all"]
    assert [entry["id"] for entry in med_scores["queries"]] == ["med_all"]

    synthetic = {
        "dummy_model": {
            "q1": np.array([1.0, 10.0]),
            "q2": np.array([5.0, 2.0]),
            "q3": np.array([9.0, 4.0]),
        }
    }
    merged = runner._apply_aggregate_groups(synthetic)
    assert np.allclose(merged["dummy_model"]["med_all"], np.array([5.0, 4.0]))


def test_experiment_runner_supports_pca_component_axes(tmp_path: Path) -> None:
    midi_a = tmp_path / "a.mid"
    midi_a.write_bytes(b"a")
    midi_b = tmp_path / "b.mid"
    midi_b.write_bytes(b"b")
    midi_c = tmp_path / "c.mid"
    midi_c.write_bytes(b"c")

    config_path = tmp_path / "config_pca_axes.yaml"
    config_path.write_text(
        """
experiment:
  name: "unit_test_pca_axes"
  output_root: "outputs"

models:
  dummy_model:
    type: "dummy"
    seed: "test"

queries:
  - id: "q1"
    name: "Q1"
    midi_path: "a.mid"
  - id: "q2"
    name: "Q2"
    midi_path: "b.mid"
  - id: "q3"
    name: "Q3"
    midi_path: "c.mid"

plots:
  pca_plot:
    axes: ["pca01", "pca02"]
    options:
      pca: true
      translate: true
      normalize_vectors: true
""".lstrip(),
        encoding="utf-8",
    )

    output_dir = ExperimentRunner(config_path).run()
    axes_path = output_dir / "dummy_model" / "json" / "pca" / "axes_pca_plot.json"
    scores_path = (
        output_dir / "dummy_model" / "json" / "pca" / "projections_pca_plot.json"
    )

    assert axes_path.exists()
    assert scores_path.exists()

    axes_payload = json.loads(axes_path.read_text(encoding="utf-8"))
    assert [item["name"] for item in axes_payload["axes"]] == ["pca01", "pca02"]
    assert [item["method"] for item in axes_payload["axes"]] == [
        "pca_component",
        "pca_component",
    ]

    scores_payload = json.loads(scores_path.read_text(encoding="utf-8"))
    assert scores_payload["axes"] == ["pca01", "pca02"]
    assert len(scores_payload["queries"]) == 3


def test_experiment_runner_supports_pca_component_without_pca(tmp_path: Path) -> None:
    midi_a = tmp_path / "a.mid"
    midi_a.write_bytes(b"a")
    midi_b = tmp_path / "b.mid"
    midi_b.write_bytes(b"b")

    config_path = tmp_path / "config_pca_no_pca.yaml"
    config_path.write_text(
        """
experiment:
  name: "unit_test_pca_no_pca"
  output_root: "outputs"

models:
  dummy_model:
    type: "dummy"
    seed: "test"

queries:
  - id: "q1"
    name: "Q1"
    midi_path: "a.mid"
  - id: "q2"
    name: "Q2"
    midi_path: "b.mid"

plots:
  bad_plot:
    axes: ["pca01"]
    options:
      pca: false
""".lstrip(),
        encoding="utf-8",
    )

    output_dir = ExperimentRunner(config_path).run()
    scores_path = (
        output_dir / "dummy_model" / "json" / "standard" / "projections_bad_plot.json"
    )
    assert scores_path.exists()


def test_experiment_runner_supports_group_lda_plot_option(tmp_path: Path) -> None:
    midi_a = tmp_path / "a.mid"
    midi_a.write_bytes(b"a")
    midi_b = tmp_path / "b.mid"
    midi_b.write_bytes(b"b")
    midi_c = tmp_path / "c.mid"
    midi_c.write_bytes(b"c")
    midi_d = tmp_path / "d.mid"
    midi_d.write_bytes(b"d")

    config_path = tmp_path / "config_lda_groups.yaml"
    config_path.write_text(
        """
experiment:
  name: "unit_test_lda_groups"
  output_root: "outputs"

models:
  dummy_model:
    type: "dummy"
    seed: "test"

queries:
  - id: "q1"
    name: "Q1"
    midi_path: "a.mid"
  - id: "q2"
    name: "Q2"
    midi_path: "b.mid"
  - id: "q3"
    name: "Q3"
    midi_path: "c.mid"
  - id: "q4"
    name: "Q4"
    midi_path: "d.mid"

average:
  g1:
    name: "G1"
    queries: ["q1", "q2"]
  g2:
    name: "G2"
    queries: ["q3", "q4"]

axes:
  - name: "g1_vs_g2"
    query_a: "g1"
    query_b: "g2"
    method: "explicit"

plots:
  lda_plot:
    axes: ["g1_vs_g2"]
    options:
      averaged_only: true
      group_filter: "average"
      lda_groups: true
""".lstrip(),
        encoding="utf-8",
    )

    output_dir = ExperimentRunner(config_path).run()
    axes_path = output_dir / "dummy_model" / "json" / "lda" / "axes_lda_plot.json"
    assert axes_path.exists()

    payload = json.loads(axes_path.read_text(encoding="utf-8"))
    assert payload["options"]["lda_groups"] is True
    assert payload["lda"]["enabled"] is True


def test_center_and_whiten_embeddings_produces_centered_output() -> None:
    runner = ExperimentRunner.__new__(ExperimentRunner)
    embeddings = {
        "q1": np.array([1.0, 0.0, 2.0]),
        "q2": np.array([0.0, 1.0, 3.0]),
        "q3": np.array([2.0, 2.0, 4.0]),
        "q4": np.array([3.0, 1.0, 5.0]),
    }

    transformed = runner._center_and_whiten_embeddings(embeddings)
    matrix = np.stack([transformed[key] for key in embeddings], axis=0)

    assert np.allclose(np.mean(matrix, axis=0), 0.0, atol=1e-7)


def test_supervised_regression_outputs(tmp_path: Path) -> None:
    midi_a = tmp_path / "a.mid"
    midi_a.write_bytes(b"a")
    midi_b = tmp_path / "b.mid"
    midi_b.write_bytes(b"b")

    config_path = tmp_path / "config_supervised_regression.yaml"
    config_path.write_text(
        """
experiment:
  name: "unit_test_supervised_regression"
  output_root: "outputs"

models:
  dummy_model:
    type: "dummy"
    seed: "test"

queries:
  - id: "q1"
    name: "Q1"
    midi_path: "a.mid"
    target: 0.1
  - id: "q2"
    name: "Q2"
    midi_path: "b.mid"
    target: 0.4
  - id: "q3"
    name: "Q3"
    midi_path: "a.mid"
    target: 0.8
  - id: "q4"
    name: "Q4"
    midi_path: "b.mid"
    target: 1.1

supervised:
  task: "regression"
  features: ["embedding", "norm", "polar"]
  top_k: 10
""".lstrip(),
        encoding="utf-8",
    )

    output_dir = ExperimentRunner(config_path).run()
    report_path = output_dir / "supervised" / "dummy_model" / "supervised_report.json"
    plot_path = output_dir / "supervised" / "dummy_model" / "feature_importance.png"

    assert report_path.exists()
    assert plot_path.exists()

    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["task"] == "regression"
    assert payload["cv_strategy"] == "leave_one_out"
    assert payload["cv_folds"] == 4
    assert payload["n_samples"] == 4
    assert payload["top_feature_importances"]


def test_supervised_cv_fold_policy() -> None:
    runner = ExperimentRunner.__new__(ExperimentRunner)
    _, label_small, splits_small = runner._determine_cv_strategy(10)
    _, label_large, splits_large = runner._determine_cv_strategy(51)
    assert label_small == "leave_one_out"
    assert splits_small == 10
    assert label_large == "kfold_5"
    assert splits_large == 5


def test_descriptive_features_extractor() -> None:
    runner = ExperimentRunner.__new__(ExperimentRunner)
    vec = np.array([1.0, -1.0, 2.0, -2.0, 0.0], dtype=np.float64)
    values, names = runner._descriptive_features(vec)
    by_name = {name: values[idx] for idx, name in enumerate(names)}

    assert "mean" in by_name
    assert "median" in by_name
    assert "q90" in by_name
    assert "gini_concentration" in by_name
    assert np.isclose(by_name["mean"], 0.0)
    assert np.isclose(by_name["median"], 0.0)
    assert np.isclose(by_name["positive_fraction"], 0.4)
    assert np.isclose(by_name["negative_fraction"], 0.4)
