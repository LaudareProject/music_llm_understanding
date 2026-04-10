from __future__ import annotations

from pathlib import Path

from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor

from experiments.common.models import (
    AnchorPLS,
    AnchorSelection,
    Projection1D,
    build_transparent_pipeline,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def default_pipeline():
    return build_transparent_pipeline(
        [
            ("pca", PCA()),
            ("anchor", AnchorSelection(method="first_vs_last", n=10, seed=42)),
            ("pls", AnchorPLS(scale=False)),
            ("projection", Projection1D(n=10)),
            ("regressor", KNeighborsRegressor(n_neighbors=5)),
        ],
        n_trials=1,
        random_state=42,
    )


def default_param_grid():
    return [
        {
            "pca__n_components": [8, 16, 32, 64],
            "anchor__method": [
                "first_vs_last",
                "random_sample_first_last",
                "random_median",
                "equally_spaced_regression",
                "random_sample_regression",
                "random_median_regression",
            ],
            "anchor__n": [4, 8, 12, 16],
            "projection__n": [4, 8, 12, 16],
            "n_trials": [1, 8, 16],
            "regressor": [KNeighborsRegressor()],
            "regressor__n_neighbors": [3, 5, 7, 9],
        },
        {
            "pca__n_components": [8, 16, 32, 64],
            "anchor__method": [
                "first_vs_last",
                "random_sample_first_last",
                "random_median",
                "equally_spaced_regression",
                "random_sample_regression",
                "random_median_regression",
            ],
            "anchor__n": [4, 8, 12, 16],
            "projection__n": [4, 8, 12, 16],
            "n_trials": [1, 8, 16],
            "regressor": [LinearRegression()],
        },
    ]


EXPERIMENTS = {
    "composer_axis": {
        "tsv_path": REPO_ROOT / "experiments" / "composer_axis" / "songs.tsv",
        "target_col": "midpoint_year",
        "group_col": "composer",
        "pipeline": default_pipeline(),
        "param_grid": default_param_grid(),
        "n_sweep": [1, 2, 4, 8, 12, 16, 24, 32],
    },
    "pitch_axis": {
        "tsv_path": REPO_ROOT / "experiments" / "pitch_axis" / "notes.tsv",
        "target_col": "midi_note",
        "group_col": "name",
        "pipeline": default_pipeline(),
        "param_grid": default_param_grid(),
        "n_sweep": [1, 2, 4, 8, 12, 16, 24, 32],
    },
    "arousal_axis": {
        "tsv_path": REPO_ROOT / "experiments" / "arousal_axis" / "songs.tsv",
        "target_col": "arousal_mean",
        "group_col": "game",
        "pipeline": default_pipeline(),
        "param_grid": default_param_grid(),
        "n_sweep": [1, 2, 4, 8, 12, 16, 24, 32],
    },
    "valence_axis": {
        "tsv_path": REPO_ROOT / "experiments" / "valence_axis" / "songs.tsv",
        "target_col": "valence_mean",
        "group_col": "game",
        "pipeline": default_pipeline(),
        "param_grid": default_param_grid(),
        "n_sweep": [1, 2, 4, 8, 12, 16, 24, 32],
    },
}
