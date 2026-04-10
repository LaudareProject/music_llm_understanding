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
DEFAULT_FEATURE_BACKEND = "aria"


def default_pipeline():
    return build_transparent_pipeline(
        [
            ("pca", PCA(whiten=True)),
            ("anchor", AnchorSelection(method="percentile_medians_regression", n=4, seed=42)),
            ("pls", AnchorPLS(scale=False)),
            ("projection", Projection1D()),
            ("regressor", KNeighborsRegressor(n_neighbors=5)),
        ],
        n_trials=5,
        random_state=42,
    )


def default_param_grid():
    return [
        {
            "pca__n_components": [0.9, None],
            "anchor__method": [
                "first_vs_last",
                "random_sample_first_last",
                "random_median",
                "percentile_medians",
                "random_sample_regression",
                "random_median_regression",
                "percentile_medians_regression",
            ],
            "anchor__n": [1, 2, 4, 8, 16, 32],
        }
    ]


EXPERIMENTS = {
    "composer_axis": {
        "tsv_path": REPO_ROOT / "experiments" / "composer_axis" / "songs.tsv",
        "target_col": "midpoint_year",
        "group_col": "composer",
        "feature_backend": DEFAULT_FEATURE_BACKEND,
        "pipeline": default_pipeline(),
        "param_grid": default_param_grid(),
    },
    "pitch_axis": {
        "tsv_path": REPO_ROOT / "experiments" / "pitch_axis" / "notes.tsv",
        "target_col": "midi_note",
        "group_col": "name",
        "feature_backend": DEFAULT_FEATURE_BACKEND,
        "pipeline": default_pipeline(),
        "param_grid": default_param_grid(),
    },
    "arousal_axis": {
        "tsv_path": REPO_ROOT / "experiments" / "arousal_axis" / "songs.tsv",
        "target_col": "arousal_mean",
        "group_col": "game",
        "feature_backend": DEFAULT_FEATURE_BACKEND,
        "pipeline": default_pipeline(),
        "param_grid": default_param_grid(),
    },
    "valence_axis": {
        "tsv_path": REPO_ROOT / "experiments" / "valence_axis" / "songs.tsv",
        "target_col": "valence_mean",
        "group_col": "game",
        "feature_backend": DEFAULT_FEATURE_BACKEND,
        "pipeline": default_pipeline(),
        "param_grid": default_param_grid(),
    },
}
