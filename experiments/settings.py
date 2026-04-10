from __future__ import annotations

from pathlib import Path

from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import VarianceThreshold
# from sklearn.neighbors import KNeighborsRegressor

from experiments.common.models import (
    AnchorPLS,
    AnchorSelection,
    Projection1D,
    build_transparent_pipeline,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FEATURE_BACKEND = "aria"
CV_FOLDS = 5
GRID_JOBS = 20


def default_pipeline():
    return build_transparent_pipeline(
        [
            ("pca", PCA(n_components=0.9, whiten=True)),
            ("anchor", AnchorSelection(method="first_vs_last", n=4, seed=42)),
            ("pls", AnchorPLS(scale=True)),
            # ("lda", LinearDiscriminantAnalysis()),
            ("projection", Projection1D()),
            # ("regressor", KNeighborsRegressor(n_neighbors=4)),
            ("regressor", LinearRegression()),
        ],
        # how many trials to do for stochastic anchor selections (spearman is averaged across trials)
        n_trials=1,
        random_state=42,
    )


def default_param_grid():
    common = {
        "anchor__method": [
            "first_vs_last",
            # "random_sample_first_last",
            # "random_median",
            # "percentile_medians",
            # "random_sample_regression",
            "random_median_regression",
            # "percentile_medians_regression",
        ],
        # "anchor__n": [1, 2, 4, 8, 16, 32],
        "anchor__n": [2, 16],
    }

    # return common
    return [
        {
            **common,
            "pca": [PCA()],
            "pca__n_components": [0.9, None],
            "pca__whiten": [True, False],
            "pls": [AnchorPLS()],
            "pls__scale": [True, False],
        },
        {
            **common,
            "pca": [PCA()],
            "pca__n_components": [0.9, None],
            "pca__whiten": [True, False],
            "pls": [LinearDiscriminantAnalysis()],
        },
        {
            **common,
            "pca": [VarianceThreshold()],
            "pls": [AnchorPLS()],
            "pls__scale": [True, False],
        },
        {
            **common,
            "pca": [VarianceThreshold()],
            "pls": [LinearDiscriminantAnalysis()],
        },
    ]


def _composer_axis_spec(dataset_name=None):
    spec = {
        "tsv_path": REPO_ROOT / "experiments" / "composer_axis" / "songs.tsv",
        "target_col": "midpoint_year",
        "group_col": "composer",
        "feature_backend": DEFAULT_FEATURE_BACKEND,
        "pipeline": default_pipeline(),
        "param_grid": default_param_grid(),
    }
    if dataset_name is not None:
        spec["row_filter_col"] = "dataset"
        spec["row_filter_value"] = dataset_name
    return spec


EXPERIMENTS = {
    "composer_axis": _composer_axis_spec(),
    "composer_axis_asap": _composer_axis_spec("asap"),
    "composer_axis_ewld": _composer_axis_spec("ewld"),
    "composer_axis_music21": _composer_axis_spec("music21"),
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
