from __future__ import annotations

from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LinearRegression, ElasticNetCV
from sklearn.feature_selection import VarianceThreshold
from sklearn.neighbors import KNeighborsRegressor


from experiments.common.models import (
    AnchorPLS,
    AnchorSelection,
    Projection1D,
    build_transparent_pipeline,
)

import warnings
warnings.filterwarnings("ignore", category=UserWarning)


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FEATURE_BACKEND = "aria"
CV_FOLDS = 5
GRID_JOBS = 20


def default_pipeline():
    return build_transparent_pipeline(
        [
            ("pca", VarianceThreshold()),
            # ("pca", PCA(n_components=0.9)),
            (
                "anchor",
                AnchorSelection(
                    method="first_vs_last",
                    n=8,
                    seed=42,
                    max_anchor_size=0.15,
                ),
            ),
            ("pls", AnchorPLS(scale=True)),
            ("projection", Projection1D(regression=True)),
            # ("regressor", KNeighborsRegressor(n_neighbors=8))
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
            "random_sample_first_last",
            "random_median",
            "percentile_medians",
        ],
        # "anchor__n": range(1, 32),
        "projection__regression": [False, True],
        "anchor__n": [1, 2, 4, 8, 16, 32],
        # "anchor__n": [2, 16],
    }
    # return common

    return [
        {
            **common,
            "pca": [PCA()],
            "pca__n_components": [0.8, 0.9, 0.95, None],
            "pca__whiten": [True, False],
            "pls": [AnchorPLS()],
            "pls__scale": [True, False],
        },
        {
            **common,
            "pca": [PCA()],
            "pca__n_components": [0.8, 0.9, 0.95, None],
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
