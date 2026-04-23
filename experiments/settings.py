from __future__ import annotations

from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LinearRegression, ElasticNetCV
from sklearn.feature_selection import VarianceThreshold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import FunctionTransformer, StandardScaler, MinMaxScaler


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
SCORE_TOLERANCE = 0.02
OUTLIER_IQR_FACTOR = 1.5


def default_pipeline():
    return build_transparent_pipeline(
        [
            ("std", StandardScaler()),
            ("pca", PCA(n_components=0.8, whiten=True)),
            (
                "anchor",
                AnchorSelection(
                    method="first_vs_last",
                    n=12,
                    seed=42,
                    max_anchor_size=0.15,
                ),
            ),
            ("pls", AnchorPLS(scale=False)),
            ("projection", Projection1D(regression=False)),
            ("regressor", LinearRegression()),
        ],
        # how many trials to do for stochastic anchor selections (spearman is averaged across trials)
        n_trials=1,
        random_state=42,
    )


def default_param_grid():
    common = {
        "std": [StandardScaler(), MinMaxScaler(), FunctionTransformer(validate=False)],
        "anchor__method": [
            "first_vs_last",
            "random_sample_first_last",
            "random_median",
            "percentile_medians",
        ],
        # "anchor__n": range(1, 32),
        "projection__regression": [False, True],
        "anchor__n": [2, 4, 8, 12, 16, 20, 24, 28, 32],
        # "anchor__n": [2, 16],
    }
    # return common

    ret = []
    for pca in ["pca", "identity", "var"]:
        for pls in ["pls", "identity", "lda"]:
            for projection in ["project1d", "identity"]:
                payload = {**common}
                if pca == "pca":
                    payload["pca"] = [PCA()]
                    payload["pca__n_components"] = [0.8, 0.9, 0.95, None]
                    payload["pca__whiten"] = [True, False]
                elif pca == "variance":
                    payload["pca"] = [VarianceThreshold()]
                elif pca == "identity":
                    payload["pca"] = [FunctionTransformer(validate=False)]

                if pls == "pls":
                    payload["pls"] = [AnchorPLS()]
                    payload["pls__scale"] = [True, False]
                elif pls == "lda":
                    payload["pls"] = [LinearDiscriminantAnalysis()]
                elif pls == "identity":
                    payload["pls"] = [FunctionTransformer(validate=False)]

                # if projection == "project1d":
                #     payload["projection"] = [Projection1D(regression=False)]
                #     # payload["projection__regression"] = [False, True]
                # elif projection == "identity":
                #     payload["projection"] = [FunctionTransformer(validate=False)]

                ret.append(payload)

    return ret


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
    "sentiment_prediction": {
        "tsv_path": REPO_ROOT / "experiments" / "valence_axis" / "songs.tsv",
        "target_col": "valence_mean",
        "group_col": "game",
        "feature_backend": DEFAULT_FEATURE_BACKEND,
        "pipeline": default_pipeline(),
        "cv_folds": 10,
    },
}
