import numpy as np
from scipy import linalg
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis,
    _class_cov,
    _class_means,
)


class OrdinalLinearDiscriminantAnalysis(LinearDiscriminantAnalysis):
    def __init__(
        self,
        solver="svd",
        shrinkage=None,
        priors=None,
        n_components=None,
        store_covariance=False,
        tol=1e-4,
        covariance_estimator=None,
        class_scores=None,
    ):
        super().__init__(
            solver=solver,
            shrinkage=shrinkage,
            priors=priors,
            n_components=n_components,
            store_covariance=store_covariance,
            tol=tol,
            covariance_estimator=covariance_estimator,
        )
        self.class_scores = (
            None if class_scores is None else np.asarray(class_scores, dtype=np.float64)
        )

    def _get_class_scores(self):
        if self.class_scores is None:
            try:
                return self.classes_.astype(np.float64)
            except ValueError as exc:
                raise ValueError("OLDA requires numeric class labels") from exc
        if self.class_scores.shape[0] != self.classes_.shape[0]:
            raise ValueError("class_scores must match number of classes")
        return self.class_scores

    def _solve_eigen(self, X, y, shrinkage, covariance_estimator):
        self.means_ = _class_means(X, y)
        self.covariance_ = _class_cov(
            X, y, self.priors_, shrinkage, covariance_estimator
        )

        class_scores = self._get_class_scores()

        mean_diff = self.means_[:, None, :] - self.means_[None, :, :]
        score_diff = (class_scores[:, None] - class_scores[None, :]) ** 2
        pair_weights = self.priors_[:, None] * self.priors_[None, :] * score_diff
        sb_ord = np.einsum("kl,klp,klq->pq", pair_weights, mean_diff, mean_diff)

        evals, evecs = linalg.eigh(sb_ord, self.covariance_)
        self.explained_variance_ratio_ = np.sort(evals / np.sum(evals))[::-1][
            : self._max_components
        ]
        evecs = evecs[:, np.argsort(evals)[::-1]]
        self.scalings_ = evecs
        self.coef_ = np.dot(self.means_, evecs).dot(evecs.T)
        self.intercept_ = -0.5 * np.diag(np.dot(self.means_, self.coef_.T)) + np.log(
            self.priors_
        )
