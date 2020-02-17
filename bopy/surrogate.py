import warnings
from abc import ABC, abstractmethod
from typing import Callable, Tuple

import GPy
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.preprocessing import StandardScaler

from .mixin import FittableMixin

__all__ = ["ScipyGPSurrogate"]


class Surrogate(FittableMixin, ABC):
    """Surrogate model base class.

    A surrogate is a probabilistic model that stands in for the true
    objective function during optimization.

    This class shouldn't be used directly, use a derived class instead."""

    def __init__(self):
        super().__init__()
        self.has_been_fitted = False
        self.n_dimensions = -1
        self.x = np.array([])
        self.y = np.array([])

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        """Fit the surrogate model to training data.

        Parameters
        ----------
        x: np.ndarray of shape (n_samples, n_dimensions)
            The training input.
        y: np.ndarray of shape (n_samples,)
            The training target.
        """
        self._validate_ok_for_fitting(x, y)
        self._update_data(x, y)
        self._fit(x, y)
        self._confirm_fit()

    def _update_data(self, x: np.ndarray, y: np.ndarray) -> None:
        self.x = x
        self.y = y

    @abstractmethod
    def _fit(self, x: np.ndarray, y: np.ndarray) -> None:
        raise NotImplementedError

    def predict(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Make a prediction on test data.

        Parameters
        ----------
        x: np.ndarray of shape (n_samples, n_dimensions)
            The test input.

        Returns
        -------
        y_pred: np.ndarray of shape (n_samples,)
            The predicted mean.
        sigma: np.ndarray of shape (n_samples, n_samples)
            The predicted covariance.
        """
        self._validate_ok_for_predicting(x)
        return self._predict(x)

    @abstractmethod
    def _predict(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError


class ScipyGPSurrogate(Surrogate):
    """Scikit-learn GP Surrogate.

    This is a wrapper around the scikit-learn
    GaussianProcessRegressor model.

    Parameters
    ----------
    gp: GaussianProcessRegressor
        The scikit-learn GP regressor.
    """

    def __init__(self, gp: GaussianProcessRegressor):
        super().__init__()
        self.gp = gp

    def _fit(self, x: np.ndarray, y: np.ndarray) -> None:
        self.gp.fit(x, y)

    def _predict(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        return self.gp.predict(x, return_cov=True)


class GPyGPSurrogate(Surrogate):
    """GPy GP Surrogate.

    This is a wrapper around a GPy GPRegression model.

    Due to the fact that GPy models are instantiated
    with training data but no such data is available when
    constructing a BayesOpt object, a level of indirection is
    required. Instead of passing the surrogate an instantiated
    GPy model, one must pass a function that will instantiate and
    then return a model given data. The signature should look like:

    def gp_initialzier(x: np.ndarray, y: np.ndarray):
        gp = GPy.models.GPRegression(...)
        ...set constraints, or priors, or whatever...
        return gp

    Parameters
    ----------
    gp_initizlizer: Callable[[np.ndarray, np.ndarray] -> GPy.models.GPRegression]
        A function that accepts training data and
        returns a GPy GP model.
    n_restarts: Integer (default = 1)
        The number of restarts during optimization.
    """

    def __init__(
        self,
        gp_initializer: Callable[[np.ndarray, np.ndarray], GPy.models.GPRegression],
        n_restarts: int = 1,
    ):
        super().__init__()
        self.gp_initializer = gp_initializer
        self.n_restarts = n_restarts
        self.gp = None

    def _fit(self, x: np.ndarray, y: np.ndarray) -> None:
        self._update_gp(x, y.reshape(-1, 1))
        self._optimize_gp()

    def _predict(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        mu, sigma = self.gp.predict_noiseless(x, full_cov=True)
        return mu.flatten(), sigma

    def _update_gp(self, x: np.ndarray, y: np.ndarray):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)

            if self.gp is None:
                self.gp = self.gp_initializer(x, y)
            else:
                self.gp.set_XY(x, y)

    def _optimize_gp(self):
        self.gp.optimize_restarts(self.n_restarts)
