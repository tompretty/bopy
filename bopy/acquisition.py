from abc import ABC, abstractmethod

import numpy as np
from scipy.stats import norm

from .exceptions import NotFittedError
from .surrogate import Surrogate

__all__ = ["LCB", "EI", "POI"]


class AcquisitionFunction(ABC):
    """An acquisition function.

    Acquisition functions navigate the exploration-exploitation
    trade off during optimization.
    """

    def __init__(self):
        self.is_fitted = False

    def __call__(self, surrogate: Surrogate, x: np.ndarray) -> np.ndarray:
        """Evaluate the acquisition function.

        Parameters
        ----------
        surrogate: Surrogate
            The surrogate model.
        x: np.ndarray of shape (n_samples, n_dimensions)
            The input locations at which to evaluate
            the acquisition function.

        Returns
        -------
        a_x: np.ndarray of shape (n_samples,)
            The value of the acquisition function
            at `x`.
        """
        if not self.is_fitted:
            raise NotFittedError("must call fit before evaluating.")

        return self._f(*surrogate.predict(x))

    @abstractmethod
    def _f(self, mean: np.ndarray, sigma: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        """Fit the acquistion function to training data.

        Parameters
        ----------
        x: np.ndarray of shape (n_samples, n_dimensions)
            The training input.
        y: np.ndarray of shape (n_samples,)
            The training output.
        """
        self._fit(x, y)
        self.is_fitted = True

    def _fit(self, x: np.ndarray, y: np.ndarray) -> None:
        pass


class LCB(AcquisitionFunction):
    """Lower confidence bound acquisition function.

    LCB implements the simple rule:
        `lcb(x) = -mean - kappa * std`

    Parameters
    ----------
    kappa: float
        The number of stds we subtract from the mean.
    """

    def __init__(self, kappa: float = 2.0):
        super().__init__()
        self.kappa = kappa

    def _f(self, mean: np.ndarray, sigma: np.ndarray) -> np.ndarray:
        return -mean - self.kappa * np.diag(sigma)


class EI(AcquisitionFunction):
    """Expected improvement acquisition function.

    EI implements the rule:
        `ei(x) = E[|eta - f(x)|+]`
    """

    def __init__(self):
        super().__init__()
        self._eta = np.inf

    def _f(self, mean: np.ndarray, sigma: np.ndarray):
        var = np.diag(sigma)
        std = np.sqrt(var)

        return -var * norm.pdf(self._eta, loc=mean, scale=std) + (
            mean - self._eta
        ) * norm.cdf(self._eta, loc=mean, scale=std)

    def _fit(self, x: np.ndarray, y: np.ndarray) -> None:
        self._eta = np.min(y)


class POI(AcquisitionFunction):
    """Probability of improvement acquisition function.

    POI implements the rule:
        `poi(x) = p(f(x) > eta)`
    """

    def __init__(self):
        super().__init__()
        self._eta = np.inf

    def _f(self, mean: np.ndarray, sigma: np.ndarray):
        var = np.diag(sigma)
        std = np.sqrt(var)

        return -norm.cdf(self._eta, mean, std)

    def _fit(self, x: np.ndarray, y: np.ndarray) -> None:
        self._eta = np.min(y)
