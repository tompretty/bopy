from abc import ABC, abstractmethod

import numpy as np
from scipy.stats import norm

from .mixin import FittableMixin
from .surrogate import Surrogate

__all__ = ["LCB", "EI", "POI"]


class AcquisitionFunction(FittableMixin, ABC):
    """An acquisition function.

    Acquisition functions navigate the exploration-exploitation
    trade off during optimization. The convention here is that
    acquistion functions are to be minimized. We should thus
    think of them as representing expected loss, rather than
    expected utility.

    This class shouldn't be used directly, use a derived class instead.
    """

    def __init__(self, surrogate: Surrogate):
        super().__init__()
        self.surrogate = surrogate
        self.has_been_fitted = False
        self.n_dimensions = -1

    def __call__(self, x: np.ndarray) -> np.ndarray:
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
        self._validate_ok_for_predicting(x)
        return self._f(*self.surrogate.predict(x))

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
        self._validate_ok_for_fitting(x, y)
        self._fit(x, y)
        self._confirm_fit()

    def _fit(self, x: np.ndarray, y: np.ndarray) -> None:
        pass


class LCB(AcquisitionFunction):
    """Lower confidence bound acquisition function.

    LCB implements the simple rule:
        `lcb(x) = mean - kappa * std`

    Parameters
    ----------
    kappa: float
        The number of stds we subtract from the mean.
    """

    def __init__(self, surrogate: Surrogate, kappa: float = 2.0):
        super().__init__(surrogate)
        self.kappa = kappa

    def _f(self, mean: np.ndarray, sigma: np.ndarray) -> np.ndarray:
        return mean - self.kappa * np.sqrt(np.diag(sigma))


class EI(AcquisitionFunction):
    """Expected improvement acquisition function.

    EI implements the rule:
        `ei(x) = -E[|eta - f(x)|+]`
    """

    def __init__(self, surrogate: Surrogate):
        super().__init__(surrogate)
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
        `poi(x) = p(f(x) >= eta)`
    """

    def __init__(self, surrogate: Surrogate):
        super().__init__(surrogate)
        self._eta = np.inf

    def _f(self, mean: np.ndarray, sigma: np.ndarray):
        var = np.diag(sigma)
        std = np.sqrt(var)

        return 1 - norm.cdf(self._eta, mean, std)

    def _fit(self, x: np.ndarray, y: np.ndarray) -> None:
        self._eta = np.min(y)


class SequentialBatchAcquisitionFunction(AcquisitionFunction):
    """Sequential batch acquisition function.

    This is the base class for acquisition functions
    to be used by a SequentialBatchOptimizer. They combine 
    a base acquistion function with a strategy to update
    it as new batch points are selected sequentially e.g. 
    the krigging believer strategy will 'fantasize' a new
    datapoint as the posterior mean of the surrogate.

    The class shouldn't be used directly. Use a derived class instead.

    Parameters
    ----------
    surrogate : Surrogate
        The surrogate model.
    base_acquisition : AcquisitionFunction
        The base acquisition function that is modified 
        as each new batch point arrives.
    """

    def __init__(self, surrogate: Surrogate, base_acquisition: AcquisitionFunction):
        super().__init__(surrogate)
        self.base_acquisition = base_acquisition

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self._f(self.base_acquisition(x))

    @abstractmethod
    def _f(self, a_x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        """Fit the acquisition function to training data."""
        self.base_acquisition.fit(x, y)

    def start_batch(self) -> None:
        """Prepare to start creating a new batch."""
        pass

    def add_to_batch(self, optimization_result) -> None:
        """Update given a new datapoint."""
        pass

    def finish_batch(self) -> None:
        """Clean up after finishing a batch."""
        pass


class KriggingBeliever(SequentialBatchAcquisitionFunction):
    """Krigging believer.

    This sequential batch acquisition function proceeds by 
    'fantasizing' new datapoints using the posterior mean of
    the surrogate model. It then updates the surrogate and
    base acquisition function with this new datapoint allowing 
    for the next datapoint to be slected by optimizing the updated
    base acquisiton.
    """

    def _f(self, a_x: np.ndarray) -> np.ndarray:
        return a_x

    def start_batch(self) -> None:
        self.n_data = len(self.surrogate.x)

    def add_to_batch(self, optimization_result) -> None:
        y_pred, _ = self.surrogate.predict(optimization_result.x_min)
        x = np.concatenate((self.surrogate.x, optimization_result.x_min))
        y = np.concatenate((self.surrogate.y, y_pred))
        self.surrogate.fit(x, y)

    def finish_batch(self) -> None:
        x = self.surrogate.x[: self.n_data]
        y = self.surrogate.y[: self.n_data]
        self.surrogate.fit(x, y)


class OneShotBatchAcquisitionFunction(AcquisitionFunction):
    def __init__(self, surrogate, base_acquisition):
        super().__init__(surrogate)
        self.base_acquisiton = base_acquisition

        self.xs = []
        self.a_xs = []

    def __call__(self, x):
        a_x = self.base_acquisiton(x)
        self._log_evaluation(x, a_x)
        return a_x

    def fit(self, x, y):
        self.base_acquisiton.fit(x, y)

    def start_optimization(self):
        self.xs = []
        self.a_xs = []

    def get_evaluations(self):
        return np.concatenate(self.xs), np.concatenate(self.a_xs)

    def _log_evaluation(self, x, a_x):
        self.xs.append(x)
        self.a_xs.append(a_x)
