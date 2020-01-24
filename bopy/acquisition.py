import numpy as np

from .surrogate import Surrogate


class AcquisitionFunction:
    """An acquisition function.

    Acquisition functions navigate the exploration-exploitation
    trade off during optimization.
    """

    def __call__(self, surrogate: Surrogate, x: np.ndarray) -> np.ndarray:
        """Evaluate the acquisition function at `x` with `surrogate`."""
        return self._f(*surrogate.predict(x))

    def _f(self, mean: np.ndarray, sigma: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class UCB(AcquisitionFunction):
    """Upper confidence bound acquisition function.

    UCB implements the simple rule:
        `ucb(x) = mean + kappa * std`

    Parameters
    ----------
    kappa: float
        The number of stds we add to the mean.
    """

    def __init__(self, kappa: float = 2.0):
        self.kappa = kappa

    def _f(self, mean: np.ndarray, sigma: np.ndarray) -> np.ndarray:
        return mean + self.kappa * np.diag(sigma)
