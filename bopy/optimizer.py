from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Tuple

import numpy as np
from scipydirect import minimize

from .acquisition import AcquisitionFunction
from .bounds import Bounds
from .surrogate import Surrogate


@dataclass
class OptimizationResult:
    """Optimization result.

    Parameters
    ----------
    x_min: np.ndarray of shape (n_dimensions, 1)
        The argmin.
    f_min: float
        The min.
    """

    x_min: np.ndarray
    f_min: float


class Optimizer(ABC):
    """An acquisition function Optimizer.

    Optimizers find the minimum of a given
    acquisition function. This minimum is
    then used as the next query location
    of the objective function.

    This class shouldn't be used directly, use a derived class instead.
    """

    def __init__(self, acquisition_function: AcquisitionFunction, bounds: Bounds):
        self.acquisition_function = acquisition_function
        self.bounds = bounds

    def optimize(self) -> OptimizationResult:
        """Optimize an acquisition function.

        Optimizes the `acquisition_function` over the `surrogate`
        model, within the `bounds`.

        Parameters
        ----------
        acquisition_function: AcquisitionFunction
            The acquisition function.
        bounds: Bounds
            The parameter bounds.

        Returns
        -------
        optimization_result: OptimizationResult
            The result of optimization.
        """
        x_min, f_min = self._optimize()
        return OptimizationResult(x_min=x_min, f_min=f_min)

    @abstractmethod
    def _optimize(self) -> Tuple[np.ndarray, float]:
        raise NotImplementedError


class DirectOptimizer(Optimizer):
    """Direct acquisition function Optimizer.

    This is a wrapper around the DIRECT
    global optimizer.

    Parameters
    ----------
    direct_args
        Args passed to DIRECT.solve
    direct_kwargs
        Kwargs passed to DIRECT.solve
    """

    def __init__(
        self, acquisition_function: AcquisitionFunction, bounds: Bounds, **direct_kwargs
    ):
        super().__init__(acquisition_function, bounds)
        self.direct_kwargs = direct_kwargs

    def _optimize(self) -> Tuple[np.ndarray, float]:
        def objective(x):
            return self.acquisition_function(x.reshape(1, -1))

        res = minimize(
            objective,
            bounds=list(zip(self.bounds.lowers, self.bounds.uppers)),
            **self.direct_kwargs
        )
        x_min = res.x
        f_min = res.fun

        return np.array([x_min]), f_min
