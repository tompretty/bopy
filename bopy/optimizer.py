from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Tuple

import numpy as np
from scipydirect import minimize

from .acquisition import AcquisitionFunction, SequentialBatchAcquisitionFunction
from .bounds import Bounds
from .surrogate import Surrogate


@dataclass
class OptimizationResult:
    """Optimization result.

    Parameters
    ----------
    x_min : np.ndarray of shape (batch_size, n_dimensions)
        The argmin.
    f_min : np.ndarray of shape (batch_size,)
        The min.
    """

    x_min: np.ndarray
    f_min: np.ndarray


class Optimizer(ABC):
    """An acquisition function Optimizer.

    Optimizers find the minimum of a given
    acquisition function. This minimum is
    then used as the next query location
    of the objective function.

    This class shouldn't be used directly, use a derived class instead.

    Parameters
    ----------
    acquisition_function : AcquisitionFunction
        The acquisition function.
    bounds : Bounds
        The parameter bounds.
    """

    def __init__(self, acquisition_function: AcquisitionFunction, bounds: Bounds):
        self.acquisition_function = acquisition_function
        self.bounds = bounds

    def optimize(self) -> OptimizationResult:
        """Optimize an acquisition function.

        Optimizes the `acquisition_function` over the `surrogate`
        model, within the `bounds`.

        Returns
        -------
        optimization_result: OptimizationResult
            The result of optimization.
        """
        x_min, f_min = self._optimize()
        return OptimizationResult(x_min=x_min, f_min=f_min)

    @abstractmethod
    def _optimize(self) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError


class DirectOptimizer(Optimizer):
    """Direct acquisition function Optimizer.

    This is a wrapper around the DIRECT
    global optimizer. Specifically, we use
    the scipydirect implementation.

    Parameters
    ----------
    acquisition_function : AcquisitionFunction
        The acquisition function.
    bounds : Bounds
        The parameter bounds.
    direct_kwargs : Dict[str, Any]
        Kwargs passed to scipydirect.minimize.
    """

    def __init__(
        self,
        acquisition_function: AcquisitionFunction,
        bounds: Bounds,
        **direct_kwargs: Dict[str, Any]
    ):
        super().__init__(acquisition_function, bounds)
        self.direct_kwargs = direct_kwargs

    def _optimize(self) -> Tuple[np.ndarray, np.ndarray]:
        def objective(x):
            return self.acquisition_function(x.reshape(1, -1))

        res = minimize(
            objective,
            bounds=list(zip(self.bounds.lowers, self.bounds.uppers)),
            **self.direct_kwargs
        )
        x_min = res.x
        f_min = res.fun

        return np.array([x_min]), np.array([f_min])


class SequentialBatchOptimizer(Optimizer):
    """Sequential Batch Optimizer.

    This is a batch optimizer that selects a batch
    by sequentially selecting points from a 
    SequentialBatchAcquisitionFunction. This proceeds
    by repeatedly optimizing then updating said 
    acquisition function.
    
    Parameters
    ----------
    acquisition_function : SequentialBatchAcquisitionFunction
        The sequential batch acquisition function to be optimized.
    bounds : Bounds
        The parameter bounds.
    base_optimizer : Optimizer
        The underlying optimizer used to optimize the acquisition
        function.
    batch_size : int
        The size of the batch.
    """

    def __init__(
        self,
        acquisition_function: SequentialBatchAcquisitionFunction,
        bounds: Bounds,
        base_optimizer: Optimizer,
        batch_size: int,
    ):
        super().__init__(acquisition_function, bounds)
        self.base_optimizer = base_optimizer
        self.batch_size = batch_size

        self.x_mins = []
        self.f_mins = []

    def _optimize(self) -> Tuple[np.ndarray, np.ndarray]:
        self.start_batch()
        self.acquisition_function.start_batch()
        for _ in range(self.batch_size):
            res = self.base_optimizer.optimize()
            self.add_to_batch(res)
            self.acquisition_function.add_to_batch(res)
        self.acquisition_function.finish_batch()

        return self.get_batch()

    def start_batch(self) -> None:
        """Prepare to start creating a batch."""
        self.x_mins = []
        self.f_mins = []

    def add_to_batch(self, optimization_result: OptimizationResult) -> None:
        """Add the newly selected point to the batch."""
        self.x_mins.append(optimization_result.x_min)
        self.f_mins.append(optimization_result.f_min)

    def get_batch(self) -> None:
        """Get the resulting batch."""
        return np.concatenate(self.x_mins), np.concatenate(self.f_mins)
