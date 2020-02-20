from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Tuple

import numpy as np

from .acquisition import AcquisitionFunction
from .bounds import Bounds
from .callback import Callback
from .initial_design import InitialDesign
from .optimizer import OptimizationResult, Optimizer
from .surrogate import Surrogate


@dataclass
class BOInitialDesignResult:
    """BO Initial Design Result.

    Parameters
    ----------
    xs_selected: np.ndarray of size (n_initial_design, n_dimensions)
        The initial points evaluated.
    f_of_xs_selected: np.ndarray of size (n_initial_design,)
        The objective function value at `xs_selected`.
    x_opt_so_far: np.ndarray of size (1, n_dimensions)
        The location of the current optimizer.
    f_of_x_opt_so_far: float
        The objective function value at `x_opt_so_far`
    """

    xs_selected: np.ndarray
    f_of_xs_selected: np.ndarray
    x_opt_so_far: np.ndarray
    f_of_x_opt_so_far: float


@dataclass
class BOTrialResult:
    """BO Trial Result.

    Parameters
    ----------
    x_selected: np.ndarray of size (1, n_dimensions)
        The optimizer of the acquisition function.
    f_of_xs_selected: float
        The objective function value at `x_selected`.
    x_opt_so_far: np.ndarray of size (1, n_dimensions)
        The location of the current optimizer.
    f_of_x_opt_so_far: float
        The objective function value at `x_opt_so_far`
    """

    x_selected: np.ndarray
    f_of_x_selected: float
    x_opt_so_far: np.ndarray
    f_of_x_opt_so_far: float


@dataclass
class BOResult:
    """BO Result.

    Parameters
    ----------
    x_opt: np.ndarray of size (1, n_dimensions)
        The location of the optimizer.
    f_of_x_opt: float
        The objective function value at `x_opt`.
    initial_design_result: BOInitialDesignResult
        The result from running the initial design.
    trial_results: List[BOTrialResult]
        The result from running each trial.
    """

    x_opt: np.ndarray
    f_of_x_opt: float
    initial_design_result: BOInitialDesignResult
    trial_results: List[BOTrialResult]


class BayesOpt:
    """Bayesian Optimization (BO).

    BO is a heuristic for global optimization of black box functions. 
    It involves training a probabilistic surrogate model of the objective 
    which can be queried as a cheap alternative to the true objective. 
    An acquistion function is used to navigate the trade-off between exploring
    areas of the space in which the model is uncertain, and exploiting the
    currently known promising areas.

    Parameters
    ----------
    objective_function : Callable[[np.ndarray], np.ndarray]
        The objective function to be optimized.
    surrogate : Surrogate
        The probabilistic surrogate model of the true objective.
    acquisition_function : AcquisitionFunction
        The acquisition function that navigates the 
        exploration-exploitation trade-off.
    optimizer : Optimizer
        The optimizer used to optimize the acquistion function
    initial_design : InitialDesign
        The strategy for initially evaluating the objective.
    bounds : Bounds
        The parameter bounds for the optimization.
    """

    def __init__(
        self,
        objective_function: Callable[[np.ndarray], np.ndarray],
        surrogate: Surrogate,
        acquisition_function: AcquisitionFunction,
        optimizer: Optimizer,
        initial_design: InitialDesign,
        bounds: Bounds,
        callbacks: Optional[List[Callback]] = None,
    ):
        self.objective_function = objective_function
        self.surrogate = surrogate
        self.acquisition_function = acquisition_function
        self.optimizer = optimizer
        self.initial_design = initial_design
        self.bounds = bounds
        self.callbacks = callbacks

        self.x = np.array([])
        self.y = np.array([])

    def run(self, n_trials: int = 10, n_initial_design: int = 5) -> BOResult:
        """Run BO.

        Parameters
        ----------
        n_trials : int, optional
            The number of BO trails to run, by default 10
        n_initial_design : int, optional
            The number of initial design points, by default 5

        Returns
        -------
        BOResult
            The results from the BO routine.
        """
        initial_design_result = self.run_initial_design(n_initial_design)
        trial_results = self.run_trials(n_trials)

        self.dispatch("on_bo_end", self)

        x_opt, f_of_x_opt = self.get_opt_so_far()

        return BOResult(
            x_opt=x_opt,
            f_of_x_opt=f_of_x_opt,
            initial_design_result=initial_design_result,
            trial_results=trial_results,
        )

    def run_initial_design(self, n_initial_design: int = 5) -> BOInitialDesignResult:
        """Run an initial design.

        Parameters
        ----------
        n_initial_design : int, optional
            The number of design points to evaluate, by default 5

        Returns
        -------
        BOInitialDesignResult
            The result of running the initial design.
        """
        x = self.initial_design.generate(self.bounds, n_initial_design)
        y = self.objective_function(x)
        self.append_to_dataset(x, y)

        self.dispatch("on_initial_design_end", self)

        self.surrogate.fit(self.x, self.y)
        self.acquisition_function.fit(self.x, self.y)

        x_opt_so_far, f_of_x_opt_so_far = self.get_opt_so_far()

        return BOInitialDesignResult(
            xs_selected=x,
            f_of_xs_selected=y,
            x_opt_so_far=x_opt_so_far,
            f_of_x_opt_so_far=f_of_x_opt_so_far,
        )

    def run_trials(self, n_trials: int = 10) -> List[BOTrialResult]:
        """Run `n_trials` BO trials.

        Parameters
        ----------
        n_trials : int, optional
            The number of trials to run, by default 10

        Returns
        -------
        List[BOTrialResult]
            A list of the results from indivdual BO trials.
        """
        return [self.run_trial() for _ in range(n_trials)]

    def run_trial(self) -> BOTrialResult:
        """Run a single BO trial.

        Returns
        -------
        BOTrialResult
            The result from a single BO trial.
        """
        result = self.optimize_acquisition()

        x = result.x_min
        y = self.objective_function(x)

        self.append_to_dataset(x, y)

        self.update_surrogate()
        self.update_acquisition()

        self.dispatch("on_trial_end", self)

        x_opt_so_far, f_of_x_opt_so_far = self.get_opt_so_far()

        return BOTrialResult(
            x_selected=x,
            f_of_x_selected=y[0],
            x_opt_so_far=x_opt_so_far,
            f_of_x_opt_so_far=f_of_x_opt_so_far,
        )

    def optimize_acquisition(self) -> OptimizationResult:
        """Optimize the acquisition function."""
        result = self.optimizer.optimize()
        self.dispatch("on_acquisition_optimized", self, result)

        return result

    def update_surrogate(self) -> None:
        """Update the surrogate model."""
        self.surrogate.fit(self.x, self.y)
        self.dispatch("on_surrogate_updated", self)

    def update_acquisition(self) -> None:
        """Update the acquisition function."""
        self.acquisition_function.fit(self.x, self.y)
        self.dispatch("on_acquisition_updated", self)

    def dispatch(self, event: str, *args: List[Any]) -> None:
        """Dispatch `event` callback with `args` arguments."""
        if self.callbacks:
            for callback in self.callbacks:
                getattr(callback, event)(*args)

    def append_to_dataset(self, x: np.ndarray, y: np.ndarray) -> None:
        """Append `x` and `y` to the dataset."""
        if len(self.x) == 0 and len(self.y) == 0:
            self.x = x
            self.y = y
        else:
            self.x = np.concatenate((self.x, x))
            self.y = np.concatenate((self.y, y))

    def get_opt_so_far(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get the current argmin and min of the objective."""
        index_of_opt = np.argmin(self.y)
        x_opt = np.atleast_2d(self.x[index_of_opt])
        f_opt = self.y[index_of_opt]

        return x_opt, f_opt
