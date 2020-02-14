from dataclasses import dataclass
from typing import List

import numpy as np

from .acquisition import AcquisitionFunction
from .bounds import Bounds
from .initial_design import InitialDesign
from .optimizer import Optimizer
from .surrogate import Surrogate

__all__ = ["BayesOpt"]


@dataclass
class BOInitialDesignResult:
    xs_selected: np.ndarray
    f_of_xs_selected: np.ndarray
    x_opt_so_far: np.ndarray
    f_of_x_opt_so_far: float


@dataclass
class BOTrialResult:
    x_selected: np.ndarray
    f_of_x_selected: float
    x_opt_so_far: np.ndarray
    f_of_x_opt_so_far: float


@dataclass
class BOResult:
    x_opt: np.ndarray
    f_of_x_opt: float
    initial_design_result: BOInitialDesignResult
    trial_results: List[BOTrialResult]


class BayesOpt:
    def __init__(
        self,
        objective_function,
        surrogate: Surrogate,
        acquisition_function: AcquisitionFunction,
        optimizer: Optimizer,
        initial_design: InitialDesign,
        bounds: Bounds,
    ):
        self.objective_function = objective_function
        self.surrogate = surrogate
        self.acquisition_function = acquisition_function
        self.optimizer = optimizer
        self.initial_design = initial_design
        self.bounds = bounds

        self.x = np.array([])
        self.y = np.array([])

    def run(self, n_trials: int = 10, n_initial_design: int = 5):
        initial_design_result = self.run_initial_design(n_initial_design)
        trial_results = self.run_trials(n_trials)

        x_opt, f_of_x_opt = self.get_opt_so_far()

        return BOResult(
            x_opt=x_opt,
            f_of_x_opt=f_of_x_opt,
            initial_design_result=initial_design_result,
            trial_results=trial_results,
        )

    def run_initial_design(self, n_initial_design: int = 5) -> BOInitialDesignResult:
        x = self.initial_design.generate(self.bounds, n_initial_design)
        y = self.objective_function(x)
        self.append_to_dataset(x, y)

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
        return [self.run_trial() for _ in range(n_trials)]

    def run_trial(self) -> BOTrialResult:
        result = self.optimizer.optimize(
            self.acquisition_function, self.surrogate, self.bounds
        )

        x = result.x_min
        y = self.objective_function(x)

        self.append_to_dataset(x, y)

        self.surrogate.fit(self.x, self.y)
        self.acquisition_function.fit(self.x, self.y)

        x_opt_so_far, f_of_x_opt_so_far = self.get_opt_so_far()

        return BOTrialResult(
            x_selected=x,
            f_of_x_selected=y[0],
            x_opt_so_far=x_opt_so_far,
            f_of_x_opt_so_far=f_of_x_opt_so_far,
        )

    def append_to_dataset(self, x: np.ndarray, y: np.ndarray) -> None:
        """Append `x` and `y` to the dataset."""
        if len(self.x) == 0 and len(self.y) == 0:
            self.x = x
            self.y = y
        else:
            self.x = np.concatenate((self.x, x))
            self.y = np.concatenate((self.y, y))

    def get_opt_so_far(self):
        index_of_opt = np.argmin(self.y)
        x_opt = np.atleast_2d(self.x[index_of_opt])
        y_opt = self.y[index_of_opt]

        return x_opt, y_opt
