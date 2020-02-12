import numpy as np

from .acquisition import AcquisitionFunction
from .bounds import Bounds
from .initial_design import InitialDesign
from .optimizer import Optimizer
from .surrogate import Surrogate

__all__ = ["BayesOpt"]


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
        self.run_initial_design(n_initial_design)
        self.run_trials(n_trials)
        # return something

    def run_initial_design(self, n_initial_design: int = 5):
        x = self.initial_design.generate(self.bounds, n_initial_design)
        y = self.objective_function(x)

        self.append_to_dataset(x, y)

        self.surrogate.fit(self.x, self.y)
        self.acquisition_function.fit(self.x, self.y)
        # return something?

    def run_trials(self, n_trials: int = 10):
        for _ in range(n_trials):
            self.run_trial()

    def run_trial(self):
        result = self.optimizer.optimize(
            self.acquisition_function, self.surrogate, self.bounds
        )

        x = result.x_min
        y = self.objective_function(x)

        self.append_to_dataset(x, y)

        self.surrogate.fit(self.x, self.y)
        self.acquisition_function.fit(self.x, self.y)

    def append_to_dataset(self, x: np.ndarray, y: np.ndarray) -> None:
        """Append `x` and `y` to the dataset."""
        if len(self.x) == 0 and len(self.y) == 0:
            self.x = x
            self.y = y
        else:
            self.x = np.concatenate((self.x, x))
            self.y = np.concatenate((self.y, y))
