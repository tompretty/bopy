import GPy
import numpy as np
import pytest

from bopy.acquisition import LCB
from bopy.bayes_opt import BayesOpt
from bopy.benchmark_functions import forrester
from bopy.bounds import Bound, Bounds
from bopy.initial_design import UniformRandomInitialDesign
from bopy.optimizer import DirectOptimizer
from bopy.surrogate import GPyGPSurrogate


@pytest.fixture
def bo():
    def gp_initializer(x, y):
        return GPy.models.GPRegression(
            x, y, kernel=GPy.kern.RBF(input_dim=1), noise_var=1e-5, normalizer=True
        )

    surrogate = GPyGPSurrogate(gp_initializer=gp_initializer)

    return BayesOpt(
        objective_function=forrester,
        surrogate=surrogate,
        acquisition_function=LCB(),
        optimizer=DirectOptimizer(maxf=100),
        initial_design=UniformRandomInitialDesign(),
        bounds=Bounds(bounds=[Bound(lower=0.0, upper=1.0)]),
    )


def test_run_initial_design_results_in_correct_shaped_dataset(bo):
    # ARRANGE
    n_initial_design = 5

    # ACT
    bo.run_initial_design(n_initial_design)

    # ASSERT
    assert bo.x.shape == (n_initial_design, 1)
    assert bo.y.shape == (n_initial_design,)


def test_run_trial_results_in_correct_shaped_dataset(bo):
    # ARRANGE
    n_initial_design = 5

    # ACT
    bo.run_initial_design(n_initial_design)
    bo.run_trial()

    # ASSERT
    assert bo.x.shape == (n_initial_design + 1, 1)
    assert bo.y.shape == (n_initial_design + 1,)


def test_run_trials_results_in_correct_shaped_dataset(bo):
    # ARRANGE
    n_initial_design = 5
    n_trials = 2

    # ACT
    bo.run_initial_design(n_initial_design)
    bo.run_trials(n_trials)

    # ASSERT
    assert bo.x.shape == (n_initial_design + n_trials, 1)
    assert bo.y.shape == (n_initial_design + n_trials,)


def test_append_to_dataset(bo):
    # ARRANGE
    n_samples = 100
    n_dimensions = 1

    x = np.zeros((n_samples, n_dimensions))
    y = np.zeros((n_samples,))

    # ACT
    bo.append_to_dataset(x, y)

    # ASSERT
    assert bo.x.shape == (n_samples, n_dimensions)
    assert bo.y.shape == (n_samples,)

    # ACT
    bo.append_to_dataset(x, y)

    # ASSERT
    assert bo.x.shape == (2 * n_samples, n_dimensions)
    assert bo.y.shape == (2 * n_samples,)
