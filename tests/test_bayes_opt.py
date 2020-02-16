import GPy
import numpy as np
import pytest

from bopy.acquisition import LCB
from bopy.bayes_opt import BayesOpt, BOInitialDesignResult, BOResult, BOTrialResult
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
    acquistion_function = LCB(surrogate=surrogate)
    bounds = Bounds(bounds=[Bound(lower=0.0, upper=1.0)])
    optimizer = DirectOptimizer(
        acquisition_function=acquistion_function, bounds=bounds, maxf=100
    )

    return BayesOpt(
        objective_function=forrester,
        surrogate=surrogate,
        acquisition_function=acquistion_function,
        optimizer=optimizer,
        initial_design=UniformRandomInitialDesign(),
        bounds=bounds,
    )


def test_run_initial_design_results_in_correct_shaped_dataset(bo):
    # ARRANGE
    n_initial_design = 5

    # ACT
    bo.run_initial_design(n_initial_design)

    # ASSERT
    assert bo.x.shape == (n_initial_design, 1)
    assert bo.y.shape == (n_initial_design,)


def test_run_initial_design_returns_the_correct_shaped_results(bo):
    # ARRANGE
    n_initial_design = 5

    # ACT
    initial_design_result = bo.run_initial_design(n_initial_design)

    # ASSERT
    assert isinstance(initial_design_result, BOInitialDesignResult)
    assert initial_design_result.xs_selected.shape == (n_initial_design, 1)
    assert initial_design_result.f_of_xs_selected.shape == (n_initial_design,)
    assert initial_design_result.x_opt_so_far.shape == (1, 1)
    assert isinstance(initial_design_result.f_of_x_opt_so_far, float)


def test_run_trial_results_in_correct_shaped_dataset(bo):
    # ARRANGE
    n_initial_design = 5

    # ACT
    bo.run_initial_design(n_initial_design)
    bo.run_trial()

    # ASSERT
    assert bo.x.shape == (n_initial_design + 1, 1)
    assert bo.y.shape == (n_initial_design + 1,)


def test_run_trial_returns_the_correct_shaped_results(bo):
    # ARRANGE
    n_initial_design = 5
    bo.run_initial_design(n_initial_design)

    # ACT
    trial_result = bo.run_trial()

    # ASSERT
    assert isinstance(trial_result, BOTrialResult)
    assert trial_result.x_selected.shape == (1, 1)
    assert isinstance(trial_result.f_of_x_selected, float)
    assert trial_result.x_opt_so_far.shape == (1, 1)
    assert isinstance(trial_result.f_of_x_opt_so_far, float)


def test_run_trials_results_in_correct_shaped_dataset(bo):
    # ARRANGE
    n_initial_design = 5
    n_trials = 2
    bo.run_initial_design(n_initial_design)

    # ACT
    bo.run_trials(n_trials)

    # ASSERT
    assert bo.x.shape == (n_initial_design + n_trials, 1)
    assert bo.y.shape == (n_initial_design + n_trials,)


def test_run_trials_returns_the_correct_shaped_results(bo):
    # ARRANGE
    n_initial_design = 5
    n_trials = 2
    bo.run_initial_design(n_initial_design)

    # ACT
    trial_results = bo.run_trials(n_trials)

    # ASSERT
    assert len(trial_results) == n_trials
    assert isinstance(trial_results[0], BOTrialResult)


def test_run_results_in_correct_shaped_dataset(bo):
    # ARRANGE
    n_initial_design = 5
    n_trials = 2

    # ACT
    bo.run(n_trials=n_trials, n_initial_design=n_initial_design)

    # ASSERT
    assert bo.x.shape == (n_initial_design + n_trials, 1)
    assert bo.y.shape == (n_initial_design + n_trials,)


def test_run_returns_correct_shaped_results(bo):
    # ARRANGE
    n_initial_design = 5
    n_trials = 2

    # ACT
    result = bo.run(n_trials=n_trials, n_initial_design=n_initial_design)

    # ASSERT
    assert isinstance(result, BOResult)
    assert result.x_opt.shape == (1, 1)
    assert isinstance(result.f_of_x_opt, float)
    assert isinstance(result.initial_design_result, BOInitialDesignResult)
    assert len(result.trial_results) == n_trials
    assert isinstance(result.trial_results[0], BOTrialResult)


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
