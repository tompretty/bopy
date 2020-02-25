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


@pytest.fixture(scope="class", autouse=True)
def surrogate():
    def gp_initializer(x, y):
        return GPy.models.GPRegression(
            x, y, kernel=GPy.kern.RBF(input_dim=1), noise_var=1e-5, normalizer=True
        )

    return GPyGPSurrogate(gp_initializer=gp_initializer)


@pytest.fixture(scope="class", autouse=True)
def acquisition(surrogate):
    return LCB(surrogate=surrogate)


@pytest.fixture(scope="class", autouse=True)
def bounds():
    return Bounds(bounds=[Bound(lower=0.0, upper=1.0)])


@pytest.fixture(scope="class", autouse=True)
def optimizer(acquisition, bounds):
    return DirectOptimizer(acquisition_function=acquisition, bounds=bounds, maxf=100)


@pytest.fixture(scope="class", autouse=True)
def bo(surrogate, acquisition, optimizer, bounds):
    return BayesOpt(
        objective_function=forrester,
        surrogate=surrogate,
        acquisition_function=acquisition,
        optimizer=optimizer,
        initial_design=UniformRandomInitialDesign(),
        bounds=bounds,
    )


n_initial_design = 5


@pytest.fixture(scope="class")
def bo_after_initial_design(bo):
    bo.run_initial_design(n_initial_design)
    return bo


class TestAfterRunningInitialDesign:
    @pytest.fixture(scope="class", autouse=True)
    def initial_design_result(self, bo):
        return bo.run_initial_design(n_initial_design)

    def test_xs_selected_is_the_correct_shape(self, initial_design_result):
        assert initial_design_result.xs_selected.shape == (n_initial_design, 1)

    def test_f_of_xs_selected_is_the_correct_shape(self, initial_design_result):
        assert initial_design_result.f_of_xs_selected.shape == (n_initial_design,)

    def test_reference_to_x_is_the_correct_shape(self, bo):
        assert bo.x.shape == (n_initial_design, 1)

    def test_reference_to_y_is_the_correct_shape(self, bo):
        assert bo.y.shape == (n_initial_design,)


class TestAfterRunningTrial:
    @pytest.fixture(scope="class", autouse=True)
    def run_trial_result(self, bo_after_initial_design):
        return bo_after_initial_design.run_trial()

    def test_xs_selected_is_the_correct_shape(self, run_trial_result):
        assert run_trial_result.x_selected.shape == (1, 1)

    def test_reference_to_x_is_the_correct_shape(self, bo):
        assert bo.x.shape == (n_initial_design + 1, 1)

    def test_reference_to_y_is_the_correct_shape(self, bo):
        assert bo.y.shape == (n_initial_design + 1,)


n_trials = 2


class TestAfterRunningTrials:
    @pytest.fixture(scope="class", autouse=True)
    def run_trials_result(self, bo_after_initial_design):
        return bo_after_initial_design.run_trials(n_trials)

    def test_reference_to_x_is_the_correct_shape(self, bo):
        assert bo.x.shape == (n_initial_design + n_trials, 1)

    def test_reference_to_y_is_the_correct_shape(self, bo):
        assert bo.y.shape == (n_initial_design + n_trials,)

    def test_run_trials_result_is_the_correct_length(self, run_trials_result):
        assert len(run_trials_result) == n_trials


class TestAfterRunning:
    @pytest.fixture(scope="class", autouse=True)
    def run_result(self, bo):
        return bo.run(n_trials=n_trials, n_initial_design=n_initial_design)

    def test_reference_to_x_is_the_correct_shape(self, bo):
        assert bo.x.shape == (n_initial_design + n_trials, 1)

    def test_reference_to_y_is_the_correct_shape(self, bo):
        assert bo.y.shape == (n_initial_design + n_trials,)

    def test_x_opt_is_the_correct_shape(self, run_result):
        assert run_result.x_opt.shape == (1, 1)

    def test_trials_results_is_the_correct_length(self, run_result):
        assert len(run_result.trial_results) == n_trials
