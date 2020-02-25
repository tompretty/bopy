import GPy
import numpy as np
import pytest

from bopy.acquisition import LCB
from bopy.bayes_opt import BayesOpt
from bopy.benchmark_functions import forrester
from bopy.bounds import Bound, Bounds
from bopy.callback import Callback
from bopy.initial_design import UniformRandomInitialDesign
from bopy.optimizer import DirectOptimizer
from bopy.surrogate import GPyGPSurrogate


@pytest.fixture(scope="module", autouse=True)
def surrogate():
    def gp_initializer(x, y):
        return GPy.models.GPRegression(
            x, y, kernel=GPy.kern.RBF(input_dim=1), noise_var=1e-5, normalizer=True
        )

    return GPyGPSurrogate(gp_initializer=gp_initializer)


@pytest.fixture(scope="module", autouse=True)
def acquisition(surrogate):
    return LCB(surrogate=surrogate)


@pytest.fixture(scope="module", autouse=True)
def bounds():
    return Bounds(bounds=[Bound(lower=0.0, upper=1.0)])


@pytest.fixture(scope="module", autouse=True)
def optimizer(acquisition, bounds):
    return DirectOptimizer(acquisition_function=acquisition, bounds=bounds, maxf=100)


@pytest.fixture(scope="module", autouse=True)
def callback():
    class TestCallback(Callback):
        def __init__(self):
            super().__init__()
            on_initial_design_end_raised = False
            on_acquisition_optimized_rasied = False
            on_acquisition_updated_rasied = False
            on_surrogate_updated_rasied = False
            on_trial_end_raised = False
            on_bo_end_raised = False

        def on_initial_design_end(self, bo):
            self.on_initial_design_end_raised = True

        def on_acquisition_optimized(self, bo, opt_result):
            self.on_acquisition_optimized_rasied = True

        def on_acquisition_updated(self, bo):
            self.on_acquisition_updated_rasied = True

        def on_surrogate_updated(self, bo):
            self.on_surrogate_updated_rasied = True

        def on_trial_end(self, bo):
            self.on_trial_end_raised = True

        def on_bo_end(self, bo):
            self.on_bo_end_raised = True

    return TestCallback()


@pytest.fixture(scope="module", autouse=True)
def bo(surrogate, acquisition, bounds, optimizer, callback):
    return BayesOpt(
        objective_function=forrester,
        surrogate=surrogate,
        acquisition_function=acquisition,
        optimizer=optimizer,
        initial_design=UniformRandomInitialDesign(),
        bounds=bounds,
        callbacks=[callback],
    )


@pytest.fixture(scope="module", autouse=True)
def bo_after_running(bo):
    bo.run(n_trials=1, n_initial_design=5)
    return bo


class TestAfterRunningBO:
    def test_on_initial_design_end_raised(self, callback):
        assert callback.on_initial_design_end_raised == True

    def test_on_acquistion_optimized_raised(self, callback):
        assert callback.on_acquisition_optimized_rasied == True

    def test_on_acuqisition_updated_raised(self, callback):
        assert callback.on_acquisition_updated_rasied == True

    def test_on_surrogate_update_raised(self, callback):
        assert callback.on_surrogate_updated_rasied == True

    def test_on_trial_end_raised(self, callback):
        assert callback.on_trial_end_raised == True

    def test_on_bo_end_raised(self, callback):
        assert callback.on_bo_end_raised == True
