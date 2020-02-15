import GPy
import numpy as np

from bopy.acquisition import LCB
from bopy.bayes_opt import BayesOpt
from bopy.benchmark_functions import forrester
from bopy.bounds import Bound, Bounds
from bopy.callback import Callback
from bopy.initial_design import UniformRandomInitialDesign
from bopy.optimizer import DirectOptimizer
from bopy.surrogate import GPyGPSurrogate


def test_all_callbacks_are_dispatched():
    # ARRANGE
    on_initial_design_end_raised = False
    on_acquisition_optimized_rasied = False
    on_acquisition_updated_rasied = False
    on_surrogate_updated_rasied = False
    on_trial_end_raised = False
    on_bo_end_raised = False

    class MyCallback(Callback):
        def on_initial_design_end(self, bo):
            nonlocal on_initial_design_end_raised
            on_initial_design_end_raised = True

        def on_acquisition_optimized(self, bo, opt_result):
            nonlocal on_acquisition_optimized_rasied
            on_acquisition_optimized_rasied = True

        def on_acquisition_updated(self, bo):
            nonlocal on_acquisition_updated_rasied
            on_acquisition_updated_rasied = True

        def on_surrogate_updated(self, bo):
            nonlocal on_surrogate_updated_rasied
            on_surrogate_updated_rasied = True

        def on_trial_end(self, bo):
            nonlocal on_trial_end_raised
            on_trial_end_raised = True

        def on_bo_end(self, bo):
            nonlocal on_bo_end_raised
            on_bo_end_raised = True

    def gp_initializer(x, y):
        return GPy.models.GPRegression(
            x, y, kernel=GPy.kern.RBF(input_dim=1), noise_var=1e-5, normalizer=True
        )

    surrogate = GPyGPSurrogate(gp_initializer=gp_initializer)

    bo = BayesOpt(
        objective_function=forrester,
        surrogate=surrogate,
        acquisition_function=LCB(),
        optimizer=DirectOptimizer(maxf=100),
        initial_design=UniformRandomInitialDesign(),
        bounds=Bounds(bounds=[Bound(lower=0.0, upper=1.0)]),
        callbacks=[MyCallback()],
    )

    n_initial_design = 5
    n_trials = 1

    # ACT
    bo.run(n_trials=n_trials, n_initial_design=n_initial_design)

    # ASSERT
    assert on_initial_design_end_raised == True
    assert on_acquisition_optimized_rasied == True
    assert on_acquisition_updated_rasied == True
    assert on_surrogate_updated_rasied == True
    assert on_trial_end_raised == True
    assert on_bo_end_raised == True
