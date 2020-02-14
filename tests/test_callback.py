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
    on_acquisition_optimized_rasied = False
    on_acquisition_updated_rasied = False
    on_surrogate_updated_rasied = False

    class MyCallback(Callback):
        def on_acquisition_optimized(self, bo, opt_result):
            nonlocal on_acquisition_optimized_rasied
            on_acquisition_optimized_rasied = True

        def on_acquisition_updated(self, bo):
            nonlocal on_acquisition_updated_rasied
            on_acquisition_updated_rasied = True

        def on_surrogate_updated(self, bo):
            nonlocal on_surrogate_updated_rasied
            on_surrogate_updated_rasied = True

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

    n_inial_design = 5
    bo.run_initial_design(n_initial_design=n_inial_design)

    # ACT
    bo.run_trial()

    # ASSERT
    assert on_acquisition_optimized_rasied == True
    assert on_acquisition_updated_rasied == True
    assert on_surrogate_updated_rasied == True
