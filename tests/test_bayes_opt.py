import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern

from bopy.acquisition import LCB
from bopy.bayes_opt import BayesOpt
from bopy.benchmark_functions import forrester
from bopy.bounds import Bound, Bounds
from bopy.initial_design import SobolSequenceInitialDesign
from bopy.optimizer import DirectOptimizer
from bopy.surrogate import ScipyGPSurrogate


def test_run_initial_design_results_in_correct_shaped_dataset():
    # ARRANGE
    bo = BayesOpt(
        objective_function=forrester,
        surrogate=ScipyGPSurrogate(gp=GaussianProcessRegressor(kernel=Matern(nu=1.5))),
        acquisition_function=LCB(),
        optimizer=DirectOptimizer(),
        initial_design=SobolSequenceInitialDesign(),
        bounds=Bounds(bounds=[Bound(lower=0.0, upper=1.0)]),
    )

    n_initial_design = 5

    # ACT
    bo.run_initial_design(n_initial_design)

    # ASSERT
    assert bo.x.shape == (n_initial_design, 1)
    assert bo.y.shape == (n_initial_design,)


def test_run_trial_results_in_correct_shaped_dataset():
    # ARRANGE
    bo = BayesOpt(
        objective_function=forrester,
        surrogate=ScipyGPSurrogate(gp=GaussianProcessRegressor(kernel=Matern(nu=1.5))),
        acquisition_function=LCB(),
        optimizer=DirectOptimizer(maxf=100),
        initial_design=SobolSequenceInitialDesign(),
        bounds=Bounds(bounds=[Bound(lower=0.0, upper=1.0)]),
    )

    n_initial_design = 5

    # ACT
    bo.run_initial_design(n_initial_design)
    bo.run_trial()

    # ASSERT
    assert bo.x.shape == (n_initial_design + 1, 1)
    assert bo.y.shape == (n_initial_design + 1,)


def test_append_to_dataset():
    # ARRANGE
    bo = BayesOpt(
        objective_function=forrester,
        surrogate=ScipyGPSurrogate(gp=GaussianProcessRegressor(kernel=Matern(nu=1.5))),
        acquisition_function=LCB(),
        optimizer=DirectOptimizer(),
        initial_design=SobolSequenceInitialDesign(),
        bounds=Bounds(bounds=[Bound(lower=0.0, upper=1.0)]),
    )

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
