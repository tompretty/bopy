import GPy
import numpy as np
import pytest

from bopy.acquisition import LCB, KriggingBeliever
from bopy.benchmark_functions import forrester
from bopy.bounds import Bound, Bounds
from bopy.initial_design import UniformRandomInitialDesign
from bopy.optimizer import DirectOptimizer, SequentialBatchOptimizer
from bopy.surrogate import GPyGPSurrogate


def direct_and_surrogate():
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

    return optimizer, surrogate


def sequential_batch_and_surrogate():
    def gp_initializer(x, y):
        return GPy.models.GPRegression(
            x, y, kernel=GPy.kern.RBF(input_dim=1), noise_var=1e-5, normalizer=True
        )

    surrogate = GPyGPSurrogate(gp_initializer=gp_initializer)

    base_acquisition = LCB(surrogate=surrogate)
    acquisition_function = KriggingBeliever(
        surrogate=surrogate, base_acquisition=base_acquisition,
    )

    bounds = Bounds(bounds=[Bound(lower=0.0, upper=1.0)])

    base_optimizer = DirectOptimizer(
        acquisition_function=base_acquisition, bounds=bounds, maxf=100
    )
    optimizer = SequentialBatchOptimizer(
        base_optimizer=base_optimizer,
        batch_size=2,
        acquisition_function=acquisition_function,
        bounds=bounds,
    )

    return optimizer, surrogate


@pytest.mark.parametrize("optimizer, surrogate", [direct_and_surrogate()])
def test_optimize_returns_correct_shaped_result(optimizer, surrogate):
    # ARRANGE
    x = np.linspace(0, 1, 5).reshape(-1, 1)
    y = forrester(x)
    surrogate.fit(x, y)

    # ACT
    optimization_result = optimizer.optimize()

    # ASSERT
    assert isinstance(optimization_result.x_min, np.ndarray)
    assert optimization_result.x_min.shape == (1, 1)
    assert isinstance(optimization_result.f_min, np.ndarray)
    assert optimization_result.f_min.shape == (1,)


@pytest.mark.parametrize("optimizer, surrogate", [sequential_batch_and_surrogate()])
def test_batch_optimize_returns_correct_shaped_result(optimizer, surrogate):
    # ARRANGE
    x = np.linspace(0, 1, 5).reshape(-1, 1)
    y = forrester(x)
    surrogate.fit(x, y)

    # ACT
    optimization_result = optimizer.optimize()

    # ASSERT
    assert isinstance(optimization_result.x_min, np.ndarray)
    assert optimization_result.x_min.shape == (2, 1)
    assert isinstance(optimization_result.f_min, np.ndarray)
    assert optimization_result.f_min.shape == (2,)
