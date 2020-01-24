import numpy as np
from sklearn.datasets import make_regression
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern

from bopy.acquisition import UCB
from bopy.surrogate import ScipyGPSurrogate


def test_output_dimensions_are_correct():
    # ARRANGE
    n_dimensions = 1
    n_samples = 100

    x, y = make_regression(n_samples, n_dimensions)

    gp = GaussianProcessRegressor(kernel=Matern(nu=1.5))
    surrogate = ScipyGPSurrogate(gp=gp)
    surrogate.fit(x, y)

    acquisition = UCB()

    # ACT
    a_x = acquisition(surrogate, x)

    # ASSERT
    assert a_x.shape == (n_samples,)
