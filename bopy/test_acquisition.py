import numpy as np
import pytest
from sklearn.datasets import make_regression
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern

from bopy.acquisition import EI, UCB
from bopy.surrogate import ScipyGPSurrogate


@pytest.mark.parametrize("acquisition", [EI(), UCB()])
def test_output_dimensions_are_correct(acquisition):
    # ARRANGE
    n_dimensions = 1
    n_samples = 100

    x, y = make_regression(n_samples, n_dimensions)

    gp = GaussianProcessRegressor(kernel=Matern(nu=1.5))
    surrogate = ScipyGPSurrogate(gp=gp)
    surrogate.fit(x, y)

    # ACT
    a_x = acquisition(surrogate, x)

    # ASSERT
    assert a_x.shape == (n_samples,)
