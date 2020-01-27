import numpy as np
import pytest
from sklearn.datasets import make_regression
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern

from bopy.acquisition import EI, POI, UCB
from bopy.exceptions import NotFittedError
from bopy.surrogate import ScipyGPSurrogate


@pytest.mark.parametrize("acquisition", [EI(), UCB(), POI()])
def test_fit_must_be_called_before_evaluating(acquisition):
    # ARRANGE
    n_dimensions = 1
    n_samples = 10

    x, y = make_regression(n_samples, n_dimensions)

    gp = GaussianProcessRegressor(kernel=Matern(nu=1.5))
    surrogate = ScipyGPSurrogate(gp=gp)
    surrogate.fit(x, y)

    # ACT/ASSERT
    with pytest.raises(NotFittedError, match="must call fit before evaluating."):
        acquisition(surrogate, x)


@pytest.mark.parametrize("acquisition", [EI(), UCB(), POI()])
def test_output_dimensions_are_correct(acquisition):
    # ARRANGE
    n_dimensions = 1
    n_samples = 10

    x, y = make_regression(n_samples, n_dimensions)

    gp = GaussianProcessRegressor(kernel=Matern(nu=1.5))
    surrogate = ScipyGPSurrogate(gp=gp)

    surrogate.fit(x, y)
    acquisition.fit(x, y)

    # ACT
    a_x = acquisition(surrogate, x)

    # ASSERT
    assert a_x.shape == (n_samples,)
