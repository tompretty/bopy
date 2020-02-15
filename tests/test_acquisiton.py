import numpy as np
import pytest
from sklearn.datasets import make_regression
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern

from bopy.acquisition import EI, LCB, POI
from bopy.exceptions import NotFittedError
from bopy.surrogate import ScipyGPSurrogate


def ei_and_surrogate():
    surrogate = ScipyGPSurrogate(gp=GaussianProcessRegressor(kernel=Matern()))
    ei = EI(surrogate=surrogate)

    return ei, surrogate


def lcb_and_surrogate():
    surrogate = ScipyGPSurrogate(gp=GaussianProcessRegressor(kernel=Matern()))
    lcb = LCB(surrogate=surrogate)

    return lcb, surrogate


def poi_and_surrogate():
    surrogate = ScipyGPSurrogate(gp=GaussianProcessRegressor(kernel=Matern()))
    poi = POI(surrogate=surrogate)

    return poi, surrogate


@pytest.mark.parametrize(
    "acquisition, surrogate",
    [ei_and_surrogate(), lcb_and_surrogate(), poi_and_surrogate()],
)
def test_fit_must_be_called_before_evaluating(acquisition, surrogate):
    # ARRANGE
    n_dimensions = 1
    n_samples = 10

    x, y = make_regression(n_samples=n_samples, n_features=n_dimensions)

    surrogate.fit(x, y)

    # ACT/ASSERT
    with pytest.raises(NotFittedError, match="must call fit before evaluating."):
        acquisition(x)


@pytest.mark.parametrize(
    "acquisition, surrogate",
    [ei_and_surrogate(), lcb_and_surrogate(), poi_and_surrogate()],
)
def test_output_dimensions_are_correct(acquisition, surrogate):
    # ARRANGE
    n_dimensions = 1
    n_samples = 10

    x, y = make_regression(n_samples, n_dimensions)

    surrogate.fit(x, y)
    acquisition.fit(x, y)

    # ACT
    a_x = acquisition(x)

    # ASSERT
    assert a_x.shape == (n_samples,)
