import numpy as np
import pytest
from sklearn.datasets import make_regression
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern

from bopy.acquisition import EI, LCB, POI
from bopy.exceptions import NotFittedError
from bopy.surrogate import ScipyGPSurrogate

n_samples = 10


@pytest.fixture(scope="module", autouse=True)
def x():
    return np.linspace(-np.pi, np.pi, n_samples).reshape(-1, 1)


@pytest.fixture(scope="module", autouse=True)
def y(x):
    return np.sin(x).flatten()


@pytest.fixture(scope="module", autouse=True)
def surrogate():
    return ScipyGPSurrogate(gp=GaussianProcessRegressor(kernel=Matern()))


@pytest.fixture(scope="module", autouse=True)
def trained_surrogate(surrogate, x, y):
    surrogate.fit(x, y)
    return surrogate


@pytest.fixture(scope="class", params=[LCB, EI, POI], ids=["LCB", "EI", "PO"])
def acquisition(request, trained_surrogate):
    return request.param(surrogate=trained_surrogate)


@pytest.fixture(scope="class")
def trained_acquisition(acquisition, x, y):
    acquisition.fit(x, y)
    return acquisition


class TestBeforeFitting:
    def test_evaluating_raises_not_fitted_error(self, acquisition, x):
        with pytest.raises(NotFittedError, match="must be fitted first"):
            acquisition(x)


class TestAfterFitting:
    def test_output_dimensions_are_correct(self, trained_acquisition, x):
        assert trained_acquisition(x).shape == (n_samples,)
