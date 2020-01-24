import numpy as np
import pytest
from sklearn.datasets import make_regression
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern

from bopy.exceptions import NotFittedError
from bopy.surrogate import ScipyGPSurrogate


def test_x_must_contain_at_least_one_sample():
    # ARRANGE
    x, y = np.array([]), np.array([])

    gp = GaussianProcessRegressor(kernel=Matern(nu=1.5))
    surrogate = ScipyGPSurrogate(gp=gp)

    # ACT/ASSERT
    with pytest.raises(ValueError, match="data must contain at least one sample"):
        surrogate.fit(x, y)


def test_x_and_y_must_be_same_size():
    # ARRANGE
    n_dimensions = 10
    n_samples_x = 15
    n_samples_y = 5

    x, y = make_regression(n_samples_x, n_dimensions)
    y = y[:n_samples_y]

    gp = GaussianProcessRegressor(kernel=Matern(nu=1.5))
    surrogate = ScipyGPSurrogate(gp=gp)

    # ACT/ASSERT
    with pytest.raises(
        ValueError, match="`x` and `y` must have the same number of samples"
    ):
        surrogate.fit(x, y)


def test_x_must_be_2d():
    # ARRANGE
    x, y = np.array([[[1]]]), np.array([1])

    gp = GaussianProcessRegressor(kernel=Matern(nu=1.5))
    surrogate = ScipyGPSurrogate(gp=gp)

    # ACT/ASSERT
    with pytest.raises(ValueError, match="`x` must be 2D"):
        surrogate.fit(x, y)


def test_y_must_be_1d():
    # ARRANGE
    x, y = np.array([[1]]), np.array([[1]])

    gp = GaussianProcessRegressor(kernel=Matern(nu=1.5))
    surrogate = ScipyGPSurrogate(gp=gp)

    # ACT/ASSERT
    with pytest.raises(ValueError, match="`y` must be 1D"):
        surrogate.fit(x, y)


def test_fit_must_be_called_before_predict():
    # ARRANGE
    n_dimensions = 10
    n_samples = 100

    x, y = make_regression(n_samples, n_dimensions)

    gp = GaussianProcessRegressor(kernel=Matern(nu=1.5))
    surrogate = ScipyGPSurrogate(gp=gp)

    # ACT/ASSERT
    with pytest.raises(NotFittedError, match="fit must be called before predict"):
        surrogate.predict(x)


def test_predict_returns_correct_dimensions():
    # ARRANGE
    n_dimensions = 10
    n_samples = 100

    x, y = make_regression(n_samples, n_dimensions)

    gp = GaussianProcessRegressor(kernel=Matern(nu=1.5))
    surrogate = ScipyGPSurrogate(gp=gp)
    surrogate.fit(x, y)

    # ACT
    y_pred, sigma = surrogate.predict(x)

    # ASSERT
    assert y_pred.shape == (n_samples,)
    assert sigma.shape == (n_samples, n_samples)


def test_predict_x_contains_at_least_one_point():
    # ARRANGE
    n_dimensions = 10
    n_samples = 100

    x_train, y_train = make_regression(n_samples, n_dimensions)
    x_test = np.array([])

    gp = GaussianProcessRegressor(kernel=Matern(nu=1.5))
    surrogate = ScipyGPSurrogate(gp=gp)
    surrogate.fit(x_train, y_train)

    # ACT/ASSERT
    with pytest.raises(ValueError, match="`x` must contain at least one sample"):
        surrogate.predict(x_test)


def test_predict_x_is_2d():
    # ARRANGE
    n_dimensions = 10
    n_samples = 100

    x_train, y_train = make_regression(n_samples, n_dimensions)
    x_test = np.zeros((n_samples, n_dimensions, 1))

    gp = GaussianProcessRegressor(kernel=Matern(nu=1.5))
    surrogate = ScipyGPSurrogate(gp=gp)
    surrogate.fit(x_train, y_train)

    # ACT/ASSERT
    with pytest.raises(ValueError, match="`x` must be 2D"):
        surrogate.predict(x_test)


def test_predict_x_requires_the_same_number_of_dimensions_as_fit_x():
    # ARRANGE
    n_dimensions_train = 10
    n_samples_train = 100

    n_dimensions_test = 100
    n_samples_test = 10

    x_train, y_train = make_regression(n_samples_train, n_dimensions_train)
    x_test, _ = make_regression(n_samples_test, n_dimensions_test)

    gp = GaussianProcessRegressor(kernel=Matern(nu=1.5))
    surrogate = ScipyGPSurrogate(gp=gp)
    surrogate.fit(x_train, y_train)

    # ACT/ASSERT
    with pytest.raises(
        ValueError,
        match="`x` must have the same number of dimensions as the training data",
    ):
        surrogate.predict(x_test)


def test_noise_free_gps_interpolate_the_training_data():
    # ARRANGE
    n_dimensions = 1
    n_samples = 100

    x, y = make_regression(n_samples, n_dimensions)

    gp = GaussianProcessRegressor(kernel=Matern(nu=1.5))
    surrogate = ScipyGPSurrogate(gp=gp)
    surrogate.fit(x, y)

    # ACT
    y_pred, sigma = surrogate.predict(x)
    stds = np.diag(sigma)

    # ASSERT
    assert np.allclose(y, y_pred)
    assert np.allclose(0, stds)


def test_gps_return_to_prior_far_away_from_the_training_data():
    # ARRANGE
    n_dimensions = 1
    n_samples = 10

    x_train = np.linspace(-0.5, 0.5, n_samples).reshape(-1, 1)
    y_train = np.sin(x_train.flatten())

    x_test = np.array([[-100], [100]])

    gp = GaussianProcessRegressor(kernel=Matern(nu=1.5))
    surrogate = ScipyGPSurrogate(gp=gp)
    surrogate.fit(x_train, y_train)

    # ACT
    y_pred, sigma = surrogate.predict(x_test)
    stds = np.diag(sigma)

    # ASSERT
    assert np.allclose(0, y_pred)
    assert np.allclose(stds[0], stds[1])
