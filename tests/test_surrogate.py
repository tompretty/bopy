import GPy
import numpy as np
import pytest
from sklearn.datasets import make_regression
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern

from bopy.exceptions import NotFittedError
from bopy.surrogate import GPyGPSurrogate, ScipyGPSurrogate


def scipy_gp_surrogate():
    return ScipyGPSurrogate(
        gp=GaussianProcessRegressor(kernel=Matern(nu=1.5), alpha=1e-5, normalize_y=True)
    )


def gpy_gp_surrogate():
    def gp_initializer(x, y):
        return GPy.models.GPRegression(
            x, y, kernel=GPy.kern.RBF(input_dim=1), noise_var=1e-5, normalizer=True
        )

    return GPyGPSurrogate(gp_initializer=gp_initializer)


def noise_free_gpy_gp_surrogate():
    def gp_initializer(x, y):
        gp = GPy.models.GPRegression(
            x, y, kernel=GPy.kern.RBF(input_dim=1), noise_var=1e-10, normalizer=True
        )
        gp.Gaussian_noise.fix()

        return gp

    return GPyGPSurrogate(gp_initializer=gp_initializer)


@pytest.mark.parametrize("surrogate", [scipy_gp_surrogate(), gpy_gp_surrogate()])
def test_training_data_must_contain_at_least_one_sample(surrogate):
    # ARRANGE
    x, y = np.array([]), np.array([])

    # ACT/ASSERT
    with pytest.raises(ValueError, match="data must contain at least one sample"):
        surrogate.fit(x, y)


@pytest.mark.parametrize("surrogate", [scipy_gp_surrogate(), gpy_gp_surrogate()])
def test_training_input_and_target_must_be_the_same_size(surrogate):
    # ARRANGE
    n_dimensions = 10
    n_samples_x = 15
    n_samples_y = 5

    x, y = make_regression(n_samples_x, n_dimensions)
    y = y[:n_samples_y]

    # ACT/ASSERT
    with pytest.raises(
        ValueError, match="`x` and `y` must have the same number of samples"
    ):
        surrogate.fit(x, y)


@pytest.mark.parametrize("surrogate", [scipy_gp_surrogate(), gpy_gp_surrogate()])
def test_training_input_must_be_2d(surrogate):
    # ARRANGE
    x, y = np.array([[[1]]]), np.array([1])

    # ACT/ASSERT
    with pytest.raises(ValueError, match="`x` must be 2D"):
        surrogate.fit(x, y)


@pytest.mark.parametrize("surrogate", [scipy_gp_surrogate(), gpy_gp_surrogate()])
def test_training_target_must_be_1d(surrogate):
    # ARRANGE
    x, y = np.array([[1]]), np.array([[1]])

    # ACT/ASSERT
    with pytest.raises(ValueError, match="`y` must be 1D"):
        surrogate.fit(x, y)


@pytest.mark.parametrize("surrogate", [scipy_gp_surrogate(), gpy_gp_surrogate()])
def test_fit_must_be_called_before_predict(surrogate):
    # ARRANGE
    n_dimensions = 10
    n_samples = 100

    x, y = make_regression(n_samples, n_dimensions)

    # ACT/ASSERT
    with pytest.raises(NotFittedError, match="fit must be called before predict"):
        surrogate.predict(x)


@pytest.mark.parametrize("surrogate", [scipy_gp_surrogate(), gpy_gp_surrogate()])
def test_predict_returns_correct_dimensions(surrogate):
    # ARRANGE
    n_dimensions = 10
    n_samples = 100

    x, y = make_regression(n_samples, n_dimensions)

    surrogate.fit(x, y)

    # ACT
    y_pred, sigma = surrogate.predict(x)

    # ASSERT
    assert y_pred.shape == (n_samples,)
    assert sigma.shape == (n_samples, n_samples)


@pytest.mark.parametrize("surrogate", [scipy_gp_surrogate(), gpy_gp_surrogate()])
def test_test_input_contains_at_least_one_sample(surrogate):
    # ARRANGE
    n_dimensions = 10
    n_samples = 100

    x_train, y_train = make_regression(n_samples, n_dimensions)
    x_test = np.array([])

    surrogate.fit(x_train, y_train)

    # ACT/ASSERT
    with pytest.raises(ValueError, match="`x` must contain at least one sample"):
        surrogate.predict(x_test)


@pytest.mark.parametrize("surrogate", [scipy_gp_surrogate(), gpy_gp_surrogate()])
def test_test_input_must_be_2d(surrogate):
    # ARRANGE
    n_dimensions = 10
    n_samples = 100

    x_train, y_train = make_regression(n_samples, n_dimensions)
    x_test = np.zeros((n_samples, n_dimensions, 1))

    surrogate.fit(x_train, y_train)

    # ACT/ASSERT
    with pytest.raises(ValueError, match="`x` must be 2D"):
        surrogate.predict(x_test)


@pytest.mark.parametrize("surrogate", [scipy_gp_surrogate(), gpy_gp_surrogate()])
def test_test_input_must_have_same_number_of_dimensions_as_training_input(surrogate):
    # ARRANGE
    n_dimensions_train = 10
    n_samples_train = 100

    n_dimensions_test = 100
    n_samples_test = 10

    x_train, y_train = make_regression(n_samples_train, n_dimensions_train)
    x_test, _ = make_regression(n_samples_test, n_dimensions_test)

    surrogate.fit(x_train, y_train)

    # ACT/ASSERT
    with pytest.raises(
        ValueError,
        match="`x` must have the same number of dimensions as the training data",
    ):
        surrogate.predict(x_test)


@pytest.mark.parametrize(
    "surrogate", [scipy_gp_surrogate(), noise_free_gpy_gp_surrogate()]
)
def test_noise_free_gps_interpolate_the_training_data(surrogate):
    # ARRANGE
    n_dimensions = 1
    n_samples = 5

    x, y = make_regression(n_samples, n_dimensions)

    surrogate.fit(x, y)

    # ACT
    y_pred, sigma = surrogate.predict(x)
    var = np.diag(sigma)

    # ASSERT
    assert np.allclose(y, y_pred, atol=1e-3)
    assert np.allclose(0, var, atol=1e-3)


@pytest.mark.parametrize("surrogate", [scipy_gp_surrogate(), gpy_gp_surrogate()])
def test_gps_return_to_the_prior_far_away_from_the_training_data(surrogate):
    # ARRANGE
    n_dimensions = 1
    n_samples = 10

    x_train = np.linspace(-0.5, 0.5, n_samples).reshape(-1, 1)
    y_train = np.sin(x_train.flatten())

    x_test = np.array([[-50], [50]])

    surrogate.fit(x_train, y_train)

    # ACT
    y_pred, sigma = surrogate.predict(x_test)
    stds = np.diag(sigma)

    # ASSERT
    assert np.allclose(0, y_pred, atol=1e-3)
    assert np.allclose(stds[0], stds[1], atol=1e-3)


def test_GPyGPSurrogate_instantiates_a_gp_after_calling_fit():
    # ARRANGE
    n_dimensions = 1
    n_samples = 100
    x, y = make_regression(n_samples, n_dimensions)
    surrogate = gpy_gp_surrogate()

    # ACT
    surrogate.fit(x, y)

    # ASSERT
    assert surrogate.gp is not None


def test_GPyGPSurrogate_updates_data_on_second_call_to_fit():
    # ARRANGE
    n_dimensions = 1
    n_samples = 100
    x1, y1 = make_regression(n_samples, n_dimensions)
    x2, y2 = make_regression(n_samples, n_dimensions)
    surrogate = gpy_gp_surrogate()
    surrogate.fit(x1, y1)

    # ACT
    surrogate.fit(x2, y2)

    # ASSERT
    assert np.array_equal(surrogate.gp.X, x2)
    assert np.array_equal(surrogate.gp.Y, y2.reshape(-1, 1))
