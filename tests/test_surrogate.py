import GPy
import numpy as np
import pytest
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern

from bopy.benchmark_functions import forrester
from bopy.exceptions import NotFittedError
from bopy.surrogate import GPyGPSurrogate, ScipyGPSurrogate

n_samples = 10


@pytest.fixture(scope="module", autouse=True)
def x():
    return np.linspace(0, 1, n_samples).reshape(-1, 1)


@pytest.fixture(scope="module", autouse=True)
def y(x):
    return forrester(x)


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


@pytest.fixture(
    scope="module",
    autouse=True,
    params=[scipy_gp_surrogate(), gpy_gp_surrogate()],
    ids=["scipy_gp", "gpy_gp"],
)
def surrogate(request):
    return request.param


@pytest.fixture(scope="class")
def trained_surrogate(surrogate, x, y):
    surrogate.fit(x, y)
    return surrogate


class TestArgumentsToFit:
    def test_x_must_contain_at_least_one_sample(self, surrogate):
        with pytest.raises(ValueError, match="`x` must contain at least one sample"):
            surrogate.fit(x=np.array([]), y=np.array([1.0]))

    def test_y_must_contain_at_least_one_sample(self, surrogate):
        with pytest.raises(ValueError, match="`y` must contain at least one sample"):
            surrogate.fit(x=np.array([[1.0]]), y=np.array([]))

    def test_x_and_y_must_contain_the_same_number_of_samples(self, surrogate):
        with pytest.raises(
            ValueError, match="`x` and `y` must have the same number of samples"
        ):
            surrogate.fit(x=np.array([[1.0]]), y=np.array([1.0, 1.0]))

    def test_x_must_be_2d(self, surrogate):
        with pytest.raises(ValueError, match="`x` must be 2D"):
            surrogate.fit(x=np.array([[[1.0]]]), y=np.array([1.0]))

    def test_y_must_be_1d(self, surrogate):
        with pytest.raises(ValueError, match="`y` must be 1D"):
            surrogate.fit(x=np.array([[1.0]]), y=np.array([[1.0]]))


class TestBeforeFitting:
    def test_calling_predict_raises_not_fitted_error(self, surrogate, x):
        with pytest.raises(NotFittedError, match="must be fitted first"):
            surrogate.predict(x)


class TestArgumentsToPredictAfterFitting:
    def test_x_must_contain_at_least_one_sample(self, trained_surrogate):
        with pytest.raises(ValueError, match="`x` must contain at least one sample"):
            trained_surrogate.predict(x=np.array([]))

    def test_x_must_be_2d(self, trained_surrogate):
        with pytest.raises(ValueError, match="`x` must be 2D"):
            trained_surrogate.predict(x=np.array([1.0]))

    def test_x_must_have_the_same_number_of_dimensions_as_the_training_data(
        self, trained_surrogate
    ):
        with pytest.raises(
            ValueError,
            match="`x` must have the same number of dimensions as the training data",
        ):
            trained_surrogate.predict(x=np.array([[1.0, 1.0]]))


class TestAfterPredicting:
    @pytest.fixture(scope="class", autouse=True)
    def predictions(self, trained_surrogate, x):
        return trained_surrogate.predict(x)

    @pytest.fixture(scope="class", autouse=True)
    def predicted_mean(self, predictions):
        return predictions[0]

    @pytest.fixture(scope="class", autouse=True)
    def predicted_var(self, predictions):
        return predictions[1]

    def test_predicted_mean_is_the_correct_shape(self, predicted_mean):
        assert predicted_mean.shape == (n_samples,)

    def test_predicted_var_is_the_correct_shape(self, predicted_var):
        assert predicted_var.shape == (n_samples, n_samples)

    def test_reference_to_x_is_stored(self, trained_surrogate, x):
        assert np.array_equal(trained_surrogate.x, x)

    def test_reference_to_y_is_stored(self, trained_surrogate, y):
        assert np.array_equal(trained_surrogate.y, y)
