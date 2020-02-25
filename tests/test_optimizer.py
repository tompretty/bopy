import GPy
import numpy as np
import pytest

from bopy.acquisition import LCB, KriggingBeliever, OneShotBatchAcquisitionFunction
from bopy.benchmark_functions import forrester
from bopy.bounds import Bound, Bounds
from bopy.optimizer import (
    DirectOptimizer,
    OneShotBatchOptimizer,
    OneShotBatchOptimizerKDPPSamplingStrategy,
    OneShotBatchOptimizerRandomSamplingStrategy,
    SequentialBatchOptimizer,
)
from bopy.surrogate import GPyGPSurrogate

n_samples = 10


@pytest.fixture(scope="module", autouse=True)
def x():
    return np.linspace(0, 1, n_samples).reshape(-1, 1)


@pytest.fixture(scope="module", autouse=True)
def y(x):
    return forrester(x)


@pytest.fixture(scope="module", autouse=True)
def surrogate():
    def gp_initializer(x, y):
        return GPy.models.GPRegression(
            x, y, kernel=GPy.kern.RBF(input_dim=1), noise_var=1e-5, normalizer=True
        )

    return GPyGPSurrogate(gp_initializer=gp_initializer)


@pytest.fixture(scope="module", autouse=True)
def trained_surrogate(surrogate, x, y):
    surrogate.fit(x, y)
    return surrogate


@pytest.fixture(scope="module", autouse=True)
def acquisition(trained_surrogate):
    return LCB(surrogate=trained_surrogate)


@pytest.fixture(scope="module", autouse=True)
def trained_acquisition(acquisition, x, y):
    acquisition.fit(x, y)
    return acquisition


@pytest.fixture(scope="module", autouse=True)
def bounds():
    return Bounds(bounds=[Bound(lower=0.0, upper=1.0)])


class TestBaseOptimizersAfterOptimize:
    @pytest.fixture(scope="class", autouse=True)
    def optimizer(self, trained_acquisition, bounds):
        return DirectOptimizer(
            acquisition_function=trained_acquisition, bounds=bounds, maxf=100
        )

    @pytest.fixture(scope="class", autouse=True)
    def optimization_result(self, optimizer):
        return optimizer.optimize()

    def test_x_min_is_a_numpy_array(self, optimization_result):
        assert isinstance(optimization_result.x_min, np.ndarray)

    def test_x_min_is_the_correct_shape(self, optimization_result):
        assert optimization_result.x_min.shape == (1, 1)

    def test_f_min_is_a_numpy_array(self, optimization_result):
        assert isinstance(optimization_result.f_min, np.ndarray)

    def test_f_min_is_the_correct_shape(self, optimization_result):
        assert optimization_result.f_min.shape == (1,)


n_batch = 2


class TestSequentialBatchOptimizerAfterOptimize:
    @pytest.fixture(scope="class", autouse=True)
    def sequential_acquisition(self, trained_surrogate, trained_acquisition):
        return KriggingBeliever(
            surrogate=trained_surrogate, base_acquisition=trained_acquisition
        )

    @pytest.fixture(scope="class", autouse=True)
    def optimizer(self, sequential_acquisition, bounds):
        return SequentialBatchOptimizer(
            acquisition_function=sequential_acquisition,
            bounds=bounds,
            base_optimizer=DirectOptimizer(
                acquisition_function=sequential_acquisition, bounds=bounds, maxf=100
            ),
            batch_size=n_batch,
        )

    @pytest.fixture(scope="class", autouse=True)
    def optimization_result(self, optimizer):
        return optimizer.optimize()

    def test_x_min_is_a_numpy_array(self, optimization_result):
        assert isinstance(optimization_result.x_min, np.ndarray)

    def test_x_min_is_the_correct_shape(self, optimization_result):
        assert optimization_result.x_min.shape == (n_batch, 1)

    def test_f_min_is_a_numpy_array(self, optimization_result):
        assert isinstance(optimization_result.f_min, np.ndarray)

    def test_f_min_is_the_correct_shape(self, optimization_result):
        assert optimization_result.f_min.shape == (n_batch,)


class TestOneShotBatchOptimizerAfterOptimize:
    @pytest.fixture(scope="class", autouse=True)
    def one_shot_acquistion(self, trained_surrogate, trained_acquisition):
        return OneShotBatchAcquisitionFunction(
            surrogate=trained_surrogate, base_acquisition=trained_acquisition
        )

    @pytest.fixture(scope="class", autouse=True)
    def one_shot_strategy(self):
        return OneShotBatchOptimizerRandomSamplingStrategy()

    @pytest.fixture(scope="class", autouse=True)
    def optimizer(self, one_shot_acquistion, one_shot_strategy, bounds):
        return OneShotBatchOptimizer(
            acquisition_function=one_shot_acquistion,
            bounds=bounds,
            batch_size=n_batch,
            strategy=one_shot_strategy,
            base_optimizer=DirectOptimizer(
                acquisition_function=one_shot_acquistion, bounds=bounds, maxf=100
            ),
        )

    @pytest.fixture(scope="class", autouse=True)
    def optimization_result(self, optimizer):
        return optimizer.optimize()

    def test_x_min_is_a_numpy_array(self, optimization_result):
        assert isinstance(optimization_result.x_min, np.ndarray)

    def test_x_min_is_the_correct_shape(self, optimization_result):
        assert optimization_result.x_min.shape == (n_batch, 1)

    def test_f_min_is_a_numpy_array(self, optimization_result):
        assert isinstance(optimization_result.f_min, np.ndarray)

    def test_f_min_is_the_correct_shape(self, optimization_result):
        assert optimization_result.f_min.shape == (n_batch,)


def random_strategy():
    return OneShotBatchOptimizerRandomSamplingStrategy()


def kdpp_strategy():
    def k(x):
        x_sq = np.sum(x * x, 1)[:, None]
        r2 = x_sq - 2 * x @ x.T + x_sq.T
        return np.exp(-r2)

    return OneShotBatchOptimizerKDPPSamplingStrategy(kernel=k)


class TestOneShotBatchOptimizerStrategiesAfterSelect:
    @pytest.fixture(
        scope="class",
        autouse=True,
        params=[random_strategy(), kdpp_strategy()],
        ids=["random", "k-dpp"],
    )
    def strategy(self, request):
        return request.param

    @pytest.fixture(scope="class", autouse=True)
    def selection(self, strategy, x, y):
        return strategy.select(x, y, n_batch)

    @pytest.fixture(scope="class", autouse=True)
    def x_selected(self, selection):
        return selection[0]

    @pytest.fixture(scope="class", autouse=True)
    def y_selected(self, selection):
        return selection[1]

    def test_x_selected_is_the_correct_shape(self, x_selected):
        assert x_selected.shape == (n_batch, 1)

    def test_y_selected_is_the_correct_shape(self, y_selected):
        assert y_selected.shape == (n_batch,)
