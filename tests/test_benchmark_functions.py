import numpy as np

from bopy.benchmark_functions import bohachevsky, forrester


def test_forrester_output_shape_is_correct():
    n_samples = 100
    n_dimensions = 1

    x = np.random.randn(n_samples, n_dimensions)
    y = forrester(x)

    assert y.shape == (n_samples,)


def test_bohachevsky_output_shape_is_correct():
    n_samples = 100
    n_dimensions = 2

    x = np.random.randn(n_samples, n_dimensions)
    y = bohachevsky(x)

    assert y.shape == (n_samples,)
