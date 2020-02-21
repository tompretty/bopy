import numpy as np
import pytest

from bopy.bounds import Bound, Bounds
from bopy.initial_design import (
    LatinHypercubeInitialDesign,
    SobolSequenceInitialDesign,
    UniformRandomInitialDesign,
)


@pytest.fixture(
    scope="module",
    autouse=True,
    params=[
        UniformRandomInitialDesign(),
        SobolSequenceInitialDesign(),
        LatinHypercubeInitialDesign(),
    ],
    ids=["uniform", "sobol", "latin"],
)
def initial_design(request):
    return request.param


n_dimensions = 10


@pytest.fixture(scope="module", autouse=True)
def lowers():
    return [10.0 * (i + 1) for i in range(n_dimensions)]


@pytest.fixture(scope="module", autouse=True)
def uppers():
    return [20.0 * (i + 1) for i in range(n_dimensions)]


@pytest.fixture(scope="module", autouse=True)
def bounds(lowers, uppers):
    return Bounds(bounds=[Bound(lower=l, upper=u) for l, u in zip(lowers, uppers)])


class TestArgumentsToGenerate:
    def test_n_points_must_be_positive(self, initial_design, bounds):
        with pytest.raises(ValueError, match="`n_points` must be positive."):
            initial_design.generate(bounds, n_points=-1)


n_points = 5


class TestGenerateResults:
    @pytest.fixture(scope="class", autouse=True)
    def points(self, initial_design, bounds):
        return initial_design.generate(bounds, n_points)

    def test_points_is_the_correct_shape(self, points):
        assert points.shape == (n_points, n_dimensions)

    def test_points_are_within_bounds(self, points, lowers, uppers):
        for i in range(n_dimensions):
            assert np.all(
                np.logical_and(lowers[i] <= points[:, i], points[:, i] <= uppers[i])
            )
