import pytest
from pytest import raises

from bopy.bounds import Bound, Bounds


class TestBoundCreation:
    def test_lower_must_be_less_than_upper(self):
        with raises(ValueError):
            Bound(lower=1.0, upper=0.0)


class TestBoundsCreation:
    def test_at_least_one_bound_is_required(self):
        with raises(ValueError, match="`bounds` must contain at least one bound."):
            Bounds(bounds=[])


class TestBoundsAfterCreation:
    n_dimensions = 10

    @pytest.fixture(scope="class", autouse=True)
    def lowers(self):
        return [i for i in range(1, self.n_dimensions + 1)]

    @pytest.fixture(scope="class", autouse=True)
    def uppers(self):
        return [2 * i for i in range(1, self.n_dimensions + 1)]

    @pytest.fixture(scope="class", autouse=True)
    def bounds(self, lowers, uppers):
        return Bounds(bounds=[Bound(lower=l, upper=u) for l, u in zip(lowers, uppers)])

    def test_n_dimensions_is_correct(self, bounds):
        assert bounds.n_dimensions == self.n_dimensions

    def test_lowers_is_correct(self, bounds, lowers):
        assert bounds.lowers == lowers

    def test_uppers_is_correct(self, bounds, uppers):
        assert bounds.uppers == uppers
