from pytest import raises

from bopy.bounds import Bound, Bounds


def test_lower_must_be_less_than_upper_bound():
    Bound(lower=0.0, upper=1.0)

    with raises(ValueError):
        Bound(lower=1.0, upper=0.0)


def test_lowers_returns_list_of_lower_bounds():
    # ARRANGE
    lowers = [0.0, 1.0, 2.0]
    uppers = [1.0, 2.0, 3.0]

    # ACT
    bounds = Bounds(bounds=[Bound(lower=l, upper=u) for l, u in zip(lowers, uppers)])

    # ASSERT
    assert bounds.lowers == lowers


def test_uppers_returns_list_of_upper_bounds():
    # ARRANGE
    lowers = [0.0, 1.0, 2.0]
    uppers = [1.0, 2.0, 3.0]

    # ACT
    bounds = Bounds(bounds=[Bound(lower=l, upper=u) for l, u in zip(lowers, uppers)])

    # ASSERT
    assert bounds.uppers == uppers
