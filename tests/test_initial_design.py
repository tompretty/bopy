import numpy as np
import pytest

from bopy.bounds import Bound, Bounds
from bopy.initial_design import (
    LatinHypercubeInitialDesign,
    SobolSequenceInitialDesign,
    UniformRandomInitialDesign,
)


@pytest.mark.parametrize(
    "design",
    [
        UniformRandomInitialDesign(),
        SobolSequenceInitialDesign(),
        LatinHypercubeInitialDesign(),
    ],
)
def test_n_points_must_be_positive(design):
    # ARRANGE
    n_points = -1
    bounds = Bounds(bounds=[Bound(lower=0.0, upper=1.0)])

    # ACT/ASSERT
    with pytest.raises(ValueError, match="`n_points` must be positive."):
        design.generate(bounds, n_points)


@pytest.mark.parametrize(
    "design",
    [
        UniformRandomInitialDesign(),
        SobolSequenceInitialDesign(),
        LatinHypercubeInitialDesign(),
    ],
)
def test_points_are_the_correct_shape(design):
    # ARRANGE
    n_dimensions = 5
    n_points = 10
    bounds = Bounds(bounds=[Bound(lower=0.0, upper=1.0) for _ in range(n_dimensions)])

    # ACT
    points = design.generate(bounds, n_points)

    # ASSERT
    assert points.shape == (n_points, n_dimensions)


@pytest.mark.parametrize(
    "design",
    [
        UniformRandomInitialDesign(),
        SobolSequenceInitialDesign(),
        LatinHypercubeInitialDesign(),
    ],
)
def test_points_are_within_bounds(design):
    # ARRANGE
    l_0, u_0 = -10.0, -5.0
    l_1, u_1 = -5.0, 5.0
    l_2, u_2 = 5.0, 10.0
    bounds = Bounds(
        bounds=[
            Bound(lower=l_0, upper=u_0),
            Bound(lower=l_1, upper=u_1),
            Bound(lower=l_2, upper=u_2),
        ]
    )

    n_points = 100

    # ACT
    points = design.generate(bounds, n_points)

    # ASSERT
    assert np.all(np.logical_and(l_0 <= points[:, 0], points[:, 0] <= u_0))
    assert np.all(np.logical_and(l_1 <= points[:, 1], points[:, 1] <= u_1))
    assert np.all(np.logical_and(l_2 <= points[:, 2], points[:, 2] <= u_2))
