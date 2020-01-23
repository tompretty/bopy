from abc import ABC, abstractmethod

import numpy as np
import pyDOE
import sobol_seq

from .bounds import Bounds

__all__ = [
    "UniformRandomInitialDesign",
    "SobolSequenceInitialDesign",
    "LatinHyperCubeInitialDesign",
]


class InitialDesign(ABC):
    """Initial Design base class.

    Initial designs generate a set of points on which
    to evaluate the objective function prior to starting
    the main BayesOpt routine.

    This class shouldn't be used directly, use a derived class instead.
    """

    def generate(self, bounds: Bounds, n_points: int) -> np.ndarray:
        """Generate `n_points` points within `bounds`."""
        if n_points <= 0:
            raise ValueError("`n_points` must be positive.")

        points = self._generate(bounds.n_dimensions, n_points)
        for i, (l, u) in enumerate(zip(bounds.lowers, bounds.uppers)):
            points[:, i] *= u - l
            points[:, i] += l
        return points

    def _generate(self, n_dimensions: int, n_points: int) -> np.ndarray:
        """Generate `n_points` points within [0, 1] ** `n_dimensions`."""
        raise NotImplementedError


class UniformRandomInitialDesign(InitialDesign):
    """Uniform Random Initial Design.


    Generates an initial design by sampling
    uniformly at random within the parameter bounds.
    """

    def _generate(self, n_dimensions: int, n_points: int) -> np.ndarray:
        return np.random.rand(n_points, n_dimensions)


class SobolSequenceInitialDesign(InitialDesign):
    """Sobol Sequence Initial Design.


    Generates an initial design by from
    a sobol sequence within the parameter bounds.
    """

    def _generate(self, n_dimensions: int, n_points: int) -> np.ndarray:
        return sobol_seq.i4_sobol_generate(n_dimensions, n_points)


class LatinHyperCubeInitialDesign(InitialDesign):
    """Latin Hyper Cube Initial Design.

    Generates an initial design by from
    a latin hyper cube over the parameter bounds.
    """

    def _generate(self, n_dimensions: int, n_points: int) -> np.ndarray:
        return pyDOE.lhs(n_dimensions, samples=n_points)
