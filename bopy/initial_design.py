from abc import ABC, abstractmethod

import numpy as np
import pyDOE
import sobol_seq

from .bounds import Bounds


class InitialDesign(ABC):
    """Initial Design base class.

    Initial designs generate a set of points on which to evaluate the
    objective function prior to starting the main BayesOpt routine.
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

    @abstractmethod
    def _generate(self, n_dimensions: int, n_points: int) -> np.ndarray:
        """Generate `n_points` points within [0, 1] ** `n_dimensions`."""


class UniformRandomInitialDesign(InitialDesign):
    """Uniform Random Initial Design.

    Generates an initial design by sampling uniformly at random within
    the parameter bounds.
    """

    def _generate(self, n_dimensions: int, n_points: int) -> np.ndarray:
        return np.random.rand(n_points, n_dimensions)


class SobolSequenceInitialDesign(InitialDesign):
    """Sobol Sequence Initial Design.

    Generates an initial design by from a sobol sequence within the
    parameter bounds.
    """

    def _generate(self, n_dimensions: int, n_points: int) -> np.ndarray:
        return sobol_seq.i4_sobol_generate(n_dimensions, n_points)


class LatinHypercubeInitialDesign(InitialDesign):
    """Latin Hypercube Initial Design.

    Generates an initial design by from a latin hyper cube over the
    parameter bounds.
    """

    def _generate(self, n_dimensions: int, n_points: int) -> np.ndarray:
        return pyDOE.lhs(n_dimensions, samples=n_points)
