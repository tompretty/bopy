from dataclasses import dataclass
from typing import List


@dataclass
class Bound:
    """A parameter bound

    Parameters
    ----------
    lower: float
        The lower bound
    upper: float
        The upper bound
    """

    lower: float
    upper: float

    def __post_init__(self):
        if self.lower >= self.upper:
            raise ValueError("`lower` must be less than `upper`")


@dataclass
class Bounds:
    """A collection of parameter bounds

    Parameters
    ----------
    bounds: List[Bound]
        The list of parameter bounds
    """

    bounds: List[Bound]

    @property
    def n_dimensions(self) -> int:
        """Get the number of dimensions"""
        return len(self.bounds)

    @property
    def lowers(self) -> List[float]:
        """Get a list of lower bounds."""
        return [b.lower for b in self.bounds]

    @property
    def uppers(self) -> List[float]:
        """Get a list of upper bounds."""
        return [b.upper for b in self.bounds]
