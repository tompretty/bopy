from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np

from .acquisition import AcquisitionFunction
from .bounds import Bound, Bounds
from .optimizer import OptimizationResult
from .surrogate import Surrogate


def plot_surrogate_1D(
    ax: plt.Axes, surrogate: Surrogate, bound: Bound, n_points: int = 100,
) -> None:
    """Plot a 1D surrogate model.

    Parameters
    ----------
    ax: plt.Axes
        matplotlib axes object on which to plot the graph.
    surrogate: Surrogate
        A trained surrogate model.
    bound: Bound
        The bound on which to plot the graph. NB: doesn't have to be the
        same as the optimization bound.
    n_points: int
        The number of points with with to plot the graph, by default 100.
    """
    x = np.linspace(bound.lower, bound.upper, n_points).reshape(-1, 1)
    y_pred, sigma = surrogate.predict(x)
    std = np.sqrt(np.diag(sigma))
    lower = y_pred - 2 * std
    upper = y_pred + 2 * std

    ax.plot(x.flatten(), y_pred, label="mean")
    ax.fill_between(x.flatten(), lower, upper, alpha=0.5, label="confidence interval")
    ax.plot(surrogate.x.flatten(), surrogate.y, "k+", label="training data")


def plot_acquisition_function_1D(
    ax: plt.Axes,
    acquisition_function: AcquisitionFunction,
    bound: Bound,
    n_points: int = 100,
) -> None:
    """Plot a 1D acquisition_function.

    Parameters
    ----------
    ax: plt.Axes
        matplotlib axes object on which to plot the graph.
    acquisition_function: AcquisitionFunction
        A trained acquisition function.
    bound: Bound
        The bound on which to plot the graph. NB: doesn't have to be the same as the
        optimization bound.
    n_points: int
        The number of points with with to plot the graph, by default 100.
    """
    x = np.linspace(bound.lower, bound.upper, n_points).reshape(-1, 1)
    a_x = acquisition_function(x)

    ax.plot(x.flatten(), a_x, label="acquisition function")


def plot_optimization_result_1D(
    ax: plt.Axes, optimization_result: OptimizationResult
) -> None:
    """Plot a 1D optimization result.

    Parameters
    ----------
    ax: plt.Axes
        matplotlib axes object on which to plot the graph.
    optimization_result:
        The optimization result.
    """
    ax.plot(
        optimization_result.x_min,
        optimization_result.f_min,
        "r*",
        label="acquisition function min",
    )


def plot_surrogate_2D(
    ax: plt.Axes,
    surrogate: Surrogate,
    bounds: Bounds,
    n_points: Tuple[int, int] = (50, 50),
) -> None:
    """Plot a 2D surrogate model.

    Parameters
    ----------
    ax: plt.Axes
        matplotlib axes object on which to plot the graph.
    surrogate: Surrogate
        A trained surrogate model.
    bounds: Bounds
        The bounds on which to plot the graph. NB: doesn't have to be the
        same as the optimization bound.
    n_points: Tuple[int, int]
        The number of x and y points on which to plot the graph,
        by default (50, 50).
    """
    n_x, n_y = n_points
    xx, yy, xs = _get_points_for_2d_grid(n_x, n_y, bounds)
    fxs, _ = surrogate.predict(xs)
    zz = fxs.reshape(n_x, n_y)

    ax.pcolor(xx, yy, zz, cmap="viridis")
    ax.plot(surrogate.x[:, 0], surrogate.x[:, 1], "bo", label="training data")


def plot_acquisition_function_2D(
    ax: plt.Axes,
    acquisition_function: AcquisitionFunction,
    bounds: Bounds,
    n_points: Tuple[int, int] = (50, 50),
) -> None:
    """Plot a 2D acquisition function.

    Parameters
    ----------
    ax: plt.Axes
        matplotlib axes object on which to plot the graph.
    acquisition_function: AcquisitionFunction
        A trained acquisition function.
    bounds: Bounds
        The bounds on which to plot the graph. NB: doesn't have to be the
        same as the optimization bound.
    n_points: Tuple[int, int]
        The number of x and y points on which to plot the graph,
        by default (50, 50).
    """
    n_x, n_y = n_points
    xx, yy, xs = _get_points_for_2d_grid(n_x, n_y, bounds)
    fxs = acquisition_function(xs)
    zz = fxs.reshape(n_x, n_y)

    ax.pcolor(xx, yy, zz, cmap="viridis")


def plot_optimization_result_2D(
    ax: plt.Axes, optimization_result: OptimizationResult
) -> None:
    """Plot a 2D optimization result.

    Parameters
    ----------
    ax: plt.Axes
        matplotlib axes object on which to plot the graph.
    optimization_result:
        The optimization result.
    """
    ax.plot(
        optimization_result.x_min[:, 0],
        optimization_result.x_min[:, 1],
        "r*",
        markersize=9,
        label="selected point(s)",
    )


def _get_points_for_2d_grid(
    n_x: int, n_y: int, bounds: Bounds
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Get meshgrids for plotting and inputs for evaluating."""
    lowers = bounds.lowers
    uppers = bounds.uppers

    x = np.linspace(lowers[0], uppers[0], n_x).reshape(-1, 1)
    y = np.linspace(lowers[1], uppers[1], n_y).reshape(-1, 1)

    xx, yy = np.meshgrid(x, y)
    xs = np.hstack((xx.reshape(-1, 1), yy.reshape(-1, 1)))

    return xx, yy, xs
