import numpy as np


def forrester(x: np.ndarray):
    """The forrester function.

    Parameters
    ----------
    x : np.ndarray of shape (n_samples, 1)
        The input locations.

    Returns
    -------
    np.ndarray of shape (n_samples,)
        The function values at `x`.
    """
    return ((6 * x - 2) ** 2 * np.sin(12 * x - 4)).flatten()


def bohachevsky(x: np.ndarray):
    """The bohachevsky function.

    Parameters
    ----------
    x : np.ndarray of shape (n_samples, 2)
        The input locations.

    Returns
    -------
    np.ndarray of shape (n_samples,)
        The function values at `x`.
    """
    x1 = x[:, 0]
    x2 = x[:, 1]

    return (
        x1 ** 2
        + 2 * x2 ** 2
        - 0.3 * np.cos(3 * np.pi * x1)
        - 0.4 * np.cos(4 * np.pi * x2)
        + 0.7
    )
