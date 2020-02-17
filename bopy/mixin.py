import numpy as np

from .exceptions import NotFittedError


class FittableMixin:
    # requires class to have self.has_been_fitted, self.n_dimensions
    def _validate_ok_for_fitting(self, x: np.ndarray, y: np.ndarray):
        if len(x) == 0:
            raise (ValueError("`x` must contain at least one sample"))
        if len(y) == 0:
            raise (ValueError("`y` must contain at least one sample"))
        if len(x) != len(y):
            raise (ValueError("`x` and `y` must have the same number of samples"))
        if len(x.shape) != 2:
            raise (ValueError("`x` must be 2D"))
        if len(y.shape) != 1:
            raise (ValueError("`y` must be 1D"))

        self.n_dimensions = x.shape[1]

    def _confirm_fit(self):
        self.has_been_fitted = True

    def _validate_ok_for_predicting(self, x: np.ndarray):
        if not self.has_been_fitted:
            raise NotFittedError("must be fitted first")
        if len(x) == 0:
            raise ValueError("`x` must contain at least one sample")
        if len(x.shape) != 2:
            raise ValueError("`x` must be 2D")
        if x.shape[1] != self.n_dimensions:
            raise ValueError(
                "`x` must have the same number of dimensions as the training data"
            )
