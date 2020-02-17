import numpy as np

from .exceptions import NotFittedError


class FittableMixin:
    """A mixin for classes that are 'fittable'.

    FittableMixin provides a means of validating the 
    values passed to a 'fit-like' and 'predict-like' function
    without prescribing a specific interface i.e. 'fit' and 
    'predict'.

    A class inheriting this mixin requires:
        - has_been_fitted boolean
        - n_dimensions int

    The mixin provides 3 functions: 
        - _validate_ok_for_fitting
        - _confirm_fit
        - _validate_ok_for_predicting

    Usage might look something like:

    def my_custom_fit_method(self, x, y):
        self._validate_ok_for_fitting(x, y)
        ...custom fitting logic...
        self._confirm_fit()

    def my_custom_predict_method(self, x):
        self._validate_ok_for_predicting(x)
        ...custom predicting logic...

       """

    def _validate_ok_for_fitting(self, x: np.ndarray, y: np.ndarray):
        """Validate `x` and `y` are suitable for fitting."""
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
        """Confirm that a fittable has been fit."""
        self.has_been_fitted = True

    def _validate_ok_for_predicting(self, x: np.ndarray):
        """Validate `x` is suitable for predicting."""
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
