from .optimizer import OptimizationResult


class Callback:
    """A Callback.

    Callbacks are called after certain events happen during the optimization.
    They provide a means to execute arbitrary code (e.g. making some plots)
    at intermediary stages in the optimization.

    Custom subclasses should overwrite the events they are interested in.
    """

    def on_initial_design_end(self, bo):
        """Event raised after the initial design has been run.

        Parameters
        ----------
        bo : BayesOpt
            The BayesOpt object.
        """
        pass

    def on_acquisition_optimized(self, bo, opt_result: OptimizationResult):
        """Event raised after the acquisition function has been updated.

        Parameters
        ----------
        bo : BayesOpt
            The BayesOpt object.
        opt_result : OptimizationResult
            The result from optimizing the acquisition function.
        """
        pass

    def on_surrogate_updated(self, bo):
        """Event raised after the surrogate has been updated.

        Parameters
        ----------
        bo : BayesOpt
            The BayesOpt object.
        """
        pass

    def on_acquisition_updated(self, bo):
        """Event raised after the acquisition function has been updated.

        Parameters
        ----------
        bo : BayesOpt
            The BayesOpt object.
        """
        pass

    def on_trial_end(self, bo):
        """Event raised after a single BO trial has been run.

        Parameters
        ----------
        bo : BayesOpt
            The BayesOpt object.
        """
        pass

    def on_bo_end(self, bo):
        """Event raised after the BayesOpt routine has been run.

        Parameters
        ----------
        bo : BayesOpt
            The BayesOpt object.
        """
        pass
