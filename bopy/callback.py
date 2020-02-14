from .optimizer import OptimizationResult


class Callback:
    def on_acquisition_optimized(self, bo, opt_result: OptimizationResult):
        pass

    def on_surrogate_updated(self, bo):
        pass

    def on_acquisition_updated(self, bo):
        pass
