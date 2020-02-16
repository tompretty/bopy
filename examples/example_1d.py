import GPy
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from bopy.acquisition import LCB
from bopy.bayes_opt import BayesOpt
from bopy.benchmark_functions import forrester
from bopy.bounds import Bound, Bounds
from bopy.callback import Callback
from bopy.initial_design import SobolSequenceInitialDesign
from bopy.optimizer import DirectOptimizer
from bopy.plotting import (
    plot_acquisition_function_1D,
    plot_optimization_result_1D,
    plot_surrogate_1D,
)
from bopy.surrogate import GPyGPSurrogate

sns.set_style("darkgrid")

# Let's set up some constants
n_inital_design = 5
n_trials = 5


# Now we define a custom callback to make plots at different stages of the optimization.
# We'll create a figure with `n_trials` rows, each of which has two columns: the first of which
# is plot just before the surrogate is updated; and the second just after.
class PlottingCallback(Callback):
    def __init__(self):
        super().__init__()
        self.fig, self.axs = plt.subplots(n_trials, 2, figsize=(10, 16))
        self.current_trial = 0

    def on_acquisition_optimized(self, bo, opt_result):
        # This is called during a BO trial, just after the acquisiton function has been optimized.
        # Here we create a plot of the surrogate, acquistion function, and the acquisition optimum.
        # We position it in first column of the `current_trial`th row.
        ax = self.axs[self.current_trial][0]

        plot_surrogate_1D(ax, bo.surrogate, bo.x, bo.y, bo.bounds.bounds[0])
        plot_acquisition_function_1D(ax, bo.acquisition_function, bo.bounds.bounds[0])
        plot_optimization_result_1D(ax, opt_result)

    def on_surrogate_updated(self, bo):
        # This is called during a BO trial, just after the surrogate has been fit to the new datum.
        # Here we create a plot of just the surrogate. We position it in the second column of the
        # `current_trial`th row.
        ax = self.axs[self.current_trial][1]

        plot_surrogate_1D(ax, bo.surrogate, bo.x, bo.y, bo.bounds.bounds[0])

    def on_trial_end(self, _):
        # This is called at the end of a BO trial. We just increment our counter keeping track
        # of the current trial.
        self.current_trial += 1

    def on_bo_end(self, _):
        # This is called right at the end of the BO routine. We save the figure to a file.
        self.fig.savefig("example_1d.svg")


# Now we'll set up the main BO object. First we need to create a surrogate model.
# We'll use a GPy GP surrogate with an RBF kernel. See the GPyGPSurrogate class for
# why we have to create this slightly funky initializer function first.
def gp_initializer(x, y):
    return GPy.models.GPRegression(
        x, y, kernel=GPy.kern.RBF(input_dim=1), noise_var=1e-10, normalizer=True
    )


surrogate = GPyGPSurrogate(gp_initializer=gp_initializer)
acquistion_function = LCB(surrogate=surrogate)
bounds = Bounds(bounds=[Bound(lower=0.0, upper=1.0)])
optimizer = DirectOptimizer(
    acquisition_function=acquistion_function, bounds=bounds, maxf=100
)

# Now we create the BO object...
bo = BayesOpt(
    objective_function=forrester,
    surrogate=surrogate,
    acquisition_function=acquistion_function,
    optimizer=optimizer,
    bounds=bounds,
    initial_design=SobolSequenceInitialDesign(),
    callbacks=[PlottingCallback()],
)

# ...and run!
bo.run(n_trials=n_trials, n_initial_design=n_inital_design)
