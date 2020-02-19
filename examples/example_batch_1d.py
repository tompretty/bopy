import GPy
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from bopy.acquisition import LCB, KriggingBeliever
from bopy.bayes_opt import BayesOpt
from bopy.benchmark_functions import forrester
from bopy.bounds import Bound, Bounds
from bopy.callback import Callback
from bopy.initial_design import SobolSequenceInitialDesign
from bopy.optimizer import DirectOptimizer, SequentialBatchOptimizer
from bopy.plotting import plot_surrogate_1D
from bopy.surrogate import GPyGPSurrogate

sns.set_style("darkgrid")

# Let's set up some constants
n_initial_design = 5
n_trials = 3
batch_size = 2

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
        # Here we create a plot of the surrogate and the batch points selected.
        # We position it in first column of the `current_trial`th row.
        ax = self.axs[self.current_trial][0]

        plot_surrogate_1D(ax, bo.surrogate, bo.bounds.bounds[0])
        ax.plot(
            opt_result.x_min,
            np.zeros(batch_size),
            "r*",
            markersize=9,
            zorder=100,
            label="acquisition min",
        )

    def on_surrogate_updated(self, bo):
        # This is called during a BO trial, just after the surrogate has been fit to the new datum.
        # Here we create a plot of just the surrogate. We position it in the second column of the
        # `current_trial`th row.
        ax = self.axs[self.current_trial][1]

        plot_surrogate_1D(ax, bo.surrogate, bo.bounds.bounds[0])

    def on_trial_end(self, _):
        # This is called at the end of a BO trial. We just increment our counter keeping track
        # of the current trial.
        self.current_trial += 1

    def on_bo_end(self, _):
        # This is called right at the end of the BO routine. We save the figure to a file.
        self.fig.savefig("example_batch_1d.svg")


# Now we'll set up the main BO object. First we need to create a surrogate model.
# We'll use a GPyGPSurrogate with an RBF kernel. See the GPyGPSurrogate class for
# why we have to create this slightly funky initializer function first.
def gp_initializer(x, y):
    return GPy.models.GPRegression(
        x, y, kernel=GPy.kern.RBF(input_dim=1), noise_var=1e-10, normalizer=True
    )


surrogate = GPyGPSurrogate(gp_initializer=gp_initializer)

# We'll use LCB as the base acquisition function, along with
# the krigging believer strategy for sequentially constructing
# a batch of points to evaluate.
base_acquisition = LCB(surrogate=surrogate)
acquisition_function = KriggingBeliever(
    surrogate=surrogate, base_acquisition=base_acquisition,
)

bounds = Bounds(bounds=[Bound(lower=0.0, upper=1.0)])

# We'll use the DirectOptimizer as the base optimizer, along with
# the SequentialBatchOptimizer for sequentially constructing
# a batch of points to evaluate.
base_optimizer = DirectOptimizer(
    acquisition_function=acquisition_function, bounds=bounds, maxf=100
)
optimizer = SequentialBatchOptimizer(
    base_optimizer=base_optimizer,
    batch_size=batch_size,
    acquisition_function=acquisition_function,
    bounds=bounds,
)

# Now we create the BO object...
bo = BayesOpt(
    objective_function=forrester,
    surrogate=surrogate,
    acquisition_function=acquisition_function,
    optimizer=optimizer,
    initial_design=SobolSequenceInitialDesign(),
    bounds=bounds,
    callbacks=[PlottingCallback()],
)

# ...and run!
bo.run(n_trials=n_trials, n_initial_design=n_initial_design)
