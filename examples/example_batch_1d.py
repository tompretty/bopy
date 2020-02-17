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

n_inital_design = 5
n_trials = 3
batch_size = 2


class PlottingCallback(Callback):
    def __init__(self):
        super().__init__()
        self.fig, self.axs = plt.subplots(n_trials, 2, figsize=(10, 16))
        self.current_trial = 0

    def on_acquisition_optimized(self, bo, opt_result):
        ax = self.axs[self.current_trial][0]

        plot_surrogate_1D(ax, bo.surrogate, bo.x, bo.y, bo.bounds.bounds[0])
        ax.plot(
            opt_result.x_min,
            np.zeros(batch_size),
            "r*",
            markersize=9,
            zorder=100,
            label="acquisition min",
        )

    def on_surrogate_updated(self, bo):
        ax = self.axs[self.current_trial][1]

        plot_surrogate_1D(ax, bo.surrogate, bo.x, bo.y, bo.bounds.bounds[0])

    def on_trial_end(self, _):
        self.current_trial += 1

    def on_bo_end(self, _):
        self.fig.savefig("example_batch_1d.svg")


def gp_initializer(x, y):
    return GPy.models.GPRegression(
        x, y, kernel=GPy.kern.RBF(input_dim=1), noise_var=1e-10, normalizer=True
    )


surrogate = GPyGPSurrogate(gp_initializer=gp_initializer)

base_acquisition = LCB(surrogate=surrogate)
acquisition_function = KriggingBeliever(
    surrogate=surrogate, base_acquisition=base_acquisition,
)

bounds = Bounds(bounds=[Bound(lower=0.0, upper=1.0)])

base_optimizer = DirectOptimizer(
    acquisition_function=base_acquisition, bounds=bounds, maxf=100
)
optimizer = SequentialBatchOptimizer(
    base_optimizer=base_optimizer,
    batch_size=2,
    acquisition_function=acquisition_function,
    bounds=bounds,
)

bo = BayesOpt(
    objective_function=forrester,
    surrogate=surrogate,
    acquisition_function=acquisition_function,
    optimizer=optimizer,
    initial_design=SobolSequenceInitialDesign(),
    bounds=bounds,
    callbacks=[PlottingCallback()],
)

bo.run(n_trials=3, n_initial_design=5)
