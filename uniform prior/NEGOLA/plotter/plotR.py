import arviz as az
from arviz.utils import get_coords, _var_names
import matplotlib.pyplot as plt
import numpy as np



def plotR(chains, n_split, labels):
    ## Load data as arviz InferenceData class
    idata = az.convert_to_inference_data(chains)
    coords = {}
    data = get_coords(az.convert_to_dataset(idata, group="posterior"), coords)
    var_names = None
    filter_vars = None
    var_names = _var_names(var_names, data, filter_vars)
    n_draws = data.dims["draw"]
    n_samples = n_draws * data.dims["chain"]
    first_draw = data.draw.values[0]  # int of where where things should start

    ## Compute where to split the data to diagnostic the convergence

    xdata = np.linspace(n_samples / n_split, n_samples, n_split)
    draw_divisions = np.linspace(n_draws // n_split, n_draws, n_split, dtype=int)

    rhat_s = np.stack(
        [
            np.array(
                az.rhat(
                    data.sel(draw=slice(first_draw + draw_div)),
                    var_names=var_names,
                    method="rank",
                )["x"]
            )
            for draw_div in draw_divisions
        ]
    )

    plt.figure()

    plt.plot(draw_divisions, rhat_s, "-o", label=labels)
    plt.axhline(1, c="k", ls="--")
    plt.xlabel("Iteration")
    plt.ylabel(r"$\hat{R}$");
    plt.grid("on")
    plt.legend()