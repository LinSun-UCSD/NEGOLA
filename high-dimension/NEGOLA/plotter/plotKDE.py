import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.cm as cm
from scipy.stats import gaussian_kde


def plotKDE(total, burn_in, n_dim, labels, figSize):
    # Remove burn-in period

    path = "results/"
    # Set up the plot grid
    fig, axes = plt.subplots(n_dim, 1, figsize=figSize, constrained_layout=True)
    if n_dim == 1:
        axes = [axes]  # Ensure axes is always iterable
    cmap = cm.get_cmap('tab10', total)
    # Iterate over each dimension
    for index in range(total):
        chains = np.load(path + "chains" + str(index) + ".npy")
        temp = chains[:, burn_in:-1, :]
        samples = temp.reshape(-1, n_dim)
        for dim in range(n_dim):
            ax = axes[dim]
            kde = gaussian_kde(samples[:, dim])
            x_vals = np.linspace(samples[:, dim].min()*0.8, samples[:, dim].max()*1.2, 100)  # Evaluation grid
            y_vals = kde(x_vals)  # KDE values

            # Use matplotlib to plot
            color = cmap(index)  # Get unique color
            ax.plot(x_vals, y_vals, color=color)

            ax.set_title(f'KDE for {labels[dim]}', fontsize=12)
            ax.set_xlabel(labels[dim])
            ax.set_ylabel("PDF")
            # ax.set_ylim(bottom=0)
            ax.grid("auto")
            # ax.legend(fontsize=10)