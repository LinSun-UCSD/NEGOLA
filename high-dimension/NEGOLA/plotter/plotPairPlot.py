import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


def plotPairPlot(chains, burn_in, n_dim, labels, figSize):
    temp = chains[:, burn_in:-1, :]
    samples = temp.reshape(-1, n_dim)
    print("mean is" )
    print(np.mean(samples, axis=0))
    print("Co-variance matrix is")
    print(np.cov(np.transpose(samples)))
    df = pd.DataFrame(samples, columns=labels)
    g = sns.pairplot(df)

    g.fig.set_size_inches(figSize[0], figSize[1])
    # font_size = 12  # Set your desired font size
    # for ax in g.axes.flat:  # Loop through all the axes in the PairPlot
    #     ax.set_xticklabels(ax.get_xticklabels(), fontsize=font_size)
    #     ax.set_yticklabels(ax.get_yticklabels(), fontsize=font_size)