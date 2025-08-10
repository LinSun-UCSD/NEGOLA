import matplotlib.pyplot as plt
import numpy as np

def plotLogProb(log_prob, figsize):
    n_chains = log_prob.shape[0]
    plt.figure(figsize=figsize)
    for i in range(n_chains):
        plt.subplot(4, 5, i + 1)
        plt.plot(log_prob[i, :])
        plt.grid("on")
        plt.xlabel("Samples")
        plt.ylabel("log probability")
    plt.tight_layout()
