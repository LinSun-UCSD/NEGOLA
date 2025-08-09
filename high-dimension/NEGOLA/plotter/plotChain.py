import matplotlib.pyplot as plt
import numpy as np
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 12
def plotChain(chains, figSize, ylim, n_dim):
    n_chains = chains.shape[0]
    initial_position = chains[:, 0]
    if n_dim == 1:
        plt.figure(figsize=figSize)
        for i in range(n_chains):
            plt.subplot(2, 5, i + 1)
            plt.scatter(0, initial_position[i, 0], marker='*', c='r',zorder=chains.shape[1])
            plt.plot(chains[i, :, 0], "-o", alpha=0.5, ms=2)
            plt.xlim((0, chains.shape[1]))
            plt.ylim((ylim[0], ylim[1]))
            plt.title("Chain " + str(i + 1))
            plt.grid("on")
            plt.xlabel("Iteration")
            plt.ylabel("x1")
    else:
        plt.figure(figsize=figSize)
        for i in range(n_chains):
            plt.subplot(4, 5, i + 1)
            plt.scatter(initial_position[i, 0], initial_position[i, 1], marker='*', c='r', zorder=chains.shape[1])
            plt.plot(chains[i, :, 0], chains[i,:, 1], "-o", alpha=0.5, ms=2)
            # plt.xlim((0, chains.shape[1]))
            # plt.ylim((ylim[0], ylim[1]))
            plt.title("Chain " + str(i + 1))
            plt.grid("on")
            plt.xlabel("Iteration")
            plt.ylabel("x1")
    plt.tight_layout()
