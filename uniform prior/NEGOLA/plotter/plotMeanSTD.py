import numpy as np
import matplotlib.pyplot as plt
import pickle


def plotMeanSTD(total, path, burn_in, n_dim, figsize):
    mean = []
    std = []
    cov = []
    for index in range(total):
        chains = np.load(path + "chains" + str(index) + ".npy")
        temp = chains[:, burn_in:-1, :]
        samples = temp.reshape(-1, n_dim)

        stageMean = np.mean(samples, axis=0)
        mean.append(stageMean)
        stageSTD = np.std(samples, axis=0)
        std.append(stageSTD)
        cov.append(stageMean / stageSTD)

    temp1 = np.array(mean) - np.array(std)
    temp2 = np.array(mean) + np.array(std)
    plt.figure(figsize=(figsize[0], figsize[1]))
    plt.rcParams["font.family"] = "times new roman"
    plt.rcParams["font.size"] = 12

    for i in range(n_dim):
        plt.subplot(n_dim,1,i+1)
        upper = temp1[:, i]
        lower = temp2[:, i]
        plt.fill_between(range(total), lower, upper, color=[0.75, 0.75, 0.75])
        plt.plot(range(total), np.array(mean)[:, i], color="b")
        plt.grid("auto")
        plt.xlabel("Time [sec]")
        # plt.plot(np.arange(total), 0.2 * np.exp(-np.arange(total) / 2) + 0.8, color='r')
        plt.axhline(y=1, color='r', linestyle='--', lw=1.5)

    plt.tight_layout()
    return mean, std, cov
