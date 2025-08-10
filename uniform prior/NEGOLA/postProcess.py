import numpy as np
import matplotlib.pyplot as plt
from plotter.plotPairPlot import plotPairPlot
from plotter.plotChain import plotChain
from plotter.plotAcceptanceRate import plotAcceptanceRate
from plotter.plotR import plotR
from plotter.plotLogProb import plotLogProb
from plotter.plotTimeHistory import plotTimeHistory
from plotter.plotKDE import plotKDE
from plotter.plotMeanSTD import plotMeanSTD

path = "results/"
# load the data

index = 0
chains = np.load(path + "chains" + str(index) + ".npy")
global_accs = np.load(path + "global_accs" + str(index) + ".npy")
local_accs = np.load(path + "local_accs" + str(index) + ".npy")
loss_vals = np.load(path + "loss_vals" + str(index) + ".npy")
log_prob = np.load(path + "log_prob" + str(index) + ".npy")
# u = np.load("results/SDOF/ug.npy")
# TrueResponse = np.load(path + "TrueResponse.npy")
# NoisyTrueResponse = np.load(path + "NoisyTrueResponse.npy")
fs = 50
n_dim = 4

# plot time history data
# figSize = (6,4)
# plotTimeHistory(u, TrueResponse, NoisyTrueResponse, fs, figSize)
# plt.savefig("1.png")
#
# plot chains
figSize = (12, 5)
ylim = [0, 2.5]
plotChain(chains, figSize, ylim, n_dim)
plt.savefig("2.png")
#
# plot pair plot
burn_in = 3000

figSize = (5, 4)
labels = ['x' + str(i) for i in range(n_dim)]
# plotPairPlot(chains, burn_in, n_dim, labels, figSize)
# plt.savefig("3.png")

# plot acceptance rate
# plotAcceptanceRate(loss_vals, global_accs, local_accs)
# plt.savefig("4.png")
# plot R^2
# n_division = 8
# plotR(chains, n_division, labels)
# plt.savefig("5.png")

# plot log prob
# plotLogProb(log_prob,figsize=(12,5))
# plt.savefig("6.png")



figsize = (10, 3)
mean = []
std = []
cov = []
chains = np.load(path + "chains" + str(0) + ".npy")
# temp = chains[:, burn_in:-1, :]
total = range(40, 4000, 1)
burn_in = 1
n_chains = 20
for index in total:
    index = index + burn_in
    temp1 = chains[:, index-40:index, :]
    samples = temp1.reshape(-1, n_dim)

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
    plt.subplot(1, 4, i + 1)
    upper = temp1[:, i]
    lower = temp2[:, i]
    plt.fill_between((np.arange(len(total))), lower, upper,
                     color=[0.75, 0.75, 0.75])
    plt.plot((np.arange(len(total))) , np.array(mean)[:, i],
             color="b")
    plt.grid("auto")
    plt.xlim((160, 4000))
    plt.hlines(1, xmin=000, xmax=80000, color='r')
    # plt.ylim((0.5, 2.5))
    ax = plt.gca()
    # ax.tick_params(axis='x', which='both', direction='in', top=True, labeltop=True)
    # ax.set_xticklabels([" " for ii in range(9)])
    plt.xlabel("Iteration")
    plt.tight_layout()

plt.savefig("mean&std.svg", dpi=800)
plt.show()