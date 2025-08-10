import _pickle as cPickle
import matplotlib.pyplot as plt
import pickle
import numpy as np
import seaborn as sns
import pandas as pd
from seaborn import scatterplot
from plotter.plotScatterTwoTheta import plotScatterTwoTheta
from plotter.plotMeanSTD import plotMeanSTD
from plotter.plotPairGrid import plotPairGrid
from plotter.plotRRMS import plotRRMS
from plotter.plotData import plotData
from plotter.createAnimation import createAnimation
from plotter.plotDistributionOfTheta import plotDistributionOfTheta
from h_measurement_eqn.h_measurement_eqn import h_measurement_eqn
import os
from tmcmc_mod import pdfs

# load trace
with open('mytrace.pickle', 'rb') as handle1:
    mytrace = pickle.load(handle1)
# mytrace stage m:
# samples, likelihood, weights, next stage ESS, next stage beta, resampled samples

DOF = 4
trueValues = [1] * DOF
stages = np.arange(0, 27)
thetaName = [f"k{i}" for i in range(1, DOF+1)]
# plotScatterTwoTheta(pickleFileName, trueValues, stages, labelsName=["k1", "k2"])
# createAnimation(mytrace, trueValues, stages, thetaName, [0, 1, 2])
# thetaChoice = range(0,9)
# thetaName = thetaName[thetaChoice]
# plotPairGrid(mytrace, stages, thetaChoice, thetaName)
# plt.savefig("pairgrid.png", dpi=800)
# plt.show()

# prior_mean = np.array([[1.14, 1.32, 0.35, 0.95, 0.87, 1.235, 0.935, 0.85]])
# figsize = (5,3)
# plotDistributionOfTheta(mytrace, thetaName, 0, trueValues, figsize, rows=1, cols=1)
# plt.savefig("prior.png", dpi=800)
# plt.show()

figsize = (10,3)
mean, std, cov = plotMeanSTD(mytrace, np.arange(0, DOF), np.arange(0, len(mytrace)), trueValues,
                             thetaName, figsize, rows=1, cols=4)
plt.savefig("mean&std.svg", dpi=800)
plt.show()

# plot RRMS
GMinput = {
    "totalStep": 500,  # earthquake record stpes
    "fs": 50,  # sampling rate
    "filename": 'NORTHR_SYL090',  # the earthquake file to load
    "path": os.getcwd() + "\\earthquake record"  # earthquake record folder
}
measure_vector = np.array([[0]])
k0 = 1
TrueResponse = np.load("TrueResponse.npy")
stage = np.arange(0,12)
trueValues = np.ones((DOF, 1), ) * k0
samples = np.arange(0, 500)
# pickleFileName = "mytrace.pickle"
# plt.rcParams["font.family"] = "Times New Roman"
# plt.rcParams["font.size"] = 12
# rrms = plotRRMS(pickleFileName, stage,samples, trueValues, h_measurement_eqn, measure_vector, k0, GMinput)
# temp = []
# for i in range(len(rrms)):
#     temp.append(rrms[i][:, 0]*100)
# plt.figure(figsize=(5,3))
# plt.boxplot(temp)
# plt.xlabel("Stage")
# plt.ylabel("RRMS [%]")
# plt.grid("on")
# plt.savefig("1.png")
plt.show()
