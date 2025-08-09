import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import scipy.stats as stats
import numpy as np


def plotBoxplot(mytrace, thetaChoice, stageNum, trueValues, thetaName, parameterBound, isSimulated, figureSize):
    plt.rcParams["font.family"] = "times new roman"
    plt.rcParams["font.size"] = 10
    plt.figure(figsize=(figureSize[0], figureSize[1]))
    for i in range(len(thetaChoice)):
        plt.subplot(3, 3, i + 1)
        data = []
        for stage in stageNum:
            Sm = mytrace[stage][0][:, thetaChoice[i]]
            data.append(Sm)
        flierprops = dict(marker='.', markerfacecolor='k', markersize=4,
                          markeredgecolor='none')
        meanpointprops = dict(marker='o', markeredgecolor='green',
                              markerfacecolor='green', markersize=3)
        bp = plt.boxplot(data, whis=[1, 99], showmeans=True, meanprops=meanpointprops, flierprops=flierprops, patch_artist=True)
        for box in bp["boxes"]:
            box.set(facecolor="gray", alpha=0.2)
        if isSimulated:
            plt.axhline(y=trueValues[i], color='r', linestyle='-', linewidth=1.2)
        if i < parameterBound.shape[0]:
            plt.axhline(y=parameterBound[thetaChoice[i], 0], color='green', linestyle='--', linewidth=1.2)
            plt.axhline(y=parameterBound[thetaChoice[i], 1], color='green', linestyle='--', linewidth=1.2)
        plt.xlabel("Stage")
        plt.ylabel(thetaName[i])
    plt.tight_layout()
