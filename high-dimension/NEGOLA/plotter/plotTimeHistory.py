import matplotlib.pyplot as plt
import numpy as np

def plotTimeHistory(u, TrueResponse, NoisyTrueResponse, fs, figSize):
    g = 9.81
    u = u[2:-1]
    plt.figure(figsize=figSize)
    plt.subplot(2,1,1)
    t = np.arange(u.shape[0])/fs
    plt.plot(t,  u)
    plt.xlim((t[0], t[-1]))
    plt.xlabel("Time [sec]")
    plt.ylabel("Accel. [g]")
    plt.grid("on")
    plt.title("Ground Motion")

    plt.subplot(2,1,2)
    t = np.arange(TrueResponse.shape[0])/fs
    plt.plot(t, TrueResponse/g, label="True Response")
    plt.plot(t, NoisyTrueResponse/g, label="Polluted Response")
    plt.xlim((t[0], t[-1]))
    plt.xlabel("Time [sec]")
    plt.ylabel("Accel. [g]")
    plt.title("Response")
    plt.legend()
    plt.grid("on")
    plt.tight_layout()