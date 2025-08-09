import matplotlib.pyplot as plt

def plotAcceptanceRate(loss_vals, local_accs, global_accs):
    plt.figure(figsize=(6, 6))
    plt.subplot(3, 1, 1)
    plt.title("NF loss")
    temp = loss_vals.reshape(-1)
    plt.plot(temp)
    plt.xlim((0, len(temp)))
    plt.grid("on")
    plt.xlabel("iteration")

    plt.subplot(3, 1, 2)
    plt.title("Local Acceptance")
    plt.plot(local_accs.mean(0))
    plt.xlim((0, local_accs.shape[1]))
    plt.ylim((0, 1.1))
    plt.grid("on")
    plt.xlabel("iteration")

    plt.subplot(3, 1, 3)
    plt.title("Global Acceptance")
    plt.plot(global_accs.mean(0))
    plt.xlim((0, global_accs.shape[1]))
    plt.xlabel("iteration")
    plt.ylim((0,1.1))
    plt.grid("on")
    plt.tight_layout()

