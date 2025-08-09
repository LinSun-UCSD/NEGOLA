import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jrand
import matplotlib.pyplot as plt
import corner
from h_measurement_eqn.normalize import normalize
import os as os
from newmark import get_mass, get_stiffness, get_classical_damping, newmark
from flowMC.proposal.MALA import MALA
from flowMC.proposal.Gaussian_random_walk import GaussianRandomWalk
from flowMC.nfmodel.rqSpline import MaskedCouplingRQSpline
import seaborn as sns
import time
start_time = time.time()
from flowMC.Sampler import Sampler

gamma = 1/2
beta = 1/4
DOF = 20
n_dim = DOF

measure_vector = np.array([[0, 1]])
damping = {
    "mode": np.array([1, 2]),
    "ratio": np.array([0.05, 0.05])
}

m0 = 1
M = get_mass(m0, DOF)

g = 9.81
GMinput = {
    "totalStep": 500,  # earthquake record step
    "fs": 50,  # sampling rate
    "filename": 'NORTHR_SYL090',  # the earthquake file to load
    "path": os.getcwd() + "\\earthquake record"  # earthquake record folder
}
temp = np.loadtxt(GMinput["path"] + "/" + GMinput["filename"] + ".txt", dtype=float)
temp = temp[2::] * 9.81
record = temp
B = jnp.ones((DOF,1))
f = -jnp.matmul(M, B)*record

# accel_rel, velocity, displacement, accel_abs = newmark(K, M, C, f, record, beta, gamma, GMinput["fs"], u0, v0, a0)
# TrueResponse = accel_rel
# TrueResponse, max = normalize(TrueResponse)
#
# key = jrand.PRNGKey(0)  # Initialize a random key
# noise = jrand.normal(key, shape=(TrueResponse.shape[0], DOF)) * 0.05
# NoisyTrueResponse = TrueResponse + noise
stepSize = 500
split = np.arange(stepSize, 2001, stepSize)


for ii in range(1):
    # define the inital position of each Markov chain
    n_chains = 20
    rng_key, subkey = jax.random.split(jax.random.PRNGKey(42))

    # define the ground motion input
    end = split[ii]
    start = end - stepSize
    ug = record[start:end]

    if ii == 0:
        u0 = jnp.zeros(DOF)  # Initial displacement
        v0 = jnp.zeros(DOF)  # Initial velocity
        a0 = jnp.zeros(DOF)
    else:
        u0 = displacement[start,:]
        v0 = velocity[start,:]
        a0 = accel_rel[start,:]
    # compute true response
    if ii >= 3:
        k = 0.8
    else:
        k = 1
    K = get_stiffness(np.ones((DOF, 1)) * k, DOF, measure_vector, k)
    C = get_classical_damping(K, M, damping)
    accel_rel, velocity, displacement, accel_abs = newmark(K, M, C, -jnp.matmul(M, B)*ug, ug, beta, gamma, GMinput["fs"], u0, v0, a0)
    TrueResponse = accel_abs
    TrueResponse, max = normalize(TrueResponse)

    key = jrand.PRNGKey(0)  # Initialize a random key
    noise = jrand.normal(key, shape=(TrueResponse.shape[0], DOF)) * 0.05
    NoisyTrueResponse = TrueResponse + noise


    def target_log_prob(theta, data=None):
        K = get_stiffness(theta, DOF, jnp.array([[0, 1]]), k)
        C = get_classical_damping(K, M, damping)
        _, _, _, y = newmark(K, M, C, -jnp.matmul(M, B)*ug, ug, beta, gamma, GMinput["fs"], u0, v0, a0)
        for i in range(y.shape[1]):
            y = y.at[:, i].set(y[:, i] / max[i])
        N, Ny = y.shape

        delta = NoisyTrueResponse[:,:] - y

        par_sigma_normalized = jnp.array([0.05] * Ny)
        LL = (
                -0.5 * N * Ny * jnp.log(2 * jnp.pi)
                - jnp.sum(N * jnp.log(par_sigma_normalized))
                - jnp.sum(0.5 * (par_sigma_normalized ** -2) * jnp.sum(delta ** 2, axis=0))
        )

        return LL



    # define the samples we want
    n_loop_training = 10
    n_loop_production = 10
    n_local_steps = 300
    n_global_steps = 300

    # define global sampler
    learning_rate = 0.001
    momentum = 0.9
    batch_size = 5000
    max_samples = 5000
    num_epochs = 5

    # define local sampler
    n_layers = 5
    hidden_size = [32, 32]
    num_bins = 8
    step_size = 0.01
    data = jnp.zeros(n_dim)
    rng_key, subkey = jax.random.split(rng_key)
    model = MaskedCouplingRQSpline(
        n_dim, n_layers, hidden_size, num_bins, subkey
    )
    # MALA_Sampler = MALA(target_dual_moon, True, step_size=step_size)
    MALA_Sampler = GaussianRandomWalk(target_log_prob, True, step_size=step_size)
    rng_key, subkey = jax.random.split(rng_key)
    nf_sampler = Sampler(
        n_dim,
        subkey,
        {'data': data},
        MALA_Sampler,
        model,
        n_loop_training=n_loop_training,
        n_loop_production=n_loop_production,
        n_local_steps=n_local_steps,
        n_global_steps=n_global_steps,
        n_chains=n_chains,
        n_epochs=num_epochs,
        learning_rate=learning_rate,
        momentum=momentum,
        batch_size=batch_size,
        use_global=True,
    )
    if ii == 0:
        initial_position = jax.random.normal(subkey, shape=(n_chains, n_dim)) * 0.2 + k*2
    else:
        temp = np.load("results/chains" + str(ii-1) + ".npy")
        initial_position = jax.numpy.array(temp[:, -1, :])

    print(initial_position)
    nf_sampler.sample(initial_position, data={'data':data})
    # save the flow
    nf_sampler.save_flow("best_model" + str(ii))
    out_train = nf_sampler.get_sampler_state(training=True)
    print("Logged during tuning:", out_train.keys())
    chains = np.array(out_train["chains"])
    global_accs = np.array(out_train["global_accs"])
    local_accs = np.array(out_train["local_accs"])
    loss_vals = np.array(out_train["loss_vals"])
    log_prob = np.array(out_train["log_prob"])
    np.save("results/chains" + str(ii) + ".npy", chains)
    np.save("results/global_accs" + str(ii) + ".npy", global_accs)
    np.save("results/local_accs" + str(ii) + ".npy", local_accs)
    np.save("results/loss_vals" + str(ii) + ".npy", loss_vals)
    np.save("results/log_prob" + str(ii) + ".npy", log_prob)
# rng_key, subkey = jax.random.split(rng_key)
# nf_samples = np.array(nf_sampler.sample_flow(subkey, 3000))
#
#
# plt.figure(figsize=(6, 6))
# plt.title("NF loss")
# plt.plot(loss_vals.reshape(-1))
# plt.xlabel("iteration")
#
# plt.figure()
# plt.title("Local Acceptance")
# plt.plot(local_accs.mean(0))
# plt.xlabel("iteration")
#
# plt.figure()
# plt.title("Global Acceptance")
# plt.plot(global_accs.mean(0))
# plt.xlabel("iteration")
# plt.tight_layout()
# plt.show(block=False)
#
#
# Plot all chains
# samples = chains.reshape(-1, n_dim)
# print(np.mean(samples,axis=0))
# print(np.sqrt(np.var(samples, axis=0)) / np.mean(samples, axis=0))
#
# figure = corner.corner(chains.reshape(-1, n_dim))
# figure.set_size_inches(7, 7)
# figure.suptitle("Visualize samples")
# plt.show()
# #
# # Plot Nf samples
# plt.figure()
# figure = corner.corner(nf_samples)
# figure.set_size_inches(7, 7)
# figure.suptitle("Visualize NF samples")
# plt.show()
