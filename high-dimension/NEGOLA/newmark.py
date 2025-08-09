import jax.numpy as jnp
import jax
from jax.scipy.linalg import solve
import numpy as np
import os
import matplotlib.pyplot as plt
def get_mass(m0, DOF):
    return m0 * np.identity(DOF)

def get_classical_damping(K, M, damping):
    eigvals = jnp.linalg.eigvalsh(jnp.dot(jnp.linalg.inv(M), K))
    omega = jnp.sort(jnp.sqrt(jnp.real(eigvals)))
    mode = damping["mode"]
    damping_ratio = damping["ratio"]

    temp = jnp.array([[1 / omega[mode[0] - 1], omega[mode[0] - 1]],
                      [1 / omega[mode[1] - 1], omega[mode[1] - 1]]])

    a = 2 * jnp.matmul(jnp.linalg.inv(temp), jnp.transpose(jnp.array([[damping_ratio[0], damping_ratio[1]]])))

    C = a[0] * M + a[1] * K

    damping_mode = a[0] / 2 * jnp.true_divide(1, omega) + a[1] / 2 * omega
    return C

def getModalData(K, M):
    DOF = K.shape[0]

    # Compute eigenvalues and eigenvectors
    values, vectors = jnp.linalg.eig(jnp.dot(jnp.linalg.inv(M), K))
    # values = values.reshape((1, len(values)))
    values = jnp.real(values).reshape((1, len(values)))
    vectors = jnp.real(vectors)
    # Initialize the result array
    y = jnp.zeros((vectors.shape[0] + values.shape[0], vectors.shape[1]))
    y = y.at[0, :].set(values[0, :])  # Set eigenvalues in the first row

    # Max eigenvalue
    maxEigen = jnp.max(y[0, :])

    # Add eigenvectors to the result array
    y = y.at[1:y.shape[0], :].set(vectors)

    # Compute norms of eigenvectors
    maxEigenVec = jnp.zeros((y.shape[1],))
    for i in range(y.shape[1]):
        norm = jnp.linalg.norm(y[1:y.shape[0], i])
        maxEigenVec = maxEigenVec.at[i].set(norm)

    return y, maxEigen, maxEigenVec


def newmark(K, M, C, f, record, beta, gamma, fs_original, u0, v0, a0):


    # Check that beta and gamma are within acceptable ranges
    if beta > 1/2 or beta < 1/4:
        raise ValueError("Beta is not in the appropriate range")
    if gamma != 1/2:
        raise ValueError("Gamma is not in the appropriate range")

    # Resample `f` for each degree of freedom (resampling not available in JAX, so we'll assume `f` is pre-resampled)


    # Initialize parameters
    dof = K.shape[0]
    samples = f.shape[1]
    dt = 1 / fs_original

    # Step 1: Compute effective stiffness parameters
    a1 = 1 / (beta * dt ** 2) * M + gamma / (beta * dt) * C
    a2 = (1 / (beta * dt)) * M + (gamma / beta - 1) * C
    a3 = (1 / (2 * beta) - 1) * M + dt / 2 * (gamma / beta - 2) * C

    Khat = a1 + K

    # Initialize response arrays
    f_hat = jnp.zeros((dof, samples))
    u = jnp.zeros((dof, samples))
    v = jnp.zeros((dof, samples))
    a_rel = jnp.zeros((dof, samples))
    a_abs = jnp.zeros((dof, samples))
    # Set initial conditions
    u = u.at[:, 0].set(u0)
    v = v.at[:, 0].set(v0)
    a_rel = a_rel.at[:, 0].set(a0)
    a_abs = a_abs.at[:, 0].set(a0+record[0])

    # Time-stepping loop
    for i in range(samples - 1):
        f_hat = f_hat.at[:, i + 1].set(f[:, i + 1] + a1 @ u[:, i] + a2 @ v[:, i] + a3 @ a_rel[:, i])
        u = u.at[:, i + 1].set(solve(Khat, f_hat[:, i + 1]))  # Solves for u(:, i+1)
        a_rel = a_rel.at[:, i + 1].set(1 / (beta * dt ** 2) * (u[:, i + 1] - u[:, i] - dt * v[:, i] - (1 / 2 - beta) * dt ** 2 * a_rel[:, i]))
        v = v.at[:, i + 1].set(v[:, i] + dt * ((1 - gamma) * a_rel[:, i] + gamma * a_rel[:, i + 1]))
        a_abs = a_abs.at[:,i+1].set(a_rel[:,i+1] + record[i+1])

    return jnp.transpose(a_rel), jnp.transpose(v), jnp.transpose(u), jnp.transpose(a_abs)

def get_stiffness(k, DOF, measure_vector, k0):
    temp = jax.numpy.zeros((2, 2, DOF))
    for i in range(len(k)):
        temp = temp.at[:, :, i].set(jax.numpy.array([[k[i], -k[i]], [-k[i], k[i]]]).reshape((2, 2)))
    K_global = jax.numpy.zeros((DOF, DOF))
    for i in range(DOF-1):
        K_global = K_global.at[i:i + 2, i:i + 2].add(temp[:, :, i + 1])
    K_global = K_global.at[0, 0].add(temp[0, 0, 0])
    return K_global


