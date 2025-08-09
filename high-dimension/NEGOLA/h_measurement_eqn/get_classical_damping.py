import numpy as np
import jax.numpy as jnp
import scipy.linalg as la
import jax
from jax import lax
# this is to get the damping matrix
def get_classical_damping(K, M, damping, plot):
    # values = la.eigvals(K, M)
    # temp = jnp.linalg.inv(M) * K
    # values = jnp.linalg.eig(temp)
    eigvals = jnp.linalg.eigvalsh(jnp.dot(jnp.linalg.inv(M), K))
    omega = jnp.sqrt(jnp.real(eigvals))

    # omega = np.sort(np.sqrt(np.real(values)))
    mode = damping["mode"]
    damping_ratio = damping["ratio"]
    # temp = np.array([[1 / omega[mode[0] - 1], omega[mode[0] - 1]], [1 / omega[mode[1] - 1], omega[mode[1] - 1]]])
    # a = 2 * np.matmul(np.linalg.inv(temp), np.transpose(np.array([[damping_ratio[0], damping_ratio[1]]])))
    # C = a[0] * M + a[1] * K  # get the classical damping
    # damping_mode = a[0] / 2 * np.true_divide(1, omega) + a[1] / 2 * omega

    temp = jnp.array([[1 / omega[mode[0] - 1], omega[mode[0] - 1]],
                      [1 / omega[mode[1] - 1], omega[mode[1] - 1]]])
    a = 2 * jnp.matmul(jnp.linalg.inv(temp), jnp.transpose(jnp.array([[damping_ratio[0], damping_ratio[1]]])))

    # Use JAX for matrix operations on `a`, `M`, and `K`
    C = a[0] * M + a[1] * K  # Calculate classical damping matrix

    # Use JAX's true divide with omega
    damping_mode = a[0] / 2 * jnp.true_divide(1, omega) + a[1] / 2 * omega
    return C, omega, a, damping_mode
