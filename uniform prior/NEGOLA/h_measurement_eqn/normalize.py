# import numpy as np
#
#
# def normalize(y):
#     yNorm = np.zeros(y.shape)
#     max = np.amax(np.abs(y), axis=0)
#     for i in range(y.shape[1]):
#         yNorm[:, i] = y[:, i]/max[i]
#     return yNorm, max
import jax.numpy as jnp

def normalize(y):
    # Calculate the max values along each column
    max_vals = jnp.amax(jnp.abs(y), axis=0)

    # Normalize y by max_vals using broadcasting
    yNorm = y / max_vals

    return yNorm, max_vals