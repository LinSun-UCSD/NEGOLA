import numpy as np
import jax
import jax.numpy as jnp
# this is calculate the stiffness matrix of a 2D shear building
def get_stiffness(k, DOF, measure_vector, k0):
    if k.shape[0] == DOF:
        # temp = np.zeros((2, 2, k.shape[0]))
        temp = jax.numpy.zeros((2, 2, k.shape[0]))
        for i in range(len(k)):

            # temp[:, :, i] = np.array([[k[i], -k[i]], [-k[i], k[i]]]).reshape((2,2))
            temp = temp.at[:, :, i].set(jax.numpy.array([[k[i], -k[i]], [-k[i], k[i]]]).reshape((2, 2)))
    elif k.shape[0] < DOF:
        temp_k = jax.numpy.ones((DOF,))*k0
        for i in range(len(measure_vector[0])):
            temp_k[measure_vector[0,i]] = k[i]
        temp = np.zeros((2, 2, temp_k.shape[0]))
        for i in range(len(temp_k)):
            temp[:, :, i] = np.array([[temp_k[i], -temp_k[i]], [-temp_k[i], temp_k[i]]]).reshape((2, 2))

    # K_global = np.zeros((DOF, DOF))
    K_global = jax.numpy.zeros((DOF, DOF))
    for i in range(DOF-1):
        # K_global[i:i+2, i:i+2] = K_global[i:i+2, i:i+2] + temp[:, :, i+1]
        K_global = K_global.at[i:i + 2, i:i + 2].add(temp[:, :, i + 1])
    K_global = K_global.at[0, 0].add(temp[0, 0, 0])
    return K_global
