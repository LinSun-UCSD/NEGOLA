import numpy as np
import jax.numpy as jnp
import jax.scipy.linalg as jsl

# this is to get the system matrices in continuous time domain
def get_continuous_state_space(K, M, C, B, output_type):
    # ndof = len(K[:, 1])
    # temp1 = np.block([[np.zeros((ndof, ndof)), np.eye(ndof)]])
    # temp2 = np.block([[np.linalg.lstsq(-M, K, rcond=None)[0], np.linalg.lstsq(-M, C, rcond=None)[0]]])
    # Ac = np.block([[temp1], [temp2]])
    # Bc = np.block([[np.zeros((ndof, 1))], [-B]])
    # Cc = np.block([np.linalg.lstsq(-M, K, rcond=None)[0], np.linalg.lstsq(-M, C, rcond=None)[0]])
    # if output_type == "abs":  # output is absolute acceleration
    #     Dc = np.zeros((ndof, 1))  # output is relative acceleration
    # elif output_type == "rel":
    #     Dc = -B
    # return Ac, Bc, Cc, Dc
    ndof = K.shape[0]

    # Define `temp1` using `jnp.block` equivalent
    temp1 = jnp.concatenate([jnp.zeros((ndof, ndof)), jnp.eye(ndof)], axis=1)

    # Replace least-squares with `solve` for direct inversion (note: ensure `M` is invertible)
    temp2_left = jsl.solve(-M, K)  # Solve -M * X = K for X
    temp2_right = jsl.solve(-M, C)  # Solve -M * X = C for X
    temp2 = jnp.concatenate([temp2_left, temp2_right], axis=1)

    # Define `Ac` using `jnp.block` equivalent
    Ac = jnp.concatenate([temp1, temp2], axis=0)

    # Define `Bc` using `jnp.block` equivalent
    Bc = jnp.concatenate([jnp.zeros((ndof, 1)), -B], axis=0)

    # Define `Cc` similar to `temp2`
    Cc = jnp.concatenate([temp2_left, temp2_right], axis=1)

    # Define `Dc` based on `output_type`
    if output_type == "abs":
        Dc = jnp.zeros((ndof, 1))
    elif output_type == "rel":
        Dc = -B

    return Ac, Bc, Cc, Dc