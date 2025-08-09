import numpy as np
import jax.numpy as jnp

# get the eigen values of Ac
def get_mode_state_space(Ac):
    # D, T = np.linalg.eig(Ac)
    # return T, D
    D, T = jnp.linalg.eig(Ac)
    # D = jnp.linalg.eigvalsh(Ac)
    return T, D