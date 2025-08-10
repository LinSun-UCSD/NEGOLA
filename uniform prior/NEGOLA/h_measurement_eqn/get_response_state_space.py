import numpy as np
from .get_mode_state_space import get_mode_state_space
import jax.numpy as jnp
import jax.scipy.linalg as jsl

# state space model:
# x(t) = Ac*x_dot(t) + Bc*u(t)
# y(t) = Cc*x(t) + Dc*u(t)

def get_response_normalized_modal_eq(S, Bc, T, Cc, Dc, u):
    # temp = np.linalg.lstsq(T, Bc, rcond=None)[0]
    # Gamma = np.diag(temp.reshape((temp.shape[0],)))
    # X = np.real(np.matmul(np.matmul(T, Gamma), S))
    # y = np.real(np.matmul(np.matmul(np.matmul(Cc, T), Gamma), S) + np.matmul(Dc, u.reshape((1, len(u)))))
    # return y, X, Gamma
    temp = jsl.solve(T, Bc)  # This assumes T is invertible

    # Create Gamma as a diagonal matrix using temp values
    Gamma = jnp.diag(temp.reshape((temp.shape[0],)))

    # Calculate X using jnp.matmul
    X = jnp.real(jnp.matmul(jnp.matmul(T, Gamma), S))

    # Calculate y with multiple matrix multiplications
    y = jnp.real(jnp.matmul(jnp.matmul(jnp.matmul(Cc, T), Gamma), S) + jnp.matmul(Dc, u.reshape((1, len(u)))))

    return y, X, Gamma

# this is to get the response from state-space
def get_response_state_space(Ac, Bc, Cc, Dc, u, t):

    # T, Lambda = get_mode_state_space(Ac)
    # n2 = Lambda.shape[0]  # state space dimension
    # # initialize
    # N = len(u)
    # s = np.zeros((n2, N), dtype=complex)
    # s_dot = np.zeros((n2, N), dtype=complex)
    # # calculate the normalized complex modal response ‘s'
    # for j in range(n2):
    #     for m in range(N - 1):
    #         c2 = (u[m + 1] - u[m]) / (t[m + 1] - t[m])
    #         c1 = u[m] - c2 * t[m]
    #         tau = t[m + 1] - t[m]
    #         temp = s[j, m] + c2 / Lambda[j] * t[m] + c1 / Lambda[j] + c2 / (Lambda[j] ** 2)
    #         s[j, m + 1] = temp * np.exp(Lambda[j] * tau) \
    #                       - c2 / Lambda[j] * t[m + 1] \
    #                       - (c1 / Lambda[j] + c2 / (Lambda[j] ** 2))
    #         s_dot[j, m + 1] = temp * np.exp(Lambda[j] * tau) * Lambda[j] - c2 / Lambda[j]
    # y, X, Gamma = get_response_normalized_modal_eq(s, Bc, T, Cc, Dc, u)
    # return y, X, s, s_dot, T, Gamma
    # Get T and Lambda from the mode state space function
    T, Lambda = get_mode_state_space(Ac)
    n2 = Lambda.shape[0]  # State space dimension

    # Initialize using jax.numpy.zeros with complex dtype
    N = len(u)
    s = jnp.zeros((n2, N), dtype=jnp.complex64)
    s_dot = jnp.zeros((n2, N), dtype=jnp.complex64)

    # Calculate the normalized complex modal response `s`
    for j in range(n2):
        for m in range(N - 1):
            c2 = (u[m + 1] - u[m]) / (t[m + 1] - t[m])
            c1 = u[m] - c2 * t[m]
            tau = t[m + 1] - t[m]
            temp = s[j, m] + c2 / Lambda[j] * t[m] + c1 / Lambda[j] + c2 / (Lambda[j] ** 2)

            # Update `s` and `s_dot` using jax.numpy operations
            s = s.at[j, m + 1].set(
                temp * jnp.exp(Lambda[j] * tau) - c2 / Lambda[j] * t[m + 1]
                - (c1 / Lambda[j] + c2 / (Lambda[j] ** 2))
            )
            s_dot = s_dot.at[j, m + 1].set(
                temp * jnp.exp(Lambda[j] * tau) * Lambda[j] - c2 / Lambda[j]
            )

    # Calculate response using the normalized modal response function
    y, X, Gamma = get_response_normalized_modal_eq(s, Bc, T, Cc, Dc, u)
    return y, X, s, s_dot, T, Gamma