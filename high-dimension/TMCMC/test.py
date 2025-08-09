import numpy as np
import pickle

import os as os
from h_measurement_eqn.h_measurement_eqn import h_measurement_eqn
from h_measurement_eqn.ismember import ismember
from h_measurement_eqn.normalize import normalize


# choose 'multiprocessing' for local workstation or 'mpi' for supercomputer
parallel_processing = 'multiprocessing'
resultPath = os.getcwd() + "\\result"
# measurement data:
g = 9.81  # gravity acceleration
# ground motion input
GMinput = {
    "totalStep": 1000,  # earthquake record step
    "fs": 50,  # sampling rate
    "filename": 'NORTHR_SYL090',  # the earthquake file to load
    "path": os.getcwd() + "\\earthquake record"  # earthquake record folder
}
# Parameters to update
ParameterInfo = {
    "ParameterName": ["k1", "k2"],  # stiffness at each floor
    "TrueParameterValues": np.ones((2, 1), ) * 1,  # true parameters
    "UpdateParameterIndex": []
}
k0 = 1
UpdateParameterName = ["k1", "k2"]
ParameterInfo["UpdateParameterIndex"] = ismember(ParameterInfo["ParameterName"], UpdateParameterName)
TrueUpdateParameterValues = ParameterInfo["TrueParameterValues"][:, 0]
# compute the true response
measure_vector = np.array([[0, 1]])  # take first floor and top floor acceleration as measurement
TrueResponse = h_measurement_eqn(ParameterInfo["TrueParameterValues"], GMinput, 1, GMinput["totalStep"], measure_vector,
                                 k0)
NoisyTrueResponse = TrueResponse + np.random.randn(TrueResponse.shape[0], TrueResponse.shape[1]) * 0.05

TrueResponse, max = normalize(TrueResponse)


def log_likelihood(particle_num, theta):
    """
    Required!
    log-likelihood function which is problem specific
    for the 2DOF example log-likelihood is

    Parameters
    ----------
    particle_num : int
        particle number.
    s : numpy array of size Np (number of parameters in all_pars)
        particle location in Np space

    Returns
    -------
    LL : float
        log-likelihood function value.

    """
    # calculate the mean
    theta.reshape(-1, 1)
    y = h_measurement_eqn(theta[0:2], GMinput, 1, GMinput["totalStep"], measure_vector, k0)

    for i in range(y.shape[1]):
        y[:, i] = y[:, i] / max[i]
    N, Ny = y.shape
    delta = NoisyTrueResponse - y
    print(N)
    print(Ny)
    print(delta)
    par_sigma_normalized = [0.05] * Ny
    if y.shape != NoisyTrueResponse.shape:
        return -np.Inf
    # par_sigma_normalized = theta[8:len(all_pars)].tolist() * Ny
    LL = -0.5 * N * Ny * np.log(2 * np.pi) - np.sum(N * np.log(par_sigma_normalized)) - np.sum(
        0.5 * (np.power(par_sigma_normalized, -2)) * np.sum(delta ** 2, axis=0))
    return LL
print(log_likelihood(0, np.array([1.2,1.2])))