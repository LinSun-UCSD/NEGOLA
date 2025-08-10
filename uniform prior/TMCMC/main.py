"""
@author: Lin Sun
email: lsun@ucsd.edu
main file to test transitional mcmc implementation
example: 8DOF system

"""
import numpy as np
import pickle
from tmcmc_mod import pdfs
from tmcmc_mod.tmcmc import run_tmcmc
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
    "totalStep": 500,  # earthquake record step
    "fs": 50,  # sampling rate
    "filename": 'NORTHR_SYL090',  # the earthquake file to load
    "path": os.getcwd() + "\\earthquake record"  # earthquake record folder
}
# Parameters to update
DOF = 4
k0 = 1
m0 = 1
ParameterInfo = {
    # "ParameterName": ["k1", "k2", "k3", "k4"],  # stiffness at each floor
    "ParameterName": [f"k{i}" for i in range(1, DOF+1)],
    "TrueParameterValues": np.ones((DOF, 1), ) * k0,  # true parameters
    "UpdateParameterIndex": []
}


# UpdateParameterName = ["k1", "k2", "k3", "k4"]
UpdateParameterName = [f"k{i}" for i in range(1, DOF+1)]
ParameterInfo["UpdateParameterIndex"] = ismember(ParameterInfo["ParameterName"], UpdateParameterName)
TrueUpdateParameterValues = ParameterInfo["TrueParameterValues"][:, 0]
# compute the true response
# measure_vector = np.array([[0,1,2,3]])
measure_vector = np.arange(DOF).reshape(1, DOF)
TrueResponse = h_measurement_eqn(ParameterInfo["TrueParameterValues"], GMinput, 1,
                                 GMinput["totalStep"], measure_vector, k0, m0)
TrueResponse, max = normalize(TrueResponse)
NoisyTrueResponse = TrueResponse + np.random.randn(TrueResponse.shape[0], TrueResponse.shape[1]) * 0.05
np.save(resultPath + "\\TrueResponse.npy", TrueResponse)
np.save(resultPath + "\\NoisyTrueResponse.npy", NoisyTrueResponse)
# number of particles (to approximate the posterior)
Np = 500

# prior distribution of parameters
# prior_mean = np.array([[2,2,2,2]])
# prior_mean = np.full((1, DOF), 2)
i = 0
all_pars = [pdfs.Uniform(0.3, 0.6)
            for i in range(DOF)]
# all_pars.append(pdfs.HalfNormal(sig=0.055))
# include measurement noise


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
    y = h_measurement_eqn(theta[0:DOF], GMinput, 1, GMinput["totalStep"], measure_vector, k0, m0)
    for i in range(y.shape[1]):
        y[:, i] = y[:, i] / max[i]
    N, Ny = y.shape
    delta = NoisyTrueResponse - y
    par_sigma_normalized = [0.05] * Ny
    if y.shape != NoisyTrueResponse.shape:
        return -np.Inf
    # par_sigma_normalized = theta[2:len(all_pars)].tolist() * Ny
    LL = -0.5 * N * Ny * np.log(2 * np.pi) - np.sum(N * np.log(par_sigma_normalized)) - np.sum(
        0.5 * (np.power(par_sigma_normalized, -2)) * np.sum(delta ** 2, axis=0))
    return LL


# run main
if __name__ == '__main__':
    """main part to run tmcmc for the 8DOF example"""
    import time as time
    start_time = time.time();
    mytrace, comm = run_tmcmc(Np, all_pars,
                              log_likelihood, parallel_processing,
                              "status_file_8DOF.txt")

    # save results
    with open('mytrace.pickle', 'wb') as handle1:
        pickle.dump(mytrace, handle1, protocol=pickle.HIGHEST_PROTOCOL)
    print("--- Execution time: %s seconds ---" + str(time.time() - start_time))
    if parallel_processing == 'mpi':
        comm.Abort(0)
