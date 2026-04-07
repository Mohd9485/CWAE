"""
@author: Mohammad Al-Jarrah
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import matplotlib
from CWAE1 import CWAE1
from CWAE2 import CWAE2
from CWAE3 import CWAE3
from smac import Scenario, HyperparameterOptimizationFacade as HPO
from mmd_loss import mmd_loss
from ConfigSpace import ConfigurationSpace, Float, Integer

# Configure matplotlib to embed fonts in PDF/PS outputs and set default font sizes.
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
plt.rc('font', size=16)

# Close any open figure windows to start fresh.
plt.close('all')


# Observation function: returns the cube of the first dy components of x.
def h(x):
    return (x[:input_dims[1]]**3)

# State transition function: lifts the latent state x[:n] to the full state space via Map.
def A(x, t=0):
    return np.concatenate((x[:latent_dims[0]].T, Map(x[:latent_dims[0]].T)), axis=1).T

# Torch version of the observation function for gradient-based filters.
def h_torch(x):
    return (x[:input_dims[1]]**3)

# Torch version of the state transition function.
def A_torch(x, t=0):
    return torch.from_numpy(A(x)).to(torch.float32)

# Analytic Jacobian of h(x) = x^3 with respect to x.
def grad_h(x):
    # Jacobian is block-diagonal: diag(3*x_i^2) in the observed block, zeros elsewhere.
    M = np.zeros((input_dims[1], input_dims[0]))
    M[:input_dims[1], :input_dims[1]] = np.diag(3 * x[:input_dims[1]]**2)
    return M

def make_nonlinear_features_map(k, m, seed=0):
    """
    Create a nonlinear map R^k -> R^m using linear, quadratic, and sinusoidal features.

    Parameters
    ----------
    k    : int, input dimension
    m    : int, output dimension
    seed : int, random seed for reproducibility
    """
    rng = np.random.default_rng(seed)

    # Linear weight matrix, scaled by 1/sqrt(k) for variance control.
    W1 = rng.normal(scale=1.0 / np.sqrt(k), size=(m, k))

    # Symmetric quadratic kernels Q_i in R^{k x k}, one per output dimension.
    Q = rng.normal(scale=0.3 / k, size=(m, k, k))
    Q = 0.5 * (Q + np.transpose(Q, (0, 2, 1)))  # symmetrize each Q_i

    # Sinusoidal weight matrix and bias for the periodic component.
    W2 = rng.normal(scale=2.0, size=(m, k))
    b2 = rng.uniform(low=-np.pi, high=np.pi, size=(m,))

    def nonlinear_map(z):
        z    = np.atleast_2d(z)                         # shape: (n, k)
        lin  = z @ W1.T                                 # linear component,     shape: (n, m)
        quad = np.einsum("nk,mkl,nl->nm", z, Q, z)     # quadratic component,  shape: (n, m)
        sinp = np.sin(z @ W2.T + b2)                   # sinusoidal component, shape: (n, m)
        out  = 1 * lin + 1 * quad + 0.5 * sinp
        return out.squeeze(0) if out.shape[0] == 1 else out

    return nonlinear_map


# --- Simulation parameters ---
n  = 10      # latent dimension
L  = 5 * n  # full state dimension
dy = 2 * n  # observation dimension
input_dims  = [L, dy]
latent_dims = [n, n]

# Build a fixed random nonlinear lifting map.
Map = make_nonlinear_features_map(latent_dims[0], input_dims[0] - latent_dims[0], seed=0 + 1)

tau = 1e-2  # time step
T = 2       # number of time steps
t = np.arange(0.0, tau * T, tau)  # time vector

# Noise levels.
sigma = 1e-2  # process noise std for the hidden state
gamma = 4e-1  # observation noise std
Noise = [sigma, gamma]

N = int(1000)  # number of ensemble particles
NUM_SIM = 2    # number of independent simulation runs

# Sample initial latent particles and embed into the full state space.
x_prior = np.random.randn(NUM_SIM, latent_dims[0], N)
X0 = np.zeros((NUM_SIM, L, N))
X0[:, :latent_dims[0]] = x_prior

# Constant observation value used across all time steps and simulations.
y_true = 1
Y_True = np.ones((NUM_SIM, T, dy, 1)) * y_true

# Build the true posterior reference by importance-weighting a large prior sample.
N_true       = int(5e6)
x_prior_true = np.random.randn(N_true, latent_dims[0])
X1 = np.concatenate((x_prior_true, Map(x_prior_true)), axis=1).T
Y1 = ((X1[:input_dims[1]] ** 3).T + gamma * np.random.randn(N_true, input_dims[1])).T

rng = np.random.default_rng()
# Calculate the weight for each particle based on its observation likelihood P(Y|X^i).
W = np.sum((y_true - Y1) * (y_true - Y1), axis=0) / (2 * gamma * gamma)
# Adjust weights by subtracting the minimum value (for numerical stability).
W = W - np.min(W)
W = np.exp(-W)
# Normalize the weights so that they sum to 1.
W = W / np.sum(W)

# Resample the particles based on the normalized weights.
index  = rng.choice(np.arange(N_true), N_true, p=W)
# Reassign the resampled particles to form the reference posterior ensemble.
X_True = X1[:, index]


# ============================================================
# SMAC hyperparameter tuning via MMD validation loss
# ============================================================

# ----------------------------------------------------------
#  Configuration space for CWAE hyperparameters.
# ----------------------------------------------------------
cs = ConfigurationSpace(seed=42)

cs.add([
    Float(  "lr1",        (1e-5, 1e-2), log=True,  default=1e-3),
    Float(  "lr2",        (1e-5, 1e-2), log=True,  default=1e-3),
    Float(  "lr3",        (1e-5, 1e-2), log=True,  default=1e-3),
    Float(  "lr4",        (1e-5, 1e-2), log=True,  default=1e-3),
    Float(  "lr5",        (1e-5, 1e-2), log=True,  default=1e-3),
    Integer("nns1",       (1, 8),                   default=6),
    Integer("nns2",       (1, 8),                   default=6),
    Integer("nns3",       (1, 8),                   default=6),
    Integer("nns4",       (1, 8),                   default=6),
    Integer("nns5",       (1, 8),                   default=6),
    Integer("nbs1",       (1, 3),                   default=3),
    Integer("nbs2",       (1, 3),                   default=3),
    Integer("nbs3",       (1, 3),                   default=3),
    Integer("nbs4",       (1, 3),                   default=3),
    Integer("nbs5",       (1, 3),                   default=3),
    Float(  "lamb",       (1e-2, 1e1),  log=True,   default=1e0),
    Integer("batch_size", (1, 16),      log=True,   default=6),
    Integer("n_critic",   (1, 15),                  default=10),
])


# ----------------------------------------------------------
#  SMAC target function: train CWAE and return MMD validation loss.
# ----------------------------------------------------------
def cwae_target(config, seed: int = 0) -> float:
    """
    Train CWAE with the given hyperparameter configuration and return the MMD
    validation loss against the reference posterior. SMAC minimises this value.
    """
    nns_ = [int(config["nns1"]), int(config["nns2"]), int(config["nns3"]), int(config["nns4"]), int(config["nns5"])]
    nns_ = [x * 16 for x in nns_]
    nbs_ = [int(config["nbs1"]), int(config["nbs2"]), int(config["nbs3"]), int(config["nbs4"]), int(config["nbs5"])]
    lr_  = [float(config["lr1"]), float(config["lr2"]), float(config["lr3"]), float(config["lr4"]), float(config["lr5"])]

    params = {
        'normalization'         : 'Standard',
        'latent_dims'           : latent_dims,
        'NUM_NEURON'            : nns_,
        'BATCH_SIZE'            : int(config["batch_size"]) * 16,
        'LearningRate'          : lr_,
        'ITERATION'             : int(2e3),
        'Final_Number_ITERATION': int(1e3),
        'num_resblocks'         : nbs_,
        'lamb'                  : float(config["lamb"]),
        'n_critic'              : int(config["n_critic"]),
    }

    try:
        loss     = 0
        n_sample = int(2e3)
        # CWAE1, CWAE2, or CWAE3 can be selected here.
        X_val = CWAE1(Y_True, X0, A_torch, h, t, Noise, params) 
        for sim in range(X_val.shape[0]):
            loss += mmd_loss(X_val[sim, -1, :, :n_sample].T, X_True[:, :n_sample].T)
        loss /= X_val.shape[0]
    except Exception as e:
        print(f"[SMAC] Trial failed: {e}")
        loss = 1e6  # penalise failed runs

    print(f"[SMAC] config={dict(config)}  MMD={loss:.6f}")
    return loss


# ----------------------------------------------------------
#  Run SMAC optimisation.
# ----------------------------------------------------------
scenario = Scenario(
    configspace    = cs,
    deterministic  = True,      # set False if CWAE has stochasticity you want averaged
    n_trials       = 1000,      # total number of configurations to evaluate
    walltime_limit = 3 * 3600,  # stop after 3 hours
    n_workers      = 1,         # increase for parallel evaluation
)

smac = HPO(
    scenario        = scenario,
    target_function = cwae_target,
)

incumbent = smac.optimize()

# ----------------------------------------------------------
#  Report best configuration.
# ----------------------------------------------------------
print("\n=== Best configuration found ===")
print(incumbent)
print(f"Validation MMD loss: {smac.validate(incumbent):.6f}")
