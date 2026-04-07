"""
@author: Mohammad Al-Jarrah
"""

import numpy as np
import matplotlib.pyplot as plt
import torch, time
import sys
import matplotlib
from CWAE1 import CWAE1
from CWAE2 import CWAE2
from CWAE3 import CWAE3
from smac import Scenario, HyperparameterOptimizationFacade as HPO
import ot

# Configure matplotlib to embed fonts in PDF/PS outputs and set default font sizes.
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
plt.rc('font', size=16)

# Close any open figure windows to start fresh.
plt.close('all')


# Observation function: maps state x to its squared Euclidean norm.
def h(x):
    return (x*x).sum(axis=0,keepdims=True)

# State transition function: zero-drift (identity motion model).
def A(x,t=0):
    return np.zeros_like(x)

# Torch version of the observation function for gradient-based filters.
def h_torch(x):
    return (x*x).sum(axis=0,keepdims=True)

# Torch version of the state transition function.
def A_torch(x,t=0):
    return torch.zeros_like(x)

# Analytic gradient of h with respect to x, used by LREnKF.
def grad_h(x):
    return np.array([[2*x[0], 2*x[1]]])

def w2_distance(x, y):
    """
    Wasserstein-2 distance between two empirical distributions.

    Parameters
    ----------
    x : np.ndarray, shape (n, d)
    y : np.ndarray, shape (m, d)

    Returns
    -------
    float
        Scalar W2 distance (not squared).
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    n, m = x.shape[0], y.shape[0]

    if n < 2 or m < 2:
        raise ValueError("x and y must each contain at least 2 samples.")

    # Uniform weights over each sample set
    a = np.ones(n) / n   # (n,)
    b = np.ones(m) / m   # (m,)

    # Squared Euclidean cost matrix  (n, m)
    M = ot.dist(x, y, metric='sqeuclidean')

    # Solve exact OT — returns the W2^2 (squared Wasserstein-2)
    w2_squared = ot.emd2(a, b, M)
    return w2_squared

# Simulation parameters.
L = 2 # number of states
dy = 1 # number of states observed
input_dims = [L,dy]
latent_dims = [L-1,dy]


tau = 1e-2 # timpe step
T = 2  # number of time steps
t = np.arange(0.0, tau * T, tau)  # Time vector.


# dynamical system
noise = np.sqrt(1e-1) # noise level std
sigma = 1              # Noise in the hidden state
sigma0 = 1             # Noise in the initial state distribution
gamma = 2e-1           # Noise in the observation
Noise = [sigma, gamma]


N = int(1000)  # Number of ensemble particles.
NUM_SIM = 2        # Number of independent simulations.

# Containers for true states, observations, and initial particles.
X0 = np.zeros((NUM_SIM, L, N))

y_true = 1
Y_True = np.ones((NUM_SIM, T, dy, 1))*y_true

# Build the true posterior reference by importance-weighting a large prior sample.
y_true = 1
N_true = int(5e6)
X1 = np.random.randn(N_true, L).T
Y1 = (X1 * X1).sum(axis=0, keepdims=True) + gamma * np.random.randn(N_true, dy).T

rng = np.random.default_rng()
# Calculate the weight for each particle based on its observation likelihood P(Y|X^i).
W = np.sum((Y_True[0, -1] - Y1) * (Y_True[0, -1] - Y1), axis=0) / (2 * gamma * gamma)
# Adjust weights by subtracting the minimum value (for numerical stability).
W = W - np.min(W)
W = np.exp(-W)
# Normalize the weights so that they sum to 1.
W = W / np.sum(W)

# Resample the particles based on the normalized weights.
index = rng.choice(np.arange(N_true), N_true, p=W)
# Reassign the resampled particles to the current time step.
X_True = X1[:, index]

# ============================================================
# SMAC hyperparameter tuning via MMD validation loss
# ============================================================
from ConfigSpace import ConfigurationSpace, Float, Integer, Categorical
from smac import Scenario, HyperparameterOptimizationFacade as HPO

# ----------------------------------------------------------
#  SMAC Configuration Space
# ----------------------------------------------------------
cs = ConfigurationSpace(seed=42)


cs.add([
    Float(  "lr1",        (1e-5, 1e-2), log=True,  default=1e-3),
    Float(  "lr2",        (1e-5, 1e-2), log=True,  default=1e-3),
    Float(  "lr3",        (1e-5, 1e-2), log=True,  default=1e-3),
    Float(  "lr4",        (1e-5, 1e-2), log=True,  default=1e-3),
    Float(  "lr5",        (1e-5, 1e-2), log=True,  default=1e-3),
    Integer("nns1",       (1, 8),    log=False,  default=6),
    Integer("nns2",       (1, 8),    log=False,  default=6),
    Integer("nns3",       (1, 8),    log=False,  default=6),
    Integer("nns4",       (1, 8),    log=False,  default=6),
    Integer("nns5",       (1, 8),    log=False,  default=6),
    Integer("nbs1",       (1, 3),                   default=3),
    Integer("nbs2",       (1, 3),                   default=3),
    Integer("nbs3",       (1, 3),                   default=3),
    Integer("nbs4",       (1, 3),                   default=3),
    Integer("nbs5",       (1, 3),                   default=3),
    Float(  "lamb",       (1e-2, 1e1),  log=True,   default=1e0),
    Integer("batch_size", (1, 16),    log=True,   default=6),
    Integer("n_critic",   (1, 15),                  default=10),
])


# ----------------------------------------------------------
#  SMAC target function
# ----------------------------------------------------------
def cwae_target(config, seed: int = 0) -> float:
    """
    Train CWAE with the given config and return MMD validation loss.
    SMAC minimises this value.
    """
    # Scale integer configs to actual hidden-layer widths and residual block counts.
    nns_ = [int(config["nns1"]),int(config["nns2"]),int(config["nns3"]),int(config["nns4"]),int(config["nns5"])]
    nns_ = [x * 16 for x in nns_]
    nbs_ = [int(config["nbs1"]),int(config["nbs2"]),int(config["nbs3"]),int(config["nbs4"]),int(config["nbs5"])]
    lr_  = [float(config["lr1"]),float(config["lr2"]),float(config["lr3"]),float(config["lr4"]),float(config["lr5"])]

    params = {
        'normalization'         : 'Standard',
        'latent_dims'           : latent_dims,
        'NUM_NEURON'            : nns_,
        'BATCH_SIZE'            : int(config["batch_size"])*16,
        'LearningRate'          : lr_,
        'ITERATION'             : int(3e3),
        'Final_Number_ITERATION': int(1e3),
        'num_resblocks'         : nbs_,
        'lamb'                  : float(config["lamb"]),
        'n_critic'              : int(config["n_critic"]),
    }

    try:
        loss = 0
        # Subsample particles for W2 estimation to keep evaluation fast.
        n_sample = int(1e3)

        X_val = CWAE3(Y_True, X0, A_torch, h, t, Noise, params) # choose CWAE1, CWAE2, or CWAE3 as the validation model 
        for sim in range(X_val.shape[0]):
            loss += w2_distance(X_val[sim,-1,:,:n_sample].T, X_True[:,:n_sample].T)
        # Average W2 loss across independent simulations.
        loss /= X_val.shape[0]
    except Exception as e:
        print(f"[SMAC] Trial failed: {e}")
        loss = 1e6          # penalise failed runs

    print(f"[SMAC] config={dict(config)}  MMD={loss:.6f}")
    return loss



# ----------------------------------------------------------
#   Run SMAC optimisation
# ----------------------------------------------------------
scenario = Scenario(
    configspace        = cs,
    deterministic      = True,      # set False if CWAE has stochasticity you want averaged
    n_trials           = 1000,        # total number of configurations to evaluate
    walltime_limit     = 3 * 3600,   # optional: stop after n hours
    n_workers          = 1,         # increase for parallel evaluation
)

smac = HPO(
    scenario   = scenario,
    target_function = cwae_target,
)

incumbent = smac.optimize()

# ----------------------------------------------------------
#  Report best configuration
# ----------------------------------------------------------
print("\n=== Best configuration found ===")
print(incumbent)
print(f"Validation W2 loss: {smac.validate(incumbent):.6f}")
