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
from LREnKF import LREnKF
import param_config
import param_config_dim
import seaborn as sns

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

# Analytic Jacobian of h(x) = x^3 with respect to x, used by LREnKF.
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
n = 3
L  = 5 * n  # full state dimension
dy = 2 * n  # observation dimension
input_dims  = [L, dy]
latent_dims = [n, n]

tau = 1e-2  # time step
T = 2       # number of time steps
t = np.arange(0.0, tau * T, tau)  # time vector

# Noise levels
sigma = 1e-2  # process noise std for the hidden state
gamma = 4e-1  # observation noise std
Noise = [sigma, gamma]

N = int(1000)  # number of ensemble particles
NUM_SIM = 1    # number of independent simulation runs

# Build a fixed random nonlinear lifting map and load nonlinear-tuned hyperparameters.
Map = make_nonlinear_features_map(latent_dims[0], input_dims[0] - latent_dims[0], seed=0 + 1)
parameters_CWAE1 = param_config.get_cwae1_parameters(N=N, latent_dims=latent_dims)
parameters_CWAE2 = param_config.get_cwae2_parameters(N=N, latent_dims=latent_dims)
parameters_CWAE3 = param_config.get_cwae3_parameters(N=N, latent_dims=latent_dims)
# parameters_CWAE1 = param_config_dim.get_cwae1_parameters(n=n, latent_dims=latent_dims)
# parameters_CWAE2 = param_config_dim.get_cwae2_parameters(n=n, latent_dims=latent_dims)
# parameters_CWAE3 = param_config_dim.get_cwae3_parameters(n=n, latent_dims=latent_dims)

# Override the number of training iterations for all CWAE variants.
it = 5000
parameters_CWAE1['ITERATION'] = it
parameters_CWAE2['ITERATION'] = it
parameters_CWAE3['ITERATION'] = it

# Sample the latent prior and embed into the full state space (remaining dims set to zero).
x_prior = np.random.randn(NUM_SIM, latent_dims[0], N)
X0 = np.zeros((NUM_SIM, L, N))
X0[:, :latent_dims[0]] = x_prior

# Constant observation value used across all time steps and simulations.
y_true = 1
Y_True = np.ones((NUM_SIM, T, dy, 1)) * y_true

# Run all filters; output shape: (NUM_SIM x T x L x N).
X_CWAE1  = CWAE1(Y_True, X0, A_torch, h, t, Noise, parameters_CWAE1)
X_CWAE2  = CWAE2(Y_True, X0, A_torch, h, t, Noise, parameters_CWAE2)
X_CWAE3  = CWAE3(Y_True, X0, A_torch, h, t, Noise, parameters_CWAE3)
X_LREnKF = LREnKF(Y_True, X0, A, h, grad_h, t, Noise, latent_dims, alpha=None, SIGMA=1e-6)

# Build the true posterior reference by importance-weighting a large prior sample.
N_true  = int(5e6)
x_prior = np.random.randn(N_true, latent_dims[0])
X1 = np.concatenate((x_prior, Map(x_prior)), axis=1).T
Y1 = ((X1[:input_dims[1]] ** 3).T + gamma * np.random.randn(N_true, input_dims[1])).T

# Sequential importance resampling to approximate the true posterior distribution given the observation Y_True.
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

#%%
# --- Plot marginal KDE for the unobserved (high-dimensional) state components ---
fontsize = 20
k = 0
bw = 1
plt.figure(figsize=(16, 6))
for l in range(12, L):
    plt.subplot(1, 3, l - 11)
    sns.kdeplot(X_True[l, :],           color="black", linestyle="--",               label="True",   bw_adjust=bw, linewidth=2.5)
    sns.kdeplot(X_LREnKF[k, -1, l, :], color="C0",    linestyle=":",                label="LREnKF", bw_adjust=bw, linewidth=2.5)
    sns.kdeplot(X_CWAE1[k, -1, l, :],  color="C2",    linestyle="-.",               label="CWAE1",  bw_adjust=bw, linewidth=2.5)
    sns.kdeplot(X_CWAE2[k, -1, l, :],  color="C6",    linestyle=(0, (3, 1, 1, 1)),  label="CWAE2",  bw_adjust=bw, linewidth=2.5)
    sns.kdeplot(X_CWAE3[k, -1, l, :],  color="C7",    linestyle=(0, (5, 1)),        label="CWAE3",  bw_adjust=bw, linewidth=2.5)
    plt.ylabel("")
    if l - 12 == 0:
        # Only add legend on the first subplot.
        plt.legend(fontsize=fontsize)
    plt.xlabel(rf'$X_{{{l+1}}}$', fontsize=fontsize)

plt.tight_layout()
