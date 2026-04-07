"""
@author: Mohammad Al-Jarrah
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import matplotlib
from SIR import SIR
from CWAE1 import CWAE1
from CWAE2 import CWAE2
from CWAE3 import CWAE3
from LREnKF import LREnKF
import param_config_dim
import ot

# Configure matplotlib to embed fonts in PDF/PS outputs and set default font sizes.
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
plt.rc('font', size=20)

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
    W2 = rng.normal(scale=1.0, size=(m, k))
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
tau = 1e-1  # time step
T = 2       # number of time steps
t = np.arange(0.0, tau * T, tau)  # time vector

# Noise levels.
sigma = 1e-2  # process noise std for the hidden state
gamma = 4e-1  # observation noise std
Noise = [sigma, gamma]

N = int(1e3)   # number of ensemble particles
NUM_SIM = 1    # number of independent simulation runs

# Latent dimensions to sweep over.
nn = [2, 4]

# Accumulators for W2 distances across all n values.
distance_LREnKF = []
distance_CWAE1  = []
distance_CWAE2  = []
distance_CWAE3  = []
distance_SIR    = []

# Storage for full particle arrays across all n values, keyed by n.
all_X = {}

for n in nn:
    print(f"\n--- Running filters for n = {n} ---")

    # Derive state and observation dimensions from the current latent size.
    L  = 5 * n  # full state dimension
    dy = 2 * n  # observation dimension
    input_dims  = [L, dy]
    latent_dims = [n, n]

    # Build a fixed random nonlinear lifting map for this n.
    Map = make_nonlinear_features_map(latent_dims[0], input_dims[0] - latent_dims[0], seed=0 + 1)

    # Load hyperparameters tuned for the current latent dimension.
    parameters_CWAE1 = param_config_dim.get_cwae1_parameters(n=n, latent_dims=latent_dims)
    parameters_CWAE2 = param_config_dim.get_cwae2_parameters(n=n, latent_dims=latent_dims)
    parameters_CWAE3 = param_config_dim.get_cwae3_parameters(n=n, latent_dims=latent_dims)

    # Build the true posterior reference by importance-weighting a large prior sample.
    y_true  = 1
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

    # Subsample the reference for W2 computation to keep cost manageable.
    p_w2_true = int(1e4)                   # number of reference particles used to compute W2
    ref = X_True[:, :p_w2_true].T         # shape: (p_w2_true, L)

    # Broadcast the scalar observation to the full filter input format.
    Y_True = np.ones((NUM_SIM, T, dy, 1)) * y_true

    # Sample initial latent particles and embed into the full state space.
    x_prior = np.random.randn(NUM_SIM, latent_dims[0], N)
    X0 = np.zeros((NUM_SIM, L, N))
    X0[:, :latent_dims[0]] = x_prior

    # Run all filtering methods; output shape: (NUM_SIM x T x L x N).
    X_LREnKF = LREnKF(Y_True, X0, A, h, grad_h, t, Noise, latent_dims, alpha=None, SIGMA=1e-6)
    X_CWAE1  = CWAE1(Y_True, X0, A_torch, h_torch, t, Noise, parameters_CWAE1)
    X_CWAE2  = CWAE2(Y_True, X0, A_torch, h_torch, t, Noise, parameters_CWAE2)
    X_CWAE3  = CWAE3(Y_True, X0, A_torch, h_torch, t, Noise, parameters_CWAE3)
    X_SIR    = SIR(Y_True, X0, A, h, t, Noise)

    # Save particle arrays for this n for later inspection.
    all_X[n] = {
        'X_LREnKF': X_LREnKF,
        'X_CWAE1':  X_CWAE1,
        'X_CWAE2':  X_CWAE2,
        'X_CWAE3':  X_CWAE3,
        'X_SIR':    X_SIR,
        'X_True':   X_True,
    }

    # Compute W2 distances averaged over NUM_SIM simulations.
    p_w2_est = min(N, p_w2_true)
    a = np.ones(p_w2_true) / p_w2_true   # uniform weights over reference particles
    b = np.ones(p_w2_est)  / p_w2_est    # uniform weights over filter particles

    w2_lrenkf = 0.0
    w2_cwae1  = 0.0
    w2_cwae2  = 0.0
    w2_cwae3  = 0.0
    w2_sir    = 0.0

    for k in range(NUM_SIM):
        M = ot.dist(ref, X_LREnKF[k, -1, :, :p_w2_est].T)
        w2_lrenkf += np.sqrt(ot.emd2(a, b, M))

        M = ot.dist(ref, X_CWAE1[k, -1, :, :p_w2_est].T)
        w2_cwae1 += np.sqrt(ot.emd2(a, b, M))

        M = ot.dist(ref, X_CWAE2[k, -1, :, :p_w2_est].T)
        w2_cwae2 += np.sqrt(ot.emd2(a, b, M))

        M = ot.dist(ref, X_CWAE3[k, -1, :, :p_w2_est].T)
        w2_cwae3 += np.sqrt(ot.emd2(a, b, M))

        M = ot.dist(ref, X_SIR[k, -1, :, :p_w2_est].T)
        w2_sir += np.sqrt(ot.emd2(a, b, M))

    # Append mean W2 over simulations for this n.
    distance_LREnKF.append(w2_lrenkf / NUM_SIM)
    distance_CWAE1.append(w2_cwae1   / NUM_SIM)
    distance_CWAE2.append(w2_cwae2   / NUM_SIM)
    distance_CWAE3.append(w2_cwae3   / NUM_SIM)
    distance_SIR.append(w2_sir       / NUM_SIM)

# --- W2 vs n plot ---
fontsize = 20
plt.figure(figsize=(12, 6))
plt.semilogy(nn, distance_LREnKF, 'D:', color="C0", label="LREnKF", lw=2.5)
plt.semilogy(nn, distance_CWAE1,  's:', color="C2", label="CWAE1",  lw=2.5)
plt.semilogy(nn, distance_CWAE2,  'h:', color="C6", label="CWAE2",  lw=2.5)
plt.semilogy(nn, distance_CWAE3,  'H:', color="C7", label="CWAE3",  lw=2.5)
plt.semilogy(nn, distance_SIR,    'v:', color="C4", label="SIR",    lw=2.5)
plt.xlabel(r'$n$', fontsize=fontsize)
plt.ylabel(r'$W_2$', fontsize=fontsize)
plt.legend(fontsize=fontsize)
plt.tight_layout()
plt.show()
