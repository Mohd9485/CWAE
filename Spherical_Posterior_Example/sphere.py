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
import param_config_sphere 

# Embed fonts in PDF/PS outputs and set default font sizes.
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
plt.rc('font', size=20)

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

# --- Simulation parameters ---
L = 2           # State dimension
dy = 1          # Observation dimension
input_dims = [L, dy]
latent_dims = [L-1, dy]

tau = 1e-1      # Time step size
T = 2           # Number of time steps
t = np.arange(0.0, tau * T, tau)  # Time vector

# Noise levels
noise = np.sqrt(1e-1)   # General noise level (std)
sigma = 1               # Process noise std for the hidden state
sigma0 = 1              # Initial state distribution noise std
gamma = 2e-1            # Observation noise std
Noise = [sigma, gamma]

NUM_SIM = 1     # Number of independent simulation runs
N = 1000        # Number of particles
fontsize = 20

# Load filter hyperparameters from config.
parameters_CWAE1 = param_config_sphere.get_cwae1_parameters(N=N, latent_dims=latent_dims)
parameters_CWAE2 = param_config_sphere.get_cwae2_parameters(N=N, latent_dims=latent_dims)
parameters_CWAE3 = param_config_sphere.get_cwae3_parameters(N=N, latent_dims=latent_dims)

# --- Main loop: run filters for each observation value and plot results ---
plt.figure(figsize=(16, 10))
for ii in [1, 2]:
    y_true = ii * 1.0
    Y_True = np.ones((NUM_SIM, T, dy, 1)) * y_true

    # Generate a large reference sample from the prior and compute the true
    # posterior via importance weighting (reference particle filter).
    N_true = int(1e6)
    X1 = np.random.randn(N_true, L).T
    Y1 = (X1*X1).sum(axis=0, keepdims=True) + gamma * np.random.randn(N_true, dy).T

    # Sequential importance resampling to approximate the true posterior distribution given the observation Y_True.
    rng = np.random.default_rng()
    # Compute unnormalized log-weights from the Gaussian observation likelihood.
    W = np.sum((Y_True[0, -1] - Y1) * (Y_True[0, -1] - Y1), axis=0) / (2 * gamma * gamma)
    # Shift by the minimum for numerical stability before exponentiating.
    W = W - np.min(W)
    W = np.exp(-W)
    # Normalize weights to form a valid probability distribution.
    W = W / np.sum(W)

    # Resample particles from the prior according to the posterior weights.
    index = rng.choice(np.arange(N_true), N_true, p=W)
    X_True = X1[:, index]

    # Initialize particle ensemble to zero (each filter samples from the prior internally).
    X0 = np.zeros((NUM_SIM, L, N))

    # Run each filter on the true observations. Output shape: (NUM_SIM, T, L, N).
    X_LREnKF = LREnKF(Y_True, X0, A, h, grad_h, t, Noise, latent_dims, alpha=None, SIGMA=1e-6)
    X_CWAE1  = CWAE1(Y_True, X0, A_torch, h_torch, t, Noise, parameters_CWAE1)
    X_CWAE2  = CWAE2(Y_True, X0, A_torch, h_torch, t, Noise, parameters_CWAE2)
    X_CWAE3  = CWAE3(Y_True, X0, A_torch, h_torch, t, Noise, parameters_CWAE3)

    # Draw the true posterior manifold: a circle of radius sqrt(y_true).
    theta = np.linspace(0, 2 * np.pi, 300)
    r = np.sqrt(ii)
    x = r * np.cos(theta)
    y = r * np.sin(theta)

    N = 1000
    k = 0
    plt.subplot(1, 2, ii)
    plt.scatter(X_True[0, :N], X_True[1, :N], color="C1", alpha=0.5, label="True", marker='x')
    plt.scatter(X_LREnKF[k, -1, 0], X_LREnKF[k, -1, 1], color="C0", alpha=0.3, label="LREnKF", marker='p')
    plt.scatter(X_CWAE1[k, -1, 0], X_CWAE1[k, -1, 1], color='C2', alpha=0.3, label="CWAE1", marker='s')
    plt.scatter(X_CWAE2[k, -1, 0], X_CWAE2[k, -1, 1], color='C6', alpha=0.3, label="CWAE2", marker='h')
    plt.scatter(X_CWAE3[k, -1, 0], X_CWAE3[k, -1, 1], color='C7', alpha=0.3, label="CWAE3", marker='H')
    plt.plot(x, y, color='black', linestyle='--', linewidth=2.5, dashes=(5, 3))
    plt.xlabel(r"$X_1$", fontsize=fontsize)
    plt.ylabel(r"$X_2$", fontsize=fontsize)
    plt.xlim(-2.99, 2.99)
    plt.ylim(-2.99, 2.99)
    if ii == 1:
        # Only add legend on the first subplot; set full opacity on markers.
        leg = plt.legend(fontsize=fontsize, loc=1)
        for lh in leg.legend_handles:
            lh.set_alpha(1)
