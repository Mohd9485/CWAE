
"""
@author: Mohammad Al-Jarrah
"""

import numpy as np
import matplotlib.pyplot as plt
import torch, time
import sys
import matplotlib
from CWAE1_NS import CWAE1_NS
from CWAE2_NS import CWAE2_NS
from CWAE3_NS import CWAE3_NS
from LREnKF import LREnKF
import param_config_NS

from torchvision import transforms

# Configure matplotlib to embed fonts in PDF/PS outputs and set default font sizes.
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
plt.rc('font', size=20)

# Close any open figure windows to start fresh.
plt.close('all')

import signal

# ── Graceful shutdown on Ctrl-C ────────────────────────────────────────────────
# These are populated as the run progresses; the signal handler saves whatever
# is complete at the moment of interruption.
_save_state = {
    "N_list":            [],
    "particles_CWAE1":   {},
    "particles_CWAE2":   {},
    "particles_CWAE3":   {},
    "particles_LREnKF":  {},
}

def _save_and_exit(sig, frame):
    print("\n\nCtrl-C detected — saving partial results...", flush=True)
    _flush_save()
    sys.exit(0)

def _flush_save():
    completed_N = _save_state["N_list"]
    if not completed_N:
        print("  No completed runs to save.", flush=True)
        return

    np.savez(
        "results_partial_full.npz",
        N_list           = np.array(completed_N),
        particles_CWAE1  = np.array([_save_state["particles_CWAE1"][n]  for n in completed_N]),
        particles_CWAE2  = np.array([_save_state["particles_CWAE2"][n]  for n in completed_N]),
        particles_CWAE3  = np.array([_save_state["particles_CWAE3"][n]  for n in completed_N]),
        particles_LREnKF = np.array([_save_state["particles_LREnKF"][n] for n in completed_N]),
    )
    print(f"  Partial results saved to mse_results_partial.npz "
          f"(N completed: {completed_N})", flush=True)

signal.signal(signal.SIGINT, _save_and_exit)

# Set a fixed random seed for reproducibility.
np.random.seed(0)
torch.manual_seed(0)

def make_indices(H, W, n, use_torch=False):
    coords_h = np.round(np.linspace(0, H, n, endpoint=False, dtype=int))
    coords_w = np.round(np.linspace(0, W, n, endpoint=False, dtype=int))
    ii, jj = np.meshgrid(coords_h, coords_w, indexing='ij')

    if use_torch:
        ii, jj = torch.from_numpy(ii), torch.from_numpy(jj)

    return ii, jj

def flat_H(H, W, n):
    coords_h = np.round(np.linspace(0, H, n, endpoint=False)).astype(int)
    coords_w = np.round(np.linspace(0, W, n, endpoint=False)).astype(int)
    # coords = [0, 5, 10, 16, 21, 26, 32, 37, 42]

    row_idx, col_idx = np.meshgrid(coords_h, coords_w, indexing='ij')  # (9, 9)
    flat_idx = row_idx.ravel() * W + col_idx.ravel()               # (81,)

    # Single-field observation matrix: (81, 2304)
    H_single = np.zeros((n * n, H * W), dtype=np.float32)
    H_single[np.arange(n * n), flat_idx] = 1.0

    # Block diagonal for 2 fields: (162, 4608)
    return np.block([[H_single, np.zeros_like(H_single)],
                [np.zeros_like(H_single), H_single]])

def h_conf(x, N, L, dy, flat=False):
    if flat: return flat_H(L[1], L[2], dy[1])@x
    ind = make_indices(L[1], L[2], dy[1], False)
    y = x[:, :, ind[0], ind[1]]

    return y.reshape(-1, N) if flat else y

def A(x,t=0):
    return x

def h_torch_conf(x, N, L, dy, flat=False):
    if flat: return torch.from_numpy(flat_H(L[1], L[2], dy[1])).to(x.device) @ x
    ind = make_indices(L[1], L[2], dy[1], True)
    y = x[:, :, ind[0], ind[1]]

    return y.reshape(-1, N) if flat else y

def grad_h_conf(x, N, L, dy, flat=False):
    if flat: return flat_H(L[1], L[2], dy[1])

    ind = make_indices(L[1], L[2], dy[1], False)
    grad_x = np.zeros((*L, N))
    grad_x[:, ind[0], ind[1], :] = 1.0

    return grad_x

def normalize_video(arr, transform):
    return np.array([transform(im) for im in arr])

def mse(x, x_true,phi):
    x = phi(x)
    x_true = phi(x_true)
    x_mean = x.mean(axis=5)
    
    numerator = ((x_mean - x_true)**2).sum(axis=(2,3,4))
    return numerator.mean(axis=1)

def relative_mse(x, x_true,phi):
    x = phi(x)
    x_true = phi(x_true)
    x_mean = x.mean(axis=5)
    
    numerator = ((x_mean - x_true)**2).sum(axis=(2,3,4))
    print(numerator)
    denominator = (x_true ** 2).sum(axis=(2,3,4)) 
    return (numerator / denominator).mean(axis=1)

#%%
# Simulation parameters.
L = (2, 48, 48)
dy = (2, 9, 9)

latent_dims = [80, 40]

tau = 1e-1
T = 2
t = np.arange(0.0, tau * T, tau)

sigma  = 0.01
gamma  = 0.02
Noise  = [sigma, gamma]

NUM_RUNS = 10
true_N   = 17000

N_list = [5000]

dev = 0
device = f"cuda:{dev}"

_mse = lambda x, y: np.mean((x - y)**2)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
data = np.load("data/CylinderBig.npy")
data = data[:, 1:49, 1:49, :]
data = normalize_video(data, transform)
data = data.astype(np.float32)

# Stores shape: {N: array of shape (NUM_RUNS,)}
mse_CWAE1  = {}
mse_CWAE2  = {}
mse_CWAE3  = {}
mse_LREnKF = {}

particles_CWAE1  = {}
particles_CWAE2  = {}
particles_CWAE3  = {}
particles_LREnKF = {}
 
true_X = []

for N in N_list:
    run_particles_CWAE1  = []
    run_particles_CWAE2  = []
    run_particles_CWAE3  = []
    run_particles_LREnKF = []
    run_true_X = []

    print(f"\n{'='*60}")
    print(f"Running ensemble size N = {N}")
    print(f"{'='*60}")

    h           = lambda x, _N=N: h_conf(x, _N, L, dy)
    grad_h      = lambda x, _N=N: grad_h_conf(x, _N, L, dy)
    h_torch     = lambda x, _N=N: h_torch_conf(x, _N, L, dy)
    h_flat      = lambda x, _N=N: h_conf(x, _N, L, dy, flat=True)
    grad_h_flat = lambda x, _N=N: grad_h_conf(x, _N, L, dy, flat=True)
    h_torch_flat= lambda x, _N=N: h_torch_conf(x, _N, L, dy, flat=True)

    parameters_CWAE1 = param_config_NS.get_cwae1_parameters(N=N, device=device, latent_dims=latent_dims)
    parameters_CWAE2 = param_config_NS.get_cwae2_parameters(N=N, device=device, latent_dims=latent_dims)
    parameters_CWAE3 = param_config_NS.get_cwae3_parameters(N=N, device=device, latent_dims=latent_dims)

    run_mse_CWAE1  = np.zeros(NUM_RUNS)
    run_mse_CWAE2  = np.zeros(NUM_RUNS)
    run_mse_CWAE3  = np.zeros(NUM_RUNS)
    run_mse_LREnKF = np.zeros(NUM_RUNS)

    for run in range(NUM_RUNS):
        print(f"\n  --- Run {run + 1}/{NUM_RUNS} ---")

        # Each run draws a fresh random ensemble
        idx = np.random.choice(true_N, N, replace=False)
        X1  = data[idx, :, :, :]
        Y1  = h(X1) + gamma * np.random.randn(N, *dy)

        X1 = np.permute_dims(X1, (1, 2, 3, 0))
        Y1 = np.permute_dims(Y1, (1, 2, 3, 0))

        NUM_SIM = 1
        Y_True = np.expand_dims(np.tile(Y1[:, :, :, -1], (2, 1, 1, 1)), (0, 5))
        X_True = X1[:, :, :, -1]
        np.save(f"x_true{run}", X_True)
        np.save(f"y_true{run}", Y_True)
        np.save(f"x_ens_true{run}", X1)

        X0 = np.expand_dims(X1, 0)

        print(f"    Running CWAE1 ...", flush=True)
        X_CWAE1_NS = CWAE1_NS(Y_True, X0, A, h_torch, t, Noise, parameters_CWAE1)

        print(f"    Running CWAE2 ...", flush=True)
        X_CWAE2_NS = CWAE2_NS(Y_True, X0, A, h_torch, t, Noise, parameters_CWAE2)

        print(f"    Running CWAE3 ...", flush=True)
        X_CWAE3_NS = CWAE3_NS(Y_True, X0, A, h_torch, t, Noise, parameters_CWAE3)

        print(f"    Running LREnKF...", flush=True)
        X_LREnKF = LREnKF(Y_True, X0, A, h_flat, grad_h_flat, t, Noise,
                           latent_dims, alpha=None, SIGMA=1e-6).reshape((NUM_SIM, T, *L, N))

        run_particles_CWAE1.append(X_CWAE1_NS[0, 1].copy())
        run_particles_CWAE2.append(X_CWAE2_NS[0, 1].copy())
        run_particles_CWAE3.append(X_CWAE3_NS[0, 1].copy())
        run_particles_LREnKF.append(X_LREnKF[0, 1].copy())

        run_true_X.append(X_True)

        _n_done = run + 1
        _save_state["particles_CWAE1"][N]  = np.stack(run_particles_CWAE1,  axis=0)
        _save_state["particles_CWAE2"][N]  = np.stack(run_particles_CWAE2,  axis=0)
        _save_state["particles_CWAE3"][N]  = np.stack(run_particles_CWAE3,  axis=0)
        _save_state["particles_LREnKF"][N] = np.stack(run_particles_LREnKF, axis=0)
        if N not in _save_state["N_list"]:
            _save_state["N_list"].append(N)

    particles_CWAE1[N]  = np.stack(run_particles_CWAE1,  axis=0)
    particles_CWAE2[N]  = np.stack(run_particles_CWAE2,  axis=0)
    particles_CWAE3[N]  = np.stack(run_particles_CWAE3,  axis=0)
    particles_LREnKF[N] = np.stack(run_particles_LREnKF, axis=0)

#%%
# ── Final save (all N completed) ──────────────────────────────────────────────
np.savez(
    "results_full.npz",
    particles_CWAE1  = np.array([particles_CWAE1[n]  for n in N_list]),
    particles_CWAE2  = np.array([particles_CWAE2[n]  for n in N_list]),
    particles_CWAE3  = np.array([particles_CWAE3[n]  for n in N_list]),
    particles_LREnKF = np.array([particles_LREnKF[n] for n in N_list]),
    true_X = np.array(true_X)
)
print("Full results saved to results_full.npz")
