# %%
import numpy as np

# --- Load files ---
particles_path  = "mse_results.npz"
particles2_path = "mse_results_CWAE2.npz"

particles_data  = np.load(particles_path)
particles2_data = np.load(particles2_path)
gt_data         = np.array([np.load(f"x_true{i}.npy") for i in range(10)])

# Inspect keys
print("Particle keys (file 1):", list(particles_data.keys()))
print("Particle keys (file 2):", list(particles2_data.keys()))

# --- Load ground truth ---
# Shape: (NRUNS, C, H, W)
gt = gt_data
NRUNS, C, H, W = gt.shape
print(f"Ground truth shape: {gt.shape}")

# --- Collect all particle arrays from both files ---
def load_particles(npz, label):
    keys = sorted(npz.keys())
    print(f"Found {len(keys)} particle arrays in {label}: {keys}")
    return {f"{label}/{k}": npz[k] for k in keys}

all_arrays = {}
all_arrays.update(load_particles(particles_data,  ""))
all_arrays.update(load_particles(particles2_data, ""))

# Stack into shape: (NARRAYS, NRUNS, 1, C, W, H, NPARTICLES)
array_keys    = ['/particles_CWAE1', '/particles_CWAE2', '/particles_CWAE3', "/particles_LREnKF"]
all_particles = np.stack([all_arrays[k][0] for k in array_keys], axis=0)
NARRAYS, NRUNS_, C_, W_, H_, _ = all_particles.shape
print(f"\nStacked particles shape: {all_particles.shape}")

all_particles_mean = np.stack([all_arrays[k][0] for k in array_keys], axis=0).mean(-1)

def print_metrics(label, mse_per_run, array_keys):
    print(f"\n{label} per run: {mse_per_run}")
    for i, key in enumerate(array_keys):
        print(f"  {key} — mean {label}: {mse_per_run[i]:.6f}")

# Reshape gt to: (1, NRUNS, 1, C, H, W, 1)
# Note: swap W/H in all_particles if dim order is (C, W, H) but gt is (C, H, W)
gt_broadcast = np.tile(gt, (4, 1, 1, 1, 1))

def phi_sq(x):
    return x*x

def phi_1(x):
    return x

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

models, nruns, channel, height, width, particles = 2 ,3, 4, 5, 5, 6 

X_particles = all_particles
X_true = gt_broadcast

mse_val_x1 = mse(X_particles,X_true,phi_1)
relative_mse_val_x1 = relative_mse(X_particles,X_true,phi_1)
print(relative_mse_val_x1)

mse_val_x2 = mse(X_particles,X_true,phi_sq)
relative_mse_val_x2 = relative_mse(X_particles,X_true,phi_sq)
print(relative_mse_val_x2)