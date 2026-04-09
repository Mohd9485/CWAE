import traceback
 
import numpy as np
import torch
import optuna
from optuna.samplers import TPESampler
from torchvision import transforms
 
from CWAE1_NS import (
    poll_stream_unif,
    normalize_video,
    CWAE1_NS,
)
from CWAE2_NS import CWAE2_NS
from CWAE3_NS import CWAE3_NS

# ============================================================
#  Global config
# ============================================================

DATA_PATH   = "data/CylinderBig.npy"
YN_SIDE     = 9        # observation grid side-length  (yn × yn patch)
GAMMA_NOISE = 0.02    # additive observation noise std  (Y-space)
SIGMA_X     = 0.01    # additive state noise std         (X-space, applied via A)
DEVICE      = "cuda:0" if torch.cuda.is_available() else "cpu"

# Phase 1 — tuning budget (applied independently to each of the 3 studies)
N_PARTICLES = 5000     # frames drawn per trial  (None → full dataset)
EPOCHS_TUNE = 30       # epochs per Optuna trial
N_TRIALS    = 999      # max trials per study
TIMEOUT_S   = 4 * 60 * 60       # wall-clock limit per study (seconds)
STORAGE     = None                # e.g. "sqlite:///cwae_ns.db" for persistence

T_STEPS     = 2

SEEDS        = [0, 1, 2]          # independent seed per model

# Ensemble evaluation (used as the Optuna objective)
N_ENSEMBLE   = 5000    # particles drawn per val frame inside CWAE1_NS
 
# One entry per model — drives both Phase 1 and Phase 2
MODELS = [
    dict(
        name       = "CWAE2",
        study_name = "cwae2_ns_tuning",
        cwae_fn    = CWAE2_NS,
        seed       = SEEDS[2],
    ),
]
 
 
# ============================================================
#  Data loading
# ============================================================
 
def load_data():
    """
    Returns
    -------
    X : np.ndarray  (N, 2, 48, 48)  float32  — full velocity fields
    Y : np.ndarray  (N, 2,  9,  9)  float32  — patch observations
    """
    print(f"Loading {DATA_PATH} ...")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    data = np.load(DATA_PATH)           # (N, H_raw, W_raw, C)
    data = data[:, 1:49, 1:49, :]      # (N, 48, 48, C)
    X    = normalize_video(data, transform).astype(np.float32)
    yn   = YN_SIDE ** 2
    Y    = poll_stream_unif(
        X,
        np.arange(0, data.shape[2], dtype=int),
        np.arange(0, data.shape[3], dtype=int),
        yn,
    ).astype(np.float32)
    print(f"  Dataset : {len(X)} frames  |  X: {X.shape[1:]}  |  Y: {Y.shape[1:]}")
    return X, Y
 
 
DATA_NORM, DY = load_data()
dy = (2, 9, 9)
N_TOTAL       = len(DATA_NORM)
C, H, W       = DATA_NORM.shape[1:]   # 2, 48, 48
yn            = YN_SIDE   
 
print(f"  Full dataset : {N_TOTAL} frames  |  Tuning subset per trial: "
      f"{N_PARTICLES if N_PARTICLES else 'all'}")
print("=" * 60)
 
 
def _grid_indices(use_torch=False):
    coords_h = np.round(np.linspace(0, H, yn, endpoint=False)).astype(int)
    coords_w = np.round(np.linspace(0, W, yn, endpoint=False)).astype(int)
    ii, jj   = np.meshgrid(coords_h, coords_w, indexing='ij')
    if use_torch:
        return torch.from_numpy(ii), torch.from_numpy(jj)
    return ii, jj
 
 
def h_np(x: np.ndarray) -> np.ndarray:
    """x: (N, C, H, W) → (N, C, yn, yn)"""
    ii, jj = _grid_indices(use_torch=False)
    return x[:, :, ii, jj]
 
 
def h_torch(x: torch.Tensor) -> torch.Tensor:
    """x: (N, C, H, W) → (N, C, yn, yn)"""
    ii, jj = _grid_indices(use_torch=True)
    return x[:, :, ii, jj]
 
 
def A(x, t=0):
    """Identity dynamics."""
    return x
 
def make_otf_inputs(n: int, rng: np.random.Generator):
    """
    Sample n frames and build CWAE*_NS-ready arrays.
 
    Steps (matching the reference pattern):
        idx    : random choice of n indices from the full dataset
        X1_raw : (N, C, H, W)          — normalised velocity fields
        Y1_raw : (N, C, yn, yn)        — h(X1_raw) + observation noise
        X1     : (C, H, W, N)          — permute_dims(X1_raw, (1,2,3,0))
        Y1     : (C, yn, yn, N)        — permute_dims(Y1_raw, (1,2,3,0))
        Y_True : (1, T, C, yn, yn, 1)  — tile last particle obs over T steps
        X0     : (1, C, H, W, N)       — add leading NUM_SIM=1 axis
 
    Returns
    -------
    Y_True : np.ndarray  (1, T_STEPS, C, yn, yn, 1)
    X0     : np.ndarray  (1, C, H, W, N)
    X_true : np.ndarray  (N, C, H, W)   — ground-truth frames for MSE
    """
    idx    = rng.choice(N_TOTAL, n, replace=False)
    X1_raw = DATA_NORM[idx]
    Y1_raw = (h_np(np.expand_dims(X1_raw[-1], 0)) + GAMMA_NOISE * rng.standard_normal((1, *dy))).astype(np.float32)

    X1 = np.permute_dims(X1_raw, (1, 2, 3, 0))   # (C, H, W, N)
    Y1 = np.permute_dims(Y1_raw, (1, 2, 3, 0))
 
    # Tile the last particle's observation over T_STEPS time steps,
    # then add the NUM_SIM=1 and N_obs=1 batch axes.
    #   Y1[:, :, :, -1] : (C, yn, yn)
    #   tile             → (T_STEPS, C, yn, yn)
    #   expand_dims      → (1, T_STEPS, C, yn, yn, 1)
    Y_True = np.expand_dims(
        np.tile(Y1[:, :, :, 0], (T_STEPS, 1, 1, 1)),
        (0, 5),
    )
 
    X0     = np.expand_dims(X1, 0)   # (1, C, H, W, N)
    X_true = X1_raw[-1]                  # (N, C, H, W)
    return Y_True, X0, X_true
 

def phi_sq(x):
    return x*x

def phi_1(x):
    return x

def mse(x, x_true,phi):
    x = phi(x)
    x_true = phi(x_true)
    x_mean = x.mean(axis=5)
    
    numerator = ((x_mean - x_true)**2).sum(axis=(2,3,4))
    print(numerator)
    return numerator.mean(axis=1)

def relative_mse(x, x_true,phi):
    x = phi(x)
    x_true = phi(x_true)
    x_mean = x.mean(axis=5)
    
    numerator = ((x_mean - x_true)**2).sum(axis=(2,3,4))
    denominator = (x_true ** 2).sum(axis=(2,3,4)) 
    print((numerator / denominator))
    return (numerator / denominator).mean(axis=1)
 
# ============================================================
#  Helpers
# ============================================================
 
def print_params(params: dict, indent: int = 4) -> None:
    pad = ' ' * indent
    print(pad + '{')
    for k, v in params.items():
        if isinstance(v, float):
            print(f"{pad}    '{k}': {v:.13g},")
        else:
            print(f"{pad}    '{k}': {v},")
    print(pad + '}')
 
 
def format_params(optuna_params: dict) -> dict:
    return {
        'batch_size':    optuna_params['batch_size'],
        'lamb':          optuna_params['adv_weight'],
        'lr1':     optuna_params['lr_enc_xy'],
        'lr2':      optuna_params['lr_enc_y'],
        'lr3':        optuna_params['lr_dec'],
        'lr4':      optuna_params['lr_dec_y'],
        'lr5':       optuna_params['lr_disc'],
    }
 
 
# ============================================================
#  Optuna objective
# ============================================================
 
def make_objective(cwae_fn):
    def objective(trial: optuna.Trial) -> float:
 
        # ── Hyperparameters ───────────────────────────────────────────────
        lr_enc_xy     = trial.suggest_float("lr_enc_xy",     1e-5, 1e-2, log=True)
        lr_enc_y      = trial.suggest_float("lr_enc_y",      1e-5, 1e-2, log=True)
        lr_dec        = trial.suggest_float("lr_dec",        1e-5, 1e-2, log=True)
        lr_dec_y      = trial.suggest_float("lr_dec_y",      1e-5, 1e-2, log=True)
        lr_disc       = trial.suggest_float("lr_disc",       1e-5, 1e-2, log=True)
        batch_size    = trial.suggest_int(  "batch_size",    1, 8) * 8
        adv_weight    = trial.suggest_float("adv_weight",    1e-3, 2.0,  log=True)
        div_weight    = 0.0
        smooth_weight = 0.0
 
        parameters = dict(
            latent_dims   = [80, 40],
            epochs        = EPOCHS_TUNE,
            batch_size    = batch_size,
            lr1           = lr_enc_xy,
            lr2           = lr_enc_y,
            lr3           = lr_dec,
            lr4           = lr_dec_y,
            lr5           = lr_disc,
            adv_weight    = adv_weight,
            div_weight    = div_weight,
            smooth_weight = smooth_weight,
            device        = DEVICE,
        )

        rng = np.random.default_rng(seed=trial.number)
        Y_True, X0, X_true = make_otf_inputs(N_PARTICLES, rng)
 
        t_vec = np.arange(T_STEPS, dtype=np.float32)   # [0., 1.]
        Noise = [SIGMA_X, GAMMA_NOISE]
 
        try:
            # cwae_fn returns (NUM_SIM, T, C, H, W, N)
            # [0, -1]              → (C, H, W, N)  last timestep, first simulation
            # permute_dims(3,0,1,2) → (N, C, H, W)
            X_out = np.permute_dims(
                cwae_fn(Y_True, X0, A, h_torch, t_vec, Noise, parameters)[0, -1],
                (3, 0, 1, 2),
            )                          # (N, C, H, W)
 
            X_out = np.permute_dims(X_out, (1, 2, 3, 0))
            loss = mse(np.expand_dims(X_out, (0, 1)), np.expand_dims(X_true, 0), phi_1)[0]
            del X_out
 
        except Exception as exc:
            print(f"[Optuna] Trial {trial.number} failed: {exc}")
            print(traceback.format_exc())
            loss = 1e6
 
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
 
        print(f"[Optuna] Trial {trial.number:4d} | MSE = {loss:.6f} | {dict(trial.params)}")
        del Y_True, X0, X_true
        return loss
 
    return objective
 
 
# ============================================================
#  Run all studies
# ============================================================
 
def run_all_studies() -> dict:
    """
    Run one independent Optuna study per model.
 
    Returns
    -------
    best_params_per_model : dict  {model_name: formatted_params_dict}
    """
    best_params_per_model = {}
 
    for cfg in MODELS:
        name       = cfg["name"]
        study_name = cfg["study_name"]
        cwae_fn    = cfg["cwae_fn"]
 
        print(f"\n{'=' * 60}")
        print(f"  Phase 1 — Optuna study for {name}  ({study_name})")
        print(f"{'=' * 60}")
 
        sampler = TPESampler(seed=42)
        study   = optuna.create_study(
            study_name     = study_name,
            direction      = "minimize",
            sampler        = sampler,
            storage        = STORAGE,
            load_if_exists = True,
        )
 
        study.optimize(
            make_objective(cwae_fn),
            n_trials          = N_TRIALS,
            timeout           = TIMEOUT_S,
            n_jobs            = 1,
            show_progress_bar = True,
        )
 
        best = study.best_trial
        fmt  = format_params(best.params)
        print(f"\n  [{name}] Best trial : #{best.number}  MSE = {best.value:.6f}")
        print_params(fmt)
 
        csv_path = f"optuna_{name.lower()}_ns_trials.csv"
        study.trials_dataframe().to_csv(csv_path, index=False)
        print(f"  [{name}] All trials saved to {csv_path}")
 
        best_params_per_model[name] = fmt
 
    return best_params_per_model
 
 
# ============================================================
#  Entry point
# ============================================================
 
if __name__ == "__main__":
 
    print("\n" + "=" * 60)
    print("  PHASE 1 — Running 3 independent Optuna studies")
    print("=" * 60)
 
    best_params_per_model = run_all_studies()
 
    print("\n=== Phase 1 complete — best params per model ===")
    for name, params in best_params_per_model.items():
        print(f"\n  {name}:")
        print_params(params)