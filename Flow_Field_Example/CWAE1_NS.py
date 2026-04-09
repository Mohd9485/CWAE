import time
from typing import Any, Sequence, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, TensorDataset
import numpy as np
import matplotlib.pyplot as plt

from torchvision import transforms

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rc('font', size=20)

relu_slope = 0.125


class TupleDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]   # tuple

# ═══════════════════════════════════════════════════════════════
#  Encoder
# ═══════════════════════════════════════════════════════════════

class EncoderXY(nn.Module):
    """(B, 2, 48, 48) + (B, 2, 9, 9) → u (B, latent_dim_u)

    Encodes the joint (x, y) input into the latent variable u.
    Both the full velocity field x and the raw observation patch y are
    processed by separate conv branches; their flat features are concatenated
    before the shared FC head.
    """

    def __init__(self, latent_dim: int = 64, latent_dim_y: int = 16):
        super().__init__()
        # ── x branch: 2×48×48 → 256×3×3 ─────────────────────────────
        self.convx = nn.Sequential(
            nn.Conv2d(2, 32, 3, stride=2, padding=1),
            nn.LeakyReLU(relu_slope, inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(relu_slope, inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(relu_slope, inplace=True),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(relu_slope, inplace=True),
        )
        # ── y branch: 2×9×9 → 128×3×3 ───────────────────────────────
        self.convy = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(relu_slope, inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(relu_slope, inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(relu_slope, inplace=True),
        )
        # flat x: 256*3*3 = 2304   flat y: 128*3*3 = 1152
        self.fc = nn.Sequential(
            nn.Linear(256 * 3 * 3 + 128 * 3 * 3, 256),
            nn.LeakyReLU(relu_slope, inplace=True),
            nn.Linear(256, latent_dim),
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        x: (B, 2, 48, 48)  — full velocity field
        y: (B, 2,  9,  9)  — raw observation patch
        Returns u: (B, latent_dim_u)
        """
        x_flat = torch.flatten(self.convx(x), start_dim=1)   # (B, 2304)
        y_flat = torch.flatten(self.convy(y), start_dim=1)   # (B, 1152)
        return self.fc(torch.cat((x_flat, y_flat), dim=1))


class EncoderY(nn.Module):
    """(B, 2, 9, 9) → z (B, latent_dim_z)

    Encodes the observation patch y into the latent variable z.
    """

    def __init__(self, latent_dim: int = 64):
        super().__init__()
        self.latent_dim = latent_dim

        self.conv = nn.Sequential(
            # 2×9×9 → 32×7×7
            nn.Conv2d(2, 32, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(relu_slope, inplace=True),

            # 32×7×7 → 64×5×5
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(relu_slope, inplace=True),

            # 64×5×5 → 128×3×3
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(relu_slope, inplace=True),
        )
        # 128 × 3 × 3 = 1152
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 3 * 3, 256),
            nn.LeakyReLU(relu_slope, inplace=True),
            nn.Linear(256, latent_dim),
        )

    def forward(self, patch: torch.Tensor) -> torch.Tensor:
        """patch: (B, 2, 9, 9) → z: (B, latent_dim_z)"""
        return self.fc(self.conv(patch))

# ═══════════════════════════════════════════════════════════════
#  Decoder  (Generator)
# ═══════════════════════════════════════════════════════════════

class Decoder(nn.Module):
    def __init__(self, latent_dim: int = 64):
        super().__init__()
        # latent_dim = latent_dim_u + latent_dim_z  (input = cat(u, z))
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256 * 3 * 3),
            nn.ReLU(inplace=True),
        )
        self.unflatten = nn.Unflatten(1, (256, 3, 3))
        self.cv = nn.Conv2d(256, 256, 3, padding=1)
        self.bn = nn.BatchNorm2d(256)

        self.deconv = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.ReplicationPad2d(1),
            nn.Conv2d(256, 128, 3, padding=0), nn.BatchNorm2d(128), nn.LeakyReLU(relu_slope, inplace=True),

            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.ReplicationPad2d(1),
            nn.Conv2d(128, 64, 3, padding=0),  nn.BatchNorm2d(64),  nn.LeakyReLU(relu_slope, inplace=True),

            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.ReplicationPad2d(1),
            nn.Conv2d(64, 32, 3, padding=0),   nn.BatchNorm2d(32),  nn.LeakyReLU(relu_slope, inplace=True),

            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.ReplicationPad2d(1),
            nn.Conv2d(32, 2, 3, padding=0),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.deconv(self.bn(self.cv(self.unflatten(self.fc(z)))))


# ═══════════════════════════════════════════════════════════════
#  Decoder Y  (reconstructs patch y from latent z)
# ═══════════════════════════════════════════════════════════════

class DecoderY(nn.Module):
    """
    z (B, latent_dim_z) → y_hat (B, 2, 9, 9)

    Decodes the patch latent z back to the 9×9 observation patch y_hat.
    Shares the decoder penalty with Decoder via the combined reconstruction
    loss:  decoder_loss = decoder_xy_loss + decoder_y_loss + div_weight * div_loss
    """
    def __init__(self, latent_dim: int = 64):
        super().__init__()
        # Project latent → small spatial seed  (128 × 3 × 3)
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.LeakyReLU(relu_slope, inplace=True),
            nn.Linear(256, 128 * 3 * 3),
            nn.LeakyReLU(relu_slope, inplace=True),
        )
        self.unflatten = nn.Unflatten(1, (128, 3, 3))
        self.cv = nn.Conv2d(128, 128, 3, padding=1)
        self.bn = nn.BatchNorm2d(128)

        # 128×3×3 → 64×5×5 → 32×7×7 → 2×9×9
        # Use upsample+conv with ReplicationPad to stay consistent with Decoder
        self.deconv = nn.Sequential(
            # 128×3×3 → 64×5×5  (upsample then crop to exact size)
            nn.Upsample(size=(5, 5), mode='bilinear', align_corners=True),
            nn.ReplicationPad2d(1),
            nn.Conv2d(128, 64, 3, padding=0), nn.BatchNorm2d(64), nn.LeakyReLU(relu_slope, inplace=True),

            # 64×5×5 → 32×7×7
            nn.Upsample(size=(7, 7), mode='bilinear', align_corners=True),
            nn.ReplicationPad2d(1),
            nn.Conv2d(64, 32, 3, padding=0), nn.BatchNorm2d(32), nn.LeakyReLU(relu_slope, inplace=True),

            # 32×7×7 → 2×9×9
            nn.Upsample(size=(9, 9), mode='bilinear', align_corners=True),
            nn.ReplicationPad2d(1),
            nn.Conv2d(32, 2, 3, padding=0),
            # No activation — velocity is unbounded
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """z: (B, latent_dim_z) → y_hat: (B, 2, 9, 9)"""
        return self.deconv(self.bn(self.cv(self.unflatten(self.fc(z)))))


# ═══════════════════════════════════════════════════════════════
#  Discriminator  (operates in latent space)
# ═══════════════════════════════════════════════════════════════

class Discriminator(nn.Module):
    """
    Distinguishes 'real' samples (drawn from N(0,I)) from 'fake' ones
    (produced by the Encoder).  Operates purely in latent space.
    """

    def __init__(self, latent_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.LeakyReLU(relu_slope, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LeakyReLU(relu_slope, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.LeakyReLU(relu_slope, inplace=True),
            nn.Linear(128, 1),
            # Raw logits — use BCEWithLogitsLoss
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


# ═══════════════════════════════════════════════════════════════
#  Full model wrapper
# ═══════════════════════════════════════════════════════════════

class VelocityGANAutoencoder1(nn.Module):
    """Convenience wrapper that holds all networks.

    Data flow
    ---------
    Encode:
        z = encoder_y(y)             (B, latent_dim_z)   — patch latent
        u = encoder_xy(x, y)         (B, latent_dim_u)   — joint latent (raw y, not z)

    Decode:
        x_hat = decoder(cat(u, z))   (B, 2, 48, 48)
        y_hat = decoder_y(z)         (B, 2, 9, 9)

    Sample at inference:
        z   = encoder_y(y_true)
        u   ~ N(0, I)
        x   = decoder(cat(u, z))
    """

    def __init__(self, latent_dims: Sequence[int]):
        super().__init__()
        self.encoder_y     = EncoderY(latent_dims[1])
        # encoder_xy receives raw (x, y) — not (x, z)
        self.encoder_xy    = EncoderXY(latent_dims[0], latent_dim_y=latent_dims[1])
        self.decoder_y     = DecoderY(latent_dims[1])
        # decoder input = cat(u, z) = cat(encoder_xy output, encoder_y output)
        self.decoder       = Decoder(latent_dims[0] + latent_dims[1])
        self.discriminator = Discriminator(latent_dims[0] + latent_dims[1])
        self.latent_dims   = latent_dims

        self.encoder_y.apply(init_weights)
        self.encoder_xy.apply(init_weights)
        self.decoder_y.apply(init_weights)
        self.decoder.apply(init_weights)

    # ── inference helpers ──────────────────────────────────────
    def encode_latent(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (u, z).
        encoder_xy receives raw (x, y); encoder_y receives raw y independently.
        """
        z = self.encoder_y(y)
        u = self.encoder_xy(x, y)       # raw y, not z
        return u, z

    def encode(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Concatenated (u, z) for the discriminator."""
        u, z = self.encode_latent(x, y)
        return torch.cat((u, z), dim=1)

    def decode(self, u: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """Decode from u (encoder_xy output) and z (encoder_y output): input = cat(u, z)."""
        return self.decoder(torch.cat((u, z), dim=1))

    def reconstruct(self, x: torch.Tensor, y: torch.Tensor):
        u, z  = self.encode_latent(x, y)
        y_hat = self.decoder_y(z)                   # (B, 2, 9, 9)
        x_hat = self.decode(u, z)                   # (B, 2, 48, 48)  — uses cat(u, z)
        latent = torch.cat((u, z), dim=1)           # for discriminator
        return x_hat, y_hat, latent

    def sample(self, y: torch.Tensor, n: int, device: str = "cpu") -> torch.Tensor:
        """Draw n samples x ~ p(x | y) by sampling u ~ N(0,I) and decoding with z = encoder_y(y)."""
        z    = self.encoder_y(y)                             # (B, latent_dim_z)
        u    = torch.randn(n, self.latent_dims[0], device=device)
        # z is (B, latent_dim_z); if B != n tile it
        if z.shape[0] != n:
            z = z.expand(n, -1)
        return self.decode(u, z)


# ═══════════════════════════════════════════════════════════════
#  Loss helpers
# ═══════════════════════════════════════════════════════════════

_bce = nn.BCEWithLogitsLoss()
_mse = nn.MSELoss()


def recon_loss(x_hat: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    return _mse(x_hat, x)


def divergence_loss(x_hat: torch.Tensor) -> torch.Tensor:
    """Incompressibility penalty  ∇·u ≈ 0 (central finite differences)."""
    du_dx = x_hat[:, 0, 1:-1, 2:] - x_hat[:, 0, 1:-1, :-2]
    dv_dy = x_hat[:, 1, 2:, 1:-1] - x_hat[:, 1, :-2, 1:-1]
    return ((du_dx + dv_dy) ** 2).mean()


def disc_loss(real_logits: torch.Tensor, fake_logits: torch.Tensor) -> torch.Tensor:
    """Standard GAN discriminator loss."""
    real_labels = torch.ones_like(real_logits)
    fake_labels = torch.zeros_like(fake_logits)
    return _bce(real_logits, real_labels) + _bce(fake_logits, fake_labels)


def gen_loss(fake_logits: torch.Tensor) -> torch.Tensor:
    """Generator (encoder) adversarial loss — fool the discriminator."""
    return _bce(fake_logits, torch.ones_like(fake_logits))


def poll_stream_unif(arr, slice_x, slice_y, n):
    len = int(np.sqrt(n))
    assert len**2 == n

    gx, gy = np.meshgrid(slice_x, slice_y)

    x_idx = np.linspace(gx[0, 0], gx[0, 0] + gx.shape[1], len, endpoint=False, dtype=int)
    y_idx = np.linspace(gy[0, 0], gy[0, 0] + gy.shape[1], len, endpoint=False, dtype=int)

    ix, iy = np.meshgrid(x_idx, y_idx)
    return np.permute_dims(arr[:, :, ix, iy], (0, 1, 3, 2))

def normalize_video(arr, transform):
    return np.array([np.stack([transform(im[:, :, 0]), transform(im[:, :, 1])], axis=0) for im in arr]).squeeze()

def init_weights(module):
    if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.kaiming_normal_(module.weight, mode='fan_out',
                                nonlinearity='leaky_relu', a=0.2)
        if module.bias is not None:
            nn.init.zeros_(module.bias)

    elif isinstance(module, nn.Linear):
        nn.init.xavier_normal_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)

    elif isinstance(module, (nn.BatchNorm2d, nn.GroupNorm, nn.InstanceNorm2d)):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)

def smoothness_loss(x_hat: torch.Tensor) -> torch.Tensor:
    """Total variation loss — penalises pixel-to-pixel jumps."""
    diff_h = (x_hat[:, :, 1:, :] - x_hat[:, :, :-1, :]).pow(2).mean()
    diff_w = (x_hat[:, :, :, 1:] - x_hat[:, :, :, :-1]).pow(2).mean()
    return diff_h + diff_w

def CWAE1_NS(Y, X0, A, h, t, Noise, parameters):
    NUM_SIM = X0.shape[0]
    N       = X0.shape[-1]
    L       = X0.shape[1:-1]
 
    T  = Y.shape[1]
    dy = Y.shape[2:-1]
 
    sigma = Noise[0]
    gamma = Noise[1]
 
    latent_dims           = parameters['latent_dims']
    Final_Number_ITERATION = parameters.get('Final_Number_ITERATION', 10)
    fix_iter              = parameters.get('fix_iter', 0)
    device_str            = parameters.get('device', 'cuda:1')
    device                = torch.device(device_str)
 
    # (NUM_SIM, T, N, *L) on CPU — mirrors OTF storage convention
    X_CWAE = torch.zeros((NUM_SIM, T, N, *L), device='cpu', dtype=torch.float32)
 
    start_time = time.time()
 
    for k in range(NUM_SIM):
        y = Y[k]
 
        # X0[k]: (*L, N) → (N, *L)
        X_CWAE[k, 0] = torch.from_numpy(
            np.permute_dims(X0[k], (3, 0, 1, 2))
        ).to(torch.float32)
 
        EPOCHS = parameters.get('epochs', 100)
 
        for i in range(T - 1):
 
            # ── 1. Propagate particles ────────────────────────────────
            X_cur = X_CWAE[k, i].to(device)
            X1    = A(X_cur, t[i]) + sigma * torch.randn(*X_cur.shape, device=device)
 
            X1_np = np.permute_dims(
                X1.detach().cpu().numpy(), (1, 2, 3, 0)
            )[np.newaxis]                            # (1, *L, N)
 
            # Build a per-step parameters dict with the current epoch count
            step_params = dict(parameters)
            step_params['epochs'] = EPOCHS
 
            model, _ = train_CWAE1_NS(X1_np, h, gamma, step_params)
 
            # Decay epochs after fix_iter steps, down to Final_Number_ITERATION
            if EPOCHS > Final_Number_ITERATION and i >= fix_iter:
                EPOCHS = max(Final_Number_ITERATION, EPOCHS // 2)
 
            Y1_true = y[i + 1, :]
            Y1_true = torch.from_numpy(Y1_true).to(torch.float32)
            Y1_tiled = np.permute_dims(Y1_true, (3, 0, 1, 2)).repeat(N, 1, 1, 1).to(device)  # (N, *dy)
 
            with torch.no_grad():
                X_mapped = model.sample(Y1_tiled, N, device=device_str)  # (N, *L)
 
            X_CWAE[k, i + 1] = X_mapped.detach().cpu()
 
            del X_cur, X1, X1_np, Y1_true, Y1_tiled, X_mapped, model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
 
            print("Simu#%d/%d, Time Step:%d/%d" % (k + 1, NUM_SIM, i + 1, T - 1))
 
    print("--- CWAE time : %s seconds ---" % (time.time() - start_time))
    return np.permute_dims(X_CWAE.numpy(), (0, 1, 3, 4, 5, 2))
 
 
def train_CWAE1_NS(
    X0: np.ndarray,
    h,
    gamma,
    parameters: dict,
) -> tuple:
 
    latent_dims   = parameters['latent_dims']
    epochs        = parameters.get('epochs',        100)
    batch_size    = parameters.get('BATCH_SIZE',    16)
    lr_enc_xy     = parameters.get('lr1',     1e-3)
    lr_enc_y      = parameters.get('lr2',      1e-3)
    lr_dec        = parameters.get('lr3',        1e-3)
    lr_dec_y      = parameters.get('lr4',      1e-3)
    lr_disc       = parameters.get('lr5',       2e-4)
    adv_weight    = parameters.get('lamb',    0.1)
    div_weight    = parameters.get('div_weight',    0.0)
    smooth_weight = parameters.get('smooth_weight', 0.0)
    val_split     = parameters.get('val_split',     0.15)
    device_str    = parameters.get('device',        'cuda:1')
    num_workers   = parameters.get('num_workers',   0)
    device        = torch.device(device_str)

    print(lr_enc_xy)
    print(adv_weight)
 
    # ── Build dataset ────────────────────────────────────────────────────
    # X0: (NUM_SIM, C, H, W, N)  →  permute  →  (NUM_SIM, N, C, H, W)
    #                              →  reshape  →  (NUM_SIM*N, C, H, W)
    NUM_SIM = X0.shape[0]
    N       = X0.shape[-1]
    L       = X0.shape[1:-1]                          # (C, H, W)
 
    X_np    = np.permute_dims(X0, (0, 4, 1, 2, 3)).reshape(NUM_SIM * N, *L)
    X_torch = torch.from_numpy(X_np).to(torch.float32)
 
    # Evaluate h in batches of N (one simulation worth of particles at a time)
    print(f"[train_CWAE1_NS] Computing h(X0) for {NUM_SIM * N} particles ...")
    Y_list = []
    for s in range(NUM_SIM):
        x_batch = X_torch[s * N : (s + 1) * N].to(device)
        with torch.no_grad():
            y_batch = h(x_batch)
        Y_list.append(y_batch.cpu())
        del x_batch, y_batch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
 
    Y_torch = torch.cat(Y_list, dim=0).to(torch.float32)
 
    if gamma > 0.0:
        Y_torch = Y_torch + gamma * torch.randn_like(Y_torch)
 
    print(f"[train_CWAE1_NS] Dataset:  X {tuple(X_torch.shape)}  "
          f"Y {tuple(Y_torch.shape)}")
 
    # ── Train / val split ────────────────────────────────────────────────
    n_total = X_torch.shape[0]
    n_val   = max(1, int(n_total * val_split))
    n_train = n_total - n_val
 
    perm    = torch.randperm(n_total)
    idx_tr  = perm
    idx_val = perm[n_train:]
 
    # Move full dataset to device once; DataLoader then slices in-place
    X_dev = X_torch.to(device)
    Y_dev = Y_torch.to(device)
 
    train_loader = DataLoader(
        TupleDataset(X_dev[idx_tr],  Y_dev[idx_tr]),
        batch_size=batch_size, shuffle=True,  num_workers=num_workers,
    )
    val_loader = DataLoader(
        TupleDataset(X_dev[idx_val], Y_dev[idx_val]),
        batch_size=batch_size, shuffle=False, num_workers=num_workers,
    )
 
    print(f"[train_CWAE_NS] Train: {n_train}   Val: {n_val}")
 
    # ── Build model and train ────────────────────────────────────────────
    model = VelocityGANAutoencoder1(latent_dims=latent_dims)
 
    history = train(
        model         = model,
        train_loader  = train_loader,
        val_loader    = val_loader,
        epochs        = epochs,
        lr_enc_xy     = lr_enc_xy,
        lr_enc_y      = lr_enc_y,
        lr_dec        = lr_dec,
        lr_dec_y      = lr_dec_y,
        lr_disc       = lr_disc,
        adv_weight    = adv_weight,
        div_weight    = div_weight,
        smooth_weight = smooth_weight,
        loss          = _mse,
        device        = device_str,
    )
 
    model.eval()
    return model, history
 
def train(
    model: VelocityGANAutoencoder1,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 100,
    lr_enc_xy: float = 1e-3,
    lr_enc_y:  float = 1e-3,
    lr_dec:    float = 1e-3,
    lr_dec_y:  float = 1e-3,
    lr_disc:   float = 2e-4,
    adv_weight: float = 0.1,
    div_weight: float = 0.01,
    smooth_weight: float = 0.1,
    yn: int = 9,
    loss: Any = recon_loss,
    device: str = "cpu",
) -> dict:
    """
    Adversarial Autoencoder training with three alternating update steps:
 
      Step 1 — Discriminator update
                   maximise ability to tell prior samples from encoded ones.
      Step 2 — Reconstruction update (per-module optimizers)
                   minimise pixel-wise MSE (+ optional divergence / smoothness penalty).
      Step 3 — Generator (Encoder adversarial) update
                   fool the Discriminator so encoded (u, z) looks like prior.
 
    Args:
        lr_enc_xy    : learning rate for EncoderXY
        lr_enc_y     : learning rate for EncoderY
        lr_dec       : learning rate for Decoder  (x reconstructor)
        lr_dec_y     : learning rate for DecoderY (y reconstructor)
        lr_disc      : learning rate for Discriminator
        adv_weight   : weight on the adversarial generator loss
        div_weight   : weight on divergence-free physics penalty (0 = off)
        smooth_weight: weight on TV smoothness penalty           (0 = off)
 
    Returns:
        history dict with per-epoch losses
    """
    model = model.to(device)
 
    opt_enc_xy = optim.Adam(model.encoder_xy.parameters(), lr=lr_enc_xy, betas=(0.9, 0.999))
    opt_enc_y  = optim.Adam(model.encoder_y.parameters(),  lr=lr_enc_y,  betas=(0.9, 0.999))
    opt_dec    = optim.Adam(model.decoder.parameters(),    lr=lr_dec,    betas=(0.9, 0.999))
    opt_dec_y  = optim.Adam(model.decoder_y.parameters(),  lr=lr_dec_y,  betas=(0.9, 0.999))
    opt_disc   = optim.Adam(model.discriminator.parameters(), lr=lr_disc, betas=(0.9, 0.999))
 
    ae_opts = [opt_enc_xy, opt_enc_y, opt_dec, opt_dec_y]
 
    sched_enc_xy = optim.lr_scheduler.ReduceLROnPlateau(opt_enc_xy, patience=10, factor=0.1)
    sched_enc_y  = optim.lr_scheduler.ReduceLROnPlateau(opt_enc_y,  patience=10, factor=0.1)
    sched_dec    = optim.lr_scheduler.ReduceLROnPlateau(opt_dec,    patience=10, factor=0.1)
    sched_dec_y  = optim.lr_scheduler.ReduceLROnPlateau(opt_dec_y,  patience=10, factor=0.1)
    sched_disc   = optim.lr_scheduler.ReduceLROnPlateau(opt_disc,   patience=10, factor=0.1)
 
    ae_scheds = [sched_enc_xy, sched_enc_y, sched_dec, sched_dec_y]
 
    history = {
        "recon_xy_loss": [], "recon_y_loss": [], "recon_loss": [],
        "disc_loss": [], "gen_loss": [], "val_recon": []
    }
 
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_recon = epoch_recon_xy = epoch_recon_y = epoch_disc = epoch_gen = 0.0
 
        for (x, y) in train_loader:
            B   = x.size(0)
 
            # ── Step 1: Discriminator ─────────────────────────────────
            model.discriminator.zero_grad()
            z_fake   = model.encode(x, y).detach()  # stop encoder grad
            z_real   = torch.randn(B, sum(model.latent_dims), device=device)
 
            d_real   = model.discriminator(z_real)
            d_fake   = model.discriminator(z_fake)
            loss_d   = disc_loss(d_real, d_fake)
 
            loss_d.backward()
            opt_disc.step()
 
            # ── Step 2: Reconstruction (per-module optimizers) ───────
            for o in ae_opts: o.zero_grad()
            u, z  = model.encode_latent(x, y)
            y_hat = model.decoder_y(z)                      # (B, 2, 9, 9)
            x_hat = model.decode(u, z)                      # (B, 2, 48, 48) — cat(u, z)
 
            loss_xy = loss(x_hat, x)
            loss_y  = loss(y_hat, y)
            loss_r  = loss_xy + loss_y + div_weight * divergence_loss(x_hat) + smooth_weight * smoothness_loss(x_hat)
 
            loss_r.backward()
            for o in ae_opts: o.step()
 
            # ── Step 3: Generator (Encoder adversarial) ──────────────
            for o in ae_opts: o.zero_grad()
            z_enc = model.encode(x, y)
            d_enc = model.discriminator(z_enc)
            loss_g = adv_weight * gen_loss(d_enc)
 
            loss_g.backward()
            for o in ae_opts: o.step()
 
            epoch_recon_xy += loss_xy.item() * B
            epoch_recon_y  += loss_y.item()  * B
            epoch_recon    += loss_r.item()  * B
            epoch_disc     += loss_d.item()  * B
            epoch_gen      += loss_g.item()  * B
 
        n_train = len(train_loader.dataset)
        epoch_recon_xy /= n_train
        epoch_recon_y  /= n_train
        epoch_recon    /= n_train
        epoch_disc     /= n_train
        epoch_gen      /= n_train
 
        # ── Validation ───────────────────────────────────────────────
        model.eval()
        val_r = 0.0
        with torch.no_grad():
            for (x, y) in val_loader:
                x = x.to(device)
                y = y.to(device)
                x_hat, y_hat, _ = model.reconstruct(x, y)
                val_r += (loss(x_hat, x) + loss(y_hat, y) + div_weight * divergence_loss(x_hat) + smooth_weight * smoothness_loss(x_hat)).item() * x.size(0)
        val_r /= len(val_loader.dataset)
 
        for s in ae_scheds: s.step(val_r)
        sched_disc.step(epoch_disc)
 
        history["recon_xy_loss"].append(epoch_recon_xy)
        history["recon_y_loss"].append(epoch_recon_y)
        history["recon_loss"].append(epoch_recon)
        history["disc_loss"].append(epoch_disc)
        history["gen_loss"].append(epoch_gen)
        history["val_recon"].append(val_r)
 
        if epoch % 10 == 0 or epoch == 1:
            print(
                f"Epoch {epoch:4d}/{epochs}  "
                f"recon_xy: {epoch_recon_xy:.5f}  "
                f"recon_y: {epoch_recon_y:.5f}  "
                f"recon: {epoch_recon:.5f}  "
                f"disc: {epoch_disc:.5f}  "
                f"gen: {epoch_gen:.5f}  "
                f"val_recon: {val_r:.5f}"
            )
 
    return history