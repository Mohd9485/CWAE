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
    """
    (B, 2, 48, 48) + z_y (B, latent_dim_y) → (B, latent_dim_xy)

    EncoderY is called first; its output z_y is passed directly into
    EncoderXY so the patch encoding is shared and not recomputed.
    EncoderXY no longer has its own patch conv branch.
    """

    def __init__(self, latent_dim: int = 64, latent_dim_y: int = 16):
        super().__init__()
        self.convx = nn.Sequential(
            # 2×48×48 → 32×24×24
            nn.Conv2d(2, 32, 3, stride=2, padding=1),
            nn.LeakyReLU(relu_slope, inplace=True),
            # 32×24×24 → 64×12×12
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(relu_slope, inplace=True),
            # 64×12×12 → 128×6×6
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(relu_slope, inplace=True),
            # 128×6×6 → 256×3×3
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(relu_slope, inplace=True),
        )
        # Input: flat convx output (256*3*3) + z_y (latent_dim_y)
        self.fc = nn.Sequential(
            nn.Linear(256 * 3 * 3 + latent_dim_y, 256),
            nn.LeakyReLU(relu_slope, inplace=True),
            nn.Linear(256, latent_dim),
        )

    def forward(self, x: torch.Tensor, z_y: torch.Tensor) -> torch.Tensor:
        """
        x:   (B, 2, 48, 48)  — full velocity field
        z_y: (B, latent_dim_y) — pre-computed patch latent from EncoderY
        """
        x_flat = torch.flatten(self.convx(x), start_dim=1)  # (B, 256*3*3)
        return self.fc(torch.cat((x_flat, z_y), dim=1))


class EncoderY(nn.Module):
    """(B, 2, 48, 48) → (B, latent_dim)"""

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
        """patch: (B, 2, 9, 9) → (B, latent_dim)"""
        return self.fc(self.conv(patch))

# ═══════════════════════════════════════════════════════════════
#  Decoder  (Generator)
# ═══════════════════════════════════════════════════════════════

class Decoder(nn.Module):
    """(B, latent_dims[0] + 2*9*9) → (B, 2, 48, 48)
    Input is the concatenation of z_xy and flattened y_hat (decoder_y output).
    """

    def __init__(self, latent_dim: int = 64):
        super().__init__()
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
#  Decoder Y  (reconstructs patch from z_y only)
# ═══════════════════════════════════════════════════════════════

class DecoderY(nn.Module):
    """
    (B, latent_dims[1]) → (B, 2, 9, 9)

    Decodes only the patch-specific latent z_y back to the 9×9 input patch.
    Shares the decoder penalty with DecoderXY via the combined reconstruction
    loss:  decoder_loss = decoder_xy_loss + decoder_y_loss + div_weight * div_loss
    """

    def __init__(self, latent_dim: int = 16):
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

    def forward(self, z_y: torch.Tensor) -> torch.Tensor:
        """z_y: (B, latent_dim) → (B, 2, 9, 9)"""
        return self.deconv(self.bn(self.cv(self.unflatten(self.fc(z_y)))))


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

class VelocityGANAutoencoder(nn.Module):
    """Convenience wrapper that holds all three networks."""

    def __init__(self, latent_dims: Sequence[int]):
        super().__init__()
        self.encoder_y     = EncoderY(latent_dims[1])
        # EncoderXY receives z_y directly — no separate patch conv branch
        self.encoder_xy    = EncoderXY(latent_dims[0], latent_dim_y=latent_dims[1])
        self.decoder_y     = DecoderY(latent_dims[1])
        # Decoder input = z_xy (latent_dims[0]) + flattened y_hat (2 * 9 * 9)
        self.decoder       = Decoder(latent_dims[0] + 2 * 9 * 9)
        self.discriminator = Discriminator(latent_dims[0] + latent_dims[1])
        self.latent_dims   = latent_dims

        self.encoder_y.apply(init_weights)
        self.encoder_xy.apply(init_weights)
        self.decoder_y.apply(init_weights)
        self.decoder.apply(init_weights)
        #self.discriminator.apply(init_weights)

    # ── inference helpers ──────────────────────────────────────
    def encode_latent(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (z_xy, z_y) — used by discriminator.
        z_y is computed first and fed into encoder_xy.
        """
        z_y  = self.encoder_y(y)
        z_xy = self.encoder_xy(x, z_y)
        return z_xy, z_y

    def encode(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Concatenated (z_xy, z_y) for the discriminator."""
        z_xy, z_y = self.encode_latent(x, y)
        return torch.cat((z_xy, z_y), dim=1)

    def decode(self, z_xy: torch.Tensor, y_hat: torch.Tensor) -> torch.Tensor:
        """Decode from z_xy and flattened y_hat: input = (z_xy, flatten(y_hat))."""
        y_flat = torch.flatten(y_hat, start_dim=1)          # (B, 2*9*9)
        return self.decoder(torch.cat((z_xy, y_flat), dim=1))

    def reconstruct(self, x: torch.Tensor, y: torch.Tensor):
        z_xy, z_y = self.encode_latent(x, y)                # z_y feeds into z_xy
        y_hat = self.decoder_y(z_y)                         # (B, 2, 9, 9)
        x_hat = self.decode(z_xy, y_hat)                    # (B, 2, 48, 48)
        z     = torch.cat((z_xy, z_y), dim=1)               # for discriminator
        return x_hat, y_hat, z

    def sample(self, y: torch.Tensor, n: int, device: str = "cpu") -> torch.Tensor:
        """Draw n samples from the prior and decode them."""
        z_y  = self.encoder_y(y)
        y_hat = self.decoder_y(z_y)                         # (B, 2, 9, 9)
        z_xy = torch.randn(n, self.latent_dims[0], device=device)
        return self.decode(z_xy, y_hat)


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
    diff_h = (x_hat[:, :, 1:, :] - x_hat[:, :, :-1, :]).pow(2)
    diff_w = (x_hat[:, :, :, 1:] - x_hat[:, :, :, :-1]).pow(2)
    return diff_h.mean() + diff_w.mean()

# ═══════════════════════════════════════════════════════════════
#  Training loop
# ═══════════════════════════════════════════════════════════════

def train(
    model: VelocityGANAutoencoder,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 100,
    lr_ae: float = 1e-3,
    lr_disc: float = 2e-4,
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
      Step 2 — Reconstruction update (Encoder + Decoder)
                   minimise pixel-wise MSE (+ optional divergence penalty).
      Step 3 — Generator (Encoder) adversarial update
                   fool the Discriminator so encoded z looks like prior.

    Args:
        lr_ae    : learning rate for Encoder + Decoder
        lr_disc  : learning rate for Discriminator (usually smaller)
        adv_weight: weight on the adversarial generator loss
        div_weight: weight on divergence-free physics penalty (0 = off)

    Returns:
        history dict with per-epoch losses
    """
    model = model.to(device)

    opt_ae   = optim.Adam(
        list(model.encoder_xy.parameters()) + list(model.encoder_y.parameters()) +
        list(model.decoder.parameters())    + list(model.decoder_y.parameters()),
        lr=lr_ae, betas=(0.9, 0.999)
    )
    opt_disc = optim.Adam(
        model.discriminator.parameters(),
        lr=lr_disc, betas=(0.9, 0.999)
    )

    sched_ae   = optim.lr_scheduler.ReduceLROnPlateau(opt_ae,   patience=10, factor=0.1)
    sched_disc = optim.lr_scheduler.ReduceLROnPlateau(opt_disc, patience=10, factor=0.1)

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

            # ── Step 2: Reconstruction (EncoderY → EncoderXY → Decoders) ──
            opt_ae.zero_grad()
            z_xy, z_y = model.encode_latent(x, y)          # z_y feeds into z_xy
            y_hat = model.decoder_y(z_y)                    # (B, 2, 9, 9)
            x_hat = model.decode(z_xy, y_hat)               # (B, 2, 48, 48)

            loss_xy = loss(x_hat, x)
            loss_y  = loss(y_hat, y)
            # Shared decoder penalty: total loss backpropagates through both decoders
            loss_r  = loss_xy + loss_y + div_weight * divergence_loss(x_hat) + smooth_weight * smoothness_loss(x_hat)

            loss_r.backward()
            opt_ae.step()

            # ── Step 3: Generator (Encoder adversarial) ──────────────
            opt_ae.zero_grad()
            z_enc = model.encode(x, y)
            d_enc = model.discriminator(z_enc)
            loss_g = adv_weight * gen_loss(d_enc)

            loss_g.backward()
            opt_ae.step()

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

        sched_ae.step(val_r)
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


# ═══════════════════════════════════════════════════════════════
#  Visualisation
# ═══════════════════════════════════════════════════════════════

def plot_losses(history: dict, save_path=None, npy_dir=None):
    if npy_dir:
        import os
        os.makedirs(npy_dir, exist_ok=True)
        for k, v in history.items():
            np.save(os.path.join(npy_dir, f"{k}.npy"), np.array(v))

    fig, axes = plt.subplots(1, 2, figsize=(20, 10))

    axes[0].plot(history["recon_xy_loss"], label="Train recon_xy (full field)")
    axes[0].plot(history["recon_y_loss"],  label="Train recon_y (patch)")
    axes[0].plot(history["recon_loss"],    label="Train recon (total)", linestyle="--")
    axes[0].plot(history["val_recon"],     label="Val recon (total)")
    axes[0].set_title("Reconstruction Loss (MSE)")
    axes[0].set_xlabel("Epoch"); axes[0].legend()

    axes[1].plot(history["disc_loss"], label="Discriminator")
    axes[1].plot(history["gen_loss"],  label="Generator (Encoder)")
    axes[1].set_title("Adversarial Losses")
    axes[1].set_xlabel("Epoch"); axes[1].legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    else:
        plt.show()


def plot_reconstruction(model, sample, device="cpu", save_path=None, npy_dir=None):
    model.eval()
    with torch.no_grad():
        x_hat, _, _ = model.reconstruct(
            sample[0].unsqueeze(0).to(device),
            sample[1].unsqueeze(0).to(device)
        )

    orig  = sample[0].cpu().numpy()
    recon = x_hat.squeeze(0).cpu().numpy()
    err   = orig - recon

    if npy_dir:
        import os
        os.makedirs(npy_dir, exist_ok=True)
        np.save(os.path.join(npy_dir, "recon_orig.npy"), orig)
        np.save(os.path.join(npy_dir, "recon_pred.npy"), recon)
        np.save(os.path.join(npy_dir, "recon_err.npy"), err)

    labels = ["u (x-vel)", "v (y-vel)"]
    fig, axes = plt.subplots(2, 3, figsize=(20, 10))

    for row, lbl in enumerate(labels):
        vmin, vmax = orig[row].min(), orig[row].max()

        im0 = axes[row, 0].imshow(orig[row], cmap="RdBu_r", vmin=vmin, vmax=vmax)
        axes[row, 0].set_title(f"{lbl} — original")
        fig.colorbar(im0, ax=axes[row, 0])

        im1 = axes[row, 1].imshow(recon[row], cmap="RdBu_r", vmin=vmin, vmax=vmax)
        axes[row, 1].set_title(f"{lbl} — reconstructed")
        fig.colorbar(im1, ax=axes[row, 1])

        im2 = axes[row, 2].imshow(err[row], cmap="bwr")
        axes[row, 2].set_title(f"{lbl} — error")
        fig.colorbar(im2, ax=axes[row, 2])

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    else:
        plt.show()


def plot_samples(model, y, n=4, device="cpu", save_path=None, npy_dir=None):
    model.eval()

    y = y.to(device)

    with torch.no_grad():
        samples_np  = model.sample(y[:n], n, device=device).cpu().numpy()

    if npy_dir:
        import os
        os.makedirs(npy_dir, exist_ok=True)
        np.save(os.path.join(npy_dir, "samples.npy"), samples_np)
        np.save(os.path.join(npy_dir, "y.npy"), y.cpu().numpy())




# ═══════════════════════════════════════════════════════════════
#  Diagnostic plots
# ═══════════════════════════════════════════════════════════════

def plot_diagnostics(
    model: "VelocityGANAutoencoder | None" = None,
    x: "torch.Tensor | None" = None,
    y: "torch.Tensor | None" = None,
    particle_counts: list[int] = [100, 200, 300, 500, 1000],
    device: str = "cpu",
    save_path: str | None = None,
    npy_dir: str | None = None,
):
    """
    Four-panel diagnostic figure for a single test sample.

    Can be called in two modes:

      1. Live inference (model + tensors):
            plot_diagnostics(model, x, y, ...)

      2. From saved .npy files (no model needed):
            plot_diagnostics(npy_dir="path/to/dir", save_path="out.png")
         Expects: diag_true.npy, diag_mean.npy, diag_mse_map.npy,
                  diag_samples.npy, diag_particle_counts.npy, diag_mse_vs_n.npy

    Panel layout (2 rows x 4 cols):
      Row 0: u-component  |  Row 1: v-component
      Col 0 - True field
      Col 1 - Particle mean  (average of max(particle_counts) samples)
      Col 2 - MSE heatmap   (pixel-wise squared error, mean vs true)
      Col 3 - Particles vs MSE line plot  (shared across both rows)

    Args:
        model:           trained VelocityGANAutoencoder (None if loading from npy)
        x:               true full field,  (2, 48, 48) or (1, 2, 48, 48)
        y:               patch observation, (2, 9, 9)  or (1, 2, 9, 9)
        particle_counts: list of N values for the particles-vs-MSE curve
        device:          torch device string
        save_path:       if given, save figure here; else call plt.show()
        npy_dir:         directory containing diag_*.npy files (activates load mode)
    """
    import os

    # ══════════════════════════════════════════════════════════════════════
    #  Mode A: load from .npy files
    # ══════════════════════════════════════════════════════════════════════
    if npy_dir is not None:
        print(f"Loading plotting data from {npy_dir}/diag_*.npy")
        true_np        = np.load(os.path.join(npy_dir, "diag_true.npy"))
        mean_np        = np.load(os.path.join(npy_dir, "diag_mean.npy"))
        mse_map        = np.load(os.path.join(npy_dir, "diag_mse_map.npy"))
        samples_np     = np.load(os.path.join(npy_dir, "diag_samples.npy"))
        particle_counts = np.load(os.path.join(npy_dir, "diag_particle_counts.npy")).tolist()
        mse_vs_n       = np.load(os.path.join(npy_dir, "diag_mse_vs_n.npy")).tolist()
        n_max          = int(max(particle_counts))

    # ══════════════════════════════════════════════════════════════════════
    #  Mode B: live inference
    # ══════════════════════════════════════════════════════════════════════
    else:
        assert model is not None and x is not None and y is not None,             "Provide either (model, x, y) for live inference or npy_dir to load saved data."

        model.eval()

        # normalise to (1, C, H, W)
        if x.dim() == 3:
            x = x.unsqueeze(0)
        if y.dim() == 3:
            y = y.unsqueeze(0)
        x = x.to(device)
        y = y.to(device)

        n_max = max(particle_counts)

        # draw n_max samples once, reuse subsets for the curve
        with torch.no_grad():
            y_tiled  = y.expand(n_max, -1, -1, -1)
            samples  = model.sample(y_tiled, n_max, device=device)

        samples_np = samples.cpu().numpy()               # (n_max, 2, 48, 48)
        true_np    = x.squeeze(0).cpu().numpy()          # (2, 48, 48)

        # particle mean using all n_max samples
        mean_np = samples_np.mean(axis=0)                # (2, 48, 48)

        # pixel-wise MSE heatmap
        mse_map = (mean_np - true_np) ** 2               # (2, 48, 48)

        # particles vs scalar MSE curve
        mse_vs_n = []
        for n in particle_counts:
            mean_n = samples_np[:n].mean(axis=0)
            mse_vs_n.append(float(((mean_n - true_np) ** 2).mean()))

        # save .npy files
        if npy_dir:
            import os
            os.makedirs(npy_dir, exist_ok=True)

            np.save(os.path.join(npy_dir, "true.npy"), true_np)
            np.save(os.path.join(npy_dir, "mean.npy"), mean_np)
            np.save(os.path.join(npy_dir, "mse_map.npy"), mse_map)
            np.save(os.path.join(npy_dir, "samples.npy"), samples_np)
            np.save(os.path.join(npy_dir, "particle_counts.npy"), np.array(particle_counts))
            np.save(os.path.join(npy_dir, "mse_vs_n.npy"), np.array(mse_vs_n))

    # ══════════════════════════════════════════════════════════════════════
    #  Build figure
    # ══════════════════════════════════════════════════════════════════════
    # Columns: True | Mean | MSE heatmap | Particles-vs-MSE
    # The last column spans both rows → use gridspec
    fig = plt.figure(figsize=(20, 10))
    gs  = fig.add_gridspec(
        2, 4,
        width_ratios=[1, 1, 1, 1.1],
        hspace=0.35, wspace=0.35,
    )

    comp_labels = ["u  (x-vel)", "v  (y-vel)"]
    cmap_field  = "RdBu_r"
    cmap_err    = "hot"

    # Shared colour limits per component
    vlim = [max(abs(true_np[c]).max(), abs(mean_np[c]).max()) for c in range(2)]

    axes_grid = [[fig.add_subplot(gs[r, c]) for c in range(3)] for r in range(2)]
    ax_curve  = fig.add_subplot(gs[:, 3])   # spans both rows

    for row in range(2):
        lbl  = comp_labels[row]
        vmax = vlim[row]

        # ── Col 0: True ───────────────────────────────────────────────────
        im0 = axes_grid[row][0].imshow(
            true_np[row], cmap=cmap_field,
            vmin=-vmax, vmax=vmax, origin="lower",
        )
        axes_grid[row][0].set_title(f"True  {lbl}")
        cb0 = fig.colorbar(im0, ax=axes_grid[row][0], fraction=0.046, pad=0.04)
        cb0.formatter = plt.matplotlib.ticker.ScalarFormatter(useMathText=True)
        cb0.formatter.set_powerlimits((0, 0)); cb0.update_ticks()

        # ── Col 1: Particle mean ──────────────────────────────────────────
        im1 = axes_grid[row][1].imshow(
            mean_np[row], cmap=cmap_field,
            vmin=-vmax, vmax=vmax, origin="lower",
        )
        axes_grid[row][1].set_title(
            f"Particle mean  {lbl}\n(N={n_max})"
        )
        cb1 = fig.colorbar(im1, ax=axes_grid[row][1], fraction=0.046, pad=0.04)
        cb1.formatter = plt.matplotlib.ticker.ScalarFormatter(useMathText=True)
        cb1.formatter.set_powerlimits((0, 0)); cb1.update_ticks()

        # ── Col 2: MSE heatmap ────────────────────────────────────────────
        im2 = axes_grid[row][2].imshow(
            mse_map[row], cmap=cmap_err, origin="lower",
        )
        axes_grid[row][2].set_title(f"MSE heatmap  {lbl}")
        cb2 = fig.colorbar(im2, ax=axes_grid[row][2], fraction=0.046, pad=0.04)
        cb2.formatter = plt.matplotlib.ticker.ScalarFormatter(useMathText=True)
        cb2.formatter.set_powerlimits((0, 0)); cb2.update_ticks()

    # ── Col 3: Particles vs MSE line plot ─────────────────────────────────
    ax_curve.plot(
        particle_counts, mse_vs_n,
        marker="o", linewidth=2, markersize=6,
        color="steelblue", label="encoder_xy",
    )
    ax_curve.set_xlabel("Number of particles (N)")
    ax_curve.set_ylabel("MSE  (mean field vs true)")
    ax_curve.set_title("Particles vs MSE")
    ax_curve.set_xticks(particle_counts)
    ax_curve.tick_params(axis="x", rotation=30)
    ax_curve.legend()
    ax_curve.grid(True, alpha=0.3)
    # Scientific notation on y-axis
    ax_curve.yaxis.set_major_formatter(plt.matplotlib.ticker.ScalarFormatter(useMathText=True))
    ax_curve.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))

    fig.suptitle(
        "Velocity Field Reconstruction Diagnostics  (encoder_xy)",
        y=1.01,
    )

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        print(f"Saved → {save_path}")
    else:
        plt.show()

    return {
        "true":            true_np,
        "mean":            mean_np,
        "mse_map":         mse_map,
        "samples":         samples_np,
        "particle_counts": particle_counts,
        "mse_vs_n":        mse_vs_n,
    }

def plot_sample_heatmap(model, x, y, n=500, device="cpu", save_path=None, npy_dir=None):
    model.eval()

    y = y.to(device)

    with torch.no_grad():
        samples_np  = model.sample(y, n, device=device).cpu().numpy()

    mean_np    = samples_np.mean(axis=0)
    std_np     = samples_np.std(axis=0)
    cv_np      = std_np / (np.abs(mean_np) + 1e-8)

    plt.imshow(mean_np[0], cmap="RdBu_r")
    plt.imshow(mean_np[-1], cmap="RdBu_r")

    if npy_dir:
        import os
        os.makedirs(npy_dir, exist_ok=True)
        np.save(os.path.join(npy_dir, "samples.npy"), samples_np)
        np.save(os.path.join(npy_dir, "mean.npy"), mean_np)
        np.save(os.path.join(npy_dir, "std.npy"), std_np)
        np.save(os.path.join(npy_dir, "cv.npy"), cv_np)
        np.save(os.path.join(npy_dir, "x_true.npy"), x.cpu().numpy())


# ═══════════════════════════════════════════════════════════════
#  Demo entry-point
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    seed = 7
    np.random.seed(seed)
    torch.manual_seed(seed)

    load = False
    fine_tune = False
    DEVICE     = "cuda:1" if torch.cuda.is_available() else "cpu"
    EPOCHS = 100
    LATENT_DIMS   = [40, 40]
    BATCH_SIZE    = 8
    adv_weight    = 0.607319
    div_weight    = 0.002
    smooth_weight = 0.03
    lr_ae         = 0.00144917
    lr_disc       = 0.00003022
    yn = 9*9
    gamma = 0.001

    print(f"Device: {DEVICE}")

    transform=transforms.Compose([ transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

    # Data
    data  = np.load("data/CylinderBig.npy")
    data = data[:, 1:49, 1:49, :]
    X = normalize_video(data, transform)
    Y = poll_stream_unif(X, np.arange(0, data.shape[2], dtype=int), np.arange(0, data.shape[3], dtype=int), yn)
    Y_noise = Y + 0.01 * np.random.normal(size=(X.shape[0], 2, 9, 9))
    X = torch.Tensor(X).to(torch.float32).to(DEVICE)
    Y = torch.Tensor(Y).to(torch.float32).to(DEVICE)
    Y_noise = torch.Tensor(Y_noise).to(torch.float32).to(DEVICE)
    split = int(0.85 * len(X))

    train_loader = DataLoader(TupleDataset(X[:split], Y_noise[:split]),
                              batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(TupleDataset(X[split:], Y_noise[split:]),
                              batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # Model
    model = VelocityGANAutoencoder(latent_dims=LATENT_DIMS)
    n_enc_y   = sum(p.numel() for p in model.encoder_y.parameters()     if p.requires_grad)
    n_enc_xy  = sum(p.numel() for p in model.encoder_xy.parameters()    if p.requires_grad)
    n_dec     = sum(p.numel() for p in model.decoder.parameters()        if p.requires_grad)
    n_dec_y   = sum(p.numel() for p in model.decoder_y.parameters()      if p.requires_grad)
    n_disc    = sum(p.numel() for p in model.discriminator.parameters()  if p.requires_grad)
    print(f"Parameters — EncoderY: {n_enc_y:,}  EncoderXY: {n_enc_xy:,}  (EncoderXY receives z_y from EncoderY)\n"
          f"            Decoder: {n_dec:,}  DecoderY: {n_dec_y:,}  Disc: {n_disc:,}")


    print(f"{seed} {LATENT_DIMS} {BATCH_SIZE} {adv_weight} {div_weight} {smooth_weight} {lr_ae} {lr_disc}")

    if load:
        model.load_state_dict(torch.load("AE1.pth").state_dict())
        model.to(DEVICE)
    
    if (load and fine_tune) or (not load and not fine_tune):
        history = train(
            model, train_loader, val_loader,
            epochs=EPOCHS,
            lr_ae=lr_ae, lr_disc=lr_disc,
            adv_weight=adv_weight,
            div_weight=div_weight,
            smooth_weight=smooth_weight,
            yn=yn,
            device=DEVICE,
            loss=_mse
        )

    model.eval()

    plt.figure()
    plt.subplot(121)
    plt.imshow(Y_noise.cpu()[-1][0], cmap="RdBu_r", origin="lower")
    plt.subplot(122)
    plt.imshow(Y_noise.cpu()[-1][1], cmap="RdBu_r", origin="lower")

    # Plots
    enc_xy = model.encode(torch.unsqueeze(X[-1], 0), torch.unsqueeze(Y[-1], 0))
    enc_y = model.encoder_y(torch.unsqueeze(Y[-1], 0))

    Y_test = torch.tile(Y[-1], (1000, 1, 1, 1))

    Z_xy_hat = torch.randn((1000,LATENT_DIMS[0]), device=DEVICE)
    Z_y_test = model.encoder_y(Y_test) 

    torch.save(model.state_dict(), "AE.pth")

    plt.figure()
    plt.subplot(121)
    plt.scatter(torch.randn(1000), torch.randn(1000), alpha=0.2)
    plt.scatter(Z_xy_hat[:1000, 0].detach().cpu(), Z_y_test[:1000, 0].detach().cpu(), marker='x', alpha=0.2)

    if not load: plot_losses(history)
    plot_reconstruction(model, (X[-1], Y[-1]), device=DEVICE,  npy_dir="data/NS/recon")
    plot_samples(model, Y_test, n=5, device=DEVICE, npy_dir="data/NS/samples")
    plot_sample_heatmap(model, X[-1], Y_test, 1000, DEVICE, npy_dir="data/NS/heatmap")

    plt.show()