"""
Multi-viewpoint augmentation for UAV/airborne IR ATR  (Module B training).

Simulates the full range of viewing conditions encountered by an ISR UAV:
  1. ElevationForeshortening   — oblique angle compresses target footprint
  2. AltitudeVariance          — UAV altitude changes apparent target scale
  3. ThermalBloom              — hot-target heat bleeds into neighbours
  4. AtmosphericScintillation  — low-altitude heat shimmer / turbulence
  5. IRFixedPatternNoise       — FLIR focal-plane-array row/column FPN

All transforms work on batched CUDA tensors (B, C, H, W) normalised to
[-1, 1] (mean=0.5, std=0.5 normalisation space used by Module B).

Usage (integrated into training loop — see module_b_classifier.py):
    from modules.augmentation_viewpoint import MultiViewpointAugmentor
    aug = MultiViewpointAugmentor().to(device)
    imgs = aug(imgs)   # (B, C, 224, 224) → (B, C, 224, 224)
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 1. Elevation foreshortening
# ---------------------------------------------------------------------------

class ElevationForeshortening(nn.Module):
    """
    Compresses the target footprint along a random approach direction,
    simulating an oblique viewing angle from the UAV sensor.

    Physics:
        A target viewed at off-nadir angle α appears compressed by
        factor cos(α) in the approach direction. At nadir (α=0) there
        is no distortion; at α=65° the target is ~42% as tall.

    Off-nadir angle distribution (from operational ISR ranges):
        0–15°   nadir      — dedicated mapping pass
        15–40°  low-oblique — standard surveillance orbit
        40–65°  mid-oblique — typical in-theatre identification

    The approach azimuth φ is sampled uniformly over [0°, 360°] so the
    model sees targets compressed from every horizontal direction.
    """

    def __init__(
        self,
        max_off_nadir_deg: float = 65.0,
        p: float = 0.70,
    ):
        super().__init__()
        self.max_rad = math.radians(max_off_nadir_deg)
        self.p = p

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if torch.rand(1).item() > self.p:
            return x

        B, C, H, W = x.shape
        device = x.device

        # Sample per-image angles
        alphas = torch.rand(B, device="cpu") * self.max_rad   # off-nadir
        phis   = torch.rand(B, device="cpu") * 2 * math.pi   # approach azimuth

        thetas = []
        for i in range(B):
            alpha = float(alphas[i])
            phi   = float(phis[i])
            f     = max(math.cos(alpha), 0.15)   # fwd compression, clamped
            fi    = 1.0 / f                       # backward expansion

            cosφ, sinφ = math.cos(phi), math.sin(phi)

            # Backward affine matrix: T_bwd = R(-φ) @ Scale(1, fi) @ R(φ)
            # where R(φ) = [[cosφ, -sinφ], [sinφ, cosφ]]
            # Gives: target compressed in approach direction in output image
            a00 = cosφ ** 2 + sinφ ** 2 * fi
            a01 = cosφ * sinφ * (fi - 1.0)
            a10 = a01
            a11 = sinφ ** 2 + cosφ ** 2 * fi

            theta = torch.tensor(
                [[a00, a01, 0.0],
                 [a10, a11, 0.0]],
                dtype=torch.float32,
            )
            thetas.append(theta)

        theta_batch = torch.stack(thetas).to(device)   # (B, 2, 3)
        grid = F.affine_grid(theta_batch, (B, C, H, W), align_corners=False)
        return F.grid_sample(
            x, grid,
            mode="bilinear",
            padding_mode="border",   # extend edge pixels — more realistic than zero-pad
            align_corners=False,
        )


# ---------------------------------------------------------------------------
# 2. Altitude-driven scale variance
# ---------------------------------------------------------------------------

class AltitudeVariance(nn.Module):
    """
    Varies the apparent size of the target to simulate UAV altitude changes.

    Two modes are randomly selected:
      zoom-in  (low altitude)  — target fills more of the 224×224 patch
      zoom-out (high altitude) — target is tiny; surrounded by background

    The zoom-out direction is especially important: standard
    RandomResizedCrop never creates patches where the target occupies
    only 5–10% of the frame, but real ISR UAVs often loiter at high
    altitude where targets appear as small blobs.
    """

    def __init__(
        self,
        min_target_fraction: float = 0.07,   # smallest apparent target (high alt)
        max_target_fraction: float = 0.90,   # largest apparent target (low alt)
        p: float = 0.65,
    ):
        super().__init__()
        self.min_frac = min_target_fraction
        self.max_frac = max_target_fraction
        self.p = p

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if torch.rand(1).item() > self.p:
            return x

        B, C, H, W = x.shape
        device = x.device
        results = []

        for i in range(B):
            frac = self.min_frac + torch.rand(1).item() * (self.max_frac - self.min_frac)
            xi   = x[i:i + 1]   # (1, C, H, W)

            if frac >= 1.0:
                results.append(xi)
                continue

            if frac > 0.5:
                # Zoom in: crop the central frac × frac region, upsample to full size
                crop_h = int(H * frac)
                crop_w = int(W * frac)
                top  = (H - crop_h) // 2
                left = (W - crop_w) // 2
                crop = xi[:, :, top: top + crop_h, left: left + crop_w]
                results.append(F.interpolate(crop, (H, W), mode="bilinear", align_corners=False))
            else:
                # Zoom out: shrink the target and embed in an IR-noise background
                small_h = int(H * frac)
                small_w = int(W * frac)
                if small_h < 4 or small_w < 4:
                    results.append(xi)
                    continue
                small   = F.interpolate(xi, (small_h, small_w), mode="bilinear", align_corners=False)

                # Background: border-extended texture of the same image
                bg = F.interpolate(xi, (H, W), mode="bilinear", align_corners=False)
                # Suppress bright objects from the background (they would confuse training)
                bg = torch.clamp(bg, min=-1.0, max=0.0)

                # Place small target at a random position
                top_max  = max(0, H - small_h)
                left_max = max(0, W - small_w)
                top  = int(torch.randint(0, max(top_max, 1),  (1,)).item())
                left = int(torch.randint(0, max(left_max, 1), (1,)).item())

                out = bg.clone()
                out[:, :, top: top + small_h, left: left + small_w] = small
                results.append(out)

        return torch.cat(results, dim=0)


# ---------------------------------------------------------------------------
# 3. Thermal bloom
# ---------------------------------------------------------------------------

class ThermalBloom(nn.Module):
    """
    Simulates the optical/thermal point-spread function of FLIR sensors:
    hot (bright) targets bleed thermal energy into neighbouring pixels.

    Applied only to pixels above a brightness threshold so cooler
    background regions are not affected.
    """

    def __init__(
        self,
        hot_threshold: float = 0.35,   # in [-1, 1] space; ~70% brightness
        bloom_sigma: float = 1.5,
        bloom_strength: float = 0.25,
        p: float = 0.50,
    ):
        super().__init__()
        self.threshold     = hot_threshold
        self.sigma         = bloom_sigma
        self.bloom_strength = bloom_strength
        self.p = p
        # Build fixed Gaussian kernel
        self._kernel = self._make_gaussian_kernel(bloom_sigma)

    @staticmethod
    def _make_gaussian_kernel(sigma: float, size: int = 7) -> torch.Tensor:
        coords = torch.arange(size, dtype=torch.float32) - size // 2
        g1d    = torch.exp(-0.5 * (coords / sigma) ** 2)
        g1d   /= g1d.sum()
        k2d    = g1d[:, None] * g1d[None, :]
        return k2d.view(1, 1, size, size)

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if torch.rand(1).item() > self.p:
            return x

        B, C, H, W = x.shape
        device = x.device

        # Collapse to single-channel luminance for bloom mask
        lum  = x.mean(dim=1, keepdim=True)    # (B, 1, H, W)
        mask = (lum > self.threshold).float()  # hot-pixel mask

        kernel = self._kernel.to(device).expand(1, 1, -1, -1)
        pad    = kernel.shape[-1] // 2

        # Blur hot pixels
        hot_region = lum * mask
        bloomed    = F.conv2d(hot_region, kernel, padding=pad)

        # Add bloom back to all channels, clamped to valid range
        bloom = (bloomed * self.bloom_strength).expand(B, C, H, W)
        return (x + bloom).clamp(-1.0, 1.0)


# ---------------------------------------------------------------------------
# 4. Atmospheric scintillation
# ---------------------------------------------------------------------------

class AtmosphericScintillation(nn.Module):
    """
    Simulates thermal shimmer / atmospheric turbulence seen by low-altitude
    UAVs: a smoothly-varying random local displacement field is applied
    to the image via bilinear grid sampling.

    Scintillation strength increases at lower altitudes (approximated here
    by randomly sampling displacement amplitude).
    """

    def __init__(
        self,
        max_displacement: float = 0.015,   # fraction of image size
        grid_size: int = 6,                 # coarseness of deformation field
        p: float = 0.40,
    ):
        super().__init__()
        self.max_disp = max_displacement
        self.grid_size = grid_size
        self.p = p

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if torch.rand(1).item() > self.p:
            return x

        B, C, H, W = x.shape
        device = x.device

        # Coarse displacement field
        amplitude = torch.rand(1).item() * self.max_disp
        flow_coarse = torch.randn(B, 2, self.grid_size, self.grid_size, device=device) * amplitude

        # Upsample to full resolution
        flow = F.interpolate(flow_coarse, (H, W), mode="bilinear", align_corners=False)

        # Build identity grid and add displacement
        base_grid = F.affine_grid(
            torch.eye(2, 3, device=device).unsqueeze(0).expand(B, -1, -1),
            (B, C, H, W),
            align_corners=False,
        )
        # flow is (B, 2, H, W); grid expects (B, H, W, 2)
        flow_hw = flow.permute(0, 2, 3, 1)
        warped_grid = (base_grid + flow_hw).clamp(-1.0, 1.0)

        return F.grid_sample(
            x, warped_grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=False,
        )


# ---------------------------------------------------------------------------
# 5. IR fixed-pattern noise
# ---------------------------------------------------------------------------

class IRFixedPatternNoise(nn.Module):
    """
    Simulates FLIR focal-plane-array (FPA) fixed-pattern noise:
      - Row FPN: same random offset for every pixel in a row
      - Column FPN: same random offset for every pixel in a column
      - Occasional bad-pixel clusters (dead or hot pixels)

    In real FLIR sensors this pattern is constant but calibrated out;
    in field conditions residual FPN remains after non-uniformity correction.
    """

    def __init__(
        self,
        row_std:    float = 0.018,   # strength of row FPN
        col_std:    float = 0.010,   # strength of column FPN
        bad_pixel_prob: float = 0.003,  # probability of any pixel being bad
        p: float = 0.45,
    ):
        super().__init__()
        self.row_std    = row_std
        self.col_std    = col_std
        self.bad_prob   = bad_pixel_prob
        self.p = p

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if torch.rand(1).item() > self.p:
            return x

        B, C, H, W = x.shape
        device = x.device

        noise = torch.zeros_like(x)

        # Row FPN: (B, C, H, 1) broadcast across W
        noise += torch.randn(B, C, H, 1, device=device) * self.row_std

        # Column FPN: (B, C, 1, W) broadcast across H
        noise += torch.randn(B, C, 1, W, device=device) * self.col_std

        # Bad pixels: sparse salt-and-pepper in thermal space
        if self.bad_prob > 0:
            bad_mask = torch.rand(B, 1, H, W, device=device) < self.bad_prob
            bad_vals = torch.where(
                torch.rand(B, 1, H, W, device=device) > 0.5,
                torch.ones(B, 1, H, W, device=device),    # hot pixel
                -torch.ones(B, 1, H, W, device=device),   # dead pixel
            )
            noise += (bad_mask.float() * bad_vals).expand(B, C, H, W)

        return (x + noise).clamp(-1.0, 1.0)


# ---------------------------------------------------------------------------
# 6. Full multi-viewpoint augmentor
# ---------------------------------------------------------------------------

class MultiViewpointAugmentor(nn.Module):
    """
    Full multi-viewpoint augmentation pipeline for UAV/airborne IR ATR.

    Applies in order:
      1. ElevationForeshortening  — oblique angle, random approach azimuth
      2. AltitudeVariance         — apparent target scale (high ↔ low altitude)
      3. ThermalBloom             — FLIR point-spread heat bleed
      4. AtmosphericScintillation — low-altitude heat shimmer warp
      5. IRFixedPatternNoise      — FLIR FPA row/column noise + dead pixels

    All transforms are probabilistic; pass p_scale ∈ (0, 1] to uniformly
    scale all activation probabilities (e.g. p_scale=0.5 for lighter aug).

    Expected input: (B, C, H, W) float32 normalised to [-1, 1].
    """

    def __init__(self, p_scale: float = 1.0):
        super().__init__()
        s = p_scale
        self.pipeline = nn.Sequential(
            ElevationForeshortening(max_off_nadir_deg=65.0, p=0.70 * s),
            AltitudeVariance(min_target_fraction=0.07, max_target_fraction=0.90, p=0.60 * s),
            ThermalBloom(p=0.50 * s),
            AtmosphericScintillation(p=0.40 * s),
            IRFixedPatternNoise(p=0.45 * s),
        )

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pipeline(x)
