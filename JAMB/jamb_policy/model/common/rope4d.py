"""
4D Rotary Position Embedding (RoPE) over (x, y, z, t) coordinates.

Each attention head's dimension is split into four axis groups; each group
gets a standard 1D rotary embedding driven by that axis of the token's
coordinate. Spatial axes (x, y, z) are in world-frame meters and share one
base frequency / scale; the time axis counts action steps and gets its own.

Tokens without a natural coordinate (CLS, diffusion-timestep token) are
excluded via `rope_mask` — their q/k vectors pass through unrotated, which
is exactly a zero rotation. We use an explicit mask rather than zero
coordinates because (0, 0, 0) is a real location in the robot workspace.

Coordinate sources (world frame, from the tracks dataset):
  - visual (Pi3 / DinoV2) patch tokens: image_patch_centre_3d_position
  - per-arm state tokens: that arm's current EEF xyz
  - per-arm action tokens: that arm's current EEF xyz, t = step index
  - track query tokens: their patch center, t = 0
"""
from typing import Optional, Tuple

import torch
import torch.nn as nn


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Standard RoPE pairing: (x1, x2) -> (-x2, x1) on interleaved pairs."""
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).flatten(-2)


class RoPE4D(nn.Module):
    """
    Precomputes per-axis inverse frequencies; builds cos/sin tables from
    token coordinates and applies the rotation to q/k tensors shaped
    [B, n_heads, N, head_dim].

    head_dim is split as: x, y, z each get `spatial_dims_per_axis` dims and
    t gets the remainder; every group size must be even. With head_dim=128
    the default split is 32/32/32/32.
    """

    def __init__(
        self,
        head_dim: int,
        spatial_base: float = 100.0,
        time_base: float = 100.0,
        spatial_scale: float = 0.5,
        time_scale: float = 16.0,
    ):
        super().__init__()
        assert head_dim % 8 == 0, f"head_dim must be divisible by 8, got {head_dim}"
        self.head_dim = head_dim

        # Split head_dim across (x, y, z, t); keep each group even
        per_axis = head_dim // 4
        per_axis -= per_axis % 2
        dims = [per_axis, per_axis, per_axis, head_dim - 3 * per_axis]
        assert all(d > 0 and d % 2 == 0 for d in dims), f"Bad axis split {dims}"
        self.axis_dims = dims

        # spatial_scale/time_scale normalize raw coordinates to ~O(1) before
        # the rotary phase: workspace xyz spans roughly +-0.5 m; t spans the
        # action horizon.
        self.spatial_scale = spatial_scale
        self.time_scale = time_scale

        for i, (name, base) in enumerate(
            zip(["x", "y", "z", "t"],
                [spatial_base, spatial_base, spatial_base, time_base])
        ):
            d = dims[i]
            inv_freq = 1.0 / (base ** (torch.arange(0, d, 2).float() / d))
            self.register_buffer(f"inv_freq_{name}", inv_freq, persistent=False)

    def build_cos_sin(
        self,
        coords: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            coords: [B, N, 4] (x, y, z in meters; t in action steps)

        Returns:
            cos, sin: [B, N, head_dim]
        """
        assert coords.shape[-1] == 4, f"Expected 4D coords, got {coords.shape}"
        scales = coords.new_tensor(
            [self.spatial_scale, self.spatial_scale, self.spatial_scale, self.time_scale]
        )
        normed = coords / scales  # [B, N, 4]

        parts_cos, parts_sin = [], []
        for i, name in enumerate(["x", "y", "z", "t"]):
            inv_freq = getattr(self, f"inv_freq_{name}")  # [d_i/2]
            phase = normed[..., i : i + 1] * inv_freq  # [B, N, d_i/2]
            phase = phase.repeat_interleave(2, dim=-1)  # [B, N, d_i]
            parts_cos.append(phase.cos())
            parts_sin.append(phase.sin())
        return torch.cat(parts_cos, dim=-1), torch.cat(parts_sin, dim=-1)

    def apply(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        rope_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Rotate q or k.

        Args:
            x: [B, n_heads, N, head_dim]
            cos, sin: [B, N, head_dim] from build_cos_sin
            rope_mask: [B, N] bool, True where rotation applies. Tokens with
                False pass through unrotated.

        Returns:
            [B, n_heads, N, head_dim]
        """
        cos = cos.unsqueeze(1)  # [B, 1, N, head_dim]
        sin = sin.unsqueeze(1)
        rotated = x * cos + _rotate_half(x) * sin
        if rope_mask is None:
            return rotated
        keep = rope_mask.unsqueeze(1).unsqueeze(-1)  # [B, 1, N, 1]
        return torch.where(keep, rotated, x)
