# NOT ACTUALLY USED.
# rope3d.py – pure‑sinusoid 3‑D RoPE implementation + minimal tests
# 

import torch
from typing import Tuple

__all__ = [
    "RotaryEmbedding3D",
    "apply_rotary_3d",
]

# -----------------------------------------------------------------------------
# helpers
# -----------------------------------------------------------------------------

def _build_inv_freq(half_dim: int, base: float = 10_000.0) -> torch.Tensor:
    """Return 1 / (base^(i/half_dim)) for i even (0,2,4,…) so len = half_dim."""
    return 1.0 / (base ** (torch.arange(0, half_dim) / half_dim))


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x[..., ::2], x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).reshape_as(x)


# -----------------------------------------------------------------------------
# core implementation
# -----------------------------------------------------------------------------

class RotaryEmbedding3D(torch.nn.Module):
    """Sinusoidal 3‑D rotary embedding (time / height / width).

    *head_dim* must be divisible by 6 (= two float lanes per axis).
    Memory footprint is O(T+H+W); there are **no learnable parameters**.
    """

    def __init__(self, head_dim: int, T: int = 32, H: int = 16, W: int = 16,
                 base: float = 10_000.0):
        super().__init__()
        self.half_axis_dim = head_dim // 6     # = k (sin/cos each)
        self.axis_dim = 2 * self.half_axis_dim # 2k features per axis

        assert head_dim % 2 == 0, "head_dim must be a multiple of 2"
        self.h_dim = self.half_axis_dim * 2
        self.w_dim = self.half_axis_dim * 2
        self.t_dim = head_dim - self.h_dim - self.w_dim

        # build inverse‑frequency tables once
        self.inv_freq_t = _build_inv_freq(self.t_dim // 2, base)
        self.inv_freq_h = _build_inv_freq(self.h_dim // 2, base)
        self.inv_freq_w = _build_inv_freq(self.w_dim // 2, base)

        # caches are filled in _build_cache
        for name in ("cos_t", "sin_t", "cos_h", "sin_h", "cos_w", "sin_w"):
            self.register_buffer(name, None, persistent=False)

        self._build_cache(T, H, W)

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    def forward(self, T: int, H: int, W: int) -> Tuple[torch.Tensor, ...]:
        # what does this do?
        """Return angle tables, extending internal cache on‑demand."""

        # we will build a new rotary embedding cache if our capacity is exceeded.

        # if too many, we will build a new cache

        if (T > self.cos_t.size(0) or H > self.cos_h.size(0)
                or W > self.cos_w.size(0)):
            self._build_cache(max(T, self.cos_t.size(0)),
                              max(H, self.cos_h.size(0)),
                              max(W, self.cos_w.size(0)))
                            
        return (self.cos_t[:T], self.sin_t[:T],
                self.cos_h[:H], self.sin_h[:H],
                self.cos_w[:W], self.sin_w[:W])

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------

    def _build_cache(self, T: int, H: int, W: int) -> None:
        device = self.inv_freq_t.device
        half = self.half_axis_dim

        def build_axis(sz: int, inv_freq: torch.Tensor):
            # inv_freq
            # einsum, we have T-sized tensor, and a inv_freq
            # 

            freqs = torch.einsum("i,j->ij", torch.arange(sz, device=device), inv_freq) # (sz, d//2)
            emb = torch.repeat_interleave(freqs, 2, dim=-1)
            return emb.cos(), emb.sin()                      # both (sz, 2k)

        self.cos_t, self.sin_t = build_axis(T, self.inv_freq_t)
        self.cos_h, self.sin_h = build_axis(H, self.inv_freq_h)
        self.cos_w, self.sin_w = build_axis(W, self.inv_freq_w)


# -----------------------------------------------------------------------------
# apply function
# -----------------------------------------------------------------------------

def apply_rotary_3d(x: torch.Tensor,
                    cos_t: torch.Tensor, sin_t: torch.Tensor,
                    cos_h: torch.Tensor, sin_h: torch.Tensor,
                    cos_w: torch.Tensor, sin_w: torch.Tensor) -> torch.Tensor:
    """Rotate tensor *x* along T, H, W axes.

    x shape : (B, T, H, W, n_heads, head_dim)
    Returned shape identical.
    """

    B, T, H, W, nH, D = x.shape
    # generate splits
    chunk = (D // 6) * 2
    xh, xw, xt = x[..., :chunk], x[..., chunk:2*chunk], x[..., 2*chunk:]

    print("rotary", xt.shape, cos_t.shape, sin_t.shape)

    xt = (xt * cos_t[None, :, None, None, None, :]) + \
         (_rotate_half(xt) * sin_t[None, :, None, None, None, :])

    # xh: (B, T, H, W, nH, D)
    # cos_h: (H, D)
    xh = (xh * cos_h[None, None, :, None, None, :]) + \
         (_rotate_half(xh) * sin_h[None, None, :, None, None, :])

    # xw: (B, T, H, W, nH, D)
    # cos_w: (W, D)
    xw = (xw * cos_w[None, None, None, :, None, :]) + \
         (_rotate_half(xw) * sin_w[None, None, None, :, None, :])

    return torch.cat((xt, xh, xw), dim=-1)


# -----------------------------------------------------------------------------
# quick tests – run with `python rope3d.py` or `pytest rope3d.py`
# -----------------------------------------------------------------------------

def _test_rotation_preserves_norm() -> None:
    head_dim, nH = 96, 2
    B, T, H, W = 3, 32, 16, 16
    rope = RotaryEmbedding3D(head_dim)

    # generate a random tensor, and apply rotary embedding over the tensors.
    x = torch.randn(B, T, H, W, nH, head_dim)

    y = apply_rotary_3d(x.clone(), *rope(T, H, W))


    assert torch.allclose(x.norm(dim=-1), y.norm(dim=-1), atol=1e-1)


def _test_cache_growth() -> None:
    rope = RotaryEmbedding3D(96, T=4, H=4, W=4)
    rope(10, 4, 4)
    rope(10, 10, 4)
    rope(10, 10, 10)


def _test_shape_integrity() -> None:
    head_dim, nH = 96, 4
    B, T, H, W = 1, 32, 16, 16
    x = torch.randn(B, T, H, W, nH, head_dim)
    rope = RotaryEmbedding3D(head_dim, T, H, W)
    y = apply_rotary_3d(x, *rope(T, H, W))
    assert y.shape == x.shape


if __name__ == "__main__":
    _test_rotation_preserves_norm()
    _test_cache_growth()
    _test_shape_integrity()
    print("All quick tests passed ✅")
