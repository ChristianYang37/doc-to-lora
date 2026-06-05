"""Orthogonal LoRA update (arXiv 2505.11881, adapted to weight space).

When a LoRA update ΔW = A@B is added to a frozen base weight W, keep only the part of
ΔW orthogonal to W:  W + ΔW_⊥ = (1−α)·W + ΔW, with
    α = ⟨A@B, Wᵀ⟩_F / (‖W‖_F² + ε)
(`Wᵀ` because F.linear computes `x @ W.T`).  α is computed via `trace(B@W@A)` WITHOUT
materializing `A@B`, in fp32.  Applied in the LoRA forward as
    out = (1−α)·(x@Wᵀ) + (x@A)@B + C
and in merge as  W ← (1−α)·W + (A@B)ᵀ.

`ENABLED` is the master switch, set at each entrypoint from `cfg.model.lora_ortho_update`
(default ON).  Note: read as a constant by torch.compile at trace time — set once at startup.
"""

from __future__ import annotations

import torch
from torch import Tensor

ENABLED = True  # set from cfg.model.lora_ortho_update at the entrypoints


def ortho_alpha(A: Tensor, B: Tensor, weight: Tensor, eps: float = 1e-8) -> Tensor:
    """α = ⟨A@B, weightᵀ⟩_F / (‖weight‖_F² + ε), per leading batch dim, in fp32.

    ``A``: ``[..., in, r]``  ``B``: ``[..., r, out]``  ``weight``: ``[out, in]`` (nn.Linear).
    Uses ``⟨A@B, Wᵀ⟩_F = trace(B@W@A) = ((B@W) * Aᵀ).sum()`` — no ``A@B`` materialization,
    no ``[r,r]`` product.  Returns ``[...]`` (scalar if A,B are 2-D).
    """
    Wf = weight.float()
    BW = torch.matmul(B.float(), Wf)                              # [..., r, in]
    num = (BW * A.float().transpose(-1, -2)).sum(dim=(-1, -2))    # [...]
    den = (Wf * Wf).sum() + eps
    return num / den


def apply_forward(base: Tensor, lora_out: Tensor, A: Tensor, B: Tensor, weight: Tensor,
                  bias, Lb: int, num_beams: int, out_features: int, in_shape) -> Tensor:
    """Return ``(1−α)·(x@Wᵀ) + lora_out`` (+ bias kept unscaled).

    ``base = F.linear(x, weight, bias)`` (already computed by the caller); we scale only
    the weight part by ``(1−α)`` (per batch element Lb), leaving any bias and the LoRA term.
    """
    alpha = ortho_alpha(A, B, weight).to(base.dtype)             # [Lb]
    wpart = base if bias is None else (base - bias)
    wpart = (wpart.reshape(Lb, num_beams, -1, out_features) * (1 - alpha).view(Lb, 1, 1, 1))
    wpart = wpart.reshape(*in_shape[:-1], out_features)
    base = wpart if bias is None else (wpart + bias)
    return base + lora_out
