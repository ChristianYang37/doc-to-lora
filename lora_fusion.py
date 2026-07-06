"""Dynamic per-layer LoRA fusion for multi-chunk SHINE.

The static multi-chunk path pre-combines page LoRAs into one rank-stacked LoRA.
This module keeps the page axis alive and lets each decoder layer compute
``softmax(q @ k)`` routing weights at forward/decode time.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
from torch import Tensor


MODE_KEY = "_fusion_mode"
KEYS_KEY = "_fusion_keys"
MASK_KEY = "_fusion_chunk_mask"
TEMP_KEY = "_fusion_temp"
TOPK_KEY = "_fusion_top_k"
RESCALE_KEY = "_fusion_rescale"
DYNAMIC_LEAF_KEY = "_fusion_dynamic"


def is_dynamic_loradict(loradict) -> bool:
    return isinstance(loradict, dict) and loradict.get(MODE_KEY) == "dynamic"


def is_dynamic_leaf(leaf) -> bool:
    return isinstance(leaf, dict) and bool(leaf.get(DYNAMIC_LEAF_KEY, False))


def make_dynamic_loradict(
    layer_loradict: dict,
    fusion_keys: Tensor,
    chunk_mask: Tensor,
    *,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    mix_rescale: float = 1.0,
) -> dict:
    """Attach dynamic-fusion metadata to a padded per-chunk loradict.

    ``fusion_keys`` is ``[B, num_layers, max_chunks, hidden]`` and
    ``chunk_mask`` is ``[B, max_chunks]``.
    """
    out = dict(layer_loradict)
    out[MODE_KEY] = "dynamic"
    out[KEYS_KEY] = fusion_keys
    out[MASK_KEY] = chunk_mask.to(torch.bool)
    out[TEMP_KEY] = float(temperature)
    out[TOPK_KEY] = None if top_k is None or int(top_k) <= 0 else int(top_k)
    out[RESCALE_KEY] = float(mix_rescale)
    return out


def _attach_leaf_metadata(leaf: dict, meta: dict, layer_idx: int, group: str, proj: str) -> dict:
    wrapped = dict(leaf)
    wrapped[DYNAMIC_LEAF_KEY] = True
    wrapped[KEYS_KEY] = meta[KEYS_KEY]
    wrapped[MASK_KEY] = meta[MASK_KEY]
    wrapped[TEMP_KEY] = meta[TEMP_KEY]
    wrapped[TOPK_KEY] = meta[TOPK_KEY]
    wrapped[RESCALE_KEY] = meta[RESCALE_KEY]
    wrapped["_fusion_cache"] = meta["_fusion_cache"]
    wrapped["_fusion_layer_idx"] = layer_idx
    wrapped["_fusion_group"] = group
    wrapped["_fusion_proj"] = proj
    return wrapped


def get_layer_loradict(loradict, layer_idx: int, cache: Optional[dict] = None):
    """Return a layer dict, wrapping leaves with dynamic routing metadata if needed."""
    if loradict is None:
        return None
    if not is_dynamic_loradict(loradict):
        return loradict[layer_idx] if isinstance(loradict, dict) and layer_idx in loradict else None
    if layer_idx not in loradict:
        return None

    if cache is None:
        cache = {}
    meta = {
        KEYS_KEY: loradict[KEYS_KEY][:, layer_idx],
        MASK_KEY: loradict[MASK_KEY],
        TEMP_KEY: loradict.get(TEMP_KEY, 1.0),
        TOPK_KEY: loradict.get(TOPK_KEY, None),
        RESCALE_KEY: loradict.get(RESCALE_KEY, 1.0),
        "_fusion_cache": cache,
    }

    layer = loradict[layer_idx]
    wrapped = {}
    for group, projs in layer.items():
        if not isinstance(projs, dict):
            continue
        wrapped[group] = {}
        for proj, leaf in projs.items():
            wrapped[group][proj] = _attach_leaf_metadata(leaf, meta, layer_idx, group, proj)
    return wrapped


def get_leaf_loradict(loradict, layer_idx: int, group: str, proj: str, cache_by_layer: Optional[dict] = None):
    cache = None
    if cache_by_layer is not None:
        cache = cache_by_layer.setdefault(layer_idx, {})
    layer = get_layer_loradict(loradict, layer_idx, cache=cache)
    if layer is None or group not in layer or proj not in layer[group]:
        return None
    return layer[group][proj]


def compute_layer_weights(q: Tensor, leaf: dict) -> Tensor:
    """Compute per-token chunk weights from layer query states and M2P keys.

    ``q`` is ``[B * beams, seq, hidden]``. Keys are ``[B, chunks, hidden]``.
    Returns ``[B, beams, seq, chunks]``.
    """
    keys = leaf[KEYS_KEY].to(device=q.device)
    mask = leaf[MASK_KEY].to(device=q.device, dtype=torch.bool)
    bsz, chunks, key_dim = keys.shape
    if q.shape[0] % bsz != 0:
        raise RuntimeError(f"input batch {q.shape[0]} must be a multiple of fusion batch {bsz}")

    beams = q.shape[0] // bsz
    qv = q.reshape(bsz, beams, -1, q.shape[-1])
    dim = min(qv.shape[-1], key_dim)
    if dim <= 0:
        raise RuntimeError("dynamic LoRA fusion received an empty q/key dimension")

    scores = torch.einsum("bnsd,bcd->bnsc", qv[..., :dim].float(), keys[..., :dim].float())
    scores = scores / math.sqrt(float(dim))
    scores = scores.masked_fill(~mask[:, None, None, :], float("-inf"))

    top_k = leaf.get(TOPK_KEY, None)
    if top_k is not None and 0 < int(top_k) < chunks:
        keep_idx = scores.topk(int(top_k), dim=-1).indices
        keep = torch.zeros_like(scores, dtype=torch.bool)
        keep.scatter_(-1, keep_idx, True)
        scores = scores.masked_fill(~keep, float("-inf"))

    temp = max(float(leaf.get(TEMP_KEY, 1.0)), 1e-6)
    weights = torch.softmax(scores / temp, dim=-1)
    weights = torch.nan_to_num(weights, nan=0.0)
    return weights.to(dtype=q.dtype) * float(leaf.get(RESCALE_KEY, 1.0))


def _cached_or_compute_weights(query_like: Tensor, leaf: dict) -> Tensor:
    cache = leaf.get("_fusion_cache", None)
    cache_key = "weights"
    weights = None if cache is None else cache.get(cache_key)
    recompute = leaf.get("_fusion_proj") == "q"

    if weights is not None:
        seq_len = query_like.shape[-2] if query_like.dim() >= 3 else 1
        if weights.shape[2] != seq_len:
            recompute = True
    if weights is None or recompute:
        weights = compute_layer_weights(query_like, leaf)
        if cache is not None:
            cache[cache_key] = weights
    return weights


def dynamic_lora_update(input: Tensor, leaf: dict, weights: Tensor) -> Tensor:
    """Apply weighted per-chunk LoRA updates. Returns only the LoRA delta."""
    A = leaf["A"].to(device=input.device, dtype=input.dtype)  # [B,C,in,r]
    B = leaf["B"].to(device=input.device, dtype=input.dtype)  # [B,C,r,out]
    C = leaf.get("C", None)
    if C is not None:
        C = C.to(device=input.device, dtype=input.dtype)

    bsz, chunks, in_features, _ = A.shape
    if input.shape[0] % bsz != 0:
        raise RuntimeError(f"input batch {input.shape[0]} must be a multiple of fusion batch {bsz}")
    beams = input.shape[0] // bsz
    x = input.reshape(bsz, beams, -1, in_features)

    x_a = torch.einsum("bnsi,bcir->bncsr", x, A)
    x_ab = torch.einsum("bncsr,bcro->bncso", x_a, B)
    w = weights.to(device=input.device, dtype=input.dtype)
    if w.shape[-1] != chunks:
        raise RuntimeError(f"fusion weights chunks {w.shape[-1]} != LoRA chunks {chunks}")
    out = (x_ab * w.permute(0, 1, 3, 2).unsqueeze(-1)).sum(dim=2)
    if C is not None:
        out = out + torch.einsum("bnsc,bco->bnso", w, C)
    return out.reshape(*input.shape[:-1], B.shape[-1])


def apply_dynamic_lora(input: Tensor, base: Tensor, leaf: dict) -> Tensor:
    weights = _cached_or_compute_weights(base, leaf)
    return base + dynamic_lora_update(input, leaf, weights)
