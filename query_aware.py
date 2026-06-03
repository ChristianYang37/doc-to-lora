"""Query-aware multi-chunk LoRA mixing + norm rescaling for SHINE.

Ported from doc-to-lora's `query_aware.py` and adapted to SHINE's `loradict`
format.  SHINE stores, per decoder layer, a nested dict::

    loradict[layer_idx] = {
        "attention": {"q"/"k"/"v"/"o": {"A":[Lb,in,r], "B":[Lb,r,out], "C":[Lb,out]?}},
        "mlp":       {"gate"/"up"/"down": {... same ...}},
    }

and applies it as ``dW = A @ B`` (+ C bias) inside ``LoraLinear.forward``.  For
multi-chunk we set the leading ``Lb`` axis to the number of context *pages*: the
metanetwork emits one LoRA per page (batched over Lb), then we

  1. (optional) nested-truncate each page to ``r_page = #tokens``   [variable rank]
  2. rescale each page's ``||A@B||_F`` to ``lam * sqrt(r_page) * sigma_max(W0)`` [rescale]
  3. QUEST-score each page vs the query and keep a top-k softmax over pages    [query-aware]
  4. combine the kept pages into ONE LoRA by weighted rank-stacking            [mix]

Everything here is pure tensor math (no model internals) so it is unit tested
on CPU; the QUEST key extraction that feeds (3) lives in the model wrapper.
"""

from __future__ import annotations

import torch
from torch import Tensor

PROJ_GROUPS = ("attention", "mlp")


# --------------------------------------------------------------------------- #
# loradict traversal helpers
# --------------------------------------------------------------------------- #
def iter_leaves(loradict: dict):
    """Yield ``(layer_idx, group, proj, leaf)`` for every projection present.

    Tolerant of layers that lack a group (e.g. Qwen3.5 linear-attention layers
    have no ``"attention"`` group under scoped LoRA).
    """
    for layer_idx, groups in loradict.items():
        for group in PROJ_GROUPS:
            if group not in groups:
                continue
            for proj, leaf in groups[group].items():
                yield layer_idx, group, proj, leaf


def frob_norm_AB(A: Tensor, B: Tensor) -> Tensor:
    """Frobenius norm of ``dW = A @ B`` without forming the in×out matrix.

    ``A``: ``[..., in, r]``  ``B``: ``[..., r, out]``.
    Uses ``||A B||_F^2 = <A^T A, B B^T>`` (both Gram matrices ``r×r``).
    Returns ``[...]``.
    """
    GA = A.transpose(-1, -2) @ A  # [..., r, r]
    GB = B @ B.transpose(-1, -2)  # [..., r, r]
    return (GA * GB).sum(dim=(-1, -2)).clamp_min(0).sqrt()


# --------------------------------------------------------------------------- #
# (1) variable per-page rank  (r_page = #tokens, nested truncation)
# --------------------------------------------------------------------------- #
def mask_loradict_to_ranks(loradict: dict, chunk_ranks: Tensor) -> dict:
    """Zero ranks ``>= r_c`` for each page (Lb axis). Effective rank -> min(r_c,R).

    ``A``: ``[Lb,in,R]`` (rank is the last dim) -> zero columns ``>= r_c``.
    ``B``: ``[Lb,R,out]`` (rank is the middle dim) -> zero rows ``>= r_c``.
    """
    out: dict = {}
    for layer_idx, groups in loradict.items():
        out[layer_idx] = {}
        for group, projs in groups.items():
            out[layer_idx][group] = {}
            for proj, leaf in projs.items():
                A, B = leaf["A"], leaf["B"]
                R = A.shape[-1]
                r = chunk_ranks.clamp(min=1, max=R).to(A.device)
                keep = (torch.arange(R, device=A.device)[None, :] < r[:, None]).to(A.dtype)
                nd = dict(leaf)
                nd["A"] = A * keep[:, None, :]   # [Lb,1,R]
                nd["B"] = B * keep[:, :, None]   # [Lb,R,1]
                out[layer_idx][group][proj] = nd
    return out


# --------------------------------------------------------------------------- #
# (2) norm rescaling
# --------------------------------------------------------------------------- #
def target_frob_norm(sigma_max: Tensor, rank, lam: float = 1.0) -> Tensor:
    """Energy / stable-rank target: ``lam * sqrt(rank) * sigma_max(W0)``."""
    r = torch.as_tensor(rank, dtype=sigma_max.dtype, device=sigma_max.device).clamp_min(1).float()
    return lam * r.sqrt() * sigma_max


def normalize_loradict(loradict: dict, target_norms: dict, eps: float = 1e-6) -> dict:
    """Rescale each page's ``||A@B||_F`` to its target (gain split as sqrt to A,B).

    ``target_norms[(layer_idx, proj)]`` is a ``[Lb]`` (or scalar/broadcastable)
    tensor of target Frobenius norms.  Differentiable: pins magnitude, frees
    direction.
    """
    out: dict = {}
    for layer_idx, groups in loradict.items():
        out[layer_idx] = {}
        for group, projs in groups.items():
            out[layer_idx][group] = {}
            for proj, leaf in projs.items():
                A, B = leaf["A"], leaf["B"]
                cur = frob_norm_AB(A, B)  # [Lb]
                tgt = target_norms[(layer_idx, proj)].to(cur.dtype).to(cur.device)
                tgt = tgt.expand_as(cur) if tgt.dim() else tgt
                s = (tgt / cur.clamp_min(eps)).clamp_min(0).sqrt()  # [Lb]
                nd = dict(leaf)
                nd["A"] = A * s[:, None, None]
                nd["B"] = B * s[:, None, None]
                if leaf.get("C") is not None:
                    nd["C"] = leaf["C"] * (s * s)[:, None]
                out[layer_idx][group][proj] = nd
    return out


# --------------------------------------------------------------------------- #
# (3) QUEST query-aware scoring  (model-agnostic math)
# --------------------------------------------------------------------------- #
def quest_scores(q: Tensor, k_min: Tensor, k_max: Tensor) -> Tensor:
    """``score_c = sum_i max(q_i M_{c,i}, q_i m_{c,i})`` = relu(q)·M + min(q,0)·m."""
    pos = torch.einsum("...d,cd->...c", q.clamp_min(0.0), k_max)
    neg = torch.einsum("...d,cd->...c", q.clamp_max(0.0), k_min)
    return pos + neg


def topk_softmax_weights(scores: Tensor, top_k=None, temperature: float = 1.0,
                         force_keep: Tensor | None = None) -> Tensor:
    """Top-k softmax (kept weights sum to 1); ``force_keep`` positions are exempt
    from the top-k budget (QUEST attention-sink + local window)."""
    n = scores.shape[-1]
    k = n if (top_k is None or top_k <= 0) else min(top_k, n)
    if force_keep is None:
        if k >= n:
            masked = scores
        else:
            thr = scores.topk(k, dim=-1).values[..., -1, None]
            masked = scores.masked_fill(scores < thr, float("-inf"))
        return torch.softmax(masked / max(temperature, 1e-6), dim=-1)
    force_keep = force_keep.to(torch.bool).expand_as(scores)
    keep = force_keep.clone()
    if k < n:
        cand = scores.masked_fill(force_keep, float("-inf"))
        thr = cand.topk(min(k, n), dim=-1).values[..., -1, None]
        keep = keep | (cand >= thr)
    else:
        keep = torch.ones_like(force_keep)
    return torch.softmax(scores.masked_fill(~keep, float("-inf")) / max(temperature, 1e-6), dim=-1)


def paged_scores_from_keys(keys: Tensor, q_mean: Tensor,
                           page_ranges: list[tuple[int, int]]) -> Tensor:
    """Per-page QUEST score from in-context keys (one sample, one layer).
    ``keys``: ``[T,d]``  ``q_mean``: ``[d]`` -> ``[n_pages]``."""
    out = []
    for s, e in page_ranges:
        if e <= s:
            out.append(keys.new_zeros(()))
            continue
        seg = keys[s:e]
        out.append(quest_scores(q_mean, seg.min(0).values[None], seg.max(0).values[None])[0])
    return torch.stack(out)


# --------------------------------------------------------------------------- #
# (4) combine pages into one LoRA (weighted rank-stack) + pruning
# --------------------------------------------------------------------------- #
def combine_chunk_loras(loradict: dict, weights: Tensor) -> dict:
    """Combine ``Lb`` per-page LoRAs into a single (Lb=1) LoRA by weighted
    rank-stacking: ``A=[in, Σr]``, ``B=[Σr, out]`` so ``A@B = Σ_c w_c A_c B_c``;
    ``C = Σ_c w_c C_c``.  ``weights``: ``[Lb]`` (sums to 1 over kept pages)."""
    w = weights.clamp_min(0)
    sw = w.sqrt()
    out: dict = {}
    for layer_idx, groups in loradict.items():
        out[layer_idx] = {}
        for group, projs in groups.items():
            out[layer_idx][group] = {}
            for proj, leaf in projs.items():
                A, B = leaf["A"], leaf["B"]  # [Lb,in,r], [Lb,r,out]
                Lb, din, r = A.shape
                dout = B.shape[-1]
                A_s = A * sw[:, None, None]
                B_s = B * sw[:, None, None]
                # stack pages along rank -> [1, in, Lb*r] and [1, Lb*r, out]
                A_c = A_s.permute(1, 0, 2).reshape(din, Lb * r)[None]
                B_c = B_s.reshape(Lb * r, dout)[None]
                nd = {"A": A_c, "B": B_c}
                if leaf.get("C") is not None:
                    nd["C"] = (leaf["C"] * w[:, None]).sum(0, keepdim=True)
                out[layer_idx][group][proj] = nd
    return out


def select_pages(loradict: dict, keep_idx: Tensor) -> dict:
    """Keep only pages (Lb rows) in ``keep_idx`` across the whole loradict."""
    out: dict = {}
    for layer_idx, groups in loradict.items():
        out[layer_idx] = {}
        for group, projs in groups.items():
            out[layer_idx][group] = {
                proj: {k: (v[keep_idx] if torch.is_tensor(v) else v) for k, v in leaf.items()}
                for proj, leaf in projs.items()
            }
    return out


# --------------------------------------------------------------------------- #
# QUEST-style context paging (model-agnostic; ported verbatim from doc-to-lora)
# --------------------------------------------------------------------------- #
def page_context_tokens(valid_len: int, n_sink: int = 4, n_local: int = 32,
                        page_size: int = 64, max_pages: int | None = None):
    """QUEST token layout: ``[sink(<=n_sink)] [middle pages] [local(<=n_local)]``.
    Returns ``(ranges, force_keep)`` exactly covering ``[0, valid_len)``;
    sink/local are force-kept, middle pages are top-k selectable."""
    L = int(valid_len)
    if L <= 0:
        return [], []
    n_sink = max(0, min(n_sink, L))
    ranges, force = [], []
    if n_sink > 0:
        ranges.append((0, n_sink)); force.append(True)
    local_start = max(n_sink, L - max(0, n_local))
    if local_start > n_sink:
        span = local_start - n_sink
        ps = max(1, page_size)
        if max_pages is not None and max_pages > 0:
            ps = max(ps, -(-span // max_pages))
        p = n_sink
        while p < local_start:
            ranges.append((p, min(p + ps, local_start))); force.append(False)
            p += ps
    if local_start < L:
        ranges.append((local_start, L)); force.append(True)
    return ranges, force


def build_paged_evidence(ctx_token_lists: list[list[int]], n_sink: int = 4,
                         n_local: int = 32, page_size: int = 64, pad_id: int = 0,
                         max_pages_per_ctx: int | None = None) -> dict:
    """Per-sample context token lists -> QUEST-paged chunks. Returns evidence_ids
    ``[Σpages, Lc]``, n_ctx_chunks ``[B]``, force_keep ``[Σ]``, chunk_ranks ``[Σ]``
    (= page token count), full_ctx_ids ``[B, Lf]`` and page_ranges (per sample)."""
    chunks, n_ctx_chunks, force_keep, chunk_ranks, page_ranges = [], [], [], [], []
    for toks in ctx_token_lists:
        ranges, force = page_context_tokens(len(toks), n_sink, n_local, page_size,
                                            max_pages=max_pages_per_ctx)
        if not ranges:
            ranges, force = [(0, 0)], [True]
        page_ranges.append(ranges)
        n_ctx_chunks.append(len(ranges))
        for (s, e), f in zip(ranges, force):
            chunks.append(toks[s:e]); force_keep.append(bool(f)); chunk_ranks.append(max(1, e - s))
    Lc = max((len(c) for c in chunks), default=1)
    Lf = max((len(t) for t in ctx_token_lists), default=1)
    ev = torch.full((len(chunks), Lc), pad_id, dtype=torch.long)
    ev_mask = torch.zeros((len(chunks), Lc), dtype=torch.long)
    for i, c in enumerate(chunks):
        if c:
            ev[i, :len(c)] = torch.tensor(c, dtype=torch.long); ev_mask[i, :len(c)] = 1
    full = torch.full((len(ctx_token_lists), Lf), pad_id, dtype=torch.long)
    full_mask = torch.zeros((len(ctx_token_lists), Lf), dtype=torch.long)
    for i, t in enumerate(ctx_token_lists):
        if t:
            full[i, :len(t)] = torch.tensor(t, dtype=torch.long); full_mask[i, :len(t)] = 1
    return {
        "evidence_ids": ev, "evidence_attention_mask": ev_mask,
        "n_ctx_chunks": torch.tensor(n_ctx_chunks, dtype=torch.int32),
        "force_keep": torch.tensor(force_keep, dtype=torch.bool),
        "chunk_ranks": torch.tensor(chunk_ranks, dtype=torch.long),
        "full_ctx_ids": full, "full_ctx_attention_mask": full_mask,
        "page_ranges": page_ranges,
    }
