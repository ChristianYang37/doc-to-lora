"""Multi-chunk SHINE generation: page the context, make one LoRA per page with the
metanetwork, QUEST-score the pages against the query, rescale, and rank-combine into
a single LoRA.  Composes ``lora_qwen35.LoraQwen35`` + ``query_aware``.

This is the "doc2lora x SHINE" path.  Default-off in the configs so plain SHINE
(single LoRA) behaviour is preserved.
"""

from __future__ import annotations

import torch

import query_aware as qa


# --------------------------------------------------------------------------- #
# rescaling targets from the base weights
# --------------------------------------------------------------------------- #
@torch.no_grad()
def build_sigma_cache(meta) -> dict:
    """Top singular value of each LoRA-target base weight: ``{(layer,proj): sigma}``."""
    cache = {}
    for li, group, proj, lin in meta.sites:
        cache[(li, proj)] = torch.linalg.svdvals(lin.weight.detach().float()).max()
    return cache


def per_chunk_targets(sigma_cache: dict, chunk_ranks, lam: float = 1.0) -> dict:
    """Energy target per (layer,proj): ``lam * sqrt(r_page) * sigma_max``  -> ``[n_pages]``."""
    r = chunk_ranks.clamp(min=1).float().sqrt()
    return {key: lam * r * sigma.to(r.device) for key, sigma in sigma_cache.items()}


# --------------------------------------------------------------------------- #
# QUEST scoring via the metamodel's full-attention q/k (GQA-aware)
# --------------------------------------------------------------------------- #
@torch.no_grad()
def _capture_qk(meta, input_ids, attn_mask):
    qk = {p: {li: lin for li, g, pp, lin in meta.sites if g == "attention" and pp == p}
          for p in ("q", "k")}
    cap, handles = {}, []

    def mk(tag):
        def hook(_m, _i, out):
            cap[tag] = out.detach()
        return hook

    for p in ("q", "k"):
        for li, lin in qk[p].items():
            handles.append(lin.register_forward_hook(mk((p, li))))
    meta(input_ids=input_ids, attention_mask=attn_mask, loradict=None, ignore_mem_token=True)
    for h in handles:
        h.remove()
    return cap, sorted(qk["q"].keys())


@torch.no_grad()
def quest_page_weights(meta, query_ids, query_mask, full_ctx_ids, full_ctx_mask,
                       page_ranges, force_keep, top_k=None, temperature=1.0):
    """Per-page QUEST weights (B=1 group). Sink/local force-kept; top-k middle.
    GQA-aware: q is averaged into kv-head groups so q·k dims match."""
    hd = meta.config.head_dim
    qcap, layers = _capture_qk(meta, query_ids, query_mask)
    kcap, _ = _capture_qk(meta, full_ctx_ids, full_ctx_mask)
    qm = query_mask[0][:, None].float()
    n_pages = len(page_ranges)
    total = torch.zeros(n_pages)
    for li in layers:
        q = qcap[("q", li)][0].float()            # [Qseq, n_q*hd]
        k = kcap[("k", li)][0].float()            # [Tseq, n_kv*hd]
        n_kv = k.shape[-1] // hd
        n_q = q.shape[-1] // hd
        grp = max(1, n_q // n_kv)
        q_mean = (q * qm).sum(0) / qm.sum().clamp_min(1.0)        # [n_q*hd]
        q_mean = q_mean.view(n_kv, grp, hd).mean(1).reshape(-1)  # collapse groups -> [n_kv*hd]
        # keys already [Tseq, n_kv*hd]; page min/max over the flat n_kv*hd channels
        total = total + qa.paged_scores_from_keys(k, q_mean, page_ranges)
    w = qa.topk_softmax_weights(total, top_k=top_k, temperature=temperature, force_keep=force_keep)
    return w, (w > 0)


# --------------------------------------------------------------------------- #
# end-to-end multi-chunk lora generation (one context group)
# --------------------------------------------------------------------------- #
@torch.no_grad()
def generate_lora_dict_multichunk(
    meta, metanetwork, context_tokens: list[int], query_ids, query_mask, metalora,
    lora_r, scale, *, n_sink=4, n_local=32, page_size=64, top_k=None, temperature=1.0,
    norm_rule="off", norm_lam=1.0, var_rank=False, sigma_cache=None,
):
    """Return one combined ``loradict`` (Lb=1) for a single context+query.

    Steps: page -> per-page memory_states (Lb=n_pages) -> metanetwork -> per-page
    loradict -> (var-rank) -> (rescale) -> QUEST weights -> prune -> rank-combine.
    """
    pad_id = getattr(meta.config, "pad_token_id", 0) or 0
    paged = qa.build_paged_evidence([context_tokens], n_sink, n_local, page_size, pad_id=pad_id)
    ev_ids, ev_mask = paged["evidence_ids"], paged["evidence_attention_mask"]
    ranks, force_keep = paged["chunk_ranks"], paged["force_keep"]
    page_ranges = paged["page_ranges"][0]

    # one LoRA per page (batched over Lb = n_pages)
    out = meta(input_ids=ev_ids, attention_mask=ev_mask, loradict=metalora)
    plain = metanetwork(out.memory_states)                      # [n_pages, numel]
    loradict = meta.generate_lora_dict(lora_r, scale, plain)    # Lb = n_pages

    if var_rank:
        loradict = qa.mask_loradict_to_ranks(loradict, ranks)
    if norm_rule != "off":
        if sigma_cache is None:
            sigma_cache = build_sigma_cache(meta)
        loradict = qa.normalize_loradict(loradict, per_chunk_targets(sigma_cache, ranks, norm_lam))

    weights, keep = quest_page_weights(meta, query_ids, query_mask, paged["full_ctx_ids"],
                                       paged["full_ctx_attention_mask"], page_ranges,
                                       force_keep, top_k=top_k, temperature=temperature)
    idx = keep.nonzero(as_tuple=True)[0]
    loradict = qa.select_pages(loradict, idx)
    w = weights[idx]
    w = w / w.sum().clamp_min(1e-6)
    return qa.combine_chunk_loras(loradict, w)
