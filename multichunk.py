"""Multi-chunk SHINE generation: page the context, make one LoRA per page with the
metanetwork, QUEST-score the pages against the query, rescale, and rank-combine into
a single LoRA.  Composes SHINE's ``Metanetwork`` + ``query_aware``.

This is the "doc2lora x SHINE" path.  It is **metamodel-agnostic**: it works with
both the original Qwen3 metamodel (``LoraQwen.LoraQwen3ForCausalLM``) and the
hybrid-Qwen3.5 hook wrapper (``lora_qwen35.LoraQwen35``).  Both go through
``Metanetwork.generate_lora_dict`` (same SHINE loradict format) and both expose
``layer.self_attn.q_proj/k_proj`` for the QUEST scoring.

Default-off in the configs so plain single-pass SHINE behaviour is preserved.
"""

from __future__ import annotations

import torch
import torch.nn as nn

import query_aware as qa
import lora_ortho
import lora_fusion


# --------------------------------------------------------------------------- #
# metamodel-agnostic navigation
# --------------------------------------------------------------------------- #
def _decoder_layers(metamodel) -> nn.ModuleList:
    for path in ("layers", "model.layers", "model.model.layers"):
        obj = metamodel
        try:
            for p in path.split("."):
                obj = getattr(obj, p)
        except AttributeError:
            continue
        if isinstance(obj, nn.ModuleList):
            return obj
    raise AttributeError("could not locate decoder layers on metamodel")


def _full_attn_qk(metamodel):
    """{layer_idx: (q_proj, k_proj)} for layers that have a self-attention sublayer."""
    out = {}
    for li, layer in enumerate(_decoder_layers(metamodel)):
        attn = getattr(layer, "self_attn", None)
        if attn is not None and hasattr(attn, "q_proj") and hasattr(attn, "k_proj"):
            out[li] = (attn.q_proj, attn.k_proj)
    return out


def _base_weight(metamodel, li, group, proj):
    layer = _decoder_layers(metamodel)[li]
    mod = layer.self_attn if group == "attention" else layer.mlp
    return getattr(mod, f"{proj}_proj").weight


def _meta_forward(metamodel, input_ids, attention_mask, loradict=None, ignore_mem_token=True):
    """Both metamodels accept (input_ids, attention_mask, loradict, ignore_mem_token)."""
    return metamodel(input_ids=input_ids, attention_mask=attention_mask,
                     loradict=loradict, ignore_mem_token=ignore_mem_token)


# --------------------------------------------------------------------------- #
# rescaling targets from the base weights
# --------------------------------------------------------------------------- #
@torch.no_grad()
def build_sigma_cache(metamodel, loradict) -> dict:
    """Top singular value of each LoRA-target base weight present in ``loradict``."""
    cache = {}
    for li, group, proj, _ in qa.iter_leaves(loradict):
        if (li, proj) not in cache:
            W = _base_weight(metamodel, li, group, proj).detach().float()
            cache[(li, proj)] = torch.linalg.svdvals(W).max()
    return cache


def per_chunk_targets(sigma_cache: dict, chunk_ranks, lam: float = 1.0) -> dict:
    r = chunk_ranks.clamp(min=1).float().sqrt()
    return {key: lam * r * sigma.to(r.device) for key, sigma in sigma_cache.items()}


# --------------------------------------------------------------------------- #
# QUEST scoring via the metamodel's full-attention q/k (GQA-aware, model-agnostic)
# --------------------------------------------------------------------------- #
@torch.no_grad()
def _capture_qk(metamodel, input_ids, attn_mask):
    qk = _full_attn_qk(metamodel)
    cap, handles = {}, []

    def mk(tag):
        def hook(_m, _i, out):
            cap[tag] = (out[0] if isinstance(out, tuple) else out).detach()
        return hook

    for li, (qp, kp) in qk.items():
        handles.append(qp.register_forward_hook(mk(("q", li))))
        handles.append(kp.register_forward_hook(mk(("k", li))))
    _meta_forward(metamodel, input_ids, attn_mask, loradict=None, ignore_mem_token=True)
    for h in handles:
        h.remove()
    return cap, sorted(qk.keys())


@torch.no_grad()
def quest_page_weights(metamodel, query_ids, query_mask, full_ctx_ids, full_ctx_mask,
                       page_ranges, force_keep, top_k=None, temperature=1.0):
    """Per-page QUEST weights (B=1 group). Sink/local force-kept; top-k middle.
    GQA-aware: q is averaged into kv-head groups so q/k channel dims match."""
    hd = metamodel.config.head_dim
    qcap, layers = _capture_qk(metamodel, query_ids, query_mask)
    kcap, _ = _capture_qk(metamodel, full_ctx_ids, full_ctx_mask)
    qm = query_mask[0][:, None].float()
    total = torch.zeros(len(page_ranges), device=query_ids.device)
    for li in layers:
        q = qcap[("q", li)][0].float()                 # [Qseq, n_q*hd]
        k = kcap[("k", li)][0].float()                 # [Tseq, n_kv*hd]
        n_kv = k.shape[-1] // hd
        grp = max(1, (q.shape[-1] // hd) // max(1, n_kv))
        q_mean = (q * qm).sum(0) / qm.sum().clamp_min(1.0)
        q_mean = q_mean.view(n_kv, grp, hd).mean(1).reshape(-1)   # [n_kv*hd]
        total = total + qa.paged_scores_from_keys(k, q_mean, page_ranges)
    w = qa.topk_softmax_weights(total, top_k=top_k, temperature=temperature, force_keep=force_keep)
    return w, (w > 0)


# --------------------------------------------------------------------------- #
# end-to-end multi-chunk lora generation (one context group)
# --------------------------------------------------------------------------- #
def generate_lora_dict_multichunk(
    metanet, context_tokens: list[int], query_ids, query_mask, metalora, *,
    n_sink=4, n_local=32, page_size=64, top_k=None, temperature=1.0,
    query_aware_mix=True, norm_rule="off", norm_lam=1.0, var_rank=False,
    mix_rescale=1.0, ortho_combine=False, sigma_cache=None,
):
    """One combined ``loradict`` (Lb=1) for a single context+query, via SHINE's
    ``Metanetwork`` (works for both the Qwen3 and Qwen3.5 metamodels).

    NOT wrapped in no_grad: at eval the callers are already @torch.no_grad(); in
    training gradients must flow through the per-page LoRA generation + combine
    (only the QUEST scoring is detached, inside ``quest_page_weights``)."""
    metamodel = metanet.metamodel
    if query_ids is not None:
        dev = query_ids.device
    else:
        try:
            dev = next(metanet.parameters()).device
        except StopIteration:
            dev = torch.device("cpu")
    pad_id = getattr(metamodel.config, "pad_token_id", 0) or 0
    paged = qa.build_paged_evidence([context_tokens], n_sink, n_local, page_size, pad_id=pad_id)
    ev_ids = paged["evidence_ids"].to(dev)
    ev_mask = paged["evidence_attention_mask"].to(dev)
    ranks, force_keep = paged["chunk_ranks"], paged["force_keep"]
    page_ranges = paged["page_ranges"][0]

    # one LoRA per page (batched over Lb = n_pages) through the metanetwork
    loradict = metanet.generate_lora_dict(ev_ids, ev_mask, metalora)   # Lb = n_pages

    if var_rank:
        loradict = qa.mask_loradict_to_ranks(loradict, ranks.to(dev))
    if norm_rule != "off":
        if sigma_cache is None:
            sigma_cache = build_sigma_cache(metamodel, loradict)
        loradict = qa.normalize_loradict(loradict, per_chunk_targets(sigma_cache, ranks.to(dev), norm_lam))

    if query_aware_mix and query_ids is not None:
        weights, keep = quest_page_weights(
            metamodel, query_ids, query_mask, paged["full_ctx_ids"].to(dev),
            paged["full_ctx_attention_mask"].to(dev), page_ranges, force_keep,
            top_k=top_k, temperature=temperature)
    else:  # uniform mix over all pages (no query available)
        n = ranks.numel()
        weights, keep = torch.full((n,), 1.0 / n, device=dev), torch.ones(n, dtype=torch.bool, device=dev)

    idx = keep.nonzero(as_tuple=True)[0]
    loradict = qa.select_pages(loradict, idx)
    w = weights[idx]
    w = w / w.sum().clamp_min(1e-6)
    w = w * float(mix_rescale)   # fixed post-softmax rescale of the combined LoRA (default 1.0 = no-op)
    return qa.combine_chunk_loras(loradict, w, ortho=ortho_combine)


def generate_lora_dict_multichunk_dynamic(
    metanet, context_tokens: list[int], metalora, *,
    n_sink=4, n_local=32, page_size=64, norm_rule="off", norm_lam=1.0,
    var_rank=False, sigma_cache=None,
):
    """Return per-page LoRAs plus M2P-generated routing keys for one sample.

    Unlike ``generate_lora_dict_multichunk`` this does not pre-combine chunks.
    The decoder layer will compute ``softmax(q @ k)`` weights at forward time.
    """
    metamodel = metanet.metamodel
    pad_id = getattr(metamodel.config, "pad_token_id", 0) or 0
    paged = qa.build_paged_evidence([context_tokens], n_sink, n_local, page_size, pad_id=pad_id)
    ev_ids = paged["evidence_ids"]
    ev_mask = paged["evidence_attention_mask"]
    try:
        dev = next(metanet.parameters()).device
    except StopIteration:
        dev = ev_ids.device
    ev_ids = ev_ids.to(dev)
    ev_mask = ev_mask.to(dev)
    ranks = paged["chunk_ranks"]

    loradict, fusion_keys = metanet.generate_lora_dict(
        ev_ids,
        ev_mask,
        metalora,
        return_fusion_keys=True,
    )

    if var_rank:
        loradict = qa.mask_loradict_to_ranks(loradict, ranks.to(dev))
    if norm_rule != "off":
        if sigma_cache is None:
            sigma_cache = build_sigma_cache(metamodel, loradict)
        loradict = qa.normalize_loradict(loradict, per_chunk_targets(sigma_cache, ranks.to(dev), norm_lam))
    return loradict, fusion_keys


# --------------------------------------------------------------------------- #
# batched entrypoint helper (one combined LoRA per sample -> stacked Lb=B)
# --------------------------------------------------------------------------- #
def _stack_loradicts(dicts: list[dict]) -> dict:
    """Stack per-sample (Lb=1) combined loradicts to Lb=B, zero-padding ragged ranks."""
    out: dict = {}
    layers = dicts[0].keys()
    for li in layers:
        out[li] = {}
        for group in dicts[0][li]:
            out[li][group] = {}
            for proj in dicts[0][li][group]:
                As = [d[li][group][proj]["A"] for d in dicts]   # each [1,in,r_i]
                Bs = [d[li][group][proj]["B"] for d in dicts]   # each [1,r_i,out]
                Cs = [d[li][group][proj].get("C") for d in dicts]
                rmax = max(a.shape[-1] for a in As)
                din, dout = As[0].shape[1], Bs[0].shape[-1]
                A = As[0].new_zeros(len(dicts), din, rmax)
                B = Bs[0].new_zeros(len(dicts), rmax, dout)
                for i, (a, b) in enumerate(zip(As, Bs)):
                    A[i, :, : a.shape[-1]] = a[0]
                    B[i, : b.shape[-2], :] = b[0]
                leaf = {"A": A, "B": B}
                if Cs[0] is not None:
                    leaf["C"] = torch.cat(Cs, dim=0)
                out[li][group][proj] = leaf
    return out


def _pad_dynamic_loradicts(dicts: list[dict], fusion_keys: list[torch.Tensor], *,
                           temperature=1.0, top_k=None, mix_rescale=1.0) -> dict:
    """Pad ragged per-sample chunk LoRAs into dynamic-fusion batch tensors."""
    B = len(dicts)
    max_chunks = max(next(iter(qa.iter_leaves(d)))[3]["A"].shape[0] for d in dicts)
    out: dict = {}
    for li in dicts[0].keys():
        out[li] = {}
        for group in dicts[0][li]:
            out[li][group] = {}
            for proj in dicts[0][li][group]:
                As = [d[li][group][proj]["A"] for d in dicts]
                Bs = [d[li][group][proj]["B"] for d in dicts]
                Cs = [d[li][group][proj].get("C") for d in dicts]
                din, r = As[0].shape[1], As[0].shape[2]
                dout = Bs[0].shape[-1]
                A = As[0].new_zeros(B, max_chunks, din, r)
                Bmat = Bs[0].new_zeros(B, max_chunks, r, dout)
                Cmat = None if Cs[0] is None else Cs[0].new_zeros(B, max_chunks, dout)
                for i, (a, b) in enumerate(zip(As, Bs)):
                    n = a.shape[0]
                    A[i, :n] = a
                    Bmat[i, :n] = b
                    if Cmat is not None:
                        Cmat[i, :n] = Cs[i]
                leaf = {"A": A, "B": Bmat}
                if Cmat is not None:
                    leaf["C"] = Cmat
                out[li][group][proj] = leaf

    num_layers, hidden = fusion_keys[0].shape[1], fusion_keys[0].shape[2]
    keys = fusion_keys[0].new_zeros(B, num_layers, max_chunks, hidden)
    mask = torch.zeros(B, max_chunks, dtype=torch.bool, device=fusion_keys[0].device)
    for i, k in enumerate(fusion_keys):
        n = k.shape[0]
        keys[i, :, :n] = k.transpose(0, 1)
        mask[i, :n] = True
    return lora_fusion.make_dynamic_loradict(
        out,
        keys,
        mask,
        temperature=temperature,
        top_k=top_k,
        mix_rescale=mix_rescale,
    )


def multichunk_dynamic_lora_for_batch(metanet, evidence_ids, evidence_mask, metalora, mc):
    B = evidence_ids.shape[0]
    per_sample, per_keys = [], []
    for i in range(B):
        ctx = evidence_ids[i][evidence_mask[i].bool()].tolist()
        loradict, keys = generate_lora_dict_multichunk_dynamic(
            metanet, ctx, metalora,
            n_sink=mc.n_sink, n_local=mc.n_local, page_size=mc.page_size,
            norm_rule=mc.norm_rule, norm_lam=mc.norm_lam, var_rank=mc.var_rank)
        per_sample.append(loradict)
        per_keys.append(keys)
    return _pad_dynamic_loradicts(
        per_sample,
        per_keys,
        temperature=getattr(mc, "dynamic_temp", getattr(mc, "mix_temp", 1.0)),
        top_k=getattr(mc, "dynamic_top_k", None),
        mix_rescale=getattr(mc, "mix_rescale", 1.0),
    )


def multichunk_lora_for_batch(metanet, evidence_ids, evidence_mask, query_ids, query_mask, metalora, mc):
    """Drop-in replacement for ``metanet.generate_lora_dict`` when ``mc.enabled``.
    Pages each sample's context, runs the multi-chunk path, returns a stacked Lb=B
    loradict.  ``query_ids=None`` -> uniform page mixing (no query available, e.g.
    reconstruction / before a multi-turn conversation).  ``mc`` is ``cfg.multichunk``
    (enabled/n_sink/n_local/page_size/query_aware_mix/mix_top_k/mix_temp/norm_rule/
    norm_lam/var_rank).  Not no_grad: grad flows in training (eval callers are no_grad)."""
    fusion_mode = str(getattr(mc, "fusion_mode", "static")).lower()
    if fusion_mode == "dynamic":
        return multichunk_dynamic_lora_for_batch(metanet, evidence_ids, evidence_mask, metalora, mc)

    B = evidence_ids.shape[0]
    if query_ids is not None and query_mask is None:
        query_mask = torch.ones_like(query_ids)
    per_sample = []
    for i in range(B):
        ctx = evidence_ids[i][evidence_mask[i].bool()].tolist()
        qi = None if query_ids is None else query_ids[i:i + 1]
        qm = None if query_ids is None else query_mask[i:i + 1]
        per_sample.append(generate_lora_dict_multichunk(
            metanet, ctx, qi, qm, metalora,
            n_sink=mc.n_sink, n_local=mc.n_local, page_size=mc.page_size,
            top_k=mc.mix_top_k, temperature=mc.mix_temp, query_aware_mix=mc.query_aware_mix,
            norm_rule=mc.norm_rule, norm_lam=mc.norm_lam, var_rank=mc.var_rank,
            mix_rescale=getattr(mc, "mix_rescale", 1.0),
            ortho_combine=getattr(mc, "ortho_combine", False)))
    return _stack_loradicts(per_sample)


# --------------------------------------------------------------------------- #
# merge a generated LoRA back into the base model weights (export a standalone model)
# --------------------------------------------------------------------------- #
@torch.no_grad()
def merge_loradict_into_model(metamodel, loradict, idx: int = 0):
    """Bake a single generated LoRA (batch index ``idx``) into the model weights IN PLACE.

    SHINE applies LoRA additively at forward (``LoraLinear.forward``: ``out = base + (x@A)@B + C``,
    with ``A:[in,r]``, ``B:[r,out]``). Since ``F.linear`` computes ``x @ W.T``, the equivalent
    merged weight is::

        W <- W + (A @ B).T          # [out,in] += ([in,r]@[r,out]).T
        bias <- bias + C            # if the layer has a bias

    SHINE's ``scale`` is already folded into ``A,B,C`` at generation, so no extra factor is needed.
    Works for both metamodels (LoraQwen3 / LoraQwen35) and for the multi-chunk combined loradict.
    After merging, run the model WITHOUT a loradict. ``copy.deepcopy(metamodel)`` first if you want
    to keep the un-merged base.
    """
    layers = _decoder_layers(metamodel)
    for li, group, proj, leaf in qa.iter_leaves(loradict):
        mod = layers[li].self_attn if group == "attention" else layers[li].mlp
        lin = getattr(mod, f"{proj}_proj")
        A = leaf["A"][idx].to(lin.weight.dtype)   # [in, r]
        B = leaf["B"][idx].to(lin.weight.dtype)   # [r, out]
        if lora_ortho.ENABLED:
            a = lora_ortho.ortho_alpha(A, B, lin.weight).to(lin.weight.dtype)   # scalar
            lin.weight.data.mul_(1 - a).add_((A @ B).t())   # W <- (1-alpha)*W + (A@B)^T
        else:
            lin.weight.data += (A @ B).t()        # [out, in]
        C = leaf.get("C")
        if C is not None and getattr(lin, "bias", None) is not None:
            lin.bias.data += C[idx].to(lin.bias.dtype)
    return metamodel
