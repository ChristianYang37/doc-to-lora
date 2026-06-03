"""Hook-based LoRA metamodel for Qwen3.5 (clean SHINE port).

Qwen3.5's text stack is HYBRID: most layers are GatedDeltaNet (linear attention,
no q/k/v/o) and only 1-in-4 are full attention.  Rather than subclass transformers
internals the way SHINE did for the uniform Qwen3 stack, this wrapper keeps the
*standard* ``Qwen3_5TextModel`` intact and injects LoRA by swapping the target
``nn.Linear`` modules for ``LoraLinear`` and routing per-module LoRA via a small
"active LoRA" state.  Memory states are read straight from ``output_hidden_states``.

It reproduces SHINE's metamodel interface so ``Metanetwork`` (metanetwork_family.py)
works unchanged: ``lora_params_numel`` / ``divide_idx`` / ``set_generate_func`` /
``generate_lora_dict`` / ``init_lora_dict`` / ``reset_mem_tokens`` / a forward that
returns ``.memory_states`` / ``.loss`` / ``.logits`` and a ``generate``.

Scoped LoRA (per the design decision): a UNIFORM per-layer target set
{q,k,v,o,gate,up,down} so the metanetwork stays uniform, but the attention LoRA is
only *applied* on the full-attention layers; on GatedDeltaNet layers those slots are
generated and ignored (a small, documented capacity overhead).
"""

from __future__ import annotations

from math import sqrt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

ATTN_PROJ = ("q", "k", "v", "o")
MLP_PROJ = ("gate", "up", "down")
_PROJ_TO_ATTR = {"q": "q_proj", "k": "k_proj", "v": "v_proj", "o": "o_proj",
                 "gate": "gate_proj", "up": "up_proj", "down": "down_proj"}


class LoraLinear(nn.Linear):
    """nn.Linear that adds ``dW = A@B (+C)`` from an externally-set active LoRA.

    ``self._active`` is ``{"A":[Lb,in,r], "B":[Lb,r,out], "C":[Lb,out]|None}`` or
    ``None``.  Batch is broadcast over ``num_beams = batch // Lb`` (SHINE semantics).
    """

    def __init__(self, in_features, out_features, bias=True, device=None, dtype=None):
        super().__init__(in_features, out_features, bias=bias, device=device, dtype=dtype)
        self._active = None

    @classmethod
    def from_linear(cls, lin: nn.Linear) -> "LoraLinear":
        m = cls(lin.in_features, lin.out_features, bias=lin.bias is not None,
                device=lin.weight.device, dtype=lin.weight.dtype)
        m.weight = lin.weight
        m.bias = lin.bias
        return m

    def forward(self, input: Tensor) -> Tensor:
        base = F.linear(input, self.weight, self.bias)
        lora = self._active
        if lora is None:
            return base
        A, B, C = lora["A"], lora["B"], lora.get("C", None)
        Lb = A.shape[0]
        num_beams = input.shape[0] // Lb
        x = input.reshape(Lb, num_beams, -1, self.in_features)
        out = torch.matmul(torch.matmul(x, A[:, None]), B[:, None])  # [Lb,beams,S,out]
        if C is not None:
            out = out + C[:, None, None, :]
        return base + out.reshape(*input.shape[:-1], self.out_features)

    # --- SHINE leaf contract ---
    def lora_params_numel(self, r):
        return self.in_features * r + self.out_features * r + (self.out_features if self.bias is not None else 0)

    def set_generate_func(self, method):
        assert method == "rl", f"only 'rl' supported, got {method}"
        self.method = method

    def generate_lora_dict(self, r, scale, plain_tensor):
        i = 0
        A = plain_tensor[:, i:i + self.in_features * r].view(-1, self.in_features, r) * sqrt(scale)
        i += self.in_features * r
        B = plain_tensor[:, i:i + self.out_features * r].view(-1, r, self.out_features) * sqrt(scale)
        i += self.out_features * r
        C = plain_tensor[:, i:i + self.out_features].view(-1, self.out_features) * scale if self.bias is not None else None
        return {"A": A, "B": B, "C": C}

    def init_lora_dict(self, r, scale, device):
        A = (torch.randn(1, self.in_features, r, device=device) * sqrt(scale)).detach().requires_grad_()
        B = torch.zeros(1, r, self.out_features, requires_grad=True, device=device)
        C = torch.zeros(1, self.out_features, requires_grad=True, device=device) if self.bias is not None else None
        return {"A": A, "B": B, "C": C}

    def divide_idx(self, r, idx_start):
        A_numel, B_numel = self.in_features * r, self.out_features * r
        return [idx_start, A_numel + idx_start], A_numel + B_numel + idx_start


def _layer_modules(layer):
    """Return (attn_module_or_None, mlp_module) for a Qwen3.5 decoder layer."""
    attn = getattr(layer, "self_attn", None)  # only full_attention layers have it
    mlp = layer.mlp
    return attn, mlp


def _find_text_decoder(module):
    """BFS the module tree for the text decoder (has layers + embed_tokens + norm)."""
    seen, queue = set(), [module]
    while queue:
        m = queue.pop(0)
        if id(m) in seen:
            continue
        seen.add(id(m))
        if (hasattr(m, "layers") and isinstance(getattr(m, "layers"), nn.ModuleList)
                and hasattr(m, "embed_tokens") and hasattr(m, "norm")):
            return m
        queue.extend(m.children())
    return None


class LoraQwen35(nn.Module):
    """Hook-based LoRA metamodel around a standard Qwen3.5 text model."""

    def __init__(self, text_model: nn.Module, lm_head: nn.Linear, config, num_mem_token: int,
                 lora_scope: str = "attention"):
        """lora_scope: "attention" -> LoRA only on q/k/v/o of the full-attention layers
        (recommended for Qwen3.5: the query-aware fusion is attention-score-based, so the
        LoRA lives exactly where attention does). "all" -> also MLP on every layer."""
        super().__init__()
        assert lora_scope in ("attention", "all")
        self.lora_scope = lora_scope
        self.model = text_model
        self.lm_head = lm_head
        self.config = config
        self.use_mem_token = num_mem_token > 0
        self.num_mem_token = max(0, num_mem_token)
        hidden = config.hidden_size
        if self.use_mem_token:
            self.mem_tokens = nn.Parameter(torch.zeros(self.num_mem_token, hidden))
        # discover layers + swap target Linears -> LoraLinear; record ordered sites
        self.layers = self.model.layers
        self.sites = []          # ordered list of (layer_idx, group, proj, LoraLinear)
        self.full_attn_layers = []
        for li, layer in enumerate(self.layers):
            attn, mlp = _layer_modules(layer)
            if attn is not None:
                self.full_attn_layers.append(li)
                for proj in ATTN_PROJ:
                    lin = getattr(attn, _PROJ_TO_ATTR[proj])
                    setattr(attn, _PROJ_TO_ATTR[proj], LoraLinear.from_linear(lin))
                    self.sites.append((li, "attention", proj, getattr(attn, _PROJ_TO_ATTR[proj])))
            if self.lora_scope == "all":
                for proj in MLP_PROJ:
                    lin = getattr(mlp, _PROJ_TO_ATTR[proj])
                    setattr(mlp, _PROJ_TO_ATTR[proj], LoraLinear.from_linear(lin))
                    self.sites.append((li, "mlp", proj, getattr(mlp, _PROJ_TO_ATTR[proj])))
        self.method = "rl"

    @classmethod
    def from_pretrained(cls, model_path, num_mem_token, lora_scope="attention", dtype=None, **kw):
        """Load a real Qwen3.5 checkpoint and wrap its text decoder.

        Qwen3.5-9B is a VL model (``Qwen3_5ForConditionalGeneration``); we locate
        the inner text decoder (the module that owns ``layers``/``embed_tokens``/
        ``norm``) and the output head, then build the hook-based LoRA metamodel.
        """
        from transformers import AutoModelForCausalLM
        full = AutoModelForCausalLM.from_pretrained(model_path, dtype=dtype, **kw)
        text = _find_text_decoder(full)
        if text is None:
            raise RuntimeError(f"Could not locate a text decoder in {type(full).__name__}")
        lm_head = full.get_output_embeddings()
        config = getattr(text, "config", getattr(full, "config", None))
        config = getattr(config, "text_config", config)  # unwrap VL config
        return cls(text, lm_head, config, num_mem_token, lora_scope=lora_scope)

    # ---- interface used by Metanetwork ----
    @property
    def device(self):
        return next(self.parameters()).device

    def set_generate_func(self, method):
        self.method = method
        for *_, lin in self.sites:
            lin.set_generate_func(method)

    def reset_mem_tokens(self):
        if self.use_mem_token:
            nn.init.zeros_(self.mem_tokens)

    def lora_params_numel(self, r):
        return sum(lin.lora_params_numel(r) for *_, lin in self.sites)

    def divide_idx(self, r, idx_start):
        idx_range = []
        idx = idx_start
        for *_, lin in self.sites:
            rng, idx = lin.divide_idx(r, idx)
            idx_range += rng
        return idx_range, idx

    def init_lora_dict(self, r, scale, device):
        d = {}
        for li, group, proj, lin in self.sites:
            d.setdefault(li, {}).setdefault(group, {})[proj] = lin.init_lora_dict(r, scale, device)
        return d

    def generate_lora_dict(self, r, scale, plain_tensor):
        d = {}
        idx = 0
        for li, group, proj, lin in self.sites:
            n = lin.lora_params_numel(r)
            d.setdefault(li, {}).setdefault(group, {})[proj] = lin.generate_lora_dict(r, scale, plain_tensor[:, idx:idx + n])
            idx += n
        return d

    # ---- LoRA routing ----
    def _apply_loradict(self, loradict):
        for li, group, proj, lin in self.sites:
            leaf = None
            if loradict is not None and li in loradict and group in loradict[li] and proj in loradict[li][group]:
                leaf = loradict[li][group][proj]
            lin._active = leaf

    def _clear_loradict(self):
        for *_, lin in self.sites:
            lin._active = None

    # ---- forward (returns memory_states like SHINE) ----
    def forward(self, input_ids=None, attention_mask=None, loradict=None, labels=None,
                ignore_mem_token=False, **kwargs):
        self._apply_loradict(loradict)
        try:
            inputs_embeds = self.model.embed_tokens(input_ids)
            collect_mem = self.use_mem_token and not ignore_mem_token
            if collect_mem:
                B = inputs_embeds.shape[0]
                inputs_embeds = torch.cat([inputs_embeds, self.mem_tokens[None].expand(B, -1, -1)], dim=1)
                if attention_mask is not None:
                    attention_mask = torch.cat(
                        [attention_mask, attention_mask.new_ones(B, self.num_mem_token)], dim=1)
            out = self.model(inputs_embeds=inputs_embeds, attention_mask=attention_mask,
                             output_hidden_states=True, use_cache=False)
            hs = out.hidden_states  # tuple: (embed, layer0, ..., layer_{L-1})
            last = self.model.norm(hs[-1]) if hasattr(self.model, "norm") else hs[-1]
            logits = self.lm_head(last)
            result = type("Out", (), {})()
            result.logits = logits
            result.loss = None
            result.memory_states = None
            if collect_mem:
                mem = torch.stack([hs[i + 1][:, -self.num_mem_token:, :]
                                   for i in range(self.config.num_hidden_layers)], dim=1)
                result.memory_states = mem  # [B, num_layers, num_mem_token, hidden]
            if labels is not None:
                logits_for_loss = logits if not (self.use_mem_token and not ignore_mem_token) else logits[:, :labels.shape[1]]
                shift_logits = logits_for_loss[:, :-1].reshape(-1, logits.shape[-1])
                shift_labels = labels[:, 1:].reshape(-1).to(shift_logits.device)
                result.loss = F.cross_entropy(shift_logits, shift_labels, ignore_index=-100)
            return result
        finally:
            self._clear_loradict()

    @torch.no_grad()
    def generate(self, input_ids=None, attention_mask=None, loradict=None,
                 ignore_mem_token=True, **gen_kwargs):
        self._apply_loradict(loradict)
        try:
            from types import MethodType
            # standard HF generate on the text model + lm_head via a thin causal head
            return _greedy_generate(self, input_ids, attention_mask, **gen_kwargs)
        finally:
            self._clear_loradict()


@torch.no_grad()
def _greedy_generate(meta: LoraQwen35, input_ids, attention_mask, max_new_tokens=16,
                     do_sample=False, **_):
    """Minimal greedy decode (no KV cache) — enough for CPU smoke; LoRA stays active."""
    ids = input_ids
    for _ in range(max_new_tokens):
        emb = meta.model.embed_tokens(ids)
        out = meta.model(inputs_embeds=emb, attention_mask=attention_mask,
                         use_cache=False, output_hidden_states=True)
        last = meta.model.norm(out.hidden_states[-1])
        nxt = meta.lm_head(last[:, -1:]).argmax(-1)  # [B,1]
        ids = torch.cat([ids, nxt], dim=1)
        if attention_mask is not None:
            attention_mask = torch.cat([attention_mask, attention_mask.new_ones(ids.shape[0], 1)], dim=1)
    return ids
