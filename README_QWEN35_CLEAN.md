# Clean SHINE — Qwen3.5 + multi-chunk query-aware LoRA (doc2lora × SHINE)

This branch is a cleaned SHINE variant with two changes:

1. **Backbone Qwen3 → Qwen3.5.**
2. **doc2lora's chunking + query-aware mixing + LoRA-norm rescaling** ported into SHINE
   (train + eval).

## Why it's not a drop-in (key findings)

- **SHINE is single-pass / one-LoRA.** Adding chunking necessarily makes it *multi-chunk*
  (one LoRA per context page, then combine) — a deliberate change to its single-pass identity.
- **Qwen3.5 is a hybrid VL model.** Its text stack has 32 layers but only **8 are
  full-attention** (`q/k/v/o`); the other **24 are `GatedDeltaNet`** (linear attention,
  `in_proj_*`, no `q/k/v/o`). SHINE's uniform full-attention subclassing doesn't apply, so
  the metamodel is **reimplemented hook-based** on the standard `Qwen3_5TextModel`.

## New modules (all CPU-verified)

| file | what | verified |
|---|---|---|
| `query_aware.py` | chunking (`build_paged_evidence`), QUEST scoring, norm rescaling (`normalize_loradict`, `frob_norm_AB`), var-rank (`mask_loradict_to_ranks`), prune + weighted **rank-combine** (`combine_chunk_loras`) | `tests/test_query_aware.py` (6/6) |
| `lora_qwen35.py` | hook-based LoRA metamodel for **hybrid Qwen3.5**; reproduces SHINE's interface (`lora_params_numel`/`generate_lora_dict`/`divide_idx`/memory-states via `output_hidden_states`/`generate`); **scoped LoRA** = MLP on all layers + attention on the 8 full-attn layers; `from_pretrained` extracts the text decoder from the VL model | `tests/test_lora_qwen35_smoke.py` |
| `multichunk.py` | end-to-end path: page → per-page LoRA → var-rank → rescale → **GQA-aware QUEST** weights → prune → combine → one LoRA | `tests/test_multichunk_smoke.py` |

Run all: `python tests/test_query_aware.py && python tests/test_lora_qwen35_smoke.py && python tests/test_multichunk_smoke.py`

## Config

`configs/Qwen3.5-9B.yaml` — `model.metamodel_class_path: lora_qwen35.LoraQwen35`, plus a
`multichunk:` block (default **off** → plain single-pass SHINE). Set `multichunk.enabled: true`
to use the doc2lora × SHINE path; `query_aware_mix`, `norm_rule: energy`, `var_rank`, `mix_top_k`
are the knobs.

## Design notes / decisions

- **Scoped LoRA, uniform-friendly:** per-layer LoRA layout is kept uniform-compatible; attention
  LoRA is *applied* only on the 8 full-attention layers (the GatedDeltaNet layers get MLP LoRA).
- **Rescaling target:** `‖A@B‖_F = lam·√(r_page)·σ_max(W0)` per page/layer/proj (energy / stable-rank).
- **Variable rank:** `r_page = #page-tokens`, capped at `lora_r` (nested truncation).
- **Combine:** weighted rank-stack so `A@B = Σ_c w_c A_c B_c` (QUEST top-k softmax weights, sink/local force-kept).
- `transformers >= 5.2.0` (first version shipping the `qwen3_5` architecture; the project's pin must be bumped from SHINE's `4.57.1`).

## Status / remaining work (needs GPU + real weights to verify)

Verified here on **CPU with a tiny random hybrid Qwen3.5** (shapes/wiring). **Not** verified:
training convergence, real Qwen3.5-9B loading, and the deep wiring of the multi-chunk path into
the large `meta_train_parallel.py` / `test.py` / `test_pwc.py` entrypoints (call `multichunk.
generate_lora_dict_multichunk` where they currently call `metanet.generate_lora_dict`, gated by
`cfg.multichunk.enabled`). The original `LoraQwen.py` (Qwen3) is left in place for reference but is
superseded by `lora_qwen35.py` for the Qwen3.5 path.
