# Clean SHINE — Qwen3.5 + multi-chunk query-aware LoRA (doc2lora × SHINE)

This branch is a cleaned SHINE variant that supports **two backbones**:

- **Qwen3** — the original SHINE metamodel (`LoraQwen.py`, all-linear: q/k/v/o + gate/up/down on
  every layer), **restored to run on transformers ≥ 5.2** (it broke on the 4.57→5.2 API churn).
- **Qwen3.5** — a hook-based metamodel (`lora_qwen35.py`) with LoRA scoped to **attention only**
  (q/k/v/o on the 8 full-attention layers). Qwen3.5 is hybrid (24/32 layers are GatedDeltaNet,
  no attention); since the query-aware fusion is **attention-score-based**, the LoRA lives exactly
  where the attention does.

Plus **doc2lora's chunking + query-aware mixing + LoRA-norm rescaling** ported into SHINE
(`query_aware.py` + `multichunk.py`).

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
| `LoraQwen.py` | **Qwen3** metamodel (original SHINE, all-linear) restored for transformers ≥ 5.2 | `tests/test_lora_qwen3_smoke.py` |
| `lora_qwen35.py` | hook-based LoRA metamodel for **hybrid Qwen3.5**; reproduces SHINE's interface (`lora_params_numel`/`generate_lora_dict`/`divide_idx`/memory-states via `output_hidden_states`/`generate`); **`lora_scope="attention"`** = q/k/v/o on the 8 full-attn layers only (`"all"` adds MLP); `from_pretrained` extracts the text decoder from the VL model | `tests/test_lora_qwen35_smoke.py` |
| `multichunk.py` | end-to-end path: page → per-page LoRA → var-rank → rescale → **GQA-aware QUEST** weights → prune → combine → one LoRA | `tests/test_multichunk_smoke.py` |

Run all: `for t in test_query_aware test_lora_qwen3_smoke test_lora_qwen35_smoke test_multichunk_smoke; do python tests/$t.py; done`

## Config & scripts

Every config (`configs/Qwen3-8B.yaml`, `Qwen3-1.7B`, `Qwen3-0.6B`, `Qwen3.5-9B`) carries a
`multichunk:` block (default **off** → plain single-pass SHINE). Set `multichunk.enabled: true`
for the doc2lora × SHINE path; knobs: `n_sink`, `n_local`, `page_size`, `query_aware_mix`,
`mix_top_k`, `mix_temp`, `norm_rule` (`off`|`energy`), `norm_lam`, `var_rank` (+ `model.lora_scope`
for Qwen3.5). The entrypoints (`test.py`/`test_pwc.py`/`test_pretrain.py`/`meta_train_parallel.py`)
read `cfg.multichunk` at their `generate_lora_dict` sites and branch to `multichunk.
multichunk_lora_for_batch` when enabled.

**Scripts:** `scripts/Qwen3.5-9B/` mirrors all Qwen3-8B scripts (with `CONFIG_NAME=Qwen3.5-9B` +
`LORA_SCOPE`); every run script (Qwen3-8B/1.7B/0.6B + Qwen3.5-9B) exposes the knobs as shell vars
(`MULTICHUNK_ENABLED`, `PAGE_SIZE`, `MIX_TOP_K`, `NORM_RULE`, `VAR_RANK`, …) wired through `${MC_ARGS}`.

## Design notes / decisions

- **Qwen3 (all-linear) vs Qwen3.5 (attention-only):** Qwen3 has attention in every layer, so the
  original all-linear LoRA + the attention-score fusion fit naturally. Qwen3.5 is hybrid, so LoRA is
  scoped to q/k/v/o of the 8 full-attention layers — the only layers the QUEST fusion can read.
- **Rescaling target:** `‖A@B‖_F = lam·√(r_page)·σ_max(W0)` per page/layer/proj (energy / stable-rank).
- **Variable rank:** `r_page = #page-tokens`, capped at `lora_r` (nested truncation).
- **Combine:** weighted rank-stack so `A@B = Σ_c w_c A_c B_c` (QUEST top-k softmax weights, sink/local force-kept).
- `transformers >= 5.2.0` (first version shipping the `qwen3_5` architecture; the project's pin must be bumped from SHINE's `4.57.1`).

## Multi-chunk on both backbones

`multichunk.py` is **metamodel-agnostic**: it drives the multi-chunk path via SHINE's `Metanetwork`
and captures q/k by module path, so it works with **both** `LoraQwen.LoraQwen3ForCausalLM` (Qwen3,
all-linear → fusion on every layer) and `lora_qwen35.LoraQwen35` (Qwen3.5, attention-only). The path
is grad-capable (training) and detaches only the QUEST scoring.

## Status / remaining work (needs GPU + real weights to verify)

Verified here on **CPU with tiny random models** (both metamodels): the math, memory-states,
per-page LoRA, QUEST scoring, rescale, combine, the batched `Lb=B` helper, and **gradient flow back
through the metanetwork** (training-capable). **Not** verified (needs your GPU + weights): training
convergence and real Qwen3-8B / Qwen3.5-9B loading. The original `LoraQwen.py` (Qwen3) is fully
restored (transformers ≥ 5.2); `lora_qwen35.py` is the Qwen3.5 path.
