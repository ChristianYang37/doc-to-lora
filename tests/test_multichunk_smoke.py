"""CPU smoke for the multi-chunk SHINE path (doc2lora x SHINE), tiny hybrid Qwen3.5.
Run: python tests/test_multichunk_smoke.py"""
import importlib.util, os, sys, torch, torch.nn as nn

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, ".."))
def _load(name):
    s = importlib.util.spec_from_file_location(name, os.path.join(_HERE, "..", name + ".py"))
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
lq = _load("lora_qwen35"); mc = _load("multichunk")

from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextModel
torch.manual_seed(0)


def build(num_mem_token=2):
    cfg = Qwen3_5TextConfig(
        vocab_size=100, hidden_size=64, intermediate_size=128, num_hidden_layers=4,
        num_attention_heads=4, num_key_value_heads=2, head_dim=16,
        layer_types=["linear_attention", "linear_attention", "linear_attention", "full_attention"],
        linear_conv_kernel_dim=4, linear_key_head_dim=16, linear_num_key_heads=2,
        linear_value_head_dim=16, linear_num_value_heads=4, rms_norm_eps=1e-6,
        rope_parameters={"rope_theta": 1e4, "partial_rotary_factor": 0.25, "rope_type": "default"},
        attn_implementation="eager")
    text = Qwen3_5TextModel(cfg).eval()
    meta = lq.LoraQwen35(text, nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False), cfg, num_mem_token).eval()
    meta.set_generate_func("rl")
    return cfg, meta


def main():
    cfg, meta = build()
    r, scale = 8, 0.01
    numel = meta.lora_params_numel(r)
    # stand-in metanetwork: memory_states [P, L, M, H] -> plain [P, numel]
    flat = cfg.num_hidden_layers * meta.num_mem_token * cfg.hidden_size
    lin = nn.Linear(flat, numel)
    metanetwork = lambda mem: lin(mem.reshape(mem.shape[0], -1))

    ctx = list(range(1, 41))                  # 40-token context
    qids = torch.randint(0, cfg.vocab_size, (1, 6)); qmask = torch.ones_like(qids)
    metalora = meta.init_lora_dict(r, scale, "cpu")

    combined = mc.generate_lora_dict_multichunk(
        meta, metanetwork, ctx, qids, qmask, metalora, r, scale,
        n_sink=4, n_local=8, page_size=8, top_k=2, norm_rule="energy", norm_lam=1.0, var_rank=True)

    # combined LoRA is Lb=1 and rank-stacked over kept pages
    q = combined[3]["attention"]["q"]
    assert q["A"].shape[0] == 1 and q["B"].shape[0] == 1
    kept_rank = q["A"].shape[-1]
    assert kept_rank % r == 0
    print("ok multichunk combine: Lb=1, kept pages=%d (combined rank=%d)" % (kept_rank // r, kept_rank))
    # attention-only scope: LoRA exists only on the full-attention layer (3), no MLP / linear layers
    assert set(combined.keys()) == {3} and set(combined[3]) == {"attention"}
    print("ok scoped attention-only: LoRA only on full-attn layer 3 (q/k/v/o), nowhere else")

    # apply the combined LoRA to a question forward + generate
    base = meta(input_ids=qids, attention_mask=qmask, loradict=None, ignore_mem_token=True).logits
    out = meta(input_ids=qids, attention_mask=qmask, loradict=combined, ignore_mem_token=True)
    assert out.logits.shape == base.shape and not torch.allclose(base, out.logits)
    gen = meta.generate(input_ids=qids, attention_mask=qmask, loradict=combined, max_new_tokens=3)
    assert gen.shape == (1, 9)
    print("ok applied combined LoRA: logits change (%.4f), generate -> %s" % (
        (base - out.logits).abs().mean(), tuple(gen.shape)))
    print("\nMULTICHUNK SMOKE PASSED")


if __name__ == "__main__":
    main()
