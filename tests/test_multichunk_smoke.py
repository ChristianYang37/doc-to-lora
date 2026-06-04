"""CPU smoke for the metamodel-agnostic multi-chunk path (doc2lora x SHINE).
Exercises BOTH metamodels on tiny random models:
  - Qwen3   : original LoraQwen.LoraQwen3ForCausalLM (all-linear)
  - Qwen3.5 : lora_qwen35.LoraQwen35 (attention-only, hybrid)
Run: python tests/test_multichunk_smoke.py
"""
import os, sys, torch, torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import multichunk as mc
import lora_qwen35 as lq
import LoraQwen
from transformers import Qwen3Config
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextModel
torch.manual_seed(0)


class MetanetStub:
    """Minimal stand-in for SHINE's Metanetwork (same generate_lora_dict contract)."""
    def __init__(self, metamodel, lora_r, scale, num_mem_token):
        self.metamodel = metamodel
        self.lora_r = lora_r
        self.scale = scale
        H = metamodel.config.hidden_size
        L = metamodel.config.num_hidden_layers
        self.lin = nn.Linear(L * num_mem_token * H, metamodel.lora_params_numel(lora_r))

    def generate_lora_dict(self, evidence_ids, evidence_attention_mask, metalora, **kw):
        out = self.metamodel(input_ids=evidence_ids, attention_mask=evidence_attention_mask,
                             loradict=metalora, ignore_mem_token=False)
        mem = out.memory_states
        plain = self.lin(mem.reshape(mem.shape[0], -1))
        return self.metamodel.generate_lora_dict(self.lora_r, self.scale, plain)


def build_qwen35(r=8, scale=0.01, mem=2):
    cfg = Qwen3_5TextConfig(
        vocab_size=100, hidden_size=64, intermediate_size=128, num_hidden_layers=4,
        num_attention_heads=4, num_key_value_heads=2, head_dim=16,
        layer_types=["linear_attention", "linear_attention", "linear_attention", "full_attention"],
        linear_conv_kernel_dim=4, linear_key_head_dim=16, linear_num_key_heads=2,
        linear_value_head_dim=16, linear_num_value_heads=4, rms_norm_eps=1e-6,
        rope_parameters={"rope_theta": 1e4, "partial_rotary_factor": 0.25, "rope_type": "default"},
        attn_implementation="eager")
    text = Qwen3_5TextModel(cfg).eval()
    meta = lq.LoraQwen35(text, nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False), cfg, mem).eval()
    meta.set_generate_func("rl")
    return meta, mem


def build_qwen3(r=8, scale=0.01, mem=2):
    cfg = Qwen3Config(vocab_size=100, hidden_size=64, intermediate_size=128, num_hidden_layers=4,
                      num_attention_heads=4, num_key_value_heads=2, head_dim=16, attn_implementation="eager")
    cfg.num_mem_token = mem
    meta = LoraQwen.LoraQwen3ForCausalLM(cfg).eval()
    meta.set_generate_func("rl")
    return meta, mem


def run(name, meta, mem, r=8, scale=0.01, full_attn_layers=None):
    metanet = MetanetStub(meta, r, scale, mem)
    metalora = meta.init_lora_dict(r, scale, "cpu")
    ctx = list(range(1, 41))
    qids = torch.randint(0, meta.config.vocab_size, (1, 6)); qmask = torch.ones_like(qids)
    combined = mc.generate_lora_dict_multichunk(
        metanet, ctx, qids, qmask, metalora,
        n_sink=4, n_local=8, page_size=8, top_k=2, query_aware_mix=True,
        norm_rule="energy", norm_lam=1.0, var_rank=True)
    # combined LoRA exists exactly on the expected layers
    assert set(combined.keys()) == set(full_attn_layers), (name, sorted(combined.keys()))
    some = next(iter(combined.values()))
    leaf = next(iter(next(iter(some.values())).values()))
    assert leaf["A"].shape[0] == 1 and leaf["A"].shape[-1] % r == 0
    print("ok %-7s multichunk: combined layers=%s, kept pages=%d" % (
        name, sorted(combined.keys()), leaf["A"].shape[-1] // r))
    # merge the combined LoRA back into the model weights -> standalone model
    import copy
    with torch.no_grad():
        applied = meta(input_ids=qids, attention_mask=qmask, loradict=combined, ignore_mem_token=True).logits
        m2 = copy.deepcopy(meta)
        mc.merge_loradict_into_model(m2, combined, idx=0)
        merged_out = m2(input_ids=qids, attention_mask=qmask, loradict=None, ignore_mem_token=True).logits
    assert torch.allclose(applied, merged_out, atol=1e-3), (applied - merged_out).abs().max()
    print("ok %-7s merge_loradict_into_model: merged-weights == LoRA-applied" % name)
    # training path: gradients flow back through the metanetwork
    metanet.lin.weight.grad = None
    leaf["A"].float().sum().backward()
    assert metanet.lin.weight.grad is not None, "no grad to metanetwork (training broken)"
    print("ok %-7s grad flows to metanetwork (training-capable)" % name)
    # batched helper (B=2), ragged ranks padded
    class MC:  # cfg.multichunk stand-in
        enabled = True; n_sink = 4; n_local = 8; page_size = 8; query_aware_mix = True
        mix_top_k = 2; mix_temp = 1.0; norm_rule = "energy"; norm_lam = 1.0; var_rank = True
    ev = torch.randint(1, meta.config.vocab_size, (2, 40)); evm = torch.ones_like(ev)
    q2 = torch.randint(0, meta.config.vocab_size, (2, 6)); q2m = torch.ones_like(q2)
    batched = mc.multichunk_lora_for_batch(metanet, ev, evm, q2, q2m, metalora, MC())
    bl = next(iter(next(iter(batched.values())).values()))
    bleaf = next(iter(bl.values()))
    assert bleaf["A"].shape[0] == 2
    print("ok %-7s batched: Lb=%d stacked loradict" % (name, bleaf["A"].shape[0]))


def main():
    m35, mem35 = build_qwen35()
    run("Qwen3.5", m35, mem35, full_attn_layers=[3])         # attention-only -> layer 3
    m3, mem3 = build_qwen3()
    run("Qwen3", m3, mem3, full_attn_layers=[0, 1, 2, 3])    # all-linear -> all layers
    print("\nMULTICHUNK (both metamodels) SMOKE PASSED")


if __name__ == "__main__":
    main()
