"""CPU dry-run smoke for the Qwen3.5 LoRA metamodel wrapper.
Builds a TINY hybrid Qwen3.5 text model (random weights) and exercises the
SHINE metamodel interface end-to-end on CPU. Run: python tests/test_lora_qwen35_smoke.py
"""
import importlib.util, os, torch

_HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("lora_qwen35", os.path.join(_HERE, "..", "lora_qwen35.py"))
lq = importlib.util.module_from_spec(spec); spec.loader.exec_module(lq)

from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextModel

torch.manual_seed(0)


def tiny_config():
    return Qwen3_5TextConfig(
        vocab_size=100, hidden_size=64, intermediate_size=128, num_hidden_layers=4,
        num_attention_heads=4, num_key_value_heads=2, head_dim=16,
        layer_types=["linear_attention", "linear_attention", "linear_attention", "full_attention"],
        linear_conv_kernel_dim=4, linear_key_head_dim=16, linear_num_key_heads=2,
        linear_value_head_dim=16, linear_num_value_heads=4,
        rms_norm_eps=1e-6, rope_parameters={"rope_theta": 10000.0, "partial_rotary_factor": 0.25, "rope_type": "default"},
        attn_implementation="eager",
    )


def build(num_mem_token=2):
    cfg = tiny_config()
    text = Qwen3_5TextModel(cfg).eval()
    lm_head = torch.nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
    meta = lq.LoraQwen35(text, lm_head, cfg, num_mem_token=num_mem_token).eval()
    meta.set_generate_func("rl")
    return cfg, meta


def main():
    cfg, meta = build()
    r, scale, B = 2, 0.001, 2

    # 1) site discovery: MLP on all 4 layers, attention only on layer 3
    n_mlp = sum(1 for *_, in meta.sites if _[1] == "mlp")
    n_attn = sum(1 for s in meta.sites if s[1] == "attention")
    assert meta.full_attn_layers == [3], meta.full_attn_layers
    assert n_mlp == 4 * 3 and n_attn == 1 * 4, (n_mlp, n_attn)
    print("ok sites: %d total (mlp=%d all-layers, attn=%d full-attn-only); full_attn_layers=%s"
          % (len(meta.sites), n_mlp, n_attn, meta.full_attn_layers))

    # 2) param layout + generate_lora_dict shapes (derive dims from the real Linears)
    site_io = {(li, g, p): (lin.in_features, lin.out_features) for li, g, p, lin in meta.sites}
    numel = meta.lora_params_numel(r)
    assert numel == sum(i * r + o * r for i, o in site_io.values()), "numel mismatch"
    plain = torch.randn(B, numel)
    ld = meta.generate_lora_dict(r, scale, plain)
    for (li, g, p), (din, dout) in site_io.items():
        leaf = ld[li][g][p]
        assert leaf["A"].shape == (B, din, r) and leaf["B"].shape == (B, r, dout), (li, g, p, leaf["A"].shape)
        assert leaf["C"] is None  # qwen3.5 projections are bias-free
    assert "attention" not in ld.get(0, {}), "layer 0 (linear-attn) must have no attention LoRA"
    print("ok generate_lora_dict: numel=%d, all %d sites shaped [B,in,r]/[B,r,out], scoped" % (numel, len(meta.sites)))

    # 3) memory-token forward -> memory_states [B, num_layers, num_mem_token, hidden]
    ev = torch.randint(0, cfg.vocab_size, (B, 12))
    ev_mask = torch.ones_like(ev)
    out = meta(input_ids=ev, attention_mask=ev_mask, loradict=meta.init_lora_dict(r, scale, "cpu"))
    assert out.memory_states.shape == (B, cfg.num_hidden_layers, meta.num_mem_token, cfg.hidden_size), out.memory_states.shape
    print("ok memory forward: memory_states %s" % (tuple(out.memory_states.shape),))

    # 4) applying the generated loradict changes the question logits
    qids = torch.randint(0, cfg.vocab_size, (B, 6))
    qmask = torch.ones_like(qids)
    base = meta(input_ids=qids, attention_mask=qmask, loradict=None, ignore_mem_token=True).logits
    lora = meta(input_ids=qids, attention_mask=qmask, loradict=ld, ignore_mem_token=True).logits
    assert base.shape == lora.shape and not torch.allclose(base, lora), "LoRA had no effect"
    print("ok LoRA application changes logits (delta=%.4f)" % (base - lora).abs().mean())

    # 5) loss + generate
    labels = qids.clone()
    loss = meta(input_ids=qids, attention_mask=qmask, loradict=ld, ignore_mem_token=True, labels=labels).loss
    assert loss is not None and torch.isfinite(loss)
    gen = meta.generate(input_ids=qids, attention_mask=qmask, loradict=ld, max_new_tokens=3)
    assert gen.shape[1] == qids.shape[1] + 3
    print("ok loss=%.3f finite; greedy generate -> %s tokens" % (loss.item(), tuple(gen.shape)))

    print("\nSMOKE PASSED")


if __name__ == "__main__":
    main()
