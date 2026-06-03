"""CPU smoke for the RESTORED original Qwen3 SHINE metamodel on transformers >=5.2.
Verifies LoraQwen.py (all-linear, all-layers) still imports + runs. Run: python tests/test_lora_qwen3_smoke.py"""
import os, sys, torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import LoraQwen
from transformers import Qwen3Config


def main():
    cfg = Qwen3Config(vocab_size=100, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, head_dim=16, attn_implementation="eager")
    cfg.num_mem_token = 2
    model = LoraQwen.LoraQwen3ForCausalLM(cfg).eval()
    model.set_generate_func("rl")
    ids = torch.randint(0, 100, (1, 8)); mask = torch.ones_like(ids)

    out = model(input_ids=ids, attention_mask=mask, ignore_mem_token=True)
    assert hasattr(out, "logits")
    out2 = model(input_ids=ids, attention_mask=mask)
    assert out2.memory_states.shape == (1, cfg.num_hidden_layers, cfg.num_mem_token, cfg.hidden_size)
    print("ok Qwen3 restore: import + forward(ignore_mem) + forward(mem)->%s" % (tuple(out2.memory_states.shape),))

    # original Qwen3 LoRA layout is uniform all-linear (q,k,v,o,gate,up,down on every layer)
    numel = model.lora_params_numel(2)
    ld = model.generate_lora_dict(2, 0.01, torch.randn(1, numel))
    groups = ld[0]
    assert set(groups["attention"]) == {"q", "k", "v", "o"} and set(groups["mlp"]) == {"gate", "up", "down"}
    print("ok Qwen3 all-linear all-layers loradict (attn q/k/v/o + mlp gate/up/down per layer)")
    print("\nQWEN3 RESTORE SMOKE PASSED")


if __name__ == "__main__":
    main()
