"""CPU tests for dynamic per-layer LoRA fusion. Run: python tests/test_lora_fusion.py"""
import importlib.util
import os

import torch


_HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("lora_fusion", os.path.join(_HERE, "..", "lora_fusion.py"))
lf = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lf)
torch.manual_seed(0)


def _leaf(batch=2, chunks=3, din=4, rank=2, dout=5, hidden=6):
    return {
        "A": torch.randn(batch, chunks, din, rank),
        "B": torch.randn(batch, chunks, rank, dout),
        "C": torch.randn(batch, chunks, dout),
        lf.DYNAMIC_LEAF_KEY: True,
        lf.KEYS_KEY: torch.randn(batch, chunks, hidden),
        lf.MASK_KEY: torch.tensor([[True, True, False], [True, True, True]]),
        lf.TEMP_KEY: 1.0,
        lf.TOPK_KEY: None,
        lf.RESCALE_KEY: 1.0,
        "_fusion_cache": {},
        "_fusion_proj": "q",
    }


def test_qk_softmax_masks_invalid_chunks():
    leaf = _leaf()
    q = torch.randn(2, 4, 6)
    w = lf.compute_layer_weights(q, leaf)
    assert w.shape == (2, 1, 4, 3)
    assert torch.allclose(w.sum(-1), torch.ones_like(w[..., 0]), atol=1e-6)
    assert torch.equal(w[0, :, :, 2], torch.zeros_like(w[0, :, :, 2]))
    print("ok dynamic qk softmax: valid chunks sum to 1 and padded chunks get 0")


def test_dynamic_lora_update_matches_manual_sum():
    leaf = _leaf()
    x = torch.randn(2, 4, 4)
    w = lf.compute_layer_weights(torch.randn(2, 4, 6), leaf)
    got = lf.dynamic_lora_update(x, leaf, w)

    ref = torch.zeros_like(got)
    for b in range(2):
        for s in range(4):
            for c in range(3):
                ref[b, s] += w[b, 0, s, c] * ((x[b, s] @ leaf["A"][b, c]) @ leaf["B"][b, c] + leaf["C"][b, c])
    assert torch.allclose(got, ref, atol=1e-5), (got - ref).abs().max()
    print("ok dynamic LoRA update: delta == sum_c softmax(qk)_c * LoRA_c(x)")


def test_layer_wrapper_attaches_dynamic_metadata():
    leaf = _leaf(batch=1)
    base = {0: {"attention": {"q": {"A": leaf["A"], "B": leaf["B"], "C": leaf["C"]}}}}
    dyn = lf.make_dynamic_loradict(
        base,
        fusion_keys=torch.randn(1, 1, 3, 6),
        chunk_mask=torch.tensor([[True, True, False]]),
    )
    wrapped = lf.get_layer_loradict(dyn, 0)
    assert lf.is_dynamic_leaf(wrapped["attention"]["q"])
    assert wrapped["attention"]["q"][lf.KEYS_KEY].shape == (1, 3, 6)
    print("ok layer wrapper: dynamic metadata is attached to LoRA leaves")


if __name__ == "__main__":
    test_qk_softmax_masks_invalid_chunks()
    test_dynamic_lora_update_matches_manual_sum()
    test_layer_wrapper_attaches_dynamic_metadata()
    print("\nALL PASSED")
