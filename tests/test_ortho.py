"""CPU tests for the orthogonal LoRA update (arXiv 2505.11881 adapted).
Covers: ortho_alpha math, ΔW_⊥ ⟂ W, both LoraLinear forwards (flag on/off),
merge arithmetic, and the combine-level sequential orthogonalization.
Run: python tests/test_ortho.py
"""
import os, sys, torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import lora_ortho as lo
import query_aware as qa
import LoraQwen
import lora_qwen35
torch.manual_seed(0)


def test_ortho_alpha():
    Lb, din, r, dout = 3, 16, 4, 12
    A, B, W = torch.randn(Lb, din, r), torch.randn(Lb, r, dout), torch.randn(dout, din)
    got = lo.ortho_alpha(A, B, W)
    ref = ((A @ B) * W.t()).sum(dim=(-1, -2)) / ((W * W).sum() + 1e-8)   # <A@B, W^T>_F / ||W||_F^2
    assert torch.allclose(got, ref, atol=1e-4), (got - ref).abs().max()
    print("ok ortho_alpha == <A@B, W^T>_F / ||W||_F^2  (no A@B in impl)")


def test_perp_orthogonal():
    din, r, dout = 16, 4, 12
    A, B, W = torch.randn(1, din, r), torch.randn(1, r, dout), torch.randn(dout, din)
    alpha = lo.ortho_alpha(A, B, W)[0]
    Weff = W.t()
    perp = (A @ B)[0] - alpha * Weff                 # ΔW_⊥
    ip = (perp * Weff).sum()
    assert ip.abs() < 1e-3 * (Weff * Weff).sum(), ip
    print("ok ΔW_⊥ = A@B − α·W^T is ⟂ W^T  (⟨ΔW_⊥,W^T⟩=%.2e)" % ip)


def test_apply_forward():
    Lb, din, r, dout, S = 2, 16, 4, 12, 5
    A, B, W = torch.randn(Lb, din, r), torch.randn(Lb, r, dout), torch.randn(dout, din)
    x = torch.randn(Lb, S, din)
    base, lora_out = F.linear(x, W), (x @ A) @ B
    got = lo.apply_forward(base, lora_out, A, B, W, None, Lb, 1, dout, x.shape)
    alpha = lo.ortho_alpha(A, B, W).to(base.dtype)
    ref = (1 - alpha).view(Lb, 1, 1) * base + lora_out
    assert torch.allclose(got, ref, atol=1e-4), (got - ref).abs().max()
    print("ok apply_forward == (1−α)·x@W^T + (x@A)@B")


def _run_loralinear(which):
    din, dout, S = 16, 12, 5
    A, B = torch.randn(2, din, 4), torch.randn(2, 4, dout)
    x = torch.randn(2, S, din)
    if which == "qwen3":
        lin = LoraQwen.LoraLinear(din, dout, bias=False)
        fwd = lambda: lin.forward(x, {"A": A, "B": B, "C": None})
    else:
        lin = lora_qwen35.LoraLinear(din, dout, bias=False)
        lin._active = {"A": A, "B": B, "C": None}
        fwd = lambda: lin.forward(x)
    base, lora = F.linear(x, lin.weight), (x @ A) @ B
    alpha = lo.ortho_alpha(A, B, lin.weight)
    lo.ENABLED = True
    on = fwd()
    assert torch.allclose(on, (1 - alpha).view(2, 1, 1) * base + lora, atol=1e-4)
    lo.ENABLED = False
    off = fwd()
    assert torch.allclose(off, base + lora, atol=1e-4)
    lo.ENABLED = True
    print("ok %-6s LoraLinear.forward: ON=(1−α)x@W+lora, OFF=baseline" % which)


def test_merge_arithmetic():
    din, r, dout, S = 16, 4, 12, 5
    A, B, W = torch.randn(din, r), torch.randn(r, dout), torch.randn(dout, din)
    x = torch.randn(S, din)
    alpha = lo.ortho_alpha(A, B, W)
    W_merged = (1 - alpha) * W + (A @ B).t()        # the merge formula
    merged = F.linear(x, W_merged)
    fwd = (1 - alpha) * F.linear(x, W) + (x @ A) @ B
    assert torch.allclose(merged, fwd, atol=1e-3), (merged - fwd).abs().max()
    print("ok merge: F.linear(x, (1−α)W+(A@B)^T) == (1−α)x@W + (x@A)@B")


def test_combine_ortho_matches_manual():
    n, din, r, dout = 4, 16, 4, 12
    A = torch.randn(n, din, r, requires_grad=True)
    B = torch.randn(n, r, dout, requires_grad=True)
    ld = {0: {"attention": {"q": {"A": A, "B": B, "C": None}}}}
    w = torch.softmax(torch.randn(n), 0)
    orth = qa.combine_chunk_loras(ld, w, ortho=True)[0]["attention"]["q"]
    dW = (orth["A"][0] @ orth["B"][0])
    # manual sequential (descending weight)
    sw = w.sqrt(); Aw = A.detach() * sw[:, None, None]; Bw = B.detach() * sw[:, None, None]
    order = torch.argsort(w, descending=True)
    acc = torch.zeros(din, dout); coef = [torch.ones(()) for _ in range(n)]; proc = []
    for pos in range(n):
        j = int(order[pos])
        if proc:
            a = torch.einsum("ik,ij,kj->", Aw[j], acc, Bw[j]) / ((acc * acc).sum() + 1e-8)
            for jj in proc:
                coef[jj] = coef[jj] * (1 - a)
        acc = acc + Aw[j] @ Bw[j]; proc.append(j)
    ref = sum(coef[j] * (Aw[j] @ Bw[j]) for j in range(n))
    assert torch.allclose(dW, ref, atol=1e-3), (dW - ref).abs().max()
    # grad flows to the chunk A,B
    dW.sum().backward()
    assert A.grad is not None and B.grad is not None
    print("ok combine ortho == manual sequential (descending) + grad flows; coef=%s"
          % [round(float(c), 3) for c in coef])


def test_combine_orthogonal_chunks_noop():
    # chunks with disjoint input-row support -> A_j@B_j Frobenius-orthogonal -> coef≈1 -> == plain concat
    n, din, r, dout = 3, 12, 2, 12
    A = torch.zeros(n, din, r); B = torch.randn(n, r, dout)
    for j in range(n):
        A[j, j * 4:(j + 1) * 4, :] = torch.randn(4, r)
    ld = {0: {"mlp": {"down": {"A": A, "B": B, "C": None}}}}
    w = torch.ones(n) / n
    plain = qa.combine_chunk_loras(ld, w, ortho=False)[0]["mlp"]["down"]
    orth = qa.combine_chunk_loras(ld, w, ortho=True)[0]["mlp"]["down"]
    dwp = plain["A"][0] @ plain["B"][0]; dwo = orth["A"][0] @ orth["B"][0]
    assert torch.allclose(dwp, dwo, atol=1e-4), (dwp - dwo).abs().max()
    print("ok orthogonal chunks: ortho combine == plain weighted concat (coef≈1)")


if __name__ == "__main__":
    test_ortho_alpha()
    test_perp_orthogonal()
    test_apply_forward()
    _run_loralinear("qwen3")
    _run_loralinear("qwen35")
    test_merge_arithmetic()
    test_combine_ortho_matches_manual()
    test_combine_orthogonal_chunks_noop()
    print("\nORTHO TESTS PASSED")
