"""CPU unit tests for SHINE's query_aware math. Run: python tests/test_query_aware.py"""
import importlib.util, os, torch

_HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("query_aware", os.path.join(_HERE, "..", "query_aware.py"))
qa = importlib.util.module_from_spec(spec); spec.loader.exec_module(qa)
torch.manual_seed(0)


def _mk_loradict(Lb=4, layers=(0, 1), din=16, r=8, dout=12, bias=True):
    d = {}
    for li in layers:
        d[li] = {"attention": {}, "mlp": {}}
        for proj in ("q", "k", "v", "o"):
            d[li]["attention"][proj] = {"A": torch.randn(Lb, din, r), "B": torch.randn(Lb, r, dout),
                                        "C": torch.randn(Lb, dout) if bias else None}
        for proj in ("gate", "up", "down"):
            d[li]["mlp"][proj] = {"A": torch.randn(Lb, din, r), "B": torch.randn(Lb, r, dout),
                                  "C": torch.randn(Lb, dout) if bias else None}
    return d


def test_frob_norm_AB():
    A = torch.randn(4, 16, 8); B = torch.randn(4, 8, 12)
    got = qa.frob_norm_AB(A, B)
    ref = torch.linalg.matrix_norm(A @ B, ord="fro")
    assert torch.allclose(got, ref, atol=1e-4), (got - ref).abs().max()
    print("ok frob_norm_AB (dW=A@B)  max_err=%.2e" % (got - ref).abs().max())


def test_normalize_pins_norm():
    d = _mk_loradict(Lb=3)
    tgt = {(li, proj): torch.rand(3) * 2 + 0.5 for li, _, proj, _ in qa.iter_leaves(d)}
    out = qa.normalize_loradict(d, tgt)
    for li, g, proj, leaf in qa.iter_leaves(out):
        nrm = qa.frob_norm_AB(leaf["A"], leaf["B"])
        assert torch.allclose(nrm, tgt[(li, proj)], atol=1e-4), (li, proj, (nrm - tgt[(li, proj)]).abs().max())
    print("ok normalize_loradict pins ||A@B||_F for every (layer,proj)")


def test_mask_to_ranks():
    d = _mk_loradict(Lb=3, layers=(0,))
    ranks = torch.tensor([2, 5, 8])
    out = qa.mask_loradict_to_ranks(d, ranks)
    leaf0, leafm = out[0]["attention"]["q"], d[0]["attention"]["q"]
    for c, rr in enumerate(ranks.tolist()):
        ref = qa.frob_norm_AB(leafm["A"][c:c+1, :, :rr], leafm["B"][c:c+1, :rr])
        got = qa.frob_norm_AB(leaf0["A"][c:c+1], leaf0["B"][c:c+1])
        assert torch.allclose(got, ref, atol=1e-5), (c, (got-ref).abs().max())
        if rr < 8:
            assert leaf0["A"][c, :, rr:].abs().max() == 0 and leaf0["B"][c, rr:].abs().max() == 0
    print("ok mask_loradict_to_ranks: effective rank == r_page per page")


def test_combine_chunk_loras():
    Lb, din, r, dout = 4, 16, 8, 12
    d = _mk_loradict(Lb=Lb, layers=(0,))
    w = torch.softmax(torch.randn(Lb), 0)  # sums to 1
    out = qa.combine_chunk_loras(d, w)
    leaf, base = out[0]["mlp"]["down"], d[0]["mlp"]["down"]
    dW = leaf["A"][0] @ leaf["B"][0]               # [in,out]
    ref = sum(w[c] * (base["A"][c] @ base["B"][c]) for c in range(Lb))
    assert torch.allclose(dW, ref, atol=1e-4), (dW - ref).abs().max()
    C_ref = sum(w[c] * base["C"][c] for c in range(Lb))
    assert torch.allclose(leaf["C"][0], C_ref, atol=1e-5)
    assert leaf["A"].shape == (1, din, Lb * r) and leaf["B"].shape == (1, Lb * r, dout)
    print("ok combine_chunk_loras: A@B == Σ w_c A_c B_c, C == Σ w_c C_c, rank=%d" % (Lb * r))


def test_quest_and_topk():
    d, n, seq = 12, 5, 7
    q = torch.randn(d); K = torch.randn(n, seq, d)
    got = qa.quest_scores(q, K.min(1).values, K.max(1).values)
    ref = torch.maximum(q * K.max(1).values, q * K.min(1).values).sum(-1)
    assert torch.allclose(got, ref, atol=1e-5)
    w = qa.topk_softmax_weights(torch.tensor([5., .1, 3., .2, 9.]), top_k=1,
                                force_keep=torch.tensor([True, False, False, False, True]))
    assert abs(w.sum() - 1) < 1e-6 and w[0] > 0 and w[4] > 0 and w[2] > 0 and w[1] == 0 and w[3] == 0
    print("ok quest_scores==brute, force-keep topk keeps sink/local + top-1 middle")


def test_paging_and_select():
    out = qa.build_paged_evidence([list(range(1, 201)), list(range(1, 21))],
                                  n_sink=4, n_local=32, page_size=16)
    assert out["n_ctx_chunks"].tolist() == [13, 2]
    assert out["chunk_ranks"][:13].tolist() == [4] + [16]*10 + [4, 32]
    assert out["page_ranges"][0][0][0] == 0 and out["page_ranges"][0][-1][1] == 200
    d = _mk_loradict(Lb=5, layers=(0,))
    sel = qa.select_pages(d, torch.tensor([0, 2, 4]))
    assert sel[0]["mlp"]["up"]["A"].shape[0] == 3
    print("ok paging: n_ctx=%s ranks0=%s ; select_pages keeps 3" % (
        out["n_ctx_chunks"].tolist(), out["chunk_ranks"][:13].tolist()))


if __name__ == "__main__":
    test_frob_norm_AB()
    test_normalize_pins_norm()
    test_mask_to_ranks()
    test_combine_chunk_loras()
    test_quest_and_topk()
    test_paging_and_select()
    print("\nALL PASSED")
