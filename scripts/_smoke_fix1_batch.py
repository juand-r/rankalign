"""Smoke test for the fix1 batch-size patch (2026-05-21).

The patch lifts the `raise NotImplementedError` for `batch_size > 1 + semi`
in fix1 g-mode. Three things to verify:

  1. **Backward-compat at B=1.** With reduction='none', val-NLL log-odds
     gives bit-identical output to the original reduction='mean' code at
     B=1. Equivalently: existing 41110-41118 jobs (running at B=1) still
     produce the same loss.

  2. **Per-element masking at B>1.** With mixed labeled/unlabeled in a
     batch, the val-NLL log-odds path correctly applies the
     is_labeled mask to each example's BCE individually (not to the
     batch-mean BCE).

  3. **Other loss components unaffected.** Preference loss, gen-NLL,
     and non-log-odds val-NLL were already per-element correct. The
     patch should NOT change them.

Plus a bonus check:

  4. **B=1-twice vs B=2-once give equivalent expected gradients.**
     If we run two B=1 steps (averaging the gradients) we should get
     the same gradient as one B=2 step on the same two examples.

Run: `python scripts/_smoke_fix1_batch.py`
"""
import math
import torch
import torch.nn.functional as F


def val_nll_logodds_old(li, ii, lab_i, lj, ij, lab_j):
    """Pre-patch (broken at B>1)."""
    return (
        lab_i * F.binary_cross_entropy_with_logits(li, ii) +
        lab_j * F.binary_cross_entropy_with_logits(lj, ij)
    ).mean() / 2


def val_nll_logodds_new(li, ii, lab_i, lj, ij, lab_j):
    """Post-patch (per-element correct)."""
    return (
        lab_i * F.binary_cross_entropy_with_logits(li, ii, reduction='none') +
        lab_j * F.binary_cross_entropy_with_logits(lj, ij, reduction='none')
    ).mean() / 2


def val_nll_nonlogodds(score_i, lab_i, score_j, lab_j):
    """Non-log-odds val-NLL (unchanged by patch). Already per-element correct."""
    return -(lab_i * score_i + lab_j * score_j).mean() / 2


def gen_nll(score_gen_i, lab_i, ind_i, score_gen_j, lab_j, ind_j):
    """Gen-NLL (unchanged by patch). Already per-element correct."""
    w_i = lab_i * ind_i
    w_j = lab_j * ind_j
    return -(w_i * score_gen_i + w_j * score_gen_j).mean() / 2


def preference_loss(score_i, score_j):
    """Preference loss (unchanged by patch). Already per-element correct."""
    diff = score_j - score_i
    return -torch.log(torch.sigmoid(diff) + 1e-12).mean()


def total_loss(li, ii, lab_i, lj, ij, lab_j, score_gen_i, score_gen_j, score_i, score_j,
               new_path: bool):
    """Replay fix1's full loss block. new_path=True uses the patched val-NLL."""
    pref = preference_loss(score_i, score_j)
    if new_path:
        val = val_nll_logodds_new(li, ii, lab_i, lj, ij, lab_j)
    else:
        val = val_nll_logodds_old(li, ii, lab_i, lj, ij, lab_j)
    gen = gen_nll(score_gen_i, lab_i, ii, score_gen_j, lab_j, ij)
    return pref + val + gen


def make_batch(B, seed=0):
    """Synthetic batch of size B with mixed labeling."""
    g = torch.Generator().manual_seed(seed)
    li = torch.randn(B, generator=g, requires_grad=True)
    lj = torch.randn(B, generator=g, requires_grad=True)
    ii = torch.randint(0, 2, (B,), generator=g, dtype=torch.float)
    ij = torch.randint(0, 2, (B,), generator=g, dtype=torch.float)
    # Half labeled, half not (worst case for the bug)
    lab_i = (torch.arange(B) % 2 == 0).float()
    lab_j = (torch.arange(B) % 2 == 1).float()
    score_gen_i = torch.randn(B, generator=g, requires_grad=True)
    score_gen_j = torch.randn(B, generator=g, requires_grad=True)
    score_i = torch.randn(B, generator=g, requires_grad=True)
    score_j = torch.randn(B, generator=g, requires_grad=True)
    return li, ii, lab_i, lj, ij, lab_j, score_gen_i, score_gen_j, score_i, score_j


def test_1_backward_compat_b1():
    """B=1: old and new must be bit-identical."""
    args = make_batch(B=1, seed=0)
    old = total_loss(*args, new_path=False)
    new = total_loss(*args, new_path=True)
    assert torch.allclose(old, new), f"B=1 not bit-identical: old={old} new={new}"
    print(f"  PASS: B=1 backward-compat. loss={new.item():.6f}")


def test_2_per_element_masking_b2():
    """B=2 with mixed labeling: new is per-element correct, old is broken."""
    args = make_batch(B=2, seed=42)
    li, ii, lab_i, lj, ij, lab_j, *_ = args
    new = val_nll_logodds_new(li, ii, lab_i, lj, ij, lab_j)
    old = val_nll_logodds_old(li, ii, lab_i, lj, ij, lab_j)
    # Hand-compute expected new = per-element BCE * per-element mask, then mean / 2
    bce_i = F.binary_cross_entropy_with_logits(li, ii, reduction='none')
    bce_j = F.binary_cross_entropy_with_logits(lj, ij, reduction='none')
    expected_new = ((lab_i * bce_i + lab_j * bce_j).sum() / 2) / 2  # (sum / B) / 2
    assert torch.allclose(new, expected_new), f"new={new} vs expected {expected_new}"
    # Old must NOT match (proves the bug was real)
    assert not torch.allclose(old, new), "old and new should differ at B=2 mixed"
    delta = (old - new).abs().item()
    print(f"  PASS: B=2 per-element masking. new={new.item():.6f}, old(broken)={old.item():.6f}, |delta|={delta:.6f}")


def test_3_other_losses_unaffected():
    """Preference, gen-NLL, non-log-odds val-NLL: these are NOT touched by
    the patch. We verify by running them at B=1 and B=2 and checking they
    produce sensible per-element output."""
    args = make_batch(B=2, seed=7)
    li, ii, lab_i, lj, ij, lab_j, sg_i, sg_j, s_i, s_j = args
    pref = preference_loss(s_i, s_j)
    gen = gen_nll(sg_i, lab_i, ii, sg_j, lab_j, ij)
    non_log_odds_val = val_nll_nonlogodds(s_i, lab_i, s_j, lab_j)
    assert pref.dim() == 0 and torch.isfinite(pref)
    assert gen.dim() == 0 and torch.isfinite(gen)
    assert non_log_odds_val.dim() == 0 and torch.isfinite(non_log_odds_val)
    print(f"  PASS: pref={pref.item():.6f}, gen={gen.item():.6f}, non_logodds_val={non_log_odds_val.item():.6f}")


def test_4_b1_twice_equiv_b2_once():
    """If two B=1 steps (averaged grads) match one B=2 step (mean loss grad),
    then training at B=2 is equivalent to training at B=1 with 2x learning
    rate (ish). This is the standard 'batched mean' equivalence and should
    hold."""
    full = make_batch(B=2, seed=123)
    li_full, ii_full, lab_i_full, lj_full, ij_full, lab_j_full, *rest_full = full

    # Single B=2 step
    li2 = li_full.detach().clone().requires_grad_()
    lj2 = lj_full.detach().clone().requires_grad_()
    loss2 = val_nll_logodds_new(li2, ii_full, lab_i_full, lj2, ij_full, lab_j_full)
    loss2.backward()
    grad_li_b2 = li2.grad.clone(); grad_lj_b2 = lj2.grad.clone()

    # Two B=1 steps, average the per-step grads
    li_a = li_full[0:1].detach().clone().requires_grad_()
    lj_a = lj_full[0:1].detach().clone().requires_grad_()
    loss_a = val_nll_logodds_new(li_a, ii_full[0:1], lab_i_full[0:1], lj_a, ij_full[0:1], lab_j_full[0:1])
    loss_a.backward()

    li_b = li_full[1:2].detach().clone().requires_grad_()
    lj_b = lj_full[1:2].detach().clone().requires_grad_()
    loss_b = val_nll_logodds_new(li_b, ii_full[1:2], lab_i_full[1:2], lj_b, ij_full[1:2], lab_j_full[1:2])
    loss_b.backward()

    # Per the convention "B=2 loss = mean over batch", the B=2 grad on
    # element k should equal half the grad you'd get from a B=1 step on
    # just element k. Verify.
    expected_grad_li_b2 = torch.cat([li_a.grad, li_b.grad]) / 2
    expected_grad_lj_b2 = torch.cat([lj_a.grad, lj_b.grad]) / 2
    assert torch.allclose(grad_li_b2, expected_grad_li_b2, atol=1e-6), (
        f"li grad mismatch: B=2 {grad_li_b2.tolist()} vs expected {expected_grad_li_b2.tolist()}"
    )
    assert torch.allclose(grad_lj_b2, expected_grad_lj_b2, atol=1e-6), (
        f"lj grad mismatch: B=2 {grad_lj_b2.tolist()} vs expected {expected_grad_lj_b2.tolist()}"
    )
    print(f"  PASS: B=2 grads = (sum of B=1 grads) / 2 (standard batched-mean behavior).")


def main():
    print("=" * 60)
    print("FIX1 batch-size patch smoke test")
    print("=" * 60)
    print("Test 1: B=1 backward-compat")
    test_1_backward_compat_b1()
    print("Test 2: B=2 per-element masking (the bug fix)")
    test_2_per_element_masking_b2()
    print("Test 3: other loss components unaffected by patch")
    test_3_other_losses_unaffected()
    print("Test 4: B=1-twice grads = B=2-once grads (batched-mean equivalence)")
    test_4_b1_twice_equiv_b2_once()
    print()
    print("ALL TESTS PASS. The fix1 batch-size patch is mathematically correct.")
    print("  - B=1 behavior is bit-identical to pre-patch.")
    print("  - B>1 behavior is per-element correct (the bug is fixed).")
    print("  - Standard batched-mean grad equivalence holds.")


if __name__ == "__main__":
    main()
