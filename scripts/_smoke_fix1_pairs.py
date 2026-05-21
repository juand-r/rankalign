"""Smoke test for the FIX1 g-mode pair construction logic.

Replays the partition + enumerate + sample stages on synthetic Z data
to confirm:
  - 4 pair shapes are partitioned correctly
  - delta filter applied
  - stratified sampling hits requested per-shape budgets
  - never produces an "invalid" pair (L+ on lo, L- on hi, etc.)
  - --force-same-x composes correctly (pairs never cross prompts)

Run: python scripts/_smoke_fix1_pairs.py
"""
from collections import defaultdict
import random
import itertools
from collections import Counter


def build_synthetic_Z(n_pos=75, n_neg=75, n_unl=1350, seed=42, n_prompts=1):
    """Return Z = [(prompt_obj, val_score, data_item, typ, is_labeled), ...]
    matching the shape that ranking_loss_ref_fix.py expects.
    val_score is sampled from N(label_mu, 1) so labeled positives lean high
    and labeled negatives lean low (informative validator).
    n_prompts > 1 distributes items across that many distinct prompts
    (round-robin), letting us exercise the fsx code path."""
    rng = random.Random(seed)

    class P:
        def __init__(self, prompt):
            self.prompt = prompt

    Z = []
    for k in range(n_pos):
        v = rng.gauss(0.5, 1.0)
        di = type("DI", (), {"correct": "yes"})()
        Z.append((P(f"prompt_{k % n_prompts}"), v, di, 0.0, True))
    for k in range(n_neg):
        v = rng.gauss(-0.5, 1.0)
        di = type("DI", (), {"correct": "no"})()
        Z.append((P(f"prompt_{k % n_prompts}"), v, di, 0.0, True))
    for k in range(n_unl):
        v = rng.gauss(0.0, 1.0)
        di = type("DI", (), {"correct": "yes" if rng.random() < 0.5 else "no"})()
        Z.append((P(f"prompt_{k % n_prompts}"), v, di, 0.0, False))
    Z.sort(key=lambda z: z[1])
    return Z


def label_class(z):
    if not z[4]:
        return 'U'
    return 'L_pos' if z[2].correct == 'yes' else 'L_neg'


def enumerate_shape(Z, lo_pool, hi_pool, delta):
    pairs = []
    for i in lo_pool:
        v_i = Z[i][1]
        for j in hi_pool:
            if i == j: continue
            v_j = Z[j][1]
            if v_i < v_j and (v_j - v_i) > delta:
                pairs.append((i, j))
    return pairs


def run_case(name, Z, delta, total_samples, shape_weights, force_same_x):
    print(f"\n{'='*60}")
    print(f"CASE: {name}  (force_same_x={force_same_x})")
    print(f"{'='*60}")

    if force_same_x:
        prompt_to_indices = defaultdict(list)
        for idx, z in enumerate(Z):
            prompt_to_indices[z[0].prompt].append(idx)
        pool = {'case_A': [], 'mixed_neg': [], 'mixed_pos': [], 'both_U': []}
        n_lpos = n_lneg = n_u = 0
        for prompt, indices in prompt_to_indices.items():
            grp_lpos, grp_lneg, grp_u = [], [], []
            for k in indices:
                c = label_class(Z[k])
                if c == 'L_pos': grp_lpos.append(k)
                elif c == 'L_neg': grp_lneg.append(k)
                else: grp_u.append(k)
            n_lpos += len(grp_lpos); n_lneg += len(grp_lneg); n_u += len(grp_u)
            pool['case_A'].extend(enumerate_shape(Z, grp_lneg, grp_lpos, delta))
            pool['mixed_neg'].extend(enumerate_shape(Z, grp_lneg, grp_u,  delta))
            pool['mixed_pos'].extend(enumerate_shape(Z, grp_u,    grp_lpos, delta))
            pool['both_U'].extend(enumerate_shape(Z, grp_u,       grp_u,    delta))
        print(f"|L+|={n_lpos}  |L-|={n_lneg}  |U|={n_u}  total={len(Z)}  prompts={len(prompt_to_indices)}")
    else:
        L_pos_ix, L_neg_ix, U_ix = [], [], []
        for k, z in enumerate(Z):
            c = label_class(z)
            if c == 'L_pos': L_pos_ix.append(k)
            elif c == 'L_neg': L_neg_ix.append(k)
            else: U_ix.append(k)
        print(f"|L+|={len(L_pos_ix)}  |L-|={len(L_neg_ix)}  |U|={len(U_ix)}  total={len(Z)}")
        pool = {
            'case_A':    enumerate_shape(Z, L_neg_ix, L_pos_ix, delta),
            'mixed_neg': enumerate_shape(Z, L_neg_ix, U_ix,     delta),
            'mixed_pos': enumerate_shape(Z, U_ix,     L_pos_ix, delta),
            'both_U':    enumerate_shape(Z, U_ix,     U_ix,     delta),
        }

    print("\nPool sizes (after delta filter):")
    for k, v in pool.items():
        print(f"  {k:10s}: {len(v):8d}")
    total_valid = sum(len(v) for v in pool.values())
    print(f"  total     : {total_valid:8d}")

    rng = random.Random(0)
    w_sum = sum(shape_weights.values())
    pair_inds = []
    sampled = {}
    eff_samples = min(total_samples, total_valid)
    for shape, w in shape_weights.items():
        target = int(round(eff_samples * w / w_sum))
        avail = pool[shape]
        take = min(target, len(avail))
        if take > 0:
            pair_inds.extend(rng.sample(avail, take))
        sampled[shape] = take

    print(f"\nSampled per shape:")
    for shape in ('case_A', 'mixed_neg', 'mixed_pos', 'both_U'):
        n = sampled.get(shape, 0)
        avail = len(pool[shape])
        w = shape_weights[shape]
        print(f"  {shape:10s}: {n:6d}/{avail:8d}  (weight={w})")

    bad = 0
    seen_shapes = Counter()
    for (i, j) in pair_inds:
        ci = label_class(Z[i])
        cj = label_class(Z[j])
        seen_shapes[(ci, cj)] += 1
        if ci == 'L_pos':
            print(f"  BUG: L+ on LO side! i={i} ({ci}) -> j={j} ({cj})")
            bad += 1
        if cj == 'L_neg':
            print(f"  BUG: L- on HI side! i={i} ({ci}) -> j={j} ({cj})")
            bad += 1
        if Z[i][1] >= Z[j][1]:
            print(f"  BUG: val(i) >= val(j)!")
            bad += 1
        if (Z[j][1] - Z[i][1]) <= delta:
            print(f"  BUG: delta filter violated!")
            bad += 1
        if force_same_x and Z[i][0].prompt != Z[j][0].prompt:
            print(f"  BUG: fsx violated! i prompt={Z[i][0].prompt!r} j prompt={Z[j][0].prompt!r}")
            bad += 1

    print(f"  Observed (lo_class, hi_class) shapes:")
    for k, v in sorted(seen_shapes.items()):
        print(f"    {k}: {v}")

    if bad == 0:
        print(f"\n  PASS: zero invalid pairs.")
    else:
        print(f"\n  FAIL: {bad} bad pairs.")
        return False
    return True


def main():
    delta = 0.15
    shape_weights = {'case_A': 0.20, 'mixed_neg': 0.20, 'mixed_pos': 0.20, 'both_U': 0.40}

    ok = True
    # 1. Persona-like (single prompt -> fsx is functionally a no-op).
    Z = build_synthetic_Z(n_prompts=1)
    ok &= run_case("persona-like (1 prompt)", Z, delta, 5110, shape_weights, force_same_x=False)
    ok &= run_case("persona-like (1 prompt) + fsx",
                   Z, delta, 5110, shape_weights, force_same_x=True)

    # 2. Multi-prompt (fsx actually filters): 50 prompts, 30 items each.
    Z = build_synthetic_Z(n_pos=75, n_neg=75, n_unl=1350, n_prompts=50)
    ok &= run_case("multi-prompt (50 prompts) no fsx",
                   Z, delta, 2000, shape_weights, force_same_x=False)
    ok &= run_case("multi-prompt (50 prompts) + fsx",
                   Z, delta, 2000, shape_weights, force_same_x=True)

    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
