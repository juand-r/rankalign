"""Smoke test for the FIX1 g-mode pair construction logic.

Replays the partition + enumerate + sample stages on synthetic Z data
to confirm:
  - 4 pair shapes are partitioned correctly
  - delta filter applied
  - stratified sampling hits requested per-shape budgets
  - never produces an "invalid" pair (L+ on lo, L- on hi, etc.)

Run: python scripts/_smoke_fix1_pairs.py
"""
import random
import itertools
from collections import Counter


def build_synthetic_Z(n_pos=75, n_neg=75, n_unl=1350, seed=42):
    """Return Z = [(prompt_obj, val_score, data_item, typ, is_labeled), ...]
    matching the shape that ranking_loss_ref_fix.py expects.
    val_score is sampled from N(label_mu, 1) so labeled positives lean high
    and labeled negatives lean low (informative validator)."""
    rng = random.Random(seed)
    Z = []
    for _ in range(n_pos):
        v = rng.gauss(0.5, 1.0)
        di = type("DI", (), {"correct": "yes"})()
        Z.append((None, v, di, 0.0, True))
    for _ in range(n_neg):
        v = rng.gauss(-0.5, 1.0)
        di = type("DI", (), {"correct": "no"})()
        Z.append((None, v, di, 0.0, True))
    for _ in range(n_unl):
        v = rng.gauss(0.0, 1.0)
        di = type("DI", (), {"correct": "yes" if rng.random() < 0.5 else "no"})()
        Z.append((None, v, di, 0.0, False))
    Z.sort(key=lambda z: z[1])
    return Z


def label_class(z):
    if not z[4]:
        return 'U'
    return 'L_pos' if z[2].correct == 'yes' else 'L_neg'


def main():
    Z = build_synthetic_Z()
    delta = 0.15
    total_samples = 5110
    shape_weights = {'case_A': 0.20, 'mixed_neg': 0.20, 'mixed_pos': 0.20, 'both_U': 0.40}

    L_pos_ix, L_neg_ix, U_ix = [], [], []
    for k, z in enumerate(Z):
        c = label_class(z)
        if c == 'L_pos': L_pos_ix.append(k)
        elif c == 'L_neg': L_neg_ix.append(k)
        else: U_ix.append(k)

    print(f"|L+|={len(L_pos_ix)}  |L-|={len(L_neg_ix)}  |U|={len(U_ix)}  total={len(Z)}")

    def enumerate_shape(lo_pool, hi_pool):
        pairs = []
        for i in lo_pool:
            v_i = Z[i][1]
            for j in hi_pool:
                if i == j: continue
                v_j = Z[j][1]
                if v_i < v_j and (v_j - v_i) > delta:
                    pairs.append((i, j))
        return pairs

    pool = {
        'case_A':    enumerate_shape(L_neg_ix, L_pos_ix),
        'mixed_neg': enumerate_shape(L_neg_ix, U_ix),
        'mixed_pos': enumerate_shape(U_ix,     L_pos_ix),
        'both_U':    enumerate_shape(U_ix,     U_ix),
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
    for shape, w in shape_weights.items():
        target = int(round(total_samples * w / w_sum))
        avail = pool[shape]
        take = min(target, len(avail))
        if take > 0:
            pair_inds.extend(rng.sample(avail, take))
        sampled[shape] = take

    deficit = total_samples - len(pair_inds)
    print(f"\nDeficit before backfill: {deficit}")

    print(f"\nSampled per shape:")
    for shape in ('case_A', 'mixed_neg', 'mixed_pos', 'both_U'):
        n = sampled.get(shape, 0)
        avail = len(pool[shape])
        w = shape_weights[shape]
        print(f"  {shape:10s}: {n:6d}/{avail:8d}  (weight={w}, target={int(round(total_samples * w / w_sum))})")

    print(f"\nValidating pair shapes...")
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
            print(f"  BUG: val(i) >= val(j)! i={i} ({Z[i][1]:.3f}) j={j} ({Z[j][1]:.3f})")
            bad += 1
        if (Z[j][1] - Z[i][1]) <= delta:
            print(f"  BUG: delta filter violated!")
            bad += 1

    print(f"\n  Observed (lo_class, hi_class) shapes:")
    for k, v in sorted(seen_shapes.items()):
        print(f"    {k}: {v}")

    if bad == 0:
        print("\n  PASS: zero invalid pairs, zero delta violations.")
    else:
        print(f"\n  FAIL: {bad} bad pairs.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
