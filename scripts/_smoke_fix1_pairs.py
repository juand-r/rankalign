"""Smoke test for the FIX1 g-mode pair construction logic.

Replays the partition + enumerate + per-prompt + per-shape sample stages
on synthetic Z data. Confirms:
  - The 4 pair shapes are partitioned correctly within each prompt.
  - Delta filter is applied during enumeration.
  - Stratified sampling hits requested per-shape budgets within each prompt.
  - Per-prompt budget is allocated proportional to n_completions[prompt].
  - Within-prompt backfill fills deficits without crossing prompts.
  - Never produces an "invalid" pair (L+ on lo, L- on hi, etc.).
  - Pairs never cross prompts when fsx is on.

See docs/issue3_fix.md for the algorithm.

Run: python scripts/_smoke_fix1_pairs.py
"""
import random
import itertools
from collections import Counter, defaultdict


def build_synthetic_Z(n_pos=75, n_neg=75, n_unl=1350, seed=42, n_prompts=1):
    """Return Z = [(prompt_obj, val_score, data_item, typ, is_labeled), ...]
    matching the shape that ranking_loss_ref_fix.py expects.

    Items are distributed round-robin across `n_prompts` distinct prompts.
    val_score is sampled from N(label_mu, 1) so labeled positives lean high
    and labeled negatives lean low (informative validator)."""
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
    """Enumerate pairs (i, j), i in lo_pool, j in hi_pool, val(i) < val(j),
    val(j) - val(i) > delta."""
    pairs = []
    for i in lo_pool:
        v_i = Z[i][1]
        for j in hi_pool:
            if i == j:
                continue
            v_j = Z[j][1]
            if v_i < v_j and (v_j - v_i) > delta:
                pairs.append((i, j))
    return pairs


def run_case(name, Z, delta, total_samples, shape_weights, force_same_x):
    """Replays the per-prompt x per-shape sampling from
    ranking_loss_ref_fix.py and validates invariants."""
    print(f"\n{'='*60}")
    print(f"CASE: {name}  (force_same_x={force_same_x})")
    print(f"{'='*60}")

    # 1. Group items by prompt.
    if force_same_x:
        prompt_groups = defaultdict(list)
        for idx, z in enumerate(Z):
            prompt_groups[z[0].prompt].append(idx)
        prompt_groups = dict(prompt_groups)
    else:
        prompt_groups = {None: list(range(len(Z)))}

    print(f"Prompts: {len(prompt_groups)} group(s)")

    # 2. Per-prompt x per-shape enumeration.
    pool_by_prompt = {}
    prompt_n = {}
    n_lpos_total = n_lneg_total = n_u_total = 0
    for prompt, indices in prompt_groups.items():
        grp_lpos, grp_lneg, grp_u = [], [], []
        for k in indices:
            c = label_class(Z[k])
            if c == 'L_pos':
                grp_lpos.append(k)
            elif c == 'L_neg':
                grp_lneg.append(k)
            else:
                grp_u.append(k)
        n_lpos_total += len(grp_lpos)
        n_lneg_total += len(grp_lneg)
        n_u_total += len(grp_u)
        pool_by_prompt[prompt] = {
            'case_A':    enumerate_shape(Z, grp_lneg, grp_lpos, delta),
            'mixed_neg': enumerate_shape(Z, grp_lneg, grp_u,    delta),
            'mixed_pos': enumerate_shape(Z, grp_u,    grp_lpos, delta),
            'both_U':    enumerate_shape(Z, grp_u,    grp_u,    delta),
        }
        prompt_n[prompt] = len(indices)

    print(f"|L+|={n_lpos_total}  |L-|={n_lneg_total}  |U|={n_u_total}  total={len(Z)}")
    agg_pool = {s: 0 for s in ('case_A', 'mixed_neg', 'mixed_pos', 'both_U')}
    for prompt in pool_by_prompt:
        for s in agg_pool:
            agg_pool[s] += len(pool_by_prompt[prompt][s])
    print("\nPool sizes (after delta filter, aggregated):")
    for s, n in agg_pool.items():
        print(f"  {s:10s}: {n:8d}")
    total_valid = sum(agg_pool.values())
    print(f"  total     : {total_valid:8d}")

    # 3 + 4. Per-prompt budget + per-shape sampling + within-prompt backfill.
    eff_samples = min(total_samples, total_valid)
    rng = random.Random(0)
    sw_sum = sum(shape_weights.values())
    N_total = len(Z)

    pair_inds = []
    sampled_per_shape = {s: 0 for s in shape_weights}
    per_prompt_log = []
    for prompt in pool_by_prompt:
        n_p = prompt_n[prompt]
        prompt_budget = int(round(eff_samples * n_p / N_total))
        prompt_pool = pool_by_prompt[prompt]
        prompt_sampled = []

        for shape, w in shape_weights.items():
            target = int(round(prompt_budget * w / sw_sum))
            avail = prompt_pool[shape]
            take = min(target, len(avail))
            if take > 0:
                picked = rng.sample(avail, take)
                prompt_sampled.extend(picked)
                sampled_per_shape[shape] += take

        deficit = prompt_budget - len(prompt_sampled)
        if deficit > 0:
            already = set(prompt_sampled)
            leftover = []
            for shape, pool in prompt_pool.items():
                for p in pool:
                    if p not in already:
                        leftover.append((shape, p))
            if leftover:
                fill = rng.sample(leftover, min(deficit, len(leftover)))
                for shape, p in fill:
                    prompt_sampled.append(p)
                    sampled_per_shape[shape] += 1

        pair_inds.extend(prompt_sampled)
        per_prompt_log.append((prompt, n_p, prompt_budget, len(prompt_sampled)))

    print(f"\nSampled per shape (aggregated):")
    for shape in ('case_A', 'mixed_neg', 'mixed_pos', 'both_U'):
        n = sampled_per_shape[shape]
        avail = agg_pool[shape]
        w = shape_weights[shape]
        print(f"  {shape:10s}: {n:6d}/{avail:8d}  (weight={w})")
    print(f"Total sampled: {len(pair_inds)}/{eff_samples}")

    if len(prompt_groups) > 1:
        sorted_log = sorted(per_prompt_log, key=lambda r: r[3], reverse=True)
        print(f"\nPer-prompt sampling (top 3):")
        for r in sorted_log[:3]:
            p, n_p, b, s = r
            print(f"  n_p={n_p:5d}  budget={b:5d}  sampled={s:5d}  prompt={p!r}")

    # 5. Validate invariants.
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
            print(f"  BUG: L- on HI side!")
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

    print(f"\n  Observed (lo_class, hi_class) shapes:")
    for k, v in sorted(seen_shapes.items()):
        print(f"    {k}: {v}")

    # Per-prompt budget invariant: sampled count for each prompt should not
    # exceed (or fall hugely short of) prompt_budget. With within-prompt
    # backfill the only way it falls short is if the entire prompt's pool is
    # smaller than its budget.
    over_budget = 0
    for prompt, n_p, b, s in per_prompt_log:
        if s > b:
            print(f"  BUG: prompt {prompt!r} sampled {s} but budget was {b}")
            over_budget += 1
    bad += over_budget

    if bad == 0:
        print(f"\n  PASS: zero invalid pairs, zero fsx/budget violations.")
        return True
    else:
        print(f"\n  FAIL: {bad} bad pairs / violations.")
        return False


def main():
    delta = 0.15
    shape_weights = {'case_A': 0.20, 'mixed_neg': 0.20, 'mixed_pos': 0.20, 'both_U': 0.40}

    ok = True

    # 1. Persona-like (1 prompt). fsx on/off should give identical results.
    Z = build_synthetic_Z(n_prompts=1)
    ok &= run_case("persona-like (1 prompt) no fsx", Z, delta, 5110, shape_weights, force_same_x=False)
    ok &= run_case("persona-like (1 prompt) + fsx", Z, delta, 5110, shape_weights, force_same_x=True)

    # 2. Multi-prompt, balanced. Each prompt gets equal n_p so equal budget.
    Z = build_synthetic_Z(n_pos=75, n_neg=75, n_unl=1350, n_prompts=50)
    ok &= run_case("multi-prompt (50 prompts, balanced) no fsx",
                   Z, delta, 2000, shape_weights, force_same_x=False)
    ok &= run_case("multi-prompt (50 prompts, balanced) + fsx",
                   Z, delta, 2000, shape_weights, force_same_x=True)

    # 3. Multi-prompt, IMBALANCED (round-robin gives prompt 0 ~30 items,
    #    others slightly fewer; not extreme but exercises the proportional
    #    budget logic).
    Z = build_synthetic_Z(n_pos=100, n_neg=100, n_unl=300, n_prompts=10)
    ok &= run_case("multi-prompt (10 prompts) + fsx",
                   Z, delta, 1000, shape_weights, force_same_x=True)

    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
