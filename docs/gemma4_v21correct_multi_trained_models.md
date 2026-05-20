# Trained Models: humaneval-v2.1correct-multi × gemma-4-31B-it

**Base model:** `google/gemma-4-31B-it`  
**Dataset:** `humaneval-v2.1correct-multi` (82 tasks, correct-only solutions)  
**Training script:** `pod-setup-train-scripts-gemma-4/run_settings_v21correct_multi.sh`  
**Training:** 3 epochs LoRA, δ=0.15, semi-supervised 0.1 (except s1)  
**Eval epoch:** 2  
**Adapter path pattern:** `v6-google--gemma-4-31B-it-delta0.15-epoch2--humaneval-v2.1correct-multi{SUFFIX}`

Run: 2026-05-19/20 across 5 RunPod H100 SXM pods.

---

## Pod Assignments

| Pod | RunPod ID | Settings |
|-----|-----------|----------|
| h100-train-4 | d4gum0mttfqkre | s1, s2 |
| h100-train-5 | e3fkcad376ki1q | s3, s4 |
| h100-train-6 | gj0nyp3g8gjk05 | s5, s6 |
| h100-train-7 | sspiiuxdyjbgmm | s7, s8 |
| h100-train-8 | owfszqz3sqenn5 | s9, s10 |

---

## Settings

| # | Name | Loss type | Supervised | fsx | vlo | TC (train) | Eval mode |
|---|------|-----------|------------|-----|-----|------------|-----------|
| 1 | SFT-lo | NLL only (pref=0) | labeled-only | — | — | — | self-tc, neg-tc |
| 2 | RankAlign | pref | semi | — | — | — | self-tc, neg-tc |
| 3 | New+fsx | comb (NLL+pref) | semi | ✓ | ✓ | — | self-tc, neg-tc |
| 4 | New+fsx+tc | comb | semi | ✓ | ✓ | self | self-tc |
| 5 | RankAlign+fsx+tc | pref | semi | ✓ | — | self | self-tc |
| 6 | RankAlign+tc | pref | semi | — | — | self | self-tc |
| 7 | New+fsx+negtc | comb | semi | ✓ | ✓ | neg | neg-tc |
| 8 | RankAlign+fsx+negtc | pref | semi | ✓ | — | neg | neg-tc |
| 9 | RankAlign+negtc | pref | semi | — | — | neg | neg-tc |
| 10 | RankAlign+fsx | pref | semi | ✓ | — | — | self-tc, neg-tc |
| 11 | New+tc | comb | semi | — | ✓ | self | self-tc |
| 12 | New+negtc | comb | semi | — | ✓ | neg | neg-tc |

*fsx = force-same-x, vlo = validator-log-odds, comb = NLL-validator + NLL-generator + preference loss*  
*Settings 11 and 12 added for future runs (not in the May-2026 gemma-4 run).*

---

## Adapter Suffixes (epoch 2)

Full adapter dir = base pattern above + suffix below.

| # | Adapter suffix |
|---|----------------|
| 1 | `-all--d2g--random--alpha1.0--full-completion--pref0.0--nllv1.0--nllg1.0--labelonly0.1` |
| 2 | `-all--d2g--random--alpha1.0--full-completion--semi0.1` |
| 3 | `-all--d2g--random--alpha1.0--full-completion--nllv1.0--nllg1.0--force-same-x--vallogodds--semi0.1` |
| 4 | `-all--d2g--random--alpha1.0--tc-self--full-completion--nllv1.0--nllg1.0--force-same-x--vallogodds--semi0.1` |
| 5 | `-all--d2g--random--alpha1.0--tc-self--full-completion--force-same-x--semi0.1` |
| 6 | `-all--d2g--random--alpha1.0--tc-self--full-completion--semi0.1` |
| 7 | `-all--d2g--random--alpha1.0--tc-neg--full-completion--nllv1.0--nllg1.0--force-same-x--vallogodds--semi0.1` |
| 8 | `-all--d2g--random--alpha1.0--tc-neg--full-completion--force-same-x--semi0.1` |
| 9 | `-all--d2g--random--alpha1.0--tc-neg--full-completion--semi0.1` |
| 10 | `-all--d2g--random--alpha1.0--full-completion--force-same-x--semi0.1` |
| 11 | `-all--d2g--random--alpha1.0--tc-self--full-completion--nllv1.0--nllg1.0--vallogodds--semi0.1` |
| 12 | `-all--d2g--random--alpha1.0--tc-neg--full-completion--nllv1.0--nllg1.0--vallogodds--semi0.1` |

---

## Score file patterns (on mll)

Location: `/datastor1/jdr/gv-gap/rankalign/outputs_gemma4_from_pod/`

| # | Score filename glob |
|---|---------------------|
| 1 | `scores_basetyp-v6-*labelonly*correct-multi*.csv` |
| 2 (self) | `scores_basetyp-v6-*semi0.1_humaneval-v2.1correct-multi*.csv` (no tc, no fsx) |
| 2 (neg) | `scores_basetypneg-v6-*semi0.1_humaneval-v2.1correct-multi*.csv` |
| 3 (self) | `scores_basetyp-v6-*force-same-x*vallogodds*correct-multi*.csv` (no tc) |
| 4 (self) | `scores_basetyp-v6-*tc-self*force-same-x*vallogodds*correct-multi*.csv` |
| 5 (self) | `scores_basetyp-v6-*tc-self*force-same-x*semi0.1*correct-multi*.csv` (no vallogodds) |
| 6 (self) | `scores_basetyp-v6-*tc-self*semi0.1*correct-multi*.csv` (no fsx) |
| 7 (neg) | `scores_basetypneg-v6-*tc-neg*force-same-x*vallogodds*correct-multi*.csv` |
| 8 (neg) | `scores_basetypneg-v6-*tc-neg*force-same-x*semi0.1*correct-multi*.csv` (no vallogodds) |
| 9 (neg) | `scores_basetypneg-v6-*tc-neg*semi0.1*correct-multi*.csv` (no fsx) |
| 10 (self) | `scores_basetyp-v6-*force-same-x*semi0.1*correct-multi*.csv` (no tc, no vallogodds) |
| 10 (neg) | `scores_basetypneg-v6-*force-same-x*semi0.1*correct-multi*.csv` |
