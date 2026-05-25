# Eval coverage matrix — per cell, 4 prefix-families

For each (dataset, model, setting) cell, counts of `scores_*.csv` files found per CSV-prefix family, vs expected N eval-tasks for that dataset. `n/a` = the TC-variant doesn't apply to a model trained with the opposite or no TC objective.

Trained-TC rule (controls n/a):

- s1, s2, s3, s13   (no TC trained)   → eval both `self` and `neg`
- s4, s5, s6, s11   (self-TC trained) → eval `self` only (Neg cols n/a)
- s7, s12           (neg-TC trained)  → eval `neg` only  (PMI cols n/a)

Annotations: `[42140]` = pending/running slurm eval job covering this cell.
`done(eN)` / `running(JID)` for the train column.

## membership  (N = 10 expected eval tasks per cell)

| model | s# | T | PMI base | PMI self | Neg base | Neg self |
|---|---|---|---|---|---|---|
| gemma-2-2b-it | s1 | done(e2) | 10/10 | 10/10 | 10/10 | 10/10 |
| gemma-2-2b-it | s2 | done(e2) | 10/10 | 10/10 | 10/10 | 10/10 |
| gemma-2-2b-it | s3 | done(e2) | 10/10 | 10/10 | 10/10 | 10/10 |
| gemma-2-2b-it | s4 | done(e2) | 10/10 | 10/10 | n/a | n/a |
| gemma-2-2b-it | s5 | done(e2) | 10/10 | 10/10 | n/a | n/a |
| gemma-2-2b-it | s6 | done(e2) | 10/10 | 10/10 | n/a | n/a |
| gemma-2-2b-it | s7 | done(e2) | n/a | n/a | 10/10 | 10/10 |
| gemma-2-2b-it | s11 | done(e2) | 10/10 | 10/10 | n/a | n/a |
| gemma-2-2b-it | s12 | done(e2) | n/a | n/a | 10/10 | 10/10 |
| gemma-2-2b-it | s13 | done(e0) | 0/10 | 0/10 | 0/10 | 0/10 |
| gemma-2-9b-it | s1 | done(e1) | 10/10 | 10/10 | 10/10 | 10/10 |
| gemma-2-9b-it | s2 | done(e2) | 10/10 | 10/10 | 10/10 | 10/10 |
| gemma-2-9b-it | s3 | done(e2) | 10/10 | 10/10 | 10/10 | 10/10 |
| gemma-2-9b-it | s4 | done(e2) | 10/10 | 10/10 | n/a | n/a |
| gemma-2-9b-it | s5 | done(e2) | 10/10 | 10/10 | n/a | n/a |
| gemma-2-9b-it | s6 | done(e2) | 10/10 | 10/10 | n/a | n/a |
| gemma-2-9b-it | s7 | done(e2) | n/a | n/a | 10/10 | 10/10 |
| gemma-2-9b-it | s11 | done(e2) | 10/10 | 10/10 | n/a | n/a |
| gemma-2-9b-it | s12 | done(e2) | n/a | n/a | 10/10 | 10/10 |
| gemma-2-9b-it | s13 | running(42262) | 0/10 | 0/10 | 0/10 | 0/10 |

## persona  (N = 6 expected eval tasks per cell)

| model | s# | T | PMI base | PMI self | Neg base | Neg self |
|---|---|---|---|---|---|---|
| gemma-2-2b-it | s1 | done(e2) | 6/6 | 6/6 | 6/6 | 6/6 |
| gemma-2-2b-it | s2 | done(e2) | 6/6 | 6/6 | 6/6 | 6/6 |
| gemma-2-2b-it | s3 | done(e2) | 6/6 | 6/6 | 6/6 | 6/6 |
| gemma-2-2b-it | s4 | done(e2) | 6/6 | 6/6 | n/a | n/a |
| gemma-2-2b-it | s5 | done(e2) | 6/6 | 6/6 | n/a | n/a |
| gemma-2-2b-it | s6 | done(e2) | 6/6 | 6/6 | n/a | n/a |
| gemma-2-2b-it | s7 | done(e2) | n/a | n/a | 6/6 | 6/6 |
| gemma-2-2b-it | s11 | done(e2) | 6/6 | 6/6 | n/a | n/a |
| gemma-2-2b-it | s12 | done(e2) | n/a | n/a | 6/6 | 6/6 |
| gemma-2-2b-it | s13 | running(42265) | 0/6 | 0/6 | 0/6 | 0/6 |
| gemma-2-9b-it | s1 | done(e0) | 6/6 | 6/6 | 6/6 | 6/6 |
| gemma-2-9b-it | s2 | done(e2) | 6/6 | 6/6 | 6/6 | 6/6 |
| gemma-2-9b-it | s3 | done(e0) | 6/6 | 6/6 | 6/6 | 6/6 |
| gemma-2-9b-it | s4 | done(e1) | 6/6 | 6/6 | n/a | n/a |
| gemma-2-9b-it | s5 | done(e2) | 6/6 | 6/6 | n/a | n/a |
| gemma-2-9b-it | s6 | done(e2) | 6/6 | 6/6 | n/a | n/a |
| gemma-2-9b-it | s7 | done(e2) | n/a | n/a | 6/6 | 6/6 |
| gemma-2-9b-it | s11 | done(e1) | 6/6 | 6/6 | n/a | n/a |
| gemma-2-9b-it | s12 | done(e1) | n/a | n/a | 6/6 | 6/6 |
| gemma-2-9b-it | s13 | running(42286) | 0/6 | 0/6 | 0/6 | 0/6 |

## ifeval  (N = 21 expected eval tasks per cell)

| model | s# | T | PMI base | PMI self | Neg base | Neg self |
|---|---|---|---|---|---|---|
| gemma-2-2b-it | s1 | — | 0/21 | 0/21 | 0/21 | 0/21 |
| gemma-2-2b-it | s2 | — | 0/21 | 0/21 | 0/21 | 0/21 |
| gemma-2-2b-it | s3 | — | 0/21 | 0/21 | 0/21 | 0/21 |
| gemma-2-2b-it | s4 | — | 0/21 | 0/21 | n/a | n/a |
| gemma-2-2b-it | s5 | — | 0/21 | 0/21 | n/a | n/a |
| gemma-2-2b-it | s6 | — | 0/21 | 0/21 | n/a | n/a |
| gemma-2-2b-it | s7 | — | n/a | n/a | 0/21 | 0/21 |
| gemma-2-2b-it | s11 | — | 0/21 | 0/21 | n/a | n/a |
| gemma-2-2b-it | s12 | — | n/a | n/a | 0/21 | 0/21 |
| gemma-2-2b-it | s13 | — | 0/21 | 0/21 | 0/21 | 0/21 |
| gemma-2-9b-it | s1 | — | 0/21 | 0/21 | 0/21 | 0/21 |
| gemma-2-9b-it | s2 | — | 0/21 | 0/21 | 0/21 | 0/21 |
| gemma-2-9b-it | s3 | — | 0/21 | 0/21 | 0/21 | 0/21 |
| gemma-2-9b-it | s4 | — | 0/21 | 0/21 | n/a | n/a |
| gemma-2-9b-it | s5 | — | 0/21 | 0/21 | n/a | n/a |
| gemma-2-9b-it | s6 | — | 0/21 | 0/21 | n/a | n/a |
| gemma-2-9b-it | s7 | — | n/a | n/a | 0/21 | 0/21 |
| gemma-2-9b-it | s11 | — | 0/21 | 0/21 | n/a | n/a |
| gemma-2-9b-it | s12 | — | n/a | n/a | 0/21 | 0/21 |
| gemma-2-9b-it | s13 | — | 0/21 | 0/21 | 0/21 | 0/21 |

## humaneval  (N = 82 expected eval tasks per cell)

| model | s# | T | PMI base | PMI self | Neg base | Neg self |
|---|---|---|---|---|---|---|
| gemma-4-31B-it | s13 | running(42114) | 0/82 | 0/82 | 0/82 | 0/82 |

## Gaps requiring submission

Total: **20 (cell × prefix) gaps with no in-flight job**.

| dataset | model | setting | column | csv prefix | NO_BASE? |
|---|---|---|---|---|---|
| membership | gemma-2-2b-it | s13 | PMI base | `basetyp-` | 0 |
| membership | gemma-2-2b-it | s13 | PMI self | `self-` | 1 |
| membership | gemma-2-2b-it | s13 | Neg base | `basetypneg-` | 0 |
| membership | gemma-2-2b-it | s13 | Neg self | `neg-` | 1 |
| membership | gemma-2-9b-it | s13 | PMI base | `basetyp-` | 0 |
| membership | gemma-2-9b-it | s13 | PMI self | `self-` | 1 |
| membership | gemma-2-9b-it | s13 | Neg base | `basetypneg-` | 0 |
| membership | gemma-2-9b-it | s13 | Neg self | `neg-` | 1 |
| persona | gemma-2-2b-it | s13 | PMI base | `basetyp-` | 0 |
| persona | gemma-2-2b-it | s13 | PMI self | `self-` | 1 |
| persona | gemma-2-2b-it | s13 | Neg base | `basetypneg-` | 0 |
| persona | gemma-2-2b-it | s13 | Neg self | `neg-` | 1 |
| persona | gemma-2-9b-it | s13 | PMI base | `basetyp-` | 0 |
| persona | gemma-2-9b-it | s13 | PMI self | `self-` | 1 |
| persona | gemma-2-9b-it | s13 | Neg base | `basetypneg-` | 0 |
| persona | gemma-2-9b-it | s13 | Neg self | `neg-` | 1 |
| humaneval | gemma-4-31B-it | s13 | PMI base | `basetyp-` | 0 |
| humaneval | gemma-4-31B-it | s13 | PMI self | `self-` | 1 |
| humaneval | gemma-4-31B-it | s13 | Neg base | `basetypneg-` | 0 |
| humaneval | gemma-4-31B-it | s13 | Neg self | `neg-` | 1 |
