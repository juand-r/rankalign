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
| gemma-2-2b-it | s3 | done(e2) | 0/10 | 10/10 | 10/10 | 10/10 |
| gemma-2-2b-it | s4 | done(e2) | 0/10 [42189] | 10/10 | n/a | n/a |
| gemma-2-2b-it | s5 | done(e2) | 9/10 [42190] | 10/10 | n/a | n/a |
| gemma-2-2b-it | s6 | done(e2) | 10/10 | 10/10 | n/a | n/a |
| gemma-2-2b-it | s7 | done(e2) | n/a | n/a | 0/10 [42191] | 6/10 [42186] |
| gemma-2-2b-it | s11 | done(e2) | 0/10 [42192] | 6/10 [42187] | n/a | n/a |
| gemma-2-2b-it | s12 | done(e2) | n/a | n/a | 10/10 [42193] | 6/10 [42188] |
| gemma-2-2b-it | s13 | — | 0/10 | 0/10 | 0/10 | 0/10 |
| gemma-2-9b-it | s1 | done(e1) | 10/10 | 0/10 | 10/10 | 0/10 |
| gemma-2-9b-it | s2 | done(e2) | 10/10 | 10/10 | 10/10 | 10/10 |
| gemma-2-9b-it | s3 | done(e2) | 10/10 | 10/10 | 10/10 | 10/10 |
| gemma-2-9b-it | s4 | done(e2) | 1/10 [42189] | 10/10 | n/a | n/a |
| gemma-2-9b-it | s5 | done(e2) | 1/10 [42190] | 10/10 | n/a | n/a |
| gemma-2-9b-it | s6 | done(e2) | 10/10 | 10/10 | n/a | n/a |
| gemma-2-9b-it | s7 | done(e2) | n/a | n/a | 2/10 [42191] | 10/10 [42186] |
| gemma-2-9b-it | s11 | done(e2) | 2/10 [42192] | 10/10 [42187] | n/a | n/a |
| gemma-2-9b-it | s12 | done(e2) | n/a | n/a | 0/10 [42193] | 10/10 [42188] |
| gemma-2-9b-it | s13 | — | 0/10 | 0/10 | 0/10 | 0/10 |

## persona  (N = 6 expected eval tasks per cell)

| model | s# | T | PMI base | PMI self | Neg base | Neg self |
|---|---|---|---|---|---|---|
| gemma-2-2b-it | s1 | done(e2) | 6/6 [42020] | 0/6 | 6/6 [42127] | 0/6 |
| gemma-2-2b-it | s2 | done(e2) | 6/6 | 0/6 | 6/6 | 0/6 |
| gemma-2-2b-it | s3 | done(e2) | 0/6 [42010] | 0/6 | 6/6 [42128] | 0/6 |
| gemma-2-2b-it | s4 | done(e2) | 6/6 [41980] | 0/6 | n/a | n/a |
| gemma-2-2b-it | s5 | done(e2) | 6/6 | 0/6 | n/a | n/a |
| gemma-2-2b-it | s6 | done(e2) | 6/6 | 0/6 | n/a | n/a |
| gemma-2-2b-it | s7 | done(e2) | n/a | n/a | 0/6 [41990] | 0/6 |
| gemma-2-2b-it | s11 | done(e2) | 6/6 [42050] | 0/6 | n/a | n/a |
| gemma-2-2b-it | s12 | done(e2) | n/a | n/a | 6/6 [42060] | 0/6 |
| gemma-2-2b-it | s13 | — | 0/6 | 0/6 | 0/6 | 0/6 |
| gemma-2-9b-it | s1 | done(e0) | 3/6 [42020] | 0/6 | 3/6 [42127] | 0/6 |
| gemma-2-9b-it | s2 | done(e2) | 6/6 | 6/6 | 6/6 | 6/6 |
| gemma-2-9b-it | s3 | done(e0) | 2/6 [42010] | 0/6 | 3/6 [42128] | 0/6 |
| gemma-2-9b-it | s4 | done(e1) | 4/6 [41980] | 0/6 | n/a | n/a |
| gemma-2-9b-it | s5 | done(e2) | 6/6 | 0/6 | n/a | n/a |
| gemma-2-9b-it | s6 | done(e2) | 6/6 | 0/6 | n/a | n/a |
| gemma-2-9b-it | s7 | done(e2) | n/a | n/a | 4/6 [41990] | 0/6 |
| gemma-2-9b-it | s11 | done(e1) | 0/6 [42050] | 0/6 | n/a | n/a |
| gemma-2-9b-it | s12 | done(e1) | n/a | n/a | 0/6 [42060] | 0/6 |
| gemma-2-9b-it | s13 | — | 0/6 | 0/6 | 0/6 | 0/6 |

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
| gemma-4-31B-it | s13 | running(42114) | 0/82 [42115] | 0/82 | 0/82 [42116] | 0/82 |

## Gaps requiring submission

Total: **27 (cell × prefix) gaps with no in-flight job**.

| dataset | model | setting | column | csv prefix | NO_BASE? |
|---|---|---|---|---|---|
| membership | gemma-2-2b-it | s3 | PMI base | `basetyp-` | 0 |
| membership | gemma-2-9b-it | s1 | PMI self | `self-` | 1 |
| membership | gemma-2-9b-it | s1 | Neg self | `neg-` | 1 |
| persona | gemma-2-2b-it | s1 | PMI self | `self-` | 1 |
| persona | gemma-2-2b-it | s1 | Neg self | `neg-` | 1 |
| persona | gemma-2-2b-it | s2 | PMI self | `self-` | 1 |
| persona | gemma-2-2b-it | s2 | Neg self | `neg-` | 1 |
| persona | gemma-2-2b-it | s3 | PMI self | `self-` | 1 |
| persona | gemma-2-2b-it | s3 | Neg self | `neg-` | 1 |
| persona | gemma-2-2b-it | s4 | PMI self | `self-` | 1 |
| persona | gemma-2-2b-it | s5 | PMI self | `self-` | 1 |
| persona | gemma-2-2b-it | s6 | PMI self | `self-` | 1 |
| persona | gemma-2-2b-it | s7 | Neg self | `neg-` | 1 |
| persona | gemma-2-2b-it | s11 | PMI self | `self-` | 1 |
| persona | gemma-2-2b-it | s12 | Neg self | `neg-` | 1 |
| persona | gemma-2-9b-it | s1 | PMI self | `self-` | 1 |
| persona | gemma-2-9b-it | s1 | Neg self | `neg-` | 1 |
| persona | gemma-2-9b-it | s3 | PMI self | `self-` | 1 |
| persona | gemma-2-9b-it | s3 | Neg self | `neg-` | 1 |
| persona | gemma-2-9b-it | s4 | PMI self | `self-` | 1 |
| persona | gemma-2-9b-it | s5 | PMI self | `self-` | 1 |
| persona | gemma-2-9b-it | s6 | PMI self | `self-` | 1 |
| persona | gemma-2-9b-it | s7 | Neg self | `neg-` | 1 |
| persona | gemma-2-9b-it | s11 | PMI self | `self-` | 1 |
| persona | gemma-2-9b-it | s12 | Neg self | `neg-` | 1 |
| humaneval | gemma-4-31B-it | s13 | PMI self | `self-` | 1 |
| humaneval | gemma-4-31B-it | s13 | Neg self | `neg-` | 1 |
