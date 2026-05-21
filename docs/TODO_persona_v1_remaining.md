# Persona-v1 — Remaining work (created late 2026-05-20)

When you pick this back up, this is what's outstanding. See chat 2026-05-20
/ -21 for the design discussion behind every item.

## State of things as of going to bed

### Trained (epoch2 on disk, evaled with matched-TC + base-typcorr)
- gemma-2-9b-it × 9 variants (#1–#9) — disc-shots zero, eval matched.
- gemma-2-2b-it × 9 variants (#1–#9) — disc-shots zero, eval matched.

### Trained tonight (in flight as of bedtime)
- **gemma-2-2b × 9 variants (#1–#9), DISC_SHOTS=auto → few**.
  Jobs 40992–41000 (`overnight/persona_v1_train_jobids_gemma-2-2b.txt`).
  Walltime 10h, started ~01:35. Should finish before morning.
  Eval has NOT been launched — needs the 2b trained-eval launcher (which
  doesn't exist yet, see TODO 4 below) and disc-shots=few at eval time.

### NOT trained yet (the gap):
- **#10 RankAlign+fsx**, **#11 New+selfTC (no fsx)**, **#12 New+negTC (no fsx)**
  for ALL 3 bases (9b-it / 2b-it / 2b). 9 more training jobs + eval.

---

## TODO list (do in order)

### 1. Audit Issue #1 status (is the mixed-pair NLL gate bug fixed yet?)
Read `scripts/ranking_loss_ref.py` around the `pair_is_labeled` gate (line
2161 area, last seen). Per `docs/IMPORTANT-RESEARCH-PLAN.md` §7 Issue #1
the bug bites **#11/#12** specifically (`comb + semi + no-fsx` mixes
labeled/unlabeled items in pairs and silently drops NLL signal).

- If fixed: run the §7 verification recipe (50-step rosch smoke test, check
  `train/nll_validator_loss` and `train/nll_generator_loss` are nonzero on
  mixed pairs) before relying on it.
- If not fixed: write the per-end masking fix per §7's "Proposed fix" code
  block. Land + verify smoke test.

#10 is unaffected (pref-only → NLL terms multiplied by 0).

### 2. Add #10/#11/#12 to the v1 train launcher
Three new `submit` lines in `scripts/run_train_persona_v1.sh`:

```bash
submit "10.RankAlign+fsx"        pref-only semi 0.1 $COMMON
submit "11.New+selfTC"           comb semi 0.1 $COMMON --self-typcorr --log-odds --no-force-same-x
submit "12.New+negTC"            comb semi 0.1 $COMMON --neg-typcorr  --log-odds --no-force-same-x
```

Verify `--no-force-same-x` works with `comb` (should — the flag controls
pair construction, orthogonal to loss).

### 3. Launch #10/#11/#12 for all 3 bases (after Issue #1 is resolved)
That's 9 jobs:
- gemma-2-9b-it × {#10, #11, #12}, disc-shots zero
- gemma-2-2b-it × {#10, #11, #12}, disc-shots zero
- gemma-2-2b    × {#10, #11, #12}, disc-shots auto (= few)

### 4. Add #10/#11/#12 to the trained-eval launcher
Three new entries in `SUFFIXES` and `TC_POLICY` in
`scripts/run_eval_persona_v1_trained.sh`:

```bash
SUFFIXES["10.RankAlign+fsx"]="--full-completion--force-same-x--semi0.1"
SUFFIXES["11.New+selfTC"]="--tc-self--full-completion--nllv1.0--nllg1.0--vallogodds--semi0.1"
SUFFIXES["12.New+negTC"]="--tc-neg--full-completion--nllv1.0--nllg1.0--vallogodds--semi0.1"
TC_POLICY["10.RankAlign+fsx"]="both"   # non-TC-trained → eval both flavors
TC_POLICY["11.New+selfTC"]="self"
TC_POLICY["12.New+negTC"]="neg"
```

(Verify suffixes match what `ranking_loss_ref.py` actually saves — these
are my best read; cross-check after the trainings finish by `ls models/`.)

### 4b. Run summarize_scores.py for the v1 evals already on disk
We now have baseline + trained evals for #1–#9 on **2b-it** and **9b-it**
(disc-shots zero throughout) plus baseline-only on **2b** in two flavors
(disc-zero in `outputs_persona-v1_gemma_2b_zero-shot-disc/`, disc-few in
`outputs/`). Aggregate mean + std err across personas with
`scripts/summarize_scores.py` (or the persona-v1-aware
`scripts/persona_v0_make_tables.py` adapted) and write per-(base, eval_TC,
metric) tables in CSV + markdown. Chat 2026-05-20 has the layout you
asked for (raw and tc as columns; one table per (model, metric, neg/self
TC); base-typcorr noted in the tables).

### 5. Run trained-eval for the gemma-2-2b runs in flight tonight
Need a new launcher path (or env var) that uses **disc-shots-few at eval**
for the 2b base, since training was disc-few. Mirror the disc-shots-auto
plumbing we added to the train launcher.

### 6. (Eventually) decide whether to redo 9b-it / 2b-it with disc-shots-few
Per chat: "Later we can decide if we also want to redo the 2b-it and 9b-it
results with zero shot also." (You meant few-shot here I think — the
existing 9b-it/2b-it are already disc-shots zero.) Defer to after
TC/headline question is answered on the existing data.

---

## Misc reminders surfaced during the night

- **Issue #1 fix-needed list (from research-plan §7)** as of 2026-05-20:
  only #11 and #12 affected. Already noted above.
- **Disc-shots history table** in `docs/datasets/persona_v1_notes.md`:
  update whenever new runs land that change disc-shots / base / phase.
- **Filename-collision patch** for `build_csv_filename()` (deferred):
  add `_disc{zero|few}` token so future parallel disc-shots evals on the
  same model+task+day don't overwrite each other. Not urgent — current
  runs use distinct dates or distinct prefixes.
