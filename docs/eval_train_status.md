# Eval/Train Status Tracker

Last updated: 2026-04-30 11:17 PM

## Workstream 1: EOS train + eval (hypernym, outputs-eos-models/)

### Training (all use `--include-eos`, models in `models-eos/`)

| # | Setting | 9b-it | 2b-it | 2b |
|---|---------|-------|-------|----|
| 1 | RankAlign (no fsx) | DONE (34873) | DONE (34875) | DONE (34881) |
| 2 | SFT-LO | DONE (prev) | DONE (34876) | DONE (34882) |
| 3 | Comb vlo semi | DONE (prev) | DONE (34877) | DONE (34883) |
| 4 | Comb vlo + tc-self semi | DONE (prev) | DONE (34878) | DONE (34884) |
| 5 | Pref-only semi | DONE (prev) | DONE (34879) | DONE (34885) |
| 6 | Pref-only semi + tc-self | DONE (34874) | DONE (34880) | DONE (34886) |

### Eval (self-TC + EOS, scores in `outputs-eos-models/`)

| # | Setting | 9b-it | 2b-it | 2b |
|---|---------|-------|-------|----|
| 1 | RankAlign (no fsx) | DONE 18/18 | DONE 18/18 | DONE 18/18 |
| 2 | SFT-LO | DONE 18/18 | DONE 18/18 | DONE 18/18 |
| 3 | Comb vlo semi | DONE 18/18 | DONE 18/18 | DONE 18/18 |
| 4 | Comb vlo + tc-self semi | DONE 18/18 | DONE 18/18 | DONE 18/18 |
| 5 | Pref-only semi | DONE 18/18 | DONE 18/18 | DONE 18/18 |
| 6 | Pref-only semi + tc-self | DONE 18/18 | DONE 18/18 | DONE 18/18 |

**ALL EOS EVALS COMPLETE.**

---

## Workstream 2: basetyp evals (non-EOS hypernym, outputs/)

### basetyp- (self-TC via base model)

| # | Setting | 9b-it | 2b-it | 2b |
|---|---------|-------|-------|----|
| 2a | SFT-LO | DONE 18/18 | DONE 18/18 | DONE 18/18 |
| 3a | RankAlign | DONE 18/18 (34957 completed) | DONE 18/18 | DONE 18/18 |
| 4a.1 | Comb vlo semi | DONE 18/18 | DONE 18/18 | DONE 18/18 |
| 4a.2 | Comb vlo + tc-self semi | RUNNING (34973) | DONE 18/18 | DONE 18/18 |

### basetypneg- (neg-TC via base model)

| # | Setting | 9b-it | 2b-it | 2b |
|---|---------|-------|-------|----|
| 3b | RankAlign | DONE 18/18 | DONE 18/18 | DONE 18/18 |
| 4b.1 | Comb vlo + tc-neg semi | DONE 18/18 | BLOCKED (training tc-neg model, 34956 RUNNING ~1h in) | DONE 18/18 |

Commands for when blocked items unblock:

```bash
# 4a.2 eval (after cp finishes):
run 1 3 --cpu 4 --mem 32G scripts/run_eval_semi.sh "../models/v6-google--gemma-2-9b-it-delta0.15-epoch2--hypernym-concat-bananas-to-dogs-double-all--d2g--random--alpha1.0--tc-self--full-completion--nllv1.0--nllg1.0--force-same-x--vallogodds--semi0.1_merged" --self-typcorr --base-typcorr --base-model google/gemma-2-9b-it --log-odds -- hypernym-bananas hypernym-bazookas hypernym-cabinets hypernym-cars hypernym-chairs hypernym-crows hypernym-diapers hypernym-dogs hypernym-dolls hypernym-ducklings hypernym-elephants hypernym-guns hypernym-hammers hypernym-helmets hypernym-jackets hypernym-kayaks hypernym-kites hypernym-mirrors

# 4b.1 eval for 2b-it (after tc-neg training finishes):
run 1 2 --cpu 4 --mem 32G scripts/run_eval_semi.sh "../models/v6-google--gemma-2-2b-it-delta0.15-epoch2--hypernym-concat-bananas-to-dogs-double-all--d2g--random--alpha1.0--tc-neg--full-completion--nllv1.0--nllg1.0--force-same-x--vallogodds--semi0.1" --neg-typcorr --base-typcorr --base-model google/gemma-2-2b-it --log-odds -- hypernym-bananas hypernym-bazookas hypernym-cabinets hypernym-cars hypernym-chairs hypernym-crows hypernym-diapers hypernym-dogs hypernym-dolls hypernym-ducklings hypernym-elephants hypernym-guns hypernym-hammers hypernym-helmets hypernym-jackets hypernym-kayaks hypernym-kites hypernym-mirrors
```

---

## Workstream 3: neg finetuned evals for 2b-it (non-EOS, outputs/)

TODO — NOT LAUNCHED.

2b-it is missing neg finetuned evals almost entirely:
- **hypernym**: only RankAlign exists (and only with `_eos`). All other variants missing.
- **ambigqa**: nothing
- **plausibleqa**: nothing
- **ifeval**: nothing

For comparison, 9b-it has neg finetuned evals for all 4 tasks, 2b has 3 tasks (no ifeval).

---

## Notes

- Workstreams 1 and 2 are **hypernym only**.
- Workstream 3 spans **all tasks** (hypernym, ambigqa, plausibleqa, ifeval).

---

## Notes

- EOS models live in `models-eos/`, non-EOS in `models/`.
- EOS eval outputs go to `outputs-eos-models/`, everything else to `outputs/`.
- 9b-it models use LoRA and need `_merged` suffix.
- 2b-it and 2b are full finetune, no `_merged`.
