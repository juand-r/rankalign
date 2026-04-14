# IFEval tc-neg Training Jobs -- Status Check Needed

**Date:** 2026-04-13

## Problem

IFEval tc-neg training jobs for 9b-it were submitted with `run 1 24` (24h walltime),
but startup/preprocessing takes ~13.5h before training even begins.
With ~6h/epoch x 3 epochs on top, total runtime is ~31.5h -- exceeding the 24h limit.

## Jobs at risk

| Job   | Variant                        | Samples | Allocated |
|-------|--------------------------------|---------|-----------|
| 27884 | comb semi 0.1, log-odds        | 5110    | 24h       |
| 27885 | sft semi 0.1                   | 5110    | 24h       |
| 27886 | comb labelonly 0.1, log-odds   | 4908    | 24h       |
| 27887 | pref-only labelonly 0.1, log-odds | 4908 | 24h       |
| 27888 | pref-only labelonly 0.1        | 4908    | 24h       |

All use: `google/gemma-2-9b-it`, LoRA, `--neg-typcorr`, `ifeval-concat`, via `run_train_semi.sh`.

## TODO

1. Check if these jobs got killed at the 24h mark
2. See how far they got (epoch0? epoch1?)
3. Resubmit with longer walltime (`run 1 48`) if needed
4. Investigate whether the ~13.5h preprocessing can be cached to speed up reruns

## RankAlign-V2G baseline training (separate)

All 8 V2G baseline jobs completed successfully -- no action needed there.
