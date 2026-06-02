# Training-run JSON logs -> setting map (TASK 1c)

Each v7 training run wrote a provenance JSON to `/datastor2/jdr/rankalign/models2/training_run_logs/<timestamp>_<model>_<task>.json`. The **setting is not in the filename** — it's classified here from the JSON's recorded `flags` (force_same_x, validator_log_odds, self/neg_typicality, semi/labeled_only, consistency_ft). 85 logs total.

Each JSON records: delta config (`delta_arg`, `delta_bins`), shape-budget weights, label partition, sampled-pair counts, score stats, seed, and (s13) the `consistency_ft` filter block. To read one: `jq . <file>` on mll.

> Multiple timestamps under one (model,task,setting) = re-runs / delta-sweep points. > These logs are gemma-2 / qwen runs (the 2026-05-24+ overnight batch). gemma-4 cu runs > logged separately — see `../provenance-cu-s2-rankalign/models_g4it/training_run_logs/`.


## gemma-2-2b

| Task | Setting | # logs | Training JSON log file(s) |
|---|---|---|---|
| ifeval | s1 (SFT-lo) | 1 | `20260525-112349_google--gemma-2-2b_ifeval-concat.json` |
| ifeval | s2 (RankAlign) | 1 | `20260525-112401_google--gemma-2-2b_ifeval-concat.json` |
| ifeval | s13 (SFT+cft) | 3 | `20260525-020112_google--gemma-2-2b_ifeval-concat.json`<br>`20260525-020941_google--gemma-2-2b_ifeval-concat.json`<br>`20260525-021651_google--gemma-2-2b_ifeval-concat.json` |
| membership | s1 (SFT-lo) | 1 | `20260525-111629_google--gemma-2-2b_membership-sans-rosch-v0.json` |
| membership | s2 (RankAlign) | 1 | `20260525-111627_google--gemma-2-2b_membership-sans-rosch-v0.json` |
| membership | s3 (New+fsx) | 1 | `20260525-111710_google--gemma-2-2b_membership-sans-rosch-v0.json` |
| membership | s4 (New+fsx+tc) | 1 | `20260525-111832_google--gemma-2-2b_membership-sans-rosch-v0.json` |
| membership | s5 (RankAlign+fsx+tc) | 1 | `20260525-111747_google--gemma-2-2b_membership-sans-rosch-v0.json` |
| membership | s6 (RankAlign+tc) | 1 | `20260525-111748_google--gemma-2-2b_membership-sans-rosch-v0.json` |
| membership | s7 (New+fsx+negtc) | 2 | `20260525-111827_google--gemma-2-2b_membership-sans-rosch-v0.json`<br>`20260525-163131_google--gemma-2-2b_membership-sans-rosch-v0.json` |
| membership | s13 (SFT+cft) | 1 | `20260524-230732_google--gemma-2-2b_membership-sans-rosch-v0.json` |
| persona | s1 (SFT-lo) | 2 | `20260525-111604_google--gemma-2-2b_persona-v1.json`<br>`20260525-181347_google--gemma-2-2b_persona-v1.json` |
| persona | s2 (RankAlign) | 2 | `20260525-111607_google--gemma-2-2b_persona-v1.json`<br>`20260525-160725_google--gemma-2-2b_persona-v1.json` |
| persona | s3 (New+fsx) | 1 | `20260525-111939_google--gemma-2-2b_persona-v1.json` |
| persona | s4 (New+fsx+tc) | 1 | `20260525-112038_google--gemma-2-2b_persona-v1.json` |
| persona | s5 (RankAlign+fsx+tc) | 2 | `20260525-111708_google--gemma-2-2b_persona-v1.json`<br>`20260525-160910_google--gemma-2-2b_persona-v1.json` |
| persona | s6 (RankAlign+tc) | 1 | `20260525-112107_google--gemma-2-2b_persona-v1.json` |
| persona | s7 (New+fsx+negtc) | 1 | `20260525-112440_google--gemma-2-2b_persona-v1.json` |
| persona | s13 (SFT+cft) | 1 | `20260525-000541_google--gemma-2-2b_persona-v1.json` |

## gemma-2-2b-it

| Task | Setting | # logs | Training JSON log file(s) |
|---|---|---|---|
| ifeval | s1 (SFT-lo) | 1 | `20260525-081421_google--gemma-2-2b-it_ifeval-concat.json` |
| ifeval | s2 (RankAlign) | 3 | `20260525-093903_google--gemma-2-2b-it_ifeval-concat.json`<br>`20260525-110804_google--gemma-2-2b-it_ifeval-concat.json`<br>`20260525-111238_google--gemma-2-2b-it_ifeval-concat.json` |
| ifeval | s3 (New+fsx) | 1 | `20260525-113756_google--gemma-2-2b-it_ifeval-concat.json` |
| ifeval | s4 (New+fsx+tc) | 1 | `20260525-114106_google--gemma-2-2b-it_ifeval-concat.json` |
| ifeval | s6 (RankAlign+tc) | 1 | `20260525-112537_google--gemma-2-2b-it_ifeval-concat.json` |
| ifeval | s7 (New+fsx+negtc) | 1 | `20260525-114105_google--gemma-2-2b-it_ifeval-concat.json` |
| ifeval | s13 (SFT+cft) | 3 | `20260525-020107_google--gemma-2-2b-it_ifeval-concat.json`<br>`20260525-020941_google--gemma-2-2b-it_ifeval-concat.json`<br>`20260525-021654_google--gemma-2-2b-it_ifeval-concat.json` |
| membership | s1 (SFT-lo) | 1 | `20260524-134650_google--gemma-2-2b-it_membership-sans-rosch-v0.json` |
| membership | s2 (RankAlign) | 1 | `20260524-131620_google--gemma-2-2b-it_membership-sans-rosch-v0.json` |
| membership | s3 (New+fsx) | 1 | `20260524-134745_google--gemma-2-2b-it_membership-sans-rosch-v0.json` |
| membership | s4 (New+fsx+tc) | 1 | `20260524-130517_google--gemma-2-2b-it_membership-sans-rosch-v0.json` |
| membership | s5 (RankAlign+fsx+tc) | 1 | `20260524-134816_google--gemma-2-2b-it_membership-sans-rosch-v0.json` |
| membership | s6 (RankAlign+tc) | 1 | `20260524-134819_google--gemma-2-2b-it_membership-sans-rosch-v0.json` |
| membership | s7 (New+fsx+negtc) | 1 | `20260524-131456_google--gemma-2-2b-it_membership-sans-rosch-v0.json` |
| membership | s11 (New+tc) | 1 | `20260524-143511_google--gemma-2-2b-it_membership-sans-rosch-v0.json` |
| membership | s12 (New+negtc) | 1 | `20260524-153231_google--gemma-2-2b-it_membership-sans-rosch-v0.json` |
| membership | s13 (SFT+cft) | 1 | `20260524-230711_google--gemma-2-2b-it_membership-sans-rosch-v0.json` |
| persona | s1 (SFT-lo) | 1 | `20260524-134630_google--gemma-2-2b-it_persona-v1.json` |
| persona | s2 (RankAlign) | 1 | `20260524-131133_google--gemma-2-2b-it_persona-v1.json` |
| persona | s3 (New+fsx) | 1 | `20260524-131943_google--gemma-2-2b-it_persona-v1.json` |
| persona | s4 (New+fsx+tc) | 2 | `20260524-044641_google--gemma-2-2b-it_persona-v1.json`<br>`20260524-114706_google--gemma-2-2b-it_persona-v1.json` |
| persona | s5 (RankAlign+fsx+tc) | 1 | `20260524-134726_google--gemma-2-2b-it_persona-v1.json` |
| persona | s6 (RankAlign+tc) | 1 | `20260524-134732_google--gemma-2-2b-it_persona-v1.json` |
| persona | s7 (New+fsx+negtc) | 1 | `20260524-130657_google--gemma-2-2b-it_persona-v1.json` |
| persona | s11 (New+tc) | 1 | `20260524-135114_google--gemma-2-2b-it_persona-v1.json` |
| persona | s12 (New+negtc) | 1 | `20260524-143758_google--gemma-2-2b-it_persona-v1.json` |
| persona | s13 (SFT+cft) | 1 | `20260524-230741_google--gemma-2-2b-it_persona-v1.json` |

## gemma-2-9b-it

| Task | Setting | # logs | Training JSON log file(s) |
|---|---|---|---|
| ifeval | s13 (SFT+cft) | 3 | `20260525-021105_google--gemma-2-9b-it_ifeval-concat.json`<br>`20260525-031614_google--gemma-2-9b-it_ifeval-concat.json`<br>`20260525-111248_google--gemma-2-9b-it_ifeval-concat.json` |
| membership | s1 (SFT-lo) | 1 | `20260524-125453_google--gemma-2-9b-it_membership-sans-rosch-v0.json` |
| membership | s2 (RankAlign) | 1 | `20260524-125325_google--gemma-2-9b-it_membership-sans-rosch-v0.json` |
| membership | s3 (New+fsx) | 1 | `20260524-125542_google--gemma-2-9b-it_membership-sans-rosch-v0.json` |
| membership | s4 (New+fsx+tc) | 1 | `20260524-124646_google--gemma-2-9b-it_membership-sans-rosch-v0.json` |
| membership | s5 (RankAlign+fsx+tc) | 1 | `20260524-125846_google--gemma-2-9b-it_membership-sans-rosch-v0.json` |
| membership | s6 (RankAlign+tc) | 1 | `20260524-125834_google--gemma-2-9b-it_membership-sans-rosch-v0.json` |
| membership | s7 (New+fsx+negtc) | 1 | `20260524-130050_google--gemma-2-9b-it_membership-sans-rosch-v0.json` |
| membership | s11 (New+tc) | 1 | `20260524-131023_google--gemma-2-9b-it_membership-sans-rosch-v0.json` |
| membership | s12 (New+negtc) | 1 | `20260524-131042_google--gemma-2-9b-it_membership-sans-rosch-v0.json` |
| membership | s13 (SFT+cft) | 1 | `20260524-231046_google--gemma-2-9b-it_membership-sans-rosch-v0.json` |
| persona | s1 (SFT-lo) | 2 | `20260524-125348_google--gemma-2-9b-it_persona-v1.json`<br>`20260525-014417_google--gemma-2-9b-it_persona-v1.json` |
| persona | s2 (RankAlign) | 1 | `20260524-125313_google--gemma-2-9b-it_persona-v1.json` |
| persona | s3 (New+fsx) | 4 | `20260524-125527_google--gemma-2-9b-it_persona-v1.json`<br>`20260525-045330_google--gemma-2-9b-it_persona-v1.json`<br>`20260525-051431_google--gemma-2-9b-it_persona-v1.json`<br>`20260525-055421_google--gemma-2-9b-it_persona-v1.json` |
| persona | s4 (New+fsx+tc) | 1 | `20260524-124722_google--gemma-2-9b-it_persona-v1.json` |
| persona | s5 (RankAlign+fsx+tc) | 1 | `20260524-125629_google--gemma-2-9b-it_persona-v1.json` |
| persona | s6 (RankAlign+tc) | 1 | `20260524-125642_google--gemma-2-9b-it_persona-v1.json` |
| persona | s7 (New+fsx+negtc) | 1 | `20260524-130039_google--gemma-2-9b-it_persona-v1.json` |
| persona | s11 (New+tc) | 1 | `20260524-131045_google--gemma-2-9b-it_persona-v1.json` |
| persona | s12 (New+negtc) | 1 | `20260524-131046_google--gemma-2-9b-it_persona-v1.json` |
| persona | s13 (SFT+cft) | 1 | `20260525-001022_google--gemma-2-9b-it_persona-v1.json` |

## gemma-4-31B-it

| Task | Setting | # logs | Training JSON log file(s) |
|---|---|---|---|
| humaneval-cu | s13 (SFT+cft) | 1 | `20260524-193331_google--gemma-4-31B-it_humaneval-v2.1correct-upper.json` |
