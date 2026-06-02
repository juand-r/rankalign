# Stray model dirs under `/datastor2/jdr/` (NOT under `…/rankalign/`)

**Checked 2026-06-02** after the user asked "did you also check `/datastor2/jdr/models{,2}`?"
The main inventory (TASK 1) covers `/datastor2/jdr/**rankalign**/models{,2}`. There are also
model dirs **one level up**, directly under `/datastor2/jdr/`. This documents them.

**Bottom line: none of these contain paper (v6/v7) models that the inventory is missing.**
They are legacy/smoke. The main TASK 1 inventory is complete for the paper.

| Path | Contents | Paper-relevant? |
|------|----------|-----------------|
| `/datastor2/jdr/models` | **882 `v5-*` dirs** — all gemma-2-2b on OLD tasks (hypernym, trivia-qa, swords, lambada, …); plus a handful of ancient `google--gemma-2-2b-delta2.5-epoch4..9` g2d hypernym runs. dir mtime **Feb 5 2026**. | ❌ No. **Zero** gemma-4 / humaneval / persona-v1 / membership-sans-rosch / ifeval-concat / v6 / v7. Pure pre-paper **v5** archive. |
| `/datastor2/jdr/models2` | **1 stray model** (2 dirs): `v7-Qwen--Qwen3.5-9B-delta0.512…-epoch0--membership-sans-rosch-v0-…-nllv1.0--nllg1.0--vallogodds--semi0.1--fix1` + its `_merged`, plus `auto_delta_log.csv`. mtime **May 23**. | ⚠ Early/smoke. epoch0 only; a comb+vlo no-fsx no-tc Qwen membership run (not one of the canonical numbered settings — `comb-notc-nofsx?`). The real Qwen membership models the paper uses are in `…/rankalign/models2` + HF (see TASK 1). |
| `/datastor2/jdr/models_g4it_smoke` | 1 gemma-4 **smoke test**: `v7-google--gemma-4-31B-it-delta2.886…-epoch0--persona-v1-…-tc-self-…-fix1` + delta log. mtime **May 24**. | ❌ Smoke test (epoch0, persona — gemma-4 was never a persona paper model). Ignore. |

Raw listings: `_raw/stray_datastor2jdr_models_v5_listing.txt` (882 v5),
`_raw/stray_models2_and_smoke.txt`.

## Why models ended up here (the "Claude got confused" question)

Path drift over time, not a correctness bug in the paper runs:
- **Feb 2026 (v5 era):** the model store was `/datastor2/jdr/models` (before the work was
  reorganized into the `…/rankalign/` repo checkout). That dir is now a frozen v5 archive.
- **May 23–24:** a couple of early/smoke runs (the stray Qwen, the gemma-4 smoke) defaulted
  their `--models-dir` to `/datastor2/jdr/models2` and `…/models_g4it_smoke` before the
  canonical `_overnight_launch.sh` default (`MODELS_DIR=/datastor2/jdr/rankalign/models2`)
  was used for the real overnight batch (May 24+).
- **The real paper models** all landed in **`/datastor2/jdr/rankalign/models2`** (v7, 271
  dirs) and **`…/rankalign/models`** (v6, 201 dirs) — which is exactly what TASK 1 inventoried.

## Action (none tonight — additive doc only)
- These stray dirs are **left untouched** (additive-only rule; and `/datastor2/jdr/models` v5
  archive may still be wanted). If you later want to tidy up, the stray `models2` Qwen-e0 and
  `models_g4it_smoke` are safe smoke-test deletions — **ask first**.
- No paper-inventory gap results from these. TASK 1 stands complete.
