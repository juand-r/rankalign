# humaneval-v2.1correct-multi — build plan

**Status:** DESIGN, preflight round 2 (round 1 = FAIL, all findings addressed
below; awaiting fresh red-team re-review). No compute run yet.
Supersedes nothing — `v2.1correct-upper` stays as the single-artifact
comparison point.

## Motivation

`v2.1correct-upper` makes correct answers atypical by renaming *every* local
to `UPPER_CASE` and `ast.unparse`-ing them (which also strips all comments &
reformats). Two problems:

1. **Label-perfect artifact.** 100% of correct are SHOUT-cased + comment-free;
   100% of wrong are natural. "Is uppercase-heavy / comment-free" is a
   near-deterministic label predictor. The headline ("TC's AUROC lift jumps
   +0.01 → +0.13 under stylization") is real but invites the obvious objection:
   *of course a typicality metric separates the classes — an OOD,
   label-perfectly-correlated surface form was injected into one class.*
2. **Single, unrealistic manipulation.** UPPER_CASE locals essentially never
   occur in real Python; the atypicality isn't representative of natural
   "atypical-but-correct" code.

Goal: keep the synthetic / unit-test-verifiable / controlled property, but
make the correct-side atypicality **diverse and realistic**, and **select
transforms by their measured effect on `log P(y|x)`** (made low while staying
plausible) rather than by intuition about "style."

## Scope & claim (what this dataset can and cannot support)

This is **not** the research-plan §3(c.2) "TC should help" dataset. §3(c.2)
specifies a 2×2: correct = high-validator **AND low** base `log P(y)`; wrong =
low-validator **AND high** base `log P(y)`, plus a mirror-image negative
control. Per locked decision 2 this build is **correct-only**: wrong rows are
byte-identical to v2.1 and are *not* steered toward high `log P(y)`, and there
is **no mirror-image negative control**.

Therefore `correct-multi` supports exactly one, weaker claim:

> **Robustness / de-artifacting check of the `correct-upper` finding.** Does
> TC's advantage over RankAlign survive when the *correct-side* atypicality is
> diverse, comment-preserving, and not a single label-perfect surface trick —
> rather than vanishing once the uppercase artifact is removed?

It **cannot** be used to claim a clean from-scratch §3(c.2) demonstration that
"TC beats RankAlign on genuine typicality." The full §3(c.2) 2×2 + mirror
control is explicitly **out of scope / future work** (would require wrong-side
construction, which decision 2 declined). This scoping is flagged to the user
in the run report; it does not change the locked correct-only parameter.

## Design

### Axis 1 — rename scheme (pick-one per example, seeded by (task_id,row_idx))

Mutually exclusive (can't be UPPER and camel at once):
`upper` · `camel` (`totalCount`) · `verbose` (`the_total_count_value`) ·
`cryptic` (`tc1`) · `hungarian` (`int_total_count`) · `numbered`
(`total_count1`).

**Rename-set correctness (no reimplementation — DRY):** the set of identifiers
to rename and the frozen set are computed by **importing
`_local_names`, `_signature_params`, `_FROZEN`, `_last_def_signature` verbatim
from `scripts/dataset_builder/build_humaneval_v2_1_correct_upper.py`** (the
trusted ast-scope reference). libcst then renames a `Name`/`arg`/binding node
iff its identifier ∈ that ast-derived rename map. libcst does not do scope
analysis and is **not** trusted to decide *which* names are renameable — it is
only the rewrite/serialization engine. A canary check asserts, per sampled
row, that the libcst-applied rename set is exactly the ast-reference set.

### Axis 2 — composable semantics-preserving transforms (stacked, seeded toggles)

Each independently toggled at prob `p`, stacked on top of the rename. Examples
get 0–several → diverse distribution, not a uniform artifact. Every transform
must be semantics-preserving **by construction**; the HumanEval unit-test
re-validation is a backstop, not the safety argument.

**Candidate set (a hypothesis — the canary measures & prunes):**
- redundant intermediate temps: `return f(x)` → `_T0 = f(x); return _T0`. The
  temp name is **reserved `_`-prefixed** (`_T0`,`_T1`,…). By construction this
  is collision-free against *renamed* identifiers too: the imported ast rename
  map skips `_`-prefixed names (`build_humaneval_v2_1_correct_upper.py:131`
  `not n.startswith("_")`), so no rename scheme (incl. `cryptic`/`numbered`)
  ever emits a `_`-prefixed name. Freshness is additionally asserted against
  the **post-rename** identifier universe (N2), not just the original scope.
- boolean-expand: `return <e>` → `if <e>: return True` / `else: return False`.
  Full AST precondition (all must hold): the target node is a `Return` whose
  `.value` is `ast.Compare` **or** `ast.UnaryOp(op=ast.Not, operand=Compare or
  BoolOp-free)`; the `Return` is a direct statement of a block (rewrite
  preserves the enclosing block/indent). **Never** `BoolOp` (`and`/`or`) or
  any non-comparison truthy expr (`a and b`, `x or default`, `lst or []`,
  `not a and b`) — those yield an operand, not a bool. Precondition checked on
  the ast tree before libcst applies the rewrite.
- mild dead/debug cruft: insert a **literal-only** no-op — `assert True`
  and/or an unused `_ = 0`. **Not** `assert <expr>` or `assert <name>` (not
  provably inert; HumanEval runs plain `python`, no `-O`).
- explicit local type annotations on a renamed local's first binding
  (`COUNT = 0` → `COUNT: int = 0`) — annotation is inert at runtime.
- *(libcst)* injected comments / inline `# ...` lines (no semantic effect).
- *(libcst)* non-PEP8 spacing (`x=1`, `f( a,b )`, trailing `;`), blank-line /
  indent-width oddity — surface only.

**Cut — idiom-neutral, ≈0 ΔlogP (different *common* idioms, not
lower-typicality; dilute the effect, add length/noise):**
comprehension↔explicit-loop · `[]`/`{}`/`""`→`list()`/`dict()`/`str()` ·
augassign-expand (`s+=1`→`s=s+1`) · redundant parens ·
`enumerate`↔`range(len(x))`.

**Excluded — unsafe** (semantics not guaranteed by construction): arithmetic
rewrites like `x*2`→`x+x` (operator overloading on lists/strings), statement
reordering (independence unprovable).

### v2 answer-format contract (libcst path)

v2 answers store line-1 at column 0; the builder re-indents (`+4 spaces` to
non-blank non-indented lines) for validation, then must store back in v2
format. The libcst path deliberately does **not** `ast.unparse` (to preserve
comments), so it replicates the column-0-line-1 store + re-indent-for-validate
contract itself (cannot reuse correct-upper's `ast.unparse` round-trip). Rows
whose answer has a column-0 `def`/garbage indentation that breaks parsing take
correct-upper's exact fallback: keep original, count as `parse_skipped`.

**Parse-source identity (N3):** ast scope/rename-map is computed on the
*exact* string `sig + "\n" + body_full_indent` (verbatim
`build_humaneval_v2_1_correct_upper.py:120-128`). libcst MUST parse the
**identical** string — the rewrite engine and the scope analysis see
byte-identical source. Set-equality of renamed *names* is insufficient under
shadowing; the canary asserts **occurrence-level** correctness: re-derive ast
scope on the libcst output and verify the set of renamed *binding+use
positions* equals what the ast reference would rename (not just the name
set). Any mismatch → revert that row, log it.

## Canary (run BEFORE the full build; gated by this preflight)

Small, cheap, decisive. **No full dataset, no training.**

1. Sample correct rows from v2.1: **≥30 correct rows across ≥10 random tasks**
   (seeded; not first-N), spanning short and long correct bodies; ≥1 long body
   per task. (≥30 not ~20 — see "Decision rule" rigor below.)
2. For each row produce: `original` (raw row, provenance only — NOT the ΔlogP
   baseline); **each candidate transform applied individually**; **2–3
   representative stacked combos**; **one deliberately-broken negative-control
   transform** (must fail validation); one **libcst no-op** = `stylize(scheme=
   None, axis2=())`. The no-op preserves **comments and semantics**; it is NOT
   guaranteed byte-identical to the raw row — `build_full_func_src`'s re-indent
   (inherited verbatim from the trusted reference) pulls any column-0 *trailing
   test-harness code* the model emitted after the solution into the function
   (~4/36 rows). This is correctness-safe (validate passes) and, crucially,
   **the noop is the ΔlogP baseline** so this re-indent is *common to baseline
   and every transformed variant and cancels in Δ* — it does not bias the
   per-transform measurement.
3. Build via the libcst path; HumanEval-validate each variant; record
   per-transform revert. Emit `canary_pairs.jsonl` with a **canary-defined,
   self-contained schema** (one record per *variant*, including `original`):
   `{task_id, row_idx, question, variant_id, transforms:[...], answer,
   validated:bool, revert_reason|null}`. **Not** `build_correct_upper.py`'s
   nested `info.mapping`/`transformed_passed` schema — that schema, and the
   `score_v2correct_upper.py` gate `p.get("transformed_passed") and
   p["info"]["mapping"]`, silently drop every pure-Axis-2 / no-local /
   `original` variant (would exit 0 with empty output). The reused scorer is
   therefore **rejected**; see step 4.
4. Score with a **committed canary scorer** (`canary_score.py`) that **copies
   verbatim** only the prompt+per-token-logp core of
   `notes/log_P_diff_plots/humaneval-v2/scripts/score_v2_humaneval.py`
   (`INSTRUCTION_COND`, `build_cond_prompt`, chat-template application,
   per-token `log P` extraction) — guaranteeing byte-identical prompts to
   v2/correct-upper (a unit test asserts the copied `INSTRUCTION_COND` /
   prompt string equals the source) — but with a canary harness that:
   (i) scores **`original` AND every variant** (so ΔlogP is computed *in-run*,
   variant − its own row's original; **no external v2/v2.1 baseline join**),
   (ii) has **no `info.mapping` / pass gate** — every emitted variant is
   scored, including no-rename rows and pure-Axis-2 transforms,
   (iii) emits `logp_cond` + `logp_uncond` per `variant_id`.
   "Reuse the prompt core, replace the harness" — the verbatim copy is the
   comparability guarantee; the rejected gate is why we don't reuse the
   harness wholesale.
5. `analyze_canary.py`: ΔlogP baseline = the **noop** variant's scores per
   `(task_id,row_idx)` (NOT raw `original` — see step 2: noop shares the
   identical pipeline so the trailing-code re-indent cancels in Δ). Metric is
   **length-normalized** per-token ΔlogP(y|x) (raw-sum reported context-only).
   **Structural no-ops are excluded from a unit's keep/cut stats:** a
   transform that didn't actually apply on a row (precondition unmet — e.g.
   `boolean_expand` on a row with no bare comparison return: ~29/36;
   `redundant_temp` on bare-Name returns: ~12/36) is tracked from
   `stylize().meta` (`axis2_applied` / non-empty rename `mapping`) and
   **dropped from that unit's Δ/sign**, with applicability (`n_used` vs `noop`
   count) reported. Folding no-ops in as Δ=0 would dilute the mean and crater
   `%neg`, structurally guaranteeing a spurious CUT (and could falsely trigger
   the degenerate "do not build" stop). Per unit: mean & spread (sign) of
   per-token Δ`log P(y|x)`, plus reported-only per-token Δ`log P(y)`, Δ(TC),
   raw-sum, revert rate, applicability, + side-by-side sample. Failed-
   validation variants → revert rate (excluded from Δ).

**Decision rule (non-circular — identical to red_team_brief §"keep"):**
Keep a transform iff **(a)** it materially lowers `log P(y|x)` *and the sign
of the per-instance ΔlogP(y|x) is consistent across a clear majority of
sampled rows*, **(b)** the transformed code is human-plausible on eyeball,
**(c)** revert rate < ~10%. **Δ(TC) and Δlog P(y) are reported for
understanding only — they NEVER gate selection.** Selection is never
conditioned on whether a transform makes TC's compensating push win, nor on
any AUROC outcome. (This is strictly no worse than the already-accepted
filter-model circularity in `V2_1_BUILD_PLAN.md`: that filter removes garbage
by either metric; this selects transforms for *plausible genuine atypicality*,
not for favorability to TC.) If a candidate's per-instance ΔlogP(y|x)
straddles zero, it is **cut** (conservative), and the sample is bumped before
deciding any borderline transform.

**Degenerate-outcome stop condition:** if **no** transform clears the
material-drop bar, the canary's conclusion is **"premise not supported — do
NOT build"**, reported as a real finding. We do not build a dataset from the
least-bad transforms.

Cost: ≥30 rows × ~10 transform variants × 2 prompts on one A100 80GB ≈ tens of
min, ≈ $1–2 (within the $50 budget). Output: `CANARY_DELTA_LOGP.md` (per-
transform table + explicit keep/cut + rationale) + `canary_pairs.jsonl` +
`canary_scores.jsonl` here.

## Build pipeline (after canary fixes the menu)

Mirror `build_humaneval_v2_1_correct_upper.py` structure (revert/log/validate
scaffolding reused), libcst rewrite engine, ast-imported rename map:
- source: `data/humaneval/v2.1/` (same as correct-upper)
- wrong rows: **unchanged, byte-identical** (locked decision 2)
- correct rows: seeded rename style + seeded subset of *kept* composable
  transforms; libcst-based so untouched code keeps comments/formatting
- re-validate every transformed answer vs HumanEval unit tests; on failure →
  drop transforms one at a time → rename-only → original. **Log per-transform
  revert rate; auto-drop any transform that reverts > ~10%** (prevents the
  dataset silently refilling with un-stylized fallbacks = uncontrolled
  artifact); record every revert + reason (mirrors correct-upper's
  `_validation_reverts.json`).
- post-build assertions: 100% transformed-correct pass HumanEval tests; wrong
  rows `diff`-identical to v2.1; correct-row count preserved (transform or
  revert, never drop); per-row libcst rename set == ast-reference set.
- output `data/humaneval/v2.1correct-multi/{humaneval_*,train}.csv` (v2.1
  schema); register `humaneval-v2.1correct-multi` task (v1→v2 pattern in
  `src/tasks/humaneval.py`).

## Required final-doc deliverable — residual-signal measurement

Decision 2's residual ("looks-stylized ⇒ correct") is **measured, not
asserted**: the final build doc must report a surface-feature separability
number — a cheap logistic / token-statistic classifier (e.g. uppercase ratio,
comment presence, identifier-length stats, whitespace features) trained to
distinguish *stylized-correct* from *untouched-wrong*, with its AUROC/accuracy
— and contrast it against the same classifier on `correct-upper` (expected:
much lower separability than uppercase-only, but quantified). Hand-waving this
is disallowed.

## Canary build findings (CPU, pre-GPU — 2026-05-16)

`build_canary.py` (committed, seed=42) ran on real v2.1: **36 correct rows ×
12 tasks → 576 variants**. Validation (trusted `validate()`) results:

- **Every real candidate (6 rename schemes + 4 Axis-2) 0/36 reverts** — the
  by-construction semantics-preservation holds on real v2.1 data, not just the
  synthetic unit tests (8/8 executional-equivalence tests, incl. a
  nested-block-placement regression test).
- **Negative control 0/36 validated (36/36 correctly rejected)** — the
  universal `raise AssertionError` injection (always-applicable, shape-
  independent) cleanly proves the `validate()` backstop is wired up.
- noop 36/36 validated (libcst round-trip preserves comments & v2 col-0).

**Measured caveat — `validate()` is a backstop, not a correctness oracle.**
An earlier negative control (flip the first comparison operator) was abandoned
after the build empirically showed it (a) is a silent no-op on bodies with no
comparison and (b) on **HumanEval/154** produced a *genuine* semantic change
that **still passed the HumanEval unit tests**. This is concrete evidence that
HumanEval re-validation has a non-zero false-negative rate on adversarial
edits. It is precisely **why** correctness here rests on transforms being
semantics-preserving *by construction* + the executional unit-test suite
(original vs every scheme×Axis-2 combo run on input batteries, 8/8), with
`validate()` only the secondary net for rare libcst edge cases on the full
build — not on `validate()` as the safety argument.

## Decisions (locked 2026-05-16)

1. **libcst** (not ast-only). Enables comment/spacing transforms and preserves
   comments+formatting on un-renamed code → removes the correct-only
   comment-strip confound in `correct-upper`. Rename *decisions* still come
   from the trusted ast reference (imported, not reimplemented).
2. **Correct-only stylization.** Wrong rows untouched, byte-identical. See
   "Scope & claim": this bounds what the artifact can claim (robustness check
   of correct-upper, not a §3(c.2) construction). Accepted; the residual
   signal is measured & disclosed per above.
3. **Name:** `humaneval-v2.1correct-multi` (confirmed).

> Compute boundary: the user authorized a $50 budget and full autonomy.
> Canary code is written + committed (reproducible) BEFORE it runs. Canary
> compute ≈ $1–2.

## Relationship to the in-flight 3-epoch run

The running gemma-4-31B-it 3-arm job (on `v2.1correct-upper`) is **kept** — it
is the single-artificial-artifact (uppercase-only) data point.
`correct-multi` answers the bounded follow-up in "Scope & claim": does the
correct-upper TC advantage survive diverse, comment-preserving, non-label-
perfect correct-side atypicality? A strong `multi` result is evidence the
correct-upper finding is **not** an artifact of the uppercase trick — it does
**not**, by itself, establish the full §3(c.2) "TC beats RankAlign on genuine
typicality" claim (that needs the out-of-scope wrong-side construction).

## Lifecycle

design (this doc) → `/raca:experiment-preflight` (red-team brief + fresh
adversarial review until PASS) → write+commit canary code → run canary (≈$1–2)
→ review ΔlogP table (`CANARY_DELTA_LOGP.md`) → finalize menu (or STOP if
degenerate) → full build + post-build assertions + residual-signal
measurement + task registration → (preflight again if design changed) → 3-arm
train/eval on `correct-multi` → compare `upper` vs `multi`.

---
## RESULTS (2026-05-18 — built & verified)

**Canary verdict** (gemma-4-31B-it, 576 variants; `CANARY_DELTA_LOGP.md`):
all 10 candidate transforms KEEP — 6 rename schemes (upper, camel, cryptic,
hungarian, numbered, verbose) + 4 composable (boolean_expand, dead_cruft,
inject_comment, redundant_temp). 0% revert; negative control 36/36 correctly
rejected; premise supported. `boolean_expand` (applies only on bare-comparison
returns) and `camel` flagged low-applicability but KEEP where they apply.

**Dataset** `data/humaneval/v2.1correct-multi/` (82 task CSVs + train.csv):
- Parity with v2.1: 82 files / 2219 test rows, **identical row counts**;
  wrong rows **byte-identical** to v2.1 (correct-only stylization, locked).
- `_BUILD_REPORT.json`: **transformed_correct_failures: 0** (every row we
  transformed still passes HumanEval, over the full set);
  **pre_existing_bad_v2_1_originals_kept: 34** — rows v2.1 labels "correct"
  whose stored answer fails strict re-validation; kept byte-identical and
  disclosed (identical handling to the trusted `v2.1correct-upper` builder's
  skip_orig_fail). 35 rows reverted during build (1.5% of correct), 0
  transforms over the revert threshold.
- **Diversity** (1202 correct test rows, 1137 actually transformed, seed=42):
  rename schemes balanced (178–227 each); composable transforms stacked per
  row = {0:123, 1:361, 2:419, 3:250, 4:49} (most rows are a *combination*);
  **all 96 (scheme × composable-subset) combinations present**, top combo 27×.
- Built reproducibly by `scripts-more/correct_multi/build_v2_1_correct_multi.py
  --menu kept_menu.json --seed 42`.

**Honest caveats:** (a) `dead_cruft`'s `assert True` can be inserted just
before a function docstring, demoting the docstring to an inert bare-string
expression — correctness-safe (0 failures) but slightly unnatural human-side;
acceptable, disclosed. (b) Single canary run, no seed-replication of ΔlogP.
(c) The residual "all-correct-stylized" signal (locked decision 2) remains —
weaker than uppercase-only but present; full residual-separability
measurement is the outstanding follow-up deliverable.

**Task:** registered as `humaneval-v2.1correct-multi` (+ per-problem
`humaneval-v2.1correct-multi-<slug>`) in `src/tasks/humaneval.py`, mirroring
`v2.1correct-upper` (same format-C prompts; only the data source differs).
