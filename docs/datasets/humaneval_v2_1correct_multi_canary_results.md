# CANARY_DELTA_LOGP — per-transform ΔlogP (gemma-4-31B-it)

Decision metric = **length-normalized** ΔlogP(y|x) (per-token nats); raw-sum reported for context only (length-confounded — not gated on). KEEP iff mean per-token ΔlogP(y|x) ≤ −0.05 AND ≥70% instances negative AND revert <10%. ΔlogP(y)/Δ(TC) reported, never gating.

Stats are over rows where the transform ACTUALLY applied (structural no-ops excluded so they don't dilute Δ/sign — applicability shown separately). ΔlogP baseline = the libcst-noop variant (same pipeline → re-indent artifact cancels).

| unit | n_used | noop | revert% | meanΔcond/tok | medΔcond/tok | %neg | meanΔcond_sum | meanΔunc/tok | meanΔTC/tok | decision |
|---|---|---|---|---|---|---|---|---|---|---|
| axis2:boolean_expand | 4 | 32 | 0% | -0.226 | -0.275 | 75% | -20.2 | +0.054 | -0.280 | KEEP (low-N, n=4 — applicability 4/36) |
| axis2:dead_cruft | 36 | 0 | 0% | -0.324 | -0.291 | 100% | -33.8 | -0.082 | -0.243 | KEEP |
| axis2:inject_comment | 36 | 0 | 0% | -0.300 | -0.235 | 100% | -32.5 | -0.256 | -0.044 | KEEP |
| axis2:redundant_temp | 24 | 12 | 0% | -0.471 | -0.408 | 100% | -51.7 | -0.307 | -0.164 | KEEP |
| scheme:camel | 12 | 24 | 0% | -0.187 | -0.186 | 100% | -14.0 | -0.211 | +0.024 | KEEP (low-N, n=12 — applicability 12/36) |
| scheme:cryptic | 31 | 5 | 0% | -0.395 | -0.295 | 97% | -42.9 | -0.233 | -0.163 | KEEP |
| scheme:hungarian | 31 | 5 | 0% | -0.294 | -0.260 | 90% | -47.0 | +0.038 | -0.332 | KEEP |
| scheme:numbered | 31 | 5 | 0% | -0.416 | -0.331 | 100% | -51.1 | -0.246 | -0.170 | KEEP |
| scheme:upper | 31 | 5 | 0% | -0.269 | -0.209 | 94% | -25.7 | -0.159 | -0.110 | KEEP |
| scheme:verbose | 31 | 5 | 0% | -0.257 | -0.200 | 87% | -59.6 | +0.160 | -0.417 | KEEP |

## Diagnostics (not keep/cut candidates)

- noop humaneval_111:1:noop: validated=True (libcst round-trip; comments must survive)
- negative control: 36 emitted, 0 wrongly validated (MUST be 0 — proves the HumanEval backstop rejects broken transforms)

## VERDICT: BUILD with kept units: axis2:boolean_expand, axis2:dead_cruft, axis2:inject_comment, axis2:redundant_temp, scheme:camel, scheme:cryptic, scheme:hungarian, scheme:numbered, scheme:upper, scheme:verbose
Eyeball the side-by-side samples for human-plausibility (criterion b) before finalizing the menu.
