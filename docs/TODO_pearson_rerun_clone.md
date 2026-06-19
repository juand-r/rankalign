# TODO (not urgent): Pearson clone of rerun_only_tables

`docs/rerun_only_tables.tex` uses **Spearman** for the `ρ` column (the assembler
`scripts/_build_rerun_only_table.py` pulls the `spearman` metric). Per user (2026-06-19),
this *should* have been **Pearson**. Decision: leave the existing table as Spearman for now,
and later make a NEW table with identical structure but `ρ = Pearson`.

How: the per-file metrics already include a `pearson` column (same source CSVs). For the
HumanEval columns, `docs/_add_humaneval_cols_to_rerun.py` would swap `spearman`→`pearson`.
For Hyponymy/IFEval, re-run the v7 builders with `IFEVAL_METRIC=pearson` /
`ROSCH_METRIC=pearson` (they already support it) to regenerate the cells CSVs, then
re-assemble. Output as e.g. `docs/rerun_only_tables_pearson.tex`.
