# gemma-2-9b-it ifeval in-domain (ID) eval — logs (2026-05-25/26)

Eval-run logs for the gemma-2-9b-it RankAlign ifeval **in-domain** eval (prompts N>21),
settings s1/s2/s3/s4/s7. The eval is reproducible from the committed `run_ra9b_ifeval_id_eval.sh`
+ the eval models on HF (`TAUR-dev/rankalign-v7-gemma2-9b-it-ifeval-s{1..7}-ep2`) + the scored
CSVs in `outputs_gemma4_from_pod-v7/ra9b_ifeval/`.

Only `s4-migration_ra9b_id_eval.log.gz` is preserved here — the s4 setting's eval ran on a
migrated pod (`qrz3m6s1hm34ob`) that was still running during cleanup. The s1/s2/s3/s7 pods had
already EXITED by the time of this backup, so their eval logs were on stopped (inaccessible)
volumes; their **results are fully in the repo** regardless. These logs are process/tqdm output,
not unique provenance.
