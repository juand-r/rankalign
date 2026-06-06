# WandB run → final-model map (paper datasets)

For each **paper** dataset × model × setting: the wandb cloud runs in `juand-r/rankalign`, so you can open the training curves for any final model. Built from `_raw/wandb_paper_runs.json` by `_build_wandb_run_map.py`.

- **Setting** classified from each run's real flags (force-same-x / typicality / vlo / cft / pref+nll).
- **state**: prefer `finished`; `failed`/`crashed`/`running` runs are counted but not the model source.
- **delta regime** (NB: all are v7-code `ranking_loss_ref_fix.py`; none are v6): `bins δX` = canonical **delta-bins(v7)**, realized adaptive delta X (matches the on-disk model name, e.g. `delta2.69`); `fixed δ0.15` = **v7b / pre-delta-bins** fixed delta. The **`delta (finished)`** column lists which regimes a cell's finished runs cover — the paper model is normally the **bins** one.
- A cell with several finished runs = re-runs / a delta sweep; newest first.


## ifeval

| Model | Setting | finished | other states | delta (finished) | newest finished run (date · regime) | URL(s) of finished runs |
|---|---|---|---|---|---|---|
| gemma-2-2b | s1 (SFT-lo) | 2 | 14 | fixed δ0.15 | `ifeval-concat-g-delta0.15-nllv1.0-nllg1.0-pref0.0-semi0.1-lr1e-05` (2026-04-07 · fixed-delta(v7b/early)) | [gl89klef](https://wandb.ai/juand-r/rankalign/runs/gl89klef) [eeu26mrz](https://wandb.ai/juand-r/rankalign/runs/eeu26mrz) |
| gemma-2-2b | s2 (RankAlign) | 1 | 1 | fixed δ0.15 | `ifeval-concat-g-delta0.15-nllv0.0-nllg0.0-lr1e-05` (2026-04-14 · fixed-delta(v7b/early)) | [2l3k0605](https://wandb.ai/juand-r/rankalign/runs/2l3k0605) |
| gemma-2-2b | s3 (New+fsx) | 1 | 7 | fixed δ0.15 | `ifeval-concat-g-delta0.15-nllv1.0-nllg1.0-labelonly0.1-lr1e-05` (2026-03-27 · fixed-delta(v7b/early)) | [wa8l06xn](https://wandb.ai/juand-r/rankalign/runs/wa8l06xn) |
| gemma-2-2b | s4 (New+fsx+tc) | 2 | 11 | fixed δ0.15 | `ifeval-concat-g-delta0.15-nllv1.0-nllg1.0-labelonly0.1-lr1e-05` (2026-03-27 · fixed-delta(v7b/early)) | [uuav0k5b](https://wandb.ai/juand-r/rankalign/runs/uuav0k5b) [cwq89rfm](https://wandb.ai/juand-r/rankalign/runs/cwq89rfm) |
| gemma-2-2b | s5 (RankAlign+fsx+tc) | 4 | 9 | fixed δ0.15 | `ifeval-concat-g-delta0.15-nllv0.0-nllg0.0-labelonly0.1-lr1e-05` (2026-03-27 · fixed-delta(v7b/early)) | [y57dgniw](https://wandb.ai/juand-r/rankalign/runs/y57dgniw) [19ha2ycb](https://wandb.ai/juand-r/rankalign/runs/19ha2ycb) [dw5de9h2](https://wandb.ai/juand-r/rankalign/runs/dw5de9h2) [8xp2646h](https://wandb.ai/juand-r/rankalign/runs/8xp2646h) |
| gemma-2-2b | s7 (New+fsx+negtc) | 2 | 2 | fixed δ0.15 | `ifeval-concat-g-delta0.15-nllv1.0-nllg1.0-semi0.1-lr1e-05` (2026-04-07 · fixed-delta(v7b/early)) | [x28wnapg](https://wandb.ai/juand-r/rankalign/runs/x28wnapg) [2li6rplc](https://wandb.ai/juand-r/rankalign/runs/2li6rplc) |
| gemma-2-2b | s8 (RankAlign+fsx+negtc) | 2 | 2 | fixed δ0.15 | `ifeval-concat-g-delta0.15-nllv0.0-nllg0.0-labelonly0.1-lr1e-05` (2026-04-07 · fixed-delta(v7b/early)) | [u4u4cu0x](https://wandb.ai/juand-r/rankalign/runs/u4u4cu0x) [fqyt5y61](https://wandb.ai/juand-r/rankalign/runs/fqyt5y61) |
| gemma-2-2b | s10 (RankAlign+fsx) | 2 | 4 | fixed δ0.15 | `ifeval-concat-g-delta0.15-nllv0.0-nllg0.0-labelonly0.1-lr1e-05` (2026-03-27 · fixed-delta(v7b/early)) | [u7suiut2](https://wandb.ai/juand-r/rankalign/runs/u7suiut2) [y8o15fc5](https://wandb.ai/juand-r/rankalign/runs/y8o15fc5) |
| gemma-2-2b | s13 (SFT+cft) | 1 | 3 | bins δ0.14 | `gemma-2-2b-ifeval-concat-g-delta0.15-bins10-nllv1.0-nllg1.0-pref0.0-labelonly0.1-lr1e-05` (2026-05-25 · delta-bins(v7)) | [rh3yzjby](https://wandb.ai/juand-r/rankalign/runs/rh3yzjby) |
| gemma-2-2b-it | s1 (SFT-lo) | 0 | 1 | — | — (no finished run) | (latest crashed: [i9cyjjo1](https://wandb.ai/juand-r/rankalign/runs/i9cyjjo1)) |
| gemma-2-2b-it | s2 (RankAlign) | 0 | 3 | — | — (no finished run) | (latest crashed: [n1ayoi9e](https://wandb.ai/juand-r/rankalign/runs/n1ayoi9e)) |
| gemma-2-2b-it | s3 (New+fsx) | 0 | 1 | — | — (no finished run) | (latest crashed: [nrm05lhm](https://wandb.ai/juand-r/rankalign/runs/nrm05lhm)) |
| gemma-2-2b-it | s4 (New+fsx+tc) | 0 | 1 | — | — (no finished run) | (latest crashed: [y85baw2k](https://wandb.ai/juand-r/rankalign/runs/y85baw2k)) |
| gemma-2-2b-it | s5 (RankAlign+fsx+tc) | 0 | 1 | — | — (no finished run) | (latest crashed: [uz3t476s](https://wandb.ai/juand-r/rankalign/runs/uz3t476s)) |
| gemma-2-2b-it | s6 (RankAlign+tc) | 0 | 1 | — | — (no finished run) | (latest crashed: [4rwkunrx](https://wandb.ai/juand-r/rankalign/runs/4rwkunrx)) |
| gemma-2-2b-it | s7 (New+fsx+negtc) | 0 | 1 | — | — (no finished run) | (latest crashed: [tnq0arek](https://wandb.ai/juand-r/rankalign/runs/tnq0arek)) |
| gemma-2-2b-it | s13 (SFT+cft) | 1 | 3 | bins δ1.04 | `gemma-2-2b-it-ifeval-concat-g-delta0.15-bins10-nllv1.0-nllg1.0-pref0.0-labelonly0.1-lr1e-05` (2026-05-25 · delta-bins(v7)) | [51dbh2id](https://wandb.ai/juand-r/rankalign/runs/51dbh2id) |
| gemma-2-9b-it | s1 (SFT-lo) | 1 | 7 | fixed δ0.15 | `ifeval-concat-g-delta0.15-nllv1.0-nllg1.0-pref0.0-semi0.1-lr1e-05` (2026-04-17 · fixed-delta(v7b/early)) | [r4gy9m8r](https://wandb.ai/juand-r/rankalign/runs/r4gy9m8r) |
| gemma-2-9b-it | s2 (RankAlign) | 1 | 2 | fixed δ0.15 | `ifeval-concat-g-delta0.15-nllv0.0-nllg0.0-lr1e-05` (2026-04-14 · fixed-delta(v7b/early)) | [7bi6xwhi](https://wandb.ai/juand-r/rankalign/runs/7bi6xwhi) |
| gemma-2-9b-it | s7 (New+fsx+negtc) | 2 | 8 | fixed δ0.15 | `ifeval-concat-g-delta0.15-nllv1.0-nllg1.0-labelonly0.1-lr1e-05` (2026-04-13 · fixed-delta(v7b/early)) | [o85l7u7c](https://wandb.ai/juand-r/rankalign/runs/o85l7u7c) [m5tee4cc](https://wandb.ai/juand-r/rankalign/runs/m5tee4cc) |
| gemma-2-9b-it | s8 (RankAlign+fsx+negtc) | 2 | 8 | fixed δ0.15 | `ifeval-concat-g-delta0.15-nllv0.0-nllg0.0-labelonly0.1-lr1e-05` (2026-04-13 · fixed-delta(v7b/early)) | [by3ymdqb](https://wandb.ai/juand-r/rankalign/runs/by3ymdqb) [zfqfvisg](https://wandb.ai/juand-r/rankalign/runs/zfqfvisg) |
| gemma-2-9b-it | s13 (SFT+cft) | 0 | 5 | — | — (no finished run) | (latest failed: [cu7raxet](https://wandb.ai/juand-r/rankalign/runs/cu7raxet)) |
| Qwen3.5-9B | s2 (RankAlign) | 0 | 4 | — | — (no finished run) | (latest failed: [tstm273n](https://wandb.ai/juand-r/rankalign/runs/tstm273n)) |

## rosch / membership

| Model | Setting | finished | other states | delta (finished) | newest finished run (date · regime) | URL(s) of finished runs |
|---|---|---|---|---|---|---|
| gemma-2-2b | s1 (SFT-lo) | 2 | 4 | fixed δ0.15 | `membership-sans-rosch-v0-g-delta0.15-nllv1.0-nllg1.0-pref0.0-lr1e-05` (2026-05-12 · fixed-delta(v7b/early)) | [wnxs2jx7](https://wandb.ai/juand-r/rankalign/runs/wnxs2jx7) [xvi9k8zl](https://wandb.ai/juand-r/rankalign/runs/xvi9k8zl) |
| gemma-2-2b | s2 (RankAlign) | 1 | 1 | fixed δ0.15 | `membership-sans-rosch-v0-g-delta0.15-nllv0.0-nllg0.0-semi0.1-lr1e-05` (2026-05-02 · fixed-delta(v7b/early)) | [wg29n5el](https://wandb.ai/juand-r/rankalign/runs/wg29n5el) |
| gemma-2-2b | s3 (New+fsx) | 1 | 1 | fixed δ0.15 | `membership-sans-rosch-v0-g-delta0.15-nllv1.0-nllg1.0-semi0.1-lr1e-05` (2026-05-02 · fixed-delta(v7b/early)) | [07ya28qr](https://wandb.ai/juand-r/rankalign/runs/07ya28qr) |
| gemma-2-2b | s4 (New+fsx+tc) | 1 | 3 | fixed δ0.15 | `membership-sans-rosch-v0-g-delta0.15-nllv1.0-nllg1.0-semi0.1-lr1e-05` (2026-05-02 · fixed-delta(v7b/early)) | [q4461i33](https://wandb.ai/juand-r/rankalign/runs/q4461i33) |
| gemma-2-2b | s5 (RankAlign+fsx+tc) | 5 | 5 | bins δ0.10 \| fixed δ0.15 | `gemma-2-2b-membership-sans-rosch-v0-g-delta0.15-bins10-nllv0.0-nllg0.0-semi0.1-lr1e-05` (2026-05-25 · delta-bins(v7)) | [31vhk3ne](https://wandb.ai/juand-r/rankalign/runs/31vhk3ne) [16hry6k5](https://wandb.ai/juand-r/rankalign/runs/16hry6k5) [9zkb2a56](https://wandb.ai/juand-r/rankalign/runs/9zkb2a56) [n2s5j0oh](https://wandb.ai/juand-r/rankalign/runs/n2s5j0oh) …(+1) |
| gemma-2-2b | s6 (RankAlign+tc) | 1 | 0 | bins δ0.10 | `gemma-2-2b-membership-sans-rosch-v0-g-delta0.15-bins10-nllv0.0-nllg0.0-semi0.1-lr1e-05` (2026-05-25 · delta-bins(v7)) | [b4nulqek](https://wandb.ai/juand-r/rankalign/runs/b4nulqek) |
| gemma-2-2b | s7 (New+fsx+negtc) | 2 | 1 | bins δ0.19 \| fixed δ0.15 | `gemma-2-2b-membership-sans-rosch-v0-g-delta0.15-bins10-nllv1.0-nllg1.0-semi0.1-lr1e-05` (2026-05-25 · delta-bins(v7)) | [kj4znyaw](https://wandb.ai/juand-r/rankalign/runs/kj4znyaw) [f1t47tk7](https://wandb.ai/juand-r/rankalign/runs/f1t47tk7) |
| gemma-2-2b | s8 (RankAlign+fsx+negtc) | 4 | 5 | fixed δ0.15 | `membership-sans-rosch-v0-g-delta0.15-nllv0.0-nllg0.0-lr1e-05` (2026-05-12 · fixed-delta(v7b/early)) | [yelb0bcs](https://wandb.ai/juand-r/rankalign/runs/yelb0bcs) [aob7lkw9](https://wandb.ai/juand-r/rankalign/runs/aob7lkw9) [9yno40ot](https://wandb.ai/juand-r/rankalign/runs/9yno40ot) [9z35pcur](https://wandb.ai/juand-r/rankalign/runs/9z35pcur) |
| gemma-2-2b | s10 (RankAlign+fsx) | 2 | 2 | fixed δ0.15 | `membership-sans-rosch-v0-g-delta0.15-nllv0.0-nllg0.0-lr1e-05` (2026-05-11 · fixed-delta(v7b/early)) | [huq8nwfp](https://wandb.ai/juand-r/rankalign/runs/huq8nwfp) [tuusssxx](https://wandb.ai/juand-r/rankalign/runs/tuusssxx) |
| gemma-2-2b | s13 (SFT+cft) | 1 | 0 | bins δ0.09 | `gemma-2-2b-membership-sans-rosch-v0-g-delta0.15-bins10-nllv1.0-nllg1.0-pref0.0-labelonly0.1-lr1e-05` (2026-05-25 · delta-bins(v7)) | [ltbt3quq](https://wandb.ai/juand-r/rankalign/runs/ltbt3quq) |
| gemma-2-2b-it | s1 (SFT-lo) | 2 | 0 | bins δ0.84 \| fixed δ0.15 | `gemma-2-2b-it-membership-sans-rosch-v0-g-delta0.15-bins10-nllv1.0-nllg1.0-pref0.0-labelonly0.1-lr1e-05` (2026-05-24 · delta-bins(v7)) | [3lzbvrlh](https://wandb.ai/juand-r/rankalign/runs/3lzbvrlh) [sdfy44vx](https://wandb.ai/juand-r/rankalign/runs/sdfy44vx) |
| gemma-2-2b-it | s2 (RankAlign) | 2 | 0 | bins δ0.85 \| fixed δ0.15 | `gemma-2-2b-it-membership-sans-rosch-v0-g-delta0.15-bins10-nllv0.0-nllg0.0-semi0.1-lr1e-05` (2026-05-24 · delta-bins(v7)) | [6xehygv4](https://wandb.ai/juand-r/rankalign/runs/6xehygv4) [sute93me](https://wandb.ai/juand-r/rankalign/runs/sute93me) |
| gemma-2-2b-it | s3 (New+fsx) | 2 | 0 | bins δ1.74 \| fixed δ0.15 | `gemma-2-2b-it-membership-sans-rosch-v0-g-delta0.15-bins10-nllv1.0-nllg1.0-semi0.1-lr1e-05` (2026-05-24 · delta-bins(v7)) | [qq20g977](https://wandb.ai/juand-r/rankalign/runs/qq20g977) [gbxtn1fg](https://wandb.ai/juand-r/rankalign/runs/gbxtn1fg) |
| gemma-2-2b-it | s4 (New+fsx+tc) | 2 | 0 | bins δ1.74 \| fixed δ0.15 | `gemma-2-2b-it-membership-sans-rosch-v0-g-delta0.15-bins10-nllv1.0-nllg1.0-semi0.1-lr1e-05` (2026-05-24 · delta-bins(v7)) | [nbwwr67u](https://wandb.ai/juand-r/rankalign/runs/nbwwr67u) [yb9j4oda](https://wandb.ai/juand-r/rankalign/runs/yb9j4oda) |
| gemma-2-2b-it | s5 (RankAlign+fsx+tc) | 2 | 0 | bins δ0.85 \| fixed δ0.15 | `gemma-2-2b-it-membership-sans-rosch-v0-g-delta0.15-bins10-nllv0.0-nllg0.0-semi0.1-lr1e-05` (2026-05-24 · delta-bins(v7)) | [jh7klif9](https://wandb.ai/juand-r/rankalign/runs/jh7klif9) [ojup07h5](https://wandb.ai/juand-r/rankalign/runs/ojup07h5) |
| gemma-2-2b-it | s6 (RankAlign+tc) | 1 | 0 | bins δ0.85 | `gemma-2-2b-it-membership-sans-rosch-v0-g-delta0.15-bins10-nllv0.0-nllg0.0-semi0.1-lr1e-05` (2026-05-24 · delta-bins(v7)) | [ils8e9xa](https://wandb.ai/juand-r/rankalign/runs/ils8e9xa) |
| gemma-2-2b-it | s7 (New+fsx+negtc) | 2 | 0 | bins δ1.74 \| fixed δ0.15 | `gemma-2-2b-it-membership-sans-rosch-v0-g-delta0.15-bins10-nllv1.0-nllg1.0-semi0.1-lr1e-05` (2026-05-24 · delta-bins(v7)) | [7171mceh](https://wandb.ai/juand-r/rankalign/runs/7171mceh) [ynvtab62](https://wandb.ai/juand-r/rankalign/runs/ynvtab62) |
| gemma-2-2b-it | s8 (RankAlign+fsx+negtc) | 1 | 0 | fixed δ0.15 | `membership-sans-rosch-v0-g-delta0.15-nllv0.0-nllg0.0-semi0.1-lr1e-05` (2026-05-02 · fixed-delta(v7b/early)) | [8k5puld1](https://wandb.ai/juand-r/rankalign/runs/8k5puld1) |
| gemma-2-2b-it | s11 (New+tc) | 1 | 0 | bins δ1.74 | `gemma-2-2b-it-membership-sans-rosch-v0-g-delta0.15-bins10-nllv1.0-nllg1.0-semi0.1-lr1e-05` (2026-05-24 · delta-bins(v7)) | [ae1k10za](https://wandb.ai/juand-r/rankalign/runs/ae1k10za) |
| gemma-2-2b-it | s12 (New+negtc) | 1 | 0 | bins δ1.74 | `gemma-2-2b-it-membership-sans-rosch-v0-g-delta0.15-bins10-nllv1.0-nllg1.0-semi0.1-lr1e-05` (2026-05-24 · delta-bins(v7)) | [ugxi38h0](https://wandb.ai/juand-r/rankalign/runs/ugxi38h0) |
| gemma-2-2b-it | s13 (SFT+cft) | 1 | 0 | bins δ0.81 | `gemma-2-2b-it-membership-sans-rosch-v0-g-delta0.15-bins10-nllv1.0-nllg1.0-pref0.0-labelonly0.1-lr1e-05` (2026-05-25 · delta-bins(v7)) | [26toxtes](https://wandb.ai/juand-r/rankalign/runs/26toxtes) |
| gemma-2-9b-it | s1 (SFT-lo) | 2 | 1 | fixed δ0.15 | `membership-sans-rosch-v0-g-delta0.15-nllv1.0-nllg1.0-pref0.0-labelonly0.1-lr1e-05` (2026-05-01 · fixed-delta(v7b/early)) | [rmq0ixu0](https://wandb.ai/juand-r/rankalign/runs/rmq0ixu0) [yfzd8lng](https://wandb.ai/juand-r/rankalign/runs/yfzd8lng) |
| gemma-2-9b-it | s2 (RankAlign) | 3 | 0 | bins δ1.42 \| fixed δ0.15 | `gemma-2-9b-it-membership-sans-rosch-v0-g-delta0.15-bins10-nllv0.0-nllg0.0-semi0.1-lr1e-05` (2026-05-24 · delta-bins(v7)) | [mowgknkz](https://wandb.ai/juand-r/rankalign/runs/mowgknkz) [u8xb90kg](https://wandb.ai/juand-r/rankalign/runs/u8xb90kg) [9g9hejja](https://wandb.ai/juand-r/rankalign/runs/9g9hejja) |
| gemma-2-9b-it | s3 (New+fsx) | 2 | 1 | bins δ2.69 \| fixed δ0.15 | `gemma-2-9b-it-membership-sans-rosch-v0-g-delta0.15-bins10-nllv1.0-nllg1.0-semi0.1-lr1e-05` (2026-05-24 · delta-bins(v7)) | [53aa9vtk](https://wandb.ai/juand-r/rankalign/runs/53aa9vtk) [8kmujdgc](https://wandb.ai/juand-r/rankalign/runs/8kmujdgc) |
| gemma-2-9b-it | s4 (New+fsx+tc) | 2 | 1 | bins δ2.69 \| fixed δ0.15 | `gemma-2-9b-it-membership-sans-rosch-v0-g-delta0.15-bins10-nllv1.0-nllg1.0-semi0.1-lr1e-05` (2026-05-24 · delta-bins(v7)) | [kdu9bokp](https://wandb.ai/juand-r/rankalign/runs/kdu9bokp) [r5ewegy2](https://wandb.ai/juand-r/rankalign/runs/r5ewegy2) |
| gemma-2-9b-it | s5 (RankAlign+fsx+tc) | 2 | 1 | bins δ1.42 \| fixed δ0.15 | `gemma-2-9b-it-membership-sans-rosch-v0-g-delta0.15-bins10-nllv0.0-nllg0.0-semi0.1-lr1e-05` (2026-05-24 · delta-bins(v7)) | [2d79vpq2](https://wandb.ai/juand-r/rankalign/runs/2d79vpq2) [qc0mlesf](https://wandb.ai/juand-r/rankalign/runs/qc0mlesf) |
| gemma-2-9b-it | s6 (RankAlign+tc) | 2 | 1 | bins δ1.42 \| fixed δ0.15 | `membership-sans-rosch-v0-g-delta0.15-nllv0.0-nllg0.0-semi0.1-lr1e-05` (2026-05-24 · fixed-delta(v7b/early)) | [ok2y8bg2](https://wandb.ai/juand-r/rankalign/runs/ok2y8bg2) [ew1w8jr0](https://wandb.ai/juand-r/rankalign/runs/ew1w8jr0) |
| gemma-2-9b-it | s7 (New+fsx+negtc) | 2 | 0 | bins δ2.69 \| fixed δ0.15 | `gemma-2-9b-it-membership-sans-rosch-v0-g-delta0.15-bins10-nllv1.0-nllg1.0-semi0.1-lr1e-05` (2026-05-24 · delta-bins(v7)) | [6cvptkrh](https://wandb.ai/juand-r/rankalign/runs/6cvptkrh) [5sbksqrr](https://wandb.ai/juand-r/rankalign/runs/5sbksqrr) |
| gemma-2-9b-it | s8 (RankAlign+fsx+negtc) | 1 | 0 | fixed δ0.15 | `membership-sans-rosch-v0-g-delta0.15-nllv0.0-nllg0.0-semi0.1-lr1e-05` (2026-05-02 · fixed-delta(v7b/early)) | [8iak506u](https://wandb.ai/juand-r/rankalign/runs/8iak506u) |
| gemma-2-9b-it | s9 (RankAlign+negtc) | 1 | 1 | fixed δ0.15 | `membership-sans-rosch-v0-g-delta0.15-nllv0.0-nllg0.0-semi0.1-lr1e-05` (2026-05-24 · fixed-delta(v7b/early)) | [4wk0qecd](https://wandb.ai/juand-r/rankalign/runs/4wk0qecd) |
| gemma-2-9b-it | s11 (New+tc) | 1 | 0 | bins δ2.69 | `gemma-2-9b-it-membership-sans-rosch-v0-g-delta0.15-bins10-nllv1.0-nllg1.0-semi0.1-lr1e-05` (2026-05-24 · delta-bins(v7)) | [neb0jg7x](https://wandb.ai/juand-r/rankalign/runs/neb0jg7x) |
| gemma-2-9b-it | s12 (New+negtc) | 1 | 0 | bins δ2.69 | `gemma-2-9b-it-membership-sans-rosch-v0-g-delta0.15-bins10-nllv1.0-nllg1.0-semi0.1-lr1e-05` (2026-05-24 · delta-bins(v7)) | [kw14nf8n](https://wandb.ai/juand-r/rankalign/runs/kw14nf8n) |
| gemma-2-9b-it | s13 (SFT+cft) | 1 | 0 | bins δ1.43 | `gemma-2-9b-it-membership-sans-rosch-v0-g-delta0.15-bins10-nllv1.0-nllg1.0-pref0.0-labelonly0.1-lr1e-05` (2026-05-25 · delta-bins(v7)) | [z98ox3ey](https://wandb.ai/juand-r/rankalign/runs/z98ox3ey) |
| Qwen3.5-9B | s2 (RankAlign) | 1 | 0 | fixed δ5.00 | `membership-sans-rosch-v0-d-delta5.0-nllv0.0-nllg0.0-lr1e-05` (2026-05-21 · fixed-delta(v7b/early)) | [x4ggmg5l](https://wandb.ai/juand-r/rankalign/runs/x4ggmg5l) |

## humaneval-cu

| Model | Setting | finished | other states | delta (finished) | newest finished run (date · regime) | URL(s) of finished runs |
|---|---|---|---|---|---|---|
| gemma-4-31B-it | s13 (SFT+cft) | 0 | 1 | — | — (no finished run) | (latest crashed: [vuvz142t](https://wandb.ai/juand-r/rankalign/runs/vuvz142t)) |

## Run-state summary (all 210 paper-task runs)

| state | count |
|---|---|
| finished | 82 |
| failed | 64 |
| crashed | 64 |

> Many runs are `failed`/`crashed` (OOM, preemption, early bugs). Only `finished` runs carry complete curves. If a cell shows finished=0, the curves for that exact model are incomplete in wandb even though the trained checkpoint exists (see training_inventory).
