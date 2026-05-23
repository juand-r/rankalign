# Main 9-job fix1 results (persona-v1, DELTA_BINS=10)

Trained with `ranking_loss_ref_fix.py`, `--delta-bins 10`, disc-shots=few,
`--no-force-same-x`, semi-supervised 0.1, log-odds.

Eval flags: `--<self|neg>-typcorr --base-typcorr --base-model <BASE> --log-odds`,
disc-shots=few, log-odds metric, no length normalization.

Cells: mean ± stderr across 3 personas per split.

- ID = psychopathy, machiavellianism, narcissism
- OOD = desire-to-create-allies, interest-in-music, interest-in-science
- raw = `gen_score` (no eval-time TC); tc = `gen_score_typcorr` (eval-time TC)

## google/gemma-2-2b (ID)

| variant / eval | gen_roc(raw) | gen_roc(tc) | val_acc | val_roc | pearson(raw) | pearson(tc) |
|------|------|------|------|------|------|------|
| #3.New / self+base | 0.909±0.016 | 0.951±0.011 | 0.996±0.002 | 1.000±0.000 | 0.691±0.014 | 0.752±0.019 |
| #3.New / neg+base | 0.909±0.016 | 0.959±0.009 | 0.996±0.002 | 1.000±0.000 | 0.691±0.014 | 0.769±0.014 |
| #4.New+selfTC / self+base | 0.831±0.013 | 0.910±0.016 | 0.994±0.001 | 1.000±0.000 | 0.574±0.011 | 0.701±0.027 |
| #7.New+negTC / neg+base | 0.841±0.025 | 0.922±0.022 | 0.997±0.002 | 1.000±0.000 | 0.571±0.031 | 0.725±0.034 |

## google/gemma-2-2b (OOD)

| variant / eval | gen_roc(raw) | gen_roc(tc) | val_acc | val_roc | pearson(raw) | pearson(tc) |
|------|------|------|------|------|------|------|
| #3.New / self+base | 0.370±0.073 | 0.466±0.055 | 0.836±0.041 | 0.940±0.043 | -0.079±0.053 | 0.043±0.050 |
| #3.New / neg+base | 0.370±0.073 | 0.575±0.061 | 0.836±0.041 | 0.940±0.043 | -0.079±0.053 | 0.191±0.065 |
| #4.New+selfTC / self+base | 0.372±0.076 | 0.487±0.066 | 0.864±0.074 | 0.957±0.030 | -0.061±0.080 | 0.070±0.079 |
| #7.New+negTC / neg+base | 0.338±0.069 | 0.521±0.061 | 0.625±0.024 | 0.806±0.092 | 0.036±0.080 | 0.176±0.080 |

## google/gemma-2-2b-it (ID)

| variant / eval | gen_roc(raw) | gen_roc(tc) | val_acc | val_roc | pearson(raw) | pearson(tc) |
|------|------|------|------|------|------|------|
| #3.New / self+base | 0.937±0.017 | 0.979±0.005 | 0.997±0.002 | 1.000±0.000 | 0.733±0.020 | 0.794±0.015 |
| #3.New / neg+base | 0.937±0.017 | 0.980±0.008 | 0.997±0.002 | 1.000±0.000 | 0.733±0.020 | 0.804±0.022 |
| #4.New+selfTC / self+base | 0.871±0.014 | 0.942±0.010 | 0.995±0.002 | 1.000±0.000 | 0.631±0.021 | 0.754±0.023 |
| #7.New+negTC / neg+base | 0.854±0.013 | 0.941±0.012 | 0.997±0.002 | 1.000±0.000 | 0.602±0.014 | 0.727±0.015 |

## google/gemma-2-2b-it (OOD)

| variant / eval | gen_roc(raw) | gen_roc(tc) | val_acc | val_roc | pearson(raw) | pearson(tc) |
|------|------|------|------|------|------|------|
| #3.New / self+base | 0.409±0.074 | 0.484±0.048 | 0.793±0.029 | 0.924±0.035 | 0.064±0.052 | 0.161±0.036 |
| #3.New / neg+base | 0.409±0.074 | 0.634±0.065 | 0.793±0.029 | 0.924±0.035 | 0.064±0.052 | 0.380±0.065 |
| #4.New+selfTC / self+base | 0.326±0.053 | 0.325±0.037 | 0.890±0.047 | 0.953±0.027 | -0.170±0.058 | -0.176±0.054 |
| #7.New+negTC / neg+base | 0.367±0.070 | 0.612±0.078 | 0.745±0.013 | 0.909±0.066 | -0.030±0.043 | 0.236±0.095 |

## google/gemma-2-9b-it (ID)

| variant / eval | gen_roc(raw) | gen_roc(tc) | val_acc | val_roc | pearson(raw) | pearson(tc) |
|------|------|------|------|------|------|------|
| #3.New / self+base | 0.873±0.014 | 0.941±0.007 | 0.999±0.001 | 1.000±0.000 | 0.629±0.005 | 0.756±0.011 |
| #3.New / neg+base | 0.873±0.014 | 0.948±0.008 | 0.999±0.001 | 1.000±0.000 | 0.629±0.005 | 0.764±0.012 |
| #4.New+selfTC / self+base | 0.821±0.021 | 0.898±0.015 | 0.999±0.001 | 1.000±0.000 | 0.530±0.027 | 0.685±0.024 |
| #7.New+negTC / neg+base | 0.839±0.026 | 0.926±0.020 | 0.999±0.001 | 1.000±0.000 | 0.564±0.030 | 0.728±0.032 |

## google/gemma-2-9b-it (OOD)

| variant / eval | gen_roc(raw) | gen_roc(tc) | val_acc | val_roc | pearson(raw) | pearson(tc) |
|------|------|------|------|------|------|------|
| #3.New / self+base | 0.409±0.095 | 0.486±0.087 | 0.880±0.046 | 0.945±0.055 | -0.101±0.079 | 0.006±0.078 |
| #3.New / neg+base | 0.409±0.095 | 0.632±0.094 | 0.880±0.046 | 0.945±0.055 | -0.101±0.079 | 0.248±0.104 |
| #4.New+selfTC / self+base | 0.369±0.078 | 0.416±0.069 | 0.904±0.061 | 0.940±0.060 | -0.153±0.057 | -0.105±0.056 |
| #7.New+negTC / neg+base | 0.401±0.087 | 0.606±0.086 | 0.909±0.057 | 0.963±0.037 | -0.116±0.079 | 0.202±0.100 |

## Coverage

- google/gemma-2-2b #3.New / self+base: ID(3), OOD(3)
- google/gemma-2-2b #3.New / neg+base: ID(3), OOD(3)
- google/gemma-2-2b #4.New+selfTC / self+base: ID(3), OOD(3)
- google/gemma-2-2b #7.New+negTC / neg+base: ID(3), OOD(3)
- google/gemma-2-2b-it #3.New / self+base: ID(3), OOD(3)
- google/gemma-2-2b-it #3.New / neg+base: ID(3), OOD(3)
- google/gemma-2-2b-it #4.New+selfTC / self+base: ID(3), OOD(3)
- google/gemma-2-2b-it #7.New+negTC / neg+base: ID(3), OOD(3)
- google/gemma-2-9b-it #3.New / self+base: ID(3), OOD(3)
- google/gemma-2-9b-it #3.New / neg+base: ID(3), OOD(3)
- google/gemma-2-9b-it #4.New+selfTC / self+base: ID(3), OOD(3)
- google/gemma-2-9b-it #7.New+negTC / neg+base: ID(3), OOD(3)

Filter stats: 72 files skipped (wrong delta, e.g. sweep models); 0 files unparsable.
