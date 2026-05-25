# Morning Status — 2026-05-25 ~13:30 UTC (08:30 CT)

Legend: ✓ = done · ✗ = failed · ⏳ = in-flight (ETA) · – = not started
Eval prefixes shown in order: basetyp · self · basetypneg · neg
(non-TC settings need all 4; tc-self need basetyp+self; tc-neg need basetypneg+neg)

## gemma-2-2b × ifeval

| Setting | Train | basetyp | self | basetypneg | neg | Notes |
|---|---|---|---|---|---|---|
| s1 | – | – | – | – | – |  |
| s2 | – | – | – | – | – |  |
| s3 | – | – | – | – | – |  |
| s4 | – | – | – | . | . |  |
| s5 | – | – | – | . | . |  |
| s6 | – | – | – | . | . |  |
| s7 | – | . | . | – | – |  |
| s11 | – | – | – | . | . |  |
| s12 | – | . | . | – | – |  |
| s13 | ✓ ep=2 | – | – | – | – |  |

## gemma-2-2b × membership

| Setting | Train | basetyp | self | basetypneg | neg | Notes |
|---|---|---|---|---|---|---|
| s1 | – | – | – | – | – |  |
| s2 | – | – | – | – | – |  |
| s3 | – | – | – | – | – |  |
| s4 | – | – | – | . | . |  |
| s5 | – | – | – | . | . |  |
| s6 | – | – | – | . | . |  |
| s7 | – | . | . | – | – |  |
| s11 | – | – | – | . | . |  |
| s12 | – | . | . | – | – |  |
| s13 | ✓ ep=2 | ✓ | – | ✓ | – |  |

## gemma-2-2b × persona

| Setting | Train | basetyp | self | basetypneg | neg | Notes |
|---|---|---|---|---|---|---|
| s1 | – | – | – | – | – |  |
| s2 | ✓ ep=1 | ✓ | – | ✓ | – |  |
| s3 | – | – | – | – | – |  |
| s4 | – | – | – | . | . |  |
| s5 | – | – | – | . | . |  |
| s6 | – | – | – | . | . |  |
| s7 | – | . | . | – | – |  |
| s11 | ✓ ep=1 | ✓ | ✓ | . | . |  |
| s12 | ✓ ep=1 | . | . | ✓ | – |  |
| s13 | ✓ ep=2 | ✓ | ✓ | ✓ | ✓ |  |

## gemma-2-2b-it × ifeval

| Setting | Train | basetyp | self | basetypneg | neg | Notes |
|---|---|---|---|---|---|---|
| s1 | ⏳ 42455 R 24:36 (ETA ≤ 5:35:24) | – | – | – | – |  |
| s2 | – | – | – | – | – |  |
| s3 | – | – | – | – | – |  |
| s4 | – | – | – | . | . |  |
| s5 | – | – | – | . | . |  |
| s6 | – | – | – | . | . |  |
| s7 | – | . | . | – | – |  |
| s11 | – | – | – | . | . |  |
| s12 | – | . | . | – | – |  |
| s13 | ✓ ep=2 | – | – | – | – |  |

## gemma-2-2b-it × membership

| Setting | Train | basetyp | self | basetypneg | neg | Notes |
|---|---|---|---|---|---|---|
| s1 | ✓ ep=2 | ✓ | ✓ | ✓ | ✓ |  |
| s2 | ✓ ep=2 | – | ✓ | ✓ | ✓ |  |
| s3 | ✓ ep=2 | ✓ | ✓ | ✓ | ✓ |  |
| s4 | ✓ ep=2 | ✓ | ✓ | . | . |  |
| s5 | ✓ ep=2 | ✓ | ✓ | . | . |  |
| s6 | ✓ ep=2 | ✓ | ✓ | . | . |  |
| s7 | ✓ ep=2 | . | . | ✓ | ✓ |  |
| s11 | ✓ ep=2 | ✓ | ✓ | . | . |  |
| s12 | ✓ ep=2 | . | . | ✓ | ✓ |  |
| s13 | ✓ ep=2 | ✓ | ✓ | ✓ | ✓ |  |

## gemma-2-2b-it × persona

| Setting | Train | basetyp | self | basetypneg | neg | Notes |
|---|---|---|---|---|---|---|
| s1 | ✓ ep=2 | ✓ | ✓ | ✓ | ✓ |  |
| s2 | ✓ ep=2 | ✓ | ✓ | ✓ | ✓ |  |
| s3 | ✓ ep=2 | ✓ | ✓ | ✓ | ✓ |  |
| s4 | ✓ ep=2 | ✓ | ✓ | . | . |  |
| s5 | ✓ ep=2 | ✓ | ✓ | . | . |  |
| s6 | ✓ ep=2 | ✓ | ✓ | . | . |  |
| s7 | ✓ ep=2 | . | . | ✓ | ✓ |  |
| s11 | ✓ ep=2 | ✓ | ✓ | . | . |  |
| s12 | ✓ ep=2 | . | . | ✓ | ✓ |  |
| s13 | ✓ ep=2 | ✓ | ✓ | ✓ | ✓ |  |

## gemma-2-9b-it × ifeval

| Setting | Train | basetyp | self | basetypneg | neg | Notes |
|---|---|---|---|---|---|---|
| s1 | – | – | – | – | – |  |
| s2 | – | – | – | – | – |  |
| s3 | – | – | – | – | – |  |
| s4 | – | – | – | . | . |  |
| s5 | – | – | – | . | . |  |
| s6 | – | – | – | . | . |  |
| s7 | – | . | . | – | – |  |
| s11 | – | – | – | . | . |  |
| s12 | – | . | . | – | – |  |
| s13 | ⏳ 42343 R 5:33:12 (ETA ≤ 8:26:48) | – | – | – | – |  |

## gemma-2-9b-it × membership

| Setting | Train | basetyp | self | basetypneg | neg | Notes |
|---|---|---|---|---|---|---|
| s1 | ✓ ep=1 | ✓ | ✓ | ✓ | ✓ |  |
| s2 | ✓ ep=2 | – | ✓ | ✓ | ✓ |  |
| s3 | ✓ ep=2 | ✓ | ✓ | ✓ | ✓ |  |
| s4 | ✓ ep=2 | ✓ | ✓ | . | . |  |
| s5 | ✓ ep=2 | ✓ | ✓ | . | . |  |
| s6 | ✓ ep=2 | – | ✓ | . | . |  |
| s7 | ✓ ep=2 | . | . | ✓ | ✓ |  |
| s11 | ✓ ep=2 | ✓ | ✓ | . | . |  |
| s12 | ✓ ep=2 | . | . | ✓ | ✓ |  |
| s13 | ✓ ep=2 | ✓ | – | ✓ | – |  |

## gemma-2-9b-it × persona

| Setting | Train | basetyp | self | basetypneg | neg | Notes |
|---|---|---|---|---|---|---|
| s1 | ⏳ 42307 R 6:55:31 (ETA ≤ 1:04:29) | ✓ | ✓ | ✓ | ✓ |  |
| s2 | ✓ ep=2 | ✓ | ✓ | ✓ | ✓ |  |
| s3 | ⏳ 42377 R 3:48:08 (ETA ≤ 5:11:52) | ✓ | ✓ | ✓ | ✓ |  |
| s4 | ✓ ep=1 | ✓ | ✓ | . | . |  |
| s5 | ✓ ep=2 | ✓ | ✓ | . | . |  |
| s6 | ✓ ep=2 | ✓ | ✓ | . | . |  |
| s7 | ✓ ep=2 | . | . | ✓ | ✓ |  |
| s11 | ✓ ep=1 | ✓ | ✓ | . | . |  |
| s12 | ✓ ep=1 | . | . | ✓ | ✓ |  |
| s13 | ⏳ 42286 R 8:32:53 (ETA ≤ 7:27:07) | ✓ | ✓ | ✓ | ✓ |  |

## gemma-4-31B-it × humaneval

| Setting | Train | basetyp | self | basetypneg | neg | Notes |
|---|---|---|---|---|---|---|
| s1 | – | – | – | – | – |  |
| s2 | – | – | – | – | – |  |
| s3 | – | – | – | – | – |  |
| s4 | – | – | – | . | . |  |
| s5 | – | – | – | . | . |  |
| s6 | – | – | – | . | . |  |
| s7 | – | . | . | – | – |  |
| s11 | – | – | – | . | . |  |
| s12 | – | . | . | – | – |  |
| s13 | ⏳ 42114 R 13:27:08 (ETA ≤ 10:32:52) | – | – | – | – |  |

