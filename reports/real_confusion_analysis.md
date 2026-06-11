# Test set error analysis

> File generated automatically by `scripts/train_models.py` from the real confusion matrix.

- **Model:** `random_forest`
- **Accuracy:** 0.9886
- **Total samples:** 440
- **Total errors:** 5
- **Generated at:** 2026-06-11T12:00:57.854804+00:00

## Pairs (actual → predicted)

| Actual crop | Predicted crop | No. of errors |
|---|---|---:|
| `lentil` | `mothbeans` | 2 |
| `blackgram` | `maize` | 1 |
| `blackgram` | `mothbeans` | 1 |
| `rice` | `jute` | 1 |

## Errors per actual crop (recall < 1)

| Crop | Errors |
|---|---:|
| `lentil` | 2 |
| `blackgram` | 2 |
| `rice` | 1 |

## Absorbing crops (precision < 1)

| Crop | False positives |
|---|---:|
| `mothbeans` | 3 |
| `maize` | 1 |
| `jute` | 1 |
