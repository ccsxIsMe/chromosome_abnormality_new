# P3 Pair Case-Isolated v1 Local-Global MixStyle

## Goal

Regularize the local-global pair comparator with pair MixStyle to improve test-time generalization.

This directly follows the weak generalization observed in `P2`.

## Config

- `configs/p3_pair_case_isolated_v1_local_global_resnet18_chrid_mixstyle_weighted_ce.yaml`

## Run

```bash
python -m src.main --config configs/p3_pair_case_isolated_v1_local_global_resnet18_chrid_mixstyle_weighted_ce.yaml
```

## Primary comparison

Compare against `P1` and `P2` on:

- `test_metrics_05.auprc`
- `test_metrics_best.f1`
- `test_metrics_best.abnormal.recall`

## Interpretation

If MixStyle helps, the main issue in `P2` was overfitting rather than architecture mismatch.
