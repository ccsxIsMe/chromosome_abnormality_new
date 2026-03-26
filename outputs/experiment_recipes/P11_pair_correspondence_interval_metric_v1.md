# P11 Pair Correspondence-Interval Metric Recipes

## Purpose

Move from binary classification toward chromosome-conditional anomaly detection.

This experiment keeps the strong difference-first correspondence model from `P10`,
but the final decision is based on distance to chromosome-specific normal prototypes
instead of only a closed-set classifier logit.

## Why this is the next step

- `P10` showed that structured supervision helps, but not enough to beat `P6`
- the remaining bottleneck is unseen subtype generalization
- for unseen inversion subtypes, the stronger prior is not abnormal class identity
  but deviation from the normal homologous-pair structure of the same chromosome

## Ready-to-run Config

- `configs/p11_pair_case_isolated_v1_correspondence_interval_multi_prototype_metric.yaml`

## Command

```powershell
python -m src.main --config configs/p11_pair_case_isolated_v1_correspondence_interval_multi_prototype_metric.yaml
```
