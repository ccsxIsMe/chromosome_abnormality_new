# N3 Non-Pair Case-Isolated v1

## Goal

Use chromosome-aware multi-prototype metric learning for abnormal detection on the full N0 split.

This baseline differs from N1 in two ways:

- it uses `chromosome_id`
- it replaces a pure softmax classifier with chromosome-specific learnable normal prototypes

## Configs

- `configs/n3_nonpair_case_isolated_v1_resnet18_multi_prototype_metric.yaml`
- `configs/n3_nonpair_case_isolated_v1_resnet50_multi_prototype_metric.yaml`

## Run

```bash
python -m src.main --config configs/n3_nonpair_case_isolated_v1_resnet18_multi_prototype_metric.yaml
python -m src.main --config configs/n3_nonpair_case_isolated_v1_resnet50_multi_prototype_metric.yaml
```

## Expected outputs

- `outputs/experiments/n3_nonpair_case_isolated_v1_resnet18_multi_prototype_metric/results.yaml`
- `outputs/experiments/n3_nonpair_case_isolated_v1_resnet50_multi_prototype_metric/results.yaml`

## Primary metrics

- `val_metrics_05.auprc`
- `test_metrics_05.auprc`
- `test_metrics_best.f1`
- `test_metrics_best.abnormal.recall`
- `test_casewise_best.f1`
- `test_casewise_best.abnormal.recall`

## Interpretation

If this baseline clearly beats N1, the gain can be attributed to chromosome-conditioned normal prototypes rather than closed-set subtype memorization.
