# P2 Pair Case-Isolated v1 Local-Global Comparator

## Goal

Use a stronger pair architecture that compares homologous chromosomes at both local feature-map level and global token level.

This is the next step after the minimal Siamese baseline.

## Configs

- `configs/p2_pair_case_isolated_v1_local_global_resnet18_chrid_weighted_ce.yaml`
- `configs/p2_pair_case_isolated_v1_local_global_resnet18_chrid_balanced_supcon.yaml`

## Run

```bash
python -m src.main --config configs/p2_pair_case_isolated_v1_local_global_resnet18_chrid_weighted_ce.yaml
python -m src.main --config configs/p2_pair_case_isolated_v1_local_global_resnet18_chrid_balanced_supcon.yaml
```

## Primary comparison

Compare against `P1` on:

- `val_metrics_05.auprc`
- `test_metrics_05.auprc`
- `test_metrics_best.f1`
- `test_metrics_best.abnormal.recall`

## Interpretation

If this model improves over the Siamese baseline, the improvement likely comes from modeling local homologous differences rather than from pair input alone.
