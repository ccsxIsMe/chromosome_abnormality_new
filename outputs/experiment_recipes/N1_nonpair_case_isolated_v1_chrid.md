# N1 Non-Pair Case-Isolated v1 with Chromosome ID

## Goal

Isolate the effect of `chromosome_id` on the full binary abnormal detection task.

This is the clean ablation against:

- `N1`: plain binary classifier without chromosome id
- `N3`: chromosome-aware multi-prototype metric detector

## Configs

- `configs/n1_nonpair_case_isolated_v1_resnet18_chrid_weighted_ce.yaml`
- `configs/n1_nonpair_case_isolated_v1_resnet50_chrid_weighted_ce.yaml`

## Run

```bash
python -m src.main --config configs/n1_nonpair_case_isolated_v1_resnet18_chrid_weighted_ce.yaml
python -m src.main --config configs/n1_nonpair_case_isolated_v1_resnet50_chrid_weighted_ce.yaml
```

## Expected outputs

- `outputs/experiments/n1_nonpair_case_isolated_v1_resnet18_chrid_weighted_ce/results.yaml`
- `outputs/experiments/n1_nonpair_case_isolated_v1_resnet50_chrid_weighted_ce/results.yaml`

## Primary comparison

Compare against the non-`chrid` N1 and the N3 prototype detector on:

- `val_metrics_05.auprc`
- `test_metrics_05.auprc`
- `test_metrics_best.f1`
- `test_metrics_best.abnormal.recall`

## Interpretation

- If `chrid + CE` already matches or beats `N3`, then most of the gain comes from chromosome prior, not prototype learning.
- If `chrid + CE` is still clearly below `N3`, then the prototype objective is contributing useful signal.
