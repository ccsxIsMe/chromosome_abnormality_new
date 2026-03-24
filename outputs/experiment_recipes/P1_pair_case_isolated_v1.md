# P1 Pair Case-Isolated v1

## Goal

Run the minimal pair baseline on the same case split as N0.

Task:

- input: homologous chromosome pair
- output: pair abnormal vs normal

## Build protocol

```powershell
powershell -ExecutionPolicy Bypass -File scripts/build_p1_pair_protocol.ps1
```

## Configs

- `configs/p1_pair_case_isolated_v1_resnet18_chrid_weighted_ce.yaml`
- `configs/p1_pair_case_isolated_v1_resnet50_chrid_weighted_ce.yaml`

## Run

```bash
python -m src.main --config configs/p1_pair_case_isolated_v1_resnet18_chrid_weighted_ce.yaml
python -m src.main --config configs/p1_pair_case_isolated_v1_resnet50_chrid_weighted_ce.yaml
```

## Primary comparison

Compare against non-pair N1 / N3 on:

- `val_metrics_05.auprc`
- `test_metrics_05.auprc`
- `test_metrics_best.f1`
- `test_metrics_best.abnormal.recall`

## Interpretation

If pair input clearly improves recall and AUPRC, then the gain is likely coming from homologous comparison rather than better closed-set memorization.
