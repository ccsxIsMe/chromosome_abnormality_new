# P5 Pair Case-Isolated v1 Siamese Pair + Side Head

## Goal

Add an auxiliary abnormal-side head to the Siamese pair detector.

## Important caveat

In the stored pair CSV, abnormal pairs are ordered as `left=normal, right=abnormal`.
Therefore, side supervision is degenerate unless we randomize pair order during training.

This config enables train-time random left/right swapping and remaps side labels online.

## Config

- `configs/p5_pair_case_isolated_v1_siamese_resnet18_chrid_pair_side.yaml`

## Run

```bash
python -m src.main --config configs/p5_pair_case_isolated_v1_siamese_resnet18_chrid_pair_side.yaml
```

## Primary comparison

Compare against `P1` on:

- `test_metrics_05.auprc`
- `test_metrics_best.f1`
- `test_metrics_best.abnormal.recall`

## Note on side metrics

Val/test side metrics are weak evidence because original evaluation CSV ordering still places abnormal chromosomes on the right.
The main purpose of this head is training-time structured regularization.
