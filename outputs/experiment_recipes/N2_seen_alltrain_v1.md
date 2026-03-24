# N2 Seen-All-Train Recipes

## Purpose

Harder seen-only abnormal subtype classification baseline.

Compared with `N2_seen_only_v1`, this version:

- trains on all 19 abnormal subtypes visible in the N0 train split
- evaluates only on seen abnormal rows from val/test

This is a more realistic closed-set benchmark because train contains many extra seen classes that do not appear in the current val/test slices.

## Data

- train: `data_protocol/n2_seen_alltrain_v1/train.csv`
- val: `data_protocol/n2_seen_alltrain_v1/val.csv`
- test: `data_protocol/n2_seen_alltrain_v1/test.csv`
- label map: `data_protocol/n2_seen_alltrain_v1/label_map.csv`

## Ready-to-run Configs

- `configs/n2_seen_alltrain_v1_resnet18_weighted_ce.yaml`
- `configs/n2_seen_alltrain_v1_resnet50_weighted_ce.yaml`

## Commands

```powershell
python src/main.py --config configs/n2_seen_alltrain_v1_resnet18_weighted_ce.yaml
python src/main.py --config configs/n2_seen_alltrain_v1_resnet50_weighted_ce.yaml
```

## Expected Output

- `outputs/experiments/n2_seen_alltrain_v1_resnet18_weighted_ce/results.yaml`
- `outputs/experiments/n2_seen_alltrain_v1_resnet50_weighted_ce/results.yaml`

## Why this matters

If the previous 6-class N2 result was high only because the task was too small and too easy, this 19-class variant should expose that quickly.
