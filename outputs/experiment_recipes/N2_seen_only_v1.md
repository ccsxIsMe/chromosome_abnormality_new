# Baseline N2 Recipes

## Purpose

Seen-only abnormal subtype classification on the non-pair protocol.

This protocol keeps only the 6 `seen_eval` subtypes from N0:

- `1::inversion`
- `2::inversion`
- `4::inversion`
- `7::inversion`
- `13::inversion`
- `19::inversion`

Label map:

- `0 -> 1::inversion`
- `1 -> 2::inversion`
- `2 -> 4::inversion`
- `3 -> 7::inversion`
- `4 -> 13::inversion`
- `5 -> 19::inversion`

## Data

- train: `data_protocol/n2_seen_only_v1/train.csv`
- val: `data_protocol/n2_seen_only_v1/val.csv`
- test: `data_protocol/n2_seen_only_v1/test.csv`

## Ready-to-run Configs

- `configs/n2_seen_only_v1_resnet18_weighted_ce.yaml`
- `configs/n2_seen_only_v1_resnet50_weighted_ce.yaml`

Both configs use:

- `num_classes = 6`
- weighted CE
- best-model metric: `macro_f1`

## Commands

```powershell
python src/main.py --config configs/n2_seen_only_v1_resnet18_weighted_ce.yaml
python src/main.py --config configs/n2_seen_only_v1_resnet50_weighted_ce.yaml
```

## Expected Output Locations

- `outputs/experiments/n2_seen_only_v1_resnet18_weighted_ce/results.yaml`
- `outputs/experiments/n2_seen_only_v1_resnet50_weighted_ce/results.yaml`

## Caveat

This is a very small closed-set benchmark:

- train: 161 images / 7 cases
- val: 51 images / 4 cases
- test: 20 images / 3 cases

It is still useful as the first answer to:

"If we remove unseen subtypes and evaluate only on seen abnormal categories, how far can a standard classifier go?"
