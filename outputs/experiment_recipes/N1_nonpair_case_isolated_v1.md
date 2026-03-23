# Baseline N1 Recipes

## Purpose

Single-image binary classification on the case-isolated non-pair protocol:

- train: `data_protocol/nonpair_case_isolated_v1/train.csv`
- val: `data_protocol/nonpair_case_isolated_v1/val.csv`
- test: `data_protocol/nonpair_case_isolated_v1/test.csv`

## Ready-to-run Configs

- `configs/n1_nonpair_case_isolated_v1_resnet18_weighted_ce.yaml`
- `configs/n1_nonpair_case_isolated_v1_resnet50_weighted_ce.yaml`

Both configs use:

- `use_pair_input = false`
- weighted cross entropy with train-set balanced weights `[0.511111, 23.0]`
- best-model selection metric: `auprc`

## Commands

```powershell
python src/main.py --config configs/n1_nonpair_case_isolated_v1_resnet18_weighted_ce.yaml
python src/main.py --config configs/n1_nonpair_case_isolated_v1_resnet50_weighted_ce.yaml
```

## Expected Output Locations

- `outputs/experiments/n1_nonpair_case_isolated_v1_resnet18_weighted_ce/results.yaml`
- `outputs/experiments/n1_nonpair_case_isolated_v1_resnet50_weighted_ce/results.yaml`

## Notes

- This is the first N1 run on the new N0 protocol.
- For N1 we evaluate only binary detection on full val/test.
- Seen/unseen subtype slicing is not used yet in the loss or model, only later for analysis.
