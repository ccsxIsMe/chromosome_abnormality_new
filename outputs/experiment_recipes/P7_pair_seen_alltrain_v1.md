# P7 Pair Seen-All-Train Recipes

## Purpose

Closed-set long-tail subtype classification on homologous chromosome pairs.

This protocol answers:

- can pair representations distinguish train-visible inversion subtypes?
- does a long-tail contrastive objective help beyond plain CE?

## Data

- build script: `scripts/build_p7_pair_seen_alltrain_protocol.ps1`
- train: `data_protocol/p7_pair_seen_alltrain_v1/train.csv`
- val: `data_protocol/p7_pair_seen_alltrain_v1/val.csv`
- test: `data_protocol/p7_pair_seen_alltrain_v1/test.csv`
- label map: `data_protocol/p7_pair_seen_alltrain_v1/label_map.csv`

## Important Rule

- do not use `chromosome_id` as input for this protocol
- current subtype label is chromosome-specific, so chromosome_id input would leak class identity

## Ready-to-run Configs

- `configs/p7_pair_seen_alltrain_v1_siamese_resnet18_ce.yaml`
- `configs/p7_pair_seen_alltrain_v1_siamese_resnet18_bpaco.yaml`

## Commands

```powershell
powershell -ExecutionPolicy Bypass -File scripts/build_p7_pair_seen_alltrain_protocol.ps1
python -m src.main --config configs/p7_pair_seen_alltrain_v1_siamese_resnet18_ce.yaml
python -m src.main --config configs/p7_pair_seen_alltrain_v1_siamese_resnet18_bpaco.yaml
```
