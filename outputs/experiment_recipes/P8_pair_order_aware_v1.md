# P8 Pair Order-Aware Recipes

## Purpose

Binary pair abnormal detection with an order-sensitive model that explicitly compares:

- direct homologous alignment
- flipped alignment along chromosome axis

This is the first model in the project that tries to use inversion-specific sequence reversal evidence instead of only pooled pair similarity.

## Ready-to-run Config

- `configs/p8_pair_case_isolated_v1_order_aware_resnet18_balanced_sampler_ce.yaml`

## Command

```powershell
python -m src.main --config configs/p8_pair_case_isolated_v1_order_aware_resnet18_balanced_sampler_ce.yaml
```
