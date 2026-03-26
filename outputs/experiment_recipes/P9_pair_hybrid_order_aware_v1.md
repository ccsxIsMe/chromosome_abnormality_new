# P9 Pair Hybrid Order-Aware Recipes

## Purpose

Fuse the strongest practical pair baseline with an explicit order-sensitive branch.

Compared with `P8`, this model does not force the sequence-order branch to solve the whole task alone.
Compared with `P6`, it adds direct vs reversed sequence evidence for inversion-like structure.

## Ready-to-run Config

- `configs/p9_pair_case_isolated_v1_hybrid_order_aware_resnet18_balanced_sampler_pair_contrastive.yaml`

## Command

```powershell
python -m src.main --config configs/p9_pair_case_isolated_v1_hybrid_order_aware_resnet18_balanced_sampler_pair_contrastive.yaml
```
