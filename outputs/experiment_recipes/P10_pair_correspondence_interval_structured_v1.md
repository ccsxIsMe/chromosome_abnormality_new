# P10 Pair Correspondence-Interval Structured Recipes

## Purpose

This is the first full difference-first structured pair model in the project.

It differs from earlier pair baselines in three ways:

- dense direct-vs-reversed correspondence modeling along chromosome order
- interval evidence aggregation from correlation maps
- explicit structural attribute supervision:
  - pericentric / paracentric
  - breakpoint1 arm
  - breakpoint2 arm
  - breakpoint1 major band
  - breakpoint2 major band

## Why this matters

The goal is to force the model to learn reusable inversion structure rather than only binary abnormality labels or subtype identity.

## Ready-to-run Config

- `configs/p10_pair_case_isolated_v1_correspondence_interval_resnet18_pair_structured.yaml`

## Command

```powershell
python -m src.main --config configs/p10_pair_case_isolated_v1_correspondence_interval_resnet18_pair_structured.yaml
```
