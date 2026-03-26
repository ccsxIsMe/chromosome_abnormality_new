# P12 Pair Normal-Only Metric Recipes

## Purpose

This is a stricter unseen-oriented experiment than `P11`.

Train uses only normal homologous pairs. The model is forced to learn:

- what normal same-chromosome pair structure looks like
- how each chromosome's normal pair manifold differs

Abnormality is evaluated only as deviation from that normal manifold.

## Build Protocol

```powershell
powershell -ExecutionPolicy Bypass -File scripts/build_p12_pair_normal_only_protocol.ps1
```

## Ready-to-run Config

- `configs/p12_pair_normal_only_v1_correspondence_interval_multi_prototype_metric.yaml`

## Command

```powershell
python -m src.main --config configs/p12_pair_normal_only_v1_correspondence_interval_multi_prototype_metric.yaml
```
