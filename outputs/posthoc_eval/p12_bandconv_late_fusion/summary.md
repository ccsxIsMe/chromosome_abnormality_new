# P12 + BandConv Late Fusion

## Settings
- p12_score_col: `calibrated_score`
- band_score_col: `score`
- normalize: `rank`
- best_alpha_p12: `1.00`
- best_alpha_band: `0.00`
- best_threshold: `0.964839`

## Test
- fusion_f1: `0.4711`
- fusion_precision_abnormal: `0.4380`
- fusion_recall_abnormal: `0.5096`
- fusion_balanced_acc: `0.7395`
- fusion_auprc: `0.3152`
- fusion_auroc: `0.8368`

## Baselines on same aligned subset
- p12_only_f1: `0.4711`
- band_only_f1: `0.0687`