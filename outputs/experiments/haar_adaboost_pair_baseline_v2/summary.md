# Haar-like + AdaBoost Pair Baseline

## Settings
- profile_length: `128`
- band_width: `32`
- kernel_sizes: `[4, 8, 16, 32, 64]`
- representation_version: `v2`
- pair_orientation_align: `True`
- n_estimators: `200`
- learning_rate: `0.5`
- weak_learner_max_depth: `1`

## Validation
- best_threshold: `0.303563`
- best_val_f1: `0.1060`

## Test
- test_f1_at_best_threshold: `0.0800`
- test_precision_abnormal: `0.0505`
- test_recall_abnormal: `0.1923`
- test_balanced_acc: `0.5115`