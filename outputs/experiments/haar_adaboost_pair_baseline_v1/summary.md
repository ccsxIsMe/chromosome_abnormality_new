# Haar-like + AdaBoost Pair Baseline

## Settings
- profile_length: `128`
- band_width: `32`
- kernel_sizes: `[4, 8, 16, 32, 64]`
- n_estimators: `200`
- learning_rate: `0.5`
- weak_learner_max_depth: `1`

## Validation
- best_threshold: `0.327852`
- best_val_f1: `0.1253`

## Test
- test_f1_at_best_threshold: `0.0393`
- test_precision_abnormal: `0.0278`
- test_recall_abnormal: `0.0673`
- test_balanced_acc: `0.4785`