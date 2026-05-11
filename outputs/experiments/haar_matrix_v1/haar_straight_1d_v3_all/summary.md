# Haar-like + AdaBoost Pair Baseline

## Settings
- profile_length: `128`
- band_width: `32`
- kernel_sizes: `[4, 8, 16, 32, 64]`
- representation_version: `v3`
- pair_orientation_align: `True`
- feature_set: `1d`
- feature_select_mode: `all`
- selected_feature_count: `757`
- n_estimators: `200`
- learning_rate: `0.5`
- weak_learner_max_depth: `1`

## Validation
- best_threshold: `0.261993`
- best_val_f1: `0.1023`

## Test
- test_f1_at_best_threshold: `0.0856`
- test_precision_abnormal: `0.0460`
- test_recall_abnormal: `0.6154`
- test_balanced_acc: `0.5090`