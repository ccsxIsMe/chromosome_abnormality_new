# P16 + Haar Frozen-Feature Boost Fusion

## Settings
- frozen_model_config: `configs/p16_pair_normal_only_v1_correspondence_interval_convnext_tiny_multi_prototype_metric.yaml`
- frozen_model_ckpt: `outputs/experiments/p16_pair_normal_only_v1_correspondence_interval_convnext_tiny_multi_prototype_metric/best_model.pth`
- classifier: `gradient_boosting`
- n_estimators: `200`
- learning_rate: `0.05`
- max_depth: `2`
- subsample: `1.0`
- use_haar: `False`
- use_embedding: `True`
- use_prototype_distances: `True`
- use_model_scalars: `True`
- embedding_pca_dim: `64`
- num_feature_columns: `76`
- feature_sanitize_info: `{'train_nan_like_before_fill': 0, 'train_nan_like_after_fill': 0, 'val_nan_like_before_fill': 0, 'val_nan_like_after_fill': 0, 'test_nan_like_before_fill': 0, 'test_nan_like_after_fill': 0}`

## Validation
- best_threshold: `0.653004`
- best_val_f1: `0.1549`

## Test
- test_f1_at_best_threshold: `0.0420`
- test_precision_abnormal: `0.0330`
- test_recall_abnormal: `0.0577`
- test_balanced_acc: `0.4892`
- test_auprc: `0.0670`
- test_auroc: `0.5516`