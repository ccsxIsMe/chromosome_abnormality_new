# P16 + Haar Frozen-Feature Boost Fusion

## Settings
- frozen_model_config: `configs/p16_pair_normal_only_v1_correspondence_interval_convnext_tiny_multi_prototype_metric_straight_out_v1.yaml`
- frozen_model_ckpt: `outputs/experiments/p16_pair_normal_only_v1_correspondence_interval_convnext_tiny_multi_prototype_metric_straight_out_v1/best_model.pth`
- classifier: `gradient_boosting`
- n_estimators: `200`
- learning_rate: `0.05`
- max_depth: `2`
- subsample: `1.0`
- use_haar: `True`
- use_embedding: `True`
- use_prototype_distances: `True`
- use_model_scalars: `True`
- haar_feature_set: `1d2d`
- haar_feature_select_mode: `all`
- selected_haar_feature_count: `2788`
- embedding_pca_dim: `64`
- num_feature_columns: `2864`
- feature_sanitize_info: `{'train_nan_like_before_fill': 0, 'train_nan_like_after_fill': 0, 'val_nan_like_before_fill': 0, 'val_nan_like_after_fill': 0, 'test_nan_like_before_fill': 0, 'test_nan_like_after_fill': 0}`

## Validation
- best_threshold: `0.579008`
- best_val_f1: `0.2114`

## Test
- test_f1_at_best_threshold: `0.3385`
- test_precision_abnormal: `0.2308`
- test_recall_abnormal: `0.6346`
- test_balanced_acc: `0.7678`
- test_auprc: `0.2736`
- test_auroc: `0.8268`