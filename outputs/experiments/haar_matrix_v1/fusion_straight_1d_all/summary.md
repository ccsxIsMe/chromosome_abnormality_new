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
- haar_feature_set: `1d`
- haar_feature_select_mode: `all`
- selected_haar_feature_count: `757`
- embedding_pca_dim: `64`
- num_feature_columns: `833`
- feature_sanitize_info: `{'train_nan_like_before_fill': 0, 'train_nan_like_after_fill': 0, 'val_nan_like_before_fill': 0, 'val_nan_like_after_fill': 0, 'test_nan_like_before_fill': 0, 'test_nan_like_after_fill': 0}`

## Validation
- best_threshold: `0.604480`
- best_val_f1: `0.1718`

## Test
- test_f1_at_best_threshold: `0.3456`
- test_precision_abnormal: `0.2450`
- test_recall_abnormal: `0.5865`
- test_balanced_acc: `0.7509`
- test_auprc: `0.2383`
- test_auroc: `0.8240`