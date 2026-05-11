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
- haar_feature_select_mode: `topk`
- selected_haar_feature_count: `128`
- embedding_pca_dim: `64`
- num_feature_columns: `204`
- feature_sanitize_info: `{'train_nan_like_before_fill': 0, 'train_nan_like_after_fill': 0, 'val_nan_like_before_fill': 0, 'val_nan_like_after_fill': 0, 'test_nan_like_before_fill': 0, 'test_nan_like_after_fill': 0}`

## Validation
- best_threshold: `0.605716`
- best_val_f1: `0.1858`

## Test
- test_f1_at_best_threshold: `0.3305`
- test_precision_abnormal: `0.2348`
- test_recall_abnormal: `0.5577`
- test_balanced_acc: `0.7363`
- test_auprc: `0.2280`
- test_auroc: `0.8137`