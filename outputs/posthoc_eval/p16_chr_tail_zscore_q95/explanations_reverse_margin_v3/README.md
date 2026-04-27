# Pair Explanation Visualization

- config: `configs/p16_pair_normal_only_v1_correspondence_interval_convnext_tiny_multi_prototype_metric.yaml`
- ckpt: `outputs/experiments/p16_pair_normal_only_v1_correspondence_interval_convnext_tiny_multi_prototype_metric/best_model.pth`
- predictions_csv: `outputs/posthoc_eval/p16_chr_tail_zscore_q95/test_predictions_chr_calibrated.csv`
- pred_column: `pred_label_global_valbest`
- score_column: `calibrated_score`
- groups: `tp,fp,fn,tn`
- num_per_group: `4`
- target_mode: `reverse_margin`
- fallback_target_mode: `pair_distance`
- min_grad_norm: `1e-10`
- cam_mode: `band`

Saved files:
- manifest: `outputs/posthoc_eval/p16_chr_tail_zscore_q95/explanations_reverse_margin_v3/manifest.csv`
- example panels: `outputs/posthoc_eval/p16_chr_tail_zscore_q95/explanations_reverse_margin_v3`