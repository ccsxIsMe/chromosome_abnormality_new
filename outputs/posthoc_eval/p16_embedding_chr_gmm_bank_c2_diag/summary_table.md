# Embedding Chromosome-Conditioned GMM Bank Summary (p16_pair_normal_only_v1_correspondence_interval_convnext_tiny_multi_prototype_metric)

- num_components_per_chr: `2`
- covariance_type: `diag`
- reg_covar: `1e-05`
- global val-best threshold: `-1226.816634`
- chr-conditioned best quantile from val: `0.9900`

| Setting | Test F1 | Test Precision_abn | Test Recall_abn | Test Balanced Acc |
|---|---:|---:|---:|---:|
| GMM bank global val-best | 0.5619 | 0.5566 | 0.5673 | 0.7731 |
| GMM bank chr-conditioned | 0.5097 | 0.4258 | 0.6346 | 0.7973 |