# P12 Embedding Chromosome-Conditioned GMM Bank Summary

- num_components_per_chr: `1`
- covariance_type: `diag`
- reg_covar: `1e-05`
- global val-best threshold: `-1111.240421`
- chr-conditioned best quantile from val: `0.9900`

| Setting | Test F1 | Test Precision_abn | Test Recall_abn | Test Balanced Acc |
|---|---:|---:|---:|---:|
| GMM bank global val-best | 0.3857 | 0.3068 | 0.5192 | 0.7322 |
| GMM bank chr-conditioned | 0.4047 | 0.2911 | 0.6635 | 0.7939 |