# P12 Embedding Chromosome-Conditioned GMM Bank Summary

- num_components_per_chr: `4`
- covariance_type: `diag`
- reg_covar: `1e-05`
- global val-best threshold: `-1145.245414`
- chr-conditioned best quantile from val: `0.9900`

| Setting | Test F1 | Test Precision_abn | Test Recall_abn | Test Balanced Acc |
|---|---:|---:|---:|---:|
| GMM bank global val-best | 0.3958 | 0.3098 | 0.5481 | 0.7454 |
| GMM bank chr-conditioned | 0.3807 | 0.2639 | 0.6827 | 0.7968 |