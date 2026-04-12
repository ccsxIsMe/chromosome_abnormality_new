# P12 Embedding Chromosome-Conditioned GMM Bank Summary

- num_components_per_chr: `2`
- covariance_type: `diag`
- reg_covar: `1e-05`
- global val-best threshold: `-1093.748947`
- chr-conditioned best quantile from val: `0.9500`

| Setting | Test F1 | Test Precision_abn | Test Recall_abn | Test Balanced Acc |
|---|---:|---:|---:|---:|
| GMM bank global val-best | 0.4157 | 0.3510 | 0.5096 | 0.7327 |
| GMM bank chr-conditioned | 0.3506 | 0.2287 | 0.7500 | 0.8158 |