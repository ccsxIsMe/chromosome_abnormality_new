# P12 Embedding Chromosome-Conditioned GMM Bank Summary

- num_components_per_chr: `8`
- covariance_type: `diag`
- reg_covar: `1e-05`
- global val-best threshold: `-1131.081928`
- chr-conditioned best quantile from val: `0.9750`

| Setting | Test F1 | Test Precision_abn | Test Recall_abn | Test Balanced Acc |
|---|---:|---:|---:|---:|
| GMM bank global val-best | 0.3943 | 0.3143 | 0.5288 | 0.7374 |
| GMM bank chr-conditioned | 0.3532 | 0.2319 | 0.7404 | 0.8128 |