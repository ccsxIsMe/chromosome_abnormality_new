# P12 Embedding Chromosome-Conditioned KMeans Prototype Bank Summary

- distance: `cosine`
- num_prototypes_per_chr: `8`
- topk_prototypes_for_score: `3`
- global val-best threshold: `0.001553`
- chr-conditioned best quantile from val: `0.9900`

| Setting | Test F1 | Test Precision_abn | Test Recall_abn | Test Balanced Acc |
|---|---:|---:|---:|---:|
| Prototype bank global val-best | 0.3985 | 0.3312 | 0.5000 | 0.7264 |
| Prototype bank chr-conditioned | 0.4035 | 0.2899 | 0.6635 | 0.7937 |