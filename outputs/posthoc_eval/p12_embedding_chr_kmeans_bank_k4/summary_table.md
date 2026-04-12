# P12 Embedding Chromosome-Conditioned KMeans Prototype Bank Summary

- distance: `cosine`
- num_prototypes_per_chr: `4`
- topk_prototypes_for_score: `1`
- global val-best threshold: `0.001522`
- chr-conditioned best quantile from val: `0.9900`

| Setting | Test F1 | Test Precision_abn | Test Recall_abn | Test Balanced Acc |
|---|---:|---:|---:|---:|
| Prototype bank global val-best | 0.4000 | 0.3333 | 0.5000 | 0.7266 |
| Prototype bank chr-conditioned | 0.3966 | 0.2811 | 0.6731 | 0.7962 |