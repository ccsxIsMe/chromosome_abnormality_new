# P12 Embedding Chromosome-Conditioned KMeans Prototype Bank Summary

- distance: `cosine`
- num_prototypes_per_chr: `16`
- topk_prototypes_for_score: `1`
- global val-best threshold: `0.001506`
- chr-conditioned best quantile from val: `0.9900`

| Setting | Test F1 | Test Precision_abn | Test Recall_abn | Test Balanced Acc |
|---|---:|---:|---:|---:|
| Prototype bank global val-best | 0.4160 | 0.3562 | 0.5000 | 0.7288 |
| Prototype bank chr-conditioned | 0.3692 | 0.2517 | 0.6923 | 0.7980 |