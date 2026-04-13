# Gray-Zone Reranker Summary

- base_score_mode: `chr_tail_zscore`
- tail_quantile: `0.95`
- gray_zone_ratio: `0.2`
- gray_zone_margin: `3.811214`
- base val-best threshold: `4.614487`
- rerank threshold from val: `0.572864`

| Setting | Test F1 | Precision_abn | Recall_abn | Balanced Acc | FP | FN |
|---|---:|---:|---:|---:|---:|---:|
| Base | 0.6070 | 0.6289 | 0.5865 | 0.7852 | 36 | 43 |
| Final reranked | 0.4813 | 0.4234 | 0.5577 | 0.7611 | 79 | 46 |