# Gray-Zone Reranker Summary

- base_score_mode: `chr_tail_zscore`
- tail_quantile: `0.95`
- gray_zone_ratio: `0.3`
- gray_zone_margin: `3.911421`
- base val-best threshold: `4.614487`
- rerank threshold from val: `0.859296`

| Setting | Test F1 | Precision_abn | Recall_abn | Balanced Acc | FP | FN |
|---|---:|---:|---:|---:|---:|---:|
| Base | 0.6070 | 0.6289 | 0.5865 | 0.7852 | 36 | 43 |
| Final reranked | 0.5429 | 0.5377 | 0.5481 | 0.7630 | 49 | 47 |