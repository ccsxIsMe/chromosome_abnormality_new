# Gray-Zone Reranker Summary

- base_score_mode: `chr_tail_zscore`
- tail_quantile: `0.95`
- gray_zone_ratio: `0.2`
- gray_zone_margin: `33.384143`
- base val-best threshold: `34.187704`
- rerank threshold from val: `0.708543`

| Setting | Test F1 | Precision_abn | Recall_abn | Balanced Acc | FP | FN |
|---|---:|---:|---:|---:|---:|---:|
| Base | 0.6256 | 0.6703 | 0.5865 | 0.7865 | 30 | 43 |
| Final reranked | 0.5498 | 0.5421 | 0.5577 | 0.7678 | 49 | 46 |