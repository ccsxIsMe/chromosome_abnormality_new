# Gray-Zone Reranker Summary

- base_score_mode: `chr_tail_zscore`
- tail_quantile: `0.9`
- gray_zone_ratio: `0.1`
- gray_zone_margin: `31.010287`
- base val-best threshold: `32.041647`
- rerank threshold from val: `0.457286`

| Setting | Test F1 | Precision_abn | Recall_abn | Balanced Acc | FP | FN |
|---|---:|---:|---:|---:|---:|---:|
| Base | 0.6321 | 0.6854 | 0.5865 | 0.7870 | 28 | 43 |
| Final reranked | 0.5572 | 0.5773 | 0.5385 | 0.7600 | 41 | 48 |