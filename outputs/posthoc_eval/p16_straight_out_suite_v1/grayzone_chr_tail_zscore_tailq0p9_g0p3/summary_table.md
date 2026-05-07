# Gray-Zone Reranker Summary

- base_score_mode: `chr_tail_zscore`
- tail_quantile: `0.9`
- gray_zone_ratio: `0.3`
- gray_zone_margin: `31.343549`
- base val-best threshold: `32.041647`
- rerank threshold from val: `0.834171`

| Setting | Test F1 | Precision_abn | Recall_abn | Balanced Acc | FP | FN |
|---|---:|---:|---:|---:|---:|---:|
| Base | 0.6321 | 0.6854 | 0.5865 | 0.7870 | 28 | 43 |
| Final reranked | 0.5659 | 0.5743 | 0.5577 | 0.7692 | 43 | 46 |