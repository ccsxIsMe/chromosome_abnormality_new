# Gray-Zone Reranker Summary

- base_score_mode: `chr_percentile`
- tail_quantile: `0.95`
- gray_zone_ratio: `0.1`
- gray_zone_margin: `0.048658`
- base val-best threshold: `0.989950`
- rerank threshold from val: `0.854084`

| Setting | Test F1 | Precision_abn | Recall_abn | Balanced Acc | FP | FN |
|---|---:|---:|---:|---:|---:|---:|
| Base | 0.5203 | 0.4507 | 0.6154 | 0.7901 | 78 | 40 |
| Final reranked | 0.0000 | 0.0000 | 0.0000 | 0.4980 | 9 | 104 |