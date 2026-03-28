# P12 Posthoc Threshold Sweep Summary

## Threshold Sweep

| Rank | Score | Threshold Rule | Threshold | Precision | Recall | F1 | Balanced Acc | FP | TP |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | raw | q99 | 0.002040 | 0.4400 | 0.5288 | 0.4803 | 0.7487 | 70 | 55 |
| 2 | raw | q97.5 | 0.001939 | 0.3314 | 0.5385 | 0.4103 | 0.7438 | 113 | 56 |
| 3 | raw | mean+3std | 0.001937 | 0.3294 | 0.5385 | 0.4088 | 0.7436 | 114 | 56 |
| 4 | raw | mean+2.5std | 0.001708 | 0.2445 | 0.5385 | 0.3363 | 0.7303 | 173 | 56 |
| 5 | raw | q95 | 0.001687 | 0.2373 | 0.5385 | 0.3294 | 0.7287 | 180 | 56 |
| 6 | raw | mean+2std | 0.001479 | 0.1927 | 0.5577 | 0.2864 | 0.7241 | 243 | 58 |
| 7 | casewise zscore | mean+2std | 1.995600 | 0.4848 | 0.1538 | 0.2336 | 0.5731 | 17 | 16 |
| 8 | casewise zscore | q95 | 2.469897 | 0.5000 | 0.1346 | 0.2121 | 0.5642 | 14 | 14 |
| 9 | casewise zscore | mean+2.5std | 2.494500 | 0.5000 | 0.1346 | 0.2121 | 0.5642 | 14 | 14 |
| 10 | casewise zscore | q97.5 | 2.927728 | 0.4800 | 0.1154 | 0.1860 | 0.5548 | 13 | 12 |
| 11 | casewise zscore | q99 | 3.238383 | 0.4800 | 0.1154 | 0.1860 | 0.5548 | 13 | 12 |
| 12 | casewise zscore | mean+3std | 2.993399 | 0.4800 | 0.1154 | 0.1860 | 0.5548 | 13 | 12 |

## Main Takeaways

- Best deploy threshold under current protocol: `raw score + train-normal q99`
- Raw anomaly score is consistently better than case-wise zscore for hard decision thresholding
- Case-wise normalization is still useful for ranking analysis, but not for final binary decision on this dataset
- Patient-level chromosome aggregation was checked quickly and did not beat the sample-level `raw q99` result
