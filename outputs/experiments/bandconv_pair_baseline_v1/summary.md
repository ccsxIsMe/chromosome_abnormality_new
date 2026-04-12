# BandConv Pair Baseline

## Settings
- representation_version: `v2`
- profile_length: `128`
- band_width: `32`
- epochs: `40`
- batch_size: `64`
- weighted_sampler: `True`

## Validation
- best_threshold: `0.712929`
- best_val_f1: `0.1836`

## Test
- test_f1_at_best_threshold: `0.0611`
- test_precision_abnormal: `0.0560`
- test_recall_abnormal: `0.0673`
- test_balanced_acc: `0.5071`
- test_auprc: `0.0485`
- test_auroc: `0.4901`