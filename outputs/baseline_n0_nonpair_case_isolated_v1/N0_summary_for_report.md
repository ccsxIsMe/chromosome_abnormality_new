# Baseline N0 Summary

## 1. Protocol Definition

Current non-pair Baseline N0 uses the case-isolated split under `data_protocol/nonpair_case_isolated_v1/`.

Abnormal subtype is defined as:

`abnormal_subtype_id = chromosome_id::abnormal_type`

In the current dataset, all abnormal samples are labeled as `inversion`, so the effective subtype granularity is:

`chromosome_id::inversion`

Examples:

- `1::inversion`
- `7::inversion`
- `X::inversion`

Evaluation slices:

- `N1` detection: all samples in each split
- `N2` seen abnormal subtype classification: abnormal samples with `subtype_status = seen`
- `N3` unseen abnormal subtype recognition: abnormal samples with `subtype_status = unseen`

## 2. Split Statistics

| Split | Total images | Normal images | Abnormal images | Total cases | Abnormal cases | Seen abnormal subtypes | Unseen abnormal subtypes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Train | 25,162 | 24,615 | 547 | 20 | 20 | 19 | 0 |
| Val | 4,738 | 4,635 | 103 | 6 | 6 | 4 | 2 |
| Test | 4,784 | 4,680 | 104 | 6 | 6 | 3 | 3 |

Overall abnormal image ratio:

- Train: `547 / 25162 = 2.17%`
- Val: `103 / 4738 = 2.17%`
- Test: `104 / 4784 = 2.17%`

## 3. Abnormal Subtype Frequency Buckets

Frequency buckets are defined by total abnormal case count in the full source pool:

- `head`: at least 3 cases
- `medium`: 2 cases
- `tail`: 1 case

| Frequency bucket | # subtypes | # abnormal cases | # abnormal images |
| --- | ---: | ---: | ---: |
| Head | 2 | 6 | 112 |
| Medium | 4 | 8 | 120 |
| Tail | 18 | 18 | 522 |

This dataset is strongly long-tailed: `18 / 24 = 75%` of subtypes are tail classes.

## 4. Protocol Roles of Abnormal Subtypes

| Protocol role | Meaning | # subtypes | # abnormal cases | # abnormal images |
| --- | --- | ---: | ---: | ---: |
| `train_only` | appears only in train | 13 | 13 | 386 |
| `seen_eval` | appears in train and also in val/test | 6 | 14 | 232 |
| `unseen` | absent from train, appears in val/test | 5 | 5 | 136 |

## 5. Full 24-Class Overview

The current source pool contains the complete set of 24 inversion subtypes:

`1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, X, Y`

Their protocol placement is:

| Subtype | Role | Frequency bucket | Train cases | Val cases | Test cases |
| --- | --- | --- | ---: | ---: | ---: |
| `1::inversion` | seen_eval | head | 2 | 1 | 0 |
| `2::inversion` | seen_eval | head | 1 | 1 | 1 |
| `3::inversion` | train_only | tail | 1 | 0 | 0 |
| `4::inversion` | seen_eval | medium | 1 | 1 | 0 |
| `5::inversion` | train_only | tail | 1 | 0 | 0 |
| `6::inversion` | unseen | tail | 0 | 0 | 1 |
| `7::inversion` | seen_eval | medium | 1 | 0 | 1 |
| `8::inversion` | train_only | tail | 1 | 0 | 0 |
| `9::inversion` | unseen | tail | 0 | 0 | 1 |
| `10::inversion` | train_only | tail | 1 | 0 | 0 |
| `11::inversion` | unseen | tail | 0 | 1 | 0 |
| `12::inversion` | train_only | tail | 1 | 0 | 0 |
| `13::inversion` | seen_eval | medium | 1 | 1 | 0 |
| `14::inversion` | train_only | tail | 1 | 0 | 0 |
| `15::inversion` | unseen | tail | 0 | 1 | 0 |
| `16::inversion` | train_only | tail | 1 | 0 | 0 |
| `17::inversion` | train_only | tail | 1 | 0 | 0 |
| `18::inversion` | train_only | tail | 1 | 0 | 0 |
| `19::inversion` | seen_eval | medium | 1 | 0 | 1 |
| `20::inversion` | unseen | tail | 0 | 0 | 1 |
| `21::inversion` | train_only | tail | 1 | 0 | 0 |
| `22::inversion` | train_only | tail | 1 | 0 | 0 |
| `X::inversion` | train_only | tail | 1 | 0 | 0 |
| `Y::inversion` | train_only | tail | 1 | 0 | 0 |

## 6. Seen-Eval Subtypes

These are the abnormal subtypes usable for `N2` seen abnormal subtype classification.

| Subtype | Frequency bucket | Train cases | Val cases | Test cases |
| --- | --- | ---: | ---: | ---: |
| `1::inversion` | head | 2 | 1 | 0 |
| `2::inversion` | head | 1 | 1 | 1 |
| `4::inversion` | medium | 1 | 1 | 0 |
| `7::inversion` | medium | 1 | 0 | 1 |
| `13::inversion` | medium | 1 | 1 | 0 |
| `19::inversion` | medium | 1 | 0 | 1 |

Seen-eval abnormal slices:

| Slice | # abnormal cases | # abnormal images |
| --- | ---: | ---: |
| Val seen | 4 | 51 |
| Test seen | 3 | 20 |

## 7. Unseen Subtypes

These are the abnormal subtypes reserved for `N3` unseen recognition.

| Subtype | Frequency bucket | Val cases | Test cases |
| --- | --- | ---: | ---: |
| `11::inversion` | tail | 1 | 0 |
| `15::inversion` | tail | 1 | 0 |
| `6::inversion` | tail | 0 | 1 |
| `9::inversion` | tail | 0 | 1 |
| `20::inversion` | tail | 0 | 1 |

Unseen abnormal slices:

| Slice | # abnormal cases | # abnormal images |
| --- | ---: | ---: |
| Val unseen | 2 | 52 |
| Test unseen | 3 | 84 |

## 8. Recommended Reporting for the Next Baselines

For `N1` detection:

| Evaluation set | Content |
| --- | --- |
| Val-all | all val images |
| Test-all | all test images |

For `N2` seen subtype classification:

| Evaluation set | Content |
| --- | --- |
| Val-seen | abnormal val images with `subtype_status = seen` |
| Test-seen | abnormal test images with `subtype_status = seen` |

For `N3` unseen recognition:

| Evaluation set | Content |
| --- | --- |
| Val-unseen | abnormal val images with `subtype_status = unseen` |
| Test-unseen | abnormal test images with `subtype_status = unseen` |

## 9. Limitation

This N0 protocol supports chromosome-level inversion subtype analysis, not breakpoint-level subtype analysis. In other words, it currently evaluates whether the model can generalize across unseen `chromosome_id::inversion` categories, but it does not yet support finer-grained subtypes such as `inv(1)(p13q21)`.
