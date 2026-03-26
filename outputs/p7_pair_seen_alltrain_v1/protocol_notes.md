# P7 Pair Seen-All-Train Protocol

Definition
- derived from data_protocol/p1_pair_case_isolated_v1
- keep only abnormal pairs
- train on all train-visible pair abnormal subtypes
- evaluate only on seen abnormal subtypes in val/test
- chromosome_id must NOT be used as model input for this protocol because class identity is chromosome-specific

- num_classes = 17
- train pairs = 491
- val pairs = 51
- test pairs = 20
- train class_counts = [60, 32, 31, 24, 34, 28, 33, 26, 20, 11, 25, 21, 36, 16, 30, 30, 34]
- recommended class_weights = [0.481373, 0.902574, 0.931689, 1.203431, 0.849481, 1.031513, 0.875223, 1.11086, 1.444118, 2.625668, 1.155294, 1.37535, 0.802288, 1.805147, 0.962745, 0.962745, 0.849481]
