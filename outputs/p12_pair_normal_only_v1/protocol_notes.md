# P12 Pair Normal-Only Protocol

Definition
- derived from data_protocol/p1_pair_case_isolated_v1
- train keeps only normal homologous pairs
- val/test stay unchanged and still contain both normal and abnormal pairs

Goal
- learn chromosome-conditional normal pair manifold only
- detect abnormalities as deviation from normal pair structure

- train normal pairs = 11814
- val total pairs = 2359
- test total pairs = 2325