# Pair CSV Image-Root Remap

Definition
- preserve the original train/val/test split, labels, and metadata
- only rewrite `left_path` and `right_path` from the old image root to the new image root
- intended for switching an existing protocol to a new image version such as straightened chromosomes

Settings
- mode: `splits_case_layout`
- source_root: ``
- target_root: `/data5/chensx/MyProject/UAE/data/straightened_out/straightened`
- case_dir_column: `case_dir`
- strict_exists: `True`

Split summary
- train: pairs=12305, normal=11814, abnormal=491, cases=20, chromosomes=23
- val: pairs=2359, normal=2256, abnormal=103, cases=6, chromosomes=23
- test: pairs=2325, normal=2221, abnormal=104, cases=6, chromosomes=23