# Legacy Projection-Split Straightened Pair Dataset

Definition
- this script follows the user-provided bend-point split-and-rotate straightening idea directly
- pair CSV structure is preserved; only `left_path` / `right_path` are rewritten
- this is additive and does not overwrite the source protocol

Settings
- output_height: `300`
- output_width: `0`
- canvas_size: `0`
- global_angle_step: `5`
- local_angle_step: `5`
- seam_trim: `3`
- white_threshold: `0.9980392156862745`
- resize_mode: `height_only`
- unique_source_images: `33978`

Split summary
- train: pairs=12305, normal=11814, abnormal=491, cases=20, chromosomes=23
- val: pairs=2359, normal=2256, abnormal=103, cases=6, chromosomes=23
- test: pairs=2325, normal=2221, abnormal=104, cases=6, chromosomes=23