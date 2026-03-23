# Baseline N0 Protocol

Subtype definition
- abnormal subtype = chromosome_id::abnormal_type
- current dataset abnormal_type is inversion only, so subtype reduces to chromosome_id::inversion

Split rules
- case-isolated
- train contains only seen abnormal subtypes
- val/test each contain both seen and unseen abnormal subtype cases

Frequency buckets
- head: >= 3 cases in the full source pool
- medium: 2 cases in the full source pool
- tail: 1 case in the full source pool
- unseen: subtype absent from train but present in val or test

Evaluation slices
- N1 detection: use all rows in each split
- N2 seen abnormal classification: evaluate abnormal rows with subtype_status=seen
- N3 unseen abnormal recognition: evaluate abnormal rows with subtype_status=unseen
