# N2 Seen-Only Protocol

Source
- derived from data_protocol/nonpair_case_isolated_v1
- keep only abnormal rows whose subtype belongs to protocol_role = seen_eval

Classes
- num_classes = 6
- class_names = 1::inversion, 2::inversion, 4::inversion, 7::inversion, 13::inversion, 19::inversion

Split summary
- train: 161 images, 7 cases, 6 classes present
- val: 51 images, 4 cases, 4 classes present
- test: 20 images, 3 cases, 3 classes present

- recommended class_weights = [0.447222, 2.439394, 1.677083, 0.894444, 1.118056, 1.341667]
