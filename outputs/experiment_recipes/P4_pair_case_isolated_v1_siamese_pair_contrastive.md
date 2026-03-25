# P4 Pair Case-Isolated v1 Siamese Pair Contrastive

## Goal

Add a classic Siamese contrastive loss on homologous-pair distance while keeping the main pair abnormality classifier.

This is a task-aligned pair loss, not a generic regularizer.

## Literature grounding

- Bromley et al., 1993: Siamese networks for pair similarity learning
- Hadsell, Chopra, LeCun, 2006: contrastive margin objective for similar / dissimilar pairs

## Configs

- `configs/p4_pair_case_isolated_v1_siamese_resnet18_chrid_pair_contrastive.yaml`
- `configs/p4_pair_case_isolated_v1_siamese_resnet50_chrid_pair_contrastive.yaml`

## Run

```bash
python -m src.main --config configs/p4_pair_case_isolated_v1_siamese_resnet18_chrid_pair_contrastive.yaml
python -m src.main --config configs/p4_pair_case_isolated_v1_siamese_resnet50_chrid_pair_contrastive.yaml
```

## Primary comparison

Compare against `P1` on:

- `test_metrics_05.auprc`
- `test_metrics_best.f1`
- `test_metrics_best.abnormal.recall`

## Interpretation

If this improves over `P1`, the gain comes from explicitly shaping homologous-pair distance, not just from pair-input classification.
