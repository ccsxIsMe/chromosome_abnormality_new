import timm

from src.models.chr_cond_model import ChromosomeConditionalClassifier
from src.models.local_global_pair_comparator import LocalGlobalPairComparator
from src.models.local_pair_comparator import LocalPairComparator
from src.models.multi_prototype_metric import MultiPrototypeMetricModel
from src.models.siamese_pair_model import SiamesePairClassifier


def build_model(
    model_name: str,
    num_classes: int = 2,
    pretrained: bool = True,
    use_chromosome_id: bool = False,
    num_chromosome_types: int = None,
    chr_embed_dim: int = 16,
    use_pair_input: bool = False,
    pair_model_type: str = "siamese",
    use_pair_mixstyle: bool = False,
    mixstyle_p: float = 0.5,
    mixstyle_alpha: float = 0.1,
    experiment_mode: str = "classifier",
    num_prototypes: int = 4,
    prototype_distance: str = "cosine",
    normalize_prototype_embedding: bool = True,
):
    if use_pair_input:
        if pair_model_type == "siamese":
            base_model = SiamesePairClassifier(
                backbone_name=model_name,
                num_classes=num_classes,
                pretrained=pretrained,
                use_chromosome_id=use_chromosome_id,
                num_chromosome_types=num_chromosome_types,
                chr_embed_dim=chr_embed_dim,
            )

        elif pair_model_type == "local":
            base_model = LocalPairComparator(
                backbone_name=model_name,
                num_classes=num_classes,
                pretrained=pretrained,
            )

        elif pair_model_type == "local_global":
            base_model = LocalGlobalPairComparator(
                backbone_name=model_name,
                num_classes=num_classes,
                pretrained=pretrained,
                use_chromosome_id=use_chromosome_id,
                num_chromosome_types=num_chromosome_types,
                chr_embed_dim=chr_embed_dim,
                use_pair_mixstyle=use_pair_mixstyle,
                mixstyle_p=mixstyle_p,
                mixstyle_alpha=mixstyle_alpha,
            )

        else:
            raise ValueError(f"Unsupported pair_model_type: {pair_model_type}")

        if experiment_mode == "multi_prototype_metric":
            if not use_chromosome_id:
                raise ValueError("multi_prototype_metric mode requires use_chromosome_id=True")

            if not hasattr(base_model, "embedding_dim"):
                raise ValueError(
                    "Base model must expose `embedding_dim` for multi_prototype_metric mode"
                )

            return MultiPrototypeMetricModel(
                base_model=base_model,
                num_chromosome_types=num_chromosome_types,
                embedding_dim=base_model.embedding_dim,
                num_prototypes=num_prototypes,
                distance=prototype_distance,
                normalize_embedding=normalize_prototype_embedding,
            )

        return base_model

    if use_chromosome_id:
        if num_chromosome_types is None:
            raise ValueError("num_chromosome_types must be provided when use_chromosome_id=True")

        return ChromosomeConditionalClassifier(
            backbone_name=model_name,
            num_classes=num_classes,
            pretrained=pretrained,
            num_chromosome_types=num_chromosome_types,
            chr_embed_dim=chr_embed_dim,
        )

    return timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)