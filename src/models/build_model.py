import timm
from src.models.chr_cond_model import ChromosomeConditionalClassifier
from src.models.siamese_pair_model import SiamesePairClassifier
from src.models.local_pair_comparator import LocalPairComparator


def build_model(
    model_name: str,
    num_classes: int = 2,
    pretrained: bool = True,
    use_chromosome_id: bool = False,
    num_chromosome_types: int = None,
    chr_embed_dim: int = 16,
    use_pair_input: bool = False,
    pair_model_type: str = "siamese",
):
    if use_pair_input:
        if pair_model_type == "siamese":
            model = SiamesePairClassifier(
                backbone_name=model_name,
                num_classes=num_classes,
                pretrained=pretrained,
                use_chromosome_id=use_chromosome_id,
                num_chromosome_types=num_chromosome_types,
                chr_embed_dim=chr_embed_dim,
            )
            return model

        elif pair_model_type == "local":
            # 这一版 local comparator 暂时不接 chromosome id
            model = LocalPairComparator(
                backbone_name=model_name,
                num_classes=num_classes,
                pretrained=pretrained,
            )
            return model

        else:
            raise ValueError(f"Unsupported pair_model_type: {pair_model_type}")

    if use_chromosome_id:
        if num_chromosome_types is None:
            raise ValueError("num_chromosome_types must be provided when use_chromosome_id=True")

        model = ChromosomeConditionalClassifier(
            backbone_name=model_name,
            num_classes=num_classes,
            pretrained=pretrained,
            num_chromosome_types=num_chromosome_types,
            chr_embed_dim=chr_embed_dim,
        )
        return model

    model = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)
    return model