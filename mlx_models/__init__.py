from .vit import ViT


def create_model(model_tag: str, *, pretrained: bool = False):
    if model_tag.startswith("vit"):
        model = ViT.from_timm(model_tag, pretrained=pretrained)
    else:
        raise ValueError(f"Unsupported {model_tag=}")
    return model
