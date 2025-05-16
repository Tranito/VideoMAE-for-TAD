import timm
import torch.nn as nn

def get_encoder(name: str, img_size: int, patch_size: int):
    encoder = timm.create_model(
        name,
        pretrained=True,
        img_size=img_size,
        patch_size=patch_size,
    )

    if hasattr(encoder, "fc_norm"):
        encoder.fc_norm = nn.Identity()
    if hasattr(encoder, "neck"):
        encoder.neck = nn.Identity()
    if hasattr(encoder, "head"):
        encoder.head = nn.Identity()
    return encoder