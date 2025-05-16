import torch
import torch.nn as nn

from models.utils.token_masking import TokenMasking
from models.classification.modeling_finetune import VisionTransformer
from functools import partial


class VITClassification(nn.Module):
    def __init__(
            self,
            img_size: int,
            model_name: str,
            num_classes: int,
            patch_size: int = 14,
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_classes = num_classes

        self.model = VisionTransformer(patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), num_classes=self.num_classes)


    def forward(self, clip):
        b, c, n, h, w = clip.shape
        assert h == w

        logit = self.model(clip)
        return logit
