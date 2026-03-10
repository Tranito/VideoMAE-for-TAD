from typing import Optional

import torch
import torch.nn as nn
from models.classification.modeling_finetune import VisionTransformer
from functools import partial


class VITClassification(nn.Module):
    def __init__(
            self,
            img_size: int,
            patch_size: int = 14,
            all_frames: int = 16,
            alpha: int = 6,
            bin_width: float = 1.0,
            model_ckpt: Optional[str] = None,
            **kwargs
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_classes = int(alpha/bin_width) + 1

        self.model = VisionTransformer(patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), num_classes=self.num_classes, use_flash_attn=True, drop_path_rate=0.3, all_frames=all_frames, img_size=img_size, **kwargs)

        if model_ckpt is None:
            print("model ckpt is None!")
            # load pre-trained VideoMAE weights
            url = "https://huggingface.co/OpenGVLab/VideoMAE2/resolve/main/distill/vit_s_k710_dl_from_giant.pth"
            ckpt = torch.hub.load_state_dict_from_url(url)['module']

            filtered_weights = {k:v for k, v in ckpt.items() if k in self.model.state_dict() and v.shape == self.model.state_dict()[k].shape}   

            filtered_weights["head.weight"] = self.model.head.weight
            filtered_weights["head.bias"] = self.model.head.bias
        else:
            weights = torch.load(model_ckpt)["state_dict"]
            filtered_weights = {k.lstrip("network.model."):v for k, v in weights.items()}
            self.model.load_state_dict(filtered_weights, strict=True)

        self.model.load_state_dict(filtered_weights, strict=True)

    def forward(self, clip):
        b, c, n, h, w = clip.shape
        assert h == w

        logit = self.model(clip)
        return logit
