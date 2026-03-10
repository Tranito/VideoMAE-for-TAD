from typing import Optional

import torch
import torch.nn as nn
from models.hybrid.modeling_finetune import VisionTransformer
from functools import partial
from timm.models.layers import drop_path, to_2tuple, trunc_normal_



class VITHybrid(nn.Module):
    def __init__(
            self,
            img_size: int,
            patch_size: int = 14,
            all_frames: int = 16,
            uncertainty_pred: bool = False,
            model_ckpt: Optional[str] = None,
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.uncertainty_pred = uncertainty_pred
        self.embed_dim = 384 # for ViT Small
        self.model = VisionTransformer(patch_size=16, 
                                       embed_dim=self.embed_dim, 
                                       depth=12,  
                                       num_heads=6,
                                       mlp_ratio=4,
                                       qkv_bias=True, 
                                       norm_layer=partial(nn.LayerNorm, eps=1e-6), 
                                       use_flash_attn=True,
                                       drop_path_rate=0.3,
                                       all_frames=all_frames, 
                                       img_size=img_size,  
                                       uncertainty_pred=self.uncertainty_pred)

        self.model_ckpt = model_ckpt
        self._load_pretrained_weights()

    def _load_pretrained_weights(self):
        # check if path to a custom pretrained model is provided, if not, load pre-traihed weights from Hugging Face
        if self.model_ckpt is None:
            url = "https://huggingface.co/OpenGVLab/VideoMAE2/resolve/main/distill/vit_s_k710_dl_from_giant.pth"
            ckpt = torch.hub.load_state_dict_from_url(url)['module']
            filtered_weights = {
                k: v for k, v in ckpt.items() 
                if k in self.model.state_dict() 
                and v.shape == self.model.state_dict()[k].shape 
                and not k.startswith('head')
            }
            self.model.load_state_dict(filtered_weights, strict=False)
        else:
            weights = torch.load(self.model_ckpt)["state_dict"]
            filtered_weights = {k.lstrip("network.model."):v for k, v in weights.items()}
            self.model.load_state_dict(filtered_weights, strict=True)

    def forward(self, clip):
        b, c, n, h, w = clip.shape
        assert h == w
        predictions = self.model(clip)
        return predictions