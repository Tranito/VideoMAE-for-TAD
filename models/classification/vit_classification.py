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

        self.model = VisionTransformer(patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), num_classes=self.num_classes, use_flash_attn=True, drop_path_rate=0.2)#, fc_drop_rate=0.2)

        # load pre-trained VideoMAE weights
        url = "https://huggingface.co/OpenGVLab/VideoMAE2/resolve/main/distill/vit_s_k710_dl_from_giant.pth"
        ckpt = torch.hub.load_state_dict_from_url(url)['module']

        filtered_weights = {k:v for k, v in ckpt.items() if k in self.model.state_dict() and v.shape == self.model.state_dict()[k].shape}   

        filtered_weights["head.weight"] = self.model.head.weight
        filtered_weights["head.bias"] = self.model.head.bias

        # uncomment when using second training strategy
        # filtered_weights["fc_norm.weight"] = self.model.fc_norm.weight
        # filtered_weights["fc_norm.bias"] = self.model.fc_norm.bias
        # filtered_weights["linear.weight"] = self.model.linear.weight
        # filtered_weights["linear.bias"] = self.model.linear.bias

        # path = "/media/vmae_s_dota_lr1e3_b56x1_dsampl1val2_ld06_aam6n3/checkpoint-16.pth"
        # ckpt = torch.load(path, map_location="cpu")

        #Only set strict to False for token masking
        self.model.load_state_dict(filtered_weights, strict=True)

    def forward(self, clip):
        b, c, n, h, w = clip.shape
        assert h == w

        logit = self.model(clip)
        return logit
