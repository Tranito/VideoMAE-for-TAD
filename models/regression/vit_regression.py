import torch
import torch.nn as nn
from models.utils.token_masking import TokenMasking
from models.regression.modeling_finetune import VisionTransformer
from functools import partial
from timm.models.layers import drop_path, to_2tuple, trunc_normal_



class VITRegression(nn.Module):
    def __init__(
            self,
            img_size: int,
            model_name: str,
            num_classes: int,
            patch_size: int = 14,
            all_frames: int = 16,
            multi_class: bool = False
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_classes = num_classes
        self.embed_dim = 384

        self.model = VisionTransformer(patch_size=16, 
                                       embed_dim=self.embed_dim, 
                                       depth=12, 
                                       num_heads=6, 
                                       mlp_ratio=4, 
                                       qkv_bias=True,
                                       norm_layer=partial(nn.LayerNorm, eps=1e-6), 
                                       num_classes=self.num_classes, 
                                       use_flash_attn=True, drop_path_rate=0.3, 
                                       all_frames=all_frames, 
                                       img_size=img_size)#, fc_drop_rate=0.2)

        self.regression_head = nn.Sequential(nn.Linear(self.embed_dim, 256),
                                   nn.GELU(),
                                   nn.Linear(256, 1))
    
        self.variance_head = nn.Sequential(nn.Linear(self.embed_dim, 128),
                                   nn.GELU(),
                                   nn.Linear(128, 1))
        self.classification_head = nn.Linear(self.embed_dim, 2)

         # Load pre-trained VideoMAE weights
        self._load_pretrained_weights()

        # Initialize heads with proper strategy
        self._init_heads()

    def _init_heads(self):
        """Initialize heads with proper weights"""
        heads = [self.regression_head, self.variance_head, self.classification_head]
        
        for head in heads:
            head.apply(self._init_head_weights)
            
    def _init_head_weights(self, m):
        """Custom initialization for heads - copied and modified from modeling_finetune.py"""
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def _load_pretrained_weights(self):
        """Load pre-trained VideoMAE weights"""
        url = "https://huggingface.co/OpenGVLab/VideoMAE2/resolve/main/distill/vit_s_k710_dl_from_giant.pth"
        ckpt = torch.hub.load_state_dict_from_url(url)['module']
        filtered_weights = {
            k: v for k, v in ckpt.items() 
            if k in self.model.state_dict() 
            and v.shape == self.model.state_dict()[k].shape 
            and not k.startswith('head')
        }
        self.model.load_state_dict(filtered_weights, strict=False)

       
    def forward(self, clip):
        b, c, n, h, w = clip.shape
        assert h == w

        features = self.model(clip)
        mean_ttc = self.regression_head(features)      
        log_variance = self.variance_head(features)    
        classification_logits = self.classification_head(features)

        return mean_ttc, log_variance, classification_logits
