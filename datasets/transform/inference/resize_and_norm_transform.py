from typing import Optional

import numpy as np
import torch
from PIL import Image
from torchvision import transforms as T
from torchvision.transforms import functional as F


class ResizeAndNormTransform:
    def __init__(
            self,
            img_size: (int, int),
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
            label_mapping: Optional[dict] = None,
    ):

        self.img_size = img_size
        self.mean = mean
        self.std = std

        if label_mapping is not None:
            self.label_mapping = torch.full((256,), 255, dtype=torch.int64)
            for from_id, to_id in label_mapping.items():
                if 0 <= from_id <= 255:
                    self.label_mapping[from_id] = to_id

    def __call__(self, image: Image.Image, target=None):
        scale = self.img_size / min(image.size)
        size = [round(image.size[1] * scale), round(image.size[0] * scale)]

        image = F.resize(image, size)
        target = F.resize(target, size, T.InterpolationMode.NEAREST)

        image_tensor = F.convert_image_dtype(F.pil_to_tensor(image), dtype=torch.float)
        image_tensor = F.normalize(image_tensor, mean=self.mean, std=self.std)

        target_tensor = torch.as_tensor(np.array(target), dtype=torch.int64)

        if hasattr(self, "label_mapping"):
            target_tensor = self.label_mapping[target_tensor]
        return image_tensor, target_tensor