from os.path import join
from typing import Union

import numpy as np
import torch
import torchvision
from torch.utils.data import get_worker_info
from torchvision.transforms import v2 as TV
from datasets.utils.custom_lightning_data_module import CustomLightningDataModule

from datasets.build_frame_dataset import build_frame_dataset
from torch.utils.data import WeightedRandomSampler 


class ImageNet1kDataModule(CustomLightningDataModule):
    def __init__(
            self,
            root,
            devices,
            batch_size: int,
            img_size: int,
            train_num_workers: int,
            val_num_workers: int = 4,
            val_batch_size: int = 32,
            dataset: str = "dota"
    ) -> None:
        super().__init__(
            root=root,
            devices=devices,
            batch_size=batch_size,
            img_size=img_size,
            train_num_workers=train_num_workers,
        )
        

        self.extra_args = {
        # Model parameters
        "input_size": 224,

        # Augmentation parameters
        "num_sample": 1,
        "aa": "rand-m6-n3-mstd0.5-inc1",
        "train_interpolation": "bicubic",

        # Evaluation parameters
        "short_side_size": 224,
        "test_num_segment": 1,

        # Random Erase parameters
        "reprob": 0.25,
        "remode": "pixel",
        "recount": 1,

        # Dataset parameters
        "num_frames": 16,
        "data_path": "/media/datasets_sveta/DoTA_refined" if dataset == "dota" else "/media/datasets_sveta/DADA2000",
        "nb_classes": 2,
        "sampling_rate": 1,
        "sampling_rate_val": 1,
        "view_fps": 10,
        "data_set": "DoTA" if dataset == "dota" else "DADA2K",

        # Optimizer parameters
        "loss": "crossentropy",

        "multi_class": False,
        }

        self.val_num_workers = val_num_workers
        self.val_batch_size = val_batch_size

        self.save_hyperparameters(ignore=['_class_path', "class_path", "init_args"])

 

    def setup(self, stage: Union[str, None] = None) -> CustomLightningDataModule:

        print(f"loading {self.extra_args['data_set']} dataset from {self.extra_args['data_path']}")


        self.train_dataset, _ = build_frame_dataset(is_train=True, test_mode=False, args=self.extra_args)
        self.val_dataset, _ = build_frame_dataset(is_train=False, test_mode=False, args=self.extra_args)

        print("Train ds sizes:", len(self.train_dataset))
        print("Val ds sizes:", len(self.val_dataset))

        return self

    def train_dataloader(self):

        # introduce weighted sampling to create a balanced trainset
        count = dict()
        count[0], count[1] = self.train_dataset._label_array.count(0), self.train_dataset._label_array.count(1)
        label_weights = {label: 1.0/count for label, count in count.items()}
        sample_weights = [label_weights[label] for label in self.train_dataset._label_array]
        generator = torch.Generator()
        generator.manual_seed(42)
        sampler = WeightedRandomSampler(weights = sample_weights, num_samples=len(self.train_dataset._label_array), replacement=True, generator=generator)

        return torch.utils.data.DataLoader(
            self.train_dataset,
            drop_last=True,
            persistent_workers=self.persistent_workers,
            num_workers=self.train_num_workers,
            pin_memory=self.pin_memory,
            batch_size=self.batch_size,
            sampler = sampler
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_dataset,
            persistent_workers=False,
            num_workers=self.val_num_workers,
            pin_memory=self.pin_memory,
            batch_size=self.val_batch_size,
        )
