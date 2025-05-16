from os.path import join
from typing import Union

import numpy as np
import torch
import torchvision
from torch.utils.data import get_worker_info
from torchvision.transforms import v2 as TV
from datasets.loader.in1k_zip_dataset import IN1KZipDataset
from datasets.utils.custom_lightning_data_module import CustomLightningDataModule

from datasets.data_load import build_frame_dataset


class ImageNet1kDataModule(CustomLightningDataModule):
    def __init__(
            self,
            root,
            devices,
            batch_size: int,
            img_size: int,
            train_num_workers: int,
            val_num_workers: int = 6,
            val_batch_size: int = 2,
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
        "data_path": "/media/datasets_sveta/DoTA_refined",
        "nb_classes": 2,
        "sampling_rate": 1,
        "sampling_rate_val": 2,
        "view_fps": 10,
        "data_set": "DoTA",

        # Optimizer parameters
        "loss": "crossentropy",
        }

        self.val_num_workers = val_num_workers
        self.val_batch_size = val_batch_size

        self.save_hyperparameters(ignore=['_class_path', "class_path", "init_args"])

        self.brightness_delta = 32 / 255.
        self.contrast_delta = 0.5
        self.saturation_delta = 0.5
        self.hue_delta = 18 / 360.
        self.color_jitter_probability = 0.8

        blur_kernel_size = int(
            np.floor(
                np.ceil(0.1 * self.img_size) - 0.5 +
                np.ceil(0.1 * self.img_size) % 2
            )
        )
        #

        self.random_aug_weak = TV.Compose([
            TV.RandomApply([
                TV.ColorJitter(
                    brightness=self.brightness_delta,
                    contrast=self.contrast_delta,
                    saturation=self.saturation_delta,
                    hue=self.hue_delta)], p=0.9),
        ])
        self.random_aug_strong = TV.Compose([
            TV.RandomApply([TV.ColorJitter(
                brightness=0.8, contrast=0.8,
                saturation=0.8, hue=0.4)], p=0.8),
            TV.RandomGrayscale(p=0.1),
            TV.RandomApply([
                TV.GaussianBlur(kernel_size=blur_kernel_size, sigma=(0.15, 3.0))], p=0.5),
        ])

    def setup(self, stage: Union[str, None] = None) -> CustomLightningDataModule:
        train_transform = torchvision.transforms.Compose([
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.Resize(self.img_size),
            torchvision.transforms.RandomResizedCrop(self.img_size),
            torchvision.transforms.ToTensor(),
            self.random_aug_weak,
        ])
        val_transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize(self.img_size),
            torchvision.transforms.CenterCrop(self.img_size),
            torchvision.transforms.ToTensor(),
        ])

        # self.train_dataset = torchvision.datasets.ImageNet(self.root, split='train', transform=train_transform)
        # self.val_dataset = torchvision.datasets.ImageNet(self.root, split='val', transform=val_transform)

        # CHANGE THESE TWO LINES
        # self.train_dataset = IN1KZipDataset(join(self.root, "in1k_train.zip"), transform=train_transform)
        # self.val_dataset = IN1KZipDataset(join(self.root, "in1k_val.zip"), transform=val_transform)
        # print(f"self.extra_args: {self.extra_args}")

        self.train_dataset, _ = build_frame_dataset(is_train=True, test_mode=False, args=self.extra_args)
        self.val_dataset, _ = build_frame_dataset(is_train=False, test_mode=False, args=self.extra_args)

        print("Train ds sizes:", len(self.train_dataset))
        print("Val ds sizes:", len(self.val_dataset))

        return self

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset,
            shuffle=True,
            drop_last=True,
            persistent_workers=self.persistent_workers,
            num_workers=self.train_num_workers,
            pin_memory=self.pin_memory,
            batch_size=self.batch_size,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_dataset,
            persistent_workers=False,
            num_workers=self.val_num_workers,
            pin_memory=self.pin_memory,
            batch_size=self.val_batch_size,
        )
