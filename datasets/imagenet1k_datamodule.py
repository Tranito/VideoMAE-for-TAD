from os.path import join
from typing import Union

import numpy as np
import torch
import torchvision
from torch.utils.data import get_worker_info
from torchvision.transforms import v2 as TV
from datasets.utils.custom_lightning_data_module import CustomLightningDataModule

from datasets.build_frame_dataset import build_frame_dataset
from torch.utils.data import WeightedRandomSampler, DistributedSampler, Subset


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
            dataset: str = "dota",
            num_frames: int = 16,
            sampling_rate: int = 1,
            sampling_rate_val: int = 1,
            view_fps: int = 10,
            regression: bool = False,
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
        "input_size": img_size,

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
        "num_frames": num_frames,
        "data_path": self.root+"/DoTA_refined" if dataset == "dota" else self.root+"/DADA2000",
        "sampling_rate": sampling_rate,
        "sampling_rate_val": sampling_rate_val,
        "view_fps": view_fps,
        "data_set": "DoTA" if dataset == "dota" else "DADA2K",
        "regression": regression,

        # Optimizer parameters
        "loss": "crossentropy",
        }

        self.val_num_workers = val_num_workers
        self.val_batch_size = val_batch_size

        self.save_hyperparameters(ignore=['_class_path', "class_path", "init_args"])

        

 

    def setup(self, stage: Union[str, None] = None) -> CustomLightningDataModule:

        print(f"loading {self.extra_args['data_set']} dataset from {self.extra_args['data_path']}")
        print(f"num_frames per window: {self.extra_args['num_frames']}, sampling_rate: {self.extra_args['sampling_rate']}, sampling_rate_val: {self.extra_args['sampling_rate_val']}, target_fps: {self.extra_args['view_fps']}")

        self.train_dataset, _ = build_frame_dataset(is_train=True, test_mode=False, args=self.extra_args)
        self.val_dataset, _ = build_frame_dataset(is_train=False, test_mode=False, args=self.extra_args)

        print("Train ds sizes:", len(self.train_dataset))
        print("Val ds sizes:", len(self.val_dataset))

        return self

    def train_dataloader(self):

        # introduce weighted sampling to create a balanced trainset
        
        unique_labels, count_labels = np.unique(np.array(self.train_dataset._new_binary_label), return_counts=True)
        count = dict(zip(unique_labels, count_labels))
        label_weights = {label: 1.0/count for label, count in count.items()}
        sample_weights = [label_weights[label] for label in self.train_dataset._new_binary_label]

        # Custom dataset that generates new balanced indices each epoch
        class DynamicallyBalancedDataset(torch.utils.data.Dataset):
            def __init__(self, base_dataset, weights):
                self.base_dataset = base_dataset
                self.weights = torch.tensor(weights, dtype=torch.double)
                self.indices = self._generate_indices()
                
            def _generate_indices(self):
                sampler = WeightedRandomSampler(
                    self.weights, 
                    num_samples=len(self.base_dataset), 
                    replacement=True
                )
                return list(sampler)
                
            def __len__(self):
                return len(self.indices)
                
            def __getitem__(self, idx):
                actual_idx = self.indices[idx]
                return self.base_dataset[actual_idx]
                
            def set_epoch(self, epoch):
                # Generate NEW balanced indices each epoch
                torch.manual_seed(epoch + 42)  # Ensure reproducibility but different each epoch
                self.indices = self._generate_indices()
        
        balanced_dataset = DynamicallyBalancedDataset(self.train_dataset, sample_weights)
        sampler = DistributedSampler(balanced_dataset, shuffle=True, drop_last=True)
        
        return torch.utils.data.DataLoader(
            balanced_dataset,
            drop_last=True,
            persistent_workers=True,
            num_workers=self.train_num_workers,
            pin_memory=False,
            batch_size=self.batch_size,
            sampler=sampler
        )

    def val_dataloader(self):
        sampler = DistributedSampler(self.val_dataset, shuffle=False, drop_last=True)
        return torch.utils.data.DataLoader(
            self.val_dataset,
            persistent_workers=False,
            num_workers=self.val_num_workers,
            pin_memory=False,
            batch_size=self.val_batch_size,
            sampler=sampler
        )
