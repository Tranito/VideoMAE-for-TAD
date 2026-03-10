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

import torch
from torch.utils.data import Sampler
from collections import defaultdict
import numpy as np
import math
    
class DashcamAccidentDataModule(CustomLightningDataModule):
    """ Lightning DataModule for dashcam accident datasets with sliding window sampling
        
        Builds datasets, creates dataloaders with distributed sampling, and optionally
        applies bin-balanced sampling for imbalanced TTC distributions.

        Args:
            root (str): the folder containing the dataset
            devices (int): specifies which GPUs used for training
            batch_size (int): batch size for training
            img_size (int): the size of the input image
            train_num_workers (int): number of workers for the training dataloader
            val_num_workers (int): number of workers for the validation dataloader
            val_batch_size (int): batch size for validation
            dataset (str): which dataset to use, currently supports "dada" (DADA-2000)
            num_frames (int): number of frames in one sliding window
            window_stride (int): controls number of video frames between start of consecutive sliding windows during training
            window_stride_val (int): controls number of video frames between start of consecutive sliding windows during validation/testing
            sliding_window_fps (int): target fps of the sliding window
            regression (bool): whether to indicate TTC labels should be provided
            bin_balancing (bool): whether to use bin balanced batches during training
            alpha (int): the TTC threshold for separating No Collision Soon and Collision Soon region
            bin_width (float): the width of each bin in seconds
        
        Example:
        num_frames=16, sliding_window_fps=10 → 1.6 second clips at 10 FPS
        window_stride=10 at 30 FPS source → new clip every 0.33 seconds
    """
    def __init__(
            self,
            root,
            devices,
            batch_size: int,
            img_size: int,
            train_num_workers: int,
            val_num_workers: int = 4,
            val_batch_size: int = 32,
            dataset: str = "dada",
            num_frames: int = 16,
            window_stride: int = 1,
            window_stride_val: int = 1,
            sliding_window_fps: int = 10,
            regression: bool = False,
            balancing: bool = False,
            alpha: int = 6,
            bin_width: float = 1,
    ) -> None:
        super().__init__(
            root=root,
            devices=devices,
            batch_size=batch_size,
            img_size=img_size,
            train_num_workers=train_num_workers,
        )
        self.extra_args = {
        # frame parameters
        "input_size": img_size,

        # augmentation parameters
        "num_sample": 1,
        "aa": "rand-m6-n3-mstd0.5-inc1",
        "train_interpolation": "bicubic",
        "reprob": 0.25,
        "remode": "pixel",
        "recount": 1,

        # dataset parameters
        "num_frames": num_frames,
        "window_stride": window_stride,
        "window_stride_val": window_stride_val,
        "sliding_window_fps": sliding_window_fps,
        "data_set": "DADA2K",
        "regression": regression,
        "alpha": alpha,
        "bin_width": bin_width,
        }

        if dataset == "dada":
            self.extra_args["data_path"] = self.root + "/DADA2000"
 
        self.val_num_workers = val_num_workers
        self.val_batch_size = val_batch_size
        self.balancing = balancing

        self.save_hyperparameters(ignore=['_class_path', "class_path", "init_args"])
 
    def setup(self, stage: Union[str, None] = None) -> CustomLightningDataModule:
        self.train_dataset = build_frame_dataset(is_train=True, test_mode=False, args=self.extra_args)
        self.val_dataset = build_frame_dataset(is_train=False, test_mode=False, args=self.extra_args)
        return self

    def train_dataloader(self):
        if self.balancing:
            # compute sample weights inversely proportional to class frequency for the WeightedRandomSampler to create bin-balanced batches
            unique_labels, count_labels = np.unique(np.array(self.train_dataset._label_array), return_counts=True)
            count = dict(zip(unique_labels, count_labels))
            label_weights = {label: 1.0/count for label, count in count.items()}
            sample_weights = [label_weights[label] for label in self.train_dataset._label_array]
            
            # Fixed seed for reproducibility
            torch.manual_seed(42)
            sampler = WeightedRandomSampler(
                weights=sample_weights,
                num_samples=len(self.train_dataset),
                replacement=True
            )

            print(f"first 10 sampled indices in train_dataloader: {list(sampler)[:10]}")  # Debug print

            balanced_dataset = Subset(self.train_dataset, indices=list(sampler))
        
            return torch.utils.data.DataLoader(
                balanced_dataset,
                drop_last=True,
                persistent_workers=True,
                num_workers=self.train_num_workers,
                pin_memory=False,
                batch_size=self.batch_size,
                shuffle=True
            )
        else:
            return torch.utils.data.DataLoader(
                self.train_dataset,
                drop_last=True,
                persistent_workers=True,
                num_workers=self.train_num_workers,
                pin_memory=False,
                batch_size=self.batch_size,
                shuffle=True
            )

    def val_dataloader(self):
        sampler = DistributedSampler(self.val_dataset, shuffle=False, drop_last=True)
        return torch.utils.data.DataLoader(
            self.val_dataset,
            persistent_workers=False,
            num_workers=self.val_num_workers,
            pin_memory=False,
            batch_size=self.val_batch_size,
            sampler=sampler,
        )