import os
from logging import info
from typing import Optional, Tuple

import lightning
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import PolynomialLR
from torchmetrics.classification import MulticlassAccuracy
import torchvision.transforms as transforms
from torchvision.transforms import functional
import numpy as np
import pickle

from models.utils.warmup_and_linear_scheduler import WarmupAndLinearScheduler

# from metrics import calculate_tta, calculate_metrics_multi_class, accuracy_per_frame
from training.optim_factory import LayerDecayValueAssigner, get_parameter_groups
from new_metrics import metrics, prediction_lead_time
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

import matplotlib.pyplot as plt
import wandb
from PIL import Image
import io

class Regression(lightning.LightningModule):
    def __init__(
            self,
            batch_size: int,
            img_size: int,
            network: nn.Module,
            lr: float = 1e-5,
            lr_multiplier: float = 0.01,
            layerwise_lr_decay: float = 0.6,
            poly_lr_decay_power: float = 0.9,
            warmup_iters: int = 1500,
            weight_decay: float = 0.05,
            ignore_index: int = 255,
            lr_mode: str = "warmuplinear",
    ):
        super().__init__()
        self.job_id = os.environ.get('SLURM_JOB_ID')
        self.batch_size = batch_size
        self.img_size = img_size
        self.lr = lr
        self.lr_multiplier = lr_multiplier
        self.layerwise_lr_decay = layerwise_lr_decay
        self.poly_lr_decay_power = poly_lr_decay_power
        self.weight_decay = weight_decay
        self.ignore_index = ignore_index
        self.warmup_iters = warmup_iters
        self.lr_mode = lr_mode
        self.save_hyperparameters()

        self.network = network

        # self.label2name = get_label2name()
        self.val_ds_names = ["val"]

        self.automatic_optimization = False

        # data augmentation on GPU
        self.blur_kernel_size = int(
            np.floor(
                np.ceil(0.1 * self.img_size) - 0.5 +
                np.ceil(0.1 * self.img_size) % 2
            )
        )
        self.augmentations = transforms.Compose([
            transforms.RandomApply([
                transforms.ColorJitter(
                    brightness=0.3,
                    contrast=0.3,
                    saturation=0.3,
                    hue=0.3)], p=0.5),
            transforms.RandomApply([
                transforms.GaussianBlur(kernel_size=self.blur_kernel_size, sigma=(0.15, 1.15))], p=0.5),
        ])

        self.norm = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

        # for saving predictions and labels for calculating eval metrics and clip info for TTA (only for DADA2K)
        self.all_preds = []
        self.all_labels = []
        self.clip_infos = []
        self.sample_idx = []
        self.all_log_variances = []

    def get_optimizers(self):
        opt = self.optimizers()
        opt.zero_grad()
        return opt

    def training_step(
            self,
            batch: Tuple[torch.Tensor, torch.Tensor],
            batch_idx: int,
    ):
        opt = self.get_optimizers()
        b_image, b_target = batch[0], batch[-2]
        # b_image = (B, C, T, H, W)
        mean_ttc, log_variance = self.network(b_image)
        mean_ttc = mean_ttc.squeeze(-1)
        log_variance = log_variance.squeeze(-1)

        # if self.global_step < 7000:
        #     s = log_variance.clamp(min=-2.0, max=0.0)
        # else:
        #     s = log_variance.clamp(min=-4.0, max=2.0)

        s = log_variance

        # ttc_mask = b_target < 5.0
        # error = torch.where(ttc_mask, b_target - mean_ttc, F.relu(5.0 - mean_ttc))
        error = b_target - mean_ttc
        loss_source = torch.mean((error**2 / (2 * (s.exp() + 1e-8))) + 0.5 * s)

        self.manual_backward(loss_source)
        opt.step()
        self.lr_schedulers().step()
        self.log("loss", loss_source, prog_bar=True)

        with torch.no_grad():
            if (self.global_step % 10) == 0:
                mae = torch.mean(torch.abs(mean_ttc.cpu() - b_target.cpu()))
                self.log(f"mae", mae, prog_bar=True)

            if (self.global_step % 100) == 0:
                self._log_img(b_image[0], "train")

    def eval_step(
            self,
            batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
            batch_idx: int,
            dataloader_idx: int,
            log_prefix: str,
    ):
        b_image, b_target, sample_idx = batch[0], batch[-2], batch[2]
        self.clip_infos.extend(zip(batch[3], batch[4]))
        
        with torch.no_grad():
            mean_ttc, log_variance = self.network(b_image)
            mean_ttc = mean_ttc.squeeze(-1)
            log_variance = log_variance.squeeze(-1)

            s = log_variance#.clamp(min=-6.0, max=6.0)
            
            self.all_preds.append(mean_ttc)
            self.all_labels.append(b_target) 
            self.sample_idx.append(sample_idx)
            self.all_log_variances.append(s)

            # ttc_mask = b_target < 5.0
            # error = torch.where(ttc_mask, b_target - mean_ttc, F.relu(5.0 - mean_ttc))
            loss_source = torch.mean(((b_target - mean_ttc)**2 / (2 * (s.exp() + 1e-8))) + 0.5 * s)

            self.log(f"{log_prefix}_loss", loss_source, prog_bar=True, sync_dist=True)

            if (batch_idx % 100) == 0:
                self._log_img(b_image[0], log_prefix) 

            mae = torch.mean(torch.abs(mean_ttc.cpu() - b_target.cpu()))
            self.log(f"{log_prefix}_mae", mae, prog_bar=True)

    @torch.no_grad()
    def _log_img(
            self,
            sourceds_image,
            log_prefix: str
    ):
        # sourceds_image = (C, T, H, W)
        # take first image 
        sourceds_image = sourceds_image[:,0]  # (C, H, W)
        inverse_normalize = transforms.Normalize(
            mean=[-(0.485/0.229), -(0.456/0.224), -(0.406/0.225)],
            std=[1/0.229, 1/0.224, 1/0.225]
        )

        sourceds_image = inverse_normalize(sourceds_image).cpu().permute(1, 2, 0).float().numpy()  # (H, W, C)
        fig, axes = plt.subplots(
            1, 1, figsize=(10,
                           10)
        )
        axes.imshow(sourceds_image)
        axes.axis("off")

        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(
            buf, format="png", bbox_inches="tight", pad_inches=0, facecolor="black"
        )
        plt.close(fig)
        buf.seek(0)
        PIL_image = Image.open(buf).convert('RGB')
        self.trainer.logger.experiment.log(  # type: ignore
            {
                f"{log_prefix}": [
                    wandb.Image(PIL_image, file_type="jpg")
                ]
            }
        )       

    def validation_step(
            self,
            batch: Tuple[torch.Tensor, torch.Tensor],
            batch_idx: int,
            dataloader_idx: int = 0,
    ):
        return self.eval_step(batch, batch_idx, dataloader_idx, "val")

    def test_step(
            self,
            batch: Tuple[torch.Tensor, torch.Tensor],
            batch_idx: int,
            dataloader_idx: int = 0,
    ):
        return self.eval_step(batch, batch_idx, dataloader_idx, "test")

    def _on_eval_epoch_end(self, log_prefix):
        
        
        if torch.distributed.is_initialized():
            torch.distributed.barrier()
        
        # Concatenate all predictions and labels
        self.all_preds = torch.cat(self.all_preds, dim=0).type(torch.float16) if self.all_preds[0].dtype == torch.int64 else torch.cat(self.all_preds, dim=0)
        self.all_labels = torch.cat(self.all_labels, dim=0)
        self.clip_infos = torch.tensor(self.clip_infos) if type(self.clip_infos[0]) == tuple else torch.cat(self.clip_infos)
        self.sample_idx = torch.cat(self.sample_idx, dim=0)
        self.all_log_variances = torch.cat(self.all_log_variances, dim=0)

        if torch.distributed.is_initialized():
            torch.distributed.barrier()
            
        # Gather data from all processes
        self.all_preds = self.all_gather(self.all_preds)
        self.all_preds = self.all_preds.reshape(-1)
        self.all_labels = self.all_gather(self.all_labels).reshape(-1)
        self.clip_infos = self.all_gather(self.clip_infos)
        self.clip_infos = self.clip_infos.reshape(-1, 2) if self.clip_infos.ndim > 2 else self.clip_infos.reshape(-1, *self.clip_infos.shape[2:])
        self.sample_idx = self.all_gather(self.sample_idx).reshape(-1) 
        self.all_log_variances = self.all_gather(self.all_log_variances).reshape(-1)
        
        if torch.distributed.is_initialized():
            torch.distributed.barrier()

        # sort preds and labels such their order for each clip is preserved for visualization
        sorted_indices = self.sample_idx.argsort()    
        self.all_preds, self.all_labels, self.clip_infos = self.all_preds[sorted_indices], self.all_labels[sorted_indices], self.clip_infos[sorted_indices]
        self.all_log_variances = self.all_log_variances[sorted_indices]

        
        if torch.distributed.is_initialized():
            torch.distributed.barrier()

        if self.global_rank == 0:
            
            # save all data from last validation for visualization
            if self.global_step > 19000:
                with open("data_regre_only_region_A_no_var_clipping.pkl", "wb") as f:
                    pickle.dump({
                        "all_preds": self.all_preds.cpu(),
                        "all_labels": self.all_labels.cpu(),
                        "clip_infos": self.clip_infos.cpu(),
                        "sample_idx": self.sample_idx.cpu(),
                        "all_log_variances": self.all_log_variances.cpu(),
                    }, f)

        if torch.distributed.is_initialized():
            torch.distributed.barrier()

        self.all_preds, self.all_labels, self.clip_infos, self.sample_idx, self.all_log_variances = [], [], [], [], []

    def on_validation_epoch_end(self):
        self._on_eval_epoch_end("val")

    def on_test_epoch_end(self):
        self._on_eval_epoch_end("test")

    def on_train_batch_end(self, outputs, batch, batch_idx):
        for i, param_group in enumerate(self.trainer.optimizers[0].param_groups):
            self.log(f"learning_rate/group_{i}", param_group["lr"], on_step=True)

    def configure_optimizers(self):
        
        # # apply layerwise decay
        assigner = LayerDecayValueAssigner(list(self.layerwise_lr_decay ** (self.network.model.get_num_layers() + 1 - i) for i in range(self.network.model.get_num_layers() + 2)))
        optim_weights = get_parameter_groups(self.network.model, weight_decay=self.weight_decay,
            get_num_layer=assigner.get_layer_id, get_layer_scale=assigner.get_scale, lr = self.lr)
        
        print(f"lr: {self.lr}")

        optimizer = torch.optim.AdamW(optim_weights, betas=(0.9, 0.999))
        print(f"lr_mode: {self.lr_mode}")
        if self.lr_mode == "poly":
            lr_scheduler = {
                "scheduler": PolynomialLR(
                    optimizer,
                    int(self.trainer.estimated_stepping_batches),
                    self.poly_lr_decay_power,
                ),
                "interval": "step",
            }
        elif self.lr_mode == "warmuplinear":
            lr_scheduler = {
                "scheduler": WarmupAndLinearScheduler(
                    optimizer,
                    start_warmup_lr=1e-5,
                    warmup_iters=self.warmup_iters,
                    base_lr=1,
                    final_lr=1e-3,
                    total_iters=self.trainer.max_steps,
                ),
                "interval": "step",
            }
        elif self.lr_mode == "cosine_annealing":
            lr_scheduler = {
                "scheduler": torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer,
                    T_max=self.trainer.max_steps,
                    eta_min=1e-6,
                ),
                "interval": "step",
            }
        else:
            raise Exception("Wrong lr_more: {}".format(self.lr_mode))

        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}