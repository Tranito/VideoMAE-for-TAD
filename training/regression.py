import os
from logging import info
from typing import Optional, Tuple
import lightning
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import PolynomialLR
import torchvision.transforms as transforms
import numpy as np
import pickle
from models.utils.warmup_and_linear_scheduler import WarmupAndLinearScheduler
from training.optim_factory import get_vit_parameter_groups
from training.new_metrics import accuracy_per_bin_regression
from .visualization_utils import log_image, create_bar_chart
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
            eta_min: float = 1e-6,
            alpha: int = 6,
            bin_width: float = 1.0,
            uncertainty_pred: bool = False,
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
        self.eta_min = eta_min
        self.alpha = alpha
        self.uncertainty_pred = uncertainty_pred
        self.bin_width = bin_width
        self.network = network

        # self.label2name = get_label2name()
        self.val_ds_names = ["val"]

        self.automatic_optimization = False

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
    
    def loss(self, predictions, targets, alpha, *args):
        """
        Computes loss for Collision Soon and No Collision Soon region separately.
        If uncertainty_pred is True, it computes the Negative Log Likelihood loss for both regions. Otherwise, it computes the MSE loss.

        Args:
            predictions: Tensor of shape (B,) containing the predicted TTC values.
            targets: Tensor of shape (B,) containing the ground truth TTC values.
            alpha: float, the threshold for Collision Soon vs No Collision Soon.
            args: if uncertainty_pred is True, args[0] should be the log_variance.

        Returns:
            loss_A: loss for Collision Soon region
            loss_B: loss for No Collision Soon region
        """
        mask_coll_soon = (targets <= alpha)
        mask_no_coll_soon = ~mask_coll_soon
        loss_A, loss_B = 0.0, 0.0

        if mask_coll_soon.any():
            error_A = targets[mask_coll_soon] - predictions[mask_coll_soon]
            if self.uncertainty_pred:
                log_variance = args[0]
                var = log_variance.exp() + 1e-8
                loss_A = (1/2 * ( (error_A ** 2) / (var[mask_coll_soon]) + log_variance[mask_coll_soon] )).mean()
            else:
                loss_A = (error_A**2).mean()

        if mask_no_coll_soon.any():
            error_B = F.relu(alpha - predictions[mask_no_coll_soon])
            if self.uncertainty_pred:
                log_variance = args[0]
                var = log_variance.exp() + 1e-8
                loss_B = (1/2 * ( (error_B ** 2) / (var[mask_no_coll_soon]) + log_variance[mask_no_coll_soon] )).mean()
            else:
                loss_B = (error_B**2).mean()

        return loss_A, loss_B

    def compute_bin_loss(self, ttc_pred, ttc_label, bin_labels, log_prefix, *args):
        """ Computes loss and TTC error for each "Collision Soon bin and logs them to wandb"""
        with torch.no_grad():
            for bin_id in range(1, int(self.alpha/self.bin_width) + 1):
                mask_bin = (bin_labels == bin_id)

                if mask_bin.any():
                    # Regression error for this bin
                    error_bin = (ttc_label[mask_bin] - ttc_pred[mask_bin]).abs()
                    mae_bin = error_bin.mean()
                    if self.uncertainty_pred:
                        log_variance = args[0].squeeze(-1)
                        var = log_variance.exp() + 1e-8
                        loss = (1/2* (  (error_bin ** 2) / (var[mask_bin]) + log_variance[mask_bin]  )).mean()
                    else:
                        loss = (error_bin**2).mean()
                    
                    self.log(f"{log_prefix}_loss_coll_<{bin_id*self.bin_width}s", loss, prog_bar=False)
                    self.log(f"{log_prefix}_mean_TTC_error_(absolute)_coll_<{bin_id*self.bin_width}s", mae_bin, prog_bar=False)

                    if self.uncertainty_pred:
                        self.log(f"{log_prefix}_log_var_coll_<{bin_id*self.bin_width}s", log_variance[mask_bin].mean(), prog_bar=False)
            
    def training_step(
            self,
            batch: Tuple[torch.Tensor, torch.Tensor],
            batch_idx: int,
    ):
        opt = self.get_optimizers()
        b_image, ttc_target, bin_labels = batch[0], batch[-1], batch[1]
        # b_image = (B, C, T, H, W)

        alpha = self.alpha

        mask_A = (ttc_target <= alpha)
        mask_B = ~mask_A

        if self.uncertainty_pred:
            predictions = self.network(b_image)
            ttc_pred, log_variance = predictions[0], predictions[1]
            ttc_pred = ttc_pred.squeeze(-1)
            log_variance = log_variance.squeeze(-1)
       
            loss_A, loss_B = self.loss(ttc_pred, ttc_target, self.alpha, log_variance)
            self.compute_bin_loss(ttc_pred, ttc_target, bin_labels, "train", log_variance)
            
        else:
            ttc_pred = self.network(b_image)
            ttc_pred = ttc_pred.squeeze(-1)       

            loss_A, loss_B = self.loss(ttc_pred, ttc_target, self.alpha)
            self.compute_bin_loss(ttc_pred, ttc_target, bin_labels, "train")
       
        loss_source = loss_A + loss_B

        self.manual_backward(loss_source)
        opt.step()
        self.lr_schedulers().step()
        self.log("loss", loss_source, prog_bar=True)

        with torch.no_grad():
            if (self.global_step % 10) == 0:
                mae = torch.mean(torch.abs(ttc_pred.cpu() - ttc_target.cpu()))
                self.log(f"mae", mae, prog_bar=True)

            if (self.global_step % 100) == 0:
                log_image(b_image[0], "train", self.trainer.logger)

    def eval_step(
            self,
            batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
            batch_idx: int,
            dataloader_idx: int,
            log_prefix: str,
    ):
        b_image, ttc_target, sample_idx, bin_labels = batch[0], batch[-1], batch[2], batch[1]
        self.clip_infos.extend(zip(batch[3], batch[4]))
        
        with torch.no_grad():
            alpha = self.alpha
            mask_A = (ttc_target <= alpha)
            mask_B = ~mask_A

            if self.uncertainty_pred:
                predictions = self.network(b_image)
                ttc_pred, log_variance = predictions[0], predictions[1]
                ttc_pred = ttc_pred.squeeze(-1)
                log_variance = log_variance.squeeze(-1)
                self.all_log_variances.append(log_variance)
        
                loss_A, loss_B = self.loss(ttc_pred, ttc_target, self.alpha, log_variance)
                self.compute_bin_loss(ttc_pred, ttc_target, bin_labels, "train", log_variance)
                
            else:
                ttc_pred = self.network(b_image)
                ttc_pred = ttc_pred.squeeze(-1)       

                loss_A, loss_B = self.loss(ttc_pred, ttc_target, self.alpha)
                self.compute_bin_loss(ttc_pred, ttc_target, bin_labels, "train")
            loss_source = loss_A + loss_B

            self.all_preds.append(ttc_pred)
            self.all_labels.append(ttc_target) 
            self.sample_idx.append(sample_idx)

            self.log(f"{log_prefix}_loss", loss_source, prog_bar=True, sync_dist=True)
            self.log(f"{log_prefix}_loss_col_soon", loss_A, sync_dist=True)
            self.log(f"{log_prefix}_loss_no_col_soon", loss_B, sync_dist=True)
            self.log(f"{log_prefix}_error_col_soon", (ttc_pred[mask_A] - ttc_target[mask_A]).abs().mean() if mask_A.any() else torch.tensor(0.,device=ttc_target.device), sync_dist=True)
            self.log(f"{log_prefix}_error_no_col_soon", (ttc_pred[mask_B] - ttc_target[mask_B]).abs().mean() if mask_B.any() else torch.tensor(0.,device=ttc_target.device), sync_dist=True)

            if self.uncertainty_pred:
                self.log(f"{log_prefix}_log_var_col_soon", log_variance[mask_A].mean() if mask_A.any() else torch.tensor(0.,device=ttc_target.device),  sync_dist=True)
                self.log(f"{log_prefix}_log_var_no_col_soon", log_variance[mask_B].mean() if mask_B.any() else torch.tensor(0.,device=ttc_target.device), prog_bar=True, sync_dist=True)

            if (batch_idx % 100) == 0:
                log_image(b_image[0], log_prefix, self.trainer.logger)

            mae = torch.mean(torch.abs(ttc_pred.cpu() - ttc_target.cpu()))
            self.log(f"{log_prefix}_mae", mae, prog_bar=True)


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
        
        self.all_preds = torch.cat(self.all_preds, dim=0).type(torch.float16) if self.all_preds[0].dtype == torch.int64 else torch.cat(self.all_preds, dim=0)
        self.all_labels = torch.cat(self.all_labels, dim=0)
        self.clip_infos = torch.tensor(self.clip_infos) 
        self.sample_idx = torch.cat(self.sample_idx, dim=0)

        if self.uncertainty_pred:
            self.all_log_variances = torch.cat(self.all_log_variances, dim=0)

        if torch.distributed.is_initialized():
            torch.distributed.barrier()
            
        # gather data from all processes (GPUs)
        self.all_preds = self.all_gather(self.all_preds)
        self.all_preds = self.all_preds.reshape(-1)
        self.all_labels = self.all_gather(self.all_labels).reshape(-1)
        self.clip_infos = self.all_gather(self.clip_infos).reshape(-1, 2)
        self.sample_idx = self.all_gather(self.sample_idx).reshape(-1)

        if self.uncertainty_pred: 
            self.all_log_variances = self.all_gather(self.all_log_variances).reshape(-1)
        
        if torch.distributed.is_initialized():
            torch.distributed.barrier()

        # sort preds and labels such their order for each video is preserved for visualization
        sorted_indices = self.sample_idx.argsort()    
        self.all_preds, self.all_labels, self.clip_infos = self.all_preds[sorted_indices], self.all_labels[sorted_indices], self.clip_infos[sorted_indices]
        self.sample_idx = self.sample_idx[sorted_indices]
        
        if self.uncertainty_pred:
            self.all_log_variances = self.all_log_variances[sorted_indices]

        if torch.distributed.is_initialized():
            torch.distributed.barrier()

        if self.global_rank == 0:
            mean_accuracy_per_bin = accuracy_per_bin_regression(self.all_preds, self.clip_infos, alpha=self.alpha, bin_width=self.bin_width, fps=30)
            create_bar_chart(torch.tensor(list(mean_accuracy_per_bin.values()))*100, log_prefix, self.trainer.logger, bin_width=self.bin_width)

            # save all data from last validation for visualization
            if self.global_step > 19000:

                data_to_save = {
                    "all_preds": self.all_preds.cpu(),
                    "all_labels": self.all_labels.cpu(),
                    "clip_infos": self.clip_infos.cpu(),
                    "sample_idx": self.sample_idx.cpu(),
                }

                if self.uncertainty_pred:
                    file_name = f"Regression_w_uncertainty_alpha{self.alpha}_binwidth{self.bin_width}.pkl"
                    data_to_save["all_log_variances"] = self.all_log_variances.cpu()
                else:
                    file_name = f"Regression_alpha{self.alpha}_binwidth{self.bin_width}.pkl"
                
                with open(file_name, "wb") as f:
                    pickle.dump(data_to_save, f)

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
        
        optim_weights = get_vit_parameter_groups(
        model=self.network.model,
        base_lr=self.lr,
        layer_decay=self.layerwise_lr_decay,
        head_lr_mult=1,
        verbose=False
        )
        
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
                    eta_min=self.eta_min,
                ),
                "interval": "step",
            }
        else:
            raise Exception("Wrong lr_more: {}".format(self.lr_mode))

        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}