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
from training.optim_factory import get_vit_parameter_groups
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from .visualization_utils import log_image, create_bar_chart
import matplotlib.pyplot as plt
import wandb
from PIL import Image
import io

class Classification(lightning.LightningModule):
    def __init__(
            self,
            img_size: int,
            network: nn.Module,
            lr: float = 1e-5,
            layerwise_lr_decay: float = 0.6,
            poly_lr_decay_power: float = 0.9,
            warmup_iters: int = 1500,
            lr_mode: str = "warmuplinear",
            eta_min: float = 1e-6,
            alpha: int = 6,
            bin_width: float = 1,
    ):
        super().__init__()
        self.job_id = os.environ.get('SLURM_JOB_ID')
        self.img_size = img_size
        self.lr = lr
        self.layerwise_lr_decay = layerwise_lr_decay
        self.poly_lr_decay_power = poly_lr_decay_power
        self.warmup_iters = warmup_iters
        self.lr_mode = lr_mode
        self.save_hyperparameters()
        self.eta_min = eta_min
        self.alpha = alpha
        self.bin_width = bin_width
        self.network = network
        self.val_ds_names = ["val"]
        self.metrics = nn.ModuleList(
            [
                MulticlassAccuracy(num_classes=self.network.num_classes, average="micro")
                for _ in range(len(self.val_ds_names))
            ]
        )
        self.automatic_optimization = False

        # for saving predictions and labels for calculating eval metrics and clip info for TTA (only for DADA2K)
        self.all_preds = []
        self.all_labels = []
        self.clip_infos = []
        self.sample_idx = []

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
        b_image, b_target = batch[0], batch[1]
        # b_image = (B, C, T, H, W)
        bin_logits = self.network(b_image)
        loss = F.cross_entropy(bin_logits, b_target)

        self.manual_backward(loss)
        opt.step()
        self.lr_schedulers().step()
        self.log("loss", loss, prog_bar=True)

        with torch.no_grad():
            # Compute loss for each bin separately
            for bin_id in range(0, int(self.alpha) + 1):
                mask_bin = (b_target == bin_id)
                if mask_bin.any():
                    ce_bin = F.cross_entropy(bin_logits[mask_bin], b_target[mask_bin])
                    if bin_id == 0:
                        # Log metrics
                        self.log(f"train_loss_no_col_soon", ce_bin, prog_bar=False)
                    else:
                        self.log(f"train_loss_col_<{bin_id}s", ce_bin, prog_bar=False)

        with torch.no_grad():
            if (self.global_step % 10) == 0:
                predicted_bin = torch.argmax(bin_logits.detach(), dim=1)
                accuracy = (predicted_bin == b_target)
                accuracy = accuracy.float().mean()
                self.log("accuracy", accuracy, prog_bar=False)

            if (self.global_step % 100) == 0:
                log_image(b_image[0], "train", self.trainer.logger)

    def eval_step(
            self,
            batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
            batch_idx: int,
            dataloader_idx: int,
            log_prefix: str,
    ):
        b_image, b_target, sample_idx = batch[0], batch[1], batch[2]
        self.clip_infos.extend(zip(batch[3], batch[4]))
        
        with torch.no_grad():
            bin_logits = self.network(b_image)
            self.all_preds.append(bin_logits)
            self.all_labels.append(b_target)
            self.sample_idx.append(sample_idx)

            loss = F.cross_entropy(bin_logits, b_target)
            self.log(f"{log_prefix}_loss", loss, prog_bar=True, sync_dist=True)

            with torch.no_grad():
                # Compute loss for each bin separately
                for bin_id in range(0, int(self.alpha) + 1):
                    mask_bin = (b_target == bin_id)
                    if mask_bin.any():
                        ce_bin = F.cross_entropy(bin_logits[mask_bin], b_target[mask_bin])
                        self.log(f"val_loss_no_col_soon", ce_bin, prog_bar=False) if bin_id == 0 \
                            else self.log(f"val_ce_bin_col_<{bin_id}s", ce_bin, prog_bar=False)

            if (batch_idx % 100) == 0:
                # only send first batch
                log_image(b_image[0], log_prefix, self.trainer.logger)

            b_pred_valid = bin_logits.argmax(dim=1)
            b_target_valid = b_target
            self.metrics[dataloader_idx].update(
                b_pred_valid, b_target_valid
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
        for metric_idx, metric in enumerate(self.metrics):
            acc = metric.compute()
            metric.reset()
            ds_name = self.val_ds_names[metric_idx]
            self.log(
                f"{log_prefix}_{ds_name}_acc", acc, sync_dist=True
            )
        
        if torch.distributed.is_initialized():
            torch.distributed.barrier()
        
        # Concatenate all predictions and labels
        self.all_preds = torch.cat(self.all_preds, dim=0).type(torch.float16) if self.all_preds[0].dtype == torch.int64 else torch.cat(self.all_preds, dim=0)
        self.all_labels = torch.cat(self.all_labels, dim=0)
        self.clip_infos = torch.tensor(self.clip_infos) if type(self.clip_infos[0]) == tuple else torch.cat(self.clip_infos)
        self.sample_idx = torch.cat(self.sample_idx)
        if torch.distributed.is_initialized():
            torch.distributed.barrier()
            
        # Gather data from all processes
        self.all_preds = self.all_gather(self.all_preds)
        self.all_preds = self.all_preds.reshape(-1, self.network.num_classes)
        self.all_labels = self.all_gather(self.all_labels).reshape(-1)
        self.clip_infos = self.all_gather(self.clip_infos)
        self.clip_infos = self.clip_infos.reshape(-1, 2) if self.clip_infos.ndim > 2 else self.clip_infos.reshape(-1)
        self.sample_idx = self.all_gather(self.sample_idx).reshape(-1) 
        
        if torch.distributed.is_initialized():
            torch.distributed.barrier()

        # sort preds, labels, clip_infos
        # this to preserve the order in each clip for calculating TTA and accuracy per frame distance w.r.t. start anomaly window/time of accident frame
        sorted_indices = self.sample_idx.argsort()    
        self.all_preds, self.all_labels, self.clip_infos = self.all_preds[sorted_indices], self.all_labels[sorted_indices], self.clip_infos[sorted_indices]
        self.sample_idx = self.sample_idx[sorted_indices]
        if torch.distributed.is_initialized():
            torch.distributed.barrier()

        if self.global_rank == 0:
            
            bin_accuracy_func = MulticlassAccuracy(num_classes=self.network.num_classes, average=None)
            bin_accuracy = bin_accuracy_func(self.all_preds.cpu(), self.all_labels.cpu())
            create_bar_chart(torch.roll(bin_accuracy, shifts=(-1))*100, log_prefix, self.trainer.logger, bin_width=self.bin_width)
            self.create_confusion_matrix_plot(self.all_preds.cpu(), self.all_labels.cpu(), log_prefix, ds_name)

            if self.global_step > 19000:
                with open(f"Classification_alpha_{self.alpha}_bins_of_{self.bin_width}s.pkl", "wb") as f:
                    pickle.dump({
                        "all_preds": self.all_preds.cpu(),
                        "all_labels": self.all_labels.cpu(),
                        "clip_infos": self.clip_infos.cpu(),
                        "sample_idx": self.sample_idx.cpu(),
                    }, f)

        # Final synchronization before clearing data
        if torch.distributed.is_initialized():
            torch.distributed.barrier()

        self.all_preds, self.all_labels, self.clip_infos, self.sample_idx = [], [], [], []

    def create_confusion_matrix_plot(self, preds, labels, log_prefix, ds_name):
        predicted_classes = preds.argmax(dim=1).cpu().numpy()
        true_labels = labels.cpu().numpy()

        unique_classes = list(np.unique(np.concatenate([true_labels, predicted_classes])))

        cm = confusion_matrix(true_labels, predicted_classes, labels=unique_classes)

        # reorder confusion matrix according to order of bins in labeling scheme
        new_order = [0] + unique_classes[1:][::-1]
        cm_reordered = cm[np.ix_(new_order, new_order)]

        # reorder display labels according to order of bins in labeling scheme
        reordered_labels = ["No Coll.\nSoon"] + [f"Coll.<\n{i*self.bin_width}s" for i in unique_classes[1:][::-1]]

        fig, ax = plt.subplots(figsize=(len(unique_classes)*1.5, len(unique_classes)*1.5))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm_reordered, display_labels=reordered_labels)
        disp.plot(ax=ax, cmap='Blues', values_format='d', text_kw={'fontsize': 20},colorbar=False)
        ax.set_xlabel('Predicted label', fontsize=16, weight='bold')
        ax.set_ylabel('True label', fontsize=16, weight='bold')
        ax.tick_params(axis='both', labelsize=16)
        
        plt.title(f'Confusion Matrix - {ds_name}')
        plt.tight_layout()
        
        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight", pad_inches=0.1, dpi=150)
        plt.close(fig)
        buf.seek(0)
        PIL_image = Image.open(buf).convert('RGB')
        
        self.trainer.logger.experiment.log({
            f"{log_prefix}_{ds_name}_confusion_matrix": wandb.Image(PIL_image, caption=f"Confusion Matrix - {ds_name}")
        })

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