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

def focal_loss_multiclass(inputs, targets, alpha=1, gamma=2):
    """
    Multi-class focal loss implementation
    - inputs: raw logits from the model
    - targets: true class labels (as integer indices, not one-hot encoded)
    """
    # Convert logits to log probabilities
    log_prob = F.log_softmax(inputs, dim=-1)
    prob = torch.exp(log_prob)  # Calculate probabilities from log probabilities

    # Gather the probabilities corresponding to the correct classes
    targets_one_hot = F.one_hot(targets, num_classes=inputs.shape[-1])
    pt = torch.sum(prob * targets_one_hot, dim=-1)

    # Apply focal adjustment
    focal_loss = -alpha * (1 - pt) ** gamma * torch.sum(log_prob * targets_one_hot, dim=-1)
    
    return focal_loss.mean()

class Classification(lightning.LightningModule):
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
            multi_class: bool = False,

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
        self.multi_class = multi_class

        self.network = network

        # self.label2name = get_label2name()
        self.val_ds_names = ["val"]
        self.metrics = nn.ModuleList(
            [
                MulticlassAccuracy(num_classes=self.network.num_classes, average="micro")
                for _ in range(len(self.val_ds_names))
            ]
        )

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
        b_image, b_target, b_smooth_target = batch[0], batch[1], batch[-1]
        # b_image = (B, C, T, H, W)
        # b_image = self.process_video_frames(b_image)
        source_logits = self.network(b_image)

        # Create mask but don't filter predictions, only use mask for loss calculation
        mask = b_target != -1
        b_smooth_target = torch.stack(b_smooth_target).T        
        # Only calculate loss on non-masked frames

        if mask.sum() > 0:
            loss_source = F.cross_entropy(source_logits[mask], b_smooth_target[mask]) #, weight = torch.tensor([1/149190, 1/28170, 1/28170, 1/28170, 1/28131, 1/24427], device=self.device))

        self.manual_backward(loss_source)
        opt.step()
        self.lr_schedulers().step()
        self.log("loss", loss_source, prog_bar=True)

        with torch.no_grad():
            if (self.global_step % 10) == 0:
                sourceds_predicted_segmentation = torch.argmax(source_logits.detach(), dim=1)
                acc_source = (sourceds_predicted_segmentation == b_target)
                acc_source = acc_source.float().mean()
                self.log("acc", acc_source, prog_bar=False)

            if (self.global_step % 100) == 0:
                self._log_img(b_image[0], "train")

    def eval_step(
            self,
            batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
            batch_idx: int,
            dataloader_idx: int,
            log_prefix: str,
    ):
        b_image, b_target, sample_idx = batch[0], batch[1], batch[2]

        mask = b_target != -1

        # b_image = (B, C, T, H, W)
        if len(batch) == 5:
            # Store all clip info but mark which ones are valid
            self.clip_infos.append(batch[4])
        else:
            # Store all clip info
            self.clip_infos.extend(zip(batch[4], batch[5]))
        
        with torch.no_grad():
            b_pred = self.network(b_image)
            
            # Store all predictions and create separate mask tensor
            self.all_preds.append(b_pred)
            self.all_labels.append(b_target)  # Store all labels including -1
            self.sample_idx.append(sample_idx)

            # Only calculate loss on valid frames
            if mask.sum() > 0:
                loss_source = F.cross_entropy(b_pred[mask], b_target[mask])
                self.log(f"{log_prefix}_loss", loss_source, prog_bar=True, sync_dist=True)

            if (batch_idx % 100) == 0:
                # only send first batch
                self._log_img(b_image[0], log_prefix) 

            # Only update metrics for valid frames
            if mask.sum() > 0:
                b_pred_valid = b_pred[mask].argmax(dim=1)
                b_target_valid = b_target[mask]
                self.metrics[dataloader_idx].update(
                    b_pred_valid, b_target_valid
                )

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
        for metric_idx, metric in enumerate(self.metrics):
            acc = metric.compute()
            metric.reset()
            ds_name = self.val_ds_names[metric_idx]
            self.log(
                f"{log_prefix}_{ds_name}_acc", acc, sync_dist=True
            )
        
        # Synchronize to ensure all processes have completed evaluation steps
        if torch.distributed.is_initialized():
            torch.distributed.barrier()
        
        # Concatenate all predictions and labels
        self.all_preds = torch.cat(self.all_preds, dim=0).type(torch.float16) if self.all_preds[0].dtype == torch.int64 else torch.cat(self.all_preds, dim=0)
        self.all_labels = torch.cat(self.all_labels, dim=0)
        self.clip_infos = torch.tensor(self.clip_infos) if type(self.clip_infos[0]) == tuple else torch.cat(self.clip_infos)
        self.sample_idx = torch.cat(self.sample_idx)

        # Create mask for valid labels (not -1)
        valid_mask = self.all_labels != -1
        
        # Calculate metrics only on valid frames
        valid_preds = self.all_preds[valid_mask]
        valid_labels = self.all_labels[valid_mask]
        
        # acc_whole_dataset, f1, auc_roc, ap, mcc_metrics, class_metrics, confmat, mcc_per_class = calculate_metrics_multi_class(self.all_preds, self.all_labels, multi_class=self.multi_class)
        acc_whole_dataset, f1, auc_roc, ap, confmat = metrics(valid_preds, valid_labels, do_softmax=True, num_classes=self.network.num_classes)
        more_metrics_to_log = [
            ("f1", f1),
            ("auc_roc", auc_roc),
            ("ap", ap),
        ]
            
        for name, value in more_metrics_to_log:
            self.log(
                f"{log_prefix}_{ds_name}_{name}", value, sync_dist=True
            )
        
        # Synchronize before all-gather operations
        if torch.distributed.is_initialized():
            torch.distributed.barrier()
            
        # Gather data from all processes
        self.all_preds = self.all_gather(self.all_preds)
        self.all_preds = self.all_preds.reshape(-1, *self.all_preds.shape[2:])
        self.all_labels = self.all_gather(self.all_labels).reshape(-1, *self.all_labels.shape[2:])
        self.clip_infos = self.all_gather(self.clip_infos)
        self.clip_infos = self.clip_infos.reshape(-1, 2) if self.clip_infos.ndim > 2 else self.clip_infos.reshape(-1, *self.clip_infos.shape[2:])
        self.sample_idx = self.all_gather(self.sample_idx).reshape(-1, *self.clip_infos.shape[2:]) 
        
        # Synchronize after gathering operations and before sorting
        if torch.distributed.is_initialized():
            torch.distributed.barrier()

        # sort preds, labels, clip_infos
        # this to preserve the order in each clip for calculating TTA and accuracy per frame distance w.r.t. start anomaly window/time of accident frame
        sorted_indices = self.sample_idx.argsort()    
        self.all_preds, self.all_labels, self.clip_infos = self.all_preds[sorted_indices], self.all_labels[sorted_indices], self.clip_infos[sorted_indices]
        
        # Update the valid mask after gathering and sorting
        valid_mask = self.all_labels != -1
        
        # Synchronize before rank-specific operations
        if torch.distributed.is_initialized():
            torch.distributed.barrier()
        
        # with open("data_kinetics_settings_new_lbl_balance_metrics_check_code_fix.pkl", "wb") as f:
        #     pickle.dump({
        #         "all_preds": self.all_preds.cpu(),
        #         "all_labels": self.all_labels.cpu(),
        #         "clip_infos": self.clip_infos.cpu(),
        #         "sample_idx": self.sample_idx.cpu()
        #     }, f)

        if self.global_rank == 0:
            # Use only valid frames for prediction lead time calculation
            valid_mask = self.all_labels != -1
            valid_preds = self.all_preds[valid_mask].cpu()
            valid_labels = self.all_labels[valid_mask].cpu()
            
            accuracy_per_label, pred_lead_time = prediction_lead_time(valid_preds, valid_labels)
            self.create_bar_chart(accuracy_per_label, log_prefix, ds_name)
            self.log("prediction lead time", pred_lead_time, sync_dist=False)

            self.create_confusion_matrix_plot(valid_preds, valid_labels, log_prefix, ds_name)

        # Final synchronization before clearing data
        if torch.distributed.is_initialized():
            torch.distributed.barrier()

        self.all_preds, self.all_labels, self.clip_infos, self.sample_idx = [], [], [], []

    def create_confusion_matrix_plot(self, preds, labels, log_prefix, ds_name):
        predicted_classes = preds.argmax(dim=1).cpu().numpy()
        true_labels = labels.cpu().numpy()
        unique_classes = np.unique(np.concatenate([true_labels, predicted_classes]))
        
        cm = confusion_matrix(true_labels, predicted_classes, labels=unique_classes)
        class_names = [str(i) for i in unique_classes]
        
        fig, ax = plt.subplots(figsize=(max(8, len(unique_classes)*2), max(6, len(unique_classes)*1.5)))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
        disp.plot(ax=ax, cmap='Blues', values_format='d')
        
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

    def create_bar_chart(self, accuracy_per_label, log_prefix, ds_name):
        fig, ax = plt.subplots(figsize=(20, 5))
        plt.title("Mean accuracy per Label")

            # Create color array - blue for all bars except last one (grey + transparent)
        colors = ['blue'] * (len(accuracy_per_label) - 1) + ['grey']
        plt.bar([str(label) for label in range(0, len(accuracy_per_label))][::-1], torch.flip(accuracy_per_label, dims=(0,)).numpy(), width=0.3, color=colors)
        plt.ylabel("Mean accuracy [-]")
        plt.xlabel("Label")
        plt.xticks(fontsize=10, rotation=90)
        ax.margins(x=-0.02)
        print(f"Mean accuracy per label (from label 5 to label 0): {torch.flip(accuracy_per_label, dims=(0,)).numpy()}")
        self.trainer.logger.experiment.log({f"{log_prefix}_{ds_name}_mean_acc_per_label": wandb.Image(fig, caption="Mean Accuracy per label")})

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
        
        # print("parameters obtained")
        # optim_weights = {param
        #     for name, param in self.network.named_parameters()
        #     if param.requires_grad}
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