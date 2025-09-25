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
            eta_min: float = 1e-6,

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

        self.eta_min = eta_min
        print(f"eta_min: {self.eta_min}")

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

        source_logits = self.network(b_image)

        b_smooth_target = torch.stack(b_smooth_target).T        

        loss_source = F.cross_entropy(source_logits, b_target)

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

        self.clip_infos.extend(zip(batch[3], batch[4]))
        
        with torch.no_grad():
            b_pred = self.network(b_image)
            
            self.all_preds.append(b_pred)
            self.all_labels.append(b_target)
            self.sample_idx.append(sample_idx)

            loss_source = F.cross_entropy(b_pred, b_target)
            self.log(f"{log_prefix}_loss", loss_source, prog_bar=True, sync_dist=True)

            if (batch_idx % 100) == 0:
                # only send first batch
                self._log_img(b_image[0], log_prefix) 

            b_pred_valid = b_pred.argmax(dim=1)
            b_target_valid = b_target
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
        
        if torch.distributed.is_initialized():
            torch.distributed.barrier()
        
        # Concatenate all predictions and labels
        self.all_preds = torch.cat(self.all_preds, dim=0).type(torch.float16) if self.all_preds[0].dtype == torch.int64 else torch.cat(self.all_preds, dim=0)
        self.all_labels = torch.cat(self.all_labels, dim=0)
        self.clip_infos = torch.tensor(self.clip_infos) if type(self.clip_infos[0]) == tuple else torch.cat(self.clip_infos)
        self.sample_idx = torch.cat(self.sample_idx)
        
        acc_whole_dataset, f1, auc_roc, ap, confmat = metrics(self.all_preds, self.all_labels, do_softmax=True, num_classes=self.network.num_classes)
        more_metrics_to_log = [
            ("f1", f1),
            ("auc_roc", auc_roc),
            ("ap", ap),
        ]
            
        for name, value in more_metrics_to_log:
            self.log(
                f"{log_prefix}_{ds_name}_{name}", value, sync_dist=True
            )
        
        if torch.distributed.is_initialized():
            torch.distributed.barrier()
            
        # Gather data from all processes
        self.all_preds = self.all_gather(self.all_preds)
        self.all_preds = self.all_preds.reshape(-1, *self.all_preds.shape[2:])
        self.all_labels = self.all_gather(self.all_labels).reshape(-1, *self.all_labels.shape[2:])
        self.clip_infos = self.all_gather(self.clip_infos)
        self.clip_infos = self.clip_infos.reshape(-1, 2) if self.clip_infos.ndim > 2 else self.clip_infos.reshape(-1, *self.clip_infos.shape[2:])
        self.sample_idx = self.all_gather(self.sample_idx).reshape(-1) 
        
        if torch.distributed.is_initialized():
            torch.distributed.barrier()

        # sort preds, labels, clip_infos
        # this to preserve the order in each clip for calculating TTA and accuracy per frame distance w.r.t. start anomaly window/time of accident frame
        sorted_indices = self.sample_idx.argsort()    
        self.all_preds, self.all_labels, self.clip_infos = self.all_preds[sorted_indices], self.all_labels[sorted_indices], self.clip_infos[sorted_indices]
        
        if torch.distributed.is_initialized():
            torch.distributed.barrier()

        if self.global_rank == 0:
            
            accuracy_per_label, _ = prediction_lead_time(self.all_preds.cpu(), self.all_labels.cpu())
            self.create_bar_chart(accuracy_per_label, log_prefix, ds_name)
            # self.log("prediction lead time", pred_lead_time, sync_dist=False)

            self.create_confusion_matrix_plot(self.all_preds.cpu(), self.all_labels.cpu(), log_prefix, ds_name)

            if self.global_step > 19000:
                with open(f"data_class_new_split_alpha3_bins_of_1s_new_lbl_balancing_lr2e-5_ld06_lr_min1e-8_corr_mdl_outputs.pkl", "wb") as f:
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
        fig, ax = plt.subplots(figsize=(5, 2.5))

            # Create color array - orange for all bars except last one (grey + transparent)
        colors = ['orange'] * (len(accuracy_per_label) - 1) + ['grey']
        labels = [str(label) for label in range(0, len(accuracy_per_label))]
        bars = plt.bar(labels[::-1], torch.flip(accuracy_per_label, dims=(0,)).numpy(), width=0.3, color=colors)
        plt.ylabel("Mean accuracy [-]")
        plt.xlabel("Labels (sec before collision)")
        plt.title("Mean accuracy per Label")
        plt.xticks(fontsize=10, rotation=90)
        plt.tight_layout()
        plt.grid()
        plt.axhline(0.5, color="green", linestyle="--")
        plt.yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        plt.xticks(labels, ["normal"] + [str(1 + i*1) + "s" for i in range(len(labels)-1)], fontsize=10, rotation=90)
        # ax.margins(x=-0.02)
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x(), yval + .005, f"{yval:.3f}", size=8)
        print(f"Mean accuracy per label (from label {len(accuracy_per_label)-1} to label 0): {torch.flip(accuracy_per_label, dims=(0,)).numpy()}")
        self.trainer.logger.experiment.log({f"{log_prefix}_{ds_name}_mean_acc_per_label": wandb.Image(fig, caption="Mean Accuracy per label")})

    def on_validation_epoch_end(self):
        self._on_eval_epoch_end("val")

    def on_test_epoch_end(self):
        self._on_eval_epoch_end("test")

    def on_train_batch_end(self, outputs, batch, batch_idx):
        for i, param_group in enumerate(self.trainer.optimizers[0].param_groups):
            self.log(f"learning_rate/group_{i}", param_group["lr"], on_step=True)

    def configure_optimizers(self):
        
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
                    eta_min=self.eta_min,
                ),
                "interval": "step",
            }
        else:
            raise Exception("Wrong lr_more: {}".format(self.lr_mode))

        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}