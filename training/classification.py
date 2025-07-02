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

from metrics import calculate_tta, calculate_metrics_multi_class, accuracy_per_frame
from training.optim_factory import LayerDecayValueAssigner, get_parameter_groups

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
            use_strong_aug_source: bool = False,
            use_pointrend: bool = True,
            ckpt_path: Optional[str] = None,
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
        self.use_strong_aug_source = use_strong_aug_source
        self.use_pointrend = use_pointrend
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

        self._load_ckpt(ckpt_path)
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
        b_image, b_target = batch[0], batch[1]
        # b_image = (B, C, T, H, W)
        # b_image = self.process_video_frames(b_image)
        source_logits = self.network(b_image)
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

        # b_image = (B, C, T, H, W)
        if len(batch) == 5:
            self.clip_infos.append(batch[4])
        else:
            self.clip_infos.extend(zip(batch[4], batch[5]))
        
        with torch.no_grad():
            # b_image = self.process_video_frames(b_image)  # (B, C, T, H, W)
            b_pred = self.network(b_image)
            self.all_preds.append(b_pred)
            self.all_labels.append(b_target)
            self.sample_idx.append(sample_idx)
            loss_source = F.cross_entropy(b_pred, b_target)
            self.log(f"{log_prefix}_loss", loss_source, prog_bar=True, sync_dist=True)

            if (batch_idx % 100) == 0:
                # only send first batch
                self._log_img(b_image[0], log_prefix) 

            b_pred = b_pred.argmax(dim=1)
        self.metrics[dataloader_idx].update(
            b_pred, b_target
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
        
        self.all_preds = torch.cat(self.all_preds, dim=0).type(torch.float16) if self.all_preds[0].dtype == torch.int64 else torch.cat(self.all_preds, dim=0)
        self.all_labels = torch.cat(self.all_labels, dim=0)
        self.clip_infos = torch.tensor(self.clip_infos) if type(self.clip_infos[0]) == tuple else torch.cat(self.clip_infos)
        self.sample_idx = torch.cat(self.sample_idx)

        acc_whole_dataset, f1, auc_roc, ap, mcc_metrics, class_metrics, confmat, mcc_per_class = calculate_metrics_multi_class(self.all_preds, self.all_labels, multi_class=self.multi_class)
        more_metrics_to_log = [
            ("f1", f1),
            ("auc_roc", auc_roc),
            ("ap", ap),
        ]
        if not class_metrics == None:
            num_classes = 3

            classes = ["non_accident", "ego_class", "non_ego_class"]

            # do not log metrics for non-accident class (0)
            for i in range(1, num_classes):
                print(f"i: {i}")
                recall = class_metrics[3][i]

                try:
                    if self.global_rank == 0:
                        recall_tensor = torch.tensor(recall, device=self.device, dtype=torch.float32)
                        recall_gathered_mean = self.all_gather(recall_tensor)
                        print(recall_gathered_mean)

                        # wandb.log({
                        # f"{log_prefix}_{classes[i]}/recall_curve": wandb.plot.line_series(
                        #     xs=np.arange(0.00, 1.001, 0.01).tolist(),
                        #     ys=[recall_gathered_mean],
                        #     title=f"Recall for {classes[i]}",
                        #     xname="Threshold"
                        # )
                        # })
                        print("recall logged!")
                except Exception as e:
                    print(e)                  
                
                per_class_metrics = [
                    ("ap", class_metrics[0][i]),
                    ("auroc", class_metrics[1][i]),
                    ("acc", class_metrics[2][i]),
                    ("mcc_auc", mcc_per_class[0][i]),
                    ("mcc_max", mcc_per_class[1][i]),
                    ("mcc_05", mcc_per_class[2][i]),
                ]
                
                for name, value in per_class_metrics:
                    self.log(
                        f"{log_prefix}_{classes[i]}/{name}", value, sync_dist=True
                    )
            
                
        else:
            mcc_auc, mcc_max, mcc_05 = mcc_metrics
            more_metrics_to_log = [
            ("f1", f1),
            ("auc_roc", auc_roc),
            ("ap", ap),
            ("mcc_auc", mcc_auc),
            ("mcc_max", mcc_max),
            ("mcc_05", mcc_05)
        ]
            
        for name, value in more_metrics_to_log:
            self.log(
                f"{log_prefix}_{ds_name}_{name}", value, sync_dist=True
            )
            
        self.all_preds = self.all_gather(self.all_preds)
        self.all_preds = self.all_preds.reshape(-1, *self.all_preds.shape[2:])

        self.all_labels = self.all_gather(self.all_labels).reshape(-1, *self.all_labels.shape[2:])

        self.clip_infos = self.all_gather(self.clip_infos)
        self.clip_infos = self.clip_infos.reshape(-1, 2) if self.clip_infos.ndim > 2 else self.clip_infos.reshape(-1, *self.clip_infos.shape[2:])

        self.sample_idx = self.all_gather(self.sample_idx).reshape(-1, *self.clip_infos.shape[2:]) 

        # sort preds, labels, clip_infos
        # this to preserve the order in each clip for calculating TTA and accuracy per frame distance w.r.t. start anomaly window/time of accident frame
        sorted_indices = self.sample_idx.argsort()    
        self.all_preds, self.all_labels, self.clip_infos = self.all_preds[sorted_indices], self.all_labels[sorted_indices], self.clip_infos[sorted_indices]

        if self.global_rank == 0:
            
            # TTA
            if self.clip_infos.ndim > 1:
                thresholds = np.arange(0, 1.0, 0.1)
                all_ttas = []
                
                for threshold in thresholds:
                    tta, itta = calculate_tta(clip_predictions=self.all_preds, clip_infos=self.clip_infos, threshold=threshold)
                    all_ttas.append(tta)

                    if threshold == 0.5:
                        itta_0_5 = itta
                        tta_0_5 = tta       
                        # Log these metrics without sync_dist (already global)
                        self.log("tta_0_5", tta_0_5, sync_dist=False)
                        self.log("itta_0_5", itta_0_5, sync_dist=False)

                mtta = np.mean(all_ttas)
                self.log("mtta", mtta, sync_dist=False)
                all_ttas = []

            accuracy_per_distance = accuracy_per_frame(self.all_preds.cpu(), self.clip_infos.cpu(), self.all_labels.cpu())
            self.create_bar_chart(accuracy_per_distance, log_prefix, ds_name)
            
        self.all_preds, self.all_labels, self.clip_infos, self.sample_idx = [], [], [], []

    def create_bar_chart(self, accuracy_per_distance, log_prefix, ds_name):
        fig, ax = plt.subplots(figsize=(50, 5))
        plt.title("mean accuracy per frame distance w.r.t anomaly window start")
        colors = ['red' if k == 0 else 'blue' for k in accuracy_per_distance.keys()]
        plt.bar([str(key) for key in accuracy_per_distance.keys()], accuracy_per_distance.values(), width=0.3, color=colors)
        plt.ylabel("Mean accuracy [%]");
        plt.xlabel("Frame interval");
        plt.xticks(fontsize=8);
        plt.xticks(rotation=90);
        ax.margins(x=-0.02)  # Add some margin to the x-axis

        # Log the matplotlib figure as a wandb.Image
        self.trainer.logger.experiment.log({f"{log_prefix}_{ds_name}_mean_acc_per_frames_dist": wandb.Image(fig, caption="Mean Accuracy per Frames Distance w.r.t. Anomaly Window Start")})

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

    def _load_ckpt(self, ckpt_path: Optional[str]):
        if ckpt_path is None:
            return

        ckpt_state: dict = torch.load(ckpt_path, map_location=self.device)

        if "state_dict" in ckpt_state:
            ckpt_state = ckpt_state["state_dict"]

        if "model" in ckpt_state:
            ckpt_state = ckpt_state["model"]

        model_state = self.state_dict()
        skipped_keys = []
        for k in ckpt_state:
            if (k in model_state) and (ckpt_state[k].shape == model_state[k].shape):
                model_state[k] = ckpt_state[k]
            else:
                skipped_keys.append(k)

        info(f"Skipped loading keys: {skipped_keys}")

        self.load_state_dict(model_state)

    def process_video_frames(self, video_data: torch.Tensor) -> torch.Tensor:
        """
        Applies color jitter, blur, and normalization per frame for each sample
        in a batch of video sequences or a single video sequence.

        This function automatically detects if the input is a batch (5 dimensions)
        or a single video (4 dimensions) and processes it accordingly.

        Args:
            video_data (torch.Tensor): Input video data.
                                    Expected shapes:
                                    - Batch: (B, C, T, H, W)
                                    - Single: (C, T, H, W)

        Returns:
            torch.Tensor: The processed video data with the same original shape.
        """
        # Check if the input is a single video (C, T, H, W) or a batch (B, C, T, H, W)
        is_single_sample = False
        if video_data.ndim == 4:
            # If it's a single sample, add a batch dimension for consistent processing
            video_batch = video_data.unsqueeze(0)  # Becomes (1, C, T, H, W)
            is_single_sample = True
        elif video_data.ndim == 5:
            video_batch = video_data
        else:
            raise ValueError(
                f"Input 'video_data' must have 4 or 5 dimensions (C, T, H, W) or (B, C, T, H, W), "
                f"but got {video_data.ndim} dimensions."
            )

        B, C, T, H, W = video_batch.shape

        # only apply augmentations during training
        if self.training:
            b_image_aug = torch.stack([
                torch.stack([self.augmentations(video_batch[b, :, t, :, :]) 
                            for t in range(T)], dim=1) 
                for b in range(B)
            ], dim=0) 
            # Resulting shape: (B, C, T, H, W)
        else:
            b_image_aug = video_batch

        b_image_norm = torch.stack([
            torch.stack([self.norm(b_image_aug[b, :, t, :, :]) 
                        for t in range(T)], dim=1) 
            for b in range(B)
        ], dim=0) 
        # Resulting shape: (B, C, T, H, W)

        if is_single_sample:
            return b_image_norm.squeeze(0) # Squeeze back to (C, T, H, W)
        else:
            return b_image_norm # Return (B, C, T, H, W)
    
