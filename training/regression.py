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
from new_metrics import metrics, prediction_lead_time, accuracy_per_bin_regression
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
            eta_min: float = 1e-6
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
        self.network = network
        self.val_ds_names = ["val"]
        self.automatic_optimization = False

        # for saving predictions and labels for calculating eval metrics and clip info for TTA (only for DADA2K)
        self.all_preds = []
        self.all_labels = []
        self.clip_infos = []
        self.sample_idx = []
        self.all_log_variances = []
        self.classification_preds = []
        self.classification_labels = []
        self.bin_labels = []
        self.hybrid_model_preds = []

    def get_model_predictions(self, classification_preds, regression_preds):
        classfication_prob = F.softmax(classification_preds, dim=1)
        #take only positive class predictions
        classfication_prob = classfication_prob[:,1]

        model_preds = []

        for i, pred in enumerate(classfication_prob):
            if pred < 0.5:
                model_preds.append(0)
            else:
                if regression_preds[i] < 0:
                    model_preds.append(1)
                if 0 <= regression_preds[i] < 1:
                    model_preds.append(1)
                elif 1 <= regression_preds[i] < 2:
                    model_preds.append(2)
                elif 2 <= regression_preds[i] < 3:
                    model_preds.append(3)
                elif 3 <= regression_preds[i] < 4:
                    model_preds.append(4)
                elif 4 <= regression_preds[i] < 5:
                    model_preds.append(5)

        return torch.tensor(model_preds)

    def training_step(
            self,
            batch: Tuple[torch.Tensor, torch.Tensor],
            batch_idx: int,
    ):
        opt = self.get_optimizers()
        b_image, b_target = batch[0], batch[-2]
        # b_image = (B, C, T, H, W)
        mean_ttc, log_variance, classification_logits = self.network(b_image)
        mean_ttc = mean_ttc.squeeze(-1)
        log_variance = log_variance.squeeze(-1)

        alpha = 3.0
        mask_A = (b_target <= alpha)
        var = (log_variance.exp() + 1e-8)

        if mask_A.any():
            error_A = b_target[mask_A] - mean_ttc[mask_A]
            loss_A = ((error_A ** 2) / (2 * var[mask_A]) + 0.5 * log_variance[mask_A]).mean()
        else:
            loss_A = 0.0

        classification_loss = F.cross_entropy(classification_logits, mask_A.long(), reduction='mean')
        loss_source = loss_A + classification_loss

        self.manual_backward(loss_source)
        opt.step()
        self.lr_schedulers().step()
        
        self.log("train loss", loss_source, prog_bar=True)
        self.log("train loss col_soon", loss_A)
        self.log("train loss classification", classification_loss)


        with torch.no_grad():
            if (self.global_step % 10) == 0:
                mae = torch.mean(torch.abs(mean_ttc.cpu() - b_target.cpu()))
                self.log(f"mae", mae, prog_bar=True)

                source_predicted_segmentation = torch.argmax(classification_logits.detach(), dim=1)
                acc_source = (source_predicted_segmentation == mask_A.float())
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
        b_image, b_target, sample_idx, bin_lbl = batch[0], batch[-1], batch[2], batch[1]
        self.clip_infos.extend(zip(batch[3], batch[4]))
        
        with torch.no_grad():
            mean_ttc, log_variance, classification_output = self.network(b_image)
            mean_ttc = mean_ttc.squeeze(-1)
            log_variance = log_variance.squeeze(-1)

            self.all_preds.append(mean_ttc)
            self.all_labels.append(b_target) 
            self.sample_idx.append(sample_idx)
            self.all_log_variances.append(log_variance)
            self.bin_labels.append(bin_lbl)

            alpha = 3.0
            mask_A = (b_target <= alpha)
            var = (log_variance.exp() + 1e-8)

            if mask_A.any():
                error_A = b_target[mask_A] - mean_ttc[mask_A]
                loss_A = ((error_A ** 2) / (2 * var[mask_A]) + 0.5 * log_variance[mask_A]).mean()
            else:
                loss_A = 0.0

            classification_loss = F.cross_entropy(classification_output, mask_A.long(), reduction='mean')
            #FOR CLASSIFICATION
            self.classification_preds.append(classification_output)
            self.classification_labels.append(mask_A.long())

            loss_source = loss_A + classification_loss

            self.log(f"{log_prefix}_loss", loss_source, prog_bar=True, sync_dist=True)
            self.log(f"{log_prefix}_loss_classification", classification_loss, sync_dist=True)
            self.log(f"{log_prefix}_loss_col_soon", loss_A, sync_dist=True)
            self.log(f"{log_prefix}_log(var)_col_soon", log_variance[mask_A].mean() if mask_A.any() else torch.tensor(0.,device=b_target.device),  sync_dist=True)

            if (batch_idx % 100) == 0:
                self._log_img(b_image[0], log_prefix) 

            mae = torch.mean(torch.abs(mean_ttc.cpu() - b_target.cpu()))
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
        
        # Concatenate all predictions and labels
        self.all_preds = torch.cat(self.all_preds, dim=0)
        self.all_labels = torch.cat(self.all_labels, dim=0)
        self.clip_infos = torch.tensor(self.clip_infos)
        self.sample_idx = torch.cat(self.sample_idx, dim=0)
        self.all_log_variances = torch.cat(self.all_log_variances, dim=0)
        #FOR CLASSIFICATION
        self.classification_preds = torch.cat(self.classification_preds, dim=0)
        self.classification_labels = torch.cat(self.classification_labels, dim=0)

        #FOR NEW EVALUATION FOR HYBRID MODEL
        self.bin_labels = torch.cat(self.bin_labels, dim=0)

        if torch.distributed.is_initialized():
            torch.distributed.barrier()
            
        # Gather data from all processes
        self.all_preds = self.all_gather(self.all_preds).reshape(-1)
        self.all_labels = self.all_gather(self.all_labels).reshape(-1)
        self.clip_infos = self.all_gather(self.clip_infos).reshape(-1, 2)
        self.sample_idx = self.all_gather(self.sample_idx).reshape(-1) 
        self.all_log_variances = self.all_gather(self.all_log_variances).reshape(-1)
        #FOR CLASSIFICATION
        self.classification_preds = self.all_gather(self.classification_preds).reshape(-1,2)
        self.classification_labels = self.all_gather(self.classification_labels).reshape(-1)

        #FOR NEW EVALUATION FOR HYBRID MODEL
        self.bin_labels = self.all_gather(self.bin_labels).reshape(-1)

        if torch.distributed.is_initialized():
            torch.distributed.barrier()

        # sort preds and labels such their order for each clip is preserved for visualization
        sorted_indices = self.sample_idx.argsort()    
        self.all_preds, self.all_labels, self.clip_infos = self.all_preds[sorted_indices], self.all_labels[sorted_indices], self.clip_infos[sorted_indices]
        self.all_log_variances = self.all_log_variances[sorted_indices]
        #FOR CLASSIFICATION
        self.classification_preds = self.classification_preds[sorted_indices]
        self.classification_labels = self.classification_labels[sorted_indices]

        #FOR NEW EVALUATION FOR HYBRID MODEL
        self.bin_labels = self.bin_labels[sorted_indices]

        if torch.distributed.is_initialized():
            torch.distributed.barrier()

        if self.global_rank == 0:

            model_preds = self.get_model_predictions(self.classification_preds, self.all_preds)
            self.hybrid_model_preds.append(model_preds)

            mean_accuracy_per_bin_func = MulticlassAccuracy(num_classes= 4, average=None)
            mean_accuracy_per_bin = mean_accuracy_per_bin_func(model_preds.cpu(), self.bin_labels.cpu())

            print(f"mean_accuracy_per_bin: {mean_accuracy_per_bin}")
            self.create_bar_chart(torch.roll(mean_accuracy_per_bin, shifts=(-1)), log_prefix, "val")
            mean_acc_val = self.classification_preds.argmax(dim=1) == self.classification_labels
            mean_acc_val = mean_acc_val.float().mean()
            self.log(f"{log_prefix}_mean_acc_classification", mean_acc_val)

            acc_per_class = MulticlassAccuracy(num_classes=2, average=None)
            acc_per_class_val = acc_per_class(self.classification_preds.softmax(dim=1).cpu(), self.classification_labels.cpu())
            self.log(f"{log_prefix}_mean_acc_classification_class_0", acc_per_class_val[0])
            self.log(f"{log_prefix}_mean_acc_classification_class_1", acc_per_class_val[1])

            # save all data from last validation for visualization
            if self.global_step > 19000:
                with open(f"data_regre_classs_lr2e-5_newsplt_ld06_2regions_simp_heads_lrmin1e-8_alpha3_new_bin_acc_eval.pkl", "wb") as f:
                    pickle.dump({
                        "all_preds": self.all_preds.cpu(),
                        "all_labels": self.all_labels.cpu(),
                        "clip_infos": self.clip_infos.cpu(),
                        "sample_idx": self.sample_idx.cpu(),
                        "all_log_variances": self.all_log_variances.cpu(),
                        "classification_preds": self.classification_preds.cpu(),
                        "classification_labels": self.classification_labels.cpu(),
                        "bin_labels": self.bin_labels.cpu(),
                        "hybrid_model_preds": torch.stack(self.hybrid_model_preds).cpu(),
                    }, f)

        if torch.distributed.is_initialized():
            torch.distributed.barrier()

        self.all_preds, self.all_labels, self.clip_infos, self.sample_idx, self.all_log_variances = [], [], [], [], []
        #FOR CLASSIFICATION
        self.classification_preds, self.classification_labels = [], []

        #FOR NEW EVALUATION FOR HYBRID MODEL
        self.bin_labels = []
        self.hybrid_model_preds = []

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

        all_params = optim_weights

        optimizer = torch.optim.AdamW(all_params, betas=(0.9, 0.999))
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
    
    def get_optimizers(self):
        opt = self.optimizers()
        opt.zero_grad()
        return opt
    
    def create_bar_chart(self, accuracy_per_label, log_prefix, ds_name):
        fig, ax = plt.subplots(figsize=(5, 2.5))

            # Create color array - orange for all bars except last one (grey + transparent)
        colors = ['grey'] + ['orange'] * (len(accuracy_per_label) - 1) 
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
        plt.xticks(labels, [str(1 + i*1) + "s" for i in range(len(labels)-1)] + ["normal"], fontsize=10, rotation=90)
        # ax.margins(x=-0.02)
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x(), yval + .005, f"{yval:.3f}", size=8)
        print(f"Mean accuracy per label (from label {len(accuracy_per_label)-1} to label 0): {torch.flip(accuracy_per_label, dims=(0,)).numpy()}")
        self.trainer.logger.experiment.log({f"{log_prefix}_{ds_name}_mean_acc_per_label": wandb.Image(fig, caption="Mean Accuracy per label")})

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