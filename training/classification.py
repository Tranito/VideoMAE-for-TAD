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

from datasets.utils.mappings import get_label2name
from datasets.utils.util import normalize
from models.utils.warmup_and_linear_scheduler import WarmupAndLinearScheduler
from training.utils.utils import get_full_names, get_param_group, process_parameters


class Classification(lightning.LightningModule):
    def __init__(
            self,
            batch_size: int,
            img_size: int,
            network: nn.Module,
            lr: float = 2e-5,
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

        self.network = network

        self.label2name = get_label2name()
        self.val_ds_names = ["val"]
        self.metrics = nn.ModuleList(
            [
                MulticlassAccuracy(num_classes=self.network.num_classes)
                for _ in range(len(self.val_ds_names))
            ]
        )

        self._load_ckpt(ckpt_path)
        self.automatic_optimization = False

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
        source_logits = self.network(b_image)
        # print(f"b_target: {b_target}")
        # print(f"source_logits: {source_logits}")
        # print(f"source_logits shape: {source_logits.shape}")

        loss_source = F.cross_entropy(source_logits, b_target)
        # print(b_image.shape, b_image.min(), b_image.max())
        # print(b_target.shape, b_target)
        # print(source_logits.shape, source_logits)
        # exit(0)
        self.manual_backward(loss_source)
        opt.step()
        self.lr_schedulers().step()

        # print(f"source_logits: {source_logits.shape}")
        # print(f"targets: {b_target}")
        
        self.log("loss", loss_source, prog_bar=True)

        with torch.no_grad():
            if (self.global_step % 10) == 0:
                sourceds_predicted_segmentation = torch.argmax(source_logits.detach(), dim=1)
                acc_source = (sourceds_predicted_segmentation == b_target)
                acc_source = acc_source.float().mean()
                self.log("acc", acc_source, prog_bar=False)

    def eval_step(
            self,
            batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
            batch_idx: int,
            dataloader_idx: int,
            log_prefix: str,
    ):
        # print(f"batch: {batch}")
        b_image, b_target = batch[0], batch[1]
        # print(b_image.shape)
        # print(b_target.shape)
        with torch.no_grad():
            b_pred = self.network(b_image)
            b_pred = b_pred.argmax(dim=1)
        self.metrics[dataloader_idx].update(
            b_pred, b_target
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

    def on_validation_epoch_end(self):
        self._on_eval_epoch_end("val")

    def on_test_epoch_end(self):
        self._on_eval_epoch_end("test")

    def on_train_batch_end(self, outputs, batch, batch_idx):
        for i, param_group in enumerate(self.trainer.optimizers[0].param_groups):
            self.log(f"learning_rate/group_{i}", param_group["lr"], on_step=True)

    def configure_optimizers(self):


        optim_weights = {param
            for name, param in self.network.named_parameters()
            if param.requires_grad}
        # print(f"optim_weights: {optim_weights}")
        # exit(0)
        print(f"lr: {self.lr}")

        optimizer = torch.optim.AdamW(optim_weights, weight_decay=self.weight_decay, lr=self.lr, betas=(0.9, 0.999))

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
