import logging
from typing import Optional

import torch
from gitignore_parser import parse_gitignore
from lightning import LightningModule
from lightning.pytorch import cli

from datasets.utils.custom_lightning_data_module import CustomLightningDataModule
import sys


class LightningCLI(cli.LightningCLI):
    def __init__(self, *args, **kwargs):
        logging.getLogger().setLevel(logging.INFO)
        torch.set_float32_matmul_precision("medium")
        super().__init__(*args, **kwargs)

    def add_arguments_to_parser(self, parser):
        parser.add_argument("--compile", type=bool, default=False)
        parser.add_argument("--root", type=str)
        parser.add_argument("--dataset", type=str, default="dada")

        parser.add_argument("--alpha", type=int, default=6)
        parser.add_argument("--bin_width", type=float, default=1.0)
        parser.add_argument("--uncertainty_pred", type=bool, default=False)
        parser.add_argument("--model_ckpt", type=Optional[str], default=None)

        parser.link_arguments("alpha", "data.init_args.alpha")
        parser.link_arguments("alpha", "model.init_args.network.init_args.alpha")
        parser.link_arguments("alpha", "model.init_args.alpha")

        parser.link_arguments("bin_width", "data.init_args.bin_width")
        parser.link_arguments("bin_width", "model.init_args.network.init_args.bin_width")
        parser.link_arguments("bin_width", "model.init_args.bin_width")

        parser.link_arguments("uncertainty_pred", "model.init_args.uncertainty_pred")
        parser.link_arguments("uncertainty_pred", "model.init_args.network.init_args.uncertainty_pred")

        parser.link_arguments("model_ckpt", "model.init_args.network.init_args.model_ckpt")

        parser.link_arguments("root", "data.init_args.root")
        parser.link_arguments("dataset", "data.init_args.dataset")
        parser.link_arguments("root", "trainer.logger.init_args.save_dir")
        parser.link_arguments("trainer.devices", "data.init_args.devices")
        parser.link_arguments("data.init_args.batch_size", "model.init_args.batch_size")
        parser.link_arguments("data.init_args.img_size", "model.init_args.img_size")
        parser.link_arguments("data.init_args.img_size", "model.init_args.network.init_args.img_size")
        parser.link_arguments("data.init_args.num_frames", "model.init_args.network.init_args.all_frames")

    def fit(self, model, **kwargs):
        if hasattr(self.trainer.logger.experiment, "log_code"):
            is_gitignored = parse_gitignore(".gitignore")
            include_fn = lambda path: path.endswith(".py") or path.endswith(".yaml")
            self.trainer.logger.experiment.log_code(
                ".", include_fn=include_fn, exclude_fn=is_gitignored
            )

        if self.config[self.config["subcommand"]]["compile"]:
            import torch._dynamo
            torch._dynamo.config.suppress_errors = False
            model = torch.compile(model)
    

        self.trainer.fit(model, **kwargs)


def cli_main():
    LightningCLI(
        LightningModule,
        CustomLightningDataModule,
        subclass_mode_model=True,
        subclass_mode_data=True,
        save_config_callback=None,
        seed_everything_default=42,
        trainer_defaults={
            "precision": "16-mixed",
            "log_every_n_steps": 1,
            "strategy": "ddp_find_unused_parameters_true"
        },
    )
    

if __name__ == "__main__":
    cli_main()
