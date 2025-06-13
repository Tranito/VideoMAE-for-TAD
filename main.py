import logging

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
        parser.add_argument("--dataset", type=str, default="dota")
        # parser.add_argument('--input_size', type=int, default=224, help='videos input size')

        # parser.link_arguments("input_size", "data.init_args.input_size")
        parser.link_arguments("root", "data.init_args.root")
        parser.link_arguments("dataset", "data.init_args.dataset")
        parser.link_arguments("root", "trainer.logger.init_args.save_dir")
        parser.link_arguments("trainer.devices", "data.init_args.devices")
        parser.link_arguments("data.init_args.batch_size", "model.init_args.batch_size")
        parser.link_arguments("data.init_args.img_size", "model.init_args.img_size")
        parser.link_arguments("data.init_args.img_size", "model.init_args.network.init_args.img_size")

        

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
