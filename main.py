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
        # parser.add_argument('--input_size', type=int, default=224, help='videos input size')

        # parser.link_arguments("input_size", "data.init_args.input_size")
        parser.link_arguments("root", "data.init_args.root")
        parser.link_arguments("root", "trainer.logger.init_args.save_dir")
        parser.link_arguments("trainer.devices", "data.init_args.devices")
        parser.link_arguments("data.init_args.batch_size", "model.init_args.batch_size")
        parser.link_arguments("data.init_args.img_size", "model.init_args.img_size")
        parser.link_arguments("data.init_args.img_size", "model.init_args.network.init_args.img_size")

        # # for loading DoTA and DADA-2000

        # Model parameters
        parser.add_argument('--input_size', default=224, type=int,
                            help='videos input size')

        # Augmentation parameters
        parser.add_argument('--num_sample', type=int, default=1,
                            help='Repeated_aug (default: 2)')
        parser.add_argument('--aa', type=str, default='rand-m6-n3-mstd0.5-inc1', metavar='NAME',
                            help='Use AutoAugment policy. "v0" or "original". " + "(default: rand-m7-n4-mstd0.5-inc1)'),
        parser.add_argument('--train_interpolation', type=str, default='bicubic',
                            help='Training interpolation (random, bilinear, bicubic default: "bicubic")')

        # Evaluation parameters
        parser.add_argument('--short_side_size', type=int, default=224)
        parser.add_argument('--test_num_segment', type=int, default=1)
        
        # Random Erase params
        parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                            help='Random erase prob (default: 0.25)')
        parser.add_argument('--remode', type=str, default='pixel',
                            help='Random erase mode (default: "pixel")')
        parser.add_argument('--recount', type=int, default=1,
                            help='Random erase count (default: 1)')

        # Dataset parameters
        parser.add_argument('--data_path', default="/media/datasets_sveta/DoTA_refined", type=str,
                            help='dataset path')
        parser.add_argument('--nb_classes', default=2, type=int,
                            help='number of the classification types')
        parser.add_argument('--sampling_rate', type=int, default=1)
        parser.add_argument('--sampling_rate_val', type=int, default=2)
        parser.add_argument('--view_fps', type=int, default=10)  # DoTA, DADA2k only!
        parser.add_argument('--data_set', default='DoTA', choices=['DoTA', 'DoTA_half', 'DoTA_amnet', 'DADA2K', 'DADA2K_half','image_folder'],
                            type=str, help='dataset')
    
        # Optimizer parameters
        parser.add_argument('--loss', default='crossentropy',
                            choices=['crossentropy', 'focal', 'focal6x100', 'focal2_6', 'focal2_2', 'smoothap', 'exponential1', "2bce"],
                            type=str, help='dataset')
        
        # parser.link_arguments("input_size", "data.init_args.input_size")
        # parser.link_arguments("num_sample", "data.init_args.num_sample")
        # parser.link_arguments("aa", "data.init_args.aa")
        # parser.link_arguments("train_interpolation", "data.init_args.train_interpolation")
        # parser.link_arguments("short_side_size", "data.init_args.short_side_size")
        # parser.link_arguments("test_num_segment", "data.init_args.test_num_segment")
        # parser.link_arguments("data_set", "data.init_args.data_set")
        # parser.link_arguments("reprob", "data.init_args.reprob")
        # parser.link_arguments("remode", "data.init_args.remode")
        # parser.link_arguments("recount", "data.init_args.recount")
        # parser.link_arguments("data_path", "data.init_args.data_path")
        # parser.link_arguments("nb_classes", "data.init_args.nb_classes")
        # parser.link_arguments("sampling_rate", "data.init_args.sampling_rate")
        # parser.link_arguments("sampling_rate_val", "data.init_args.sampling_rate_val")
        # parser.link_arguments("view_fps", "data.init_args.view_fps")
        # parser.link_arguments("loss", "data.init_args.loss")
        

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
        
        # print(f"self.config[\"data\"][\"init_args\"]: {self.config}" )


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
