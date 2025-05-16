import os
import argparse

def get_args():
    parser = argparse.ArgumentParser('VideoMAE fine-tuning and evaluation script for video classification', add_help=False)
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
    parser.add_argument('--num_frames', type=int, default=16)
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

    return parser.parse_args()