import cv2
import numpy as np
import torch
import pandas as pd
import json
from natsort import natsorted
from PIL import Image
from torchvision import transforms
import warnings
from torch.utils.data import Dataset
from tqdm import tqdm
import warnings


from datasets.random_erasing import RandomErasing
import datasets.video_transforms as video_transforms 
import datasets.volume_transforms as volume_transforms

from datasets.dataset_loading.sequencing import RegularSequencer, RegularSequencerWithStart
from datasets.dataset_loading.data_utils import smooth_labels, compute_time_vector
import os
import zipfile
from types import SimpleNamespace
from datasets.dota import FrameClsDataset_DoTA
from datasets.dada import FrameClsDataset_DADA
            
def build_frame_dataset(is_train, test_mode, args):
    args = SimpleNamespace(**args)
    # print(args)
    if args.data_set.startswith('DoTA'):
        mode = None
        anno_path = None
        orig_fps = 10
        if is_train is True:
            mode = 'train'
            if "_half" in args.data_set:
                anno_path = 'half_train_split.txt'
            elif "_amnet" in args.data_set:
                anno_path = 'amnet_train_split300.txt'
            else: 
                anno_path = 'train_split.txt'
            sampling_rate = args.sampling_rate
        elif test_mode is True:
            mode = 'test'
            anno_path = 'val_split.txt'
            sampling_rate = 1 # args.sampling_rate_val if args.sampling_rate_val > 0 else args.sampling_rate
        else:  
            mode = 'validation'
            anno_path = 'val_split.txt'
            sampling_rate = args.sampling_rate_val if args.sampling_rate_val > 0 else args.sampling_rate

        dataset = FrameClsDataset_DoTA(
            anno_path=anno_path,
            data_path=args.data_path,
            mode=mode,
            view_len=args.num_frames,
            view_step=sampling_rate,
            orig_fps=orig_fps,  # for DoTA
            target_fps=args.view_fps,  # 10
            num_segment=1,
            test_num_segment=args.test_num_segment,
            test_num_crop=1,  # 1
            num_crop=1,
            keep_aspect_ratio=True,
            crop_size=args.input_size,
            short_side_size=args.short_side_size,
            args=args)
        if args.multi_class is True:
            nb_classes = 3
        else:
            nb_classes = 2

    elif args.data_set.startswith('DADA2K'):
        mode = None
        anno_path = None
        orig_fps = 30
        if is_train is True:
            mode = 'train'
            anno_path = 'DADA2K_my_split/half_training.txt' if "_half" in args.data_set else "DADA2K_my_split/training.txt"
            sampling_rate = args.sampling_rate
        elif test_mode is True:
            mode = 'test'
            anno_path = "DADA2K_my_split/validation.txt"
            sampling_rate = args.sampling_rate_val if args.sampling_rate_val > 0 else args.sampling_rate
        else:
            mode = 'validation'
            anno_path = "DADA2K_my_split/validation.txt"
            sampling_rate = args.sampling_rate_val if args.sampling_rate_val > 0 else args.sampling_rate

        dataset = FrameClsDataset_DADA(
            anno_path=anno_path,
            data_path=args.data_path,
            mode=mode,
            view_len=args.num_frames,
            view_step=sampling_rate,
            orig_fps=orig_fps,  # original FPS of the dataset
            target_fps=args.view_fps,  # 10
            num_segment=1,
            test_num_segment=args.test_num_segment,
            test_num_crop=1,  # 1
            num_crop=1,
            keep_aspect_ratio=True,
            crop_size=args.input_size,
            short_side_size=args.short_side_size,
            args=args)
        if args.multi_class is True:
            nb_classes = 3
        else:
            nb_classes = 2

    else:
        raise NotImplementedError()
    assert nb_classes == args.nb_classes
    print("Number of the class = %d" % np.unique(np.array(dataset.label_array)).shape[0])

    return dataset, nb_classes
