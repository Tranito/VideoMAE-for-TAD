import cv2
import numpy as np
import torch
import pandas as pd
from natsort import natsorted
from torchvision import transforms
import warnings
from torch.utils.data import Dataset
from tqdm import tqdm
import warnings
from datasets.random_erasing import RandomErasing
import datasets.video_transforms as video_transforms 
import datasets.volume_transforms as volume_transforms
from datasets.dataset_loading.sequencing import RegularSequencer
import os
from datasets.transform.tensor_normalize import tensor_normalize
# import simplejpeg
from concurrent.futures import ThreadPoolExecutor
from collections import OrderedDict



class FrameClsDataset_DADA(Dataset):
    """Loads the DADA-2000 train and validation set for collision prediction. 
       Each sample corresponds to a sliding window of num_frames frames, 
       sampled at sliding_window_fps from the original video with orig_fps frames per second.
       
       Parameters:
        split_file_path (str): Path to the split file listing the video clips to use.
        data_path (str): Root path to the dataset containing the video frames and annotations.
        mode (str): One of 'train', 'validation', or 'test' to specify
        num_frames (int): The number of frames in each sliding window sample.
        sliding_window_fps (int): The frames per second at which to sample frames for the sliding window.
        orig_fps (int): The original frames per second of the videos in the dataset
        window_stride (int): The number of frames between the start of consecutive sliding windows.
        video_ext (str): The file extension of the video frames (default: ".png").
        crop_size (int): The size to which each frame will be cropped/resized (default: 224).

       Returns:
        buffer (Tensor): A tensor of shape (C, T, H, W) containing the sliding window frames.
        label (int or float): The bin label for the sample.
        index (int): The index of the sample in the dataset.
        clipID (int): The ID of the original video clip this sample belongs to. 
                      This ID corresponds to the index of the clip in the new_training/new_validation split 
        clip_toa (int): The frame number of the collision in the original video clip.
        sample_ttc_label (float, optional): The TTC label for the sample if regression is used, otherwise an empty list.
       """
    ego_categories = [str(cat) for cat in list(range(1, 19)) + [61, 62]]

    def __init__(self, split_file_path, data_path, mode='train',
                 num_frames=8, sliding_window_fps=10, orig_fps=30, window_stride=10,
                 crop_size=224, video_ext=".png",
                 args=None):
        self.split_file_path = split_file_path
        self.data_path = data_path
        self.mode = mode
        self.num_frames = num_frames
        self.sliding_window_fps = sliding_window_fps
        self.orig_fps = orig_fps
        self.window_stride = window_stride
        self.crop_size = crop_size
        self.video_ext = video_ext
        self.args = args
        self.aug = False
        self.rand_erase = False
        self.regression = args.regression
        self.alpha = args.alpha
        self.bin_width = args.bin_width

        assert (self.bin_width*self.orig_fps) % 1 == 0, \
                        f"bin_width must result in an integer number of frames at {self.orig_fps} fps (got {self.bin_width*self.orig_fps} frames)"

        if self.mode in ['train']:
            self.aug = True
            if self.args.reprob > 0:
                self.rand_erase = True

        self.image_cache = OrderedDict()
        self.max_cache_size = 1000

        self._read_anno()
        self._prepare_views()
        assert len(self.dataset_samples) > 0
        assert len(self._label_array) > 0

        self.label_array = self._label_array
        if mode == "train":
            print(40*"="+"\nBIN DISTRIBUTION TRAINING SET:")
        else:
            print(40*"="+"\nBIN DISTRIBUTION VALIDATION SET:")
        unique_labels, counts = np.unique(np.array(self._label_array), return_counts=True)
        for label, count in zip(unique_labels, counts):
            if label == 0:
                print(f"No Coll. Soon: {count} samples")
            else:
                print(f"Coll. within {label*self.bin_width}s: {count} samples")
        print(40*"=")


        if (mode == 'train'):
            pass

        elif (mode == 'validation'):
            print("mode is validation.")
            self.data_transform = video_transforms.Compose([
                volume_transforms.ClipToTensor(),
                 video_transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                            std=[0.229, 0.224, 0.225])
            ])
        elif mode == 'test':
            self.data_resize = video_transforms.Compose([
                video_transforms.Resize(size=(self.crop_size, self.crop_size), interpolation='bilinear')
            ])
            self.data_transform = video_transforms.Compose([
                volume_transforms.ClipToTensor(),
                 video_transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                            std=[0.229, 0.224, 0.225])
            ])
            self.test_seg = [(0, 0)]
            self.test_dataset = self.dataset_samples
            self.test_label_array = self.label_array

    def _read_anno(self):
        clip_timesteps = []
        clip_bin_labels = []
        clip_ego = []
        clip_night = []
        clip_toa = []
        clip_acc = []
        clip_ttc_label = []
        errors = []

        with open(os.path.join(self.data_path, self.split_file_path), 'r') as file:
            clip_names = [line.rstrip() for line in file]
        df = pd.read_csv(os.path.join(self.data_path, "annotation", "full_anno.csv"))

        for clip in tqdm(clip_names, "Part 1/2. Reading and checking clips"):
            clip_type, clip_subfolder = clip.split("/")
            row = df[(df["video"] == int(clip_subfolder)) & (df["type"] == int(clip_type))]
            info = f"clip: {clip}, type: {clip_type}, subfolder: {clip_subfolder}, rows found: {row}"
            description_csv = row["texts"]
            assert len(row) == 1, f"Multiple results! \n{info}"
            if len(row) != 1:
                errors.append(info)
            row = row.iloc[0]
            frames_dir = os.path.join(self.data_path, "frames", clip, "images")
            framenames = natsorted([f for f in os.listdir(frames_dir) if os.path.splitext(f)[1] == self.video_ext])
            timesteps = natsorted([int(os.path.splitext(f)[0].split("_")[-1]) for f in framenames])
            toa = int(row["accident frame"])

            # only select the timesteps up to and including the frame before collision
            timesteps = timesteps[:toa-1]
            if_acc_video = int(row["whether an accident occurred (1/0)"])

            bin_label = []
            if toa > -1:
                if self.regression:
                    ttc_labels = [t/30 for t in range(1, len(timesteps)+1)][::-1]

                bin_label = []

                # create bin labels
                for t in range(1, len(timesteps)+1):
                    frame_difference = toa - t

                    # outside the prediction window
                    if frame_difference > self.alpha*self.orig_fps:
                        bin_label.append(0)
                    else:
                        if frame_difference % (self.bin_width*self.orig_fps) == 0:
                            # bin boundaries are (t_lower, t_upper], where a frame on the upper boundary still belons to that bin
                            bin_label.append( int(frame_difference // (self.bin_width*self.orig_fps)) )
                        else:
                            bin_label.append( int(frame_difference // (self.bin_width*self.orig_fps)) + 1 )
                                            
            if_ego = clip_type in self.ego_categories
            if_night = int(row["light(day,night)1-2"]) == 2

            clip_timesteps.append(timesteps)
            clip_bin_labels.append(bin_label)
            clip_ego.append(if_ego)
            clip_night.append(if_night)
            clip_toa.append(toa)
            clip_acc.append(if_acc_video)
            clip_ttc_label.append(ttc_labels) if self.regression else clip_ttc_label.append([])

        for line in errors:
            print(line)
        if len(errors) > 0:
            print(f"\n====\nerrors: {len(errors)}. You can add saving the error list in the code.")
            exit(0)

        assert len(clip_names) == len(clip_timesteps) == len(clip_bin_labels)
        
        clip_acc = np.array(clip_acc)
        # only keep videos with a collision
        valid_idx = np.where(clip_acc == 1)[0]

        self.clip_names = [clip_names[i] for i in valid_idx]
        self.clip_timesteps = [clip_timesteps[i] for i in valid_idx]
        self.clip_bin_labels = [clip_bin_labels[i] for i in valid_idx]
        self.clip_ego = [clip_ego[i] for i in valid_idx]
        self.clip_night = [clip_night[i] for i in valid_idx]
        self.clip_toa = [clip_toa[i] for i in valid_idx]
        self.clip_ttc_label = [clip_ttc_label[i] for i in valid_idx] if self.regression else None

    def _prepare_views(self):
        dataset_sequences = []
        label_array = []
        ttc_label = []

        sequencer = RegularSequencer(seq_frequency=self.sliding_window_fps, seq_length=self.num_frames, step=self.window_stride)
        N = len(self.clip_names)
        for i in tqdm(range(N), desc="Part 2/2. Preparing views"):
            timesteps = self.clip_timesteps[i]
            sequences = sequencer.get_sequences(timesteps_nb=len(timesteps), input_frequency=self.orig_fps)
            if sequences is None:
                continue
            dataset_sequences.extend([(i, seq) for seq in sequences])
            label_array.extend([self.clip_bin_labels[i][seq[-1]] for seq in sequences])
            if self.regression:
                ttc_label.extend([self.clip_ttc_label[i][seq[-1]] for seq in sequences])
            else:
                ttc_label.extend([])
        self.dataset_samples = dataset_sequences
        self._label_array = label_array
        self.sample_ttc_label = ttc_label


    def __getitem__(self, index):
        if self.mode == 'train':
            args = self.args
            sample = self.dataset_samples[index]
            buffer, _, __ = self.load_images(sample, keep_aspect_ratio=True)  # T H W C
            if len(buffer) == 0:
                while len(buffer) == 0:
                    warnings.warn("video {} not correctly loaded during training".format(sample))
                    index = np.random.randint(self.__len__())
                    sample = self.dataset_samples[index]
                    buffer, _, __ = self.load_images(sample, keep_aspect_ratio=True)

            if args.num_sample > 1:
                frame_list = []
                label_list = []
                index_list = []
                clipID_list = []
                clip_toa_list = []

                for _ in range(args.num_sample):
                    new_frames = self._aug_frame(buffer, args)
                    label = self.label_array[index]
                    clipID = self.dataset_samples[index][0]
                    clip_toa = self.clip_toa[clipID]
                    clipID_list.append(clipID)
                    clip_toa_list.append(clip_toa)
                    frame_list.append(new_frames)
                    label_list.append(label)
                    index_list.append(index)

                return frame_list, label_list, index_list, clipID_list, clip_toa_list
            else:
                buffer = self._aug_frame(buffer, args)
            clipID = self.dataset_samples[index][0]
            sample_ttc_label = self.sample_ttc_label[index] if self.regression else []
            return buffer, self.label_array[index], index, clipID, self.clip_toa[clipID], sample_ttc_label

        elif self.mode == 'validation':
            sample = self.dataset_samples[index]
            buffer, _, __ = self.load_images(sample, keep_aspect_ratio=True)
            if len(buffer) == 0:
                while len(buffer) == 0:
                    warnings.warn("video {} not correctly loaded during validation".format(sample))
                    index = np.random.randint(self.__len__())
                    sample = self.dataset_samples[index]
                    buffer, _, __ = self.load_images(sample, keep_aspect_ratio=True)
            do_pad = video_transforms.pad_wide_clips(buffer[0].shape[0], buffer[0].shape[1], self.crop_size)
            buffer = [do_pad(img) for img in buffer]       
            buffer = self.data_transform(buffer)
            clipID = self.dataset_samples[index][0]
            sample_ttc_label = self.sample_ttc_label[index] if self.regression else []

            return buffer, self.label_array[index], index, clipID, self.clip_toa[clipID], sample_ttc_label
        elif self.mode == 'test':
            sample = self.test_dataset[index]
            buffer, clip_name, frame_name = self.load_images(sample, keep_aspect_ratio=True)
            while len(buffer) == 0:
                warnings.warn("video {} not found during testing".format(str(self.test_dataset[index])))
                index = np.random.randint(self.__len__())
                sample = self.test_dataset[index]
                buffer, clip_name, frame_name = self.load_images(sample, keep_aspect_ratio=True)
            do_pad = video_transforms.pad_wide_clips(buffer[0].shape[0], buffer[0].shape[1], self.crop_size)
            buffer = [do_pad(img) for img in buffer]
            buffer = self.data_transform(buffer)
            clipID = self.dataset_samples[index][0]
            sample_ttc_label = self.sample_ttc_label[index] if self.regression else []
                
            return buffer, self.label_array[index], index, clipID, self.clip_toa[clipID], sample_ttc_label
        else:
            raise NameError('mode {} unkown'.format(self.mode))

    # data augmentation for training frames
    def _aug_frame(
        self,
        buffer,
        args,
    ):
        h, w, _ = buffer[0].shape
        do_pad = video_transforms.pad_wide_clips(h, w, self.crop_size, is_train=True)
        buffer = [do_pad(img) for img in buffer]

        aug_transform = video_transforms.create_random_augment(
            input_size=(self.crop_size, self.crop_size),
            auto_augment=args.aa,
            interpolation=args.train_interpolation,
            do_transforms=video_transforms.DRIVE_TRANSFORMS
        )

        buffer = [transforms.ToPILImage()(frame) for frame in buffer]
        buffer = aug_transform(buffer)
        buffer = [transforms.ToTensor()(img) for img in buffer]
        buffer = torch.stack(buffer) # T C H W
        buffer = buffer.permute(0, 2, 3, 1) # T H W C
        # T H W C 
        buffer = tensor_normalize(
            buffer, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        )
        # T H W C -> C T H W.
        buffer = buffer.permute(3, 0, 1, 2)

        if self.rand_erase:
            erase_transform = RandomErasing(
                args.reprob,
                mode=args.remode,
                max_count=args.recount,
                num_splits=args.recount,
                device="cpu",
            )
            buffer = buffer.permute(1, 0, 2, 3)
            buffer = erase_transform(buffer)
            buffer = buffer.permute(1, 0, 2, 3)

        return buffer
    
    def load_and_resize(self, file_path, crop_size, keep_aspect_ratio=True):
        
        if len(self.image_cache) > self.max_cache_size:
            self.image_cache.popitem(last=False)
        
        # store frequently loaded images in cache during validation to reduce loading times
        if file_path in self.image_cache and self.mode != "train":
            img = self.image_cache.pop(file_path)
            self.image_cache[file_path] = img
            return img
        else:
            img = cv2.imread(file_path)
            
            if keep_aspect_ratio == True:
                width = crop_size
                height = int(img.shape[0] * width / img.shape[1])
                img = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.uint8)
                if self.mode != "train":
                    self.image_cache[file_path] = img
            else:
                img = cv2.resize(img, (crop_size, crop_size), interpolation=cv2.INTER_AREA)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.uint8)
                if self.mode != "train":
                    self.image_cache[file_path] = img
            return img

    def load_images(self, dataset_sample, keep_aspect_ratio=True):
        clip_id, frame_seq = dataset_sample
        clip_name = self.clip_names[clip_id]
        timesteps = [self.clip_timesteps[clip_id][idx] for idx in frame_seq]
        filenames = [f"{str(ts).zfill(4)}{self.video_ext}" for ts in timesteps]
        clip_path = os.path.join(self.data_path, "frames", clip_name, "images")
        file_paths =  [os.path.join(clip_path, fname) for fname in filenames]
        with ThreadPoolExecutor() as executor:
            imgs = list(
                executor.map(
                    lambda fb: self.load_and_resize(
                        fb,
                        self.crop_size,
                        keep_aspect_ratio=keep_aspect_ratio
                    ),
                    file_paths,
                )
            )
        return imgs, clip_name, filenames[-1]

    def __len__(self):
        if self.mode != 'test':
            return len(self.dataset_samples)
        else:
            return len(self.test_dataset)