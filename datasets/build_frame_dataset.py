import numpy as np
from types import SimpleNamespace
from datasets.dada import FrameClsDataset_DADA
            
def build_frame_dataset(is_train, test_mode, args):
    """Builds a dataset for collision prediction using a specified configuration.
        
        Args:
            is_train (bool): Whether to build the training dataset.
            test_mode (bool): Whether to build the test dataset.
            args (dict): A dictionary containing the configuration parameters for the dataset.

        Returns:
            dataset (Dataset): The constructed dataset based on the provided configuration.
    """
    args = SimpleNamespace(**args)
    
    if args.data_set.startswith('DADA2K'):
        mode = None
        split_file_path = None
        orig_fps = 30
        if is_train is True:
            mode = 'train'
            split_file_path = 'DADA2K_my_split/half_training.txt' if "_half" in args.data_set else "DADA2K_my_split/new_training.txt"
            window_stride = args.window_stride
        elif test_mode is True:
            mode = 'test'
            split_file_path = "DADA2K_my_split/new_validation.txt"
            window_stride = args.window_stride_val if args.window_stride_val > 0 else args.window_stride
        else:
            mode = 'validation'
            split_file_path = "DADA2K_my_split/new_validation.txt"
            window_stride = args.window_stride_val if args.window_stride_val > 0 else args.window_stride
        dataset = FrameClsDataset_DADA(
            split_file_path=split_file_path,
            data_path=args.data_path,
            mode=mode,
            num_frames=args.num_frames,
            window_stride=window_stride,
            orig_fps=orig_fps,  # original FPS of the dataset
            sliding_window_fps=args.sliding_window_fps,  # 10
            crop_size=args.input_size,
            args=args)

    else:
        raise NotImplementedError()
    
    print("NUMBER OF BINS = %d" % np.unique(np.array(dataset.label_array)).shape[0])

    return dataset
