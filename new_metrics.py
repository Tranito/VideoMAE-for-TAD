import numpy as np
import torch
import torchmetrics
from itertools import groupby
from collections import defaultdict

def accuracy_per_bin_regression(preds, clip_infos, alpha=5, bin_width=1, fps=30):
    """
    Computes per-bin accuracy for the regression model.

    Unlike classification, regression cannot use PyTorch's MultiClassAccuracy,
    because it outputs continuous TTC values rather than discrete bin logits.
    Bin accuracy is defined as the fraction of TTC predictions that fall within the
    ground-truth bin's TTC interval.
    
    Args:
        preds (tensor): tensor of predicted TTC values
        clip_infos (tensor): tensor of tuples (clip_id, toa) for each prediction
        alpha (int): the threshold for separating No Collision Soon and Collision Soon region
        bin_width (float): the width of each bin in seconds
        fps (int): frames per second of the video

    Returns:
        mean_accuracy_per_bin: dict mapping bin labels to mean accuracy for that bin across the dataset
    """
    assert type(alpha) == int, "alpha must be an integer"
    assert alpha > 0, "alpha must be greater than 0"

    accumulative_acc_per_bin = defaultdict(int)
    bin_count = defaultdict(int)
    mean_accuracy_per_bin = dict()
    preds_grouped = defaultdict(list)

    # group predictions per video using the clip id (video identifier) and toa (time of accident) information
    for pred, (clip_id, toa) in zip(preds, clip_infos):
        preds_grouped[(int(clip_id), int(toa))].append(pred)

    # reverse predictions to calculate bin accuracy starting from bin closest to collision to No Collision Soon bin
    for clip_info, preds in preds_grouped.items():
        preds_reversed = torch.flip(torch.tensor(preds), dims=(0,))

        # group the data into bins of (bin_width*fps) frames until reaching the alpha threshold and group the rest into the last bin (No Coll. Soon)
        # then compute accuracy per bin as the fraction of predictions falling between the bin's TTC boundaries
        if len(preds_reversed) <= alpha*fps:
            preds_subsets = [preds_reversed[i:i+int(fps*bin_width)] for i in range(0, len(preds_reversed), int(fps*bin_width))]
        else:
            subsets = [preds_reversed[i:i+int(fps*bin_width)] for i in range(0, int(alpha*fps), int(fps*bin_width))]
            last_subset = preds_reversed[int(alpha*fps):]
            preds_subsets = subsets + [last_subset]

        for i in range(0, len(preds_subsets)):
            if i*bin_width < int(alpha):
                mask = (preds_subsets[i] >=i*bin_width) & (preds_subsets[i] < (i+1)*bin_width)
            else:
                mask = (preds_subsets[i] >= alpha)

            accuracy = sum(mask)/len(mask) if len(mask) > 0 else 0
            accumulative_acc_per_bin[i] += accuracy
            bin_count[i] += 1

    # compute mean accuracy per bin across the dataset
    for key, value in bin_count.items():
        if key == int(alpha/bin_width):
            mean_accuracy_per_bin[f"no_coll_soon"] = accumulative_acc_per_bin[key]/value if value > 0 else 0
        else:
            mean_accuracy_per_bin[f"coll_{(key+1)*bin_width:.2f}s"] = accumulative_acc_per_bin[key]/value if value > 0 else 0        

    return mean_accuracy_per_bin
