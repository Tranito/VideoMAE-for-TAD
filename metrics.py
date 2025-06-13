import numpy as np
from sklearn.metrics import (
    accuracy_score,
    recall_score,
    precision_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
    roc_curve,
    matthews_corrcoef,
    auc
)


import torch
import torchmetrics
from itertools import groupby
from collections import defaultdict
THRESHOLDS = np.arange(0.00, 1.001, 0.01).tolist()


def calculate_metrics(preds, labels, do_softmax=True):
    if do_softmax:
        preds = torch.nn.functional.softmax(preds, dim=1)
    
    values = preds[:, 1]
    
    _, preds = torch.max(preds, 1)

    metr_acc = torchmetrics.functional.accuracy(preds=preds, target=labels, task="binary").item()
    recall = torchmetrics.functional.recall(preds=preds, target=labels, task="binary").item()
    precision = torchmetrics.functional.precision(preds=preds, target=labels, task="binary").item()
    f1 = torchmetrics.functional.f1_score(preds=preds, target=labels, task="binary").item()
    confmat = torchmetrics.functional.confusion_matrix(preds=preds, target=labels, task="binary").detach().tolist()

    auroc = torchmetrics.functional.auroc(
        preds=values,
        target=labels,
        task="binary",
        thresholds=THRESHOLDS
    ).item()
    ap = torchmetrics.functional.average_precision(
        preds=values,
        target=labels,
        task="binary",
        thresholds=THRESHOLDS
    ).item()
    pr_curve = torchmetrics.functional.precision_recall_curve(
        preds=values,
        target=labels,
        task="binary",
        thresholds=THRESHOLDS
    )
    roc_curve = torchmetrics.functional.roc(
        preds=values,
        target=labels,
        task="binary",
        thresholds=THRESHOLDS
    )
    # MCC
    thresh_metrics = calculate_MORE_metrics(preds=values.detach().cpu(), labels=labels.detach().cpu())
    mcc_thresholded_vals, p_thresholded_vals, r_thresholded_vals, acc_thresholded_vals, f1_thresholded_vals = thresh_metrics[
                                                                                                              -5:]
    mcc_max = max(mcc_thresholded_vals)
    mcc_max_idx = mcc_thresholded_vals.index(mcc_max)
    idx_05 = THRESHOLDS.index(0.5)
    mcc_05 = mcc_thresholded_vals[idx_05]
    mcc_auc = auc(THRESHOLDS, mcc_thresholded_vals)
    return metr_acc, recall, precision, f1, confmat, auroc, ap, pr_curve, roc_curve, (mcc_auc, mcc_max, THRESHOLDS[mcc_max_idx], mcc_05)

def calculate_MORE_metrics(preds, labels):
    """
    Calculate binary classification metrics.
    
    Parameters:
      preds : array-like, shape (n_samples,)
          Predicted probabilities for class 1.
      labels : array-like, shape (n_samples,)
          True binary labels (0 or 1).
    
    Returns:
      metr_acc : float
          Accuracy at threshold 0.5.
      recall_val : float
          Recall at threshold 0.5.
      precision_val : float
          Precision at threshold 0.5.
      f1_val : float
          F1 score at threshold 0.5.
      confmat : list of lists
          Confusion matrix (computed at threshold 0.5).
      auroc : float
          Area under the ROC curve.
      ap : float
          Average precision score.
      pr_curve_vals : tuple
          (precision, recall, thresholds) for the precision-recall curve.
      roc_curve_vals : tuple
          (fpr, tpr, thresholds) for the ROC curve.
      mcc_thresholded_vals : list
          List of Matthews Correlation Coefficient for each threshold.
      p_thresholded_vals : list
          List of precision values for each threshold.
      r_thresholded_vals : list
          List of recall values for each threshold.
    """


    # Ensure preds is a numpy array
    preds = np.array(preds)
    
    # Compute binary predictions using threshold 0.5
    binary_preds = (preds >= 0.5).astype(int)
    
    # Metrics at threshold 0.5
    metr_acc   = accuracy_score(labels, binary_preds)
    recall_val = recall_score(labels, binary_preds)
    precision_val = precision_score(labels, binary_preds)
    f1_val     = f1_score(labels, binary_preds)
    confmat    = confusion_matrix(labels, binary_preds).tolist()
    
    # Threshold-independent metrics (computed using continuous probability values)
    auroc = roc_auc_score(labels, preds)
    ap    = average_precision_score(labels, preds)
    pr_curve_vals = precision_recall_curve(labels, preds)
    roc_curve_vals = roc_curve(labels, preds)
    
    # Compute MCC, precision, and recall for each threshold in THRESHOLDS
    mcc_thresholded_vals = []
    p_thresholded_vals   = []
    r_thresholded_vals   = []
    acc_thresholded_vals = []
    f1_thresholded_vals  = []
    
    for t in THRESHOLDS:
        binary_preds_t = (preds >= t).astype(int)
        mcc_val = matthews_corrcoef(labels, binary_preds_t)
        # Use zero_division=0 to avoid errors if no positives are predicted
        p_val   = precision_score(labels, binary_preds_t, zero_division=0)
        r_val   = recall_score(labels, binary_preds_t, zero_division=0)
        acc_val = accuracy_score(labels, binary_preds_t)
        f1_val  = f1_score(labels, binary_preds_t, zero_division=0)
        mcc_thresholded_vals.append(mcc_val)
        p_thresholded_vals.append(p_val)
        r_thresholded_vals.append(r_val)
        acc_thresholded_vals.append(acc_val)
        f1_thresholded_vals.append(f1_val)
    
    return (metr_acc, precision_val, recall_val, f1_val, ap, auroc, 
            confmat, pr_curve_vals, roc_curve_vals,
            mcc_thresholded_vals, p_thresholded_vals, r_thresholded_vals, 
            acc_thresholded_vals, f1_thresholded_vals)

def calculate_metrics_multi_class(preds, labels, do_softmax=True, multi_class=False):
    
    multi_class = multi_class
    if do_softmax:
        preds = torch.nn.functional.softmax(preds, dim=1)

    # print(f"preds in calculate_metrics_multi_class: {preds}, labels: {labels}")

    if multi_class:
        values = preds
        task = "multiclass"
        average = "macro"
    else:
        values = preds[:, 1]
        task = "binary"
        average = "binary"
    
    _, preds = torch.max(preds, 1)

    metr_acc = torchmetrics.functional.accuracy(preds=preds, target=labels, task=task , average=average).item()
    f1 = torchmetrics.functional.f1_score(preds=preds, target=labels, task=task, average=average).item()

    auroc = torchmetrics.functional.auroc(
        preds=values,
        target=labels,
        task=task,
        average=average,
        thresholds=THRESHOLDS
    ).item()
    ap = torchmetrics.functional.average_precision(
        preds=values,
        target=labels,
        task=task,
        average=average,
        thresholds=THRESHOLDS
    ).item()

    # MCC
    mcc_thresholded_vals, per_class_metrics = calculate_MORE_metrics_multi_class(preds=values.detach().cpu(), 
                                                                                 labels=labels.detach().cpu(), 
                                                                                 multi_class=multi_class)

    mcc_max = max(mcc_thresholded_vals)
    mcc_max_idx = mcc_thresholded_vals.index(mcc_max)
    idx_05 = THRESHOLDS.index(0.5)
    mcc_05 = mcc_thresholded_vals[idx_05]
    mcc_auc = auc(THRESHOLDS, mcc_thresholded_vals)
    return metr_acc, f1, auroc, ap, (mcc_auc, mcc_max, mcc_05), per_class_metrics

def calculate_MORE_metrics_multi_class(preds, labels, multi_class=False):
    """
    Calculate binary classification metrics.
    
    Parameters:
      preds : array-like, shape (n_samples,)
          Predicted probabilities for class 1.
      labels : array-like, shape (n_samples,)
          True binary labels (0 or 1).
    
    Returns:
      mcc_thresholded_vals : list
          List of Matthews Correlation Coefficient for each threshold.
      ap_per_class : list
          AP per class at threshold 0.5.
      auroc_per_class : list
          Area under the ROC curve per class.
      acc_per_class : list
          Accuracy per class at threshold 0.5.
      recall_per_class_t : list
          Recall per class for different thresholds.
    """

    # Ensure preds is a numpy array
    preds = np.array(preds)

    # print(f"preds in calculate_MORE_metrics_multi_class: {preds}, labels: {labels}")

    # Compute MCC, precision, and recall for each threshold in THRESHOLDS
    mcc_thresholded_vals = []
    
    for t in THRESHOLDS:
        binary_preds_t = (preds >= t).astype(int)
        mcc_val = matthews_corrcoef(labels, binary_preds_t)
        mcc_thresholded_vals.append(mcc_val)
    
    if multi_class:
        # For multi-class, we need to
        ap_per_class = []
        auroc_per_class = []
        acc_per_class = []
        recall_per_class_t = []
        recall_t = []

        for i in range(preds.shape[1]):

            # extract predictions and labels for the current class
            preds_class = preds[:, i]
            labels_class = labels == i

            # compute binary predictions using threshold 0.5
            binary_preds = (preds_class >= 0.5).astype(int)
            
            # metrics at threshold 0.5
            metr_acc   = accuracy_score(labels_class, binary_preds)
            acc_per_class.append(metr_acc)
            
            # threshold-independent metrics (computed using continuous probability values)
            auroc = roc_auc_score(labels_class, preds_class)
            auroc_per_class.append(auroc)

            ap    = average_precision_score(labels_class, preds_class)
            ap_per_class.append(ap)

            for t in THRESHOLDS:
                binary_preds_t = (preds_class >= t).astype(int)
                recall = torchmetrics.functional.recall(preds=binary_preds_t, target=labels_class, task="binary").item()
                recall_t.append(recall)

            recall_per_class_t.apppend(recall_t)
        return mcc_thresholded_vals, (ap_per_class, auroc_per_class, acc_per_class, recall_per_class_t)
    else:
        return mcc_thresholded_vals, None

def calculate_tta(clip_predictions, clip_infos, fps=10, threshold = 0.5):
    """
    Calculate time-to-accident (TTA) based on clip predictions and time-of-accident frame (TOA)

    Args:
    ----------
        clip_predictions (dict): Dictionary with clipID as keys and a list of predictions as values
        clip_infos (tuple): Tuple containing clip_id and toa
        fps (int): Frames per second of the video
        threhsold (float): Threshold to classify prediction as an anomaly

    Returns:
    ----------
        mtta (float): mean time-to-accident across all positive (accident) clips
        mean_itta (float): mean immanent time-to-accident across all positive (accident) clips
    """

    # data preprocessing
    preds = clip_predictions[:, 1].detach().cpu()
    grouped = defaultdict(list)
    for pred, (clip_id, toa) in zip(preds, clip_infos):
        grouped[(int(clip_id), int(toa))].append(pred)
    
    time, counter, time_itta, counter_itta = 0.0, 0, 0.0, 0

    for (clip_id, toa), clip_predictions in grouped.items():
        predictions = np.array(clip_predictions)

        # tta
        pred_bins = (predictions >= threshold).astype(int)
        idx_first = np.where(pred_bins == 1)[0]

        if (toa != -1) and len(idx_first) > 0:
            time += max( (toa - (idx_first[0] + 45)) / fps, 0)
            counter += 1

        # itta
        # select last frame which is below threshold before toa
        pred_bins_itta = (predictions[:toa] < threshold).astype(int)

        # if no frame below the threshold before the toa, iTTA is the same as the TTA
        if toa!= -1:
            if np.all(pred_bins_itta == 0):
                time_itta += max((toa - (idx_first[0] + 45)) / fps, 0)
                counter_itta += 1

            elif np.any(pred_bins_itta == 1):
                idx_first_itta = np.where(pred_bins_itta == 1)[0][-1]

                # if the last frame is frame before toa, model is not able to predict an accident
                if idx_first_itta < (toa - 1):
                    time_itta += max((toa - (idx_first_itta + 1 + 45)) / fps, 0)

                counter_itta += 1

    mtta = time / counter if counter > 0 else 0
    mean_itta = time_itta / counter_itta if counter_itta > 0 else 0

    return mtta, mean_itta

def accuracy_per_10_frames(preds, clip_infos, labels, threshold=0.5):
    grouped = defaultdict(list)
    labels_grouped = defaultdict(list)
    clip_infos = torch.tensor(clip_infos).cpu() if type(clip_infos[0]) == tuple else torch.cat(clip_infos).cpu()

    preds = torch.nn.functional.softmax(preds, dim=1)
    
    # if binary classification take only predictions of accident class
    if preds.shape[1] == 2:
        preds = preds[:,1]

    # Group predictions and labels by clip
    if clip_infos.ndim > 1:
        for pred, (clip_id, toa), label in zip(preds, clip_infos, labels):
            grouped[(int(clip_id), int(toa))].append(pred)
            labels_grouped[(int(clip_id), int(toa))].append(label)
    else:
        for pred, clip_id, label in zip(preds, clip_infos, labels):
            grouped[int(clip_id)].append(pred)
            labels_grouped[int(clip_id)].append(label)

    # For each interval, keep running sum of correct and total
    correct_counts = defaultdict(int)
    total_counts = defaultdict(int)

    for clip_info, item in grouped.items():
        labels_item = labels_grouped[clip_info]

        if item[0].ndim > 0:
            # for multi-class classification, take the class with the highest probability
            item_argmax = torch.argmax(torch.tensor(item), dim=1)
            item_bins = np.array([argmax if item[clipID][argmax] >= 0.5 else 0 for clipID, argmax in zip(range(len(item)), item_argmax)])
        else:
            item_bins = (np.array(item) >= threshold)

        labels_arr = np.array(labels_item)

        n = len(item_bins)
        for interval in range(0, n - 10 + 1, 10):
            preds_slice = item_bins[interval:interval+10]
            labels_slice = labels_arr[interval:interval+10]
            correct = np.sum(preds_slice == labels_slice)
            correct_counts[(interval, interval+9)] += correct
            total_counts[(interval, interval+9)] += preds_slice.size

    # Compute mean accuracy for each interval
    mean_accuracy = {}
    for interval in correct_counts:
        mean_accuracy[interval] = correct_counts[interval] / total_counts[interval] if total_counts[interval] > 0 else np.nan
    return mean_accuracy

def accuracy_per_frame(preds, clip_infos, labels, threshold=0.5):
    """
    Returns the accuracy per frame with respect to the start of the anomaly window

    Args:
    ----------
        preds (tensor): Tensor of all predictions from the dataset
        clip_infos (list): List of clip information of each prediction
        labels (tensor): Tensor of labels for each prediction        
        threhsold (float): Threshold to classify prediction as an anomaly

    Returns:
    ----------
        accuracy_per_frame (dict): Dictionary with frame distance to anomaly start as keys and mean accuracy as values
    """

    grouped = defaultdict(list)
    labels_grouped = defaultdict(list)
    clip_infos = torch.tensor(clip_infos).cpu() if type(clip_infos[0]) == tuple else torch.cat(clip_infos).cpu()

    preds = torch.nn.functional.softmax(preds, dim=1)
    
    # if binary classification take only predictions of accident class
    if preds.shape[1] == 2:
        preds = preds[:,1]

    # Group predictions and labels by clip
    if clip_infos.ndim > 1:
        for pred, (clip_id, toa), label in zip(preds, clip_infos, labels):
            grouped[(int(clip_id), int(toa))].append(pred)
            labels_grouped[(int(clip_id), int(toa))].append(label)
    else:
        for pred, clip_id, label in zip(preds, clip_infos, labels):
            grouped[int(clip_id)].append(pred)
            labels_grouped[int(clip_id)].append(label)

    # For each interval, keep running sum of correct and total
    correct_counts = defaultdict(int)
    total_counts = defaultdict(int)

    for clip_info, item in grouped.items():
        labels_item = labels_grouped[clip_info]

        if item[0].ndim > 0:
            # for multi-class classification, take the class with the highest probability
            item_argmax = torch.argmax(torch.tensor(item), dim=1)
            item_bins = np.array([argmax if item[clipID][argmax] >= 0.5 else 0 for clipID, argmax in zip(range(len(item)), item_argmax)])
        else:
            item_bins = (np.array(item) >= threshold)

        labels_arr = np.array(labels_item)

        anomaly_start_idx = np.where(labels_arr > 0)[0][0]  # get index where anomaly window starts
        for frame_idx in range(len(item_bins)):

            distance = frame_idx - anomaly_start_idx
            # print(f"item_bins[frame_idx]: {item_bins[frame_idx]}, labels_arr[frame_idx]: {labels_arr[frame_idx]}")
            # print(item_bins[frame_idx] == labels_arr[frame_idx])
            correct = int(item_bins[frame_idx] == labels_arr[frame_idx])
            # print(int(item_bins[frame_idx] == labels_arr[frame_idx]))
            correct_counts[distance] += correct
            total_counts[distance] += 1

    # Compute mean accuracy for each distance
    accuracy_per_distance = {}
    for distance in correct_counts:
        accuracy_per_distance[distance] = correct_counts[distance] / total_counts[distance] if total_counts[distance] > 0 else np.nan

    # sort dictionary by distance
    accuracy_per_distance = {k: accuracy_per_distance[k] for k in sorted(accuracy_per_distance)}
    return accuracy_per_distance