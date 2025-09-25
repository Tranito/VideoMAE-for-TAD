import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
)

import torch
import torchmetrics
from itertools import groupby
from collections import defaultdict
THRESHOLDS = np.arange(0.00, 1.001, 0.01).tolist()


def metrics(preds, labels, do_softmax=True, num_classes=2):
    
    if do_softmax:
        preds = torch.nn.functional.softmax(preds, dim=1)

    values = preds
    task = "multiclass"
    average = "macro"
    num_classes = num_classes
    
    _, preds = torch.max(preds, 1)

    metr_acc = torchmetrics.functional.accuracy(preds=preds, target=labels, task=task , average=average, num_classes=num_classes).item()
    f1 = torchmetrics.functional.f1_score(preds=preds, target=labels, task=task, average=average, num_classes=num_classes).item()
    confmat = torchmetrics.functional.confusion_matrix(preds=preds, target=labels, task=task, num_classes=num_classes).detach()

    auroc = torchmetrics.functional.auroc(
        preds=values,
        target=labels,
        task=task,
        average=average,
        thresholds=THRESHOLDS,
        num_classes=num_classes,
    ).item()
    ap = torchmetrics.functional.average_precision(
        preds=values,
        target=labels,
        task=task,
        average=average,
        thresholds=THRESHOLDS,
        num_classes=num_classes,
    ).item()

    return metr_acc, f1, auroc, ap, confmat

def prediction_lead_time(preds, labels):

    preds_softmax = torch.nn.functional.softmax(preds, dim=1)
    print(f"Predictions shape: {preds_softmax.shape}, Labels: {np.unique(labels)}")
    accuracy_per_label  = torchmetrics.functional.accuracy(preds=preds_softmax, 
                                                           target=labels, 
                                                           task="multiclass" , 
                                                           average=None, 
                                                           num_classes=preds_softmax.shape[1])
    below_threshold = torch.where(accuracy_per_label < 0.5)[0]
    if len(below_threshold) == 0:
        pred_lead_time = 0
    else:
        index = below_threshold[0].item()
        pred_lead_time = index - 1

    return accuracy_per_label, pred_lead_time

def mean_mae_per_label(preds, labels):

    mean_mae_per_label = []
    print(labels)
    for label in np.unique(labels):
        mask = labels == label
        preds_label = preds[mask]
        labels_label = labels[mask]

        mean_mae = np.abs(preds_label - labels_label).mean()
        mean_mae_per_label.append(mean_mae)

    return torch.tensor(mean_mae_per_label)

def accuracy_per_bin_regression(preds, clip_infos, ttc_boundary=5):
  """Calculates the accuracy per label bin across the whole dataset"""
  assert type(ttc_boundary) == int, "ttc_boundary must be an integer"
  assert ttc_boundary > 0, "ttc_boundary must be greater than 0"

  accumulative_acc_per_bin = defaultdict(int)
  bin_count = defaultdict(int)
  mean_accuracy_per_bin = dict()
  preds_grouped = defaultdict(list)

  # group predictions per clip

  for pred, (clip_id, toa) in zip(preds, clip_infos):
    preds_grouped[(int(clip_id), int(toa))].append(pred)

  for clip_info, preds in preds_grouped.items():
    preds_reversed = torch.flip(torch.tensor(preds), dims=(0,))

    if len(preds_reversed) <= ttc_boundary*30:
      preds_subsets = [preds_reversed[i:i+30] for i in range(0, len(preds_reversed), 30)]
    else:
      subsets = [preds_reversed[i:i+30] for i in range(0, int(ttc_boundary*30), 30)]
      last_subset = preds_reversed[int(ttc_boundary*30):]
      preds_subsets = subsets + [last_subset]
      # print(len(preds_subsets))

    for i in range(0, len(preds_subsets)):
      if i < int(ttc_boundary):
          mask = (preds_subsets[i] >=i) & (preds_subsets[i] < i+1 )
      else:
          mask = (preds_subsets[i] >= ttc_boundary)

      accuracy = sum(mask)/len(mask)
      accumulative_acc_per_bin[i] += accuracy
      bin_count[i] += 1

  for key, value in bin_count.items():
    mean_accuracy_per_bin[key] = accumulative_acc_per_bin[key]/value

  return mean_accuracy_per_bin
