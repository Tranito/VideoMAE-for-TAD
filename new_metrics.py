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
