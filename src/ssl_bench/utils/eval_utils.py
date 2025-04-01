from tqdm import tqdm

import logging
import numpy as np
import os
import pickle

import torch
from sklearn.metrics import auc, roc_curve
from sklearn.metrics import confusion_matrix as sklearn_cm
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_auc_score

from ssl_bench.utils.train_utils import AverageMeter


def eval_model(args, data_loader, model, weights=None):
    """Evaluate the model on the given data_loader.

    Args:
        args (argparse.ArgumentParser): Input arguments
        data_loader (torch.utils.data.DataLoader): DataLoader for the dataset
        model (torch.nn.Module): Model to evaluate
        weights (torch.Tensor): Weights for the loss function

    Returns:
        dict: Dictionary containing the evaluation metrics
    """
    model.eval()
    loss_meter = AverageMeter()
    data_loader = tqdm(data_loader, disable=False)

    weights = weights.to(args.device) if weights is not None else None

    with torch.no_grad():
        all_labels, all_probs = [], []

        for inputs, labels in data_loader:
            inputs, labels = inputs.to(args.device).float(), labels.to(args.device).long()
            probs, loss = model.eval_forward(inputs, labels)
            all_probs.append(probs)
            all_labels.append(labels)

            loss_meter.update(loss.item(), inputs.size(0))

        all_labels = torch.cat(all_labels).cpu().numpy()
        all_probs = torch.cat(all_probs).cpu().numpy()

        data_loader.close()

    metrics = evaluate_all_metrics(all_probs, all_labels)
    metrics['loss'] = loss_meter.avg

    return metrics


def evaluate_all_metrics(probs, labels):
    preds = probs.argmax(axis=1)
    return {
        'plain_accuracy': calculate_plain_accuracy(preds, labels),
        'balanced_accuracy': calculate_balanced_accuracy(preds, labels),
        'auroc': calculate_auroc(probs, labels),
        'auprc': calculate_auprc(probs, labels),
        'tpr_at_fpr_5': calculate_tpr_at_fpr_5(probs, labels),
    }


def calculate_plain_accuracy(preds, labels):
    return (preds == labels).mean() * 100


def calculate_balanced_accuracy(preds, labels):
    cm = sklearn_cm(labels, preds)
    num_classes = cm.shape[0]

    recalls = []
    for i in range(num_classes):
        recall = cm[i, i] / np.sum(cm[i]) if np.sum(cm[i]) > 0 else 0
        recalls.append(recall)

    return np.mean(recalls) * 100


def calculate_auroc(probs, labels):
    if probs.shape[1] == 2:
        return roc_auc_score(labels, probs[:, 1]) * 100
    return roc_auc_score(labels, probs, multi_class="ovr") * 100


def calculate_auprc(probs, labels):
    if probs.shape[1] == 2:
        precision, recall, _ = precision_recall_curve(labels, probs[:, 1])
        return auc(recall, precision) * 100

    auprc_score = 0
    for class_idx in range(probs.shape[1]):
        class_targets = (labels == class_idx).astype(int)
        precision, recall, _ = precision_recall_curve(class_targets, probs[:, class_idx])
        auprc_score += auc(recall, precision)
    return (auprc_score / probs.shape[1]) * 100


def calculate_tpr_at_fpr_5(probs, labels):
    num_classes = probs.shape[1]
    tpr_scores = []

    for class_idx in range(num_classes):
        binary_labels = (labels == class_idx).astype(int)
        fpr, tpr, _ = roc_curve(binary_labels, probs[:, class_idx])
        tpr_at_fpr_5 = tpr[np.where(fpr < 0.05)[0][-1]] if np.any(fpr < 0.05) else 0.0
        tpr_scores.append(tpr_at_fpr_5)

    return (np.mean(tpr_scores) * 100)


def save_pickle(save_dir, file_name, data):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    with open(os.path.join(save_dir, file_name), 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)


def get_mean_and_std(dataset):
    '''Compute the mean and std value of dataset.'''
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=4)

    mean = torch.zeros(3)
    std = torch.zeros(3)
    logger = logging.getLogger(__name__)
    logger.info('==> Computing mean and std..')
    for inputs, targets in dataloader:
        # TODO use dim param of functions to eliminate this loop
        for i in range(3):
            mean[i] += inputs[:, i, :, :].mean()
            std[i] += inputs[:, i, :, :].std()
    # TODO why might these values be closer to 0 than expected?
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std
