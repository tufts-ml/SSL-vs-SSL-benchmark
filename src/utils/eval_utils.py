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

from src.utils.train_utils import AverageMeter


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
    losses = AverageMeter()
    data_loader = tqdm(data_loader, disable=False)

    weights = weights.to(args.device) if weights is not None else None

    with torch.no_grad():
        total_targets, total_outputs = [], []

        for inputs, targets in data_loader:
            inputs, targets = inputs.to(args.device).float(), targets.to(args.device).long()
            output, loss = model.eval_forward(inputs, targets)
            total_outputs.append(output)
            total_targets.append(targets)

            losses.update(loss.item(), inputs.size(0))

        total_targets = torch.cat(total_targets).cpu().numpy()
        total_outputs = torch.cat(total_outputs).cpu().numpy()

        data_loader.close()

    metrics = evaluate_all_metrics(total_outputs, total_targets)
    metrics['loss'] = losses.avg

    return metrics


def evaluate_all_metrics(outputs, targets):
    predictions = outputs.argmax(axis=1)
    return {
        'plain_accuracy': calculate_plain_accuracy(predictions, targets),
        'balanced_accuracy': calculate_balanced_accuracy(predictions, targets),
        'auroc': calculate_auroc(outputs, targets),
        'auprc': calculate_auprc(outputs, targets),
        'tpr_at_fpr_5': calculate_tpr_at_fpr_5(outputs, targets),
    }


def calculate_plain_accuracy(predictions, target):
    """
    Compute plain accuracy
    Args:
        predictions (np.array): predicted class indices
        target (np.array): ground truth class indices
    Returns:
        float: accuracy percentage
    """
    return (predictions == target).mean() * 100


def calculate_balanced_accuracy(predictions, target):
    """
    Compute balanced accuracy using confusion matrix.
    Args:
        predictions (np.array): predicted class indices
        target (np.array): ground truth class indices
    Returns:
        float: balanced accuracy percentage
    """
    confusion_matrix = sklearn_cm(target, predictions)
    n_class = confusion_matrix.shape[0]

    recalls = []
    for i in range(n_class):
        recall = confusion_matrix[i, i] / \
            np.sum(confusion_matrix[i]) if np.sum(confusion_matrix[i]) > 0 else 0
        recalls.append(recall)

    balanced_accuracy = np.mean(recalls) * 100

    return balanced_accuracy


def calculate_auroc(output, target):
    """
    Compute Area Under the Receiver Operating Characteristic Curve (AUROC)
    Args:
        output (np.array): probabiltities from model (N, num_classes)
        target (np.array): ground truth class indices (N,)
    Returns:
        float: AUROC score
    """
    if output.shape[1] == 2:
        auroc_score = roc_auc_score(target, output[:, 1])
    else:
        auroc_score = roc_auc_score(target, output, multi_class="ovr")

    return auroc_score * 100


def calculate_auprc(output, target):
    """
    Compute Area Under the Precision-Recall Curve (AUPRC)
    Args:
        output (np.array): logits from model (N, num_classes)
        target (np.array): ground truth class indices (N,)
    Returns:
        float: AUPRC score
    """

    if output.shape[1] == 2:
        precision, recall, _ = precision_recall_curve(target, output[:, 1])
        auprc_score = auc(recall, precision)
    else:
        auprc_score = 0
        for i in range(output.shape[1]):
            precision, recall, _ = precision_recall_curve(
                target == i, output[:, i])
            auprc_score += auc(recall, precision)
        auprc_score /= output

    return auprc_score * 100


def calculate_tpr_at_fpr_5(probs, labels):
    num_classes = probs.shape[1]
    tpr_scores = []

    for class_idx in range(num_classes):
        # Convert labels to binary (one-vs-rest)
        binary_labels = (labels == class_idx).astype(int)

        fpr, tpr, _ = roc_curve(binary_labels, probs[:, class_idx])
        tpr_at_fpr_5 = tpr[np.where(fpr < 0.05)[0][-1]] if np.any(fpr < 0.05) else 0.0
        tpr_scores.append(tpr_at_fpr_5)

    return (np.mean(tpr_scores) * 100)  # Average TPR across all classes


def save_pickle(save_dir, save_file_name, data):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    data_save_fullpath = os.path.join(save_dir, save_file_name)
    with open(data_save_fullpath, 'wb') as handle:
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
