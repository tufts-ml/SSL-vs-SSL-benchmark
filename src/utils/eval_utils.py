import time
from tqdm import tqdm
import torch.nn.functional as func

import logging
import numpy as np
import os
import pickle

import torch
from sklearn.metrics import auc
from sklearn.metrics import confusion_matrix as sklearn_cm
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_auc_score

from src.utils.train_utils import AverageMeter


logger = logging.getLogger(__name__)


def eval_model(args, data_loader, raw_model, epoch,
               evaluation_criterion='plain_accuracy', weights=None):
    """Evaluate model on validation or test set

    Args:
        args (Namespace): parsed arguments
        data_loader (torch.utils.data.DataLoader): data loader for validation or test set
        raw_model (torch.nn.Module): model to evaluate
        epoch (int): current epoch
        evaluation_criterion (str, optional): evaluation criterion. Defaults to 'plain_accuracy'.
        weights (torch.Tensor, optional): class weights for loss computation.

    Returns:
        tuple: loss, raw_performance, total_targets, total_raw_outputs
    """

    if evaluation_criterion == 'plain_accuracy':
        evaluation_method = calculate_plain_accuracy
    elif evaluation_criterion == 'balanced_accuracy':
        evaluation_method = calculate_balanced_accuracy
    else:
        raise NameError('Evaluation criterion not supported yet')

    raw_model.backbone.eval()

    losses = AverageMeter()

    data_loader = tqdm(data_loader, disable=False)

    with torch.no_grad():
        total_targets = []
        total_raw_outputs = []

        for batch_idx, (inputs, targets) in enumerate(data_loader):
            inputs = inputs.to(args.device).float()
            targets = targets.to(args.device).long()

            logits, _, _, _ = raw_model.forward(inputs, targets)

            total_targets.append(targets.cpu().numpy())  # Convert to NumPy
            total_raw_outputs.append(logits.cpu().numpy())

            loss = func.cross_entropy(
                    logits, targets, weight=weights
                ) if weights is not None else func.cross_entropy(logits, targets)
            losses.update(loss.item(), inputs.shape[0])

        total_targets = np.concatenate(total_targets, axis=0)
        total_raw_outputs = np.concatenate(total_raw_outputs, axis=0)

        # Convert raw outputs (logits) to class predictions
        total_predictions = total_raw_outputs.argmax(axis=1)

        raw_performance = evaluation_method(total_predictions, total_targets)

        data_loader.close()

    return losses.avg, raw_performance, total_targets, total_raw_outputs


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
        output (np.array): logits from model (N, num_classes)
        target (np.array): ground truth class indices (N,)
    Returns:
        float: AUROC score
    """
    probabilities = func.softmax(torch.tensor(output), dim=1).numpy()

    if output.shape[1] == 1:  # Binary classification
        auroc_score = roc_auc_score(target, probabilities[:, 1])
    else:  # Multi-class classification
        auroc_score = roc_auc_score(target, probabilities, multi_class="ovr")

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
    probabilities = func.softmax(torch.tensor(output), dim=1).numpy()

    if output.shape[1] == 1:  # Binary classification
        precision, recall, _ = precision_recall_curve(target, probabilities[:, 1])
        auprc_score = auc(recall, precision)
    else:  # Multi-class classification
        auprc_score = 0
        for class_idx in range(probabilities.shape[1]):
            class_target = (target == class_idx).astype(int)
            precision, recall, _ = precision_recall_curve(class_target, probabilities[:, class_idx])
            auprc_score += auc(recall, precision)
        auprc_score /= probabilities.shape[1]  # Average across classes

    return auprc_score * 100


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
    logger.info('==> Computing mean and std..')
    for inputs, targets in dataloader:
        for i in range(3):
            mean[i] += inputs[:, i, :, :].mean()
            std[i] += inputs[:, i, :, :].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std
