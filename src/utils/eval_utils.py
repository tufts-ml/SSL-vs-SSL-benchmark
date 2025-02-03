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
        weights (torch.Tensor, optional): class weights for

    Raises:
        NameError: _description_

    Returns:
        tuple: loss, raw_performance, total_targets, total_raw_outputs
    """

    if evaluation_criterion == 'plain_accuracy':
        evaluation_method = calculate_plain_accuracy
    elif evaluation_criterion == 'balanced_accuracy':
        evaluation_method = calculate_balanced_accuracy
    else:
        raise NameError('not supported yet')

    raw_model.eval()

    end_time = time.time()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    data_loader = tqdm(data_loader, disable=False)

    with torch.no_grad():
        total_targets = []
        total_raw_outputs = []

        for batch_idx, (inputs, targets) in enumerate(data_loader):
            data_time.update(time.time() - end_time)

            inputs = inputs.to(args.device).float()
            targets = targets.to(args.device).long()
            raw_outputs = raw_model(inputs)

            total_targets.append(targets.detach().cpu())
            total_raw_outputs.append(raw_outputs.detach().cpu())

            if weights is not None:
                loss = func.cross_entropy(raw_outputs, targets, weights)
            else:
                loss = func.cross_entropy(raw_outputs, targets)

            losses.update(loss.item(), inputs.shape[0])
            batch_time.update(time.time() - end_time)

            # update end time
            end_time = time.time()

        total_targets = np.concatenate(total_targets, axis=0)
        total_raw_outputs = np.concatenate(total_raw_outputs, axis=0)

        raw_performance = evaluation_method(total_raw_outputs, total_targets)

        data_loader.close()

    return losses.avg, raw_performance, total_targets, total_raw_outputs


def calculate_plain_accuracy(output, target):

    accuracy = (output.argmax(1) == target).mean()*100

    return accuracy


def calculate_balanced_accuracy(output, target):
    confusion_matrix = sklearn_cm(target, output.argmax(1))
    n_class = confusion_matrix.shape[0]
    print('Inside calculate_balanced_accuracy, {} classes passed in'.format(n_class), flush=True)

    recalls = []
    for i in range(n_class):
        recall = confusion_matrix[i, i]/np.sum(confusion_matrix[i])
        recalls.append(recall)
        print('class{} recall: {}'.format(i, recall), flush=True)

    balanced_accuracy = np.mean(np.array(recalls))

    return balanced_accuracy * 100


def calculate_auroc(output, target):
    if output.shape[1] == 1:  # Binary classification with single output (logit)
        probabilities = func.softmax(torch.tensor(output), dim=1).numpy()
        auroc_score = roc_auc_score(target, probabilities[:, 1])
    else:  # Multi-class classification
        probabilities = func.softmax(torch.tensor(output), dim=1).numpy()
        auroc_score = roc_auc_score(target, probabilities, multi_class="ovr")

    return auroc_score * 100


def calculate_auprc(output, target):
    if output.shape[1] == 1:  # Binary classification with single output (logit)
        probabilities = func.softmax(torch.tensor(output), dim=1).numpy()
        precision, recall, _ = precision_recall_curve(target, probabilities[:, 1])
        auprc_score = auc(recall, precision)

    else:  # Multi-class classification
        probabilities = func.softmax(torch.tensor(output), dim=1).numpy()
        auprc_score = 0
        for class_idx in range(probabilities.shape[1]):
            # Compute precision-recall curve for each class in a one-vs-rest manner
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


class AverageMeter(object):
    """Computes and stores the average and current value
       Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count