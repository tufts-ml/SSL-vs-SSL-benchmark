import time
from tqdm import tqdm
import torch.nn.functional as func

import logging
import numpy as np
import os
import pickle

import torch
from sklearn.metrics import confusion_matrix as sklearn_cm


logger = logging.getLogger(__name__)

__all__ = ['get_mean_and_std', 'AverageMeter', 'eval_model',
           'save_pickle', 'calculate_plain_accuracy', 'calculate_balanced_accuracy', 
           'EarlyStopping']


class EarlyStopping:
    """Early stops the training if validation acc doesn't improve after a given patience."""

    def __init__(self, patience=20, initial_count=0, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 20
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0

        """

        self.patience = patience
        self.counter = initial_count
        self.best_score = None
        self.early_stop = False
        self.delta = delta

    def __call__(self, val_acc):

        score = val_acc

        if self.best_score is None:
            self.best_score = score

        elif score <= self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

        else:
            self.best_score = score
            self.counter = 0


def eval_model(args, data_loader, raw_model, epoch, evaluation_criterion, weights=None):

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
