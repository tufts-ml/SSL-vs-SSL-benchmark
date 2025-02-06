import time
from tqdm import tqdm
import numpy as np
import os
import shutil
import torch
import random
import math
from torch.optim.lr_scheduler import LambdaLR
from src.utils.eval_utils import AverageMeter


class EarlyStopping:
    """Early stops the training if validation acc doesn't improve after a given patience."""

    def __init__(self, patience=20, initial_count=0, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
            initial_count (int): Initial count value for early stopping.
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
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


def save_checkpoint(state, is_best, checkpoint, filename='last_checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint,
                                               'model_best.pth.tar'))


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# TODO - Check if this is needed after Ray Tune integration
def sample_loguniform(low=0, high=1, size=1, coefficient=1, base=10):
    power_value = np.random.uniform(low, high, size)[0]
    return coefficient*np.power(base, power_value)


# TODO - Check if this is needed after Ray Tune integration
def sample_uniform(low=0.0, high=1.0, size=1, decimal=1):
    return round(np.random.uniform(low=low, high=high, size=size)[0], 1)


def get_cosine_schedule_with_warmup(optimizer,
                                    lr_warmup_epochs,
                                    lr_cycle_epochs,  # total train epochs
                                    num_cycles=7./16.,
                                    last_epoch=-1):
    """Get cosine scheduler with warmup

    Args:
        optimizer (torch.optim.Optimizer): optimizer
        lr_warmup_epochs (int): number of warmup epochs
        lr_cycle_epochs (int): number of total epochs
        num_cycles (float): number of cosine cycles
        last_epoch (int): last epoch number
    """
    def _lr_lambda(current_epoch):
        if current_epoch < lr_warmup_epochs:
            return float(current_epoch) / float(max(1, lr_warmup_epochs))
        no_progress = float(current_epoch - lr_warmup_epochs) / \
            float(max(1, float(lr_cycle_epochs) - lr_warmup_epochs))
        return max(0., math.cos(math.pi * num_cycles * no_progress))

    return LambdaLR(optimizer, _lr_lambda, last_epoch)


def get_fixed_lr(optimizer, lr_warmup_epochs, lr_cycle_epochs, num_cycles=7./16., last_epoch=-1):
    """Get fixed learning rate scheduler

    Args:
        optimizer (torch.optim.Optimizer): optimizer
        lr_warmup_epochs (int): number of warmup epochs
        lr_cycle_epochs (int): number of total epochs
        num_cycles (float): number of cosine cycles
        last_epoch (int): last epoch number
    """
    def _lr_lambda(current_epoch):
        return 1.0
    return LambdaLR(optimizer, _lr_lambda, last_epoch)


# TODO - Check if this is needed after refactoring
def train_one_epoch(args, weights, labeledtrain_loader, model, optimizer, scheduler, epoch):
    """
    Generic training loop compatible with MethodWrapper subclasses.
    This function trains the model for one epoch using labeled data.

    Args:
        args (Namespace): Parsed arguments with training settings.
        weights (torch.Tensor): Class weights for the labeled loss.
        labeledtrain_loader (DataLoader): DataLoader for labeled training data.
        model (MethodWrapper): Model wrapped with MethodWrapper.
        optimizer (Optimizer): Optimizer for the model.
        scheduler (Scheduler): Learning rate scheduler.
        epoch (int): Current epoch number.

    Returns:
        list: A list of labeled loss values for this epoch.
    """
    model.train()
    args.writer.add_scalar('train/lr', scheduler.get_last_lr()[0], epoch)

    # Tracking losses and timing
    labeled_loss_this_epoch = []
    end_time = time.time()
    labeledtrain_iter = iter(labeledtrain_loader)

    batch_time = AverageMeter()
    data_time = AverageMeter()
    labeled_loss = AverageMeter()

    # Number of steps per epoch
    n_steps_per_epoch = args.nimg_per_epoch // args.labeledtrain_batchsize
    p_bar = tqdm(range(n_steps_per_epoch), disable=False)

    for batch_idx in range(n_steps_per_epoch):
        try:
            l_input, l_labels = next(labeledtrain_iter)
        except StopIteration:
            labeledtrain_iter = iter(labeledtrain_loader)
            l_input, l_labels = next(labeledtrain_iter)

        data_time.update(time.time() - end_time)

        # Move data to the device
        l_input, l_labels = l_input.to(args.device).float(), l_labels.to(args.device).long()

        # Forward pass through the model
        # Assuming no unlabeled data for this example
        loss, s_loss, _ = model(l_input, l_labels, None)

        # Calculate supervised loss and backpropagate
        if s_loss != 0:
            labeled_loss.update(s_loss.item())
            labeled_loss_this_epoch.append(s_loss.item())
            s_loss.backward()

        optimizer.step()
        model.zero_grad()

        batch_time.update(time.time() - end_time)
        end_time = time.time()

        # Update progress bar
        p_bar.set_description(
            f"Train Epoch: {epoch}/{args.train_epoch}. "
            f"Iter: {batch_idx + 1}/{n_steps_per_epoch}. "
            f"LR: {scheduler.get_last_lr()[0]:.4f}. "
            f"Data: {data_time.avg:.3f}s. Batch: {batch_time.avg:.3f}s. "
            f"Loss_x: {labeled_loss.avg:.4f}."
        )
        p_bar.update()

    p_bar.close()
    scheduler.step()

    return labeled_loss_this_epoch


def str2bool(s):
    if s == 'True':
        return True
    elif s == 'False':
        return False
    else:
        raise NameError('Bad string')
    
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