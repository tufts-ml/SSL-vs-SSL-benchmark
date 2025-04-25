import numpy as np
import os
import shutil
import torch
import random
import math
from torch.optim.lr_scheduler import LambdaLR
from torchvision.models import resnet18, ResNet18_Weights
import torch.nn.init as init
from ssl_bench.methods.LabelOnlyBaseline import LabelOnlyBaseline
from ssl_bench.methods.MixUp import MixUp
from ssl_bench.methods.BarlowTwins import BarlowTwins
from ssl_bench.methods.SimCLR import SimCLR
import torch.optim as optim


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


def get_model(args):
    """Get neural network model

    Args:
        args (Namespace): parsed arguments

    Returns:
        torch.nn.Module: model specified by args
    """

    if args.arch == 'resnet18':
        weights = ResNet18_Weights.DEFAULT if args.use_pretrained else None
        model = resnet18(weights=weights)

        # Freeze layers only if using a pretrained model
        if args.use_pretrained and args.freeze_backbone:
            print("Freezing layers")
            for param in model.parameters():
                param.requires_grad = False

        # Replace the last fully connected layer
        model.fc = torch.nn.Linear(512, args.num_classes)
        init.normal_(model.fc.weight, mean=0.0, std=0.0001)
        init.zeros_(model.fc.bias)

        # Ensure the new last layer is trainable
        for param in model.fc.parameters():
            param.requires_grad = True

    elif args.arch == 'wideresnet':
        import backbone.wideresnet as models
        model_depth = 28
        model_width = 2

        model = models.build_wideresnet(depth=model_depth,
                                        widen_factor=model_width,
                                        dropout=0.0,
                                        num_classes=args.num_classes)

        if args.use_pretrained and args.freeze_backbone:
            print("Freezing layers")
            for param in model.parameters():
                param.requires_grad = False

        model.fc = torch.nn.Linear(512, args.num_classes)
        init.normal_(model.fc.weight, mean=0.0, std=0.0001)
        init.zeros_(model.fc.bias)

        # Ensure the new last layer is trainable
        for param in model.fc.parameters():
            param.requires_grad = True

    else:
        raise NameError('Not implemented yet')

    implementation_map = {
        'LabelOnlyBaseline': LabelOnlyBaseline,
        'MixUp': MixUp,
        'BarlowTwins': BarlowTwins,
        'SimCLR': SimCLR,
    }

    model_class = implementation_map.get(args.implementation)

    if model_class is None:
        raise NameError(f"Invalid implementation: {args.implementation}")

    return model_class(model, args)


def get_optimizer(args, model: torch.nn.Module):
    """Get optimizer for learning

    Args:
        args (Namespace): parsed arguments

    Returns:
        torch.optim.Optimizer: optimizer specified by args
    """
    no_decay = ['bias', 'bn']
    grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(
            nd in n for nd in no_decay)], 'weight_decay': args.wd},
        {'params': [p for n, p in model.named_parameters() if any(
            nd in n for nd in no_decay)], 'weight_decay': args.wd}
    ]

    if args.optimizer_type == 'SGD':
        optimizer = optim.SGD(grouped_parameters, lr=args.lr,
                              momentum=0.9, nesterov=args.nesterov)

    elif args.optimizer_type == 'Adam':
        optimizer = optim.Adam(grouped_parameters, lr=args.lr)

    else:
        raise NameError('Not supported optimizer setting')

    return optimizer


def get_lr_scheduler(optimizer, args):
    """Get learning rate scheduler

    Args:
        optimizer (torch.optim.Optimizer): optimizer
        args (Namespace): parsed arguments

    Returns:
        torch.optim.lr_scheduler.LambdaLR: learning rate scheduler
    """
    if args.lr_schedule_type == 'CosineLR':
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, args.lr_warmup_epochs, args.lr_cycle_epochs)
    elif args.lr_schedule_type == 'FixedLR':
        scheduler = get_fixed_lr(optimizer, args.lr_warmup_epochs, args.lr_cycle_epochs)
    else:
        raise NameError('Invalid lr_schedule_type')

    return scheduler


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
