import numpy as np
import os
import shutil
import torch
import random
import math
from torch.optim.lr_scheduler import LambdaLR


def sample_loguniform(low=0, high=1, size=1, coefficient=1, base=10):
    power_value = np.random.uniform(low, high, size)[0]
    return coefficient*np.power(base, power_value)


def sample_uniform(low=0.0, high=1.0, size=1, decimal=1):
    return round(np.random.uniform(low=low, high=high, size=size)[0], 1)


def str2bool(s):
    if s == 'True':
        return True
    elif s == 'False':
        return False
    else:
        raise NameError('Bad string')


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
    def _lr_lambda(current_epoch):
        if current_epoch < lr_warmup_epochs:
            return float(current_epoch) / float(max(1, lr_warmup_epochs))
        no_progress = float(current_epoch - lr_warmup_epochs) / \
            float(max(1, float(lr_cycle_epochs) - lr_warmup_epochs))
        return max(0., math.cos(math.pi * num_cycles * no_progress))

    return LambdaLR(optimizer, _lr_lambda, last_epoch)


def get_fixed_lr(optimizer, lr_warmup_epochs, lr_cycle_epochs, num_cycles=7./16., last_epoch=-1):
    def _lr_lambda(current_epoch):
        return 1.0
    return LambdaLR(optimizer, _lr_lambda, last_epoch)
