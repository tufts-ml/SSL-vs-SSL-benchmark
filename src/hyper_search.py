import argparse


import src.config as config
from torch.utils.tensorboard import SummaryWriter
from src.LabelOnlyBaseline.libml.utils import train_one_epoch, eval_model
from src.LabelOnlyBaseline.libml.utils import EarlyStopping
from src.LabelOnlyBaseline.libml.utils import save_checkpoint
from src.LabelOnlyBaseline.libml.utils import get_cosine_schedule_with_warmup, get_fixed_lr

import os
import json
import numpy as np
import torch
import time


def parse_args():
    parser = argparse.ArgumentParser()

    # data settings
    parser.add_argument('--dataset_name', default='TissueMNIST', type=str, help='name of dataset')
    # dataset paths
    parser.add_argument('--l_train_dataset_path', default='', type=str)
    parser.add_argument('--u_train_dataset_path', default='', type=str)
    parser.add_argument('--val_dataset_path', default='', type=str)
    parser.add_argument('--test_dataset_path', default='', type=str)
    # data loading settings
    parser.add_argument('--labeledtrain_batchsize', default=50, type=int)
    parser.add_argument('--unlabeledtrain_batchsize', default=50, type=int)
    parser.add_argument('--num_workers', default=12, type=int)

    # architecture settings
    parser.add_argument('--arch', default='resnet18', type=str, help='backbone to use')
    # pretrained weights for resnet18
    parser.add_argument('--use_pretrained', action="store_true")

    # training process settings
    parser.add_argument('--train_epoch', default=300, type=int, help='total epochs to run')
    parser.add_argument('--start_epoch', default=0, type=int,
                        help='manual epoch number (useful on restarts)')
    # learning rate
    parser.add_argument('--lr', default=3e-4, type=float, help='learning rate')
    # learning rate schedule following MixMatch and FixMatch repo
    parser.add_argument('--lr_warmup_epochs', default=0, type=float,
                        help='warmup epoch for learning rate schedule')
    parser.add_argument('--lr_schedule_type', default='CosineLR',
                        choices=['CosineLR', 'FixedLR'], type=str)
    parser.add_argument('--lr_cycle_epochs', default=10000, type=int)
    # optimization
    parser.add_argument('--wd', default=5e-4, type=float, help='weight decay')
    parser.add_argument('--optimizer_type', default='SGD', choices=['SGD', 'Adam'], type=str)
    parser.add_argument('--patience', default=20, type=int, help='Earlystop patience')
    # loss parameters?
    parser.add_argument('--temperature', default=0.95, type=float,
                        help='temperature for label guessing')
    parser.add_argument('--alpha', default=0.75, type=float)
    parser.add_argument('--lambda_u_max', default=1, type=float,
                        help='coefficient of unlabeled loss')
    # unlabeled loss parameters
    parser.add_argument('--unlabeledloss_warmup_schedule_type', default='NoWarmup',
                        choices=['NoWarmup', 'Linear', 'Sigmoid', ], type=str)
    # unlabeled warmup following MixMatch and FixMatch repo
    parser.add_argument('--unlabeledloss_warmup_pos', default=0.4, type=float,
                        help='position at which unlabeled loss warmup ends')
    # default hypers not to search for now
    parser.add_argument('--nesterov', action='store_true', default=True,
                        help='use nesterov momentum')
    parser.add_argument('--use_ema', action='store_true', default=True,
                        help='use EMA model')
    parser.add_argument('--ema_decay', default=0.999, type=float,
                        help='EMA decay rate')
    parser.add_argument('--total_hour', default=50, type=int, help='total hours to run')

    parser.add_argument('--train_dir', help='directory to output the result')

    args = parser.parse_args()

    # total size of labeled + unlabeled set for TissueMNIST
    args.nimg_per_epoch = config[args.dataset_name]['nimg_per_epoch']
    args.num_classes = config[args.dataset_name]['num_classes']
    return args


def get_dataloaders(args):
    """Get DataLoaders

    Args:
        args (Namespace): parsed arguments

    Returns:
        tuple: 4 DataLoaders, which may be none
               train_loader, unlabel_loader, valid_loader, test_loader
    """
    # TODO implement
    return None


def get_model(args):
    """Get neural network model

    Args:
        args (Namespace): parsed arguments

    Returns:
        torch.nn.Module: model specified by args
    """
    # TODO implement
    return None


def get_optimizer(args):
    """Get optimizer for learning

    Args:
        args (Namespace): parsed arguments

    Returns:
        torch.optim.Optimizer: optimizer specified by args
    """
    # TODO implement
    return None


def train(args):
    precalculated_class_weights = config[args.dataset_name]['class_weights']
    weights = torch.Tensor(precalculated_class_weights)
    weights = weights.to(args.device)

    model = get_model(args)
    model = model.to(args.device)

    optimizer = get_optimizer(args)
    train_loader, unlabel_loader, val_loader, test_loader = get_dataloaders(args)

    writer = SummaryWriter(args.train_dir)

    # Initialize scheduler based on args
    if args.lr_schedule_type == 'CosineLR':
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, args.lr_warmup_epochs, args.lr_cycle_epochs)
    elif args.lr_schedule_type == 'FixedLR':
        scheduler = get_fixed_lr(optimizer, args.lr_warmup_epochs, args.lr_cycle_epochs)
    else:
        raise NameError('Invalid lr_schedule_type')

    # Initialize tracking variables
    best_val_acc = 0
    best_test_acc = 0
    args.start_epoch = 0
    current_count = 0
    total_time = 0

    # Early stopping
    early_stopping = EarlyStopping(patience=args.patience, initial_count=current_count)

    start_time = time.time()

    for epoch in range(args.start_epoch, args.train_epoch):
        # Train
        train_losses = train_one_epoch(args, weights, train_loader,
                                       model, optimizer, scheduler, epoch)

        # Evaluate
        val_loss, val_acc, val_labels, val_preds = eval_model(args, val_loader, model, epoch)
        test_loss, test_acc, test_labels, test_preds = eval_model(args, test_loader, model, epoch)

        # Update best scores
        is_best = val_acc > best_val_acc
        if is_best:
            best_val_acc = val_acc
            best_test_acc = test_acc

        # Log metrics
        writer.add_scalar('train/loss', np.mean(train_losses), epoch)
        writer.add_scalar('val/accuracy', val_acc, epoch)
        writer.add_scalar('val/loss', val_loss, epoch)
        writer.add_scalar('test/accuracy', test_acc, epoch)
        writer.add_scalar('test/loss', test_loss, epoch)

        # Save checkpoint
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_val_acc': best_val_acc,
            'best_test_acc': best_test_acc,
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
        }, is_best, args.train_dir)

        # Early stopping check
        if early_stopping(val_acc):
            print(f'Early stopping triggered after epoch {epoch}')
            break

        # Time tracking
        epoch_time = time.time() - start_time
        total_time += epoch_time
        start_time = time.time()

    # Save final summary
    summary = {
        'best_val_accuracy': best_val_acc,
        'best_test_accuracy': best_test_acc,
        'total_epochs': epoch + 1,
        'total_time': total_time
    }

    with open(os.path.join(args.train_dir, 'training_summary.json'), 'w') as f:
        json.dump(summary, f)

    writer.close()
    return best_val_acc, best_test_acc


def main(args):
    # TODO Ray Tune hyperparameter search
    # https://pytorch.org/tutorials/beginner/hyperparameter_tuning_tutorial.html
    # TODO test eval
    _, _, _, test_loader = get_dataloaders(args)


if __name__ == "__main__":
    main(parse_args())
