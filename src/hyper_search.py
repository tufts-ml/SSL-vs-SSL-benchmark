import argparse
import json
import os
import time


import src.config as config


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
    model = get_model(args)
    optimizer = get_optimizer(args)
    # TODO init SummaryWriter with unique name, then pass hyperparams


def main(args):
    hypercombo_iteratethrough_list = []
    hypercombo_iteratethrough_time_list = []

    start_time = time.time()
    total_used_time = 0

    while total_used_time <= args.total_hour * 3600:
        lr = sample_loguniform(low=-5, high=-2, size=1, coefficient=3, base=10)
        wd = sample_loguniform(low=-6, high=-3, size=1, coefficient=4, base=10)

        print(f'Running with lr: {lr}, wd: {wd}')

        hypercombo_iteratethrough_list.append({'lr': lr, 'wd': wd})
        save_pickle(os.path.join(args.train_dir, 'global_stats'),
                    'hypercombo_iteratethrough_list.pkl',
                    hypercombo_iteratethrough_list)

        this_hypercombo_starttime = time.time()

        args.lr = lr
        args.wd = wd
        experiment_name = f"lr-{args.lr}_wd-{args.wd}"
        args.experiment_dir = os.path.join(args.train_dir, 'hypercombos', experiment_name)

        val_acc, test_acc = train(args)

        elapsed_time = time.time() - start_time
        total_used_time += elapsed_time
        start_time = time.time()

        print(f'Best val accuracy: {val_acc}, Best test accuracy: {test_acc}')

        brief_summary = {
            "dataset_name": args.dataset_name,
            "best_val_raw_acc": val_acc,
            "best_test_raw_acc_at_val": test_acc
        }

        with open(os.path.join(args.experiment_dir, "brief_summary.json"), "w") as f:
            json.dump(brief_summary, f)

        if total_used_time > args.total_hour * 3600:
            break

        hypercombo_iteratethrough_time_list.append(time.time() - this_hypercombo_starttime)
        save_pickle(os.path.join(args.train_dir, 'global_stats'),
                    'hypercombo_iteratethrough_time_list.pkl',
                    hypercombo_iteratethrough_time_list)

    save_pickle(os.path.join(args.train_dir, 'global_stats'), 'total_time.pkl', [total_used_time])


if __name__ == "__main__":
    main(parse_args())
