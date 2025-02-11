import argparse
from src.config import dataset_configs


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--method', type=str, required=True, help='Method name')

    # data settings
    parser.add_argument('--dataset_name', default='TissueMNIST', type=str, help='name of dataset')
    # dataset paths
    parser.add_argument('--l_train_dataset_path', default='', type=str)
    parser.add_argument('--u_train_dataset_path', default='', type=str)
    parser.add_argument('--val_dataset_path', default='', type=str)
    parser.add_argument('--test_dataset_path', default='', type=str)
    parser.add_argument('--root_dataset_path', default='', type=str)
    # data loading settings
    parser.add_argument('--labeledtrain_batchsize', default=128, type=int)
    parser.add_argument('--unlabeledtrain_batchsize', default=128, type=int)
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

    # total size of labeled + unlabeled set
    args.nimg_per_epoch = dataset_configs[args.dataset_name]['nimg_per_epoch']
    args.num_classes = dataset_configs[args.dataset_name]['num_classes']
    return args
