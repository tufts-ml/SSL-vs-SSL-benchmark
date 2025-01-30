import argparse
from functools import partial
from pathlib import Path
import torch
from ray import tune
from ray.tune.schedulers import ASHAScheduler
import ray.cloudpickle as pickle


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


def test_accuracy(model, device, args):
    model.eval()
    correct = 0
    total = 0

    _, _, _, test_loader = get_dataloaders(args)

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return correct / total


def main(args):
    # TODO Ray Tune hyperparameter search
    # https://pytorch.org/tutorials/beginner/hyperparameter_tuning_tutorial.html
    config = {
        "lr": tune.loguniform(1e-5, 1e-2),
        "wd": tune.loguniform(1e-6, 1e-3),
    }
    scheduler = ASHAScheduler(
        metric="val_acc",
        mode="max",
        max_t=args.train_epoch,
        grace_period=1,
        reduction_factor=2,
    )

    result = tune.run(
        partial(train, args=args),
        config=config,
        num_samples=20,  # TODO adjust
        scheduler=scheduler,
        resources_per_trial={"cpu": 2, "gpu": 1},  # TODO adjust
        local_dir=args.train_dir,
    )

    best_trial = result.get_best_trial("val_acc", "max", "last")
    print(f"Best trial config: {best_trial.config}")
    print(f"Best trial final validation accuracy: {best_trial.last_result['val_acc']}")
    print(f"Best trial final test accuracy: {best_trial.last_result['test_acc']}")
    # TODO test eval
    best_trained_model = get_model(args)
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
    best_trained_model.to(device)

    best_checkpoint = result.get_best_checkpoint(trial=best_trial, metric="accuracy", mode="max")
    with best_checkpoint.as_directory() as checkpoint_dir:
        data_path = Path(checkpoint_dir) / "data.pkl"
        with open(data_path, "rb") as fp:
            best_checkpoint_data = pickle.load(fp)

    best_trained_model.load_state_dict(best_checkpoint_data["net_state_dict"])
    test_acc = test_accuracy(best_trained_model, device, args)
    print("Best trial test set accuracy: {}".format(test_acc))

    _, _, _, test_loader = get_dataloaders(args)


if __name__ == "__main__":
    main(parse_args())
