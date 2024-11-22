import argparse
from torchvision import transforms
from src.dataset_csv import ImageCSVDataset
from src.clahe import apply_clahe


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
    parser.add_argument('--root_dataset_folder', default='', type=str)
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
    dataset_name = args.dataset_name
    root_dataset_folder = args.root_dataset_folder

    train_csv_path = args.l_train_dataset_path
    val_csv_path = args.val_dataset_path
    test_csv_path = args.test_dataset_path
    unlab_csv_path = args.u_train_dataset_path

    dataset_mean = config[args.dataset_name]['dataset_mean']
    dataset_std = config[args.dataset_name]['dataset_std']
    image_size = config[args.dataset_name]['image_size']

    if dataset_name == "TMED2":
        transform_labeledtrain = transforms.Compose([
        # transforms.ToPILImage(),
        transforms.Grayscale(num_output_channels=3),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(size=image_size,
                              padding=int(image_size*0.125),
                              padding_mode='reflect'),
        transforms.ToTensor(),
        transforms.Normalize(mean=dataset_mean, std=dataset_std)
        ])

        transform_eval = transforms.Compose([
        # transforms.ToPILImage(),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=dataset_mean, std=dataset_std)
        ])

    elif dataset_name == "CheXpert":
        pass
    elif dataset_name == "IDRID":
        transform_labeledtrain = transforms.Compose([
        transforms.Resize(size=image_size),
        transforms.Lambda(apply_clahe),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(size=image_size,
                              padding=int(image_size*0.125),
                              padding_mode='reflect'),
        transforms.ToTensor(),
        transforms.Normalize(mean=dataset_mean, std=dataset_std)
        ])

        transform_eval = transforms.Compose([
            transforms.Resize(size=image_size),
            transforms.Lambda(apply_clahe),
            transforms.ToTensor(),
            transforms.Normalize(mean=dataset_mean, std=dataset_std)
        ])
    else:
        raise NotImplementedError(f"Implement dataloading logic for the \
            following dataset: {dataset_name}")
    
    # handle if they're none
    if args.l_train_dataset_path != '':
        train_loader = ImageCSVDataset(csv_file=args.l_train_dataset_path,
                                        root_dir=args.root_dataset_path,
                                        transform=transform_labeledtrain)
    else:
        train_loader = None
    
    if args.val_dataset_path != '':
        valid_loader = ImageCSVDataset(csv_file=args.val_dataset_path,
                                        root_dir=args.root_dataset_path,
                                        transform=transform_eval)
    else:
        valid_loader = None
    
    if args.u_train_dataset_path != '':
        raise NotImplementedError("Implement Dataloading logic")
    else:
        unlab_dataset = None

    if args.test_dataset_path != '':
        test_loader = ImageCSVDataset(csv_file=args.test_dataset_path,
                                        root_dir=args.root_dataset_path,
                                        transform=transform_eval)
    else:
        test_loader = None

    return train_loader, unlabel_loader, valid_loader, test_loader


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
    # TODO Ray Tune hyperparameter search
    # https://pytorch.org/tutorials/beginner/hyperparameter_tuning_tutorial.html
    # TODO test eval
    _, _, _, test_loader = get_dataloaders(args)


if __name__ == "__main__":
    main(parse_args())
