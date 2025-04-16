import torch
from torchvision import transforms
from torchvision.models import ResNet18_Weights

from ssl_bench.config import dataset_configs, get_transformations
from ssl_bench.utils.apply_clahe import apply_clahe
from ssl_bench.dataset_csv import LabeledImageCSVDataset, UnlabeledImageCSVDataset, CheXpertDataset


class TransformTwice:
    def __init__(self, transform_fn):
        self.transform_fn = transform_fn

    def __call__(self, x):
        out1 = self.transform_fn(x)
        out2 = self.transform_fn(x)

        return out1, out2


def get_dataloaders(args):
    """Get DataLoaders

    Args:
        args (Namespace): parsed arguments

    Returns:
        tuple: 4 DataLoaders, which may be none
               train_loader, unlabel_loader, valid_loader, test_loader
    """
    dataset_name = args.dataset_name

    dataset_mean = dataset_configs[args.dataset_name]['dataset_mean']
    dataset_std = dataset_configs[args.dataset_name]['dataset_std']
    image_size = dataset_configs[args.dataset_name]['image_size']

    l_train_transform, u_train_transform, val_transform, test_transform = get_transformations(args)

    # Process unlabeled data
    if args.u_train_dataset_path != '':
        unlabel_dataset = UnlabeledImageCSVDataset(csv_file=args.u_train_dataset_path,
                                                   root_dir=args.u_root_dataset_path,
                                                   transform=u_train_transform)
    else:
        unlabel_dataset = None

    # Process labeled data
    if dataset_name == "CheXpert":
        dataset_class = CheXpertDataset
    else:
        dataset_class = LabeledImageCSVDataset
    if args.l_train_dataset_path != '':
        train_dataset = dataset_class(csv_file=args.l_train_dataset_path,
                                      root_dir=args.l_root_dataset_path,
                                      transform=l_train_transform)
    else:
        train_dataset = None

    if args.val_dataset_path != '':
        valid_dataset = dataset_class(csv_file=args.val_dataset_path,
                                      root_dir=args.l_root_dataset_path,
                                      transform=val_transform)
    else:
        valid_dataset = None

    if args.test_dataset_path != '':
        test_dataset = dataset_class(csv_file=args.test_dataset_path,
                                     root_dir=args.l_root_dataset_path,
                                     transform=test_transform)
    else:
        test_dataset = None

    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=args.labeledtrain_batchsize,
                                               shuffle=True,
                                               num_workers=args.num_workers,
                                               pin_memory=True,
                                               drop_last=True)
    if unlabel_dataset is not None:
        print("Length of unlabeled dataset: ", len(unlabel_dataset))
        unlabel_loader = torch.utils.data.DataLoader(unlabel_dataset,
                                                     batch_size=args.unlabeledtrain_batchsize,
                                                     shuffle=True,
                                                     num_workers=args.num_workers,
                                                     pin_memory=True,
                                                     drop_last=True)
    else:
        unlabel_loader = None

    if valid_dataset is not None:
        valid_loader = torch.utils.data.DataLoader(valid_dataset, 32,
                                                   shuffle=False, drop_last=False,
                                                   num_workers=args.num_workers,
                                                   pin_memory=True)
    else:
        valid_loader = None

    if test_dataset is not None:
        test_loader = torch.utils.data.DataLoader(test_dataset, 32,
                                                  shuffle=False, drop_last=False,
                                                  num_workers=args.num_workers,
                                                  pin_memory=True)
    else:
        test_loader = None

    return train_loader, unlabel_loader, valid_loader, test_loader
