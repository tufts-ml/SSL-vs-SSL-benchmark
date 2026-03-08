import torch
import torchvision
from torchvision import transforms
from torchvision.models import ResNet18_Weights

from ssl_bench.config import dataset_configs
from ssl_bench.utils.apply_clahe import apply_clahe
from ssl_bench.dataset_csv import LabeledImageCSVDataset, UnlabeledImageCSVDataset, CheXpertDataset

from torchvision.transforms import RandAugment


class TransformFixMatch:
    def __init__(self, weak_transform, strong_transform):
        self.weak = weak_transform
        self.strong = strong_transform

    def __call__(self, x):
        return self.weak(x), self.strong(x)


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

    transform_weak = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((390, 400)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(size=image_size, padding=4, padding_mode='reflect'),
        transforms.ToTensor(),
        transforms.Normalize(mean=dataset_mean, std=dataset_std)
    ])

    transform_strong = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((390, 400)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(size=image_size, padding=4, padding_mode='reflect'),
        RandAugment(),
        transforms.ToTensor(),
        transforms.Normalize(mean=dataset_mean, std=dataset_std)
    ])

    # data transformations for TMED2
    if dataset_name == "TMED2":
        transform_labeledtrain = transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.Lambda(apply_clahe),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=image_size,
                                  padding=int(image_size*0.125),
                                  padding_mode='reflect'),
            transforms.ToTensor(),
            transforms.Normalize(mean=dataset_mean, std=dataset_std)
        ])

        transform_eval = transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.Lambda(apply_clahe),
            transforms.ToTensor(),
            transforms.Normalize(mean=dataset_mean, std=dataset_std)
        ])

    # data transformations for CheXpert
    elif dataset_name == "CheXpertEffusion":
        transform_labeledtrain = transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.RandomHorizontalFlip(),
            transforms.Resize(400),
            transforms.CenterCrop(320),
            transforms.ToTensor(),
            transforms.Normalize(mean=dataset_mean, std=dataset_std)
        ])

        transform_eval = transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.RandomHorizontalFlip(),
            transforms.Resize(400),
            transforms.CenterCrop(320),
            transforms.ToTensor(),
            transforms.Normalize(mean=dataset_mean, std=dataset_std)
        ])

    # data transformations for IDRID
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
    elif ("CIFAR" in dataset_name):
        transform_labeledtrain = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize(mean=dataset_mean, std=dataset_std)])
        transform_eval = transform_labeledtrain
    else:
        raise NotImplementedError(f"Implement dataloading logic for the \
            following dataset: {dataset_name}")

    imagenet_mean = (0.485, 0.456, 0.406)
    imagenet_std = (0.229, 0.224, 0.225)

    if args.use_pretrained:
        print("Using pretrained model and transforms")
        pretrained_transforms = ResNet18_Weights.IMAGENET1K_V1.transforms()
        transform_labeledtrain = transforms.Compose([
            transforms.Resize(size=image_size),
            transforms.RandomCrop(size=image_size,
                                  padding=int(image_size*0.125),
                                  padding_mode='reflect'),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
        ])
        transform_eval = transforms.Compose([
            transforms.Resize(size=image_size),
            transforms.CenterCrop(image_size),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
            pretrained_transforms,
        ])

    if args.implementation == "FixMatch":
        unlabeled_transform = TransformFixMatch(transform_weak, transform_strong)
    elif args.implementation == "BarlowTwins":
        unlabeled_transform = TransformTwice(transform_weak)
    # else:
    #     raise NotImplementedError(f"Not implemented: {args.implementation}")

    # Process unlabeled data
    if args.u_train_dataset_path != '':
        unlabel_dataset = UnlabeledImageCSVDataset(
            csv_file=args.u_train_dataset_path,
            root_dir=args.u_root_dataset_path,
            transform=unlabeled_transform
        )
    else:
        unlabel_dataset = None

    # Process labeled data
    if dataset_name == "CheXpert":
        dataset_class = CheXpertDataset
    elif dataset_name == "CIFAR10":
        train_dataset = torchvision.datasets.CIFAR10(
            root=args.l_train_dataset_path, train=True, download=True,
            transform=transform_labeledtrain)
        test_dataset = torchvision.datasets.CIFAR10(
            root=args.test_dataset_path, train=False, download=True, transform=transform_eval)

        train_size = int(len(train_dataset) * 0.8) # 80% training data
        valid_size = len(train_dataset) - train_size # 20% validation data
        train_dataset, valid_dataset = torch.utils.data.random_split(train_dataset, [train_size, valid_size])
        dataset_class = None
    elif dataset_name == "CIFAR100":
        train_dataset = torchvision.datasets.CIFAR100(
            root=args.l_train_dataset_path, train=True, download=True,
            transform=transform_labeledtrain)
        test_dataset = torchvision.datasets.CIFAR100(
            root=args.test_dataset_path, train=False, download=True, transform=transform_eval)
        train_size = int(len(train_dataset) * 0.8) # 80% training data
        valid_size = len(train_dataset) - train_size # 20% validation data
        train_dataset, valid_dataset = torch.utils.data.random_split(train_dataset, [train_size, valid_size])
        dataset_class = None
    else:
        dataset_class = LabeledImageCSVDataset

    if dataset_class is not None:
        if args.l_train_dataset_path != '':
            train_dataset = dataset_class(csv_file=args.l_train_dataset_path,
                                          root_dir=args.l_root_dataset_path,
                                          transform=transform_labeledtrain)
        else:
            train_dataset = None

        if args.val_dataset_path != '':
            valid_dataset = dataset_class(csv_file=args.val_dataset_path,
                                          root_dir=args.l_root_dataset_path,
                                          transform=transform_eval)
        else:
            valid_dataset = None

        if args.test_dataset_path != '':
            test_dataset = dataset_class(csv_file=args.test_dataset_path,
                                         root_dir=args.l_root_dataset_path,
                                         transform=transform_eval)
        else:
            test_dataset = None

    # Create dataloaders
    if train_dataset is not None:
        train_loader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=args.labeledtrain_batchsize,
                                                   shuffle=True,
                                                   num_workers=args.num_workers,
                                                   pin_memory=True,
                                                   drop_last=True)
    else:
        train_loader = None

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
