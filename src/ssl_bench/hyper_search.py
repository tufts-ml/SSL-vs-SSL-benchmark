import json
import logging
import os
import time

from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import torch.nn.init as init

from ssl_bench.config import dataset_configs, method_configs, HyperparamSpace
from ssl_bench.utils.train_utils import (AverageMeter, save_checkpoint,
                                         get_cosine_schedule_with_warmup,
                                         get_fixed_lr, EarlyStopping)
from ssl_bench.utils.apply_clahe import apply_clahe
from ssl_bench.utils.eval_utils import (
    calculate_plain_accuracy,
    eval_model,
)
from ssl_bench.utils.arg_parser import parse_args
from ssl_bench.methods.LabelOnlyBaseline import LabelOnlyBaseline
from ssl_bench.methods.MixUp import MixUp
from ssl_bench.methods.BarlowTwins import BarlowTwins
from ssl_bench.dataset_csv import LabeledImageCSVDataset, UnlabeledImageCSVDataset, CheXpertDataset

# TODO - Move this to a separate file?


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
    logger.info(f"Loading dataset: {args.dataset_name}")

    dataset_name = args.dataset_name

    dataset_mean = dataset_configs[args.dataset_name]['dataset_mean']
    dataset_std = dataset_configs[args.dataset_name]['dataset_std']
    image_size = dataset_configs[args.dataset_name]['image_size']

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
    elif dataset_name == "CheXpert":
        transform_labeledtrain = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Grayscale(num_output_channels=3),
            transforms.RandomHorizontalFlip(),
            transforms.Resize(400),
            transforms.CenterCrop(320),
            transforms.ToTensor(),
            transforms.Normalize(mean=dataset_mean, std=dataset_std)
        ])

        transform_eval = transforms.Compose([
            transforms.ToPILImage(),
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

    else:
        raise NotImplementedError(f"Implement dataloading logic for the \
            following dataset: {dataset_name}")

    if args.use_pretrained:
        print("Using pretrained model and transforms")
        pretrained_transforms = ResNet18_Weights.IMAGENET1K_V1.transforms()
        transform_labeledtrain = transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            pretrained_transforms,
        ])
        transform_eval = transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            pretrained_transforms,
        ])

    # TODO - currently this transforms twice for BarlowTwins, will need to
    # for other methods using unlabeled data

    # Process unlabeled data
    if args.u_train_dataset_path != '':
        unlabel_dataset = UnlabeledImageCSVDataset(csv_file=args.u_train_dataset_path,
                                                   root_dir=args.u_root_dataset_path,
                                                   transform=TransformTwice(transform_labeledtrain))
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
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=args.labeledtrain_batchsize,
                                               shuffle=True,
                                               num_workers=args.num_workers,
                                               pin_memory=True,
                                               drop_last=True)
    if unlabel_dataset is not None:
        unlabel_loader = torch.utils.data.DataLoader(unlabel_dataset,
                                                     batch_size=args.unlabeledtrain_batchsize,
                                                     shuffle=True,
                                                     num_workers=args.num_workers,
                                                     pin_memory=True,
                                                     drop_last=True)
    else:
        unlabel_loader = None

    if valid_dataset is not None:
        valid_loader = torch.utils.data.DataLoader(valid_dataset, 128,
                                                   shuffle=False, drop_last=False,
                                                   num_workers=args.num_workers,
                                                   pin_memory=True)
    else:
        valid_loader = None

    if test_dataset is not None:
        test_loader = torch.utils.data.DataLoader(test_dataset, 128,
                                                  shuffle=False, drop_last=False,
                                                  num_workers=args.num_workers,
                                                  pin_memory=True)
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
    logger.info(f"Initializing model architecture: {args.arch}")

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

    else:
        raise NameError('Not implemented yet')

    implementation_map = {
        'LabelOnlyBaseline': LabelOnlyBaseline,
        'MixUp': MixUp,
        'BarlowTwins': BarlowTwins,
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


def setup_training(args, method_config: HyperparamSpace):
    """Set up training environment

    Args:
        args (Namespace): parsed arguments
        method_config (HyperparamSpace): hyperparameter space for method

    Raises:
        ValueError: if dataset not found in config

    Returns:
        tuple: model, optimizer, scheduler, train_loader, unlabel_loader, val_loader, test_loader
    """
    if args.dataset_name not in dataset_configs:
        raise ValueError(f"Dataset {args.dataset_name} not found in config.")

    # Set hyperparameters
    hyper_strs = []
    for key, value in method_config.rvs().items():
        setattr(args, key, value)
        hyper_strs.append(f'{key}={value}')

    model_dir = "_".join(hyper_strs)
    args.train_dir = os.path.join(args.base_train_dir, model_dir)
    os.makedirs(args.train_dir, exist_ok=True)

    # Load class weights
    args.weights = torch.Tensor(dataset_configs[args.dataset_name]['class_weights']).to(args.device)

    # Initialize model, optimizer, and scheduler
    model = get_model(args).to(args.device)
    optimizer = get_optimizer(args, model)
    scheduler = get_lr_scheduler(optimizer, args)
    train_loader, unlabel_loader, val_loader, test_loader = get_dataloaders(args)

    return model, optimizer, scheduler, train_loader, unlabel_loader, val_loader, test_loader


def train_one_epoch(args, model, optimizer, scheduler, train_loader,
                    epoch, unlabel_loader=None):
    """Train model for one epoch

    Args:
        args (Namespace): parsed arguments
        model (torch.nn.Module): model to train
        optimizer (torch.optim.Optimizer): optimizer
        scheduler (torch.optim.lr_scheduler.LambdaLR): learning rate scheduler
        train_loader (torch.utils.data.DataLoader): training data loader
        unlabel_loader (torch.utils.data.DataLoader): unlabeled data loader
        epoch (int): current epoch

    Returns:
        float: average loss for the epoch
        torch.Tensor: logits from the model
        torch.Tensor: labels from the dataset
    """

    model.train()
    args.writer.add_scalar('train/lr', scheduler.get_last_lr()[0], epoch)
    print(f"Epoch {epoch+1} - Learning Rate: {scheduler.get_last_lr()[0]}")

    batch_time, data_time, labeled_loss = AverageMeter(), AverageMeter(), AverageMeter()

    all_logits = []
    all_labels = []

    labeledtrain_iter = iter(train_loader)
    n_steps_per_epoch = args.nimg_per_epoch // args.labeledtrain_batchsize
    p_bar = tqdm(range(n_steps_per_epoch), disable=False)

    start_time = time.time()

    for batch_idx in range(n_steps_per_epoch):
        try:
            l_input, l_labels = next(labeledtrain_iter)
        except StopIteration:
            labeledtrain_iter = iter(train_loader)
            l_input, l_labels = next(labeledtrain_iter)

        data_time.update(time.time() - start_time)

        optimizer.zero_grad()  # Zero gradients before backward pass

        logits, loss, supervised_loss, unsupervised_loss = model.forward(
            l_input, l_labels)

        total_loss = supervised_loss + (unsupervised_loss if unsupervised_loss is not None else 0)

        all_logits.append(logits.detach().cpu())
        all_labels.append(l_labels.detach().cpu())

        # Weighted update for correct loss averaging
        batch_size = l_labels.size(0)
        labeled_loss.update(supervised_loss.item(), batch_size)

        total_loss.backward()
        optimizer.step()

        batch_time.update(time.time() - start_time)
        start_time = time.time()

        p_bar.set_description(f"Epoch {epoch+1} - Loss: {labeled_loss.avg:.4f}")
        p_bar.update()

    p_bar.close()
    scheduler.step()

    all_logits = torch.cat(all_logits)
    all_labels = torch.cat(all_labels)

    return labeled_loss.avg, all_logits, all_labels


def train_unlabel_one_epoch(args, model, optimizer, scheduler, train_loader,
                            unlabel_loader, epoch):
    """Train model for one epoch with unlabeled data

    Args:
        args (Namespace): parsed arguments
        model (torch.nn.Module): model to train
        optimizer (torch.optim.Optimizer): optimizer
        scheduler (torch.optim.lr_scheduler.LambdaLR): learning rate scheduler
        train_loader (torch.utils.data.DataLoader): training data loader
        unlabel_loader (torch.utils.data.DataLoader): unlabeled data loader
        epoch (int): current epoch

    Returns:
        float: average loss for the epoch
        torch.Tensor: logits from the model
    """
    model.train()
    args.writer.add_scalar('train/lr', scheduler.get_last_lr()[0], epoch)
    print(f"Epoch {epoch+1} - Learning Rate: {scheduler.get_last_lr()[0]}")

    batch_time, data_time, unlabeled_loss = AverageMeter(), AverageMeter(), AverageMeter()

    unlabeled_train_iter = iter(unlabel_loader)
    n_steps_per_epoch = len(unlabel_loader) // args.unlabeledtrain_batchsize

    p_bar = tqdm(range(n_steps_per_epoch), disable=False)

    start_time = time.time()

    for batch_idx in range(n_steps_per_epoch):
        try:
            u_input = next(unlabeled_train_iter)
        except StopIteration:
            unlabeled_train_iter = iter(unlabel_loader)
            u_input = next(unlabeled_train_iter)

        data_time.update(time.time() - start_time)

        optimizer.zero_grad()

        _, loss, _, _ = model.forward(u_input)

        batch_size = u_input.size(0)
        unlabeled_loss.update(loss.item(), batch_size)

        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - start_time)
        start_time = time.time()

        p_bar.set_description(f"Epoch {epoch+1} - Loss: {unlabeled_loss.avg:.4f}")
        p_bar.update()

    p_bar.close()
    scheduler.step()

    return unlabeled_loss.avg, None, None


def train(args, method_config):
    """ Train model

    Args:
        args (Namespace): parsed arguments
        method_config (HyperparamSpace): hyperparameter space for method

    Returns:
        float: best validation accuracy
        float: test accuracy
    """

    start_time = time.time()

    model, optimizer, scheduler, train_loader, unlabel_loader, \
        val_loader, test_loader = setup_training(args, method_config)
    writer = SummaryWriter(args.train_dir)
    args.writer = writer
    best_val_acc, total_time = 0, 0
    current_count = 0  # for early stopping, when continue training
    early_stopping = EarlyStopping(patience=args.patience, initial_count=current_count)

    for epoch in range(args.start_epoch, args.train_epoch):
        start_time = time.time()

        if unlabel_loader is not None:
            train_loss, logits, labels = train_unlabel_one_epoch(
                args, model, optimizer, scheduler, train_loader, unlabel_loader, epoch)

            print("Trained one epoch with unlabeled data")
        else:
            train_loss, logits, labels = train_one_epoch(
                args, model, optimizer, scheduler, train_loader, epoch)

        train_pred = logits.cpu().detach().numpy().argmax(axis=1)
        train_targets = labels.cpu().detach().numpy()
        train_acc = calculate_plain_accuracy(train_pred, train_targets)

        # Validation
        val_metrics = eval_model(args, val_loader, model, args.weights)

        if val_metrics['plain_accuracy'] > best_val_acc:
            best_val_acc = val_metrics['plain_accuracy']

        early_stopping(val_metrics['plain_accuracy'])

        # Logging
        print(f"Epoch {epoch+1} - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, ")
        writer.add_scalar('train/loss', train_loss, epoch)
        writer.add_scalar('train/accuracy', train_acc, epoch)
        writer.add_scalar('val/accuracy', val_metrics['plain_accuracy'], epoch)
        writer.add_scalar('val/loss', val_metrics['loss'], epoch)
        writer.add_scalar('val/balanced_accuracy', val_metrics['balanced_accuracy'], epoch)
        writer.add_scalar('val/auroc', val_metrics['auroc'], epoch)
        writer.add_scalar('val/auprc', val_metrics['auprc'], epoch)

        save_checkpoint(
            {
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_val_acc': best_val_acc,
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict()
            },
            val_metrics['plain_accuracy'] > best_val_acc,
            args.train_dir
        )

        total_time += time.time() - start_time

        print(f"Validation Accuracy: {val_metrics['plain_accuracy']:.4f}")
        print(f"Early Stopping Count: {early_stopping.counter}")

        if early_stopping.early_stop:
            print("Early stopping")
            break

        if total_time >= args.total_hour * 3600:
            print("Training time exceeded")
            break

    # Final Testing
    test_metrics = eval_model(args, test_loader, model, args.weights)
    test_acc = test_metrics['plain_accuracy']

    writer.add_scalar('test/accuracy', test_acc, epoch)
    writer.add_scalar('test/balanced_accuracy', test_metrics['balanced_accuracy'], epoch)
    writer.add_scalar('test/auroc', test_metrics['auroc'], epoch)
    writer.add_scalar('test/auprc', test_metrics['auprc'], epoch)
    writer.add_scalar('test/tpr_at_fpr_5', test_metrics['tpr_at_fpr_5'], epoch)

    with open(os.path.join(args.train_dir, 'training_summary.json'), 'w') as f:
        json.dump({'best_val_accuracy': best_val_acc, 'test_accuracy': test_acc,
                   'total_epochs': epoch + 1, 'total_time': total_time}, f)

    writer.close()
    return best_val_acc, test_metrics['plain_accuracy']


def main(args):
    method_config = method_configs[args.implementation]

    # loop until desired duration has elapsed
    start_time = time.time()
    best_val_acc = 0
    while time.time() - start_time <= args.total_hour * 3600:
        cur_best_val_acc, cur_test_acc = train(args, method_config)
        if cur_best_val_acc > best_val_acc:
            best_val_acc = cur_best_val_acc
            print(f"New Best Model! Val Acc: {best_val_acc}, Test Acc: {cur_test_acc}")
            # TODO how do we find the new best model later?


if __name__ == "__main__":
    # Set up logging
    logger = logging.getLogger(__name__)

    args = parse_args()
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    args.base_train_dir = args.train_dir

    print(f"Device: {args.device}")
    print(f"Base Train Dir: {args.base_train_dir}")
    print(f"Arguments: {vars(args)}")

    main(args)
