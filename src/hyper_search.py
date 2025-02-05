import argparse
from functools import partial
from pathlib import Path
import torch
from ray import tune
from ray.tune.schedulers import ASHAScheduler
import ray.cloudpickle as pickle
import time
import torch
import numpy as np
import json
import os
from torchvision import transforms


from src.utils.train_utils import save_checkpoint, get_cosine_schedule_with_warmup, get_fixed_lr
from src.utils.arg_parser import parse_args
from src.utils.apply_clahe import apply_clahe
import src.config as config
from torch.utils.tensorboard import SummaryWriter
from src.utils.eval_utils import (
    calculate_auprc,
    calculate_auroc,
    calculate_balanced_accuracy,
    eval_model,
    EarlyStopping
)
from src.libml.utils import train_one_epoch
from src.methods.LabelOnlyBaseline import LabelOnlyBaseline
from src.dataset_csv import LabeledImageCSVDataset, UnlabeledImageCSVDataset, CheXpertDataset


# TODO - Move this to a separate file?
def get_dataloaders(args):
    """Get DataLoaders

    Args:
        args (Namespace): parsed arguments

    Returns:
        tuple: 4 DataLoaders, which may be none
               train_loader, unlabel_loader, valid_loader, test_loader
    """
    dataset_name = args.dataset_name

    dataset_mean = config[args.dataset_name]['dataset_mean']
    dataset_std = config[args.dataset_name]['dataset_std']
    image_size = config[args.dataset_name]['image_size']

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

    # Process unlabeled data
    if args.u_train_dataset_path != '':
        unlabel_loader = UnlabeledImageCSVDataset(csv_file=args.u_train_dataset_path,
                                                  root_dir=args.root_dataset_path,
                                                  transform=transform_labeledtrain)
    else:
        unlabel_loader = None

    # Process labeled data
    if dataset_name == "CheXpert":
        dataset_class = CheXpertDataset
    else:
        dataset_class = LabeledImageCSVDataset
    if args.l_train_dataset_path != '':
        train_loader = dataset_class(csv_file=args.l_train_dataset_path,
                                     root_dir=args.root_dataset_path,
                                     transform=transform_labeledtrain)
    else:
        train_loader = None

    if args.val_dataset_path != '':
        valid_loader = dataset_class(csv_file=args.val_dataset_path,
                                     root_dir=args.root_dataset_path,
                                     transform=transform_eval)
    else:
        valid_loader = None

    if args.test_dataset_path != '':
        test_loader = dataset_class(csv_file=args.test_dataset_path,
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
    if args.arch == 'resnet18':
        from torchvision import models

        model = models.resnet18(pretrained=args.use_pretrained)
        model.fc = torch.nn.Linear(512, args.num_classes)

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

    # TODO - Implement other methods, this should dynamically load the method
    return LabelOnlyBaseline(model, args)


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

    os.makedirs(args.train_dir, exist_ok=True)
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

        # Update best scores
        is_best = val_acc > best_val_acc
        if is_best:
            best_val_acc = val_acc

        # Calculate metrics
        balanced_acc = calculate_balanced_accuracy(val_labels, val_preds)
        auroc = calculate_auroc(val_labels, val_preds)
        auprc = calculate_auprc(val_labels, val_preds)

        # Log metrics
        writer.add_scalar('train/loss', np.mean(train_losses), epoch)
        writer.add_scalar('val/accuracy', val_acc, epoch)
        writer.add_scalar('val/loss', val_loss, epoch)
        writer.add_scalar('val/balanced_accuracy', balanced_acc, epoch)
        writer.add_scalar('val/auroc', auroc, epoch)
        writer.add_scalar('val/auprc', auprc, epoch)

        # Save checkpoint
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_val_acc': best_val_acc,
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

    test_loss, test_acc, test_labels, test_preds = eval_model(args, test_loader, model, epoch)
    writer.add_scalar('test/accuracy', test_acc, epoch)
    writer.add_scalar('test/loss', test_loss, epoch)

    # Save final summary
    summary = {
        'best_val_accuracy': best_val_acc,
        'test_accuracy': test_acc,
        'total_epochs': epoch + 1,
        'total_time': total_time
    }

    with open(os.path.join(args.train_dir, 'training_summary.json'), 'w') as f:
        json.dump(summary, f)

    writer.close()
    return best_val_acc, test_acc


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
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # TODO Ray Tune hyperparameter search
    # https://pytorch.org/tutorials/beginner/hyperparameter_tuning_tutorial.html
    method_config = config.method_config[args.method]
    scheduler = ASHAScheduler(
        metric="val_acc",
        mode="max",
        max_t=args.train_epoch,
        grace_period=1,
        reduction_factor=2,
    )

    result = tune.run(
        partial(train, args=args),
        config=method_config,
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
    args = parse_args()
    main(args)
