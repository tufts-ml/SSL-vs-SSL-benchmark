import json
import logging
import os
import time


import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


from ssl_bench.config import dataset_configs, method_configs, HyperparamSpace
from ssl_bench.dataload import get_dataloaders
from ssl_bench.utils.train_utils import (AverageMeter, save_checkpoint,
                                         EarlyStopping, get_model,
                                         get_optimizer, get_lr_scheduler)
from ssl_bench.utils.eval_utils import (
    calculate_plain_accuracy,
    eval_model,
)
from ssl_bench.utils.arg_parser import parse_args


def setup_training(args, method_config: HyperparamSpace):
    """Set up training environment

    Args:
        args (Namespace): parsed arguments
        method_config (HyperparamSpace): hyperparameter space for method

    Raises:
        ValueError: if dataset not found in config

    Returns:
        tuple: model, optimizer, scheduler, train_loader, val_loader, test_loader
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
    logger.info(f"Loading dataset: {args.dataset_name}")
    train_loader, _, val_loader, test_loader = get_dataloaders(args)

    return model, optimizer, scheduler, train_loader, val_loader, test_loader


def train_one_epoch(args, model, optimizer, scheduler, train_loader, epoch):
    """Train model for one epoch

    Args:
        args (Namespace): parsed arguments
        model (torch.nn.Module): model to train
        optimizer (torch.optim.Optimizer): optimizer
        scheduler (torch.optim.lr_scheduler.LambdaLR): learning rate scheduler
        train_loader (torch.utils.data.DataLoader): training data loader
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

        # l_input, l_labels = l_input.to(args.device).float(), l_labels.to(args.device).long()

        optimizer.zero_grad()  # Zero gradients before backward pass

        logits, loss, supervised_loss, unsupervised_loss = model.forward(
            l_input, l_labels)

        total_loss = supervised_loss + (unsupervised_loss if unsupervised_loss is not None else 0)

        # Accumulate logits & labels
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

    model, optimizer, scheduler, train_loader, val_loader, test_loader = setup_training(
        args, method_config)
    writer = SummaryWriter(args.train_dir)
    args.writer = writer
    best_val_acc, total_time = 0, 0
    current_count = 0  # for early stopping, when continue training
    early_stopping = EarlyStopping(patience=args.patience, initial_count=current_count)

    for epoch in range(args.start_epoch, args.train_epoch):
        start_time = time.time()
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
