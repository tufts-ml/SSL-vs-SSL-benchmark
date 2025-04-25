import json
import logging
import os
import time


from sklearn.linear_model import LogisticRegression
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np


from ssl_bench.config import dataset_configs, method_configs, HyperparamSpace
from ssl_bench.dataload import get_dataloaders
from ssl_bench.utils.train_utils import (AverageMeter, save_checkpoint,
                                         EarlyStopping, get_model,
                                         get_optimizer, get_lr_scheduler)
from ssl_bench.utils.eval_utils import (
    calculate_balanced_accuracy,
    eval_model,
    log_metrics_to_tensorboard
)
from ssl_bench.utils.arg_parser import parse_args


def set_seed(seed):
    """Set random seed for reproducibility

    Args:
        seed (int): random seed
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


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
    logging.info(f"Hypers: {hyper_strs}")

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


def train_one_epoch(
    args, model, optimizer, scheduler, epoch, label_loader=None, unlabel_loader=None
):
    """Train model for one epoch

    Args:
        args (Namespace): parsed arguments
        model (torch.nn.Module): model to train
        optimizer (torch.optim.Optimizer): optimizer
        scheduler (torch.optim.lr_scheduler.LambdaLR): learning rate scheduler
        epoch (int): current epoch
        label_loader (torch.utils.data.DataLoader): training data loader
        unlabel_loader (torch.utils.data.DataLoader): unlabeled data loader

    Returns:
        float: average loss for the epoch
        torch.Tensor: logits from the model
        torch.Tensor: labels from the dataset
    """

    model.train()
    args.writer.add_scalar('train/lr', scheduler.get_last_lr()[0], epoch)
    print(f"Epoch {epoch+1} - Learning Rate: {scheduler.get_last_lr()[0]}")

    batch_time, data_time, total_loss = AverageMeter(), AverageMeter(), AverageMeter()

    all_logits = []
    all_labels = []

    if label_loader is not None:
        labeledtrain_iter = iter(label_loader)

    if unlabel_loader is not None:
        unlabeledtrain_iter = iter(unlabel_loader)

    n_steps_per_epoch = args.nimg_per_epoch // args.labeledtrain_batchsize

    p_bar = tqdm(range(n_steps_per_epoch), disable=False)

    start_time = time.time()

    for _ in range(n_steps_per_epoch):
        data_time.update(time.time() - start_time)

        optimizer.zero_grad()  # Zero gradients before backward pass

        if label_loader is not None:
            try:
                l_input, l_labels = next(labeledtrain_iter)
            except StopIteration:
                labeledtrain_iter = iter(label_loader)
                l_input, l_labels = next(labeledtrain_iter)
        else:
            l_input, l_labels = None, None

        if unlabel_loader is not None:
            try:
                u_input = next(unlabeledtrain_iter)
            except StopIteration:
                unlabeledtrain_iter = iter(unlabel_loader)
                u_input = next(unlabeledtrain_iter)
        else:
            u_input = None

        logits, loss, supervised_loss, unsupervised_loss = model.forward(
            l_input, l_labels, u_input)

        if logits is not None:
            all_logits.append(logits.detach().cpu())
        if l_labels is not None:
            all_labels.append(l_labels.detach().cpu())

        # Weighted update for correct loss averaging
        batch_size = l_labels.size(0) if l_labels is not None else u_input.size(0)
        total_loss.update(loss, batch_size)

        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - start_time)
        start_time = time.time()

        p_bar.set_description(f"Epoch {epoch+1} - Loss: {total_loss.avg:.4f}")
        p_bar.update()

    p_bar.close()
    scheduler.step()

    all_logits = torch.cat(all_logits) if all_logits else None
    all_labels = torch.cat(all_labels) if all_labels else None

    return total_loss.avg, all_logits, all_labels


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

    model, optimizer, scheduler, label_loader, unlabel_loader, \
        val_loader, test_loader = setup_training(args, method_config)
    writer = SummaryWriter(args.train_dir)
    args.writer = writer
    best_val_acc, total_time = 0, 0
    early_stopping = EarlyStopping(patience=args.patience)

    clf = None

    for epoch in range(args.start_epoch, args.train_epoch):
        start_time = time.time()

        train_loss, probs, labels = train_one_epoch(
            args, model, optimizer, scheduler, epoch, label_loader, unlabel_loader)

        if args.implementation in ['BarlowTwins', 'SimCLR']:
            model.eval()

            all_features = []
            all_labels = []

            for batch_idx, (l_input, l_labels) in enumerate(label_loader):
                l_input, l_labels = l_input.to(
                    args.device, non_blocking=True), l_labels.to(args.device, non_blocking=True)

                with torch.no_grad():
                    features = model.eval_forward(l_input, l_labels)[0]

                all_features.append(features.cpu())
                all_labels.append(l_labels.cpu())

            all_features = torch.cat(all_features)
            all_labels = torch.cat(all_labels)

            val_features = []
            val_labels = []
            for batch_idx, (l_input, l_labels) in enumerate(val_loader):
                l_input, l_labels = l_input.to(
                    args.device, non_blocking=True), l_labels.to(args.device, non_blocking=True)

                with torch.no_grad():
                    features = model.eval_forward(l_input, l_labels)[0]

                val_features.append(features.cpu())
                val_labels.append(l_labels.cpu())
            val_features = torch.cat(val_features)
            val_labels = torch.cat(val_labels)

            if (epoch % 5) == 0:
                clf = fit_logistic_regression(
                    args, model, all_features, all_labels, val_features, val_labels)

            probs = clf.predict_proba(all_features)
            probs = torch.tensor(probs, dtype=torch.float32).to(args.device)
            labels = all_labels.to(args.device)

        train_pred = probs.cpu().detach().numpy().argmax(axis=1)
        train_targets = labels.cpu().detach().numpy()
        train_acc = calculate_balanced_accuracy(train_pred, train_targets)

        val_metrics = eval_model(args, val_loader, model, args.weights, clf)

        if val_metrics['balanced_accuracy'] > best_val_acc:
            best_val_acc = val_metrics['balanced_accuracy']

        early_stopping(val_metrics['balanced_accuracy'])

        # Logging
        print(f"Epoch {epoch+1} - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, ")
        writer.add_scalar('train/loss', train_loss, epoch)
        writer.add_scalar('train/balanced_accuracy', train_acc, epoch)
        log_metrics_to_tensorboard(writer, epoch, 'val', val_metrics)

        save_checkpoint(
            {
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_val_acc': best_val_acc,
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict()
            },
            val_metrics['balanced_accuracy'] > best_val_acc,
            args.train_dir
        )

        total_time += time.time() - start_time

        print(f"Validation Accuracy: {val_metrics['balanced_accuracy']:.4f}")
        print(f"Early Stopping Count: {early_stopping.counter}")

        if early_stopping.early_stop:
            print("Early stopping")
            break

        if total_time >= args.total_hour * 3600:
            print("Training time exceeded")
            break

    # Final Testing
    test_metrics = eval_model(args, test_loader, model, args.weights, clf)
    test_acc = test_metrics['balanced_accuracy']

    log_metrics_to_tensorboard(writer, epoch, 'test', test_metrics)

    with open(os.path.join(args.train_dir, 'training_summary.json'), 'w') as f:
        json.dump({'best_val_accuracy': best_val_acc, 'test_accuracy': test_acc,
                   'total_epochs': epoch + 1, 'total_time': total_time}, f)

    writer.close()
    return best_val_acc, test_metrics['balanced_accuracy']


def fit_logistic_regression(args, model, train_features, train_labels, val_features, val_labels):
    """Fit logistic regression model

    Args:
        args (Namespace): parsed arguments
        model (torch.nn.Module): model to train
        train_features (torch.Tensor): training features
        train_labels (torch.Tensor): training labels
        val_features (torch.Tensor): validation features
        val_labels (torch.Tensor): validation labels

    Returns:
        LogisticRegression: fitted logistic regression model
    """
    train_features, train_labels = train_features.cpu().numpy(), train_labels.cpu().numpy()
    val_features, val_labels = val_features.cpu().numpy(), val_labels.cpu().numpy()

    reg = 10 ** np.random.uniform(-3, 3, size=10)
    best_val_acc = 0
    best_clf = None

    for i in range(10):
        clf = LogisticRegression(random_state=args.seed,
                                 C=reg[i], max_iter=1000,
                                 class_weight='balanced')
        clf.fit(train_features, train_labels)

        val_probs = clf.predict_proba(val_features)
        val_probs = torch.tensor(val_probs, dtype=torch.float32).to(args.device)
        val_labels = val_labels.to(args.device)
        val_pred = val_probs.cpu().detach().numpy().argmax(axis=1)
        val_targets = val_labels.cpu().detach().numpy()
        val_acc = calculate_balanced_accuracy(val_pred, val_targets)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_clf = clf
            print(f"New Best Classifier! Val Acc: {best_val_acc}")
            print(f"L2 Regularization: {clf.C}")

    return best_clf


def main(args):
    method_config = method_configs[args.implementation]
    log_path = os.path.join(args.train_dir, 'logging.log')
    os.makedirs(args.train_dir, exist_ok=True)
    logging.basicConfig(filename=log_path, encoding='utf-8', level=logging.INFO)
    logging.info(f"Args: {args}")
    set_seed(args.seed)

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
