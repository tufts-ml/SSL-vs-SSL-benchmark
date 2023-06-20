import time
from tqdm import tqdm
import torch.nn.functional as F

import logging
import numpy as np
import os
import pickle

import torch
from sklearn.metrics import confusion_matrix as sklearn_cm


logger = logging.getLogger(__name__)

__all__ = ['get_mean_and_std', 'AverageMeter', 'train_one_epoch', 'eval_model', 'save_pickle', 'calculate_plain_accuracy', 'calculate_balanced_accuracy', 'EarlyStopping']


class EarlyStopping:
    """Early stops the training if validation acc doesn't improve after a given patience."""
    
    def __init__(self, patience=20, initial_count=0, delta=0):
        
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 20
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            
        """
        
        self.patience = patience
        self.counter = initial_count
        self.best_score = None
        self.early_stop = False
        self.delta = delta
        
    
    def __call__(self, val_acc):
        
        score = val_acc
        
        if self.best_score is None:
            self.best_score = score
        
        elif score <= self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        
        else:
            self.best_score = score
            self.counter = 0
            
            


def interleave(x, size):
    s = list(x.shape)
    return x.reshape([-1, size] + s[1:]).transpose(0, 1).reshape([-1] + s[1:])


def de_interleave(x, size):
    s = list(x.shape)
    return x.reshape([size, -1] + s[1:]).transpose(0, 1).reshape([-1] + s[1:])


    
def train_one_epoch(args, weights, labeledtrain_loader, unlabeledtrain_loader, model, optimizer, scheduler, epoch):
    
    model.train()

    args.writer.add_scalar('train/lr', scheduler.get_last_lr()[0], epoch)
     #unlabeledloss warmup schedule choice
    if args.unlabeledloss_warmup_schedule_type == 'NoWarmup':
        current_warmup = 1
    elif args.unlabeledloss_warmup_schedule_type == 'Linear':
        current_warmup = np.clip(epoch/(float(args.unlabeledloss_warmup_pos) * args.train_epoch), 0, 1)
    elif args.unlabeledloss_warmup_schedule_type == 'Sigmoid':
        current_warmup = math.exp(-5 * (1 - min(epoch/(float(args.unlabeledloss_warmup_pos) * args.train_epoch), 1))**2)
    else:
        raise NameError('Not supported unlabeledloss warmup schedule')



    TotalLoss_this_epoch, LabeledLoss_this_epoch, UnlabeledLossUnscaled_this_epoch, UnlabeledLossScaled_this_epoch = [], [], [], []
    
    end_time = time.time()
    
    labeledtrain_iter = iter(labeledtrain_loader)
    unlabeledtrain_iter = iter(unlabeledtrain_loader)
        
    batch_time = AverageMeter()
    data_time = AverageMeter()
    total_loss = AverageMeter()
    labeled_loss = AverageMeter()
    unlabeled_loss_unscaled = AverageMeter()
    unlabeled_loss_scaled = AverageMeter()
    mask_probs = AverageMeter() #how frequently the unlabeled samples' confidence score greater than pre-defined threshold
    
    n_steps_per_epoch = args.nimg_per_epoch//(args.labeledtrain_batchsize+args.unlabeledtrain_batchsize)
    
    p_bar = tqdm(range(n_steps_per_epoch), disable=False)
    
    for batch_idx in range(n_steps_per_epoch):
 
        try:
            l_input, l_labels = labeledtrain_iter.next()
        except:
            labeledtrain_iter = iter(labeledtrain_loader)
            l_input, l_labels = labeledtrain_iter.next()
        
        try:
            (u_input_weak, u_input_strong), u_labels = unlabeledtrain_iter.next()
        except:
            unlabeledtrain_iter = iter(unlabeledtrain_loader)
            (u_input_weak, u_input_strong), u_labels = unlabeledtrain_iter.next()
        
        
        data_time.update(time.time() - end_time)
        
        ##############################################################################################################
        #For FM:
        #reference: https://github.com/kekmodel/FixMatch-pytorch/blob/master/train.py
        inputs = interleave(torch.cat((l_input, u_input_weak, u_input_strong)), 2*args.mu+1).to(args.device)
        l_labels = l_labels.to(args.device).long()
        
        logits = model(inputs)
        logits = de_interleave(logits, 2*args.mu+1)
        logits_x = logits[:args.labeledtrain_batchsize]
        logits_u_w, logits_u_s = logits[args.labeledtrain_batchsize:].chunk(2)
        
        del logits
        
        labeledtrain_loss = F.cross_entropy(logits_x, l_labels, weights, reduction='mean')
        
        #label guessing
        pseudo_label = torch.softmax(logits_u_w.detach()/args.temperature, dim=-1)
        max_probs, targets_u = torch.max(pseudo_label, dim=-1)
        mask = max_probs.ge(args.threshold).float()
        
        unlabeledtrain_loss = (F.cross_entropy(logits_u_s, targets_u, reduction='none') * mask).mean()
        
        current_lambda_u = args.lambda_u_max * current_warmup #FixMatch algo did not use unlabeled loss rampup schedule

        args.writer.add_scalar('train/lambda_u', current_lambda_u, epoch)
       
        loss = labeledtrain_loss + current_lambda_u * unlabeledtrain_loss
        
#         print('mask is {}'.format(mask))
        
        if args.em > 0:
            raise NameError('Need to think about how to use em regularization in FixMatch')
# #             loss -= args.em * ((combined_outputs.softmax(1) * F.log_softmax(combined_outputs, 1)).sum(1) * unlabeled_mask).mean()
#             loss -= args.em * ((logits_u.softmax(1) * F.log_softmax(logits_u, 1)).sum(1)).mean()
        
        ###############################################################################################################
        
        loss.backward()
        
        total_loss.update(loss.item())
        labeled_loss.update(labeledtrain_loss.item())
        unlabeled_loss_unscaled.update(unlabeledtrain_loss.item())
        unlabeled_loss_scaled.update(unlabeledtrain_loss.item() * current_lambda_u)
        mask_probs.update(mask.mean().item())

        TotalLoss_this_epoch.append(loss.item())
        LabeledLoss_this_epoch.append(labeledtrain_loss.item())
        UnlabeledLossUnscaled_this_epoch.append(unlabeledtrain_loss.item())
        UnlabeledLossScaled_this_epoch.append(unlabeledtrain_loss.item() * current_lambda_u)
        
        optimizer.step()
        
#         #update ema model
#         ema_model.update(model)
        
        model.zero_grad()
        
        
        batch_time.update(time.time() - end_time)
        
        #update end time
        end_time = time.time()


        #tqdm display for each minibatch update
        p_bar.set_description("Train Epoch: {epoch}/{epochs:4}. Iter: {batch:4}/{iter:4}. LR: {lr:.4f}. Data: {data:.3f}s. Batch: {bt:.3f}s. Loss: {total_loss:.4f}. Loss_x: {labeled_loss:.4f}. Loss_u: {unlabeled_loss_unscaled:.4f}. Mask: {mask:.2f}. ".format(
                epoch=epoch + 1,
                epochs=args.train_epoch,
                batch=batch_idx + 1,
                iter=n_steps_per_epoch,
                lr=scheduler.get_last_lr()[0],
                data=data_time.avg,
                bt=batch_time.avg,
                total_loss=total_loss.avg,
                labeled_loss=labeled_loss.avg,
                unlabeled_loss_unscaled=unlabeled_loss_unscaled.avg,
                mask=mask_probs.avg))
        p_bar.update()
        
        
    
    args.writer.add_scalar('train/gt_mask', mask_probs.avg, epoch)

    scheduler.step()

    p_bar.close()
        
    
    return TotalLoss_this_epoch, LabeledLoss_this_epoch, UnlabeledLossUnscaled_this_epoch, UnlabeledLossScaled_this_epoch
        
   
    
    
    


#shared helper fct across different algos
def eval_model(args, data_loader, raw_model, epoch, evaluation_criterion, weights=None):
    
    if evaluation_criterion == 'plain_accuracy':
        evaluation_method = calculate_plain_accuracy
    elif evaluation_criterion == 'balanced_accuracy':
        evaluation_method = calculate_balanced_accuracy
    else:
        raise NameError('not supported yet')
    
    raw_model.eval()

    end_time = time.time()
    
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    
    data_loader = tqdm(data_loader, disable=False)
    
    with torch.no_grad():
        total_targets = []
        total_raw_outputs = []
        
        for batch_idx, (inputs, targets) in enumerate(data_loader):
            data_time.update(time.time() - end_time)
            
            inputs = inputs.to(args.device).float()
            targets = targets.to(args.device).long()
            raw_outputs = raw_model(inputs)
            
            total_targets.append(targets.detach().cpu())
            total_raw_outputs.append(raw_outputs.detach().cpu())
            
            if weights is not None:
                print('calculating weighted loss inside eval')
                loss = F.cross_entropy(raw_outputs, targets, weights)
            else:
                loss = F.cross_entropy(raw_outputs, targets)
            
            losses.update(loss.item(), inputs.shape[0])
            batch_time.update(time.time() - end_time)
            
            #update end time
            end_time = time.time()
            
            
        total_targets = np.concatenate(total_targets, axis=0)
        total_raw_outputs = np.concatenate(total_raw_outputs, axis=0)
        
        raw_performance = evaluation_method(total_raw_outputs, total_targets)

        
        data_loader.close()
        
        
    return losses.avg, raw_performance, total_targets, total_raw_outputs
    

#shared helper fct across different algos
def calculate_plain_accuracy(output, target):
    
    accuracy = (output.argmax(1) == target).mean()*100
    
    return accuracy


def calculate_balanced_accuracy(output, target):
    
    confusion_matrix = sklearn_cm(target, output.argmax(1))
    n_class = confusion_matrix.shape[0]
    print('Inside calculate_balanced_accuracy, {} classes passed in'.format(n_class), flush=True)
    
    recalls = []
    for i in range(n_class): 
        recall = confusion_matrix[i,i]/np.sum(confusion_matrix[i])
        recalls.append(recall)
        print('class{} recall: {}'.format(i, recall), flush=True)
        
    balanced_accuracy = np.mean(np.array(recalls))
    
    return balanced_accuracy * 100


#shared helper fct across different algos
def save_pickle(save_dir, save_file_name, data):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    data_save_fullpath = os.path.join(save_dir, save_file_name)
    with open(data_save_fullpath, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
#shared helper fct across different algos
def get_mean_and_std(dataset):
    '''Compute the mean and std value of dataset.'''
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=4)

    mean = torch.zeros(3)
    std = torch.zeros(3)
    logger.info('==> Computing mean and std..')
    for inputs, targets in dataloader:
        for i in range(3):
            mean[i] += inputs[:, i, :, :].mean()
            std[i] += inputs[:, i, :, :].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std


#shared helper fct across different algos
class AverageMeter(object):
    """Computes and stores the average and current value
       Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
