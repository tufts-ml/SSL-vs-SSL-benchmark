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
            
            
    
#https://github.com/YU1ut/MixMatch-pytorch/blob/cc7ef42cffe61288d06eec1428268b384674009a/train.py
def interleave_offsets(batch, nu):
    groups = [batch // (nu + 1)] * (nu + 1)
    for x in range(batch - sum(groups)):
        groups[-x - 1] += 1
    offsets = [0]
    for g in groups:
        offsets.append(offsets[-1] + g)
    assert offsets[-1] == batch
    return offsets

#https://github.com/YU1ut/MixMatch-pytorch/blob/cc7ef42cffe61288d06eec1428268b384674009a/train.py
def interleave(xy, batch):
    nu = len(xy) - 1
    offsets = interleave_offsets(batch, nu)
    xy = [[v[offsets[p]:offsets[p + 1]] for p in range(nu + 1)] for v in xy]
    for i in range(1, nu + 1):
        xy[0][i], xy[i][i] = xy[i][i], xy[0][i]
    return [torch.cat(v, dim=0) for v in xy]

    
def train_one_epoch(args, weights, labeledtrain_loader, unlabeledtrain_loader, model, optimizer, scheduler, epoch):
    
    '''
    this implementation follow: https://github.com/perrying/realistic-ssl-evaluation-pytorch/blob/master/lib/algs/pseudo_label.py
    #same as Oliver et al 2018, which use vanilla logits when unlabeled samples has maximum probability below threshold
    '''
    
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
            l_inputs, l_labels = labeledtrain_iter.next()
        except:
            labeledtrain_iter = iter(labeledtrain_loader)
            l_inputs, l_labels = labeledtrain_iter.next()
        
        try:
            (inputs_u, inputs_u2), u_labels = unlabeledtrain_iter.next()
        except:
            unlabeledtrain_iter = iter(unlabeledtrain_loader)
            (inputs_u, inputs_u2), u_labels = unlabeledtrain_iter.next()
        
        
        data_time.update(time.time() - end_time)
        
        ##############################################################################################################
        #For MM
        #reference: https://github.com/YU1ut/MixMatch-pytorch/blob/master/train.py
        
        batch_size = l_inputs.size(0)
        
        # Transform label to one-hot
        l_labels = torch.zeros(batch_size, args.num_classes).scatter_(1, l_labels.view(-1,1).long(), 1)
         
        #put data to device
        l_inputs, l_labels = l_inputs.to(args.device).float(), l_labels.to(args.device).long()
        
        inputs_u = inputs_u.to(args.device)
        inputs_u2 = inputs_u2.to(args.device)
            
        
        #label guessing
        with torch.no_grad():
            #compute guessed labels of unlabel samples
#             print('start label guessing')
            u_output1 = model(inputs_u)
            u_output2 = model(inputs_u2)
#             print('end label guessing')
            #for debugging:
#             print('u_output1 is {}'.format(u_output1))
#             print('u_output2 is {}'.format(u_output2))


            p = (torch.softmax(u_output1, dim=1) + torch.softmax(u_output2, dim=1)) / 2
            pt = p**(1/args.temperature)
            
            u_targets = pt/pt.sum(dim=1, keepdim=True)
            u_targets = u_targets.detach()
            
        
        combined_inputs = torch.cat([l_inputs, inputs_u, inputs_u2], dim=0)
        combined_labels = torch.cat([l_labels, u_targets, u_targets], dim=0)
        
        l = np.random.beta(args.alpha, args.alpha)
        l = max(l, 1-l)
        
        idx = torch.randperm(combined_inputs.size(0))
        
        input_a, input_b = combined_inputs, combined_inputs[idx]
        target_a, target_b = combined_labels, combined_labels[idx]
        
        mixed_input = l * input_a + (1-l) * input_b
        mixed_target = l * target_a + (1-l) * target_b
        
        #interleave labeled and unlabeled samples between batches to get correct batchnorm calculation
        mixed_input = list(torch.split(mixed_input, batch_size))
        mixed_input = interleave(mixed_input, batch_size)
        
        logits = [model(mixed_input[0])]
        for input in mixed_input[1:]:
            logits.append(model(input))
            
        #put interleaved samples back
        logits = interleave(logits, batch_size)
        logits_x = logits[0]
        logits_u = torch.cat(logits[1:], dim=0)
        
        #unlabeled loss
        probs_u = torch.softmax(logits_u, dim=1)
        unlabeledtrain_loss = torch.mean((probs_u - mixed_target[batch_size:])**2)
        
        #Lx, Lu, w = criterion(logits_x, mixed_target[:batch_size], logits_u, mixed_target[batch_size:], epoch+batch_idx/args.train_iteration)

        #labeled loss
#         labeledtrain_loss = -torch.mean(torch.sum(F.log_softmax(logits_x, dim=1) * mixed_target[:batch_size], dim=1))
        labeledtrain_loss = F.cross_entropy(logits_x, mixed_target[:batch_size], weights, reduction='mean')
        
        #from FixMatch and MixMatch: warmup = tf.clip_by_value(tf.to_float(self.step) / (warmup_pos * (FLAGS.train_kimg << 10)), 0, 1)   
        
        current_lambda_u = args.lambda_u_max * current_warmup
        args.writer.add_scalar('train/lambda_u', current_lambda_u, epoch)

        loss = labeledtrain_loss + current_lambda_u * unlabeledtrain_loss
        
        if args.em > 0:
#             loss -= args.em * ((combined_outputs.softmax(1) * F.log_softmax(combined_outputs, 1)).sum(1) * unlabeled_mask).mean()
            loss -= args.em * ((logits_u.softmax(1) * F.log_softmax(logits_u, 1)).sum(1)).mean()
        
        ###############################################################################################################
        
        #just to see how the confidence on unlabeled data change over time
        pseudo_label = torch.softmax(logits_u.detach(), dim=-1)
        max_probs, _ = torch.max(pseudo_label, dim=-1)
        mask = max_probs.ge(0.95).float()
        
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
        
        #update ema model
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
        
        
        
##for debugging
#     print('fc.weight: {}'.format(model.fc.weight.cpu().detach().numpy()))
#     print('output.bias: {}'.format(model.output.bias.cpu().detach().numpy()))
        
#     print('ema fc.weight: {}'.format(ema_model.ema.fc.weight.cpu().detach().numpy()))
#     print('ema output.bias: {}'.format(ema_model.ema.output.bias.cpu().detach().numpy()))
        
    p_bar.close()
    scheduler.step()

    
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
