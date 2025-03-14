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


    
def train_one_epoch(args, weights, labeledtrain_loader, unlabeledtrain_loader, model, optimizer, scheduler, epoch, prob_list, queue_feats, queue_probs, queue_ptr):
    
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



    TotalLoss_this_epoch, LabeledLoss_this_epoch, UnlabeledLossUnscaled_this_epoch, UnlabeledLossScaled_this_epoch, ContrastiveLoss_this_epoch = [], [], [], [], []
    
    end_time = time.time()
    
    labeledtrain_iter = iter(labeledtrain_loader)
    unlabeledtrain_iter = iter(unlabeledtrain_loader)
        
    batch_time = AverageMeter()
    data_time = AverageMeter()
    total_loss = AverageMeter()
    labeled_loss = AverageMeter()
    unlabeled_loss_unscaled = AverageMeter()
    unlabeled_loss_scaled = AverageMeter()
    contrastive_loss = AverageMeter()
    mask_probs = AverageMeter() #how frequently the unlabeled samples' confidence score greater than pre-defined threshold
    
    n_steps_per_epoch = args.nimg_per_epoch//(args.labeledtrain_batchsize+args.unlabeledtrain_batchsize)
    
    p_bar = tqdm(range(n_steps_per_epoch), disable=False)
    
    for batch_idx in range(n_steps_per_epoch):
 
        try:
            ims_x_weak, lbs_x = labeledtrain_iter.next()
        except:
            labeledtrain_iter = iter(labeledtrain_loader)
            ims_x_weak, lbs_x = labeledtrain_iter.next()
        
        try:
            (ims_u_weak, ims_u_strong0, ims_u_strong1), u_labels = unlabeledtrain_iter.next()
        except:
            unlabeledtrain_iter = iter(unlabeledtrain_loader)
            (ims_u_weak, ims_u_strong0, ims_u_strong1), u_labels = unlabeledtrain_iter.next()
        
        
        data_time.update(time.time() - end_time)
        
        ##############################################################################################################
        bt = ims_x_weak.size(0)
        btu = ims_u_weak.size(0)

        lbs_x = lbs_x.to(args.device).long()

        
        imgs = torch.cat([ims_x_weak, ims_u_weak, ims_u_strong0, ims_u_strong1], dim=0).to(args.device)
        
        logits, features = model(imgs)
        
        logits_x = logits[:bt]
        logits_u_w, logits_u_s0, logits_u_s1 = torch.split(logits[bt:], btu)
        
        feats_x = features[:bt]
        feats_u_w, feats_u_s0, feats_u_s1 = torch.split(features[bt:], btu)
        
        labeledtrain_loss = F.cross_entropy(logits_x, lbs_x, weights, reduction='mean')
        
        
        with torch.no_grad():
            logits_u_w = logits_u_w.detach()
            feats_x = feats_x.detach()
            feats_u_w = feats_u_w.detach()
            
            probs = torch.softmax(logits_u_w, dim=1)   
            # DA
            prob_list.append(probs.mean(0))
            
            if len(prob_list)>32:
                prob_list.pop(0)
                
            prob_avg = torch.stack(prob_list,dim=0).mean(0)
            probs = probs / prob_avg
            probs = probs / probs.sum(dim=1, keepdim=True)   
            
            probs_orig = probs.clone()
            
            if epoch>0 or batch_idx>args.queue_batch: # memory-smoothing 
                A = torch.exp(torch.mm(feats_u_w, queue_feats.t())/args.temperature)
                A = A/A.sum(1,keepdim=True)        
                probs = args.alpha*probs + (1-args.alpha)*torch.mm(A, queue_probs)               
                
            scores, lbs_u_guess = torch.max(probs, dim=1)
            mask = scores.ge(args.threshold).float() 
            
            feats_w = torch.cat([feats_u_w,feats_x],dim=0)
            onehot = torch.zeros(bt,args.num_classes).to(args.device).scatter(1,lbs_x.view(-1,1),1)
            probs_w = torch.cat([probs_orig,onehot],dim=0)
            
            # update memory bank
            n = bt+btu  
            queue_feats[queue_ptr:queue_ptr + n,:] = feats_w
            queue_probs[queue_ptr:queue_ptr + n,:] = probs_w      
            queue_ptr = (queue_ptr+n)%args.queue_size
            
            
        # embedding similarity
        sim = torch.exp(torch.mm(feats_u_s0, feats_u_s1.t())/args.temperature) 
        sim_probs = sim / sim.sum(1, keepdim=True)
        
        # pseudo-label graph with self-loop
        Q = torch.mm(probs, probs.t())
        Q.fill_diagonal_(1) 
        pos_mask = (Q>=args.contrast_th).float()
        
        Q = Q * pos_mask
        Q = Q / Q.sum(1, keepdim=True)
        
        # contrastive loss
        loss_contrast = - (torch.log(sim_probs + 1e-7) * Q).sum(1)
        loss_contrast = loss_contrast.mean()  
        
        # unsupervised classification loss
        unlabeledtrain_loss = - torch.sum((F.log_softmax(logits_u_s0,dim=1) * probs),dim=1) * mask
        unlabeledtrain_loss = unlabeledtrain_loss.mean()
        
        
        
        current_lambda_u = args.lambda_u_max * current_warmup #FixMatch algo did not use unlabeled loss rampup schedule

        args.writer.add_scalar('train/lambda_u', current_lambda_u, epoch)
               
        loss = labeledtrain_loss + current_lambda_u * unlabeledtrain_loss + args.lam_c * loss_contrast
        
        
        loss.backward()
    
        
        total_loss.update(loss.item())
        labeled_loss.update(labeledtrain_loss.item())
        unlabeled_loss_unscaled.update(unlabeledtrain_loss.item())
        unlabeled_loss_scaled.update(unlabeledtrain_loss.item() * current_lambda_u)
        contrastive_loss.update(loss_contrast.item())
        mask_probs.update(mask.mean().item())

        TotalLoss_this_epoch.append(loss.item())
        LabeledLoss_this_epoch.append(labeledtrain_loss.item())
        UnlabeledLossUnscaled_this_epoch.append(unlabeledtrain_loss.item())
        UnlabeledLossScaled_this_epoch.append(unlabeledtrain_loss.item() * current_lambda_u)
        ContrastiveLoss_this_epoch.append(loss_contrast.item())
        
        optimizer.step()
        
        model.zero_grad()
        
        
        batch_time.update(time.time() - end_time)
        
        #update end time
        end_time = time.time()


        #tqdm display for each minibatch update
        p_bar.set_description("Train Epoch: {epoch}/{epochs:4}. Iter: {batch:4}/{iter:4}. LR: {lr:.4f}. Data: {data:.3f}s. Batch: {bt:.3f}s. Loss: {total_loss:.4f}. Loss_x: {labeled_loss:.4f}. Loss_u: {unlabeled_loss_unscaled:.4f}. Loss_contrast: {contrastive_loss:.4f}. Mask: {mask:.2f}. ".format(
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
                contrastive_loss=contrastive_loss.avg,
                mask=mask_probs.avg))
        p_bar.update()
        
        
        

    args.writer.add_scalar('train/gt_mask', mask_probs.avg, epoch)

    scheduler.step()

    p_bar.close()
        
    
    return TotalLoss_this_epoch, LabeledLoss_this_epoch, UnlabeledLossUnscaled_this_epoch, UnlabeledLossScaled_this_epoch, ContrastiveLoss_this_epoch, queue_feats, queue_probs, queue_ptr, prob_list
        
   
    
    

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
            raw_outputs,_ = raw_model(inputs)
            
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

        print('raw {} this evaluation step: {}'.format(evaluation_criterion, raw_performance), flush=True)
        
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
