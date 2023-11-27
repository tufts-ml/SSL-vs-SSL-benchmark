import argparse
import os
import pandas

import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from libml.model import ModelMoCo
from libml import utils
from libml.utils import EarlyStopping

from datetime import datetime
from functools import partial
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torchvision.models import resnet
from tqdm import tqdm
import argparse
import json
import math
import os
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression

import time

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('Home device: {}'.format(device))



from sklearn.metrics import confusion_matrix as sklearn_cm

def calculate_balanced_accuracy(output, target):
    
    confusion_matrix = sklearn_cm(target, output)
    n_class = confusion_matrix.shape[0]

    recalls = []
    for i in range(n_class): 
        recall = confusion_matrix[i,i]/np.sum(confusion_matrix[i])
        recalls.append(recall)

    balanced_accuracy = np.mean(np.array(recalls))
    
    return balanced_accuracy * 100

def log_n_uniform(low=-3, high=0, size=1, coefficient=1, base=10):
    power_value = np.random.uniform(low, high, size)[0]
    
    return coefficient*np.power(base, power_value)
    
def uniform(low=0.0, high=1.0, size=1, decimal=1):
    
    return np.random.uniform(low=low, high=high, size=size)[0]
    
def train(net, data_loader, train_optimizer, epoch, args):
    net.train()
    adjust_learning_rate(optimizer, epoch, args)

    total_loss, total_num, train_bar = 0.0, 0, tqdm(data_loader)
    for im_1, im_2, target in train_bar:
        im_1, im_2 = im_1.cuda(non_blocking=True), im_2.cuda(non_blocking=True)

        loss = net(im_1, im_2)
        train_optimizer.zero_grad()
        loss.backward()
        train_optimizer.step()

        total_num += data_loader.batch_size
        total_loss += loss.item() * data_loader.batch_size
        #train_bar.set_description('Train Epoch: [{}/{}], lr: {:.6f}, Loss: {:.4f}'.format(epoch, args.epochs, optimizer.param_groups[0]['lr'], total_loss / total_num))

    return total_loss / total_num

# lr scheduler for training
def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate based on schedule"""
    lr = args.lr
    if args.cos:  # cosine lr schedule
        lr *= 0.5 * (1. + math.cos(math.pi * epoch / args.epochs))
    else:  # stepwise lr schedule
        for milestone in args.schedule:
            lr *= 0.1 if epoch >= milestone else 1.
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr



def test(net, memory_data_loader, test_data_loader):
    net.eval()
    total_top1, total_top5, total_num, feature_bank, label_bank = 0.0, 0.0, 0, [], []
    with torch.no_grad():
        # generate feature bank
        for data, _, target in tqdm(memory_data_loader, desc='Feature extracting'):
            feature = net.get_rep(data.to(device, non_blocking=True))
            feature_bank.append(feature)
            label_bank.append(target)
            
        # [D, N]
        feature_bank = torch.cat(feature_bank, dim=0).t().contiguous()
        label_bank = torch.cat(label_bank, dim=0).t().contiguous()
        # [N]

        feature_labels = torch.tensor(label_bank, device=feature_bank.device) 
        feature_bank = feature_bank.T.detach().cpu().numpy()
        label_bank = label_bank.numpy()
        clf = LogisticRegression(random_state=0, class_weight='balanced').fit(feature_bank, label_bank)
        
        test_bar = tqdm(test_data_loader)
        label_test_bank = []
        label_test_pred = []
        for data, _, target in test_bar:
            total_num = total_num + len(data)
            data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
            feature = net.get_rep(data.to(device, non_blocking=True))
            #top_1 = len(np.where(clf.predict(feature.cpu().detach().numpy()) == target.cpu().numpy())[0])
            #total_top1 += top_1
            #print(total_top1)
            label_test_bank = label_test_bank + list(target.cpu().detach().numpy())
            label_test_pred = label_test_pred + list(clf.predict(feature.cpu().detach().numpy()))
            
        balanced_accuracy = calculate_balanced_accuracy(label_test_pred, label_test_bank)
        print(balanced_accuracy)
            

    return balanced_accuracy

# test for one epoch, use weighted knn to find the most similar images' label to assign the test image
def pred_val(net, memory_data_loader, test_data_loader):
    net.eval()
    total_top1, total_top5, total_num, feature_bank, label_bank = 0.0, 0.0, 0, [], []
    val_max_all = []
    reg_all = []
    with torch.no_grad():
        # generate feature bank
        for data, _, target in tqdm(memory_data_loader, desc='Feature extracting'):
            #print(target)
            feature = net.get_rep(data.to(device, non_blocking=True))
            feature_bank.append(feature)
            label_bank.append(target)
            
        # [D, N]
        feature_bank = torch.cat(feature_bank, dim=0).t().contiguous()
        label_bank = torch.cat(label_bank, dim=0).t().contiguous()
        # [N]

        feature_labels = torch.tensor(label_bank, device=feature_bank.device) 
        feature_bank = feature_bank.T.detach().cpu().numpy()
        label_bank = label_bank.numpy()
        for i in range(10):
            reg = log_n_uniform(-1, 1)
            clf = LogisticRegression(random_state=0, class_weight='balanced', C = reg).fit(feature_bank, label_bank)
            
            test_bar = tqdm(test_data_loader)
            label_test_bank = []
            label_test_pred = []
            for data, _, target in test_bar:
                total_num = total_num + len(data)
                data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
                feature = net.get_rep(data.to(device, non_blocking=True))
                #top_1 = len(np.where(clf.predict(feature.cpu().detach().numpy()) == target.cpu().numpy())[0])
                #total_top1 += top_1
                #print(total_top1)
                label_test_bank = label_test_bank + list(target.cpu().detach().numpy())
                label_test_pred = label_test_pred + list(clf.predict(feature.cpu().detach().numpy()))
                
            balanced_accuracy = calculate_balanced_accuracy(label_test_pred, label_test_bank)
            print(balanced_accuracy, reg)
            val_max_all.append(balanced_accuracy)
            reg_all.append(reg)
          
    print(int(np.where(val_max_all == max(val_max_all))[0][0]))  
    return val_max_all[int(np.where(val_max_all == max(val_max_all))[0][0])], reg_all[int(np.where(val_max_all == max(val_max_all))[0][0])]

# test for one epoch, use weighted knn to find the most similar images' label to assign the test image
def pred_test(net, memory_data_loader, test_data_loader, reg):
    net.eval()
    total_top1, total_top5, total_num, feature_bank, label_bank = 0.0, 0.0, 0, [], []
    with torch.no_grad():
        # generate feature bank
        for data, _, target in tqdm(memory_data_loader, desc='Feature extracting'):
            #print(target)
            feature = net.get_rep(data.to(device, non_blocking=True))
            feature_bank.append(feature)
            label_bank.append(target)
            
        # [D, N]
        feature_bank = torch.cat(feature_bank, dim=0).t().contiguous()
        label_bank = torch.cat(label_bank, dim=0).t().contiguous()
        # [N]

        feature_labels = torch.tensor(label_bank, device=feature_bank.device) 
        feature_bank = feature_bank.T.detach().cpu().numpy()
        label_bank = label_bank.numpy()
        clf = LogisticRegression(random_state=0, class_weight='balanced', C = reg).fit(feature_bank, label_bank)
        
        test_bar = tqdm(test_data_loader)
        label_test_bank = []
        label_test_pred = []
        for data, _, target in test_bar:
            total_num = total_num + len(data)
            data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
            feature = net.get_rep(data.to(device, non_blocking=True))
            #top_1 = len(np.where(clf.predict(feature.cpu().detach().numpy()) == target.cpu().numpy())[0])
            #total_top1 += top_1
            #print(total_top1)
            label_test_bank = label_test_bank + list(target.cpu().detach().numpy())
            label_test_pred = label_test_pred + list(clf.predict(feature.cpu().detach().numpy()))
            
        balanced_accuracy = calculate_balanced_accuracy(label_test_pred, label_test_bank)
        print(balanced_accuracy)
            

    return balanced_accuracy

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train MoCo on CIFAR-10')
    parser.add_argument('-a', '--arch', default='resnet18')
    # lr: 0.06 for batch 512 (or 0.03 for batch 256)
    parser.add_argument('--lr', '--learning-rate', default=0.06, type=float, metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--epochs', default=200, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('--schedule', default=[120, 160], nargs='*', type=int, help='learning rate schedule (when to drop lr by 10x); does not take effect if --cos is on')
    parser.add_argument('--cos', action='store_true', help='use cosine lr schedule')
    parser.add_argument('--batch_size', default=512, type=int, metavar='N', help='mini-batch size')
    parser.add_argument('--wd', default=5e-4, type=float, metavar='W', help='weight decay')
    # moco specific configs:
    parser.add_argument('--moco-dim', default=128, type=int, help='feature dimension')
    parser.add_argument('--moco-k', default=16384, type=int, help='queue size; number of negative keys')
    parser.add_argument('--moco-m', default=0.99, type=float, help='moco momentum of updating key encoder')
    parser.add_argument('--moco-t', default=0.1, type=float, help='softmax temperature')
    parser.add_argument('--bn-splits', default=8, type=int, help='simulate multi-gpu behavior of BatchNorm in one gpu; 1 is SyncBatchNorm in multi-gpu')
    parser.add_argument('--symmetric', action='store_true', help='use a symmetric loss function that backprops to both crops')
    # knn monitor
    parser.add_argument('--knn-k', default=10, type=int, help='k in kNN monitor')
    parser.add_argument('--knn-t', default=0.1, type=float, help='softmax temperature in kNN monitor; could be different with moco-t')
    # utils
    parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
    parser.add_argument('--results-dir', default='result', type=str, metavar='PATH', help='path to cache (default: none)')
    
    parser.add_argument('--dataset_name', default='CIFAR10', type=str, help='Dataset name')
    parser.add_argument('--patience', default=20, type=int, help='Earlystop patience')
    parser.add_argument('--seed', default=0, type=int, help='seed')
    
    
    args = parser.parse_args()  # running in command line

    args.epochs = 200
    args.cos = True
    args.schedule = []  # cos in use
    args.symmetric = False


    
    # data prepare
    train_data, memory_data, test_data, test_data_2 = utils.get_medical_dataset()
    
    train_loader = DataLoader(train_data, batch_size=args.batch_size, num_workers = 12, shuffle=True, pin_memory=True, drop_last=True)
    memory_loader = DataLoader(memory_data, batch_size=args.batch_size, num_workers = 12, shuffle=False, pin_memory=True)
    val_loader = DataLoader(test_data, batch_size=args.batch_size, num_workers = 12, shuffle=False, pin_memory=True)
    test_loader = DataLoader(test_data_2, batch_size=args.batch_size, num_workers = 12, shuffle=False, pin_memory=True)
    
    #train_loader, memory_loader, test_loader = utils.get_dataset(args)
    


    seed = args.seed
    np.random.seed(seed=seed)
    start = time.time()
    epoch_all = 0
    num_hyper_param = 0
    res_10000 = np.zeros((10000, 5))
    hyper_param = np.zeros((100, 4))
    print(seed)
    all_time = 50
        
    flag = 0
        
    while (time.time() - start)/3600 <= all_time:
        lr = log_n_uniform(-4.5,-1.5)
        wd = log_n_uniform(-6.5,-3.5)
        temperature = uniform(0.07, 0.12)
        mad = uniform(0.99,0.9999)
        flag = flag + 1
        
        hyper_param[num_hyper_param][0] = lr
        hyper_param[num_hyper_param][1] = wd
        hyper_param[num_hyper_param][2] = mad
        hyper_param[num_hyper_param][3] = temperature
        num_hyper_param += 1
        np.save("MoCo_hyper_param_"+str(seed), hyper_param)
        

        model = ModelMoCo(
                    dim=args.moco_dim,
                    K=args.moco_k,
                    m=mad,
                    T=temperature,
                    arch=args.arch,
                    bn_splits=args.bn_splits,
                    symmetric=args.symmetric).cuda()
    
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_loader), eta_min=0, last_epoch=-1)

        res = np.zeros((args.epochs, 2))
    
        early_stopping = EarlyStopping(patience=args.patience, verbose=True)
        best_acc = 0

        train_loss = 0
        val_acc_1 = 0
        test_acc_1 = 0
        epoch = 0
    
            
        for epoch in range(1, args.epochs + 1):
            val_acc_1, reg = pred_val(model.encoder_q, memory_loader, val_loader)
            print(epoch, val_acc_1)
            
            if best_acc < val_acc_1:
                best_acc = val_acc_1
            
            test_acc_1 = pred_test(model.encoder_q, memory_loader, test_loader, reg)

            train_loss = train(model, train_loader, optimizer, epoch, args)
            
            error_rate = 1 - val_acc_1
            early_stopping(error_rate, model) # you can replace error_rate with val loss
            if early_stopping.early_stop:
                break
            

            
            res_10000[epoch_all][0] = train_loss
            res_10000[epoch_all][1] = val_acc_1
            res_10000[epoch_all][2] = test_acc_1
            res_10000[epoch_all][3] = (time.time() - start)/60
            res_10000[epoch_all][4] = flag
            np.save("MoCo_"+str(seed), res_10000)
            
            epoch_all += 1
    
                    