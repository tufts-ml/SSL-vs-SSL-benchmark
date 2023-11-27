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
import torchvision

import torch
import torchvision
from torch import nn

from lightly.loss import DINOLoss
from lightly.models.modules import DINOProjectionHead
from lightly.models.utils import deactivate_requires_grad, update_momentum
#from lightly.utils.scheduler import cosine_schedule

import copy

import libml.utils as utils
from libml.utils import EarlyStopping
import pandas as pd
from sklearn.linear_model import LogisticRegression
import time
from sklearn.metrics import confusion_matrix as sklearn_cm


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('Home device: {}'.format(device))

def cosine_schedule(
    step: float, max_steps: float, start_value: float, end_value: float
) -> float:
    """
    Use cosine decay to gradually modify start_value to reach target end_value during iterations.

    Args:
        step:
            Current step number.
        max_steps:
            Total number of steps.
        start_value:
            Starting value.
        end_value:
            Target value.

    Returns:
        Cosine decay value.

    """
    if step < 0:
        raise ValueError("Current step number can't be negative")
    if max_steps < 1:
        raise ValueError("Total step number must be >= 1")
    if step > max_steps:
        warnings.warn(
            f"Current step number {step} exceeds max_steps {max_steps}.",
            category=RuntimeWarning,
        )

    if max_steps == 1:
        # Avoid division by zero
        decay = end_value
    elif step == max_steps:
        # Special case for Pytorch Lightning which updates LR scheduler also for epoch
        # after last training epoch.
        decay = end_value
    else:
        decay = (
            end_value
            - (end_value - start_value)
            * (np.cos(np.pi * step / (max_steps - 1)) + 1)
            / 2
        )
    return decay

class DINO(torch.nn.Module):
    def __init__(self, backbone, input_dim):
        super().__init__()
        self.student_backbone = backbone
        self.student_head = DINOProjectionHead(
            input_dim, 512, 64, 2048, freeze_last_layer=1
        )
        self.teacher_backbone = copy.deepcopy(backbone)
        self.teacher_head = DINOProjectionHead(input_dim, 512, 64, 2048)
        deactivate_requires_grad(self.teacher_backbone)
        deactivate_requires_grad(self.teacher_head)

    def forward(self, x):
        y = self.student_backbone(x).flatten(start_dim=1)
        z = self.student_head(y)
        return z

    def forward_teacher(self, x):
        y = self.teacher_backbone(x).flatten(start_dim=1)
        z = self.teacher_head(y)
        return z

    def return_rep(self, x):
        y = self.teacher_backbone(x).flatten(start_dim=1)
        return F.normalize(y, dim=-1)

def calculate_balanced_accuracy(output, target):
    
    confusion_matrix = sklearn_cm(target, output)
    n_class = confusion_matrix.shape[0]
    #print('Inside calculate_balanced_accuracy, {} classes passed in'.format(n_class), flush=True)
    
    recalls = []
    for i in range(n_class): 
        recall = confusion_matrix[i,i]/np.sum(confusion_matrix[i])
        recalls.append(recall)
        #print('class{} recall: {}'.format(i, recall), flush=True)
        
    balanced_accuracy = np.mean(np.array(recalls))
    
    return balanced_accuracy * 100

def log_n_uniform(low=-3, high=0, size=1, coefficient=1, base=10):
    power_value = np.random.uniform(low, high, size)[0]
    
    return coefficient*np.power(base, power_value)
    
def uniform(low=0.0, high=1.0, size=1, decimal=1):
    
    return np.random.uniform(low=low, high=high, size=size)[0]
    
    

def train(args, net, data_loader, train_optimizer, scheduler, criterion):
    net.train()
    total_loss, total_num, train_bar = 0.0, 0, tqdm(data_loader)
    momentum_val = cosine_schedule(epoch, epochs, 0.996, 1)
    for pos_1, pos_2, target in train_bar:
        pos_1, pos_2 = pos_1.cuda(non_blocking=True), pos_2.cuda(non_blocking=True)
        update_momentum(model.student_backbone, model.teacher_backbone, m=momentum_val)
        update_momentum(model.student_head, model.teacher_head, m=momentum_val)
        teacher_out = [model.forward_teacher(pos_1), model.forward_teacher(pos_2)]
        #print((teacher_out[0][0]**2).sum())
        student_out = [model.forward(pos_1), model.forward(pos_2)]
        #print(teacher_out, student_out)
        loss = criterion(teacher_out, student_out, epoch=epoch)
        total_loss += loss.detach()
        loss.backward()
        # We only cancel gradients of student head.
        model.student_head.cancel_last_layer_gradients(current_epoch=epoch)
        optimizer.step()
        optimizer.zero_grad()

        
    print("---------------------------------------------")
    if epoch >= 0:
        scheduler.step()

    return total_loss / total_num



def test(net, memory_data_loader, test_data_loader):
    net.eval()
    total_top1, total_top5, total_num, feature_bank, label_bank = 0.0, 0.0, 0, [], []
    with torch.no_grad():
        # generate feature bank
        for data, _, target in tqdm(memory_data_loader, desc='Feature extracting'):
            feature = net.return_rep(data.to(device, non_blocking=True))
            #print(feature.shape)
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
            feature = net.return_rep(data.to(device, non_blocking=True))
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
            feature = net.return_rep(data.to(device, non_blocking=True))
            print(feature.shape)
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
                feature = net.return_rep(data.to(device, non_blocking=True))
                #top_1 = len(np.where(clf.predict(feature.cpu().detach().numpy()) == target.cpu().numpy())[0])
                #total_top1 += top_1
                #print(total_top1)
                label_test_bank = label_test_bank + list(target.cpu().detach().numpy())
                label_test_pred = label_test_pred + list(clf.predict(feature.cpu().detach().numpy()))
                
            balanced_accuracy = calculate_balanced_accuracy(label_test_pred, label_test_bank)
            print(balanced_accuracy, reg)
            val_max_all.append(balanced_accuracy)
            reg_all.append(reg)
            
    return val_max_all[int(np.where(val_max_all == max(val_max_all))[0][0])], reg_all[int(np.where(val_max_all == max(val_max_all))[0][0])]

# test for one epoch, use weighted knn to find the most similar images' label to assign the test image
def pred_test(net, memory_data_loader, test_data_loader, reg):
    net.eval()
    total_top1, total_top5, total_num, feature_bank, label_bank = 0.0, 0.0, 0, [], []
    with torch.no_grad():
        # generate feature bank
        for data, _, target in tqdm(memory_data_loader, desc='Feature extracting'):
            #print(target)
            feature = net.return_rep(data.to(device, non_blocking=True))
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
            feature = net.return_rep(data.to(device, non_blocking=True))
            #top_1 = len(np.where(clf.predict(feature.cpu().detach().numpy()) == target.cpu().numpy())[0])
            #total_top1 += top_1
            #print(total_top1)
            label_test_bank = label_test_bank + list(target.cpu().detach().numpy())
            label_test_pred = label_test_pred + list(clf.predict(feature.cpu().detach().numpy()))
            
        balanced_accuracy = calculate_balanced_accuracy(label_test_pred, label_test_bank)
        print(balanced_accuracy)
            

    return balanced_accuracy
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train SimCLR')
    parser.add_argument('--root', type=str, default='../data', help='Path to data directory')
    parser.add_argument('--feature_dim', default=128, type=int, help='Feature dim for latent vector')
    parser.add_argument('--temperature', default=0.1, type=float, help='Temperature used in softmax')
    parser.add_argument('--k', default=10, type=int, help='Top k most similar images used to predict the label')
    parser.add_argument('--batch_size', default=256, type=int, help='Number of images in each mini-batch')
    parser.add_argument('--lr', default=0.001, type=float, metavar='LR', help='initial learning rate')
    parser.add_argument('--wd', default=1e-6, type=float, metavar='W', help='weight decay (default: 1e-4)')
    parser.add_argument('--epochs', default=400, type=int, help='Number of sweeps over the dataset to train')
    parser.add_argument('--dataset_name', default='cifar10', type=str, help='Choose loss function')
    parser.add_argument('--patience', default=10, type=int, help='Earlystop patience')
    parser.add_argument('--seed', default=0, type=int, help='seed')

    # args parse
    args = parser.parse_args()
    feature_dim, temperature, k = args.feature_dim, args.temperature, args.k
    batch_size, epochs = args.batch_size, args.epochs
    dataset_name = args.dataset_name


    #configuring an adaptive beta if using annealing method
    
    result = np.zeros(epochs)
    
    # data prepare

    print("Loading data")
    train_data, memory_data, test_data, test_data_2 = utils.get_medical_dataset()

    train_loader = DataLoader(train_data, batch_size=batch_size, num_workers = 12, shuffle=True, pin_memory=True, drop_last=True)
    memory_loader = DataLoader(memory_data, batch_size=batch_size, num_workers = 12, shuffle=False, pin_memory=True)
    val_loader = DataLoader(test_data, batch_size=batch_size, num_workers = 12, shuffle=False, pin_memory=True)
    test_loader = DataLoader(test_data_2, batch_size=batch_size, num_workers = 12, shuffle=False, pin_memory=True)
    print("Training data loading")
    
    
    # model setup and optimizer config


    # training loop
    #os.makedirs('results/{}'.format(dataset_name))

        
    seed = args.seed
    np.random.seed(seed=seed)
    start = time.time()
    epoch_all = 0
    num_hyper_param = 0
    res_10000 = np.zeros((10000, 5))
    hyper_param = np.zeros((1000, 4))
    print(seed)
    
    all_time = 50
        
    flag = 0
        
    while (time.time() - start)/3600 <= all_time:
        lr = log_n_uniform(-4.5,-1.5)
        wd = log_n_uniform(-6.5,-3.5)
        temperature = uniform(0.07, 0.12)
        center_momentum = uniform(0.8, 1.0)
        flag = flag + 1

        hyper_param[num_hyper_param][0] = lr
        hyper_param[num_hyper_param][1] = wd
        hyper_param[num_hyper_param][2] = temperature
        hyper_param[num_hyper_param][3] = center_momentum
        num_hyper_param += 1
        np.save("DINO_hyper_param"+str(seed), hyper_param)
        
                
        resnet = torchvision.models.resnet18()
        backbone = nn.Sequential(*list(resnet.children())[:-1])
        input_dim = 512
        model = DINO(backbone, input_dim)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
    
        #criterion = SwaVLoss(temperature = temperature)
        criterion = DINOLoss(output_dim=2048, warmup_teacher_temp_epochs=5, student_temp = temperature, center_momentum = center_momentum)
        criterion = criterion.to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_loader), eta_min=0, last_epoch=-1)

        res = np.zeros((epochs, 2))
    
        early_stopping = EarlyStopping(patience=args.patience, verbose=True)
        best_acc = 0

        train_loss = 0
        val_acc_1 = 0
        test_acc_1 = 0
        epoch = 0
    
            
        results = {'epoch' : [], 'train_loss': [], 'test_acc@1': [], "time": []}
    
        for epoch in range(1, epochs + 1):
            
            val_acc_1, reg = pred_val(model, memory_loader, val_loader)
            print(epoch, val_acc_1)
            
            if best_acc < val_acc_1:
                best_acc = val_acc_1
                test_acc_1 = pred_test(model, memory_loader, test_loader, reg)


            #test_acc_1 = test(model, memory_loader, test_loader_2)
            train_loss = train(args, model, train_loader, optimizer, scheduler, criterion)
            print(train_loss)

        
            
            error_rate = 1 - val_acc_1
            early_stopping(error_rate, model) # you can replace error_rate with val loss
            if early_stopping.early_stop:
                break
            
            
            res_10000[epoch_all][0] = train_loss
            res_10000[epoch_all][1] = val_acc_1
            res_10000[epoch_all][2] = test_acc_1
            res_10000[epoch_all][3] = (time.time() - start)/60
            res_10000[epoch_all][4] = flag
            np.save("DINO_"+str(seed), res_10000)
            
            epoch_all += 1
                
                
            