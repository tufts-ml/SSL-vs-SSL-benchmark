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
from libml.utils import EarlyStopping

import libml.utils as utils
from libml.model import Model

import torch
from libml.byol import BYOL
from torchvision import models
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix as sklearn_cm

import pandas as pd
import time

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('Home device: {}'.format(device))


from sklearn.metrics import confusion_matrix as sklearn_cm

def calculate_balanced_accuracy(output, target):
    
    confusion_matrix = sklearn_cm(target, output)
    n_class = confusion_matrix.shape[0]
    print('Inside calculate_balanced_accuracy, {} classes passed in'.format(n_class), flush=True)
    
    recalls = []
    for i in range(n_class): 
        recall = confusion_matrix[i,i]/np.sum(confusion_matrix[i])
        recalls.append(recall)
        print('class{} recall: {}'.format(i, recall), flush=True)
        
    balanced_accuracy = np.mean(np.array(recalls))
    
    return balanced_accuracy * 100

def log_n_uniform(low=-3, high=0, size=1, coefficient=1, base=10):
    power_value = np.random.uniform(low, high, size)[0]
    
    return coefficient*np.power(base, power_value)
    
def uniform(low=0.0, high=1.0, size=1, decimal=1):
    
    return np.random.uniform(low=low, high=high, size=size)[0]


def train(args, net, data_loader, train_optimizer, scheduler):
    net.train()
    total_loss, total_num, train_bar = 0.0, 0, tqdm(data_loader)
    for pos_1, pos_2, target in train_bar:
        pos_1, pos_2 = pos_1.to(device,non_blocking=True), pos_2.to(device,non_blocking=True)
        loss = net(pos_1, pos_2)
        #(loss)

        train_optimizer.zero_grad()
        loss.backward()
        train_optimizer.step()


        total_num += batch_size
        total_loss += loss.item() * batch_size

        #train_bar.set_description('Train Epoch: [{}/{}] Loss: {:.4f}'.format(epoch, epochs, total_loss / total_num))

    if epoch >= 10:
        scheduler.step()
    
    #print(total_loss / total_num)

    return total_loss / total_num


# test for one epoch, use weighted knn to find the most similar images' label to assign the test image
def test(net, memory_data_loader, test_data_loader):
    net.eval()
    total_top1, total_top5, total_num, feature_bank, label_bank = 0.0, 0.0, 0, [], []
    with torch.no_grad():
        # generate feature bank
        for data, _, target in tqdm(memory_data_loader, desc='Feature extracting'):
            feature, rep = net(data.to(device, non_blocking=True), _, return_embedding = True)
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
            feature, rep = net(data, _, return_embedding = True)
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
    parser.add_argument('--batch_size', default=256, type=int, help='Number of images in each mini-batch')
    parser.add_argument('--epochs', default=400, type=int, help='Number of sweeps over the dataset to train')
    parser.add_argument('--dataset_name', default='cifar10', type=str, help='Choose loss function')
    parser.add_argument('--patience', default=20, type=int, help='Earlystop patience')


    # args parse
    args = parser.parse_args()
    batch_size, epochs = args.batch_size, args.epochs
    dataset_name = args.dataset_name



    #configuring an adaptive beta if using annealing method

    
    # data prepare
    #train_data, memory_data, test_data = utils.get_dataset(dataset_name, root=args.root)
    train_data, memory_data, test_data, test_data_2 = utils.get_medical_dataset()

    train_loader = DataLoader(train_data, batch_size=batch_size, num_workers = 0, shuffle=True, pin_memory=True, drop_last=True)
    memory_loader = DataLoader(memory_data, batch_size=batch_size, num_workers = 0, shuffle=False, pin_memory=True)
    val_loader = DataLoader(test_data, batch_size=batch_size, num_workers = 0, shuffle=False, pin_memory=True)
    test_loader = DataLoader(test_data_2, batch_size=batch_size, num_workers = 0, shuffle=False, pin_memory=True)
    
    for seed in range(0, 5):
        np.random.seed(seed=seed)
        start = time.time()
        epoch_all = 0
        num_hyper_param = 0
        res_10000 = np.zeros((10000, 5))
        hyper_param = np.zeros((100, 3))
        print(seed)
        all_time = 50
            
        flag = 0
            
        while (time.time() - start)/3600 <= all_time:
            lr = log_n_uniform(-4.5,-1.5)
            wd = log_n_uniform(-6.5,-3.5)
            flag = flag + 1

            hyper_param[num_hyper_param][0] = lr
            hyper_param[num_hyper_param][1] = wd
            num_hyper_param += 1
            np.save("SimSiam_hyper_param"+str(seed), hyper_param)
            
            resnet = models.resnet18(pretrained=False)
            
            model = BYOL(
            resnet,
            image_size = 32,
            hidden_layer = 'avgpool',
            projection_size = 256,           # the projection size
            projection_hidden_size = 4096,   # the hidden dimension of the MLP for both the projection and prediction
            use_momentum = False    # the moving average decay factor for the target encoder, already set at what paper recommends
            ).cuda()
        
            optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_loader), eta_min=0, last_epoch=-1)


            early_stopping = EarlyStopping(patience=args.patience, verbose=True)
            best_acc = 0
    
            train_loss = 0
            val_acc_1 = 0
            test_acc_1 = 0
            epoch = 0
        
                
            results = {'epoch' : [], 'train_loss': [], 'test_acc@1': [], "time": []}
        
            for epoch in range(1, epochs + 1):
                val_acc_1 = test(model, memory_loader, val_loader)
                print(epoch, val_acc_1)
                
                if best_acc < val_acc_1:
                    best_acc = val_acc_1
                    test_acc_1 = test(model, memory_loader, test_loader)

                #test_acc_1 = test(model, memory_loader, test_loader_2)
                train_loss = train(args, model, train_loader, optimizer, scheduler)
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
                np.save("SimSiam_"+str(seed), res_10000)
                
                epoch_all += 1
                
                
                
                
                
