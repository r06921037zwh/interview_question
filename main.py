# -*- coding: utf-8 -*-
"""
Created on Fri Jun 12 13:48:37 2020

@author: zhewei
"""
import pandas as pd
import numpy as np
import random
import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from sklearn.preprocessing import StandardScaler
from dataset import interviewDataset

# read in data and prepare for training
def dataProcess(filename='interviewData.csv', test_percent=0.2):  
    # read csv file
    df = pd.read_csv(filename)
    
    # make target
    y_data = [int(i == 1) for i in list(df['hardbin_FT1'])]
    y_data = np.array(y_data, 'float64')
    
    # sample train and test dataset
    neg_index = []
    pos_index = []
    for i, y in enumerate(y_data):
        if y != 1:
            neg_index.append(i)
        else:
            pos_index.append(i)
            
    test_pos_index = random.sample(pos_index, int(test_percent*len(pos_index)))
    test_neg_index = random.sample(neg_index, int(test_percent*len(neg_index)))
    train_pos_index = [i for i in pos_index if i not in test_pos_index]
    train_neg_index = [i for i in neg_index if i not in test_neg_index]

    # drop unnecessary features
    df = df.drop(['totaldietestseconds_WS1','wafername_WS1','ecid', 'hardbin_FT1'], 1)
    X_data = df.to_numpy()

    # Feature Normalization.
    # All features should have the same range of values (-1,1)
    sc = StandardScaler()
    X_data = sc.fit_transform(X_data)
    
    # make train and test data
    upsample_times = int(len(train_pos_index) / len(train_neg_index))
    X_train, y_train = [], []
    X_test, y_test = [], []
    for i, v in enumerate(X_data):
        if i in train_pos_index:
            X_train.append(X_data[i, :])
            y_train.append(y_data[i])
        elif i in train_neg_index:
            for _ in range(upsample_times):
                X_train.append(X_data[i,:])
                y_train.append(y_data[i])
        elif i in test_pos_index or i in test_neg_index:
            X_test.append(X_data[i,:])
            y_test.append(y_data[i])
    

    # convert the arrays to torch tensors
    X_train, y_train = torch.tensor(X_train, dtype=torch.float), torch.tensor(y_train, dtype=torch.float).unsqueeze(1)
    X_test, y_test = torch.tensor(X_test, dtype=torch.float), torch.tensor(y_test, dtype=torch.float).unsqueeze(1)
    #print(X_train.shape)
    #print(y_train.shape)
    return X_train, y_train, X_test, y_test


def train(args, model, device, train_loader, optimizer, criterion):
    model.train()
    for batch_idx, (X_train, y_train) in enumerate(train_loader):
        optimizer.zero_grad()
        X_train, y_train = X_train.to(device), y_train.to(device)
        y_pred = model(X_train)
        loss = criterion(y_pred, y_train)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('[{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                   batch_idx * len(X_train), len(train_loader.dataset),
                   100. * batch_idx / len(train_loader), loss.item()))
            
def test(model, device, test_loader, criterion):
    model.eval()  
    with torch.no_grad():
        correct = 0
        predict_results = []
        ground_truth = []
        for X_test, y_test in test_loader:
            X_test, y_test = X_test.to(device), y_test.to(device)
            outputs = model(X_test)
            loss = criterion(outputs, y_test)
            y_pred = (outputs > 0.5).float()
            correct += (y_pred == y_test).sum().item()
            predict_results.extend(list(y_pred.cpu().numpy()))
            ground_truth.extend(list(y_test.cpu().numpy()))
        # Print statistics 
        print("Test Loss: {:.3f}, Accuracy: {:.3f}\n".format(loss.item(), 100. * (correct/len(test_loader.dataset))))
        cal_confusion_matrix(ground_truth, predict_results)

def cal_confusion_matrix(y_true, y_pred):
    from sklearn.metrics import confusion_matrix
    #y_true = [2, 0, 2, 2, 0, 1]
    #y_pred = [0, 0, 2, 2, 0, 2]
    print("----- Confusion Matrix -----")
    cm = confusion_matrix(y_true, y_pred)
    df = pd.DataFrame(data=cm, index=['True-Fail', 'True-Pass'], columns=['Pred-Fail', 'Pred-Pass'])
    print(df)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    precision = tp/(tp+fp)
    recall = tp/(tp+fn)
    print("Accuracy : {}".format((tn+tp)/(tn+fp+fn+tp)))
    print("Precision : {}".format(precision))
    print("Recall : {}".format(recall))
    print("F1 score : {}".format(2./(1./precision+1./recall)))
    
# build the network      
class Net(nn.Module):
    def __init__(self, input_size):
        super(Net, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 100),
            nn.ReLU(),
            nn.Linear(100, 10),
            nn.ReLU(),
            nn.Linear(10, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='Interview Data Classification')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=64,
                        help='input batch size for testing (default: 64)')
    parser.add_argument('--epochs', type=int, default=20,
                        help='number of epochs to train (default: 20)')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='learning rate (default: 0.01)')
    parser.add_argument('--gamma', type=float, default=0.7,
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--log-interval', type=int, default=5, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='Saving the current Model')
    
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # perform data process
    X_train, y_train, X_test, y_test = dataProcess(filename='interviewData.csv', test_percent=0.2)
    
    # Load the train and test dataset to dataloader
    train_dataset = interviewDataset(X_train, y_train)
    test_dataset = interviewDataset(X_test, y_test)
    print(len(train_dataset))
    
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=args.batch_size,
                              shuffle=True)
    
    test_loader = DataLoader(dataset=test_dataset,
                              batch_size=args.batch_size,
                              shuffle=False)
    '''
    # Test data loader
    print("There is {} batches in the dataset".format(len(test_loader)))
    for (x,y) in test_loader:
        print("For one iteration (batch), there is:")
        print("Data:    {}".format(x.shape))
        print("Labels:  {}".format(y.shape))
        break
    '''
    
    model = Net(X_train.shape[1]).to(device)
    print(model)
    # Binary Cross-entropy
    criterion = torch.nn.BCELoss(reduction='mean')   
    
    # optimizer choices
    #optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    # Learning Rate (lr) adjustment
    #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=args.gamma)
    
    for epoch in range(1, args.epochs + 1):
        print("======= Epoch {} ======".format(epoch))
        train(args, model, device, train_loader, optimizer, criterion)
        test(model, device, test_loader, criterion)
        #scheduler.step()
    
    if args.save_model:
        torch.save(model.state_dict(), "hw.pt")

if __name__ == '__main__':
    main()
